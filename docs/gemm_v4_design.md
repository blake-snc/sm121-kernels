# GEMM v4 — warp-specialized + TMA BF16 GEMM for SM121a

**Motivation.** Existing `gemm_bf16_mma_v3`
hits 27-50 TFLOPS across our transformer-prefill shapes (25-42% of SM121a's
~119 TFLOPS BF16 peak). cuBLAS hits 75-102 TFLOPS on the same shapes
(63-86% of peak). The v4 kernel closes that gap.

## Target

Hit ≥60 TFLOPS at M=256-512 for all heavy shapes:

| Shape | v3 best | cuBLAS | v4 target |
|---|---|---|---|
| QKV proj (8192×4096) | 34 | 84 | ≥60 |
| O proj (4096×3072) | 50 | 102 | ≥75 |
| MLP gate/up (12288×4096) | 27 | 75 | ≥60 |
| MLP down (4096×12288) | 40 | 70 | ≥55 |
| GDN out_proj (4096×4096) | 46 | 102 | ≥75 |
| GDN in_proj (12288×4096) | 27 | 67 | ≥55 |
| lm_head (248320×4096) | 12 | 39 | ≥30 |

## Why v3 underperforms

Each of the 8 warps in v3 does BOTH loading (cp.async) and compute (MMA).
The cp.async issues run in the foreground; MMAs can't issue until the warp
finishes its load. The 2-stage pipeline overlaps stage-N's compute with
stage-(N+1)'s load AT THE BLOCK LEVEL, but within each warp the two
phases serialize. Net effect: load latency isn't fully hidden behind compute.

cuBLAS (and CUTLASS reference kernels) use **warp specialization**:
dedicated warps do nothing but TMA loads (producer); other warps do nothing
but MMA (consumer). Coordination via mbarrier. True overlap between load and
compute because they happen in different physical warp slots.

## v4 design

### Layout

- **5 warps (160 threads)**: 1 producer (DMA) + 4 consumer (MMA).
  Matches the proven pattern from `fa_bf16_v20_tma_bc128.ptx` and v11 attention.
- **Output tile: 128 × 128** (same as v3 — fits SMEM, MMA shape divides cleanly).
- **K-block: 32** (matches MMA m16n8k16's K dimension, same as v3).
- **MMA warp layout: 2×2** (warp_m ∈ [0,1], warp_n ∈ [0,1]).
  Each MMA warp handles a **64×64 output sub-tile**.
  Per-warp: 4 m-groups × 8 n-groups = 32 MMAs per K-block, 64 fp32 accumulators per thread.

### Pipeline

3-stage software pipeline:

```
K-iter 0: stage 0 load    stage 1 idle    stage 2 idle
K-iter 1: stage 0 compute stage 1 load    stage 2 idle
K-iter 2: stage 0 compute stage 1 compute stage 2 load
K-iter 3: stage 0 reuse   stage 1 compute stage 2 compute     (wrap around)
```

Each stage holds one (A_tile, B_tile) pair. Producer warp issues TMA loads
in advance; consumer warps wait on mbarrier parity per stage. With 3 stages
the producer can be 2 K-iters ahead of the consumer, hiding most TMA latency.

### SMEM layout (fits 99 KB budget)

```
Stage 0: A_tile [128×32] BF16 =  8 KB | B_tile [32×128] BF16 =  8 KB | total 16 KB
Stage 1: same = 16 KB
Stage 2: same = 16 KB
Mbarriers: 6 × 8 B = 48 B (per-stage A barrier + B barrier)
Total: 48 KB + 48 B
Reserve: 51 KB headroom for register spill / future tile growth
```

### MMA inner loop (consumer warp, per K-iter)

```
1. Wait on stage's A_mbar (mbarrier.try_wait.parity)
2. Wait on stage's B_mbar
3. ldmatrix.x4 A fragment (4 m-groups × 8 lanes per ldmatrix call)
4. ldmatrix.x2.trans B fragment (8 n-groups)
5. 32 × MMA_BF16_M16N8K16 (4 m-groups × 8 n-groups)
6. Signal stage's DONE_mbar (mbarrier.arrive)
```

### TMA loads (producer warp, per K-iter)

```
1. Compute next stage's TMA coords (m_off + 0, k_off + iter*32) for A
   and (k_off + iter*32, n_off + 0) for B
2. mbarrier.arrive.expect_tx for A_mbar with byte count
3. cp.async.bulk.tensor.2d for A → A_tile[stage]
4. mbarrier.arrive.expect_tx for B_mbar
5. cp.async.bulk.tensor.2d for B → B_tile[stage]
6. (TMA auto-signals mbarrier on completion; producer doesn't need explicit arrive)
7. Wait on stage's DONE_mbar so consumer is done with it before next overwrite
```

### Output write (after K-loop)

Each MMA warp writes its 64×64 sub-tile to global memory:
- 64 fp32 accumulators → convert to BF16 → store via st.global.v4.b32 (4×BF16 per thread)
- Each warp covers (warp_m × 64) to (warp_m × 64 + 64) rows, (warp_n × 64) to (warp_n × 64 + 64) cols of the CTA tile
- CTA tile starts at (ctaid.y × 128, ctaid.x × 128)

### Constraints

- M divisible by 128 (output tile)
- N divisible by 128 (output tile)
- K divisible by 32 (K-block)
- Caller allocates TMA descriptors for A and B at dispatch time

## Implementation plan

Multi-stage so each step compiles + validates before adding complexity:

**Step 1 — Scaffolding.** Entry + param loading + mbarrier init + warp dispatch (producer vs consumer). No actual compute yet. Validates the warp split + sync pattern.

**Step 2 — Single-stage TMA + MMA.** Producer does one TMA pair (A0, B0), consumer waits + MMA, signals done. K-loop iterates this for the full K dim. SMEM = 1 stage = 16 KB. Validates TMA correctness + ldmatrix addressing + MMA accumulation.

**Step 3 — Output write.** After K-loop, MMA warps convert accumulators to BF16 and write to C. Validates correctness end-to-end on small shape (M=128, N=128, K=32 → single CTA, single K-block).

**Step 4 — Full correctness gate.** Bit-equality vs v3 on multiple shapes (16×16 CTA grid for M=2048 N=2048; vary K from 32 to 4096). Compute-sanitizer clean.

**Step 5 — Throughput bench.** Measure at the hot shapes. If single-stage already hits ≥40 TFLOPS, the architecture works. If significantly below v3, root-cause before adding pipeline stages.

**Step 6 — 2-stage pipeline.** Add stage 1. Producer issues stage N+1's TMA while consumer computes stage N. Should jump throughput by 30-50%.

**Step 7 — 3-stage pipeline.** Add stage 2. Producer can be 2 ahead. Final architecture.

**Step 8 — Tile size + warp layout sweep.** Try (128, 128, 2×2 warps) vs (128, 256, 2×4 warps) vs (256, 128, 4×2 warps). Pick best.

**Step 9 — Integrate into auto-dispatch.** Update `gemm_bf16_mma_auto` to prefer v4 when shape allows. Keep v3 as fallback.

**Step 10 — vLLM head-to-head bench.** Same shapes, same hardware, our new GEMM vs cuBLAS-via-PyTorch.

## Risks / open questions

1. **TMA descriptor setup**: SM121a uses `cp.async.bulk.tensor.2d` which needs a TMA descriptor (CUtensorMap) created host-side via `cuTensorMapEncodeTiled`. Need to verify the Rust dispatch can build that for arbitrary A/B shapes per launch. (FA V20 already does this; reuse the pattern.)

2. **Register pressure**: 64 fp32 accumulators per MMA thread + working registers for loop arithmetic + ldmatrix targets. Budget is 256 regs/thread; need to verify under 200 to leave headroom.

3. **B layout in SMEM**: ldmatrix.x2.trans requires specific addressing for column-major B fragment access. The smoke-test pattern works for K_block=16; need to extend to K_block=32.

4. **mbarrier parity bit overflow**: With 3-stage pipeline, parity bits per stage need careful tracking. Each stage's mbarrier gets `mbarrier.arrive.expect_tx` from producer + `mbarrier.try_wait.parity` from consumer; parity flips per round-trip.

5. **N dimension scaling**: at very large N (e.g., lm_head N=248K), the 128×128 tile produces 248K/128 × M/128 CTAs. Need to confirm CTA count doesn't overwhelm the scheduler.

## Validation gates

- **Step 3**: bit-equality on M=128 N=128 K=32 vs v3 (single CTA test).
- **Step 4**: bit-equality on full shape sweep vs v3 (multiple CTAs, K up to 4096).
- **Step 5**: throughput ≥40 TFLOPS at QKV M=512.
- **Step 7**: throughput ≥60 TFLOPS at QKV M=512 (catches up to cuBLAS within 30%).
- **Step 10**: documented head-to-head bench showing our throughput vs cuBLAS at all hot shapes.

## Memory entry on landing

Headline: GEMM v4 (warp-specialized + TMA + 3-stage pipeline) hits X TFLOPS at QKV M=512 (cuBLAS Y, headroom Z%). Direct 2-3× per-worker throughput unlock on the existing serial serving path, no scheduler changes.

## Step 5/6/7 results (2026-05-21, 9-warp 1P+8C, 128×128 tile, K-block=32)

Throughput at M=512 across the pipeline depths:

| Shape | v3 (8-warp cp.async) | v4 single-stage | v4 2-stage | v4 3-stage | 3-stage / v3 |
|---|---|---|---|---|---|
| QKV (8192×4096) | 32.8 | 17.8 (-46%) | 24.0 (+35% vs 1S) | 25.3 (+5%) | 77% |
| O   (4096×3072) | 40.9 | 23.6 (-42%) | 33.3 (+41%) | 34.7 (+4%) | 85% |
| MLP gate (12288²) | 27.4 | 18.2 (-34%) | 22.5 (+24%) | 26.9 (+20%) | 98% |
| MLP down (K=12288) | 37.1 | 20.8 (-44%) | 19.3 (-7%) | 31.9 (+65%) | 86% |
| GDN in (12288²) | 25.6 | 21.8 (-15%) | 23.3 (+7%) | 25.9 (+11%) | 97% |
| GDN out (4096²) | 39.6 | 21.0 (-47%) | 29.4 (+40%) | 31.8 (+8%) | 79% |
| lm_head (248K×4K) | 21.7 | 15.8 (-27%) | 18.5 (+17%) | 25.2 (+36%) | **116%** |

Correctness: 9/9 shapes bit-exact vs v3 at every pipeline depth.

### Findings

- **Warp specialization works on SM121a.** 2-stage → 3-stage adds another 5-65% on top of 1-stage→2-stage's 24-41%. The architecture is sound.
- **3-stage closes deep-K starvation.** MLP down (K=12288 → 384 K-iters) was -7% under 2-stage (producer couldn't stay ahead), then +65% under 3-stage. The pattern is exactly what 3-stage is designed for.
- **v4 ≥ v3 on large-N shapes.** lm_head (N=248K) hits 116% of v3 — first shape where v4 cleanly beats v3.
- **v4 < v3 on small-N/standard-K shapes (15-23% behind).** QKV / O proj / GDN out are still bottlenecked. Hypothesis: per-warp compute (32 MMAs per K-block at K=32) is small relative to TMA setup cost; v3's distributed 8-warp loading parallelizes loads better than v4's centralized 1-producer TMA.

### Why v4 doesn't decisively beat v3 yet

v3 distributes loading across all 8 warps doing cp.async. v4 centralizes loading in 1 producer warp doing TMA. For small per-stage compute (32 MMAs / K-iter), v3's 8-way load parallelism wins. v4's win comes from larger compute-per-stage, which we haven't unlocked yet.

### Why the cheap occupancy knob does NOT work (measured 2026-06-11)

NCU on `gemm_bf16_mma_v3` at 4096³: it is **occupancy-bound, not memory-bound** — 35%
SM throughput, 33% achieved occupancy (L2 only 59%, hit rate 86%). The limiter is
**registers**: ptxas allocates 94 regs/thread → only **2 blocks/SM**. The obvious fix —
cap registers to fit a 3rd block — was tried (`.maxnreg 84` → 3 blocks/SM, 50% occupancy)
and **regressed ~8.5%** (49→45 TFLOPS at 4096³). Reason: the 94 registers are mostly the
**64 FP32 accumulators** intrinsic to the 128×128 tile; forcing fewer spills them (64 B
spill measured), and the hot loop pays more for the spill traffic than it gains from
occupancy. **Conclusion: occupancy here is gated by the accumulator footprint, so the
only way to raise it is to change the *tile/accumulator structure* (Step 8 below), not a
register knob or a maxnreg flag.** This is the empirical justification for the tile work.

### Step 8 candidates (in priority order if pursuing further)

1. **Larger tile (128×256) — BUILT and SHIPPED as `gemm_bf16_mma_v5`.** This was
   pursued not as a v4 modification but as a standalone v3-derived kernel (64×64 per
   warp, 128 FP32 accumulators/thread, threadblock swizzle for L2 reuse). It is the new
   square-shape best — ~54–56 TFLOPS at 4096³ vs v3's ~49 — and `gemm_bf16_mma_auto`
   now routes to it for M,N ≥ 2048 (N%256==0). It confirmed the prediction above: adding
   accumulators trades occupancy for compute-per-thread, and v5 lands register-bound at
   1 block/SM with an occupancy ceiling near ~56 TFLOPS. The remaining candidates below
   are unbuilt.
2. **Larger K-block (32→64)** — same warp/tile layout, doubles per-K-iter compute. SMEM jumps to 96 KB (tight). 30% refactor.
3. **2-producer split (A and B in parallel)** — halves per-producer load latency. ~25% refactor.
4. **4-stage pipeline** — diminishing returns expected after 3-stage. Lowest priority.

### Integration recommendation (Step 9, not yet shipped)

v4 currently wins decisively only at lm_head shape (N ≥ ~64K). Integrating v4 conditionally in `gemm_bf16_mma_auto` for `N ≥ 65536` would deliver the lm_head speedup without affecting QKV/O paths. Conservative deploy: wait until Step 8 tile work flips QKV/O before integrating broadly.

**Measured (2026-06-11, `benchmark` example):** the `N ≥ 65536` gate is not just caution — at large *square* shapes v4 actively **regresses**: 4096³ measures v4 ≈ 28 TFLOPS vs v3 ≈ 49 TFLOPS (41 vs 44 at 2048³). v3 stays the right default for square/QKV/O GEMM; v4 is strictly an lm_head-shape specialization until Step 8 lands. (For context, `fa_bf16_v21` sustains ~75 TFLOPS on the same tensor cores, so even v3's 49 leaves headroom — the open BF16-GEMM-to-ceiling project.)
