# MMA-Optimized MLA Decode — Design Document

Target: lift `fa_bf16_mla_decode` from 0.57 TFLOPS (scalar) / ~2.5× TMA speedup to
**25–40 TFLOPS** (BF16) and **60–90 TFLOPS** (FP8) by moving the Q·c_kv and P·c_kv
dot products onto tensor cores via m16n8k16 / m16n8k32 MMA.

Last updated: 2026-04-19, after the TMA-MLA kernel landed.

---

## Why this is non-trivial

MLA decode has `seq_q = 1`. BF16 MMA on SM121a is `m16n8k16` — requires **16
queries** in the `A` operand. For a single query per head, the natural packing
is **16 heads per CTA** (each head as a "query row"). But a naive
16-heads-per-CTA grid `(H/16, batch)` gives only 16–32 CTAs at common serving
dims (DeepSeek V3 H=128, B=1–4), leaving **33–67 % GPU occupancy** on 48 SMs.

### First attempt, what went wrong

A 16-heads-per-CTA scalar TMA variant I wrote in this session ran **7× SLOWER
than scalar**. Reason: 16 CTAs × 48 SMs = 33 % occupancy, and the compute
isn't faster because it's still scalar. Confirmed in commit `f928797`.

### The correct two-pronged fix

1. **16 heads per CTA** (mandatory for MMA).
2. **Split-K across CTAs** (à la FlashDecoding) to recover occupancy for small
   batches. Each CTA owns `(head_group, kv_slice, batch)`; combine across
   kv_slice via log-sum-exp post-pass.

For B=1, H=128, Skv=32K:
- Grid without split-K: 8 × 1 = 8 CTAs → 17 % occupancy
- Grid with 8× split-K: 8 × 8 × 1 = 64 CTAs → saturates 48 SMs

---

## Layout

### SMEM (44 KB — requires `cuFuncSetAttribute`)

```
SMEM_Q_C       0     - 16383   (16 rows × 512 BF16, SWIZZLE_128B for ldmatrix)
SMEM_Q_R      16384  - 18431   (16 rows × 64  BF16, SWIZZLE_128B)
SMEM_KV0_C    18432  - 26623   (8 positions × 512 BF16, SWIZZLE_128B)
SMEM_KV0_R    26624  - 27647   (8 × 64 BF16)
SMEM_KV1_C    27648  - 35839
SMEM_KV1_R    35840  - 36863
SMEM_MBARS    36864  - 36927   (4 mbars × 16 B)
SMEM_P_TILE   36928  - 37183   (16 heads × 8 positions BF16, staged for PV)
```

**Swizzle**: unlike the scalar TMA variant (NO_SWIZZLE because scalar reads),
the MMA variant uses **SWIZZLE_128B** because MMA consumes via `ldmatrix` which
knows the XOR pattern. 64-col tile_cols means 8 TMAs per KV chunk (matching V21).

### Registers

Per thread (warp 0 doing MMA):
- Q_c fragment: 4 × b32 regs for m16n8k16 A operand (16 rows × 16 cols / 32 lanes = 8 elements → 4 × b32)
- Q_r fragment: 4 × b32 regs (same pattern on 64 dims = 4 K-iters)
- Score accumulators: 4 × f32 regs (m16n8: 16×8 / 32 = 4 per thread)
- O accumulators: need to cover 16 heads × 512 dims output. Too big for regs
  (8192 values). Stage in **smem_o_accum** (32 KB FP32) with warp-specialized
  writes + warp-parallel reduction for final normalization.

### Grid / block

- **Grid**: `(num_heads / 16, num_kv_splits, batch)` — split-K for occupancy.
- **Block**: 128 threads = 4 warps. Warp specialization à la V21:
  - Warp 0: QK MMA
  - Warp 1: softmax + P-tile staging
  - Warp 2: PV MMA
  - Warp 3: DMA + output reduction across kv_splits

---

## MMA dataflow

### QK (per 8-position KV chunk)

For each `kc` in `0..32` (covering D_C=512 in 16-dim slices):
```
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {Aq0, Aq1, Aq2, Aq3}, [SMEM_Q_C + swizzled_addr]
ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {Bk0, Bk1}, [SMEM_KV_C + swizzled_addr]
mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
  {D0, D1, D2, D3}, {Aq0, Aq1, Aq2, Aq3}, {Bk0, Bk1}, {D0, D1, D2, D3}
```
32 MMAs per chunk for Q·c_kv, 4 more for Q·k_rope. `D0..D3` accumulate the score tile.

### Softmax + P staging

Standard online softmax per row (8 rows per thread since m16 / 2 rows). Convert
FP32 scores → BF16 P matrix into `SMEM_P_TILE`.

### PV (per 16-position chunk)

Load P from SMEM, V (= c_kv) from SMEM. For each `kn` covering D_C output
dims in 8-col slices:
```
ldmatrix ... A = P[16 heads × 16 positions]
ldmatrix.trans ... B = c_kv[16 positions × 8 output-dims]
mma.sync.aligned.m16n8k16 ... D_out_tile
```
64 MMAs per 16-pos chunk to cover D_C=512 output. Output tile accumulates
into `smem_o_accum` (FP32).

### Split-K combine

After all chunks: each CTA holds partial O for its kv_slice. Split-K combine
via grid-level log-sum-exp:

```
for kv_split in 0..num_splits:
    O_final[h, :] = Σ exp(log_sum_exp[split] - max) * O_partial[split, h, :]
    / Σ exp(log_sum_exp[split] - max)
```

Implement as a small combine kernel that reads partial O + per-split LSE from global.

---

## Register budget

m16n8k16 MMA uses ~20 regs per warp for A/B/D operands. With 4 warps × 20 = 80
shared registers (warp-specialized). Softmax state ~16 regs. O accumulator in
SMEM (not regs). Total register usage per thread ≤ 128, well within SM121's 256
budget. 2 CTAs/SM achievable.

---

## Step-by-step implementation plan

1. **Start from `fa_bf16_mla_decode_tma.ptx`**. It already has the TMA double-buffered
   load path for c_kv and k_rope working.
2. **Change grid to `(num_heads/16, batch)`** (single-CTA-per-head-group, no split-K yet).
   Re-test correctness.
3. **Replace scalar QK with MMA**. First without PV changes — use MMA to compute
   scores, then still store scores to SMEM and use existing scalar PV. Validate
   correctness at intermediate step.
4. **Replace scalar PV with MMA**. Stage P tile in SMEM. Use ldmatrix.trans on
   c_kv. Validate.
5. **Add warp specialization**: warp 0 = QK, warp 2 = PV, warps 1+3 = DMA/softmax.
6. **Add split-K dimension to grid**. Implement combine kernel.
7. **Measure**. Target: ≥ 10× over scalar reference, ≥ 5× over TMA scalar variant.

---

## Risk areas

- **SWIZZLE_128B addressing**: V21 has a bug-free reference; port the swizzle XOR
  pattern directly. `ldmatrix` needs 16B-aligned addresses that alias correctly.
- **P staging**: the softmax output must be in BF16 to be ldmatrix-consumed by
  PV MMA. The conversion `f32 → bf16` and SMEM store pattern must match what
  `ldmatrix` for the A operand expects.
- **Accumulator precision**: Scores at 8K magnitudes in FP32 are fine. Output
  accumulator at 1024 positions × BF16 c_kv × BF16 P can accumulate to ~± 1K in
  FP32 — safe.
- **Split-K LSE**: numerically subtle. Use `max(LSE_i) → renorm` pattern from
  FlashDecoding combine kernel (we have that in `flash_decoding_combine.ptx`).

---

## Acceptance criteria

1. Correctness: matches `fa_bf16_mla_decode` (scalar) to within `max_diff ≤ 0.1`.
2. Performance at DeepSeek V3 scale (B=2, H=128, Skv=1024): target ≥ 30 TFLOPS
   (i.e. ≥ 10× speedup vs scalar, ≥ 4× vs current TMA).
3. Correctness regression test in `tests/test_mla_tma.rs`.

---

## Why this design over alternatives

- **Vs dense 128 heads/CTA**: too much SMEM (128 × 512 × 2 = 128 KB Q_c alone, exceeds 99 KB limit).
- **Vs full split-Q scheme**: seq_q=1, no queries to split.
- **Vs duplicate-Q-to-16x MMA**: wastes 15/16 MMA lanes, no perf gain over scalar.
- **Vs MMA-only without TMA**: memory bandwidth still bottleneck (we proved this
  is 2.5× by itself).

The two-pronged 16-heads-per-CTA + split-K grid recovers both MMA utilization
AND SM occupancy for small batches.
