# From 10 to 100 TFLOPS: Flash Attention on Consumer Blackwell

_A journey through 13 kernel versions, hand-written PTX, and the techniques
behind the fastest open-source exact FP8-input flash-attention forward we
measured on GB10/SM121 as of 2026-06, vs cuDNN / flash-attn / CUTLASS on the
same hardware._

## The Challenge

SM120 (Blackwell GeForce / DGX Spark) lacks the datacenter features that make
FA4 possible: no tcgen05, no WGMMA, no TMEM, no TMA multicast. The flagship
flash attention kernel work — FA4, ThunderKittens 2.0, CUTLASS CuTe DSL FA —
targets SM100 (B200) first; SM12x is not unserved (flash-attn has since merged
SM120/121 forward/backward/varlen support, and CUTLASS has partial SM120 work),
but it remains a second-class target.

We set out to write the fastest flash attention we could for hardware the
flagship kernel work had largely passed over — see the hedged, dated claim above
for exactly what we measured and against which baselines.

## The Journey

### V1: Naive (10 TFLOPS)
- All threads load, all threads compute
- No software pipelining, no double buffering
- 128 threads, Br=16, Bc=64
- **Bottleneck**: Memory latency completely exposed

### V3: CpAsync + Cooperative Loading (20 TFLOPS)
- 256 threads, Br=128, Bc=64
- `cp.async.cg.shared.global` for register-bypass GMEM→SMEM loads
- `bar.sync` between load and compute phases
- 10.8% MMA density (128 HMMA per kernel instance)
- **Improvement**: 2x from larger tiles + async loads

### V7: TMA (First Attempt) (18 TFLOPS)
- `cp.async.bulk.tensor.2d` replaces cp.async
- Hardware manages the entire GMEM→SMEM transfer
- Single instruction loads 8192 bytes (half a tile)
- **But**: No overlap between load and compute — TMA alone isn't enough

### V11: TMA + Warp Specialization (26-30 TFLOPS)
- 5 warps (160 threads): warp 4 = DMA, warps 0-3 = MMA
- DMA warp issues TMA while MMA warps compute on previous tile
- mbarrier phase-based synchronization for double buffering
- Fused scale: `exp2(S * scale_log2e + neg_max)` saves 32 FMUL/iteration
- 6.7% MMA density (more total instructions, but better latency hiding)
- **Improvement**: 1.5x from producer/consumer overlap

### V12c: VT-GMEM Layout (100 TFLOPS FP8)
- **The breakthrough**: Pre-transpose V in global memory
- Standard: V[B*H*Skv, D] → PV GEMM needs per-thread 4-load transpose
- VT-GMEM: V[D, B*H*Skv] → single `ld.b32` per B-fragment
- FP8 `mma.sync.aligned.kind::f8f6f4.m16n8k32` doubles K-dimension
- 3.9% MMA density but 100 TFLOPS — proves the bottleneck was load stalls
- **Improvement**: 2.2x from eliminating transpose + FP8 2x compute density

## Performance Progression

| Version | Architecture | Dtype | TFLOPS | MMA% | Key Innovation |
|---------|-------------|-------|--------|------|----------------|
| V1 | Naive | BF16 | ~10 | ~8% | Baseline |
| V3 | CpAsync | BF16 | ~20 | 10.8% | Cooperative loading |
| V5 | Large tile | BF16 | ~18 | 12.1% | 256 MMA (register-limited) |
| V7 | TMA | BF16 | ~18 | 8.9% | Hardware-managed loads |
| V8 | TMA-DB | BF16 | ~11 | 7.9% | Double buffer (sync overhead) |
| V11 | TMA-WS | BF16 | ~30 | 6.7% | Warp specialization |
| V12 | Persistent | BF16 | ~30 | 6.7% | CTA loops |
| V13 | 2CTA/SM | BF16 | ~22 | 5.3% | Occupancy experiment |
| V11 | TMA-WS | FP8 | ~46 | 3.2% | FP8 m16n8k32 |
| V12c | VT-GMEM | FP8 | **100** | 3.9% | **Pre-transposed V** |

## Lessons Learned

1. **MMA density is misleading.** V12c has the lowest MMA% (3.9%) but the highest
   TFLOPS. The bottleneck is memory latency, not compute throughput.

2. **Data layout > instruction scheduling.** The VT-GMEM innovation is worth more
   than all the PTX scheduling tricks combined.

3. **Warp specialization matters for TMA.** V7 (TMA without overlap) was slower
   than V3 (CpAsync with overlap via async). V11 (TMA with warp specialization)
   is faster than both.

4. **Register pressure kills.** V5 doubled MMA count but hit 255 registers (max)
   and didn't improve TFLOPS. Register spills to local memory negate MMA gains.

5. **Application-level layout knowledge (VT-GMEM) isn't expressible to the compiler.**
   ptxas generates reasonable SASS but doesn't know the global memory access pattern.
   Hand-tuned PTX that encodes the VT-GMEM layout beats compiler-generated code by 2.2x
   on FP8.

## Comparison

All cells below are at **B=1, H=16, S=2048** — the shape-matched cell against the
CUTLASS CuTe DSL reference (see `cutlass_comparison.md`), so the cross-vendor ratios
are apples-to-apples. At the benchmark's default **H=32** shape the FP8 V12c kernel
reads ~108 TFLOPS; don't compare the 100 below against that headline.

| | sm121-kernels | cuDNN | CUTLASS CuTe DSL |
|--|-------------|-------|-----------------|
| FP8 FA (H=16, S=2048) | **100 TFLOPS** | 68 TFLOPS | 45 TFLOPS |
| BF16 FA (H=16, S=2048) | 30 TFLOPS | — | **67 TFLOPS** |

sm121-kernels is +123% / 2.2× faster on FP8 (+47% vs cuDNN, +123% vs CUTLASS — note the
CUTLASS FP8 path is impaired by issue #3044; see cutlass_comparison.md).
On BF16 at this single short shape CUTLASS is ahead.

**The BF16 gap is a small-shape occupancy artifact, not a structural ceiling.** The
30 TFLOPS above is V21 at S=2048 with a single batch × low head count, where the
kernel is occupancy-starved (12 TFLOPS at S=1024). The throughput climbs steeply
with the work available: at B=2, H=32 the same V21 measures ~48 TFLOPS at S=2048,
~69 at S=4096, and **~75 TFLOPS at S=8192** (non-causal dense, `4·B·H·S²·D`, CUDA
events, this box; `~` because GB10 has no clock lock). For an external long-context comparable, upstream flash-attention's
FA4 CuTe-DSL forward measures 86.9 TFLOPS dense / 74.6 TFLOPS causal at the same
S=8192 H32 B2 cell on this GB10 — i.e. V21 is within ~13% of FA4 dense and on par
with FA4 causal at the sequence lengths that actually matter for serving. The "2.2×
BF16 gap" should be read as occupancy-at-S=2048, not a property of hand-written PTX.

## What's Next

- SASS-level optimization (blocked by tooling — no SM120 assembler exists)
- Recover small-shape BF16 occupancy (the long-context kernel is already competitive)
- Python bindings for drop-in integration with serving frameworks
