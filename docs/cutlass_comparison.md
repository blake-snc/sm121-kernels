# sm121-kernels vs CUTLASS 4.5 Performance Comparison

## SM121a (DGX Spark), D=128, non-causal, CUDA events timing

### Flash Attention

| Config | Metric | sm121-kernels | CUTLASS 4.5 CuTe DSL | Ratio |
|--------|--------|--------------|----------------------|-------|
| B=2 H=16 S=1024 BF16 | TFLOPS | 10.0 (V11 TMA) | — | — |
| B=2 H=16 S=2048 BF16 | TFLOPS | 28.8 (V11 TMA) | — | — |
| B=1 H=16 S=2048 BF16 | TFLOPS | ~30 (V11 TMA) | 67.3 | 0.45x |
| B=2 H=16 S=1024 FP8 | TFLOPS | 64.9 (V12c VT) | — | — |
| B=2 H=16 S=2048 FP8 | TFLOPS | 98.3 (V12c VT) | — | — |
| B=1 H=16 S=2048 FP8 | TFLOPS | ~100 (V12c VT) | 44.9 | **2.2x** |

> **Config note (unambiguous)**: the **67.3 TFLOPS** CUTLASS BF16 number — and the corresponding
> ~30 TFLOPS V11 number it is compared against — were measured at **B=1, H=16, S=2048, D=128,
> non-causal** (the H=16 row in the table above). Any superlative or ratio derived from 67.3 is at
> H=16; do not mix it with H=32 figures.

### Analysis

**BF16 Flash Attention**: CUTLASS CuTe DSL generates higher-performance BF16 kernels than our hand-written V11 PTX. Their advantage comes from:
- Optimized software pipelining (CuTe DSL auto-generates multi-stage pipelines)
- Better register allocation (MLIR compiler optimizations)
- More aggressive instruction scheduling

Our V11 at 30 TFLOPS is limited by the hand-written PTX scheduling — ptxas doesn't always generate optimal SASS from manually ordered PTX.

> **Scope of the 0.45× BF16 ratio.** Both numbers above are at **B=1, H=16, S=2048** against the
> archived V11 TMA kernel. The production BF16 kernel (V21) is occupancy-sensitive and scales with
> available work: at B=2, H=32 it measures ~48 TFLOPS at S=2048, ~69 at S=4096, and **~75 TFLOPS
> at S=8192** (non-causal dense, CUDA events, this GB10; `~` because GB10 has no clock lock). For a long-context external comparable,
> upstream flash-attention's FA4 forward measures 86.9 TFLOPS dense / 74.6 causal at the same
> S=8192 H32 B2 cell on this box — i.e. V21 is within ~13% of FA4 dense and on par with FA4 causal
> at serving-relevant sequence lengths. The BF16 deficit is a short-sequence occupancy effect, not a
> structural property of hand-written PTX.

**FP8 Flash Attention**: sm121-kernels is **2.2x faster** here.

> **IMPORTANT — read this caveat with the 2.2× number.** This comparison is against a
> **known-limited CUTLASS FP8 path**: CUTLASS's FP8 FA on SM120 is impaired by Issue #3044
> (MmaAtomSM80Type missing `kind::f8f6f4` lowering; open as of 2026-06), so its FP8 kernel falls
> back to suboptimal code generation. The 2.2× is therefore against an impaired baseline, not
> CUTLASS at its best. **cuDNN is the primary FP8 reference** for the headline comparison.

Our hand-tuned PTX with the VT-GMEM layout eliminates the per-thread transpose bottleneck entirely.

**Key takeaway**: Hand-tuned PTX wins where the compiler toolchain has gaps (FP8 MMA on SM120). Compiler-generated code wins where the toolchain is mature (BF16 MMA with SM80 instructions). As CUTLASS fixes #3044, their FP8 performance will likely improve — but the VT-GMEM layout innovation is kernel-level, not toolchain-level.

## NVFP4 GEMM

CUTLASS 4.5 has SM120 NVFP4 support only via the C++ template library (example 80b — sparse blockscaled GEMM). No CuTe DSL dense NVFP4 GEMM for SM120 yet. Direct comparison not possible — our `gemm_nvfp4_mma.ptx` is a dense (non-sparse) NVFP4 GEMM.

## FP8 MMA Status (Issue #3044)

Still open as of 2026-06. FP8 `mma.sync.aligned.kind::f8f6f4.m16n8k32` on SM120 requires MLIR lowering in CUTLASS's closed-source CuTe DSL backend. Not yet landed in the 4.5.x releases.

## Methodology

- Both benchmarks run on the same DGX Spark (SM121a)
- CUTLASS reference (authoritative attribution for the 67.3 TFLOPS BF16 and 44.9 TFLOPS FP8
  numbers): **CUTLASS 4.5 CuTe DSL**, example
  `examples/python/CuTeDSL/blackwell_geforce/benchmark_fp8_vs_bf16.py`.
    Measured on the same DGX Spark from a CUTLASS 4.5 CuTe DSL checkout; the CUTLASS harness is
    not bundled here, so the 67.3/44.9 figures are point-in-time references on our box rather
    than reproducible from this tree. Reproduce the sm121-kernels side with the bundled
    `benchmark` example.
  - Attribution note: the CUTLASS column is the CUTLASS CuTe DSL example above (not flash-attn-4,
    which is a separate codebase) — use this attribution consistently.
- sm121-kernels: `cargo run --release --example benchmark`
- Timing: CUDA events, median reported, after 5 warm-up iterations: 200 measured
  iterations for flash-attention, 100 for GEMM (matches `docs/benchmark_methodology.md`)
