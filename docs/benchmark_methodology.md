# Benchmark Methodology

## Hardware

- **Device**: NVIDIA DGX Spark (GB10)
- **GPU**: SM121a (Blackwell GeForce architecture)
- **Memory**: 128 GB LPDDR5x unified memory, 273 GB/s bandwidth
- **SMs**: 48
- **CUDA**: 13.0+
- **PTX ISA**: 8.8

## Measurement

All benchmarks use CUDA event timing with warm-up runs to eliminate JIT and cache effects.

```
cargo run --release --example benchmark
```

### Methodology

1. **Warm-up**: 5 iterations (discarded), all kernel families
2. **Measurement**, per kernel family (as wired in the `benchmark` example):
   - Flash attention variants (BF16 V3/V21/V22, FP8 V11 TMA/V12a/V12c VT): 200 iterations
   - FP8 FA d128 baseline: 100 iterations
   - GEMM (BF16 MMA, FP8 MMA): 100 iterations
   - Top-k sampling, MoE routing: 100 iterations
3. **Timing**: CUDA events (cudaEventRecord/ElapsedTime), microsecond precision
4. **Reported**: Median, mean, min, max latency; TFLOPS; GB/s
5. **TFLOPS computation**: `4 * B * H * Sq * Skv * D / latency_seconds / 1e12`
   (two GEMMs — QK^T and PV — at 2 FLOPs per MAC)
   (causal=false; the benchmarked kernels are non-causal, so the full Sq·Skv term is
   correct — causal variants halve the FLOP count)

### Configuration

Default benchmark parameters:
- **B** = 1 (batch size)
- **H** = 32 (num heads)
- **Sq = Skv** = 2048 (sequence length)
- **D** = 128 (head dimension)
- **scale** = 1/sqrt(128) ≈ 0.0884

### Reference baselines

Compared against:
- **cuDNN** (NVIDIA's closed-source kernel library): cuDNN fused scaled-dot-product-attention
  via the cudnn-frontend graph API (SDPA fprop), measured on the same DGX Spark with the
  engine/graph config logged by the harness at run time. The cuDNN comparison harness is
  **not bundled in this repo**, so the cuDNN figure is a point-in-time reference on our box,
  not a number reproducible from this tree; reproduce the sm121-kernels side with the bundled
  `benchmark` example.
- **CUTLASS 4.5 CuTe DSL**: the BF16 (67.3 TFLOPS) and FP8 CuTe DSL reference numbers come from the
  CUTLASS example `examples/python/CuTeDSL/blackwell_geforce/benchmark_fp8_vs_bf16.py`. See
  `cutlass_comparison.md` for the pinned commit and exact command (single, consistent attribution
  across both docs).

### Clock/thermal methodology

GB10 (DGX Spark) does not expose application-level clock locking the way datacenter parts do
(`nvidia-smi -lgc` is not available / has no effect on this consumer Blackwell part), so **clocks
are not locked**. Instead:

- A short thermal soak (a few seconds of the kernel under test) precedes the measured run, so the
  GPU has reached a steady thermal/clock state before timing begins.
- The achieved SM clock should be **reported via `nvidia-smi` during the run** (sampled while the
  measured iterations execute), so the reader can see the clock the TFLOPS numbers were obtained at
  rather than assuming a fixed boost clock.

This is the honest current state: results are steady-state numbers at the achieved (not pinned)
clock; report the observed SM clock alongside the TFLOPS.

## Kernel Inventory

| Kernel | Dtype | Arch | Key Feature | Typical TFLOPS |
|--------|-------|------|-------------|----------------|
| V3 | BF16 | CpAsync | All-threads-load, bar.sync | ~20 |
| V11 | BF16 | TMA | Warp-specialized, mbarrier | ~26 |
| V12c VT | FP8 | TMA | Pre-transposed V layout | ~100 |
| V13 Bc=32 | BF16 | TMA | 2 CTAs/SM experiment | ~22 |
| V3 Paged | BF16 | CpAsync | Paged KV cache support | ~20 |
| V3 Split | BF16 | CpAsync | FlashDecoding split-KV | ~20 |
| Combine | — | — | Reduce split-KV partials | N/A |

## Reproducing

```bash
# Build
cargo build --release

# Run full benchmark suite
cargo run --release --example benchmark

# Run specific kernel
cargo run --release --example benchmark -- --kernel v11

# Run tests (verify correctness first)
cargo test --release -- --test-threads=1
```

## Notes

- All kernels target SM121a exclusively. They will not work on other GPU architectures.
- Tests must run with `--test-threads=1` to avoid races on shared cubin temp files.
- The `build.rs` must be touched (`touch crates/sm121-kernels/build.rs`) to force PTX recompilation when only `.ptx` files change.
- Dynamic SMEM (>48KB) requires `cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES)` at launch.
- **Iteration counts**: the per-family counts in the Methodology section above (flash attention =
  200 measured iterations, GEMM/sampling/MoE = 100) are authoritative, and the benchmark binary's
  printed banner now matches them ("Warmup: 5 iterations; Timed: 200 iterations (attention), 100
  iterations (GEMM/sampling/MoE)").
