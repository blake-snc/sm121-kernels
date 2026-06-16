# sm121-kernels

**CUDA/PTX kernels for the NVIDIA DGX Spark (GB10, SM121).**

To our knowledge, this is the first open-source library of hand-written PTX kernels targeting SM121 — the compute capability of the DGX Spark's GB10 superchip, and a close sibling of SM120 (the RTX 50-series). Every kernel is hand-written PTX assembly, assembled to SASS at build time by `ptxas`, embedded in a Rust binary, and dispatched through [`cudarc`](https://github.com/coreylowman/cudarc). Zero runtime CUDA-toolkit dependency: only the driver (`libcuda.so`) is needed on the box that runs it.

**259 hand-written PTX kernel files** (each assembled to its own SASS cubin at build time; a few are templated into multiple dtype/shape variants) across:

- **Flash attention** — BF16 + FP8 forward (causal, GQA, paged-KV, split-KV, varlen, SWA, softcap, MLA), backward (BF16), FP8-KV decode and chunked-prefill paths
- **GEMM** — BF16 MMA (incl. register-blocked and warp-specialized + TMA variants), FP8, W8A16, W4A16, NVFP4/MXFP4/MXFP8 block-scaled, deterministic split-K variants, backward
- **Gated DeltaNet (GDN) linear attention** — to our knowledge the only *raw hand-PTX* GDN kernel suite for SM12x (GDN otherwise lives in Triton via `flash-linear-attention`, or in CUDA): chunked prefill, MMA decode, TMA decode, state update, conv1d+SiLU, alpha/beta — the full Qwen3-Next-style hybrid-layer surface
- **MoE** — routing, permute/unpermute, grouped GEMM in BF16/FP8/MXFP8/MXFP4/NVFP4
- **Quantization** — FP8 per-token and 1×128-block, MXFP8, MXFP4, NVFP4 (+ KV-cache variants)
- **KV cache** — append/gather/scatter in BF16 and FP8, paged layouts
- **Elementwise** — RMSNorm, RoPE (incl. per-sequence positions), SiLU/GeLU, top-k sampling, masked argmax, and friends

On top of the kernels, the crate also ships a Rust-level multi-node layer (`NcclTransport` for AllReduce via cudarc's NCCL bindings, plus ring-attention orchestration over the attention kernels) — no custom PTX of its own. It is single-rank validated; multi-rank is experimental and not yet validated end to end.

These kernels are built to run a full Qwen3-Next-style **GDN-hybrid** assistant on a single DGX Spark — a model surface no off-the-shelf toolchain covers end to end on SM121. The point of the library is **complete, deterministic, driver-only coverage** of that surface on a chip the mainstream stack doesn't fully support — not out-benchmarking NVIDIA's mature attention/GEMM libraries.

## Why hand-written PTX for this chip?

SM120/SM121 sits in an awkward spot: it has TMA and the MMAv2 (`mma.sync.aligned`) tensor-core ISA, but **not** the tcgen05/TMEM/WGMMA path that FlashAttention-4, CUTLASS's flagship Blackwell kernels, and most "Blackwell" tooling target. As of June 2026: upstream FlashAttention-4 does not run on SM120 (ports exist only as open PRs), and NVIDIA's TensorRT-LLM FMHA cubins are SM100/SM103-only with no SM120/121 plan stated ([TensorRT-LLM #11799](https://github.com/NVIDIA/TensorRT-LLM/issues/11799)). First-party SM120 flash attention from CUTLASS and a fully stable Triton path on SM120 were both still in progress at that date (see [docs/cutlass_comparison.md](docs/cutlass_comparison.md) for the dated, linked status of each). Writing PTX directly against the MMAv2 + TMA ISA sidesteps the gap — and documents, kernel by kernel, what this hardware can actually do.

## Performance

These are single-kernel microbenchmarks — the project's value is the coverage and deployment story above, with **FP8 flash attention** as the standout result.

Flash-attention forward throughput on a DGX Spark (GB10, SM121a), D=128, non-causal, FLOP count `4·B·H·Sq·Skv·D` (two GEMMs at 2 FLOPs/MAC; see [docs/benchmark_methodology.md](docs/benchmark_methodology.md)). **Each row lists its own measured shape — do not compare rows at different shapes.**

| Kernel | Dtype | Shape (B,H,S) | TFLOPS | Reference |
|---|---|---|---|---|
| `fa_fp8_v12c_vt` (VT-GMEM) | FP8 e4m3 | 1, 32, 2048 | **~108** | vs cuDNN SDPA fprop on the same machine |
| `fa_bf16_v21_streaming_p` | BF16 | 1, 32, 2048 | ~35 | default BF16 kernel — occupancy-limited at this small shape |
| `fa_bf16_v21_streaming_p` | BF16 | 2, 32, 4096 | ~69 | same kernel, more work in flight |
| `fa_bf16_v21_streaming_p` | BF16 | 2, 32, 8192 | **~75** | long-context — see note |
| `fa_bf16_v22_db` | BF16 | 1, 32, 1024 | ~21 | best short-context BF16 (does not scale to long context) |
| CUTLASS 4.5 CuTe DSL reference | BF16 | 1, **16**, 2048 | 67.3 | CUTLASS's BF16 reference at its own small shape — see note below on why this is not a structural gap |

> Measured 2026-06 on GB10/SM121a with **random** inputs (the bundled `benchmark` example; reproduce the small shape with `SPARK_BENCH_B=1 SPARK_BENCH_H=32 SPARK_BENCH_S=2048 cargo run --release --example benchmark`, the long-context rows with `SPARK_BENCH_B=2 SPARK_BENCH_H=32 SPARK_BENCH_S=8192 ...`). The CUTLASS reference was measured at H=16 (not H=32) — see [docs/cutlass_comparison.md](docs/cutlass_comparison.md) for its exact config and commit. Numbers are `~` because GB10 does not expose clock locking; see [docs/benchmark_methodology.md](docs/benchmark_methodology.md).
>
> **On the BF16 gap:** V21 is occupancy-sensitive — its throughput climbs steeply with the amount of work in flight. At B=2, H=32 it measures ~48 TFLOPS at S=2048, ~69 at S=4096, and ~75 at S=8192; at the smaller default shapes (B=1, or H=16) it reads 12–35 TFLOPS because short sequences leave SMs idle. At long context (S=8192) the ~75 TFLOPS is within ~13% of upstream flash-attention's FA4 forward measured on this same GB10 (86.9 dense / 74.6 causal). The headline 67.3-vs-35 comparison is two kernels at *different* small shapes where ours is occupancy-starved, not evidence of a structural ceiling in hand-written PTX. We document the full scaling in [docs/optimization_journey.md](docs/optimization_journey.md).

To our knowledge the FP8 V12c kernel is the fastest open-source **exact** FP8-input flash-attention **forward** we have measured on GB10/SM121 at the shape above as of June 2026, versus cuDNN, flash-attention, and CUTLASS on the same hardware. Quantized-attention designs (e.g. SageAttention) and approximate methods can post higher numbers; baseline harnesses for cuDNN/flash-attention are not bundled in this repo, so the competitor side is not reproducible from this repo alone (their configs/versions are documented in [docs/benchmark_methodology.md](docs/benchmark_methodology.md)). Run `cargo run --release --example benchmark` to reproduce **our** side on your hardware.

Beyond attention, the BF16 GEMM kernels reach ~49 TFLOPS at 4096³, and the 128×256-tiled `gemm_bf16_mma_v5` reaches ~54–56; GEMM, GDN, MoE, and quantization throughput is tabulated in [docs/reproducer.md](docs/reproducer.md) and [docs/kernel_inventory.md](docs/kernel_inventory.md).

Determinism: GEMM/GEMV split-K reductions have deterministic-by-construction variants, selectable at runtime with `SPARK_DETERMINISTIC=1` — same inputs, bytewise-identical outputs, run to run.

## Quick start

Requirements: CUDA Toolkit 13.0+ (`ptxas` is used at build time), `cpp`, stable Rust, and an SM121/SM120 GPU to run anything. See [docs/BUILD.md](docs/BUILD.md).

```bash
cargo build --release                      # assembles all PTX -> SASS, embeds it
cargo run --release --example benchmark    # kernel benchmark suite
cargo run --release --example rust_api_demo

# tests need golden vectors first (~1.2 GB, generated locally with PyTorch):
pip install -r tests/reference/requirements.txt
python tests/reference/generate_golden.py
cargo test --release -- --test-threads=1
```

A C API is available behind `--features c-api` (`cbindgen`-generated header; see `examples/c_api_demo.c`), and a thin Python ctypes wrapper lives in `python/`.

## Docs

- [docs/kernel_inventory.md](docs/kernel_inventory.md) — the authoritative kernel list
- [docs/sm120_architecture_guide.md](docs/sm120_architecture_guide.md) — what SM120/SM121 has and doesn't have
- [docs/design.md](docs/design.md) / [docs/implementation-guide.md](docs/implementation-guide.md) — kernel design + build-it-yourself guide
- [docs/CORRECTNESS.md](docs/CORRECTNESS.md) — validation story and tolerances
- [docs/optimization_journey.md](docs/optimization_journey.md) — how the FA kernels got fast, including the failures
- [docs/cutlass_comparison.md](docs/cutlass_comparison.md), [docs/benchmark_methodology.md](docs/benchmark_methodology.md), [docs/reproducer.md](docs/reproducer.md)

## Related work

- [flash-attention](https://github.com/Dao-AILab/flash-attention) has merged SM120/SM121 forward/backward/varlen support (PRs #2329/#2330/#2333 — contributed by this project's author); paged-KV, split-KV, and TMA warp-specialized PRs are open, and third-party FA4-SM120 ports exist as open PRs.
- [Avarok's inference engine](https://github.com/Avarok-Cybersecurity/atlas) is a Rust inference engine for the DGX Spark with a large suite of custom **CUDA C++** SM121 kernels (AGPLv3). Different layer of the stack: it is an engine with FlashInfer-based attention; this is a permissively-licensed kernel library with hand-written PTX.
- [gau-nernst's fa-5090](https://gau-nernst.github.io/fa-5090/) demonstrated near-SOL BF16 attention on the RTX 5090 and is the spiritual ancestor of the PTX-level approach here.
- Kernel Hub hosts a small number of individual sm_121 kernels (elementwise); CUTLASS, FlashInfer, and Triton each have partial and evolving SM120/121 support — see the dated gap map in [docs/cutlass_comparison.md](docs/cutlass_comparison.md).

## License

Dual-licensed under either of [Apache License 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT), at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.

*NVIDIA, DGX, and DGX Spark are trademarks of NVIDIA Corporation. This project is an independent open-source effort and is not affiliated with or endorsed by NVIDIA.*
