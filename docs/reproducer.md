# Reproducer recipe

How to reproduce the sm121-kernels kernel benchmark numbers from scratch.

## Hardware

You need a Blackwell GPU with compute capability **12.1 (SM121a)**. Validated on:
- **DGX Spark GB10** (unified LPDDR5X, 121.69 GiB shared with Grace CPU, 48 SMs)

The kernels target `sm_121a` specifically. A consumer RTX 50-series (SM120) host is **not** validated. Numbers in this doc are DGX Spark numbers.

## Software pinning

| Dependency | Pinned version |
|---|---|
| Base OS | Ubuntu 24.04 (sbsa for DGX Spark; amd64 untested) |
| NVIDIA driver | r580+ (must provide `libcuda.so` with CUDA 13.0 user-mode driver) |
| CUDA toolkit | 13.0 (`cuda-toolkit-13-0`; ptxas-13.0 is what produced the shipped cubins) |
| Rust toolchain | 1.93.1 stable |
| cudarc crate | 0.15 (driver-only loading, no cuRT/cuBLAS/cuDNN at run time) |
| Build profile | release with `lto = true, opt-level = 3, codegen-units = 1` (set in `Cargo.toml`) |

Comparison baselines cited in the docs were measured on the same machine with:

| Comparison-side dependency | Pinned version |
|---|---|
| vLLM | 0.18.1rc1.dev255+gd1678e6ad.d20260417 (editable install) |
| Triton | with PR #9572 merged upstream (`triton-lang/triton@6fe3ed795`) — no local patch needed for fresh installs |
| Python | 3.13.3 |
| Local act_quant_fusion patch | one-line `hasattr` guard on `silu_and_mul_per_block_quant`; only needed for the dev build's stale .so (mirrors existing `silu_and_mul_nvfp4_quant_supported` pattern) |

## Headline numbers (what to expect)

Kernel-level, single-shape, CUDA event timing (5 warm-up + 200 attention / 100
GEMM & sampling timed iterations, median; `~` because GB10 has no clock lock):

| Kernel | Result | Source |
|---|---|---|
| FP8 V12c flash attention | **~108 TFLOPS** at B=1, H=32, S=2048, D=128 | `cargo run --release --example benchmark` |
| BF16 flash attention v21 | **~75 TFLOPS** at B=2, H=32, S=8192 (long-context; ~35 at S=2048) | same |
| BF16 GEMM v3 | **~49 TFLOPS** at 4096³ | same |
| BF16 GEMM v5 (128×256) | **~54–56 TFLOPS** at 4096³ | same |

## Two reproducer paths

### A. From source (recommended for development)

```bash
git clone https://github.com/bledden/sm121-kernels.git
cd sm121-kernels

# Step 0 — DGX Spark page-cache drain (every run after sustained workloads):
#   the unified-memory page cache fills up and silently degrades CUDA init.
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

# Step 1 — build (~3 min release on Spark)
cargo build --release -p sm121-kernels

# Step 2 — kernel benchmarks (~2 min)
cargo run --release --example benchmark
```

### B. From Docker (recommended for one-shot reproduction)

```bash
# Build the image (~15 min cold cache; CUDA toolkit + Rust install dominate).
docker build -t sm121-kernels:latest .

# Run the kernel benchmark suite (the default CMD; no checkpoints needed).
docker run --rm --gpus all sm121-kernels:latest

# Banner-wrapped run with environment info (scripts/reproducer.sh):
docker run --rm --gpus all sm121-kernels:latest spark-reproducer

# Interactive shell for ad-hoc exploration:
docker run --rm --gpus all -it sm121-kernels:latest bash
```

The container ships the following binaries on `PATH`:

| Binary | Source | What it does |
|---|---|---|
| `spark-bench` | `examples/benchmark.rs` | All-kernel benchmark suite (CUDA event timing, TFLOPS table) |
| `spark-demo` | `examples/rust_api_demo.rs` | Minimal "hello kernel" smoke test |
| `spark-reproducer` | `scripts/reproducer.sh` | Wraps `spark-bench` with banners + environment info |

The runtime image **does not ship the CUDA toolkit** — only `libcuda1` (the user-mode driver). This is the sm121-kernels project's distinctive deployment story: all PTX → SASS compilation happens at build time inside the builder stage, the cubins are `include_bytes!`-embedded into the Rust binary, and the runtime stage loads them via cudarc's dynamic-loading mode against `libcuda.so` from the host driver.

## Verification checks

After running, the benchmark suite should print results matching the expected numbers above (±5% — the variation comes from system noise, page-cache state, and concurrent load).

If benchmark numbers are **significantly below expected** (e.g. >20% drop):
1. `sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'` and retry — the DGX Spark page cache is the most common culprit.
2. Confirm no other CUDA workload is co-tenant (`nvidia-smi`, `pgrep -af python`).

If the kernel benchmark suite **fails on a specific kernel**:
1. `compute-sanitizer --tool memcheck cargo run --release --example benchmark` to localize.
2. Verify your driver supports SM121 — `nvidia-smi --query-gpu=compute_cap` should report `12.1`.
3. Verify ptxas-13.0 is what got used at build time: in the builder stage, `ptxas --version` should report `13.0.88` or newer.

## What's deliberately NOT reproducible

- **Nsight Compute kernel-level metrics (TC %, mem %, occupancy)** — these need the kernel-module flag `NVreg_RestrictProfilingToAdminUsers=0` set on the host, which requires a reboot. Timing-only `nsys` profiles work without it.
- **Multi-node clustering** — requires a second DGX Spark; the `NcclTransport` machinery is in place but unvalidated end-to-end.
