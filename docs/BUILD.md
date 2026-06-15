# Building sm121-kernels

## Requirements

| Tool | Version | Why |
|---|---|---|
| CUDA Toolkit | 13.0+ | `ptxas` assembles the PTX to SASS at **build** time (`sm_121a` needs 13.x); the toolkit is NOT needed at runtime |
| NVIDIA driver | r580+ | runtime needs only `libcuda.so` |
| `cpp` (GNU cpp) | any recent | resolves `#include`/macros in `.ptx`/`.ptxh` (PTX has no native include) |
| Rust | stable | see `rust-toolchain.toml` |
| GPU | SM121 (DGX Spark / GB10) or SM120 (RTX 50-series) | required to RUN anything; build works without a GPU |
| Python + PyTorch | 3.10+ | only for generating test golden vectors |

## Build

```bash
cargo build --release                       # assembles every ptx/**/*.ptx -> cubin, embeds via include_bytes!
cargo build --release --features c-api      # + extern "C" surface (header via cbindgen)
cargo build --release --features experimental
```

`build.rs` auto-discovers PTX under `ptx/`, preprocesses with `cpp -P`, assembles with
`ptxas --gpu-name sm_121a -O3`, and generates the embedded-kernels table. If you edit only a
`.ptx` file and the build doesn't pick it up, `touch crates/sm121-kernels/build.rs`.

## Tests

All integration tests require an SM121/SM120 GPU, and most compare against golden vectors that
are generated locally (they are not committed — ~1.2 GB):

```bash
pip install -r tests/reference/requirements.txt
python tests/reference/generate_golden.py        # ~1.2 GB into tests/reference/data/; a CUDA GPU IS required (tensors are allocated with device="cuda")
cargo test --release -- --test-threads=1         # GPU required, single-threaded (one CUDA context)
```

Sanitizer pass (recommended for kernel changes):

```bash
compute-sanitizer --tool memcheck cargo test --release -- --test-threads=1
```

## Benchmarks

```bash
cargo run --release --example benchmark
```

CUDA-event timing; per-kernel warmup/iteration counts are documented in
`docs/benchmark_methodology.md`.

## PTX development loop

```bash
ptxas --gpu-name sm_121a -O3 --warn-on-spills -o /dev/null ptx/<dir>/<kernel>.ptx   # syntax check
cuobjdump -sass /tmp/test.cubin                                                      # inspect SASS
scripts/gen_ptx_variants.sh && scripts/check_codegen_identity.sh                     # templated families
```

## Docker

```bash
docker build -t sm121-kernels .
docker run --gpus all sm121-kernels          # runs the benchmark suite
```

## DGX Spark note: page-cache fragmentation

On long-uptime DGX Spark systems the unified-memory page cache can fragment such that CUDA
context creation fails with `CUDA_ERROR_OUT_OF_MEMORY` despite tens of GB free. Fix:

```bash
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
```

The device-init error message in this library detects and explains this case.
