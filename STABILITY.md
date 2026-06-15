# API Stability and Versioning

## Target platform

- **GPU:** NVIDIA SM121a — DGX Spark GB10 / Blackwell consumer (compute capability 12.1) —
  is the optimized fast path (precompiled SASS cubins, no JIT). Other SM12x GeForce
  Blackwell parts are supported via forward-compatibility: each kernel also embeds its
  preprocessed PTX, which the driver JIT-compiles when the sm_121a cubin is rejected.
- **MSRV:** current stable Rust (see `rust-toolchain.toml`).
- **Runtime:** CUDA driver only (`libcuda.so`). Kernels are compiled to SASS at
  build time, so no CUDA toolkit is required to run the published crate.

## Stable surface (semver-tracked)

The public, semver-tracked API is the kernel dispatch functions re-exported from
these modules, together with their parameter structs and shared types:

- `activation`, `attention`, `distributed` (shipped; experimental stability),
  `embedding`, `gemm`, `kv_cache`, `linear_attention`, `moe`, `norm`,
  `quantization`, `rope`, `sampling`
- `error::SparkError` and `error::Result`
- `device` and `module` (small public infrastructure: device init, capability
  check, kernel module loading)

Every item on this surface carries `///` documentation and is enforced by
`#![deny(missing_docs)]` in `crates/sm121-kernels/src/lib.rs`.

## Not stable / excluded

- Anything behind the `experimental` cargo feature. These are superseded kernel
  generations kept for comparison and may change or be removed in any release.
- The production model/serving layers this library was extracted from are not
  part of this repo. The `ffi` C-API surface (gated behind `c-api`) is exempt
  from the docs gate via `#[allow(missing_docs)]` and has no stability guarantee.

## Versioning

Pre-1.0:

- **Minor** versions (`0.x.0`) may make breaking changes to the public API.
- **Patch** versions (`0.x.y`) are additive or bug-fix only.

A `cargo public-api` snapshot guards the public surface against accidental
changes. The API baseline (`public-api.txt`) is committed on the first green
CI run of the public repo; see the `api-snapshot` / `api-check` targets in the
`Makefile` and the (non-blocking) `public-api` job in
`.github/workflows/ci.yml`. Any diff against the baseline should be reviewed
against the rules above and the baseline regenerated deliberately.

## Correctness contract

Kernels are validated against PyTorch golden vectors in `tests/reference/` at
documented absolute tolerances. The tolerance table lives in
`docs/CORRECTNESS.md`; the authoritative kernel list, codegen status, and
per-kernel performance/correctness notes are in `docs/kernel_inventory.md`.
