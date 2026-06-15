# Contributing to sm121-kernels

Thanks for your interest. sm121-kernels is a hand-written PTX kernel library for SM121
(DGX Spark GB10 / Blackwell consumer). Contributions are welcome — especially new
kernels, perf improvements, and broader correctness coverage.

## Build & test

```bash
make verify     # no-GPU gate: PTX cross-assembly + codegen identity + build + clippy + docs
make golden     # regenerate golden vectors (PyTorch, deterministic seed=42) — needs CUDA
make test       # kernel correctness vs goldens — needs an SM121a (or any SM12x) device
make ci-gpu     # golden + test + compute-sanitizer + benchmark
```

`make verify` needs only CUDA Toolkit 13.0+ (`ptxas`), `cpp`, and Rust stable — no GPU.
It mirrors the no-GPU CI job, so run it before opening a PR. See
[docs/CORRECTNESS.md](docs/CORRECTNESS.md) for the full reproduction story and
[STABILITY.md](STABILITY.md) for the API/versioning policy.

## What every change must clear

- **Correctness, gated.** New/changed kernels are validated against a PyTorch golden
  vector at a documented tolerance (`tests/reference/generate_golden.py` +
  `crates/sm121-kernels/tests/`). Add a generator + a test for any new kernel.
- **Memory-safe.** `compute-sanitizer --tool memcheck` must be clean on the kernel's test.
- **No accidental API breakage.** The public surface is `#![deny(missing_docs)]` and
  semver-tracked (see STABILITY.md). Document new public items. Superseded kernel
  generations go behind the `experimental` cargo feature, not the default surface.
- **Codegen stays inert.** The collapsible FA families are generated from
  `ptx/attention/templates/*.ptx.in`; `make codegen-check` asserts the generated cubins
  are byte-identical to the archived hand-PTX (ptxas is deterministic). Edit the template,
  not a generated file, and keep the gate green.
- **Lint/format clean.** `cargo fmt --check` and `cargo clippy --release --all-targets -- -D warnings`.

## Kernel coding conventions

- One `.ptx` (or one template variant) per `(dtype, causal/paged/gqa/...)` combination —
  no runtime branching inside kernels. Prefer compile-time dispatch over runtime.
- Vectorized memory access (128-bit / `ld.global.v4`) on the hot paths.
- New features should preserve parity or provide a fallback — don't silently drop a
  capability behind an optimization.
- Parametrize tests over realistic shapes **including non-power-of-two** dimensions.
- See `ptx/common/*.ptxh` for the shared swizzle/reduction/MMA macros and
  `docs/sm120_architecture_guide.md` for the SM121 hardware constraints
  (MMAv2, 99 KB SMEM, no tcgen05/WGMMA).

## Pull requests

Use the PR template. A good PR description has:

- **Summary** — what changed and why (link the issue).
- **Context** — the root cause / motivation, with the literal error or measurement.
- **Changes** — `New:` / `Modified:` bullets.
- **Test plan** — checkboxes, and **honest about what was and wasn't run** (e.g. "memcheck
  clean; multi-day soak not run"). A truthful "not validated" beats an overstated claim.

Performance claims must be apples-to-apples (same dtype, same shapes, with a reproducer)
and qualified honestly — see `docs/benchmark_methodology.md`.

## License

Unless you explicitly state otherwise, any contribution intentionally submitted for
inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual
licensed as above, without any additional terms or conditions.
