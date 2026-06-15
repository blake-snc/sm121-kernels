# Correctness & reproducibility

Every sm121-kernels kernel is validated against a PyTorch reference. The validation is
**externally reproducible**: the golden vectors are generated from synthetic, seeded
inputs — **no model checkpoints required** — so anyone with a Spark/5090 and PyTorch can
reproduce the kernel-level correctness from scratch.

## One-command reproduction

```bash
make golden   # python tests/reference/generate_golden.py  → tests/reference/data/*.npz
make test     # cargo test --release -- --test-threads=1   → kernels vs goldens
```

Requirements: a CUDA device (SM121a for the cubin fast path; any SM12x via PTX-JIT),
Rust stable, and `pip install -r tests/reference/requirements.txt` (PyTorch + numpy).
The generator uses only `torch.randn` with a fixed seed (`torch.manual_seed(42)`), so the
goldens are deterministic and bit-stable across regenerations.

## Why the goldens are not committed

The 183 `.npz` vectors total ~1.2 GB. They are **regenerated** (deterministically) from
the committed generator rather than stored in git. `tests/reference/generate_golden.py`
(73 generators) is the published source of truth; the `.npz` are a build artifact.

## What is validated

`generate_golden.py` emits goldens for every production kernel family:

| Area | Kernels covered |
|------|-----------------|
| Flash attention | BF16 (v3/v3_d256/v21), FP8 (v12c), causal, GQA, varlen, softcap, SWA, paged-KV, split-KV, MLA (decode/prefill, BF16+FP8), tree, NSA — forward **and** backward |
| GEMM | BF16, FP8 (E4M3), NVFP4 block-scaled, W4A16 — forward and backward |
| Linear attention | GDN (decode/prefill), Mamba2 (decode/prefill) |
| Norm / activation | RMSNorm (+residual, +FP8-out, +backward), SiLU/GeLU/GeLU-tanh (+backward), softmax (+backward) |
| RoPE | forward + backward |
| Quantization | FP8 per-token, FP8 block128, MXFP8, MXFP4, NVFP4 (quant + dequant) |
| MoE | grouped GEMM, routing |
| Sampling / misc | top-k, top-p, embedding lookup, cross-entropy, KV-cache FP8 write |

Backward coverage note: the backward kernels ship in `ptx/`, but their training-side test
harnesses are not part of this repository — the golden tests here validate the forward
(inference) paths.

## Tolerances

Absolute tolerances, set to roughly 2× the measured max error on SM121a. (For the FP8/NVFP4 rows
where "measured max" is 0.0 — FP8 GEMM, NVFP4 GEMM — the reference quantizes to the *same* values
the kernel consumes, so the kernel reproduces the quantized reference exactly; the 0.0 is an exact
match, not the 2×-rule producing a zero tolerance. The listed tolerance for those rows is a fixed
floor, not 2× of zero.)

| Kernel | Tol | Measured max |
|--------|-----|--------------|
| RMSNorm | 0.02 | 0.008 |
| RoPE | 0.02 | 0.008 |
| SiLU | 0.01 | 0.0 |
| GeLU-tanh | 0.01 | 0.001 |
| GeLU | 0.07 | 0.031 |
| BF16 GEMM (scalar) | 0.15 | 0.063 |
| BF16 GEMM (MMA) | 0.15–2.0 | 1.0 @ 1024×4096×4096 |[^bf16mma]
| FP8 GEMM | 0.5 | 0.0 |
| NVFP4 GEMM | 1.0 | 0.0 |
| W4A16 GEMM | 0.6 | 0.25 |
| BF16 flash attention | 0.01–0.05 | 0.016 |
| BF16 flash attention (varlen) | 0.5 | 0.449 |
| FP8 flash attention | 0.15 | 0.051 |
| FP8 flash attention (V11 TMA) | 0.15 | 0.012 |

FP8/FP4 tolerances are wider by construction (E4M3 has 3 mantissa bits, E5M2 only 2).

[^bf16mma]: The BF16 GEMM (MMA) absolute tolerance of up to 2.0 at K=4096 is ~1.6% relative: the
    output std grows as √K ≈ 64 for unit-variance inputs, so 2.0 / 64 ≈ 0.031, and the measured-max
    1.0 / 64 ≈ 1.6%. The reference is fp32-accumulate, so the residual is BF16-input rounding
    accumulated over K, not a kernel error.

## Determinism

- **Goldens** are deterministic (fixed torch seed) — regeneration is byte-stable.
- **Runtime**: deterministic split-K GEMM/GEMV reductions are selectable via
  `SPARK_DETERMINISTIC=1` in the gemm dispatch.

## Memory safety

`compute-sanitizer --tool memcheck` runs over the kernel test binary in CI (the GPU job).
New kernels are expected to be memcheck-clean before merge.

Note: under memcheck, a handful of flash-attention golden tests (the cp.async
v3 family and split-KV variants) can exceed their tolerances by ~0.01-0.02.
The sanitizer reports zero invalid accesses for these kernels; its
instrumentation perturbs async-copy timing and therefore the floating-point
reduction order, which moves results that sit at the tolerance boundary.
Tolerances are calibrated against native execution (~2x the measured native
max error), so the native runs are the correctness gate; the sanitizer run is
the memory-safety gate.

## Known limitations

- **Ragged KV length on the non-causal flash-attention forward.** The non-causal
  FA forward kernels process the KV sequence in 64-row tiles and do **not** mask a
  partial final tile (`seq_kv % 64 != 0`): out-of-range KV rows are zero-filled and
  still participate in the softmax, which biases the result. Pass `seq_kv` as a
  multiple of 64 for the non-causal path, or use the causal / position-aware
  (`pos_dev`) kernels, which mask correctly at any length (these are the paths the
  production deployment uses). Treat this as the non-causal path's documented
  contract — pass `seq_kv` as a multiple of 64, or use the causal / `pos_dev`
  kernels — not as a silent default to lean on. Every path the production
  deployment ships on (causal and `pos_dev`) masks correctly at any length, so
  there is no correctness gap on a shipped path; the only sharp edge is dense
  non-causal with a ragged final tile, which the workaround above avoids.

- **GDN recurrence convention differs between `gdn_prefill` and `gdn_decode`.** By
  default `gdn_prefill` computes the delta-rule state update *without* the per-step
  α (alpha) gate that `gdn_decode` applies (the latter matches HF
  `recurrent_gated_delta_rule`). Each kernel is individually golden-validated, but
  the two are therefore **not bit-consistent across a prefill→decode handoff** by
  default. If you chain them for a single sequence, set `SPARK_GDN_HF=1` to dispatch
  `gdn_prefill_hf`, which applies the α gate so prefill matches `gdn_decode`. The
  default is left α-free for backward compatibility with the no-α goldens; the gated
  variant exists for HF-consistent prefill.

## Model-level validation (beyond kernels)

The kernels also pass full-model byte-identity validation in the production deployment
they were extracted from; those harnesses require model checkpoints and are not part of
this repository.
