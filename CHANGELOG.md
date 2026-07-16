# Changelog

## 0.2.0 (2026-07-15)

### Breaking

- The Mamba2 selective-scan kernels (`mamba2_selective_scan_prefill`,
  `mamba2_selective_scan_decode`) rename their `a_log` argument to `a`. The kernels
  have always expected the already negated and exponentiated decay rate
  `A = -exp(A_log)`; the old name invited passing raw `A_log`, which produces
  plausible but wrong numerics. Callers pass the same value as before under the
  new name. A negative test now feeds raw `A_log` and asserts the mismatch, and
  the golden vectors were regenerated under the new argument key.

### Changed

- FP8 flash-attention forward: the V^T scratch row stride is now auto-padded by
  512 elements whenever the natural stride is a multiple of 4096, avoiding a
  power-of-two DRAM channel alias that serialized one channel at S=4096
  (measured on GB10: 89.9 to 109.9 TFLOPS at that shape; other shapes
  unchanged). `QWEN_VT_PAD` overrides the pad explicitly; `QWEN_VT_PAD=0`
  restores the old layout. The V^T layout contract is documented at the
  dispatch boundary.

### Docs

- README wording cleanup.

## 0.1.0 (2026-06-17)

Initial public release.
