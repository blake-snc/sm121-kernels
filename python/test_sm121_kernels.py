"""End-to-end validation of sm121_kernels Python bindings.

Run via:
    python3 python/test_sm121_kernels.py

Or under pytest:
    pytest python/test_sm121_kernels.py -v

On DGX Spark, the CUDA-context creation can fail with OOM if buff/cache is
fragmented. If that happens, run `sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'`
and retry.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running both from repo root and from python/
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
import sm121_kernels as sk  # noqa: E402

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("[skip] torch not available; skipping CUDA tests")
    sys.exit(0)


def test_load():
    """Library + bindings import clean."""
    assert sk._lib is not None
    assert sk._lib_path is not None and sk._lib_path.exists()
    assert sk.__version__ == "0.1.0"


def test_ptr_helper_torch():
    """_ptr() extracts data_ptr from a CUDA torch tensor."""
    if not torch.cuda.is_available():
        print("[skip] no CUDA")
        return
    t = torch.zeros(16, dtype=torch.float32, device="cuda:0")
    p = sk._ptr(t)
    assert p == t.data_ptr()
    assert p != 0


def test_ptr_helper_none():
    """_ptr(None) returns 0 (for optional tensor params)."""
    assert sk._ptr(None) == 0


def test_ptr_helper_int():
    """_ptr(int) passthrough."""
    assert sk._ptr(0xdeadbeef) == 0xdeadbeef


def test_ptr_helper_rejects_cpu_tensor():
    """_ptr() rejects CPU tensors."""
    t = torch.zeros(16)  # CPU
    try:
        sk._ptr(t)
        raise AssertionError("expected ValueError for CPU tensor")
    except ValueError as e:
        assert "CUDA" in str(e)


def test_ptr_helper_rejects_non_contiguous():
    """_ptr() rejects non-contiguous tensors."""
    if not torch.cuda.is_available():
        print("[skip] no CUDA")
        return
    t = torch.zeros(16, 16, device="cuda:0").T  # transpose makes it non-contiguous
    try:
        sk._ptr(t)
        raise AssertionError("expected ValueError for non-contiguous tensor")
    except ValueError as e:
        assert "contiguous" in str(e)


def test_init_destroy():
    """Init + destroy round-trip on device 0."""
    if not torch.cuda.is_available():
        print("[skip] no CUDA")
        return
    try:
        ctx = sk.init(device=0)
    except sk.SparkError as e:
        if e.status == sk.SPARK_STATUS_ERROR_CUDA_LAUNCH:
            print("[skip] CUDA OOM at init — run `sudo drop_caches` on DGX Spark")
            return
        raise
    sk.synchronize(ctx)
    sk.destroy(ctx)


def test_flash_attention_smoke():
    """Flash attention BF16: shape preserved, no NaN, no Inf."""
    if not torch.cuda.is_available():
        print("[skip] no CUDA")
        return
    try:
        ctx = sk.init(device=0)
    except sk.SparkError as e:
        if e.status == sk.SPARK_STATUS_ERROR_CUDA_LAUNCH:
            print("[skip] CUDA OOM")
            return
        raise

    try:
        torch.manual_seed(0)
        b, h, s, d = 1, 8, 256, 128
        q = torch.randn(b, h, s, d, dtype=torch.bfloat16, device="cuda:0")
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        o = torch.empty_like(q)

        sk.flash_attention(ctx, q, k, v, o, scale=1.0 / d**0.5,
                           causal=False, dtype=sk.DTYPE_BF16)
        sk.synchronize(ctx)

        assert not torch.isnan(o).any(), "output has NaN"
        assert not torch.isinf(o).any(), "output has Inf"
        assert o.shape == q.shape

        # Sanity: output magnitude in reasonable range
        m = o.abs().mean().item()
        assert 0.01 < m < 100.0, f"output mean magnitude {m} outside sane range"
    finally:
        sk.destroy(ctx)


def test_gemm_bf16():
    """BF16 GEMM correctness test against torch reference.

    Uses 128×128×128 (smallest size satisfying gemm_bf16_mma's 128×64 tile
    requirement). Tolerance is loose (5.0) to allow for BF16 truncation +
    summation order differences vs FP32 reference.
    """
    if not torch.cuda.is_available():
        print("[skip] no CUDA")
        return
    try:
        ctx = sk.init(device=0)
    except sk.SparkError as e:
        if e.status == sk.SPARK_STATUS_ERROR_CUDA_LAUNCH:
            print("[skip] CUDA OOM")
            return
        raise

    try:
        torch.manual_seed(0)
        m, n, k = 128, 128, 128
        a = torch.randn(m, k, dtype=torch.bfloat16, device="cuda:0")
        b = torch.randn(k, n, dtype=torch.bfloat16, device="cuda:0")
        c = torch.zeros(m, n, dtype=torch.bfloat16, device="cuda:0")

        sk.gemm(ctx, a, b, c, m, n, k, dtype=sk.DTYPE_BF16)
        sk.synchronize(ctx)

        assert not torch.isnan(c).any(), "output has NaN"
        assert not torch.isinf(c).any(), "output has Inf"

        ref = a.float() @ b.float()
        diff = (c.float() - ref).abs()
        max_d = diff.max().item()
        # All elements should match torch's reference within BF16 tolerance.
        assert max_d < 5.0, f"max abs diff {max_d:.4f} too large vs torch reference"

        # Check that we're not dropping output: at least 95% of c should be
        # non-zero (BF16 randn product is rarely exactly zero).
        nonzero_frac = (c != 0).float().mean().item()
        assert nonzero_frac > 0.95, f"only {nonzero_frac:.1%} of output non-zero"
    finally:
        sk.destroy(ctx)


if __name__ == "__main__":
    print("=== Running sm121_kernels Python tests ===\n")
    tests = [t for t in globals() if t.startswith("test_")]
    passed, failed, skipped = 0, 0, 0
    for tname in tests:
        t = globals()[tname]
        print(f"  {tname} ... ", end="", flush=True)
        try:
            t()
            print("ok")
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed, of {len(tests)} tests")
    sys.exit(0 if failed == 0 else 1)
