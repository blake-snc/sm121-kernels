"""sm121-kernels Python bindings.

A high-performance GPU kernel library for SM121a (DGX Spark / RTX 5090 / Blackwell GeForce).
Hand-written PTX kernels compiled at build time, dispatched via ctypes.

Quick start:
    import torch
    import sm121_kernels as sk

    ctx = sk.init(device=0)

    q = torch.randn(1, 32, 2048, 128, dtype=torch.bfloat16, device='cuda')
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    o = torch.empty_like(q)
    sk.flash_attention(ctx, q, k, v, o, scale=1.0/128**0.5, causal=False)
    sk.synchronize(ctx)

    sk.destroy(ctx)

All tensors must be:
- CUDA tensors on device 0 (or matching device passed to init)
- Contiguous in memory
- 16-byte aligned (PyTorch defaults satisfy this)

Supported kernels: flash attention (BF16/FP8, causal/non-causal, varlen), GEMM
(BF16/FP8/NVFP4/W4A16), RMSNorm, RoPE, fused activations (SiLU/GeLU/GeLU-tanh
× mul), top-k sampling, MoE routing.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Library loading
# ---------------------------------------------------------------------------

_LIB_PATHS = [
    # Wheel install: bundled inside the sm121_kernels package
    Path(__file__).parent / "libsm121_kernels.so",
    # Editable install / dev: source-tree relative (python/sm121_kernels/__init__.py)
    Path(__file__).parent.parent.parent / "target" / "release" / "libsm121_kernels.so",
    # System install
    Path("/usr/local/lib/libsm121_kernels.so"),
]

# Allow override via env var
_env = os.environ.get("SPARK_KERNELS_LIB")
if _env:
    _LIB_PATHS.insert(0, Path(_env))

_lib: Optional[ctypes.CDLL] = None
_lib_path: Optional[Path] = None
for _p in _LIB_PATHS:
    if _p.exists():
        _lib = ctypes.CDLL(str(_p))
        _lib_path = _p
        break

if _lib is None:
    raise RuntimeError(
        "libsm121_kernels.so not found. Build with: cargo build --release --features c-api\n"
        f"Searched: {[str(p) for p in _LIB_PATHS]}\n"
        "Or set SPARK_KERNELS_LIB to the absolute path."
    )

# ---------------------------------------------------------------------------
# Enums + status
# ---------------------------------------------------------------------------

# SparkStatus
SPARK_STATUS_SUCCESS = 0
SPARK_STATUS_ERROR_INVALID_ARGUMENT = 1
SPARK_STATUS_ERROR_CUDA_LAUNCH = 2
SPARK_STATUS_ERROR_KERNEL_NOT_FOUND = 3
SPARK_STATUS_ERROR_INTERNAL = 4

_STATUS_NAMES = {
    0: "SUCCESS",
    1: "ERROR_INVALID_ARGUMENT",
    2: "ERROR_CUDA_LAUNCH",
    3: "ERROR_KERNEL_NOT_FOUND",
    4: "ERROR_INTERNAL",
}

# SparkDtype
DTYPE_BF16 = 0
DTYPE_FP8E4M3 = 1
DTYPE_FP32 = 2
DTYPE_U32 = 3
DTYPE_NVFP4 = 4
DTYPE_W4A16 = 5

# SparkActivationType
ACTIVATION_SILU_MUL = 0
ACTIVATION_GELU_MUL = 1
ACTIVATION_GELU_TANH_MUL = 2


class SparkError(RuntimeError):
    """Raised when a sm121-kernels call returns a non-success status."""
    def __init__(self, status: int, op: str):
        self.status = status
        self.op = op
        super().__init__(
            f"sm121-kernels {op} failed: {_STATUS_NAMES.get(status, f'status={status}')}"
        )


def _check(status: int, op: str) -> None:
    if status != SPARK_STATUS_SUCCESS:
        raise SparkError(status, op)


# ---------------------------------------------------------------------------
# Opaque context handle
# ---------------------------------------------------------------------------

class SparkCtx(ctypes.Structure):
    """Opaque CUDA context handle. Treat as opaque; do not inspect fields."""
    pass


SparkCtxPtr = ctypes.POINTER(SparkCtx)


# ---------------------------------------------------------------------------
# Parameter structs (mirror include/sm121_kernels.h)
# ---------------------------------------------------------------------------

class _FlashAttnParams(ctypes.Structure):
    _fields_ = [
        ("q",         ctypes.c_void_p),
        ("k",         ctypes.c_void_p),
        ("v",         ctypes.c_void_p),
        ("o",         ctypes.c_void_p),
        ("batch",     ctypes.c_uint),
        ("num_heads", ctypes.c_uint),
        ("seq_q",     ctypes.c_uint),
        ("seq_kv",    ctypes.c_uint),
        ("head_dim",  ctypes.c_uint),
        ("scale",     ctypes.c_float),
        ("dtype",     ctypes.c_int),
        ("causal",    ctypes.c_int),
    ]


class _GemmParams(ctypes.Structure):
    _fields_ = [
        ("a",     ctypes.c_void_p),
        ("b",     ctypes.c_void_p),
        ("c",     ctypes.c_void_p),
        ("m",     ctypes.c_uint),
        ("n",     ctypes.c_uint),
        ("k",     ctypes.c_uint),
        ("dtype", ctypes.c_int),
    ]


class _Nvfp4GemmParams(ctypes.Structure):
    _fields_ = [
        ("a",        ctypes.c_void_p),
        ("b",        ctypes.c_void_p),
        ("c",        ctypes.c_void_p),
        ("scale_a",  ctypes.c_void_p),
        ("scale_b",  ctypes.c_void_p),
        ("m",        ctypes.c_uint),
        ("n",        ctypes.c_uint),
        ("k",        ctypes.c_uint),
    ]


class _W4a16GemmParams(ctypes.Structure):
    _fields_ = [
        ("a",      ctypes.c_void_p),
        ("w",      ctypes.c_void_p),
        ("c",      ctypes.c_void_p),
        ("scales", ctypes.c_void_p),
        ("zeros",  ctypes.c_void_p),
        ("m",      ctypes.c_uint),
        ("n",      ctypes.c_uint),
        ("k",      ctypes.c_uint),
    ]


class _TopkParams(ctypes.Structure):
    _fields_ = [
        ("logits",      ctypes.c_void_p),
        ("indices",     ctypes.c_void_p),
        ("values",      ctypes.c_void_p),
        ("batch_size",  ctypes.c_uint),
        ("vocab_size",  ctypes.c_uint),
        ("k",           ctypes.c_uint),
        ("temperature", ctypes.c_float),
    ]


class _MoeRoutingParams(ctypes.Structure):
    _fields_ = [
        ("logits",      ctypes.c_void_p),
        ("expert_ids",  ctypes.c_void_p),
        ("weights",     ctypes.c_void_p),
        ("num_tokens",  ctypes.c_uint),
        ("num_experts", ctypes.c_uint),
        ("top_k",       ctypes.c_uint),
    ]


class _RmsNormParams(ctypes.Structure):
    _fields_ = [
        ("x",          ctypes.c_void_p),
        ("out",        ctypes.c_void_p),
        ("weight",     ctypes.c_void_p),
        ("hidden_dim", ctypes.c_uint),
        ("eps",        ctypes.c_float),
        ("num_rows",   ctypes.c_uint),
    ]


class _RopeParams(ctypes.Structure):
    _fields_ = [
        ("x",         ctypes.c_void_p),
        ("cos_cache", ctypes.c_void_p),
        ("sin_cache", ctypes.c_void_p),
        ("batch",     ctypes.c_uint),
        ("seq_len",   ctypes.c_uint),
        ("heads",     ctypes.c_uint),
        ("dim",       ctypes.c_uint),
    ]


class _ActivationParams(ctypes.Structure):
    _fields_ = [
        ("input",           ctypes.c_void_p),
        ("out",             ctypes.c_void_p),
        ("total_out_elems", ctypes.c_uint),
        ("d",               ctypes.c_uint),
        ("activation",      ctypes.c_int),
    ]


class _VarlenFlashAttnParams(ctypes.Structure):
    _fields_ = [
        ("q",             ctypes.c_void_p),
        ("k",             ctypes.c_void_p),
        ("v",             ctypes.c_void_p),
        ("o",             ctypes.c_void_p),
        ("cu_seqlens_q",  ctypes.c_void_p),
        ("cu_seqlens_k",  ctypes.c_void_p),
        ("batch",         ctypes.c_uint),
        ("num_heads",     ctypes.c_uint),
        ("max_seqlen_q",  ctypes.c_uint),
        ("scale",         ctypes.c_float),
    ]


# ---------------------------------------------------------------------------
# C function signatures
# ---------------------------------------------------------------------------

_lib.spark_init.argtypes = [ctypes.c_int, ctypes.POINTER(SparkCtxPtr)]
_lib.spark_init.restype = ctypes.c_int

_lib.spark_destroy.argtypes = [SparkCtxPtr]
_lib.spark_destroy.restype = ctypes.c_int

_lib.spark_synchronize.argtypes = [SparkCtxPtr]
_lib.spark_synchronize.restype = ctypes.c_int

_lib.spark_flash_attention.argtypes = [SparkCtxPtr, ctypes.POINTER(_FlashAttnParams)]
_lib.spark_flash_attention.restype = ctypes.c_int

_lib.spark_flash_attention_varlen.argtypes = [SparkCtxPtr, ctypes.POINTER(_VarlenFlashAttnParams)]
_lib.spark_flash_attention_varlen.restype = ctypes.c_int

_lib.spark_gemm.argtypes = [SparkCtxPtr, ctypes.POINTER(_GemmParams)]
_lib.spark_gemm.restype = ctypes.c_int

_lib.spark_gemm_nvfp4.argtypes = [SparkCtxPtr, ctypes.POINTER(_Nvfp4GemmParams)]
_lib.spark_gemm_nvfp4.restype = ctypes.c_int

_lib.spark_gemm_w4a16.argtypes = [SparkCtxPtr, ctypes.POINTER(_W4a16GemmParams)]
_lib.spark_gemm_w4a16.restype = ctypes.c_int

_lib.spark_topk_sampling.argtypes = [SparkCtxPtr, ctypes.POINTER(_TopkParams)]
_lib.spark_topk_sampling.restype = ctypes.c_int

_lib.spark_moe_routing.argtypes = [SparkCtxPtr, ctypes.POINTER(_MoeRoutingParams)]
_lib.spark_moe_routing.restype = ctypes.c_int

_lib.spark_rmsnorm.argtypes = [SparkCtxPtr, ctypes.POINTER(_RmsNormParams)]
_lib.spark_rmsnorm.restype = ctypes.c_int

_lib.spark_rope.argtypes = [SparkCtxPtr, ctypes.POINTER(_RopeParams)]
_lib.spark_rope.restype = ctypes.c_int

_lib.spark_activation.argtypes = [SparkCtxPtr, ctypes.POINTER(_ActivationParams)]
_lib.spark_activation.restype = ctypes.c_int


# ---------------------------------------------------------------------------
# Tensor pointer extraction
# ---------------------------------------------------------------------------

def _ptr(t) -> int:
    """Extract device pointer from a torch.Tensor or numpy GPU array.

    Returns 0 for None (used for optional tensors).
    """
    if t is None:
        return 0
    # torch.Tensor
    if hasattr(t, "data_ptr") and hasattr(t, "is_cuda"):
        if not t.is_cuda:
            raise ValueError(f"tensor must be on CUDA device, got {t.device}")
        if not t.is_contiguous():
            raise ValueError("tensor must be contiguous; call .contiguous() first")
        return t.data_ptr()
    # cupy / numpy GPU array — has __cuda_array_interface__
    if hasattr(t, "__cuda_array_interface__"):
        return t.__cuda_array_interface__["data"][0]
    # Raw integer pointer (advanced)
    if isinstance(t, int):
        return t
    raise TypeError(
        f"expected torch.Tensor, cuda array, or int pointer, got {type(t).__name__}"
    )


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

def init(device: int = 0) -> SparkCtxPtr:
    """Initialize sm121-kernels context on the given CUDA device.

    Returns an opaque handle that must be passed to all kernel calls and
    eventually freed with destroy(ctx).

    Raises SparkError on init failure (e.g. non-SM121 device, OOM).
    """
    ctx = SparkCtxPtr()
    _check(_lib.spark_init(device, ctypes.byref(ctx)), "spark_init")
    return ctx


def destroy(ctx: SparkCtxPtr) -> None:
    """Destroy a sm121-kernels context."""
    _check(_lib.spark_destroy(ctx), "spark_destroy")


def synchronize(ctx: SparkCtxPtr) -> None:
    """Wait for all pending operations on the context's CUDA stream."""
    _check(_lib.spark_synchronize(ctx), "spark_synchronize")


# ---------------------------------------------------------------------------
# Kernel wrappers
# ---------------------------------------------------------------------------

def flash_attention(
    ctx: SparkCtxPtr,
    q,
    k,
    v,
    o,
    scale: float,
    *,
    causal: bool = False,
    dtype: int = DTYPE_BF16,
) -> None:
    """Flash attention forward: O = softmax(QK^T * scale) V.

    Q, K, V, O shape: [batch, num_heads, seq, 128]. Only head_dim=128 supported.
    The C API dispatches the baseline forward kernels:
      BF16 path (dtype=DTYPE_BF16)     -> fa_bf16_v3_d128
      FP8 path  (dtype=DTYPE_FP8E4M3)  -> fa_fp8_d128
    The faster V21 (BF16) and V12c VT-GMEM (FP8) variants are exercised by the
    Rust benchmark suite (`cargo run --release --example benchmark`), not this
    C-API entry point.
    """
    # Infer shapes from tensors (assume torch convention [B, H, S, D])
    if hasattr(q, "shape"):
        b, h, sq, d = q.shape
        sk_ = k.shape[2]
    else:
        raise ValueError("q must be a tensor with .shape attribute")

    p = _FlashAttnParams(
        q=_ptr(q), k=_ptr(k), v=_ptr(v), o=_ptr(o),
        batch=b, num_heads=h, seq_q=sq, seq_kv=sk_,
        head_dim=d, scale=scale, dtype=dtype, causal=int(bool(causal)),
    )
    _check(_lib.spark_flash_attention(ctx, ctypes.byref(p)), "spark_flash_attention")


def flash_attention_varlen(
    ctx: SparkCtxPtr,
    q, k, v, o,
    cu_seqlens_q,
    cu_seqlens_k,
    batch: int,
    num_heads: int,
    max_seqlen_q: int,
    scale: float,
) -> None:
    """Variable-length BF16 Flash Attention (non-causal).

    Q, K, V, O are packed: shape [total_tokens, num_heads, 128].
    cu_seqlens_q / cu_seqlens_k are i32 prefix-sums of per-sequence lengths.
    """
    p = _VarlenFlashAttnParams(
        q=_ptr(q), k=_ptr(k), v=_ptr(v), o=_ptr(o),
        cu_seqlens_q=_ptr(cu_seqlens_q), cu_seqlens_k=_ptr(cu_seqlens_k),
        batch=batch, num_heads=num_heads, max_seqlen_q=max_seqlen_q, scale=scale,
    )
    _check(_lib.spark_flash_attention_varlen(ctx, ctypes.byref(p)),
           "spark_flash_attention_varlen")


def gemm(
    ctx: SparkCtxPtr,
    a, b, c,
    m: int, n: int, k: int,
    dtype: int = DTYPE_BF16,
) -> None:
    """GEMM: C = A * B.

    A: [M, K], B: [K, N], C: [M, N], all row-major.
    BF16 (dtype=DTYPE_BF16): MMA m16n8k16, ~42 TFLOPS at 4096³.
    FP8 (dtype=DTYPE_FP8E4M3): MMA m16n8k32, ~65 TFLOPS at 2048³.
    """
    p = _GemmParams(a=_ptr(a), b=_ptr(b), c=_ptr(c), m=m, n=n, k=k, dtype=dtype)
    _check(_lib.spark_gemm(ctx, ctypes.byref(p)), "spark_gemm")


def gemm_nvfp4(
    ctx: SparkCtxPtr,
    a, b, c,
    scale_a, scale_b,
    m: int, n: int, k: int,
) -> None:
    """NVFP4 block-scaled GEMM. A/B are NVFP4-packed bytes; scale_a/b are UE8M0."""
    p = _Nvfp4GemmParams(
        a=_ptr(a), b=_ptr(b), c=_ptr(c),
        scale_a=_ptr(scale_a), scale_b=_ptr(scale_b),
        m=m, n=n, k=k,
    )
    _check(_lib.spark_gemm_nvfp4(ctx, ctypes.byref(p)), "spark_gemm_nvfp4")


def gemm_w4a16(
    ctx: SparkCtxPtr,
    a, w, c,
    scales, zeros,
    m: int, n: int, k: int,
) -> None:
    """W4A16 dequant GEMM. A is BF16, w is INT4 packed, scales/zeros are BF16."""
    p = _W4a16GemmParams(
        a=_ptr(a), w=_ptr(w), c=_ptr(c),
        scales=_ptr(scales), zeros=_ptr(zeros),
        m=m, n=n, k=k,
    )
    _check(_lib.spark_gemm_w4a16(ctx, ctypes.byref(p)), "spark_gemm_w4a16")


def rmsnorm(
    ctx: SparkCtxPtr,
    x, out, weight,
    hidden_dim: int,
    num_rows: int,
    eps: float = 1.0e-6,
) -> None:
    """RMSNorm: out = x * rsqrt(mean(x^2) + eps) * weight."""
    p = _RmsNormParams(
        x=_ptr(x), out=_ptr(out), weight=_ptr(weight),
        hidden_dim=hidden_dim, eps=eps, num_rows=num_rows,
    )
    _check(_lib.spark_rmsnorm(ctx, ctypes.byref(p)), "spark_rmsnorm")


def rope(
    ctx: SparkCtxPtr,
    x,
    cos_cache,
    sin_cache,
    batch: int,
    seq_len: int,
    heads: int,
    dim: int,
) -> None:
    """RoPE (Rotary Position Embedding) — applied in-place to x."""
    p = _RopeParams(
        x=_ptr(x), cos_cache=_ptr(cos_cache), sin_cache=_ptr(sin_cache),
        batch=batch, seq_len=seq_len, heads=heads, dim=dim,
    )
    _check(_lib.spark_rope(ctx, ctypes.byref(p)), "spark_rope")


def activation(
    ctx: SparkCtxPtr,
    inp, out,
    total_out_elems: int,
    d: int,
    activation_type: int = ACTIVATION_SILU_MUL,
) -> None:
    """Fused activation: out = act(input[..., :d]) * input[..., d:]."""
    p = _ActivationParams(
        input=_ptr(inp), out=_ptr(out),
        total_out_elems=total_out_elems, d=d,
        activation=activation_type,
    )
    _check(_lib.spark_activation(ctx, ctypes.byref(p)), "spark_activation")


def topk_sampling(
    ctx: SparkCtxPtr,
    logits, indices, values,
    batch_size: int,
    vocab_size: int,
    k: int,
    temperature: float = 1.0,
) -> None:
    """Top-k sampling: select top-k values + indices per batch row."""
    p = _TopkParams(
        logits=_ptr(logits), indices=_ptr(indices), values=_ptr(values),
        batch_size=batch_size, vocab_size=vocab_size, k=k,
        temperature=temperature,
    )
    _check(_lib.spark_topk_sampling(ctx, ctypes.byref(p)), "spark_topk_sampling")


def moe_routing(
    ctx: SparkCtxPtr,
    logits, expert_ids, weights,
    num_tokens: int,
    num_experts: int,
    top_k: int,
) -> None:
    """MoE expert routing: logits → top-k expert IDs + softmax-normalized weights."""
    p = _MoeRoutingParams(
        logits=_ptr(logits), expert_ids=_ptr(expert_ids), weights=_ptr(weights),
        num_tokens=num_tokens, num_experts=num_experts, top_k=top_k,
    )
    _check(_lib.spark_moe_routing(ctx, ctypes.byref(p)), "spark_moe_routing")


# ---------------------------------------------------------------------------
# Info helper
# ---------------------------------------------------------------------------

__version__ = "0.1.0"


def info() -> None:
    """Print library info."""
    print(f"sm121-kernels {__version__} (Python bindings)")
    print(f"  Library: {_lib_path}")
    print(f"  Platform: SM121a (DGX Spark / RTX 5090 / Blackwell GeForce)")
    print(f"  Kernels: 13 C-API functions (attention, GEMM x4 dtypes, RMSNorm,")
    print(f"           RoPE, activation x3, top-k, MoE routing)")
    print(f"  Note: the C API dispatches the baseline forward kernels")
    print(f"        (fa_bf16_v3_d128 / fa_fp8_d128); the library's fastest")
    print(f"        variants (V21, V12c) are measured by the Rust benchmark suite.")


if __name__ == "__main__":
    info()
    print()
    try:
        ctx = init()
        print("✓ Init OK")
        synchronize(ctx)
        print("✓ Sync OK")
        destroy(ctx)
        print("✓ Destroy OK")
    except SparkError as e:
        print(f"✗ {e}")
        if e.status == SPARK_STATUS_ERROR_CUDA_LAUNCH:
            print("  Hint: on DGX Spark, run `sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'`")
            print("        to clear page-cache fragmentation, then retry.")
