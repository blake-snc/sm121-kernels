//! C API for sm121-kernels.
//!
//! Provides `extern "C"` functions for use from C, Python (via ctypes/cffi), or other languages.
//! All device memory pointers are passed as raw `*const u8` / `*mut u8` (cast from CUdeviceptr).
//!
//! # Safety
//!
//! All functions in this module are `unsafe extern "C"`. Callers must ensure:
//! - Device memory pointers are valid CUdeviceptr values for the correct device
//! - Buffer sizes match the declared dimensions
//! - The SparkCtx handle was obtained from `spark_init` and has not been destroyed
#![allow(clippy::missing_safety_doc)]

pub mod types;
pub use types::*;

use std::os::raw::c_int;

use cudarc::driver::{CudaStream, LaunchConfig, PushKernelArg};

use crate::module;

/// Internal context holding CUDA resources.
pub struct SparkCtx {
    ctx: std::sync::Arc<cudarc::driver::CudaContext>,
    stream: std::sync::Arc<CudaStream>,
}

fn error_to_status(e: crate::SparkError) -> SparkStatus {
    match e {
        crate::SparkError::InvalidArgument(_) => SparkStatus::ErrorInvalidArgument,
        crate::SparkError::KernelNotFound(_) => SparkStatus::ErrorKernelNotFound,
        crate::SparkError::Driver(_) | crate::SparkError::LaunchFailed(_) => {
            SparkStatus::ErrorCudaLaunch
        }
        crate::SparkError::UnsupportedArch { .. } => SparkStatus::ErrorInvalidArgument,
        crate::SparkError::Io(_) | crate::SparkError::Other(_) => SparkStatus::ErrorInternal,
    }
}

/// Initialize a sm121-kernels context on the given CUDA device.
///
/// Returns `SparkStatus::Success` and writes a valid handle to `*out`.
/// The handle must be freed with `spark_destroy`.
#[no_mangle]
pub unsafe extern "C" fn spark_init(device: c_int, out: *mut *mut SparkCtx) -> SparkStatus {
    if out.is_null() {
        return SparkStatus::ErrorInvalidArgument;
    }
    match crate::device::init_device(device as usize) {
        Ok(ctx) => {
            let stream = ctx.default_stream();
            let handle = Box::new(SparkCtx { ctx, stream });
            *out = Box::into_raw(handle);
            SparkStatus::Success
        }
        Err(e) => error_to_status(e),
    }
}

/// Destroy a sm121-kernels context and free associated resources.
#[no_mangle]
pub unsafe extern "C" fn spark_destroy(handle: *mut SparkCtx) -> SparkStatus {
    if handle.is_null() {
        return SparkStatus::ErrorInvalidArgument;
    }
    drop(Box::from_raw(handle));
    SparkStatus::Success
}

/// Launch flash attention.
///
/// Dispatches based on `params.dtype` and `params.causal`.
/// BF16 uses V3 (8-warp, Br=128). FP8 uses 1-warp (Br=16).
/// Only `head_dim=128` is supported.
#[no_mangle]
pub unsafe extern "C" fn spark_flash_attention(
    handle: *mut SparkCtx,
    params: *const SparkFlashAttnParams,
) -> SparkStatus {
    if handle.is_null() || params.is_null() {
        return SparkStatus::ErrorInvalidArgument;
    }
    let h = &*handle;
    let p = &*params;

    if p.head_dim != 128 {
        return SparkStatus::ErrorInvalidArgument;
    }

    let result = match (p.dtype, p.causal != 0) {
        (SparkDtype::BF16, false) => launch_fa_bf16_v3(h, p),
        (SparkDtype::BF16, true) => launch_fa_bf16_v3_causal(h, p),
        (SparkDtype::FP8E4M3, false) => launch_fa_fp8(h, p),
        (SparkDtype::FP8E4M3, true) => launch_fa_fp8_causal(h, p),
        _ => return SparkStatus::ErrorInvalidArgument,
    };

    match result {
        Ok(()) => SparkStatus::Success,
        Err(e) => error_to_status(e),
    }
}

/// Launch GEMM: C = A x B.
///
/// Dispatches based on `params.dtype`. BF16 and FP8 use MMA variants.
#[no_mangle]
pub unsafe extern "C" fn spark_gemm(
    handle: *mut SparkCtx,
    params: *const SparkGemmParams,
) -> SparkStatus {
    if handle.is_null() || params.is_null() {
        return SparkStatus::ErrorInvalidArgument;
    }
    let h = &*handle;
    let p = &*params;

    let result = match p.dtype {
        SparkDtype::BF16 => launch_gemm_bf16_mma(h, p),
        SparkDtype::FP8E4M3 => launch_gemm_fp8_mma(h, p),
        _ => return SparkStatus::ErrorInvalidArgument,
    };

    match result {
        Ok(()) => SparkStatus::Success,
        Err(e) => error_to_status(e),
    }
}

/// Launch top-k sampling.
#[no_mangle]
pub unsafe extern "C" fn spark_topk_sampling(
    handle: *mut SparkCtx,
    params: *const SparkTopkParams,
) -> SparkStatus {
    if handle.is_null() || params.is_null() {
        return SparkStatus::ErrorInvalidArgument;
    }
    let h = &*handle;
    let p = &*params;

    match launch_topk(h, p) {
        Ok(()) => SparkStatus::Success,
        Err(e) => error_to_status(e),
    }
}

/// Launch MoE expert routing.
#[no_mangle]
pub unsafe extern "C" fn spark_moe_routing(
    handle: *mut SparkCtx,
    params: *const SparkMoeRoutingParams,
) -> SparkStatus {
    if handle.is_null() || params.is_null() {
        return SparkStatus::ErrorInvalidArgument;
    }
    let h = &*handle;
    let p = &*params;

    match launch_moe(h, p) {
        Ok(()) => SparkStatus::Success,
        Err(e) => error_to_status(e),
    }
}

/// Launch RMSNorm.
#[no_mangle]
pub unsafe extern "C" fn spark_rmsnorm(
    handle: *mut SparkCtx,
    params: *const SparkRmsNormParams,
) -> SparkStatus {
    if handle.is_null() || params.is_null() {
        return SparkStatus::ErrorInvalidArgument;
    }
    let h = &*handle;
    let p = &*params;

    match launch_rmsnorm(h, p) {
        Ok(()) => SparkStatus::Success,
        Err(e) => error_to_status(e),
    }
}

/// Launch RoPE (in-place).
#[no_mangle]
pub unsafe extern "C" fn spark_rope(
    handle: *mut SparkCtx,
    params: *const SparkRopeParams,
) -> SparkStatus {
    if handle.is_null() || params.is_null() {
        return SparkStatus::ErrorInvalidArgument;
    }
    let h = &*handle;
    let p = &*params;

    match launch_rope(h, p) {
        Ok(()) => SparkStatus::Success,
        Err(e) => error_to_status(e),
    }
}

/// Launch fused activation (SiLU*mul, GeLU*mul, or GeLU-tanh*mul).
#[no_mangle]
pub unsafe extern "C" fn spark_activation(
    handle: *mut SparkCtx,
    params: *const SparkActivationParams,
) -> SparkStatus {
    if handle.is_null() || params.is_null() {
        return SparkStatus::ErrorInvalidArgument;
    }
    let h = &*handle;
    let p = &*params;

    match launch_activation(h, p) {
        Ok(()) => SparkStatus::Success,
        Err(e) => error_to_status(e),
    }
}

/// Synchronize the device stream (wait for all pending operations).
#[no_mangle]
pub unsafe extern "C" fn spark_synchronize(handle: *mut SparkCtx) -> SparkStatus {
    if handle.is_null() {
        return SparkStatus::ErrorInvalidArgument;
    }
    let h = &*handle;
    match h.stream.synchronize() {
        Ok(()) => SparkStatus::Success,
        Err(e) => error_to_status(crate::SparkError::Driver(e)),
    }
}

/// Launch NVFP4 block-scaled GEMM.
#[no_mangle]
pub unsafe extern "C" fn spark_gemm_nvfp4(
    handle: *mut SparkCtx,
    params: *const SparkNvfp4GemmParams,
) -> SparkStatus {
    if handle.is_null() || params.is_null() {
        return SparkStatus::ErrorInvalidArgument;
    }
    let h = &*handle;
    let p = &*params;

    match launch_gemm_nvfp4(h, p) {
        Ok(()) => SparkStatus::Success,
        Err(e) => error_to_status(e),
    }
}

/// Launch W4A16 dequant GEMM.
#[no_mangle]
pub unsafe extern "C" fn spark_gemm_w4a16(
    handle: *mut SparkCtx,
    params: *const SparkW4a16GemmParams,
) -> SparkStatus {
    if handle.is_null() || params.is_null() {
        return SparkStatus::ErrorInvalidArgument;
    }
    let h = &*handle;
    let p = &*params;

    match launch_gemm_w4a16(h, p) {
        Ok(()) => SparkStatus::Success,
        Err(e) => error_to_status(e),
    }
}

/// Launch variable-length BF16 Flash Attention (non-causal).
#[no_mangle]
pub unsafe extern "C" fn spark_flash_attention_varlen(
    handle: *mut SparkCtx,
    params: *const SparkVarlenFlashAttnParams,
) -> SparkStatus {
    if handle.is_null() || params.is_null() {
        return SparkStatus::ErrorInvalidArgument;
    }
    let h = &*handle;
    let p = &*params;

    match launch_fa_bf16_varlen(h, p) {
        Ok(()) => SparkStatus::Success,
        Err(e) => error_to_status(e),
    }
}

// --- C API validation helpers ---

fn validate_fa_dims(p: &SparkFlashAttnParams) -> crate::Result<()> {
    if p.batch == 0 || p.num_heads == 0 || p.seq_q == 0 || p.seq_kv == 0 {
        return Err(crate::SparkError::InvalidArgument(format!(
            "FA dims must be > 0: batch={}, heads={}, seq_q={}, seq_kv={}",
            p.batch, p.num_heads, p.seq_q, p.seq_kv
        )));
    }
    if p.q.is_null() || p.k.is_null() || p.v.is_null() || p.o.is_null() {
        return Err(crate::SparkError::InvalidArgument(
            "FA buffer pointers must not be null".into(),
        ));
    }
    Ok(())
}

fn validate_gemm_dims_ffi(m: u32, n: u32, k: u32) -> crate::Result<()> {
    if m == 0 || n == 0 || k == 0 {
        return Err(crate::SparkError::InvalidArgument(format!(
            "GEMM dims must be > 0: M={m}, N={n}, K={k}"
        )));
    }
    Ok(())
}

fn validate_mma_alignment_ffi(m: u32, n: u32, k: u32, k_align: u32) -> crate::Result<()> {
    if !m.is_multiple_of(32) || !n.is_multiple_of(32) {
        return Err(crate::SparkError::InvalidArgument(format!(
            "MMA GEMM requires M, N divisible by 32: M={m}, N={n}"
        )));
    }
    if !k.is_multiple_of(k_align) {
        return Err(crate::SparkError::InvalidArgument(format!(
            "MMA GEMM requires K divisible by {k_align}: K={k}"
        )));
    }
    Ok(())
}

// --- Internal launch helpers ---

unsafe fn launch_fa_bf16_v3(h: &SparkCtx, p: &SparkFlashAttnParams) -> crate::Result<()> {
    validate_fa_dims(p)?;
    let func = module::load_kernel(&h.ctx, "fa_bf16_v3_d128", "flash_attn_bf16_v3_d128")?;
    let grid_x = p.seq_q.div_ceil(128);
    let cfg = LaunchConfig {
        grid_dim: (grid_x, p.num_heads, p.batch),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let q = p.q as u64;
    let k = p.k as u64;
    let v = p.v as u64;
    let o = p.o as u64;
    h.stream
        .launch_builder(&func)
        .arg(&q)
        .arg(&k)
        .arg(&v)
        .arg(&o)
        .arg(&p.seq_q)
        .arg(&p.seq_kv)
        .arg(&p.num_heads)
        .arg(&p.scale)
        .launch(cfg)?;
    Ok(())
}

unsafe fn launch_fa_bf16_v3_causal(h: &SparkCtx, p: &SparkFlashAttnParams) -> crate::Result<()> {
    validate_fa_dims(p)?;
    let func = module::load_kernel(
        &h.ctx,
        "fa_bf16_v3_d128_causal",
        "flash_attn_bf16_v3_d128_causal",
    )?;
    let grid_x = p.seq_q.div_ceil(128);
    let cfg = LaunchConfig {
        grid_dim: (grid_x, p.num_heads, p.batch),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let q = p.q as u64;
    let k = p.k as u64;
    let v = p.v as u64;
    let o = p.o as u64;
    h.stream
        .launch_builder(&func)
        .arg(&q)
        .arg(&k)
        .arg(&v)
        .arg(&o)
        .arg(&p.seq_q)
        .arg(&p.seq_kv)
        .arg(&p.num_heads)
        .arg(&p.scale)
        .launch(cfg)?;
    Ok(())
}

unsafe fn launch_fa_fp8(h: &SparkCtx, p: &SparkFlashAttnParams) -> crate::Result<()> {
    validate_fa_dims(p)?;
    let func = module::load_kernel(&h.ctx, "fa_fp8_d128", "flash_attn_fp8_d128")?;
    let grid_x = p.seq_q.div_ceil(16);
    let cfg = LaunchConfig {
        grid_dim: (grid_x, p.num_heads, p.batch),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    let q = p.q as u64;
    let k = p.k as u64;
    let v = p.v as u64;
    let o = p.o as u64;
    h.stream
        .launch_builder(&func)
        .arg(&q)
        .arg(&k)
        .arg(&v)
        .arg(&o)
        .arg(&p.seq_q)
        .arg(&p.seq_kv)
        .arg(&p.num_heads)
        .arg(&p.scale)
        .launch(cfg)?;
    Ok(())
}

unsafe fn launch_fa_fp8_causal(h: &SparkCtx, p: &SparkFlashAttnParams) -> crate::Result<()> {
    validate_fa_dims(p)?;
    let func = module::load_kernel(&h.ctx, "fa_fp8_d128_causal", "flash_attn_fp8_d128_causal")?;
    let grid_x = p.seq_q.div_ceil(16);
    let cfg = LaunchConfig {
        grid_dim: (grid_x, p.num_heads, p.batch),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    let q = p.q as u64;
    let k = p.k as u64;
    let v = p.v as u64;
    let o = p.o as u64;
    h.stream
        .launch_builder(&func)
        .arg(&q)
        .arg(&k)
        .arg(&v)
        .arg(&o)
        .arg(&p.seq_q)
        .arg(&p.seq_kv)
        .arg(&p.num_heads)
        .arg(&p.scale)
        .launch(cfg)?;
    Ok(())
}

unsafe fn launch_gemm_bf16_mma(h: &SparkCtx, p: &SparkGemmParams) -> crate::Result<()> {
    validate_gemm_dims_ffi(p.m, p.n, p.k)?;
    // gemm_bf16_mma uses 128×64 tile with 4 warps. Caller must align M to 128
    // and N to 64; K to 16. The Rust dispatch (gemm/mod.rs::gemm_bf16_mma)
    // enforces the same. Previous C-API dispatch used (32, 1, 1) blocks and
    // n.div_ceil(32)/m.div_ceil(32) grid which only filled 1/8 of output —
    // this was a real bug.
    if !p.m.is_multiple_of(128) || !p.n.is_multiple_of(64) {
        return Err(crate::SparkError::InvalidArgument(format!(
            "spark_gemm BF16 requires M divisible by 128 and N by 64: M={}, N={}",
            p.m, p.n
        )));
    }
    validate_mma_alignment_ffi(p.m, p.n, p.k, 16)?;
    if p.a.is_null() || p.b.is_null() || p.c.is_null() {
        return Err(crate::SparkError::InvalidArgument(
            "GEMM buffer pointers must not be null".into(),
        ));
    }
    let func = module::load_kernel(&h.ctx, "gemm_bf16_mma", "gemm_bf16_mma")?;
    // 128×64 tile, 4 warps (128 threads) per block, 12KB SMEM. Matches
    // Rust dispatch in gemm/mod.rs.
    let grid_x = p.n / 64;
    let grid_y = p.m / 128;
    let cfg = LaunchConfig {
        grid_dim: (grid_x, grid_y, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 12288,
    };
    let a = p.a as u64;
    let b = p.b as u64;
    let c = p.c as u64;
    h.stream
        .launch_builder(&func)
        .arg(&a)
        .arg(&b)
        .arg(&c)
        .arg(&p.m)
        .arg(&p.n)
        .arg(&p.k)
        .launch(cfg)?;
    Ok(())
}

unsafe fn launch_gemm_fp8_mma(h: &SparkCtx, p: &SparkGemmParams) -> crate::Result<()> {
    validate_gemm_dims_ffi(p.m, p.n, p.k)?;
    validate_mma_alignment_ffi(p.m, p.n, p.k, 32)?;
    if p.a.is_null() || p.b.is_null() || p.c.is_null() {
        return Err(crate::SparkError::InvalidArgument(
            "GEMM buffer pointers must not be null".into(),
        ));
    }
    let func = module::load_kernel(&h.ctx, "gemm_fp8_mma", "gemm_fp8_mma")?;
    let grid_x = p.n.div_ceil(32);
    let grid_y = p.m.div_ceil(32);
    let cfg = LaunchConfig {
        grid_dim: (grid_x, grid_y, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 2048,
    };
    let a = p.a as u64;
    let b = p.b as u64;
    let c = p.c as u64;
    h.stream
        .launch_builder(&func)
        .arg(&a)
        .arg(&b)
        .arg(&c)
        .arg(&p.m)
        .arg(&p.n)
        .arg(&p.k)
        .launch(cfg)?;
    Ok(())
}

unsafe fn launch_topk(h: &SparkCtx, p: &SparkTopkParams) -> crate::Result<()> {
    if p.batch_size == 0 || p.vocab_size == 0 || p.k == 0 {
        return Err(crate::SparkError::InvalidArgument(format!(
            "topk dims must be > 0: batch_size={}, vocab_size={}, k={}",
            p.batch_size, p.vocab_size, p.k
        )));
    }
    if p.logits.is_null() || p.indices.is_null() || p.values.is_null() {
        return Err(crate::SparkError::InvalidArgument(
            "topk buffer pointers must not be null".into(),
        ));
    }
    let func = module::load_kernel(&h.ctx, "topk_sampling", "topk_sampling")?;
    let cfg = LaunchConfig {
        grid_dim: (p.batch_size, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let logits = p.logits as u64;
    let indices = p.indices as u64;
    let values = p.values as u64;
    h.stream
        .launch_builder(&func)
        .arg(&logits)
        .arg(&indices)
        .arg(&values)
        .arg(&p.vocab_size)
        .arg(&p.k)
        .arg(&p.temperature)
        .launch(cfg)?;
    Ok(())
}

unsafe fn launch_moe(h: &SparkCtx, p: &SparkMoeRoutingParams) -> crate::Result<()> {
    if p.num_tokens == 0 || p.num_experts == 0 || p.top_k == 0 {
        return Err(crate::SparkError::InvalidArgument(format!(
            "MoE dims must be > 0: num_tokens={}, num_experts={}, top_k={}",
            p.num_tokens, p.num_experts, p.top_k
        )));
    }
    if p.top_k > p.num_experts {
        return Err(crate::SparkError::InvalidArgument(format!(
            "top_k ({}) must be <= num_experts ({})",
            p.top_k, p.num_experts
        )));
    }
    if p.logits.is_null() || p.expert_ids.is_null() || p.weights.is_null() {
        return Err(crate::SparkError::InvalidArgument(
            "MoE buffer pointers must not be null".into(),
        ));
    }
    let func = module::load_kernel(&h.ctx, "moe_routing", "moe_routing")?;
    let cfg = LaunchConfig {
        grid_dim: (p.num_tokens, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    let logits = p.logits as u64;
    let expert_ids = p.expert_ids as u64;
    let weights = p.weights as u64;
    h.stream
        .launch_builder(&func)
        .arg(&logits)
        .arg(&expert_ids)
        .arg(&weights)
        .arg(&p.num_tokens)
        .arg(&p.num_experts)
        .arg(&p.top_k)
        .launch(cfg)?;
    Ok(())
}

unsafe fn launch_rmsnorm(h: &SparkCtx, p: &SparkRmsNormParams) -> crate::Result<()> {
    if p.hidden_dim == 0 || p.num_rows == 0 {
        return Err(crate::SparkError::InvalidArgument(format!(
            "RMSNorm dims must be > 0: hidden_dim={}, num_rows={}",
            p.hidden_dim, p.num_rows
        )));
    }
    if p.x.is_null() || p.out.is_null() || p.weight.is_null() {
        return Err(crate::SparkError::InvalidArgument(
            "RMSNorm buffer pointers must not be null".into(),
        ));
    }
    let func = module::load_kernel(&h.ctx, "rmsnorm_bf16", "rmsnorm_bf16")?;
    let cfg = LaunchConfig {
        grid_dim: (p.num_rows, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let x = p.x as u64;
    let out = p.out as u64;
    let weight = p.weight as u64;
    h.stream
        .launch_builder(&func)
        .arg(&x)
        .arg(&out)
        .arg(&weight)
        .arg(&p.hidden_dim)
        .arg(&p.eps)
        .launch(cfg)?;
    Ok(())
}

unsafe fn launch_rope(h: &SparkCtx, p: &SparkRopeParams) -> crate::Result<()> {
    if p.batch == 0 || p.seq_len == 0 || p.heads == 0 || p.dim == 0 {
        return Err(crate::SparkError::InvalidArgument(format!(
            "RoPE dims must be > 0: batch={}, seq_len={}, heads={}, dim={}",
            p.batch, p.seq_len, p.heads, p.dim
        )));
    }
    if !p.dim.is_multiple_of(2) {
        return Err(crate::SparkError::InvalidArgument(format!(
            "RoPE dim must be even: dim={}",
            p.dim
        )));
    }
    let half_dim = p.dim / 2;
    if half_dim > 1024 {
        return Err(crate::SparkError::InvalidArgument(format!(
            "RoPE dim/2 ({half_dim}) exceeds max threads per block (1024)"
        )));
    }
    if p.x.is_null() || p.cos_cache.is_null() || p.sin_cache.is_null() {
        return Err(crate::SparkError::InvalidArgument(
            "RoPE buffer pointers must not be null".into(),
        ));
    }
    let func = module::load_kernel(&h.ctx, "rope_bf16", "rope_bf16")?;
    // half_dim already computed above during validation
    let num_positions = p.batch * p.seq_len * p.heads;
    let cfg = LaunchConfig {
        grid_dim: (num_positions, 1, 1),
        block_dim: (half_dim, 1, 1),
        shared_mem_bytes: 0,
    };
    let x = p.x as u64;
    let cos_cache = p.cos_cache as u64;
    let sin_cache = p.sin_cache as u64;
    h.stream
        .launch_builder(&func)
        .arg(&x)
        .arg(&cos_cache)
        .arg(&sin_cache)
        .arg(&p.seq_len)
        .arg(&p.heads)
        .arg(&p.dim)
        .arg(&half_dim)
        .launch(cfg)?;
    Ok(())
}

unsafe fn launch_gemm_nvfp4(h: &SparkCtx, p: &SparkNvfp4GemmParams) -> crate::Result<()> {
    validate_gemm_dims_ffi(p.m, p.n, p.k)?;
    validate_mma_alignment_ffi(p.m, p.n, p.k, 64)?;
    if p.a.is_null() || p.b.is_null() || p.c.is_null() || p.scale_a.is_null() || p.scale_b.is_null()
    {
        return Err(crate::SparkError::InvalidArgument(
            "NVFP4 GEMM buffer pointers must not be null".into(),
        ));
    }
    let func = module::load_kernel(&h.ctx, "gemm_nvfp4_mma", "gemm_nvfp4_mma")?;
    let grid_x = p.n.div_ceil(32);
    let grid_y = p.m.div_ceil(32);
    let cfg = LaunchConfig {
        grid_dim: (grid_x, grid_y, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 2048,
    };
    let a = p.a as u64;
    let b = p.b as u64;
    let c = p.c as u64;
    let scale_a = p.scale_a as u64;
    let scale_b = p.scale_b as u64;
    h.stream
        .launch_builder(&func)
        .arg(&a)
        .arg(&b)
        .arg(&c)
        .arg(&scale_a)
        .arg(&scale_b)
        .arg(&p.m)
        .arg(&p.n)
        .arg(&p.k)
        .launch(cfg)?;
    Ok(())
}

unsafe fn launch_gemm_w4a16(h: &SparkCtx, p: &SparkW4a16GemmParams) -> crate::Result<()> {
    validate_gemm_dims_ffi(p.m, p.n, p.k)?;
    validate_mma_alignment_ffi(p.m, p.n, p.k, 16)?;
    if p.a.is_null() || p.w.is_null() || p.c.is_null() || p.scales.is_null() || p.zeros.is_null() {
        return Err(crate::SparkError::InvalidArgument(
            "W4A16 GEMM buffer pointers must not be null".into(),
        ));
    }
    let func = module::load_kernel(&h.ctx, "gemm_w4a16_mma", "gemm_w4a16_mma")?;
    let grid_x = p.n.div_ceil(32);
    let grid_y = p.m.div_ceil(32);
    let cfg = LaunchConfig {
        grid_dim: (grid_x, grid_y, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 2176,
    };
    let a = p.a as u64;
    let w = p.w as u64;
    let c = p.c as u64;
    let scales = p.scales as u64;
    let zeros = p.zeros as u64;
    h.stream
        .launch_builder(&func)
        .arg(&a)
        .arg(&w)
        .arg(&c)
        .arg(&scales)
        .arg(&zeros)
        .arg(&p.m)
        .arg(&p.n)
        .arg(&p.k)
        .launch(cfg)?;
    Ok(())
}

unsafe fn launch_fa_bf16_varlen(h: &SparkCtx, p: &SparkVarlenFlashAttnParams) -> crate::Result<()> {
    if p.batch == 0 || p.num_heads == 0 || p.max_seqlen_q == 0 {
        return Err(crate::SparkError::InvalidArgument(format!(
            "varlen FA dims must be > 0: batch={}, num_heads={}, max_seqlen_q={}",
            p.batch, p.num_heads, p.max_seqlen_q
        )));
    }
    if p.q.is_null() || p.k.is_null() || p.v.is_null() || p.o.is_null() {
        return Err(crate::SparkError::InvalidArgument(
            "varlen FA buffer pointers must not be null".into(),
        ));
    }
    if p.cu_seqlens_q.is_null() || p.cu_seqlens_k.is_null() {
        return Err(crate::SparkError::InvalidArgument(
            "varlen FA cu_seqlens pointers must not be null".into(),
        ));
    }
    let func = module::load_kernel(&h.ctx, "fa_bf16_varlen_d128", "flash_attn_bf16_varlen_d128")?;
    let grid_x = p.max_seqlen_q.div_ceil(16);
    let cfg = LaunchConfig {
        grid_dim: (grid_x, p.num_heads, p.batch),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 18432,
    };
    let q = p.q as u64;
    let k = p.k as u64;
    let v = p.v as u64;
    let o = p.o as u64;
    let cu_seqlens_q = p.cu_seqlens_q as u64;
    let cu_seqlens_k = p.cu_seqlens_k as u64;
    h.stream
        .launch_builder(&func)
        .arg(&q)
        .arg(&k)
        .arg(&v)
        .arg(&o)
        .arg(&cu_seqlens_q)
        .arg(&cu_seqlens_k)
        .arg(&p.num_heads)
        .arg(&p.max_seqlen_q)
        .arg(&p.scale)
        .launch(cfg)?;
    Ok(())
}

unsafe fn launch_activation(h: &SparkCtx, p: &SparkActivationParams) -> crate::Result<()> {
    if p.total_out_elems == 0 || p.d == 0 {
        return Err(crate::SparkError::InvalidArgument(format!(
            "activation dims must be > 0: total_out_elems={}, d={}",
            p.total_out_elems, p.d
        )));
    }
    if p.input.is_null() || p.out.is_null() {
        return Err(crate::SparkError::InvalidArgument(
            "activation buffer pointers must not be null".into(),
        ));
    }
    let kernel_name = match p.activation {
        SparkActivationType::SiluMul => "silu_mul_bf16",
        SparkActivationType::GeluMul => "gelu_mul_bf16",
        SparkActivationType::GeluTanhMul => "gelu_tanh_mul_bf16",
    };
    let func = module::load_kernel(&h.ctx, kernel_name, kernel_name)?;
    let threads_per_block: u32 = 256;
    let num_blocks = p.total_out_elems.div_ceil(threads_per_block);
    let cfg = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (threads_per_block, 1, 1),
        shared_mem_bytes: 0,
    };
    let input = p.input as u64;
    let out = p.out as u64;
    h.stream
        .launch_builder(&func)
        .arg(&input)
        .arg(&out)
        .arg(&p.total_out_elems)
        .arg(&p.d)
        .launch(cfg)?;
    Ok(())
}
