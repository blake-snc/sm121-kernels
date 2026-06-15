use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaSlice, CudaStream, CudaView, CudaViewMut, DevicePtr, LaunchConfig,
    PushKernelArg,
};

use crate::error::{Result, SparkError};
use crate::module;

/// Per-head dimension of the K/V cache entries these kernels operate on.
pub const KV_HEAD_DIM: u32 = 128;

/// Fused BF16→FP8 quantization + paged KV-cache write.
/// Writes one new token's K/V for all batch×head pairs into a paged FP8 cache.
///
/// Write value: `cache[slot] = cast_to_fp8_e4m3(bf16_val / scale)` per element.
///
/// Layouts:
///   new_k, new_v: [B, H, D=128] BF16
///   k_cache, v_cache: [num_pages, page_size, H, D] FP8 E4M3 (1 byte/elem)
///   page_indices, slot_in_page: [B] u32
#[allow(clippy::too_many_arguments)]
pub fn kv_cache_fp8_write(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    new_k: &CudaSlice<u16>,
    new_v: &CudaSlice<u16>,
    k_cache: &mut CudaSlice<u8>,
    v_cache: &mut CudaSlice<u8>,
    page_indices: &CudaSlice<u32>,
    slot_in_page: &CudaSlice<u32>,
    batch: u32,
    num_heads: u32,
    page_size: u32,
    k_scale: f32,
    v_scale: f32,
) -> Result<()> {
    if batch == 0 || num_heads == 0 || page_size == 0 {
        return Err(SparkError::InvalidArgument(
            "batch, num_heads, page_size must be > 0".into(),
        ));
    }

    use cudarc::driver::sys::*;

    let (new_k_ptr, _a) = new_k.device_ptr(stream);
    let (new_v_ptr, _b) = new_v.device_ptr(stream);
    let (k_cache_ptr, _c) = k_cache.device_ptr(stream);
    let (v_cache_ptr, _d) = v_cache.device_ptr(stream);
    let (page_indices_ptr, _e) = page_indices.device_ptr(stream);
    let (slot_in_page_ptr, _f) = slot_in_page.device_ptr(stream);

    let cu_func = module::load_kernel_raw(ctx, "kv_cache_fp8_write", "kv_cache_fp8_write")?;
    let cu_stream = stream.cu_stream();

    let params: [*mut core::ffi::c_void; 10] = [
        &new_k_ptr as *const u64 as *mut _,
        &new_v_ptr as *const u64 as *mut _,
        &k_cache_ptr as *const u64 as *mut _,
        &v_cache_ptr as *const u64 as *mut _,
        &page_indices_ptr as *const u64 as *mut _,
        &slot_in_page_ptr as *const u64 as *mut _,
        &num_heads as *const u32 as *mut _,
        &page_size as *const u32 as *mut _,
        &k_scale as *const f32 as *mut _,
        &v_scale as *const f32 as *mut _,
    ];

    let result = unsafe {
        cuLaunchKernel(
            cu_func,
            num_heads,
            batch,
            1,
            32,
            1,
            1,
            0,
            cu_stream,
            params.as_ptr() as *mut _,
            core::ptr::null_mut(),
        )
    };

    if result != CUresult::CUDA_SUCCESS {
        return Err(SparkError::LaunchFailed(format!(
            "KV FP8 write launch: {:?}",
            result
        )));
    }
    Ok(())
}

/// Fused BF16→NVFP4 quantization + paged KV-cache write (GLM-5 format).
/// Expects PRE-COMPUTED FP8 E4M3 scales (one per 16-element block).
/// Packed output: 4-bit FP4 E2M1 values, 2 per byte, with per-block scales.
#[allow(clippy::too_many_arguments)]
pub fn kv_cache_nvfp4_write(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    new_k: &CudaSlice<u16>,
    new_v: &CudaSlice<u16>,
    k_cache: &mut CudaSlice<u8>,
    v_cache: &mut CudaSlice<u8>,
    k_scales_in: &CudaSlice<u8>,
    v_scales_in: &CudaSlice<u8>,
    k_scales_out: &mut CudaSlice<u8>,
    v_scales_out: &mut CudaSlice<u8>,
    page_indices: &CudaSlice<u32>,
    slot_in_page: &CudaSlice<u32>,
    batch: u32,
    num_heads: u32,
    page_size: u32,
) -> Result<()> {
    if batch == 0 || num_heads == 0 || page_size == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }

    use cudarc::driver::sys::*;

    let (new_k_ptr, _a) = new_k.device_ptr(stream);
    let (new_v_ptr, _b) = new_v.device_ptr(stream);
    let (k_cache_ptr, _c) = k_cache.device_ptr(stream);
    let (v_cache_ptr, _d) = v_cache.device_ptr(stream);
    let (k_sin_ptr, _e) = k_scales_in.device_ptr(stream);
    let (v_sin_ptr, _f) = v_scales_in.device_ptr(stream);
    let (k_sout_ptr, _g) = k_scales_out.device_ptr(stream);
    let (v_sout_ptr, _h) = v_scales_out.device_ptr(stream);
    let (pi_ptr, _i) = page_indices.device_ptr(stream);
    let (sp_ptr, _j) = slot_in_page.device_ptr(stream);

    let cu_func = module::load_kernel_raw(ctx, "kv_cache_nvfp4_write", "kv_cache_nvfp4_write")?;
    let cu_stream = stream.cu_stream();

    let params: [*mut core::ffi::c_void; 12] = [
        &new_k_ptr as *const u64 as *mut _,
        &new_v_ptr as *const u64 as *mut _,
        &k_cache_ptr as *const u64 as *mut _,
        &v_cache_ptr as *const u64 as *mut _,
        &k_sin_ptr as *const u64 as *mut _,
        &v_sin_ptr as *const u64 as *mut _,
        &k_sout_ptr as *const u64 as *mut _,
        &v_sout_ptr as *const u64 as *mut _,
        &pi_ptr as *const u64 as *mut _,
        &sp_ptr as *const u64 as *mut _,
        &num_heads as *const u32 as *mut _,
        &page_size as *const u32 as *mut _,
    ];

    let result = unsafe {
        cuLaunchKernel(
            cu_func,
            num_heads,
            batch,
            1,
            32,
            1,
            1,
            0,
            cu_stream,
            params.as_ptr() as *mut _,
            core::ptr::null_mut(),
        )
    };

    if result != CUresult::CUDA_SUCCESS {
        return Err(SparkError::LaunchFailed(format!(
            "nvfp4 KV write launch: {:?}",
            result
        )));
    }
    Ok(())
}

/// Single-launch multi-head BF16 KV append:
///   `dst[h, pos, :] = src[h, :]` for h in 0..num_heads_kv
///
/// Layouts:
///   `src`: `[num_heads_kv, head_dim]` BF16
///   `dst`: `[num_heads_kv, max_seq, head_dim]` BF16 (in-place mutation)
///   `pos_ptr`: device-resident u32
///
/// Replaces a host-side per-head dtod loop with a single GPU kernel launch,
/// allowing inclusion in CUDA Graph capture (position read at launch time).
#[allow(clippy::too_many_arguments)]
/// Strided-append a new K and V row into a head-major KV cache.
///
/// Cache layout: [num_kv_heads, max_seq, head_dim] BF16. Writes
/// `src_k[h*head_dim..]` to `dst_k[h*max_seq*head_dim + position*head_dim..]`
/// (and likewise for V) for each kv_head h. Replaces 2*num_kv_heads
/// dtod-async memcpys with a single kernel launch.
/// Full-view variant of `kv_append_strided_bf16`: src and dst both views.
/// Lets the caller slice into both batched src and contiguous BatchedKvCache.
#[allow(clippy::too_many_arguments)]
pub fn kv_append_strided_bf16_full_view(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src_k: &CudaView<u16>,
    src_v: &CudaView<u16>,
    dst_k: &mut CudaViewMut<u16>,
    dst_v: &mut CudaViewMut<u16>,
    position: u32,
    max_seq: u32,
    num_kv_heads: u32,
    head_dim: u32,
) -> Result<()> {
    if num_kv_heads == 0 || head_dim == 0 || max_seq == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    if position >= max_seq {
        return Err(SparkError::InvalidArgument(format!(
            "position {position} >= max_seq {max_seq}"
        )));
    }
    if head_dim > 1024 {
        return Err(SparkError::InvalidArgument(format!(
            "head_dim ({head_dim}) > max threads per block (1024)"
        )));
    }
    let src_need = (num_kv_heads * head_dim) as usize;
    let dst_need = (num_kv_heads * max_seq * head_dim) as usize;
    if src_k.len() < src_need || src_v.len() < src_need {
        return Err(SparkError::InvalidArgument("src buffer too small".into()));
    }
    if dst_k.len() < dst_need || dst_v.len() < dst_need {
        return Err(SparkError::InvalidArgument("dst buffer too small".into()));
    }
    let func = module::load_kernel(ctx, "kv_append_strided_bf16", "kv_append_strided_bf16")?;
    let cfg = LaunchConfig {
        grid_dim: (num_kv_heads, 1, 1),
        block_dim: (head_dim, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(src_k)
            .arg(src_v)
            .arg(dst_k)
            .arg(dst_v)
            .arg(&position)
            .arg(&max_seq)
            .arg(&head_dim)
            .launch(cfg)?;
    }
    Ok(())
}

/// FP8 (e4m3) counterpart of `kv_append_strided_bf16_full_view`. Fused
/// BF16→FP8 quantize + strided write into a per-sequence VIEW of a batched
/// cache buffer. Used by the batched (M=128) full-attn KV append when the cache
/// is FP8: `dst_k`/`dst_v` are `slice_mut`s of one slot's region. Reuses the
/// `kv_append_strided_fp8` PTX kernel (head_dim/2 threads, pair-quantize); the
/// per-tensor `k_scale`/`v_scale` are applied during the quantize.
#[allow(clippy::too_many_arguments)]
pub fn kv_append_strided_fp8_full_view(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src_k: &CudaView<u16>,
    src_v: &CudaView<u16>,
    dst_k: &mut CudaViewMut<u8>,
    dst_v: &mut CudaViewMut<u8>,
    position: u32,
    max_seq: u32,
    num_kv_heads: u32,
    head_dim: u32,
    k_scale: f32,
    v_scale: f32,
) -> Result<()> {
    if num_kv_heads == 0 || head_dim == 0 || max_seq == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    if position >= max_seq {
        return Err(SparkError::InvalidArgument(format!(
            "position {position} >= max_seq {max_seq}"
        )));
    }
    if head_dim & 1 != 0 {
        return Err(SparkError::InvalidArgument(format!(
            "head_dim ({head_dim}) must be even (FP8 pair-quantize)"
        )));
    }
    if (head_dim / 2) > 1024 {
        return Err(SparkError::InvalidArgument(format!(
            "head_dim/2 ({}) > max threads per block",
            head_dim / 2
        )));
    }
    let src_need = (num_kv_heads * head_dim) as usize;
    let dst_need = (num_kv_heads * max_seq * head_dim) as usize;
    if src_k.len() < src_need || src_v.len() < src_need {
        return Err(SparkError::InvalidArgument("src buffer too small".into()));
    }
    if dst_k.len() < dst_need || dst_v.len() < dst_need {
        return Err(SparkError::InvalidArgument("dst buffer too small".into()));
    }
    let func = module::load_kernel(ctx, "kv_append_strided_fp8", "kv_append_strided_fp8")?;
    let cfg = LaunchConfig {
        grid_dim: (num_kv_heads, 1, 1),
        block_dim: (head_dim / 2, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(src_k)
            .arg(src_v)
            .arg(dst_k)
            .arg(dst_v)
            .arg(&position)
            .arg(&max_seq)
            .arg(&head_dim)
            .arg(&k_scale)
            .arg(&v_scale)
            .launch(cfg)?;
    }
    Ok(())
}

/// View-accepting variant of `kv_append_strided_bf16`. The src buffers can be
/// CudaView slices into a batched tensor; dst remains a mutable cache slice.
#[allow(clippy::too_many_arguments)]
pub fn kv_append_strided_bf16_src_view(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src_k: &CudaView<u16>,
    src_v: &CudaView<u16>,
    dst_k: &mut CudaSlice<u16>,
    dst_v: &mut CudaSlice<u16>,
    position: u32,
    max_seq: u32,
    num_kv_heads: u32,
    head_dim: u32,
) -> Result<()> {
    if num_kv_heads == 0 || head_dim == 0 || max_seq == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    if position >= max_seq {
        return Err(SparkError::InvalidArgument(format!(
            "position {position} >= max_seq {max_seq}"
        )));
    }
    if head_dim > 1024 {
        return Err(SparkError::InvalidArgument(format!(
            "head_dim ({head_dim}) > max threads per block (1024)"
        )));
    }
    let src_need = (num_kv_heads * head_dim) as usize;
    let dst_need = (num_kv_heads * max_seq * head_dim) as usize;
    if src_k.len() < src_need || src_v.len() < src_need {
        return Err(SparkError::InvalidArgument("src buffer too small".into()));
    }
    if dst_k.len() < dst_need || dst_v.len() < dst_need {
        return Err(SparkError::InvalidArgument("dst buffer too small".into()));
    }
    let func = module::load_kernel(ctx, "kv_append_strided_bf16", "kv_append_strided_bf16")?;
    let cfg = LaunchConfig {
        grid_dim: (num_kv_heads, 1, 1),
        block_dim: (head_dim, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(src_k)
            .arg(src_v)
            .arg(dst_k)
            .arg(dst_v)
            .arg(&position)
            .arg(&max_seq)
            .arg(&head_dim)
            .launch(cfg)?;
    }
    Ok(())
}

/// Append one token's K and V (`[num_kv_heads, head_dim]`) into contiguous BF16
/// caches laid out `[num_kv_heads, max_seq, head_dim]`, writing each head at
/// `position`. The `max_seq` stride keeps per-head sequences separated.
pub fn kv_append_strided_bf16(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src_k: &CudaSlice<u16>,
    src_v: &CudaSlice<u16>,
    dst_k: &mut CudaSlice<u16>,
    dst_v: &mut CudaSlice<u16>,
    position: u32,
    max_seq: u32,
    num_kv_heads: u32,
    head_dim: u32,
) -> Result<()> {
    if num_kv_heads == 0 || head_dim == 0 || max_seq == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    if position >= max_seq {
        return Err(SparkError::InvalidArgument(format!(
            "position {position} >= max_seq {max_seq}"
        )));
    }
    if head_dim > 1024 {
        return Err(SparkError::InvalidArgument(format!(
            "head_dim ({head_dim}) > max threads per block (1024)"
        )));
    }
    let src_need = (num_kv_heads * head_dim) as usize;
    let dst_need = (num_kv_heads * max_seq * head_dim) as usize;
    if src_k.len() < src_need || src_v.len() < src_need {
        return Err(SparkError::InvalidArgument("src buffer too small".into()));
    }
    if dst_k.len() < dst_need || dst_v.len() < dst_need {
        return Err(SparkError::InvalidArgument("dst buffer too small".into()));
    }
    let func = module::load_kernel(ctx, "kv_append_strided_bf16", "kv_append_strided_bf16")?;
    let cfg = LaunchConfig {
        grid_dim: (num_kv_heads, 1, 1),
        block_dim: (head_dim, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(src_k)
            .arg(src_v)
            .arg(dst_k)
            .arg(dst_v)
            .arg(&position)
            .arg(&max_seq)
            .arg(&head_dim)
            .launch(cfg)?;
    }
    Ok(())
}

/// Position-from-device-pointer variant of `kv_append_strided_bf16`. Reads the
/// current append position from a u32 device pointer instead of taking it as
/// a kernel param. Lets the dispatch be replayed under a CUDA Graph across
/// positions when the host updates `*pos_ptr` between replays.
#[allow(clippy::too_many_arguments)]
pub fn kv_append_strided_bf16_pos_dev(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src_k: &CudaSlice<u16>,
    src_v: &CudaSlice<u16>,
    dst_k: &mut CudaSlice<u16>,
    dst_v: &mut CudaSlice<u16>,
    pos_ptr: &CudaSlice<u32>,
    max_seq: u32,
    num_kv_heads: u32,
    head_dim: u32,
) -> Result<()> {
    if num_kv_heads == 0 || head_dim == 0 || max_seq == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    if head_dim > 1024 {
        return Err(SparkError::InvalidArgument(format!(
            "head_dim ({head_dim}) > max threads per block (1024)"
        )));
    }
    if pos_ptr.is_empty() {
        return Err(SparkError::InvalidArgument("pos_ptr empty".into()));
    }
    let src_need = (num_kv_heads * head_dim) as usize;
    let dst_need = (num_kv_heads * max_seq * head_dim) as usize;
    if src_k.len() < src_need || src_v.len() < src_need {
        return Err(SparkError::InvalidArgument("src buffer too small".into()));
    }
    if dst_k.len() < dst_need || dst_v.len() < dst_need {
        return Err(SparkError::InvalidArgument("dst buffer too small".into()));
    }
    let func = module::load_kernel(
        ctx,
        "kv_append_strided_bf16_pos_dev",
        "kv_append_strided_bf16_pos_dev",
    )?;
    let cfg = LaunchConfig {
        grid_dim: (num_kv_heads, 1, 1),
        block_dim: (head_dim, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(src_k)
            .arg(src_v)
            .arg(dst_k)
            .arg(dst_v)
            .arg(pos_ptr)
            .arg(&max_seq)
            .arg(&head_dim)
            .launch(cfg)?;
    }
    Ok(())
}

/// Strided-append a new K and V row from BF16 src into a head-major FP8 KV cache.
///
/// For each kv_head h, dst_*[h*max_seq*head_dim + position*head_dim ..] receives
/// the FP8 e4m3 quantization of src_*[h*head_dim ..]: `quant(x) = x / scale`,
/// stored as 1 byte per element. The dequant on read is `x * scale`.
///
/// `head_dim` must be even (kernel uses pair-quantize cvt.rn.satfinite.e4m3x2.f32).
#[allow(clippy::too_many_arguments)]
pub fn kv_append_strided_fp8(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src_k: &CudaSlice<u16>,
    src_v: &CudaSlice<u16>,
    dst_k: &mut CudaSlice<u8>,
    dst_v: &mut CudaSlice<u8>,
    position: u32,
    max_seq: u32,
    num_kv_heads: u32,
    head_dim: u32,
    k_scale: f32,
    v_scale: f32,
) -> Result<()> {
    if num_kv_heads == 0 || head_dim == 0 || max_seq == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    if position >= max_seq {
        return Err(SparkError::InvalidArgument(format!(
            "position {position} >= max_seq {max_seq}"
        )));
    }
    if head_dim & 1 != 0 {
        return Err(SparkError::InvalidArgument(format!(
            "head_dim ({head_dim}) must be even (FP8 pair-quantize)"
        )));
    }
    if (head_dim / 2) > 1024 {
        return Err(SparkError::InvalidArgument(format!(
            "head_dim/2 ({}) > max threads per block",
            head_dim / 2
        )));
    }
    let src_need = (num_kv_heads * head_dim) as usize;
    let dst_need = (num_kv_heads * max_seq * head_dim) as usize;
    if src_k.len() < src_need || src_v.len() < src_need {
        return Err(SparkError::InvalidArgument("src buffer too small".into()));
    }
    if dst_k.len() < dst_need || dst_v.len() < dst_need {
        return Err(SparkError::InvalidArgument("dst buffer too small".into()));
    }
    let func = module::load_kernel(ctx, "kv_append_strided_fp8", "kv_append_strided_fp8")?;
    let cfg = LaunchConfig {
        grid_dim: (num_kv_heads, 1, 1),
        block_dim: (head_dim / 2, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(src_k)
            .arg(src_v)
            .arg(dst_k)
            .arg(dst_v)
            .arg(&position)
            .arg(&max_seq)
            .arg(&head_dim)
            .arg(&k_scale)
            .arg(&v_scale)
            .launch(cfg)?;
    }
    Ok(())
}

/// Position-from-device-pointer variant of `kv_append_strided_fp8`. Same
/// fused BF16→FP8 quantize + strided write, but reads `position` from a
/// u32 device pointer. Required for CUDA Graph capture compatibility —
/// the captured graph holds the `pos_ptr` device address constant; the
/// host updates `*pos_ptr` between replays.
#[allow(clippy::too_many_arguments)]
pub fn kv_append_strided_fp8_pos_dev(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src_k: &CudaSlice<u16>,
    src_v: &CudaSlice<u16>,
    dst_k: &mut CudaSlice<u8>,
    dst_v: &mut CudaSlice<u8>,
    pos_ptr: &CudaSlice<u32>,
    max_seq: u32,
    num_kv_heads: u32,
    head_dim: u32,
    k_scale: f32,
    v_scale: f32,
) -> Result<()> {
    if num_kv_heads == 0 || head_dim == 0 || max_seq == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    if head_dim & 1 != 0 {
        return Err(SparkError::InvalidArgument(format!(
            "head_dim ({head_dim}) must be even (FP8 pair-quantize)"
        )));
    }
    if (head_dim / 2) > 1024 {
        return Err(SparkError::InvalidArgument(format!(
            "head_dim/2 ({}) > max threads per block",
            head_dim / 2
        )));
    }
    if pos_ptr.is_empty() {
        return Err(SparkError::InvalidArgument("pos_ptr empty".into()));
    }
    let src_need = (num_kv_heads * head_dim) as usize;
    let dst_need = (num_kv_heads * max_seq * head_dim) as usize;
    if src_k.len() < src_need || src_v.len() < src_need {
        return Err(SparkError::InvalidArgument("src buffer too small".into()));
    }
    if dst_k.len() < dst_need || dst_v.len() < dst_need {
        return Err(SparkError::InvalidArgument("dst buffer too small".into()));
    }
    let func = module::load_kernel(
        ctx,
        "kv_append_strided_fp8_pos_dev",
        "kv_append_strided_fp8_pos_dev",
    )?;
    let cfg = LaunchConfig {
        grid_dim: (num_kv_heads, 1, 1),
        block_dim: (head_dim / 2, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(src_k)
            .arg(src_v)
            .arg(dst_k)
            .arg(dst_v)
            .arg(pos_ptr)
            .arg(&max_seq)
            .arg(&head_dim)
            .arg(&k_scale)
            .arg(&v_scale)
            .launch(cfg)?;
    }
    Ok(())
}

/// Append one token's multi-head BF16 tensor (`[num_heads_kv, head_dim]`) into a
/// `[num_heads_kv, max_seq, head_dim]` cache, reading the write position from
/// `pos_ptr` on the device so the append needs no host sync.
pub fn kv_append_multihead_bf16_pos_dev(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<u16>,
    dst: &mut CudaSlice<u16>,
    pos_ptr: &CudaSlice<u32>,
    num_heads_kv: u32,
    max_seq: u32,
    head_dim: u32,
) -> Result<()> {
    if num_heads_kv == 0 || max_seq == 0 || head_dim == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    let src_need = (num_heads_kv * head_dim) as usize;
    let dst_need = (num_heads_kv * max_seq * head_dim) as usize;
    if src.len() < src_need {
        return Err(SparkError::InvalidArgument("src too small".into()));
    }
    if dst.len() < dst_need {
        return Err(SparkError::InvalidArgument("dst too small".into()));
    }
    if pos_ptr.is_empty() {
        return Err(SparkError::InvalidArgument("pos_ptr empty".into()));
    }
    let func = module::load_kernel(
        ctx,
        "kv_append_multihead_bf16_pos_dev",
        "kv_append_multihead_bf16_pos_dev",
    )?;
    let cfg = LaunchConfig {
        grid_dim: (num_heads_kv, 1, 1),
        block_dim: (head_dim, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(src)
            .arg(dst)
            .arg(pos_ptr)
            .arg(&max_seq)
            .arg(&head_dim)
            .launch(cfg)?;
    }
    Ok(())
}
