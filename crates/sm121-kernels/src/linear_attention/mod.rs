use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg,
};

use crate::error::{Result, SparkError};
use crate::module;

/// Per-head dimension of GatedDeltaNet value/state used by these kernels.
pub const GDN_HEAD_DIM: u32 = 128;

/// Compute Qwen3-Next / Mamba2 GatedDeltaNet decay scalars for one decode token:
///   dt[i]    = softplus(a_logits[i] + dt_bias[i])
///   alpha[i] = exp(-exp(a_log[i]) * dt[i])
///   beta[i]  = sigmoid(b_logits[i])
///
/// Inputs (BF16 stored as u16):  a_logits, b_logits, dt_bias, a_log — all `[n]`.
/// Outputs (FP32):               alpha, beta — both `[n]`.
///
/// Launch: 1 block, `n` threads (n ≤ 1024; typically 32 V heads).
#[allow(clippy::too_many_arguments)]
/// M-batched variant: M sequences in one launch.
///
/// Layouts:
///   a_logits, b_logits: [M, n] BF16
///   dt_bias, a_log:     [n] BF16 (shared across M sequences — model weights)
///   alpha, beta:        [M, n] FP32
///
/// Grid: (ceil(n / 128), M, 1).
#[allow(clippy::too_many_arguments)]
pub fn gdn_alpha_beta_bf16_batched(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    a_logits: &CudaSlice<u16>,
    b_logits: &CudaSlice<u16>,
    dt_bias: &CudaSlice<u16>,
    a_log: &CudaSlice<u16>,
    alpha: &mut CudaSlice<f32>,
    beta: &mut CudaSlice<f32>,
    n: u32,
    batch_m: u32,
) -> Result<()> {
    if n == 0 || batch_m == 0 {
        return Err(SparkError::InvalidArgument(
            "n and batch_m must be > 0".into(),
        ));
    }
    let m = batch_m as usize;
    let need = (n as usize) * m;
    if a_logits.len() < need || b_logits.len() < need || alpha.len() < need || beta.len() < need {
        return Err(SparkError::InvalidArgument(
            "batched alpha_beta a/b/alpha/beta buffer too small".into(),
        ));
    }
    if dt_bias.len() < n as usize || a_log.len() < n as usize {
        return Err(SparkError::InvalidArgument(
            "dt_bias / a_log buffer too small".into(),
        ));
    }

    use cudarc::driver::sys::*;

    let (al, _a) = a_logits.device_ptr(stream);
    let (bl, _b) = b_logits.device_ptr(stream);
    let (db, _c) = dt_bias.device_ptr(stream);
    let (alg, _d) = a_log.device_ptr(stream);
    let (alpha_p, _e) = alpha.device_ptr_mut(stream);
    let (beta_p, _f) = beta.device_ptr_mut(stream);
    let cu_func = module::load_kernel_raw(
        ctx,
        "gdn_alpha_beta_bf16_batched",
        "gdn_alpha_beta_bf16_batched",
    )?;
    let cu_stream = stream.cu_stream();

    const BLOCK: u32 = 128;
    let grid_x = n.div_ceil(BLOCK);

    let params: [*mut core::ffi::c_void; 7] = [
        &al as *const u64 as *mut _,
        &bl as *const u64 as *mut _,
        &db as *const u64 as *mut _,
        &alg as *const u64 as *mut _,
        &alpha_p as *const u64 as *mut _,
        &beta_p as *const u64 as *mut _,
        &n as *const u32 as *mut _,
    ];
    let result = unsafe {
        cuLaunchKernel(
            cu_func,
            grid_x,
            batch_m,
            1,
            BLOCK,
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
            "gdn_alpha_beta_bf16_batched launch: {:?}",
            result
        )));
    }
    Ok(())
}

/// Single-token (M=1) variant of the batched GatedDeltaNet decay computation:
/// `dt = softplus(a_logits + dt_bias)`, `alpha = exp(-exp(a_log) * dt)`,
/// `beta = sigmoid(b_logits)` over `n` value heads. Inputs BF16, outputs FP32.
pub fn gdn_alpha_beta_bf16(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    a_logits: &CudaSlice<u16>,
    b_logits: &CudaSlice<u16>,
    dt_bias: &CudaSlice<u16>,
    a_log: &CudaSlice<u16>,
    alpha: &mut CudaSlice<f32>,
    beta: &mut CudaSlice<f32>,
    n: u32,
) -> Result<()> {
    if n == 0 || n > 1024 {
        return Err(SparkError::InvalidArgument(format!(
            "n must be in 1..=1024, got {n}"
        )));
    }
    let need = n as usize;
    if a_logits.len() < need
        || b_logits.len() < need
        || dt_bias.len() < need
        || a_log.len() < need
        || alpha.len() < need
        || beta.len() < need
    {
        return Err(SparkError::InvalidArgument(
            "alpha/beta buffer too small".into(),
        ));
    }

    use cudarc::driver::sys::*;

    let (al, _a) = a_logits.device_ptr(stream);
    let (bl, _b) = b_logits.device_ptr(stream);
    let (db, _c) = dt_bias.device_ptr(stream);
    let (alg, _d) = a_log.device_ptr(stream);
    let (alpha_p, _e) = alpha.device_ptr_mut(stream);
    let (beta_p, _f) = beta.device_ptr_mut(stream);
    let cu_func = module::load_kernel_raw(ctx, "gdn_alpha_beta_bf16", "gdn_alpha_beta_bf16")?;
    let cu_stream = stream.cu_stream();

    let params: [*mut core::ffi::c_void; 7] = [
        &al as *const u64 as *mut _,
        &bl as *const u64 as *mut _,
        &db as *const u64 as *mut _,
        &alg as *const u64 as *mut _,
        &alpha_p as *const u64 as *mut _,
        &beta_p as *const u64 as *mut _,
        &n as *const u32 as *mut _,
    ];
    let result = unsafe {
        cuLaunchKernel(
            cu_func,
            1,
            1,
            1,
            n,
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
            "gdn_alpha_beta_bf16 launch: {:?}",
            result
        )));
    }
    Ok(())
}

/// Per-head L2 normalization with scale + GQA-style interleave replication.
///
///   inv = scale / sqrt(sum(x[h, :]^2) + eps)
///   y[h*factor + r, d] = x[h, d] * inv     for r in 0..factor
///
/// Used by Qwen3-Next / GDN-hybrid GDN to L2-norm Q (with scale=1/sqrt(D_k)) and
/// K (with scale=1.0), then replicate from `n_qk` heads → `n_v` heads
/// (factor = n_v / n_qk).
///
/// Shapes (BF16 stored as u16):
///   x: [num_heads_in,            head_dim]
///   y: [num_heads_in * factor,   head_dim]
///
/// Launch: 1 block per input head, head_dim threads. head_dim must be a
/// multiple of 32 and ≤ 1024.
#[allow(clippy::too_many_arguments)]
pub fn l2norm_scale_replicate_bf16(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<u16>,
    y: &mut CudaSlice<u16>,
    num_heads_in: u32,
    head_dim: u32,
    factor: u32,
    scale: f32,
    eps: f32,
) -> Result<()> {
    if num_heads_in == 0 || head_dim == 0 || factor == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    if !head_dim.is_multiple_of(32) || head_dim > 1024 {
        return Err(SparkError::InvalidArgument(format!(
            "head_dim must be multiple of 32 and ≤ 1024, got {head_dim}"
        )));
    }
    let in_need = (num_heads_in * head_dim) as usize;
    let out_need = (num_heads_in * factor * head_dim) as usize;
    if x.len() < in_need {
        return Err(SparkError::InvalidArgument(format!(
            "x buffer too small: {} < {in_need}",
            x.len()
        )));
    }
    if y.len() < out_need {
        return Err(SparkError::InvalidArgument(format!(
            "y buffer too small: {} < {out_need}",
            y.len()
        )));
    }

    use cudarc::driver::sys::*;

    let (x_ptr, _a) = x.device_ptr(stream);
    let (y_ptr, _b) = y.device_ptr_mut(stream);
    let cu_func = module::load_kernel_raw(
        ctx,
        "l2norm_scale_replicate_bf16",
        "l2norm_scale_replicate_bf16",
    )?;
    let cu_stream = stream.cu_stream();

    let params: [*mut core::ffi::c_void; 6] = [
        &x_ptr as *const u64 as *mut _,
        &y_ptr as *const u64 as *mut _,
        &head_dim as *const u32 as *mut _,
        &factor as *const u32 as *mut _,
        &scale as *const f32 as *mut _,
        &eps as *const f32 as *mut _,
    ];
    let result = unsafe {
        cuLaunchKernel(
            cu_func,
            num_heads_in,
            1,
            1,
            head_dim,
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
            "l2norm_scale_replicate_bf16 launch: {:?}",
            result
        )));
    }
    Ok(())
}

/// Depthwise causal conv1d (kernel=4) + SiLU, single-token decode.
///
/// For each channel c (0..total_qkv) independently:
///   y[c] = w[c, 0]*state[0, c] + w[c, 1]*state[1, c]
///        + w[c, 2]*state[2, c] + w[c, 3]*current[c]
///   out[c] = y * sigmoid(y)
/// And state advances: state[0]=old state[1], state[1]=old state[2],
/// state[2]=current.
///
/// Layouts:
///   state:   [3, total_qkv] BF16 (in/out)
///   current: [total_qkv]    BF16
///   weight:  [total_qkv, 1, 4] BF16 (HF Conv1d depthwise)
///   out:     [total_qkv]    BF16
///
/// Used by Qwen3-Next / GDN-hybrid GatedDeltaNet for the QKV-stream conv1d.
#[allow(clippy::too_many_arguments)]
/// M-batched variant: M independent sequences in one launch.
///
/// Layouts:
///   state:   [M, 3, total_qkv] BF16 (in/out, advances per seq)
///   current: [M, total_qkv] BF16
///   weight:  [total_qkv, 1, 4] BF16 (shared across M)
///   out:     [M, total_qkv] BF16
///
/// Grid: (ceil(total_qkv / 128), M, 1).
pub fn conv1d_silu_bf16_batched(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    state: &mut CudaSlice<u16>,
    current: &CudaSlice<u16>,
    weight: &CudaSlice<u16>,
    out: &mut CudaSlice<u16>,
    total_qkv: u32,
    batch_m: u32,
) -> Result<()> {
    if total_qkv == 0 || batch_m == 0 {
        return Err(SparkError::InvalidArgument(
            "total_qkv and batch_m must be > 0".into(),
        ));
    }
    let n = total_qkv as usize;
    let m = batch_m as usize;
    if state.len() < m * 3 * n {
        return Err(SparkError::InvalidArgument(format!(
            "batched conv1d state buffer too small: need {} got {}",
            m * 3 * n,
            state.len()
        )));
    }
    if current.len() < m * n || out.len() < m * n {
        return Err(SparkError::InvalidArgument(
            "batched conv1d current/out buffer too small".into(),
        ));
    }
    if weight.len() < 4 * n {
        return Err(SparkError::InvalidArgument(format!(
            "conv1d weight buffer too small: need {} got {}",
            4 * n,
            weight.len()
        )));
    }

    use cudarc::driver::sys::*;

    let (state_ptr, _s) = state.device_ptr_mut(stream);
    let (current_ptr, _c) = current.device_ptr(stream);
    let (weight_ptr, _w) = weight.device_ptr(stream);
    let (out_ptr, _o) = out.device_ptr_mut(stream);

    let cu_func =
        module::load_kernel_raw(ctx, "conv1d_silu_bf16_batched", "conv1d_silu_bf16_batched")?;
    let cu_stream = stream.cu_stream();

    const BLOCK: u32 = 128;
    let grid_x = total_qkv.div_ceil(BLOCK);

    let params: [*mut core::ffi::c_void; 5] = [
        &state_ptr as *const u64 as *mut _,
        &current_ptr as *const u64 as *mut _,
        &weight_ptr as *const u64 as *mut _,
        &out_ptr as *const u64 as *mut _,
        &total_qkv as *const u32 as *mut _,
    ];

    let result = unsafe {
        cuLaunchKernel(
            cu_func,
            grid_x,
            batch_m,
            1,
            BLOCK,
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
            "conv1d_silu_bf16_batched launch failed: {:?}",
            result
        )));
    }
    Ok(())
}

/// Single-token (M=1) variant of `conv1d_silu_bf16_batched`: applies the
/// depthwise causal Conv1d (kernel width 4) over the QKV stream followed by
/// SiLU, advancing the 3-tap rolling `state` in place. Used by Qwen3-Next /
/// GDN-hybrid GatedDeltaNet during decode.
pub fn conv1d_silu_bf16(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    state: &mut CudaSlice<u16>,
    current: &CudaSlice<u16>,
    weight: &CudaSlice<u16>,
    out: &mut CudaSlice<u16>,
    total_qkv: u32,
) -> Result<()> {
    if total_qkv == 0 {
        return Err(SparkError::InvalidArgument("total_qkv must be > 0".into()));
    }
    let n = total_qkv as usize;
    if state.len() < 3 * n {
        return Err(SparkError::InvalidArgument(format!(
            "conv1d state buffer too small: need {} got {}",
            3 * n,
            state.len()
        )));
    }
    if current.len() < n || out.len() < n {
        return Err(SparkError::InvalidArgument(
            "conv1d current/out buffer too small".into(),
        ));
    }
    if weight.len() < 4 * n {
        return Err(SparkError::InvalidArgument(format!(
            "conv1d weight buffer too small: need {} got {}",
            4 * n,
            weight.len()
        )));
    }

    use cudarc::driver::sys::*;

    let (state_ptr, _s) = state.device_ptr_mut(stream);
    let (current_ptr, _c) = current.device_ptr(stream);
    let (weight_ptr, _w) = weight.device_ptr(stream);
    let (out_ptr, _o) = out.device_ptr_mut(stream);

    let cu_func = module::load_kernel_raw(ctx, "conv1d_silu_bf16", "conv1d_silu_bf16")?;
    let cu_stream = stream.cu_stream();

    const BLOCK: u32 = 128;
    let grid = total_qkv.div_ceil(BLOCK);

    let params: [*mut core::ffi::c_void; 5] = [
        &state_ptr as *const u64 as *mut _,
        &current_ptr as *const u64 as *mut _,
        &weight_ptr as *const u64 as *mut _,
        &out_ptr as *const u64 as *mut _,
        &total_qkv as *const u32 as *mut _,
    ];

    let result = unsafe {
        cuLaunchKernel(
            cu_func,
            grid,
            1,
            1,
            BLOCK,
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
            "conv1d_silu_bf16 launch: {:?}",
            result
        )));
    }
    Ok(())
}

/// Row-offset variant of `conv1d_silu_bf16` for chunked prefill: reads the
/// `current` token directly from row `row_offset_elems` of a larger
/// `[M, total_qkv]` input buffer and writes the result to the same row of a
/// larger output buffer, avoiding the per-row gather/scatter `memcpy_dtod`
/// copies. Byte-identical to copy-in / conv / copy-out, since the same kernel
/// runs on the same bytes — only the launch pointers carry the row offset.
#[allow(clippy::too_many_arguments)]
pub fn conv1d_silu_bf16_row(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    state: &mut CudaSlice<u16>,
    current_full: &CudaSlice<u16>,
    out_full: &mut CudaSlice<u16>,
    weight: &CudaSlice<u16>,
    total_qkv: u32,
    row_offset_elems: usize,
) -> Result<()> {
    if total_qkv == 0 {
        return Err(SparkError::InvalidArgument("total_qkv must be > 0".into()));
    }
    let n = total_qkv as usize;
    if state.len() < 3 * n {
        return Err(SparkError::InvalidArgument(format!(
            "conv1d state buffer too small: need {} got {}",
            3 * n,
            state.len()
        )));
    }
    if current_full.len() < row_offset_elems + n || out_full.len() < row_offset_elems + n {
        return Err(SparkError::InvalidArgument(
            "conv1d row-view current/out buffer too small for offset".into(),
        ));
    }
    if weight.len() < 4 * n {
        return Err(SparkError::InvalidArgument(format!(
            "conv1d weight buffer too small: need {} got {}",
            4 * n,
            weight.len()
        )));
    }

    use cudarc::driver::sys::*;

    let (state_ptr, _s) = state.device_ptr_mut(stream);
    let (current_base, _c) = current_full.device_ptr(stream);
    let (weight_ptr, _w) = weight.device_ptr(stream);
    let (out_base, _o) = out_full.device_ptr_mut(stream);
    let byte_off = (row_offset_elems * 2) as u64;
    let current_ptr = current_base + byte_off;
    let out_ptr = out_base + byte_off;

    let cu_func = module::load_kernel_raw(ctx, "conv1d_silu_bf16", "conv1d_silu_bf16")?;
    let cu_stream = stream.cu_stream();

    const BLOCK: u32 = 128;
    let grid = total_qkv.div_ceil(BLOCK);

    let params: [*mut core::ffi::c_void; 5] = [
        &state_ptr as *const u64 as *mut _,
        &current_ptr as *const u64 as *mut _,
        &weight_ptr as *const u64 as *mut _,
        &out_ptr as *const u64 as *mut _,
        &total_qkv as *const u32 as *mut _,
    ];

    let result = unsafe {
        cuLaunchKernel(
            cu_func,
            grid,
            1,
            1,
            BLOCK,
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
            "conv1d_silu_bf16_row launch: {:?}",
            result
        )));
    }
    Ok(())
}

/// Chunked depthwise causal Conv1d (kernel=4) + SiLU over `m` consecutive tokens
/// of ONE sequence in a single launch. Byte-identical to calling
/// `conv1d_silu_bf16` (the M=1 kernel) `m` times while threading the 3-tap state:
/// the kernel keeps the sliding window in registers (same f32 fma order + SiLU)
/// and writes only the final state. Replaces the per-token prefill conv loop.
///
/// `current`/`out` are the full `[m, total_qkv]` row-major buffers; `state` is
/// `[3, total_qkv]` (the 3 tokens before this chunk, updated in place to the
/// last 3 tokens of the chunk).
#[allow(clippy::too_many_arguments)]
pub fn conv1d_silu_chunk_bf16(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    state: &mut CudaSlice<u16>,
    current: &CudaSlice<u16>,
    weight: &CudaSlice<u16>,
    out: &mut CudaSlice<u16>,
    total_qkv: u32,
    m: u32,
) -> Result<()> {
    if total_qkv == 0 || m == 0 {
        return Err(SparkError::InvalidArgument(
            "total_qkv and m must be > 0".into(),
        ));
    }
    let n = total_qkv as usize;
    let mm = m as usize;
    if state.len() < 3 * n {
        return Err(SparkError::InvalidArgument(format!(
            "chunk conv1d state buffer too small: need {} got {}",
            3 * n,
            state.len()
        )));
    }
    if current.len() < mm * n || out.len() < mm * n {
        return Err(SparkError::InvalidArgument(
            "chunk conv1d current/out buffer too small".into(),
        ));
    }
    if weight.len() < 4 * n {
        return Err(SparkError::InvalidArgument(format!(
            "conv1d weight buffer too small: need {} got {}",
            4 * n,
            weight.len()
        )));
    }

    use cudarc::driver::sys::*;

    let (state_ptr, _s) = state.device_ptr_mut(stream);
    let (current_ptr, _c) = current.device_ptr(stream);
    let (weight_ptr, _w) = weight.device_ptr(stream);
    let (out_ptr, _o) = out.device_ptr_mut(stream);

    let cu_func = module::load_kernel_raw(ctx, "conv1d_silu_chunk_bf16", "conv1d_silu_chunk_bf16")?;
    let cu_stream = stream.cu_stream();

    const BLOCK: u32 = 128;
    let grid = total_qkv.div_ceil(BLOCK);

    let params: [*mut core::ffi::c_void; 6] = [
        &state_ptr as *const u64 as *mut _,
        &current_ptr as *const u64 as *mut _,
        &weight_ptr as *const u64 as *mut _,
        &out_ptr as *const u64 as *mut _,
        &total_qkv as *const u32 as *mut _,
        &m as *const u32 as *mut _,
    ];

    let result = unsafe {
        cuLaunchKernel(
            cu_func,
            grid,
            1,
            1,
            BLOCK,
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
            "conv1d_silu_chunk_bf16 launch: {:?}",
            result
        )));
    }
    Ok(())
}

/// Backward for `gdn_alpha_beta_bf16` — produces gradients wrt the inputs
/// `a_logits`, `b_logits`, `dt_bias`, `a_log` from upstream `dalpha`/`dbeta`.
///
/// Forward chain: x = a_logits + dt_bias; dt = softplus(x); A_pos = exp(a_log);
/// alpha = exp(-A_pos * dt); beta = sigmoid(b_logits).
///
/// Backward:
///   db_logits = dbeta * beta * (1 - beta)
///   dx        = -A_pos * dalpha * alpha * sigmoid(x)
///   da_logits = dx; ddt_bias = dx
///   dA_log    = -dt * dalpha * alpha * A_pos
#[allow(clippy::too_many_arguments)]
pub fn gdn_alpha_beta_backward_bf16(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    a_logits: &CudaSlice<u16>,
    b_logits: &CudaSlice<u16>,
    dt_bias: &CudaSlice<u16>,
    a_log: &CudaSlice<u16>,
    dalpha: &CudaSlice<f32>,
    dbeta: &CudaSlice<f32>,
    da_logits: &mut CudaSlice<u16>,
    db_logits: &mut CudaSlice<u16>,
    ddt_bias: &mut CudaSlice<u16>,
    d_a_log: &mut CudaSlice<u16>,
    n: u32,
) -> Result<()> {
    if n == 0 || n > 1024 {
        return Err(SparkError::InvalidArgument(format!(
            "n must be in 1..=1024, got {n}"
        )));
    }
    let need = n as usize;
    if a_logits.len() < need
        || b_logits.len() < need
        || dt_bias.len() < need
        || a_log.len() < need
        || dalpha.len() < need
        || dbeta.len() < need
        || da_logits.len() < need
        || db_logits.len() < need
        || ddt_bias.len() < need
        || d_a_log.len() < need
    {
        return Err(SparkError::InvalidArgument(
            "gdn_alpha_beta_backward: buffer too small".into(),
        ));
    }

    let func = module::load_kernel(
        ctx,
        "gdn_alpha_beta_backward_bf16",
        "gdn_alpha_beta_backward_bf16",
    )?;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (n, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(a_logits)
            .arg(b_logits)
            .arg(dt_bias)
            .arg(a_log)
            .arg(dalpha)
            .arg(dbeta)
            .arg(da_logits)
            .arg(db_logits)
            .arg(ddt_bias)
            .arg(d_a_log)
            .arg(&n)
            .launch(cfg)?;
    }
    Ok(())
}

/// Backward for `gdn_decode` — the single-token GDN recurrent state update.
///
/// Forward (post the alpha-decay fix in commit `564201e`):
///   `m_t    = S_{t-1} @ k_t`
///   `diff_t = v_t - alpha_t * m_t`
///   `S_t    = alpha_t * S_{t-1} + beta_t * diff_t ⊗ k_t`
///   `y_t    = S_t @ q_t`
///
/// Backward: given upstream `dy_t` and downstream `dS_t` (from later timesteps),
/// computes `dq, dk, dv, dalpha, dbeta` for this step and `dS_{t-1}` for the
/// previous step. `dS_t_inout` is in/out: input dS_t, output dS_{t-1}.
///
/// Caller must pre-allocate and pre-zero output buffers. Cached forward state
/// (`s_old`, `s_new`) must be saved during training-mode forward.
#[allow(clippy::too_many_arguments)]
pub fn gdn_decode_backward(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    s_old: &CudaSlice<f32>,
    s_new: &CudaSlice<f32>,
    q: &CudaSlice<u16>,
    k: &CudaSlice<u16>,
    v: &CudaSlice<u16>,
    alpha: &CudaSlice<f32>,
    beta: &CudaSlice<f32>,
    dy: &CudaSlice<u16>,
    ds_t_inout: &mut CudaSlice<f32>,
    dq: &mut CudaSlice<u16>,
    dk: &mut CudaSlice<u16>,
    dv: &mut CudaSlice<u16>,
    dalpha: &mut CudaSlice<f32>,
    dbeta: &mut CudaSlice<f32>,
    batch: u32,
    num_heads: u32,
) -> Result<()> {
    if batch == 0 || num_heads == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    let bh = (batch * num_heads) as usize;
    let d_need = bh * 128;
    let s_need = bh * 128 * 128;
    if s_old.len() < s_need || s_new.len() < s_need || ds_t_inout.len() < s_need {
        return Err(SparkError::InvalidArgument(
            "state buffer too small (need B*H*D*D)".into(),
        ));
    }
    if q.len() < d_need
        || k.len() < d_need
        || v.len() < d_need
        || dy.len() < d_need
        || dq.len() < d_need
        || dk.len() < d_need
        || dv.len() < d_need
    {
        return Err(SparkError::InvalidArgument(
            "D-vector buffer too small".into(),
        ));
    }
    if alpha.len() < bh || beta.len() < bh || dalpha.len() < bh || dbeta.len() < bh {
        return Err(SparkError::InvalidArgument(
            "alpha/beta/dalpha/dbeta buffer too small".into(),
        ));
    }

    // SMEM ~ 69 KB > 48 KB default per-block limit. Use raw cuLaunchKernel
    // + cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES) like the forward.
    use cudarc::driver::sys::*;
    const SMEM_TOTAL: i32 = 69248;

    let cu_func =
        module::load_kernel_raw(ctx, "gdn_decode_backward_bf16", "gdn_decode_backward_bf16")?;
    let attr_result = unsafe {
        cuFuncSetAttribute(
            cu_func,
            CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            SMEM_TOTAL,
        )
    };
    if attr_result != CUresult::CUDA_SUCCESS {
        return Err(SparkError::LaunchFailed(format!(
            "cuFuncSetAttribute: {:?}",
            attr_result
        )));
    }

    let (so_p, _g1) = s_old.device_ptr(stream);
    let (sn_p, _g2) = s_new.device_ptr(stream);
    let (q_p, _g3) = q.device_ptr(stream);
    let (k_p, _g4) = k.device_ptr(stream);
    let (v_p, _g5) = v.device_ptr(stream);
    let (al_p, _g6) = alpha.device_ptr(stream);
    let (be_p, _g7) = beta.device_ptr(stream);
    let (dy_p, _g8) = dy.device_ptr(stream);
    let (ds_p, _g9) = ds_t_inout.device_ptr_mut(stream);
    let (dq_p, _g10) = dq.device_ptr_mut(stream);
    let (dk_p, _g11) = dk.device_ptr_mut(stream);
    let (dv_p, _g12) = dv.device_ptr_mut(stream);
    let (da_p, _g13) = dalpha.device_ptr_mut(stream);
    let (db_p, _g14) = dbeta.device_ptr_mut(stream);

    let params: [*mut core::ffi::c_void; 15] = [
        &so_p as *const u64 as *mut _,
        &sn_p as *const u64 as *mut _,
        &q_p as *const u64 as *mut _,
        &k_p as *const u64 as *mut _,
        &v_p as *const u64 as *mut _,
        &al_p as *const u64 as *mut _,
        &be_p as *const u64 as *mut _,
        &dy_p as *const u64 as *mut _,
        &ds_p as *const u64 as *mut _,
        &dq_p as *const u64 as *mut _,
        &dk_p as *const u64 as *mut _,
        &dv_p as *const u64 as *mut _,
        &da_p as *const u64 as *mut _,
        &db_p as *const u64 as *mut _,
        &num_heads as *const u32 as *mut _,
    ];

    let result = unsafe {
        cuLaunchKernel(
            cu_func,
            num_heads,
            batch,
            1,
            128,
            1,
            1,
            SMEM_TOTAL as u32,
            stream.cu_stream(),
            params.as_ptr() as *mut _,
            core::ptr::null_mut(),
        )
    };
    if result != CUresult::CUDA_SUCCESS {
        return Err(SparkError::LaunchFailed(format!(
            "gdn_decode_backward launch: {:?}",
            result
        )));
    }
    Ok(())
}

/// Backward for `l2norm_scale_replicate_bf16` (Qwen3-Next / GDN-hybrid GDN Q/K norm).
///
/// Forward:
///   inv = scale / sqrt(sum_d(x[h, :]^2) + eps)
///   y[h*factor + r, d] = x[h, d] * inv     for r in 0..factor
///
/// Backward (this kernel):
///   dy_comb[h, d] = sum_{r=0..factor-1} dy[h*factor+r, d]
///   inner         = sum_d(x[h, d] * dy_comb[h, d])
///   dx[h, d]      = inv * dy_comb[h, d] - (inv / s2) * x[h, d] * inner
///
/// Inputs:
/// - `x`:  [num_heads_in, head_dim] BF16 — original forward input
/// - `dy`: [num_heads_in*factor, head_dim] BF16 — upstream gradient
/// Output:
/// - `dx`: [num_heads_in, head_dim] BF16
#[allow(clippy::too_many_arguments)]
pub fn l2norm_scale_replicate_backward_bf16(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<u16>,
    dy: &CudaSlice<u16>,
    dx: &mut CudaSlice<u16>,
    num_heads_in: u32,
    head_dim: u32,
    factor: u32,
    scale: f32,
    eps: f32,
) -> Result<()> {
    if num_heads_in == 0 || head_dim == 0 || factor == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    if !head_dim.is_multiple_of(32) || head_dim > 1024 {
        return Err(SparkError::InvalidArgument(format!(
            "head_dim must be multiple of 32 and ≤ 1024, got {head_dim}"
        )));
    }
    let in_need = (num_heads_in * head_dim) as usize;
    let dy_need = (num_heads_in * factor * head_dim) as usize;
    if x.len() < in_need || dx.len() < in_need {
        return Err(SparkError::InvalidArgument("x/dx buffer too small".into()));
    }
    if dy.len() < dy_need {
        return Err(SparkError::InvalidArgument("dy buffer too small".into()));
    }

    let func = module::load_kernel(
        ctx,
        "l2norm_scale_replicate_backward_bf16",
        "l2norm_scale_replicate_backward_bf16",
    )?;
    let cfg = LaunchConfig {
        grid_dim: (num_heads_in, 1, 1),
        block_dim: (head_dim, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(x)
            .arg(dy)
            .arg(dx)
            .arg(&head_dim)
            .arg(&factor)
            .arg(&scale)
            .arg(&eps)
            .launch(cfg)?;
    }
    Ok(())
}

/// Backward (dx) for depthwise causal conv1d with kernel_size=4.
///
/// Forward (training prefill mode, hypothetical — see GDN-bw plan):
///   y[t, c] = sum_{j=0..K-1} W[c, j] * x[t - (K-1-j), c]
///
/// Backward:
///   dx[t, c] = sum_{t' = t..min(t+K-1, T-1)} W[c, K-1-(t'-t)] * dy[t', c]
///
/// Note: this kernel computes the conv1d backward only — call
/// `silu_backward_bf16` first to convert dy_post_silu into dy_pre_silu, then
/// pass the dy_pre_silu to this kernel.
///
/// Inputs:
/// - `dy`: [T, C] BF16 — upstream gradient at the pre-SiLU position
/// - `w`:  [C, K=4] BF16 — depthwise conv weights
/// Output:
/// - `dx`: [T, C] BF16 — gradient wrt input
#[allow(clippy::too_many_arguments)]
pub fn conv1d_backward_dx_bf16(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    dy: &CudaSlice<u16>,
    w: &CudaSlice<u16>,
    dx: &mut CudaSlice<u16>,
    seq_len: u32,
    channels: u32,
) -> Result<()> {
    if seq_len == 0 || channels == 0 {
        return Err(SparkError::InvalidArgument(
            "seq_len/channels must be > 0".into(),
        ));
    }
    let need = (seq_len * channels) as usize;
    if dy.len() < need || dx.len() < need {
        return Err(SparkError::InvalidArgument("dy/dx buffer too small".into()));
    }
    if w.len() < (channels * 4) as usize {
        return Err(SparkError::InvalidArgument(
            "w buffer too small (need C*4)".into(),
        ));
    }

    let func = module::load_kernel(ctx, "conv1d_backward_dx_bf16", "conv1d_backward_dx_bf16")?;
    let cfg = LaunchConfig {
        grid_dim: (seq_len, channels.div_ceil(128), 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(dy)
            .arg(w)
            .arg(dx)
            .arg(&seq_len)
            .arg(&channels)
            .launch(cfg)?;
    }
    Ok(())
}

/// Backward (dW) for depthwise causal conv1d with kernel_size=4.
///
/// `dW[c, j] = sum_{t = K-1-j .. T-1} dy[t, c] * x[t - (K-1-j), c]`
///
/// Inputs:
/// - `dy`: [T, C] BF16 — upstream gradient at the pre-SiLU position
/// - `x`:  [T, C] BF16 — ORIGINAL conv1d input (must be cached during forward)
/// Output:
/// - `dw`: [C, K=4] BF16 — gradient wrt conv weights
#[allow(clippy::too_many_arguments)]
pub fn conv1d_backward_dw_bf16(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    dy: &CudaSlice<u16>,
    x: &CudaSlice<u16>,
    dw: &mut CudaSlice<u16>,
    seq_len: u32,
    channels: u32,
) -> Result<()> {
    if seq_len == 0 || channels == 0 {
        return Err(SparkError::InvalidArgument(
            "seq_len/channels must be > 0".into(),
        ));
    }
    let need = (seq_len * channels) as usize;
    if dy.len() < need || x.len() < need {
        return Err(SparkError::InvalidArgument("dy/x buffer too small".into()));
    }
    if dw.len() < (channels * 4) as usize {
        return Err(SparkError::InvalidArgument(
            "dw buffer too small (need C*4)".into(),
        ));
    }

    let func = module::load_kernel(ctx, "conv1d_backward_dw_bf16", "conv1d_backward_dw_bf16")?;
    let total = channels * 4;
    let cfg = LaunchConfig {
        grid_dim: (total.div_ceil(128), 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(dy)
            .arg(x)
            .arg(dw)
            .arg(&seq_len)
            .arg(&channels)
            .launch(cfg)?;
    }
    Ok(())
}

/// Launch Gated DeltaNet decode (BF16, single token).
///
/// Recurrent state update:
///   S_t = alpha * S_{t-1} + beta * (v - S_{t-1} @ k) * k^T
///   y_t = S_t @ q
///
/// State is updated in-place in `state` (FP32, [B, H, D, D]).
/// Used by Qwen3-Next-80B linear-attention layers (36 of 40).
#[allow(clippy::too_many_arguments)]
pub fn gdn_decode(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u16>,
    k: &CudaSlice<u16>,
    v: &CudaSlice<u16>,
    alpha: &CudaSlice<f32>,
    beta: &CudaSlice<f32>,
    state: &mut CudaSlice<u16>,
    y: &mut CudaSlice<u16>,
    batch: u32,
    num_heads: u32,
) -> Result<()> {
    if batch == 0 || num_heads == 0 {
        return Err(SparkError::InvalidArgument(
            "batch, num_heads must be > 0".into(),
        ));
    }
    let qkv_need = (batch * num_heads * GDN_HEAD_DIM) as usize;
    let scalar_need = (batch * num_heads) as usize;
    let state_need = (batch * num_heads * GDN_HEAD_DIM * GDN_HEAD_DIM) as usize;
    if q.len() < qkv_need || k.len() < qkv_need || v.len() < qkv_need || y.len() < qkv_need {
        return Err(SparkError::InvalidArgument(
            "GDN q/k/v/y buffer too small".into(),
        ));
    }
    if alpha.len() < scalar_need || beta.len() < scalar_need {
        return Err(SparkError::InvalidArgument(
            "GDN alpha/beta buffer too small".into(),
        ));
    }
    if state.len() < state_need {
        return Err(SparkError::InvalidArgument(format!(
            "GDN state buffer too small: need {state_need} (got {})",
            state.len()
        )));
    }

    use cudarc::driver::sys::*;

    let (q_ptr, _a) = q.device_ptr(stream);
    let (k_ptr, _b) = k.device_ptr(stream);
    let (v_ptr, _c) = v.device_ptr(stream);
    let (alpha_ptr, _al) = alpha.device_ptr(stream);
    let (beta_ptr, _be) = beta.device_ptr(stream);
    let (state_ptr, _st) = state.device_ptr(stream);
    let (y_ptr, _y) = y.device_ptr(stream);

    let cu_func = module::load_kernel_raw(ctx, "gdn_decode", "gdn_decode")?;
    let cu_stream = stream.cu_stream();

    // SMEM = 64KB state + 1.5KB qkv ≈ 67KB. Above 48KB default, requires opt-in.
    const SMEM_TOTAL: i32 = 67072;
    let attr_result = unsafe {
        cuFuncSetAttribute(
            cu_func,
            CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            SMEM_TOTAL,
        )
    };
    if attr_result != CUresult::CUDA_SUCCESS {
        return Err(SparkError::LaunchFailed(format!(
            "cuFuncSetAttribute: {:?}",
            attr_result
        )));
    }

    let params: [*mut core::ffi::c_void; 8] = [
        &q_ptr as *const u64 as *mut _,
        &k_ptr as *const u64 as *mut _,
        &v_ptr as *const u64 as *mut _,
        &alpha_ptr as *const u64 as *mut _,
        &beta_ptr as *const u64 as *mut _,
        &state_ptr as *const u64 as *mut _,
        &y_ptr as *const u64 as *mut _,
        &num_heads as *const u32 as *mut _,
    ];

    let result = unsafe {
        cuLaunchKernel(
            cu_func,
            num_heads,
            batch,
            1,
            128,
            1,
            1,
            SMEM_TOTAL as u32,
            cu_stream,
            params.as_ptr() as *mut _,
            core::ptr::null_mut(),
        )
    };

    if result != CUresult::CUDA_SUCCESS {
        return Err(SparkError::LaunchFailed(format!(
            "GDN launch: {:?}",
            result
        )));
    }
    Ok(())
}

/// SSM state dimension (`d_state`) for the Mamba2 selective-scan kernels.
pub const MAMBA2_D_STATE: u32 = 128;

/// Launch Mamba2 selective scan decode (single token).
/// Used by Nemotron-H and Mamba2 family models.
#[allow(clippy::too_many_arguments)]
pub fn mamba2_selective_scan_decode(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<f32>,
    delta: &CudaSlice<f32>,
    a_log: &CudaSlice<f32>,
    b: &CudaSlice<f32>,
    c: &CudaSlice<f32>,
    h: &mut CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    batch: u32,
    num_heads: u32,
) -> Result<()> {
    if batch == 0 || num_heads == 0 {
        return Err(SparkError::InvalidArgument(
            "batch, num_heads must be > 0".into(),
        ));
    }
    let scalar_need = (batch * num_heads) as usize;
    let state_need = (batch * num_heads * MAMBA2_D_STATE) as usize;
    if x.len() < scalar_need || delta.len() < scalar_need || y.len() < scalar_need {
        return Err(SparkError::InvalidArgument(
            "Mamba2 x/delta/y buffer too small".into(),
        ));
    }
    if a_log.len() < state_need
        || b.len() < state_need
        || c.len() < state_need
        || h.len() < state_need
    {
        return Err(SparkError::InvalidArgument(
            "Mamba2 state buffer too small".into(),
        ));
    }

    use cudarc::driver::sys::*;

    let (x_ptr, _a) = x.device_ptr(stream);
    let (delta_ptr, _b) = delta.device_ptr(stream);
    let (a_log_ptr, _c) = a_log.device_ptr(stream);
    let (b_ptr, _d) = b.device_ptr(stream);
    let (c_ptr, _e) = c.device_ptr(stream);
    let (h_ptr, _h) = h.device_ptr(stream);
    let (y_ptr, _y) = y.device_ptr(stream);

    let cu_func = module::load_kernel_raw(
        ctx,
        "mamba2_selective_scan_decode",
        "mamba2_selective_scan_decode",
    )?;
    let cu_stream = stream.cu_stream();

    let params: [*mut core::ffi::c_void; 8] = [
        &x_ptr as *const u64 as *mut _,
        &delta_ptr as *const u64 as *mut _,
        &a_log_ptr as *const u64 as *mut _,
        &b_ptr as *const u64 as *mut _,
        &c_ptr as *const u64 as *mut _,
        &h_ptr as *const u64 as *mut _,
        &y_ptr as *const u64 as *mut _,
        &num_heads as *const u32 as *mut _,
    ];

    let result = unsafe {
        cuLaunchKernel(
            cu_func,
            num_heads,
            batch,
            1,
            128,
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
            "Mamba2 launch: {:?}",
            result
        )));
    }
    Ok(())
}

/// Launch Gated DeltaNet prefill (sequential reference, seq_q > 1).
/// Keeps state resident in SMEM across all seq_q tokens in one kernel call.
/// Used by Qwen3-Next prefill path.
#[allow(clippy::too_many_arguments)]
pub fn gdn_prefill(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u16>,
    k: &CudaSlice<u16>,
    v: &CudaSlice<u16>,
    alpha: &CudaSlice<f32>,
    beta: &CudaSlice<f32>,
    state: &mut CudaSlice<u16>,
    y: &mut CudaSlice<u16>,
    batch: u32,
    num_heads: u32,
    seq_q: u32,
) -> Result<()> {
    if batch == 0 || num_heads == 0 || seq_q == 0 {
        return Err(SparkError::InvalidArgument(
            "batch, num_heads, seq_q must be > 0".into(),
        ));
    }

    use cudarc::driver::sys::*;

    let (q_ptr, _a) = q.device_ptr(stream);
    let (k_ptr, _b) = k.device_ptr(stream);
    let (v_ptr, _c) = v.device_ptr(stream);
    let (alpha_ptr, _al) = alpha.device_ptr(stream);
    let (beta_ptr, _be) = beta.device_ptr(stream);
    let (state_ptr, _st) = state.device_ptr(stream);
    let (y_ptr, _y) = y.device_ptr(stream);

    // HF-correctness gate (default OFF -> production byte-identical). When
    // SPARK_GDN_HF is set, dispatch the variant that applies the alpha decay to
    // the S.k term, matching HF transformers' recurrent_gated_delta_rule and the
    // gdn_decode kernel. The default kernel omits that decay and diverges from HF
    // over the prefill sequence (see NVIDIA_REVIEW_FINDINGS.md F3). Enabling this
    // changes served outputs and requires a model re-eval before deployment.
    let (kname, kentry) = if std::env::var("SPARK_GDN_HF").is_ok() {
        ("gdn_prefill_hf", "gdn_prefill_hf")
    } else {
        ("gdn_prefill", "gdn_prefill")
    };
    let cu_func = module::load_kernel_raw(ctx, kname, kentry)?;
    let cu_stream = stream.cu_stream();

    const SMEM_TOTAL: i32 = 67072;
    let attr_result = unsafe {
        cuFuncSetAttribute(
            cu_func,
            CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            SMEM_TOTAL,
        )
    };
    if attr_result != CUresult::CUDA_SUCCESS {
        return Err(SparkError::LaunchFailed(format!(
            "cuFuncSetAttribute: {:?}",
            attr_result
        )));
    }

    let params: [*mut core::ffi::c_void; 9] = [
        &q_ptr as *const u64 as *mut _,
        &k_ptr as *const u64 as *mut _,
        &v_ptr as *const u64 as *mut _,
        &alpha_ptr as *const u64 as *mut _,
        &beta_ptr as *const u64 as *mut _,
        &state_ptr as *const u64 as *mut _,
        &y_ptr as *const u64 as *mut _,
        &num_heads as *const u32 as *mut _,
        &seq_q as *const u32 as *mut _,
    ];

    let result = unsafe {
        cuLaunchKernel(
            cu_func,
            num_heads,
            batch,
            1,
            128,
            1,
            1,
            SMEM_TOTAL as u32,
            cu_stream,
            params.as_ptr() as *mut _,
            core::ptr::null_mut(),
        )
    };

    if result != CUresult::CUDA_SUCCESS {
        return Err(SparkError::LaunchFailed(format!(
            "GDN prefill launch: {:?}",
            result
        )));
    }
    Ok(())
}

/// Launch Mamba2 selective scan prefill (sequential reference).
/// Holds state element per thread across all seq_q tokens in one kernel call.
#[allow(clippy::too_many_arguments)]
pub fn mamba2_selective_scan_prefill(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<f32>,
    delta: &CudaSlice<f32>,
    a_log: &CudaSlice<f32>,
    b: &CudaSlice<f32>,
    c: &CudaSlice<f32>,
    h: &mut CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    batch: u32,
    num_heads: u32,
    seq_q: u32,
) -> Result<()> {
    if batch == 0 || num_heads == 0 || seq_q == 0 {
        return Err(SparkError::InvalidArgument(
            "batch, num_heads, seq_q must be > 0".into(),
        ));
    }

    use cudarc::driver::sys::*;

    let (x_ptr, _a) = x.device_ptr(stream);
    let (delta_ptr, _b) = delta.device_ptr(stream);
    let (a_log_ptr, _c) = a_log.device_ptr(stream);
    let (b_ptr, _d) = b.device_ptr(stream);
    let (c_ptr, _e) = c.device_ptr(stream);
    let (h_ptr, _h) = h.device_ptr(stream);
    let (y_ptr, _y) = y.device_ptr(stream);

    let cu_func = module::load_kernel_raw(
        ctx,
        "mamba2_selective_scan_prefill",
        "mamba2_selective_scan_prefill",
    )?;
    let cu_stream = stream.cu_stream();

    let params: [*mut core::ffi::c_void; 9] = [
        &x_ptr as *const u64 as *mut _,
        &delta_ptr as *const u64 as *mut _,
        &a_log_ptr as *const u64 as *mut _,
        &b_ptr as *const u64 as *mut _,
        &c_ptr as *const u64 as *mut _,
        &h_ptr as *const u64 as *mut _,
        &y_ptr as *const u64 as *mut _,
        &num_heads as *const u32 as *mut _,
        &seq_q as *const u32 as *mut _,
    ];

    let result = unsafe {
        cuLaunchKernel(
            cu_func,
            num_heads,
            batch,
            1,
            128,
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
            "Mamba2 prefill launch: {:?}",
            result
        )));
    }
    Ok(())
}

/// Create an FP32 2D TMA descriptor for GDN state.
/// View: [B*H*D, D] — so one tile of [D, D] = the full state matrix for
/// one (batch, head). Coord = (0, (b*H + h) * D).
fn create_tma_desc_fp32(
    global_ptr: *mut core::ffi::c_void,
    total_rows: u32,
    inner_cols: u32,
) -> Result<cudarc::driver::sys::CUtensorMap> {
    use cudarc::driver::sys::*;

    let mut tma = CUtensorMap_st::default();
    let global_dim: [u64; 2] = [inner_cols as u64, total_rows as u64];
    let global_strides: [u64; 1] = [(inner_cols as u64) * 4];
    let box_dim: [u32; 2] = [inner_cols, inner_cols]; // square tile [D, D]
    let elem_strides: [u32; 2] = [1, 1];

    let result = unsafe {
        cuTensorMapEncodeTiled(
            &mut tma,
            CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
            2,
            global_ptr,
            global_dim.as_ptr(),
            global_strides.as_ptr(),
            box_dim.as_ptr(),
            elem_strides.as_ptr(),
            CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
            CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
            CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        )
    };
    if result != CUresult::CUDA_SUCCESS {
        return Err(SparkError::LaunchFailed(format!(
            "TMA FP32 encode: {:?}",
            result
        )));
    }
    Ok(tma)
}

/// TMA-accelerated Gated DeltaNet decode.
/// Same math as `gdn_decode` but uses a single 64KB TMA bulk transfer for
/// state load (and TMA bulk store for state write-back), replacing 128×128
/// scalar `ld.global.f32`s.
#[allow(clippy::too_many_arguments)]
pub fn gdn_decode_tma(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u16>,
    k: &CudaSlice<u16>,
    v: &CudaSlice<u16>,
    alpha: &CudaSlice<f32>,
    beta: &CudaSlice<f32>,
    state: &mut CudaSlice<f32>,
    y: &mut CudaSlice<u16>,
    batch: u32,
    num_heads: u32,
) -> Result<()> {
    if batch == 0 || num_heads == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }

    use cudarc::driver::sys::*;

    let (q_ptr, _a) = q.device_ptr(stream);
    let (k_ptr, _b) = k.device_ptr(stream);
    let (v_ptr, _c) = v.device_ptr(stream);
    let (alpha_ptr, _al) = alpha.device_ptr(stream);
    let (beta_ptr, _be) = beta.device_ptr(stream);
    let (state_ptr, _st) = state.device_ptr(stream);
    let (y_ptr, _y) = y.device_ptr(stream);

    let total_rows = batch * num_heads * GDN_HEAD_DIM;
    let state_tma = create_tma_desc_fp32(state_ptr as *mut _, total_rows, GDN_HEAD_DIM)?;
    let state_tma_u32: [u32; 32] = unsafe { core::mem::transmute(state_tma) };
    let state_tma_dev = stream
        .memcpy_stod(&state_tma_u32)
        .map_err(SparkError::Driver)?;
    let (state_tma_dptr, _) = state_tma_dev.device_ptr(stream);

    let cu_func = module::load_kernel_raw(ctx, "gdn_decode_tma", "gdn_decode_tma")?;
    let cu_stream = stream.cu_stream();

    const SMEM_TOTAL: i32 = 67088;
    let attr_result = unsafe {
        cuFuncSetAttribute(
            cu_func,
            CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            SMEM_TOTAL,
        )
    };
    if attr_result != CUresult::CUDA_SUCCESS {
        return Err(SparkError::LaunchFailed(format!(
            "cuFuncSetAttribute: {:?}",
            attr_result
        )));
    }

    let params: [*mut core::ffi::c_void; 9] = [
        &q_ptr as *const u64 as *mut _,
        &k_ptr as *const u64 as *mut _,
        &v_ptr as *const u64 as *mut _,
        &alpha_ptr as *const u64 as *mut _,
        &beta_ptr as *const u64 as *mut _,
        &state_tma_dptr as *const u64 as *mut _,
        &state_ptr as *const u64 as *mut _, // raw ptr also passed (used for TMA store coord)
        &y_ptr as *const u64 as *mut _,
        &num_heads as *const u32 as *mut _,
    ];

    let result = unsafe {
        cuLaunchKernel(
            cu_func,
            num_heads,
            batch,
            1,
            128,
            1,
            1,
            SMEM_TOTAL as u32,
            cu_stream,
            params.as_ptr() as *mut _,
            core::ptr::null_mut(),
        )
    };

    if result != CUresult::CUDA_SUCCESS {
        return Err(SparkError::LaunchFailed(format!(
            "GDN TMA launch: {:?}",
            result
        )));
    }
    Ok(())
}

/// Launch linear attention chunk-parallel prefill (one chunk).
///
/// Computes one chunk of linear attention with state passing:
///   For all i in [0, C):
///     S_{c0+i} = S_{c0+i-1} + v_{c0+i} k_{c0+i}^T
///     y_{c0+i} = S_{c0+i} q_{c0+i}
///
/// The chunk is processed in parallel via the algebraic decomposition
/// `Y = lowerTri(Q K^T) V + Q S_init^T` and `S_new = S_init + V^T K`. Each
/// CTA processes one (batch, head) for one chunk; host loops chunks
/// sequentially.
///
/// First iteration: scalar implementation (correct, not yet MMA-accelerated).
/// MMA acceleration is the next pass — same algorithm, MMAs replace the inner
/// scalar loops.
///
/// Layouts (per chunk, C=32, D=128):
/// - `k`, `v`, `q`: `[B, H, C, D]` BF16
/// - `y`: `[B, H, C, D]` BF16 (output)
/// - `s_in`: `[B, H, D, D]` FP32 (state at chunk start)
/// - `s_out`: `[B, H, D, D]` FP32 (state at chunk end; can alias s_in)
#[allow(clippy::too_many_arguments)]
pub fn linear_attn_chunk_prefill(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    k: &CudaSlice<u16>,
    v: &CudaSlice<u16>,
    q: &CudaSlice<u16>,
    y: &mut CudaSlice<u16>,
    s_in: &CudaSlice<u16>,
    s_out: &mut CudaSlice<u16>,
    batch: u32,
    num_heads: u32,
) -> Result<()> {
    if batch == 0 || num_heads == 0 {
        return Err(SparkError::InvalidArgument(
            "batch, num_heads must be > 0".into(),
        ));
    }

    use cudarc::driver::sys::*;

    let (k_ptr, _k1) = k.device_ptr(stream);
    let (v_ptr, _v1) = v.device_ptr(stream);
    let (q_ptr, _q1) = q.device_ptr(stream);
    let (y_ptr, _y1) = y.device_ptr(stream);
    let (s_in_ptr, _s1) = s_in.device_ptr(stream);
    let (s_out_ptr, _s2) = s_out.device_ptr(stream);

    let cu_func = module::load_kernel_raw(
        ctx,
        "linear_attn_chunk_prefill",
        "linear_attn_chunk_prefill",
    )?;
    let cu_stream = stream.cu_stream();

    const SMEM_TOTAL: i32 = 90112;
    let attr = unsafe {
        cuFuncSetAttribute(
            cu_func,
            CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            SMEM_TOTAL,
        )
    };
    if attr != CUresult::CUDA_SUCCESS {
        return Err(SparkError::LaunchFailed(format!(
            "cuFuncSetAttribute: {:?}",
            attr
        )));
    }

    let params: [*mut core::ffi::c_void; 7] = [
        &k_ptr as *const u64 as *mut _,
        &v_ptr as *const u64 as *mut _,
        &q_ptr as *const u64 as *mut _,
        &y_ptr as *const u64 as *mut _,
        &s_in_ptr as *const u64 as *mut _,
        &s_out_ptr as *const u64 as *mut _,
        &num_heads as *const u32 as *mut _,
    ];

    let result = unsafe {
        cuLaunchKernel(
            cu_func,
            num_heads,
            batch,
            1,
            128,
            1,
            1,
            SMEM_TOTAL as u32,
            cu_stream,
            params.as_ptr() as *mut _,
            core::ptr::null_mut(),
        )
    };
    if result != CUresult::CUDA_SUCCESS {
        return Err(SparkError::LaunchFailed(format!(
            "linear_attn_chunk_prefill launch: {:?}",
            result,
        )));
    }
    Ok(())
}

/// MMA-accelerated state update for linear attention chunk-scan:
///   S_new = S_init + V^T @ K
/// where V, K: [B, H, C=32, D=128] BF16; S_init/S_new: [B, H, D, D] FP32.
///
/// Standalone primitive — replaces the scalar V^T@K loop in the chunk-scan
/// kernel. Use as: scalar `linear_attn_chunk_prefill` for Y, then this for S.
///
/// 4 warps in 2×2 layout, 80 KB SMEM. 64 BF16 m16n8k16 MMAs per warp.
#[allow(clippy::too_many_arguments)]
pub fn linear_attn_state_update_mma(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    v: &CudaSlice<u16>,
    k: &CudaSlice<u16>,
    s_in: &CudaSlice<u16>,
    s_out: &mut CudaSlice<u16>,
    batch: u32,
    num_heads: u32,
) -> Result<()> {
    if batch == 0 || num_heads == 0 {
        return Err(SparkError::InvalidArgument(
            "batch, num_heads must be > 0".into(),
        ));
    }
    use cudarc::driver::sys::*;

    let (v_ptr, _v1) = v.device_ptr(stream);
    let (k_ptr, _k1) = k.device_ptr(stream);
    let (s_in_ptr, _s1) = s_in.device_ptr(stream);
    let (s_out_ptr, _s2) = s_out.device_ptr(stream);

    let cu_func = module::load_kernel_raw(
        ctx,
        "linear_attn_state_update_mma",
        "linear_attn_state_update_mma",
    )?;
    let cu_stream = stream.cu_stream();

    const SMEM_TOTAL: i32 = 81920;
    let attr = unsafe {
        cuFuncSetAttribute(
            cu_func,
            CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            SMEM_TOTAL,
        )
    };
    if attr != CUresult::CUDA_SUCCESS {
        return Err(SparkError::LaunchFailed(format!(
            "cuFuncSetAttribute: {:?}",
            attr
        )));
    }

    let params: [*mut core::ffi::c_void; 5] = [
        &v_ptr as *const u64 as *mut _,
        &k_ptr as *const u64 as *mut _,
        &s_in_ptr as *const u64 as *mut _,
        &s_out_ptr as *const u64 as *mut _,
        &num_heads as *const u32 as *mut _,
    ];

    let result = unsafe {
        cuLaunchKernel(
            cu_func,
            num_heads,
            batch,
            1,
            128,
            1,
            1,
            SMEM_TOTAL as u32,
            cu_stream,
            params.as_ptr() as *mut _,
            core::ptr::null_mut(),
        )
    };
    if result != CUresult::CUDA_SUCCESS {
        return Err(SparkError::LaunchFailed(format!(
            "linear_attn_state_update_mma launch: {:?}",
            result,
        )));
    }
    Ok(())
}

/// MMA-accelerated linear attention inter-chunk output:
///   Y_inter = Q @ S^T
/// Q: [B, H, C=32, D=128] BF16; S: [B, H, D, D] FP32; Y_inter: [B, H, C, D] FP32.
///
/// Together with `linear_attn_state_update_mma`, captures ~80% of per-chunk
/// FLOPS in MMA. Caller adds Y_intra (scalar) + Y_inter and converts to BF16.
///
/// Precision note: S downcast to BF16 in SMEM during transpose-load. ~16
/// mantissa bits lost per chunk; final Y converts to BF16 anyway so the
/// effective loss is at the same point. State S in global stays FP32.
#[allow(clippy::too_many_arguments)]
pub fn linear_attn_y_inter_mma(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u16>,
    s: &CudaSlice<u16>,     // [B, H, D, D] FP16 state
    y: &mut CudaSlice<f32>, // [B, H, C, D] FP32 output
    batch: u32,
    num_heads: u32,
) -> Result<()> {
    if batch == 0 || num_heads == 0 {
        return Err(SparkError::InvalidArgument(
            "batch, num_heads must be > 0".into(),
        ));
    }
    use cudarc::driver::sys::*;

    let (q_ptr, _q1) = q.device_ptr(stream);
    let (s_ptr, _s1) = s.device_ptr(stream);
    let (y_ptr, _y1) = y.device_ptr(stream);

    let cu_func =
        module::load_kernel_raw(ctx, "linear_attn_y_inter_mma", "linear_attn_y_inter_mma")?;
    let cu_stream = stream.cu_stream();

    const SMEM_TOTAL: i32 = 40960;
    let attr = unsafe {
        cuFuncSetAttribute(
            cu_func,
            CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            SMEM_TOTAL,
        )
    };
    if attr != CUresult::CUDA_SUCCESS {
        return Err(SparkError::LaunchFailed(format!(
            "cuFuncSetAttribute: {:?}",
            attr
        )));
    }

    let params: [*mut core::ffi::c_void; 4] = [
        &q_ptr as *const u64 as *mut _,
        &s_ptr as *const u64 as *mut _,
        &y_ptr as *const u64 as *mut _,
        &num_heads as *const u32 as *mut _,
    ];

    let result = unsafe {
        cuLaunchKernel(
            cu_func,
            num_heads,
            batch,
            1,
            128,
            1,
            1,
            SMEM_TOTAL as u32,
            cu_stream,
            params.as_ptr() as *mut _,
            core::ptr::null_mut(),
        )
    };
    if result != CUresult::CUDA_SUCCESS {
        return Err(SparkError::LaunchFailed(format!(
            "linear_attn_y_inter_mma launch: {:?}",
            result,
        )));
    }
    Ok(())
}
