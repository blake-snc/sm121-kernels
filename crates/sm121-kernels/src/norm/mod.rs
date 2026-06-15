use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};

use crate::error::{Result, SparkError};
use crate::module;

/// Launch RMSNorm BF16 kernel.
///
/// `x`: input tensor [num_rows, hidden_dim] in BF16 (stored as u16)
/// `out`: output tensor [num_rows, hidden_dim] in BF16
/// `weight`: weight tensor [hidden_dim] in BF16
/// `hidden_dim`: size of last dimension
/// `eps`: epsilon for numerical stability
/// `num_rows`: number of rows to normalize
#[allow(clippy::too_many_arguments)]
pub fn rmsnorm_bf16(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<u16>,
    out: &mut CudaSlice<u16>,
    weight: &CudaSlice<u16>,
    hidden_dim: u32,
    eps: f32,
    num_rows: u32,
) -> Result<()> {
    if hidden_dim == 0 {
        return Err(SparkError::InvalidArgument("hidden_dim must be > 0".into()));
    }
    if num_rows == 0 {
        return Err(SparkError::InvalidArgument("num_rows must be > 0".into()));
    }
    let need = num_rows as usize * hidden_dim as usize;
    if x.len() < need {
        return Err(SparkError::InvalidArgument(format!(
            "x buffer too small: {} < {need}",
            x.len()
        )));
    }
    if out.len() < need {
        return Err(SparkError::InvalidArgument(format!(
            "out buffer too small: {} < {need}",
            out.len()
        )));
    }
    if weight.len() < hidden_dim as usize {
        return Err(SparkError::InvalidArgument(format!(
            "weight buffer too small: {} < {hidden_dim}",
            weight.len()
        )));
    }
    let func = module::load_kernel(ctx, "rmsnorm_bf16", "rmsnorm_bf16")?;

    let threads_per_block: u32 = 256;
    let cfg = LaunchConfig {
        grid_dim: (num_rows, 1, 1),
        block_dim: (threads_per_block, 1, 1),
        shared_mem_bytes: 0, // Static shared memory declared in PTX
    };

    unsafe {
        stream
            .launch_builder(&func)
            .arg(x)
            .arg(out)
            .arg(weight)
            .arg(&hidden_dim)
            .arg(&eps)
            .launch(cfg)?;
    }

    Ok(())
}

/// Backward pass for `rmsnorm_bf16`.
///
/// For `y = x * rsqrt(mean(x^2) + eps) * weight`, computes:
/// - `dx[r, i] = r * (g[r, i] - x[r, i] * r^2 * mean(g[r, :] * x[r, :]))`
/// - `dweight[i] = sum_over_rows(dy[r, i] * x[r, i] * r[r])`
///
/// where `r[r] = rsqrt(mean(x[r, :]^2) + eps)` and `g = dy * weight`.
///
/// `x`, `weight`, `dy`: same shapes as the forward.
/// `dx`: BF16 [num_rows, hidden_dim] (output, written densely).
/// `dweight`: **f32** [hidden_dim] (output via atomicAdd — caller MUST
///   `memset_zeros` before calling so the accumulation starts from 0).
///
/// Why dweight is f32: BF16 atomicAdd loses precision on the accumulator
/// when many rows contribute. Caller can do a separate f32→bf16 cast
/// after the kernel returns if BF16 dweight is needed.
#[allow(clippy::too_many_arguments)]
pub fn rmsnorm_backward_bf16(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<u16>,
    weight: &CudaSlice<u16>,
    dy: &CudaSlice<u16>,
    dx: &mut CudaSlice<u16>,
    dweight: &mut CudaSlice<f32>,
    hidden_dim: u32,
    eps: f32,
    num_rows: u32,
) -> Result<()> {
    if hidden_dim == 0 {
        return Err(SparkError::InvalidArgument("hidden_dim must be > 0".into()));
    }
    if num_rows == 0 {
        return Err(SparkError::InvalidArgument("num_rows must be > 0".into()));
    }
    let need = num_rows as usize * hidden_dim as usize;
    if x.len() < need || dy.len() < need || dx.len() < need {
        return Err(SparkError::InvalidArgument(format!(
            "buffers too small: x/dy/dx need {need}"
        )));
    }
    if weight.len() < hidden_dim as usize {
        return Err(SparkError::InvalidArgument(format!(
            "weight buffer too small: {} < {hidden_dim}",
            weight.len()
        )));
    }
    if dweight.len() < hidden_dim as usize {
        return Err(SparkError::InvalidArgument(format!(
            "dweight buffer too small: {} < {hidden_dim}",
            dweight.len()
        )));
    }

    let func = module::load_kernel(ctx, "rmsnorm_backward_bf16", "rmsnorm_backward_bf16")?;
    let threads_per_block: u32 = 256;
    let cfg = LaunchConfig {
        grid_dim: (num_rows, 1, 1),
        block_dim: (threads_per_block, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(x)
            .arg(weight)
            .arg(dy)
            .arg(dx)
            .arg(dweight)
            .arg(&hidden_dim)
            .arg(&eps)
            .launch(cfg)?;
    }
    Ok(())
}

/// Launch fused residual + RMSNorm BF16 kernel.
///
/// Operation:
///   residual[i] := residual[i] + x[i]         (in-place residual update)
///   out[i]      := (residual[i] / rms) * weight[i]
///
/// Saves one kernel launch and one global-memory round-trip per transformer block
/// compared to separate residual-add and RMSNorm launches.
///
/// `x`: input contribution [num_rows, hidden_dim] BF16 (read-only)
/// `residual`: running residual [num_rows, hidden_dim] BF16 (updated in-place)
/// `out`: normalized output [num_rows, hidden_dim] BF16
/// `weight`: [hidden_dim] BF16
#[allow(clippy::too_many_arguments)]
pub fn rmsnorm_residual_bf16(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<u16>,
    residual: &mut CudaSlice<u16>,
    out: &mut CudaSlice<u16>,
    weight: &CudaSlice<u16>,
    hidden_dim: u32,
    eps: f32,
    num_rows: u32,
) -> Result<()> {
    if hidden_dim == 0 {
        return Err(SparkError::InvalidArgument("hidden_dim must be > 0".into()));
    }
    if num_rows == 0 {
        return Err(SparkError::InvalidArgument("num_rows must be > 0".into()));
    }
    let need = num_rows as usize * hidden_dim as usize;
    for (name, buf_len) in [
        ("x", x.len()),
        ("residual", residual.len()),
        ("out", out.len()),
    ] {
        if buf_len < need {
            return Err(SparkError::InvalidArgument(format!(
                "{name} buffer too small: {buf_len} < {need}"
            )));
        }
    }
    if weight.len() < hidden_dim as usize {
        return Err(SparkError::InvalidArgument(format!(
            "weight buffer too small: {} < {hidden_dim}",
            weight.len()
        )));
    }

    let func = module::load_kernel(ctx, "rmsnorm_residual_bf16", "rmsnorm_residual_bf16")?;

    let threads_per_block: u32 = 256;
    let cfg = LaunchConfig {
        grid_dim: (num_rows, 1, 1),
        block_dim: (threads_per_block, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&func)
            .arg(x)
            .arg(residual)
            .arg(out)
            .arg(weight)
            .arg(&hidden_dim)
            .arg(&eps)
            .launch(cfg)?;
    }

    Ok(())
}

/// Launch RMSNorm with FP8 E4M3 output for quantized pipelines.
///
/// `tmp[i] = (x[i] / rms) * weight[i]` (F32)
/// `out[i] = round_to_e4m3(tmp[i] * inv_scale)` (saturating)
///
/// `x`: input [num_rows, hidden_dim] BF16
/// `out`: output [num_rows, hidden_dim] FP8 E4M3 (as u8)
/// `weight`: [hidden_dim] BF16
/// `inv_scale`: pre-FP8 multiplier (typically 1 / fp8_scale where fp8_scale is
///              the scale the downstream FP8 kernel will use to dequantize)
#[allow(clippy::too_many_arguments)]
pub fn rmsnorm_bf16_fp8out(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<u16>,
    out: &mut CudaSlice<u8>,
    weight: &CudaSlice<u16>,
    hidden_dim: u32,
    eps: f32,
    inv_scale: f32,
    num_rows: u32,
) -> Result<()> {
    if hidden_dim == 0 || num_rows == 0 {
        return Err(SparkError::InvalidArgument(
            "hidden_dim and num_rows must be > 0".into(),
        ));
    }
    if hidden_dim & 1 != 0 {
        return Err(SparkError::InvalidArgument(
            "hidden_dim must be even for FP8 output".into(),
        ));
    }
    let need = num_rows as usize * hidden_dim as usize;
    if x.len() < need || out.len() < need {
        return Err(SparkError::InvalidArgument(format!(
            "buffers too small: need {need}"
        )));
    }
    if weight.len() < hidden_dim as usize {
        return Err(SparkError::InvalidArgument("weight too small".into()));
    }

    let func = module::load_kernel(ctx, "rmsnorm_bf16_fp8out", "rmsnorm_bf16_fp8out")?;
    let cfg = LaunchConfig {
        grid_dim: (num_rows, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(x)
            .arg(out)
            .arg(weight)
            .arg(&hidden_dim)
            .arg(&eps)
            .arg(&inv_scale)
            .launch(cfg)?;
    }
    Ok(())
}

/// Launch RMSNorm with per-token dynamic FP8 E4M3 output (DeepSeek V3 pattern).
///
/// For each row:
///   rms         = sqrt(mean(x^2) + eps)
///   tmp[i]      = (x[i] / rms) * weight[i]     (FP32)
///   row_max     = max |tmp|
///   row_scale   = row_max / 448.0              (written to `scales[row]`)
///   out[i]      = round_to_e4m3(tmp[i] / row_scale)
///
/// Consumer applies `row_scale` at dequant time (FP8 GEMM / attention).
///
/// `hidden_dim` must be even and ≤ 8192 (SMEM capacity constraint).
#[allow(clippy::too_many_arguments)]
pub fn rmsnorm_bf16_fp8out_pertoken(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<u16>,
    out: &mut CudaSlice<u8>,
    weight: &CudaSlice<u16>,
    scales: &mut CudaSlice<f32>,
    hidden_dim: u32,
    eps: f32,
    num_rows: u32,
) -> Result<()> {
    if hidden_dim == 0 || num_rows == 0 {
        return Err(SparkError::InvalidArgument(
            "hidden_dim and num_rows must be > 0".into(),
        ));
    }
    if hidden_dim & 1 != 0 {
        return Err(SparkError::InvalidArgument(
            "hidden_dim must be even for FP8 output".into(),
        ));
    }
    if hidden_dim > 8192 {
        return Err(SparkError::InvalidArgument(
            "hidden_dim must be ≤ 8192 (SMEM staging cap)".into(),
        ));
    }
    let need = num_rows as usize * hidden_dim as usize;
    if x.len() < need || out.len() < need {
        return Err(SparkError::InvalidArgument(format!(
            "buffers too small: need {need}"
        )));
    }
    if weight.len() < hidden_dim as usize {
        return Err(SparkError::InvalidArgument("weight too small".into()));
    }
    if scales.len() < num_rows as usize {
        return Err(SparkError::InvalidArgument("scales too small".into()));
    }

    let func = module::load_kernel(
        ctx,
        "rmsnorm_bf16_fp8out_pertoken",
        "rmsnorm_bf16_fp8out_pertoken",
    )?;
    let cfg = LaunchConfig {
        grid_dim: (num_rows, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(x)
            .arg(out)
            .arg(weight)
            .arg(scales)
            .arg(&hidden_dim)
            .arg(&eps)
            .launch(cfg)?;
    }
    Ok(())
}

/// In-place variant of `gated_rmsnorm_silu_bf16`: writes the gated output
/// back into the `y` buffer. Alias-safe per kernel design (each thread reads
/// its own slot of y and z, computes locally, writes its own slot of out).
/// Uses raw `cuLaunchKernel` to bypass cudarc's safe-API aliasing check.
pub fn gated_rmsnorm_silu_bf16_inplace(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    y: &mut CudaSlice<u16>,
    z: &CudaSlice<u16>,
    weight: &CudaSlice<u16>,
    num_heads: u32,
    head_dim: u32,
    eps: f32,
) -> Result<()> {
    use cudarc::driver::sys::*;
    use cudarc::driver::DevicePtr;

    if num_heads == 0 || head_dim == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    if !head_dim.is_multiple_of(32) || head_dim > 1024 {
        return Err(SparkError::InvalidArgument(format!(
            "head_dim must be multiple of 32 and ≤ 1024, got {head_dim}"
        )));
    }
    let need = (num_heads * head_dim) as usize;
    if y.len() < need || z.len() < need {
        return Err(SparkError::InvalidArgument("y/z buffer too small".into()));
    }
    if weight.len() < head_dim as usize {
        return Err(SparkError::InvalidArgument(
            "weight buffer too small".into(),
        ));
    }

    let cu_func =
        crate::module::load_kernel_raw(ctx, "gated_rmsnorm_silu_bf16", "gated_rmsnorm_silu_bf16")?;
    let cu_stream = stream.cu_stream();
    let (y_ptr, _g1) = y.device_ptr(stream);
    let (z_ptr, _g2) = z.device_ptr(stream);
    let (w_ptr, _g3) = weight.device_ptr(stream);
    let params: [*mut core::ffi::c_void; 6] = [
        &y_ptr as *const u64 as *mut _,
        &z_ptr as *const u64 as *mut _,
        &w_ptr as *const u64 as *mut _,
        &y_ptr as *const u64 as *mut _, // out aliases y
        &head_dim as *const u32 as *mut _,
        &eps as *const f32 as *mut _,
    ];
    let r = unsafe {
        cuLaunchKernel(
            cu_func,
            num_heads,
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
    if r != CUresult::CUDA_SUCCESS {
        return Err(SparkError::LaunchFailed(format!(
            "gated_rmsnorm_silu_bf16_inplace: {:?}",
            r
        )));
    }
    Ok(())
}

/// Qwen3-Next GatedRMSNorm with silu(z) gate, applied AFTER GDN recurrence:
///   out[h, d] = (y[h, d] / rms_h) * weight[d] * silu(z[h, d])
/// where rms_h = sqrt(mean(y[h, :]^2) + eps), per V head independently;
/// weight is shared across heads (length head_dim); silu(x) = x * sigmoid(x).
///
/// Shapes (BF16 stored as u16):
///   y:      [num_heads, head_dim]
///   z:      [num_heads, head_dim]
///   weight: [head_dim]                (broadcast across heads)
///   out:    [num_heads, head_dim]
///
/// Launch: 1 block per head, head_dim threads per block. head_dim must be a
/// multiple of 32 and ≤ 1024.
#[allow(clippy::too_many_arguments)]
pub fn gated_rmsnorm_silu_bf16(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    y: &CudaSlice<u16>,
    z: &CudaSlice<u16>,
    weight: &CudaSlice<u16>,
    out: &mut CudaSlice<u16>,
    num_heads: u32,
    head_dim: u32,
    eps: f32,
) -> Result<()> {
    if num_heads == 0 || head_dim == 0 {
        return Err(SparkError::InvalidArgument(
            "num_heads, head_dim must be > 0".into(),
        ));
    }
    if !head_dim.is_multiple_of(32) || head_dim > 1024 {
        return Err(SparkError::InvalidArgument(format!(
            "head_dim must be multiple of 32 and ≤ 1024, got {head_dim}"
        )));
    }
    let need = (num_heads * head_dim) as usize;
    if y.len() < need || z.len() < need || out.len() < need {
        return Err(SparkError::InvalidArgument(
            "y/z/out buffer too small".into(),
        ));
    }
    if weight.len() < head_dim as usize {
        return Err(SparkError::InvalidArgument(
            "weight buffer too small".into(),
        ));
    }

    let func = module::load_kernel(ctx, "gated_rmsnorm_silu_bf16", "gated_rmsnorm_silu_bf16")?;

    let cfg = LaunchConfig {
        grid_dim: (num_heads, 1, 1),
        block_dim: (head_dim, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(y)
            .arg(z)
            .arg(weight)
            .arg(out)
            .arg(&head_dim)
            .arg(&eps)
            .launch(cfg)?;
    }
    Ok(())
}

/// Backward for `gated_rmsnorm_silu_bf16` — computes `dy` and `dz`.
///
/// Forward: `out = (y/rms_h) * weight * silu(z)`, per head independently.
/// This kernel computes `dy` and `dz`; the `dweight` reduction is in a
/// separate kernel (`gated_rmsnorm_silu_backward_dweight_bf16`) since
/// it's cross-head.
///
/// Inputs (all BF16 unless noted):
/// - `y`, `z`: `[num_heads, head_dim]` — original forward inputs
/// - `weight`: `[head_dim]` — original forward weight
/// - `dout`: `[num_heads, head_dim]` — upstream gradient
/// Outputs:
/// - `dy`, `dz`: `[num_heads, head_dim]`
#[allow(clippy::too_many_arguments)]
pub fn gated_rmsnorm_silu_backward_bf16(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    y: &CudaSlice<u16>,
    z: &CudaSlice<u16>,
    weight: &CudaSlice<u16>,
    dout: &CudaSlice<u16>,
    dy: &mut CudaSlice<u16>,
    dz: &mut CudaSlice<u16>,
    num_heads: u32,
    head_dim: u32,
    eps: f32,
) -> Result<()> {
    if num_heads == 0 || head_dim == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    if !head_dim.is_multiple_of(32) || head_dim > 1024 {
        return Err(SparkError::InvalidArgument(format!(
            "head_dim must be multiple of 32 and ≤ 1024, got {head_dim}"
        )));
    }
    let need = (num_heads * head_dim) as usize;
    if y.len() < need || z.len() < need || dout.len() < need || dy.len() < need || dz.len() < need {
        return Err(SparkError::InvalidArgument(
            "y/z/dout/dy/dz buffer too small".into(),
        ));
    }
    if weight.len() < head_dim as usize {
        return Err(SparkError::InvalidArgument(
            "weight buffer too small".into(),
        ));
    }

    let func = module::load_kernel(
        ctx,
        "gated_rmsnorm_silu_backward_bf16",
        "gated_rmsnorm_silu_backward_bf16",
    )?;
    let cfg = LaunchConfig {
        grid_dim: (num_heads, 1, 1),
        block_dim: (head_dim, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(y)
            .arg(z)
            .arg(weight)
            .arg(dout)
            .arg(dy)
            .arg(dz)
            .arg(&head_dim)
            .arg(&eps)
            .launch(cfg)?;
    }
    Ok(())
}

/// Compute `dweight[d] = sum_h dout[h, d] * y_norm[h, d] * silu_z[h, d]`
/// — the cross-head reduction for `gated_rmsnorm_silu` backward.
///
/// One thread per `d` (looping over `h` internally). Compiles with the
/// dy/dz kernel above.
#[allow(clippy::too_many_arguments)]
pub fn gated_rmsnorm_silu_backward_dweight_bf16(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    y: &CudaSlice<u16>,
    z: &CudaSlice<u16>,
    dout: &CudaSlice<u16>,
    dweight: &mut CudaSlice<u16>,
    num_heads: u32,
    head_dim: u32,
    eps: f32,
) -> Result<()> {
    if num_heads == 0 || head_dim == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    let need = (num_heads * head_dim) as usize;
    if y.len() < need || z.len() < need || dout.len() < need {
        return Err(SparkError::InvalidArgument(
            "y/z/dout buffer too small".into(),
        ));
    }
    if dweight.len() < head_dim as usize {
        return Err(SparkError::InvalidArgument(
            "dweight buffer too small".into(),
        ));
    }

    let func = module::load_kernel(
        ctx,
        "gated_rmsnorm_silu_backward_bf16", // same PTX file as the dy/dz kernel
        "gated_rmsnorm_silu_backward_dweight_bf16",
    )?;
    let cfg = LaunchConfig {
        grid_dim: (head_dim.div_ceil(128), 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(y)
            .arg(z)
            .arg(dout)
            .arg(dweight)
            .arg(&num_heads)
            .arg(&head_dim)
            .arg(&eps)
            .launch(cfg)?;
    }
    Ok(())
}
