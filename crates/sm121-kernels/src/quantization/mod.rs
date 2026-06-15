use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};

use crate::error::{Result, SparkError};
use crate::module;

/// Dynamic per-token FP8 E4M3 quantization of BF16 activations.
///
/// For each row (token), computes `abs_max`, derives `scale = abs_max / 448`,
/// and stores `out[i] = round_to_e4m3(x[i] / scale)` along with the per-row
/// scale. Downstream FP8 GEMM multiplies back by this scale.
///
/// `x`: input [num_rows, hidden_dim] BF16
/// `out`: output [num_rows, hidden_dim] FP8 E4M3 (u8)
/// `scales`: output [num_rows] F32 (one scale per row)
#[allow(clippy::too_many_arguments)]
/// Quantize a BF16 tensor to FP8 e4m3 with a single per-tensor scale
/// (computed by the caller, typically `max_abs(x) / 448.0`).
///
/// `q[i] = round(x[i] / scale)`, saturating to e4m3 range.
/// Caller-side dequant: `x_hat[i] = scale * f32(q[i])`.
pub fn quant_bf16_to_fp8_pertensor(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<u16>,
    q: &mut CudaSlice<u8>,
    n: u32,
    scale: f32,
) -> Result<()> {
    if n == 0 {
        return Err(SparkError::InvalidArgument("n must be > 0".into()));
    }
    if x.len() < n as usize {
        return Err(SparkError::InvalidArgument("x too small".into()));
    }
    if q.len() < n as usize {
        return Err(SparkError::InvalidArgument("q too small".into()));
    }
    let func = module::load_kernel(
        ctx,
        "quant_bf16_to_fp8_pertensor",
        "quant_bf16_to_fp8_pertensor",
    )?;
    let threads = 256u32;
    let elems_per_block = threads * 2;
    let blocks = n.div_ceil(elems_per_block).max(1);
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(x)
            .arg(q)
            .arg(&n)
            .arg(&scale)
            .launch(cfg)?;
    }
    Ok(())
}

/// Quantize BF16 → FP8 E4M3 with one scale per row (per-token): each row is
/// scaled by its own absmax so `out[row, i] = fp8(x[row, i] / scales[row])`.
/// Writes the chosen `scales` for later dequantization.
pub fn quant_bf16_to_fp8_pertoken(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<u16>,
    out: &mut CudaSlice<u8>,
    scales: &mut CudaSlice<f32>,
    num_rows: u32,
    hidden_dim: u32,
) -> Result<()> {
    if num_rows == 0 || hidden_dim == 0 {
        return Err(SparkError::InvalidArgument(
            "num_rows and hidden_dim must be > 0".into(),
        ));
    }
    if hidden_dim & 1 != 0 {
        return Err(SparkError::InvalidArgument(
            "hidden_dim must be even for FP8 output".into(),
        ));
    }
    let need = num_rows as usize * hidden_dim as usize;
    if x.len() < need {
        return Err(SparkError::InvalidArgument(format!(
            "x too small: {} < {need}",
            x.len()
        )));
    }
    if out.len() < need {
        return Err(SparkError::InvalidArgument(format!(
            "out too small: {} < {need}",
            out.len()
        )));
    }
    if scales.len() < num_rows as usize {
        return Err(SparkError::InvalidArgument(format!(
            "scales too small: {} < {num_rows}",
            scales.len()
        )));
    }

    let func = module::load_kernel(
        ctx,
        "quant_bf16_to_fp8_pertoken",
        "quant_bf16_to_fp8_pertoken",
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
            .arg(scales)
            .arg(&hidden_dim)
            .launch(cfg)?;
    }
    Ok(())
}

/// Dequantize FP8 E4M3 → BF16 with per-row scales.
///
/// `out[row, i] = x[row, i] * scales[row]`
///
/// `x`: input [num_rows, hidden_dim] FP8 E4M3 (u8)
/// `scales`: [num_rows] F32
/// `out`: output [num_rows, hidden_dim] BF16
/// Per-tensor FP8 e4m3 → BF16 dequantization (single scale value).
/// `out[i] = scale * f32(x[i])`.
pub fn dequant_fp8_bf16_pertensor(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<u8>,
    out: &mut CudaSlice<u16>,
    n: u32,
    scale: f32,
) -> Result<()> {
    if n == 0 {
        return Err(SparkError::InvalidArgument("n must be > 0".into()));
    }
    let need = n as usize;
    if x.len() < need || out.len() < need {
        return Err(SparkError::InvalidArgument(format!(
            "buffers too small: need {need}"
        )));
    }
    let func = module::load_kernel(
        ctx,
        "dequant_fp8_bf16_pertensor",
        "dequant_fp8_bf16_pertensor",
    )?;
    let cfg = LaunchConfig {
        grid_dim: (n.div_ceil(256), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(x)
            .arg(out)
            .arg(&n)
            .arg(&scale)
            .launch(cfg)?;
    }
    Ok(())
}

/// Dequantize FP8 E4M3 → BF16 with one scale per row (per-token):
/// `out[row, i] = x[row, i] * scales[row]`. Inverse of `quant_bf16_to_fp8_pertoken`.
pub fn dequant_fp8_bf16(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<u8>,
    scales: &CudaSlice<f32>,
    out: &mut CudaSlice<u16>,
    num_rows: u32,
    hidden_dim: u32,
) -> Result<()> {
    if num_rows == 0 || hidden_dim == 0 {
        return Err(SparkError::InvalidArgument(
            "num_rows and hidden_dim must be > 0".into(),
        ));
    }
    let need = num_rows as usize * hidden_dim as usize;
    if x.len() < need || out.len() < need {
        return Err(SparkError::InvalidArgument(format!(
            "buffers too small: need {need}"
        )));
    }
    if scales.len() < num_rows as usize {
        return Err(SparkError::InvalidArgument("scales too small".into()));
    }

    let func = module::load_kernel(ctx, "dequant_fp8_bf16", "dequant_fp8_bf16")?;
    let cfg = LaunchConfig {
        grid_dim: (num_rows, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(x)
            .arg(scales)
            .arg(out)
            .arg(&hidden_dim)
            .launch(cfg)?;
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────
// Block-scaled FP8 (DeepSeek V3 1×128)
// ─────────────────────────────────────────────────────────────────

/// DeepSeek V3-style block-scaled FP8 E4M3 quantization: 128-element blocks
/// along the last dim, each with its own FP32 scale.
///
/// `x`: [num_rows, hidden] BF16 (hidden multiple of 128)
/// `out`: [num_rows, hidden] FP8 E4M3 (u8)
/// `scales`: [num_rows, hidden/128] F32
pub fn quant_bf16_to_fp8_block128(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<u16>,
    out: &mut CudaSlice<u8>,
    scales: &mut CudaSlice<f32>,
    num_rows: u32,
    hidden_dim: u32,
) -> Result<()> {
    if hidden_dim & 127 != 0 {
        return Err(SparkError::InvalidArgument(
            "hidden_dim must be multiple of 128 for FP8 block128 quant".into(),
        ));
    }
    if num_rows == 0 {
        return Err(SparkError::InvalidArgument("num_rows must be > 0".into()));
    }
    let num_blocks = hidden_dim / 128;
    let need = num_rows as usize * hidden_dim as usize;
    if x.len() < need || out.len() < need {
        return Err(SparkError::InvalidArgument("x/out too small".into()));
    }
    if scales.len() < (num_rows * num_blocks) as usize {
        return Err(SparkError::InvalidArgument("scales too small".into()));
    }

    let func = module::load_kernel(
        ctx,
        "quant_bf16_to_fp8_block128",
        "quant_bf16_to_fp8_block128",
    )?;
    let cfg = LaunchConfig {
        grid_dim: (num_rows, num_blocks, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(x)
            .arg(out)
            .arg(scales)
            .arg(&hidden_dim)
            .launch(cfg)?;
    }
    Ok(())
}

/// Dequantize block-scaled FP8 (1×128) back to BF16.
pub fn dequant_fp8_block128_bf16(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<u8>,
    scales: &CudaSlice<f32>,
    out: &mut CudaSlice<u16>,
    num_rows: u32,
    hidden_dim: u32,
) -> Result<()> {
    if hidden_dim & 127 != 0 {
        return Err(SparkError::InvalidArgument(
            "hidden_dim must be multiple of 128".into(),
        ));
    }
    let num_blocks = hidden_dim / 128;
    let func = module::load_kernel(
        ctx,
        "dequant_fp8_block128_bf16",
        "dequant_fp8_block128_bf16",
    )?;
    let cfg = LaunchConfig {
        grid_dim: (num_rows, num_blocks, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(x)
            .arg(scales)
            .arg(out)
            .arg(&hidden_dim)
            .launch(cfg)?;
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────
// MXFP8 (32-element blocks, UE8M0 scales)
// ─────────────────────────────────────────────────────────────────

/// OCP MX microscaling FP8 E4M3 quantization. 32-element blocks with UE8M0
/// (exponent-only) scales. Used by FlashInfer / SGLang mixed-precision paths.
///
/// `out`: [num_rows, hidden] FP8 E4M3 (u8)
/// `scales`: [num_rows, hidden/32] u8 (UE8M0 byte per block)
pub fn quant_bf16_to_mxfp8(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<u16>,
    out: &mut CudaSlice<u8>,
    scales: &mut CudaSlice<u8>,
    num_rows: u32,
    hidden_dim: u32,
) -> Result<()> {
    if hidden_dim & 31 != 0 {
        return Err(SparkError::InvalidArgument(
            "hidden_dim must be multiple of 32 for MXFP8".into(),
        ));
    }
    let num_blocks = hidden_dim / 32;
    let func = module::load_kernel(ctx, "quant_bf16_to_mxfp8", "quant_bf16_to_mxfp8")?;
    let cfg = LaunchConfig {
        grid_dim: (num_rows, num_blocks, 1),
        block_dim: (8, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(x)
            .arg(out)
            .arg(scales)
            .arg(&hidden_dim)
            .launch(cfg)?;
    }
    Ok(())
}

/// Dequantize MXFP8 → BF16.
pub fn dequant_mxfp8_bf16(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<u8>,
    scales: &CudaSlice<u8>,
    out: &mut CudaSlice<u16>,
    num_rows: u32,
    hidden_dim: u32,
) -> Result<()> {
    if hidden_dim & 31 != 0 {
        return Err(SparkError::InvalidArgument(
            "hidden_dim must be multiple of 32".into(),
        ));
    }
    let num_blocks = hidden_dim / 32;
    let func = module::load_kernel(ctx, "dequant_mxfp8_bf16", "dequant_mxfp8_bf16")?;
    let cfg = LaunchConfig {
        grid_dim: (num_rows, num_blocks, 1),
        block_dim: (8, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(x)
            .arg(scales)
            .arg(out)
            .arg(&hidden_dim)
            .launch(cfg)?;
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────
// MXFP4 (32-element blocks, UE8M0 scales, FP4 E2M1 values)
// ─────────────────────────────────────────────────────────────────

/// OCP MX microscaling FP4 E2M1 quantization. 32-element blocks with UE8M0
/// scales. Used by gpt-oss-120b and similar MXFP4 MoE models.
///
/// `out`: [num_rows, hidden/2] u8 (packed FP4 pairs: low nibble, high nibble)
/// `scales`: [num_rows, hidden/32] u8 (UE8M0)
pub fn quant_bf16_to_mxfp4(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<u16>,
    out: &mut CudaSlice<u8>,
    scales: &mut CudaSlice<u8>,
    num_rows: u32,
    hidden_dim: u32,
) -> Result<()> {
    if hidden_dim & 31 != 0 {
        return Err(SparkError::InvalidArgument(
            "hidden_dim must be multiple of 32 for MXFP4".into(),
        ));
    }
    let num_blocks = hidden_dim / 32;
    let func = module::load_kernel(ctx, "quant_bf16_to_mxfp4", "quant_bf16_to_mxfp4")?;
    let cfg = LaunchConfig {
        grid_dim: (num_rows, num_blocks, 1),
        block_dim: (8, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(x)
            .arg(out)
            .arg(scales)
            .arg(&hidden_dim)
            .launch(cfg)?;
    }
    Ok(())
}

/// Dequantize MXFP4 → BF16.
pub fn dequant_mxfp4_bf16(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<u8>,
    scales: &CudaSlice<u8>,
    out: &mut CudaSlice<u16>,
    num_rows: u32,
    hidden_dim: u32,
) -> Result<()> {
    if hidden_dim & 31 != 0 {
        return Err(SparkError::InvalidArgument(
            "hidden_dim must be multiple of 32".into(),
        ));
    }
    let num_blocks = hidden_dim / 32;
    let func = module::load_kernel(ctx, "dequant_mxfp4_bf16", "dequant_mxfp4_bf16")?;
    let cfg = LaunchConfig {
        grid_dim: (num_rows, num_blocks, 1),
        block_dim: (8, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(x)
            .arg(scales)
            .arg(out)
            .arg(&hidden_dim)
            .launch(cfg)?;
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────
// NVFP4 (16-element blocks, FP8 E4M3 scales, FP4 E2M1 values)
// ─────────────────────────────────────────────────────────────────

/// NVIDIA NVFP4 quantization. 16-element blocks with FP8 E4M3 scales.
/// Compatible with CUTLASS NVFP4 GEMM.
///
/// `out`: [num_rows, hidden/2] u8 (packed FP4)
/// `scales`: [num_rows, hidden/16] u8 (FP8 e4m3)
pub fn quant_bf16_to_nvfp4(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<u16>,
    out: &mut CudaSlice<u8>,
    scales: &mut CudaSlice<u8>,
    num_rows: u32,
    hidden_dim: u32,
) -> Result<()> {
    if hidden_dim & 15 != 0 {
        return Err(SparkError::InvalidArgument(
            "hidden_dim must be multiple of 16 for NVFP4".into(),
        ));
    }
    let num_blocks = hidden_dim / 16;
    let func = module::load_kernel(ctx, "quant_bf16_to_nvfp4", "quant_bf16_to_nvfp4")?;
    let cfg = LaunchConfig {
        grid_dim: (num_rows, num_blocks, 1),
        block_dim: (4, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(x)
            .arg(out)
            .arg(scales)
            .arg(&hidden_dim)
            .launch(cfg)?;
    }
    Ok(())
}

/// Dequantize NVFP4 → BF16.
pub fn dequant_nvfp4_bf16(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<u8>,
    scales: &CudaSlice<u8>,
    out: &mut CudaSlice<u16>,
    num_rows: u32,
    hidden_dim: u32,
) -> Result<()> {
    if hidden_dim & 15 != 0 {
        return Err(SparkError::InvalidArgument(
            "hidden_dim must be multiple of 16".into(),
        ));
    }
    let num_blocks = hidden_dim / 16;
    let func = module::load_kernel(ctx, "dequant_nvfp4_bf16", "dequant_nvfp4_bf16")?;
    let cfg = LaunchConfig {
        grid_dim: (num_rows, num_blocks, 1),
        block_dim: (4, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(x)
            .arg(scales)
            .arg(out)
            .arg(&hidden_dim)
            .launch(cfg)?;
    }
    Ok(())
}
