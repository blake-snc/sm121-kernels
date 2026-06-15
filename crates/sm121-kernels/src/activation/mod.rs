use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaSlice, CudaStream, CudaView, DevicePtr, DevicePtrMut, LaunchConfig,
    PushKernelArg,
};

use crate::error::{Result, SparkError};
use crate::module;

fn validate_activation_args(
    input: &CudaSlice<u16>,
    out: &CudaSlice<u16>,
    total_out_elems: u32,
    d: u32,
) -> Result<()> {
    if total_out_elems == 0 {
        return Err(SparkError::InvalidArgument(
            "total_out_elems must be > 0".into(),
        ));
    }
    if d == 0 {
        return Err(SparkError::InvalidArgument("d must be > 0".into()));
    }
    // Vectorized kernels process 2 outputs per thread — require even D and even total.
    if d & 1 != 0 {
        return Err(SparkError::InvalidArgument(format!(
            "d must be even for vectorized activation kernels (got {d})"
        )));
    }
    // Input is [N, 2*D], output is [N, D]
    let input_need = total_out_elems as usize * 2;
    if input.len() < input_need {
        return Err(SparkError::InvalidArgument(format!(
            "input buffer too small: {} < {input_need}",
            input.len()
        )));
    }
    if out.len() < total_out_elems as usize {
        return Err(SparkError::InvalidArgument(format!(
            "out buffer too small: {} < {total_out_elems}",
            out.len()
        )));
    }
    Ok(())
}

/// Compute grid size for a vectorized elementwise kernel that consumes 2 outputs per thread.
fn vector2_grid(total_out_elems: u32, threads_per_block: u32) -> u32 {
    // Each thread handles 2 elements, so we need ceil(total / (2 * threads))
    total_out_elems.div_ceil(threads_per_block * 2).max(1)
}

/// Launch SiLU*mul BF16 kernel.
///
/// `input`: [N, 2*D] BF16 — first D elements are gate, second D are up
/// `out`: [N, D] BF16
/// `total_out_elems`: N * D
/// `d`: D (half of input dim per row)
pub fn silu_mul_bf16(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    input: &CudaSlice<u16>,
    out: &mut CudaSlice<u16>,
    total_out_elems: u32,
    d: u32,
) -> Result<()> {
    validate_activation_args(input, out, total_out_elems, d)?;
    let func = module::load_kernel(ctx, "silu_mul_bf16", "silu_mul_bf16")?;

    let threads_per_block: u32 = 256;
    let num_blocks = vector2_grid(total_out_elems, threads_per_block);
    let cfg = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (threads_per_block, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&func)
            .arg(input)
            .arg(out)
            .arg(&total_out_elems)
            .arg(&d)
            .launch(cfg)?;
    }

    Ok(())
}

/// Plain SiLU forward: `y[i] = x[i] * sigmoid(x[i])`.
///
/// Companion to `silu_backward_bf16`. Used by simple MLPs that don't use
/// the gated SwiGLU variant (`silu_mul_bf16`).
pub fn silu_bf16(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<u16>,
    y: &mut CudaSlice<u16>,
    n: u32,
) -> Result<()> {
    if n == 0 {
        return Err(SparkError::InvalidArgument("n must be > 0".into()));
    }
    if x.len() < n as usize || y.len() < n as usize {
        return Err(SparkError::InvalidArgument(format!(
            "buf too small: x={} y={} need={n}",
            x.len(),
            y.len()
        )));
    }
    let func = module::load_kernel(ctx, "silu_bf16", "silu_bf16")?;
    let threads_per_block: u32 = 256;
    let num_blocks = n.div_ceil(threads_per_block).min(65535).max(1);
    let cfg = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (threads_per_block, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(x)
            .arg(y)
            .arg(&n)
            .launch(cfg)?;
    }
    Ok(())
}

/// Backward pass for plain SiLU: `y = x * sigmoid(x)`.
/// `dx[i] = dy[i] * sigmoid(x) * (1 + x * (1 - sigmoid(x)))`.
///
/// Layout: x, dy, dx all `[n]` BF16. Element-wise.
pub fn silu_backward_bf16(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<u16>,
    dy: &CudaSlice<u16>,
    dx: &mut CudaSlice<u16>,
    n: u32,
) -> Result<()> {
    if n == 0 {
        return Err(SparkError::InvalidArgument("n must be > 0".into()));
    }
    let need = n as usize;
    if x.len() < need || dy.len() < need || dx.len() < need {
        return Err(SparkError::InvalidArgument(format!(
            "buffers too small: need {need}"
        )));
    }
    let func = module::load_kernel(ctx, "silu_backward_bf16", "silu_backward_bf16")?;
    let threads_per_block: u32 = 256;
    let num_blocks = n.div_ceil(threads_per_block);
    let cfg = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (threads_per_block, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(x)
            .arg(dy)
            .arg(dx)
            .arg(&n)
            .launch(cfg)?;
    }
    Ok(())
}

/// Backward pass for PyTorch GeLU-tanh approximation:
/// `y = x * 0.5 * (1 + tanh(K*(x + 0.044715*x^3)))` with `K = sqrt(2/pi)`.
/// Analytical derivative: `dy/dx = 0.5*(1+t) + 0.5*x*(1-t^2)*K*(1+0.134145*x^2)`.
///
/// Layout: x, dy, dx all `[n]` BF16.
pub fn gelu_tanh_backward_bf16(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<u16>,
    dy: &CudaSlice<u16>,
    dx: &mut CudaSlice<u16>,
    n: u32,
) -> Result<()> {
    if n == 0 {
        return Err(SparkError::InvalidArgument("n must be > 0".into()));
    }
    let need = n as usize;
    if x.len() < need || dy.len() < need || dx.len() < need {
        return Err(SparkError::InvalidArgument(format!(
            "buffers too small: need {need}"
        )));
    }
    let func = module::load_kernel(ctx, "gelu_tanh_backward_bf16", "gelu_tanh_backward_bf16")?;
    let threads_per_block: u32 = 256;
    let num_blocks = n.div_ceil(threads_per_block);
    let cfg = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (threads_per_block, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(x)
            .arg(dy)
            .arg(dx)
            .arg(&n)
            .launch(cfg)?;
    }
    Ok(())
}

/// SiLU*mul on SEPARATE gate and up tensors: `out[i] = silu(gate[i]) * up[i]`.
///
/// Avoids the dtod-copy "fuse gate||up into [N, 2D]" pattern that `silu_mul_bf16`
/// requires. Used by the MoE per-expert decode path where gate and up arrive as
/// outputs of two separate batched grouped GEMVs.
///
/// All three buffers must have at least `n` elements; `n` must be even.
pub fn silu_mul_split_bf16(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    gate: &CudaSlice<u16>,
    up: &CudaSlice<u16>,
    out: &mut CudaSlice<u16>,
    n: u32,
) -> Result<()> {
    if n == 0 {
        return Err(SparkError::InvalidArgument("n must be > 0".into()));
    }
    if n & 1 != 0 {
        return Err(SparkError::InvalidArgument(format!(
            "silu_mul_split_bf16: n must be even, got {n}"
        )));
    }
    if gate.len() < n as usize || up.len() < n as usize || out.len() < n as usize {
        return Err(SparkError::InvalidArgument(format!(
            "silu_mul_split_bf16: buffer too small (gate={}, up={}, out={}, need {n})",
            gate.len(),
            up.len(),
            out.len()
        )));
    }
    let func = module::load_kernel(ctx, "silu_mul_split_bf16", "silu_mul_split_bf16")?;
    let threads = 256u32;
    let blocks = n.div_ceil(threads * 2).min(2048).max(1);
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(gate)
            .arg(up)
            .arg(out)
            .arg(&n)
            .launch(cfg)?;
    }
    Ok(())
}

/// In-place broadcast multiply by a single BF16 scalar device pointer:
/// `buf[i] *= *scalar`. Used by Gemma-4-style per-layer-scalar residuals.
/// Replaces the host roundtrip (dtoh + host mul + htod).
pub fn scale_bf16_inplace_dev(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    buf: &mut CudaSlice<u16>,
    scalar: &CudaSlice<u16>,
    n: u32,
) -> Result<()> {
    if n == 0 {
        return Err(SparkError::InvalidArgument("n must be > 0".into()));
    }
    if n & 1 != 0 {
        return Err(SparkError::InvalidArgument(format!(
            "scale_bf16_inplace_dev: n must be even, got {n}"
        )));
    }
    if buf.len() < n as usize {
        return Err(SparkError::InvalidArgument(format!(
            "scale_bf16_inplace_dev: buf too small ({} < {n})",
            buf.len()
        )));
    }
    if scalar.is_empty() {
        return Err(SparkError::InvalidArgument("scalar buffer empty".into()));
    }
    let func = module::load_kernel(ctx, "scale_bf16_inplace_dev", "scale_bf16_inplace_dev")?;
    let threads = 256u32;
    let blocks = n.div_ceil(threads * 2).min(2048).max(1);
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(buf)
            .arg(scalar)
            .arg(&n)
            .launch(cfg)?;
    }
    Ok(())
}

/// In-place per-row scale + per-tensor scale broadcast for the
/// W8A16-via-FP8-v3 post-multiply. Computes `buf[i, j] *= row_scales[i] * b_scale`.
///
/// Used after `gemm_fp8_mma_v3` to recover true C magnitude when A is
/// per-token quantized and B is per-tensor quantized:
///
///   accum[i, j] = sum_k (a[i, k]/a_scales[i] * b[k, j]/b_scale)
///   C_true[i, j] = (a_scales[i] * b_scale) * accum[i, j]
///
/// `m` is the row count (= number of slots in batched M=128 decode);
/// `n` must be even (always true for our decode shapes). Throughput is
/// memory-bound: scales like 2× pass over `buf`.
pub fn scale_bf16_rows_inplace(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    buf: &mut CudaSlice<u16>,
    row_scales: &CudaSlice<f32>,
    m: u32,
    n: u32,
    b_scale: f32,
) -> Result<()> {
    if m == 0 || n == 0 || n & 1 != 0 {
        return Err(SparkError::InvalidArgument(format!(
            "scale_bf16_rows_inplace: m={m} > 0, n={n} > 0 + even required"
        )));
    }
    if buf.len() < (m as usize) * (n as usize) {
        return Err(SparkError::InvalidArgument(format!(
            "scale_bf16_rows_inplace: buf too small ({} < {m}*{n})",
            buf.len()
        )));
    }
    if row_scales.len() < m as usize {
        return Err(SparkError::InvalidArgument(format!(
            "scale_bf16_rows_inplace: row_scales too small ({} < {m})",
            row_scales.len()
        )));
    }
    let func = module::load_kernel(ctx, "scale_bf16_rows_inplace", "scale_bf16_rows_inplace")?;
    let threads: u32 = 256;
    let pairs_per_block = threads * 2; // 2 BF16 per thread (packed b32)
    let grid_x = n.div_ceil(pairs_per_block);
    let cfg = LaunchConfig {
        grid_dim: (grid_x.max(1), m, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(buf)
            .arg(row_scales)
            .arg(&m)
            .arg(&n)
            .arg(&b_scale)
            .launch(cfg)?;
    }
    Ok(())
}

/// View variant of `gelu_tanh_mul_split_bf16`: accepts `up` as a `CudaView`
/// so callers can pass a sub-slice of a larger contiguous buffer (e.g., a
/// per-layer slice of a stacked aux-inputs tensor).
pub fn gelu_tanh_mul_split_view_bf16(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    gate: &CudaSlice<u16>,
    up: &CudaView<u16>,
    out: &mut CudaSlice<u16>,
    n: u32,
) -> Result<()> {
    if n == 0 || n & 1 != 0 {
        return Err(SparkError::InvalidArgument(format!(
            "gelu_tanh_mul_split_view_bf16: n must be >0 and even, got {n}"
        )));
    }
    if gate.len() < n as usize || up.len() < n as usize || out.len() < n as usize {
        return Err(SparkError::InvalidArgument(format!(
            "gelu_tanh_mul_split_view_bf16: buffer too small (gate={}, up={}, out={}, need {n})",
            gate.len(),
            up.len(),
            out.len()
        )));
    }
    let func = module::load_kernel(ctx, "gelu_tanh_mul_split_bf16", "gelu_tanh_mul_split_bf16")?;
    let threads = 256u32;
    let blocks = n.div_ceil(threads * 2).min(2048).max(1);
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(gate)
            .arg(up)
            .arg(out)
            .arg(&n)
            .launch(cfg)?;
    }
    Ok(())
}

/// `gelu_tanh(gate) * up` on SEPARATE gate and up tensors. Drop-in counterpart
/// to `silu_mul_split_bf16` for Gemma-style MLPs that use `gelu_pytorch_tanh`.
/// Avoids the dtod-copy "fuse gate||up into [N, 2D]" step that `gelu_tanh_mul_bf16`
/// requires.
///
/// All three buffers must have at least `n` elements; `n` must be even.
pub fn gelu_tanh_mul_split_bf16(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    gate: &CudaSlice<u16>,
    up: &CudaSlice<u16>,
    out: &mut CudaSlice<u16>,
    n: u32,
) -> Result<()> {
    if n == 0 {
        return Err(SparkError::InvalidArgument("n must be > 0".into()));
    }
    if n & 1 != 0 {
        return Err(SparkError::InvalidArgument(format!(
            "gelu_tanh_mul_split_bf16: n must be even, got {n}"
        )));
    }
    if gate.len() < n as usize || up.len() < n as usize || out.len() < n as usize {
        return Err(SparkError::InvalidArgument(format!(
            "gelu_tanh_mul_split_bf16: buffer too small (gate={}, up={}, out={}, need {n})",
            gate.len(),
            up.len(),
            out.len()
        )));
    }
    let func = module::load_kernel(ctx, "gelu_tanh_mul_split_bf16", "gelu_tanh_mul_split_bf16")?;
    let threads = 256u32;
    let blocks = n.div_ceil(threads * 2).min(2048).max(1);
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(gate)
            .arg(up)
            .arg(out)
            .arg(&n)
            .launch(cfg)?;
    }
    Ok(())
}

/// Elementwise BF16 add: `out[i] = a[i] + b[i]` for `i in [0, n)`.
///
/// All three buffers must have at least `n` elements. Grid is sized for `n` total
/// elements with grid-stride looping, so safe for any size.
pub fn add_bf16(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    a: &CudaSlice<u16>,
    b: &CudaSlice<u16>,
    out: &mut CudaSlice<u16>,
    n: u32,
) -> Result<()> {
    if n == 0 {
        return Err(SparkError::InvalidArgument("n must be > 0".into()));
    }
    if a.len() < n as usize || b.len() < n as usize || out.len() < n as usize {
        return Err(SparkError::InvalidArgument(format!(
            "add_bf16: buffer too small (a={}, b={}, out={}, need {n})",
            a.len(),
            b.len(),
            out.len()
        )));
    }
    let func = module::load_kernel(ctx, "add_bf16", "add_bf16")?;
    let threads = 256u32;
    let blocks = n.div_ceil(threads).min(2048).max(1);
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(a)
            .arg(b)
            .arg(out)
            .arg(&n)
            .launch(cfg)?;
    }
    Ok(())
}

/// Weighted F32 accumulate: `out[i] = out[i] + alpha * src[i]` for `i in [0, n)`.
///
/// Used by the per-expert MoE decode path to accumulate weighted expert outputs
/// into a single F32 accumulator before final BF16 cast.
pub fn f32_axpy(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    out: &mut CudaSlice<f32>,
    src: &CudaSlice<f32>,
    alpha: f32,
    n: u32,
) -> Result<()> {
    if n == 0 {
        return Err(SparkError::InvalidArgument("n must be > 0".into()));
    }
    if out.len() < n as usize || src.len() < n as usize {
        return Err(SparkError::InvalidArgument(format!(
            "f32_axpy: buffer too small (out={}, src={}, need {n})",
            out.len(),
            src.len()
        )));
    }
    let func = module::load_kernel(ctx, "f32_axpy", "f32_axpy")?;
    let threads = 256u32;
    let blocks = n.div_ceil(threads * 2).min(2048).max(1);
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(out)
            .arg(src)
            .arg(&alpha)
            .arg(&n)
            .launch(cfg)?;
    }
    Ok(())
}

/// Fused weighted column sum + BF16 cast: for each column j in `[0, n)`:
///   `out_bf16[j] = bf16( sum_{i=0..num_rows} weights[i] * in_f32[i, j] )`
///
/// Replaces `num_rows × f32_axpy + f32_to_bf16` launches with a single kernel.
/// Used by the MoE per-expert decode path to combine the [num_active, hidden]
/// down-projection outputs with their routing weights into final BF16 mlp_out.
pub fn f32_weighted_sum_to_bf16(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    in_f32: &CudaSlice<f32>,
    weights: &CudaSlice<f32>,
    out_bf16: &mut CudaSlice<u16>,
    num_rows: u32,
    n: u32,
) -> Result<()> {
    if num_rows == 0 || n == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    if in_f32.len() < (num_rows as usize) * (n as usize) {
        return Err(SparkError::InvalidArgument("in_f32 too small".into()));
    }
    if weights.len() < num_rows as usize {
        return Err(SparkError::InvalidArgument("weights too small".into()));
    }
    if out_bf16.len() < n as usize {
        return Err(SparkError::InvalidArgument("out too small".into()));
    }
    let func = module::load_kernel(ctx, "f32_weighted_sum_to_bf16", "f32_weighted_sum_to_bf16")?;
    let threads = 256u32;
    let blocks = n.div_ceil(threads).max(1);
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(in_f32)
            .arg(weights)
            .arg(out_bf16)
            .arg(&num_rows)
            .arg(&n)
            .launch(cfg)?;
    }
    Ok(())
}

/// In-place divide by sum: w[i] /= sum(w) for i in 0..n.
///
/// For MoE with `norm_topk_prob=True` (Qwen3 family). One launch replaces a
/// `dtoh weights → host renorm → htod weights` roundtrip per MoE layer.
/// Single block, single thread (n is tiny — top_k ≤ 16).
pub fn f32_renormalize_topk(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    w: &mut CudaSlice<f32>,
    n: u32,
) -> Result<()> {
    if n == 0 || n > 16 {
        return Err(SparkError::InvalidArgument(format!(
            "n must be in 1..=16, got {n}"
        )));
    }
    if w.len() < n as usize {
        return Err(SparkError::InvalidArgument("w buffer too small".into()));
    }
    let func = module::load_kernel(ctx, "f32_renormalize_topk", "f32_renormalize_topk")?;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream.launch_builder(&func).arg(w).arg(&n).launch(cfg)?;
    }
    Ok(())
}

/// In-place fused sigmoid-scalar AXPY in BF16:
///   `out[i] := out[i] + sigmoid(*scalar) * src[i]`
///
/// `scalar` is a single device-resident BF16 logit (no host roundtrip).
/// Used by Qwen3-Next / GDN-hybrid MoE shared-expert combine.
pub fn sigmoid_scalar_axpy_bf16(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    out: &mut CudaSlice<u16>,
    src: &CudaSlice<u16>,
    scalar: &CudaSlice<u16>,
    n: u32,
) -> Result<()> {
    if n == 0 {
        return Err(SparkError::InvalidArgument("n must be > 0".into()));
    }
    if out.len() < n as usize || src.len() < n as usize {
        return Err(SparkError::InvalidArgument(
            "out/src buffer too small".into(),
        ));
    }
    if scalar.is_empty() {
        return Err(SparkError::InvalidArgument("scalar buffer empty".into()));
    }
    let func = module::load_kernel(ctx, "sigmoid_scalar_axpy_bf16", "sigmoid_scalar_axpy_bf16")?;
    let threads = 256u32;
    let blocks = n.div_ceil(threads).max(1);
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(out)
            .arg(src)
            .arg(scalar)
            .arg(&n)
            .launch(cfg)?;
    }
    Ok(())
}

/// View variant of `f32_axpy`: src can be a sub-view of a larger buffer (e.g.,
/// a single per-expert row of a `[num_active, n]` grouped GEMV output).
pub fn f32_axpy_view(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    out: &mut CudaSlice<f32>,
    src: &CudaView<f32>,
    alpha: f32,
    n: u32,
) -> Result<()> {
    if n == 0 {
        return Err(SparkError::InvalidArgument("n must be > 0".into()));
    }
    if out.len() < n as usize || src.len() < n as usize {
        return Err(SparkError::InvalidArgument(format!(
            "f32_axpy_view: buffer too small (out={}, src={}, need {n})",
            out.len(),
            src.len()
        )));
    }
    let func = module::load_kernel(ctx, "f32_axpy", "f32_axpy")?;
    let threads = 256u32;
    let blocks = n.div_ceil(threads * 2).min(2048).max(1);
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(out)
            .arg(src)
            .arg(&alpha)
            .arg(&n)
            .launch(cfg)?;
    }
    Ok(())
}

/// In-place elementwise BF16 add: `out_inout[i] += src[i]` for `i in [0, n)`.
///
/// Uses the same `add_bf16` PTX kernel; just reuses the `out_inout` buffer for
/// both the "a" input and the "out" output. Skips one dtod-copy that the
/// `out = a + b; copy out -> a` pattern would require.
/// In-place BF16 axpy with host scalar: `out[i] = out[i] + alpha * src[i]`.
/// Accumulation in FP32, result rounded to BF16. Used by the Gemma-4 MoE
/// block decode path to accumulate weighted per-expert outputs.
pub fn bf16_axpy_host(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    out_inout: &mut CudaSlice<u16>,
    src: &CudaSlice<u16>,
    n: u32,
    alpha: f32,
) -> Result<()> {
    if n == 0 {
        return Err(SparkError::InvalidArgument("n must be > 0".into()));
    }
    if out_inout.len() < n as usize || src.len() < n as usize {
        return Err(SparkError::InvalidArgument(format!(
            "bf16_axpy_host: buffer too small (out={}, src={}, need {n})",
            out_inout.len(),
            src.len()
        )));
    }
    let func = module::load_kernel(ctx, "bf16_axpy_host", "bf16_axpy_host")?;
    let threads = 256u32;
    let blocks = n.div_ceil(threads).min(65535).max(1);
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(out_inout)
            .arg(src)
            .arg(&n)
            .arg(&alpha)
            .launch(cfg)?;
    }
    Ok(())
}

/// Element-wise BF16 add accumulating into `out_inout`: `out_inout[i] += src[i]`
/// for the first `n` elements. Used to fold a LoRA delta into a base activation.
pub fn add_bf16_inplace(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    out_inout: &mut CudaSlice<u16>,
    src: &CudaSlice<u16>,
    n: u32,
) -> Result<()> {
    if n == 0 {
        return Err(SparkError::InvalidArgument("n must be > 0".into()));
    }
    if out_inout.len() < n as usize || src.len() < n as usize {
        return Err(SparkError::InvalidArgument(format!(
            "add_bf16_inplace: buffer too small (out_inout={}, src={}, need {n})",
            out_inout.len(),
            src.len()
        )));
    }
    let func = module::load_kernel(ctx, "add_bf16", "add_bf16")?;
    let threads = 256u32;
    let blocks = n.div_ceil(threads).min(2048).max(1);
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };
    // Pass the same device pointer as both `a` (input) and `out` (output) by
    // grabbing the raw u64 device pointer twice. This is safe: each thread
    // does load → add → store at the same index, no aliasing within a thread.
    unsafe {
        let (out_ptr, _g) = out_inout.device_ptr_mut(stream);
        let (src_ptr, _g2) = src.device_ptr(stream);
        stream
            .launch_builder(&func)
            .arg(&out_ptr) // a (= out_inout)
            .arg(&src_ptr) // b (= src)
            .arg(&out_ptr) // out (= out_inout)
            .arg(&n)
            .launch(cfg)?;
    }
    Ok(())
}

/// Elementwise cast F32 → BF16: `out[i] = bf16(in[i])`.
pub fn f32_to_bf16(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    input: &CudaSlice<f32>,
    out: &mut CudaSlice<u16>,
    n: u32,
) -> Result<()> {
    if n == 0 {
        return Err(SparkError::InvalidArgument("n must be > 0".into()));
    }
    if input.len() < n as usize || out.len() < n as usize {
        return Err(SparkError::InvalidArgument(format!(
            "f32_to_bf16: buffer too small (in={}, out={}, need {n})",
            input.len(),
            out.len()
        )));
    }
    let func = module::load_kernel(ctx, "f32_to_bf16", "f32_to_bf16")?;
    let threads = 256u32;
    let blocks = n.div_ceil(threads).min(2048).max(1);
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(input)
            .arg(out)
            .arg(&n)
            .launch(cfg)?;
    }
    Ok(())
}

/// Replicate `in [M, hidden]` along a new top_k axis: `out[j*top_k+k, :] = in[j, :]`.
/// Output is laid out flat as `[M*top_k, hidden]`. `hidden` must be even.
pub fn broadcast_top_k_bf16(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    input: &CudaSlice<u16>,
    out: &mut CudaSlice<u16>,
    m: u32,
    top_k: u32,
    hidden: u32,
) -> Result<()> {
    if m == 0 || top_k == 0 || hidden == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    if hidden & 1 != 0 {
        return Err(SparkError::InvalidArgument(format!(
            "broadcast_top_k_bf16: hidden ({hidden}) must be even (paired b32 stores)"
        )));
    }
    let need_in = (m as usize) * (hidden as usize);
    let need_out = (m as usize) * (top_k as usize) * (hidden as usize);
    if input.len() < need_in {
        return Err(SparkError::InvalidArgument(format!(
            "broadcast_top_k_bf16: input too small ({}, need {need_in})",
            input.len()
        )));
    }
    if out.len() < need_out {
        return Err(SparkError::InvalidArgument(format!(
            "broadcast_top_k_bf16: out too small ({}, need {need_out})",
            out.len()
        )));
    }
    let func = module::load_kernel(ctx, "broadcast_top_k_bf16", "broadcast_top_k_bf16")?;
    let threads = 256u32;
    let total_pairs = (need_out / 2) as u32;
    let blocks = total_pairs.div_ceil(threads).min(2048).max(1);
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(input)
            .arg(out)
            .arg(&m)
            .arg(&top_k)
            .arg(&hidden)
            .launch(cfg)?;
    }
    Ok(())
}

/// Elementwise cast BF16 → F32: `out[i] = f32(in[i])`.
pub fn bf16_to_f32(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    input: &CudaSlice<u16>,
    out: &mut CudaSlice<f32>,
    n: u32,
) -> Result<()> {
    if n == 0 {
        return Err(SparkError::InvalidArgument("n must be > 0".into()));
    }
    if input.len() < n as usize || out.len() < n as usize {
        return Err(SparkError::InvalidArgument(format!(
            "bf16_to_f32: buffer too small (in={}, out={}, need {n})",
            input.len(),
            out.len()
        )));
    }
    let func = module::load_kernel(ctx, "bf16_to_f32", "bf16_to_f32")?;
    let threads = 256u32;
    let blocks = n.div_ceil(threads).min(2048).max(1);
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(input)
            .arg(out)
            .arg(&n)
            .launch(cfg)?;
    }
    Ok(())
}

/// Launch GeLU*mul BF16 kernel (tanh approximation, matches PyTorch default).
pub fn gelu_mul_bf16(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    input: &CudaSlice<u16>,
    out: &mut CudaSlice<u16>,
    total_out_elems: u32,
    d: u32,
) -> Result<()> {
    validate_activation_args(input, out, total_out_elems, d)?;
    let func = module::load_kernel(ctx, "gelu_mul_bf16", "gelu_mul_bf16")?;

    let threads_per_block: u32 = 256;
    let num_blocks = vector2_grid(total_out_elems, threads_per_block);
    let cfg = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (threads_per_block, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&func)
            .arg(input)
            .arg(out)
            .arg(&total_out_elems)
            .arg(&d)
            .launch(cfg)?;
    }

    Ok(())
}

/// Launch GeLU-tanh*mul BF16 kernel (explicit tanh approximation).
pub fn gelu_tanh_mul_bf16(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    input: &CudaSlice<u16>,
    out: &mut CudaSlice<u16>,
    total_out_elems: u32,
    d: u32,
) -> Result<()> {
    validate_activation_args(input, out, total_out_elems, d)?;
    let func = module::load_kernel(ctx, "gelu_tanh_mul_bf16", "gelu_tanh_mul_bf16")?;

    let threads_per_block: u32 = 256;
    let num_blocks = vector2_grid(total_out_elems, threads_per_block);
    let cfg = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (threads_per_block, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&func)
            .arg(input)
            .arg(out)
            .arg(&total_out_elems)
            .arg(&d)
            .launch(cfg)?;
    }

    Ok(())
}

/// De-interleave GDN-hybrid gated-attention Q-projection output.
///
/// `q_full` is `[num_heads, 2*head_dim]` BF16 with Q and output_gate
/// interleaved per head. Writes:
///   `q_only[h, :]` = `q_full[h, 0..head_dim]`
///   `gate[h, :]`   = `q_full[h, head_dim..2*head_dim]`
/// Replace 3*M dtod copies (per fused-QKV split) with
/// ONE kernel launch. Reads per-seq interleaved [M, n1+n2+n3] BF16, writes
/// to three contiguous [M, n_i] dst buffers.
///
/// Grid: (ceil(total / 128), M, 1). Block: (128, 1, 1).
#[allow(clippy::too_many_arguments)]
pub fn batched_split_3way_bf16(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<u16>,
    dst0: &mut CudaSlice<u16>,
    dst1: &mut CudaSlice<u16>,
    dst2: &mut CudaSlice<u16>,
    n1: u32,
    n2: u32,
    n3: u32,
    batch_m: u32,
) -> Result<()> {
    if n1 == 0 || n2 == 0 || n3 == 0 || batch_m == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    let total = n1 + n2 + n3;
    let m = batch_m as usize;
    if src.len() < m * (total as usize) {
        return Err(SparkError::InvalidArgument("src buffer too small".into()));
    }
    if dst0.len() < m * (n1 as usize)
        || dst1.len() < m * (n2 as usize)
        || dst2.len() < m * (n3 as usize)
    {
        return Err(SparkError::InvalidArgument("dst buffer too small".into()));
    }

    let func = module::load_kernel(ctx, "batched_split_3way_bf16", "batched_split_3way_bf16")?;
    const BLOCK: u32 = 128;
    let grid_x = total.div_ceil(BLOCK);
    let cfg = LaunchConfig {
        grid_dim: (grid_x, batch_m, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(src)
            .arg(dst0)
            .arg(dst1)
            .arg(dst2)
            .arg(&n1)
            .arg(&n2)
            .arg(&n3)
            .launch(cfg)?;
    }
    Ok(())
}

/// Split an interleaved `[num_heads, 2 * head_dim]` BF16 tensor into separate
/// query and gate halves: per head the first `head_dim` lanes go to `q_only`
/// and the next `head_dim` lanes go to `gate`. Used to unpack GatedDeltaNet
/// fused query+gate projections.
pub fn split_q_gate_bf16(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q_full: &CudaSlice<u16>,
    q_only: &mut CudaSlice<u16>,
    gate: &mut CudaSlice<u16>,
    num_heads: u32,
    head_dim: u32,
) -> Result<()> {
    if num_heads == 0 || head_dim == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    let need = (num_heads * head_dim) as usize;
    if q_full.len() < 2 * need || q_only.len() < need || gate.len() < need {
        return Err(SparkError::InvalidArgument(
            "split_q_gate buffer too small".into(),
        ));
    }
    let func = module::load_kernel(ctx, "split_q_gate_bf16", "split_q_gate_bf16")?;
    let cfg = LaunchConfig {
        grid_dim: (num_heads, 1, 1),
        block_dim: (head_dim, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(q_full)
            .arg(q_only)
            .arg(gate)
            .arg(&head_dim)
            .launch(cfg)?;
    }
    Ok(())
}

/// In-place variant of `attn_output_gate`: `attn_out *= sigmoid(gate_logits)`.
/// Each thread reads its own slot of attn_out + gate_logits and writes back to
/// the same slot — alias-safe because no thread reads another thread's output.
/// Uses raw `cuLaunchKernel` to bypass cudarc's safe-API aliasing check.
pub fn attn_output_gate_inplace(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    attn_out: &mut CudaSlice<u16>,
    gate_logits: &CudaSlice<u16>,
    num_tokens: u32,
    hidden_dim: u32,
) -> Result<()> {
    use cudarc::driver::sys::*;
    use cudarc::driver::DevicePtr;

    if num_tokens == 0 || hidden_dim == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    let need = (num_tokens * hidden_dim) as usize;
    if attn_out.len() < need || gate_logits.len() < need {
        return Err(SparkError::InvalidArgument(
            "attn_out/gate buffer too small".into(),
        ));
    }

    let cu_func = crate::module::load_kernel_raw(ctx, "attn_output_gate", "attn_output_gate")?;
    let cu_stream = stream.cu_stream();
    let (a_ptr, _g1) = attn_out.device_ptr(stream);
    let (g_ptr, _g2) = gate_logits.device_ptr(stream);
    // Pass attn_out as both `param_attn_out` and `param_out` — alias-safe per
    // kernel design (per-thread read-then-write of the same slot).
    let params: [*mut core::ffi::c_void; 4] = [
        &a_ptr as *const u64 as *mut _,
        &g_ptr as *const u64 as *mut _,
        &a_ptr as *const u64 as *mut _,
        &hidden_dim as *const u32 as *mut _,
    ];
    let r = unsafe {
        cuLaunchKernel(
            cu_func,
            num_tokens,
            1,
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
    if r != CUresult::CUDA_SUCCESS {
        return Err(SparkError::LaunchFailed(format!(
            "attn_output_gate_inplace: {:?}",
            r
        )));
    }
    Ok(())
}

/// GDN-hybrid Gated Attention output gate: y = attn_out * sigmoid(gate_logits).
/// Fuses sigmoid + multiply into one pass, avoiding the intermediate allocation.
pub fn attn_output_gate(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    attn_out: &CudaSlice<u16>,
    gate_logits: &CudaSlice<u16>,
    out: &mut CudaSlice<u16>,
    num_tokens: u32,
    hidden_dim: u32,
) -> Result<()> {
    if num_tokens == 0 || hidden_dim == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }

    let func = module::load_kernel(ctx, "attn_output_gate", "attn_output_gate")?;
    let cfg = LaunchConfig {
        grid_dim: (num_tokens, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(attn_out)
            .arg(gate_logits)
            .arg(out)
            .arg(&hidden_dim)
            .launch(cfg)?;
    }
    Ok(())
}

/// Backward for `attn_output_gate` (9B GDN-hybrid / GDN-hybrid hybrid attention layer).
///
/// Forward: `out = attn_out * sigmoid(gate)`
/// Backward (one fused kernel):
///   `da = dy * sigmoid(gate)`
///   `dg = dy * attn_out * sigmoid(gate) * (1 - sigmoid(gate))`
///
/// Inputs:
/// - `attn_out`:    [N, D] BF16 — the ORIGINAL pre-gate values (must be cached
///                                  by the training-driver during forward;
///                                  this kernel does NOT recover them from the
///                                  gated output).
/// - `gate_logits`: [N, D] BF16 — pre-sigmoid gate logits.
/// - `dy`:          [N, D] BF16 — upstream gradient.
///
/// Outputs:
/// - `da`: [N, D] BF16 — gradient wrt attn_out.
/// - `dg`: [N, D] BF16 — gradient wrt gate_logits.
#[allow(clippy::too_many_arguments)]
pub fn attn_output_gate_backward_bf16(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    attn_out: &CudaSlice<u16>,
    gate_logits: &CudaSlice<u16>,
    dy: &CudaSlice<u16>,
    da: &mut CudaSlice<u16>,
    dg: &mut CudaSlice<u16>,
    num_tokens: u32,
    hidden_dim: u32,
) -> Result<()> {
    if num_tokens == 0 || hidden_dim == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    let need = (num_tokens * hidden_dim) as usize;
    if attn_out.len() < need
        || gate_logits.len() < need
        || dy.len() < need
        || da.len() < need
        || dg.len() < need
    {
        return Err(SparkError::InvalidArgument(
            "attn_output_gate_backward: one or more buffers too small".into(),
        ));
    }

    let func = module::load_kernel(
        ctx,
        "attn_output_gate_backward_bf16",
        "attn_output_gate_backward_bf16",
    )?;
    let cfg = LaunchConfig {
        grid_dim: (num_tokens, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(attn_out)
            .arg(gate_logits)
            .arg(dy)
            .arg(da)
            .arg(dg)
            .arg(&hidden_dim)
            .launch(cfg)?;
    }
    Ok(())
}
