use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaSlice, CudaStream, CudaViewMut, LaunchConfig, PushKernelArg,
};

use crate::error::{Result, SparkError};
use crate::module;

/// Launch RoPE BF16 kernel (in-place).
///
/// `x`: input/output tensor [B * S * H * D] in BF16 (stored as u16), modified in-place
/// `cos_cache`: [S, D/2] in f32
/// `sin_cache`: [S, D/2] in f32
/// `batch`: B
/// `seq_len`: S
/// `heads`: H
/// `dim`: D (must be even)
#[allow(clippy::too_many_arguments)]
pub fn rope_bf16(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    x: &mut CudaSlice<u16>,
    cos_cache: &CudaSlice<f32>,
    sin_cache: &CudaSlice<f32>,
    batch: u32,
    seq_len: u32,
    heads: u32,
    dim: u32,
) -> Result<()> {
    if batch == 0 || seq_len == 0 || heads == 0 || dim == 0 {
        return Err(SparkError::InvalidArgument(format!(
            "all dimensions must be > 0: batch={batch}, seq_len={seq_len}, heads={heads}, dim={dim}"
        )));
    }
    if !dim.is_multiple_of(2) {
        return Err(SparkError::InvalidArgument(format!(
            "dim must be even: dim={dim}"
        )));
    }
    let x_need = batch as usize * seq_len as usize * heads as usize * dim as usize;
    if x.len() < x_need {
        return Err(SparkError::InvalidArgument(format!(
            "x buffer too small: {} < {x_need}",
            x.len()
        )));
    }
    let cache_need = seq_len as usize * dim as usize / 2;
    if cos_cache.len() < cache_need {
        return Err(SparkError::InvalidArgument(format!(
            "cos_cache too small: {} < {cache_need}",
            cos_cache.len()
        )));
    }
    if sin_cache.len() < cache_need {
        return Err(SparkError::InvalidArgument(format!(
            "sin_cache too small: {} < {cache_need}",
            sin_cache.len()
        )));
    }
    let half_dim = dim / 2;
    if half_dim > 1024 {
        return Err(SparkError::InvalidArgument(format!(
            "dim/2 ({half_dim}) exceeds max threads per block (1024)"
        )));
    }
    let func = module::load_kernel(ctx, "rope_bf16", "rope_bf16")?;

    let num_positions = batch * seq_len * heads;

    let cfg = LaunchConfig {
        grid_dim: (num_positions, 1, 1),
        block_dim: (half_dim, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&func)
            .arg(x)
            .arg(cos_cache)
            .arg(sin_cache)
            .arg(&seq_len)
            .arg(&heads)
            .arg(&dim)
            .arg(&half_dim)
            .launch(cfg)?;
    }

    Ok(())
}

/// Backward pass for `rope_bf16` (interleaved-pair RoPE).
///
/// Forward: `(y[2i], y[2i+1]) = (x[2i]*c - x[2i+1]*s, x[2i]*s + x[2i+1]*c)`
/// Backward: `(dx[2i], dx[2i+1]) = (dy[2i]*c + dy[2i+1]*s, -dy[2i]*s + dy[2i+1]*c)`
///
/// `dy`: upstream gradient `[B, S, H, D]` BF16.
/// `dx`: output gradient `[B, S, H, D]` BF16.
/// `cos_cache`, `sin_cache`: same f32 caches the forward used.
#[allow(clippy::too_many_arguments)]
pub fn rope_backward_bf16(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    dy: &CudaSlice<u16>,
    dx: &mut CudaSlice<u16>,
    cos_cache: &CudaSlice<f32>,
    sin_cache: &CudaSlice<f32>,
    batch: u32,
    seq_len: u32,
    heads: u32,
    dim: u32,
) -> Result<()> {
    if batch == 0 || seq_len == 0 || heads == 0 || dim == 0 {
        return Err(SparkError::InvalidArgument("all dims must be > 0".into()));
    }
    if !dim.is_multiple_of(2) {
        return Err(SparkError::InvalidArgument(format!(
            "dim must be even: {dim}"
        )));
    }
    let need = batch as usize * seq_len as usize * heads as usize * dim as usize;
    if dy.len() < need || dx.len() < need {
        return Err(SparkError::InvalidArgument(format!(
            "dy/dx buffer too small: need {need}"
        )));
    }
    let cache_need = seq_len as usize * dim as usize / 2;
    if cos_cache.len() < cache_need || sin_cache.len() < cache_need {
        return Err(SparkError::InvalidArgument(format!(
            "cache too small: need {cache_need}"
        )));
    }
    let half_dim = dim / 2;
    if half_dim > 1024 {
        return Err(SparkError::InvalidArgument(format!(
            "dim/2 ({half_dim}) > 1024 max threads"
        )));
    }
    let func = module::load_kernel(ctx, "rope_backward_bf16", "rope_backward_bf16")?;
    let num_positions = batch * seq_len * heads;
    let cfg = LaunchConfig {
        grid_dim: (num_positions, 1, 1),
        block_dim: (half_dim, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(dy)
            .arg(dx)
            .arg(cos_cache)
            .arg(sin_cache)
            .arg(&seq_len)
            .arg(&heads)
            .arg(&dim)
            .arg(&half_dim)
            .launch(cfg)?;
    }
    Ok(())
}

/// Partial in-place RoPE for the first `rotary_dim` of each head, with
/// position read from a device pointer at launch time (CUDA-Graph-capturable).
/// Computes inv_freq[i] = theta^(-2i/rotary_dim) on the fly using ex2/lg2,
/// avoiding the need for a precomputed cos/sin table.
///
/// `buf`: `[heads, head_dim]` BF16, modified in-place.
/// `pos_ptr`: device-resident u32 with the current position.
/// `theta`: rope_theta (e.g. 10000000.0 for Qwen3-Next).
/// `rotary_dim`: number of leading dims of each head to rotate (must be even
/// and ≤ head_dim and ≤ 1024).
#[allow(clippy::too_many_arguments)]
pub fn rope_partial_bf16_pos_dev(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    buf: &mut CudaSlice<u16>,
    pos_ptr: &CudaSlice<u32>,
    theta: f32,
    heads: u32,
    head_dim: u32,
    rotary_dim: u32,
) -> Result<()> {
    if heads == 0 || head_dim == 0 || rotary_dim == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    if rotary_dim & 1 != 0 || rotary_dim > head_dim {
        return Err(SparkError::InvalidArgument(format!(
            "rotary_dim must be even and ≤ head_dim, got {rotary_dim}/{head_dim}"
        )));
    }
    let need = (heads * head_dim) as usize;
    if buf.len() < need {
        return Err(SparkError::InvalidArgument("buf too small".into()));
    }
    if pos_ptr.is_empty() {
        return Err(SparkError::InvalidArgument("pos_ptr empty".into()));
    }
    let func = module::load_kernel(
        ctx,
        "rope_partial_bf16_pos_dev",
        "rope_partial_bf16_pos_dev",
    )?;
    let half = rotary_dim / 2;
    let cfg = LaunchConfig {
        grid_dim: (heads, 1, 1),
        block_dim: (half, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(buf)
            .arg(pos_ptr)
            .arg(&theta)
            .arg(&head_dim)
            .arg(&rotary_dim)
            .launch(cfg)?;
    }
    Ok(())
}

/// Backward for `rope_partial_bf16_pos_dev`. In-place: overwrites the upstream
/// `dy` gradient buffer with `dx`. Same launch shape as the forward.
///
/// Used by 9B GDN-hybrid (and any future hybrid-GDN model) training. The rotary
/// slice gets the rotation-by-negated-angle backward; the trailing
/// `head_dim - rotary_dim` positions are identity (dx = dy) and not touched.
#[allow(clippy::too_many_arguments)]
pub fn rope_partial_backward_bf16(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    buf: &mut CudaSlice<u16>,
    pos_ptr: &CudaSlice<u32>,
    theta: f32,
    heads: u32,
    head_dim: u32,
    rotary_dim: u32,
) -> Result<()> {
    if heads == 0 || head_dim == 0 || rotary_dim == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    if rotary_dim & 1 != 0 || rotary_dim > head_dim {
        return Err(SparkError::InvalidArgument(format!(
            "rotary_dim must be even and ≤ head_dim, got {rotary_dim}/{head_dim}"
        )));
    }
    let need = (heads * head_dim) as usize;
    if buf.len() < need {
        return Err(SparkError::InvalidArgument("buf too small".into()));
    }
    if pos_ptr.is_empty() {
        return Err(SparkError::InvalidArgument("pos_ptr empty".into()));
    }
    let func = module::load_kernel(
        ctx,
        "rope_partial_backward_bf16",
        "rope_partial_backward_bf16",
    )?;
    let half = rotary_dim / 2;
    let cfg = LaunchConfig {
        grid_dim: (heads, 1, 1),
        block_dim: (half, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(buf)
            .arg(pos_ptr)
            .arg(&theta)
            .arg(&head_dim)
            .arg(&rotary_dim)
            .launch(cfg)?;
    }
    Ok(())
}

/// Per-sequence partial RoPE — variant of `rope_partial_bf16_pos_dev` that
/// reads a different position per sequence from a `[M] u32` device pointer.
///
/// `buf`: `[M, heads_per_seq, head_dim]` BF16, modified in-place.
/// `pos_per_seq`: `[M]` u32 device — per-sequence current decode position.
/// `heads_per_seq`: number of heads owned by each sequence (so that
///   `seq_idx = global_head / heads_per_seq` and the kernel reads
///   `pos_per_seq[seq_idx]`).
/// `theta`: rope_theta (e.g., 10000000.0 for Qwen3-Next).
/// `rotary_dim`: number of leading dims of each head to rotate.
///
/// Used by continuous-batching servers to handle mixed-arrival workloads
/// where each of the M active slots is at a different position.
#[allow(clippy::too_many_arguments)]
pub fn rope_partial_bf16_per_seq(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    buf: &mut CudaSlice<u16>,
    pos_per_seq: &CudaSlice<u32>,
    theta: f32,
    batch_m: u32,
    heads_per_seq: u32,
    head_dim: u32,
    rotary_dim: u32,
) -> Result<()> {
    if batch_m == 0 || heads_per_seq == 0 || head_dim == 0 || rotary_dim == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    if rotary_dim & 1 != 0 || rotary_dim > head_dim {
        return Err(SparkError::InvalidArgument(format!(
            "rotary_dim must be even and <= head_dim, got {rotary_dim}/{head_dim}"
        )));
    }
    let total_heads = batch_m * heads_per_seq;
    let need = (total_heads * head_dim) as usize;
    if buf.len() < need {
        return Err(SparkError::InvalidArgument(format!(
            "buf too small: have {}, need {}",
            buf.len(),
            need
        )));
    }
    if pos_per_seq.len() < batch_m as usize {
        return Err(SparkError::InvalidArgument(format!(
            "pos_per_seq too small: have {}, need {}",
            pos_per_seq.len(),
            batch_m
        )));
    }
    let func = module::load_kernel(
        ctx,
        "rope_partial_bf16_per_seq",
        "rope_partial_bf16_per_seq",
    )?;
    let half = rotary_dim / 2;
    let cfg = LaunchConfig {
        grid_dim: (total_heads, 1, 1),
        block_dim: (half, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(buf)
            .arg(pos_per_seq)
            .arg(&theta)
            .arg(&head_dim)
            .arg(&rotary_dim)
            .arg(&heads_per_seq)
            .launch(cfg)?;
    }
    Ok(())
}

/// Per-sequence variant of `rope_proportional_bf16_pos_dev` for Gemma-4
/// full-attention layers. Reads pos_per_seq[seq_idx] (= ctaid.x /
/// heads_per_seq) instead of broadcasting a single position. Otherwise
/// identical math (partial rotation over the first `rope_angles` pair
/// indices). Enables heterogeneous-position batched decode.
#[allow(clippy::too_many_arguments)]
pub fn rope_proportional_bf16_per_seq(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    buf: &mut CudaSlice<u16>,
    pos_per_seq: &CudaSlice<u32>,
    theta: f32,
    batch_m: u32,
    heads_per_seq: u32,
    head_dim: u32,
    rope_angles: u32,
) -> Result<()> {
    if batch_m == 0 || heads_per_seq == 0 || head_dim == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    if head_dim & 1 != 0 {
        return Err(SparkError::InvalidArgument(format!(
            "head_dim must be even, got {head_dim}"
        )));
    }
    if rope_angles > head_dim / 2 {
        return Err(SparkError::InvalidArgument(format!(
            "rope_angles must be <= head_dim/2, got {rope_angles}/{}",
            head_dim / 2
        )));
    }
    let total_heads = batch_m * heads_per_seq;
    let need = (total_heads * head_dim) as usize;
    if buf.len() < need {
        return Err(SparkError::InvalidArgument(format!(
            "buf too small: have {}, need {}",
            buf.len(),
            need
        )));
    }
    if pos_per_seq.len() < batch_m as usize {
        return Err(SparkError::InvalidArgument(format!(
            "pos_per_seq too small: have {}, need {}",
            pos_per_seq.len(),
            batch_m
        )));
    }
    let func = module::load_kernel(
        ctx,
        "rope_proportional_bf16_per_seq",
        "rope_proportional_bf16_per_seq",
    )?;
    let half = head_dim / 2;
    let cfg = LaunchConfig {
        grid_dim: (total_heads, 1, 1),
        block_dim: (half, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(buf)
            .arg(pos_per_seq)
            .arg(&theta)
            .arg(&head_dim)
            .arg(&rope_angles)
            .arg(&heads_per_seq)
            .launch(cfg)?;
    }
    Ok(())
}

/// Proportional RoPE (Gemma-4 full-attention layers). Pair convention is
/// `(i, i + head_dim/2)` with the first `rope_angles` indices receiving a
/// nonzero rotation; `i in [rope_angles, head_dim/2)` get cos=1, sin=0
/// (identity, dims pass through unchanged).
///
/// `rope_angles = partial_rotary_factor * head_dim / 2`. For Gemma-4-e4b
/// full-attention layers: `head_dim=512`, `partial_rotary_factor=0.25` →
/// `rope_angles=64`.
///
/// `buf`: `[heads, head_dim]` BF16, modified in-place. `head_dim` must be even.
/// `pos_ptr`: device-resident u32 with the current position.
/// `theta`: rope_theta (e.g. 1000000.0 for Gemma-4 full attention).
/// `rope_angles`: number of nonzero inv_freq entries (must be ≤ head_dim/2).
#[allow(clippy::too_many_arguments)]
/// View-accepting variant of `rope_partial_bf16_pos_dev`. Operates on a
/// `&mut CudaViewMut<u16>` so the caller can pass a slice into a batched
/// buffer (e.g., one sequence's row inside a `[M, q_dim]` stacked tensor).
pub fn rope_partial_bf16_pos_dev_view(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    buf: &mut CudaViewMut<u16>,
    pos_ptr: &CudaSlice<u32>,
    theta: f32,
    heads: u32,
    head_dim: u32,
    rotary_dim: u32,
) -> Result<()> {
    if heads == 0 || head_dim == 0 || rotary_dim == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    if rotary_dim & 1 != 0 || rotary_dim > head_dim {
        return Err(SparkError::InvalidArgument(format!(
            "rotary_dim must be even and ≤ head_dim, got {rotary_dim}/{head_dim}"
        )));
    }
    let need = (heads * head_dim) as usize;
    if buf.len() < need {
        return Err(SparkError::InvalidArgument("buf too small".into()));
    }
    if pos_ptr.is_empty() {
        return Err(SparkError::InvalidArgument("pos_ptr empty".into()));
    }
    let func = module::load_kernel(
        ctx,
        "rope_partial_bf16_pos_dev",
        "rope_partial_bf16_pos_dev",
    )?;
    let half = rotary_dim / 2;
    let cfg = LaunchConfig {
        grid_dim: (heads, 1, 1),
        block_dim: (half, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(buf)
            .arg(pos_ptr)
            .arg(&theta)
            .arg(&head_dim)
            .arg(&rotary_dim)
            .launch(cfg)?;
    }
    Ok(())
}

/// View-accepting variant of `rope_proportional_bf16_pos_dev`.
pub fn rope_proportional_bf16_pos_dev_view(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    buf: &mut CudaViewMut<u16>,
    pos_ptr: &CudaSlice<u32>,
    theta: f32,
    heads: u32,
    head_dim: u32,
    rope_angles: u32,
) -> Result<()> {
    if heads == 0 || head_dim == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    if head_dim & 1 != 0 {
        return Err(SparkError::InvalidArgument(format!(
            "head_dim must be even, got {head_dim}"
        )));
    }
    let half = head_dim / 2;
    if rope_angles > half {
        return Err(SparkError::InvalidArgument(format!(
            "rope_angles ({rope_angles}) must be ≤ head_dim/2 ({half})"
        )));
    }
    let need = (heads * head_dim) as usize;
    if buf.len() < need {
        return Err(SparkError::InvalidArgument("buf too small".into()));
    }
    if pos_ptr.is_empty() {
        return Err(SparkError::InvalidArgument("pos_ptr empty".into()));
    }
    let func = module::load_kernel(
        ctx,
        "rope_proportional_bf16_pos_dev",
        "rope_proportional_bf16_pos_dev",
    )?;
    let cfg = LaunchConfig {
        grid_dim: (heads, 1, 1),
        block_dim: (half, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(buf)
            .arg(pos_ptr)
            .arg(&theta)
            .arg(&head_dim)
            .arg(&rope_angles)
            .launch(cfg)?;
    }
    Ok(())
}

/// Apply partial rotary embedding in place over the first `rope_angles` pair
/// indices of each head, reading the (single, broadcast) token position from
/// `pos_ptr` on the device so decode needs no host sync. `theta` is the RoPE base.
pub fn rope_proportional_bf16_pos_dev(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    buf: &mut CudaSlice<u16>,
    pos_ptr: &CudaSlice<u32>,
    theta: f32,
    heads: u32,
    head_dim: u32,
    rope_angles: u32,
) -> Result<()> {
    if heads == 0 || head_dim == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    if head_dim & 1 != 0 {
        return Err(SparkError::InvalidArgument(format!(
            "head_dim must be even, got {head_dim}"
        )));
    }
    let half = head_dim / 2;
    if rope_angles > half {
        return Err(SparkError::InvalidArgument(format!(
            "rope_angles ({rope_angles}) must be ≤ head_dim/2 ({half})"
        )));
    }
    let need = (heads * head_dim) as usize;
    if buf.len() < need {
        return Err(SparkError::InvalidArgument("buf too small".into()));
    }
    if pos_ptr.is_empty() {
        return Err(SparkError::InvalidArgument("pos_ptr empty".into()));
    }
    let func = module::load_kernel(
        ctx,
        "rope_proportional_bf16_pos_dev",
        "rope_proportional_bf16_pos_dev",
    )?;
    let cfg = LaunchConfig {
        grid_dim: (heads, 1, 1),
        block_dim: (half, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(buf)
            .arg(pos_ptr)
            .arg(&theta)
            .arg(&head_dim)
            .arg(&rope_angles)
            .launch(cfg)?;
    }
    Ok(())
}
