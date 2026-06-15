use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};

use crate::error::{Result, SparkError};
use crate::module;

/// Launch top-k sampling kernel.
///
/// Performs temperature-scaled argmax (greedy decoding when k=1) or top-k selection.
///
/// `logits`: [batch_size, vocab_size] BF16 logits
/// `indices`: [batch_size, k] u32 output token indices
/// `values`: [batch_size, k] BF16 output scaled logit values
#[allow(clippy::too_many_arguments)]
pub fn topk_sampling(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    logits: &CudaSlice<u16>,
    indices: &mut CudaSlice<u32>,
    values: &mut CudaSlice<u16>,
    batch_size: u32,
    vocab_size: u32,
    k: u32,
    temperature: f32,
) -> Result<()> {
    if batch_size == 0 {
        return Err(SparkError::InvalidArgument("batch_size must be > 0".into()));
    }
    if vocab_size == 0 {
        return Err(SparkError::InvalidArgument("vocab_size must be > 0".into()));
    }
    if k == 0 {
        return Err(SparkError::InvalidArgument("k must be > 0".into()));
    }
    if logits.len() < batch_size as usize * vocab_size as usize {
        return Err(SparkError::InvalidArgument(format!(
            "logits buffer too small: {} < {}",
            logits.len(),
            batch_size as usize * vocab_size as usize
        )));
    }
    if indices.len() < batch_size as usize * k as usize {
        return Err(SparkError::InvalidArgument(format!(
            "indices buffer too small: {} < {}",
            indices.len(),
            batch_size as usize * k as usize
        )));
    }
    if values.len() < batch_size as usize * k as usize {
        return Err(SparkError::InvalidArgument(format!(
            "values buffer too small: {} < {}",
            values.len(),
            batch_size as usize * k as usize
        )));
    }
    let func = module::load_kernel(ctx, "topk_sampling", "topk_sampling")?;

    let cfg = LaunchConfig {
        grid_dim: (batch_size, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&func)
            .arg(logits)
            .arg(indices)
            .arg(values)
            .arg(&vocab_size)
            .arg(&k)
            .arg(&temperature)
            .launch(cfg)?;
    }

    Ok(())
}

/// Launch row-wise softmax BF16 kernel with temperature.
///
/// `out[i] = exp((x[i] - max(x)) / T) / sum(exp((x[j] - max(x)) / T))`
///
/// `x`: input [num_rows, vocab_size] BF16
/// `out`: output [num_rows, vocab_size] BF16
/// `vocab_size`: dimension to softmax over
/// `num_rows`: number of rows (typically batch size)
/// `temperature`: temperature scaling (1.0 = standard softmax)
pub fn softmax_bf16(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<u16>,
    out: &mut CudaSlice<u16>,
    num_rows: u32,
    vocab_size: u32,
    temperature: f32,
) -> Result<()> {
    if num_rows == 0 || vocab_size == 0 {
        return Err(SparkError::InvalidArgument(
            "num_rows and vocab_size must be > 0".into(),
        ));
    }
    if temperature <= 0.0 {
        return Err(SparkError::InvalidArgument(
            "temperature must be positive".into(),
        ));
    }
    let need = num_rows as usize * vocab_size as usize;
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

    let func = module::load_kernel(ctx, "softmax_bf16", "softmax_bf16")?;
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
            .arg(&vocab_size)
            .arg(&temperature)
            .launch(cfg)?;
    }

    Ok(())
}

/// Backward pass for `softmax_bf16`.
///
/// Forward: `y = softmax(x, dim=-1)`
/// Backward: `dx[i] = y[i] * (dy[i] - sum_j(dy[j] * y[j]))`
///
/// Takes the saved forward output `y` (cheaper than recomputing softmax).
///
/// `y`: [num_rows, dim] BF16 — softmax forward output (saved from forward).
/// `dy`: [num_rows, dim] BF16 — upstream gradient.
/// `dx`: [num_rows, dim] BF16 — output gradient.
pub fn softmax_backward_bf16(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    y: &CudaSlice<u16>,
    dy: &CudaSlice<u16>,
    dx: &mut CudaSlice<u16>,
    num_rows: u32,
    dim: u32,
) -> Result<()> {
    if num_rows == 0 || dim == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    let need = num_rows as usize * dim as usize;
    if y.len() < need || dy.len() < need || dx.len() < need {
        return Err(SparkError::InvalidArgument(format!(
            "buffers too small: need {need}"
        )));
    }
    let func = module::load_kernel(ctx, "softmax_backward_bf16", "softmax_backward_bf16")?;
    let cfg = LaunchConfig {
        grid_dim: (num_rows, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(y)
            .arg(dy)
            .arg(dx)
            .arg(&dim)
            .launch(cfg)?;
    }
    Ok(())
}

/// Cross-entropy loss backward.
///
/// Forward: `loss[b] = -log(softmax(logits[b])[target[b]])`
/// Backward: `dlogits[b, i] = (softmax(logits[b])[i] - 1[i==target[b]]) / batch`
///
/// `logits`: [batch, vocab] BF16
/// `targets`: [batch] u32
/// `dlogits`: [batch, vocab] BF16 (output)
pub fn cross_entropy_backward_bf16(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    logits: &CudaSlice<u16>,
    targets: &CudaSlice<u32>,
    dlogits: &mut CudaSlice<u16>,
    batch: u32,
    vocab: u32,
) -> Result<()> {
    if batch == 0 || vocab == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    let need = batch as usize * vocab as usize;
    if logits.len() < need || dlogits.len() < need {
        return Err(SparkError::InvalidArgument(format!(
            "logits/dlogits too small: need {need}"
        )));
    }
    if targets.len() < batch as usize {
        return Err(SparkError::InvalidArgument(format!(
            "targets too small: {} < {batch}",
            targets.len()
        )));
    }
    let func = module::load_kernel(
        ctx,
        "cross_entropy_backward_bf16",
        "cross_entropy_backward_bf16",
    )?;
    let cfg = LaunchConfig {
        grid_dim: (batch, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(logits)
            .arg(targets)
            .arg(dlogits)
            .arg(&vocab)
            .arg(&batch)
            .launch(cfg)?;
    }
    Ok(())
}

/// Launch top-p (nucleus) filtering BF16 kernel.
///
/// Takes an input probability distribution (post-softmax), truncates to the
/// smallest set with cumulative probability >= `p_thresh`, zeros out the rest,
/// renormalizes. Useful as a preprocessor before random multinomial sampling.
///
/// `probs`: input [num_rows, vocab] BF16 probabilities (must sum to 1)
/// `out`: output [num_rows, vocab] BF16 filtered + renormalized probabilities
pub fn topp_filter_bf16(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    probs: &CudaSlice<u16>,
    out: &mut CudaSlice<u16>,
    num_rows: u32,
    vocab_size: u32,
    p_thresh: f32,
) -> Result<()> {
    if num_rows == 0 || vocab_size == 0 {
        return Err(SparkError::InvalidArgument(
            "num_rows and vocab_size must be > 0".into(),
        ));
    }
    if !(0.0..=1.0).contains(&p_thresh) {
        return Err(SparkError::InvalidArgument(
            "p_thresh must be in [0, 1]".into(),
        ));
    }
    let need = num_rows as usize * vocab_size as usize;
    if probs.len() < need || out.len() < need {
        return Err(SparkError::InvalidArgument(format!(
            "probs/out too small: need {need}"
        )));
    }

    let func = module::load_kernel(ctx, "topp_filter_bf16", "topp_filter_bf16")?;
    let cfg = LaunchConfig {
        grid_dim: (num_rows, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(probs)
            .arg(out)
            .arg(&vocab_size)
            .arg(&p_thresh)
            .launch(cfg)?;
    }
    Ok(())
}

/// Launch cross-entropy loss BF16 kernel.
///
/// For each row: loss = -log(softmax(logits)[target]).
/// Numerically stable via max subtraction. Uses natural log.
///
/// `logits`: [num_tokens, vocab] BF16
/// `targets`: [num_tokens] u32 ground-truth token ids
/// `losses`: [num_tokens] f32 output per-token loss
pub fn cross_entropy_bf16(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    logits: &CudaSlice<u16>,
    targets: &CudaSlice<u32>,
    losses: &mut CudaSlice<f32>,
    num_tokens: u32,
    vocab_size: u32,
) -> Result<()> {
    if num_tokens == 0 || vocab_size == 0 {
        return Err(SparkError::InvalidArgument(
            "num_tokens and vocab_size must be > 0".into(),
        ));
    }
    let need_logits = num_tokens as usize * vocab_size as usize;
    if logits.len() < need_logits {
        return Err(SparkError::InvalidArgument(format!(
            "logits too small: need {need_logits}"
        )));
    }
    if targets.len() < num_tokens as usize {
        return Err(SparkError::InvalidArgument("targets too small".into()));
    }
    if losses.len() < num_tokens as usize {
        return Err(SparkError::InvalidArgument("losses too small".into()));
    }

    let func = module::load_kernel(ctx, "cross_entropy_bf16", "cross_entropy_bf16")?;
    let cfg = LaunchConfig {
        grid_dim: (num_tokens, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(logits)
            .arg(targets)
            .arg(losses)
            .arg(&vocab_size)
            .launch(cfg)?;
    }
    Ok(())
}

/// Greedy argmax sampling: token[b] = argmax_v logits[b, v].
///
/// `logits`: [batch, vocab] FP32
/// `tokens`: [batch] u32 output token ids
pub fn argmax_f32(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    logits: &CudaSlice<f32>,
    tokens: &mut CudaSlice<u32>,
    batch: u32,
    vocab_size: u32,
) -> Result<()> {
    if batch == 0 || vocab_size == 0 {
        return Err(SparkError::InvalidArgument(
            "batch and vocab_size must be > 0".into(),
        ));
    }
    let need = batch as usize * vocab_size as usize;
    if logits.len() < need {
        return Err(SparkError::InvalidArgument(format!(
            "logits too small: need {need}"
        )));
    }
    if tokens.len() < batch as usize {
        return Err(SparkError::InvalidArgument("tokens too small".into()));
    }

    let func = module::load_kernel(ctx, "argmax_f32", "argmax_f32")?;
    let cfg = LaunchConfig {
        grid_dim: (batch, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(logits)
            .arg(tokens)
            .arg(&vocab_size)
            .launch(cfg)?;
    }
    Ok(())
}

/// Device-side masked argmax for constrained decoding.
///
/// `logits`: [batch, vocab] BF16
/// `mask`:   [batch, ceil(vocab/8)] u8 bitset — bit j of byte (j/8) is 1
///           iff token j is allowed at the current FSA state.
/// `tokens`: [batch] u32 output token ids (argmax over allowed only).
///
/// Eliminates the dtoh + host scan of a host-side masked argmax
/// by running everything on device. Measured grammar-decode latency drops
/// from ~150 ms/tok (host roundtrip) to within ~10% of free argmax.
pub fn masked_argmax_bf16(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    logits: &CudaSlice<u16>,
    mask: &CudaSlice<u8>,
    tokens: &mut CudaSlice<u32>,
    batch: u32,
    vocab_size: u32,
) -> Result<()> {
    if batch == 0 || vocab_size == 0 {
        return Err(SparkError::InvalidArgument(
            "batch and vocab_size must be > 0".into(),
        ));
    }
    let need_logits = batch as usize * vocab_size as usize;
    if logits.len() < need_logits {
        return Err(SparkError::InvalidArgument(format!(
            "logits too small: need {need_logits}, got {}",
            logits.len()
        )));
    }
    let mask_bytes_per_row = vocab_size.div_ceil(8) as usize;
    let need_mask = batch as usize * mask_bytes_per_row;
    if mask.len() < need_mask {
        return Err(SparkError::InvalidArgument(format!(
            "mask too small: need {need_mask}, got {}",
            mask.len()
        )));
    }
    if tokens.len() < batch as usize {
        return Err(SparkError::InvalidArgument("tokens too small".into()));
    }
    let func = module::load_kernel(ctx, "masked_argmax_bf16", "masked_argmax_bf16")?;
    let cfg = LaunchConfig {
        grid_dim: (batch, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(logits)
            .arg(mask)
            .arg(tokens)
            .arg(&vocab_size)
            .launch(cfg)?;
    }
    Ok(())
}

/// Two-stage device-side masked argmax. Higher parallelism than the single-block
/// `masked_argmax_bf16` — stage 1 launches `n_blocks` per batch, each scanning
/// `vocab / n_blocks` tokens; stage 2 reduces the `n_blocks` partials into a
/// global argmax. Recommended `n_blocks`: 4, 8, or 16 (must be ≤ 32 so stage 2's
/// single warp suffices).
///
/// Caller provides a u32 scratch buffer of length `batch * n_blocks * 2`.
#[allow(clippy::too_many_arguments)]
pub fn masked_argmax_bf16_v2(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    logits: &CudaSlice<u16>,
    mask: &CudaSlice<u8>,
    scratch_u32: &mut CudaSlice<u32>,
    tokens: &mut CudaSlice<u32>,
    batch: u32,
    vocab_size: u32,
    n_blocks: u32,
) -> Result<()> {
    if batch == 0 || vocab_size == 0 || n_blocks == 0 {
        return Err(SparkError::InvalidArgument(
            "batch/vocab/n_blocks must be > 0".into(),
        ));
    }
    if n_blocks > 32 {
        return Err(SparkError::InvalidArgument("n_blocks must be ≤ 32".into()));
    }
    let need_logits = batch as usize * vocab_size as usize;
    if logits.len() < need_logits {
        return Err(SparkError::InvalidArgument(format!(
            "logits too small: need {need_logits}, got {}",
            logits.len()
        )));
    }
    let mask_bytes_per_row = vocab_size.div_ceil(8) as usize;
    if mask.len() < batch as usize * mask_bytes_per_row {
        return Err(SparkError::InvalidArgument("mask too small".into()));
    }
    let scratch_need = batch as usize * n_blocks as usize * 2;
    if scratch_u32.len() < scratch_need {
        return Err(SparkError::InvalidArgument(format!(
            "scratch too small: need {scratch_need}, got {}",
            scratch_u32.len()
        )));
    }
    if tokens.len() < batch as usize {
        return Err(SparkError::InvalidArgument("tokens too small".into()));
    }

    // Stage 1
    let func1 = module::load_kernel(ctx, "masked_argmax_bf16_v2", "masked_argmax_bf16_stage1")?;
    let cfg1 = LaunchConfig {
        grid_dim: (n_blocks, batch, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func1)
            .arg(logits)
            .arg(mask)
            .arg(&mut *scratch_u32)
            .arg(&vocab_size)
            .arg(&n_blocks)
            .launch(cfg1)?;
    }
    // Stage 2
    let func2 = module::load_kernel(ctx, "masked_argmax_bf16_v2", "masked_argmax_bf16_stage2")?;
    let cfg2 = LaunchConfig {
        grid_dim: (batch, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func2)
            .arg(&*scratch_u32)
            .arg(tokens)
            .arg(&n_blocks)
            .launch(cfg2)?;
    }
    Ok(())
}

/// Helper: build a bitmask (vocab/8 u8s) from a sorted list of allowed token
/// ids. For grammar-constrained decoding, `allowed` is what
/// `ConstrainedDecoder::allowed()` returns.
pub fn allowed_to_bitmask(allowed: &[u32], vocab_size: u32) -> Vec<u8> {
    let bytes = vocab_size.div_ceil(8) as usize;
    let mut out = vec![0u8; bytes];
    for &tid in allowed {
        if tid >= vocab_size {
            continue;
        }
        let byte = (tid / 8) as usize;
        let bit = tid & 7;
        out[byte] |= 1 << bit;
    }
    out
}

/// Softcap (logit clamping): out = cap * tanh(input / cap).
///
/// Used by Gemma-style models to bound the magnitude of attention logits or
/// final logits to ±cap while preserving rank order.
///
/// `input`/`out` are 1D BF16 buffers of length `n` (must be even).
/// `cap` is the softcap magnitude (typically 30.0 for Gemma final logits).
/// `out` may alias `input` (kernel reads + writes same address per element).
pub fn softcap_bf16(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    input: &CudaSlice<u16>,
    out: &mut CudaSlice<u16>,
    n: u32,
    cap: f32,
) -> Result<()> {
    if n == 0 {
        return Err(SparkError::InvalidArgument("n must be > 0".into()));
    }
    if !n.is_multiple_of(2) {
        return Err(SparkError::InvalidArgument(
            "n must be even (kernel processes pairs)".into(),
        ));
    }
    if !cap.is_finite() || cap <= 0.0 {
        return Err(SparkError::InvalidArgument(format!(
            "cap must be finite and positive: {cap}"
        )));
    }
    if input.len() < n as usize {
        return Err(SparkError::InvalidArgument(format!(
            "input buffer too small: {} < {}",
            input.len(),
            n
        )));
    }
    if out.len() < n as usize {
        return Err(SparkError::InvalidArgument(format!(
            "out buffer too small: {} < {}",
            out.len(),
            n
        )));
    }

    let func = module::load_kernel(ctx, "softcap_bf16", "softcap_bf16")?;
    let threads_per_block: u32 = 256;
    // 2 elements per thread → divide n by 2 then ceil-div by threads.
    let pairs = n.div_ceil(2);
    let num_blocks = pairs.div_ceil(threads_per_block);
    // Cap grid at a sensible size; the kernel loops to cover all elements.
    let num_blocks = num_blocks.min(1024);
    let cfg = LaunchConfig {
        grid_dim: (num_blocks, 1, 1),
        block_dim: (threads_per_block, 1, 1),
        shared_mem_bytes: 0,
    };

    let in_ptr = input as *const _ as u64;
    let out_ptr = out as *const _ as u64;
    let _ = (in_ptr, out_ptr);

    unsafe {
        stream
            .launch_builder(&func)
            .arg(input)
            .arg(out)
            .arg(&n)
            .arg(&cap)
            .launch(cfg)?;
    }
    Ok(())
}
