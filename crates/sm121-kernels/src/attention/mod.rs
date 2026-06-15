use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaSlice, CudaStream, CudaView, CudaViewMut, DevicePtr, LaunchConfig,
    PushKernelArg,
};

use crate::error::{Result, SparkError};
use crate::module;

// DSA sparse-attention reference (future-work, only test-referenced) calls the
// gated pre-v3 baseline `flash_attn_bf16_d128`, so it shares the experimental gate.
#[cfg(feature = "experimental")]
pub mod dsa;

const HEAD_DIM: u32 = 128;

fn validate_attn_dims(batch: u32, num_heads: u32, seq_q: u32, seq_kv: u32) -> Result<()> {
    if batch == 0 {
        return Err(SparkError::InvalidArgument("batch must be > 0".into()));
    }
    if num_heads == 0 {
        return Err(SparkError::InvalidArgument("num_heads must be > 0".into()));
    }
    if seq_q == 0 {
        return Err(SparkError::InvalidArgument("seq_q must be > 0".into()));
    }
    if seq_kv == 0 {
        return Err(SparkError::InvalidArgument("seq_kv must be > 0".into()));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_attn_bf16_bufs(
    q: &CudaSlice<u16>,
    k: &CudaSlice<u16>,
    v: &CudaSlice<u16>,
    o: &mut CudaSlice<u16>,
    batch: u32,
    num_heads: u32,
    seq_q: u32,
    seq_kv: u32,
) -> Result<()> {
    let q_need = batch as usize * num_heads as usize * seq_q as usize * HEAD_DIM as usize;
    let kv_need = batch as usize * num_heads as usize * seq_kv as usize * HEAD_DIM as usize;
    if q.len() < q_need {
        return Err(SparkError::InvalidArgument(format!(
            "Q buffer too small: {} < {q_need}",
            q.len()
        )));
    }
    if k.len() < kv_need {
        return Err(SparkError::InvalidArgument(format!(
            "K buffer too small: {} < {kv_need}",
            k.len()
        )));
    }
    if v.len() < kv_need {
        return Err(SparkError::InvalidArgument(format!(
            "V buffer too small: {} < {kv_need}",
            v.len()
        )));
    }
    if o.len() < q_need {
        return Err(SparkError::InvalidArgument(format!(
            "O buffer too small: {} < {q_need}",
            o.len()
        )));
    }
    Ok(())
}

/// Launch BF16 Flash Attention (non-causal) with d=128.
///
/// Q, K, V: [batch, num_heads, seq, 128] row-major bf16 (u16 raw bits).
/// O: [batch, num_heads, seq_q, 128] row-major bf16 (u16 raw bits, output).
///
/// Uses FlashAttention-2 online softmax algorithm.
/// 1 warp (32 threads) per block, Br=16, Bc=64.
/// Flash Attention backward (BF16, non-GQA). Correctness-first
/// orchestration that uses scratch buffers to materialize P / dP / dS and
/// chains existing GEMM-backward kernels.
///
/// For O = softmax(Q @ K^T * scale) @ V:
///   D[m]    = sum_n(dP[m,n] * P[m,n])
///   dV      = P^T @ dO              (use gemm_bf16_backward_dB(P, dO, dV))
///   dP      = dO @ V^T              (use gemm_bf16_backward_dA(dO, V, dP))
///   dS[m,n] = P[m,n] * (dP[m,n] - D[m])
///   dQ      = (dS @ K) * scale      (use gemm_bf16 then scale)
///   dK      = (dS^T @ Q) * scale    (use gemm_bf16_backward_dB then scale)
///
/// `causal=true` enables causal masking (n > m → P[m, n] = 0).
///
/// Inputs: Q, K, V, dO all [B, H, S, d] BF16.
/// Outputs: dQ, dK, dV all [B, H, S, d] BF16. Caller need NOT zero-init.
///
/// **Performance**: this is a SCALAR/correctness-first impl using
/// per-(B,H) GEMM dispatches + 2 helper kernels per row. A follow-up will
/// fuse this into a single MMA-tiled FA backward kernel.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_backward_bf16(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u16>,
    k: &CudaSlice<u16>,
    v: &CudaSlice<u16>,
    do_: &CudaSlice<u16>,
    dq: &mut CudaSlice<u16>,
    dk: &mut CudaSlice<u16>,
    dv: &mut CudaSlice<u16>,
    batch: u32,
    num_heads: u32,
    seq: u32,
    d: u32,
    scale: f32,
    causal: bool,
) -> Result<()> {
    if batch == 0 || num_heads == 0 || seq == 0 || d == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    let bhsd = (batch * num_heads * seq * d) as usize;
    if q.len() < bhsd
        || k.len() < bhsd
        || v.len() < bhsd
        || do_.len() < bhsd
        || dq.len() < bhsd
        || dk.len() < bhsd
        || dv.len() < bhsd
    {
        return Err(SparkError::InvalidArgument(format!(
            "buffers too small: need {bhsd} BF16 elements"
        )));
    }
    let recompute_p = if causal {
        module::load_kernel(
            ctx,
            "fa_bw_recompute_p_causal_bf16",
            "fa_bw_recompute_p_causal_bf16",
        )?
    } else {
        module::load_kernel(ctx, "fa_bw_recompute_p_bf16", "fa_bw_recompute_p_bf16")?
    };
    let compute_ds = module::load_kernel(ctx, "fa_bw_compute_ds_bf16", "fa_bw_compute_ds_bf16")?;
    let scale_kernel =
        module::load_kernel(ctx, "scale_bf16_inplace_host", "scale_bf16_inplace_host")?;

    // Scratch: P, dP, dS each [S, S] BF16 per batch-head, but we can reuse
    // across batch-head pairs (process them sequentially). Allocate once.
    let scratch_n = (seq * seq) as usize;
    let mut p_scratch = stream
        .alloc_zeros::<u16>(scratch_n)
        .map_err(|e| SparkError::LaunchFailed(format!("alloc P scratch: {e:?}")))?;
    let dp_scratch = stream
        .alloc_zeros::<u16>(scratch_n)
        .map_err(|e| SparkError::LaunchFailed(format!("alloc dP scratch: {e:?}")))?;
    let mut ds_scratch = stream
        .alloc_zeros::<u16>(scratch_n)
        .map_err(|e| SparkError::LaunchFailed(format!("alloc dS scratch: {e:?}")))?;

    // Per (batch, head), do the 6-step backward sequence using view variants.
    let bh_stride = (seq * d) as usize;
    let n_dq = seq * d;
    let cfg_p = LaunchConfig {
        grid_dim: (seq, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    let cfg_ds = LaunchConfig {
        grid_dim: (seq, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    let cfg_scale = LaunchConfig {
        grid_dim: (n_dq.div_ceil(256), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    for bh in 0..(batch * num_heads) as usize {
        let off = bh * bh_stride;
        let q_v = q.slice(off..off + bh_stride);
        let k_v = k.slice(off..off + bh_stride);
        let v_v = v.slice(off..off + bh_stride);
        let do_v = do_.slice(off..off + bh_stride);
        let mut dq_v = dq.slice_mut(off..off + bh_stride);
        let mut dk_v = dk.slice_mut(off..off + bh_stride);
        let mut dv_v = dv.slice_mut(off..off + bh_stride);

        // 1. P = softmax(Q @ K^T * scale)
        unsafe {
            stream
                .launch_builder(&recompute_p)
                .arg(&q_v)
                .arg(&k_v)
                .arg(&mut p_scratch)
                .arg(&seq)
                .arg(&d)
                .arg(&scale)
                .launch(cfg_p)?;
        }

        // 2. dV = P^T @ dO
        crate::gemm::gemm_bf16_backward_dB_view(
            ctx,
            stream,
            &p_scratch.as_view(),
            &do_v,
            &mut dv_v,
            seq,
            d,
            seq,
        )?;

        // 3. dP = dO @ V^T
        let mut dp_view = dp_scratch.as_view_mut();
        crate::gemm::gemm_bf16_backward_dA_view(
            ctx,
            stream,
            &do_v,
            &v_v,
            &mut dp_view,
            seq,
            d,
            seq,
        )?;

        // 4. dS = P * (dP - rowsum(dP*P))
        unsafe {
            stream
                .launch_builder(&compute_ds)
                .arg(&p_scratch)
                .arg(&dp_scratch)
                .arg(&mut ds_scratch)
                .arg(&seq)
                .launch(cfg_ds)?;
        }

        // 5. dQ = dS @ K, then *= scale
        crate::gemm::gemm_bf16_view(
            ctx,
            stream,
            &ds_scratch.as_view(),
            &k_v,
            &mut dq_v,
            seq,
            d,
            seq,
        )?;
        unsafe {
            stream
                .launch_builder(&scale_kernel)
                .arg(&mut dq_v)
                .arg(&n_dq)
                .arg(&scale)
                .launch(cfg_scale)?;
        }

        // 6. dK = dS^T @ Q, then *= scale
        crate::gemm::gemm_bf16_backward_dB_view(
            ctx,
            stream,
            &ds_scratch.as_view(),
            &q_v,
            &mut dk_v,
            seq,
            d,
            seq,
        )?;
        unsafe {
            stream
                .launch_builder(&scale_kernel)
                .arg(&mut dk_v)
                .arg(&n_dq)
                .arg(&scale)
                .launch(cfg_scale)?;
        }
    }
    Ok(())
}

/// Flash Attention backward (BF16) with GQA support.
///
/// `q`: [B, H_q, S, d] BF16
/// `k`, `v`: [B, H_kv, S, d] BF16  (H_kv divides H_q)
/// `do_`: [B, H_q, S, d] BF16
/// `dq`: [B, H_q, S, d] BF16 (output)
/// `dk`, `dv`: [B, H_kv, S, d] BF16 (output)
/// `causal`: enable causal masking
///
/// Implementation (correctness MVP):
/// 1. Expand K, V to H_q heads via `repeat_kv_groups_bf16`
/// 2. Run non-GQA `flash_attn_backward_bf16` on expanded buffers
/// 3. Reduce dK_expanded, dV_expanded back to H_kv heads via
///    `sum_kv_groups_bf16`
///
/// Follow-up: direct GQA backward without expand-then-contract.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_backward_bf16_gqa(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u16>,
    k: &CudaSlice<u16>,
    v: &CudaSlice<u16>,
    do_: &CudaSlice<u16>,
    dq: &mut CudaSlice<u16>,
    dk: &mut CudaSlice<u16>,
    dv: &mut CudaSlice<u16>,
    batch: u32,
    num_heads_q: u32,
    num_heads_kv: u32,
    seq: u32,
    d: u32,
    scale: f32,
    causal: bool,
) -> Result<()> {
    if num_heads_kv == 0 || !num_heads_q.is_multiple_of(num_heads_kv) {
        return Err(SparkError::InvalidArgument(format!(
            "num_heads_kv ({num_heads_kv}) must divide num_heads_q ({num_heads_q})"
        )));
    }
    let q_elems = (batch * num_heads_q * seq * d) as usize;
    let kv_elems = (batch * num_heads_kv * seq * d) as usize;

    // Allocate expanded K, V and dK, dV scratch [B, H_q, S, d]
    let mut k_exp = stream
        .alloc_zeros::<u16>(q_elems)
        .map_err(|e| SparkError::LaunchFailed(format!("alloc k_exp: {e:?}")))?;
    let mut v_exp = stream
        .alloc_zeros::<u16>(q_elems)
        .map_err(|e| SparkError::LaunchFailed(format!("alloc v_exp: {e:?}")))?;
    let mut dk_exp = stream
        .alloc_zeros::<u16>(q_elems)
        .map_err(|e| SparkError::LaunchFailed(format!("alloc dk_exp: {e:?}")))?;
    let mut dv_exp = stream
        .alloc_zeros::<u16>(q_elems)
        .map_err(|e| SparkError::LaunchFailed(format!("alloc dv_exp: {e:?}")))?;

    // 1. Expand K, V from [B, H_kv, S, d] to [B, H_q, S, d]
    let repeat = module::load_kernel(ctx, "repeat_kv_groups_bf16", "repeat_kv_groups_bf16")?;
    let cfg_repeat = LaunchConfig {
        grid_dim: ((q_elems as u32).div_ceil(256), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&repeat)
            .arg(k)
            .arg(&mut k_exp)
            .arg(&batch)
            .arg(&num_heads_kv)
            .arg(&num_heads_q)
            .arg(&seq)
            .arg(&d)
            .launch(cfg_repeat)?;
        stream
            .launch_builder(&repeat)
            .arg(v)
            .arg(&mut v_exp)
            .arg(&batch)
            .arg(&num_heads_kv)
            .arg(&num_heads_q)
            .arg(&seq)
            .arg(&d)
            .launch(cfg_repeat)?;
    }
    let _ = kv_elems;

    // 2. Non-GQA backward on expanded tensors
    flash_attn_backward_bf16(
        ctx,
        stream,
        q,
        &k_exp,
        &v_exp,
        do_,
        dq,
        &mut dk_exp,
        &mut dv_exp,
        batch,
        num_heads_q,
        seq,
        d,
        scale,
        causal,
    )?;

    // 3. Sum-reduce dK_expanded → dK, dV_expanded → dV
    let sum_groups = module::load_kernel(ctx, "sum_kv_groups_bf16", "sum_kv_groups_bf16")?;
    let n_kv = batch * num_heads_kv * seq * d;
    let cfg_sum = LaunchConfig {
        grid_dim: (n_kv.div_ceil(256), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&sum_groups)
            .arg(&dk_exp)
            .arg(dk)
            .arg(&batch)
            .arg(&num_heads_kv)
            .arg(&num_heads_q)
            .arg(&seq)
            .arg(&d)
            .launch(cfg_sum)?;
        stream
            .launch_builder(&sum_groups)
            .arg(&dv_exp)
            .arg(dv)
            .arg(&batch)
            .arg(&num_heads_kv)
            .arg(&num_heads_q)
            .arg(&seq)
            .arg(&d)
            .launch(cfg_sum)?;
    }
    Ok(())
}

/// FP8 KV cache Flash Attention backward (BF16 Q + FP8 K/V → BF16 dQ/dK/dV).
///
/// Used during training when KV cache is stored in FP8 (Tensor Engine convention).
/// Dequantizes K/V to BF16, runs BF16 backward, returns BF16 gradients.
/// dK/dV are BF16; quantization back to FP8 for storage is the caller's job.
///
/// Same shape conventions as `flash_attn_backward_bf16`.
/// `kv_scale`: per-tensor scale used to dequantize K and V.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_backward_fp8kv_bf16(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u16>,
    k_fp8: &CudaSlice<u8>,
    v_fp8: &CudaSlice<u8>,
    do_: &CudaSlice<u16>,
    dq: &mut CudaSlice<u16>,
    dk: &mut CudaSlice<u16>,
    dv: &mut CudaSlice<u16>,
    batch: u32,
    num_heads: u32,
    seq: u32,
    d: u32,
    scale: f32,
    kv_scale: f32,
    causal: bool,
) -> Result<()> {
    let kv_elems = (batch * num_heads * seq * d) as usize;
    let mut k_bf16 = stream
        .alloc_zeros::<u16>(kv_elems)
        .map_err(|e| SparkError::LaunchFailed(format!("alloc k_bf16: {e:?}")))?;
    let mut v_bf16 = stream
        .alloc_zeros::<u16>(kv_elems)
        .map_err(|e| SparkError::LaunchFailed(format!("alloc v_bf16: {e:?}")))?;
    crate::quantization::dequant_fp8_bf16_pertensor(
        ctx,
        stream,
        k_fp8,
        &mut k_bf16,
        kv_elems as u32,
        kv_scale,
    )?;
    crate::quantization::dequant_fp8_bf16_pertensor(
        ctx,
        stream,
        v_fp8,
        &mut v_bf16,
        kv_elems as u32,
        kv_scale,
    )?;
    flash_attn_backward_bf16(
        ctx, stream, q, &k_bf16, &v_bf16, do_, dq, dk, dv, batch, num_heads, seq, d, scale, causal,
    )
}

/// View variant of permute helper for slicing into a packed total buffer.
pub fn permute_thd_to_htd_bf16_view(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &cudarc::driver::CudaView<u16>,
    dst: &mut CudaSlice<u16>,
    t: u32,
    h: u32,
    d: u32,
) -> Result<()> {
    let func = module::load_kernel(ctx, "permute_thd_htd_bf16", "permute_thd_to_htd_bf16")?;
    let cfg = LaunchConfig {
        grid_dim: (t, h, 1),
        block_dim: (256.min(d).max(32), 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(src)
            .arg(dst)
            .arg(&t)
            .arg(&h)
            .arg(&d)
            .launch(cfg)?;
    }
    Ok(())
}

/// Helper: permute `[H, T, D]` -> `[T, H, D]`. Inverse of
/// `permute_thd_to_htd_bf16`.
pub fn permute_htd_to_thd_bf16_view(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<u16>,
    dst: &mut cudarc::driver::CudaViewMut<u16>,
    t: u32,
    h: u32,
    d: u32,
) -> Result<()> {
    let func = module::load_kernel(ctx, "permute_thd_htd_bf16", "permute_htd_to_thd_bf16")?;
    let cfg = LaunchConfig {
        grid_dim: (t, h, 1),
        block_dim: (256.min(d).max(32), 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(src)
            .arg(dst)
            .arg(&t)
            .arg(&h)
            .arg(&d)
            .launch(cfg)?;
    }
    Ok(())
}

/// Flash Attention backward (BF16, varlen / packed sequences).
///
/// Layout:
/// - `q`, `k`, `v`, `do_`, `dq`, `dk`, `dv`: `[total_tokens, num_heads, d]`
///   — heads-inner packed layout matching the existing varlen forward.
/// - `cu_seqlens`: `[batch + 1]` u32 host-readable cumulative offsets.
///
/// Implementation: per-sequence orchestration. For each sequence i:
/// 1. Permute its `[S_i, H, D]` slice of `q/k/v/do_` into `[H, S_i, D]`
///    scratch buffers.
/// 2. Run the standard `flash_attn_backward_bf16` with batch=1, H, S=S_i.
/// 3. Permute the resulting `[H, S_i, D]` `dq/dk/dv` back to `[S_i, H, D]`
///    in the packed output buffers.
///
/// This adds 2 permutes per sequence — bandwidth-bound but small relative
/// to the O(S²) backward compute. Once a fused varlen FA backward kernel
/// exists this orchestrator can be retired.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_backward_bf16_varlen(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u16>,
    k: &CudaSlice<u16>,
    v: &CudaSlice<u16>,
    do_: &CudaSlice<u16>,
    dq: &mut CudaSlice<u16>,
    dk: &mut CudaSlice<u16>,
    dv: &mut CudaSlice<u16>,
    cu_seqlens: &[u32], // host-side cumulative seq lengths, length = batch + 1
    num_heads: u32,
    d: u32,
    scale: f32,
    causal: bool,
) -> Result<()> {
    if cu_seqlens.len() < 2 {
        return Err(SparkError::InvalidArgument(
            "cu_seqlens must have at least 2 entries (batch >= 1)".into(),
        ));
    }
    if cu_seqlens[0] != 0 {
        return Err(SparkError::InvalidArgument(
            "cu_seqlens[0] must be 0".into(),
        ));
    }
    let total_tokens = *cu_seqlens.last().unwrap();
    let need = (total_tokens * num_heads * d) as usize;
    for (name, buf) in [("q", q), ("k", k), ("v", v), ("do_", do_)] {
        if buf.len() < need {
            return Err(SparkError::InvalidArgument(format!(
                "{name} too small: {} < {need}",
                buf.len()
            )));
        }
    }
    for (name, buf) in [("dq", dq.len()), ("dk", dk.len()), ("dv", dv.len())] {
        if buf < need {
            return Err(SparkError::InvalidArgument(format!(
                "{name} too small: {buf} < {need}"
            )));
        }
    }

    // Find max sequence length to allocate scratch once.
    let mut max_s: u32 = 0;
    for w in cu_seqlens.windows(2) {
        max_s = max_s.max(w[1] - w[0]);
    }
    let scratch_n = (num_heads * max_s * d) as usize;

    let mut q_htd = stream
        .alloc_zeros::<u16>(scratch_n)
        .map_err(|e| SparkError::LaunchFailed(format!("alloc q_htd: {e:?}")))?;
    let mut k_htd = stream
        .alloc_zeros::<u16>(scratch_n)
        .map_err(|e| SparkError::LaunchFailed(format!("alloc k_htd: {e:?}")))?;
    let mut v_htd = stream
        .alloc_zeros::<u16>(scratch_n)
        .map_err(|e| SparkError::LaunchFailed(format!("alloc v_htd: {e:?}")))?;
    let mut do_htd = stream
        .alloc_zeros::<u16>(scratch_n)
        .map_err(|e| SparkError::LaunchFailed(format!("alloc do_htd: {e:?}")))?;
    let mut dq_htd = stream
        .alloc_zeros::<u16>(scratch_n)
        .map_err(|e| SparkError::LaunchFailed(format!("alloc dq_htd: {e:?}")))?;
    let mut dk_htd = stream
        .alloc_zeros::<u16>(scratch_n)
        .map_err(|e| SparkError::LaunchFailed(format!("alloc dk_htd: {e:?}")))?;
    let mut dv_htd = stream
        .alloc_zeros::<u16>(scratch_n)
        .map_err(|e| SparkError::LaunchFailed(format!("alloc dv_htd: {e:?}")))?;

    let hd = (num_heads * d) as usize;

    for i in 0..(cu_seqlens.len() - 1) {
        let s_start = cu_seqlens[i] as usize;
        let s_end = cu_seqlens[i + 1] as usize;
        let s_i = (s_end - s_start) as u32;
        if s_i == 0 {
            continue;
        }
        let off = s_start * hd;
        let len = (s_i as usize) * hd;
        let q_seq = q.slice(off..off + len);
        let k_seq = k.slice(off..off + len);
        let v_seq = v.slice(off..off + len);
        let do_seq = do_.slice(off..off + len);

        // 1. Permute [S_i, H, D] -> [H, S_i, D] for inputs.
        permute_thd_to_htd_bf16_view(ctx, stream, &q_seq, &mut q_htd, s_i, num_heads, d)?;
        permute_thd_to_htd_bf16_view(ctx, stream, &k_seq, &mut k_htd, s_i, num_heads, d)?;
        permute_thd_to_htd_bf16_view(ctx, stream, &v_seq, &mut v_htd, s_i, num_heads, d)?;
        permute_thd_to_htd_bf16_view(ctx, stream, &do_seq, &mut do_htd, s_i, num_heads, d)?;

        // 2. Run standard backward on [1, H, S_i, D].
        flash_attn_backward_bf16(
            ctx,
            stream,
            &q_htd,
            &k_htd,
            &v_htd,
            &do_htd,
            &mut dq_htd,
            &mut dk_htd,
            &mut dv_htd,
            1,
            num_heads,
            s_i,
            d,
            scale,
            causal,
        )?;

        // 3. Permute outputs [H, S_i, D] back to [S_i, H, D] in dq/dk/dv.
        let mut dq_seq = dq.slice_mut(off..off + len);
        let mut dk_seq = dk.slice_mut(off..off + len);
        let mut dv_seq = dv.slice_mut(off..off + len);
        permute_htd_to_thd_bf16_view(ctx, stream, &dq_htd, &mut dq_seq, s_i, num_heads, d)?;
        permute_htd_to_thd_bf16_view(ctx, stream, &dk_htd, &mut dk_seq, s_i, num_heads, d)?;
        permute_htd_to_thd_bf16_view(ctx, stream, &dv_htd, &mut dv_seq, s_i, num_heads, d)?;
    }

    Ok(())
}

/// Helper: gather paged K or V into contiguous `[B, H_kv, S_kv, D]` layout
/// for the backward orchestrator.
#[allow(clippy::too_many_arguments)]
fn paged_kv_gather_bf16(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    kv_paged: &CudaSlice<u16>,
    kv_out: &mut CudaSlice<u16>,
    page_table: &CudaSlice<u32>,
    batch: u32,
    num_kv_heads: u32,
    d: u32,
    page_size: u32,
    max_pages: u32,
    s_kv: u32,
) -> Result<()> {
    let func = module::load_kernel(ctx, "paged_kv_gather_bf16", "paged_kv_gather_bf16")?;
    let cfg = LaunchConfig {
        grid_dim: (s_kv, batch, 1),
        block_dim: (256.min((num_kv_heads * d).max(32)), 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(kv_paged)
            .arg(kv_out)
            .arg(page_table)
            .arg(&num_kv_heads)
            .arg(&d)
            .arg(&page_size)
            .arg(&max_pages)
            .arg(&s_kv)
            .launch(cfg)?;
    }
    Ok(())
}

/// Helper: scatter contiguous `[B, H_kv, S_kv, D]` back to paged layout.
/// Plain (non-atomic) write — caller must guarantee no page sharing across
/// batches. For training (which never shares pages), this is fine.
#[allow(clippy::too_many_arguments)]
fn paged_kv_scatter_bf16(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    dkv_in: &CudaSlice<u16>,
    dkv_paged: &mut CudaSlice<u16>,
    page_table: &CudaSlice<u32>,
    batch: u32,
    num_kv_heads: u32,
    d: u32,
    page_size: u32,
    max_pages: u32,
    s_kv: u32,
) -> Result<()> {
    let func = module::load_kernel(ctx, "paged_kv_scatter_atomic_bf16", "paged_kv_scatter_bf16")?;
    let cfg = LaunchConfig {
        grid_dim: (s_kv, batch, 1),
        block_dim: (256.min((num_kv_heads * d).max(32)), 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(dkv_in)
            .arg(dkv_paged)
            .arg(page_table)
            .arg(&num_kv_heads)
            .arg(&d)
            .arg(&page_size)
            .arg(&max_pages)
            .arg(&s_kv)
            .launch(cfg)?;
    }
    Ok(())
}

/// Flash Attention backward (BF16) against a paged KV cache.
///
/// Layouts:
/// - `q`: `[B, H_q, Sq, D]` BF16 (replicated, contiguous)
/// - `k_paged`, `v_paged`: `[num_pages, page_size, H_kv, D]` BF16
/// - `page_table`: `[B, max_pages]` u32, per-batch list of physical page indices
/// - `do_`: `[B, H_q, Sq, D]` BF16
/// - `dq`: `[B, H_q, Sq, D]` BF16 (output)
/// - `dk_paged`, `dv_paged`: `[num_pages, page_size, H_kv, D]` BF16 (output)
///
/// Implementation: gather pages into a contiguous `[B, H_kv, S_kv, D]`
/// buffer, run the GQA backward, scatter dK/dV back. Pages are NOT
/// atomically updated — this orchestrator assumes no page sharing across
/// batches (the typical training case). For inference KV caches with
/// prefix sharing, callers must serialize backward calls per shared page
/// or convert to a non-shared layout first.
///
/// `s_kv` is the gathered length; must be `pages_per_seq * page_size` and
/// `pages_per_seq = ceil(seq_kv / page_size)`. The kernel reads/writes the
/// full `S_kv` slice; callers concerned with padding should pre-zero V
/// past the valid token count and ignore dK/dV in that region.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_backward_bf16_paged(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u16>,
    k_paged: &CudaSlice<u16>,
    v_paged: &CudaSlice<u16>,
    do_: &CudaSlice<u16>,
    dq: &mut CudaSlice<u16>,
    dk_paged: &mut CudaSlice<u16>,
    dv_paged: &mut CudaSlice<u16>,
    page_table: &CudaSlice<u32>,
    batch: u32,
    num_heads: u32,
    num_kv_heads: u32,
    seq_q: u32,
    seq_kv: u32,
    d: u32,
    page_size: u32,
    max_pages: u32,
    scale: f32,
    causal: bool,
) -> Result<()> {
    if !num_heads.is_multiple_of(num_kv_heads) {
        return Err(SparkError::InvalidArgument(
            "num_heads must be divisible by num_kv_heads".into(),
        ));
    }
    if seq_q != seq_kv {
        return Err(SparkError::InvalidArgument(
            "paged backward requires seq_q == seq_kv (training case)".into(),
        ));
    }
    let pages_per_seq = seq_kv.div_ceil(page_size);
    let s_kv = pages_per_seq * page_size;

    // Gather K/V into [B, H_kv, S_kv, D] contiguous.
    let gather_n = (batch * num_kv_heads * s_kv * d) as usize;
    let mut k_flat = stream
        .alloc_zeros::<u16>(gather_n)
        .map_err(|e| SparkError::LaunchFailed(format!("alloc k_flat: {e:?}")))?;
    let mut v_flat = stream
        .alloc_zeros::<u16>(gather_n)
        .map_err(|e| SparkError::LaunchFailed(format!("alloc v_flat: {e:?}")))?;
    let mut dk_flat = stream
        .alloc_zeros::<u16>(gather_n)
        .map_err(|e| SparkError::LaunchFailed(format!("alloc dk_flat: {e:?}")))?;
    let mut dv_flat = stream
        .alloc_zeros::<u16>(gather_n)
        .map_err(|e| SparkError::LaunchFailed(format!("alloc dv_flat: {e:?}")))?;

    paged_kv_gather_bf16(
        ctx,
        stream,
        k_paged,
        &mut k_flat,
        page_table,
        batch,
        num_kv_heads,
        d,
        page_size,
        max_pages,
        s_kv,
    )?;
    paged_kv_gather_bf16(
        ctx,
        stream,
        v_paged,
        &mut v_flat,
        page_table,
        batch,
        num_kv_heads,
        d,
        page_size,
        max_pages,
        s_kv,
    )?;

    // Run standard GQA backward (or non-GQA if H_q == H_kv).
    if num_heads == num_kv_heads {
        flash_attn_backward_bf16(
            ctx,
            stream,
            q,
            &k_flat,
            &v_flat,
            do_,
            dq,
            &mut dk_flat,
            &mut dv_flat,
            batch,
            num_heads,
            s_kv,
            d,
            scale,
            causal,
        )?;
    } else {
        flash_attn_backward_bf16_gqa(
            ctx,
            stream,
            q,
            &k_flat,
            &v_flat,
            do_,
            dq,
            &mut dk_flat,
            &mut dv_flat,
            batch,
            num_heads,
            num_kv_heads,
            s_kv,
            d,
            scale,
            causal,
        )?;
    }

    // Scatter dK/dV back to paged layout.
    paged_kv_scatter_bf16(
        ctx,
        stream,
        &dk_flat,
        dk_paged,
        page_table,
        batch,
        num_kv_heads,
        d,
        page_size,
        max_pages,
        s_kv,
    )?;
    paged_kv_scatter_bf16(
        ctx,
        stream,
        &dv_flat,
        dv_paged,
        page_table,
        batch,
        num_kv_heads,
        d,
        page_size,
        max_pages,
        s_kv,
    )?;

    Ok(())
}

/// V1-generation non-causal BF16 flash attention (head_dim=128, single warp per
/// block, Br=16). Superseded by `flash_attn_bf16_v3_d128`; gated behind the
/// `experimental` feature and not part of the stable surface.
#[cfg(feature = "experimental")]
pub fn flash_attn_bf16_d128(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u16>,
    k: &CudaSlice<u16>,
    v: &CudaSlice<u16>,
    o: &mut CudaSlice<u16>,
    batch: u32,
    num_heads: u32,
    seq_q: u32,
    seq_kv: u32,
    scale: f32,
) -> Result<()> {
    validate_attn_dims(batch, num_heads, seq_q, seq_kv)?;
    validate_attn_bf16_bufs(q, k, v, o, batch, num_heads, seq_q, seq_kv)?;
    let func = module::load_kernel(ctx, "fa_bf16_d128", "flash_attn_bf16_d128")?;

    let grid_x = seq_q.div_ceil(16); // Q blocks (Br=16)
    let cfg = LaunchConfig {
        grid_dim: (grid_x, num_heads, batch),
        block_dim: (32, 1, 1), // 1 warp
        shared_mem_bytes: 18432,
    };

    unsafe {
        stream
            .launch_builder(&func)
            .arg(q)
            .arg(k)
            .arg(v)
            .arg(o)
            .arg(&seq_q)
            .arg(&seq_kv)
            .arg(&num_heads)
            .arg(&scale)
            .launch(cfg)?;
    }

    Ok(())
}

/// Launch BF16 Flash Attention V3 (non-causal) with d=128.
///
/// Q, K, V: [batch, num_heads, seq, 128] row-major bf16 (u16 raw bits).
/// O: [batch, num_heads, seq_q, 128] row-major bf16 (u16 raw bits, output).
///
/// V3 uses 8 warps (256 threads) per block, Br=128, Bc=64.
/// 32 KB shared memory. Designed for higher throughput than V1.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_bf16_v3_d128(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u16>,
    k: &CudaSlice<u16>,
    v: &CudaSlice<u16>,
    o: &mut CudaSlice<u16>,
    batch: u32,
    num_heads: u32,
    seq_q: u32,
    seq_kv: u32,
    scale: f32,
) -> Result<()> {
    validate_attn_dims(batch, num_heads, seq_q, seq_kv)?;
    validate_attn_bf16_bufs(q, k, v, o, batch, num_heads, seq_q, seq_kv)?;
    let func = module::load_kernel(ctx, "fa_bf16_v3_d128", "flash_attn_bf16_v3_d128")?;

    let grid_x = seq_q.div_ceil(128); // Q blocks (Br=128)
    let cfg = LaunchConfig {
        grid_dim: (grid_x, num_heads, batch),
        block_dim: (256, 1, 1), // 8 warps
        shared_mem_bytes: 0,    // smem is statically declared in PTX
    };

    unsafe {
        stream
            .launch_builder(&func)
            .arg(q)
            .arg(k)
            .arg(v)
            .arg(o)
            .arg(&seq_q)
            .arg(&seq_kv)
            .arg(&num_heads)
            .arg(&scale)
            .launch(cfg)?;
    }

    Ok(())
}

/// Launch BF16 Flash Attention V22 DB (non-causal) with d=128.
///
/// Same algorithm and output as `flash_attn_bf16_v3_d128`, but with a 2-stage
/// double-buffered async pipeline (K0/K1/V0/V1 ping-pong buffers + a dedicated
/// P region) to hide the KV-load latency that caps v3 at long sequences.
///
/// Uses 80 KB dynamic shared memory, so it must set
/// `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES` before launch and pass the
/// SMEM size to the launch (the PTX declares `.extern .shared`).
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_bf16_v22_db(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u16>,
    k: &CudaSlice<u16>,
    v: &CudaSlice<u16>,
    o: &mut CudaSlice<u16>,
    batch: u32,
    num_heads: u32,
    seq_q: u32,
    seq_kv: u32,
    scale: f32,
) -> Result<()> {
    validate_attn_dims(batch, num_heads, seq_q, seq_kv)?;
    validate_attn_bf16_bufs(q, k, v, o, batch, num_heads, seq_q, seq_kv)?;

    use cudarc::driver::sys::*;

    let cu_func = module::load_kernel_raw(ctx, "fa_bf16_v22_db", "flash_attn_bf16_v22_db")?;

    // 80 KB dynamic SMEM: K0/K1/V0/V1/P = 5 * 64*128*2 = 81920 bytes.
    const SMEM_TOTAL: i32 = 81920;
    let attr_result = unsafe {
        cuFuncSetAttribute(
            cu_func,
            CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            SMEM_TOTAL,
        )
    };
    if attr_result != CUresult::CUDA_SUCCESS {
        return Err(SparkError::LaunchFailed(format!(
            "cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES={}) failed: {:?}",
            SMEM_TOTAL, attr_result
        )));
    }

    let (q_dptr, _q_sync) = q.device_ptr(stream);
    let (k_dptr, _k_sync) = k.device_ptr(stream);
    let (v_dptr, _v_sync) = v.device_ptr(stream);
    let (o_dptr, _o_sync) = o.device_ptr(stream);

    let grid_x = seq_q.div_ceil(128); // Br=128
    let cu_stream = stream.cu_stream();

    let params: [*mut core::ffi::c_void; 8] = [
        &q_dptr as *const u64 as *mut _,
        &k_dptr as *const u64 as *mut _,
        &v_dptr as *const u64 as *mut _,
        &o_dptr as *const u64 as *mut _,
        &seq_q as *const u32 as *mut _,
        &seq_kv as *const u32 as *mut _,
        &num_heads as *const u32 as *mut _,
        &scale as *const f32 as *mut _,
    ];

    let result = unsafe {
        cuLaunchKernel(
            cu_func,
            grid_x,
            num_heads,
            batch, // grid
            256,
            1,
            1,                 // block (8 warps)
            SMEM_TOTAL as u32, // dynamic SMEM
            cu_stream,
            params.as_ptr() as *mut _,
            core::ptr::null_mut(),
        )
    };
    if result != CUresult::CUDA_SUCCESS {
        return Err(SparkError::LaunchFailed(format!(
            "cuLaunchKernel failed for V22 DB: {:?}",
            result
        )));
    }
    Ok(())
}

/// Launch BF16 Flash Attention (non-causal) with d=256.
///
/// Q, K, V: [batch, num_heads, seq, 256] row-major bf16 (u16 raw bits).
/// O: [batch, num_heads, seq_q, 256] row-major bf16 (output).
/// scale: typically 1/sqrt(256).
///
/// Built for GDN-hybrid's gated full-attention layers (10/40 layers, head_dim=256).
/// V3 architecture (Br=128, Bc=64, 8 warps × 16 rows each). Uses 65536 bytes SMEM (peak), 249/256 regs,
/// 0 spills. ~2000 lines of unrolled MMA (16 k-chunks × 8 ng QK + 4 kp × 32 ng PV).
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_bf16_v3_d256(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u16>,
    k: &CudaSlice<u16>,
    v: &CudaSlice<u16>,
    o: &mut CudaSlice<u16>,
    batch: u32,
    num_heads: u32,
    seq_q: u32,
    seq_kv: u32,
    scale: f32,
) -> Result<()> {
    const D: u32 = 256;
    validate_attn_dims(batch, num_heads, seq_q, seq_kv)?;
    let q_need = batch as usize * num_heads as usize * seq_q as usize * D as usize;
    let kv_need = batch as usize * num_heads as usize * seq_kv as usize * D as usize;
    if q.len() < q_need {
        return Err(SparkError::InvalidArgument(format!(
            "Q buffer too small: {} < {q_need}",
            q.len()
        )));
    }
    if k.len() < kv_need {
        return Err(SparkError::InvalidArgument(format!(
            "K buffer too small: {} < {kv_need}",
            k.len()
        )));
    }
    if v.len() < kv_need {
        return Err(SparkError::InvalidArgument(format!(
            "V buffer too small: {} < {kv_need}",
            v.len()
        )));
    }
    if o.len() < q_need {
        return Err(SparkError::InvalidArgument(format!(
            "O buffer too small: {} < {q_need}",
            o.len()
        )));
    }
    let func = module::load_kernel(ctx, "fa_bf16_v3_d256", "flash_attn_bf16_v3_d256")?;

    let grid_x = seq_q.div_ceil(128); // Br=128
    let cfg = LaunchConfig {
        grid_dim: (grid_x, num_heads, batch),
        block_dim: (256, 1, 1), // 8 warps
        shared_mem_bytes: 0,    // smem is statically declared in PTX
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(q)
            .arg(k)
            .arg(v)
            .arg(o)
            .arg(&seq_q)
            .arg(&seq_kv)
            .arg(&num_heads)
            .arg(&scale)
            .launch(cfg)?;
    }
    Ok(())
}

/// GQA variant of `flash_attn_bf16_v3_d256`. K/V have `num_heads_kv` heads; each Q
/// head q maps to head_kv = (q * num_heads_kv) / num_heads. Required for GDN-hybrid's
/// gated full-attention layers (16 Q heads, 2 KV heads, ratio 8).
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_bf16_v3_d256_gqa(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u16>,
    k: &CudaSlice<u16>,
    v: &CudaSlice<u16>,
    o: &mut CudaSlice<u16>,
    batch: u32,
    num_heads: u32,
    num_heads_kv: u32,
    seq_q: u32,
    seq_kv: u32,
    scale: f32,
) -> Result<()> {
    flash_attn_bf16_v3_d256_gqa_inner(
        ctx,
        stream,
        q,
        k,
        v,
        o,
        batch,
        num_heads,
        num_heads_kv,
        seq_q,
        seq_kv,
        scale,
        false,
        0,
    )
}

/// GQA + causal combined variant. `q_pos_offset` defaults to 0 (prefill); pass
/// `current_position` for decode (`seq_q=1`) so the causal mask gates the
/// correct KV positions even when current_seq isn't a multiple of BC=64.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_bf16_v3_d256_gqa_causal(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u16>,
    k: &CudaSlice<u16>,
    v: &CudaSlice<u16>,
    o: &mut CudaSlice<u16>,
    batch: u32,
    num_heads: u32,
    num_heads_kv: u32,
    seq_q: u32,
    seq_kv: u32,
    scale: f32,
) -> Result<()> {
    flash_attn_bf16_v3_d256_gqa_inner(
        ctx,
        stream,
        q,
        k,
        v,
        o,
        batch,
        num_heads,
        num_heads_kv,
        seq_q,
        seq_kv,
        scale,
        true,
        0,
    )
}

/// Decode variant: GQA + causal with explicit q_pos_offset.
#[allow(clippy::too_many_arguments)]
/// pos_dev variant of `flash_attn_bf16_v3_d256_gqa_causal_with_offset`:
/// `q_pos_offset` is read from a device-resident u32 at launch time, allowing
/// the kernel to be CUDA-Graph-captured and replayed across decode steps where
/// `position` changes.
/// BF16 GQA flash-attention DECODE kernel for d=128 (Sq=1).
///
/// Q: [batch, num_q_heads, 128] BF16 — single token per batch position
/// K, V: [batch, num_kv_heads, kv_stride, 128] BF16 — KV cache, allocated at
///       kv_stride rows per kv_head; the kernel only reads positions
///       0..seq_kv-1 and uses kv_stride for the per-head address stride.
/// O: [batch, num_q_heads, 128] BF16
/// scale: typically 1/sqrt(128).
///
/// `seq_kv` is the actual valid KV length (e.g. position+1 during decode).
/// `kv_stride` is the max sequence length the cache was allocated for.
///
/// Decode-only (Sq=1). For prefill, use a Br≥128 kernel.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_bf16_decode_d128_gqa(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u16>,
    k: &CudaSlice<u16>,
    v: &CudaSlice<u16>,
    o: &mut CudaSlice<u16>,
    batch: u32,
    num_heads: u32,
    num_kv_heads: u32,
    seq_kv: u32,
    kv_stride: u32,
    scale: f32,
) -> Result<()> {
    const D: u32 = 128;
    if batch == 0 || num_heads == 0 || num_kv_heads == 0 || seq_kv == 0 || kv_stride == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    if !num_heads.is_multiple_of(num_kv_heads) {
        return Err(SparkError::InvalidArgument(format!(
            "num_kv_heads ({num_kv_heads}) must divide num_heads ({num_heads})"
        )));
    }
    if seq_kv > kv_stride {
        return Err(SparkError::InvalidArgument(format!(
            "seq_kv ({seq_kv}) > kv_stride ({kv_stride})"
        )));
    }
    let q_need = batch as usize * num_heads as usize * D as usize;
    let kv_need = batch as usize * num_kv_heads as usize * kv_stride as usize * D as usize;
    if q.len() < q_need || k.len() < kv_need || v.len() < kv_need || o.len() < q_need {
        return Err(SparkError::InvalidArgument(format!(
            "buffer sizes: q={} k={} v={} o={}, need q/o={q_need} k/v={kv_need}",
            q.len(),
            k.len(),
            v.len(),
            o.len(),
        )));
    }
    let func = module::load_kernel(ctx, "fa_bf16_decode_d128_gqa", "fa_bf16_decode_d128_gqa")?;
    let cfg = LaunchConfig {
        grid_dim: (num_heads, batch, 1),
        block_dim: (D, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(q)
            .arg(k)
            .arg(v)
            .arg(o)
            .arg(&num_heads)
            .arg(&num_kv_heads)
            .arg(&seq_kv)
            .arg(&kv_stride)
            .arg(&scale)
            .launch(cfg)?;
    }
    Ok(())
}

/// BF16 GQA flash-attention DECODE for asymmetric K/V dims: d_qk=192, d_v=128.
/// Used by DeepSeek V3-style attention where K is `concat(k_nope, k_rope)` (128+64=192)
/// and V keeps its own 128-dim head.
///
/// Q: [batch, num_heads, 192] BF16
/// K: [batch, num_kv_heads, kv_stride, 192] BF16
/// V: [batch, num_kv_heads, kv_stride, 128] BF16
/// O: [batch, num_heads, 128] BF16
///
/// `scale` is typically `1 / sqrt(d_qk)`. `kv_stride` is the cache's allocation
/// length per kv_head (the kernel iterates k in `0..seq_kv` only).
///
/// Decode-only (Sq=1). 192 threads per block (6 warps); threads 0..127 hold
/// the V accumulator and store the output. Threads 128..191 only contribute
/// to the QK reduction.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_bf16_decode_d192_dv128_gqa(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u16>,
    k: &CudaSlice<u16>,
    v: &CudaSlice<u16>,
    o: &mut CudaSlice<u16>,
    batch: u32,
    num_heads: u32,
    num_kv_heads: u32,
    seq_kv: u32,
    kv_stride: u32,
    scale: f32,
) -> Result<()> {
    const D_QK: u32 = 192;
    const D_V: u32 = 128;
    if batch == 0 || num_heads == 0 || num_kv_heads == 0 || seq_kv == 0 || kv_stride == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    if !num_heads.is_multiple_of(num_kv_heads) {
        return Err(SparkError::InvalidArgument(format!(
            "num_kv_heads ({num_kv_heads}) must divide num_heads ({num_heads})"
        )));
    }
    if seq_kv > kv_stride {
        return Err(SparkError::InvalidArgument(format!(
            "seq_kv ({seq_kv}) > kv_stride ({kv_stride})"
        )));
    }
    let q_need = batch as usize * num_heads as usize * D_QK as usize;
    let o_need = batch as usize * num_heads as usize * D_V as usize;
    let k_need = batch as usize * num_kv_heads as usize * kv_stride as usize * D_QK as usize;
    let v_need = batch as usize * num_kv_heads as usize * kv_stride as usize * D_V as usize;
    if q.len() < q_need || k.len() < k_need || v.len() < v_need || o.len() < o_need {
        return Err(SparkError::InvalidArgument(format!(
            "buffer sizes: q={} k={} v={} o={}, need q={q_need} k={k_need} v={v_need} o={o_need}",
            q.len(),
            k.len(),
            v.len(),
            o.len(),
        )));
    }
    let func = module::load_kernel(
        ctx,
        "fa_bf16_decode_d192_dv128_gqa",
        "fa_bf16_decode_d192_dv128_gqa",
    )?;
    let cfg = LaunchConfig {
        grid_dim: (num_heads, batch, 1),
        block_dim: (D_QK, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(q)
            .arg(k)
            .arg(v)
            .arg(o)
            .arg(&num_heads)
            .arg(&num_kv_heads)
            .arg(&seq_kv)
            .arg(&kv_stride)
            .arg(&scale)
            .launch(cfg)?;
    }
    Ok(())
}

/// Causal BF16 flash attention (head_dim=256, GQA) with the query position read
/// from a device pointer. Each query row uses position `*pos_ptr` for the causal
/// mask, letting the position vary at launch time without a host sync.
pub fn flash_attn_bf16_v3_d256_gqa_causal_pos_dev(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u16>,
    k: &CudaSlice<u16>,
    v: &CudaSlice<u16>,
    o: &mut CudaSlice<u16>,
    pos_ptr: &CudaSlice<u32>,
    batch: u32,
    num_heads: u32,
    num_heads_kv: u32,
    seq_q: u32,
    seq_kv: u32,
    scale: f32,
) -> Result<()> {
    const D: u32 = 256;
    validate_attn_dims(batch, num_heads, seq_q, seq_kv)?;
    if num_heads_kv == 0 || !num_heads.is_multiple_of(num_heads_kv) {
        return Err(SparkError::InvalidArgument(format!(
            "num_heads_kv ({num_heads_kv}) must divide num_heads ({num_heads})"
        )));
    }
    let q_need = batch as usize * num_heads as usize * seq_q as usize * D as usize;
    let kv_need = batch as usize * num_heads_kv as usize * seq_kv as usize * D as usize;
    if q.len() < q_need || k.len() < kv_need || v.len() < kv_need || o.len() < q_need {
        return Err(SparkError::InvalidArgument("attn buffer too small".into()));
    }
    if (pos_ptr.len() as u32) < batch {
        // The kernel indexes pos_ptr[blockIdx.z] for blockIdx.z in 0..batch, so
        // a shorter buffer is an out-of-bounds read, not just "empty".
        return Err(SparkError::InvalidArgument(format!(
            "pos_ptr too small: {} < batch {batch}",
            pos_ptr.len()
        )));
    }
    let func = module::load_kernel(
        ctx,
        "fa_bf16_v3_d256_gqa_causal_pos_dev",
        "flash_attn_bf16_v3_d256_gqa_causal_pos_dev",
    )?;
    let grid_x = seq_q.div_ceil(128);
    let cfg = LaunchConfig {
        grid_dim: (grid_x, num_heads, batch),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(q)
            .arg(k)
            .arg(v)
            .arg(o)
            .arg(&seq_q)
            .arg(&seq_kv)
            .arg(&num_heads)
            .arg(&num_heads_kv)
            .arg(pos_ptr)
            .arg(&scale)
            .launch(cfg)?;
    }
    Ok(())
}

/// FP8-KV causal flash attention (head_dim=256, GQA, pos_dev) — BF16 Q x e4m3
/// K/V. The KV cache is stored as FP8 e4m3 (1 byte/elem); the kernel dequantizes
/// K/V in the gmem->smem load and folds the per-tensor `kv_scale` into the QK
/// logit scale (K side) and the output normalization (V side). Numerically
/// equivalent to the BF16 d256 kernel run on dequantized (fp8 * kv_scale) K/V.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_bf16_fp8kv_v3_d256_gqa_causal_pos_dev(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u16>,
    k: &CudaSlice<u8>,
    v: &CudaSlice<u8>,
    o: &mut CudaSlice<u16>,
    pos_ptr: &CudaSlice<u32>,
    batch: u32,
    num_heads: u32,
    num_heads_kv: u32,
    seq_q: u32,
    seq_kv: u32,
    scale: f32,
    kv_scale: f32,
) -> Result<()> {
    const D: u32 = 256;
    validate_attn_dims(batch, num_heads, seq_q, seq_kv)?;
    if num_heads_kv == 0 || !num_heads.is_multiple_of(num_heads_kv) {
        return Err(SparkError::InvalidArgument(format!(
            "num_heads_kv ({num_heads_kv}) must divide num_heads ({num_heads})"
        )));
    }
    let q_need = batch as usize * num_heads as usize * seq_q as usize * D as usize;
    let kv_need = batch as usize * num_heads_kv as usize * seq_kv as usize * D as usize;
    if q.len() < q_need || k.len() < kv_need || v.len() < kv_need || o.len() < q_need {
        return Err(SparkError::InvalidArgument(
            "attn fp8kv buffer too small".into(),
        ));
    }
    if (pos_ptr.len() as u32) < batch {
        // The kernel indexes pos_ptr[blockIdx.z] for blockIdx.z in 0..batch, so
        // a shorter buffer is an out-of-bounds read, not just "empty".
        return Err(SparkError::InvalidArgument(format!(
            "pos_ptr too small: {} < batch {batch}",
            pos_ptr.len()
        )));
    }
    let func = module::load_kernel(
        ctx,
        "fa_bf16_fp8kv_v3_d256_gqa_causal_pos_dev",
        "flash_attn_bf16_fp8kv_v3_d256_gqa_causal_pos_dev",
    )?;
    let grid_x = seq_q.div_ceil(128);
    let cfg = LaunchConfig {
        grid_dim: (grid_x, num_heads, batch),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(q)
            .arg(k)
            .arg(v)
            .arg(o)
            .arg(&seq_q)
            .arg(&seq_kv)
            .arg(&num_heads)
            .arg(&num_heads_kv)
            .arg(pos_ptr)
            .arg(&scale)
            .arg(&kv_scale)
            .launch(cfg)?;
    }
    Ok(())
}

/// Causal BF16 flash attention (head_dim=256, GQA) where query row `i` is masked
/// against KV positions up to `q_pos_offset + i`. The offset places a chunk of
/// queries at an arbitrary start position, as needed for chunked prefill.
pub fn flash_attn_bf16_v3_d256_gqa_causal_with_offset(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u16>,
    k: &CudaSlice<u16>,
    v: &CudaSlice<u16>,
    o: &mut CudaSlice<u16>,
    batch: u32,
    num_heads: u32,
    num_heads_kv: u32,
    seq_q: u32,
    seq_kv: u32,
    scale: f32,
    q_pos_offset: u32,
) -> Result<()> {
    flash_attn_bf16_v3_d256_gqa_inner(
        ctx,
        stream,
        q,
        k,
        v,
        o,
        batch,
        num_heads,
        num_heads_kv,
        seq_q,
        seq_kv,
        scale,
        true,
        q_pos_offset,
    )
}

#[allow(clippy::too_many_arguments)]
fn flash_attn_bf16_v3_d256_gqa_inner(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u16>,
    k: &CudaSlice<u16>,
    v: &CudaSlice<u16>,
    o: &mut CudaSlice<u16>,
    batch: u32,
    num_heads: u32,
    num_heads_kv: u32,
    seq_q: u32,
    seq_kv: u32,
    scale: f32,
    causal: bool,
    q_pos_offset: u32,
) -> Result<()> {
    const D: u32 = 256;
    validate_attn_dims(batch, num_heads, seq_q, seq_kv)?;
    if num_heads_kv == 0 || !num_heads.is_multiple_of(num_heads_kv) {
        return Err(SparkError::InvalidArgument(format!(
            "num_heads_kv ({num_heads_kv}) must divide num_heads ({num_heads})"
        )));
    }
    let q_need = batch as usize * num_heads as usize * seq_q as usize * D as usize;
    let kv_need = batch as usize * num_heads_kv as usize * seq_kv as usize * D as usize;
    if q.len() < q_need || k.len() < kv_need || v.len() < kv_need || o.len() < q_need {
        return Err(SparkError::InvalidArgument(format!(
            "buffer too small for d=256 GQA (q={} k={} v={} o={} need q={} kv={})",
            q.len(),
            k.len(),
            v.len(),
            o.len(),
            q_need,
            kv_need
        )));
    }
    let (cubin_name, entry) = if causal {
        (
            "fa_bf16_v3_d256_gqa_causal",
            "flash_attn_bf16_v3_d256_gqa_causal",
        )
    } else {
        ("fa_bf16_v3_d256_gqa", "flash_attn_bf16_v3_d256_gqa")
    };
    let func = module::load_kernel(ctx, cubin_name, entry)?;
    let grid_x = seq_q.div_ceil(128);
    let cfg = LaunchConfig {
        grid_dim: (grid_x, num_heads, batch),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    if causal {
        unsafe {
            stream
                .launch_builder(&func)
                .arg(q)
                .arg(k)
                .arg(v)
                .arg(o)
                .arg(&seq_q)
                .arg(&seq_kv)
                .arg(&num_heads)
                .arg(&num_heads_kv)
                .arg(&q_pos_offset)
                .arg(&scale)
                .launch(cfg)?;
        }
    } else {
        unsafe {
            stream
                .launch_builder(&func)
                .arg(q)
                .arg(k)
                .arg(v)
                .arg(o)
                .arg(&seq_q)
                .arg(&seq_kv)
                .arg(&num_heads)
                .arg(&num_heads_kv)
                .arg(&scale)
                .launch(cfg)?;
        }
    }
    Ok(())
}

/// Causal variant of `flash_attn_bf16_v3_d256`. Same kernel + per-element causal mask
/// (S[i,j] = -inf where kv_pos_j > q_pos_i) and a KV-block iteration cap so blocks
/// fully outside the causal window are skipped.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_bf16_v3_d256_causal(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u16>,
    k: &CudaSlice<u16>,
    v: &CudaSlice<u16>,
    o: &mut CudaSlice<u16>,
    batch: u32,
    num_heads: u32,
    seq_q: u32,
    seq_kv: u32,
    scale: f32,
) -> Result<()> {
    const D: u32 = 256;
    validate_attn_dims(batch, num_heads, seq_q, seq_kv)?;
    let q_need = batch as usize * num_heads as usize * seq_q as usize * D as usize;
    let kv_need = batch as usize * num_heads as usize * seq_kv as usize * D as usize;
    if q.len() < q_need || k.len() < kv_need || v.len() < kv_need || o.len() < q_need {
        return Err(SparkError::InvalidArgument(format!(
            "buffer too small for d=256 causal (q={} k={} v={} o={} need q={} kv={})",
            q.len(),
            k.len(),
            v.len(),
            o.len(),
            q_need,
            kv_need
        )));
    }
    let func = module::load_kernel(
        ctx,
        "fa_bf16_v3_d256_causal",
        "flash_attn_bf16_v3_d256_causal",
    )?;
    let grid_x = seq_q.div_ceil(128);
    let cfg = LaunchConfig {
        grid_dim: (grid_x, num_heads, batch),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let q_pos_offset: u32 = 0; // prefill default; decode users use the GQA variant
    unsafe {
        stream
            .launch_builder(&func)
            .arg(q)
            .arg(k)
            .arg(v)
            .arg(o)
            .arg(&seq_q)
            .arg(&seq_kv)
            .arg(&num_heads)
            .arg(&q_pos_offset)
            .arg(&scale)
            .launch(cfg)?;
    }
    Ok(())
}

/// 128 threads, BR=128, BC=128. Each warp handles 32 Q rows (2 MMA row groups).
/// 512 regs/thread available. Matches CUTLASS CuTe DSL architecture.
#[allow(clippy::too_many_arguments)]
#[cfg(feature = "experimental")]
pub fn flash_attn_bf16_v18_4warp(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u16>,
    k: &CudaSlice<u16>,
    v: &CudaSlice<u16>,
    o: &mut CudaSlice<u16>,
    batch: u32,
    num_heads: u32,
    seq_q: u32,
    seq_kv: u32,
    scale: f32,
) -> Result<()> {
    let func = module::load_kernel(ctx, "fa_bf16_v18_4warp", "flash_attn_bf16_v18_4warp")?;
    let grid_x = seq_q.div_ceil(128);
    let cfg = LaunchConfig {
        grid_dim: (grid_x, num_heads, batch),
        block_dim: (128, 1, 1), // 4 warps!
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(q)
            .arg(k)
            .arg(v)
            .arg(o)
            .arg(&seq_q)
            .arg(&seq_kv)
            .arg(&num_heads)
            .arg(&scale)
            .launch(cfg)?;
    }
    Ok(())
}

/// Launch BF16 Flash Attention V17 (BC=128) with d=128.
/// Same as V3 but with BC=128 (double K/V tile width), halving KV loop iterations.
#[allow(clippy::too_many_arguments)]
#[cfg(feature = "experimental")]
pub fn flash_attn_bf16_v17_bc128(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u16>,
    k: &CudaSlice<u16>,
    v: &CudaSlice<u16>,
    o: &mut CudaSlice<u16>,
    batch: u32,
    num_heads: u32,
    seq_q: u32,
    seq_kv: u32,
    scale: f32,
) -> Result<()> {
    let func = module::load_kernel(ctx, "fa_bf16_v17_bc128", "flash_attn_bf16_v17_bc128")?;
    let grid_x = seq_q.div_ceil(128);
    let cfg = LaunchConfig {
        grid_dim: (grid_x, num_heads, batch),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(q)
            .arg(k)
            .arg(v)
            .arg(o)
            .arg(&seq_q)
            .arg(&seq_kv)
            .arg(&num_heads)
            .arg(&scale)
            .launch(cfg)?;
    }
    Ok(())
}

/// Launch BF16 Flash Attention V3 (causal) with d=128.
///
/// Same interface as V3 non-causal but applies causal masking:
/// S[i,j] = -inf when kv_pos_j > q_pos_i.
///
/// V3 uses 8 warps (256 threads) per block, Br=128, Bc=64.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_bf16_v3_d128_causal(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u16>,
    k: &CudaSlice<u16>,
    v: &CudaSlice<u16>,
    o: &mut CudaSlice<u16>,
    batch: u32,
    num_heads: u32,
    seq_q: u32,
    seq_kv: u32,
    scale: f32,
) -> Result<()> {
    validate_attn_dims(batch, num_heads, seq_q, seq_kv)?;
    validate_attn_bf16_bufs(q, k, v, o, batch, num_heads, seq_q, seq_kv)?;
    let func = module::load_kernel(
        ctx,
        "fa_bf16_v3_d128_causal",
        "flash_attn_bf16_v3_d128_causal",
    )?;

    let grid_x = seq_q.div_ceil(128); // Q blocks (Br=128)
    let cfg = LaunchConfig {
        grid_dim: (grid_x, num_heads, batch),
        block_dim: (256, 1, 1), // 8 warps
        shared_mem_bytes: 0,    // smem is statically declared in PTX
    };

    unsafe {
        stream
            .launch_builder(&func)
            .arg(q)
            .arg(k)
            .arg(v)
            .arg(o)
            .arg(&seq_q)
            .arg(&seq_kv)
            .arg(&num_heads)
            .arg(&scale)
            .launch(cfg)?;
    }

    Ok(())
}

/// Launch FP8 Flash Attention (non-causal) with d=128.
///
/// Q, K, V: [batch, num_heads, seq, 128] row-major FP8 e4m3 (u8 raw bits).
/// O: [batch, num_heads, seq_q, 128] row-major bf16 (u16 raw bits, output).
///
/// 1 warp (32 threads) per block, Br=16, Bc=64.
#[allow(clippy::too_many_arguments)]
#[cfg(feature = "experimental")]
pub fn flash_attn_fp8_d128(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u8>,
    k: &CudaSlice<u8>,
    v: &CudaSlice<u8>,
    o: &mut CudaSlice<u16>,
    batch: u32,
    num_heads: u32,
    seq_q: u32,
    seq_kv: u32,
    scale: f32,
) -> Result<()> {
    validate_attn_dims(batch, num_heads, seq_q, seq_kv)?;
    let q_need = batch as usize * num_heads as usize * seq_q as usize * HEAD_DIM as usize;
    let kv_need = batch as usize * num_heads as usize * seq_kv as usize * HEAD_DIM as usize;
    if q.len() < q_need {
        return Err(SparkError::InvalidArgument(format!(
            "Q buffer too small: {} < {q_need}",
            q.len()
        )));
    }
    if k.len() < kv_need {
        return Err(SparkError::InvalidArgument(format!(
            "K buffer too small: {} < {kv_need}",
            k.len()
        )));
    }
    if v.len() < kv_need {
        return Err(SparkError::InvalidArgument(format!(
            "V buffer too small: {} < {kv_need}",
            v.len()
        )));
    }
    if o.len() < q_need {
        return Err(SparkError::InvalidArgument(format!(
            "O buffer too small: {} < {q_need}",
            o.len()
        )));
    }
    let func = module::load_kernel(ctx, "fa_fp8_d128", "flash_attn_fp8_d128")?;

    let grid_x = seq_q.div_ceil(16); // Q blocks (Br=16)
    let cfg = LaunchConfig {
        grid_dim: (grid_x, num_heads, batch),
        block_dim: (32, 1, 1), // 1 warp
        shared_mem_bytes: 0,   // smem is statically declared in PTX
    };

    unsafe {
        stream
            .launch_builder(&func)
            .arg(q)
            .arg(k)
            .arg(v)
            .arg(o)
            .arg(&seq_q)
            .arg(&seq_kv)
            .arg(&num_heads)
            .arg(&scale)
            .launch(cfg)?;
    }

    Ok(())
}

/// Launch FP8 Flash Attention (causal) with d=128.
///
/// Q, K, V: [batch, num_heads, seq, 128] row-major FP8 e4m3 (u8 raw bits).
/// O: [batch, num_heads, seq_q, 128] row-major bf16 (u16 raw bits, output).
///
/// Applies causal masking: S[i,j] = -inf when kv_pos_j > q_pos_i.
/// 1 warp (32 threads) per block, Br=16, Bc=64.
#[allow(clippy::too_many_arguments)]
#[cfg(feature = "experimental")]
pub fn flash_attn_fp8_d128_causal(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u8>,
    k: &CudaSlice<u8>,
    v: &CudaSlice<u8>,
    o: &mut CudaSlice<u16>,
    batch: u32,
    num_heads: u32,
    seq_q: u32,
    seq_kv: u32,
    scale: f32,
) -> Result<()> {
    validate_attn_dims(batch, num_heads, seq_q, seq_kv)?;
    let q_need = batch as usize * num_heads as usize * seq_q as usize * HEAD_DIM as usize;
    let kv_need = batch as usize * num_heads as usize * seq_kv as usize * HEAD_DIM as usize;
    if q.len() < q_need {
        return Err(SparkError::InvalidArgument(format!(
            "Q buffer too small: {} < {q_need}",
            q.len()
        )));
    }
    if k.len() < kv_need {
        return Err(SparkError::InvalidArgument(format!(
            "K buffer too small: {} < {kv_need}",
            k.len()
        )));
    }
    if v.len() < kv_need {
        return Err(SparkError::InvalidArgument(format!(
            "V buffer too small: {} < {kv_need}",
            v.len()
        )));
    }
    if o.len() < q_need {
        return Err(SparkError::InvalidArgument(format!(
            "O buffer too small: {} < {q_need}",
            o.len()
        )));
    }
    let func = module::load_kernel(ctx, "fa_fp8_d128_causal", "flash_attn_fp8_d128_causal")?;

    let grid_x = seq_q.div_ceil(16); // Q blocks (Br=16)
    let cfg = LaunchConfig {
        grid_dim: (grid_x, num_heads, batch),
        block_dim: (32, 1, 1), // 1 warp
        shared_mem_bytes: 0,   // smem is statically declared in PTX
    };

    unsafe {
        stream
            .launch_builder(&func)
            .arg(q)
            .arg(k)
            .arg(v)
            .arg(o)
            .arg(&seq_q)
            .arg(&seq_kv)
            .arg(&num_heads)
            .arg(&scale)
            .launch(cfg)?;
    }

    Ok(())
}

/// Launch BF16 Flash Attention (varlen, non-causal) with d=128.
///
/// Q, K, V: [total_tokens, num_heads, 128] row-major BF16.
/// O: [total_q, num_heads, 128] row-major BF16 (output).
/// cu_seqlens_q: [batch+1] u32, cumulative Q sequence lengths.
/// cu_seqlens_k: [batch+1] u32, cumulative K/V sequence lengths.
///
/// 1 warp (32 threads) per block, Br=16, Bc=64.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_bf16_varlen_d128(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u16>,
    k: &CudaSlice<u16>,
    v: &CudaSlice<u16>,
    o: &mut CudaSlice<u16>,
    cu_seqlens_q: &CudaSlice<u32>,
    cu_seqlens_k: &CudaSlice<u32>,
    batch: u32,
    num_heads: u32,
    max_seqlen_q: u32,
    scale: f32,
) -> Result<()> {
    if batch == 0 {
        return Err(SparkError::InvalidArgument("batch must be > 0".into()));
    }
    if num_heads == 0 {
        return Err(SparkError::InvalidArgument("num_heads must be > 0".into()));
    }
    if max_seqlen_q == 0 {
        return Err(SparkError::InvalidArgument(
            "max_seqlen_q must be > 0".into(),
        ));
    }
    let cu_need = (batch + 1) as usize;
    if cu_seqlens_q.len() < cu_need {
        return Err(SparkError::InvalidArgument(format!(
            "cu_seqlens_q too small: {} < {cu_need}",
            cu_seqlens_q.len()
        )));
    }
    if cu_seqlens_k.len() < cu_need {
        return Err(SparkError::InvalidArgument(format!(
            "cu_seqlens_k too small: {} < {cu_need}",
            cu_seqlens_k.len()
        )));
    }
    // NOTE: cu_seqlens are GPU buffers (CudaSlice), so we can't validate contents
    // (monotonicity, starts at 0) without a D2H copy. Caller is responsible for
    // ensuring cu_seqlens[0] == 0, cu_seqlens is monotonically non-decreasing,
    // and cu_seqlens[batch] == total_tokens. Invalid cu_seqlens causes silent
    // wrong results (incorrect masking/offset calculations in the kernel).
    let func = module::load_kernel(ctx, "fa_bf16_varlen_d128", "flash_attn_bf16_varlen_d128")?;

    let grid_x = max_seqlen_q.div_ceil(16); // Q blocks (Br=16)
    let cfg = LaunchConfig {
        grid_dim: (grid_x, num_heads, batch),
        block_dim: (32, 1, 1), // 1 warp
        shared_mem_bytes: 18432,
    };

    unsafe {
        stream
            .launch_builder(&func)
            .arg(q)
            .arg(k)
            .arg(v)
            .arg(o)
            .arg(cu_seqlens_q)
            .arg(cu_seqlens_k)
            .arg(&num_heads)
            .arg(&max_seqlen_q)
            .arg(&scale)
            .launch(cfg)?;
    }

    Ok(())
}

/// Requires 90KB dynamic shared memory (K0+K1+V0+V1 buffers).
///
/// 5 warps (160 threads) per block, Br=64, Bc=64.
#[allow(clippy::too_many_arguments)]
#[cfg(feature = "experimental")]
pub fn flash_attn_bf16_v8_tma_db_d128(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u16>,
    k: &CudaSlice<u16>,
    v: &CudaSlice<u16>,
    o: &mut CudaSlice<u16>,
    batch: u32,
    num_heads: u32,
    seq_q: u32,
    seq_kv: u32,
    scale: f32,
) -> Result<()> {
    validate_attn_dims(batch, num_heads, seq_q, seq_kv)?;
    validate_attn_bf16_bufs(q, k, v, o, batch, num_heads, seq_q, seq_kv)?;

    use cudarc::driver::sys::*;

    let grid_x = seq_q.div_ceil(64);

    let (q_dptr, _q_sync) = q.device_ptr(stream);
    let (k_dptr, _k_sync) = k.device_ptr(stream);
    let (v_dptr, _v_sync) = v.device_ptr(stream);
    let (o_dptr, _o_sync) = o.device_ptr(stream);

    let total_q_rows = batch * num_heads * seq_q;
    let total_kv_rows = batch * num_heads * seq_kv;
    let q_tma = create_tma_desc(q_dptr as *mut _, total_q_rows, HEAD_DIM, 64, HEAD_DIM)?;
    let k_tma = create_tma_desc(k_dptr as *mut _, total_kv_rows, HEAD_DIM, 64, HEAD_DIM)?;
    let v_tma = create_tma_desc(v_dptr as *mut _, total_kv_rows, HEAD_DIM, 64, HEAD_DIM)?;

    let q_tma_u32: [u32; 32] = unsafe { core::mem::transmute(q_tma) };
    let k_tma_u32: [u32; 32] = unsafe { core::mem::transmute(k_tma) };
    let v_tma_u32: [u32; 32] = unsafe { core::mem::transmute(v_tma) };
    let q_tma_dev = stream.memcpy_stod(&q_tma_u32).map_err(SparkError::Driver)?;
    let k_tma_dev = stream.memcpy_stod(&k_tma_u32).map_err(SparkError::Driver)?;
    let v_tma_dev = stream.memcpy_stod(&v_tma_u32).map_err(SparkError::Driver)?;

    let (q_tma_dptr, _) = q_tma_dev.device_ptr(stream);
    let (k_tma_dptr, _) = k_tma_dev.device_ptr(stream);
    let (v_tma_dptr, _) = v_tma_dev.device_ptr(stream);

    let cu_func = module::load_kernel_raw(
        ctx,
        "fa_bf16_v8_tma_db_d128",
        "flash_attn_bf16_v8_tma_db_d128",
    )?;
    let cu_stream = stream.cu_stream();

    // V8 uses 90KB dynamic SMEM — must set attribute before launch
    const SMEM_TOTAL: i32 = 82032;
    let attr_result = unsafe {
        cuFuncSetAttribute(
            cu_func,
            CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            SMEM_TOTAL,
        )
    };
    if attr_result != CUresult::CUDA_SUCCESS {
        return Err(SparkError::LaunchFailed(format!(
            "cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES={}) failed: {:?}",
            SMEM_TOTAL, attr_result
        )));
    }

    let params: [*mut core::ffi::c_void; 8] = [
        &q_tma_dptr as *const u64 as *mut _,
        &k_tma_dptr as *const u64 as *mut _,
        &v_tma_dptr as *const u64 as *mut _,
        &o_dptr as *const u64 as *mut _,
        &seq_q as *const u32 as *mut _,
        &seq_kv as *const u32 as *mut _,
        &num_heads as *const u32 as *mut _,
        &scale as *const f32 as *mut _,
    ];

    let result = unsafe {
        cuLaunchKernel(
            cu_func,
            grid_x,
            num_heads,
            batch, // grid
            160,
            1,
            1,                 // block (5 warps: 4 MMA + 1 DMA)
            SMEM_TOTAL as u32, // dynamic SMEM
            cu_stream,
            params.as_ptr() as *mut _,
            core::ptr::null_mut(),
        )
    };

    if result != CUresult::CUDA_SUCCESS {
        return Err(SparkError::LaunchFailed(format!(
            "cuLaunchKernel failed for V8 TMA DB: {:?}",
            result
        )));
    }

    Ok(())
}

/// Create a 2D TMA tensor map descriptor for a BF16 attention tensor.
///
/// The tensor is laid out as [batch, num_heads, seq_len, head_dim] in memory.
/// TMA sees it as a 2D tile: rows=seq_len, cols=head_dim, per head per batch.
/// The batch/head dimensions are handled by the grid coordinates passed to the
/// TMA instruction at launch time.
///
/// Returns a 128-byte aligned tensor map descriptor that can be passed as a
/// kernel parameter.
/// Create a TMA tensor map descriptor for a BF16 2D tile.
fn create_tma_desc(
    global_ptr: *mut core::ffi::c_void,
    seq_len: u32,
    head_dim: u32,
    tile_rows: u32,
    tile_cols: u32,
) -> Result<cudarc::driver::sys::CUtensorMap> {
    use cudarc::driver::sys::*;

    let mut tma = CUtensorMap_st::default();

    // TMA 2D: dim[0] = innermost = head_dim, dim[1] = seq_len
    let global_dim: [u64; 2] = [head_dim as u64, seq_len as u64];
    // stride[0] is implicit (= element size). Only stride[1..] needed.
    let global_strides: [u64; 1] = [(head_dim as u64) * 2]; // bytes per row
    let box_dim: [u32; 2] = [tile_cols, tile_rows];
    let elem_strides: [u32; 2] = [1, 1];

    //    eprintln!(
    //    //         "TMA desc: ptr={:?} globalDim={:?} globalStrides={:?} boxDim={:?} elemStrides={:?}",
    //        global_ptr, global_dim, global_strides, box_dim, elem_strides
    //    );
    //
    let result = unsafe {
        cuTensorMapEncodeTiled(
            &mut tma,
            CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
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
            "cuTensorMapEncodeTiled failed: {:?}",
            result
        )));
    }

    Ok(tma)
}

/// Create a 3D TMA tensor map descriptor for BF16 with NO swizzle.
/// Use this when the tensor has three logical dimensions and you need to
/// tile along the outermost (e.g., K is [B, Skv, H, D], view as
/// [D, H, B*Skv] so consecutive "outer" units advance through positions
/// within the same head — a 2D descriptor can't express this because the
/// physical stride between consecutive positions is H*D not D).
///
/// Orphaned helper — the NSA/MLA-prefill TMA retrofits that used this were
/// removed for perf regressions, but the helper is kept for future kernels
/// whose per-CTA volume actually amortizes TMA overhead.
#[allow(dead_code)]
fn create_tma_desc_3d_bf16(
    global_ptr: *mut core::ffi::c_void,
    inner_dim: u32,  // innermost = D
    middle_dim: u32, // middle   = H
    outer_dim: u32,  // outermost = B*Skv
    tile_inner: u32,
    tile_middle: u32,
    tile_outer: u32,
) -> Result<cudarc::driver::sys::CUtensorMap> {
    use cudarc::driver::sys::*;
    let mut tma = CUtensorMap_st::default();
    let global_dim: [u64; 3] = [inner_dim as u64, middle_dim as u64, outer_dim as u64];
    let global_strides: [u64; 2] = [
        (inner_dim as u64) * 2,                       // stride[1] = D * 2 bytes
        (middle_dim as u64) * (inner_dim as u64) * 2, // stride[2] = H * D * 2 bytes
    ];
    let box_dim: [u32; 3] = [tile_inner, tile_middle, tile_outer];
    let elem_strides: [u32; 3] = [1, 1, 1];

    let result = unsafe {
        cuTensorMapEncodeTiled(
            &mut tma,
            CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            3,
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
            "cuTensorMapEncodeTiled 3D: {:?}",
            result
        )));
    }
    Ok(tma)
}

/// Create a TMA tensor map descriptor for a BF16 2D tile with SWIZZLE_128B.
///
/// boxDim[0] must be 64 (64 cols x 2B = 128 bytes = one swizzle segment).
/// The kernel issues two TMA loads per tile: coord_x=0 (left half) and
/// coord_x=64 (right half) to cover the full 128-column head dimension.
fn create_tma_desc_swizzle(
    global_ptr: *mut core::ffi::c_void,
    seq_len: u32,
    head_dim: u32,
    tile_rows: u32,
    tile_cols: u32, // must be 64 for SWIZZLE_128B
) -> Result<cudarc::driver::sys::CUtensorMap> {
    use cudarc::driver::sys::*;

    let mut tma = CUtensorMap_st::default();

    // TMA 2D: dim[0] = innermost = head_dim, dim[1] = seq_len
    let global_dim: [u64; 2] = [head_dim as u64, seq_len as u64];
    // stride[0] is implicit (= element size). Only stride[1..] needed.
    let global_strides: [u64; 1] = [(head_dim as u64) * 2]; // bytes per row
    let box_dim: [u32; 2] = [tile_cols, tile_rows];
    let elem_strides: [u32; 2] = [1, 1];

    //    eprintln!(
    //    //         "TMA desc (swizzle): ptr={:?} globalDim={:?} globalStrides={:?} boxDim={:?} elemStrides={:?}",
    //        global_ptr, global_dim, global_strides, box_dim, elem_strides
    //    );
    //
    let result = unsafe {
        cuTensorMapEncodeTiled(
            &mut tma,
            CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            2,
            global_ptr,
            global_dim.as_ptr(),
            global_strides.as_ptr(),
            box_dim.as_ptr(),
            elem_strides.as_ptr(),
            CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
            CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
            CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        )
    };

    if result != CUresult::CUDA_SUCCESS {
        return Err(SparkError::LaunchFailed(format!(
            "cuTensorMapEncodeTiled (swizzle) failed: {:?}",
            result
        )));
    }

    Ok(tma)
}

/// V11 builds on V9 (TMA + double-buffered KV + SWIZZLE_128B) with one optimization:
/// the scale factor (1/sqrt(D)) is fused into the softmax FMA, eliminating 32 explicit
/// FMUL instructions per KV iteration. Instead of `S *= scale` followed by
/// `fma(S_scaled, LOG2E, -max*LOG2E)`, V11 precomputes `scale_log2e = scale * LOG2E`
/// and uses `fma(S_unscaled, scale_log2e, -max_scaled*LOG2E)` directly.
/// The running row max is stored as a scaled value throughout.
///
/// SASS: 138 FMUL (vs V9's 168) = 30 fewer per KV iteration. 165 registers, 2 blocks/SM.
///
/// 5 warps (160 threads) per block, Br=64, Bc=64.
/// 90KB dynamic shared memory.
#[allow(clippy::too_many_arguments)]
#[cfg(feature = "experimental")]
pub fn flash_attn_bf16_v11_fused_scale(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u16>,
    k: &CudaSlice<u16>,
    v: &CudaSlice<u16>,
    o: &mut CudaSlice<u16>,
    batch: u32,
    num_heads: u32,
    seq_q: u32,
    seq_kv: u32,
    scale: f32,
) -> Result<()> {
    validate_attn_dims(batch, num_heads, seq_q, seq_kv)?;
    validate_attn_bf16_bufs(q, k, v, o, batch, num_heads, seq_q, seq_kv)?;

    use cudarc::driver::sys::*;

    let grid_x = seq_q.div_ceil(64);

    let (q_dptr, _q_sync) = q.device_ptr(stream);
    let (k_dptr, _k_sync) = k.device_ptr(stream);
    let (v_dptr, _v_sync) = v.device_ptr(stream);
    let (o_dptr, _o_sync) = o.device_ptr(stream);

    let total_q_rows = batch * num_heads * seq_q;
    let total_kv_rows = batch * num_heads * seq_kv;

    let q_tma = create_tma_desc_swizzle(q_dptr as *mut _, total_q_rows, HEAD_DIM, 64, 64)?;
    let k_tma = create_tma_desc_swizzle(k_dptr as *mut _, total_kv_rows, HEAD_DIM, 64, 64)?;
    let v_tma = create_tma_desc_swizzle(v_dptr as *mut _, total_kv_rows, HEAD_DIM, 64, 64)?;

    let q_tma_u32: [u32; 32] = unsafe { core::mem::transmute(q_tma) };
    let k_tma_u32: [u32; 32] = unsafe { core::mem::transmute(k_tma) };
    let v_tma_u32: [u32; 32] = unsafe { core::mem::transmute(v_tma) };
    let q_tma_dev = stream.memcpy_stod(&q_tma_u32).map_err(SparkError::Driver)?;
    let k_tma_dev = stream.memcpy_stod(&k_tma_u32).map_err(SparkError::Driver)?;
    let v_tma_dev = stream.memcpy_stod(&v_tma_u32).map_err(SparkError::Driver)?;

    let (q_tma_dptr, _) = q_tma_dev.device_ptr(stream);
    let (k_tma_dptr, _) = k_tma_dev.device_ptr(stream);
    let (v_tma_dptr, _) = v_tma_dev.device_ptr(stream);

    let cu_func = module::load_kernel_raw(
        ctx,
        "fa_bf16_v11_fused_scale",
        "flash_attn_bf16_v11_fused_scale",
    )?;
    let cu_stream = stream.cu_stream();

    const SMEM_TOTAL: i32 = 82032;
    let attr_result = unsafe {
        cuFuncSetAttribute(
            cu_func,
            CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            SMEM_TOTAL,
        )
    };
    if attr_result != CUresult::CUDA_SUCCESS {
        return Err(SparkError::LaunchFailed(format!(
            "cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES={}) failed: {:?}",
            SMEM_TOTAL, attr_result
        )));
    }

    let params: [*mut core::ffi::c_void; 8] = [
        &q_tma_dptr as *const u64 as *mut _,
        &k_tma_dptr as *const u64 as *mut _,
        &v_tma_dptr as *const u64 as *mut _,
        &o_dptr as *const u64 as *mut _,
        &seq_q as *const u32 as *mut _,
        &seq_kv as *const u32 as *mut _,
        &num_heads as *const u32 as *mut _,
        &scale as *const f32 as *mut _,
    ];

    let result = unsafe {
        cuLaunchKernel(
            cu_func,
            grid_x,
            num_heads,
            batch,
            160,
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
            "cuLaunchKernel failed for V11 fused scale: {:?}",
            result
        )));
    }

    Ok(())
}

/// Launch BF16 Flash Attention V12 (persistent) with TMA + SWIZZLE_128B + fused scale.
///
/// Persistent variant of V11: grid.x = min(num_q_blocks, 48) so each CTA iterates over
/// multiple Q tiles with stride gridDim.x. Reduces wave fragmentation for long sequences
/// (seq_q > 3072). For shorter sequences grid.x = num_q_blocks (identical to V11).
///
/// 168 registers, 0 spills, 90KB dynamic shared memory.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_bf16_v12_persistent(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u16>,
    k: &CudaSlice<u16>,
    v: &CudaSlice<u16>,
    o: &mut CudaSlice<u16>,
    batch: u32,
    num_heads: u32,
    seq_q: u32,
    seq_kv: u32,
    scale: f32,
) -> Result<()> {
    validate_attn_dims(batch, num_heads, seq_q, seq_kv)?;
    validate_attn_bf16_bufs(q, k, v, o, batch, num_heads, seq_q, seq_kv)?;

    use cudarc::driver::sys::*;

    // Persistent: cap grid_x at 48 SMs (1 block/SM due to 90KB SMEM per block).
    // Each block processes ceil(num_q_blocks / grid_x) Q tiles.
    const NUM_SMS: u32 = 48;
    let num_q_blocks = seq_q.div_ceil(64);
    let grid_x = num_q_blocks.min(NUM_SMS);

    let (q_dptr, _q_sync) = q.device_ptr(stream);
    let (k_dptr, _k_sync) = k.device_ptr(stream);
    let (v_dptr, _v_sync) = v.device_ptr(stream);
    let (o_dptr, _o_sync) = o.device_ptr(stream);

    let total_q_rows = batch * num_heads * seq_q;
    let total_kv_rows = batch * num_heads * seq_kv;

    let q_tma = create_tma_desc_swizzle(q_dptr as *mut _, total_q_rows, HEAD_DIM, 64, 64)?;
    let k_tma = create_tma_desc_swizzle(k_dptr as *mut _, total_kv_rows, HEAD_DIM, 64, 64)?;
    let v_tma = create_tma_desc_swizzle(v_dptr as *mut _, total_kv_rows, HEAD_DIM, 64, 64)?;

    let q_tma_u32: [u32; 32] = unsafe { core::mem::transmute(q_tma) };
    let k_tma_u32: [u32; 32] = unsafe { core::mem::transmute(k_tma) };
    let v_tma_u32: [u32; 32] = unsafe { core::mem::transmute(v_tma) };
    let q_tma_dev = stream.memcpy_stod(&q_tma_u32).map_err(SparkError::Driver)?;
    let k_tma_dev = stream.memcpy_stod(&k_tma_u32).map_err(SparkError::Driver)?;
    let v_tma_dev = stream.memcpy_stod(&v_tma_u32).map_err(SparkError::Driver)?;

    let (q_tma_dptr, _) = q_tma_dev.device_ptr(stream);
    let (k_tma_dptr, _) = k_tma_dev.device_ptr(stream);
    let (v_tma_dptr, _) = v_tma_dev.device_ptr(stream);

    let cu_func = module::load_kernel_raw(
        ctx,
        "fa_bf16_v12_persistent",
        "flash_attn_bf16_v12_persistent",
    )?;
    let cu_stream = stream.cu_stream();

    const SMEM_TOTAL: i32 = 82032;
    let attr_result = unsafe {
        cuFuncSetAttribute(
            cu_func,
            CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            SMEM_TOTAL,
        )
    };
    if attr_result != CUresult::CUDA_SUCCESS {
        return Err(SparkError::LaunchFailed(format!(
            "cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES={}) failed: {:?}",
            SMEM_TOTAL, attr_result
        )));
    }

    let params: [*mut core::ffi::c_void; 8] = [
        &q_tma_dptr as *const u64 as *mut _,
        &k_tma_dptr as *const u64 as *mut _,
        &v_tma_dptr as *const u64 as *mut _,
        &o_dptr as *const u64 as *mut _,
        &seq_q as *const u32 as *mut _,
        &seq_kv as *const u32 as *mut _,
        &num_heads as *const u32 as *mut _,
        &scale as *const f32 as *mut _,
    ];

    let result = unsafe {
        cuLaunchKernel(
            cu_func,
            grid_x,
            num_heads,
            batch,
            160,
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
            "cuLaunchKernel failed for V12 persistent: {:?}",
            result
        )));
    }

    Ok(())
}

/// Flash Attention V13: BF16, non-causal, Bc=32 for 2-CTA/SM occupancy.
///
/// Uses Bc=32 (half of V11's Bc=64) to reduce static SMEM to 37 KB, allowing
/// 2 CTAs per SM simultaneously (vs 1 CTA/SM for V11 with 90 KB).
/// Single-buffer K/V (no double-buffering) — compute-to-memory ratio is ~33:1
/// so pipeline stalls are negligible.
///
/// 5 warps (160 threads) per block, Br=64, Bc=32.
/// 37 KB static shared memory (no cuFuncSetAttribute needed).
#[allow(clippy::too_many_arguments)]
#[cfg(feature = "experimental")]
pub fn flash_attn_bf16_v13_bc32_d128(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u16>,
    k: &CudaSlice<u16>,
    v: &CudaSlice<u16>,
    o: &mut CudaSlice<u16>,
    batch: u32,
    num_heads: u32,
    seq_q: u32,
    seq_kv: u32,
    scale: f32,
) -> Result<()> {
    validate_attn_dims(batch, num_heads, seq_q, seq_kv)?;
    validate_attn_bf16_bufs(q, k, v, o, batch, num_heads, seq_q, seq_kv)?;

    use cudarc::driver::sys::*;

    let grid_x = seq_q.div_ceil(64);

    let (q_dptr, _q_sync) = q.device_ptr(stream);
    let (k_dptr, _k_sync) = k.device_ptr(stream);
    let (v_dptr, _v_sync) = v.device_ptr(stream);
    let (o_dptr, _o_sync) = o.device_ptr(stream);

    let total_q_rows = batch * num_heads * seq_q;
    let total_kv_rows = batch * num_heads * seq_kv;

    // Q: tile_rows=64, K/V: tile_rows=32 (Bc=32), tile_cols=64 (half of D=128)
    let q_tma = create_tma_desc_swizzle(q_dptr as *mut _, total_q_rows, HEAD_DIM, 64, 64)?;
    let k_tma = create_tma_desc_swizzle(k_dptr as *mut _, total_kv_rows, HEAD_DIM, 32, 64)?;
    let v_tma = create_tma_desc_swizzle(v_dptr as *mut _, total_kv_rows, HEAD_DIM, 32, 64)?;

    let q_tma_u32: [u32; 32] = unsafe { core::mem::transmute(q_tma) };
    let k_tma_u32: [u32; 32] = unsafe { core::mem::transmute(k_tma) };
    let v_tma_u32: [u32; 32] = unsafe { core::mem::transmute(v_tma) };
    let q_tma_dev = stream.memcpy_stod(&q_tma_u32).map_err(SparkError::Driver)?;
    let k_tma_dev = stream.memcpy_stod(&k_tma_u32).map_err(SparkError::Driver)?;
    let v_tma_dev = stream.memcpy_stod(&v_tma_u32).map_err(SparkError::Driver)?;

    let (q_tma_dptr, _) = q_tma_dev.device_ptr(stream);
    let (k_tma_dptr, _) = k_tma_dev.device_ptr(stream);
    let (v_tma_dptr, _) = v_tma_dev.device_ptr(stream);

    let cu_func = module::load_kernel_raw(
        ctx,
        "fa_bf16_v13_bc32_d128",
        "flash_attn_bf16_v13_bc32_d128",
    )?;
    let cu_stream = stream.cu_stream();

    let params: [*mut core::ffi::c_void; 8] = [
        &q_tma_dptr as *const u64 as *mut _,
        &k_tma_dptr as *const u64 as *mut _,
        &v_tma_dptr as *const u64 as *mut _,
        &o_dptr as *const u64 as *mut _,
        &seq_q as *const u32 as *mut _,
        &seq_kv as *const u32 as *mut _,
        &num_heads as *const u32 as *mut _,
        &scale as *const f32 as *mut _,
    ];

    let result = unsafe {
        cuLaunchKernel(
            cu_func,
            grid_x,
            num_heads,
            batch,
            160,
            1,
            1,
            0, // static SMEM (37 KB) — no dynamic SMEM needed
            cu_stream,
            params.as_ptr() as *mut _,
            core::ptr::null_mut(),
        )
    };

    if result != CUresult::CUDA_SUCCESS {
        return Err(SparkError::LaunchFailed(format!(
            "cuLaunchKernel failed for V13 Bc32: {:?}",
            result
        )));
    }

    Ok(())
}

/// Launch BF16 Flash Attention V11 (causal) with TMA + SWIZZLE_128B + fused scale.
///
/// Causal masking: S[i,j] = -inf when kv_col j > q_row i (future tokens).
/// Only the diagonal KV block (kv_block == q_block) applies per-element masking;
/// future blocks (kv_block > q_block) are skipped after waiting for TMA completion.
///
/// 169 registers, 0 spills, 2 blocks/SM on SM121a. 90KB dynamic shared memory.
#[allow(clippy::too_many_arguments)]
#[cfg(feature = "experimental")]
pub fn flash_attn_bf16_v11_causal(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u16>,
    k: &CudaSlice<u16>,
    v: &CudaSlice<u16>,
    o: &mut CudaSlice<u16>,
    batch: u32,
    num_heads: u32,
    seq_q: u32,
    seq_kv: u32,
    scale: f32,
) -> Result<()> {
    validate_attn_dims(batch, num_heads, seq_q, seq_kv)?;
    validate_attn_bf16_bufs(q, k, v, o, batch, num_heads, seq_q, seq_kv)?;

    use cudarc::driver::sys::*;

    let grid_x = seq_q.div_ceil(64);

    let (q_dptr, _q_sync) = q.device_ptr(stream);
    let (k_dptr, _k_sync) = k.device_ptr(stream);
    let (v_dptr, _v_sync) = v.device_ptr(stream);
    let (o_dptr, _o_sync) = o.device_ptr(stream);

    let total_q_rows = batch * num_heads * seq_q;
    let total_kv_rows = batch * num_heads * seq_kv;

    let q_tma = create_tma_desc_swizzle(q_dptr as *mut _, total_q_rows, HEAD_DIM, 64, 64)?;
    let k_tma = create_tma_desc_swizzle(k_dptr as *mut _, total_kv_rows, HEAD_DIM, 64, 64)?;
    let v_tma = create_tma_desc_swizzle(v_dptr as *mut _, total_kv_rows, HEAD_DIM, 64, 64)?;

    let q_tma_u32: [u32; 32] = unsafe { core::mem::transmute(q_tma) };
    let k_tma_u32: [u32; 32] = unsafe { core::mem::transmute(k_tma) };
    let v_tma_u32: [u32; 32] = unsafe { core::mem::transmute(v_tma) };
    let q_tma_dev = stream.memcpy_stod(&q_tma_u32).map_err(SparkError::Driver)?;
    let k_tma_dev = stream.memcpy_stod(&k_tma_u32).map_err(SparkError::Driver)?;
    let v_tma_dev = stream.memcpy_stod(&v_tma_u32).map_err(SparkError::Driver)?;

    let (q_tma_dptr, _) = q_tma_dev.device_ptr(stream);
    let (k_tma_dptr, _) = k_tma_dev.device_ptr(stream);
    let (v_tma_dptr, _) = v_tma_dev.device_ptr(stream);

    let cu_func = module::load_kernel_raw(ctx, "fa_bf16_v11_causal", "flash_attn_bf16_v11_causal")?;
    let cu_stream = stream.cu_stream();

    const SMEM_TOTAL: i32 = 82032;
    let attr_result = unsafe {
        cuFuncSetAttribute(
            cu_func,
            CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            SMEM_TOTAL,
        )
    };
    if attr_result != CUresult::CUDA_SUCCESS {
        return Err(SparkError::LaunchFailed(format!(
            "cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES={}) failed: {:?}",
            SMEM_TOTAL, attr_result
        )));
    }

    let params: [*mut core::ffi::c_void; 8] = [
        &q_tma_dptr as *const u64 as *mut _,
        &k_tma_dptr as *const u64 as *mut _,
        &v_tma_dptr as *const u64 as *mut _,
        &o_dptr as *const u64 as *mut _,
        &seq_q as *const u32 as *mut _,
        &seq_kv as *const u32 as *mut _,
        &num_heads as *const u32 as *mut _,
        &scale as *const f32 as *mut _,
    ];

    let result = unsafe {
        cuLaunchKernel(
            cu_func,
            grid_x,
            num_heads,
            batch,
            160,
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
            "cuLaunchKernel failed for V11 causal: {:?}",
            result
        )));
    }

    Ok(())
}

/// Launch BF16 Flash Attention V11 varlen non-causal (TMA + SWIZZLE_128B + fused scale).
///
/// Variable-length non-causal attention with head-major layout.
/// Same interface as `flash_attn_bf16_v11_varlen_causal` but processes all KV blocks
/// without masking — full cross-attention between Q and KV sequences.
#[allow(clippy::too_many_arguments)]
#[cfg(feature = "experimental")]
pub fn flash_attn_bf16_v11_varlen(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u16>,
    k: &CudaSlice<u16>,
    v: &CudaSlice<u16>,
    o: &mut CudaSlice<u16>,
    cu_seqlens_q: &CudaSlice<u32>,
    cu_seqlens_k: &CudaSlice<u32>,
    batch: u32,
    num_heads: u32,
    max_seqlen_q: u32,
    total_q: u32,
    total_kv: u32,
    scale: f32,
) -> Result<()> {
    if batch == 0 {
        return Err(SparkError::InvalidArgument("batch must be > 0".into()));
    }
    if num_heads == 0 {
        return Err(SparkError::InvalidArgument("num_heads must be > 0".into()));
    }
    if max_seqlen_q == 0 || total_q == 0 || total_kv == 0 {
        return Err(SparkError::InvalidArgument(
            "max_seqlen_q, total_q, total_kv must be > 0".into(),
        ));
    }
    let cu_need = (batch + 1) as usize;
    if cu_seqlens_q.len() < cu_need || cu_seqlens_k.len() < cu_need {
        return Err(SparkError::InvalidArgument(format!(
            "cu_seqlens too small: need {cu_need}"
        )));
    }

    use cudarc::driver::sys::*;

    let grid_x = max_seqlen_q.div_ceil(64);

    let (q_dptr, _q_sync) = q.device_ptr(stream);
    let (k_dptr, _k_sync) = k.device_ptr(stream);
    let (v_dptr, _v_sync) = v.device_ptr(stream);
    let (o_dptr, _o_sync) = o.device_ptr(stream);
    let (cu_q_dptr, _cu_q_sync) = cu_seqlens_q.device_ptr(stream);
    let (cu_k_dptr, _cu_k_sync) = cu_seqlens_k.device_ptr(stream);

    let q_tma = create_tma_desc_swizzle(q_dptr as *mut _, num_heads * total_q, HEAD_DIM, 64, 64)?;
    let k_tma = create_tma_desc_swizzle(k_dptr as *mut _, num_heads * total_kv, HEAD_DIM, 64, 64)?;
    let v_tma = create_tma_desc_swizzle(v_dptr as *mut _, num_heads * total_kv, HEAD_DIM, 64, 64)?;

    let q_tma_u32: [u32; 32] = unsafe { core::mem::transmute(q_tma) };
    let k_tma_u32: [u32; 32] = unsafe { core::mem::transmute(k_tma) };
    let v_tma_u32: [u32; 32] = unsafe { core::mem::transmute(v_tma) };
    let q_tma_dev = stream.memcpy_stod(&q_tma_u32).map_err(SparkError::Driver)?;
    let k_tma_dev = stream.memcpy_stod(&k_tma_u32).map_err(SparkError::Driver)?;
    let v_tma_dev = stream.memcpy_stod(&v_tma_u32).map_err(SparkError::Driver)?;

    let (q_tma_dptr, _) = q_tma_dev.device_ptr(stream);
    let (k_tma_dptr, _) = k_tma_dev.device_ptr(stream);
    let (v_tma_dptr, _) = v_tma_dev.device_ptr(stream);

    let cu_func = module::load_kernel_raw(ctx, "fa_bf16_v11_varlen", "flash_attn_bf16_v11_varlen")?;
    let cu_stream = stream.cu_stream();

    const SMEM_TOTAL: i32 = 82032;
    let attr_result = unsafe {
        cuFuncSetAttribute(
            cu_func,
            CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            SMEM_TOTAL,
        )
    };
    if attr_result != CUresult::CUDA_SUCCESS {
        return Err(SparkError::LaunchFailed(format!(
            "cuFuncSetAttribute failed: {:?}",
            attr_result
        )));
    }

    let params: [*mut core::ffi::c_void; 9] = [
        &q_tma_dptr as *const u64 as *mut _,
        &k_tma_dptr as *const u64 as *mut _,
        &v_tma_dptr as *const u64 as *mut _,
        &o_dptr as *const u64 as *mut _,
        &cu_q_dptr as *const u64 as *mut _,
        &cu_k_dptr as *const u64 as *mut _,
        &total_q as *const u32 as *mut _,
        &total_kv as *const u32 as *mut _,
        &scale as *const f32 as *mut _,
    ];

    let result = unsafe {
        cuLaunchKernel(
            cu_func,
            grid_x,
            num_heads,
            batch,
            160,
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
            "cuLaunchKernel failed for V11 varlen: {:?}",
            result
        )));
    }

    Ok(())
}

/// Launch BF16 Flash Attention V11 varlen causal (TMA + SWIZZLE_128B + fused scale).
///
/// Variable-length causal attention with head-major layout:
///   Q, K, V, O: `[num_heads, total_tokens, D]` row-major BF16 (head-major varlen)
///
/// TMA descriptor is created over `[num_heads * total_tokens, D]` (flat 2D view),
/// so consecutive tokens for the same head are contiguous in memory.
///
/// - `cu_seqlens_q`: GPU buffer `[batch+1]` u32, cu_seqlens_q[b]..cu_seqlens_q[b+1] is Q range for batch b
/// - `cu_seqlens_k`: GPU buffer `[batch+1]` u32, analogous for KV
/// - `total_q`: total Q tokens = cu_seqlens_q[batch]
/// - `total_kv`: total KV tokens = cu_seqlens_k[batch]
/// - `max_seqlen_q`: max Q sequence length across batch elements (used for grid size)
#[allow(clippy::too_many_arguments)]
#[cfg(feature = "experimental")]
pub fn flash_attn_bf16_v11_varlen_causal(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u16>,
    k: &CudaSlice<u16>,
    v: &CudaSlice<u16>,
    o: &mut CudaSlice<u16>,
    cu_seqlens_q: &CudaSlice<u32>,
    cu_seqlens_k: &CudaSlice<u32>,
    batch: u32,
    num_heads: u32,
    max_seqlen_q: u32,
    total_q: u32,
    total_kv: u32,
    scale: f32,
) -> Result<()> {
    if batch == 0 {
        return Err(SparkError::InvalidArgument("batch must be > 0".into()));
    }
    if num_heads == 0 {
        return Err(SparkError::InvalidArgument("num_heads must be > 0".into()));
    }
    if max_seqlen_q == 0 || total_q == 0 || total_kv == 0 {
        return Err(SparkError::InvalidArgument(
            "max_seqlen_q, total_q, total_kv must be > 0".into(),
        ));
    }
    let cu_need = (batch + 1) as usize;
    if cu_seqlens_q.len() < cu_need || cu_seqlens_k.len() < cu_need {
        return Err(SparkError::InvalidArgument(format!(
            "cu_seqlens too small: need {cu_need}"
        )));
    }

    use cudarc::driver::sys::*;

    let grid_x = max_seqlen_q.div_ceil(64);

    let (q_dptr, _q_sync) = q.device_ptr(stream);
    let (k_dptr, _k_sync) = k.device_ptr(stream);
    let (v_dptr, _v_sync) = v.device_ptr(stream);
    let (o_dptr, _o_sync) = o.device_ptr(stream);
    let (cu_q_dptr, _cu_q_sync) = cu_seqlens_q.device_ptr(stream);
    let (cu_k_dptr, _cu_k_sync) = cu_seqlens_k.device_ptr(stream);

    // TMA descriptors: head-major flat view [num_heads * total_tokens, D]
    let q_tma = create_tma_desc_swizzle(q_dptr as *mut _, num_heads * total_q, HEAD_DIM, 64, 64)?;
    let k_tma = create_tma_desc_swizzle(k_dptr as *mut _, num_heads * total_kv, HEAD_DIM, 64, 64)?;
    let v_tma = create_tma_desc_swizzle(v_dptr as *mut _, num_heads * total_kv, HEAD_DIM, 64, 64)?;

    let q_tma_u32: [u32; 32] = unsafe { core::mem::transmute(q_tma) };
    let k_tma_u32: [u32; 32] = unsafe { core::mem::transmute(k_tma) };
    let v_tma_u32: [u32; 32] = unsafe { core::mem::transmute(v_tma) };
    let q_tma_dev = stream.memcpy_stod(&q_tma_u32).map_err(SparkError::Driver)?;
    let k_tma_dev = stream.memcpy_stod(&k_tma_u32).map_err(SparkError::Driver)?;
    let v_tma_dev = stream.memcpy_stod(&v_tma_u32).map_err(SparkError::Driver)?;

    let (q_tma_dptr, _) = q_tma_dev.device_ptr(stream);
    let (k_tma_dptr, _) = k_tma_dev.device_ptr(stream);
    let (v_tma_dptr, _) = v_tma_dev.device_ptr(stream);

    let cu_func = module::load_kernel_raw(
        ctx,
        "fa_bf16_v11_varlen_causal",
        "flash_attn_bf16_v11_varlen_causal",
    )?;
    let cu_stream = stream.cu_stream();

    const SMEM_TOTAL: i32 = 82032;
    let attr_result = unsafe {
        cuFuncSetAttribute(
            cu_func,
            CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            SMEM_TOTAL,
        )
    };
    if attr_result != CUresult::CUDA_SUCCESS {
        return Err(SparkError::LaunchFailed(format!(
            "cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES={}) failed: {:?}",
            SMEM_TOTAL, attr_result
        )));
    }

    let params: [*mut core::ffi::c_void; 9] = [
        &q_tma_dptr as *const u64 as *mut _,
        &k_tma_dptr as *const u64 as *mut _,
        &v_tma_dptr as *const u64 as *mut _,
        &o_dptr as *const u64 as *mut _,
        &cu_q_dptr as *const u64 as *mut _,
        &cu_k_dptr as *const u64 as *mut _,
        &total_q as *const u32 as *mut _,
        &total_kv as *const u32 as *mut _,
        &scale as *const f32 as *mut _,
    ];

    let result = unsafe {
        cuLaunchKernel(
            cu_func,
            grid_x,
            num_heads,
            batch,
            160,
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
            "cuLaunchKernel failed for V11 varlen causal: {:?}",
            result
        )));
    }

    Ok(())
}

/// Launch BF16 Flash Attention V11 GQA (non-causal) with TMA + SWIZZLE_128B + fused scale.
///
/// Grouped Query Attention: Q has `num_heads_q` heads, K/V have `num_heads_kv` heads.
/// Requires `num_heads_q % num_heads_kv == 0`. Each KV head is shared by
/// `num_heads_q / num_heads_kv` Q heads.
///
/// Q, O: [batch, num_heads_q, seq_q, 128] row-major bf16
/// K, V: [batch, num_heads_kv, seq_kv, 128] row-major bf16
///
/// 165 registers, 0 spills, 2 blocks/SM on SM121a.
#[allow(clippy::too_many_arguments)]
#[cfg(feature = "experimental")]
pub fn flash_attn_bf16_v11_gqa(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u16>,
    k: &CudaSlice<u16>,
    v: &CudaSlice<u16>,
    o: &mut CudaSlice<u16>,
    batch: u32,
    num_heads_q: u32,
    num_heads_kv: u32,
    seq_q: u32,
    seq_kv: u32,
    scale: f32,
) -> Result<()> {
    if batch == 0 || num_heads_q == 0 || num_heads_kv == 0 || seq_q == 0 || seq_kv == 0 {
        return Err(SparkError::InvalidArgument(
            "batch, num_heads_q, num_heads_kv, seq_q, seq_kv must be > 0".into(),
        ));
    }
    if !num_heads_q.is_multiple_of(num_heads_kv) {
        return Err(SparkError::InvalidArgument(format!(
            "num_heads_q ({num_heads_q}) must be divisible by num_heads_kv ({num_heads_kv})"
        )));
    }

    use cudarc::driver::sys::*;

    let grid_x = seq_q.div_ceil(64);

    let (q_dptr, _q_sync) = q.device_ptr(stream);
    let (k_dptr, _k_sync) = k.device_ptr(stream);
    let (v_dptr, _v_sync) = v.device_ptr(stream);
    let (o_dptr, _o_sync) = o.device_ptr(stream);

    let total_q_rows = batch * num_heads_q * seq_q;
    let total_kv_rows = batch * num_heads_kv * seq_kv;

    let q_tma = create_tma_desc_swizzle(q_dptr as *mut _, total_q_rows, HEAD_DIM, 64, 64)?;
    let k_tma = create_tma_desc_swizzle(k_dptr as *mut _, total_kv_rows, HEAD_DIM, 64, 64)?;
    let v_tma = create_tma_desc_swizzle(v_dptr as *mut _, total_kv_rows, HEAD_DIM, 64, 64)?;

    let q_tma_u32: [u32; 32] = unsafe { core::mem::transmute(q_tma) };
    let k_tma_u32: [u32; 32] = unsafe { core::mem::transmute(k_tma) };
    let v_tma_u32: [u32; 32] = unsafe { core::mem::transmute(v_tma) };
    let q_tma_dev = stream.memcpy_stod(&q_tma_u32).map_err(SparkError::Driver)?;
    let k_tma_dev = stream.memcpy_stod(&k_tma_u32).map_err(SparkError::Driver)?;
    let v_tma_dev = stream.memcpy_stod(&v_tma_u32).map_err(SparkError::Driver)?;

    let (q_tma_dptr, _) = q_tma_dev.device_ptr(stream);
    let (k_tma_dptr, _) = k_tma_dev.device_ptr(stream);
    let (v_tma_dptr, _) = v_tma_dev.device_ptr(stream);

    let cu_func = module::load_kernel_raw(ctx, "fa_bf16_v11_gqa", "flash_attn_bf16_v11_gqa")?;
    let cu_stream = stream.cu_stream();

    const SMEM_TOTAL: i32 = 82032;
    let attr_result = unsafe {
        cuFuncSetAttribute(
            cu_func,
            CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            SMEM_TOTAL,
        )
    };
    if attr_result != CUresult::CUDA_SUCCESS {
        return Err(SparkError::LaunchFailed(format!(
            "cuFuncSetAttribute failed: {:?}",
            attr_result
        )));
    }

    let params: [*mut core::ffi::c_void; 9] = [
        &q_tma_dptr as *const u64 as *mut _,
        &k_tma_dptr as *const u64 as *mut _,
        &v_tma_dptr as *const u64 as *mut _,
        &o_dptr as *const u64 as *mut _,
        &seq_q as *const u32 as *mut _,
        &seq_kv as *const u32 as *mut _,
        &num_heads_q as *const u32 as *mut _,
        &num_heads_kv as *const u32 as *mut _,
        &scale as *const f32 as *mut _,
    ];

    let result = unsafe {
        cuLaunchKernel(
            cu_func,
            grid_x,
            num_heads_q,
            batch,
            160,
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
            "cuLaunchKernel failed for V11 GQA: {:?}",
            result
        )));
    }

    Ok(())
}

/// Launch BF16 Flash Attention V11 GQA causal with TMA + SWIZZLE_128B + fused scale.
///
/// Grouped Query Attention with causal masking.
/// Q has `num_heads_q` heads, K/V have `num_heads_kv` heads.
/// Requires `num_heads_q % num_heads_kv == 0`.
///
/// 166 registers, 0 spills, 2 blocks/SM on SM121a.
#[allow(clippy::too_many_arguments)]
#[cfg(feature = "experimental")]
pub fn flash_attn_bf16_v11_gqa_causal(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u16>,
    k: &CudaSlice<u16>,
    v: &CudaSlice<u16>,
    o: &mut CudaSlice<u16>,
    batch: u32,
    num_heads_q: u32,
    num_heads_kv: u32,
    seq_q: u32,
    seq_kv: u32,
    scale: f32,
) -> Result<()> {
    if batch == 0 || num_heads_q == 0 || num_heads_kv == 0 || seq_q == 0 || seq_kv == 0 {
        return Err(SparkError::InvalidArgument(
            "batch, num_heads_q, num_heads_kv, seq_q, seq_kv must be > 0".into(),
        ));
    }
    if !num_heads_q.is_multiple_of(num_heads_kv) {
        return Err(SparkError::InvalidArgument(format!(
            "num_heads_q ({num_heads_q}) must be divisible by num_heads_kv ({num_heads_kv})"
        )));
    }

    use cudarc::driver::sys::*;

    let grid_x = seq_q.div_ceil(64);

    let (q_dptr, _q_sync) = q.device_ptr(stream);
    let (k_dptr, _k_sync) = k.device_ptr(stream);
    let (v_dptr, _v_sync) = v.device_ptr(stream);
    let (o_dptr, _o_sync) = o.device_ptr(stream);

    let total_q_rows = batch * num_heads_q * seq_q;
    let total_kv_rows = batch * num_heads_kv * seq_kv;

    let q_tma = create_tma_desc_swizzle(q_dptr as *mut _, total_q_rows, HEAD_DIM, 64, 64)?;
    let k_tma = create_tma_desc_swizzle(k_dptr as *mut _, total_kv_rows, HEAD_DIM, 64, 64)?;
    let v_tma = create_tma_desc_swizzle(v_dptr as *mut _, total_kv_rows, HEAD_DIM, 64, 64)?;

    let q_tma_u32: [u32; 32] = unsafe { core::mem::transmute(q_tma) };
    let k_tma_u32: [u32; 32] = unsafe { core::mem::transmute(k_tma) };
    let v_tma_u32: [u32; 32] = unsafe { core::mem::transmute(v_tma) };
    let q_tma_dev = stream.memcpy_stod(&q_tma_u32).map_err(SparkError::Driver)?;
    let k_tma_dev = stream.memcpy_stod(&k_tma_u32).map_err(SparkError::Driver)?;
    let v_tma_dev = stream.memcpy_stod(&v_tma_u32).map_err(SparkError::Driver)?;

    let (q_tma_dptr, _) = q_tma_dev.device_ptr(stream);
    let (k_tma_dptr, _) = k_tma_dev.device_ptr(stream);
    let (v_tma_dptr, _) = v_tma_dev.device_ptr(stream);

    let cu_func = module::load_kernel_raw(
        ctx,
        "fa_bf16_v11_gqa_causal",
        "flash_attn_bf16_v11_gqa_causal",
    )?;
    let cu_stream = stream.cu_stream();

    const SMEM_TOTAL: i32 = 82032;
    let attr_result = unsafe {
        cuFuncSetAttribute(
            cu_func,
            CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            SMEM_TOTAL,
        )
    };
    if attr_result != CUresult::CUDA_SUCCESS {
        return Err(SparkError::LaunchFailed(format!(
            "cuFuncSetAttribute failed: {:?}",
            attr_result
        )));
    }

    let params: [*mut core::ffi::c_void; 9] = [
        &q_tma_dptr as *const u64 as *mut _,
        &k_tma_dptr as *const u64 as *mut _,
        &v_tma_dptr as *const u64 as *mut _,
        &o_dptr as *const u64 as *mut _,
        &seq_q as *const u32 as *mut _,
        &seq_kv as *const u32 as *mut _,
        &num_heads_q as *const u32 as *mut _,
        &num_heads_kv as *const u32 as *mut _,
        &scale as *const f32 as *mut _,
    ];

    let result = unsafe {
        cuLaunchKernel(
            cu_func,
            grid_x,
            num_heads_q,
            batch,
            160,
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
            "cuLaunchKernel failed for V11 GQA causal: {:?}",
            result
        )));
    }

    Ok(())
}

/// Launch BF16 Flash Attention V11 varlen GQA (non-causal) with TMA + SWIZZLE_128B.
///
/// Variable-length GQA with head-major layout.
/// Q, O use [H_q, total_q, D]; K, V use [H_kv, total_kv, D].
/// TMA descriptors: Q over [H_q * total_q, D], K/V over [H_kv * total_kv, D].
///
/// 166 registers, 0 spills, 2 blocks/SM on SM121a.
#[allow(clippy::too_many_arguments)]
#[cfg(feature = "experimental")]
pub fn flash_attn_bf16_v11_varlen_gqa(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u16>,
    k: &CudaSlice<u16>,
    v: &CudaSlice<u16>,
    o: &mut CudaSlice<u16>,
    cu_seqlens_q: &CudaSlice<u32>,
    cu_seqlens_k: &CudaSlice<u32>,
    batch: u32,
    num_heads_q: u32,
    num_heads_kv: u32,
    max_seqlen_q: u32,
    total_q: u32,
    total_kv: u32,
    scale: f32,
) -> Result<()> {
    if batch == 0 || num_heads_q == 0 || num_heads_kv == 0 {
        return Err(SparkError::InvalidArgument(
            "batch, num_heads_q, num_heads_kv must be > 0".into(),
        ));
    }
    if !num_heads_q.is_multiple_of(num_heads_kv) {
        return Err(SparkError::InvalidArgument(format!(
            "num_heads_q ({num_heads_q}) must be divisible by num_heads_kv ({num_heads_kv})"
        )));
    }
    if max_seqlen_q == 0 || total_q == 0 || total_kv == 0 {
        return Err(SparkError::InvalidArgument(
            "max_seqlen_q, total_q, total_kv must be > 0".into(),
        ));
    }
    let cu_need = (batch + 1) as usize;
    if cu_seqlens_q.len() < cu_need || cu_seqlens_k.len() < cu_need {
        return Err(SparkError::InvalidArgument(format!(
            "cu_seqlens too small: need {cu_need}"
        )));
    }

    use cudarc::driver::sys::*;

    let grid_x = max_seqlen_q.div_ceil(64);

    let (q_dptr, _q_sync) = q.device_ptr(stream);
    let (k_dptr, _k_sync) = k.device_ptr(stream);
    let (v_dptr, _v_sync) = v.device_ptr(stream);
    let (o_dptr, _o_sync) = o.device_ptr(stream);
    let (cu_q_dptr, _cu_q_sync) = cu_seqlens_q.device_ptr(stream);
    let (cu_k_dptr, _cu_k_sync) = cu_seqlens_k.device_ptr(stream);

    // TMA descriptors: head-major flat view
    let q_tma = create_tma_desc_swizzle(q_dptr as *mut _, num_heads_q * total_q, HEAD_DIM, 64, 64)?;
    let k_tma =
        create_tma_desc_swizzle(k_dptr as *mut _, num_heads_kv * total_kv, HEAD_DIM, 64, 64)?;
    let v_tma =
        create_tma_desc_swizzle(v_dptr as *mut _, num_heads_kv * total_kv, HEAD_DIM, 64, 64)?;

    let q_tma_u32: [u32; 32] = unsafe { core::mem::transmute(q_tma) };
    let k_tma_u32: [u32; 32] = unsafe { core::mem::transmute(k_tma) };
    let v_tma_u32: [u32; 32] = unsafe { core::mem::transmute(v_tma) };
    let q_tma_dev = stream.memcpy_stod(&q_tma_u32).map_err(SparkError::Driver)?;
    let k_tma_dev = stream.memcpy_stod(&k_tma_u32).map_err(SparkError::Driver)?;
    let v_tma_dev = stream.memcpy_stod(&v_tma_u32).map_err(SparkError::Driver)?;

    let (q_tma_dptr, _) = q_tma_dev.device_ptr(stream);
    let (k_tma_dptr, _) = k_tma_dev.device_ptr(stream);
    let (v_tma_dptr, _) = v_tma_dev.device_ptr(stream);

    let cu_func = module::load_kernel_raw(
        ctx,
        "fa_bf16_v11_varlen_gqa",
        "flash_attn_bf16_v11_varlen_gqa",
    )?;
    let cu_stream = stream.cu_stream();

    const SMEM_TOTAL: i32 = 82032;
    let attr_result = unsafe {
        cuFuncSetAttribute(
            cu_func,
            CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            SMEM_TOTAL,
        )
    };
    if attr_result != CUresult::CUDA_SUCCESS {
        return Err(SparkError::LaunchFailed(format!(
            "cuFuncSetAttribute failed: {:?}",
            attr_result
        )));
    }

    let params: [*mut core::ffi::c_void; 11] = [
        &q_tma_dptr as *const u64 as *mut _,
        &k_tma_dptr as *const u64 as *mut _,
        &v_tma_dptr as *const u64 as *mut _,
        &o_dptr as *const u64 as *mut _,
        &cu_q_dptr as *const u64 as *mut _,
        &cu_k_dptr as *const u64 as *mut _,
        &total_q as *const u32 as *mut _,
        &total_kv as *const u32 as *mut _,
        &num_heads_q as *const u32 as *mut _,
        &num_heads_kv as *const u32 as *mut _,
        &scale as *const f32 as *mut _,
    ];

    let result = unsafe {
        cuLaunchKernel(
            cu_func,
            grid_x,
            num_heads_q,
            batch,
            160,
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
            "cuLaunchKernel failed for V11 varlen GQA: {:?}",
            result
        )));
    }

    Ok(())
}

/// Launch BF16 Flash Attention V11 varlen GQA causal with TMA + SWIZZLE_128B.
///
/// Variable-length GQA with causal masking and head-major layout.
/// Q, O use [H_q, total_q, D]; K, V use [H_kv, total_kv, D].
///
/// 166 registers, 0 spills, 2 blocks/SM on SM121a.
#[allow(clippy::too_many_arguments)]
#[cfg(feature = "experimental")]
pub fn flash_attn_bf16_v11_varlen_gqa_causal(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u16>,
    k: &CudaSlice<u16>,
    v: &CudaSlice<u16>,
    o: &mut CudaSlice<u16>,
    cu_seqlens_q: &CudaSlice<u32>,
    cu_seqlens_k: &CudaSlice<u32>,
    batch: u32,
    num_heads_q: u32,
    num_heads_kv: u32,
    max_seqlen_q: u32,
    total_q: u32,
    total_kv: u32,
    scale: f32,
) -> Result<()> {
    if batch == 0 || num_heads_q == 0 || num_heads_kv == 0 {
        return Err(SparkError::InvalidArgument(
            "batch, num_heads_q, num_heads_kv must be > 0".into(),
        ));
    }
    if !num_heads_q.is_multiple_of(num_heads_kv) {
        return Err(SparkError::InvalidArgument(format!(
            "num_heads_q ({num_heads_q}) must be divisible by num_heads_kv ({num_heads_kv})"
        )));
    }
    if max_seqlen_q == 0 || total_q == 0 || total_kv == 0 {
        return Err(SparkError::InvalidArgument(
            "max_seqlen_q, total_q, total_kv must be > 0".into(),
        ));
    }
    let cu_need = (batch + 1) as usize;
    if cu_seqlens_q.len() < cu_need || cu_seqlens_k.len() < cu_need {
        return Err(SparkError::InvalidArgument(format!(
            "cu_seqlens too small: need {cu_need}"
        )));
    }

    use cudarc::driver::sys::*;

    let grid_x = max_seqlen_q.div_ceil(64);

    let (q_dptr, _q_sync) = q.device_ptr(stream);
    let (k_dptr, _k_sync) = k.device_ptr(stream);
    let (v_dptr, _v_sync) = v.device_ptr(stream);
    let (o_dptr, _o_sync) = o.device_ptr(stream);
    let (cu_q_dptr, _cu_q_sync) = cu_seqlens_q.device_ptr(stream);
    let (cu_k_dptr, _cu_k_sync) = cu_seqlens_k.device_ptr(stream);

    // TMA descriptors: head-major flat view
    let q_tma = create_tma_desc_swizzle(q_dptr as *mut _, num_heads_q * total_q, HEAD_DIM, 64, 64)?;
    let k_tma =
        create_tma_desc_swizzle(k_dptr as *mut _, num_heads_kv * total_kv, HEAD_DIM, 64, 64)?;
    let v_tma =
        create_tma_desc_swizzle(v_dptr as *mut _, num_heads_kv * total_kv, HEAD_DIM, 64, 64)?;

    let q_tma_u32: [u32; 32] = unsafe { core::mem::transmute(q_tma) };
    let k_tma_u32: [u32; 32] = unsafe { core::mem::transmute(k_tma) };
    let v_tma_u32: [u32; 32] = unsafe { core::mem::transmute(v_tma) };
    let q_tma_dev = stream.memcpy_stod(&q_tma_u32).map_err(SparkError::Driver)?;
    let k_tma_dev = stream.memcpy_stod(&k_tma_u32).map_err(SparkError::Driver)?;
    let v_tma_dev = stream.memcpy_stod(&v_tma_u32).map_err(SparkError::Driver)?;

    let (q_tma_dptr, _) = q_tma_dev.device_ptr(stream);
    let (k_tma_dptr, _) = k_tma_dev.device_ptr(stream);
    let (v_tma_dptr, _) = v_tma_dev.device_ptr(stream);

    let cu_func = module::load_kernel_raw(
        ctx,
        "fa_bf16_v11_varlen_gqa_causal",
        "flash_attn_bf16_v11_varlen_gqa_causal",
    )?;
    let cu_stream = stream.cu_stream();

    const SMEM_TOTAL: i32 = 82032;
    let attr_result = unsafe {
        cuFuncSetAttribute(
            cu_func,
            CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            SMEM_TOTAL,
        )
    };
    if attr_result != CUresult::CUDA_SUCCESS {
        return Err(SparkError::LaunchFailed(format!(
            "cuFuncSetAttribute failed: {:?}",
            attr_result
        )));
    }

    let params: [*mut core::ffi::c_void; 11] = [
        &q_tma_dptr as *const u64 as *mut _,
        &k_tma_dptr as *const u64 as *mut _,
        &v_tma_dptr as *const u64 as *mut _,
        &o_dptr as *const u64 as *mut _,
        &cu_q_dptr as *const u64 as *mut _,
        &cu_k_dptr as *const u64 as *mut _,
        &total_q as *const u32 as *mut _,
        &total_kv as *const u32 as *mut _,
        &num_heads_q as *const u32 as *mut _,
        &num_heads_kv as *const u32 as *mut _,
        &scale as *const f32 as *mut _,
    ];

    let result = unsafe {
        cuLaunchKernel(
            cu_func,
            grid_x,
            num_heads_q,
            batch,
            160,
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
            "cuLaunchKernel failed for V11 varlen GQA causal: {:?}",
            result
        )));
    }

    Ok(())
}

/// Create a TMA descriptor for FP8 (uint8) 2D tiles with SWIZZLE_128B.
///
/// For FP8, each row is 128 bytes (128 cols × 1B), matching exactly one
/// SWIZZLE_128B segment. So boxDim[0]=128 covers the full head dimension
/// in a single TMA load (no left/right split needed).
fn create_tma_desc_fp8_swizzle(
    global_ptr: *mut core::ffi::c_void,
    seq_len: u32,
    head_dim: u32,
    tile_rows: u32,
    tile_cols: u32, // must be 128 for FP8 SWIZZLE_128B
) -> Result<cudarc::driver::sys::CUtensorMap> {
    use cudarc::driver::sys::*;

    let mut tma = CUtensorMap_st::default();

    let global_dim: [u64; 2] = [head_dim as u64, seq_len as u64];
    let global_strides: [u64; 1] = [head_dim as u64]; // bytes per row (1 byte/element)
    let box_dim: [u32; 2] = [tile_cols, tile_rows];
    let elem_strides: [u32; 2] = [1, 1];

    let result = unsafe {
        cuTensorMapEncodeTiled(
            &mut tma,
            CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8,
            2,
            global_ptr,
            global_dim.as_ptr(),
            global_strides.as_ptr(),
            box_dim.as_ptr(),
            elem_strides.as_ptr(),
            CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
            CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
            CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        )
    };

    if result != CUresult::CUDA_SUCCESS {
        return Err(SparkError::LaunchFailed(format!(
            "cuTensorMapEncodeTiled (fp8 swizzle) failed: {:?}",
            result
        )));
    }

    Ok(tma)
}

/// Create a TMA descriptor for FP8 (uint8) 2D tiles with SWIZZLE_NONE.
///
/// Used for V tiles where transposed B-fragment loading uses simple
/// stride-128 byte access (no swizzle XOR needed).
#[cfg(feature = "experimental")]
fn create_tma_desc_fp8(
    global_ptr: *mut core::ffi::c_void,
    seq_len: u32,
    head_dim: u32,
    tile_rows: u32,
    tile_cols: u32,
) -> Result<cudarc::driver::sys::CUtensorMap> {
    use cudarc::driver::sys::*;

    let mut tma = CUtensorMap_st::default();

    let global_dim: [u64; 2] = [head_dim as u64, seq_len as u64];
    let global_strides: [u64; 1] = [head_dim as u64]; // bytes per row
    let box_dim: [u32; 2] = [tile_cols, tile_rows];
    let elem_strides: [u32; 2] = [1, 1];

    let result = unsafe {
        cuTensorMapEncodeTiled(
            &mut tma,
            CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8,
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
            "cuTensorMapEncodeTiled (fp8) failed: {:?}",
            result
        )));
    }

    Ok(tma)
}

/// Flash Attention FP8 V11 TMA: FP8 e4m3 QKV input → BF16 O output.
///
/// Q, K: SWIZZLE_128B TMA, single load per 64×128 tile (128B row = one segment).
/// V: SWIZZLE_NONE TMA, transposed B-fragment via stride-128 byte loads.
/// Uses FP8 m16n8k32 MMA (32 QK + 32 PV per KV block).
/// 150 registers, 0 spills, 2 blocks/SM.
///
/// Q, K, V: [batch, num_heads, seq, 128] row-major fp8 e4m3 (u8 raw bits).
/// O: [batch, num_heads, seq_q, 128] row-major bf16 (u16 raw bits, output).
///
/// 5 warps (160 threads) per block, Br=64, Bc=64.
/// ~46KB dynamic shared memory.
#[allow(clippy::too_many_arguments)]
#[cfg(feature = "experimental")]
pub fn flash_attn_fp8_v11_tma(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u8>,
    k: &CudaSlice<u8>,
    v: &CudaSlice<u8>,
    o: &mut CudaSlice<u16>,
    batch: u32,
    num_heads: u32,
    seq_q: u32,
    seq_kv: u32,
    scale: f32,
) -> Result<()> {
    validate_attn_dims(batch, num_heads, seq_q, seq_kv)?;

    let q_need = batch as usize * num_heads as usize * seq_q as usize * HEAD_DIM as usize;
    let kv_need = batch as usize * num_heads as usize * seq_kv as usize * HEAD_DIM as usize;
    if q.len() < q_need {
        return Err(SparkError::InvalidArgument(format!(
            "Q buffer too small: {} < {q_need}",
            q.len()
        )));
    }
    if k.len() < kv_need {
        return Err(SparkError::InvalidArgument(format!(
            "K buffer too small: {} < {kv_need}",
            k.len()
        )));
    }
    if v.len() < kv_need {
        return Err(SparkError::InvalidArgument(format!(
            "V buffer too small: {} < {kv_need}",
            v.len()
        )));
    }
    if o.len() < q_need {
        return Err(SparkError::InvalidArgument(format!(
            "O buffer too small: {} < {q_need}",
            o.len()
        )));
    }

    use cudarc::driver::sys::*;

    let grid_x = seq_q.div_ceil(64);

    let (q_dptr, _q_sync) = q.device_ptr(stream);
    let (k_dptr, _k_sync) = k.device_ptr(stream);
    let (v_dptr, _v_sync) = v.device_ptr(stream);
    let (o_dptr, _o_sync) = o.device_ptr(stream);

    let total_q_rows = batch * num_heads * seq_q;
    let total_kv_rows = batch * num_heads * seq_kv;

    // Q/K: SWIZZLE_128B, boxDim=[128,64] — single load covers full 128-col tile
    let q_tma = create_tma_desc_fp8_swizzle(q_dptr as *mut _, total_q_rows, HEAD_DIM, 64, 128)?;
    let k_tma = create_tma_desc_fp8_swizzle(k_dptr as *mut _, total_kv_rows, HEAD_DIM, 64, 128)?;
    // V: SWIZZLE_NONE, boxDim=[128,64] — transposed B-frag via stride-128 byte loads
    let v_tma = create_tma_desc_fp8(v_dptr as *mut _, total_kv_rows, HEAD_DIM, 64, 128)?;

    let q_tma_u32: [u32; 32] = unsafe { core::mem::transmute(q_tma) };
    let k_tma_u32: [u32; 32] = unsafe { core::mem::transmute(k_tma) };
    let v_tma_u32: [u32; 32] = unsafe { core::mem::transmute(v_tma) };
    let q_tma_dev = stream.memcpy_stod(&q_tma_u32).map_err(SparkError::Driver)?;
    let k_tma_dev = stream.memcpy_stod(&k_tma_u32).map_err(SparkError::Driver)?;
    let v_tma_dev = stream.memcpy_stod(&v_tma_u32).map_err(SparkError::Driver)?;

    let (q_tma_dptr, _) = q_tma_dev.device_ptr(stream);
    let (k_tma_dptr, _) = k_tma_dev.device_ptr(stream);
    let (v_tma_dptr, _) = v_tma_dev.device_ptr(stream);

    let cu_func = module::load_kernel_raw(ctx, "fa_fp8_v11_tma", "flash_attn_fp8_v11_tma")?;
    let cu_stream = stream.cu_stream();

    const SMEM_TOTAL: i32 = 45168;
    let attr_result = unsafe {
        cuFuncSetAttribute(
            cu_func,
            CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            SMEM_TOTAL,
        )
    };
    if attr_result != CUresult::CUDA_SUCCESS {
        return Err(SparkError::LaunchFailed(format!(
            "cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES={}) failed: {:?}",
            SMEM_TOTAL, attr_result
        )));
    }

    let params: [*mut core::ffi::c_void; 8] = [
        &q_tma_dptr as *const u64 as *mut _,
        &k_tma_dptr as *const u64 as *mut _,
        &v_tma_dptr as *const u64 as *mut _,
        &o_dptr as *const u64 as *mut _,
        &seq_q as *const u32 as *mut _,
        &seq_kv as *const u32 as *mut _,
        &num_heads as *const u32 as *mut _,
        &scale as *const f32 as *mut _,
    ];

    let result = unsafe {
        cuLaunchKernel(
            cu_func,
            grid_x,
            num_heads,
            batch,
            160,
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
            "cuLaunchKernel failed for FP8 V11 TMA: {:?}",
            result
        )));
    }

    Ok(())
}

/// Create a TMA descriptor for FP8 V in transposed [D, B*H*Skv] GMEM layout.
///
/// V_T[d, t] at byte d*total_kv_rows + t, where d=0..127 and t is the flat token index.
/// TMA loads a [tile_d × tile_tokens] = [128 × 64] tile using coord = {kv_row, 0}.
fn create_tma_desc_fp8_vt(
    global_ptr: *mut core::ffi::c_void,
    total_kv_rows: u32, // B * H * Skv (flat token count, = D-row stride in bytes)
    head_dim: u32,      // D = 128
    tile_tokens: u32,   // 64
    tile_d: u32,        // 128
) -> Result<cudarc::driver::sys::CUtensorMap> {
    use cudarc::driver::sys::*;

    let mut tma = CUtensorMap_st::default();

    // dim[0] = fast (token axis = B*H*Skv), dim[1] = slow (D axis = 128)
    let global_dim: [u64; 2] = [total_kv_rows as u64, head_dim as u64];
    // stride[0] = bytes per D-row = total_kv_rows
    let global_strides: [u64; 1] = [total_kv_rows as u64];
    let box_dim: [u32; 2] = [tile_tokens, tile_d];
    let elem_strides: [u32; 2] = [1, 1];

    let result = unsafe {
        cuTensorMapEncodeTiled(
            &mut tma,
            CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8,
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
            "cuTensorMapEncodeTiled (fp8 vt) failed: {:?}",
            result
        )));
    }

    Ok(tma)
}

/// Flash Attention FP8 V12a: SMEM cooperative transpose of V after TMA load.
///
/// Same GMEM layout as V11 (V in [B,H,Skv,D] row-major). After each TMA V load, all
/// 160 threads cooperatively transpose V[64,128] → V_T[128,64] in SMEM, then use
/// `ld.shared.b32` for PV B-fragments (same as V12c). No API change vs V11.
///
/// Trade-off: adds ~52 iterations/thread of transpose work per KV block vs V11's
/// 7-instruction/fragment `ld.u8+bfi` approach. Break-even depends on memory latency.
///
/// Q, K, V: [batch, num_heads, seq, 128] row-major fp8 e4m3 (u8 raw bits).
/// O: [batch, num_heads, seq_q, 128] row-major bf16 (u16 raw bits, output).
///
/// 5 warps (160 threads) per block, Br=64, Bc=64.
/// ~61.5KB dynamic shared memory.
#[allow(clippy::too_many_arguments)]
#[cfg(feature = "experimental")]
pub fn flash_attn_fp8_v12a_transpose(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u8>,
    k: &CudaSlice<u8>,
    v: &CudaSlice<u8>,
    o: &mut CudaSlice<u16>,
    batch: u32,
    num_heads: u32,
    seq_q: u32,
    seq_kv: u32,
    scale: f32,
) -> Result<()> {
    validate_attn_dims(batch, num_heads, seq_q, seq_kv)?;

    let q_need = batch as usize * num_heads as usize * seq_q as usize * HEAD_DIM as usize;
    let kv_need = batch as usize * num_heads as usize * seq_kv as usize * HEAD_DIM as usize;
    if q.len() < q_need {
        return Err(SparkError::InvalidArgument(format!(
            "Q buffer too small: {} < {q_need}",
            q.len()
        )));
    }
    if k.len() < kv_need {
        return Err(SparkError::InvalidArgument(format!(
            "K buffer too small: {} < {kv_need}",
            k.len()
        )));
    }
    if v.len() < kv_need {
        return Err(SparkError::InvalidArgument(format!(
            "V buffer too small: {} < {kv_need}",
            v.len()
        )));
    }
    if o.len() < q_need {
        return Err(SparkError::InvalidArgument(format!(
            "O buffer too small: {} < {q_need}",
            o.len()
        )));
    }

    use cudarc::driver::sys::*;

    let grid_x = seq_q.div_ceil(64);

    let (q_dptr, _q_sync) = q.device_ptr(stream);
    let (k_dptr, _k_sync) = k.device_ptr(stream);
    let (v_dptr, _v_sync) = v.device_ptr(stream);
    let (o_dptr, _o_sync) = o.device_ptr(stream);

    let total_q_rows = batch * num_heads * seq_q;
    let total_kv_rows = batch * num_heads * seq_kv;

    // Q/K: SWIZZLE_128B. V: SWIZZLE_NONE (standard layout, transpose happens in SMEM).
    let q_tma = create_tma_desc_fp8_swizzle(q_dptr as *mut _, total_q_rows, HEAD_DIM, 64, 128)?;
    let k_tma = create_tma_desc_fp8_swizzle(k_dptr as *mut _, total_kv_rows, HEAD_DIM, 64, 128)?;
    let v_tma = create_tma_desc_fp8(v_dptr as *mut _, total_kv_rows, HEAD_DIM, 64, 128)?;

    let q_tma_u32: [u32; 32] = unsafe { core::mem::transmute(q_tma) };
    let k_tma_u32: [u32; 32] = unsafe { core::mem::transmute(k_tma) };
    let v_tma_u32: [u32; 32] = unsafe { core::mem::transmute(v_tma) };
    let q_tma_dev = stream.memcpy_stod(&q_tma_u32).map_err(SparkError::Driver)?;
    let k_tma_dev = stream.memcpy_stod(&k_tma_u32).map_err(SparkError::Driver)?;
    let v_tma_dev = stream.memcpy_stod(&v_tma_u32).map_err(SparkError::Driver)?;

    let (q_tma_dptr, _) = q_tma_dev.device_ptr(stream);
    let (k_tma_dptr, _) = k_tma_dev.device_ptr(stream);
    let (v_tma_dptr, _) = v_tma_dev.device_ptr(stream);

    let cu_func = module::load_kernel_raw(
        ctx,
        "fa_fp8_v12a_transpose",
        "flash_attn_fp8_v12a_transpose",
    )?;
    let cu_stream = stream.cu_stream();

    const SMEM_TOTAL: i32 = 61552;
    let attr_result = unsafe {
        cuFuncSetAttribute(
            cu_func,
            CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            SMEM_TOTAL,
        )
    };
    if attr_result != CUresult::CUDA_SUCCESS {
        return Err(SparkError::LaunchFailed(format!(
            "cuFuncSetAttribute failed for V12a: {:?}",
            attr_result
        )));
    }

    let params: [*mut core::ffi::c_void; 8] = [
        &q_tma_dptr as *const u64 as *mut _,
        &k_tma_dptr as *const u64 as *mut _,
        &v_tma_dptr as *const u64 as *mut _,
        &o_dptr as *const u64 as *mut _,
        &seq_q as *const u32 as *mut _,
        &seq_kv as *const u32 as *mut _,
        &num_heads as *const u32 as *mut _,
        &scale as *const f32 as *mut _,
    ];

    let result = unsafe {
        cuLaunchKernel(
            cu_func,
            grid_x,
            num_heads,
            batch,
            160,
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
            "cuLaunchKernel failed for FP8 V12a transpose: {:?}",
            result
        )));
    }

    Ok(())
}

/// Flash Attention FP8 V12c: V pre-transposed in GMEM as [D, B*H*Skv].
///
/// V must be passed pre-transposed: original V[B,H,Skv,D] → V_T[D,B,H,Skv] contiguous.
/// In Python: `v_t = v.permute(3,0,1,2).contiguous()`.
///
/// TMA loads a [128D × 64 tokens] tile with a single `cp.async.bulk` using coord={kv_row,0}.
/// PV B-fragments use `ld.shared.b32` (32 total, replacing V11's 64×7=448 instructions).
/// Zero per-block overhead vs V12a's cooperative SMEM transpose.
///
/// Q, K: [batch, num_heads, seq, 128] row-major fp8 e4m3 (u8 raw bits).
/// V_T: [D=128, batch, num_heads, Skv] row-major fp8 e4m3 (u8 raw bits, pre-transposed).
/// O: [batch, num_heads, seq_q, 128] row-major bf16 (u16 raw bits, output).
///
/// 5 warps (160 threads) per block, Br=64, Bc=64.
/// ~45KB dynamic shared memory (same as V11).
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_fp8_v12c_vt(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u8>,
    k: &CudaSlice<u8>,
    v_t: &CudaSlice<u8>, // V pre-transposed: [D=128, B*H*Skv]
    o: &mut CudaSlice<u16>,
    batch: u32,
    num_heads: u32,
    seq_q: u32,
    seq_kv: u32,
    scale: f32,
) -> Result<()> {
    validate_attn_dims(batch, num_heads, seq_q, seq_kv)?;

    let q_need = batch as usize * num_heads as usize * seq_q as usize * HEAD_DIM as usize;
    let kv_need = batch as usize * num_heads as usize * seq_kv as usize * HEAD_DIM as usize;
    if q.len() < q_need {
        return Err(SparkError::InvalidArgument(format!(
            "Q buffer too small: {} < {q_need}",
            q.len()
        )));
    }
    if k.len() < kv_need {
        return Err(SparkError::InvalidArgument(format!(
            "K buffer too small: {} < {kv_need}",
            k.len()
        )));
    }
    if v_t.len() < kv_need {
        return Err(SparkError::InvalidArgument(format!(
            "V_T buffer too small: {} < {kv_need}",
            v_t.len()
        )));
    }
    if o.len() < q_need {
        return Err(SparkError::InvalidArgument(format!(
            "O buffer too small: {} < {q_need}",
            o.len()
        )));
    }

    use cudarc::driver::sys::*;

    let grid_x = seq_q.div_ceil(64);

    let (q_dptr, _q_sync) = q.device_ptr(stream);
    let (k_dptr, _k_sync) = k.device_ptr(stream);
    let (vt_dptr, _vt_sync) = v_t.device_ptr(stream);
    let (o_dptr, _o_sync) = o.device_ptr(stream);

    let total_q_rows = batch * num_heads * seq_q;
    let total_kv_rows = batch * num_heads * seq_kv;

    // Q/K: SWIZZLE_128B. V_T: transposed layout [D, B*H*Skv], SWIZZLE_NONE.
    let q_tma = create_tma_desc_fp8_swizzle(q_dptr as *mut _, total_q_rows, HEAD_DIM, 64, 128)?;
    let k_tma = create_tma_desc_fp8_swizzle(k_dptr as *mut _, total_kv_rows, HEAD_DIM, 64, 128)?;
    // V_T descriptor: global_dim=[B*H*Skv, D=128], box=[64 tokens, 128 D], coord={kv_row, 0}
    let vt_tma = create_tma_desc_fp8_vt(vt_dptr as *mut _, total_kv_rows, HEAD_DIM, 64, 128)?;

    let q_tma_u32: [u32; 32] = unsafe { core::mem::transmute(q_tma) };
    let k_tma_u32: [u32; 32] = unsafe { core::mem::transmute(k_tma) };
    let vt_tma_u32: [u32; 32] = unsafe { core::mem::transmute(vt_tma) };
    let q_tma_dev = stream.memcpy_stod(&q_tma_u32).map_err(SparkError::Driver)?;
    let k_tma_dev = stream.memcpy_stod(&k_tma_u32).map_err(SparkError::Driver)?;
    let vt_tma_dev = stream
        .memcpy_stod(&vt_tma_u32)
        .map_err(SparkError::Driver)?;

    let (q_tma_dptr, _) = q_tma_dev.device_ptr(stream);
    let (k_tma_dptr, _) = k_tma_dev.device_ptr(stream);
    let (vt_tma_dptr, _) = vt_tma_dev.device_ptr(stream);

    let cu_func = module::load_kernel_raw(ctx, "fa_fp8_v12c_vt", "flash_attn_fp8_v12c_vt")?;
    let cu_stream = stream.cu_stream();

    const SMEM_TOTAL: i32 = 45168;
    let attr_result = unsafe {
        cuFuncSetAttribute(
            cu_func,
            CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            SMEM_TOTAL,
        )
    };
    if attr_result != CUresult::CUDA_SUCCESS {
        return Err(SparkError::LaunchFailed(format!(
            "cuFuncSetAttribute failed for V12c: {:?}",
            attr_result
        )));
    }

    let params: [*mut core::ffi::c_void; 8] = [
        &q_tma_dptr as *const u64 as *mut _,
        &k_tma_dptr as *const u64 as *mut _,
        &vt_tma_dptr as *const u64 as *mut _,
        &o_dptr as *const u64 as *mut _,
        &seq_q as *const u32 as *mut _,
        &seq_kv as *const u32 as *mut _,
        &num_heads as *const u32 as *mut _,
        &scale as *const f32 as *mut _,
    ];

    let result = unsafe {
        cuLaunchKernel(
            cu_func,
            grid_x,
            num_heads,
            batch,
            160,
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
            "cuLaunchKernel failed for FP8 V12c vt: {:?}",
            result
        )));
    }

    Ok(())
}

/// Launch FP8 Flash Attention (causal, V_T layout) with d=128.
///
/// Same interface as `flash_attn_fp8_v12c_vt` but applies causal masking:
/// S[i,j] = -inf when kv_pos_j > q_pos_i. Uses effective_kv_blocks =
/// min(num_kv_blocks, q_block+1) to skip all-future KV blocks entirely.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_fp8_v12c_vt_causal(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u8>,
    k: &CudaSlice<u8>,
    v_t: &CudaSlice<u8>, // V pre-transposed: [D=128, B*H*Skv]
    o: &mut CudaSlice<u16>,
    batch: u32,
    num_heads: u32,
    seq_q: u32,
    seq_kv: u32,
    scale: f32,
) -> Result<()> {
    validate_attn_dims(batch, num_heads, seq_q, seq_kv)?;

    let q_need = batch as usize * num_heads as usize * seq_q as usize * HEAD_DIM as usize;
    let kv_need = batch as usize * num_heads as usize * seq_kv as usize * HEAD_DIM as usize;
    if q.len() < q_need {
        return Err(SparkError::InvalidArgument(format!(
            "Q buffer too small: {} < {q_need}",
            q.len()
        )));
    }
    if k.len() < kv_need {
        return Err(SparkError::InvalidArgument(format!(
            "K buffer too small: {} < {kv_need}",
            k.len()
        )));
    }
    if v_t.len() < kv_need {
        return Err(SparkError::InvalidArgument(format!(
            "V_T buffer too small: {} < {kv_need}",
            v_t.len()
        )));
    }
    if o.len() < q_need {
        return Err(SparkError::InvalidArgument(format!(
            "O buffer too small: {} < {q_need}",
            o.len()
        )));
    }

    use cudarc::driver::sys::*;

    let grid_x = seq_q.div_ceil(64);

    let (q_dptr, _q_sync) = q.device_ptr(stream);
    let (k_dptr, _k_sync) = k.device_ptr(stream);
    let (vt_dptr, _vt_sync) = v_t.device_ptr(stream);
    let (o_dptr, _o_sync) = o.device_ptr(stream);

    let total_q_rows = batch * num_heads * seq_q;
    let total_kv_rows = batch * num_heads * seq_kv;

    let q_tma = create_tma_desc_fp8_swizzle(q_dptr as *mut _, total_q_rows, HEAD_DIM, 64, 128)?;
    let k_tma = create_tma_desc_fp8_swizzle(k_dptr as *mut _, total_kv_rows, HEAD_DIM, 64, 128)?;
    let vt_tma = create_tma_desc_fp8_vt(vt_dptr as *mut _, total_kv_rows, HEAD_DIM, 64, 128)?;

    let q_tma_u32: [u32; 32] = unsafe { core::mem::transmute(q_tma) };
    let k_tma_u32: [u32; 32] = unsafe { core::mem::transmute(k_tma) };
    let vt_tma_u32: [u32; 32] = unsafe { core::mem::transmute(vt_tma) };
    let q_tma_dev = stream.memcpy_stod(&q_tma_u32).map_err(SparkError::Driver)?;
    let k_tma_dev = stream.memcpy_stod(&k_tma_u32).map_err(SparkError::Driver)?;
    let vt_tma_dev = stream
        .memcpy_stod(&vt_tma_u32)
        .map_err(SparkError::Driver)?;

    let (q_tma_dptr, _) = q_tma_dev.device_ptr(stream);
    let (k_tma_dptr, _) = k_tma_dev.device_ptr(stream);
    let (vt_tma_dptr, _) = vt_tma_dev.device_ptr(stream);

    let cu_func = module::load_kernel_raw(
        ctx,
        "fa_fp8_v12c_vt_causal",
        "flash_attn_fp8_v12c_vt_causal",
    )?;
    let cu_stream = stream.cu_stream();

    const SMEM_TOTAL: i32 = 45168;
    let attr_result = unsafe {
        cuFuncSetAttribute(
            cu_func,
            CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            SMEM_TOTAL,
        )
    };
    if attr_result != CUresult::CUDA_SUCCESS {
        return Err(SparkError::LaunchFailed(format!(
            "cuFuncSetAttribute failed for FP8 V12c causal: {:?}",
            attr_result
        )));
    }

    let params: [*mut core::ffi::c_void; 8] = [
        &q_tma_dptr as *const u64 as *mut _,
        &k_tma_dptr as *const u64 as *mut _,
        &vt_tma_dptr as *const u64 as *mut _,
        &o_dptr as *const u64 as *mut _,
        &seq_q as *const u32 as *mut _,
        &seq_kv as *const u32 as *mut _,
        &num_heads as *const u32 as *mut _,
        &scale as *const f32 as *mut _,
    ];

    let result = unsafe {
        cuLaunchKernel(
            cu_func,
            grid_x,
            num_heads,
            batch,
            160,
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
            "cuLaunchKernel failed for FP8 V12c causal: {:?}",
            result
        )));
    }

    Ok(())
}

/// Launch FP8 Flash Attention (varlen non-causal, V_T layout) with d=128.
///
/// Variable-length non-causal attention with head-major layout and pre-transposed V:
///   Q, K: `[num_heads, total_tokens, D]` row-major FP8 (head-major varlen)
///   V_T: `[D, num_heads * total_kv]` pre-transposed FP8
///   O: `[num_heads, total_q, D]` row-major BF16
///
/// TMA descriptors are created over `[num_heads * total_tokens, D]` (flat 2D view).
///
/// - `cu_seqlens_q`: GPU buffer `[batch+1]` u32
/// - `cu_seqlens_k`: GPU buffer `[batch+1]` u32
/// - `total_q`: total Q tokens
/// - `total_kv`: total KV tokens
/// - `max_seqlen_q`: max Q sequence length (used for grid size)
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_fp8_v12c_vt_varlen(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u8>,
    k: &CudaSlice<u8>,
    v_t: &CudaSlice<u8>, // V pre-transposed: [D=128, num_heads * total_kv]
    o: &mut CudaSlice<u16>,
    cu_seqlens_q: &CudaSlice<u32>,
    cu_seqlens_k: &CudaSlice<u32>,
    batch: u32,
    num_heads: u32,
    max_seqlen_q: u32,
    total_q: u32,
    total_kv: u32,
    scale: f32,
) -> Result<()> {
    if batch == 0 {
        return Err(SparkError::InvalidArgument("batch must be > 0".into()));
    }
    if num_heads == 0 {
        return Err(SparkError::InvalidArgument("num_heads must be > 0".into()));
    }
    if max_seqlen_q == 0 || total_q == 0 || total_kv == 0 {
        return Err(SparkError::InvalidArgument(
            "max_seqlen_q, total_q, total_kv must be > 0".into(),
        ));
    }
    let cu_need = (batch + 1) as usize;
    if cu_seqlens_q.len() < cu_need || cu_seqlens_k.len() < cu_need {
        return Err(SparkError::InvalidArgument(format!(
            "cu_seqlens too small: need {cu_need}"
        )));
    }

    use cudarc::driver::sys::*;

    let grid_x = max_seqlen_q.div_ceil(64);

    let (q_dptr, _q_sync) = q.device_ptr(stream);
    let (k_dptr, _k_sync) = k.device_ptr(stream);
    let (vt_dptr, _vt_sync) = v_t.device_ptr(stream);
    let (o_dptr, _o_sync) = o.device_ptr(stream);
    let (cu_q_dptr, _cu_q_sync) = cu_seqlens_q.device_ptr(stream);
    let (cu_k_dptr, _cu_k_sync) = cu_seqlens_k.device_ptr(stream);

    let q_tma =
        create_tma_desc_fp8_swizzle(q_dptr as *mut _, num_heads * total_q, HEAD_DIM, 64, 128)?;
    let k_tma =
        create_tma_desc_fp8_swizzle(k_dptr as *mut _, num_heads * total_kv, HEAD_DIM, 64, 128)?;
    let vt_tma =
        create_tma_desc_fp8_vt(vt_dptr as *mut _, num_heads * total_kv, HEAD_DIM, 64, 128)?;

    let q_tma_u32: [u32; 32] = unsafe { core::mem::transmute(q_tma) };
    let k_tma_u32: [u32; 32] = unsafe { core::mem::transmute(k_tma) };
    let vt_tma_u32: [u32; 32] = unsafe { core::mem::transmute(vt_tma) };
    let q_tma_dev = stream.memcpy_stod(&q_tma_u32).map_err(SparkError::Driver)?;
    let k_tma_dev = stream.memcpy_stod(&k_tma_u32).map_err(SparkError::Driver)?;
    let vt_tma_dev = stream
        .memcpy_stod(&vt_tma_u32)
        .map_err(SparkError::Driver)?;

    let (q_tma_dptr, _) = q_tma_dev.device_ptr(stream);
    let (k_tma_dptr, _) = k_tma_dev.device_ptr(stream);
    let (vt_tma_dptr, _) = vt_tma_dev.device_ptr(stream);

    let cu_func = module::load_kernel_raw(
        ctx,
        "fa_fp8_v12c_vt_varlen",
        "flash_attn_fp8_v12c_vt_varlen",
    )?;
    let cu_stream = stream.cu_stream();

    const SMEM_TOTAL: i32 = 45168;
    let attr_result = unsafe {
        cuFuncSetAttribute(
            cu_func,
            CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            SMEM_TOTAL,
        )
    };
    if attr_result != CUresult::CUDA_SUCCESS {
        return Err(SparkError::LaunchFailed(format!(
            "cuFuncSetAttribute failed for FP8 V12c VT varlen: {:?}",
            attr_result
        )));
    }

    let params: [*mut core::ffi::c_void; 9] = [
        &q_tma_dptr as *const u64 as *mut _,
        &k_tma_dptr as *const u64 as *mut _,
        &vt_tma_dptr as *const u64 as *mut _,
        &o_dptr as *const u64 as *mut _,
        &cu_q_dptr as *const u64 as *mut _,
        &cu_k_dptr as *const u64 as *mut _,
        &total_q as *const u32 as *mut _,
        &total_kv as *const u32 as *mut _,
        &scale as *const f32 as *mut _,
    ];

    let result = unsafe {
        cuLaunchKernel(
            cu_func,
            grid_x,
            num_heads,
            batch,
            160,
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
            "cuLaunchKernel failed for FP8 V12c VT varlen: {:?}",
            result
        )));
    }

    Ok(())
}

/// Launch FP8 Flash Attention (varlen causal, V_T layout) with d=128.
///
/// Variable-length causal attention with head-major layout and pre-transposed V:
///   Q, K: `[num_heads, total_tokens, D]` row-major FP8 (head-major varlen)
///   V_T: `[D, num_heads * total_kv]` pre-transposed FP8
///   O: `[num_heads, total_q, D]` row-major BF16
///
/// TMA descriptors are created over `[num_heads * total_tokens, D]` (flat 2D view).
///
/// - `cu_seqlens_q`: GPU buffer `[batch+1]` u32
/// - `cu_seqlens_k`: GPU buffer `[batch+1]` u32
/// - `total_q`: total Q tokens
/// - `total_kv`: total KV tokens
/// - `max_seqlen_q`: max Q sequence length (used for grid size)
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_fp8_v12c_vt_varlen_causal(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u8>,
    k: &CudaSlice<u8>,
    v_t: &CudaSlice<u8>, // V pre-transposed: [D=128, num_heads * total_kv]
    o: &mut CudaSlice<u16>,
    cu_seqlens_q: &CudaSlice<u32>,
    cu_seqlens_k: &CudaSlice<u32>,
    batch: u32,
    num_heads: u32,
    max_seqlen_q: u32,
    total_q: u32,
    total_kv: u32,
    scale: f32,
) -> Result<()> {
    if batch == 0 {
        return Err(SparkError::InvalidArgument("batch must be > 0".into()));
    }
    if num_heads == 0 {
        return Err(SparkError::InvalidArgument("num_heads must be > 0".into()));
    }
    if max_seqlen_q == 0 || total_q == 0 || total_kv == 0 {
        return Err(SparkError::InvalidArgument(
            "max_seqlen_q, total_q, total_kv must be > 0".into(),
        ));
    }
    let cu_need = (batch + 1) as usize;
    if cu_seqlens_q.len() < cu_need || cu_seqlens_k.len() < cu_need {
        return Err(SparkError::InvalidArgument(format!(
            "cu_seqlens too small: need {cu_need}"
        )));
    }

    use cudarc::driver::sys::*;

    let grid_x = max_seqlen_q.div_ceil(64);

    let (q_dptr, _q_sync) = q.device_ptr(stream);
    let (k_dptr, _k_sync) = k.device_ptr(stream);
    let (vt_dptr, _vt_sync) = v_t.device_ptr(stream);
    let (o_dptr, _o_sync) = o.device_ptr(stream);
    let (cu_q_dptr, _cu_q_sync) = cu_seqlens_q.device_ptr(stream);
    let (cu_k_dptr, _cu_k_sync) = cu_seqlens_k.device_ptr(stream);

    let q_tma =
        create_tma_desc_fp8_swizzle(q_dptr as *mut _, num_heads * total_q, HEAD_DIM, 64, 128)?;
    let k_tma =
        create_tma_desc_fp8_swizzle(k_dptr as *mut _, num_heads * total_kv, HEAD_DIM, 64, 128)?;
    let vt_tma =
        create_tma_desc_fp8_vt(vt_dptr as *mut _, num_heads * total_kv, HEAD_DIM, 64, 128)?;

    let q_tma_u32: [u32; 32] = unsafe { core::mem::transmute(q_tma) };
    let k_tma_u32: [u32; 32] = unsafe { core::mem::transmute(k_tma) };
    let vt_tma_u32: [u32; 32] = unsafe { core::mem::transmute(vt_tma) };
    let q_tma_dev = stream.memcpy_stod(&q_tma_u32).map_err(SparkError::Driver)?;
    let k_tma_dev = stream.memcpy_stod(&k_tma_u32).map_err(SparkError::Driver)?;
    let vt_tma_dev = stream
        .memcpy_stod(&vt_tma_u32)
        .map_err(SparkError::Driver)?;

    let (q_tma_dptr, _) = q_tma_dev.device_ptr(stream);
    let (k_tma_dptr, _) = k_tma_dev.device_ptr(stream);
    let (vt_tma_dptr, _) = vt_tma_dev.device_ptr(stream);

    let cu_func = module::load_kernel_raw(
        ctx,
        "fa_fp8_v12c_vt_varlen_causal",
        "flash_attn_fp8_v12c_vt_varlen_causal",
    )?;
    let cu_stream = stream.cu_stream();

    const SMEM_TOTAL: i32 = 45168;
    let attr_result = unsafe {
        cuFuncSetAttribute(
            cu_func,
            CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            SMEM_TOTAL,
        )
    };
    if attr_result != CUresult::CUDA_SUCCESS {
        return Err(SparkError::LaunchFailed(format!(
            "cuFuncSetAttribute failed for FP8 V12c VT varlen causal: {:?}",
            attr_result
        )));
    }

    let params: [*mut core::ffi::c_void; 9] = [
        &q_tma_dptr as *const u64 as *mut _,
        &k_tma_dptr as *const u64 as *mut _,
        &vt_tma_dptr as *const u64 as *mut _,
        &o_dptr as *const u64 as *mut _,
        &cu_q_dptr as *const u64 as *mut _,
        &cu_k_dptr as *const u64 as *mut _,
        &total_q as *const u32 as *mut _,
        &total_kv as *const u32 as *mut _,
        &scale as *const f32 as *mut _,
    ];

    let result = unsafe {
        cuLaunchKernel(
            cu_func,
            grid_x,
            num_heads,
            batch,
            160,
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
            "cuLaunchKernel failed for FP8 V12c VT varlen causal: {:?}",
            result
        )));
    }

    Ok(())
}

/// Launch FP8 Flash Attention V12c VT GQA (non-causal) with d=128.
///
/// Grouped Query Attention: Q has `num_heads_q` heads, KV have `num_heads_kv` heads.
/// Requires `num_heads_q % num_heads_kv == 0`.
///
/// Q, O: [batch, num_heads_q, seq_q, 128] row-major FP8/BF16
/// K:    [batch, num_heads_kv, seq_kv, 128] row-major FP8
/// V_T:  [D=128, batch * num_heads_kv * seq_kv] pre-transposed FP8
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_fp8_v12c_vt_gqa(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u8>,
    k: &CudaSlice<u8>,
    v_t: &CudaSlice<u8>,
    o: &mut CudaSlice<u16>,
    batch: u32,
    num_heads_q: u32,
    num_heads_kv: u32,
    seq_q: u32,
    seq_kv: u32,
    scale: f32,
) -> Result<()> {
    if batch == 0 || num_heads_q == 0 || num_heads_kv == 0 || seq_q == 0 || seq_kv == 0 {
        return Err(SparkError::InvalidArgument(
            "batch, num_heads_q, num_heads_kv, seq_q, seq_kv must be > 0".into(),
        ));
    }
    if !num_heads_q.is_multiple_of(num_heads_kv) {
        return Err(SparkError::InvalidArgument(format!(
            "num_heads_q ({num_heads_q}) must be divisible by num_heads_kv ({num_heads_kv})"
        )));
    }

    use cudarc::driver::sys::*;

    let grid_x = seq_q.div_ceil(64);

    let (q_dptr, _q_sync) = q.device_ptr(stream);
    let (k_dptr, _k_sync) = k.device_ptr(stream);
    let (vt_dptr, _vt_sync) = v_t.device_ptr(stream);
    let (o_dptr, _o_sync) = o.device_ptr(stream);

    let total_q_rows = batch * num_heads_q * seq_q;
    let total_kv_rows = batch * num_heads_kv * seq_kv;

    let q_tma = create_tma_desc_fp8_swizzle(q_dptr as *mut _, total_q_rows, HEAD_DIM, 64, 128)?;
    let k_tma = create_tma_desc_fp8_swizzle(k_dptr as *mut _, total_kv_rows, HEAD_DIM, 64, 128)?;
    let vt_tma = create_tma_desc_fp8_vt(vt_dptr as *mut _, total_kv_rows, HEAD_DIM, 64, 128)?;

    let q_tma_u32: [u32; 32] = unsafe { core::mem::transmute(q_tma) };
    let k_tma_u32: [u32; 32] = unsafe { core::mem::transmute(k_tma) };
    let vt_tma_u32: [u32; 32] = unsafe { core::mem::transmute(vt_tma) };
    let q_tma_dev = stream.memcpy_stod(&q_tma_u32).map_err(SparkError::Driver)?;
    let k_tma_dev = stream.memcpy_stod(&k_tma_u32).map_err(SparkError::Driver)?;
    let vt_tma_dev = stream
        .memcpy_stod(&vt_tma_u32)
        .map_err(SparkError::Driver)?;

    let (q_tma_dptr, _) = q_tma_dev.device_ptr(stream);
    let (k_tma_dptr, _) = k_tma_dev.device_ptr(stream);
    let (vt_tma_dptr, _) = vt_tma_dev.device_ptr(stream);

    let cu_func = module::load_kernel_raw(ctx, "fa_fp8_v12c_vt_gqa", "flash_attn_fp8_v12c_vt_gqa")?;
    let cu_stream = stream.cu_stream();

    const SMEM_TOTAL: i32 = 45168;
    let attr_result = unsafe {
        cuFuncSetAttribute(
            cu_func,
            CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            SMEM_TOTAL,
        )
    };
    if attr_result != CUresult::CUDA_SUCCESS {
        return Err(SparkError::LaunchFailed(format!(
            "cuFuncSetAttribute failed for FP8 V12c VT GQA: {:?}",
            attr_result
        )));
    }

    let params: [*mut core::ffi::c_void; 9] = [
        &q_tma_dptr as *const u64 as *mut _,
        &k_tma_dptr as *const u64 as *mut _,
        &vt_tma_dptr as *const u64 as *mut _,
        &o_dptr as *const u64 as *mut _,
        &seq_q as *const u32 as *mut _,
        &seq_kv as *const u32 as *mut _,
        &num_heads_q as *const u32 as *mut _,
        &num_heads_kv as *const u32 as *mut _,
        &scale as *const f32 as *mut _,
    ];

    let result = unsafe {
        cuLaunchKernel(
            cu_func,
            grid_x,
            num_heads_q,
            batch,
            160,
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
            "cuLaunchKernel failed for FP8 V12c VT GQA: {:?}",
            result
        )));
    }

    Ok(())
}

/// Launch FP8 Flash Attention V12c VT GQA causal with d=128.
///
/// Same as `flash_attn_fp8_v12c_vt_gqa` but with causal masking.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_fp8_v12c_vt_gqa_causal(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u8>,
    k: &CudaSlice<u8>,
    v_t: &CudaSlice<u8>,
    o: &mut CudaSlice<u16>,
    batch: u32,
    num_heads_q: u32,
    num_heads_kv: u32,
    seq_q: u32,
    seq_kv: u32,
    scale: f32,
) -> Result<()> {
    if batch == 0 || num_heads_q == 0 || num_heads_kv == 0 || seq_q == 0 || seq_kv == 0 {
        return Err(SparkError::InvalidArgument(
            "batch, num_heads_q, num_heads_kv, seq_q, seq_kv must be > 0".into(),
        ));
    }
    if !num_heads_q.is_multiple_of(num_heads_kv) {
        return Err(SparkError::InvalidArgument(format!(
            "num_heads_q ({num_heads_q}) must be divisible by num_heads_kv ({num_heads_kv})"
        )));
    }

    use cudarc::driver::sys::*;

    let grid_x = seq_q.div_ceil(64);

    let (q_dptr, _q_sync) = q.device_ptr(stream);
    let (k_dptr, _k_sync) = k.device_ptr(stream);
    let (vt_dptr, _vt_sync) = v_t.device_ptr(stream);
    let (o_dptr, _o_sync) = o.device_ptr(stream);

    let total_q_rows = batch * num_heads_q * seq_q;
    let total_kv_rows = batch * num_heads_kv * seq_kv;

    let q_tma = create_tma_desc_fp8_swizzle(q_dptr as *mut _, total_q_rows, HEAD_DIM, 64, 128)?;
    let k_tma = create_tma_desc_fp8_swizzle(k_dptr as *mut _, total_kv_rows, HEAD_DIM, 64, 128)?;
    let vt_tma = create_tma_desc_fp8_vt(vt_dptr as *mut _, total_kv_rows, HEAD_DIM, 64, 128)?;

    let q_tma_u32: [u32; 32] = unsafe { core::mem::transmute(q_tma) };
    let k_tma_u32: [u32; 32] = unsafe { core::mem::transmute(k_tma) };
    let vt_tma_u32: [u32; 32] = unsafe { core::mem::transmute(vt_tma) };
    let q_tma_dev = stream.memcpy_stod(&q_tma_u32).map_err(SparkError::Driver)?;
    let k_tma_dev = stream.memcpy_stod(&k_tma_u32).map_err(SparkError::Driver)?;
    let vt_tma_dev = stream
        .memcpy_stod(&vt_tma_u32)
        .map_err(SparkError::Driver)?;

    let (q_tma_dptr, _) = q_tma_dev.device_ptr(stream);
    let (k_tma_dptr, _) = k_tma_dev.device_ptr(stream);
    let (vt_tma_dptr, _) = vt_tma_dev.device_ptr(stream);

    let cu_func = module::load_kernel_raw(
        ctx,
        "fa_fp8_v12c_vt_gqa_causal",
        "flash_attn_fp8_v12c_vt_gqa_causal",
    )?;
    let cu_stream = stream.cu_stream();

    const SMEM_TOTAL: i32 = 45168;
    let attr_result = unsafe {
        cuFuncSetAttribute(
            cu_func,
            CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            SMEM_TOTAL,
        )
    };
    if attr_result != CUresult::CUDA_SUCCESS {
        return Err(SparkError::LaunchFailed(format!(
            "cuFuncSetAttribute failed for FP8 V12c VT GQA causal: {:?}",
            attr_result
        )));
    }

    let params: [*mut core::ffi::c_void; 9] = [
        &q_tma_dptr as *const u64 as *mut _,
        &k_tma_dptr as *const u64 as *mut _,
        &vt_tma_dptr as *const u64 as *mut _,
        &o_dptr as *const u64 as *mut _,
        &seq_q as *const u32 as *mut _,
        &seq_kv as *const u32 as *mut _,
        &num_heads_q as *const u32 as *mut _,
        &num_heads_kv as *const u32 as *mut _,
        &scale as *const f32 as *mut _,
    ];

    let result = unsafe {
        cuLaunchKernel(
            cu_func,
            grid_x,
            num_heads_q,
            batch,
            160,
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
            "cuLaunchKernel failed for FP8 V12c VT GQA causal: {:?}",
            result
        )));
    }

    Ok(())
}

/// Launch FP8 Flash Attention V12c VT varlen GQA (non-causal) with d=128.
///
/// Variable-length GQA with head-major layout and pre-transposed V:
///   Q:   [num_heads_q, total_q, D] row-major FP8
///   K:   [num_heads_kv, total_kv, D] row-major FP8
///   V_T: [D, num_heads_kv * total_kv] pre-transposed FP8
///   O:   [num_heads_q, total_q, D] row-major BF16
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_fp8_v12c_vt_varlen_gqa(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u8>,
    k: &CudaSlice<u8>,
    v_t: &CudaSlice<u8>,
    o: &mut CudaSlice<u16>,
    cu_seqlens_q: &CudaSlice<u32>,
    cu_seqlens_k: &CudaSlice<u32>,
    batch: u32,
    num_heads_q: u32,
    num_heads_kv: u32,
    max_seqlen_q: u32,
    total_q: u32,
    total_kv: u32,
    scale: f32,
) -> Result<()> {
    if batch == 0 {
        return Err(SparkError::InvalidArgument("batch must be > 0".into()));
    }
    if num_heads_q == 0 || num_heads_kv == 0 {
        return Err(SparkError::InvalidArgument(
            "num_heads_q and num_heads_kv must be > 0".into(),
        ));
    }
    if !num_heads_q.is_multiple_of(num_heads_kv) {
        return Err(SparkError::InvalidArgument(format!(
            "num_heads_q ({num_heads_q}) must be divisible by num_heads_kv ({num_heads_kv})"
        )));
    }
    if max_seqlen_q == 0 || total_q == 0 || total_kv == 0 {
        return Err(SparkError::InvalidArgument(
            "max_seqlen_q, total_q, total_kv must be > 0".into(),
        ));
    }
    let cu_need = (batch + 1) as usize;
    if cu_seqlens_q.len() < cu_need || cu_seqlens_k.len() < cu_need {
        return Err(SparkError::InvalidArgument(format!(
            "cu_seqlens too small: need {cu_need}"
        )));
    }

    use cudarc::driver::sys::*;

    let grid_x = max_seqlen_q.div_ceil(64);

    let (q_dptr, _q_sync) = q.device_ptr(stream);
    let (k_dptr, _k_sync) = k.device_ptr(stream);
    let (vt_dptr, _vt_sync) = v_t.device_ptr(stream);
    let (o_dptr, _o_sync) = o.device_ptr(stream);
    let (cu_q_dptr, _cu_q_sync) = cu_seqlens_q.device_ptr(stream);
    let (cu_k_dptr, _cu_k_sync) = cu_seqlens_k.device_ptr(stream);

    let q_tma =
        create_tma_desc_fp8_swizzle(q_dptr as *mut _, num_heads_q * total_q, HEAD_DIM, 64, 128)?;
    let k_tma =
        create_tma_desc_fp8_swizzle(k_dptr as *mut _, num_heads_kv * total_kv, HEAD_DIM, 64, 128)?;
    let vt_tma = create_tma_desc_fp8_vt(
        vt_dptr as *mut _,
        num_heads_kv * total_kv,
        HEAD_DIM,
        64,
        128,
    )?;

    let q_tma_u32: [u32; 32] = unsafe { core::mem::transmute(q_tma) };
    let k_tma_u32: [u32; 32] = unsafe { core::mem::transmute(k_tma) };
    let vt_tma_u32: [u32; 32] = unsafe { core::mem::transmute(vt_tma) };
    let q_tma_dev = stream.memcpy_stod(&q_tma_u32).map_err(SparkError::Driver)?;
    let k_tma_dev = stream.memcpy_stod(&k_tma_u32).map_err(SparkError::Driver)?;
    let vt_tma_dev = stream
        .memcpy_stod(&vt_tma_u32)
        .map_err(SparkError::Driver)?;

    let (q_tma_dptr, _) = q_tma_dev.device_ptr(stream);
    let (k_tma_dptr, _) = k_tma_dev.device_ptr(stream);
    let (vt_tma_dptr, _) = vt_tma_dev.device_ptr(stream);

    let cu_func = module::load_kernel_raw(
        ctx,
        "fa_fp8_v12c_vt_varlen_gqa",
        "flash_attn_fp8_v12c_vt_varlen_gqa",
    )?;
    let cu_stream = stream.cu_stream();

    const SMEM_TOTAL: i32 = 45168;
    let attr_result = unsafe {
        cuFuncSetAttribute(
            cu_func,
            CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            SMEM_TOTAL,
        )
    };
    if attr_result != CUresult::CUDA_SUCCESS {
        return Err(SparkError::LaunchFailed(format!(
            "cuFuncSetAttribute failed for FP8 V12c VT varlen GQA: {:?}",
            attr_result
        )));
    }

    let params: [*mut core::ffi::c_void; 11] = [
        &q_tma_dptr as *const u64 as *mut _,
        &k_tma_dptr as *const u64 as *mut _,
        &vt_tma_dptr as *const u64 as *mut _,
        &o_dptr as *const u64 as *mut _,
        &cu_q_dptr as *const u64 as *mut _,
        &cu_k_dptr as *const u64 as *mut _,
        &total_q as *const u32 as *mut _,
        &total_kv as *const u32 as *mut _,
        &num_heads_q as *const u32 as *mut _,
        &num_heads_kv as *const u32 as *mut _,
        &scale as *const f32 as *mut _,
    ];

    let result = unsafe {
        cuLaunchKernel(
            cu_func,
            grid_x,
            num_heads_q,
            batch,
            160,
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
            "cuLaunchKernel failed for FP8 V12c VT varlen GQA: {:?}",
            result
        )));
    }

    Ok(())
}

/// Launch FP8 Flash Attention V12c VT varlen GQA causal with d=128.
///
/// Same as `flash_attn_fp8_v12c_vt_varlen_gqa` but with causal masking.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_fp8_v12c_vt_varlen_gqa_causal(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u8>,
    k: &CudaSlice<u8>,
    v_t: &CudaSlice<u8>,
    o: &mut CudaSlice<u16>,
    cu_seqlens_q: &CudaSlice<u32>,
    cu_seqlens_k: &CudaSlice<u32>,
    batch: u32,
    num_heads_q: u32,
    num_heads_kv: u32,
    max_seqlen_q: u32,
    total_q: u32,
    total_kv: u32,
    scale: f32,
) -> Result<()> {
    if batch == 0 {
        return Err(SparkError::InvalidArgument("batch must be > 0".into()));
    }
    if num_heads_q == 0 || num_heads_kv == 0 {
        return Err(SparkError::InvalidArgument(
            "num_heads_q and num_heads_kv must be > 0".into(),
        ));
    }
    if !num_heads_q.is_multiple_of(num_heads_kv) {
        return Err(SparkError::InvalidArgument(format!(
            "num_heads_q ({num_heads_q}) must be divisible by num_heads_kv ({num_heads_kv})"
        )));
    }
    if max_seqlen_q == 0 || total_q == 0 || total_kv == 0 {
        return Err(SparkError::InvalidArgument(
            "max_seqlen_q, total_q, total_kv must be > 0".into(),
        ));
    }
    let cu_need = (batch + 1) as usize;
    if cu_seqlens_q.len() < cu_need || cu_seqlens_k.len() < cu_need {
        return Err(SparkError::InvalidArgument(format!(
            "cu_seqlens too small: need {cu_need}"
        )));
    }

    use cudarc::driver::sys::*;

    let grid_x = max_seqlen_q.div_ceil(64);

    let (q_dptr, _q_sync) = q.device_ptr(stream);
    let (k_dptr, _k_sync) = k.device_ptr(stream);
    let (vt_dptr, _vt_sync) = v_t.device_ptr(stream);
    let (o_dptr, _o_sync) = o.device_ptr(stream);
    let (cu_q_dptr, _cu_q_sync) = cu_seqlens_q.device_ptr(stream);
    let (cu_k_dptr, _cu_k_sync) = cu_seqlens_k.device_ptr(stream);

    let q_tma =
        create_tma_desc_fp8_swizzle(q_dptr as *mut _, num_heads_q * total_q, HEAD_DIM, 64, 128)?;
    let k_tma =
        create_tma_desc_fp8_swizzle(k_dptr as *mut _, num_heads_kv * total_kv, HEAD_DIM, 64, 128)?;
    let vt_tma = create_tma_desc_fp8_vt(
        vt_dptr as *mut _,
        num_heads_kv * total_kv,
        HEAD_DIM,
        64,
        128,
    )?;

    let q_tma_u32: [u32; 32] = unsafe { core::mem::transmute(q_tma) };
    let k_tma_u32: [u32; 32] = unsafe { core::mem::transmute(k_tma) };
    let vt_tma_u32: [u32; 32] = unsafe { core::mem::transmute(vt_tma) };
    let q_tma_dev = stream.memcpy_stod(&q_tma_u32).map_err(SparkError::Driver)?;
    let k_tma_dev = stream.memcpy_stod(&k_tma_u32).map_err(SparkError::Driver)?;
    let vt_tma_dev = stream
        .memcpy_stod(&vt_tma_u32)
        .map_err(SparkError::Driver)?;

    let (q_tma_dptr, _) = q_tma_dev.device_ptr(stream);
    let (k_tma_dptr, _) = k_tma_dev.device_ptr(stream);
    let (vt_tma_dptr, _) = vt_tma_dev.device_ptr(stream);

    let cu_func = module::load_kernel_raw(
        ctx,
        "fa_fp8_v12c_vt_varlen_gqa_causal",
        "flash_attn_fp8_v12c_vt_varlen_gqa_causal",
    )?;
    let cu_stream = stream.cu_stream();

    const SMEM_TOTAL: i32 = 45168;
    let attr_result = unsafe {
        cuFuncSetAttribute(
            cu_func,
            CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            SMEM_TOTAL,
        )
    };
    if attr_result != CUresult::CUDA_SUCCESS {
        return Err(SparkError::LaunchFailed(format!(
            "cuFuncSetAttribute failed for FP8 V12c VT varlen GQA causal: {:?}",
            attr_result
        )));
    }

    let params: [*mut core::ffi::c_void; 11] = [
        &q_tma_dptr as *const u64 as *mut _,
        &k_tma_dptr as *const u64 as *mut _,
        &vt_tma_dptr as *const u64 as *mut _,
        &o_dptr as *const u64 as *mut _,
        &cu_q_dptr as *const u64 as *mut _,
        &cu_k_dptr as *const u64 as *mut _,
        &total_q as *const u32 as *mut _,
        &total_kv as *const u32 as *mut _,
        &num_heads_q as *const u32 as *mut _,
        &num_heads_kv as *const u32 as *mut _,
        &scale as *const f32 as *mut _,
    ];

    let result = unsafe {
        cuLaunchKernel(
            cu_func,
            grid_x,
            num_heads_q,
            batch,
            160,
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
            "cuLaunchKernel failed for FP8 V12c VT varlen GQA causal: {:?}",
            result
        )));
    }

    Ok(())
}

/// Launch BF16 Flash Attention (causal) with d=128.
///
/// Same interface as `flash_attn_bf16_d128` but applies causal masking:
/// S[i,j] = -inf when kv_pos_j > q_pos_i.
#[allow(clippy::too_many_arguments)]
#[cfg(feature = "experimental")]
pub fn flash_attn_bf16_d128_causal(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u16>,
    k: &CudaSlice<u16>,
    v: &CudaSlice<u16>,
    o: &mut CudaSlice<u16>,
    batch: u32,
    num_heads: u32,
    seq_q: u32,
    seq_kv: u32,
    scale: f32,
) -> Result<()> {
    validate_attn_dims(batch, num_heads, seq_q, seq_kv)?;
    validate_attn_bf16_bufs(q, k, v, o, batch, num_heads, seq_q, seq_kv)?;
    let func = module::load_kernel(ctx, "fa_bf16_d128_causal", "flash_attn_bf16_d128_causal")?;

    let grid_x = seq_q.div_ceil(16);
    let cfg = LaunchConfig {
        grid_dim: (grid_x, num_heads, batch),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 18432,
    };

    unsafe {
        stream
            .launch_builder(&func)
            .arg(q)
            .arg(k)
            .arg(v)
            .arg(o)
            .arg(&seq_q)
            .arg(&seq_kv)
            .arg(&num_heads)
            .arg(&scale)
            .launch(cfg)?;
    }

    Ok(())
}

///
/// V3 architecture: 8 warps (256 threads), all threads load cooperatively.
/// No warp specialization. Q/K/V use raw pointers (no TMA).
/// K/V in paged layout: [num_pages, page_size, num_kv_heads, D] bf16.
/// Page table: [B, max_pages] u32.
///
/// Br=128, Bc=64, D=128.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_bf16_v3_paged_kv(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u16>,
    k: &CudaSlice<u16>,
    v: &CudaSlice<u16>,
    o: &mut CudaSlice<u16>,
    page_table: &CudaSlice<u32>,
    batch: u32,
    num_heads: u32,
    num_kv_heads: u32,
    seq_q: u32,
    seq_kv: u32,
    page_size: u32,
    max_pages: u32,
    scale: f32,
) -> Result<()> {
    if batch == 0 || num_heads == 0 || seq_q == 0 || seq_kv == 0 {
        return Err(SparkError::InvalidArgument("zero dimension".into()));
    }
    if !num_heads.is_multiple_of(num_kv_heads) {
        return Err(SparkError::InvalidArgument(
            "num_heads must be divisible by num_kv_heads".into(),
        ));
    }

    let func = module::load_kernel(ctx, "fa_bf16_v3_paged_kv", "flash_attn_bf16_v3_paged_kv")?;

    let grid_x = seq_q.div_ceil(128); // Br=128

    let cfg = LaunchConfig {
        grid_dim: (grid_x, num_heads, batch),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0, // Static SMEM in PTX
    };

    unsafe {
        stream
            .launch_builder(&func)
            .arg(q)
            .arg(k)
            .arg(v)
            .arg(o)
            .arg(page_table)
            .arg(&seq_q)
            .arg(&seq_kv)
            .arg(&num_heads)
            .arg(&num_kv_heads)
            .arg(&page_size)
            .arg(&max_pages)
            .arg(&scale)
            .launch(cfg)?;
    }

    Ok(())
}

/// Launch BF16 Flash Attention V3 with paged KV cache (causal).
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_bf16_v3_paged_kv_causal(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u16>,
    k: &CudaSlice<u16>,
    v: &CudaSlice<u16>,
    o: &mut CudaSlice<u16>,
    page_table: &CudaSlice<u32>,
    batch: u32,
    num_heads: u32,
    num_kv_heads: u32,
    seq_q: u32,
    seq_kv: u32,
    page_size: u32,
    max_pages: u32,
    scale: f32,
) -> Result<()> {
    if batch == 0 || num_heads == 0 || seq_q == 0 || seq_kv == 0 {
        return Err(SparkError::InvalidArgument("zero dimension".into()));
    }
    if !num_heads.is_multiple_of(num_kv_heads) {
        return Err(SparkError::InvalidArgument(
            "num_heads must be divisible by num_kv_heads".into(),
        ));
    }

    let func = module::load_kernel(
        ctx,
        "fa_bf16_v3_paged_kv_causal",
        "flash_attn_bf16_v3_paged_kv_causal",
    )?;

    let grid_x = seq_q.div_ceil(128);
    let cfg = LaunchConfig {
        grid_dim: (grid_x, num_heads, batch),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&func)
            .arg(q)
            .arg(k)
            .arg(v)
            .arg(o)
            .arg(page_table)
            .arg(&seq_q)
            .arg(&seq_kv)
            .arg(&num_heads)
            .arg(&num_kv_heads)
            .arg(&page_size)
            .arg(&max_pages)
            .arg(&scale)
            .launch(cfg)?;
    }

    Ok(())
}

/// Launch split-KV forward kernel (one split).
/// Writes FP32 partial O + LSE for this split's KV range.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_bf16_v3_split_kv(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u16>,               // [B, H, Sq, D] bf16
    k: &CudaSlice<u16>,               // [B, H, Skv, D] bf16
    v: &CudaSlice<u16>,               // [B, H, Skv, D] bf16
    o_partial: &mut CudaSlice<f32>,   // [num_splits, B, H, Sq, D] f32
    lse_partial: &mut CudaSlice<f32>, // [num_splits, B, H, Sq] f32
    batch: u32,
    num_heads: u32,
    seq_q: u32,
    seq_kv: u32,
    scale: f32,
    num_splits: u32,
    split_idx: u32,
) -> Result<()> {
    let func = module::load_kernel(ctx, "fa_bf16_v3_split_kv", "flash_attn_bf16_v3_split_kv")?;

    let grid_x = seq_q.div_ceil(128);
    let cfg = LaunchConfig {
        grid_dim: (grid_x, num_heads, batch),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&func)
            .arg(q)
            .arg(k)
            .arg(v)
            .arg(o_partial)
            .arg(&seq_q)
            .arg(&seq_kv)
            .arg(&num_heads)
            .arg(&batch)
            .arg(&scale)
            .arg(lse_partial)
            .arg(&num_splits)
            .arg(&split_idx)
            .launch(cfg)?;
    }

    Ok(())
}

/// Launch FlashDecoding combine kernel.
/// Reduces split-KV partials to final BF16 output.
#[allow(clippy::too_many_arguments)]
pub fn flash_decoding_combine(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    o_partial: &CudaSlice<f32>,   // [num_splits, B, H, Sq, D] f32
    lse_partial: &CudaSlice<f32>, // [num_splits, B, H, Sq] f32
    o: &mut CudaSlice<u16>,       // [B, H, Sq, D] bf16
    batch: u32,
    num_heads: u32,
    seq_q: u32,
    num_splits: u32,
) -> Result<()> {
    let func = module::load_kernel(ctx, "flash_decoding_combine", "flash_decoding_combine")?;

    let cfg = LaunchConfig {
        grid_dim: (seq_q, num_heads, batch),
        block_dim: (128, 1, 1), // D=128 threads
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&func)
            .arg(o_partial)
            .arg(lse_partial)
            .arg(o)
            .arg(&batch)
            .arg(&num_heads)
            .arg(&seq_q)
            .arg(&num_splits)
            .launch(cfg)?;
    }

    Ok(())
}

/// Launch FP8 Flash Attention with paged KV cache.
/// V3 architecture: all-threads-load + bar.sync.
/// 160 threads (5 warps), Br=64, Bc=64, D=128.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_fp8_v3_paged_kv(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u8>,      // [B, H, Sq, D] FP8 e4m3
    k: &CudaSlice<u8>,      // [num_pages, page_size, num_kv_heads, D] FP8 e4m3
    v: &CudaSlice<u8>,      // [num_pages, page_size, num_kv_heads, D] FP8 e4m3
    o: &mut CudaSlice<u16>, // [B, H, Sq, D] bf16
    page_table: &CudaSlice<u32>,
    batch: u32,
    num_heads: u32,
    num_kv_heads: u32,
    seq_q: u32,
    seq_kv: u32,
    page_size: u32,
    max_pages: u32,
    scale: f32,
) -> Result<()> {
    if batch == 0 || num_heads == 0 || seq_q == 0 || seq_kv == 0 {
        return Err(SparkError::InvalidArgument("zero dimension".into()));
    }
    if !num_heads.is_multiple_of(num_kv_heads) {
        return Err(SparkError::InvalidArgument(
            "num_heads must be divisible by num_kv_heads".into(),
        ));
    }

    let func = module::load_kernel(ctx, "fa_fp8_v3_paged_kv", "flash_attn_fp8_v3_paged_kv")?;

    let grid_x = seq_q.div_ceil(128); // Br=128 (8 warps, 256 threads)

    let cfg = LaunchConfig {
        grid_dim: (grid_x, num_heads, batch),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&func)
            .arg(q)
            .arg(k)
            .arg(v)
            .arg(o)
            .arg(page_table)
            .arg(&seq_q)
            .arg(&seq_kv)
            .arg(&num_heads)
            .arg(&num_kv_heads)
            .arg(&page_size)
            .arg(&max_pages)
            .arg(&scale)
            .launch(cfg)?;
    }

    Ok(())
}

/// FP8 Paged KV cache + Causal flash attention (V3, 256 threads, BR=128).
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_fp8_v3_paged_kv_causal(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u8>,
    k: &CudaSlice<u8>,
    v: &CudaSlice<u8>,
    o: &mut CudaSlice<u16>,
    page_table: &CudaSlice<u32>,
    batch: u32,
    num_heads: u32,
    num_kv_heads: u32,
    seq_q: u32,
    seq_kv: u32,
    page_size: u32,
    max_pages: u32,
    scale: f32,
) -> Result<()> {
    let func = module::load_kernel(
        ctx,
        "fa_fp8_v3_paged_kv_causal",
        "flash_attn_fp8_v3_paged_kv_causal",
    )?;

    let grid_x = seq_q.div_ceil(128);
    let cfg = LaunchConfig {
        grid_dim: (grid_x, num_heads, batch),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&func)
            .arg(q)
            .arg(k)
            .arg(v)
            .arg(o)
            .arg(page_table)
            .arg(&seq_q)
            .arg(&seq_kv)
            .arg(&num_heads)
            .arg(&num_kv_heads)
            .arg(&page_size)
            .arg(&max_pages)
            .arg(&scale)
            .launch(cfg)?;
    }

    Ok(())
}

/// Launch split-KV forward kernel (causal) for one split.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_bf16_v3_split_kv_causal(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u16>,
    k: &CudaSlice<u16>,
    v: &CudaSlice<u16>,
    o_partial: &mut CudaSlice<f32>,
    lse_partial: &mut CudaSlice<f32>,
    batch: u32,
    num_heads: u32,
    seq_q: u32,
    seq_kv: u32,
    scale: f32,
    num_splits: u32,
    split_idx: u32,
) -> Result<()> {
    let func = module::load_kernel(
        ctx,
        "fa_bf16_v3_split_kv_causal",
        "flash_attn_bf16_v3_split_kv_causal",
    )?;

    let grid_x = seq_q.div_ceil(128);
    let cfg = LaunchConfig {
        grid_dim: (grid_x, num_heads, batch),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&func)
            .arg(q)
            .arg(k)
            .arg(v)
            .arg(o_partial)
            .arg(&seq_q)
            .arg(&seq_kv)
            .arg(&num_heads)
            .arg(&batch)
            .arg(&scale)
            .arg(lse_partial)
            .arg(&num_splits)
            .arg(&split_idx)
            .launch(cfg)?;
    }

    Ok(())
}

/// Launch split-KV + paged KV forward kernel (one split).
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_bf16_v3_split_paged_kv(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u16>,
    k: &CudaSlice<u16>,
    v: &CudaSlice<u16>,
    o_partial: &mut CudaSlice<f32>,
    lse_partial: &mut CudaSlice<f32>,
    page_table: &CudaSlice<u32>,
    batch: u32,
    num_heads: u32,
    num_kv_heads: u32,
    seq_q: u32,
    seq_kv: u32,
    page_size: u32,
    max_pages: u32,
    scale: f32,
    num_splits: u32,
    split_idx: u32,
) -> Result<()> {
    let func = module::load_kernel(
        ctx,
        "fa_bf16_v3_split_paged_kv",
        "flash_attn_bf16_v3_split_paged_kv",
    )?;

    let grid_x = seq_q.div_ceil(128);
    let cfg = LaunchConfig {
        grid_dim: (grid_x, num_heads, batch),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&func)
            .arg(q)
            .arg(k)
            .arg(v)
            .arg(o_partial)
            .arg(page_table)
            .arg(&seq_q)
            .arg(&seq_kv)
            .arg(&num_heads)
            .arg(&num_kv_heads)
            .arg(&page_size)
            .arg(&max_pages)
            .arg(&scale)
            .arg(lse_partial)
            .arg(&num_splits)
            .arg(&split_idx)
            .arg(&batch)
            .launch(cfg)?;
    }

    Ok(())
}

/// Split-KV + Paged KV cache + Causal flash attention (BF16, V3).
/// Same interface as split_paged_kv but with causal masking.
pub fn flash_attn_bf16_v3_split_paged_kv_causal(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u16>,
    k: &CudaSlice<u16>,
    v: &CudaSlice<u16>,
    o_partial: &mut CudaSlice<f32>,
    lse_partial: &mut CudaSlice<f32>,
    page_table: &CudaSlice<u32>,
    batch: u32,
    num_heads: u32,
    num_kv_heads: u32,
    seq_q: u32,
    seq_kv: u32,
    page_size: u32,
    max_pages: u32,
    scale: f32,
    num_splits: u32,
    split_idx: u32,
) -> Result<()> {
    let func = module::load_kernel(
        ctx,
        "fa_bf16_v3_split_paged_kv_causal",
        "flash_attn_bf16_v3_split_paged_kv_causal",
    )?;

    let grid_x = seq_q.div_ceil(128);
    let cfg = LaunchConfig {
        grid_dim: (grid_x, num_heads, batch),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&func)
            .arg(q)
            .arg(k)
            .arg(v)
            .arg(o_partial)
            .arg(page_table)
            .arg(&seq_q)
            .arg(&seq_kv)
            .arg(&num_heads)
            .arg(&num_kv_heads)
            .arg(&page_size)
            .arg(&max_pages)
            .arg(&scale)
            .arg(lse_partial)
            .arg(&num_splits)
            .arg(&split_idx)
            .arg(&batch)
            .launch(cfg)?;
    }

    Ok(())
}

/// FP8 Split-KV + Paged KV cache flash attention (V3, 256 threads, BR=128).
/// Writes FP32 partial O + LSE for FlashDecoding combine.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_fp8_v3_split_paged_kv(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u8>,
    k: &CudaSlice<u8>,
    v: &CudaSlice<u8>,
    o_partial: &mut CudaSlice<f32>,
    lse_partial: &mut CudaSlice<f32>,
    page_table: &CudaSlice<u32>,
    batch: u32,
    num_heads: u32,
    num_kv_heads: u32,
    seq_q: u32,
    seq_kv: u32,
    page_size: u32,
    max_pages: u32,
    scale: f32,
    num_splits: u32,
    split_idx: u32,
) -> Result<()> {
    let func = module::load_kernel(
        ctx,
        "fa_fp8_v3_split_paged_kv",
        "flash_attn_fp8_v3_split_paged_kv",
    )?;

    let grid_x = seq_q.div_ceil(128);
    let cfg = LaunchConfig {
        grid_dim: (grid_x, num_heads, batch),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&func)
            .arg(q)
            .arg(k)
            .arg(v)
            .arg(o_partial)
            .arg(page_table)
            .arg(&seq_q)
            .arg(&seq_kv)
            .arg(&num_heads)
            .arg(&num_kv_heads)
            .arg(&page_size)
            .arg(&max_pages)
            .arg(&scale)
            .arg(lse_partial)
            .arg(&num_splits)
            .arg(&split_idx)
            .arg(&batch)
            .launch(cfg)?;
    }

    Ok(())
}

/// FP8 Split-KV + Paged KV + Causal flash attention (V3, 256 threads, BR=128).
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_fp8_v3_split_paged_kv_causal(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u8>,
    k: &CudaSlice<u8>,
    v: &CudaSlice<u8>,
    o_partial: &mut CudaSlice<f32>,
    lse_partial: &mut CudaSlice<f32>,
    page_table: &CudaSlice<u32>,
    batch: u32,
    num_heads: u32,
    num_kv_heads: u32,
    seq_q: u32,
    seq_kv: u32,
    page_size: u32,
    max_pages: u32,
    scale: f32,
    num_splits: u32,
    split_idx: u32,
) -> Result<()> {
    let func = module::load_kernel(
        ctx,
        "fa_fp8_v3_split_paged_kv_causal",
        "flash_attn_fp8_v3_split_paged_kv_causal",
    )?;

    let grid_x = seq_q.div_ceil(128);
    let cfg = LaunchConfig {
        grid_dim: (grid_x, num_heads, batch),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&func)
            .arg(q)
            .arg(k)
            .arg(v)
            .arg(o_partial)
            .arg(page_table)
            .arg(&seq_q)
            .arg(&seq_kv)
            .arg(&num_heads)
            .arg(&num_kv_heads)
            .arg(&page_size)
            .arg(&max_pages)
            .arg(&scale)
            .arg(lse_partial)
            .arg(&num_splits)
            .arg(&split_idx)
            .arg(&batch)
            .launch(cfg)?;
    }

    Ok(())
}

/// Launch BF16 Flash Attention V20 (TMA, BC=128, single-buffered) with d=128.
/// 5 warps (160 threads): 4 MMA + 1 DMA. BR=64, BC=128.
/// TMA with swizzle for K/V loading. ~96KB dynamic SMEM.
#[allow(clippy::too_many_arguments)]
#[cfg(feature = "experimental")]
pub fn flash_attn_bf16_v20_tma_bc128(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u16>,
    k: &CudaSlice<u16>,
    v: &CudaSlice<u16>,
    o: &mut CudaSlice<u16>,
    batch: u32,
    num_heads: u32,
    seq_q: u32,
    seq_kv: u32,
    scale: f32,
) -> Result<()> {
    validate_attn_dims(batch, num_heads, seq_q, seq_kv)?;
    validate_attn_bf16_bufs(q, k, v, o, batch, num_heads, seq_q, seq_kv)?;

    use cudarc::driver::sys::*;

    let grid_x = seq_q.div_ceil(64); // BR=64

    let (q_dptr, _q_sync) = q.device_ptr(stream);
    let (k_dptr, _k_sync) = k.device_ptr(stream);
    let (v_dptr, _v_sync) = v.device_ptr(stream);
    let (o_dptr, _o_sync) = o.device_ptr(stream);

    let total_q_rows = batch * num_heads * seq_q;
    let total_kv_rows = batch * num_heads * seq_kv;

    // TMA tile: BC=128 rows, 64 cols per half
    let q_tma = create_tma_desc_swizzle(q_dptr as *mut _, total_q_rows, HEAD_DIM, 64, 64)?;
    let k_tma = create_tma_desc_swizzle(k_dptr as *mut _, total_kv_rows, HEAD_DIM, 128, 64)?;
    let v_tma = create_tma_desc_swizzle(v_dptr as *mut _, total_kv_rows, HEAD_DIM, 128, 64)?;

    let q_tma_u32: [u32; 32] = unsafe { core::mem::transmute(q_tma) };
    let k_tma_u32: [u32; 32] = unsafe { core::mem::transmute(k_tma) };
    let v_tma_u32: [u32; 32] = unsafe { core::mem::transmute(v_tma) };
    let q_tma_dev = stream.memcpy_stod(&q_tma_u32).map_err(SparkError::Driver)?;
    let k_tma_dev = stream.memcpy_stod(&k_tma_u32).map_err(SparkError::Driver)?;
    let v_tma_dev = stream.memcpy_stod(&v_tma_u32).map_err(SparkError::Driver)?;

    let (q_tma_dptr, _) = q_tma_dev.device_ptr(stream);
    let (k_tma_dptr, _) = k_tma_dev.device_ptr(stream);
    let (v_tma_dptr, _) = v_tma_dev.device_ptr(stream);

    let cu_func = module::load_kernel_raw(
        ctx,
        "fa_bf16_v20_tma_bc128",
        "flash_attn_bf16_v20_tma_bc128",
    )?;
    let cu_stream = stream.cu_stream();

    const SMEM_TOTAL: i32 = 98368;
    let attr_result = unsafe {
        cuFuncSetAttribute(
            cu_func,
            CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            SMEM_TOTAL,
        )
    };
    if attr_result != CUresult::CUDA_SUCCESS {
        return Err(SparkError::LaunchFailed(format!(
            "cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES={}) failed: {:?}",
            SMEM_TOTAL, attr_result
        )));
    }

    let params: [*mut core::ffi::c_void; 8] = [
        &q_tma_dptr as *const u64 as *mut _,
        &k_tma_dptr as *const u64 as *mut _,
        &v_tma_dptr as *const u64 as *mut _,
        &o_dptr as *const u64 as *mut _,
        &seq_q as *const u32 as *mut _,
        &seq_kv as *const u32 as *mut _,
        &num_heads as *const u32 as *mut _,
        &scale as *const f32 as *mut _,
    ];

    let result = unsafe {
        cuLaunchKernel(
            cu_func,
            grid_x,
            num_heads,
            batch,
            160,
            1,
            1, // 5 warps (4 MMA + 1 DMA)
            SMEM_TOTAL as u32,
            cu_stream,
            params.as_ptr() as *mut _,
            core::ptr::null_mut(),
        )
    };

    if result != CUresult::CUDA_SUCCESS {
        return Err(SparkError::LaunchFailed(format!(
            "cuLaunchKernel failed for V20 TMA BC=128: {:?}",
            result
        )));
    }

    Ok(())
}

/// Launch BF16 Flash Attention V21 (streaming P, no P SMEM buffer) with d=128.
/// Same as V11 but fuses P computation with PV GEMM, eliminating P SMEM round-trip.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_bf16_v21_streaming_p(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u16>,
    k: &CudaSlice<u16>,
    v: &CudaSlice<u16>,
    o: &mut CudaSlice<u16>,
    batch: u32,
    num_heads: u32,
    seq_q: u32,
    seq_kv: u32,
    scale: f32,
) -> Result<()> {
    validate_attn_dims(batch, num_heads, seq_q, seq_kv)?;
    validate_attn_bf16_bufs(q, k, v, o, batch, num_heads, seq_q, seq_kv)?;

    use cudarc::driver::sys::*;

    let grid_x = seq_q.div_ceil(64);

    let (q_dptr, _q_sync) = q.device_ptr(stream);
    let (k_dptr, _k_sync) = k.device_ptr(stream);
    let (v_dptr, _v_sync) = v.device_ptr(stream);
    let (o_dptr, _o_sync) = o.device_ptr(stream);

    let total_q_rows = batch * num_heads * seq_q;
    let total_kv_rows = batch * num_heads * seq_kv;

    let q_tma = create_tma_desc_swizzle(q_dptr as *mut _, total_q_rows, HEAD_DIM, 64, 64)?;
    let k_tma = create_tma_desc_swizzle(k_dptr as *mut _, total_kv_rows, HEAD_DIM, 64, 64)?;
    let v_tma = create_tma_desc_swizzle(v_dptr as *mut _, total_kv_rows, HEAD_DIM, 64, 64)?;

    let q_tma_u32: [u32; 32] = unsafe { core::mem::transmute(q_tma) };
    let k_tma_u32: [u32; 32] = unsafe { core::mem::transmute(k_tma) };
    let v_tma_u32: [u32; 32] = unsafe { core::mem::transmute(v_tma) };
    let q_tma_dev = stream.memcpy_stod(&q_tma_u32).map_err(SparkError::Driver)?;
    let k_tma_dev = stream.memcpy_stod(&k_tma_u32).map_err(SparkError::Driver)?;
    let v_tma_dev = stream.memcpy_stod(&v_tma_u32).map_err(SparkError::Driver)?;

    let (q_tma_dptr, _) = q_tma_dev.device_ptr(stream);
    let (k_tma_dptr, _) = k_tma_dev.device_ptr(stream);
    let (v_tma_dptr, _) = v_tma_dev.device_ptr(stream);

    let cu_func = module::load_kernel_raw(
        ctx,
        "fa_bf16_v21_streaming_p",
        "flash_attn_bf16_v21_streaming_p",
    )?;
    let cu_stream = stream.cu_stream();

    // V21 removes P SMEM buffer — use the SMEM_TOTAL from the PTX
    // The agent will set this. For now use V11's value (safe upper bound).
    const SMEM_TOTAL: i32 = 82032;
    let attr_result = unsafe {
        cuFuncSetAttribute(
            cu_func,
            CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            SMEM_TOTAL,
        )
    };
    if attr_result != CUresult::CUDA_SUCCESS {
        return Err(SparkError::LaunchFailed(format!(
            "cuFuncSetAttribute failed: {:?}",
            attr_result
        )));
    }

    let params: [*mut core::ffi::c_void; 8] = [
        &q_tma_dptr as *const u64 as *mut _,
        &k_tma_dptr as *const u64 as *mut _,
        &v_tma_dptr as *const u64 as *mut _,
        &o_dptr as *const u64 as *mut _,
        &seq_q as *const u32 as *mut _,
        &seq_kv as *const u32 as *mut _,
        &num_heads as *const u32 as *mut _,
        &scale as *const f32 as *mut _,
    ];

    let result = unsafe {
        cuLaunchKernel(
            cu_func,
            grid_x,
            num_heads,
            batch,
            160,
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
            "V21 streaming P launch failed: {:?}",
            result
        )));
    }

    Ok(())
}

/// Launch BF16 Flash Attention V21 CAUSAL (streaming P + causal masking) with d=128.
/// Same architecture as V21 streaming_p + causal masking on diagonal KV block,
/// and effective_kv_blocks = min(num_kv_blocks, q_block+1) to skip future blocks.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_bf16_v21_causal(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u16>,
    k: &CudaSlice<u16>,
    v: &CudaSlice<u16>,
    o: &mut CudaSlice<u16>,
    batch: u32,
    num_heads: u32,
    seq_q: u32,
    seq_kv: u32,
    scale: f32,
) -> Result<()> {
    validate_attn_dims(batch, num_heads, seq_q, seq_kv)?;
    validate_attn_bf16_bufs(q, k, v, o, batch, num_heads, seq_q, seq_kv)?;

    use cudarc::driver::sys::*;

    let grid_x = seq_q.div_ceil(64);

    let (q_dptr, _q_sync) = q.device_ptr(stream);
    let (k_dptr, _k_sync) = k.device_ptr(stream);
    let (v_dptr, _v_sync) = v.device_ptr(stream);
    let (o_dptr, _o_sync) = o.device_ptr(stream);

    let total_q_rows = batch * num_heads * seq_q;
    let total_kv_rows = batch * num_heads * seq_kv;

    let q_tma = create_tma_desc_swizzle(q_dptr as *mut _, total_q_rows, HEAD_DIM, 64, 64)?;
    let k_tma = create_tma_desc_swizzle(k_dptr as *mut _, total_kv_rows, HEAD_DIM, 64, 64)?;
    let v_tma = create_tma_desc_swizzle(v_dptr as *mut _, total_kv_rows, HEAD_DIM, 64, 64)?;

    let q_tma_u32: [u32; 32] = unsafe { core::mem::transmute(q_tma) };
    let k_tma_u32: [u32; 32] = unsafe { core::mem::transmute(k_tma) };
    let v_tma_u32: [u32; 32] = unsafe { core::mem::transmute(v_tma) };
    let q_tma_dev = stream.memcpy_stod(&q_tma_u32).map_err(SparkError::Driver)?;
    let k_tma_dev = stream.memcpy_stod(&k_tma_u32).map_err(SparkError::Driver)?;
    let v_tma_dev = stream.memcpy_stod(&v_tma_u32).map_err(SparkError::Driver)?;

    let (q_tma_dptr, _) = q_tma_dev.device_ptr(stream);
    let (k_tma_dptr, _) = k_tma_dev.device_ptr(stream);
    let (v_tma_dptr, _) = v_tma_dev.device_ptr(stream);

    let cu_func = module::load_kernel_raw(ctx, "fa_bf16_v21_causal", "flash_attn_bf16_v21_causal")?;
    let cu_stream = stream.cu_stream();

    const SMEM_TOTAL: i32 = 82032;
    let attr_result = unsafe {
        cuFuncSetAttribute(
            cu_func,
            CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            SMEM_TOTAL,
        )
    };
    if attr_result != CUresult::CUDA_SUCCESS {
        return Err(SparkError::LaunchFailed(format!(
            "cuFuncSetAttribute failed: {:?}",
            attr_result
        )));
    }

    let params: [*mut core::ffi::c_void; 8] = [
        &q_tma_dptr as *const u64 as *mut _,
        &k_tma_dptr as *const u64 as *mut _,
        &v_tma_dptr as *const u64 as *mut _,
        &o_dptr as *const u64 as *mut _,
        &seq_q as *const u32 as *mut _,
        &seq_kv as *const u32 as *mut _,
        &num_heads as *const u32 as *mut _,
        &scale as *const f32 as *mut _,
    ];

    let result = unsafe {
        cuLaunchKernel(
            cu_func,
            grid_x,
            num_heads,
            batch,
            160,
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
            "V21 causal launch failed: {:?}",
            result
        )));
    }

    Ok(())
}

/// Launch BF16 Flash Attention V21 GQA (streaming P + Grouped Query Attention) with d=128.
/// Q has `num_heads_q` heads, K/V have `num_heads_kv` heads. Requires H_q % H_kv == 0.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_bf16_v21_gqa(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u16>,
    k: &CudaSlice<u16>,
    v: &CudaSlice<u16>,
    o: &mut CudaSlice<u16>,
    batch: u32,
    num_heads_q: u32,
    num_heads_kv: u32,
    seq_q: u32,
    seq_kv: u32,
    scale: f32,
) -> Result<()> {
    if batch == 0 || num_heads_q == 0 || num_heads_kv == 0 || seq_q == 0 || seq_kv == 0 {
        return Err(SparkError::InvalidArgument(
            "batch, num_heads_q, num_heads_kv, seq_q, seq_kv must be > 0".into(),
        ));
    }
    if !num_heads_q.is_multiple_of(num_heads_kv) {
        return Err(SparkError::InvalidArgument(format!(
            "num_heads_q ({num_heads_q}) must be divisible by num_heads_kv ({num_heads_kv})"
        )));
    }

    use cudarc::driver::sys::*;

    let grid_x = seq_q.div_ceil(64);

    let (q_dptr, _q_sync) = q.device_ptr(stream);
    let (k_dptr, _k_sync) = k.device_ptr(stream);
    let (v_dptr, _v_sync) = v.device_ptr(stream);
    let (o_dptr, _o_sync) = o.device_ptr(stream);

    let total_q_rows = batch * num_heads_q * seq_q;
    let total_kv_rows = batch * num_heads_kv * seq_kv;

    let q_tma = create_tma_desc_swizzle(q_dptr as *mut _, total_q_rows, HEAD_DIM, 64, 64)?;
    let k_tma = create_tma_desc_swizzle(k_dptr as *mut _, total_kv_rows, HEAD_DIM, 64, 64)?;
    let v_tma = create_tma_desc_swizzle(v_dptr as *mut _, total_kv_rows, HEAD_DIM, 64, 64)?;

    let q_tma_u32: [u32; 32] = unsafe { core::mem::transmute(q_tma) };
    let k_tma_u32: [u32; 32] = unsafe { core::mem::transmute(k_tma) };
    let v_tma_u32: [u32; 32] = unsafe { core::mem::transmute(v_tma) };
    let q_tma_dev = stream.memcpy_stod(&q_tma_u32).map_err(SparkError::Driver)?;
    let k_tma_dev = stream.memcpy_stod(&k_tma_u32).map_err(SparkError::Driver)?;
    let v_tma_dev = stream.memcpy_stod(&v_tma_u32).map_err(SparkError::Driver)?;

    let (q_tma_dptr, _) = q_tma_dev.device_ptr(stream);
    let (k_tma_dptr, _) = k_tma_dev.device_ptr(stream);
    let (v_tma_dptr, _) = v_tma_dev.device_ptr(stream);

    let cu_func = module::load_kernel_raw(ctx, "fa_bf16_v21_gqa", "flash_attn_bf16_v21_gqa")?;
    let cu_stream = stream.cu_stream();

    const SMEM_TOTAL: i32 = 82032;
    let attr_result = unsafe {
        cuFuncSetAttribute(
            cu_func,
            CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            SMEM_TOTAL,
        )
    };
    if attr_result != CUresult::CUDA_SUCCESS {
        return Err(SparkError::LaunchFailed(format!(
            "cuFuncSetAttribute failed: {:?}",
            attr_result
        )));
    }

    let params: [*mut core::ffi::c_void; 9] = [
        &q_tma_dptr as *const u64 as *mut _,
        &k_tma_dptr as *const u64 as *mut _,
        &v_tma_dptr as *const u64 as *mut _,
        &o_dptr as *const u64 as *mut _,
        &seq_q as *const u32 as *mut _,
        &seq_kv as *const u32 as *mut _,
        &num_heads_q as *const u32 as *mut _,
        &num_heads_kv as *const u32 as *mut _,
        &scale as *const f32 as *mut _,
    ];

    let result = unsafe {
        cuLaunchKernel(
            cu_func,
            grid_x,
            num_heads_q,
            batch,
            160,
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
            "V21 GQA launch failed: {:?}",
            result
        )));
    }

    Ok(())
}

/// Launch BF16 Flash Attention V21 GQA + CAUSAL (streaming P + GQA + causal masking).
/// Combines head remap + causal diagonal block masking + effective_kv_blocks limit.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_bf16_v21_gqa_causal(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u16>,
    k: &CudaSlice<u16>,
    v: &CudaSlice<u16>,
    o: &mut CudaSlice<u16>,
    batch: u32,
    num_heads_q: u32,
    num_heads_kv: u32,
    seq_q: u32,
    seq_kv: u32,
    scale: f32,
) -> Result<()> {
    if batch == 0 || num_heads_q == 0 || num_heads_kv == 0 || seq_q == 0 || seq_kv == 0 {
        return Err(SparkError::InvalidArgument(
            "batch, num_heads_q, num_heads_kv, seq_q, seq_kv must be > 0".into(),
        ));
    }
    if !num_heads_q.is_multiple_of(num_heads_kv) {
        return Err(SparkError::InvalidArgument(format!(
            "num_heads_q ({num_heads_q}) must be divisible by num_heads_kv ({num_heads_kv})"
        )));
    }

    use cudarc::driver::sys::*;

    let grid_x = seq_q.div_ceil(64);

    let (q_dptr, _q_sync) = q.device_ptr(stream);
    let (k_dptr, _k_sync) = k.device_ptr(stream);
    let (v_dptr, _v_sync) = v.device_ptr(stream);
    let (o_dptr, _o_sync) = o.device_ptr(stream);

    let total_q_rows = batch * num_heads_q * seq_q;
    let total_kv_rows = batch * num_heads_kv * seq_kv;

    let q_tma = create_tma_desc_swizzle(q_dptr as *mut _, total_q_rows, HEAD_DIM, 64, 64)?;
    let k_tma = create_tma_desc_swizzle(k_dptr as *mut _, total_kv_rows, HEAD_DIM, 64, 64)?;
    let v_tma = create_tma_desc_swizzle(v_dptr as *mut _, total_kv_rows, HEAD_DIM, 64, 64)?;

    let q_tma_u32: [u32; 32] = unsafe { core::mem::transmute(q_tma) };
    let k_tma_u32: [u32; 32] = unsafe { core::mem::transmute(k_tma) };
    let v_tma_u32: [u32; 32] = unsafe { core::mem::transmute(v_tma) };
    let q_tma_dev = stream.memcpy_stod(&q_tma_u32).map_err(SparkError::Driver)?;
    let k_tma_dev = stream.memcpy_stod(&k_tma_u32).map_err(SparkError::Driver)?;
    let v_tma_dev = stream.memcpy_stod(&v_tma_u32).map_err(SparkError::Driver)?;

    let (q_tma_dptr, _) = q_tma_dev.device_ptr(stream);
    let (k_tma_dptr, _) = k_tma_dev.device_ptr(stream);
    let (v_tma_dptr, _) = v_tma_dev.device_ptr(stream);

    let cu_func = module::load_kernel_raw(
        ctx,
        "fa_bf16_v21_gqa_causal",
        "flash_attn_bf16_v21_gqa_causal",
    )?;
    let cu_stream = stream.cu_stream();

    const SMEM_TOTAL: i32 = 82032;
    let attr_result = unsafe {
        cuFuncSetAttribute(
            cu_func,
            CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            SMEM_TOTAL,
        )
    };
    if attr_result != CUresult::CUDA_SUCCESS {
        return Err(SparkError::LaunchFailed(format!(
            "cuFuncSetAttribute failed: {:?}",
            attr_result
        )));
    }

    let params: [*mut core::ffi::c_void; 9] = [
        &q_tma_dptr as *const u64 as *mut _,
        &k_tma_dptr as *const u64 as *mut _,
        &v_tma_dptr as *const u64 as *mut _,
        &o_dptr as *const u64 as *mut _,
        &seq_q as *const u32 as *mut _,
        &seq_kv as *const u32 as *mut _,
        &num_heads_q as *const u32 as *mut _,
        &num_heads_kv as *const u32 as *mut _,
        &scale as *const f32 as *mut _,
    ];

    let result = unsafe {
        cuLaunchKernel(
            cu_func,
            grid_x,
            num_heads_q,
            batch,
            160,
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
            "V21 GQA causal launch failed: {:?}",
            result
        )));
    }

    Ok(())
}

// ─── V21 varlen dispatch family ──────────────────────────────────────
// Head-major layout: Q/K/V/O = [num_heads, total_tokens, D] flat bf16.
// cu_seqlens_q/k: [batch+1] u32, per-batch cumulative token counts.
// TMA descriptor is over [num_heads*total_tokens, D].
// Non-GQA variants assume num_heads_q == num_heads_kv.

/// Launch BF16 Flash Attention V21 varlen (streaming P + variable-length sequences).
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_bf16_v21_varlen(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u16>,
    k: &CudaSlice<u16>,
    v: &CudaSlice<u16>,
    o: &mut CudaSlice<u16>,
    cu_seqlens_q: &CudaSlice<u32>,
    cu_seqlens_k: &CudaSlice<u32>,
    batch: u32,
    num_heads: u32,
    max_seqlen_q: u32,
    total_q: u32,
    total_kv: u32,
    scale: f32,
) -> Result<()> {
    v21_varlen_dispatch(
        ctx,
        stream,
        q,
        k,
        v,
        o,
        cu_seqlens_q,
        cu_seqlens_k,
        batch,
        num_heads,
        num_heads,
        max_seqlen_q,
        total_q,
        total_kv,
        scale,
        "fa_bf16_v21_varlen",
        "flash_attn_bf16_v21_varlen",
        false,
    )
}

/// Launch BF16 Flash Attention V21 varlen + CAUSAL.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_bf16_v21_varlen_causal(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u16>,
    k: &CudaSlice<u16>,
    v: &CudaSlice<u16>,
    o: &mut CudaSlice<u16>,
    cu_seqlens_q: &CudaSlice<u32>,
    cu_seqlens_k: &CudaSlice<u32>,
    batch: u32,
    num_heads: u32,
    max_seqlen_q: u32,
    total_q: u32,
    total_kv: u32,
    scale: f32,
) -> Result<()> {
    v21_varlen_dispatch(
        ctx,
        stream,
        q,
        k,
        v,
        o,
        cu_seqlens_q,
        cu_seqlens_k,
        batch,
        num_heads,
        num_heads,
        max_seqlen_q,
        total_q,
        total_kv,
        scale,
        "fa_bf16_v21_varlen_causal",
        "flash_attn_bf16_v21_varlen_causal",
        false,
    )
}

/// Launch BF16 Flash Attention V21 varlen + GQA.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_bf16_v21_varlen_gqa(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u16>,
    k: &CudaSlice<u16>,
    v: &CudaSlice<u16>,
    o: &mut CudaSlice<u16>,
    cu_seqlens_q: &CudaSlice<u32>,
    cu_seqlens_k: &CudaSlice<u32>,
    batch: u32,
    num_heads_q: u32,
    num_heads_kv: u32,
    max_seqlen_q: u32,
    total_q: u32,
    total_kv: u32,
    scale: f32,
) -> Result<()> {
    v21_varlen_dispatch(
        ctx,
        stream,
        q,
        k,
        v,
        o,
        cu_seqlens_q,
        cu_seqlens_k,
        batch,
        num_heads_q,
        num_heads_kv,
        max_seqlen_q,
        total_q,
        total_kv,
        scale,
        "fa_bf16_v21_varlen_gqa",
        "flash_attn_bf16_v21_varlen_gqa",
        true,
    )
}

/// Launch BF16 Flash Attention V21 varlen + GQA + CAUSAL (all three variant features).
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_bf16_v21_varlen_gqa_causal(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u16>,
    k: &CudaSlice<u16>,
    v: &CudaSlice<u16>,
    o: &mut CudaSlice<u16>,
    cu_seqlens_q: &CudaSlice<u32>,
    cu_seqlens_k: &CudaSlice<u32>,
    batch: u32,
    num_heads_q: u32,
    num_heads_kv: u32,
    max_seqlen_q: u32,
    total_q: u32,
    total_kv: u32,
    scale: f32,
) -> Result<()> {
    v21_varlen_dispatch(
        ctx,
        stream,
        q,
        k,
        v,
        o,
        cu_seqlens_q,
        cu_seqlens_k,
        batch,
        num_heads_q,
        num_heads_kv,
        max_seqlen_q,
        total_q,
        total_kv,
        scale,
        "fa_bf16_v21_varlen_gqa_causal",
        "flash_attn_bf16_v21_varlen_gqa_causal",
        true,
    )
}

/// Shared dispatch helper for V21 varlen family. `is_gqa` selects the 11-param
/// (GQA) vs 9-param (non-GQA) kernel signature.
#[allow(clippy::too_many_arguments)]
fn v21_varlen_dispatch(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u16>,
    k: &CudaSlice<u16>,
    v: &CudaSlice<u16>,
    o: &mut CudaSlice<u16>,
    cu_seqlens_q: &CudaSlice<u32>,
    cu_seqlens_k: &CudaSlice<u32>,
    batch: u32,
    num_heads_q: u32,
    num_heads_kv: u32,
    max_seqlen_q: u32,
    total_q: u32,
    total_kv: u32,
    scale: f32,
    cubin_name: &'static str,
    entry_name: &str,
    is_gqa: bool,
) -> Result<()> {
    if batch == 0 || num_heads_q == 0 || num_heads_kv == 0 {
        return Err(SparkError::InvalidArgument(
            "batch, num_heads_q, num_heads_kv must be > 0".into(),
        ));
    }
    if max_seqlen_q == 0 || total_q == 0 || total_kv == 0 {
        return Err(SparkError::InvalidArgument(
            "max_seqlen_q, total_q, total_kv must be > 0".into(),
        ));
    }
    if !num_heads_q.is_multiple_of(num_heads_kv) {
        return Err(SparkError::InvalidArgument(format!(
            "num_heads_q ({num_heads_q}) must be divisible by num_heads_kv ({num_heads_kv})"
        )));
    }
    let cu_need = (batch + 1) as usize;
    if cu_seqlens_q.len() < cu_need || cu_seqlens_k.len() < cu_need {
        return Err(SparkError::InvalidArgument(format!(
            "cu_seqlens too small: need {cu_need}"
        )));
    }

    use cudarc::driver::sys::*;

    let grid_x = max_seqlen_q.div_ceil(64);

    let (q_dptr, _q_sync) = q.device_ptr(stream);
    let (k_dptr, _k_sync) = k.device_ptr(stream);
    let (v_dptr, _v_sync) = v.device_ptr(stream);
    let (o_dptr, _o_sync) = o.device_ptr(stream);
    let (cu_q_dptr, _cu_q_sync) = cu_seqlens_q.device_ptr(stream);
    let (cu_k_dptr, _cu_k_sync) = cu_seqlens_k.device_ptr(stream);

    let q_tma = create_tma_desc_swizzle(q_dptr as *mut _, num_heads_q * total_q, HEAD_DIM, 64, 64)?;
    let k_tma =
        create_tma_desc_swizzle(k_dptr as *mut _, num_heads_kv * total_kv, HEAD_DIM, 64, 64)?;
    let v_tma =
        create_tma_desc_swizzle(v_dptr as *mut _, num_heads_kv * total_kv, HEAD_DIM, 64, 64)?;

    let q_tma_u32: [u32; 32] = unsafe { core::mem::transmute(q_tma) };
    let k_tma_u32: [u32; 32] = unsafe { core::mem::transmute(k_tma) };
    let v_tma_u32: [u32; 32] = unsafe { core::mem::transmute(v_tma) };
    let q_tma_dev = stream.memcpy_stod(&q_tma_u32).map_err(SparkError::Driver)?;
    let k_tma_dev = stream.memcpy_stod(&k_tma_u32).map_err(SparkError::Driver)?;
    let v_tma_dev = stream.memcpy_stod(&v_tma_u32).map_err(SparkError::Driver)?;

    let (q_tma_dptr, _) = q_tma_dev.device_ptr(stream);
    let (k_tma_dptr, _) = k_tma_dev.device_ptr(stream);
    let (v_tma_dptr, _) = v_tma_dev.device_ptr(stream);

    let cu_func = module::load_kernel_raw(ctx, cubin_name, entry_name)?;
    let cu_stream = stream.cu_stream();

    const SMEM_TOTAL: i32 = 82032;
    let attr_result = unsafe {
        cuFuncSetAttribute(
            cu_func,
            CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            SMEM_TOTAL,
        )
    };
    if attr_result != CUresult::CUDA_SUCCESS {
        return Err(SparkError::LaunchFailed(format!(
            "cuFuncSetAttribute failed: {:?}",
            attr_result
        )));
    }

    // Build params. GQA adds num_heads_q and num_heads_kv at the end.
    let base = [
        &q_tma_dptr as *const u64 as *mut core::ffi::c_void,
        &k_tma_dptr as *const u64 as *mut core::ffi::c_void,
        &v_tma_dptr as *const u64 as *mut core::ffi::c_void,
        &o_dptr as *const u64 as *mut core::ffi::c_void,
        &cu_q_dptr as *const u64 as *mut core::ffi::c_void,
        &cu_k_dptr as *const u64 as *mut core::ffi::c_void,
        &total_q as *const u32 as *mut core::ffi::c_void,
        &total_kv as *const u32 as *mut core::ffi::c_void,
    ];

    let result = if is_gqa {
        let params: [*mut core::ffi::c_void; 11] = [
            base[0],
            base[1],
            base[2],
            base[3],
            base[4],
            base[5],
            base[6],
            base[7],
            &num_heads_q as *const u32 as *mut _,
            &num_heads_kv as *const u32 as *mut _,
            &scale as *const f32 as *mut _,
        ];
        unsafe {
            cuLaunchKernel(
                cu_func,
                grid_x,
                num_heads_q,
                batch,
                160,
                1,
                1,
                SMEM_TOTAL as u32,
                cu_stream,
                params.as_ptr() as *mut _,
                core::ptr::null_mut(),
            )
        }
    } else {
        let params: [*mut core::ffi::c_void; 9] = [
            base[0],
            base[1],
            base[2],
            base[3],
            base[4],
            base[5],
            base[6],
            base[7],
            &scale as *const f32 as *mut _,
        ];
        unsafe {
            cuLaunchKernel(
                cu_func,
                grid_x,
                num_heads_q,
                batch,
                160,
                1,
                1,
                SMEM_TOTAL as u32,
                cu_stream,
                params.as_ptr() as *mut _,
                core::ptr::null_mut(),
            )
        }
    };

    if result != CUresult::CUDA_SUCCESS {
        return Err(SparkError::LaunchFailed(format!(
            "{} launch failed: {:?}",
            entry_name, result
        )));
    }

    Ok(())
}

/// Launch BF16 Flash Attention V21 + SOFTCAP + CAUSAL (Gemma 2 global layers).
/// S' = softcap * tanh(S * scale / softcap); applied after QK GEMM, before causal mask.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_bf16_v21_softcap_causal(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u16>,
    k: &CudaSlice<u16>,
    v: &CudaSlice<u16>,
    o: &mut CudaSlice<u16>,
    batch: u32,
    num_heads: u32,
    seq_q: u32,
    seq_kv: u32,
    scale: f32,
    softcap: f32,
) -> Result<()> {
    v21_scalar_mask_dispatch(
        ctx,
        stream,
        q,
        k,
        v,
        o,
        batch,
        num_heads,
        seq_q,
        seq_kv,
        scale,
        MaskKind::SoftcapCausal { softcap },
    )
}

/// Launch BF16 Flash Attention V21 + Sliding Window Attention (SWA, causal).
/// Each query attends only to K/V positions in [max(0, q - window + 1), q].
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_bf16_v21_swa(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u16>,
    k: &CudaSlice<u16>,
    v: &CudaSlice<u16>,
    o: &mut CudaSlice<u16>,
    batch: u32,
    num_heads: u32,
    seq_q: u32,
    seq_kv: u32,
    scale: f32,
    window: u32,
) -> Result<()> {
    if window == 0 {
        return Err(SparkError::InvalidArgument("window must be > 0".into()));
    }
    v21_scalar_mask_dispatch(
        ctx,
        stream,
        q,
        k,
        v,
        o,
        batch,
        num_heads,
        seq_q,
        seq_kv,
        scale,
        MaskKind::Swa { window },
    )
}

/// Launch BF16 Flash Attention V21 + SWA + Softcap (Gemma 2 local layers).
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_bf16_v21_swa_softcap(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u16>,
    k: &CudaSlice<u16>,
    v: &CudaSlice<u16>,
    o: &mut CudaSlice<u16>,
    batch: u32,
    num_heads: u32,
    seq_q: u32,
    seq_kv: u32,
    scale: f32,
    window: u32,
    softcap: f32,
) -> Result<()> {
    if window == 0 {
        return Err(SparkError::InvalidArgument("window must be > 0".into()));
    }
    v21_scalar_mask_dispatch(
        ctx,
        stream,
        q,
        k,
        v,
        o,
        batch,
        num_heads,
        seq_q,
        seq_kv,
        scale,
        MaskKind::SwaSoftcap { window, softcap },
    )
}

enum MaskKind {
    SoftcapCausal { softcap: f32 },
    Swa { window: u32 },
    SwaSoftcap { window: u32, softcap: f32 },
}

#[allow(clippy::too_many_arguments)]
fn v21_scalar_mask_dispatch(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u16>,
    k: &CudaSlice<u16>,
    v: &CudaSlice<u16>,
    o: &mut CudaSlice<u16>,
    batch: u32,
    num_heads: u32,
    seq_q: u32,
    seq_kv: u32,
    scale: f32,
    mask: MaskKind,
) -> Result<()> {
    validate_attn_dims(batch, num_heads, seq_q, seq_kv)?;
    validate_attn_bf16_bufs(q, k, v, o, batch, num_heads, seq_q, seq_kv)?;

    use cudarc::driver::sys::*;

    let grid_x = seq_q.div_ceil(64);

    let (q_dptr, _q_sync) = q.device_ptr(stream);
    let (k_dptr, _k_sync) = k.device_ptr(stream);
    let (v_dptr, _v_sync) = v.device_ptr(stream);
    let (o_dptr, _o_sync) = o.device_ptr(stream);

    let total_q_rows = batch * num_heads * seq_q;
    let total_kv_rows = batch * num_heads * seq_kv;

    let q_tma = create_tma_desc_swizzle(q_dptr as *mut _, total_q_rows, HEAD_DIM, 64, 64)?;
    let k_tma = create_tma_desc_swizzle(k_dptr as *mut _, total_kv_rows, HEAD_DIM, 64, 64)?;
    let v_tma = create_tma_desc_swizzle(v_dptr as *mut _, total_kv_rows, HEAD_DIM, 64, 64)?;

    let q_tma_u32: [u32; 32] = unsafe { core::mem::transmute(q_tma) };
    let k_tma_u32: [u32; 32] = unsafe { core::mem::transmute(k_tma) };
    let v_tma_u32: [u32; 32] = unsafe { core::mem::transmute(v_tma) };
    let q_tma_dev = stream.memcpy_stod(&q_tma_u32).map_err(SparkError::Driver)?;
    let k_tma_dev = stream.memcpy_stod(&k_tma_u32).map_err(SparkError::Driver)?;
    let v_tma_dev = stream.memcpy_stod(&v_tma_u32).map_err(SparkError::Driver)?;

    let (q_tma_dptr, _) = q_tma_dev.device_ptr(stream);
    let (k_tma_dptr, _) = k_tma_dev.device_ptr(stream);
    let (v_tma_dptr, _) = v_tma_dev.device_ptr(stream);

    let (cubin_name, entry_name) = match mask {
        MaskKind::SoftcapCausal { .. } => (
            "fa_bf16_v21_softcap_causal",
            "flash_attn_bf16_v21_softcap_causal",
        ),
        MaskKind::Swa { .. } => ("fa_bf16_v21_swa", "flash_attn_bf16_v21_swa"),
        MaskKind::SwaSoftcap { .. } => {
            ("fa_bf16_v21_swa_softcap", "flash_attn_bf16_v21_swa_softcap")
        }
    };

    let cu_func = module::load_kernel_raw(ctx, cubin_name, entry_name)?;
    let cu_stream = stream.cu_stream();

    const SMEM_TOTAL: i32 = 82032;
    let attr_result = unsafe {
        cuFuncSetAttribute(
            cu_func,
            CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            SMEM_TOTAL,
        )
    };
    if attr_result != CUresult::CUDA_SUCCESS {
        return Err(SparkError::LaunchFailed(format!(
            "cuFuncSetAttribute failed: {:?}",
            attr_result
        )));
    }

    let result = match mask {
        MaskKind::SoftcapCausal { softcap } => {
            let params: [*mut core::ffi::c_void; 9] = [
                &q_tma_dptr as *const u64 as *mut _,
                &k_tma_dptr as *const u64 as *mut _,
                &v_tma_dptr as *const u64 as *mut _,
                &o_dptr as *const u64 as *mut _,
                &seq_q as *const u32 as *mut _,
                &seq_kv as *const u32 as *mut _,
                &num_heads as *const u32 as *mut _,
                &scale as *const f32 as *mut _,
                &softcap as *const f32 as *mut _,
            ];
            unsafe {
                cuLaunchKernel(
                    cu_func,
                    grid_x,
                    num_heads,
                    batch,
                    160,
                    1,
                    1,
                    SMEM_TOTAL as u32,
                    cu_stream,
                    params.as_ptr() as *mut _,
                    core::ptr::null_mut(),
                )
            }
        }
        MaskKind::Swa { window } => {
            let params: [*mut core::ffi::c_void; 9] = [
                &q_tma_dptr as *const u64 as *mut _,
                &k_tma_dptr as *const u64 as *mut _,
                &v_tma_dptr as *const u64 as *mut _,
                &o_dptr as *const u64 as *mut _,
                &seq_q as *const u32 as *mut _,
                &seq_kv as *const u32 as *mut _,
                &num_heads as *const u32 as *mut _,
                &scale as *const f32 as *mut _,
                &window as *const u32 as *mut _,
            ];
            unsafe {
                cuLaunchKernel(
                    cu_func,
                    grid_x,
                    num_heads,
                    batch,
                    160,
                    1,
                    1,
                    SMEM_TOTAL as u32,
                    cu_stream,
                    params.as_ptr() as *mut _,
                    core::ptr::null_mut(),
                )
            }
        }
        MaskKind::SwaSoftcap { window, softcap } => {
            let params: [*mut core::ffi::c_void; 10] = [
                &q_tma_dptr as *const u64 as *mut _,
                &k_tma_dptr as *const u64 as *mut _,
                &v_tma_dptr as *const u64 as *mut _,
                &o_dptr as *const u64 as *mut _,
                &seq_q as *const u32 as *mut _,
                &seq_kv as *const u32 as *mut _,
                &num_heads as *const u32 as *mut _,
                &scale as *const f32 as *mut _,
                &window as *const u32 as *mut _,
                &softcap as *const f32 as *mut _,
            ];
            unsafe {
                cuLaunchKernel(
                    cu_func,
                    grid_x,
                    num_heads,
                    batch,
                    160,
                    1,
                    1,
                    SMEM_TOTAL as u32,
                    cu_stream,
                    params.as_ptr() as *mut _,
                    core::ptr::null_mut(),
                )
            }
        }
    };

    if result != CUresult::CUDA_SUCCESS {
        return Err(SparkError::LaunchFailed(format!(
            "{} launch failed: {:?}",
            entry_name, result
        )));
    }

    Ok(())
}

// ────────────────────────────────────────────────────────────────────────
// MLA (Multi-head Latent Attention) — DeepSeek V3, GLM-4.5/5
// ────────────────────────────────────────────────────────────────────────

/// MLA compressed (no-RoPE) latent dimension per head.
pub const MLA_D_C: u32 = 512;
/// MLA RoPE (decoupled positional) dimension per head.
pub const MLA_D_R: u32 = 64;

/// Launch weight-absorbed MLA decode (BF16, seq_q=1).
///
/// Buffers:
///   q_c: [batch, num_heads, D_C=512] BF16 (compressed query per head)
///   q_r: [batch, num_heads, D_R=64]  BF16 (RoPE query per head)
///   c_kv: [batch, seq_kv, D_C]       BF16 (compressed KV cache, shared across heads)
///   k_rope: [batch, seq_kv, D_R]     BF16 (RoPE K cache, shared across heads)
///   o: [batch, num_heads, D_C]       BF16 (output)
///
/// This is a scalar-reference implementation. Correct but not peak performance —
/// an MMA-based optimization is future work.
#[allow(clippy::too_many_arguments)]
pub fn mla_decode_bf16(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q_c: &CudaSlice<u16>,
    q_r: &CudaSlice<u16>,
    c_kv: &CudaSlice<u16>,
    k_rope: &CudaSlice<u16>,
    o: &mut CudaSlice<u16>,
    batch: u32,
    num_heads: u32,
    seq_kv: u32,
    scale: f32,
) -> Result<()> {
    if batch == 0 || num_heads == 0 || seq_kv == 0 {
        return Err(SparkError::InvalidArgument(
            "batch, num_heads, seq_kv must be > 0".into(),
        ));
    }
    let q_c_need = batch as usize * num_heads as usize * MLA_D_C as usize;
    let q_r_need = batch as usize * num_heads as usize * MLA_D_R as usize;
    let ckv_need = batch as usize * seq_kv as usize * MLA_D_C as usize;
    let krope_need = batch as usize * seq_kv as usize * MLA_D_R as usize;
    let o_need = batch as usize * num_heads as usize * MLA_D_C as usize;
    if q_c.len() < q_c_need
        || q_r.len() < q_r_need
        || c_kv.len() < ckv_need
        || k_rope.len() < krope_need
        || o.len() < o_need
    {
        return Err(SparkError::InvalidArgument(
            "MLA buffer size mismatch".into(),
        ));
    }

    use cudarc::driver::sys::*;

    let (q_c_ptr, _a) = q_c.device_ptr(stream);
    let (q_r_ptr, _b) = q_r.device_ptr(stream);
    let (c_kv_ptr, _c) = c_kv.device_ptr(stream);
    let (k_rope_ptr, _d) = k_rope.device_ptr(stream);
    let (o_ptr, _e) = o.device_ptr(stream);

    let cu_func = module::load_kernel_raw(ctx, "fa_bf16_mla_decode", "fa_bf16_mla_decode")?;
    let cu_stream = stream.cu_stream();

    let params: [*mut core::ffi::c_void; 8] = [
        &q_c_ptr as *const u64 as *mut _,
        &q_r_ptr as *const u64 as *mut _,
        &c_kv_ptr as *const u64 as *mut _,
        &k_rope_ptr as *const u64 as *mut _,
        &o_ptr as *const u64 as *mut _,
        &num_heads as *const u32 as *mut _,
        &seq_kv as *const u32 as *mut _,
        &scale as *const f32 as *mut _,
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
    // No explicit cuStreamSynchronize: launches are async by design. The
    // previous sync-after-launch pattern was a debug artifact that forced
    // the host to wait for each MLA decode to finish before submitting the
    // next kernel — turned out to be a 27-syncs-per-token bottleneck on
    // DSV2-Lite-Chat decode.
    if result != CUresult::CUDA_SUCCESS {
        return Err(SparkError::LaunchFailed(format!(
            "MLA decode launch: {:?}",
            result
        )));
    }
    Ok(())
}

/// Launch weight-absorbed MLA prefill (BF16, causal, seq_q > 1).
///
/// Buffers:
///   q_c: [batch, seq_q, num_heads, D_C=512] BF16
///   q_r: [batch, seq_q, num_heads, D_R=64]  BF16
///   c_kv: [batch, seq_kv, D_C]              BF16 (shared across heads)
///   k_rope: [batch, seq_kv, D_R]            BF16 (shared across heads)
///   o: [batch, seq_q, num_heads, D_C]       BF16
///
/// Causal: s in [0, q_idx] contributes; s > q_idx is masked out.
/// Scalar-reference implementation (correct, not peak performance).
#[allow(clippy::too_many_arguments)]
pub fn mla_prefill_bf16(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q_c: &CudaSlice<u16>,
    q_r: &CudaSlice<u16>,
    c_kv: &CudaSlice<u16>,
    k_rope: &CudaSlice<u16>,
    o: &mut CudaSlice<u16>,
    batch: u32,
    num_heads: u32,
    seq_q: u32,
    seq_kv: u32,
    scale: f32,
) -> Result<()> {
    if batch == 0 || num_heads == 0 || seq_q == 0 || seq_kv == 0 {
        return Err(SparkError::InvalidArgument(
            "batch, num_heads, seq_q, seq_kv must be > 0".into(),
        ));
    }
    let q_c_need = batch as usize * seq_q as usize * num_heads as usize * MLA_D_C as usize;
    let q_r_need = batch as usize * seq_q as usize * num_heads as usize * MLA_D_R as usize;
    let ckv_need = batch as usize * seq_kv as usize * MLA_D_C as usize;
    let krope_need = batch as usize * seq_kv as usize * MLA_D_R as usize;
    let o_need = q_c_need;
    if q_c.len() < q_c_need
        || q_r.len() < q_r_need
        || c_kv.len() < ckv_need
        || k_rope.len() < krope_need
        || o.len() < o_need
    {
        return Err(SparkError::InvalidArgument(
            "MLA prefill buffer size mismatch".into(),
        ));
    }

    use cudarc::driver::sys::*;

    let (q_c_ptr, _a) = q_c.device_ptr(stream);
    let (q_r_ptr, _b) = q_r.device_ptr(stream);
    let (c_kv_ptr, _c) = c_kv.device_ptr(stream);
    let (k_rope_ptr, _d) = k_rope.device_ptr(stream);
    let (o_ptr, _e) = o.device_ptr(stream);

    let cu_func = module::load_kernel_raw(ctx, "fa_bf16_mla_prefill", "fa_bf16_mla_prefill")?;
    let cu_stream = stream.cu_stream();

    let params: [*mut core::ffi::c_void; 9] = [
        &q_c_ptr as *const u64 as *mut _,
        &q_r_ptr as *const u64 as *mut _,
        &c_kv_ptr as *const u64 as *mut _,
        &k_rope_ptr as *const u64 as *mut _,
        &o_ptr as *const u64 as *mut _,
        &num_heads as *const u32 as *mut _,
        &seq_q as *const u32 as *mut _,
        &seq_kv as *const u32 as *mut _,
        &scale as *const f32 as *mut _,
    ];

    let result = unsafe {
        cuLaunchKernel(
            cu_func,
            seq_q,
            num_heads,
            batch,
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
            "MLA prefill launch: {:?}",
            result
        )));
    }
    Ok(())
}

/// Launch weight-absorbed MLA decode (FP8 KV + BF16 Q, seq_q=1).
///
/// c_kv and k_rope are FP8 E4M3 with per-tensor dequant scale.
/// Used by DeepSeek V3 with FP8 KV cache.
#[allow(clippy::too_many_arguments)]
/// FP8 auto-dispatch: picks between scalar FP8 MLA and FP8-KV MMA.
///
/// Decision tree:
///   - num_heads % 16 != 0       → scalar `mla_decode_fp8`
///   - seq_kv >= 32 AND H ≥ 16   → `mla_decode_fp8kv_mma` (1.3-1.6× over scalar)
///   - otherwise                 → scalar (MMA overhead not amortized)
#[allow(clippy::too_many_arguments)]
pub fn mla_decode_fp8_auto(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q_c: &CudaSlice<u16>,
    q_r: &CudaSlice<u16>,
    c_kv: &CudaSlice<u8>,
    k_rope: &CudaSlice<u8>,
    o: &mut CudaSlice<u16>,
    batch: u32,
    num_heads: u32,
    seq_kv: u32,
    scale: f32,
    kv_scale: f32,
) -> Result<()> {
    if !num_heads.is_multiple_of(16) || num_heads < 16 || seq_kv < 32 {
        return mla_decode_fp8(
            ctx, stream, q_c, q_r, c_kv, k_rope, o, batch, num_heads, seq_kv, scale, kv_scale,
        );
    }
    mla_decode_fp8kv_mma(
        ctx, stream, q_c, q_r, c_kv, k_rope, o, batch, num_heads, seq_kv, scale, kv_scale,
    )
}

/// Hybrid BF16 MMA with FP8 KV read — 16 heads/CTA, m16n8k16 QK + m16n8k8 PV.
/// Reads FP8 E4M3 c_kv/k_rope with per-tensor kv_scale, dequantizes to BF16 in
/// SMEM during load, then uses the proven BF16 MMA path. Requires num_heads % 16 == 0.
#[allow(clippy::too_many_arguments)]
pub fn mla_decode_fp8kv_mma(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q_c: &CudaSlice<u16>,
    q_r: &CudaSlice<u16>,
    c_kv: &CudaSlice<u8>,
    k_rope: &CudaSlice<u8>,
    o: &mut CudaSlice<u16>,
    batch: u32,
    num_heads: u32,
    seq_kv: u32,
    scale: f32,
    kv_scale: f32,
) -> Result<()> {
    if batch == 0 || num_heads == 0 || seq_kv == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    if !num_heads.is_multiple_of(16) {
        return Err(SparkError::InvalidArgument(format!(
            "mla_decode_fp8kv_mma requires num_heads % 16 == 0 (got {num_heads})"
        )));
    }

    use cudarc::driver::sys::*;

    let (q_c_ptr, _a) = q_c.device_ptr(stream);
    let (q_r_ptr, _b) = q_r.device_ptr(stream);
    let (c_kv_ptr, _c) = c_kv.device_ptr(stream);
    let (k_rope_ptr, _d) = k_rope.device_ptr(stream);
    let (o_ptr, _e) = o.device_ptr(stream);

    let cu_func =
        module::load_kernel_raw(ctx, "fa_fp8kv_mla_decode_mma", "fa_fp8kv_mla_decode_mma")?;
    let cu_stream = stream.cu_stream();

    const SMEM_TOTAL: i32 = 27968;
    let attr = unsafe {
        cuFuncSetAttribute(
            cu_func,
            CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            SMEM_TOTAL,
        )
    };
    if attr != CUresult::CUDA_SUCCESS {
        return Err(SparkError::LaunchFailed(format!("attr: {:?}", attr)));
    }

    let params: [*mut core::ffi::c_void; 9] = [
        &q_c_ptr as *const u64 as *mut _,
        &q_r_ptr as *const u64 as *mut _,
        &c_kv_ptr as *const u64 as *mut _,
        &k_rope_ptr as *const u64 as *mut _,
        &o_ptr as *const u64 as *mut _,
        &num_heads as *const u32 as *mut _,
        &seq_kv as *const u32 as *mut _,
        &scale as *const f32 as *mut _,
        &kv_scale as *const f32 as *mut _,
    ];

    let head_groups = num_heads / 16;
    let r = unsafe {
        cuLaunchKernel(
            cu_func,
            head_groups,
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
    if r != CUresult::CUDA_SUCCESS {
        return Err(SparkError::LaunchFailed(format!("fp8kv launch: {:?}", r)));
    }
    Ok(())
}

/// FP8-KV variant of `mla_decode_bf16`: the compressed KV and RoPE K caches are
/// stored as FP8 E4M3 and dequantized by `kv_scale` during the score/output
/// accumulation. Query inputs stay BF16. Scalar-reference implementation.
pub fn mla_decode_fp8(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q_c: &CudaSlice<u16>,
    q_r: &CudaSlice<u16>,
    c_kv: &CudaSlice<u8>,
    k_rope: &CudaSlice<u8>,
    o: &mut CudaSlice<u16>,
    batch: u32,
    num_heads: u32,
    seq_kv: u32,
    scale: f32,
    kv_scale: f32,
) -> Result<()> {
    if batch == 0 || num_heads == 0 || seq_kv == 0 {
        return Err(SparkError::InvalidArgument(
            "batch, num_heads, seq_kv must be > 0".into(),
        ));
    }

    use cudarc::driver::sys::*;

    let (q_c_ptr, _a) = q_c.device_ptr(stream);
    let (q_r_ptr, _b) = q_r.device_ptr(stream);
    let (c_kv_ptr, _c) = c_kv.device_ptr(stream);
    let (k_rope_ptr, _d) = k_rope.device_ptr(stream);
    let (o_ptr, _e) = o.device_ptr(stream);

    let cu_func = module::load_kernel_raw(ctx, "fa_fp8_mla_decode", "fa_fp8_mla_decode")?;
    let cu_stream = stream.cu_stream();

    let params: [*mut core::ffi::c_void; 9] = [
        &q_c_ptr as *const u64 as *mut _,
        &q_r_ptr as *const u64 as *mut _,
        &c_kv_ptr as *const u64 as *mut _,
        &k_rope_ptr as *const u64 as *mut _,
        &o_ptr as *const u64 as *mut _,
        &num_heads as *const u32 as *mut _,
        &seq_kv as *const u32 as *mut _,
        &scale as *const f32 as *mut _,
        &kv_scale as *const f32 as *mut _,
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
            "MLA FP8 decode launch: {:?}",
            result
        )));
    }
    Ok(())
}

/// Launch weight-absorbed MLA decode with PAGED KV cache (BF16, seq_q=1).
///
/// c_kv and k_rope live in a paged pool; page_table maps logical positions
/// to physical pages.  Used by vLLM/SGLang-style serving with paged caches.
///
/// Buffers:
///   q_c: [batch, num_heads, D_C=512]
///   q_r: [batch, num_heads, D_R=64]
///   c_kv: [num_pages, page_size, D_C]
///   k_rope: [num_pages, page_size, D_R]
///   page_table: [batch, max_pages] u32 (physical page indices)
///   seq_lens: [batch] u32 (actual length per batch)
///   o: [batch, num_heads, D_C]
#[allow(clippy::too_many_arguments)]
pub fn mla_decode_bf16_paged(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q_c: &CudaSlice<u16>,
    q_r: &CudaSlice<u16>,
    c_kv: &CudaSlice<u16>,
    k_rope: &CudaSlice<u16>,
    page_table: &CudaSlice<u32>,
    seq_lens: &CudaSlice<u32>,
    o: &mut CudaSlice<u16>,
    batch: u32,
    num_heads: u32,
    max_pages: u32,
    page_size: u32,
    scale: f32,
) -> Result<()> {
    if batch == 0 || num_heads == 0 || max_pages == 0 || page_size == 0 {
        return Err(SparkError::InvalidArgument(
            "batch, num_heads, max_pages, page_size must be > 0".into(),
        ));
    }

    use cudarc::driver::sys::*;

    let (q_c_ptr, _a) = q_c.device_ptr(stream);
    let (q_r_ptr, _b) = q_r.device_ptr(stream);
    let (c_kv_ptr, _c) = c_kv.device_ptr(stream);
    let (k_rope_ptr, _d) = k_rope.device_ptr(stream);
    let (page_table_ptr, _pt) = page_table.device_ptr(stream);
    let (seq_lens_ptr, _sl) = seq_lens.device_ptr(stream);
    let (o_ptr, _e) = o.device_ptr(stream);

    let cu_func =
        module::load_kernel_raw(ctx, "fa_bf16_mla_decode_paged", "fa_bf16_mla_decode_paged")?;
    let cu_stream = stream.cu_stream();

    let params: [*mut core::ffi::c_void; 11] = [
        &q_c_ptr as *const u64 as *mut _,
        &q_r_ptr as *const u64 as *mut _,
        &c_kv_ptr as *const u64 as *mut _,
        &k_rope_ptr as *const u64 as *mut _,
        &page_table_ptr as *const u64 as *mut _,
        &seq_lens_ptr as *const u64 as *mut _,
        &o_ptr as *const u64 as *mut _,
        &num_heads as *const u32 as *mut _,
        &max_pages as *const u32 as *mut _,
        &page_size as *const u32 as *mut _,
        &scale as *const f32 as *mut _,
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
            "MLA paged decode launch: {:?}",
            result
        )));
    }
    Ok(())
}

/// Launch tree attention (BF16) — used by EAGLE-3 / Medusa speculative decoding.
/// Each draft query attends only to positions marked with mask[q, k] = 1.
/// Mask is shared across batches and heads (standard EAGLE-3 convention).
///
/// Buffers:
///   q, k, v, o: standard [B, seq, H, D=128] BF16
///   mask: [seq_q, seq_kv] u8 (1 = attend, 0 = -inf)
#[allow(clippy::too_many_arguments)]
pub fn tree_attention_bf16(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u16>,
    k: &CudaSlice<u16>,
    v: &CudaSlice<u16>,
    mask: &CudaSlice<u8>,
    o: &mut CudaSlice<u16>,
    batch: u32,
    num_heads: u32,
    seq_q: u32,
    seq_kv: u32,
    scale: f32,
) -> Result<()> {
    if batch == 0 || num_heads == 0 || seq_q == 0 || seq_kv == 0 {
        return Err(SparkError::InvalidArgument(
            "batch, num_heads, seq_q, seq_kv must be > 0".into(),
        ));
    }
    let mask_need = (seq_q * seq_kv) as usize;
    if mask.len() < mask_need {
        return Err(SparkError::InvalidArgument(format!(
            "mask too small: need {mask_need}"
        )));
    }

    use cudarc::driver::sys::*;

    let (q_ptr, _a) = q.device_ptr(stream);
    let (k_ptr, _b) = k.device_ptr(stream);
    let (v_ptr, _c) = v.device_ptr(stream);
    let (mask_ptr, _m) = mask.device_ptr(stream);
    let (o_ptr, _e) = o.device_ptr(stream);

    let cu_func = module::load_kernel_raw(ctx, "fa_bf16_tree", "fa_bf16_tree")?;
    let cu_stream = stream.cu_stream();

    let params: [*mut core::ffi::c_void; 9] = [
        &q_ptr as *const u64 as *mut _,
        &k_ptr as *const u64 as *mut _,
        &v_ptr as *const u64 as *mut _,
        &mask_ptr as *const u64 as *mut _,
        &o_ptr as *const u64 as *mut _,
        &num_heads as *const u32 as *mut _,
        &seq_q as *const u32 as *mut _,
        &seq_kv as *const u32 as *mut _,
        &scale as *const f32 as *mut _,
    ];

    let result = unsafe {
        cuLaunchKernel(
            cu_func,
            seq_q,
            num_heads,
            batch,
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
            "tree attn launch: {:?}",
            result
        )));
    }
    Ok(())
}

/// Native Sparse Attention (NSA) — BF16 reference.
/// Each query attends only to pre-selected KV blocks (via block_idx).
///
/// Used by DeepSeek V3.2-Exp research branch. Block indices are typically
/// produced by a separate top-K block-scoring pass (not provided by this
/// kernel — caller is responsible for the selection).
///
/// Buffers:
///   q, k, v, o: standard [B, seq_q/kv, H, D=128] BF16
///   block_idx: [B, seq_q, H, k_top] u32 — which KV blocks each query attends to
#[allow(clippy::too_many_arguments)]
pub fn nsa_attention_bf16(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u16>,
    k: &CudaSlice<u16>,
    v: &CudaSlice<u16>,
    block_idx: &CudaSlice<u32>,
    o: &mut CudaSlice<u16>,
    batch: u32,
    num_heads: u32,
    seq_q: u32,
    seq_kv: u32,
    k_top: u32,
    block_size: u32,
    scale: f32,
) -> Result<()> {
    if batch == 0 || num_heads == 0 || seq_q == 0 || seq_kv == 0 || k_top == 0 || block_size == 0 {
        return Err(SparkError::InvalidArgument("all dims must be > 0".into()));
    }
    let idx_need = (batch * seq_q * num_heads * k_top) as usize;
    if block_idx.len() < idx_need {
        return Err(SparkError::InvalidArgument(format!(
            "block_idx too small: need {idx_need}"
        )));
    }

    use cudarc::driver::sys::*;

    let (q_ptr, _a) = q.device_ptr(stream);
    let (k_ptr, _b) = k.device_ptr(stream);
    let (v_ptr, _c) = v.device_ptr(stream);
    let (idx_ptr, _i) = block_idx.device_ptr(stream);
    let (o_ptr, _e) = o.device_ptr(stream);

    let cu_func = module::load_kernel_raw(ctx, "fa_bf16_nsa", "fa_bf16_nsa")?;
    let cu_stream = stream.cu_stream();

    let params: [*mut core::ffi::c_void; 11] = [
        &q_ptr as *const u64 as *mut _,
        &k_ptr as *const u64 as *mut _,
        &v_ptr as *const u64 as *mut _,
        &idx_ptr as *const u64 as *mut _,
        &o_ptr as *const u64 as *mut _,
        &num_heads as *const u32 as *mut _,
        &seq_q as *const u32 as *mut _,
        &seq_kv as *const u32 as *mut _,
        &k_top as *const u32 as *mut _,
        &block_size as *const u32 as *mut _,
        &scale as *const f32 as *mut _,
    ];

    let result = unsafe {
        cuLaunchKernel(
            cu_func,
            seq_q,
            num_heads,
            batch,
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
            "NSA launch: {:?}",
            result
        )));
    }
    Ok(())
}

/// Compute per-block mean of K for NSA / MoBA block scoring.
///
/// Input:  k: [B, seq_kv, H, D=128] BF16
/// Output: out: [B, num_blocks, H, D] BF16 where num_blocks = ceil(seq_kv/block_size)
#[allow(clippy::too_many_arguments)]
pub fn k_block_mean(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    k: &CudaSlice<u16>,
    out: &mut CudaSlice<u16>,
    batch: u32,
    num_heads: u32,
    seq_kv: u32,
    block_size: u32,
) -> Result<()> {
    if batch == 0 || num_heads == 0 || seq_kv == 0 || block_size == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    let num_blocks = seq_kv.div_ceil(block_size);

    use cudarc::driver::sys::*;

    let (k_ptr, _a) = k.device_ptr(stream);
    let (out_ptr, _b) = out.device_ptr(stream);

    let cu_func = module::load_kernel_raw(ctx, "k_block_mean", "k_block_mean")?;
    let cu_stream = stream.cu_stream();

    let params: [*mut core::ffi::c_void; 6] = [
        &k_ptr as *const u64 as *mut _,
        &out_ptr as *const u64 as *mut _,
        &batch as *const u32 as *mut _,
        &num_heads as *const u32 as *mut _,
        &seq_kv as *const u32 as *mut _,
        &block_size as *const u32 as *mut _,
    ];

    let result = unsafe {
        cuLaunchKernel(
            cu_func,
            num_blocks,
            num_heads,
            batch,
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
            "k_block_mean launch: {:?}",
            result
        )));
    }
    Ok(())
}

/// Select top-K block indices per query from per-block importance scores.
/// Final stage of the NSA / MoBA block-selection pipeline.
///
/// Input:  scores: [B, Sq, H, num_blocks] FP32
/// Output: indices: [B, Sq, H, K] u32 (sorted descending by score)
///
/// Constraint: this reference handles num_blocks ≤ 32.
#[allow(clippy::too_many_arguments)]
pub fn topk_block_idx(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    scores: &CudaSlice<f32>,
    indices: &mut CudaSlice<u32>,
    batch: u32,
    num_heads: u32,
    seq_q: u32,
    num_blocks: u32,
    k_top: u32,
) -> Result<()> {
    if batch == 0 || num_heads == 0 || seq_q == 0 || num_blocks == 0 || k_top == 0 {
        return Err(SparkError::InvalidArgument("all dims must be > 0".into()));
    }
    if num_blocks > 32 {
        return Err(SparkError::InvalidArgument(format!(
            "topk_block_idx reference supports num_blocks ≤ 32, got {num_blocks}"
        )));
    }
    if k_top > num_blocks {
        return Err(SparkError::InvalidArgument(format!(
            "k_top ({k_top}) must be ≤ num_blocks ({num_blocks})"
        )));
    }

    use cudarc::driver::sys::*;

    let (scores_ptr, _a) = scores.device_ptr(stream);
    let (idx_ptr, _b) = indices.device_ptr(stream);

    let cu_func = module::load_kernel_raw(ctx, "topk_block_idx", "topk_block_idx")?;
    let cu_stream = stream.cu_stream();

    let params: [*mut core::ffi::c_void; 4] = [
        &scores_ptr as *const u64 as *mut _,
        &idx_ptr as *const u64 as *mut _,
        &num_blocks as *const u32 as *mut _,
        &k_top as *const u32 as *mut _,
    ];

    let result = unsafe {
        cuLaunchKernel(
            cu_func,
            seq_q,
            num_heads,
            batch,
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
            "topk_block launch: {:?}",
            result
        )));
    }
    Ok(())
}

/// MTP draft-heads fused matmul+argmax for DeepSeek V3/V4 speculative decoding.
///
/// Input:  hidden [B, D_hidden], W [K, V, D_hidden] — K draft lm_heads
/// Output: draft [B, K] u32 (top-1 token id per head)
#[allow(clippy::too_many_arguments)]
pub fn mtp_draft_heads(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    hidden: &CudaSlice<u16>,
    w: &CudaSlice<u16>,
    draft: &mut CudaSlice<u32>,
    batch: u32,
    num_draft_heads: u32,
    vocab: u32,
    d_hidden: u32,
) -> Result<()> {
    if batch == 0 || num_draft_heads == 0 || vocab == 0 || d_hidden == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    if hidden.len() < batch as usize * d_hidden as usize {
        return Err(SparkError::InvalidArgument("hidden too small".into()));
    }
    if w.len() < num_draft_heads as usize * vocab as usize * d_hidden as usize {
        return Err(SparkError::InvalidArgument("W too small".into()));
    }
    if draft.len() < batch as usize * num_draft_heads as usize {
        return Err(SparkError::InvalidArgument("draft too small".into()));
    }

    use cudarc::driver::sys::*;

    let (h_ptr, _a) = hidden.device_ptr(stream);
    let (w_ptr, _b) = w.device_ptr(stream);
    let (d_ptr, _c) = draft.device_ptr(stream);

    let cu_func = module::load_kernel_raw(ctx, "mtp_draft_heads", "mtp_draft_heads")?;
    let cu_stream = stream.cu_stream();

    let params: [*mut core::ffi::c_void; 5] = [
        &h_ptr as *const u64 as *mut _,
        &w_ptr as *const u64 as *mut _,
        &d_ptr as *const u64 as *mut _,
        &d_hidden as *const u32 as *mut _,
        &vocab as *const u32 as *mut _,
    ];

    let result = unsafe {
        cuLaunchKernel(
            cu_func,
            num_draft_heads,
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
            "MTP launch: {:?}",
            result
        )));
    }
    Ok(())
}

/// TMA-accelerated MLA BF16 decode.
///
/// Same math + same (num_heads, batch) grid as `mla_decode_bf16` — keeps
/// the scalar reference's parallelism so the GPU stays saturated. The
/// only change is the c_kv / k_rope GMEM→SMEM path uses double-buffered
/// TMA (8-position chunks, 8 halves per chunk for D_C=512) instead of
/// per-thread scalar ld.global.v2.b32. Compute stays scalar.
#[allow(clippy::too_many_arguments)]
pub fn mla_decode_bf16_tma(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q_c: &CudaSlice<u16>,
    q_r: &CudaSlice<u16>,
    c_kv: &CudaSlice<u16>,
    k_rope: &CudaSlice<u16>,
    o: &mut CudaSlice<u16>,
    batch: u32,
    num_heads: u32,
    seq_kv: u32,
    scale: f32,
) -> Result<()> {
    if batch == 0 || num_heads == 0 || seq_kv == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }

    use cudarc::driver::sys::*;

    let (q_c_ptr, _a) = q_c.device_ptr(stream);
    let (q_r_ptr, _b) = q_r.device_ptr(stream);
    let (c_kv_ptr, _c) = c_kv.device_ptr(stream);
    let (k_rope_ptr, _d) = k_rope.device_ptr(stream);
    let (o_ptr, _e) = o.device_ptr(stream);

    let _total_q_rows = batch * num_heads;
    let total_kv_rows = batch * seq_kv;

    // TMA descriptors — NO_SWIZZLE so scalar SMEM reads see logical layout.
    // Q is scalar-loaded (one-shot, 1 KB), so no TMA descriptor for it.
    // c_kv / k_rope: tile [64 cols inner, 8 rows outer]; 8 halves per chunk.
    let c_kv_tma = create_tma_desc(c_kv_ptr as *mut _, total_kv_rows, MLA_D_C, 8, 64)?;
    let k_rope_tma = create_tma_desc(k_rope_ptr as *mut _, total_kv_rows, MLA_D_R, 8, 64)?;

    let c_kv_u32: [u32; 32] = unsafe { core::mem::transmute(c_kv_tma) };
    let k_rope_u32: [u32; 32] = unsafe { core::mem::transmute(k_rope_tma) };
    let c_kv_dev = stream.memcpy_stod(&c_kv_u32).map_err(SparkError::Driver)?;
    let k_rope_dev = stream
        .memcpy_stod(&k_rope_u32)
        .map_err(SparkError::Driver)?;

    let (c_kv_dptr, _) = c_kv_dev.device_ptr(stream);
    let (k_rope_dptr, _) = k_rope_dev.device_ptr(stream);

    let cu_func = module::load_kernel_raw(ctx, "fa_bf16_mla_decode_tma", "fa_bf16_mla_decode_tma")?;
    let cu_stream = stream.cu_stream();

    const SMEM_TOTAL: i32 = 19616;
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
        &q_c_ptr as *const u64 as *mut _,
        &q_r_ptr as *const u64 as *mut _,
        &c_kv_dptr as *const u64 as *mut _,
        &k_rope_dptr as *const u64 as *mut _,
        &o_ptr as *const u64 as *mut _,
        &num_heads as *const u32 as *mut _,
        &seq_kv as *const u32 as *mut _,
        &scale as *const f32 as *mut _,
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
            "MLA TMA launch: {:?}",
            result
        )));
    }
    Ok(())
}

/// DeepSeek Sparse Attention (DSA) — DeepSeek V3.2 production variant.
/// Each query attends to a selected set of INDIVIDUAL positions (not blocks).
///
/// Implemented as a thin wrapper over `nsa_attention_bf16` with block_size=1:
/// the "block_idx" then contains individual position indices, which matches
/// DSA's per-token indexer output exactly.
///
/// Input:
///   position_idx: [B, seq_q, H, K] u32 — selected positions per query
///   (produced by a companion per-token indexer, not yet implemented as a kernel)
#[allow(clippy::too_many_arguments)]
pub fn dsa_attention_bf16(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u16>,
    k: &CudaSlice<u16>,
    v: &CudaSlice<u16>,
    position_idx: &CudaSlice<u32>,
    o: &mut CudaSlice<u16>,
    batch: u32,
    num_heads: u32,
    seq_q: u32,
    seq_kv: u32,
    k_top: u32,
    scale: f32,
) -> Result<()> {
    // block_size=1 means each "block" is one KV position.
    nsa_attention_bf16(
        ctx,
        stream,
        q,
        k,
        v,
        position_idx,
        o,
        batch,
        num_heads,
        seq_q,
        seq_kv,
        k_top,
        1,
        scale,
    )
}

/// Medusa draft heads — same kernel as MTP draft heads.
///
/// Medusa and MTP produce K parallel lm_head projections from a shared
/// hidden state; the only difference is training-time (MTP is jointly
/// trained with the base model, Medusa heads are trained separately).
/// Inference uses identical dataflow, so we dispatch to the same kernel.
#[allow(clippy::too_many_arguments)]
pub fn medusa_draft_heads(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    hidden: &CudaSlice<u16>,
    w: &CudaSlice<u16>,
    draft: &mut CudaSlice<u32>,
    batch: u32,
    num_draft_heads: u32,
    vocab: u32,
    d_hidden: u32,
) -> Result<()> {
    mtp_draft_heads(
        ctx,
        stream,
        hidden,
        w,
        draft,
        batch,
        num_draft_heads,
        vocab,
        d_hidden,
    )
}

/// Launch weight-absorbed MLA prefill with FP8 KV cache (causal, seq_q > 1).
/// Used by DeepSeek V3 / V4 FP8 KV serving during prefill.
#[allow(clippy::too_many_arguments)]
pub fn mla_prefill_fp8(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q_c: &CudaSlice<u16>,
    q_r: &CudaSlice<u16>,
    c_kv: &CudaSlice<u8>,
    k_rope: &CudaSlice<u8>,
    o: &mut CudaSlice<u16>,
    batch: u32,
    num_heads: u32,
    seq_q: u32,
    seq_kv: u32,
    scale: f32,
    kv_scale: f32,
) -> Result<()> {
    if batch == 0 || num_heads == 0 || seq_q == 0 || seq_kv == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }

    use cudarc::driver::sys::*;

    let (q_c_ptr, _a) = q_c.device_ptr(stream);
    let (q_r_ptr, _b) = q_r.device_ptr(stream);
    let (c_kv_ptr, _c) = c_kv.device_ptr(stream);
    let (k_rope_ptr, _d) = k_rope.device_ptr(stream);
    let (o_ptr, _e) = o.device_ptr(stream);

    let cu_func = module::load_kernel_raw(ctx, "fa_fp8_mla_prefill", "fa_fp8_mla_prefill")?;
    let cu_stream = stream.cu_stream();

    let params: [*mut core::ffi::c_void; 10] = [
        &q_c_ptr as *const u64 as *mut _,
        &q_r_ptr as *const u64 as *mut _,
        &c_kv_ptr as *const u64 as *mut _,
        &k_rope_ptr as *const u64 as *mut _,
        &o_ptr as *const u64 as *mut _,
        &num_heads as *const u32 as *mut _,
        &seq_q as *const u32 as *mut _,
        &seq_kv as *const u32 as *mut _,
        &scale as *const f32 as *mut _,
        &kv_scale as *const f32 as *mut _,
    ];

    let result = unsafe {
        cuLaunchKernel(
            cu_func,
            seq_q,
            num_heads,
            batch,
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
            "MLA FP8 prefill launch: {:?}",
            result
        )));
    }
    Ok(())
}

/// Combine partial MLA outputs from split-K decode via log-sum-exp.
///
/// For each (b, h, d):
///   weight[i] = exp(lse[i] - max_i lse[i])
///   o_final[b,h,d] = (Σ weight[i] * o_partial[i,b,h,d]) / (Σ weight[i])
///
/// Used by FlashDecoding-style MLA (long-context / multi-DGX-Spark ring
/// attention) to reduce per-split partials into the final output.
///
/// Input:
///   o_partial: [num_splits, B, H, D_C=512] FP32
///   lse: [num_splits, B, H] FP32
/// Output:
///   o_final: [B, H, D_C] BF16
///
/// Constraint: num_splits ≤ 64 (fits the SMEM scratch).
#[allow(clippy::too_many_arguments)]
pub fn mla_split_kv_combine(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    o_partial: &CudaSlice<f32>,
    lse: &CudaSlice<f32>,
    o_final: &mut CudaSlice<u16>,
    batch: u32,
    num_heads: u32,
    num_splits: u32,
) -> Result<()> {
    if batch == 0 || num_heads == 0 || num_splits == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    if num_splits > 64 {
        return Err(SparkError::InvalidArgument(format!(
            "mla_split_kv_combine supports up to 64 splits (got {num_splits})"
        )));
    }

    use cudarc::driver::sys::*;

    let (o_p_ptr, _a) = o_partial.device_ptr(stream);
    let (l_ptr, _b) = lse.device_ptr(stream);
    let (o_f_ptr, _c) = o_final.device_ptr(stream);

    let cu_func = module::load_kernel_raw(ctx, "mla_split_kv_combine", "mla_split_kv_combine")?;
    let cu_stream = stream.cu_stream();

    let params: [*mut core::ffi::c_void; 5] = [
        &o_p_ptr as *const u64 as *mut _,
        &l_ptr as *const u64 as *mut _,
        &o_f_ptr as *const u64 as *mut _,
        &num_splits as *const u32 as *mut _,
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
            "MLA combine launch: {:?}",
            result
        )));
    }
    Ok(())
}

// NSA TMA and MLA prefill TMA retrofits were removed — both ran slower than
// scalar at production configs (0.53-0.88× scalar). Per-CTA transfer volumes
// were too modest to amortize the TMA issue overhead. MLA decode TMA stays
// in (2.5× win) because its per-CTA volume covers all seq_kv.
// See `docs/remaining_work.md` for the documented TMA rule of thumb.

/// Auto-dispatch with manual override: callers who have benchmarked can pin
/// `num_splits` directly. Use `Some(n)` to force split-K with n splits,
/// `Some(1)` to force the no-split MMA+TMA+PQ path, `None` for heuristic.
#[allow(clippy::too_many_arguments)]
pub fn mla_decode_bf16_auto_with_splits(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q_c: &CudaSlice<u16>,
    q_r: &CudaSlice<u16>,
    c_kv: &CudaSlice<u16>,
    k_rope: &CudaSlice<u16>,
    o: &mut CudaSlice<u16>,
    batch: u32,
    num_heads: u32,
    seq_kv: u32,
    scale: f32,
    num_splits: Option<u32>,
) -> Result<()> {
    if !num_heads.is_multiple_of(16) {
        return mla_decode_bf16(
            ctx, stream, q_c, q_r, c_kv, k_rope, o, batch, num_heads, seq_kv, scale,
        );
    }
    let num_splits = match num_splits {
        Some(n) => n,
        None => {
            if seq_kv < 64 {
                1
            } else {
                let base_ctas = (num_heads / 16) * batch;
                let target = if base_ctas >= 48 {
                    2
                } else {
                    (48 / base_ctas.max(1)).max(2)
                };
                let num_chunks = seq_kv.div_ceil(8);
                target.min(num_chunks).min(16)
            }
        }
    };
    if num_splits <= 1 {
        return mla_decode_bf16_mma_tma_pq(
            ctx, stream, q_c, q_r, c_kv, k_rope, o, batch, num_heads, seq_kv, scale,
        );
    }
    mla_decode_bf16_mma_split(
        ctx, stream, q_c, q_r, c_kv, k_rope, o, batch, num_heads, seq_kv, num_splits, scale,
    )
}

/// Auto-dispatch: picks the fastest BF16 MLA decode variant for the given dims.
///
/// Decision tree (tuned from measured benchmarks on SM121a):
///   - num_heads % 16 != 0          → scalar `mla_decode_bf16`
///   - seq_kv < 64                  → `mla_decode_bf16_mma_tma` (split doesn't amortize)
///   - otherwise                    → `mla_decode_bf16_mma_split` with num_splits tuned
///                                    to target ~48 CTAs on SM121a's 48 SMs:
///                                    num_splits = clamp(48 / (H/16 * B), 2, 16)
///                                    clamped to num_chunks = ceil(seq_kv/8).
///
/// Matches the split-K sweet spots found in `examples/mla_tma_bench.rs` runs.
#[allow(clippy::too_many_arguments)]
pub fn mla_decode_bf16_auto(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q_c: &CudaSlice<u16>,
    q_r: &CudaSlice<u16>,
    c_kv: &CudaSlice<u16>,
    k_rope: &CudaSlice<u16>,
    o: &mut CudaSlice<u16>,
    batch: u32,
    num_heads: u32,
    seq_kv: u32,
    scale: f32,
) -> Result<()> {
    if !num_heads.is_multiple_of(16) {
        return mla_decode_bf16(
            ctx, stream, q_c, q_r, c_kv, k_rope, o, batch, num_heads, seq_kv, scale,
        );
    }
    if seq_kv < 64 {
        // PQ variant matches MMA+TMA within noise at tiny configs; use it uniformly
        // in the no-split path for a small but consistent gain at batched scales.
        return mla_decode_bf16_mma_tma_pq(
            ctx, stream, q_c, q_r, c_kv, k_rope, o, batch, num_heads, seq_kv, scale,
        );
    }
    // 48 SMs on SM121a; aim for ~48 active CTAs.
    let base_ctas = (num_heads / 16) * batch;
    let target_splits = if base_ctas >= 48 {
        2
    } else {
        (48 / base_ctas.max(1)).max(2)
    };
    let num_chunks = seq_kv.div_ceil(8);
    let num_splits = target_splits.min(num_chunks).min(16);
    if num_splits <= 1 {
        return mla_decode_bf16_mma_tma_pq(
            ctx, stream, q_c, q_r, c_kv, k_rope, o, batch, num_heads, seq_kv, scale,
        );
    }
    mla_decode_bf16_mma_split(
        ctx, stream, q_c, q_r, c_kv, k_rope, o, batch, num_heads, seq_kv, num_splits, scale,
    )
}

/// Split-K MMA+TMA BF16 MLA decode. Grid: (H/16, B, num_splits).
/// Outputs partial O (FP32) + LSE per split, then reduces via `mla_split_kv_combine`.
#[allow(clippy::too_many_arguments)]
pub fn mla_decode_bf16_mma_split(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q_c: &CudaSlice<u16>,
    q_r: &CudaSlice<u16>,
    c_kv: &CudaSlice<u16>,
    k_rope: &CudaSlice<u16>,
    o: &mut CudaSlice<u16>,
    batch: u32,
    num_heads: u32,
    seq_kv: u32,
    num_splits: u32,
    scale: f32,
) -> Result<()> {
    if batch == 0 || num_heads == 0 || seq_kv == 0 || num_splits == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    if !num_heads.is_multiple_of(16) {
        return Err(SparkError::InvalidArgument(format!(
            "mla_decode_bf16_mma_split requires num_heads % 16 == 0 (got {num_heads})"
        )));
    }
    if num_splits > 64 {
        return Err(SparkError::InvalidArgument(format!(
            "num_splits must be ≤ 64 (got {num_splits})"
        )));
    }

    use cudarc::driver::sys::*;
    let (q_c_ptr, _a) = q_c.device_ptr(stream);
    let (q_r_ptr, _b) = q_r.device_ptr(stream);
    let (c_kv_ptr, _c) = c_kv.device_ptr(stream);
    let (k_rope_ptr, _d) = k_rope.device_ptr(stream);
    let o_ptr: u64 = {
        let (p, _) = o.device_ptr(stream);
        p
    };

    let total_kv_rows = batch * seq_kv;
    let c_kv_tma = create_tma_desc_swizzle(c_kv_ptr as *mut _, total_kv_rows, MLA_D_C, 8, 64)?;
    let k_rope_tma = create_tma_desc_swizzle(k_rope_ptr as *mut _, total_kv_rows, MLA_D_R, 8, 64)?;
    let c_kv_u32: [u32; 32] = unsafe { core::mem::transmute(c_kv_tma) };
    let k_rope_u32: [u32; 32] = unsafe { core::mem::transmute(k_rope_tma) };
    let c_kv_dev = stream.memcpy_stod(&c_kv_u32).map_err(SparkError::Driver)?;
    let k_rope_dev = stream
        .memcpy_stod(&k_rope_u32)
        .map_err(SparkError::Driver)?;
    let (c_kv_dptr, _) = c_kv_dev.device_ptr(stream);
    let (k_rope_dptr, _) = k_rope_dev.device_ptr(stream);

    let partial_len = (num_splits * batch * num_heads) as usize * MLA_D_C as usize;
    let lse_len = (num_splits * batch * num_heads) as usize;
    let partial_o = stream
        .alloc_zeros::<f32>(partial_len)
        .map_err(SparkError::Driver)?;
    let lse = stream
        .alloc_zeros::<f32>(lse_len)
        .map_err(SparkError::Driver)?;
    let (po_ptr, _pp) = partial_o.device_ptr(stream);
    let (lse_ptr, _ll) = lse.device_ptr(stream);

    let cu_func = module::load_kernel_raw(
        ctx,
        "fa_bf16_mla_decode_mma_tma_split",
        "fa_bf16_mla_decode_mma_tma_split",
    )?;
    let cu_stream = stream.cu_stream();

    const SMEM_TOTAL: i32 = 37216;
    let attr = unsafe {
        cuFuncSetAttribute(
            cu_func,
            CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            SMEM_TOTAL,
        )
    };
    if attr != CUresult::CUDA_SUCCESS {
        return Err(SparkError::LaunchFailed(format!("attr: {:?}", attr)));
    }

    let params: [*mut core::ffi::c_void; 11] = [
        &q_c_ptr as *const u64 as *mut _,
        &q_r_ptr as *const u64 as *mut _,
        &c_kv_dptr as *const u64 as *mut _,
        &k_rope_dptr as *const u64 as *mut _,
        &o_ptr as *const u64 as *mut _, // unused by split kernel but declared in sig
        &po_ptr as *const u64 as *mut _,
        &lse_ptr as *const u64 as *mut _,
        &num_heads as *const u32 as *mut _,
        &seq_kv as *const u32 as *mut _,
        &num_splits as *const u32 as *mut _,
        &scale as *const f32 as *mut _,
    ];

    let head_groups = num_heads / 16;
    let r = unsafe {
        cuLaunchKernel(
            cu_func,
            head_groups,
            batch,
            num_splits,
            128,
            1,
            1,
            SMEM_TOTAL as u32,
            cu_stream,
            params.as_ptr() as *mut _,
            core::ptr::null_mut(),
        )
    };
    if r != CUresult::CUDA_SUCCESS {
        return Err(SparkError::LaunchFailed(format!("split launch: {:?}", r)));
    }

    // Combine partials → final BF16 O.
    mla_split_kv_combine(
        ctx, stream, &partial_o, &lse, o, batch, num_heads, num_splits,
    )
}

/// MMA + TMA + persistent Q_r. Identical to `mla_decode_bf16_mma_tma` but
/// hoists the 4 Q_r ldmatrix calls out of the chunk loop (Q_r data is
/// loop-invariant). 20 extra persistent registers, no spills, same
/// occupancy.
#[allow(clippy::too_many_arguments)]
pub fn mla_decode_bf16_mma_tma_pq(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q_c: &CudaSlice<u16>,
    q_r: &CudaSlice<u16>,
    c_kv: &CudaSlice<u16>,
    k_rope: &CudaSlice<u16>,
    o: &mut CudaSlice<u16>,
    batch: u32,
    num_heads: u32,
    seq_kv: u32,
    scale: f32,
) -> Result<()> {
    if !num_heads.is_multiple_of(16) {
        return Err(SparkError::InvalidArgument(format!(
            "mla_decode_bf16_mma_tma_pq requires num_heads % 16 == 0 (got {num_heads})"
        )));
    }
    use cudarc::driver::sys::*;
    let (q_c_ptr, _a) = q_c.device_ptr(stream);
    let (q_r_ptr, _b) = q_r.device_ptr(stream);
    let (c_kv_ptr, _c) = c_kv.device_ptr(stream);
    let (k_rope_ptr, _d) = k_rope.device_ptr(stream);
    let (o_ptr, _e) = o.device_ptr(stream);

    let total_kv_rows = batch * seq_kv;
    let c_kv_tma = create_tma_desc_swizzle(c_kv_ptr as *mut _, total_kv_rows, MLA_D_C, 8, 64)?;
    let k_rope_tma = create_tma_desc_swizzle(k_rope_ptr as *mut _, total_kv_rows, MLA_D_R, 8, 64)?;
    let c_kv_u32: [u32; 32] = unsafe { core::mem::transmute(c_kv_tma) };
    let k_rope_u32: [u32; 32] = unsafe { core::mem::transmute(k_rope_tma) };
    let c_kv_dev = stream.memcpy_stod(&c_kv_u32).map_err(SparkError::Driver)?;
    let k_rope_dev = stream
        .memcpy_stod(&k_rope_u32)
        .map_err(SparkError::Driver)?;
    let (c_kv_dptr, _) = c_kv_dev.device_ptr(stream);
    let (k_rope_dptr, _) = k_rope_dev.device_ptr(stream);

    let cu_func = module::load_kernel_raw(
        ctx,
        "fa_bf16_mla_decode_mma_tma_pq",
        "fa_bf16_mla_decode_mma_tma_pq",
    )?;
    let cu_stream = stream.cu_stream();

    const SMEM_TOTAL: i32 = 37216;
    let attr = unsafe {
        cuFuncSetAttribute(
            cu_func,
            CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            SMEM_TOTAL,
        )
    };
    if attr != CUresult::CUDA_SUCCESS {
        return Err(SparkError::LaunchFailed(format!("attr: {:?}", attr)));
    }

    let params: [*mut core::ffi::c_void; 8] = [
        &q_c_ptr as *const u64 as *mut _,
        &q_r_ptr as *const u64 as *mut _,
        &c_kv_dptr as *const u64 as *mut _,
        &k_rope_dptr as *const u64 as *mut _,
        &o_ptr as *const u64 as *mut _,
        &num_heads as *const u32 as *mut _,
        &seq_kv as *const u32 as *mut _,
        &scale as *const f32 as *mut _,
    ];

    let head_groups = num_heads / 16;
    let r = unsafe {
        cuLaunchKernel(
            cu_func,
            head_groups,
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
    if r != CUresult::CUDA_SUCCESS {
        return Err(SparkError::LaunchFailed(format!("pq launch: {:?}", r)));
    }
    Ok(())
}

/// MMA + TMA double-buffered BF16 MLA decode — prefetches chunk N+1 while
/// compute runs on chunk N. Uses SWIZZLE_128B TMA descriptors so the ldmatrix
/// layout matches. Requires num_heads % 16 == 0.
#[allow(clippy::too_many_arguments)]
pub fn mla_decode_bf16_mma_tma(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q_c: &CudaSlice<u16>,
    q_r: &CudaSlice<u16>,
    c_kv: &CudaSlice<u16>,
    k_rope: &CudaSlice<u16>,
    o: &mut CudaSlice<u16>,
    batch: u32,
    num_heads: u32,
    seq_kv: u32,
    scale: f32,
) -> Result<()> {
    if batch == 0 || num_heads == 0 || seq_kv == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    if !num_heads.is_multiple_of(16) {
        return Err(SparkError::InvalidArgument(format!(
            "mla_decode_bf16_mma_tma requires num_heads % 16 == 0 (got {num_heads})"
        )));
    }

    use cudarc::driver::sys::*;

    let (q_c_ptr, _a) = q_c.device_ptr(stream);
    let (q_r_ptr, _b) = q_r.device_ptr(stream);
    let (c_kv_ptr, _c) = c_kv.device_ptr(stream);
    let (k_rope_ptr, _d) = k_rope.device_ptr(stream);
    let (o_ptr, _e) = o.device_ptr(stream);

    let total_kv_rows = batch * seq_kv;
    // SWIZZLE_128B descriptors — ldmatrix in the MMA kernel expects the
    // XOR-swizzled layout. tile_cols must be 64 (= 128 B = one swizzle row).
    let c_kv_tma = create_tma_desc_swizzle(c_kv_ptr as *mut _, total_kv_rows, MLA_D_C, 8, 64)?;
    let k_rope_tma = create_tma_desc_swizzle(k_rope_ptr as *mut _, total_kv_rows, MLA_D_R, 8, 64)?;
    let c_kv_u32: [u32; 32] = unsafe { core::mem::transmute(c_kv_tma) };
    let k_rope_u32: [u32; 32] = unsafe { core::mem::transmute(k_rope_tma) };
    let c_kv_dev = stream.memcpy_stod(&c_kv_u32).map_err(SparkError::Driver)?;
    let k_rope_dev = stream
        .memcpy_stod(&k_rope_u32)
        .map_err(SparkError::Driver)?;
    let (c_kv_dptr, _) = c_kv_dev.device_ptr(stream);
    let (k_rope_dptr, _) = k_rope_dev.device_ptr(stream);

    let cu_func = module::load_kernel_raw(
        ctx,
        "fa_bf16_mla_decode_mma_tma",
        "fa_bf16_mla_decode_mma_tma",
    )?;
    let cu_stream = stream.cu_stream();

    const SMEM_TOTAL: i32 = 37216;
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

    let params: [*mut core::ffi::c_void; 8] = [
        &q_c_ptr as *const u64 as *mut _,
        &q_r_ptr as *const u64 as *mut _,
        &c_kv_dptr as *const u64 as *mut _,
        &k_rope_dptr as *const u64 as *mut _,
        &o_ptr as *const u64 as *mut _,
        &num_heads as *const u32 as *mut _,
        &seq_kv as *const u32 as *mut _,
        &scale as *const f32 as *mut _,
    ];

    let head_groups = num_heads / 16;
    let result = unsafe {
        cuLaunchKernel(
            cu_func,
            head_groups,
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
            "MMA TMA launch: {:?}",
            result
        )));
    }
    Ok(())
}

/// MMA-accelerated BF16 MLA decode.
///
/// 16 heads per CTA, tensor core QK (m16n8k16) + PV (m16n8k8) with
/// SWIZZLE_128B SMEM + ldmatrix. v1: single-buffered, scalar loads, no
/// warp specialization, no split-K. Requires num_heads % 16 == 0.
#[allow(clippy::too_many_arguments)]
pub fn mla_decode_bf16_mma(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q_c: &CudaSlice<u16>,
    q_r: &CudaSlice<u16>,
    c_kv: &CudaSlice<u16>,
    k_rope: &CudaSlice<u16>,
    o: &mut CudaSlice<u16>,
    batch: u32,
    num_heads: u32,
    seq_kv: u32,
    scale: f32,
) -> Result<()> {
    if batch == 0 || num_heads == 0 || seq_kv == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    if !num_heads.is_multiple_of(16) {
        return Err(SparkError::InvalidArgument(format!(
            "mla_decode_bf16_mma requires num_heads % 16 == 0 (got {num_heads})"
        )));
    }

    use cudarc::driver::sys::*;

    let (q_c_ptr, _a) = q_c.device_ptr(stream);
    let (q_r_ptr, _b) = q_r.device_ptr(stream);
    let (c_kv_ptr, _c) = c_kv.device_ptr(stream);
    let (k_rope_ptr, _d) = k_rope.device_ptr(stream);
    let (o_ptr, _e) = o.device_ptr(stream);

    let cu_func = module::load_kernel_raw(ctx, "fa_bf16_mla_decode_mma", "fa_bf16_mla_decode_mma")?;
    let cu_stream = stream.cu_stream();

    const SMEM_TOTAL: i32 = 27968;
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

    let params: [*mut core::ffi::c_void; 8] = [
        &q_c_ptr as *const u64 as *mut _,
        &q_r_ptr as *const u64 as *mut _,
        &c_kv_ptr as *const u64 as *mut _,
        &k_rope_ptr as *const u64 as *mut _,
        &o_ptr as *const u64 as *mut _,
        &num_heads as *const u32 as *mut _,
        &seq_kv as *const u32 as *mut _,
        &scale as *const f32 as *mut _,
    ];

    let head_groups = num_heads / 16;
    let result = unsafe {
        cuLaunchKernel(
            cu_func,
            head_groups,
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
            "MLA MMA launch: {:?}",
            result
        )));
    }
    Ok(())
}

/// QK-ONLY validation entry point for the MMA-MLA decode rewrite.
///
/// Runs the Q_c @ c_kv^T + Q_r @ k_rope^T path on tensor cores (m16n8k16)
/// with 16 heads per CTA and writes the scaled scores to `scores` for
/// byte-exact comparison against a PyTorch reference. This is a test
/// harness, NOT a shipping attention kernel — no softmax, no PV.
/// Restricted to num_heads % 16 == 0.
#[allow(clippy::too_many_arguments)]
pub fn mla_decode_bf16_mma_qk(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q_c: &CudaSlice<u16>,
    q_r: &CudaSlice<u16>,
    c_kv: &CudaSlice<u16>,
    k_rope: &CudaSlice<u16>,
    scores: &mut CudaSlice<f32>,
    batch: u32,
    num_heads: u32,
    seq_kv: u32,
    scale: f32,
) -> Result<()> {
    if batch == 0 || num_heads == 0 || seq_kv == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    if !num_heads.is_multiple_of(16) {
        return Err(SparkError::InvalidArgument(format!(
            "mla_decode_bf16_mma_qk requires num_heads % 16 == 0 (got {num_heads})"
        )));
    }

    use cudarc::driver::sys::*;

    let (q_c_ptr, _a) = q_c.device_ptr(stream);
    let (q_r_ptr, _b) = q_r.device_ptr(stream);
    let (c_kv_ptr, _c) = c_kv.device_ptr(stream);
    let (k_rope_ptr, _d) = k_rope.device_ptr(stream);
    let (s_ptr, _e) = scores.device_ptr(stream);

    let cu_func = module::load_kernel_raw(
        ctx,
        "fa_bf16_mla_decode_mma_qk",
        "fa_bf16_mla_decode_mma_qk",
    )?;
    let cu_stream = stream.cu_stream();

    const SMEM_TOTAL: i32 = 28160;
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

    let params: [*mut core::ffi::c_void; 8] = [
        &q_c_ptr as *const u64 as *mut _,
        &q_r_ptr as *const u64 as *mut _,
        &c_kv_ptr as *const u64 as *mut _,
        &k_rope_ptr as *const u64 as *mut _,
        &s_ptr as *const u64 as *mut _,
        &num_heads as *const u32 as *mut _,
        &seq_kv as *const u32 as *mut _,
        &scale as *const f32 as *mut _,
    ];

    let head_groups = num_heads / 16;
    let result = unsafe {
        cuLaunchKernel(
            cu_func,
            head_groups,
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
            "MMA QK launch: {:?}",
            result
        )));
    }
    Ok(())
}

/// Warp-specialized 4-warp FP8 MMA MLA decode (v3). Same FP8 MMA QK as v2 but
/// distributes PV (64 N-tiles for D_C=512) and final normalize (512 cols)
/// across 4 warps. Targets long-Skv workloads where v1's 4-warp parallelism
/// dominated v2's single-warp throughput.
#[allow(clippy::too_many_arguments)]
pub fn fa_fp8_mla_decode_mma_v3(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q_c: &CudaSlice<u8>,
    q_r: &CudaSlice<u8>,
    c_kv: &CudaSlice<u8>,
    k_rope: &CudaSlice<u8>,
    o: &mut CudaSlice<u16>,
    batch: u32,
    num_heads: u32,
    seq_kv: u32,
    scale: f32,
    q_scale: f32,
    kv_scale: f32,
) -> Result<()> {
    if batch == 0 || num_heads == 0 || seq_kv == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    if !num_heads.is_multiple_of(16) {
        return Err(SparkError::InvalidArgument(format!(
            "num_heads must be multiple of 16 (got {num_heads})"
        )));
    }
    let func = module::load_kernel(ctx, "fa_fp8_mla_decode_mma_v3", "fa_fp8_mla_decode_mma_v3")?;
    let cfg = LaunchConfig {
        grid_dim: (num_heads / 16, batch, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(q_c)
            .arg(q_r)
            .arg(c_kv)
            .arg(k_rope)
            .arg(o)
            .arg(&num_heads)
            .arg(&seq_kv)
            .arg(&scale)
            .arg(&q_scale)
            .arg(&kv_scale)
            .launch(cfg)?;
    }
    Ok(())
}

/// Auto-dispatch FP8 MLA decode: routes to v3 (warp-specialized 4-warp FP8 MMA)
/// which dominates v1 (BF16-MMA hybrid) and v2 (single-warp FP8 MMA) at every
/// Skv tested (1.13×–4.5× faster). v1 BF16 Q args retained for API compatibility
/// but unused in current routing.
///
/// Bench (B=1, H=16, Skv=128/512/2048):
///   v1: 254 / 323 / 1006 μs  (BF16-MMA hybrid, 4-warp)
///   v2:  98 / 363 / 1596 μs  (FP8 MMA, single-warp)
///   v3:  56 / 208 /  891 μs  (FP8 MMA, 4-warp warp-specialized) ← always wins
#[allow(clippy::too_many_arguments)]
pub fn fa_fp8_mla_decode_auto(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    _q_c_bf16: &CudaSlice<u16>, // unused (kept for API stability)
    _q_r_bf16: &CudaSlice<u16>,
    q_c_fp8: &CudaSlice<u8>,
    q_r_fp8: &CudaSlice<u8>,
    c_kv: &CudaSlice<u8>,
    k_rope: &CudaSlice<u8>,
    o: &mut CudaSlice<u16>,
    batch: u32,
    num_heads: u32,
    seq_kv: u32,
    scale: f32,
    q_scale: f32,
    kv_scale: f32,
) -> Result<()> {
    fa_fp8_mla_decode_mma_v3(
        ctx, stream, q_c_fp8, q_r_fp8, c_kv, k_rope, o, batch, num_heads, seq_kv, scale, q_scale,
        kv_scale,
    )
}

/// True FP8 MMA MLA decode (v2 PoC). Q_c, Q_r, c_kv, k_rope all FP8 E4M3.
/// Uses validated FP8 MMA m16n8k32 pattern for QK. PV is NOT yet integrated —
/// this PoC writes QK scores to the output for chunk 0 only, to validate the
/// FP8 MMA QK path in MLA structure end-to-end. Full PV is the next iteration.
///
/// Layout:
///  - `q_c`: [B, H, 512] FP8
///  - `q_r`: [B, H, 64]  FP8
///  - `c_kv`: [B, Skv, 512] FP8
///  - `k_rope`: [B, Skv, 64] FP8
///  - `o`: [B, H, 512] BF16 (only first 8 cols valid as QK scores in this PoC)
///  - `scale`, `q_scale`, `kv_scale`: combined post-MMA
///
/// Constraint: num_heads % 16 == 0
#[allow(clippy::too_many_arguments)]
pub fn fa_fp8_mla_decode_mma_v2(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q_c: &CudaSlice<u8>,
    q_r: &CudaSlice<u8>,
    c_kv: &CudaSlice<u8>,
    k_rope: &CudaSlice<u8>,
    o: &mut CudaSlice<u16>,
    batch: u32,
    num_heads: u32,
    seq_kv: u32,
    scale: f32,
    q_scale: f32,
    kv_scale: f32,
) -> Result<()> {
    if batch == 0 || num_heads == 0 || seq_kv == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    if !num_heads.is_multiple_of(16) {
        return Err(SparkError::InvalidArgument(format!(
            "num_heads must be multiple of 16 (got {num_heads})"
        )));
    }
    let func = module::load_kernel(ctx, "fa_fp8_mla_decode_mma_v2", "fa_fp8_mla_decode_mma_v2")?;
    let cfg = LaunchConfig {
        grid_dim: (num_heads / 16, batch, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(q_c)
            .arg(q_r)
            .arg(c_kv)
            .arg(k_rope)
            .arg(o)
            .arg(&num_heads)
            .arg(&seq_kv)
            .arg(&scale)
            .arg(&q_scale)
            .arg(&kv_scale)
            .launch(cfg)?;
    }
    Ok(())
}

/// GQA variant of `fa_bf16_fp8kv_decode_d128`. Multiple query heads share each
/// KV head (`num_q_heads % num_kv_heads == 0`). Maps:
///   `kv_head = q_head / (num_q_heads / num_kv_heads)`
///
/// Layout:
///  - `q`: [B, num_q_heads, 128] BF16
///  - `k`, `v`: [B, num_kv_heads, Skv, 128] FP8 E4M3
///  - `o`: [B, num_q_heads, 128] BF16
#[allow(clippy::too_many_arguments)]
pub fn fa_bf16_fp8kv_decode_d128_gqa(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u16>,
    k: &CudaSlice<u8>,
    v: &CudaSlice<u8>,
    o: &mut CudaSlice<u16>,
    batch: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    seq_kv: u32,
    kv_stride: u32,
    scale: f32,
    kv_scale: f32,
) -> Result<()> {
    if batch == 0 || num_q_heads == 0 || num_kv_heads == 0 || seq_kv == 0 || kv_stride == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    if !num_q_heads.is_multiple_of(num_kv_heads) {
        return Err(SparkError::InvalidArgument(format!(
            "num_q_heads ({num_q_heads}) must be multiple of num_kv_heads ({num_kv_heads})"
        )));
    }
    if seq_kv > kv_stride {
        return Err(SparkError::InvalidArgument(format!(
            "seq_kv ({seq_kv}) > kv_stride ({kv_stride})"
        )));
    }
    let need_qo = (batch * num_q_heads * 128) as usize;
    if q.len() < need_qo || o.len() < need_qo {
        return Err(SparkError::InvalidArgument("q/o buffer too small".into()));
    }
    let need_kv = (batch * num_kv_heads * kv_stride * 128) as usize;
    if k.len() < need_kv || v.len() < need_kv {
        return Err(SparkError::InvalidArgument("k/v buffer too small".into()));
    }
    let func = module::load_kernel(
        ctx,
        "fa_bf16_fp8kv_decode_d128_gqa",
        "fa_bf16_fp8kv_decode_d128_gqa",
    )?;
    let cfg = LaunchConfig {
        grid_dim: (num_q_heads, batch, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(q)
            .arg(k)
            .arg(v)
            .arg(o)
            .arg(&num_q_heads)
            .arg(&num_kv_heads)
            .arg(&seq_kv)
            .arg(&kv_stride)
            .arg(&scale)
            .arg(&kv_scale)
            .launch(cfg)?;
    }
    Ok(())
}

/// d=256 GQA flash attention DECODE with FP8 e4m3 KV cache + per-tensor
/// `kv_scale` + sliding window + position from device pointer.
///
/// Designed for Gemma-4-style sliding-attention layers where head_dim=256
/// and the KV cache is stored as FP8 to halve attention bandwidth.
///
/// Layout:
///  - `q`: [B, num_q_heads, 256] BF16
///  - `k`, `v`: [B, num_kv_heads, kv_stride, 256] FP8 E4M3
///  - `o`: [B, num_q_heads, 256] BF16
///  - `pos_ptr`: device pointer to a single u32 (current q position)
///  - `sliding_window`: 0 = full attention; >0 = clamp k_min to
///    `max(0, q_pos + 1 - sliding_window)`
///
/// Internally computes `seq_kv = *pos_ptr + 1`. Caller must ensure
/// `(*pos_ptr) + 1 <= kv_stride`.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_bf16_fp8kv_decode_d256_gqa_pos_dev(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u16>,
    k: &CudaSlice<u8>,
    v: &CudaSlice<u8>,
    o: &mut CudaSlice<u16>,
    batch: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    kv_stride: u32,
    pos_ptr: &CudaSlice<u32>,
    sliding_window: u32,
    scale: f32,
    kv_scale: f32,
) -> Result<()> {
    const D: u32 = 256;
    if batch == 0 || num_q_heads == 0 || num_kv_heads == 0 || kv_stride == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    if !num_q_heads.is_multiple_of(num_kv_heads) {
        return Err(SparkError::InvalidArgument(format!(
            "num_q_heads ({num_q_heads}) must be multiple of num_kv_heads ({num_kv_heads})"
        )));
    }
    if (pos_ptr.len() as u32) < batch {
        // The kernel indexes pos_ptr[blockIdx.z] for blockIdx.z in 0..batch, so
        // a shorter buffer is an out-of-bounds read, not just "empty".
        return Err(SparkError::InvalidArgument(format!(
            "pos_ptr too small: {} < batch {batch}",
            pos_ptr.len()
        )));
    }
    let q_need = batch as usize * num_q_heads as usize * D as usize;
    let kv_need = batch as usize * num_kv_heads as usize * kv_stride as usize * D as usize;
    if q.len() < q_need || o.len() < q_need {
        return Err(SparkError::InvalidArgument("q/o buffer too small".into()));
    }
    if k.len() < kv_need || v.len() < kv_need {
        return Err(SparkError::InvalidArgument("k/v buffer too small".into()));
    }
    let func = module::load_kernel(
        ctx,
        "fa_bf16_fp8kv_decode_d256_gqa_pos_dev",
        "fa_bf16_fp8kv_decode_d256_gqa_pos_dev",
    )?;
    let cfg = LaunchConfig {
        grid_dim: (num_q_heads, batch, 1),
        block_dim: (D, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(q)
            .arg(k)
            .arg(v)
            .arg(o)
            .arg(&num_q_heads)
            .arg(&num_kv_heads)
            .arg(&kv_stride)
            .arg(pos_ptr)
            .arg(&sliding_window)
            .arg(&scale)
            .arg(&kv_scale)
            .launch(cfg)?;
    }
    Ok(())
}

/// d=512 GQA flash attention DECODE with FP8 e4m3 KV cache + per-tensor
/// `kv_scale` + position from device pointer. **Full attention** (no
/// sliding window) — designed for Gemma-4 full-attention layers.
///
/// Layout:
///  - `q`: [B, num_q_heads, 512] BF16
///  - `k`, `v`: [B, num_kv_heads, kv_stride, 512] FP8 E4M3
///  - `o`: [B, num_q_heads, 512] BF16
///  - `pos_ptr`: device pointer to a single u32 (current q position)
///
/// Internally computes `seq_kv = *pos_ptr + 1`. Caller must ensure
/// `(*pos_ptr) + 1 <= kv_stride`. 512 threads per block (16 warps).
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_bf16_fp8kv_decode_d512_gqa_pos_dev(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u16>,
    k: &CudaSlice<u8>,
    v: &CudaSlice<u8>,
    o: &mut CudaSlice<u16>,
    batch: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    kv_stride: u32,
    pos_ptr: &CudaSlice<u32>,
    scale: f32,
    kv_scale: f32,
) -> Result<()> {
    const D: u32 = 512;
    if batch == 0 || num_q_heads == 0 || num_kv_heads == 0 || kv_stride == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    if !num_q_heads.is_multiple_of(num_kv_heads) {
        return Err(SparkError::InvalidArgument(format!(
            "num_q_heads ({num_q_heads}) must be multiple of num_kv_heads ({num_kv_heads})"
        )));
    }
    if (pos_ptr.len() as u32) < batch {
        // The kernel indexes pos_ptr[blockIdx.z] for blockIdx.z in 0..batch, so
        // a shorter buffer is an out-of-bounds read, not just "empty".
        return Err(SparkError::InvalidArgument(format!(
            "pos_ptr too small: {} < batch {batch}",
            pos_ptr.len()
        )));
    }
    let q_need = batch as usize * num_q_heads as usize * D as usize;
    let kv_need = batch as usize * num_kv_heads as usize * kv_stride as usize * D as usize;
    if q.len() < q_need || o.len() < q_need {
        return Err(SparkError::InvalidArgument("q/o buffer too small".into()));
    }
    if k.len() < kv_need || v.len() < kv_need {
        return Err(SparkError::InvalidArgument("k/v buffer too small".into()));
    }
    let func = module::load_kernel(
        ctx,
        "fa_bf16_fp8kv_decode_d512_gqa_pos_dev",
        "fa_bf16_fp8kv_decode_d512_gqa_pos_dev",
    )?;
    let cfg = LaunchConfig {
        grid_dim: (num_q_heads, batch, 1),
        block_dim: (D, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(q)
            .arg(k)
            .arg(v)
            .arg(o)
            .arg(&num_q_heads)
            .arg(&num_kv_heads)
            .arg(&kv_stride)
            .arg(pos_ptr)
            .arg(&scale)
            .arg(&kv_scale)
            .launch(cfg)?;
    }
    Ok(())
}

/// Causal variant of `fa_bf16_fp8kv_decode_d128`. Per-batch `q_pos` clamps the
/// KV loop bound so each batch position only attends to KV positions
/// `[0, q_pos[batch]]` inclusive. Same scalar compute pattern.
///
/// Layout: same as non-causal version + `q_pos: [B] u32`.
#[allow(clippy::too_many_arguments)]
pub fn fa_bf16_fp8kv_decode_d128_causal(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u16>,
    k: &CudaSlice<u8>,
    v: &CudaSlice<u8>,
    o: &mut CudaSlice<u16>,
    q_pos: &CudaSlice<u32>,
    batch: u32,
    num_heads: u32,
    seq_kv: u32,
    scale: f32,
    kv_scale: f32,
) -> Result<()> {
    if batch == 0 || num_heads == 0 || seq_kv == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    if q_pos.len() < batch as usize {
        return Err(SparkError::InvalidArgument("q_pos too small".into()));
    }
    let need_qo = (batch * num_heads * 128) as usize;
    if q.len() < need_qo || o.len() < need_qo {
        return Err(SparkError::InvalidArgument("q/o buffer too small".into()));
    }
    let need_kv = (batch * num_heads * seq_kv * 128) as usize;
    if k.len() < need_kv || v.len() < need_kv {
        return Err(SparkError::InvalidArgument("k/v buffer too small".into()));
    }
    let func = module::load_kernel(
        ctx,
        "fa_bf16_fp8kv_decode_d128_causal",
        "fa_bf16_fp8kv_decode_d128_causal",
    )?;
    let cfg = LaunchConfig {
        grid_dim: (num_heads, batch, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(q)
            .arg(k)
            .arg(v)
            .arg(o)
            .arg(q_pos)
            .arg(&num_heads)
            .arg(&seq_kv)
            .arg(&scale)
            .arg(&kv_scale)
            .launch(cfg)?;
    }
    Ok(())
}

/// FP8 MMA m16n8k32 microkernel for attention QK validation.
/// Computes QK = scale * (Q @ K^T) where Q is [16, 512] FP8 and K is [8, 512] FP8.
/// Output [16, 8] FP32. Validates the FP8 MMA pattern in attention shape before
/// integrating into the full MLA decode kernel.
pub fn fp8_mma_qk_microkernel(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u8>,
    k: &CudaSlice<u8>,
    qk_out: &mut CudaSlice<f32>,
    scale: f32,
) -> Result<()> {
    if q.len() < 16 * 512 || k.len() < 8 * 512 || qk_out.len() < 16 * 8 {
        return Err(SparkError::InvalidArgument("buffer sizes wrong".into()));
    }
    let func = module::load_kernel(ctx, "fp8_mma_qk_microkernel", "fp8_mma_qk_microkernel")?;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(q)
            .arg(k)
            .arg(qk_out)
            .arg(&scale)
            .launch(cfg)?;
    }
    Ok(())
}

/// Standard MHA decode (Sq=1) with BF16 Q/O and FP8 E4M3 KV cache.
/// Per-tensor `kv_scale` dequantizes K and V at SMEM staging. D fixed at 128.
/// Scalar compute (no MMA) — slower than V21 BF16 path but unblocks FP8-KV
/// inference for non-MLA models. MMA-optimized variant is follow-on.
///
/// Layout:
///  - `q`:  [B, H, 128] BF16
///  - `k`:  [B, H, Skv, 128] FP8 E4M3 (u8)
///  - `v`:  [B, H, Skv, 128] FP8 E4M3 (u8)
///  - `o`:  [B, H, 128] BF16
///  - `scale`: 1/sqrt(D) typically
///  - `kv_scale`: per-tensor dequant scale
#[allow(clippy::too_many_arguments)]
pub fn fa_bf16_fp8kv_decode_d128(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u16>,
    k: &CudaSlice<u8>,
    v: &CudaSlice<u8>,
    o: &mut CudaSlice<u16>,
    batch: u32,
    num_heads: u32,
    seq_kv: u32,
    scale: f32,
    kv_scale: f32,
) -> Result<()> {
    if batch == 0 || num_heads == 0 || seq_kv == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    let need_qo = (batch * num_heads * 128) as usize;
    if q.len() < need_qo || o.len() < need_qo {
        return Err(SparkError::InvalidArgument("q/o buffer too small".into()));
    }
    let need_kv = (batch * num_heads * seq_kv * 128) as usize;
    if k.len() < need_kv || v.len() < need_kv {
        return Err(SparkError::InvalidArgument("k/v buffer too small".into()));
    }

    let func = module::load_kernel(
        ctx,
        "fa_bf16_fp8kv_decode_d128",
        "fa_bf16_fp8kv_decode_d128",
    )?;
    let cfg = LaunchConfig {
        grid_dim: (num_heads, batch, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(q)
            .arg(k)
            .arg(v)
            .arg(o)
            .arg(&num_heads)
            .arg(&seq_kv)
            .arg(&scale)
            .arg(&kv_scale)
            .launch(cfg)?;
    }
    Ok(())
}

/// BF16 GQA flash-attention DECODE for d=256 with optional sliding-window
/// attention. Used by Gemma-4 sliding attention layers (head_dim=256,
/// window=512). Set `sliding_window=0` for full attention (no window).
///
/// Q: [batch, num_heads, 256] BF16
/// K: [batch, num_kv_heads, kv_stride, 256] BF16 (KV cache)
/// V: [batch, num_kv_heads, kv_stride, 256] BF16
/// O: [batch, num_heads, 256] BF16
///
/// `q_pos` is the absolute position of the (single) query token; used as the
/// upper bound for the sliding window when sliding_window > 0. The kernel
/// attends to k ∈ [max(0, q_pos+1-window), seq_kv).
///
/// Decode-only (Sq=1).
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_bf16_decode_d256_gqa(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u16>,
    k: &CudaSlice<u16>,
    v: &CudaSlice<u16>,
    o: &mut CudaSlice<u16>,
    batch: u32,
    num_heads: u32,
    num_kv_heads: u32,
    seq_kv: u32,
    kv_stride: u32,
    q_pos: u32,
    sliding_window: u32,
    scale: f32,
) -> Result<()> {
    const D: u32 = 256;
    if batch == 0 || num_heads == 0 || num_kv_heads == 0 || seq_kv == 0 || kv_stride == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    if !num_heads.is_multiple_of(num_kv_heads) {
        return Err(SparkError::InvalidArgument(format!(
            "num_kv_heads ({num_kv_heads}) must divide num_heads ({num_heads})"
        )));
    }
    if seq_kv > kv_stride {
        return Err(SparkError::InvalidArgument(format!(
            "seq_kv ({seq_kv}) > kv_stride ({kv_stride})"
        )));
    }
    let q_need = batch as usize * num_heads as usize * D as usize;
    let kv_need = batch as usize * num_kv_heads as usize * kv_stride as usize * D as usize;
    if q.len() < q_need || k.len() < kv_need || v.len() < kv_need || o.len() < q_need {
        return Err(SparkError::InvalidArgument(format!(
            "buffer sizes: q={} k={} v={} o={}, need q/o={q_need} k/v={kv_need}",
            q.len(),
            k.len(),
            v.len(),
            o.len(),
        )));
    }
    let func = module::load_kernel(ctx, "fa_bf16_decode_d256_gqa", "fa_bf16_decode_d256_gqa")?;
    let cfg = LaunchConfig {
        grid_dim: (num_heads, batch, 1),
        block_dim: (D, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(q)
            .arg(k)
            .arg(v)
            .arg(o)
            .arg(&num_heads)
            .arg(&num_kv_heads)
            .arg(&seq_kv)
            .arg(&kv_stride)
            .arg(&q_pos)
            .arg(&sliding_window)
            .arg(&scale)
            .launch(cfg)?;
    }
    Ok(())
}

/// BF16 GQA flash-attention DECODE for d=512. Used by Gemma-4 full-attention
/// layers. No sliding window — full attention.
///
/// Q: [batch, num_heads, 512] BF16
/// K, V: [batch, num_kv_heads, kv_stride, 512] BF16
/// O: [batch, num_heads, 512] BF16
///
/// Decode-only (Sq=1).
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_bf16_decode_d512_gqa(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u16>,
    k: &CudaSlice<u16>,
    v: &CudaSlice<u16>,
    o: &mut CudaSlice<u16>,
    batch: u32,
    num_heads: u32,
    num_kv_heads: u32,
    seq_kv: u32,
    kv_stride: u32,
    scale: f32,
) -> Result<()> {
    const D: u32 = 512;
    if batch == 0 || num_heads == 0 || num_kv_heads == 0 || seq_kv == 0 || kv_stride == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    if !num_heads.is_multiple_of(num_kv_heads) {
        return Err(SparkError::InvalidArgument(format!(
            "num_kv_heads ({num_kv_heads}) must divide num_heads ({num_heads})"
        )));
    }
    if seq_kv > kv_stride {
        return Err(SparkError::InvalidArgument(format!(
            "seq_kv ({seq_kv}) > kv_stride ({kv_stride})"
        )));
    }
    let q_need = batch as usize * num_heads as usize * D as usize;
    let kv_need = batch as usize * num_kv_heads as usize * kv_stride as usize * D as usize;
    if q.len() < q_need || k.len() < kv_need || v.len() < kv_need || o.len() < q_need {
        return Err(SparkError::InvalidArgument(format!(
            "buffer sizes: q={} k={} v={} o={}, need q/o={q_need} k/v={kv_need}",
            q.len(),
            k.len(),
            v.len(),
            o.len(),
        )));
    }
    let func = module::load_kernel(ctx, "fa_bf16_decode_d512_gqa", "fa_bf16_decode_d512_gqa")?;
    let cfg = LaunchConfig {
        grid_dim: (num_heads, batch, 1),
        block_dim: (D, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(q)
            .arg(k)
            .arg(v)
            .arg(o)
            .arg(&num_heads)
            .arg(&num_kv_heads)
            .arg(&seq_kv)
            .arg(&kv_stride)
            .arg(&scale)
            .launch(cfg)?;
    }
    Ok(())
}

/// Position-from-device-pointer variant of `flash_attn_bf16_decode_d256_gqa`.
/// Reads `q_pos` from `pos_ptr` and computes `seq_kv = *pos_ptr + 1`
/// internally. Caller must ensure `(*pos_ptr) + 1 <= kv_stride`. Same math
/// as the param-based sibling otherwise.
#[allow(clippy::too_many_arguments)]
/// View-accepting variant of `flash_attn_bf16_decode_d256_gqa_pos_dev`. q/o
/// can be views into batched buffers; k/v remain CudaSlice (per-seq KV cache).
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_bf16_decode_d256_gqa_pos_dev_view(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaView<u16>,
    k: &CudaSlice<u16>,
    v: &CudaSlice<u16>,
    o: &mut CudaViewMut<u16>,
    batch: u32,
    num_heads: u32,
    num_kv_heads: u32,
    kv_stride: u32,
    pos_ptr: &CudaSlice<u32>,
    sliding_window: u32,
    scale: f32,
) -> Result<()> {
    const D: u32 = 256;
    if batch == 0 || num_heads == 0 || num_kv_heads == 0 || kv_stride == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    if !num_heads.is_multiple_of(num_kv_heads) {
        return Err(SparkError::InvalidArgument(format!(
            "num_kv_heads ({num_kv_heads}) must divide num_heads ({num_heads})"
        )));
    }
    let func = module::load_kernel(
        ctx,
        "fa_bf16_decode_d256_gqa_pos_dev",
        "fa_bf16_decode_d256_gqa_pos_dev",
    )?;
    let cfg = LaunchConfig {
        grid_dim: (num_heads, batch, 1),
        block_dim: (D, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(q)
            .arg(k)
            .arg(v)
            .arg(o)
            .arg(&num_heads)
            .arg(&num_kv_heads)
            .arg(&kv_stride)
            .arg(pos_ptr)
            .arg(&sliding_window)
            .arg(&scale)
            .launch(cfg)?;
    }
    Ok(())
}

/// View-accepting variant of `flash_attn_bf16_decode_d512_gqa_pos_dev`.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_bf16_decode_d512_gqa_pos_dev_view(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaView<u16>,
    k: &CudaSlice<u16>,
    v: &CudaSlice<u16>,
    o: &mut CudaViewMut<u16>,
    batch: u32,
    num_heads: u32,
    num_kv_heads: u32,
    kv_stride: u32,
    pos_ptr: &CudaSlice<u32>,
    sliding_window: u32,
    scale: f32,
) -> Result<()> {
    const D: u32 = 512;
    if batch == 0 || num_heads == 0 || num_kv_heads == 0 || kv_stride == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    if !num_heads.is_multiple_of(num_kv_heads) {
        return Err(SparkError::InvalidArgument(format!(
            "num_kv_heads ({num_kv_heads}) must divide num_heads ({num_heads})"
        )));
    }
    let func = module::load_kernel(
        ctx,
        "fa_bf16_decode_d512_gqa_pos_dev",
        "fa_bf16_decode_d512_gqa_pos_dev",
    )?;
    let cfg = LaunchConfig {
        grid_dim: (num_heads, batch, 1),
        block_dim: (D, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(q)
            .arg(k)
            .arg(v)
            .arg(o)
            .arg(&num_heads)
            .arg(&num_kv_heads)
            .arg(&kv_stride)
            .arg(pos_ptr)
            .arg(&sliding_window)
            .arg(&scale)
            .launch(cfg)?;
    }
    Ok(())
}

/// Single-token BF16 decode attention (head_dim=256, GQA) with the current
/// position read from `pos_ptr` (`seq_kv = *pos_ptr + 1`) and an optional
/// `sliding_window` cap on how far back KV is attended. Reads the full
/// `[.., kv_stride, ..]` KV cache in place.
pub fn flash_attn_bf16_decode_d256_gqa_pos_dev(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u16>,
    k: &CudaSlice<u16>,
    v: &CudaSlice<u16>,
    o: &mut CudaSlice<u16>,
    batch: u32,
    num_heads: u32,
    num_kv_heads: u32,
    kv_stride: u32,
    pos_ptr: &CudaSlice<u32>,
    sliding_window: u32,
    scale: f32,
) -> Result<()> {
    const D: u32 = 256;
    if batch == 0 || num_heads == 0 || num_kv_heads == 0 || kv_stride == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    if !num_heads.is_multiple_of(num_kv_heads) {
        return Err(SparkError::InvalidArgument(format!(
            "num_kv_heads ({num_kv_heads}) must divide num_heads ({num_heads})"
        )));
    }
    if (pos_ptr.len() as u32) < batch {
        // The kernel indexes pos_ptr[blockIdx.z] for blockIdx.z in 0..batch, so
        // a shorter buffer is an out-of-bounds read, not just "empty".
        return Err(SparkError::InvalidArgument(format!(
            "pos_ptr too small: {} < batch {batch}",
            pos_ptr.len()
        )));
    }
    let q_need = batch as usize * num_heads as usize * D as usize;
    let kv_need = batch as usize * num_kv_heads as usize * kv_stride as usize * D as usize;
    if q.len() < q_need || k.len() < kv_need || v.len() < kv_need || o.len() < q_need {
        return Err(SparkError::InvalidArgument(format!(
            "buffer sizes: q={} k={} v={} o={}, need q/o={q_need} k/v={kv_need}",
            q.len(),
            k.len(),
            v.len(),
            o.len(),
        )));
    }
    let func = module::load_kernel(
        ctx,
        "fa_bf16_decode_d256_gqa_pos_dev",
        "fa_bf16_decode_d256_gqa_pos_dev",
    )?;
    let cfg = LaunchConfig {
        grid_dim: (num_heads, batch, 1),
        block_dim: (D, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(q)
            .arg(k)
            .arg(v)
            .arg(o)
            .arg(&num_heads)
            .arg(&num_kv_heads)
            .arg(&kv_stride)
            .arg(pos_ptr)
            .arg(&sliding_window)
            .arg(&scale)
            .launch(cfg)?;
    }
    Ok(())
}

/// Position-from-device-pointer variant of `flash_attn_bf16_decode_d512_gqa`.
/// Reads `q_pos` from `pos_ptr` and computes `seq_kv = *pos_ptr + 1`
/// internally. Caller must ensure `(*pos_ptr) + 1 <= kv_stride`.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_bf16_decode_d512_gqa_pos_dev(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u16>,
    k: &CudaSlice<u16>,
    v: &CudaSlice<u16>,
    o: &mut CudaSlice<u16>,
    batch: u32,
    num_heads: u32,
    num_kv_heads: u32,
    kv_stride: u32,
    pos_ptr: &CudaSlice<u32>,
    scale: f32,
) -> Result<()> {
    const D: u32 = 512;
    if batch == 0 || num_heads == 0 || num_kv_heads == 0 || kv_stride == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    if !num_heads.is_multiple_of(num_kv_heads) {
        return Err(SparkError::InvalidArgument(format!(
            "num_kv_heads ({num_kv_heads}) must divide num_heads ({num_heads})"
        )));
    }
    if (pos_ptr.len() as u32) < batch {
        // The kernel indexes pos_ptr[blockIdx.z] for blockIdx.z in 0..batch, so
        // a shorter buffer is an out-of-bounds read, not just "empty".
        return Err(SparkError::InvalidArgument(format!(
            "pos_ptr too small: {} < batch {batch}",
            pos_ptr.len()
        )));
    }
    let q_need = batch as usize * num_heads as usize * D as usize;
    let kv_need = batch as usize * num_kv_heads as usize * kv_stride as usize * D as usize;
    if q.len() < q_need || k.len() < kv_need || v.len() < kv_need || o.len() < q_need {
        return Err(SparkError::InvalidArgument(format!(
            "buffer sizes: q={} k={} v={} o={}, need q/o={q_need} k/v={kv_need}",
            q.len(),
            k.len(),
            v.len(),
            o.len(),
        )));
    }
    let func = module::load_kernel(
        ctx,
        "fa_bf16_decode_d512_gqa_pos_dev",
        "fa_bf16_decode_d512_gqa_pos_dev",
    )?;
    let cfg = LaunchConfig {
        grid_dim: (num_heads, batch, 1),
        block_dim: (D, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(q)
            .arg(k)
            .arg(v)
            .arg(o)
            .arg(&num_heads)
            .arg(&num_kv_heads)
            .arg(&kv_stride)
            .arg(pos_ptr)
            .arg(&scale)
            .launch(cfg)?;
    }
    Ok(())
}
