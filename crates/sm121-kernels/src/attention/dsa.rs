//! DeepSeek Sparse Attention (DSA) kernel dispatchers.
//!
//! Two-stage design: a BF16 indexer score kernel (`dsa_indexer_score_bf16`)
//! ranks KV positions, then sparse-mask attention runs as a gather-then-dense
//! wrapper (`dsa_sparse_attention_bf16`) over the selected positions. A native
//! masked-FA PTX kernel is a possible future perf path.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};

use crate::error::{Result, SparkError};
use crate::module;

use super::flash_attn_bf16_d128;

const H_IDX: u32 = 32;
const D_IDX: u32 = 128;
const TILE_T: u32 = 128;

/// Compute the DSA indexer's per-(b, s, t) raw index scores.
///
/// Inputs:
/// - `q`:       `[B, S, H_idx, D_idx]` BF16 — post-RoPE indexer query
/// - `k`:       `[B, T,        D_idx]` BF16 — indexer's own key cache (no head dim)
/// - `weights`: `[B, S, H_idx]` FP32       — must already absorb `softmax_scale * n_heads^-0.5`
///
/// Output:
/// - `out`:     `[B, S, T]` FP32 — raw index scores (top-k selection is the caller's job)
///
/// Fixed at compile time for the MVP: `H_idx = 32`, `D_idx = 128`.
/// A generic-dimension dispatch can ship later.
#[allow(clippy::too_many_arguments)]
pub fn dsa_indexer_score_bf16(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u16>,
    k: &CudaSlice<u16>,
    weights: &CudaSlice<f32>,
    out: &mut CudaSlice<f32>,
    batch: u32,
    seq_q: u32,
    seq_kv: u32,
) -> Result<()> {
    if batch == 0 || seq_q == 0 || seq_kv == 0 {
        return Err(SparkError::InvalidArgument(
            "batch / seq_q / seq_kv must be > 0".into(),
        ));
    }

    // Validate buffer sizes against the fixed (H_idx, D_idx) defaults.
    let q_need = (batch * seq_q * H_IDX * D_IDX) as usize;
    let k_need = (batch * seq_kv * D_IDX) as usize;
    let w_need = (batch * seq_q * H_IDX) as usize;
    let o_need = (batch * seq_q * seq_kv) as usize;
    if q.len() < q_need {
        return Err(SparkError::InvalidArgument(format!(
            "q too small: {} < {}",
            q.len(),
            q_need
        )));
    }
    if k.len() < k_need {
        return Err(SparkError::InvalidArgument(format!(
            "k too small: {} < {}",
            k.len(),
            k_need
        )));
    }
    if weights.len() < w_need {
        return Err(SparkError::InvalidArgument(format!(
            "weights too small: {} < {}",
            weights.len(),
            w_need
        )));
    }
    if out.len() < o_need {
        return Err(SparkError::InvalidArgument(format!(
            "out too small: {} < {}",
            out.len(),
            o_need
        )));
    }

    // kernel_name = PTX file stem (ptx/attention/dsa_indexer_bf16.ptx),
    // entry_point = the .entry symbol inside it.
    let func = module::load_kernel(ctx, "dsa_indexer_bf16", "dsa_indexer_score_bf16")?;

    let grid_x = batch * seq_q;
    let grid_y = seq_kv.div_ceil(TILE_T);

    // SMEM: Q [32, 128] BF16 (8 KB) + weights [32] FP32 (128 B) = 8320 B.
    // We pass the exact size so ptxas can configure cuFuncSetAttribute.
    let smem_bytes = H_IDX * D_IDX * 2 + H_IDX * 4;

    let cfg = LaunchConfig {
        grid_dim: (grid_x, grid_y, 1),
        block_dim: (TILE_T, 1, 1),
        shared_mem_bytes: smem_bytes,
    };

    unsafe {
        stream
            .launch_builder(&func)
            .arg(q)
            .arg(k)
            .arg(weights)
            .arg(out)
            .arg(&seq_q)
            .arg(&seq_kv)
            .launch(cfg)?;
    }
    Ok(())
}

/// Sparse-mask attention via gather-then-dense (MVP).
///
/// For each (b, s):
///   1. Gather K, V at the top-k positions from `top_k_indices[b, s, :]`.
///   2. Permute Q for this (b, s) into [1, H, 1, D].
///   3. Repeat-interleave K, V from H_kv → H (MVP GQA bridge).
///   4. Call existing `flash_attn_bf16_d128` with the gathered tensors.
///   5. Write per-(b, s) result back to the flat output buffer.
///
/// This is mathematically equivalent to standard FA with a -inf mask on
/// non-selected positions (the gathered top-k positions get standard
/// softmax weight; non-gathered positions are dropped). The MVP runs one
/// FA launch per (b, s) — N kernel launches for N queries. A native
/// fused masked-FA kernel is a possible future perf path.
///
/// Inputs:
/// - `q`:    `[B, S, H, D=128]` BF16
/// - `k`:    `[B, T_full, H_kv, D=128]` BF16
/// - `v`:    `[B, T_full, H_kv, D=128]` BF16
/// - `top_k_indices`: `[B, S, topk]` INT32 — selected K positions per query
/// - `scale`: softmax scale (typically 1/sqrt(D))
///
/// Output:
/// - `out`:  `[B, S, H, D=128]` BF16 — attention result
#[allow(clippy::too_many_arguments)]
pub fn dsa_sparse_attention_bf16(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u16>,
    k: &CudaSlice<u16>,
    v: &CudaSlice<u16>,
    top_k_indices: &CudaSlice<i32>,
    out: &mut CudaSlice<u16>,
    batch: u32,
    seq_q: u32,
    seq_kv_full: u32,
    num_heads: u32,
    num_kv_heads: u32,
    topk: u32,
    scale: f32,
) -> Result<()> {
    if batch == 0 || seq_q == 0 || seq_kv_full == 0 || num_heads == 0 || topk == 0 {
        return Err(SparkError::InvalidArgument(
            "batch / seq_q / seq_kv_full / num_heads / topk must be > 0".into(),
        ));
    }
    if !num_heads.is_multiple_of(num_kv_heads) {
        return Err(SparkError::InvalidArgument(
            "num_heads must be divisible by num_kv_heads".into(),
        ));
    }
    let d: u32 = 128;
    let gqa_factor = num_heads / num_kv_heads;

    // Pull indices to host once (per launch). Per-(b, s) loop on the host side;
    // each iteration issues GPU work. CPU-side cost is O(B*S*topk) ints copied
    // — fine for the MVP.
    let idx_host = stream
        .memcpy_dtov(top_k_indices)
        .map_err(|e| SparkError::LaunchFailed(format!("dtoh top_k_indices: {e:?}")))?;

    // Scratch buffers for the gathered K, V tiles + Q reshape + per-query O.
    // Allocate once per launch, reuse across the (b, s) loop.
    let mut k_gather = stream
        .alloc_zeros::<u16>((num_heads * topk * d) as usize)
        .map_err(|e| SparkError::LaunchFailed(format!("alloc k_gather: {e:?}")))?;
    let mut v_gather = stream
        .alloc_zeros::<u16>((num_heads * topk * d) as usize)
        .map_err(|e| SparkError::LaunchFailed(format!("alloc v_gather: {e:?}")))?;
    let mut q_one = stream
        .alloc_zeros::<u16>((num_heads * d) as usize)
        .map_err(|e| SparkError::LaunchFailed(format!("alloc q_one: {e:?}")))?;
    let mut o_one = stream
        .alloc_zeros::<u16>((num_heads * d) as usize)
        .map_err(|e| SparkError::LaunchFailed(format!("alloc o_one: {e:?}")))?;

    // Per-(b, s) loop. Each iteration is one FA launch.
    let q_host = stream
        .memcpy_dtov(q)
        .map_err(|e| SparkError::LaunchFailed(format!("dtoh q: {e:?}")))?;
    let k_host = stream
        .memcpy_dtov(k)
        .map_err(|e| SparkError::LaunchFailed(format!("dtoh k: {e:?}")))?;
    let v_host = stream
        .memcpy_dtov(v)
        .map_err(|e| SparkError::LaunchFailed(format!("dtoh v: {e:?}")))?;
    let mut out_host = vec![0u16; (batch * seq_q * num_heads * d) as usize];

    for b in 0..batch as usize {
        for s in 0..seq_q as usize {
            // Build per-query Q [num_heads, d]: source layout is [B, S, H, D].
            let mut q_one_host = vec![0u16; (num_heads * d) as usize];
            for h in 0..num_heads as usize {
                for di in 0..d as usize {
                    let src = ((b * seq_q as usize + s) * num_heads as usize + h) * d as usize + di;
                    let dst = h * d as usize + di;
                    q_one_host[dst] = q_host[src];
                }
            }

            // Build gathered K [num_heads, topk, d] and V [num_heads, topk, d]
            // via repeat-interleave from [num_kv_heads, topk, d].
            let mut k_gh = vec![0u16; (num_heads * topk * d) as usize];
            let mut v_gh = vec![0u16; (num_heads * topk * d) as usize];
            for ti in 0..topk as usize {
                let bs_idx = (b * seq_q as usize + s) * topk as usize + ti;
                let kv_pos = idx_host[bs_idx] as usize;
                for h_kv in 0..num_kv_heads as usize {
                    for di in 0..d as usize {
                        // K source: [B, T_full, H_kv, D]
                        let src = ((b * seq_kv_full as usize + kv_pos) * num_kv_heads as usize
                            + h_kv)
                            * d as usize
                            + di;
                        let k_val = k_host[src];
                        let v_val = v_host[src];
                        // Destination: [num_heads, topk, D] — repeat-interleave H_kv → H
                        for r in 0..gqa_factor as usize {
                            let h = h_kv * gqa_factor as usize + r;
                            let dst = (h * topk as usize + ti) * d as usize + di;
                            k_gh[dst] = k_val;
                            v_gh[dst] = v_val;
                        }
                    }
                }
            }

            // Upload + FA launch + readback per (b, s). MVP: definitely
            // slow due to N htod copies, but correct. A follow-up can move
            // the entire gather to a GPU kernel.
            stream
                .memcpy_htod(&q_one_host, &mut q_one)
                .map_err(|e| SparkError::LaunchFailed(format!("htod q_one: {e:?}")))?;
            stream
                .memcpy_htod(&k_gh, &mut k_gather)
                .map_err(|e| SparkError::LaunchFailed(format!("htod k_gh: {e:?}")))?;
            stream
                .memcpy_htod(&v_gh, &mut v_gather)
                .map_err(|e| SparkError::LaunchFailed(format!("htod v_gh: {e:?}")))?;

            flash_attn_bf16_d128(
                ctx, stream, &q_one, &k_gather, &v_gather, &mut o_one, 1, num_heads, 1, topk, scale,
            )?;

            // Read back and write to output buffer.
            let o_one_host = stream
                .memcpy_dtov(&o_one)
                .map_err(|e| SparkError::LaunchFailed(format!("dtoh o_one: {e:?}")))?;
            for h in 0..num_heads as usize {
                for di in 0..d as usize {
                    let src = h * d as usize + di;
                    let dst = ((b * seq_q as usize + s) * num_heads as usize + h) * d as usize + di;
                    out_host[dst] = o_one_host[src];
                }
            }
        }
    }

    stream
        .memcpy_htod(&out_host, out)
        .map_err(|e| SparkError::LaunchFailed(format!("htod final out: {e:?}")))?;
    Ok(())
}

/// End-to-end DSA forward — indexer + top-k selection + sparse attention.
///
/// Integration wrapper that bundles the indexer + sparse attention:
///   1. `dsa_indexer_score_bf16` produces index_scores [B, S, T_full].
///   2. CPU-side argsort (MVP) picks the top-`topk` indices per (b, s).
///      A GPU top-k kernel is a follow-on perf item.
///   3. `dsa_sparse_attention_bf16` runs masked attention over those top-k.
///
/// Inputs:
/// - `q_idx`:     `[B, S, H_idx, D_idx]` BF16 — indexer query (post-RoPE)
/// - `k_idx`:     `[B, T_full, D_idx]`   BF16 — indexer key cache (no head dim)
/// - `weights`:   `[B, S, H_idx]`         FP32 — indexer weights (softmax_scale baked in)
/// - `q_attn`:    `[B, S, H, D=128]`      BF16 — main attention query
/// - `k_attn`:    `[B, T_full, H_kv, D=128]` BF16 — main attention K cache
/// - `v_attn`:    `[B, T_full, H_kv, D=128]` BF16 — main attention V cache
/// - `topk`: how many K positions to keep per query
/// - `scale_attn`: softmax scale for the main attention (typically 1/sqrt(D))
///
/// Output:
/// - `out`: `[B, S, H, D=128]` BF16 — attention result
///
/// Scratch:
/// - `index_scores_scratch`: `[B, S, T_full]` FP32 — owned-by-caller, allocated once.
/// - `topk_indices_scratch`: `[B, S, topk]` INT32 — owned-by-caller.
#[allow(clippy::too_many_arguments)]
pub fn dsa_attention_bf16(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    q_idx: &CudaSlice<u16>,
    k_idx: &CudaSlice<u16>,
    weights: &CudaSlice<f32>,
    q_attn: &CudaSlice<u16>,
    k_attn: &CudaSlice<u16>,
    v_attn: &CudaSlice<u16>,
    index_scores_scratch: &mut CudaSlice<f32>,
    topk_indices_scratch: &mut CudaSlice<i32>,
    out: &mut CudaSlice<u16>,
    batch: u32,
    seq_q: u32,
    seq_kv_full: u32,
    num_heads: u32,
    num_kv_heads: u32,
    topk: u32,
    scale_attn: f32,
) -> Result<()> {
    // Stage 1: indexer scores.
    dsa_indexer_score_bf16(
        ctx,
        stream,
        q_idx,
        k_idx,
        weights,
        index_scores_scratch,
        batch,
        seq_q,
        seq_kv_full,
    )?;

    // Stage 2: CPU-side top-k. Dtov the scores, argsort per (b, s), htod
    // the indices. MVP — a fused GPU top-k kernel is a follow-on.
    let scores_host = stream
        .memcpy_dtov(index_scores_scratch)
        .map_err(|e| SparkError::LaunchFailed(format!("dtoh index_scores: {e:?}")))?;
    let mut idx_host = vec![0i32; (batch * seq_q * topk) as usize];
    let topk_usize = topk as usize;
    let kv_usize = seq_kv_full as usize;
    for b in 0..batch as usize {
        for s in 0..seq_q as usize {
            let base = (b * seq_q as usize + s) * kv_usize;
            let row = &scores_host[base..base + kv_usize];
            // Sort indices by score descending; take first topk.
            let mut order: Vec<u32> = (0..kv_usize as u32).collect();
            order.sort_unstable_by(|&i, &j| {
                row[j as usize]
                    .partial_cmp(&row[i as usize])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            for (i, val) in order.iter().take(topk_usize).enumerate() {
                idx_host[(b * seq_q as usize + s) * topk_usize + i] = *val as i32;
            }
        }
    }
    stream
        .memcpy_htod(&idx_host, topk_indices_scratch)
        .map_err(|e| SparkError::LaunchFailed(format!("htod topk_indices: {e:?}")))?;

    // Stage 3: sparse attention over the selected positions.
    dsa_sparse_attention_bf16(
        ctx,
        stream,
        q_attn,
        k_attn,
        v_attn,
        topk_indices_scratch,
        out,
        batch,
        seq_q,
        seq_kv_full,
        num_heads,
        num_kv_heads,
        topk,
        scale_attn,
    )?;
    Ok(())
}
