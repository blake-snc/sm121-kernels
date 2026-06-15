//! Ring attention orchestrator (single-GPU implementation).
//!
//! Computes flash attention by sweeping the KV dimension in `num_chunks` ring
//! steps, accumulating partial outputs + LSEs, then merging via flash-decoding
//! combine. On a single GPU this is functionally identical to a single
//! attention call and exists primarily to:
//!   1. Demonstrate the per-step structure that maps 1:1 to multi-node ring,
//!   2. Validate that split-K + combine gives byte-equivalent output to the
//!      monolithic kernel (correctness invariant for multi-node port),
//!   3. Provide a host-side API surface that multi-node NCCL integration can
//!      slot into without changing kernel-level code.
//!
//! For true multi-node ring attention on `N` DGX Spark nodes:
//!   - Each node holds K_local, V_local of shape [B, H, Skv/N, D]
//!   - Step `k` (k in 0..N): node i computes attention against K from node
//!     `(i - k) mod N` (received via NCCL send/recv from node i-1, sent to
//!     node i+1) → produces O_partial[k] + LSE_partial[k]
//!   - After N steps each node has N partials; combine merges them locally.
//!
//! The kernel-level changes for that future port are zero — same split-K and
//! combine kernels. The orchestrator below is the integration point.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};

use crate::attention::{flash_attn_bf16_v3_split_kv, flash_decoding_combine};
use crate::error::{Result, SparkError};

/// Single-GPU ring attention. Sweeps KV in `num_chunks` steps, each computing a
/// partial output + LSE, then merges via online LSE combine. Output matches
/// (within FP rounding tolerance) `flash_attn_bf16_v3` called once with the
/// full KV.
///
/// Layouts:
/// - `q`:  `[B, H, Sq, D]` BF16
/// - `k`:  `[B, H, Skv, D]` BF16 (the orchestrator slices internally; in
///   multi-node ring this would be the per-node local slice rotated each step)
/// - `v`:  `[B, H, Skv, D]` BF16
/// - `o`:  `[B, H, Sq, D]` BF16 (output)
///
/// `num_chunks` must divide `Skv` evenly. D=128.
#[allow(clippy::too_many_arguments)]
pub fn ring_attention_bf16(
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
    num_chunks: u32,
) -> Result<()> {
    if num_chunks == 0 || !seq_kv.is_multiple_of(num_chunks) {
        return Err(SparkError::InvalidArgument(format!(
            "num_chunks ({num_chunks}) must divide seq_kv ({seq_kv}) evenly"
        )));
    }
    if num_chunks > 64 {
        // Combine kernel reads num_splits sequentially per output element;
        // very high split counts dominate the runtime.
        return Err(SparkError::InvalidArgument(format!(
            "num_chunks={num_chunks} exceeds practical limit (64)"
        )));
    }

    // Allocate partial buffers once. Reused across the (degenerate) ring rotation.
    let total_partial = (num_chunks * batch * num_heads * seq_q * 128) as usize;
    let total_lse = (num_chunks * batch * num_heads * seq_q) as usize;
    let mut o_partial = stream.alloc_zeros::<f32>(total_partial)?;
    let mut lse_partial = stream.alloc_zeros::<f32>(total_lse)?;

    // Ring loop: each step computes partial against the chunk_idx'th KV slice.
    // Single-GPU degenerate case — the kernel uses (split_idx, num_splits) to
    // select its KV range. Multi-node would replace this with a fixed local
    // (split_idx=node_rank) and a between-step NCCL send/recv KV rotation.
    for chunk_idx in 0..num_chunks {
        flash_attn_bf16_v3_split_kv(
            ctx,
            stream,
            q,
            k,
            v,
            &mut o_partial,
            &mut lse_partial,
            batch,
            num_heads,
            seq_q,
            seq_kv,
            scale,
            num_chunks,
            chunk_idx,
        )?;
    }

    // Combine: merge num_chunks partials into final BF16 output via online LSE.
    flash_decoding_combine(
        ctx,
        stream,
        &o_partial,
        &lse_partial,
        o,
        batch,
        num_heads,
        seq_q,
        num_chunks,
    )?;

    Ok(())
}
