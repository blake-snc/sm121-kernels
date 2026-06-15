//! Multi-node ring attention scaffolding (compile-gated behind `nccl` feature).
//!
//! Provides a `RingTransport` trait that any communication backend (NCCL, MPI,
//! custom RDMA, gloo) can implement. The `ring_attention_bf16_distributed`
//! function then orchestrates the ring rotation + per-step compute against
//! the existing split-K + combine kernels.
//!
//! Why a trait instead of direct NCCL integration? cudarc's `nccl::Comm` does
//! not impl `NcclType` for `u16` (the BF16 byte representation), and its raw
//! `ncclComm_t` handle is private. Wrapping NCCL via this trait avoids both
//! issues and lets users plug in any backend.
//!
//! # Reference NCCL implementation
//!
//! ```rust,ignore
//! use cudarc::nccl::{Comm, result as ncr, sys as ncs};
//! use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
//!
//! struct NcclTransport { comm: Comm }
//!
//! impl RingTransport for NcclTransport {
//!     fn world_size(&self) -> u32 { self.comm.world_size() as u32 }
//!     fn rank(&self) -> u32 { self.comm.rank() as u32 }
//!
//!     fn send_recv_bf16(
//!         &self, stream: &Arc<CudaStream>,
//!         send_buf: &CudaSlice<u16>, recv_buf: &mut CudaSlice<u16>,
//!         send_to: u32, recv_from: u32,
//!     ) -> Result<()> {
//!         // Use raw cudarc::nccl::result::send/recv with ncclBfloat16
//!         // (cudarc Comm::send doesn't impl NcclType for u16 — use raw API).
//!         unsafe {
//!             let (sp, _g1) = send_buf.device_ptr(stream);
//!             let (rp, _g2) = recv_buf.device_ptr_mut(stream);
//!             let n = send_buf.len();
//!             // ncr::send / ncr::recv with ncclBfloat16 datatype
//!             // (raw_handle access requires upstream cudarc PR or transmute)
//!         }
//!         Ok(())
//!     }
//! }
//! ```
//!
//! See `examples/ring_attention_demo.rs` for the single-GPU equivalent.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};

use crate::attention::{flash_attn_bf16_v3_split_kv, flash_decoding_combine};
use crate::error::{Result, SparkError};

/// Communication backend abstraction for ring attention. Implement for NCCL,
/// MPI, custom RDMA, or any point-to-point transport.
pub trait RingTransport {
    fn world_size(&self) -> u32;
    fn rank(&self) -> u32;

    /// Send `send_buf` to rank `send_to`, simultaneously receive into
    /// `recv_buf` from rank `recv_from`. May be implemented as parallel
    /// send/recv pair or a single sendrecv primitive.
    fn send_recv_bf16(
        &self,
        stream: &Arc<CudaStream>,
        send_buf: &CudaSlice<u16>,
        recv_buf: &mut CudaSlice<u16>,
        send_to: u32,
        recv_from: u32,
    ) -> Result<()>;
}

/// Multi-node ring attention. Uses `transport` to ring-rotate KV between
/// ranks. Each rank computes `world_size` partials and combines locally.
///
/// Layouts (per rank):
/// - `q`: `[B, H, Sq, D]` BF16 (replicated across ranks)
/// - `k_local`, `v_local`: `[B, H, Skv_local, D]` BF16 (this rank's slice)
/// - `o`: `[B, H, Sq, D]` BF16 (output)
#[allow(clippy::too_many_arguments)]
pub fn ring_attention_bf16_distributed(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    transport: &dyn RingTransport,
    q: &CudaSlice<u16>,
    k_local: &CudaSlice<u16>,
    v_local: &CudaSlice<u16>,
    o: &mut CudaSlice<u16>,
    batch: u32,
    num_heads: u32,
    seq_q: u32,
    seq_kv_local: u32,
    scale: f32,
) -> Result<()> {
    let world_size = transport.world_size();
    let rank = transport.rank();
    if world_size == 0 {
        return Err(SparkError::InvalidArgument("world_size must be > 0".into()));
    }

    let kv_chunk_elems = (batch * num_heads * seq_kv_local * 128) as usize;
    let mut k_send = stream
        .alloc_zeros::<u16>(kv_chunk_elems)
        .map_err(SparkError::Driver)?;
    let mut v_send = stream
        .alloc_zeros::<u16>(kv_chunk_elems)
        .map_err(SparkError::Driver)?;
    let mut k_recv = stream
        .alloc_zeros::<u16>(kv_chunk_elems)
        .map_err(SparkError::Driver)?;
    let mut v_recv = stream
        .alloc_zeros::<u16>(kv_chunk_elems)
        .map_err(SparkError::Driver)?;

    // Bootstrap: copy local KV into send buffers.
    {
        let bytes = kv_chunk_elems * 2;
        unsafe {
            let (src_k, _g1) = k_local.device_ptr(stream);
            let (dst_k, _g2) = k_send.device_ptr_mut(stream);
            cudarc::driver::result::memcpy_dtod_async(dst_k, src_k, bytes, stream.cu_stream())
                .map_err(SparkError::Driver)?;
            let (src_v, _g3) = v_local.device_ptr(stream);
            let (dst_v, _g4) = v_send.device_ptr_mut(stream);
            cudarc::driver::result::memcpy_dtod_async(dst_v, src_v, bytes, stream.cu_stream())
                .map_err(SparkError::Driver)?;
        }
    }

    let partial_o_len = (world_size * batch * num_heads * seq_q * 128) as usize;
    let lse_len = (world_size * batch * num_heads * seq_q) as usize;
    let mut o_partial = stream
        .alloc_zeros::<f32>(partial_o_len)
        .map_err(SparkError::Driver)?;
    let mut lse_partial = stream
        .alloc_zeros::<f32>(lse_len)
        .map_err(SparkError::Driver)?;

    let next = (rank + 1) % world_size;
    let prev = (rank + world_size - 1) % world_size;

    for step in 0..world_size {
        let owner = (rank + world_size - step) % world_size;
        flash_attn_bf16_v3_split_kv(
            ctx,
            stream,
            q,
            &k_send,
            &v_send,
            &mut o_partial,
            &mut lse_partial,
            batch,
            num_heads,
            seq_q,
            seq_kv_local,
            scale,
            world_size,
            owner,
        )?;

        if step + 1 < world_size {
            transport.send_recv_bf16(stream, &k_send, &mut k_recv, next, prev)?;
            transport.send_recv_bf16(stream, &v_send, &mut v_recv, next, prev)?;
            std::mem::swap(&mut k_send, &mut k_recv);
            std::mem::swap(&mut v_send, &mut v_recv);
        }
    }

    flash_decoding_combine(
        ctx,
        stream,
        &o_partial,
        &lse_partial,
        o,
        batch,
        num_heads,
        seq_q,
        world_size,
    )?;

    Ok(())
}

/// Trivial single-GPU `RingTransport` for testing — just memcpy between buffers
/// (no actual cross-GPU/cross-node communication). Useful for validating the
/// orchestrator end-to-end without real multi-node hardware.
pub struct LoopbackTransport {
    pub world_size: u32,
    pub rank: u32,
}

impl RingTransport for LoopbackTransport {
    fn world_size(&self) -> u32 {
        self.world_size
    }
    fn rank(&self) -> u32 {
        self.rank
    }
    fn send_recv_bf16(
        &self,
        stream: &Arc<CudaStream>,
        send_buf: &CudaSlice<u16>,
        recv_buf: &mut CudaSlice<u16>,
        _send_to: u32,
        _recv_from: u32,
    ) -> Result<()> {
        // Loopback: just copy send → recv in-place.
        let bytes = send_buf.len() * 2;
        unsafe {
            let (src, _g1) = send_buf.device_ptr(stream);
            let (dst, _g2) = recv_buf.device_ptr_mut(stream);
            cudarc::driver::result::memcpy_dtod_async(dst, src, bytes, stream.cu_stream())
                .map_err(SparkError::Driver)?;
        }
        Ok(())
    }
}
