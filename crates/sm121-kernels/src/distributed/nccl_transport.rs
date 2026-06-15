//! NCCL implementation of `RingTransport` for multi-node DGX Spark clustering.
//!
//! Bridges cudarc's safe `Comm` to our `RingTransport` trait. BF16 send/recv
//! uses the raw `nccl::result::send`/`recv` API with `ncclBfloat16` so we don't
//! need the `f16` feature flag (we transmit the byte-equivalent `u16` buffer).
//!
//! # Bringing up a 2-rank cluster
//!
//! Rank 0 generates a unique `Id`, broadcasts its 128-byte payload to all other
//! ranks via an out-of-band channel (file-shared, gRPC, MPI), then every rank
//! constructs `NcclTransport::new(stream, rank, world_size, id)`.
//!
//! For single-node multi-process bring-up:
//! ```bash
//! NCCL_SOCKET_IFNAME=lo RANK=0 WORLD_SIZE=2 NCCL_ID_FILE=/tmp/nccl.id \
//!   cargo run --release --example nccl_allreduce_sanity
//! NCCL_SOCKET_IFNAME=lo RANK=1 WORLD_SIZE=2 NCCL_ID_FILE=/tmp/nccl.id \
//!   cargo run --release --example nccl_allreduce_sanity
//! ```

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use cudarc::nccl::{result as ncr, sys as ncs, Comm, Id, ReduceOp};

use crate::distributed::ring_attention_distributed::RingTransport;
use crate::error::{Result, SparkError};

/// NCCL-backed ring transport. Wraps cudarc's `Comm` and forwards the
/// `RingTransport` trait calls to it.
pub struct NcclTransport {
    comm: Comm,
    rank: u32,
    world_size: u32,
}

impl NcclTransport {
    /// Construct a transport for the given rank. `id` must be the same payload
    /// across all ranks (rank 0 calls `Id::new()` and broadcasts the
    /// `id.internal()` byte array to peers).
    pub fn new(stream: Arc<CudaStream>, rank: u32, world_size: u32, id: Id) -> Result<Self> {
        let comm = Comm::from_rank(stream, rank as usize, world_size as usize, id)
            .map_err(|e| SparkError::Other(format!("nccl Comm::from_rank: {e:?}")))?;
        Ok(Self {
            comm,
            rank,
            world_size,
        })
    }

    /// Create a fresh unique id (call on rank 0 only, then broadcast its
    /// `internal()` payload to all peers via an OOB channel).
    pub fn new_id() -> Result<Id> {
        Id::new().map_err(|e| SparkError::Other(format!("nccl Id::new: {e:?}")))
    }

    /// Reconstruct an Id from a 128-byte payload (use on non-root ranks after
    /// receiving the bytes from rank 0). `c_char` differs across platforms
    /// (i8 on x86_64, u8 on aarch64), so callers should transmute their byte
    /// buffer with `bytemuck::cast` or a manual cast at the call site.
    pub fn id_from_bytes(bytes: [std::ffi::c_char; 128]) -> Id {
        Id::uninit(bytes)
    }

    /// AllReduce SUM helper for the sanity test. Operates on f32 in-place style
    /// (separate send/recv buffers), via the safe wrapper.
    pub fn all_reduce_sum_f32(
        &self,
        send_buf: &CudaSlice<f32>,
        recv_buf: &mut CudaSlice<f32>,
    ) -> Result<()> {
        self.comm
            .all_reduce(send_buf, recv_buf, &ReduceOp::Sum)
            .map_err(|e| SparkError::Other(format!("nccl all_reduce: {e:?}")))?;
        Ok(())
    }

    /// Underlying cudarc Comm (for advanced collectives outside the
    /// RingTransport trait).
    pub fn comm(&self) -> &Comm {
        &self.comm
    }

    /// Context the comm was constructed against.
    pub fn context(&self) -> &Arc<CudaContext> {
        self.comm.context()
    }
}

impl RingTransport for NcclTransport {
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
        send_to: u32,
        recv_from: u32,
    ) -> Result<()> {
        let count = send_buf.len();
        if recv_buf.len() != count {
            return Err(SparkError::InvalidArgument(format!(
                "nccl send_recv_bf16 length mismatch: send={} recv={}",
                count,
                recv_buf.len()
            )));
        }

        let raw_comm = self.comm.comm_handle();
        let raw_stream = stream.cu_stream() as ncs::cudaStream_t;

        let (src_ptr, _g1) = send_buf.device_ptr(stream);
        let (dst_ptr, _g2) = recv_buf.device_ptr_mut(stream);

        unsafe {
            ncr::group_start()
                .map_err(|e| SparkError::Other(format!("nccl group_start: {e:?}")))?;

            ncr::send(
                src_ptr as *const _,
                count,
                ncs::ncclDataType_t::ncclBfloat16,
                send_to as i32,
                raw_comm,
                raw_stream,
            )
            .map_err(|e| SparkError::Other(format!("nccl send: {e:?}")))?;

            ncr::recv(
                dst_ptr as *mut _,
                count,
                ncs::ncclDataType_t::ncclBfloat16,
                recv_from as i32,
                raw_comm,
                raw_stream,
            )
            .map_err(|e| SparkError::Other(format!("nccl recv: {e:?}")))?;

            ncr::group_end().map_err(|e| SparkError::Other(format!("nccl group_end: {e:?}")))?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Helper: cudarc 0.15 makes `Comm::comm` (the raw `ncclComm_t`) private. We
// need the raw handle for the BF16 send/recv path (since `Comm::send` requires
// `T: NcclType` and BF16 is gated on the `f16` feature). Workaround: extension
// trait that does a `transmute` against the layout (we control cudarc version).
// ---------------------------------------------------------------------------

trait CommHandleExt {
    fn comm_handle(&self) -> ncs::ncclComm_t;
}

impl CommHandleExt for Comm {
    fn comm_handle(&self) -> ncs::ncclComm_t {
        // SAFETY: `cudarc::nccl::Comm` layout (verified for 0.15.2):
        //   { comm: ncclComm_t, stream: Arc<CudaStream>, rank: usize, world_size: usize }
        // The first field is the raw handle. Read it via pointer cast.
        // If cudarc adds `pub fn comm(&self) -> ncclComm_t`, replace this.
        unsafe { *(self as *const Comm as *const ncs::ncclComm_t) }
    }
}
