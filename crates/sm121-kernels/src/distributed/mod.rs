//! Distributed primitives — ring attention, expert parallel, sequence parallel.
//!
//! # Status
//!
//! - **Ring attention (single-GPU simulation)**: shipped. Wraps the existing
//!   split-K + flash-decoding-combine kernels into a single-call API. Each
//!   "ring step" computes attention against a contiguous KV chunk; the combine
//!   merges all chunks via online LSE merging. On a single GPU this is
//!   functionally identical to a full attention call but exposes the per-step
//!   structure that maps 1:1 to multi-node ring attention.
//!
//! - **Multi-node ring attention (NCCL)**: not yet wired. The single-GPU
//!   orchestrator is structured so that the only differences for true multi-
//!   node are: (a) each node holds only its KV slice in memory, and (b) the
//!   ring rotation between steps is an NCCL send/recv pair instead of a
//!   pointer-bump. cudarc has NCCL bindings (`cudarc::nccl`) but we don't yet
//!   ship a multi-node example.

pub mod nccl_transport;
pub mod ring_attention;
pub mod ring_attention_distributed;

pub use nccl_transport::NcclTransport;
pub use ring_attention::ring_attention_bf16;
pub use ring_attention_distributed::{
    ring_attention_bf16_distributed, LoopbackTransport, RingTransport,
};
