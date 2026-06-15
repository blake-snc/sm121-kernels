//! Public-mirror lib.rs: kept kernel modules only.
#![deny(missing_docs)]
/// Fused element-wise activation kernels (SiLU, GeLU, gating, casts).
pub mod activation;
/// Flash-attention and decode-attention kernel dispatch.
pub mod attention;
/// Device initialization and SM121 capability checks.
pub mod device;
#[allow(missing_docs)]
pub mod distributed;
/// Token embedding lookup kernels.
pub mod embedding;
/// Error and result types shared across the crate.
pub mod error;
#[cfg(feature = "c-api")]
#[allow(missing_docs)]
pub mod ffi;
/// GEMM and GEMV kernel dispatch.
pub mod gemm;
/// Paged and contiguous KV-cache write/append kernels.
pub mod kv_cache;
/// Linear-attention kernels (GatedDeltaNet, Mamba2 selective scan).
pub mod linear_attention;
/// Kernel module loading and the cubin cache.
pub mod module;
/// Mixture-of-experts routing and grouped GEMM/GEMV kernels.
pub mod moe;
/// RMSNorm and related normalization kernels.
pub mod norm;
/// Weight/activation quantization and dequantization kernels.
pub mod quantization;
/// Rotary position embedding kernels.
pub mod rope;
/// Sampling kernels (argmax, top-k, top-p).
pub mod sampling;
pub use error::{Result, SparkError};
