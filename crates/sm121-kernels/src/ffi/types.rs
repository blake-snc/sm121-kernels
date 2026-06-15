use std::os::raw::{c_float, c_int, c_uint};

/// Status codes returned by C API functions.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparkStatus {
    Success = 0,
    ErrorInvalidArgument = 1,
    ErrorCudaLaunch = 2,
    ErrorKernelNotFound = 3,
    ErrorInternal = 4,
}

/// Data type enumeration.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparkDtype {
    BF16 = 0,
    FP8E4M3 = 1,
    FP32 = 2,
    U32 = 3,
    NVFP4 = 4,
    W4A16 = 5,
}

/// Parameters for Flash Attention dispatch.
///
/// All device memory pointers (q, k, v, o) must be:
/// - Non-null valid CUdeviceptr values on the active device
/// - 16-byte aligned (required by ldmatrix/cp.async instructions)
/// - Sized to hold batch * num_heads * seq_{q,kv} * head_dim elements
///
/// Only head_dim=128 is currently supported.
#[repr(C)]
pub struct SparkFlashAttnParams {
    pub q: *const u8,
    pub k: *const u8,
    pub v: *const u8,
    pub o: *mut u8,
    pub batch: c_uint,
    pub num_heads: c_uint,
    pub seq_q: c_uint,
    pub seq_kv: c_uint,
    pub head_dim: c_uint,
    pub scale: c_float,
    pub dtype: SparkDtype,
    pub causal: c_int,
}

/// Parameters for GEMM dispatch.
///
/// Device pointers must be 16-byte aligned.
/// BF16 MMA requires M, N divisible by 32 and K divisible by 16.
/// FP8 MMA requires K divisible by 32.
#[repr(C)]
pub struct SparkGemmParams {
    pub a: *const u8,
    pub b: *const u8,
    pub c: *mut u8,
    pub m: c_uint,
    pub n: c_uint,
    pub k: c_uint,
    pub dtype: SparkDtype,
}

/// Parameters for top-k sampling dispatch.
#[repr(C)]
pub struct SparkTopkParams {
    pub logits: *const u8,
    pub indices: *mut u32,
    pub values: *mut u8,
    pub batch_size: c_uint,
    pub vocab_size: c_uint,
    pub k: c_uint,
    pub temperature: c_float,
}

/// Parameters for MoE routing dispatch.
#[repr(C)]
pub struct SparkMoeRoutingParams {
    pub logits: *const u8,
    pub expert_ids: *mut u32,
    pub weights: *mut u8,
    pub num_tokens: c_uint,
    pub num_experts: c_uint,
    pub top_k: c_uint,
}

/// Parameters for RMSNorm dispatch.
#[repr(C)]
pub struct SparkRmsNormParams {
    pub x: *const u8,
    pub out: *mut u8,
    pub weight: *const u8,
    pub hidden_dim: c_uint,
    pub eps: c_float,
    pub num_rows: c_uint,
}

/// Parameters for RoPE dispatch.
#[repr(C)]
pub struct SparkRopeParams {
    pub x: *mut u8,
    pub cos_cache: *const u8,
    pub sin_cache: *const u8,
    pub batch: c_uint,
    pub seq_len: c_uint,
    pub heads: c_uint,
    pub dim: c_uint,
}

/// Activation type for dispatch.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparkActivationType {
    SiluMul = 0,
    GeluMul = 1,
    GeluTanhMul = 2,
}

/// Parameters for fused activation dispatch.
#[repr(C)]
pub struct SparkActivationParams {
    pub input: *const u8,
    pub out: *mut u8,
    pub total_out_elems: c_uint,
    pub d: c_uint,
    pub activation: SparkActivationType,
}

/// Parameters for NVFP4 block-scaled GEMM dispatch.
#[repr(C)]
pub struct SparkNvfp4GemmParams {
    pub a: *const u8,
    pub b: *const u8,
    pub c: *mut u8,
    pub scale_a: *const u8,
    pub scale_b: *const u8,
    pub m: c_uint,
    pub n: c_uint,
    pub k: c_uint,
}

/// Parameters for W4A16 dequant GEMM dispatch.
#[repr(C)]
pub struct SparkW4a16GemmParams {
    pub a: *const u8,
    pub w: *const u8,
    pub c: *mut u8,
    pub scales: *const u8,
    pub zeros: *const u8,
    pub m: c_uint,
    pub n: c_uint,
    pub k: c_uint,
}

/// Parameters for variable-length Flash Attention dispatch.
#[repr(C)]
pub struct SparkVarlenFlashAttnParams {
    pub q: *const u8,
    pub k: *const u8,
    pub v: *const u8,
    pub o: *mut u8,
    pub cu_seqlens_q: *const u32,
    pub cu_seqlens_k: *const u32,
    pub batch: c_uint,
    pub num_heads: c_uint,
    pub max_seqlen_q: c_uint,
    pub scale: c_float,
}
