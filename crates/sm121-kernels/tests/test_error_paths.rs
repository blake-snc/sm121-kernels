//! Edge case and error path tests for sm121-kernels dispatch functions.
//!
//! These tests verify that invalid inputs are rejected at the Rust level
//! with appropriate SparkError::InvalidArgument errors, without launching
//! any GPU kernels.

use sm121_kernels::{activation, attention, device, gemm, moe, norm, rope, sampling, SparkError};

fn is_invalid_arg(e: &SparkError) -> bool {
    matches!(e, SparkError::InvalidArgument(_))
}

// ─── GEMM error paths ───────────────────────────────────────────────

#[test]
fn test_gemm_bf16_zero_dims() {
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();
    let a = stream.alloc_zeros::<u16>(1).unwrap();
    let b = stream.alloc_zeros::<u16>(1).unwrap();
    let mut c = stream.alloc_zeros::<u16>(1).unwrap();

    let err = gemm::gemm_bf16(&ctx, &stream, &a, &b, &mut c, 0, 16, 16).unwrap_err();
    assert!(is_invalid_arg(&err), "M=0 should fail: {err}");

    let err = gemm::gemm_bf16(&ctx, &stream, &a, &b, &mut c, 16, 0, 16).unwrap_err();
    assert!(is_invalid_arg(&err), "N=0 should fail: {err}");

    let err = gemm::gemm_bf16(&ctx, &stream, &a, &b, &mut c, 16, 16, 0).unwrap_err();
    assert!(is_invalid_arg(&err), "K=0 should fail: {err}");
}

#[test]
fn test_gemm_bf16_buffer_too_small() {
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();
    // M=16, N=16, K=16 → needs A=256, B=256, C=256 elements
    let small = stream.alloc_zeros::<u16>(1).unwrap();
    let ok = stream.alloc_zeros::<u16>(256).unwrap();
    let mut c_ok = stream.alloc_zeros::<u16>(256).unwrap();
    let mut c_small = stream.alloc_zeros::<u16>(1).unwrap();

    let err = gemm::gemm_bf16(&ctx, &stream, &small, &ok, &mut c_ok, 16, 16, 16).unwrap_err();
    assert!(is_invalid_arg(&err), "A too small: {err}");

    let err = gemm::gemm_bf16(&ctx, &stream, &ok, &small, &mut c_ok, 16, 16, 16).unwrap_err();
    assert!(is_invalid_arg(&err), "B too small: {err}");

    let err = gemm::gemm_bf16(&ctx, &stream, &ok, &ok, &mut c_small, 16, 16, 16).unwrap_err();
    assert!(is_invalid_arg(&err), "C too small: {err}");
}

#[test]
fn test_gemm_bf16_mma_misaligned() {
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();
    let buf = stream.alloc_zeros::<u16>(1024 * 1024).unwrap();
    let mut c = stream.alloc_zeros::<u16>(1024 * 1024).unwrap();

    // BF16 MMA requires M%128==0, N%64==0, K%16==0
    let err = gemm::gemm_bf16_mma(&ctx, &stream, &buf, &buf, &mut c, 33, 64, 16).unwrap_err();
    assert!(is_invalid_arg(&err), "M=33 not divisible by 128: {err}");

    let err = gemm::gemm_bf16_mma(&ctx, &stream, &buf, &buf, &mut c, 128, 33, 16).unwrap_err();
    assert!(is_invalid_arg(&err), "N=33 not divisible by 64: {err}");

    let err = gemm::gemm_bf16_mma(&ctx, &stream, &buf, &buf, &mut c, 128, 64, 17).unwrap_err();
    assert!(is_invalid_arg(&err), "K=17 not divisible by 16: {err}");
}

#[test]
fn test_gemm_fp8_mma_misaligned() {
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();
    let buf = stream.alloc_zeros::<u8>(1024 * 1024).unwrap();
    let mut c = stream.alloc_zeros::<u16>(1024 * 1024).unwrap();

    // FP8 MMA requires M%32==0, N%32==0, K%32==0
    let err = gemm::gemm_fp8_mma(&ctx, &stream, &buf, &buf, &mut c, 33, 32, 32).unwrap_err();
    assert!(is_invalid_arg(&err), "M=33: {err}");

    let err = gemm::gemm_fp8_mma(&ctx, &stream, &buf, &buf, &mut c, 32, 33, 32).unwrap_err();
    assert!(is_invalid_arg(&err), "N=33: {err}");

    let err = gemm::gemm_fp8_mma(&ctx, &stream, &buf, &buf, &mut c, 32, 32, 33).unwrap_err();
    assert!(is_invalid_arg(&err), "K=33: {err}");
}

#[test]
fn test_gemm_nvfp4_mma_misaligned() {
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();
    let buf = stream.alloc_zeros::<u8>(1024 * 1024).unwrap();
    let mut c = stream.alloc_zeros::<u16>(1024 * 1024).unwrap();

    // NVFP4 MMA requires M%32==0, N%32==0, K%64==0
    let err = gemm::gemm_nvfp4_mma(&ctx, &stream, &buf, &buf, &mut c, &buf, &buf, 33, 32, 64)
        .unwrap_err();
    assert!(is_invalid_arg(&err), "M=33: {err}");

    let err = gemm::gemm_nvfp4_mma(&ctx, &stream, &buf, &buf, &mut c, &buf, &buf, 32, 32, 63)
        .unwrap_err();
    assert!(is_invalid_arg(&err), "K=63: {err}");
}

#[test]
fn test_gemm_w4a16_mma_misaligned() {
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();
    let a = stream.alloc_zeros::<u16>(1024 * 1024).unwrap();
    let w = stream.alloc_zeros::<u8>(1024 * 1024).unwrap();
    let mut c = stream.alloc_zeros::<u16>(1024 * 1024).unwrap();
    let scales = stream.alloc_zeros::<u16>(1024).unwrap();
    let zeros = stream.alloc_zeros::<u16>(1024).unwrap();

    // W4A16 MMA requires M%32==0, N%32==0, K%16==0
    let err = gemm::gemm_w4a16_mma(&ctx, &stream, &a, &w, &mut c, &scales, &zeros, 33, 32, 16)
        .unwrap_err();
    assert!(is_invalid_arg(&err), "M=33: {err}");

    let err = gemm::gemm_w4a16_mma(&ctx, &stream, &a, &w, &mut c, &scales, &zeros, 32, 32, 15)
        .unwrap_err();
    assert!(is_invalid_arg(&err), "K=15: {err}");
}

// ─── Attention error paths ──────────────────────────────────────────

// Exercises the experimental pre-v3 baseline kernel.
#[cfg(feature = "experimental")]
#[test]
fn test_attention_zero_dims() {
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();
    let buf = stream.alloc_zeros::<u16>(1).unwrap();
    let mut o = stream.alloc_zeros::<u16>(1).unwrap();

    let err = attention::flash_attn_bf16_d128(
        &ctx, &stream, &buf, &buf, &buf, &mut o, 0, 1, 128, 128, 0.088,
    )
    .unwrap_err();
    assert!(is_invalid_arg(&err), "batch=0: {err}");

    let err = attention::flash_attn_bf16_d128(
        &ctx, &stream, &buf, &buf, &buf, &mut o, 1, 0, 128, 128, 0.088,
    )
    .unwrap_err();
    assert!(is_invalid_arg(&err), "num_heads=0: {err}");

    let err = attention::flash_attn_bf16_d128(
        &ctx, &stream, &buf, &buf, &buf, &mut o, 1, 1, 0, 128, 0.088,
    )
    .unwrap_err();
    assert!(is_invalid_arg(&err), "seq_q=0: {err}");

    let err = attention::flash_attn_bf16_d128(
        &ctx, &stream, &buf, &buf, &buf, &mut o, 1, 1, 128, 0, 0.088,
    )
    .unwrap_err();
    assert!(is_invalid_arg(&err), "seq_kv=0: {err}");
}

#[test]
fn test_attention_v3_zero_dims() {
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();
    let buf = stream.alloc_zeros::<u16>(1).unwrap();
    let mut o = stream.alloc_zeros::<u16>(1).unwrap();

    let err = attention::flash_attn_bf16_v3_d128(
        &ctx, &stream, &buf, &buf, &buf, &mut o, 0, 1, 128, 128, 0.088,
    )
    .unwrap_err();
    assert!(is_invalid_arg(&err), "V3 batch=0: {err}");

    let err = attention::flash_attn_bf16_v3_d128_causal(
        &ctx, &stream, &buf, &buf, &buf, &mut o, 1, 0, 128, 128, 0.088,
    )
    .unwrap_err();
    assert!(is_invalid_arg(&err), "V3 causal num_heads=0: {err}");
}

// Exercises the experimental pre-v3 FP8 baseline kernels.
#[cfg(feature = "experimental")]
#[test]
fn test_attention_fp8_zero_dims() {
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();
    let buf = stream.alloc_zeros::<u8>(1).unwrap();
    let mut o = stream.alloc_zeros::<u16>(1).unwrap();

    let err = attention::flash_attn_fp8_d128(
        &ctx, &stream, &buf, &buf, &buf, &mut o, 0, 1, 128, 128, 0.088,
    )
    .unwrap_err();
    assert!(is_invalid_arg(&err), "FP8 batch=0: {err}");

    let err = attention::flash_attn_fp8_d128_causal(
        &ctx, &stream, &buf, &buf, &buf, &mut o, 1, 1, 0, 128, 0.088,
    )
    .unwrap_err();
    assert!(is_invalid_arg(&err), "FP8 causal seq_q=0: {err}");
}

// Exercises the experimental pre-v3 baseline kernel.
#[cfg(feature = "experimental")]
#[test]
fn test_attention_buffer_too_small() {
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();
    // batch=1, heads=1, seq=16, d=128 → Q/K/V need 16*128=2048 elements
    let small = stream.alloc_zeros::<u16>(1).unwrap();
    let ok = stream.alloc_zeros::<u16>(2048).unwrap();
    let mut o_ok = stream.alloc_zeros::<u16>(2048).unwrap();
    let mut o_small = stream.alloc_zeros::<u16>(1).unwrap();

    let err = attention::flash_attn_bf16_d128(
        &ctx, &stream, &small, &ok, &ok, &mut o_ok, 1, 1, 16, 16, 0.088,
    )
    .unwrap_err();
    assert!(is_invalid_arg(&err), "Q too small: {err}");

    let err = attention::flash_attn_bf16_d128(
        &ctx, &stream, &ok, &small, &ok, &mut o_ok, 1, 1, 16, 16, 0.088,
    )
    .unwrap_err();
    assert!(is_invalid_arg(&err), "K too small: {err}");

    let err = attention::flash_attn_bf16_d128(
        &ctx, &stream, &ok, &ok, &small, &mut o_ok, 1, 1, 16, 16, 0.088,
    )
    .unwrap_err();
    assert!(is_invalid_arg(&err), "V too small: {err}");

    let err = attention::flash_attn_bf16_d128(
        &ctx,
        &stream,
        &ok,
        &ok,
        &ok,
        &mut o_small,
        1,
        1,
        16,
        16,
        0.088,
    )
    .unwrap_err();
    assert!(is_invalid_arg(&err), "O too small: {err}");
}

#[test]
fn test_attention_varlen_zero_dims() {
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();
    let buf = stream.alloc_zeros::<u16>(1).unwrap();
    let cu = stream.alloc_zeros::<u32>(2).unwrap();
    let mut o = stream.alloc_zeros::<u16>(1).unwrap();

    let err = attention::flash_attn_bf16_varlen_d128(
        &ctx, &stream, &buf, &buf, &buf, &mut o, &cu, &cu, 0, 1, 16, 0.088,
    )
    .unwrap_err();
    assert!(is_invalid_arg(&err), "varlen batch=0: {err}");

    let err = attention::flash_attn_bf16_varlen_d128(
        &ctx, &stream, &buf, &buf, &buf, &mut o, &cu, &cu, 1, 0, 16, 0.088,
    )
    .unwrap_err();
    assert!(is_invalid_arg(&err), "varlen num_heads=0: {err}");
}

// ─── RMSNorm error paths ────────────────────────────────────────────

#[test]
fn test_rmsnorm_zero_dims() {
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();
    let buf = stream.alloc_zeros::<u16>(1).unwrap();
    let mut out = stream.alloc_zeros::<u16>(1).unwrap();

    let err = norm::rmsnorm_bf16(&ctx, &stream, &buf, &mut out, &buf, 0, 1e-5, 1).unwrap_err();
    assert!(is_invalid_arg(&err), "hidden_dim=0: {err}");

    let err = norm::rmsnorm_bf16(&ctx, &stream, &buf, &mut out, &buf, 128, 1e-5, 0).unwrap_err();
    assert!(is_invalid_arg(&err), "num_rows=0: {err}");
}

#[test]
fn test_rmsnorm_buffer_too_small() {
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();
    let small = stream.alloc_zeros::<u16>(1).unwrap();
    let ok = stream.alloc_zeros::<u16>(256).unwrap();
    let mut out_ok = stream.alloc_zeros::<u16>(256).unwrap();
    let mut out_small = stream.alloc_zeros::<u16>(1).unwrap();
    let w = stream.alloc_zeros::<u16>(128).unwrap();
    let w_small = stream.alloc_zeros::<u16>(1).unwrap();

    // hidden_dim=128, num_rows=2 → need 256 elements
    let err = norm::rmsnorm_bf16(&ctx, &stream, &small, &mut out_ok, &w, 128, 1e-5, 2).unwrap_err();
    assert!(is_invalid_arg(&err), "x too small: {err}");

    let err = norm::rmsnorm_bf16(&ctx, &stream, &ok, &mut out_small, &w, 128, 1e-5, 2).unwrap_err();
    assert!(is_invalid_arg(&err), "out too small: {err}");

    let err =
        norm::rmsnorm_bf16(&ctx, &stream, &ok, &mut out_ok, &w_small, 128, 1e-5, 2).unwrap_err();
    assert!(is_invalid_arg(&err), "weight too small: {err}");
}

// ─── RoPE error paths ───────────────────────────────────────────────

#[test]
fn test_rope_odd_dim() {
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();
    let mut x = stream.alloc_zeros::<u16>(1024).unwrap();
    let cos = stream.alloc_zeros::<f32>(1024).unwrap();
    let sin = stream.alloc_zeros::<f32>(1024).unwrap();

    let err = rope::rope_bf16(&ctx, &stream, &mut x, &cos, &sin, 1, 1, 1, 3).unwrap_err();
    assert!(is_invalid_arg(&err), "dim=3 (odd): {err}");
}

#[test]
fn test_rope_zero_dims() {
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();
    let mut x = stream.alloc_zeros::<u16>(1).unwrap();
    let cos = stream.alloc_zeros::<f32>(1).unwrap();
    let sin = stream.alloc_zeros::<f32>(1).unwrap();

    let err = rope::rope_bf16(&ctx, &stream, &mut x, &cos, &sin, 0, 1, 1, 2).unwrap_err();
    assert!(is_invalid_arg(&err), "batch=0: {err}");

    let err = rope::rope_bf16(&ctx, &stream, &mut x, &cos, &sin, 1, 0, 1, 2).unwrap_err();
    assert!(is_invalid_arg(&err), "seq_len=0: {err}");

    let err = rope::rope_bf16(&ctx, &stream, &mut x, &cos, &sin, 1, 1, 0, 2).unwrap_err();
    assert!(is_invalid_arg(&err), "heads=0: {err}");

    let err = rope::rope_bf16(&ctx, &stream, &mut x, &cos, &sin, 1, 1, 1, 0).unwrap_err();
    assert!(is_invalid_arg(&err), "dim=0: {err}");
}

#[test]
fn test_rope_dim_exceeds_thread_limit() {
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();
    // dim=2050 → dim/2 = 1025 > 1024 max threads
    let mut x = stream.alloc_zeros::<u16>(2050).unwrap();
    let cos = stream.alloc_zeros::<f32>(1025).unwrap();
    let sin = stream.alloc_zeros::<f32>(1025).unwrap();

    let err = rope::rope_bf16(&ctx, &stream, &mut x, &cos, &sin, 1, 1, 1, 2050).unwrap_err();
    assert!(is_invalid_arg(&err), "dim/2 > 1024: {err}");
}

// ─── Activation error paths ─────────────────────────────────────────

#[test]
fn test_activation_zero_dims() {
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();
    let buf = stream.alloc_zeros::<u16>(1).unwrap();
    let mut out = stream.alloc_zeros::<u16>(1).unwrap();

    let err = activation::silu_mul_bf16(&ctx, &stream, &buf, &mut out, 0, 1).unwrap_err();
    assert!(is_invalid_arg(&err), "SiLU total_out_elems=0: {err}");

    let err = activation::silu_mul_bf16(&ctx, &stream, &buf, &mut out, 1, 0).unwrap_err();
    assert!(is_invalid_arg(&err), "SiLU d=0: {err}");

    let err = activation::gelu_mul_bf16(&ctx, &stream, &buf, &mut out, 0, 1).unwrap_err();
    assert!(is_invalid_arg(&err), "GeLU total_out_elems=0: {err}");

    let err = activation::gelu_tanh_mul_bf16(&ctx, &stream, &buf, &mut out, 0, 1).unwrap_err();
    assert!(is_invalid_arg(&err), "GeLU-tanh total_out_elems=0: {err}");
}

#[test]
fn test_activation_buffer_too_small() {
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();
    // total_out_elems=256, d=16 → input needs 512, output needs 256
    let small_in = stream.alloc_zeros::<u16>(1).unwrap();
    let ok_in = stream.alloc_zeros::<u16>(512).unwrap();
    let mut out_ok = stream.alloc_zeros::<u16>(256).unwrap();
    let mut out_small = stream.alloc_zeros::<u16>(1).unwrap();

    let err =
        activation::silu_mul_bf16(&ctx, &stream, &small_in, &mut out_ok, 256, 16).unwrap_err();
    assert!(is_invalid_arg(&err), "input too small: {err}");

    let err =
        activation::silu_mul_bf16(&ctx, &stream, &ok_in, &mut out_small, 256, 16).unwrap_err();
    assert!(is_invalid_arg(&err), "output too small: {err}");
}

// ─── Sampling error paths ───────────────────────────────────────────

#[test]
fn test_sampling_zero_dims() {
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();
    let logits = stream.alloc_zeros::<u16>(1).unwrap();
    let mut indices = stream.alloc_zeros::<u32>(1).unwrap();
    let mut values = stream.alloc_zeros::<u16>(1).unwrap();

    let err = sampling::topk_sampling(
        &ctx,
        &stream,
        &logits,
        &mut indices,
        &mut values,
        0,
        100,
        1,
        1.0,
    )
    .unwrap_err();
    assert!(is_invalid_arg(&err), "batch_size=0: {err}");

    let err = sampling::topk_sampling(
        &ctx,
        &stream,
        &logits,
        &mut indices,
        &mut values,
        1,
        0,
        1,
        1.0,
    )
    .unwrap_err();
    assert!(is_invalid_arg(&err), "vocab_size=0: {err}");

    let err = sampling::topk_sampling(
        &ctx,
        &stream,
        &logits,
        &mut indices,
        &mut values,
        1,
        100,
        0,
        1.0,
    )
    .unwrap_err();
    assert!(is_invalid_arg(&err), "k=0: {err}");
}

#[test]
fn test_sampling_buffer_too_small() {
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();
    // batch=2, vocab=100, k=5 → logits=200, indices=10, values=10
    let logits_small = stream.alloc_zeros::<u16>(1).unwrap();
    let logits_ok = stream.alloc_zeros::<u16>(200).unwrap();
    let mut indices_small = stream.alloc_zeros::<u32>(1).unwrap();
    let mut indices_ok = stream.alloc_zeros::<u32>(10).unwrap();
    let mut values_small = stream.alloc_zeros::<u16>(1).unwrap();
    let mut values_ok = stream.alloc_zeros::<u16>(10).unwrap();

    let err = sampling::topk_sampling(
        &ctx,
        &stream,
        &logits_small,
        &mut indices_ok,
        &mut values_ok,
        2,
        100,
        5,
        1.0,
    )
    .unwrap_err();
    assert!(is_invalid_arg(&err), "logits too small: {err}");

    let err = sampling::topk_sampling(
        &ctx,
        &stream,
        &logits_ok,
        &mut indices_small,
        &mut values_ok,
        2,
        100,
        5,
        1.0,
    )
    .unwrap_err();
    assert!(is_invalid_arg(&err), "indices too small: {err}");

    let err = sampling::topk_sampling(
        &ctx,
        &stream,
        &logits_ok,
        &mut indices_ok,
        &mut values_small,
        2,
        100,
        5,
        1.0,
    )
    .unwrap_err();
    assert!(is_invalid_arg(&err), "values too small: {err}");
}

// ─── MoE error paths ───────────────────────────────────────────────

#[test]
fn test_moe_zero_dims() {
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();
    let logits = stream.alloc_zeros::<u16>(1).unwrap();
    let mut ids = stream.alloc_zeros::<u32>(1).unwrap();
    let mut weights = stream.alloc_zeros::<u16>(1).unwrap();

    let err =
        moe::moe_routing(&ctx, &stream, &logits, &mut ids, &mut weights, 0, 8, 2).unwrap_err();
    assert!(is_invalid_arg(&err), "num_tokens=0: {err}");

    let err =
        moe::moe_routing(&ctx, &stream, &logits, &mut ids, &mut weights, 1, 0, 1).unwrap_err();
    assert!(is_invalid_arg(&err), "num_experts=0: {err}");

    let err =
        moe::moe_routing(&ctx, &stream, &logits, &mut ids, &mut weights, 1, 8, 0).unwrap_err();
    assert!(is_invalid_arg(&err), "top_k=0: {err}");
}

#[test]
fn test_moe_topk_exceeds_experts() {
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();
    let logits = stream.alloc_zeros::<u16>(16).unwrap();
    let mut ids = stream.alloc_zeros::<u32>(16).unwrap();
    let mut weights = stream.alloc_zeros::<u16>(16).unwrap();

    let err =
        moe::moe_routing(&ctx, &stream, &logits, &mut ids, &mut weights, 1, 4, 5).unwrap_err();
    assert!(is_invalid_arg(&err), "top_k=5 > num_experts=4: {err}");
}

#[test]
fn test_moe_buffer_too_small() {
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();
    // num_tokens=4, num_experts=8, top_k=2 → logits=32, ids=8, weights=8
    let logits_small = stream.alloc_zeros::<u16>(1).unwrap();
    let logits_ok = stream.alloc_zeros::<u16>(32).unwrap();
    let mut ids_small = stream.alloc_zeros::<u32>(1).unwrap();
    let mut ids_ok = stream.alloc_zeros::<u32>(8).unwrap();
    let mut weights_small = stream.alloc_zeros::<u16>(1).unwrap();
    let mut weights_ok = stream.alloc_zeros::<u16>(8).unwrap();

    let err = moe::moe_routing(
        &ctx,
        &stream,
        &logits_small,
        &mut ids_ok,
        &mut weights_ok,
        4,
        8,
        2,
    )
    .unwrap_err();
    assert!(is_invalid_arg(&err), "logits too small: {err}");

    let err = moe::moe_routing(
        &ctx,
        &stream,
        &logits_ok,
        &mut ids_small,
        &mut weights_ok,
        4,
        8,
        2,
    )
    .unwrap_err();
    assert!(is_invalid_arg(&err), "expert_ids too small: {err}");

    let err = moe::moe_routing(
        &ctx,
        &stream,
        &logits_ok,
        &mut ids_ok,
        &mut weights_small,
        4,
        8,
        2,
    )
    .unwrap_err();
    assert!(is_invalid_arg(&err), "weights too small: {err}");
}
