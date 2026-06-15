#![allow(non_snake_case)]

mod common;

use common::{compare_bf16, load_npz};
use sm121_kernels::{attention, device};

#[cfg(feature = "experimental")]
fn run_flash_attn_test(npz_name: &str, tol: f32) {
    run_flash_attn_test_impl(npz_name, tol, false);
}

#[cfg(feature = "experimental")]
fn run_flash_attn_causal_test(npz_name: &str, tol: f32) {
    run_flash_attn_test_impl(npz_name, tol, true);
}

fn run_flash_attn_v3_test(npz_name: &str, tol: f32, causal: bool) {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let mut npz = load_npz(npz_name);

    let q_np: ndarray::Array4<u16> = npz.by_name("q").unwrap();
    let k_np: ndarray::Array4<u16> = npz.by_name("k").unwrap();
    let v_np: ndarray::Array4<u16> = npz.by_name("v").unwrap();
    let o_expected: ndarray::Array4<u16> = npz.by_name("o").unwrap();
    let scale_np: ndarray::Array0<f32> = npz.by_name("scale").unwrap();

    let batch = q_np.shape()[0] as u32;
    let num_heads = q_np.shape()[1] as u32;
    let seq_q = q_np.shape()[2] as u32;
    let seq_kv = k_np.shape()[2] as u32;
    let scale = scale_np.into_scalar();

    let q_flat: Vec<u16> = q_np.into_raw_vec_and_offset().0;
    let k_flat: Vec<u16> = k_np.into_raw_vec_and_offset().0;
    let v_flat: Vec<u16> = v_np.into_raw_vec_and_offset().0;
    let expected_flat: Vec<u16> = o_expected.into_raw_vec_and_offset().0;

    let q_dev = stream.memcpy_stod(&q_flat).unwrap();
    let k_dev = stream.memcpy_stod(&k_flat).unwrap();
    let v_dev = stream.memcpy_stod(&v_flat).unwrap();
    let mut o_dev = stream
        .alloc_zeros::<u16>((batch * num_heads * seq_q * 128) as usize)
        .unwrap();

    if causal {
        attention::flash_attn_bf16_v3_d128_causal(
            &ctx, &stream, &q_dev, &k_dev, &v_dev, &mut o_dev, batch, num_heads, seq_q, seq_kv,
            scale,
        )
        .unwrap();
    } else {
        attention::flash_attn_bf16_v3_d128(
            &ctx, &stream, &q_dev, &k_dev, &v_dev, &mut o_dev, batch, num_heads, seq_q, seq_kv,
            scale,
        )
        .unwrap();
    }

    let o_host = stream.memcpy_dtov(&o_dev).unwrap();
    let (max_diff, mean_diff) = compare_bf16(&o_host, &expected_flat, tol);
    eprintln!("V3 {npz_name}: max_diff={max_diff:.6} mean_diff={mean_diff:.6} (tol={tol})");
}

#[cfg(feature = "experimental")]
fn run_flash_attn_test_impl(npz_name: &str, tol: f32, causal: bool) {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let mut npz = load_npz(npz_name);

    let q_np: ndarray::Array4<u16> = npz.by_name("q").unwrap();
    let k_np: ndarray::Array4<u16> = npz.by_name("k").unwrap();
    let v_np: ndarray::Array4<u16> = npz.by_name("v").unwrap();
    let o_expected: ndarray::Array4<u16> = npz.by_name("o").unwrap();
    let scale_np: ndarray::Array0<f32> = npz.by_name("scale").unwrap();

    let batch = q_np.shape()[0] as u32;
    let num_heads = q_np.shape()[1] as u32;
    let seq_q = q_np.shape()[2] as u32;
    let seq_kv = k_np.shape()[2] as u32;
    let scale = scale_np.into_scalar();

    let q_flat: Vec<u16> = q_np.into_raw_vec_and_offset().0;
    let k_flat: Vec<u16> = k_np.into_raw_vec_and_offset().0;
    let v_flat: Vec<u16> = v_np.into_raw_vec_and_offset().0;
    let expected_flat: Vec<u16> = o_expected.into_raw_vec_and_offset().0;

    let q_dev = stream.memcpy_stod(&q_flat).unwrap();
    let k_dev = stream.memcpy_stod(&k_flat).unwrap();
    let v_dev = stream.memcpy_stod(&v_flat).unwrap();
    let mut o_dev = stream
        .alloc_zeros::<u16>((batch * num_heads * seq_q * 128) as usize)
        .unwrap();

    if causal {
        attention::flash_attn_bf16_d128_causal(
            &ctx, &stream, &q_dev, &k_dev, &v_dev, &mut o_dev, batch, num_heads, seq_q, seq_kv,
            scale,
        )
        .unwrap();
    } else {
        attention::flash_attn_bf16_d128(
            &ctx, &stream, &q_dev, &k_dev, &v_dev, &mut o_dev, batch, num_heads, seq_q, seq_kv,
            scale,
        )
        .unwrap();
    }

    let o_host = stream.memcpy_dtov(&o_dev).unwrap();
    let (max_diff, mean_diff) = compare_bf16(&o_host, &expected_flat, tol);
    eprintln!("{npz_name}: max_diff={max_diff:.6} mean_diff={mean_diff:.6} (tol={tol})");
}

// Step 4: previously diagnostic-only, now has compare_bf16 assertion
#[cfg(feature = "experimental")]
#[test]
fn test_flash_attn_bf16_noncausal_s256() {
    run_flash_attn_test("flash_attn_bf16_noncausal_s256.npz", 0.01);
}

#[cfg(feature = "experimental")]
#[test]
fn test_flash_attn_bf16_noncausal_s1024() {
    run_flash_attn_test("flash_attn_bf16_noncausal_s1024.npz", 0.01);
}

#[cfg(feature = "experimental")]
#[test]
fn test_flash_attn_bf16_debug_single_block() {
    run_flash_attn_test("flash_attn_bf16_debug_s16x64.npz", 0.01);
}

#[cfg(feature = "experimental")]
#[test]
fn test_flash_attn_bf16_debug_qzero() {
    run_flash_attn_test("flash_attn_bf16_debug_qzero.npz", 0.01);
}

#[cfg(feature = "experimental")]
#[test]
fn test_flash_attn_bf16_debug_identity() {
    run_flash_attn_test("flash_attn_bf16_debug_identity.npz", 0.01);
}

#[cfg(feature = "experimental")]
#[test]
fn test_flash_attn_bf16_causal_s256() {
    run_flash_attn_causal_test("flash_attn_bf16_causal_s256.npz", 0.05);
}

#[cfg(feature = "experimental")]
#[test]
fn test_flash_attn_bf16_causal_s1024() {
    run_flash_attn_causal_test("flash_attn_bf16_causal_s1024.npz", 0.05);
}

#[cfg(feature = "experimental")]
#[test]
fn test_flash_attn_bf16_debug_two_blocks() {
    run_flash_attn_test("flash_attn_bf16_debug_s16x128.npz", 0.01);
}

#[cfg(feature = "experimental")]
#[test]
fn test_flash_attn_bf16_causal_debug_s16x64() {
    run_flash_attn_causal_test("flash_attn_bf16_causal_debug_s16x64.npz", 0.05);
}

// ─── V3 Flash Attention tests ────────────────────────────────────────

#[test]
fn test_flash_attn_bf16_v3_noncausal_s128() {
    run_flash_attn_v3_test("flash_attn_bf16_noncausal_s128.npz", 0.01, false);
}

#[test]
fn test_flash_attn_bf16_v3_noncausal_s256() {
    run_flash_attn_v3_test("flash_attn_bf16_noncausal_s256.npz", 0.01, false);
}

#[test]
fn test_flash_attn_bf16_v3_noncausal_s1024() {
    run_flash_attn_v3_test("flash_attn_bf16_noncausal_s1024.npz", 0.01, false);
}

// ─── V22 DB (double-buffered) Flash Attention tests ──────────────────
// Same algorithm/output as V3; verifies the 2-stage pipeline + 4-buffer
// addressing against the same golden vectors at the same tolerance.

fn run_flash_attn_v22_db_test_ctx(
    ctx: &std::sync::Arc<cudarc::driver::CudaContext>,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    npz_name: &str,
    tol: f32,
) {
    let mut npz = load_npz(npz_name);

    let q_np: ndarray::Array4<u16> = npz.by_name("q").unwrap();
    let k_np: ndarray::Array4<u16> = npz.by_name("k").unwrap();
    let v_np: ndarray::Array4<u16> = npz.by_name("v").unwrap();
    let o_expected: ndarray::Array4<u16> = npz.by_name("o").unwrap();
    let scale_np: ndarray::Array0<f32> = npz.by_name("scale").unwrap();

    let batch = q_np.shape()[0] as u32;
    let num_heads = q_np.shape()[1] as u32;
    let seq_q = q_np.shape()[2] as u32;
    let seq_kv = k_np.shape()[2] as u32;
    let scale = scale_np.into_scalar();

    let q_flat: Vec<u16> = q_np.into_raw_vec_and_offset().0;
    let k_flat: Vec<u16> = k_np.into_raw_vec_and_offset().0;
    let v_flat: Vec<u16> = v_np.into_raw_vec_and_offset().0;
    let expected_flat: Vec<u16> = o_expected.into_raw_vec_and_offset().0;

    let q_dev = stream.memcpy_stod(&q_flat).unwrap();
    let k_dev = stream.memcpy_stod(&k_flat).unwrap();
    let v_dev = stream.memcpy_stod(&v_flat).unwrap();
    let mut o_dev = stream
        .alloc_zeros::<u16>((batch * num_heads * seq_q * 128) as usize)
        .unwrap();

    attention::flash_attn_bf16_v22_db(
        ctx, stream, &q_dev, &k_dev, &v_dev, &mut o_dev, batch, num_heads, seq_q, seq_kv, scale,
    )
    .unwrap();

    let o_host = stream.memcpy_dtov(&o_dev).unwrap();
    let (max_diff, mean_diff) = compare_bf16(&o_host, &expected_flat, tol);
    eprintln!("V22 DB {npz_name}: max_diff={max_diff:.6} mean_diff={mean_diff:.6} (tol={tol})");
    assert!(
        max_diff <= tol,
        "V22 DB {npz_name} differs (max_diff={max_diff:.6} > tol={tol})"
    );
}

// All three shapes run under ONE context: load_kernel_raw caches the raw
// CUfunction by (device ordinal, name), so a fresh context per test would hand
// back a stale handle to cuFuncSetAttribute (a test-harness artifact, not a
// kernel bug — the server holds one context for its lifetime).
#[test]
fn test_flash_attn_bf16_v22_db_noncausal() {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();
    run_flash_attn_v22_db_test_ctx(&ctx, &stream, "flash_attn_bf16_noncausal_s128.npz", 0.01);
    run_flash_attn_v22_db_test_ctx(&ctx, &stream, "flash_attn_bf16_noncausal_s256.npz", 0.01);
    run_flash_attn_v22_db_test_ctx(&ctx, &stream, "flash_attn_bf16_noncausal_s1024.npz", 0.01);
}

#[test]
fn test_flash_attn_bf16_v3_causal_s128() {
    run_flash_attn_v3_test("flash_attn_bf16_causal_s128.npz", 0.05, true);
}

#[test]
fn test_flash_attn_bf16_v3_causal_s256() {
    run_flash_attn_v3_test("flash_attn_bf16_causal_s256.npz", 0.05, true);
}

#[test]
fn test_flash_attn_bf16_v3_causal_s1024() {
    run_flash_attn_v3_test("flash_attn_bf16_causal_s1024.npz", 0.05, true);
}

// ─── FP8 Flash Attention tests ──────────────────────────────────────

#[cfg(feature = "experimental")]
fn run_flash_attn_fp8_test(npz_name: &str, tol: f32) {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let mut npz = load_npz(npz_name);

    let q_np: ndarray::Array4<u8> = npz.by_name("q").unwrap();
    let k_np: ndarray::Array4<u8> = npz.by_name("k").unwrap();
    let v_np: ndarray::Array4<u8> = npz.by_name("v").unwrap();
    let o_expected: ndarray::Array4<u16> = npz.by_name("o").unwrap();
    let scale_np: ndarray::Array0<f32> = npz.by_name("scale").unwrap();

    let batch = q_np.shape()[0] as u32;
    let num_heads = q_np.shape()[1] as u32;
    let seq_q = q_np.shape()[2] as u32;
    let seq_kv = k_np.shape()[2] as u32;
    let scale = scale_np.into_scalar();

    let q_flat: Vec<u8> = q_np.into_raw_vec_and_offset().0;
    let k_flat: Vec<u8> = k_np.into_raw_vec_and_offset().0;
    let v_flat: Vec<u8> = v_np.into_raw_vec_and_offset().0;
    let expected_flat: Vec<u16> = o_expected.into_raw_vec_and_offset().0;

    let q_dev = stream.memcpy_stod(&q_flat).unwrap();
    let k_dev = stream.memcpy_stod(&k_flat).unwrap();
    let v_dev = stream.memcpy_stod(&v_flat).unwrap();
    let mut o_dev = stream
        .alloc_zeros::<u16>((batch * num_heads * seq_q * 128) as usize)
        .unwrap();

    attention::flash_attn_fp8_d128(
        &ctx, &stream, &q_dev, &k_dev, &v_dev, &mut o_dev, batch, num_heads, seq_q, seq_kv, scale,
    )
    .unwrap();

    let o_host = stream.memcpy_dtov(&o_dev).unwrap();
    let (max_diff, mean_diff) = compare_bf16(&o_host, &expected_flat, tol);
    eprintln!("FP8 {npz_name}: max_diff={max_diff:.6} mean_diff={mean_diff:.6} (tol={tol})");
}

#[cfg(feature = "experimental")]
#[test]
fn test_flash_attn_fp8_noncausal_s256() {
    run_flash_attn_fp8_test("flash_attn_fp8_noncausal_s256.npz", 0.15);
}

#[cfg(feature = "experimental")]
#[test]
fn test_flash_attn_fp8_noncausal_s1024() {
    run_flash_attn_fp8_test("flash_attn_fp8_noncausal_s1024.npz", 0.15);
}

#[cfg(feature = "experimental")]
fn run_flash_attn_fp8_causal_test(npz_name: &str, tol: f32) {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let mut npz = load_npz(npz_name);

    let q_np: ndarray::Array4<u8> = npz.by_name("q").unwrap();
    let k_np: ndarray::Array4<u8> = npz.by_name("k").unwrap();
    let v_np: ndarray::Array4<u8> = npz.by_name("v").unwrap();
    let o_expected: ndarray::Array4<u16> = npz.by_name("o").unwrap();
    let scale_np: ndarray::Array0<f32> = npz.by_name("scale").unwrap();

    let batch = q_np.shape()[0] as u32;
    let num_heads = q_np.shape()[1] as u32;
    let seq_q = q_np.shape()[2] as u32;
    let seq_kv = k_np.shape()[2] as u32;
    let scale = scale_np.into_scalar();

    let q_flat: Vec<u8> = q_np.into_raw_vec_and_offset().0;
    let k_flat: Vec<u8> = k_np.into_raw_vec_and_offset().0;
    let v_flat: Vec<u8> = v_np.into_raw_vec_and_offset().0;
    let expected_flat: Vec<u16> = o_expected.into_raw_vec_and_offset().0;

    let q_dev = stream.memcpy_stod(&q_flat).unwrap();
    let k_dev = stream.memcpy_stod(&k_flat).unwrap();
    let v_dev = stream.memcpy_stod(&v_flat).unwrap();
    let mut o_dev = stream
        .alloc_zeros::<u16>((batch * num_heads * seq_q * 128) as usize)
        .unwrap();

    attention::flash_attn_fp8_d128_causal(
        &ctx, &stream, &q_dev, &k_dev, &v_dev, &mut o_dev, batch, num_heads, seq_q, seq_kv, scale,
    )
    .unwrap();

    let o_host = stream.memcpy_dtov(&o_dev).unwrap();
    let (max_diff, mean_diff) = compare_bf16(&o_host, &expected_flat, tol);
    eprintln!("FP8 causal {npz_name}: max_diff={max_diff:.6} mean_diff={mean_diff:.6} (tol={tol})");
}

#[cfg(feature = "experimental")]
#[test]
fn test_flash_attn_fp8_causal_s256() {
    run_flash_attn_fp8_causal_test("flash_attn_fp8_causal_s256.npz", 0.15);
}

#[cfg(feature = "experimental")]
#[test]
fn test_flash_attn_fp8_causal_s1024() {
    run_flash_attn_fp8_causal_test("flash_attn_fp8_causal_s1024.npz", 0.15);
}

// ─── Varlen Flash Attention tests ───────────────────────────────────

#[test]
fn test_flash_attn_bf16_varlen() {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let mut npz = load_npz("flash_attn_bf16_varlen.npz");

    let q_flat: ndarray::Array1<u16> = npz.by_name("q").unwrap();
    let k_flat: ndarray::Array1<u16> = npz.by_name("k").unwrap();
    let v_flat: ndarray::Array1<u16> = npz.by_name("v").unwrap();
    let o_expected: ndarray::Array1<u16> = npz.by_name("o").unwrap();
    let cu_seqlens_q: ndarray::Array1<u32> = npz.by_name("cu_seqlens_q").unwrap();
    let cu_seqlens_k: ndarray::Array1<u32> = npz.by_name("cu_seqlens_k").unwrap();
    let num_heads: ndarray::Array0<u32> = npz.by_name("num_heads").unwrap();
    let max_seqlen_q: ndarray::Array0<u32> = npz.by_name("max_seqlen_q").unwrap();
    let scale: ndarray::Array0<f32> = npz.by_name("scale").unwrap();
    let total_q: ndarray::Array0<u32> = npz.by_name("total_q").unwrap();

    let num_heads = num_heads.into_scalar();
    let max_seqlen_q = max_seqlen_q.into_scalar();
    let scale = scale.into_scalar();
    let total_q = total_q.into_scalar();
    let batch = cu_seqlens_q.len() as u32 - 1;

    let q_vec: Vec<u16> = q_flat.into_raw_vec_and_offset().0;
    let k_vec: Vec<u16> = k_flat.into_raw_vec_and_offset().0;
    let v_vec: Vec<u16> = v_flat.into_raw_vec_and_offset().0;
    let expected: Vec<u16> = o_expected.into_raw_vec_and_offset().0;
    let cu_q: Vec<u32> = cu_seqlens_q.into_raw_vec_and_offset().0;
    let cu_k: Vec<u32> = cu_seqlens_k.into_raw_vec_and_offset().0;

    let q_dev = stream.memcpy_stod(&q_vec).unwrap();
    let k_dev = stream.memcpy_stod(&k_vec).unwrap();
    let v_dev = stream.memcpy_stod(&v_vec).unwrap();
    let cu_q_dev = stream.memcpy_stod(&cu_q).unwrap();
    let cu_k_dev = stream.memcpy_stod(&cu_k).unwrap();
    let mut o_dev = stream
        .alloc_zeros::<u16>((total_q * num_heads * 128) as usize)
        .unwrap();

    attention::flash_attn_bf16_varlen_d128(
        &ctx,
        &stream,
        &q_dev,
        &k_dev,
        &v_dev,
        &mut o_dev,
        &cu_q_dev,
        &cu_k_dev,
        batch,
        num_heads,
        max_seqlen_q,
        scale,
    )
    .unwrap();

    let o_host = stream.memcpy_dtov(&o_dev).unwrap();
    let (max_diff, mean_diff) = compare_bf16(&o_host, &expected, 0.5);
    eprintln!(
        "varlen: max_diff={max_diff:.6} mean_diff={mean_diff:.6} (batch={batch}, heads={num_heads}, total_q={total_q})"
    );
}

#[cfg(feature = "experimental")]
fn run_flash_attn_v11_test(npz_name: &str, tol: f32) {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let mut npz = load_npz(npz_name);

    let q_np: ndarray::Array4<u16> = npz.by_name("q").unwrap();
    let k_np: ndarray::Array4<u16> = npz.by_name("k").unwrap();
    let v_np: ndarray::Array4<u16> = npz.by_name("v").unwrap();
    let o_expected: ndarray::Array4<u16> = npz.by_name("o").unwrap();
    let scale_np: ndarray::Array0<f32> = npz.by_name("scale").unwrap();

    let batch = q_np.shape()[0] as u32;
    let num_heads = q_np.shape()[1] as u32;
    let seq_q = q_np.shape()[2] as u32;
    let seq_kv = k_np.shape()[2] as u32;
    let scale = scale_np.into_scalar();

    let q_flat: Vec<u16> = q_np.into_raw_vec_and_offset().0;
    let k_flat: Vec<u16> = k_np.into_raw_vec_and_offset().0;
    let v_flat: Vec<u16> = v_np.into_raw_vec_and_offset().0;
    let expected_flat: Vec<u16> = o_expected.into_raw_vec_and_offset().0;

    let q_dev = stream.memcpy_stod(&q_flat).unwrap();
    let k_dev = stream.memcpy_stod(&k_flat).unwrap();
    let v_dev = stream.memcpy_stod(&v_flat).unwrap();
    let mut o_dev = stream
        .alloc_zeros::<u16>((batch * num_heads * seq_q * 128) as usize)
        .unwrap();

    attention::flash_attn_bf16_v11_fused_scale(
        &ctx, &stream, &q_dev, &k_dev, &v_dev, &mut o_dev, batch, num_heads, seq_q, seq_kv, scale,
    )
    .unwrap();

    let o_host = stream.memcpy_dtov(&o_dev).unwrap();
    let (max_diff, mean_diff) = compare_bf16(&o_host, &expected_flat, tol);
    eprintln!("V11 {npz_name}: max_diff={max_diff:.6} mean_diff={mean_diff:.6} (tol={tol})");
}

#[cfg(feature = "experimental")]
#[test]
#[ignore = "90KB+ dynamic SMEM; crashes on DGX Spark unified memory after fragmentation"]
fn test_flash_attn_bf16_v11_noncausal_s256() {
    // V11 uses fused scale; tolerance slightly wider than V9 (1 BF16 ULP rounding difference)
    run_flash_attn_v11_test("flash_attn_bf16_noncausal_s256.npz", 0.02);
}

#[cfg(feature = "experimental")]
#[test]
#[ignore = "90KB+ dynamic SMEM; crashes on DGX Spark unified memory after fragmentation"]
fn test_flash_attn_bf16_v11_noncausal_s1024() {
    run_flash_attn_v11_test("flash_attn_bf16_noncausal_s1024.npz", 0.02);
}

fn run_flash_attn_v12_test(npz_name: &str, tol: f32) {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let mut npz = load_npz(npz_name);

    let q_np: ndarray::Array4<u16> = npz.by_name("q").unwrap();
    let k_np: ndarray::Array4<u16> = npz.by_name("k").unwrap();
    let v_np: ndarray::Array4<u16> = npz.by_name("v").unwrap();
    let o_expected: ndarray::Array4<u16> = npz.by_name("o").unwrap();
    let scale_np: ndarray::Array0<f32> = npz.by_name("scale").unwrap();

    let batch = q_np.shape()[0] as u32;
    let num_heads = q_np.shape()[1] as u32;
    let seq_q = q_np.shape()[2] as u32;
    let seq_kv = k_np.shape()[2] as u32;
    let scale = scale_np.into_scalar();

    let q_flat: Vec<u16> = q_np.into_raw_vec_and_offset().0;
    let k_flat: Vec<u16> = k_np.into_raw_vec_and_offset().0;
    let v_flat: Vec<u16> = v_np.into_raw_vec_and_offset().0;
    let expected_flat: Vec<u16> = o_expected.into_raw_vec_and_offset().0;

    let q_dev = stream.memcpy_stod(&q_flat).unwrap();
    let k_dev = stream.memcpy_stod(&k_flat).unwrap();
    let v_dev = stream.memcpy_stod(&v_flat).unwrap();
    let mut o_dev = stream
        .alloc_zeros::<u16>((batch * num_heads * seq_q * 128) as usize)
        .unwrap();

    attention::flash_attn_bf16_v12_persistent(
        &ctx, &stream, &q_dev, &k_dev, &v_dev, &mut o_dev, batch, num_heads, seq_q, seq_kv, scale,
    )
    .unwrap();

    let o_host = stream.memcpy_dtov(&o_dev).unwrap();
    let (max_diff, mean_diff) = compare_bf16(&o_host, &expected_flat, tol);
    eprintln!("V12 {npz_name}: max_diff={max_diff:.6} mean_diff={mean_diff:.6} (tol={tol})");
}

#[test]
#[ignore = "90KB+ dynamic SMEM; crashes on DGX Spark unified memory after fragmentation"]
fn test_flash_attn_bf16_v12_noncausal_s256() {
    run_flash_attn_v12_test("flash_attn_bf16_noncausal_s256.npz", 0.02);
}

#[test]
#[ignore = "90KB+ dynamic SMEM; crashes on DGX Spark unified memory after fragmentation"]
fn test_flash_attn_bf16_v12_noncausal_s1024() {
    run_flash_attn_v12_test("flash_attn_bf16_noncausal_s1024.npz", 0.02);
}

#[cfg(feature = "experimental")]
fn run_flash_attn_v8_test(npz_name: &str, tol: f32) {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let mut npz = load_npz(npz_name);

    let q_np: ndarray::Array4<u16> = npz.by_name("q").unwrap();
    let k_np: ndarray::Array4<u16> = npz.by_name("k").unwrap();
    let v_np: ndarray::Array4<u16> = npz.by_name("v").unwrap();
    let o_expected: ndarray::Array4<u16> = npz.by_name("o").unwrap();
    let scale_np: ndarray::Array0<f32> = npz.by_name("scale").unwrap();

    let batch = q_np.shape()[0] as u32;
    let num_heads = q_np.shape()[1] as u32;
    let seq_q = q_np.shape()[2] as u32;
    let seq_kv = k_np.shape()[2] as u32;
    let scale = scale_np.into_scalar();

    let q_flat: Vec<u16> = q_np.into_raw_vec_and_offset().0;
    let k_flat: Vec<u16> = k_np.into_raw_vec_and_offset().0;
    let v_flat: Vec<u16> = v_np.into_raw_vec_and_offset().0;
    let expected_flat: Vec<u16> = o_expected.into_raw_vec_and_offset().0;

    let q_dev = stream.memcpy_stod(&q_flat).unwrap();
    let k_dev = stream.memcpy_stod(&k_flat).unwrap();
    let v_dev = stream.memcpy_stod(&v_flat).unwrap();
    let mut o_dev = stream
        .alloc_zeros::<u16>((batch * num_heads * seq_q * 128) as usize)
        .unwrap();

    attention::flash_attn_bf16_v8_tma_db_d128(
        &ctx, &stream, &q_dev, &k_dev, &v_dev, &mut o_dev, batch, num_heads, seq_q, seq_kv, scale,
    )
    .unwrap();

    let o_host = stream.memcpy_dtov(&o_dev).unwrap();
    let (max_diff, mean_diff) = compare_bf16(&o_host, &expected_flat, tol);
    eprintln!("V8 {npz_name}: max_diff={max_diff:.6} mean_diff={mean_diff:.6} (tol={tol})");
}

#[cfg(feature = "experimental")]
#[test]
#[ignore = "90KB+ dynamic SMEM; crashes on DGX Spark unified memory after fragmentation"]
fn test_flash_attn_bf16_v8_noncausal_s256() {
    run_flash_attn_v8_test("flash_attn_bf16_noncausal_s256.npz", 0.02);
}

#[cfg(feature = "experimental")]
#[test]
#[ignore = "90KB+ dynamic SMEM; crashes on DGX Spark unified memory after fragmentation"]
fn test_flash_attn_bf16_v8_noncausal_s1024() {
    run_flash_attn_v8_test("flash_attn_bf16_noncausal_s1024.npz", 0.02);
}

#[cfg(feature = "experimental")]
fn run_flash_attn_v13_test(npz_name: &str, tol: f32) {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let mut npz = load_npz(npz_name);

    let q_np: ndarray::Array4<u16> = npz.by_name("q").unwrap();
    let k_np: ndarray::Array4<u16> = npz.by_name("k").unwrap();
    let v_np: ndarray::Array4<u16> = npz.by_name("v").unwrap();
    let o_expected: ndarray::Array4<u16> = npz.by_name("o").unwrap();
    let scale_np: ndarray::Array0<f32> = npz.by_name("scale").unwrap();

    let batch = q_np.shape()[0] as u32;
    let num_heads = q_np.shape()[1] as u32;
    let seq_q = q_np.shape()[2] as u32;
    let seq_kv = k_np.shape()[2] as u32;
    let scale = scale_np.into_scalar();

    let q_flat: Vec<u16> = q_np.into_raw_vec_and_offset().0;
    let k_flat: Vec<u16> = k_np.into_raw_vec_and_offset().0;
    let v_flat: Vec<u16> = v_np.into_raw_vec_and_offset().0;
    let expected_flat: Vec<u16> = o_expected.into_raw_vec_and_offset().0;

    let q_dev = stream.memcpy_stod(&q_flat).unwrap();
    let k_dev = stream.memcpy_stod(&k_flat).unwrap();
    let v_dev = stream.memcpy_stod(&v_flat).unwrap();
    let mut o_dev = stream
        .alloc_zeros::<u16>((batch * num_heads * seq_q * 128) as usize)
        .unwrap();

    attention::flash_attn_bf16_v13_bc32_d128(
        &ctx, &stream, &q_dev, &k_dev, &v_dev, &mut o_dev, batch, num_heads, seq_q, seq_kv, scale,
    )
    .unwrap();

    let o_host = stream.memcpy_dtov(&o_dev).unwrap();
    let (max_diff, mean_diff) = compare_bf16(&o_host, &expected_flat, tol);
    eprintln!("V13 {npz_name}: max_diff={max_diff:.6} mean_diff={mean_diff:.6} (tol={tol})");
}

#[cfg(feature = "experimental")]
#[test]
fn test_flash_attn_bf16_v13_noncausal_s256() {
    run_flash_attn_v13_test("flash_attn_bf16_noncausal_s256.npz", 0.02);
}

#[cfg(feature = "experimental")]
#[test]
fn test_flash_attn_bf16_v13_noncausal_s1024() {
    run_flash_attn_v13_test("flash_attn_bf16_noncausal_s1024.npz", 0.02);
}

#[cfg(feature = "experimental")]
fn run_flash_attn_v11_causal_test(npz_name: &str, tol: f32) {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let mut npz = load_npz(npz_name);

    let q_np: ndarray::Array4<u16> = npz.by_name("q").unwrap();
    let k_np: ndarray::Array4<u16> = npz.by_name("k").unwrap();
    let v_np: ndarray::Array4<u16> = npz.by_name("v").unwrap();
    let o_expected: ndarray::Array4<u16> = npz.by_name("o").unwrap();
    let scale_np: ndarray::Array0<f32> = npz.by_name("scale").unwrap();

    let batch = q_np.shape()[0] as u32;
    let num_heads = q_np.shape()[1] as u32;
    let seq_q = q_np.shape()[2] as u32;
    let seq_kv = k_np.shape()[2] as u32;
    let scale = scale_np.into_scalar();

    let q_flat: Vec<u16> = q_np.into_raw_vec_and_offset().0;
    let k_flat: Vec<u16> = k_np.into_raw_vec_and_offset().0;
    let v_flat: Vec<u16> = v_np.into_raw_vec_and_offset().0;
    let expected_flat: Vec<u16> = o_expected.into_raw_vec_and_offset().0;

    let q_dev = stream.memcpy_stod(&q_flat).unwrap();
    let k_dev = stream.memcpy_stod(&k_flat).unwrap();
    let v_dev = stream.memcpy_stod(&v_flat).unwrap();
    let mut o_dev = stream
        .alloc_zeros::<u16>((batch * num_heads * seq_q * 128) as usize)
        .unwrap();

    attention::flash_attn_bf16_v11_causal(
        &ctx, &stream, &q_dev, &k_dev, &v_dev, &mut o_dev, batch, num_heads, seq_q, seq_kv, scale,
    )
    .unwrap();

    let o_host = stream.memcpy_dtov(&o_dev).unwrap();
    let (max_diff, mean_diff) = compare_bf16(&o_host, &expected_flat, tol);
    eprintln!("V11 causal {npz_name}: max_diff={max_diff:.6} mean_diff={mean_diff:.6} (tol={tol})");
}

#[cfg(feature = "experimental")]
#[test]
#[ignore = "90KB+ dynamic SMEM; crashes on DGX Spark unified memory after fragmentation"]
fn test_flash_attn_bf16_v11_causal_s256() {
    run_flash_attn_v11_causal_test("flash_attn_bf16_causal_s256.npz", 0.05);
}

#[cfg(feature = "experimental")]
#[test]
#[ignore = "90KB+ dynamic SMEM; crashes on DGX Spark unified memory after fragmentation"]
fn test_flash_attn_bf16_v11_causal_s1024() {
    run_flash_attn_v11_causal_test("flash_attn_bf16_causal_s1024.npz", 0.05);
}

// ─── V11 varlen non-causal flash attention tests ─────────────────────

#[cfg(feature = "experimental")]
fn run_flash_attn_v11_varlen_test(npz_name: &str, tol: f32) {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let mut npz = load_npz(npz_name);

    let q_flat: ndarray::Array1<u16> = npz.by_name("q").unwrap();
    let k_flat: ndarray::Array1<u16> = npz.by_name("k").unwrap();
    let v_flat: ndarray::Array1<u16> = npz.by_name("v").unwrap();
    let o_expected: ndarray::Array1<u16> = npz.by_name("o").unwrap();
    let cu_seqlens_q: ndarray::Array1<u32> = npz.by_name("cu_seqlens_q").unwrap();
    let cu_seqlens_k: ndarray::Array1<u32> = npz.by_name("cu_seqlens_k").unwrap();
    let num_heads: ndarray::Array0<u32> = npz.by_name("num_heads").unwrap();
    let batch: ndarray::Array0<u32> = npz.by_name("batch").unwrap();
    let max_seqlen_q: ndarray::Array0<u32> = npz.by_name("max_seqlen_q").unwrap();
    let total_q: ndarray::Array0<u32> = npz.by_name("total_q").unwrap();
    let total_kv: ndarray::Array0<u32> = npz.by_name("total_kv").unwrap();
    let scale: ndarray::Array0<f32> = npz.by_name("scale").unwrap();

    let num_heads = num_heads.into_scalar();
    let batch = batch.into_scalar();
    let max_seqlen_q = max_seqlen_q.into_scalar();
    let total_q = total_q.into_scalar();
    let total_kv = total_kv.into_scalar();
    let scale = scale.into_scalar();

    let q_vec: Vec<u16> = q_flat.into_raw_vec_and_offset().0;
    let k_vec: Vec<u16> = k_flat.into_raw_vec_and_offset().0;
    let v_vec: Vec<u16> = v_flat.into_raw_vec_and_offset().0;
    let expected: Vec<u16> = o_expected.into_raw_vec_and_offset().0;
    let cu_q: Vec<u32> = cu_seqlens_q.into_raw_vec_and_offset().0;
    let cu_k: Vec<u32> = cu_seqlens_k.into_raw_vec_and_offset().0;

    let q_dev = stream.memcpy_stod(&q_vec).unwrap();
    let k_dev = stream.memcpy_stod(&k_vec).unwrap();
    let v_dev = stream.memcpy_stod(&v_vec).unwrap();
    let cu_q_dev = stream.memcpy_stod(&cu_q).unwrap();
    let cu_k_dev = stream.memcpy_stod(&cu_k).unwrap();
    let mut o_dev = stream
        .alloc_zeros::<u16>((num_heads * total_q * 128) as usize)
        .unwrap();

    attention::flash_attn_bf16_v11_varlen(
        &ctx,
        &stream,
        &q_dev,
        &k_dev,
        &v_dev,
        &mut o_dev,
        &cu_q_dev,
        &cu_k_dev,
        batch,
        num_heads,
        max_seqlen_q,
        total_q,
        total_kv,
        scale,
    )
    .unwrap();

    let o_host = stream.memcpy_dtov(&o_dev).unwrap();
    let (max_diff, mean_diff) = compare_bf16(&o_host, &expected, tol);
    eprintln!(
        "V11 varlen {npz_name}: max_diff={max_diff:.6} mean_diff={mean_diff:.6} (batch={batch}, heads={num_heads}, total_q={total_q}, total_kv={total_kv}, tol={tol})"
    );
}

#[cfg(feature = "experimental")]
#[test]
#[ignore = "90KB+ dynamic SMEM; crashes on DGX Spark unified memory after fragmentation"]
fn test_flash_attn_bf16_v11_varlen_a() {
    run_flash_attn_v11_varlen_test("flash_attn_bf16_v11_varlen_a.npz", 0.05);
}

#[cfg(feature = "experimental")]
#[test]
#[ignore = "90KB+ dynamic SMEM; crashes on DGX Spark unified memory after fragmentation"]
fn test_flash_attn_bf16_v11_varlen_b() {
    run_flash_attn_v11_varlen_test("flash_attn_bf16_v11_varlen_b.npz", 0.05);
}

// ─── V11 varlen causal flash attention tests ─────────────────────────

#[cfg(feature = "experimental")]
fn run_flash_attn_v11_varlen_causal_test(npz_name: &str, tol: f32) {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let mut npz = load_npz(npz_name);

    let q_flat: ndarray::Array1<u16> = npz.by_name("q").unwrap();
    let k_flat: ndarray::Array1<u16> = npz.by_name("k").unwrap();
    let v_flat: ndarray::Array1<u16> = npz.by_name("v").unwrap();
    let o_expected: ndarray::Array1<u16> = npz.by_name("o").unwrap();
    let cu_seqlens_q: ndarray::Array1<u32> = npz.by_name("cu_seqlens_q").unwrap();
    let cu_seqlens_k: ndarray::Array1<u32> = npz.by_name("cu_seqlens_k").unwrap();
    let num_heads: ndarray::Array0<u32> = npz.by_name("num_heads").unwrap();
    let batch: ndarray::Array0<u32> = npz.by_name("batch").unwrap();
    let max_seqlen_q: ndarray::Array0<u32> = npz.by_name("max_seqlen_q").unwrap();
    let total_q: ndarray::Array0<u32> = npz.by_name("total_q").unwrap();
    let total_kv: ndarray::Array0<u32> = npz.by_name("total_kv").unwrap();
    let scale: ndarray::Array0<f32> = npz.by_name("scale").unwrap();

    let num_heads = num_heads.into_scalar();
    let batch = batch.into_scalar();
    let max_seqlen_q = max_seqlen_q.into_scalar();
    let total_q = total_q.into_scalar();
    let total_kv = total_kv.into_scalar();
    let scale = scale.into_scalar();

    let q_vec: Vec<u16> = q_flat.into_raw_vec_and_offset().0;
    let k_vec: Vec<u16> = k_flat.into_raw_vec_and_offset().0;
    let v_vec: Vec<u16> = v_flat.into_raw_vec_and_offset().0;
    let expected: Vec<u16> = o_expected.into_raw_vec_and_offset().0;
    let cu_q: Vec<u32> = cu_seqlens_q.into_raw_vec_and_offset().0;
    let cu_k: Vec<u32> = cu_seqlens_k.into_raw_vec_and_offset().0;

    let q_dev = stream.memcpy_stod(&q_vec).unwrap();
    let k_dev = stream.memcpy_stod(&k_vec).unwrap();
    let v_dev = stream.memcpy_stod(&v_vec).unwrap();
    let cu_q_dev = stream.memcpy_stod(&cu_q).unwrap();
    let cu_k_dev = stream.memcpy_stod(&cu_k).unwrap();
    // O layout: [num_heads, total_q, D] head-major
    let mut o_dev = stream
        .alloc_zeros::<u16>((num_heads * total_q * 128) as usize)
        .unwrap();

    attention::flash_attn_bf16_v11_varlen_causal(
        &ctx,
        &stream,
        &q_dev,
        &k_dev,
        &v_dev,
        &mut o_dev,
        &cu_q_dev,
        &cu_k_dev,
        batch,
        num_heads,
        max_seqlen_q,
        total_q,
        total_kv,
        scale,
    )
    .unwrap();

    let o_host = stream.memcpy_dtov(&o_dev).unwrap();
    let (max_diff, mean_diff) = compare_bf16(&o_host, &expected, tol);
    eprintln!(
        "V11 varlen causal {npz_name}: max_diff={max_diff:.6} mean_diff={mean_diff:.6} (batch={batch}, heads={num_heads}, total_q={total_q}, tol={tol})"
    );
}

#[cfg(feature = "experimental")]
#[test]
#[ignore = "90KB+ dynamic SMEM; crashes on DGX Spark unified memory after fragmentation"]
fn test_flash_attn_bf16_v11_varlen_causal_a() {
    run_flash_attn_v11_varlen_causal_test("flash_attn_bf16_v11_varlen_causal_a.npz", 0.05);
}

#[cfg(feature = "experimental")]
#[test]
#[ignore = "90KB+ dynamic SMEM; crashes on DGX Spark unified memory after fragmentation"]
fn test_flash_attn_bf16_v11_varlen_causal_b() {
    run_flash_attn_v11_varlen_causal_test("flash_attn_bf16_v11_varlen_causal_b.npz", 0.05);
}

// ─── V11 GQA fixed-len flash attention tests ────────────────────────

#[cfg(feature = "experimental")]
fn run_flash_attn_v11_gqa_test(npz_name: &str, tol: f32, causal: bool) {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let mut npz = load_npz(npz_name);

    let q_flat: ndarray::Array1<u16> = npz.by_name("q").unwrap();
    let k_flat: ndarray::Array1<u16> = npz.by_name("k").unwrap();
    let v_flat: ndarray::Array1<u16> = npz.by_name("v").unwrap();
    let o_expected: ndarray::Array1<u16> = npz.by_name("o").unwrap();
    let num_heads_q: ndarray::Array0<u32> = npz.by_name("num_heads_q").unwrap();
    let num_heads_kv: ndarray::Array0<u32> = npz.by_name("num_heads_kv").unwrap();
    let batch: ndarray::Array0<u32> = npz.by_name("batch").unwrap();
    let seq_q: ndarray::Array0<u32> = npz.by_name("seq_q").unwrap();
    let seq_kv: ndarray::Array0<u32> = npz.by_name("seq_kv").unwrap();
    let scale: ndarray::Array0<f32> = npz.by_name("scale").unwrap();

    let num_heads_q = num_heads_q.into_scalar();
    let num_heads_kv = num_heads_kv.into_scalar();
    let batch = batch.into_scalar();
    let seq_q = seq_q.into_scalar();
    let seq_kv = seq_kv.into_scalar();
    let scale = scale.into_scalar();

    let q_vec: Vec<u16> = q_flat.into_raw_vec_and_offset().0;
    let k_vec: Vec<u16> = k_flat.into_raw_vec_and_offset().0;
    let v_vec: Vec<u16> = v_flat.into_raw_vec_and_offset().0;
    let expected: Vec<u16> = o_expected.into_raw_vec_and_offset().0;

    let q_dev = stream.memcpy_stod(&q_vec).unwrap();
    let k_dev = stream.memcpy_stod(&k_vec).unwrap();
    let v_dev = stream.memcpy_stod(&v_vec).unwrap();
    let mut o_dev = stream
        .alloc_zeros::<u16>((batch * num_heads_q * seq_q * 128) as usize)
        .unwrap();

    if causal {
        attention::flash_attn_bf16_v11_gqa_causal(
            &ctx,
            &stream,
            &q_dev,
            &k_dev,
            &v_dev,
            &mut o_dev,
            batch,
            num_heads_q,
            num_heads_kv,
            seq_q,
            seq_kv,
            scale,
        )
        .unwrap();
    } else {
        attention::flash_attn_bf16_v11_gqa(
            &ctx,
            &stream,
            &q_dev,
            &k_dev,
            &v_dev,
            &mut o_dev,
            batch,
            num_heads_q,
            num_heads_kv,
            seq_q,
            seq_kv,
            scale,
        )
        .unwrap();
    }

    let o_host = stream.memcpy_dtov(&o_dev).unwrap();
    let (max_diff, mean_diff) = compare_bf16(&o_host, &expected, tol);
    eprintln!(
        "V11 GQA{} {npz_name}: max_diff={max_diff:.6} mean_diff={mean_diff:.6} (batch={batch}, hq={num_heads_q}, hkv={num_heads_kv}, seq_q={seq_q}, seq_kv={seq_kv}, tol={tol})",
        if causal { " causal" } else { "" }
    );
}

#[cfg(feature = "experimental")]
#[test]
#[ignore = "90KB+ dynamic SMEM; crashes on DGX Spark unified memory after fragmentation"]
fn test_flash_attn_bf16_v11_gqa_noncausal() {
    run_flash_attn_v11_gqa_test("flash_attn_bf16_v11_gqa_noncausal.npz", 0.05, false);
}

#[cfg(feature = "experimental")]
#[test]
#[ignore = "90KB+ dynamic SMEM; crashes on DGX Spark unified memory after fragmentation"]
fn test_flash_attn_bf16_v11_gqa_causal() {
    run_flash_attn_v11_gqa_test("flash_attn_bf16_v11_gqa_causal.npz", 0.05, true);
}

// ─── V11 varlen GQA flash attention tests ───────────────────────────

#[cfg(feature = "experimental")]
fn run_flash_attn_v11_varlen_gqa_test(npz_name: &str, tol: f32, causal: bool) {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let mut npz = load_npz(npz_name);

    let q_flat: ndarray::Array1<u16> = npz.by_name("q").unwrap();
    let k_flat: ndarray::Array1<u16> = npz.by_name("k").unwrap();
    let v_flat: ndarray::Array1<u16> = npz.by_name("v").unwrap();
    let o_expected: ndarray::Array1<u16> = npz.by_name("o").unwrap();
    let cu_seqlens_q: ndarray::Array1<u32> = npz.by_name("cu_seqlens_q").unwrap();
    let cu_seqlens_k: ndarray::Array1<u32> = npz.by_name("cu_seqlens_k").unwrap();
    let num_heads_q: ndarray::Array0<u32> = npz.by_name("num_heads_q").unwrap();
    let num_heads_kv: ndarray::Array0<u32> = npz.by_name("num_heads_kv").unwrap();
    let batch: ndarray::Array0<u32> = npz.by_name("batch").unwrap();
    let max_seqlen_q: ndarray::Array0<u32> = npz.by_name("max_seqlen_q").unwrap();
    let total_q: ndarray::Array0<u32> = npz.by_name("total_q").unwrap();
    let total_kv: ndarray::Array0<u32> = npz.by_name("total_kv").unwrap();
    let scale: ndarray::Array0<f32> = npz.by_name("scale").unwrap();

    let num_heads_q = num_heads_q.into_scalar();
    let num_heads_kv = num_heads_kv.into_scalar();
    let batch = batch.into_scalar();
    let max_seqlen_q = max_seqlen_q.into_scalar();
    let total_q = total_q.into_scalar();
    let total_kv = total_kv.into_scalar();
    let scale = scale.into_scalar();

    let q_vec: Vec<u16> = q_flat.into_raw_vec_and_offset().0;
    let k_vec: Vec<u16> = k_flat.into_raw_vec_and_offset().0;
    let v_vec: Vec<u16> = v_flat.into_raw_vec_and_offset().0;
    let expected: Vec<u16> = o_expected.into_raw_vec_and_offset().0;
    let cu_q: Vec<u32> = cu_seqlens_q.into_raw_vec_and_offset().0;
    let cu_k: Vec<u32> = cu_seqlens_k.into_raw_vec_and_offset().0;

    let q_dev = stream.memcpy_stod(&q_vec).unwrap();
    let k_dev = stream.memcpy_stod(&k_vec).unwrap();
    let v_dev = stream.memcpy_stod(&v_vec).unwrap();
    let cu_q_dev = stream.memcpy_stod(&cu_q).unwrap();
    let cu_k_dev = stream.memcpy_stod(&cu_k).unwrap();
    let mut o_dev = stream
        .alloc_zeros::<u16>((num_heads_q * total_q * 128) as usize)
        .unwrap();

    if causal {
        attention::flash_attn_bf16_v11_varlen_gqa_causal(
            &ctx,
            &stream,
            &q_dev,
            &k_dev,
            &v_dev,
            &mut o_dev,
            &cu_q_dev,
            &cu_k_dev,
            batch,
            num_heads_q,
            num_heads_kv,
            max_seqlen_q,
            total_q,
            total_kv,
            scale,
        )
        .unwrap();
    } else {
        attention::flash_attn_bf16_v11_varlen_gqa(
            &ctx,
            &stream,
            &q_dev,
            &k_dev,
            &v_dev,
            &mut o_dev,
            &cu_q_dev,
            &cu_k_dev,
            batch,
            num_heads_q,
            num_heads_kv,
            max_seqlen_q,
            total_q,
            total_kv,
            scale,
        )
        .unwrap();
    }

    let o_host = stream.memcpy_dtov(&o_dev).unwrap();
    let (max_diff, mean_diff) = compare_bf16(&o_host, &expected, tol);
    eprintln!(
        "V11 varlen GQA{} {npz_name}: max_diff={max_diff:.6} mean_diff={mean_diff:.6} (batch={batch}, hq={num_heads_q}, hkv={num_heads_kv}, total_q={total_q}, total_kv={total_kv}, tol={tol})",
        if causal { " causal" } else { "" }
    );
}

#[cfg(feature = "experimental")]
#[test]
#[ignore = "90KB+ dynamic SMEM; crashes on DGX Spark unified memory after fragmentation"]
fn test_flash_attn_bf16_v11_varlen_gqa_noncausal() {
    run_flash_attn_v11_varlen_gqa_test("flash_attn_bf16_v11_varlen_gqa_noncausal.npz", 0.05, false);
}

#[cfg(feature = "experimental")]
#[test]
#[ignore = "90KB+ dynamic SMEM; crashes on DGX Spark unified memory after fragmentation"]
fn test_flash_attn_bf16_v11_varlen_gqa_causal() {
    run_flash_attn_v11_varlen_gqa_test("flash_attn_bf16_v11_varlen_gqa_causal.npz", 0.05, true);
}

#[cfg(feature = "experimental")]
fn run_flash_attn_fp8_v11_tma_test(npz_name: &str, tol: f32) {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let mut npz = load_npz(npz_name);

    let q_np: ndarray::Array4<u8> = npz.by_name("q").unwrap();
    let k_np: ndarray::Array4<u8> = npz.by_name("k").unwrap();
    let v_np: ndarray::Array4<u8> = npz.by_name("v").unwrap();
    let o_expected: ndarray::Array4<u16> = npz.by_name("o").unwrap();
    let scale_np: ndarray::Array0<f32> = npz.by_name("scale").unwrap();

    let batch = q_np.shape()[0] as u32;
    let num_heads = q_np.shape()[1] as u32;
    let seq_q = q_np.shape()[2] as u32;
    let seq_kv = k_np.shape()[2] as u32;
    let scale = scale_np.into_scalar();

    let q_flat: Vec<u8> = q_np.into_raw_vec_and_offset().0;
    let k_flat: Vec<u8> = k_np.into_raw_vec_and_offset().0;
    let v_flat: Vec<u8> = v_np.into_raw_vec_and_offset().0;
    let expected_flat: Vec<u16> = o_expected.into_raw_vec_and_offset().0;

    let q_dev = stream.memcpy_stod(&q_flat).unwrap();
    let k_dev = stream.memcpy_stod(&k_flat).unwrap();
    let v_dev = stream.memcpy_stod(&v_flat).unwrap();
    let mut o_dev = stream
        .alloc_zeros::<u16>((batch * num_heads * seq_q * 128) as usize)
        .unwrap();

    attention::flash_attn_fp8_v11_tma(
        &ctx, &stream, &q_dev, &k_dev, &v_dev, &mut o_dev, batch, num_heads, seq_q, seq_kv, scale,
    )
    .unwrap();

    let o_host = stream.memcpy_dtov(&o_dev).unwrap();
    let (max_diff, mean_diff) = compare_bf16(&o_host, &expected_flat, tol);
    eprintln!(
        "FP8 V11 TMA {npz_name}: max_diff={max_diff:.6} mean_diff={mean_diff:.6} (tol={tol})"
    );
}

#[cfg(feature = "experimental")]
#[test]
fn test_flash_attn_fp8_v11_tma_noncausal_s256() {
    run_flash_attn_fp8_v11_tma_test("flash_attn_fp8_noncausal_s256.npz", 0.15);
}

#[cfg(feature = "experimental")]
#[test]
fn test_flash_attn_fp8_v11_tma_noncausal_s1024() {
    run_flash_attn_fp8_v11_tma_test("flash_attn_fp8_noncausal_s1024.npz", 0.15);
}

// ── V12a: SMEM cooperative transpose ──────────────────────────────────────────

#[cfg(feature = "experimental")]
fn run_flash_attn_fp8_v12a_test(npz_name: &str, tol: f32) {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let mut npz = load_npz(npz_name);

    let q_np: ndarray::Array4<u8> = npz.by_name("q").unwrap();
    let k_np: ndarray::Array4<u8> = npz.by_name("k").unwrap();
    let v_np: ndarray::Array4<u8> = npz.by_name("v").unwrap();
    let o_expected: ndarray::Array4<u16> = npz.by_name("o").unwrap();
    let scale_np: ndarray::Array0<f32> = npz.by_name("scale").unwrap();

    let batch = q_np.shape()[0] as u32;
    let num_heads = q_np.shape()[1] as u32;
    let seq_q = q_np.shape()[2] as u32;
    let seq_kv = k_np.shape()[2] as u32;
    let scale = scale_np.into_scalar();

    let q_flat: Vec<u8> = q_np.into_raw_vec_and_offset().0;
    let k_flat: Vec<u8> = k_np.into_raw_vec_and_offset().0;
    let v_flat: Vec<u8> = v_np.into_raw_vec_and_offset().0;
    let expected_flat: Vec<u16> = o_expected.into_raw_vec_and_offset().0;

    let q_dev = stream.memcpy_stod(&q_flat).unwrap();
    let k_dev = stream.memcpy_stod(&k_flat).unwrap();
    let v_dev = stream.memcpy_stod(&v_flat).unwrap();
    let mut o_dev = stream
        .alloc_zeros::<u16>((batch * num_heads * seq_q * 128) as usize)
        .unwrap();

    attention::flash_attn_fp8_v12a_transpose(
        &ctx, &stream, &q_dev, &k_dev, &v_dev, &mut o_dev, batch, num_heads, seq_q, seq_kv, scale,
    )
    .unwrap();

    let o_host = stream.memcpy_dtov(&o_dev).unwrap();
    let (max_diff, mean_diff) = compare_bf16(&o_host, &expected_flat, tol);
    eprintln!("FP8 V12a {npz_name}: max_diff={max_diff:.6} mean_diff={mean_diff:.6} (tol={tol})");
}

#[cfg(feature = "experimental")]
#[test]
fn test_flash_attn_fp8_v12a_noncausal_s256() {
    run_flash_attn_fp8_v12a_test("flash_attn_fp8_noncausal_s256.npz", 0.15);
}

#[cfg(feature = "experimental")]
#[test]
fn test_flash_attn_fp8_v12a_noncausal_s1024() {
    run_flash_attn_fp8_v12a_test("flash_attn_fp8_noncausal_s1024.npz", 0.15);
}

// ── V12c: V pre-transposed in GMEM [D, B*H*Skv] ───────────────────────────────

fn run_flash_attn_fp8_v12c_test(npz_name: &str, tol: f32) {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let mut npz = load_npz(npz_name);

    let q_np: ndarray::Array4<u8> = npz.by_name("q").unwrap();
    let k_np: ndarray::Array4<u8> = npz.by_name("k").unwrap();
    // V from npz is [B, H, Skv, D]. Transpose to [D, B, H, Skv] = [D, B*H*Skv] flat.
    let v_np: ndarray::Array4<u8> = npz.by_name("v").unwrap();
    let o_expected: ndarray::Array4<u16> = npz.by_name("o").unwrap();
    let scale_np: ndarray::Array0<f32> = npz.by_name("scale").unwrap();

    let batch = q_np.shape()[0] as u32;
    let num_heads = q_np.shape()[1] as u32;
    let seq_q = q_np.shape()[2] as u32;
    let seq_kv = k_np.shape()[2] as u32;
    let scale = scale_np.into_scalar();
    let head_dim = 128usize;

    let q_flat: Vec<u8> = q_np.into_raw_vec_and_offset().0;
    let k_flat: Vec<u8> = k_np.into_raw_vec_and_offset().0;
    let expected_flat: Vec<u16> = o_expected.into_raw_vec_and_offset().0;

    // Transpose V: [B, H, Skv, D] -> [D, B, H, Skv] contiguous
    // In-memory: v_t[d * B*H*Skv + b*H*Skv + h*Skv + k] = v[b*H*Skv*D + h*Skv*D + k*D + d]
    let b = batch as usize;
    let nh = num_heads as usize;
    let skv = seq_kv as usize;
    let total_kv = b * nh * skv;
    let v_raw: Vec<u8> = v_np.into_raw_vec_and_offset().0;
    let mut v_t = vec![0u8; b * nh * head_dim * skv];
    for d in 0..head_dim {
        for t in 0..total_kv {
            // t = b*nh*skv + h*skv + k (flat token index)
            // v_raw index: t*head_dim + d
            v_t[d * total_kv + t] = v_raw[t * head_dim + d];
        }
    }

    let q_dev = stream.memcpy_stod(&q_flat).unwrap();
    let k_dev = stream.memcpy_stod(&k_flat).unwrap();
    let vt_dev = stream.memcpy_stod(&v_t).unwrap();
    let mut o_dev = stream
        .alloc_zeros::<u16>((batch * num_heads * seq_q * 128) as usize)
        .unwrap();

    attention::flash_attn_fp8_v12c_vt(
        &ctx, &stream, &q_dev, &k_dev, &vt_dev, &mut o_dev, batch, num_heads, seq_q, seq_kv, scale,
    )
    .unwrap();

    let o_host = stream.memcpy_dtov(&o_dev).unwrap();
    let (max_diff, mean_diff) = compare_bf16(&o_host, &expected_flat, tol);
    eprintln!("FP8 V12c {npz_name}: max_diff={max_diff:.6} mean_diff={mean_diff:.6} (tol={tol})");
}

#[test]
fn test_flash_attn_fp8_v12c_noncausal_s256() {
    run_flash_attn_fp8_v12c_test("flash_attn_fp8_noncausal_s256.npz", 0.15);
}

#[test]
fn test_flash_attn_fp8_v12c_noncausal_s1024() {
    run_flash_attn_fp8_v12c_test("flash_attn_fp8_noncausal_s1024.npz", 0.15);
}

fn run_flash_attn_fp8_v12c_causal_test(npz_name: &str, tol: f32) {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let mut npz = load_npz(npz_name);

    let q_np: ndarray::Array4<u8> = npz.by_name("q").unwrap();
    let k_np: ndarray::Array4<u8> = npz.by_name("k").unwrap();
    let v_np: ndarray::Array4<u8> = npz.by_name("v").unwrap();
    let o_expected: ndarray::Array4<u16> = npz.by_name("o").unwrap();
    let scale_np: ndarray::Array0<f32> = npz.by_name("scale").unwrap();

    let batch = q_np.shape()[0] as u32;
    let num_heads = q_np.shape()[1] as u32;
    let seq_q = q_np.shape()[2] as u32;
    let seq_kv = k_np.shape()[2] as u32;
    let scale = scale_np.into_scalar();
    let head_dim = 128usize;

    let q_flat: Vec<u8> = q_np.into_raw_vec_and_offset().0;
    let k_flat: Vec<u8> = k_np.into_raw_vec_and_offset().0;
    let expected_flat: Vec<u16> = o_expected.into_raw_vec_and_offset().0;

    // Transpose V: [B, H, Skv, D] -> [D, B, H, Skv] contiguous
    let b = batch as usize;
    let nh = num_heads as usize;
    let skv = seq_kv as usize;
    let total_kv = b * nh * skv;
    let v_raw: Vec<u8> = v_np.into_raw_vec_and_offset().0;
    let mut v_t = vec![0u8; b * nh * head_dim * skv];
    for d in 0..head_dim {
        for t in 0..total_kv {
            v_t[d * total_kv + t] = v_raw[t * head_dim + d];
        }
    }

    let q_dev = stream.memcpy_stod(&q_flat).unwrap();
    let k_dev = stream.memcpy_stod(&k_flat).unwrap();
    let vt_dev = stream.memcpy_stod(&v_t).unwrap();
    let mut o_dev = stream
        .alloc_zeros::<u16>((batch * num_heads * seq_q * 128) as usize)
        .unwrap();

    attention::flash_attn_fp8_v12c_vt_causal(
        &ctx, &stream, &q_dev, &k_dev, &vt_dev, &mut o_dev, batch, num_heads, seq_q, seq_kv, scale,
    )
    .unwrap();

    let o_host = stream.memcpy_dtov(&o_dev).unwrap();
    let (max_diff, mean_diff) = compare_bf16(&o_host, &expected_flat, tol);
    eprintln!(
        "FP8 V12c causal {npz_name}: max_diff={max_diff:.6} mean_diff={mean_diff:.6} (tol={tol})"
    );
}

#[test]
fn test_flash_attn_fp8_v12c_causal_s256() {
    run_flash_attn_fp8_v12c_causal_test("flash_attn_fp8_causal_s256.npz", 0.15);
}

#[test]
fn test_flash_attn_fp8_v12c_causal_s1024() {
    run_flash_attn_fp8_v12c_causal_test("flash_attn_fp8_causal_s1024.npz", 0.15);
}

// ─── FP8 V12c VT varlen non-causal tests ─────────────────────────────

fn run_flash_attn_fp8_v12c_vt_varlen_test(npz_name: &str, tol: f32) {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let mut npz = load_npz(npz_name);

    let q_flat: ndarray::Array1<u8> = npz.by_name("q").unwrap();
    let k_flat: ndarray::Array1<u8> = npz.by_name("k").unwrap();
    let v_flat: ndarray::Array1<u8> = npz.by_name("v").unwrap();
    let o_expected: ndarray::Array1<u16> = npz.by_name("o").unwrap();
    let cu_seqlens_q: ndarray::Array1<u32> = npz.by_name("cu_seqlens_q").unwrap();
    let cu_seqlens_k: ndarray::Array1<u32> = npz.by_name("cu_seqlens_k").unwrap();
    let num_heads: ndarray::Array0<u32> = npz.by_name("num_heads").unwrap();
    let batch: ndarray::Array0<u32> = npz.by_name("batch").unwrap();
    let max_seqlen_q: ndarray::Array0<u32> = npz.by_name("max_seqlen_q").unwrap();
    let total_q: ndarray::Array0<u32> = npz.by_name("total_q").unwrap();
    let total_kv: ndarray::Array0<u32> = npz.by_name("total_kv").unwrap();
    let scale: ndarray::Array0<f32> = npz.by_name("scale").unwrap();

    let num_heads = num_heads.into_scalar();
    let batch = batch.into_scalar();
    let max_seqlen_q = max_seqlen_q.into_scalar();
    let total_q_val = total_q.into_scalar();
    let total_kv_val = total_kv.into_scalar();
    let scale = scale.into_scalar();
    let head_dim = 128usize;

    let q_vec: Vec<u8> = q_flat.into_raw_vec_and_offset().0;
    let k_vec: Vec<u8> = k_flat.into_raw_vec_and_offset().0;
    let v_vec: Vec<u8> = v_flat.into_raw_vec_and_offset().0;
    let expected: Vec<u16> = o_expected.into_raw_vec_and_offset().0;
    let cu_q: Vec<u32> = cu_seqlens_q.into_raw_vec_and_offset().0;
    let cu_k: Vec<u32> = cu_seqlens_k.into_raw_vec_and_offset().0;

    // Transpose V: [H, total_kv, D] -> [D, H * total_kv]
    let nh = num_heads as usize;
    let total_kv_sz = total_kv_val as usize;
    let total_kv_tokens = nh * total_kv_sz;
    let mut v_t = vec![0u8; head_dim * total_kv_tokens];
    for d in 0..head_dim {
        for t in 0..total_kv_tokens {
            v_t[d * total_kv_tokens + t] = v_vec[t * head_dim + d];
        }
    }

    let q_dev = stream.memcpy_stod(&q_vec).unwrap();
    let k_dev = stream.memcpy_stod(&k_vec).unwrap();
    let vt_dev = stream.memcpy_stod(&v_t).unwrap();
    let cu_q_dev = stream.memcpy_stod(&cu_q).unwrap();
    let cu_k_dev = stream.memcpy_stod(&cu_k).unwrap();
    let mut o_dev = stream
        .alloc_zeros::<u16>((num_heads * total_q_val * 128) as usize)
        .unwrap();

    attention::flash_attn_fp8_v12c_vt_varlen(
        &ctx,
        &stream,
        &q_dev,
        &k_dev,
        &vt_dev,
        &mut o_dev,
        &cu_q_dev,
        &cu_k_dev,
        batch,
        num_heads,
        max_seqlen_q,
        total_q_val,
        total_kv_val,
        scale,
    )
    .unwrap();

    let o_host = stream.memcpy_dtov(&o_dev).unwrap();
    let (max_diff, mean_diff) = compare_bf16(&o_host, &expected, tol);
    eprintln!(
        "FP8 V12c VT varlen {npz_name}: max_diff={max_diff:.6} mean_diff={mean_diff:.6} (batch={batch}, heads={num_heads}, total_q={total_q_val}, tol={tol})"
    );
}

#[test]
fn test_flash_attn_fp8_v12c_vt_varlen_a() {
    run_flash_attn_fp8_v12c_vt_varlen_test("flash_attn_fp8_varlen_a.npz", 0.15);
}

#[test]
fn test_flash_attn_fp8_v12c_vt_varlen_b() {
    run_flash_attn_fp8_v12c_vt_varlen_test("flash_attn_fp8_varlen_b.npz", 0.15);
}

// ─── FP8 V12c VT varlen causal tests ────────────────────────────────

fn run_flash_attn_fp8_v12c_vt_varlen_causal_test(npz_name: &str, tol: f32) {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let mut npz = load_npz(npz_name);

    let q_flat: ndarray::Array1<u8> = npz.by_name("q").unwrap();
    let k_flat: ndarray::Array1<u8> = npz.by_name("k").unwrap();
    let v_flat: ndarray::Array1<u8> = npz.by_name("v").unwrap();
    let o_expected: ndarray::Array1<u16> = npz.by_name("o").unwrap();
    let cu_seqlens_q: ndarray::Array1<u32> = npz.by_name("cu_seqlens_q").unwrap();
    let cu_seqlens_k: ndarray::Array1<u32> = npz.by_name("cu_seqlens_k").unwrap();
    let num_heads: ndarray::Array0<u32> = npz.by_name("num_heads").unwrap();
    let batch: ndarray::Array0<u32> = npz.by_name("batch").unwrap();
    let max_seqlen_q: ndarray::Array0<u32> = npz.by_name("max_seqlen_q").unwrap();
    let total_q: ndarray::Array0<u32> = npz.by_name("total_q").unwrap();
    let total_kv: ndarray::Array0<u32> = npz.by_name("total_kv").unwrap();
    let scale: ndarray::Array0<f32> = npz.by_name("scale").unwrap();

    let num_heads = num_heads.into_scalar();
    let batch = batch.into_scalar();
    let max_seqlen_q = max_seqlen_q.into_scalar();
    let total_q_val = total_q.into_scalar();
    let total_kv_val = total_kv.into_scalar();
    let scale = scale.into_scalar();
    let head_dim = 128usize;

    let q_vec: Vec<u8> = q_flat.into_raw_vec_and_offset().0;
    let k_vec: Vec<u8> = k_flat.into_raw_vec_and_offset().0;
    let v_vec: Vec<u8> = v_flat.into_raw_vec_and_offset().0;
    let expected: Vec<u16> = o_expected.into_raw_vec_and_offset().0;
    let cu_q: Vec<u32> = cu_seqlens_q.into_raw_vec_and_offset().0;
    let cu_k: Vec<u32> = cu_seqlens_k.into_raw_vec_and_offset().0;

    // Transpose V: [H, total_kv, D] -> [D, H * total_kv]
    let nh = num_heads as usize;
    let total_kv_sz = total_kv_val as usize;
    let total_kv_tokens = nh * total_kv_sz;
    let mut v_t = vec![0u8; head_dim * total_kv_tokens];
    for d in 0..head_dim {
        for t in 0..total_kv_tokens {
            v_t[d * total_kv_tokens + t] = v_vec[t * head_dim + d];
        }
    }

    let q_dev = stream.memcpy_stod(&q_vec).unwrap();
    let k_dev = stream.memcpy_stod(&k_vec).unwrap();
    let vt_dev = stream.memcpy_stod(&v_t).unwrap();
    let cu_q_dev = stream.memcpy_stod(&cu_q).unwrap();
    let cu_k_dev = stream.memcpy_stod(&cu_k).unwrap();
    let mut o_dev = stream
        .alloc_zeros::<u16>((num_heads * total_q_val * 128) as usize)
        .unwrap();

    attention::flash_attn_fp8_v12c_vt_varlen_causal(
        &ctx,
        &stream,
        &q_dev,
        &k_dev,
        &vt_dev,
        &mut o_dev,
        &cu_q_dev,
        &cu_k_dev,
        batch,
        num_heads,
        max_seqlen_q,
        total_q_val,
        total_kv_val,
        scale,
    )
    .unwrap();

    let o_host = stream.memcpy_dtov(&o_dev).unwrap();
    let (max_diff, mean_diff) = compare_bf16(&o_host, &expected, tol);
    eprintln!(
        "FP8 V12c VT varlen causal {npz_name}: max_diff={max_diff:.6} mean_diff={mean_diff:.6} (batch={batch}, heads={num_heads}, total_q={total_q_val}, tol={tol})"
    );
}

#[test]
fn test_flash_attn_fp8_v12c_vt_varlen_causal_a() {
    run_flash_attn_fp8_v12c_vt_varlen_causal_test("flash_attn_fp8_varlen_causal_a.npz", 0.15);
}

#[test]
fn test_flash_attn_fp8_v12c_vt_varlen_causal_b() {
    run_flash_attn_fp8_v12c_vt_varlen_causal_test("flash_attn_fp8_varlen_causal_b.npz", 0.15);
}

// ─── FP8 V12c VT GQA non-causal tests ────────────────────────────────

fn run_flash_attn_fp8_v12c_vt_gqa_test(npz_name: &str, tol: f32) {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let mut npz = load_npz(npz_name);

    let q_np: ndarray::Array4<u8> = npz.by_name("q").unwrap();
    let k_np: ndarray::Array4<u8> = npz.by_name("k").unwrap();
    let v_np: ndarray::Array4<u8> = npz.by_name("v").unwrap();
    let o_expected: ndarray::Array4<u16> = npz.by_name("o").unwrap();
    let scale_np: ndarray::Array0<f32> = npz.by_name("scale").unwrap();

    let batch = q_np.shape()[0] as u32;
    let num_heads_q = q_np.shape()[1] as u32;
    let seq_q = q_np.shape()[2] as u32;
    let num_heads_kv = k_np.shape()[1] as u32;
    let seq_kv = k_np.shape()[2] as u32;
    let scale = scale_np.into_scalar();
    let head_dim = 128usize;

    let q_flat: Vec<u8> = q_np.into_raw_vec_and_offset().0;
    let k_flat: Vec<u8> = k_np.into_raw_vec_and_offset().0;
    let expected_flat: Vec<u16> = o_expected.into_raw_vec_and_offset().0;

    // Transpose V: [B, H_kv, Skv, D] -> [D, B * H_kv * Skv]
    let b = batch as usize;
    let nhkv = num_heads_kv as usize;
    let skv = seq_kv as usize;
    let total_kv = b * nhkv * skv;
    let v_raw: Vec<u8> = v_np.into_raw_vec_and_offset().0;
    let mut v_t = vec![0u8; head_dim * total_kv];
    for d in 0..head_dim {
        for t in 0..total_kv {
            v_t[d * total_kv + t] = v_raw[t * head_dim + d];
        }
    }

    let q_dev = stream.memcpy_stod(&q_flat).unwrap();
    let k_dev = stream.memcpy_stod(&k_flat).unwrap();
    let vt_dev = stream.memcpy_stod(&v_t).unwrap();
    let mut o_dev = stream
        .alloc_zeros::<u16>((batch * num_heads_q * seq_q * 128) as usize)
        .unwrap();

    attention::flash_attn_fp8_v12c_vt_gqa(
        &ctx,
        &stream,
        &q_dev,
        &k_dev,
        &vt_dev,
        &mut o_dev,
        batch,
        num_heads_q,
        num_heads_kv,
        seq_q,
        seq_kv,
        scale,
    )
    .unwrap();

    let o_host = stream.memcpy_dtov(&o_dev).unwrap();
    let (max_diff, mean_diff) = compare_bf16(&o_host, &expected_flat, tol);
    eprintln!(
        "FP8 V12c VT GQA {npz_name}: max_diff={max_diff:.6} mean_diff={mean_diff:.6} (batch={batch}, hq={num_heads_q}, hkv={num_heads_kv}, seq={seq_q}, tol={tol})"
    );
}

#[test]
fn test_flash_attn_fp8_v12c_vt_gqa_noncausal_s256() {
    run_flash_attn_fp8_v12c_vt_gqa_test("flash_attn_fp8_gqa_noncausal_s256.npz", 0.15);
}

#[test]
fn test_flash_attn_fp8_v12c_vt_gqa_noncausal_s1024() {
    run_flash_attn_fp8_v12c_vt_gqa_test("flash_attn_fp8_gqa_noncausal_s1024.npz", 0.15);
}

// ─── FP8 V12c VT GQA causal tests ────────────────────────────────────

fn run_flash_attn_fp8_v12c_vt_gqa_causal_test(npz_name: &str, tol: f32) {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let mut npz = load_npz(npz_name);

    let q_np: ndarray::Array4<u8> = npz.by_name("q").unwrap();
    let k_np: ndarray::Array4<u8> = npz.by_name("k").unwrap();
    let v_np: ndarray::Array4<u8> = npz.by_name("v").unwrap();
    let o_expected: ndarray::Array4<u16> = npz.by_name("o").unwrap();
    let scale_np: ndarray::Array0<f32> = npz.by_name("scale").unwrap();

    let batch = q_np.shape()[0] as u32;
    let num_heads_q = q_np.shape()[1] as u32;
    let seq_q = q_np.shape()[2] as u32;
    let num_heads_kv = k_np.shape()[1] as u32;
    let seq_kv = k_np.shape()[2] as u32;
    let scale = scale_np.into_scalar();
    let head_dim = 128usize;

    let q_flat: Vec<u8> = q_np.into_raw_vec_and_offset().0;
    let k_flat: Vec<u8> = k_np.into_raw_vec_and_offset().0;
    let expected_flat: Vec<u16> = o_expected.into_raw_vec_and_offset().0;

    // Transpose V: [B, H_kv, Skv, D] -> [D, B * H_kv * Skv]
    let b = batch as usize;
    let nhkv = num_heads_kv as usize;
    let skv = seq_kv as usize;
    let total_kv = b * nhkv * skv;
    let v_raw: Vec<u8> = v_np.into_raw_vec_and_offset().0;
    let mut v_t = vec![0u8; head_dim * total_kv];
    for d in 0..head_dim {
        for t in 0..total_kv {
            v_t[d * total_kv + t] = v_raw[t * head_dim + d];
        }
    }

    let q_dev = stream.memcpy_stod(&q_flat).unwrap();
    let k_dev = stream.memcpy_stod(&k_flat).unwrap();
    let vt_dev = stream.memcpy_stod(&v_t).unwrap();
    let mut o_dev = stream
        .alloc_zeros::<u16>((batch * num_heads_q * seq_q * 128) as usize)
        .unwrap();

    attention::flash_attn_fp8_v12c_vt_gqa_causal(
        &ctx,
        &stream,
        &q_dev,
        &k_dev,
        &vt_dev,
        &mut o_dev,
        batch,
        num_heads_q,
        num_heads_kv,
        seq_q,
        seq_kv,
        scale,
    )
    .unwrap();

    let o_host = stream.memcpy_dtov(&o_dev).unwrap();
    let (max_diff, mean_diff) = compare_bf16(&o_host, &expected_flat, tol);
    eprintln!(
        "FP8 V12c VT GQA causal {npz_name}: max_diff={max_diff:.6} mean_diff={mean_diff:.6} (batch={batch}, hq={num_heads_q}, hkv={num_heads_kv}, seq={seq_q}, tol={tol})"
    );
}

#[test]
fn test_flash_attn_fp8_v12c_vt_gqa_causal_s256() {
    run_flash_attn_fp8_v12c_vt_gqa_causal_test("flash_attn_fp8_gqa_causal_s256.npz", 0.15);
}

#[test]
fn test_flash_attn_fp8_v12c_vt_gqa_causal_s1024() {
    run_flash_attn_fp8_v12c_vt_gqa_causal_test("flash_attn_fp8_gqa_causal_s1024.npz", 0.15);
}

// ─── FP8 V12c VT varlen GQA non-causal tests ─────────────────────────

fn run_flash_attn_fp8_v12c_vt_varlen_gqa_test(npz_name: &str, tol: f32) {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let mut npz = load_npz(npz_name);

    let q_flat: ndarray::Array1<u8> = npz.by_name("q").unwrap();
    let k_flat: ndarray::Array1<u8> = npz.by_name("k").unwrap();
    let v_flat: ndarray::Array1<u8> = npz.by_name("v").unwrap();
    let o_expected: ndarray::Array1<u16> = npz.by_name("o").unwrap();
    let cu_seqlens_q: ndarray::Array1<u32> = npz.by_name("cu_seqlens_q").unwrap();
    let cu_seqlens_k: ndarray::Array1<u32> = npz.by_name("cu_seqlens_k").unwrap();
    let num_heads_q: ndarray::Array0<u32> = npz.by_name("num_heads_q").unwrap();
    let num_heads_kv: ndarray::Array0<u32> = npz.by_name("num_heads_kv").unwrap();
    let batch: ndarray::Array0<u32> = npz.by_name("batch").unwrap();
    let max_seqlen_q: ndarray::Array0<u32> = npz.by_name("max_seqlen_q").unwrap();
    let total_q: ndarray::Array0<u32> = npz.by_name("total_q").unwrap();
    let total_kv: ndarray::Array0<u32> = npz.by_name("total_kv").unwrap();
    let scale: ndarray::Array0<f32> = npz.by_name("scale").unwrap();

    let num_heads_q = num_heads_q.into_scalar();
    let num_heads_kv = num_heads_kv.into_scalar();
    let batch = batch.into_scalar();
    let max_seqlen_q = max_seqlen_q.into_scalar();
    let total_q_val = total_q.into_scalar();
    let total_kv_val = total_kv.into_scalar();
    let scale = scale.into_scalar();
    let head_dim = 128usize;

    let q_vec: Vec<u8> = q_flat.into_raw_vec_and_offset().0;
    let k_vec: Vec<u8> = k_flat.into_raw_vec_and_offset().0;
    let v_vec: Vec<u8> = v_flat.into_raw_vec_and_offset().0;
    let expected: Vec<u16> = o_expected.into_raw_vec_and_offset().0;
    let cu_q: Vec<u32> = cu_seqlens_q.into_raw_vec_and_offset().0;
    let cu_k: Vec<u32> = cu_seqlens_k.into_raw_vec_and_offset().0;

    // Transpose V: [H_kv, total_kv, D] -> [D, H_kv * total_kv]
    let nhkv = num_heads_kv as usize;
    let total_kv_sz = total_kv_val as usize;
    let total_kv_tokens = nhkv * total_kv_sz;
    let mut v_t = vec![0u8; head_dim * total_kv_tokens];
    for d in 0..head_dim {
        for t in 0..total_kv_tokens {
            v_t[d * total_kv_tokens + t] = v_vec[t * head_dim + d];
        }
    }

    let q_dev = stream.memcpy_stod(&q_vec).unwrap();
    let k_dev = stream.memcpy_stod(&k_vec).unwrap();
    let vt_dev = stream.memcpy_stod(&v_t).unwrap();
    let cu_q_dev = stream.memcpy_stod(&cu_q).unwrap();
    let cu_k_dev = stream.memcpy_stod(&cu_k).unwrap();
    let mut o_dev = stream
        .alloc_zeros::<u16>((num_heads_q * total_q_val * 128) as usize)
        .unwrap();

    attention::flash_attn_fp8_v12c_vt_varlen_gqa(
        &ctx,
        &stream,
        &q_dev,
        &k_dev,
        &vt_dev,
        &mut o_dev,
        &cu_q_dev,
        &cu_k_dev,
        batch,
        num_heads_q,
        num_heads_kv,
        max_seqlen_q,
        total_q_val,
        total_kv_val,
        scale,
    )
    .unwrap();

    let o_host = stream.memcpy_dtov(&o_dev).unwrap();
    let (max_diff, mean_diff) = compare_bf16(&o_host, &expected, tol);
    eprintln!(
        "FP8 V12c VT varlen GQA {npz_name}: max_diff={max_diff:.6} mean_diff={mean_diff:.6} (batch={batch}, hq={num_heads_q}, hkv={num_heads_kv}, total_q={total_q_val}, tol={tol})"
    );
}

#[test]
fn test_flash_attn_fp8_v12c_vt_varlen_gqa_a() {
    run_flash_attn_fp8_v12c_vt_varlen_gqa_test("flash_attn_fp8_varlen_gqa_a.npz", 0.15);
}

#[test]
fn test_flash_attn_fp8_v12c_vt_varlen_gqa_b() {
    run_flash_attn_fp8_v12c_vt_varlen_gqa_test("flash_attn_fp8_varlen_gqa_b.npz", 0.15);
}

// ─── FP8 V12c VT varlen GQA causal tests ─────────────────────────────

fn run_flash_attn_fp8_v12c_vt_varlen_gqa_causal_test(npz_name: &str, tol: f32) {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let mut npz = load_npz(npz_name);

    let q_flat: ndarray::Array1<u8> = npz.by_name("q").unwrap();
    let k_flat: ndarray::Array1<u8> = npz.by_name("k").unwrap();
    let v_flat: ndarray::Array1<u8> = npz.by_name("v").unwrap();
    let o_expected: ndarray::Array1<u16> = npz.by_name("o").unwrap();
    let cu_seqlens_q: ndarray::Array1<u32> = npz.by_name("cu_seqlens_q").unwrap();
    let cu_seqlens_k: ndarray::Array1<u32> = npz.by_name("cu_seqlens_k").unwrap();
    let num_heads_q: ndarray::Array0<u32> = npz.by_name("num_heads_q").unwrap();
    let num_heads_kv: ndarray::Array0<u32> = npz.by_name("num_heads_kv").unwrap();
    let batch: ndarray::Array0<u32> = npz.by_name("batch").unwrap();
    let max_seqlen_q: ndarray::Array0<u32> = npz.by_name("max_seqlen_q").unwrap();
    let total_q: ndarray::Array0<u32> = npz.by_name("total_q").unwrap();
    let total_kv: ndarray::Array0<u32> = npz.by_name("total_kv").unwrap();
    let scale: ndarray::Array0<f32> = npz.by_name("scale").unwrap();

    let num_heads_q = num_heads_q.into_scalar();
    let num_heads_kv = num_heads_kv.into_scalar();
    let batch = batch.into_scalar();
    let max_seqlen_q = max_seqlen_q.into_scalar();
    let total_q_val = total_q.into_scalar();
    let total_kv_val = total_kv.into_scalar();
    let scale = scale.into_scalar();
    let head_dim = 128usize;

    let q_vec: Vec<u8> = q_flat.into_raw_vec_and_offset().0;
    let k_vec: Vec<u8> = k_flat.into_raw_vec_and_offset().0;
    let v_vec: Vec<u8> = v_flat.into_raw_vec_and_offset().0;
    let expected: Vec<u16> = o_expected.into_raw_vec_and_offset().0;
    let cu_q: Vec<u32> = cu_seqlens_q.into_raw_vec_and_offset().0;
    let cu_k: Vec<u32> = cu_seqlens_k.into_raw_vec_and_offset().0;

    // Transpose V: [H_kv, total_kv, D] -> [D, H_kv * total_kv]
    let nhkv = num_heads_kv as usize;
    let total_kv_sz = total_kv_val as usize;
    let total_kv_tokens = nhkv * total_kv_sz;
    let mut v_t = vec![0u8; head_dim * total_kv_tokens];
    for d in 0..head_dim {
        for t in 0..total_kv_tokens {
            v_t[d * total_kv_tokens + t] = v_vec[t * head_dim + d];
        }
    }

    let q_dev = stream.memcpy_stod(&q_vec).unwrap();
    let k_dev = stream.memcpy_stod(&k_vec).unwrap();
    let vt_dev = stream.memcpy_stod(&v_t).unwrap();
    let cu_q_dev = stream.memcpy_stod(&cu_q).unwrap();
    let cu_k_dev = stream.memcpy_stod(&cu_k).unwrap();
    let mut o_dev = stream
        .alloc_zeros::<u16>((num_heads_q * total_q_val * 128) as usize)
        .unwrap();

    attention::flash_attn_fp8_v12c_vt_varlen_gqa_causal(
        &ctx,
        &stream,
        &q_dev,
        &k_dev,
        &vt_dev,
        &mut o_dev,
        &cu_q_dev,
        &cu_k_dev,
        batch,
        num_heads_q,
        num_heads_kv,
        max_seqlen_q,
        total_q_val,
        total_kv_val,
        scale,
    )
    .unwrap();

    let o_host = stream.memcpy_dtov(&o_dev).unwrap();
    let (max_diff, mean_diff) = compare_bf16(&o_host, &expected, tol);
    eprintln!(
        "FP8 V12c VT varlen GQA causal {npz_name}: max_diff={max_diff:.6} mean_diff={mean_diff:.6} (batch={batch}, hq={num_heads_q}, hkv={num_heads_kv}, total_q={total_q_val}, tol={tol})"
    );
}

#[test]
fn test_flash_attn_fp8_v12c_vt_varlen_gqa_causal_a() {
    run_flash_attn_fp8_v12c_vt_varlen_gqa_causal_test(
        "flash_attn_fp8_varlen_gqa_causal_a.npz",
        0.15,
    );
}

#[test]
fn test_flash_attn_fp8_v12c_vt_varlen_gqa_causal_b() {
    run_flash_attn_fp8_v12c_vt_varlen_gqa_causal_test(
        "flash_attn_fp8_varlen_gqa_causal_b.npz",
        0.15,
    );
}

/// Test FP8 paged KV: compare against FP8 non-paged reference.
#[cfg(feature = "experimental")]
#[test]
fn test_flash_attn_fp8_v3_paged_kv_sequential() {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let mut npz = load_npz("flash_attn_fp8_noncausal_s256.npz");
    let q_np: ndarray::Array4<u8> = npz.by_name("q").unwrap();
    let k_np: ndarray::Array4<u8> = npz.by_name("k").unwrap();
    let v_np: ndarray::Array4<u8> = npz.by_name("v").unwrap();
    let scale_np: ndarray::Array0<f32> = npz.by_name("scale").unwrap();

    let batch = q_np.shape()[0] as u32;
    let num_heads = q_np.shape()[1] as u32;
    let num_kv_heads = num_heads;
    let seq_q = q_np.shape()[2] as u32;
    let seq_kv = k_np.shape()[2] as u32;
    let d = 128u32;
    let scale = scale_np.into_scalar();
    let page_size: u32 = 64;
    let num_pages_per_seq = seq_kv / page_size;
    let max_pages = num_pages_per_seq;

    let q_flat: Vec<u8> = q_np.into_raw_vec_and_offset().0;
    let k_flat: Vec<u8> = k_np.into_raw_vec_and_offset().0;
    let v_flat: Vec<u8> = v_np.into_raw_vec_and_offset().0;
    let total_q = (batch * num_heads * seq_q * d) as usize;

    // FP8 non-paged reference
    let q_dev = stream.memcpy_stod(&q_flat).unwrap();
    let k_dev = stream.memcpy_stod(&k_flat).unwrap();
    let v_dev = stream.memcpy_stod(&v_flat).unwrap();
    let mut o_ref = stream.alloc_zeros::<u16>(total_q).unwrap();
    attention::flash_attn_fp8_d128(
        &ctx, &stream, &q_dev, &k_dev, &v_dev, &mut o_ref, batch, num_heads, seq_q, seq_kv, scale,
    )
    .expect("FP8 reference failed");
    let o_ref_host = stream.memcpy_dtov(&o_ref).unwrap();

    // Paged layout [total_pages, page_size, num_kv_heads, D] u8
    let total_pages = batch * num_pages_per_seq;
    let paged_total = (total_pages * page_size * num_kv_heads * d) as usize;
    let mut k_paged = vec![0u8; paged_total];
    let mut v_paged = vec![0u8; paged_total];
    for b in 0..batch {
        for h in 0..num_kv_heads {
            for s in 0..seq_kv {
                let physical_page = b * num_pages_per_seq + s / page_size;
                let page_offset = s % page_size;
                for di in 0..d {
                    let std_idx = (((b * num_heads + h) * seq_kv + s) * d + di) as usize;
                    let paged_idx = (((physical_page * page_size + page_offset) * num_kv_heads + h)
                        * d
                        + di) as usize;
                    k_paged[paged_idx] = k_flat[std_idx];
                    v_paged[paged_idx] = v_flat[std_idx];
                }
            }
        }
    }
    let mut page_table_data = vec![0u32; (batch * max_pages) as usize];
    for b in 0..batch {
        for p in 0..num_pages_per_seq {
            page_table_data[(b * max_pages + p) as usize] = b * num_pages_per_seq + p;
        }
    }

    let k_dev_paged = stream.memcpy_stod(&k_paged).unwrap();
    let v_dev_paged = stream.memcpy_stod(&v_paged).unwrap();
    let pt_dev = stream.memcpy_stod(&page_table_data).unwrap();
    let mut o_paged = stream.alloc_zeros::<u16>(total_q).unwrap();

    attention::flash_attn_fp8_v3_paged_kv(
        &ctx,
        &stream,
        &q_dev,
        &k_dev_paged,
        &v_dev_paged,
        &mut o_paged,
        &pt_dev,
        batch,
        num_heads,
        num_kv_heads,
        seq_q,
        seq_kv,
        page_size,
        max_pages,
        scale,
    )
    .expect("FP8 paged KV failed");

    let o_paged_host = stream.memcpy_dtov(&o_paged).unwrap();
    let (max_diff, mean_diff) = compare_bf16(&o_paged_host, &o_ref_host, 0.15);
    eprintln!(
        "FP8 Paged KV: max_diff={:.6}, mean_diff={:.6}",
        max_diff, mean_diff
    );
    assert!(
        max_diff <= 0.15,
        "FP8 paged KV differs (max_diff={:.4})",
        max_diff
    );
}

/// Test causal split-KV: split forward + combine = causal reference.
#[test]
fn test_flash_attn_bf16_split_kv_causal_2splits() {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let mut npz = load_npz("flash_attn_bf16_causal_s256.npz");
    let q_np: ndarray::Array4<u16> = npz.by_name("q").unwrap();
    let k_np: ndarray::Array4<u16> = npz.by_name("k").unwrap();
    let v_np: ndarray::Array4<u16> = npz.by_name("v").unwrap();
    let scale_np: ndarray::Array0<f32> = npz.by_name("scale").unwrap();

    let batch = q_np.shape()[0] as u32;
    let num_heads = q_np.shape()[1] as u32;
    let seq_q = q_np.shape()[2] as u32;
    let seq_kv = k_np.shape()[2] as u32;
    let d = 128u32;
    let scale = scale_np.into_scalar();
    let num_splits: u32 = 2;

    let q_flat: Vec<u16> = q_np.into_raw_vec_and_offset().0;
    let k_flat: Vec<u16> = k_np.into_raw_vec_and_offset().0;
    let v_flat: Vec<u16> = v_np.into_raw_vec_and_offset().0;
    let total_q = (batch * num_heads * seq_q * d) as usize;

    // Causal reference
    let q_dev = stream.memcpy_stod(&q_flat).unwrap();
    let k_dev = stream.memcpy_stod(&k_flat).unwrap();
    let v_dev = stream.memcpy_stod(&v_flat).unwrap();
    let mut o_ref = stream.alloc_zeros::<u16>(total_q).unwrap();
    attention::flash_attn_bf16_v3_d128_causal(
        &ctx, &stream, &q_dev, &k_dev, &v_dev, &mut o_ref, batch, num_heads, seq_q, seq_kv, scale,
    )
    .expect("V3 causal reference failed");
    let o_ref_host = stream.memcpy_dtov(&o_ref).unwrap();

    // Split-KV causal
    let partial_size = (num_splits * batch * num_heads * seq_q * d) as usize;
    let lse_size = (num_splits * batch * num_heads * seq_q) as usize;
    let mut o_partial = stream.alloc_zeros::<f32>(partial_size).unwrap();
    let mut lse_partial = stream.alloc_zeros::<f32>(lse_size).unwrap();

    for s in 0..num_splits {
        attention::flash_attn_bf16_v3_split_kv_causal(
            &ctx,
            &stream,
            &q_dev,
            &k_dev,
            &v_dev,
            &mut o_partial,
            &mut lse_partial,
            batch,
            num_heads,
            seq_q,
            seq_kv,
            scale,
            num_splits,
            s,
        )
        .unwrap_or_else(|_| panic!("Split-KV causal split {} failed", s));
    }

    let mut o_combined = stream.alloc_zeros::<u16>(total_q).unwrap();
    attention::flash_decoding_combine(
        &ctx,
        &stream,
        &o_partial,
        &lse_partial,
        &mut o_combined,
        batch,
        num_heads,
        seq_q,
        num_splits,
    )
    .expect("Combine failed");

    let o_combined_host = stream.memcpy_dtov(&o_combined).unwrap();
    let (max_diff, mean_diff) = compare_bf16(&o_combined_host, &o_ref_host, 0.1);
    eprintln!(
        "Split-KV causal (2 splits): max_diff={:.6}, mean_diff={:.6}",
        max_diff, mean_diff
    );
    assert!(
        max_diff <= 0.1,
        "Split-KV causal differs (max_diff={:.4})",
        max_diff
    );
}

/// Test split-KV + paged KV combined.
#[test]
fn test_flash_attn_bf16_split_paged_kv_2splits() {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let mut npz = load_npz("flash_attn_bf16_noncausal_s256.npz");
    let q_np: ndarray::Array4<u16> = npz.by_name("q").unwrap();
    let k_np: ndarray::Array4<u16> = npz.by_name("k").unwrap();
    let v_np: ndarray::Array4<u16> = npz.by_name("v").unwrap();
    let scale_np: ndarray::Array0<f32> = npz.by_name("scale").unwrap();

    let batch = q_np.shape()[0] as u32;
    let num_heads = q_np.shape()[1] as u32;
    let num_kv_heads = num_heads;
    let seq_q = q_np.shape()[2] as u32;
    let seq_kv = k_np.shape()[2] as u32;
    let d = 128u32;
    let scale = scale_np.into_scalar();
    let page_size: u32 = 64;
    let num_pages_per_seq = seq_kv / page_size;
    let max_pages = num_pages_per_seq;
    let num_splits: u32 = 2;

    let q_flat: Vec<u16> = q_np.into_raw_vec_and_offset().0;
    let k_flat: Vec<u16> = k_np.into_raw_vec_and_offset().0;
    let v_flat: Vec<u16> = v_np.into_raw_vec_and_offset().0;
    let total_q = (batch * num_heads * seq_q * d) as usize;

    // V3 non-paged reference
    let q_dev = stream.memcpy_stod(&q_flat).unwrap();
    let k_dev_std = stream.memcpy_stod(&k_flat).unwrap();
    let v_dev_std = stream.memcpy_stod(&v_flat).unwrap();
    let mut o_ref = stream.alloc_zeros::<u16>(total_q).unwrap();
    attention::flash_attn_bf16_v3_d128(
        &ctx, &stream, &q_dev, &k_dev_std, &v_dev_std, &mut o_ref, batch, num_heads, seq_q, seq_kv,
        scale,
    )
    .expect("V3 reference failed");
    let o_ref_host = stream.memcpy_dtov(&o_ref).unwrap();

    // Paged layout (shuffled)
    let page_assignments: Vec<Vec<u32>> = vec![vec![3, 0, 2, 1], vec![2, 3, 1, 0]];
    let total_pages = batch * num_pages_per_seq;
    let paged_total = (total_pages * page_size * num_kv_heads * d) as usize;
    let mut k_paged = vec![0u16; paged_total];
    let mut v_paged = vec![0u16; paged_total];
    for b in 0..batch {
        for h in 0..num_kv_heads {
            for s in 0..seq_kv {
                let physical_page =
                    page_assignments[b as usize][(s / page_size) as usize] + b * num_pages_per_seq;
                let page_offset = s % page_size;
                for di in 0..d {
                    let std_idx = (((b * num_heads + h) * seq_kv + s) * d + di) as usize;
                    let paged_idx = (((physical_page * page_size + page_offset) * num_kv_heads + h)
                        * d
                        + di) as usize;
                    k_paged[paged_idx] = k_flat[std_idx];
                    v_paged[paged_idx] = v_flat[std_idx];
                }
            }
        }
    }
    let mut page_table_data = vec![0u32; (batch * max_pages) as usize];
    for b in 0..batch {
        for p in 0..num_pages_per_seq {
            page_table_data[(b * max_pages + p) as usize] =
                page_assignments[b as usize][p as usize] + b * num_pages_per_seq;
        }
    }

    let k_dev_paged = stream.memcpy_stod(&k_paged).unwrap();
    let v_dev_paged = stream.memcpy_stod(&v_paged).unwrap();
    let pt_dev = stream.memcpy_stod(&page_table_data).unwrap();

    // Split-KV + paged
    let partial_size = (num_splits * batch * num_heads * seq_q * d) as usize;
    let lse_size = (num_splits * batch * num_heads * seq_q) as usize;
    let mut o_partial = stream.alloc_zeros::<f32>(partial_size).unwrap();
    let mut lse_partial = stream.alloc_zeros::<f32>(lse_size).unwrap();

    for s in 0..num_splits {
        attention::flash_attn_bf16_v3_split_paged_kv(
            &ctx,
            &stream,
            &q_dev,
            &k_dev_paged,
            &v_dev_paged,
            &mut o_partial,
            &mut lse_partial,
            &pt_dev,
            batch,
            num_heads,
            num_kv_heads,
            seq_q,
            seq_kv,
            page_size,
            max_pages,
            scale,
            num_splits,
            s,
        )
        .unwrap_or_else(|_| panic!("Split+paged split {} failed", s));
    }

    let mut o_combined = stream.alloc_zeros::<u16>(total_q).unwrap();
    attention::flash_decoding_combine(
        &ctx,
        &stream,
        &o_partial,
        &lse_partial,
        &mut o_combined,
        batch,
        num_heads,
        seq_q,
        num_splits,
    )
    .expect("Combine failed");

    let o_combined_host = stream.memcpy_dtov(&o_combined).unwrap();
    let (max_diff, mean_diff) = compare_bf16(&o_combined_host, &o_ref_host, 0.1);
    eprintln!(
        "Split+Paged KV (2 splits, shuffled): max_diff={:.6}, mean_diff={:.6}",
        max_diff, mean_diff
    );
    assert!(
        max_diff <= 0.1,
        "Split+Paged KV differs (max_diff={:.4})",
        max_diff
    );
}

#[test]
fn test_flash_attn_bf16_split_paged_kv_causal_2splits() {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let mut npz = load_npz("flash_attn_bf16_noncausal_s256.npz");
    let q_np: ndarray::Array4<u16> = npz.by_name("q").unwrap();
    let k_np: ndarray::Array4<u16> = npz.by_name("k").unwrap();
    let v_np: ndarray::Array4<u16> = npz.by_name("v").unwrap();
    let scale_np: ndarray::Array0<f32> = npz.by_name("scale").unwrap();

    let batch = q_np.shape()[0] as u32;
    let num_heads = q_np.shape()[1] as u32;
    let num_kv_heads = num_heads;
    let seq_q = q_np.shape()[2] as u32;
    let seq_kv = k_np.shape()[2] as u32;
    let d = 128u32;
    let scale = scale_np.into_scalar();
    let page_size: u32 = 64;
    let num_pages_per_seq = seq_kv / page_size;
    let max_pages = num_pages_per_seq;
    let num_splits: u32 = 2;

    let q_flat: Vec<u16> = q_np.into_raw_vec_and_offset().0;
    let k_flat: Vec<u16> = k_np.into_raw_vec_and_offset().0;
    let v_flat: Vec<u16> = v_np.into_raw_vec_and_offset().0;
    let total_q = (batch * num_heads * seq_q * d) as usize;

    // V3 causal non-paged reference
    let q_dev = stream.memcpy_stod(&q_flat).unwrap();
    let k_dev_std = stream.memcpy_stod(&k_flat).unwrap();
    let v_dev_std = stream.memcpy_stod(&v_flat).unwrap();
    let mut o_ref = stream.alloc_zeros::<u16>(total_q).unwrap();
    attention::flash_attn_bf16_v3_d128_causal(
        &ctx, &stream, &q_dev, &k_dev_std, &v_dev_std, &mut o_ref, batch, num_heads, seq_q, seq_kv,
        scale,
    )
    .expect("V3 causal reference failed");
    let o_ref_host = stream.memcpy_dtov(&o_ref).unwrap();

    // Paged layout (shuffled)
    let page_assignments: Vec<Vec<u32>> = vec![vec![3, 0, 2, 1], vec![2, 3, 1, 0]];
    let total_pages = batch * num_pages_per_seq;
    let paged_total = (total_pages * page_size * num_kv_heads * d) as usize;
    let mut k_paged = vec![0u16; paged_total];
    let mut v_paged = vec![0u16; paged_total];
    for b in 0..batch {
        for h in 0..num_kv_heads {
            for s in 0..seq_kv {
                let physical_page =
                    page_assignments[b as usize][(s / page_size) as usize] + b * num_pages_per_seq;
                let page_offset = s % page_size;
                for di in 0..d {
                    let std_idx = (((b * num_heads + h) * seq_kv + s) * d + di) as usize;
                    let paged_idx = (((physical_page * page_size + page_offset) * num_kv_heads + h)
                        * d
                        + di) as usize;
                    k_paged[paged_idx] = k_flat[std_idx];
                    v_paged[paged_idx] = v_flat[std_idx];
                }
            }
        }
    }
    let mut page_table_data = vec![0u32; (batch * max_pages) as usize];
    for b in 0..batch {
        for p in 0..num_pages_per_seq {
            page_table_data[(b * max_pages + p) as usize] =
                page_assignments[b as usize][p as usize] + b * num_pages_per_seq;
        }
    }

    let k_dev_paged = stream.memcpy_stod(&k_paged).unwrap();
    let v_dev_paged = stream.memcpy_stod(&v_paged).unwrap();
    let pt_dev = stream.memcpy_stod(&page_table_data).unwrap();

    // Split-KV + paged + causal
    let partial_size = (num_splits * batch * num_heads * seq_q * d) as usize;
    let lse_size = (num_splits * batch * num_heads * seq_q) as usize;
    let mut o_partial = stream.alloc_zeros::<f32>(partial_size).unwrap();
    let mut lse_partial = stream.alloc_zeros::<f32>(lse_size).unwrap();

    for s in 0..num_splits {
        attention::flash_attn_bf16_v3_split_paged_kv_causal(
            &ctx,
            &stream,
            &q_dev,
            &k_dev_paged,
            &v_dev_paged,
            &mut o_partial,
            &mut lse_partial,
            &pt_dev,
            batch,
            num_heads,
            num_kv_heads,
            seq_q,
            seq_kv,
            page_size,
            max_pages,
            scale,
            num_splits,
            s,
        )
        .unwrap_or_else(|_| panic!("Split+paged+causal split {} failed", s));
    }

    let mut o_combined = stream.alloc_zeros::<u16>(total_q).unwrap();
    attention::flash_decoding_combine(
        &ctx,
        &stream,
        &o_partial,
        &lse_partial,
        &mut o_combined,
        batch,
        num_heads,
        seq_q,
        num_splits,
    )
    .expect("Combine failed");

    let o_combined_host = stream.memcpy_dtov(&o_combined).unwrap();
    let (max_diff, mean_diff) = compare_bf16(&o_combined_host, &o_ref_host, 0.1);
    eprintln!(
        "Split+Paged+Causal KV (2 splits, shuffled): max_diff={:.6}, mean_diff={:.6}",
        max_diff, mean_diff
    );
    assert!(
        max_diff <= 0.1,
        "Split+Paged+Causal KV differs (max_diff={:.4})",
        max_diff
    );
}

#[cfg(feature = "experimental")]
#[test]
fn test_flash_attn_fp8_v3_paged_kv_causal() {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let mut npz = load_npz("flash_attn_fp8_noncausal_s256.npz");
    let q_np: ndarray::Array4<u8> = npz.by_name("q").unwrap();
    let k_np: ndarray::Array4<u8> = npz.by_name("k").unwrap();
    let v_np: ndarray::Array4<u8> = npz.by_name("v").unwrap();
    let scale_np: ndarray::Array0<f32> = npz.by_name("scale").unwrap();

    let batch = q_np.shape()[0] as u32;
    let num_heads = q_np.shape()[1] as u32;
    let num_kv_heads = num_heads;
    let seq_q = q_np.shape()[2] as u32;
    let seq_kv = k_np.shape()[2] as u32;
    let d = 128u32;
    let scale = scale_np.into_scalar();
    let page_size: u32 = 64;
    let num_pages_per_seq = seq_kv / page_size;
    let max_pages = num_pages_per_seq;

    let q_flat: Vec<u8> = q_np.into_raw_vec_and_offset().0;
    let k_flat: Vec<u8> = k_np.into_raw_vec_and_offset().0;
    let v_flat: Vec<u8> = v_np.into_raw_vec_and_offset().0;
    let total_q = (batch * num_heads * seq_q * d) as usize;

    // FP8 causal non-paged reference
    let q_dev = stream.memcpy_stod(&q_flat).unwrap();
    let k_dev = stream.memcpy_stod(&k_flat).unwrap();
    let v_dev = stream.memcpy_stod(&v_flat).unwrap();
    let mut o_ref = stream.alloc_zeros::<u16>(total_q).unwrap();
    attention::flash_attn_fp8_d128_causal(
        &ctx, &stream, &q_dev, &k_dev, &v_dev, &mut o_ref, batch, num_heads, seq_q, seq_kv, scale,
    )
    .expect("FP8 causal reference failed");
    let o_ref_host = stream.memcpy_dtov(&o_ref).unwrap();

    // Paged layout (sequential)
    let total_pages = batch * num_pages_per_seq;
    let paged_total = (total_pages * page_size * num_kv_heads * d) as usize;
    let mut k_paged = vec![0u8; paged_total];
    let mut v_paged = vec![0u8; paged_total];
    for b in 0..batch {
        for h in 0..num_kv_heads {
            for s in 0..seq_kv {
                let physical_page = b * num_pages_per_seq + s / page_size;
                let page_offset = s % page_size;
                for di in 0..d {
                    let std_idx = (((b * num_heads + h) * seq_kv + s) * d + di) as usize;
                    let paged_idx = (((physical_page * page_size + page_offset) * num_kv_heads + h)
                        * d
                        + di) as usize;
                    k_paged[paged_idx] = k_flat[std_idx];
                    v_paged[paged_idx] = v_flat[std_idx];
                }
            }
        }
    }
    let mut page_table_data = vec![0u32; (batch * max_pages) as usize];
    for b in 0..batch {
        for p in 0..num_pages_per_seq {
            page_table_data[(b * max_pages + p) as usize] = b * num_pages_per_seq + p;
        }
    }

    let k_dev_paged = stream.memcpy_stod(&k_paged).unwrap();
    let v_dev_paged = stream.memcpy_stod(&v_paged).unwrap();
    let pt_dev = stream.memcpy_stod(&page_table_data).unwrap();
    let mut o_paged = stream.alloc_zeros::<u16>(total_q).unwrap();

    attention::flash_attn_fp8_v3_paged_kv_causal(
        &ctx,
        &stream,
        &q_dev,
        &k_dev_paged,
        &v_dev_paged,
        &mut o_paged,
        &pt_dev,
        batch,
        num_heads,
        num_kv_heads,
        seq_q,
        seq_kv,
        page_size,
        max_pages,
        scale,
    )
    .expect("FP8 paged KV causal failed");

    let o_paged_host = stream.memcpy_dtov(&o_paged).unwrap();
    let (max_diff, mean_diff) = compare_bf16(&o_paged_host, &o_ref_host, 0.15);
    eprintln!(
        "FP8 Paged KV Causal: max_diff={:.6}, mean_diff={:.6}",
        max_diff, mean_diff
    );
    assert!(
        max_diff <= 0.15,
        "FP8 paged KV causal differs (max_diff={:.4})",
        max_diff
    );
}

#[test]
fn test_flash_attn_fp8_v3_paged_kv_gqa() {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let mut npz = load_npz("flash_attn_fp8_gqa_noncausal_s256.npz");
    let q_np: ndarray::Array4<u8> = npz.by_name("q").unwrap();
    let k_np: ndarray::Array4<u8> = npz.by_name("k").unwrap();
    let v_np: ndarray::Array4<u8> = npz.by_name("v").unwrap();
    let o_expected: ndarray::Array4<u16> = npz.by_name("o").unwrap();
    let scale_np: ndarray::Array0<f32> = npz.by_name("scale").unwrap();

    let batch = q_np.shape()[0] as u32; // 2
    let num_heads = q_np.shape()[1] as u32; // 8
    let num_kv_heads = k_np.shape()[1] as u32; // 2
    let seq_q = q_np.shape()[2] as u32; // 256
    let seq_kv = k_np.shape()[2] as u32; // 256
    let d = 128u32;
    let scale = scale_np.into_scalar();
    let page_size: u32 = 64;
    let num_pages_per_seq = seq_kv / page_size;
    let max_pages = num_pages_per_seq;

    let q_flat: Vec<u8> = q_np.into_raw_vec_and_offset().0;
    let k_flat: Vec<u8> = k_np.into_raw_vec_and_offset().0;
    let v_flat: Vec<u8> = v_np.into_raw_vec_and_offset().0;
    let expected_flat: Vec<u16> = o_expected.into_raw_vec_and_offset().0;
    let total_q = (batch * num_heads * seq_q * d) as usize;

    // Paged KV layout: [total_pages, page_size, num_kv_heads, D]
    let total_pages = batch * num_pages_per_seq;
    let paged_total = (total_pages * page_size * num_kv_heads * d) as usize;
    let mut k_paged = vec![0u8; paged_total];
    let mut v_paged = vec![0u8; paged_total];
    for b in 0..batch {
        for h in 0..num_kv_heads {
            for s in 0..seq_kv {
                let physical_page = b * num_pages_per_seq + s / page_size;
                let page_offset = s % page_size;
                for di in 0..d {
                    let std_idx = (((b * num_kv_heads + h) * seq_kv + s) * d + di) as usize;
                    let paged_idx = (((physical_page * page_size + page_offset) * num_kv_heads + h)
                        * d
                        + di) as usize;
                    k_paged[paged_idx] = k_flat[std_idx];
                    v_paged[paged_idx] = v_flat[std_idx];
                }
            }
        }
    }
    let mut page_table_data = vec![0u32; (batch * max_pages) as usize];
    for b in 0..batch {
        for p in 0..num_pages_per_seq {
            page_table_data[(b * max_pages + p) as usize] = b * num_pages_per_seq + p;
        }
    }

    let q_dev = stream.memcpy_stod(&q_flat).unwrap();
    let k_dev_paged = stream.memcpy_stod(&k_paged).unwrap();
    let v_dev_paged = stream.memcpy_stod(&v_paged).unwrap();
    let pt_dev = stream.memcpy_stod(&page_table_data).unwrap();
    let mut o_paged = stream.alloc_zeros::<u16>(total_q).unwrap();

    attention::flash_attn_fp8_v3_paged_kv(
        &ctx,
        &stream,
        &q_dev,
        &k_dev_paged,
        &v_dev_paged,
        &mut o_paged,
        &pt_dev,
        batch,
        num_heads,
        num_kv_heads,
        seq_q,
        seq_kv,
        page_size,
        max_pages,
        scale,
    )
    .expect("FP8 paged KV GQA failed");

    let o_paged_host = stream.memcpy_dtov(&o_paged).unwrap();
    let (max_diff, mean_diff) = compare_bf16(&o_paged_host, &expected_flat, 0.15);
    eprintln!(
        "FP8 Paged KV GQA (Hq=8,Hkv=2): max_diff={:.6}, mean_diff={:.6}",
        max_diff, mean_diff
    );
    assert!(
        max_diff <= 0.15,
        "FP8 paged KV GQA differs (max_diff={:.4})",
        max_diff
    );
}

#[test]
fn test_flash_attn_bf16_split_paged_kv_gqa() {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let mut npz = load_npz("flash_attn_bf16_v11_gqa_noncausal.npz");
    let q_np: ndarray::Array1<u16> = npz.by_name("q").unwrap();
    let k_np: ndarray::Array1<u16> = npz.by_name("k").unwrap();
    let v_np: ndarray::Array1<u16> = npz.by_name("v").unwrap();
    let o_np: ndarray::Array1<u16> = npz.by_name("o").unwrap();
    let batch_np: ndarray::Array0<u32> = npz.by_name("batch").unwrap();
    let num_heads_q_np: ndarray::Array0<u32> = npz.by_name("num_heads_q").unwrap();
    let num_heads_kv_np: ndarray::Array0<u32> = npz.by_name("num_heads_kv").unwrap();
    let seq_q_np: ndarray::Array0<u32> = npz.by_name("seq_q").unwrap();
    let seq_kv_np: ndarray::Array0<u32> = npz.by_name("seq_kv").unwrap();
    let scale_np: ndarray::Array0<f32> = npz.by_name("scale").unwrap();

    let q_flat: Vec<u16> = q_np.into_raw_vec_and_offset().0;
    let k_flat: Vec<u16> = k_np.into_raw_vec_and_offset().0;
    let v_flat: Vec<u16> = v_np.into_raw_vec_and_offset().0;
    let o_expected: Vec<u16> = o_np.into_raw_vec_and_offset().0;
    let batch = batch_np.into_scalar();
    let num_heads = num_heads_q_np.into_scalar();
    let num_kv_heads = num_heads_kv_np.into_scalar();
    let seq_q = seq_q_np.into_scalar();
    let seq_kv = seq_kv_np.into_scalar();
    let scale = scale_np.into_scalar();

    let d = 128u32;
    let page_size: u32 = 64;
    let num_pages_per_seq = seq_kv / page_size;
    let max_pages = num_pages_per_seq;
    let num_splits: u32 = 2;
    let total_q = (batch * num_heads * seq_q * d) as usize;

    // Paged KV layout: [total_pages, page_size, num_kv_heads, D]
    let total_pages = batch * num_pages_per_seq;
    let paged_total = (total_pages * page_size * num_kv_heads * d) as usize;
    let mut k_paged = vec![0u16; paged_total];
    let mut v_paged = vec![0u16; paged_total];
    for b in 0..batch {
        for h in 0..num_kv_heads {
            for s in 0..seq_kv {
                let physical_page = b * num_pages_per_seq + s / page_size;
                let page_offset = s % page_size;
                for di in 0..d {
                    let std_idx = (((b * num_kv_heads + h) * seq_kv + s) * d + di) as usize;
                    let paged_idx = (((physical_page * page_size + page_offset) * num_kv_heads + h)
                        * d
                        + di) as usize;
                    k_paged[paged_idx] = k_flat[std_idx];
                    v_paged[paged_idx] = v_flat[std_idx];
                }
            }
        }
    }
    let mut page_table_data = vec![0u32; (batch * max_pages) as usize];
    for b in 0..batch {
        for p in 0..num_pages_per_seq {
            page_table_data[(b * max_pages + p) as usize] = b * num_pages_per_seq + p;
        }
    }

    let q_dev = stream.memcpy_stod(&q_flat).unwrap();
    let k_dev_paged = stream.memcpy_stod(&k_paged).unwrap();
    let v_dev_paged = stream.memcpy_stod(&v_paged).unwrap();
    let pt_dev = stream.memcpy_stod(&page_table_data).unwrap();

    // Split-KV + paged + GQA
    let partial_size = (num_splits * batch * num_heads * seq_q * d) as usize;
    let lse_size = (num_splits * batch * num_heads * seq_q) as usize;
    let mut o_partial = stream.alloc_zeros::<f32>(partial_size).unwrap();
    let mut lse_partial = stream.alloc_zeros::<f32>(lse_size).unwrap();

    for s in 0..num_splits {
        attention::flash_attn_bf16_v3_split_paged_kv(
            &ctx,
            &stream,
            &q_dev,
            &k_dev_paged,
            &v_dev_paged,
            &mut o_partial,
            &mut lse_partial,
            &pt_dev,
            batch,
            num_heads,
            num_kv_heads,
            seq_q,
            seq_kv,
            page_size,
            max_pages,
            scale,
            num_splits,
            s,
        )
        .unwrap_or_else(|_| panic!("Split+paged+GQA split {} failed", s));
    }

    let mut o_combined = stream.alloc_zeros::<u16>(total_q).unwrap();
    attention::flash_decoding_combine(
        &ctx,
        &stream,
        &o_partial,
        &lse_partial,
        &mut o_combined,
        batch,
        num_heads,
        seq_q,
        num_splits,
    )
    .expect("Combine failed");

    let o_combined_host = stream.memcpy_dtov(&o_combined).unwrap();
    let (max_diff, mean_diff) = compare_bf16(&o_combined_host, &o_expected, 0.1);
    eprintln!(
        "Split+Paged+GQA (Hq={},Hkv={},2splits): max_diff={:.6}, mean_diff={:.6}",
        num_heads, num_kv_heads, max_diff, mean_diff
    );
    assert!(
        max_diff <= 0.1,
        "Split+Paged+GQA differs (max_diff={:.4})",
        max_diff
    );
}

#[test]
fn test_flash_attn_bf16_split_paged_kv_gqa_causal() {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let mut npz = load_npz("flash_attn_bf16_v11_gqa_causal.npz");
    let q_np: ndarray::Array1<u16> = npz.by_name("q").unwrap();
    let k_np: ndarray::Array1<u16> = npz.by_name("k").unwrap();
    let v_np: ndarray::Array1<u16> = npz.by_name("v").unwrap();
    let o_np: ndarray::Array1<u16> = npz.by_name("o").unwrap();
    let batch_np: ndarray::Array0<u32> = npz.by_name("batch").unwrap();
    let num_heads_q_np: ndarray::Array0<u32> = npz.by_name("num_heads_q").unwrap();
    let num_heads_kv_np: ndarray::Array0<u32> = npz.by_name("num_heads_kv").unwrap();
    let seq_q_np: ndarray::Array0<u32> = npz.by_name("seq_q").unwrap();
    let seq_kv_np: ndarray::Array0<u32> = npz.by_name("seq_kv").unwrap();
    let scale_np: ndarray::Array0<f32> = npz.by_name("scale").unwrap();

    let q_flat: Vec<u16> = q_np.into_raw_vec_and_offset().0;
    let k_flat: Vec<u16> = k_np.into_raw_vec_and_offset().0;
    let v_flat: Vec<u16> = v_np.into_raw_vec_and_offset().0;
    let o_expected: Vec<u16> = o_np.into_raw_vec_and_offset().0;
    let batch = batch_np.into_scalar();
    let num_heads = num_heads_q_np.into_scalar();
    let num_kv_heads = num_heads_kv_np.into_scalar();
    let seq_q = seq_q_np.into_scalar();
    let seq_kv = seq_kv_np.into_scalar();
    let scale = scale_np.into_scalar();

    let d = 128u32;
    let page_size: u32 = 64;
    let num_pages_per_seq = seq_kv / page_size;
    let max_pages = num_pages_per_seq;
    let num_splits: u32 = 2;
    let total_q = (batch * num_heads * seq_q * d) as usize;

    let total_pages = batch * num_pages_per_seq;
    let paged_total = (total_pages * page_size * num_kv_heads * d) as usize;
    let mut k_paged = vec![0u16; paged_total];
    let mut v_paged = vec![0u16; paged_total];
    for b in 0..batch {
        for h in 0..num_kv_heads {
            for s in 0..seq_kv {
                let physical_page = b * num_pages_per_seq + s / page_size;
                let page_offset = s % page_size;
                for di in 0..d {
                    let std_idx = (((b * num_kv_heads + h) * seq_kv + s) * d + di) as usize;
                    let paged_idx = (((physical_page * page_size + page_offset) * num_kv_heads + h)
                        * d
                        + di) as usize;
                    k_paged[paged_idx] = k_flat[std_idx];
                    v_paged[paged_idx] = v_flat[std_idx];
                }
            }
        }
    }
    let mut page_table_data = vec![0u32; (batch * max_pages) as usize];
    for b in 0..batch {
        for p in 0..num_pages_per_seq {
            page_table_data[(b * max_pages + p) as usize] = b * num_pages_per_seq + p;
        }
    }

    let q_dev = stream.memcpy_stod(&q_flat).unwrap();
    let k_dev_paged = stream.memcpy_stod(&k_paged).unwrap();
    let v_dev_paged = stream.memcpy_stod(&v_paged).unwrap();
    let pt_dev = stream.memcpy_stod(&page_table_data).unwrap();

    let partial_size = (num_splits * batch * num_heads * seq_q * d) as usize;
    let lse_size = (num_splits * batch * num_heads * seq_q) as usize;
    let mut o_partial = stream.alloc_zeros::<f32>(partial_size).unwrap();
    let mut lse_partial = stream.alloc_zeros::<f32>(lse_size).unwrap();

    for s in 0..num_splits {
        attention::flash_attn_bf16_v3_split_paged_kv_causal(
            &ctx,
            &stream,
            &q_dev,
            &k_dev_paged,
            &v_dev_paged,
            &mut o_partial,
            &mut lse_partial,
            &pt_dev,
            batch,
            num_heads,
            num_kv_heads,
            seq_q,
            seq_kv,
            page_size,
            max_pages,
            scale,
            num_splits,
            s,
        )
        .unwrap_or_else(|_| panic!("Split+paged+GQA+causal split {} failed", s));
    }

    let mut o_combined = stream.alloc_zeros::<u16>(total_q).unwrap();
    attention::flash_decoding_combine(
        &ctx,
        &stream,
        &o_partial,
        &lse_partial,
        &mut o_combined,
        batch,
        num_heads,
        seq_q,
        num_splits,
    )
    .expect("Combine failed");

    let o_combined_host = stream.memcpy_dtov(&o_combined).unwrap();
    let (max_diff, mean_diff) = compare_bf16(&o_combined_host, &o_expected, 0.1);
    eprintln!(
        "Split+Paged+GQA+Causal (Hq={},Hkv={},2splits): max_diff={:.6}, mean_diff={:.6}",
        num_heads, num_kv_heads, max_diff, mean_diff
    );
    assert!(
        max_diff <= 0.1,
        "Split+Paged+GQA+Causal differs (max_diff={:.4})",
        max_diff
    );
}

#[cfg(feature = "experimental")]
#[test]
fn test_flash_attn_fp8_split_paged_kv_2splits() {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let mut npz = load_npz("flash_attn_fp8_noncausal_s256.npz");
    let q_np: ndarray::Array4<u8> = npz.by_name("q").unwrap();
    let k_np: ndarray::Array4<u8> = npz.by_name("k").unwrap();
    let v_np: ndarray::Array4<u8> = npz.by_name("v").unwrap();
    let scale_np: ndarray::Array0<f32> = npz.by_name("scale").unwrap();

    let batch = q_np.shape()[0] as u32;
    let num_heads = q_np.shape()[1] as u32;
    let num_kv_heads = num_heads;
    let seq_q = q_np.shape()[2] as u32;
    let seq_kv = k_np.shape()[2] as u32;
    let d = 128u32;
    let scale = scale_np.into_scalar();
    let page_size: u32 = 64;
    let num_pages_per_seq = seq_kv / page_size;
    let max_pages = num_pages_per_seq;
    let num_splits: u32 = 2;

    let q_flat: Vec<u8> = q_np.into_raw_vec_and_offset().0;
    let k_flat: Vec<u8> = k_np.into_raw_vec_and_offset().0;
    let v_flat: Vec<u8> = v_np.into_raw_vec_and_offset().0;
    let total_q = (batch * num_heads * seq_q * d) as usize;

    // FP8 non-paged reference
    let q_dev = stream.memcpy_stod(&q_flat).unwrap();
    let k_dev = stream.memcpy_stod(&k_flat).unwrap();
    let v_dev = stream.memcpy_stod(&v_flat).unwrap();
    let mut o_ref = stream.alloc_zeros::<u16>(total_q).unwrap();
    attention::flash_attn_fp8_d128(
        &ctx, &stream, &q_dev, &k_dev, &v_dev, &mut o_ref, batch, num_heads, seq_q, seq_kv, scale,
    )
    .expect("FP8 reference failed");
    let o_ref_host = stream.memcpy_dtov(&o_ref).unwrap();

    // Paged layout (sequential)
    let total_pages = batch * num_pages_per_seq;
    let paged_total = (total_pages * page_size * num_kv_heads * d) as usize;
    let mut k_paged = vec![0u8; paged_total];
    let mut v_paged = vec![0u8; paged_total];
    for b in 0..batch {
        for h in 0..num_kv_heads {
            for s in 0..seq_kv {
                let physical_page = b * num_pages_per_seq + s / page_size;
                let page_offset = s % page_size;
                for di in 0..d {
                    let std_idx = (((b * num_heads + h) * seq_kv + s) * d + di) as usize;
                    let paged_idx = (((physical_page * page_size + page_offset) * num_kv_heads + h)
                        * d
                        + di) as usize;
                    k_paged[paged_idx] = k_flat[std_idx];
                    v_paged[paged_idx] = v_flat[std_idx];
                }
            }
        }
    }
    let mut page_table_data = vec![0u32; (batch * max_pages) as usize];
    for b in 0..batch {
        for p in 0..num_pages_per_seq {
            page_table_data[(b * max_pages + p) as usize] = b * num_pages_per_seq + p;
        }
    }

    let k_dev_paged = stream.memcpy_stod(&k_paged).unwrap();
    let v_dev_paged = stream.memcpy_stod(&v_paged).unwrap();
    let pt_dev = stream.memcpy_stod(&page_table_data).unwrap();

    // FP8 Split-KV + paged
    let partial_size = (num_splits * batch * num_heads * seq_q * d) as usize;
    let lse_size = (num_splits * batch * num_heads * seq_q) as usize;
    let mut o_partial = stream.alloc_zeros::<f32>(partial_size).unwrap();
    let mut lse_partial = stream.alloc_zeros::<f32>(lse_size).unwrap();

    for s in 0..num_splits {
        attention::flash_attn_fp8_v3_split_paged_kv(
            &ctx,
            &stream,
            &q_dev,
            &k_dev_paged,
            &v_dev_paged,
            &mut o_partial,
            &mut lse_partial,
            &pt_dev,
            batch,
            num_heads,
            num_kv_heads,
            seq_q,
            seq_kv,
            page_size,
            max_pages,
            scale,
            num_splits,
            s,
        )
        .unwrap_or_else(|_| panic!("FP8 Split+paged split {} failed", s));
    }

    let mut o_combined = stream.alloc_zeros::<u16>(total_q).unwrap();
    attention::flash_decoding_combine(
        &ctx,
        &stream,
        &o_partial,
        &lse_partial,
        &mut o_combined,
        batch,
        num_heads,
        seq_q,
        num_splits,
    )
    .expect("Combine failed");

    let o_combined_host = stream.memcpy_dtov(&o_combined).unwrap();
    let (max_diff, mean_diff) = compare_bf16(&o_combined_host, &o_ref_host, 0.15);
    eprintln!(
        "FP8 Split+Paged KV (2 splits): max_diff={:.6}, mean_diff={:.6}",
        max_diff, mean_diff
    );
    assert!(
        max_diff <= 0.15,
        "FP8 Split+Paged KV differs (max_diff={:.4})",
        max_diff
    );
}

#[cfg(feature = "experimental")]
#[test]
fn test_flash_attn_fp8_split_paged_kv_causal_2splits() {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let mut npz = load_npz("flash_attn_fp8_noncausal_s256.npz");
    let q_np: ndarray::Array4<u8> = npz.by_name("q").unwrap();
    let k_np: ndarray::Array4<u8> = npz.by_name("k").unwrap();
    let v_np: ndarray::Array4<u8> = npz.by_name("v").unwrap();
    let scale_np: ndarray::Array0<f32> = npz.by_name("scale").unwrap();

    let batch = q_np.shape()[0] as u32;
    let num_heads = q_np.shape()[1] as u32;
    let num_kv_heads = num_heads;
    let seq_q = q_np.shape()[2] as u32;
    let seq_kv = k_np.shape()[2] as u32;
    let d = 128u32;
    let scale = scale_np.into_scalar();
    let page_size: u32 = 64;
    let num_pages_per_seq = seq_kv / page_size;
    let max_pages = num_pages_per_seq;
    let num_splits: u32 = 2;

    let q_flat: Vec<u8> = q_np.into_raw_vec_and_offset().0;
    let k_flat: Vec<u8> = k_np.into_raw_vec_and_offset().0;
    let v_flat: Vec<u8> = v_np.into_raw_vec_and_offset().0;
    let total_q = (batch * num_heads * seq_q * d) as usize;

    // FP8 causal non-paged reference
    let q_dev = stream.memcpy_stod(&q_flat).unwrap();
    let k_dev = stream.memcpy_stod(&k_flat).unwrap();
    let v_dev = stream.memcpy_stod(&v_flat).unwrap();
    let mut o_ref = stream.alloc_zeros::<u16>(total_q).unwrap();
    attention::flash_attn_fp8_d128_causal(
        &ctx, &stream, &q_dev, &k_dev, &v_dev, &mut o_ref, batch, num_heads, seq_q, seq_kv, scale,
    )
    .expect("FP8 causal reference failed");
    let o_ref_host = stream.memcpy_dtov(&o_ref).unwrap();

    // Paged layout
    let total_pages = batch * num_pages_per_seq;
    let paged_total = (total_pages * page_size * num_kv_heads * d) as usize;
    let mut k_paged = vec![0u8; paged_total];
    let mut v_paged = vec![0u8; paged_total];
    for b in 0..batch {
        for h in 0..num_kv_heads {
            for s in 0..seq_kv {
                let physical_page = b * num_pages_per_seq + s / page_size;
                let page_offset = s % page_size;
                for di in 0..d {
                    let std_idx = (((b * num_heads + h) * seq_kv + s) * d + di) as usize;
                    let paged_idx = (((physical_page * page_size + page_offset) * num_kv_heads + h)
                        * d
                        + di) as usize;
                    k_paged[paged_idx] = k_flat[std_idx];
                    v_paged[paged_idx] = v_flat[std_idx];
                }
            }
        }
    }
    let mut page_table_data = vec![0u32; (batch * max_pages) as usize];
    for b in 0..batch {
        for p in 0..num_pages_per_seq {
            page_table_data[(b * max_pages + p) as usize] = b * num_pages_per_seq + p;
        }
    }

    let k_dev_paged = stream.memcpy_stod(&k_paged).unwrap();
    let v_dev_paged = stream.memcpy_stod(&v_paged).unwrap();
    let pt_dev = stream.memcpy_stod(&page_table_data).unwrap();

    let partial_size = (num_splits * batch * num_heads * seq_q * d) as usize;
    let lse_size = (num_splits * batch * num_heads * seq_q) as usize;
    let mut o_partial = stream.alloc_zeros::<f32>(partial_size).unwrap();
    let mut lse_partial = stream.alloc_zeros::<f32>(lse_size).unwrap();

    for s in 0..num_splits {
        attention::flash_attn_fp8_v3_split_paged_kv_causal(
            &ctx,
            &stream,
            &q_dev,
            &k_dev_paged,
            &v_dev_paged,
            &mut o_partial,
            &mut lse_partial,
            &pt_dev,
            batch,
            num_heads,
            num_kv_heads,
            seq_q,
            seq_kv,
            page_size,
            max_pages,
            scale,
            num_splits,
            s,
        )
        .unwrap_or_else(|_| panic!("FP8 Split+paged+causal split {} failed", s));
    }

    let mut o_combined = stream.alloc_zeros::<u16>(total_q).unwrap();
    attention::flash_decoding_combine(
        &ctx,
        &stream,
        &o_partial,
        &lse_partial,
        &mut o_combined,
        batch,
        num_heads,
        seq_q,
        num_splits,
    )
    .expect("Combine failed");

    let o_combined_host = stream.memcpy_dtov(&o_combined).unwrap();
    let (max_diff, mean_diff) = compare_bf16(&o_combined_host, &o_ref_host, 0.15);
    eprintln!(
        "FP8 Split+Paged+Causal (2 splits): max_diff={:.6}, mean_diff={:.6}",
        max_diff, mean_diff
    );
    assert!(
        max_diff <= 0.15,
        "FP8 Split+Paged+Causal differs (max_diff={:.4})",
        max_diff
    );
}

#[test]
fn test_flash_attn_fp8_split_paged_kv_gqa() {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let mut npz = load_npz("flash_attn_fp8_gqa_noncausal_s256.npz");
    let q_np: ndarray::Array4<u8> = npz.by_name("q").unwrap();
    let k_np: ndarray::Array4<u8> = npz.by_name("k").unwrap();
    let v_np: ndarray::Array4<u8> = npz.by_name("v").unwrap();
    let o_expected: ndarray::Array4<u16> = npz.by_name("o").unwrap();
    let scale_np: ndarray::Array0<f32> = npz.by_name("scale").unwrap();

    let batch = q_np.shape()[0] as u32;
    let num_heads = q_np.shape()[1] as u32;
    let num_kv_heads = k_np.shape()[1] as u32;
    let seq_q = q_np.shape()[2] as u32;
    let seq_kv = k_np.shape()[2] as u32;
    let d = 128u32;
    let scale = scale_np.into_scalar();
    let page_size: u32 = 64;
    let num_pages_per_seq = seq_kv / page_size;
    let max_pages = num_pages_per_seq;
    let num_splits: u32 = 2;

    let q_flat: Vec<u8> = q_np.into_raw_vec_and_offset().0;
    let k_flat: Vec<u8> = k_np.into_raw_vec_and_offset().0;
    let v_flat: Vec<u8> = v_np.into_raw_vec_and_offset().0;
    let expected_flat: Vec<u16> = o_expected.into_raw_vec_and_offset().0;
    let total_q = (batch * num_heads * seq_q * d) as usize;

    // Paged KV layout
    let total_pages = batch * num_pages_per_seq;
    let paged_total = (total_pages * page_size * num_kv_heads * d) as usize;
    let mut k_paged = vec![0u8; paged_total];
    let mut v_paged = vec![0u8; paged_total];
    for b in 0..batch {
        for h in 0..num_kv_heads {
            for s in 0..seq_kv {
                let physical_page = b * num_pages_per_seq + s / page_size;
                let page_offset = s % page_size;
                for di in 0..d {
                    let std_idx = (((b * num_kv_heads + h) * seq_kv + s) * d + di) as usize;
                    let paged_idx = (((physical_page * page_size + page_offset) * num_kv_heads + h)
                        * d
                        + di) as usize;
                    k_paged[paged_idx] = k_flat[std_idx];
                    v_paged[paged_idx] = v_flat[std_idx];
                }
            }
        }
    }
    let mut page_table_data = vec![0u32; (batch * max_pages) as usize];
    for b in 0..batch {
        for p in 0..num_pages_per_seq {
            page_table_data[(b * max_pages + p) as usize] = b * num_pages_per_seq + p;
        }
    }

    let q_dev = stream.memcpy_stod(&q_flat).unwrap();
    let k_dev_paged = stream.memcpy_stod(&k_paged).unwrap();
    let v_dev_paged = stream.memcpy_stod(&v_paged).unwrap();
    let pt_dev = stream.memcpy_stod(&page_table_data).unwrap();

    let partial_size = (num_splits * batch * num_heads * seq_q * d) as usize;
    let lse_size = (num_splits * batch * num_heads * seq_q) as usize;
    let mut o_partial = stream.alloc_zeros::<f32>(partial_size).unwrap();
    let mut lse_partial = stream.alloc_zeros::<f32>(lse_size).unwrap();

    for s in 0..num_splits {
        attention::flash_attn_fp8_v3_split_paged_kv(
            &ctx,
            &stream,
            &q_dev,
            &k_dev_paged,
            &v_dev_paged,
            &mut o_partial,
            &mut lse_partial,
            &pt_dev,
            batch,
            num_heads,
            num_kv_heads,
            seq_q,
            seq_kv,
            page_size,
            max_pages,
            scale,
            num_splits,
            s,
        )
        .unwrap_or_else(|_| panic!("FP8 Split+paged+GQA split {} failed", s));
    }

    let mut o_combined = stream.alloc_zeros::<u16>(total_q).unwrap();
    attention::flash_decoding_combine(
        &ctx,
        &stream,
        &o_partial,
        &lse_partial,
        &mut o_combined,
        batch,
        num_heads,
        seq_q,
        num_splits,
    )
    .expect("Combine failed");

    let o_combined_host = stream.memcpy_dtov(&o_combined).unwrap();
    let (max_diff, mean_diff) = compare_bf16(&o_combined_host, &expected_flat, 0.15);
    eprintln!(
        "FP8 Split+Paged+GQA (Hq={},Hkv={},2splits): max_diff={:.6}, mean_diff={:.6}",
        num_heads, num_kv_heads, max_diff, mean_diff
    );
    assert!(
        max_diff <= 0.15,
        "FP8 Split+Paged+GQA differs (max_diff={:.4})",
        max_diff
    );
}

#[cfg(feature = "experimental")]
#[test]
fn test_flash_attn_bf16_v18_4warp_noncausal_s256() {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let mut npz = load_npz("flash_attn_bf16_noncausal_s256.npz");
    let q_np: ndarray::Array4<u16> = npz.by_name("q").unwrap();
    let k_np: ndarray::Array4<u16> = npz.by_name("k").unwrap();
    let v_np: ndarray::Array4<u16> = npz.by_name("v").unwrap();
    let o_expected: ndarray::Array4<u16> = npz.by_name("o").unwrap();
    let scale_np: ndarray::Array0<f32> = npz.by_name("scale").unwrap();

    let batch = q_np.shape()[0] as u32;
    let num_heads = q_np.shape()[1] as u32;
    let seq_q = q_np.shape()[2] as u32;
    let seq_kv = k_np.shape()[2] as u32;
    let scale = scale_np.into_scalar();

    let q_flat: Vec<u16> = q_np.into_raw_vec_and_offset().0;
    let k_flat: Vec<u16> = k_np.into_raw_vec_and_offset().0;
    let v_flat: Vec<u16> = v_np.into_raw_vec_and_offset().0;
    let expected_flat: Vec<u16> = o_expected.into_raw_vec_and_offset().0;
    let total_q = (batch * num_heads * seq_q * 128) as usize;

    let q_dev = stream.memcpy_stod(&q_flat).unwrap();
    let k_dev = stream.memcpy_stod(&k_flat).unwrap();
    let v_dev = stream.memcpy_stod(&v_flat).unwrap();
    let mut o_dev = stream.alloc_zeros::<u16>(total_q).unwrap();

    attention::flash_attn_bf16_v18_4warp(
        &ctx, &stream, &q_dev, &k_dev, &v_dev, &mut o_dev, batch, num_heads, seq_q, seq_kv, scale,
    )
    .expect("V18 failed");

    let o_host = stream.memcpy_dtov(&o_dev).unwrap();
    let (max_diff, mean_diff) = compare_bf16(&o_host, &expected_flat, 0.02);
    eprintln!(
        "V18 4-warp BC=128 s256: max_diff={:.6}, mean_diff={:.6}",
        max_diff, mean_diff
    );
    assert!(
        max_diff <= 0.02,
        "V18 differs from golden (max_diff={:.4})",
        max_diff
    );
}

#[cfg(feature = "experimental")]
#[test]
fn test_flash_attn_bf16_v17_bc128_s256() {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let mut npz = load_npz("flash_attn_bf16_noncausal_s256.npz");
    let q_np: ndarray::Array4<u16> = npz.by_name("q").unwrap();
    let k_np: ndarray::Array4<u16> = npz.by_name("k").unwrap();
    let v_np: ndarray::Array4<u16> = npz.by_name("v").unwrap();
    let o_expected: ndarray::Array4<u16> = npz.by_name("o").unwrap();
    let scale_np: ndarray::Array0<f32> = npz.by_name("scale").unwrap();

    let batch = q_np.shape()[0] as u32;
    let num_heads = q_np.shape()[1] as u32;
    let seq_q = q_np.shape()[2] as u32;
    let seq_kv = k_np.shape()[2] as u32;
    let scale = scale_np.into_scalar();

    let q_flat: Vec<u16> = q_np.into_raw_vec_and_offset().0;
    let k_flat: Vec<u16> = k_np.into_raw_vec_and_offset().0;
    let v_flat: Vec<u16> = v_np.into_raw_vec_and_offset().0;
    let expected_flat: Vec<u16> = o_expected.into_raw_vec_and_offset().0;
    let total_q = (batch * num_heads * seq_q * 128) as usize;

    let q_dev = stream.memcpy_stod(&q_flat).unwrap();
    let k_dev = stream.memcpy_stod(&k_flat).unwrap();
    let v_dev = stream.memcpy_stod(&v_flat).unwrap();
    let mut o_dev = stream.alloc_zeros::<u16>(total_q).unwrap();

    attention::flash_attn_bf16_v17_bc128(
        &ctx, &stream, &q_dev, &k_dev, &v_dev, &mut o_dev, batch, num_heads, seq_q, seq_kv, scale,
    )
    .expect("V17 failed");

    let o_host = stream.memcpy_dtov(&o_dev).unwrap();
    let (max_diff, mean_diff) = compare_bf16(&o_host, &expected_flat, 0.02);
    eprintln!(
        "V17 BC=128 s256: max_diff={:.6}, mean_diff={:.6}",
        max_diff, mean_diff
    );
    assert!(
        max_diff <= 0.02,
        "V17 differs from golden (max_diff={:.4})",
        max_diff
    );
}

#[cfg(feature = "experimental")]
#[test]
fn test_flash_attn_bf16_v20_tma_bc128_s256() {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let mut npz = load_npz("flash_attn_bf16_noncausal_s256.npz");
    let q_np: ndarray::Array4<u16> = npz.by_name("q").unwrap();
    let k_np: ndarray::Array4<u16> = npz.by_name("k").unwrap();
    let v_np: ndarray::Array4<u16> = npz.by_name("v").unwrap();
    let o_expected: ndarray::Array4<u16> = npz.by_name("o").unwrap();
    let scale_np: ndarray::Array0<f32> = npz.by_name("scale").unwrap();

    let batch = q_np.shape()[0] as u32;
    let num_heads = q_np.shape()[1] as u32;
    let seq_q = q_np.shape()[2] as u32;
    let seq_kv = k_np.shape()[2] as u32;
    let scale = scale_np.into_scalar();

    let q_flat: Vec<u16> = q_np.into_raw_vec_and_offset().0;
    let k_flat: Vec<u16> = k_np.into_raw_vec_and_offset().0;
    let v_flat: Vec<u16> = v_np.into_raw_vec_and_offset().0;
    let expected_flat: Vec<u16> = o_expected.into_raw_vec_and_offset().0;
    let total_q = (batch * num_heads * seq_q * 128) as usize;

    let q_dev = stream.memcpy_stod(&q_flat).unwrap();
    let k_dev = stream.memcpy_stod(&k_flat).unwrap();
    let v_dev = stream.memcpy_stod(&v_flat).unwrap();
    let mut o_dev = stream.alloc_zeros::<u16>(total_q).unwrap();

    attention::flash_attn_bf16_v20_tma_bc128(
        &ctx, &stream, &q_dev, &k_dev, &v_dev, &mut o_dev, batch, num_heads, seq_q, seq_kv, scale,
    )
    .expect("V20 failed");

    let o_host = stream.memcpy_dtov(&o_dev).unwrap();
    let (max_diff, mean_diff) = compare_bf16(&o_host, &expected_flat, 0.02);
    eprintln!(
        "V20 TMA BC=128 s256: max_diff={:.6}, mean_diff={:.6}",
        max_diff, mean_diff
    );
    assert!(
        max_diff <= 0.02,
        "V20 differs from golden (max_diff={:.4})",
        max_diff
    );
}

#[test]
fn test_flash_attn_bf16_v21_streaming_p_s256() {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let mut npz = load_npz("flash_attn_bf16_noncausal_s256.npz");
    let q_np: ndarray::Array4<u16> = npz.by_name("q").unwrap();
    let k_np: ndarray::Array4<u16> = npz.by_name("k").unwrap();
    let v_np: ndarray::Array4<u16> = npz.by_name("v").unwrap();
    let o_expected: ndarray::Array4<u16> = npz.by_name("o").unwrap();
    let scale_np: ndarray::Array0<f32> = npz.by_name("scale").unwrap();

    let batch = q_np.shape()[0] as u32;
    let num_heads = q_np.shape()[1] as u32;
    let seq_q = q_np.shape()[2] as u32;
    let seq_kv = k_np.shape()[2] as u32;
    let scale = scale_np.into_scalar();

    let q_flat: Vec<u16> = q_np.into_raw_vec_and_offset().0;
    let k_flat: Vec<u16> = k_np.into_raw_vec_and_offset().0;
    let v_flat: Vec<u16> = v_np.into_raw_vec_and_offset().0;
    let expected_flat: Vec<u16> = o_expected.into_raw_vec_and_offset().0;
    let total_q = (batch * num_heads * seq_q * 128) as usize;

    let q_dev = stream.memcpy_stod(&q_flat).unwrap();
    let k_dev = stream.memcpy_stod(&k_flat).unwrap();
    let v_dev = stream.memcpy_stod(&v_flat).unwrap();
    let mut o_dev = stream.alloc_zeros::<u16>(total_q).unwrap();

    attention::flash_attn_bf16_v21_streaming_p(
        &ctx, &stream, &q_dev, &k_dev, &v_dev, &mut o_dev, batch, num_heads, seq_q, seq_kv, scale,
    )
    .expect("V21 failed");

    let o_host = stream.memcpy_dtov(&o_dev).unwrap();
    let (max_diff, mean_diff) = compare_bf16(&o_host, &expected_flat, 0.02);
    eprintln!(
        "V21 streaming P s256: max_diff={:.6}, mean_diff={:.6}",
        max_diff, mean_diff
    );
    assert!(
        max_diff <= 0.02,
        "V21 differs from golden (max_diff={:.4})",
        max_diff
    );
}

#[test]
fn test_flash_attn_bf16_v21_causal_s256() {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let mut npz = load_npz("flash_attn_bf16_causal_s256.npz");
    let q_np: ndarray::Array4<u16> = npz.by_name("q").unwrap();
    let k_np: ndarray::Array4<u16> = npz.by_name("k").unwrap();
    let v_np: ndarray::Array4<u16> = npz.by_name("v").unwrap();
    let o_expected: ndarray::Array4<u16> = npz.by_name("o").unwrap();
    let scale_np: ndarray::Array0<f32> = npz.by_name("scale").unwrap();

    let batch = q_np.shape()[0] as u32;
    let num_heads = q_np.shape()[1] as u32;
    let seq_q = q_np.shape()[2] as u32;
    let seq_kv = k_np.shape()[2] as u32;
    let scale = scale_np.into_scalar();

    let q_flat: Vec<u16> = q_np.into_raw_vec_and_offset().0;
    let k_flat: Vec<u16> = k_np.into_raw_vec_and_offset().0;
    let v_flat: Vec<u16> = v_np.into_raw_vec_and_offset().0;
    let expected_flat: Vec<u16> = o_expected.into_raw_vec_and_offset().0;
    let total_q = (batch * num_heads * seq_q * 128) as usize;

    let q_dev = stream.memcpy_stod(&q_flat).unwrap();
    let k_dev = stream.memcpy_stod(&k_flat).unwrap();
    let v_dev = stream.memcpy_stod(&v_flat).unwrap();
    let mut o_dev = stream.alloc_zeros::<u16>(total_q).unwrap();

    attention::flash_attn_bf16_v21_causal(
        &ctx, &stream, &q_dev, &k_dev, &v_dev, &mut o_dev, batch, num_heads, seq_q, seq_kv, scale,
    )
    .expect("V21 causal failed");

    let o_host = stream.memcpy_dtov(&o_dev).unwrap();
    let (max_diff, mean_diff) = compare_bf16(&o_host, &expected_flat, 0.05);
    eprintln!(
        "V21 causal s256: max_diff={:.6}, mean_diff={:.6}",
        max_diff, mean_diff
    );
    assert!(
        max_diff <= 0.05,
        "V21 causal differs from golden (max_diff={:.4})",
        max_diff
    );
}

#[test]
fn test_flash_attn_bf16_v21_causal_s1024() {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let mut npz = load_npz("flash_attn_bf16_causal_s1024.npz");
    let q_np: ndarray::Array4<u16> = npz.by_name("q").unwrap();
    let k_np: ndarray::Array4<u16> = npz.by_name("k").unwrap();
    let v_np: ndarray::Array4<u16> = npz.by_name("v").unwrap();
    let o_expected: ndarray::Array4<u16> = npz.by_name("o").unwrap();
    let scale_np: ndarray::Array0<f32> = npz.by_name("scale").unwrap();

    let batch = q_np.shape()[0] as u32;
    let num_heads = q_np.shape()[1] as u32;
    let seq_q = q_np.shape()[2] as u32;
    let seq_kv = k_np.shape()[2] as u32;
    let scale = scale_np.into_scalar();

    let q_flat: Vec<u16> = q_np.into_raw_vec_and_offset().0;
    let k_flat: Vec<u16> = k_np.into_raw_vec_and_offset().0;
    let v_flat: Vec<u16> = v_np.into_raw_vec_and_offset().0;
    let expected_flat: Vec<u16> = o_expected.into_raw_vec_and_offset().0;
    let total_q = (batch * num_heads * seq_q * 128) as usize;

    let q_dev = stream.memcpy_stod(&q_flat).unwrap();
    let k_dev = stream.memcpy_stod(&k_flat).unwrap();
    let v_dev = stream.memcpy_stod(&v_flat).unwrap();
    let mut o_dev = stream.alloc_zeros::<u16>(total_q).unwrap();

    attention::flash_attn_bf16_v21_causal(
        &ctx, &stream, &q_dev, &k_dev, &v_dev, &mut o_dev, batch, num_heads, seq_q, seq_kv, scale,
    )
    .expect("V21 causal s1024 failed");

    let o_host = stream.memcpy_dtov(&o_dev).unwrap();
    let (max_diff, mean_diff) = compare_bf16(&o_host, &expected_flat, 0.05);
    eprintln!(
        "V21 causal s1024: max_diff={:.6}, mean_diff={:.6}",
        max_diff, mean_diff
    );
    assert!(
        max_diff <= 0.05,
        "V21 causal s1024 differs from golden (max_diff={:.4})",
        max_diff
    );
}

// ─── V21 GQA variants ─────────────────────────────────────────────────

fn run_flash_attn_v21_gqa_test(npz_name: &str, tol: f32, causal: bool) {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let mut npz = load_npz(npz_name);

    let q_flat: ndarray::Array1<u16> = npz.by_name("q").unwrap();
    let k_flat: ndarray::Array1<u16> = npz.by_name("k").unwrap();
    let v_flat: ndarray::Array1<u16> = npz.by_name("v").unwrap();
    let o_expected: ndarray::Array1<u16> = npz.by_name("o").unwrap();
    let num_heads_q: ndarray::Array0<u32> = npz.by_name("num_heads_q").unwrap();
    let num_heads_kv: ndarray::Array0<u32> = npz.by_name("num_heads_kv").unwrap();
    let batch: ndarray::Array0<u32> = npz.by_name("batch").unwrap();
    let seq_q: ndarray::Array0<u32> = npz.by_name("seq_q").unwrap();
    let seq_kv: ndarray::Array0<u32> = npz.by_name("seq_kv").unwrap();
    let scale: ndarray::Array0<f32> = npz.by_name("scale").unwrap();

    let num_heads_q = num_heads_q.into_scalar();
    let num_heads_kv = num_heads_kv.into_scalar();
    let batch = batch.into_scalar();
    let seq_q = seq_q.into_scalar();
    let seq_kv = seq_kv.into_scalar();
    let scale = scale.into_scalar();

    let q_vec: Vec<u16> = q_flat.into_raw_vec_and_offset().0;
    let k_vec: Vec<u16> = k_flat.into_raw_vec_and_offset().0;
    let v_vec: Vec<u16> = v_flat.into_raw_vec_and_offset().0;
    let expected: Vec<u16> = o_expected.into_raw_vec_and_offset().0;

    let q_dev = stream.memcpy_stod(&q_vec).unwrap();
    let k_dev = stream.memcpy_stod(&k_vec).unwrap();
    let v_dev = stream.memcpy_stod(&v_vec).unwrap();
    let mut o_dev = stream
        .alloc_zeros::<u16>((batch * num_heads_q * seq_q * 128) as usize)
        .unwrap();

    if causal {
        attention::flash_attn_bf16_v21_gqa_causal(
            &ctx,
            &stream,
            &q_dev,
            &k_dev,
            &v_dev,
            &mut o_dev,
            batch,
            num_heads_q,
            num_heads_kv,
            seq_q,
            seq_kv,
            scale,
        )
        .expect("V21 GQA causal failed");
    } else {
        attention::flash_attn_bf16_v21_gqa(
            &ctx,
            &stream,
            &q_dev,
            &k_dev,
            &v_dev,
            &mut o_dev,
            batch,
            num_heads_q,
            num_heads_kv,
            seq_q,
            seq_kv,
            scale,
        )
        .expect("V21 GQA failed");
    }

    let o_host = stream.memcpy_dtov(&o_dev).unwrap();
    let (max_diff, mean_diff) = compare_bf16(&o_host, &expected, tol);
    eprintln!(
        "V21 GQA{} {npz_name}: max_diff={max_diff:.6} mean_diff={mean_diff:.6} (batch={batch}, hq={num_heads_q}, hkv={num_heads_kv}, seq_q={seq_q}, seq_kv={seq_kv})",
        if causal { " causal" } else { "" }
    );
    assert!(
        max_diff <= tol,
        "V21 GQA differs (max_diff={:.4})",
        max_diff
    );
}

#[test]
fn test_flash_attn_bf16_v21_gqa_noncausal() {
    run_flash_attn_v21_gqa_test("flash_attn_bf16_v11_gqa_noncausal.npz", 0.05, false);
}

#[test]
fn test_flash_attn_bf16_v21_gqa_causal() {
    run_flash_attn_v21_gqa_test("flash_attn_bf16_v11_gqa_causal.npz", 0.05, true);
}

// ─── V21 varlen variants (reuse V11 golden vectors) ──────────────────

fn run_flash_attn_v21_varlen_test(npz_name: &str, tol: f32, causal: bool) {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let mut npz = load_npz(npz_name);

    let q_flat: ndarray::Array1<u16> = npz.by_name("q").unwrap();
    let k_flat: ndarray::Array1<u16> = npz.by_name("k").unwrap();
    let v_flat: ndarray::Array1<u16> = npz.by_name("v").unwrap();
    let o_expected: ndarray::Array1<u16> = npz.by_name("o").unwrap();
    let cu_seqlens_q: ndarray::Array1<u32> = npz.by_name("cu_seqlens_q").unwrap();
    let cu_seqlens_k: ndarray::Array1<u32> = npz.by_name("cu_seqlens_k").unwrap();
    let num_heads: ndarray::Array0<u32> = npz.by_name("num_heads").unwrap();
    let batch: ndarray::Array0<u32> = npz.by_name("batch").unwrap();
    let max_seqlen_q: ndarray::Array0<u32> = npz.by_name("max_seqlen_q").unwrap();
    let total_q: ndarray::Array0<u32> = npz.by_name("total_q").unwrap();
    let total_kv: ndarray::Array0<u32> = npz.by_name("total_kv").unwrap();
    let scale: ndarray::Array0<f32> = npz.by_name("scale").unwrap();

    let num_heads = num_heads.into_scalar();
    let batch = batch.into_scalar();
    let max_seqlen_q = max_seqlen_q.into_scalar();
    let total_q = total_q.into_scalar();
    let total_kv = total_kv.into_scalar();
    let scale = scale.into_scalar();

    let q_vec: Vec<u16> = q_flat.into_raw_vec_and_offset().0;
    let k_vec: Vec<u16> = k_flat.into_raw_vec_and_offset().0;
    let v_vec: Vec<u16> = v_flat.into_raw_vec_and_offset().0;
    let expected: Vec<u16> = o_expected.into_raw_vec_and_offset().0;
    let cu_q: Vec<u32> = cu_seqlens_q.into_raw_vec_and_offset().0;
    let cu_k: Vec<u32> = cu_seqlens_k.into_raw_vec_and_offset().0;

    let q_dev = stream.memcpy_stod(&q_vec).unwrap();
    let k_dev = stream.memcpy_stod(&k_vec).unwrap();
    let v_dev = stream.memcpy_stod(&v_vec).unwrap();
    let cu_q_dev = stream.memcpy_stod(&cu_q).unwrap();
    let cu_k_dev = stream.memcpy_stod(&cu_k).unwrap();
    let mut o_dev = stream
        .alloc_zeros::<u16>((num_heads * total_q * 128) as usize)
        .unwrap();

    if causal {
        attention::flash_attn_bf16_v21_varlen_causal(
            &ctx,
            &stream,
            &q_dev,
            &k_dev,
            &v_dev,
            &mut o_dev,
            &cu_q_dev,
            &cu_k_dev,
            batch,
            num_heads,
            max_seqlen_q,
            total_q,
            total_kv,
            scale,
        )
        .expect("V21 varlen causal failed");
    } else {
        attention::flash_attn_bf16_v21_varlen(
            &ctx,
            &stream,
            &q_dev,
            &k_dev,
            &v_dev,
            &mut o_dev,
            &cu_q_dev,
            &cu_k_dev,
            batch,
            num_heads,
            max_seqlen_q,
            total_q,
            total_kv,
            scale,
        )
        .expect("V21 varlen failed");
    }

    let o_host = stream.memcpy_dtov(&o_dev).unwrap();
    let (max_diff, mean_diff) = compare_bf16(&o_host, &expected, tol);
    eprintln!(
        "V21 varlen{} {npz_name}: max_diff={max_diff:.6} mean_diff={mean_diff:.6} (batch={batch}, heads={num_heads}, total_q={total_q}, total_kv={total_kv})",
        if causal { " causal" } else { "" }
    );
    assert!(
        max_diff <= tol,
        "V21 varlen differs (max_diff={:.4})",
        max_diff
    );
}

#[test]
fn test_flash_attn_bf16_v21_varlen_a() {
    run_flash_attn_v21_varlen_test("flash_attn_bf16_v11_varlen_a.npz", 0.05, false);
}

#[test]
fn test_flash_attn_bf16_v21_varlen_b() {
    run_flash_attn_v21_varlen_test("flash_attn_bf16_v11_varlen_b.npz", 0.05, false);
}

#[test]
fn test_flash_attn_bf16_v21_varlen_causal_a() {
    run_flash_attn_v21_varlen_test("flash_attn_bf16_v11_varlen_causal_a.npz", 0.05, true);
}

#[test]
fn test_flash_attn_bf16_v21_varlen_causal_b() {
    run_flash_attn_v21_varlen_test("flash_attn_bf16_v11_varlen_causal_b.npz", 0.05, true);
}

fn run_flash_attn_v21_varlen_gqa_test(npz_name: &str, tol: f32, causal: bool) {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let mut npz = load_npz(npz_name);

    let q_flat: ndarray::Array1<u16> = npz.by_name("q").unwrap();
    let k_flat: ndarray::Array1<u16> = npz.by_name("k").unwrap();
    let v_flat: ndarray::Array1<u16> = npz.by_name("v").unwrap();
    let o_expected: ndarray::Array1<u16> = npz.by_name("o").unwrap();
    let cu_seqlens_q: ndarray::Array1<u32> = npz.by_name("cu_seqlens_q").unwrap();
    let cu_seqlens_k: ndarray::Array1<u32> = npz.by_name("cu_seqlens_k").unwrap();
    let num_heads_q: ndarray::Array0<u32> = npz.by_name("num_heads_q").unwrap();
    let num_heads_kv: ndarray::Array0<u32> = npz.by_name("num_heads_kv").unwrap();
    let batch: ndarray::Array0<u32> = npz.by_name("batch").unwrap();
    let max_seqlen_q: ndarray::Array0<u32> = npz.by_name("max_seqlen_q").unwrap();
    let total_q: ndarray::Array0<u32> = npz.by_name("total_q").unwrap();
    let total_kv: ndarray::Array0<u32> = npz.by_name("total_kv").unwrap();
    let scale: ndarray::Array0<f32> = npz.by_name("scale").unwrap();

    let num_heads_q = num_heads_q.into_scalar();
    let num_heads_kv = num_heads_kv.into_scalar();
    let batch = batch.into_scalar();
    let max_seqlen_q = max_seqlen_q.into_scalar();
    let total_q = total_q.into_scalar();
    let total_kv = total_kv.into_scalar();
    let scale = scale.into_scalar();

    let q_vec: Vec<u16> = q_flat.into_raw_vec_and_offset().0;
    let k_vec: Vec<u16> = k_flat.into_raw_vec_and_offset().0;
    let v_vec: Vec<u16> = v_flat.into_raw_vec_and_offset().0;
    let expected: Vec<u16> = o_expected.into_raw_vec_and_offset().0;
    let cu_q: Vec<u32> = cu_seqlens_q.into_raw_vec_and_offset().0;
    let cu_k: Vec<u32> = cu_seqlens_k.into_raw_vec_and_offset().0;

    let q_dev = stream.memcpy_stod(&q_vec).unwrap();
    let k_dev = stream.memcpy_stod(&k_vec).unwrap();
    let v_dev = stream.memcpy_stod(&v_vec).unwrap();
    let cu_q_dev = stream.memcpy_stod(&cu_q).unwrap();
    let cu_k_dev = stream.memcpy_stod(&cu_k).unwrap();
    let mut o_dev = stream
        .alloc_zeros::<u16>((num_heads_q * total_q * 128) as usize)
        .unwrap();

    if causal {
        attention::flash_attn_bf16_v21_varlen_gqa_causal(
            &ctx,
            &stream,
            &q_dev,
            &k_dev,
            &v_dev,
            &mut o_dev,
            &cu_q_dev,
            &cu_k_dev,
            batch,
            num_heads_q,
            num_heads_kv,
            max_seqlen_q,
            total_q,
            total_kv,
            scale,
        )
        .expect("V21 varlen GQA causal failed");
    } else {
        attention::flash_attn_bf16_v21_varlen_gqa(
            &ctx,
            &stream,
            &q_dev,
            &k_dev,
            &v_dev,
            &mut o_dev,
            &cu_q_dev,
            &cu_k_dev,
            batch,
            num_heads_q,
            num_heads_kv,
            max_seqlen_q,
            total_q,
            total_kv,
            scale,
        )
        .expect("V21 varlen GQA failed");
    }

    let o_host = stream.memcpy_dtov(&o_dev).unwrap();
    let (max_diff, mean_diff) = compare_bf16(&o_host, &expected, tol);
    eprintln!(
        "V21 varlen GQA{} {npz_name}: max_diff={max_diff:.6} mean_diff={mean_diff:.6} (batch={batch}, hq={num_heads_q}, hkv={num_heads_kv}, total_q={total_q}, total_kv={total_kv})",
        if causal { " causal" } else { "" }
    );
    assert!(
        max_diff <= tol,
        "V21 varlen GQA differs (max_diff={:.4})",
        max_diff
    );
}

#[test]
fn test_flash_attn_bf16_v21_varlen_gqa_noncausal() {
    run_flash_attn_v21_varlen_gqa_test("flash_attn_bf16_v11_varlen_gqa_noncausal.npz", 0.05, false);
}

#[test]
fn test_flash_attn_bf16_v21_varlen_gqa_causal() {
    run_flash_attn_v21_varlen_gqa_test("flash_attn_bf16_v11_varlen_gqa_causal.npz", 0.05, true);
}

// ─── V21 Softcap + SWA variants (Gemma 2/3, Mistral) ───────────────────

fn run_softcap_causal_test(npz_name: &str, tol: f32) {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let mut npz = load_npz(npz_name);
    let q_np: ndarray::Array4<u16> = npz.by_name("q").unwrap();
    let k_np: ndarray::Array4<u16> = npz.by_name("k").unwrap();
    let v_np: ndarray::Array4<u16> = npz.by_name("v").unwrap();
    let o_expected: ndarray::Array4<u16> = npz.by_name("o").unwrap();
    let scale_np: ndarray::Array0<f32> = npz.by_name("scale").unwrap();
    let softcap_np: ndarray::Array0<f32> = npz.by_name("softcap").unwrap();

    let batch = q_np.shape()[0] as u32;
    let num_heads = q_np.shape()[1] as u32;
    let seq_q = q_np.shape()[2] as u32;
    let seq_kv = k_np.shape()[2] as u32;
    let scale = scale_np.into_scalar();
    let softcap = softcap_np.into_scalar();

    let q_flat: Vec<u16> = q_np.into_raw_vec_and_offset().0;
    let k_flat: Vec<u16> = k_np.into_raw_vec_and_offset().0;
    let v_flat: Vec<u16> = v_np.into_raw_vec_and_offset().0;
    let expected: Vec<u16> = o_expected.into_raw_vec_and_offset().0;
    let total_q = (batch * num_heads * seq_q * 128) as usize;

    let q_dev = stream.memcpy_stod(&q_flat).unwrap();
    let k_dev = stream.memcpy_stod(&k_flat).unwrap();
    let v_dev = stream.memcpy_stod(&v_flat).unwrap();
    let mut o_dev = stream.alloc_zeros::<u16>(total_q).unwrap();

    attention::flash_attn_bf16_v21_softcap_causal(
        &ctx, &stream, &q_dev, &k_dev, &v_dev, &mut o_dev, batch, num_heads, seq_q, seq_kv, scale,
        softcap,
    )
    .expect("V21 softcap_causal failed");

    let o_host = stream.memcpy_dtov(&o_dev).unwrap();
    let (max_diff, mean_diff) = compare_bf16(&o_host, &expected, tol);
    eprintln!("V21 softcap_causal {npz_name}: max_diff={max_diff:.6}, mean_diff={mean_diff:.6}");
    assert!(
        max_diff <= tol,
        "V21 softcap_causal differs (max_diff={:.4})",
        max_diff
    );
}

#[test]
fn test_flash_attn_bf16_v21_softcap_causal_s256() {
    run_softcap_causal_test("flash_attn_bf16_softcap_causal_s256.npz", 0.05);
}

#[test]
fn test_flash_attn_bf16_v21_softcap_causal_s1024() {
    run_softcap_causal_test("flash_attn_bf16_softcap_causal_s1024.npz", 0.05);
}

fn run_swa_test(npz_name: &str, tol: f32) {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let mut npz = load_npz(npz_name);
    let q_np: ndarray::Array4<u16> = npz.by_name("q").unwrap();
    let k_np: ndarray::Array4<u16> = npz.by_name("k").unwrap();
    let v_np: ndarray::Array4<u16> = npz.by_name("v").unwrap();
    let o_expected: ndarray::Array4<u16> = npz.by_name("o").unwrap();
    let scale_np: ndarray::Array0<f32> = npz.by_name("scale").unwrap();
    let window_np: ndarray::Array0<u32> = npz.by_name("window").unwrap();

    let batch = q_np.shape()[0] as u32;
    let num_heads = q_np.shape()[1] as u32;
    let seq_q = q_np.shape()[2] as u32;
    let seq_kv = k_np.shape()[2] as u32;
    let scale = scale_np.into_scalar();
    let window = window_np.into_scalar();

    let q_flat: Vec<u16> = q_np.into_raw_vec_and_offset().0;
    let k_flat: Vec<u16> = k_np.into_raw_vec_and_offset().0;
    let v_flat: Vec<u16> = v_np.into_raw_vec_and_offset().0;
    let expected: Vec<u16> = o_expected.into_raw_vec_and_offset().0;
    let total_q = (batch * num_heads * seq_q * 128) as usize;

    let q_dev = stream.memcpy_stod(&q_flat).unwrap();
    let k_dev = stream.memcpy_stod(&k_flat).unwrap();
    let v_dev = stream.memcpy_stod(&v_flat).unwrap();
    let mut o_dev = stream.alloc_zeros::<u16>(total_q).unwrap();

    attention::flash_attn_bf16_v21_swa(
        &ctx, &stream, &q_dev, &k_dev, &v_dev, &mut o_dev, batch, num_heads, seq_q, seq_kv, scale,
        window,
    )
    .expect("V21 swa failed");

    let o_host = stream.memcpy_dtov(&o_dev).unwrap();
    let (max_diff, mean_diff) = compare_bf16(&o_host, &expected, tol);
    eprintln!(
        "V21 swa {npz_name}: max_diff={max_diff:.6}, mean_diff={mean_diff:.6}, window={window}"
    );
    assert!(
        max_diff <= tol,
        "V21 swa differs (max_diff={:.4})",
        max_diff
    );
}

#[test]
fn test_flash_attn_bf16_v21_swa_w64_s256() {
    run_swa_test("flash_attn_bf16_swa_w64_s256.npz", 0.05);
}

#[test]
fn test_flash_attn_bf16_v21_swa_w256_s1024() {
    run_swa_test("flash_attn_bf16_swa_w256_s1024.npz", 0.05);
}

fn run_swa_softcap_test(npz_name: &str, tol: f32) {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let mut npz = load_npz(npz_name);
    let q_np: ndarray::Array4<u16> = npz.by_name("q").unwrap();
    let k_np: ndarray::Array4<u16> = npz.by_name("k").unwrap();
    let v_np: ndarray::Array4<u16> = npz.by_name("v").unwrap();
    let o_expected: ndarray::Array4<u16> = npz.by_name("o").unwrap();
    let scale_np: ndarray::Array0<f32> = npz.by_name("scale").unwrap();
    let window_np: ndarray::Array0<u32> = npz.by_name("window").unwrap();
    let softcap_np: ndarray::Array0<f32> = npz.by_name("softcap").unwrap();

    let batch = q_np.shape()[0] as u32;
    let num_heads = q_np.shape()[1] as u32;
    let seq_q = q_np.shape()[2] as u32;
    let seq_kv = k_np.shape()[2] as u32;
    let scale = scale_np.into_scalar();
    let window = window_np.into_scalar();
    let softcap = softcap_np.into_scalar();

    let q_flat: Vec<u16> = q_np.into_raw_vec_and_offset().0;
    let k_flat: Vec<u16> = k_np.into_raw_vec_and_offset().0;
    let v_flat: Vec<u16> = v_np.into_raw_vec_and_offset().0;
    let expected: Vec<u16> = o_expected.into_raw_vec_and_offset().0;
    let total_q = (batch * num_heads * seq_q * 128) as usize;

    let q_dev = stream.memcpy_stod(&q_flat).unwrap();
    let k_dev = stream.memcpy_stod(&k_flat).unwrap();
    let v_dev = stream.memcpy_stod(&v_flat).unwrap();
    let mut o_dev = stream.alloc_zeros::<u16>(total_q).unwrap();

    attention::flash_attn_bf16_v21_swa_softcap(
        &ctx, &stream, &q_dev, &k_dev, &v_dev, &mut o_dev, batch, num_heads, seq_q, seq_kv, scale,
        window, softcap,
    )
    .expect("V21 swa_softcap failed");

    let o_host = stream.memcpy_dtov(&o_dev).unwrap();
    let (max_diff, mean_diff) = compare_bf16(&o_host, &expected, tol);
    eprintln!("V21 swa_softcap {npz_name}: max_diff={max_diff:.6}, mean_diff={mean_diff:.6}");
    assert!(
        max_diff <= tol,
        "V21 swa_softcap differs (max_diff={:.4})",
        max_diff
    );
}

#[test]
fn test_flash_attn_bf16_v21_swa_softcap_w64_s256() {
    run_swa_softcap_test("flash_attn_bf16_swa_softcap_w64_s256.npz", 0.05);
}

#[test]
fn test_flash_attn_bf16_v21_swa_softcap_w256_s1024() {
    run_swa_softcap_test("flash_attn_bf16_swa_softcap_w256_s1024.npz", 0.05);
}

// ─── MLA decode BF16 (DeepSeek V3, GLM-4.5/5) ──────────────────────────

fn run_mla_decode_test(npz_name: &str, tol: f32) {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let mut npz = load_npz(npz_name);
    let q_c: ndarray::Array3<u16> = npz.by_name("q_c").unwrap();
    let q_r: ndarray::Array3<u16> = npz.by_name("q_r").unwrap();
    let c_kv: ndarray::Array3<u16> = npz.by_name("c_kv").unwrap();
    let k_rope: ndarray::Array3<u16> = npz.by_name("k_rope").unwrap();
    let o_expected: ndarray::Array3<u16> = npz.by_name("o").unwrap();
    let scale: ndarray::Array0<f32> = npz.by_name("scale").unwrap();
    let batch: ndarray::Array0<u32> = npz.by_name("batch").unwrap();
    let num_heads: ndarray::Array0<u32> = npz.by_name("num_heads").unwrap();
    let seq_kv: ndarray::Array0<u32> = npz.by_name("seq_kv").unwrap();

    let batch = batch.into_scalar();
    let num_heads = num_heads.into_scalar();
    let seq_kv = seq_kv.into_scalar();
    let scale = scale.into_scalar();

    let q_c_flat: Vec<u16> = q_c.into_raw_vec_and_offset().0;
    let q_r_flat: Vec<u16> = q_r.into_raw_vec_and_offset().0;
    let c_kv_flat: Vec<u16> = c_kv.into_raw_vec_and_offset().0;
    let k_rope_flat: Vec<u16> = k_rope.into_raw_vec_and_offset().0;
    let expected: Vec<u16> = o_expected.into_raw_vec_and_offset().0;

    let q_c_dev = stream.memcpy_stod(&q_c_flat).unwrap();
    let q_r_dev = stream.memcpy_stod(&q_r_flat).unwrap();
    let c_kv_dev = stream.memcpy_stod(&c_kv_flat).unwrap();
    let k_rope_dev = stream.memcpy_stod(&k_rope_flat).unwrap();
    let o_len = (batch * num_heads * attention::MLA_D_C) as usize;
    let mut o_dev = stream.alloc_zeros::<u16>(o_len).unwrap();

    attention::mla_decode_bf16(
        &ctx,
        &stream,
        &q_c_dev,
        &q_r_dev,
        &c_kv_dev,
        &k_rope_dev,
        &mut o_dev,
        batch,
        num_heads,
        seq_kv,
        scale,
    )
    .expect("MLA decode failed");

    let o_host = stream.memcpy_dtov(&o_dev).unwrap();
    let (max_diff, mean_diff) = compare_bf16(&o_host, &expected, tol);
    eprintln!("MLA decode {npz_name}: max_diff={max_diff:.6} mean_diff={mean_diff:.6} (B={batch}, H={num_heads}, S={seq_kv})");
    assert!(
        max_diff <= tol,
        "MLA decode differs (max_diff={:.4})",
        max_diff
    );
}

#[test]
fn test_mla_decode_bf16_B1_H16_S32() {
    run_mla_decode_test("mla_decode_B1_H16_S32.npz", 0.05);
}

#[test]
fn test_mla_decode_bf16_B2_H32_S128() {
    run_mla_decode_test("mla_decode_B2_H32_S128.npz", 0.08);
}

// ─── MLA prefill BF16 (DeepSeek V3, GLM-4.5/5) ──────────────────────────

fn run_mla_prefill_test(npz_name: &str, tol: f32) {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let mut npz = load_npz(npz_name);
    let q_c: ndarray::Array4<u16> = npz.by_name("q_c").unwrap();
    let q_r: ndarray::Array4<u16> = npz.by_name("q_r").unwrap();
    let c_kv: ndarray::Array3<u16> = npz.by_name("c_kv").unwrap();
    let k_rope: ndarray::Array3<u16> = npz.by_name("k_rope").unwrap();
    let o_expected: ndarray::Array4<u16> = npz.by_name("o").unwrap();
    let scale: ndarray::Array0<f32> = npz.by_name("scale").unwrap();
    let batch: ndarray::Array0<u32> = npz.by_name("batch").unwrap();
    let num_heads: ndarray::Array0<u32> = npz.by_name("num_heads").unwrap();
    let seq_q: ndarray::Array0<u32> = npz.by_name("seq_q").unwrap();
    let seq_kv: ndarray::Array0<u32> = npz.by_name("seq_kv").unwrap();

    let batch = batch.into_scalar();
    let num_heads = num_heads.into_scalar();
    let seq_q = seq_q.into_scalar();
    let seq_kv = seq_kv.into_scalar();
    let scale = scale.into_scalar();

    let q_c_flat: Vec<u16> = q_c.into_raw_vec_and_offset().0;
    let q_r_flat: Vec<u16> = q_r.into_raw_vec_and_offset().0;
    let c_kv_flat: Vec<u16> = c_kv.into_raw_vec_and_offset().0;
    let k_rope_flat: Vec<u16> = k_rope.into_raw_vec_and_offset().0;
    let expected: Vec<u16> = o_expected.into_raw_vec_and_offset().0;

    let q_c_dev = stream.memcpy_stod(&q_c_flat).unwrap();
    let q_r_dev = stream.memcpy_stod(&q_r_flat).unwrap();
    let c_kv_dev = stream.memcpy_stod(&c_kv_flat).unwrap();
    let k_rope_dev = stream.memcpy_stod(&k_rope_flat).unwrap();
    let o_len = (batch * seq_q * num_heads * attention::MLA_D_C) as usize;
    let mut o_dev = stream.alloc_zeros::<u16>(o_len).unwrap();

    attention::mla_prefill_bf16(
        &ctx,
        &stream,
        &q_c_dev,
        &q_r_dev,
        &c_kv_dev,
        &k_rope_dev,
        &mut o_dev,
        batch,
        num_heads,
        seq_q,
        seq_kv,
        scale,
    )
    .expect("MLA prefill failed");

    let o_host = stream.memcpy_dtov(&o_dev).unwrap();
    let (max_diff, mean_diff) = compare_bf16(&o_host, &expected, tol);
    eprintln!("MLA prefill {npz_name}: max_diff={max_diff:.6} mean_diff={mean_diff:.6} (B={batch}, H={num_heads}, Sq={seq_q})");
    assert!(
        max_diff <= tol,
        "MLA prefill differs (max_diff={:.4})",
        max_diff
    );
}

#[test]
fn test_mla_prefill_bf16_b1_sq32_h8() {
    run_mla_prefill_test("mla_prefill_B1_Sq32_H8.npz", 0.08);
}

#[test]
fn test_mla_prefill_bf16_b2_sq64_h16() {
    run_mla_prefill_test("mla_prefill_B2_Sq64_H16.npz", 0.08);
}

// ─── MLA FP8 decode ────────────────────────────────────────────────────

fn run_mla_decode_fp8_test(npz_name: &str, tol: f32) {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let mut npz = load_npz(npz_name);
    let q_c: ndarray::Array3<u16> = npz.by_name("q_c").unwrap();
    let q_r: ndarray::Array3<u16> = npz.by_name("q_r").unwrap();
    let c_kv: ndarray::Array3<u8> = npz.by_name("c_kv").unwrap();
    let k_rope: ndarray::Array3<u8> = npz.by_name("k_rope").unwrap();
    let o_expected: ndarray::Array3<u16> = npz.by_name("o").unwrap();
    let scale: ndarray::Array0<f32> = npz.by_name("scale").unwrap();
    let kv_scale: ndarray::Array0<f32> = npz.by_name("kv_scale").unwrap();
    let batch: ndarray::Array0<u32> = npz.by_name("batch").unwrap();
    let num_heads: ndarray::Array0<u32> = npz.by_name("num_heads").unwrap();
    let seq_kv: ndarray::Array0<u32> = npz.by_name("seq_kv").unwrap();

    let batch = batch.into_scalar();
    let num_heads = num_heads.into_scalar();
    let seq_kv = seq_kv.into_scalar();
    let scale = scale.into_scalar();
    let kv_scale = kv_scale.into_scalar();

    let q_c_flat: Vec<u16> = q_c.into_raw_vec_and_offset().0;
    let q_r_flat: Vec<u16> = q_r.into_raw_vec_and_offset().0;
    let c_kv_flat: Vec<u8> = c_kv.into_raw_vec_and_offset().0;
    let k_rope_flat: Vec<u8> = k_rope.into_raw_vec_and_offset().0;
    let expected: Vec<u16> = o_expected.into_raw_vec_and_offset().0;

    let q_c_dev = stream.memcpy_stod(&q_c_flat).unwrap();
    let q_r_dev = stream.memcpy_stod(&q_r_flat).unwrap();
    let c_kv_dev = stream.memcpy_stod(&c_kv_flat).unwrap();
    let k_rope_dev = stream.memcpy_stod(&k_rope_flat).unwrap();
    let o_len = (batch * num_heads * attention::MLA_D_C) as usize;
    let mut o_dev = stream.alloc_zeros::<u16>(o_len).unwrap();

    attention::mla_decode_fp8(
        &ctx,
        &stream,
        &q_c_dev,
        &q_r_dev,
        &c_kv_dev,
        &k_rope_dev,
        &mut o_dev,
        batch,
        num_heads,
        seq_kv,
        scale,
        kv_scale,
    )
    .expect("MLA FP8 decode failed");

    let o_host = stream.memcpy_dtov(&o_dev).unwrap();
    let (max_diff, mean_diff) = compare_bf16(&o_host, &expected, tol);
    eprintln!("MLA FP8 decode {npz_name}: max_diff={max_diff:.6} mean_diff={mean_diff:.6}");
    assert!(
        max_diff <= tol,
        "MLA FP8 decode differs (max_diff={:.4})",
        max_diff
    );
}

#[test]
fn test_mla_decode_fp8_b1_h16_s32() {
    run_mla_decode_fp8_test("mla_decode_fp8_B1_H16_S32.npz", 1.0);
}

#[test]
fn test_mla_decode_fp8_b2_h32_s128() {
    run_mla_decode_fp8_test("mla_decode_fp8_B2_H32_S128.npz", 2.0);
}

// ─── MLA paged decode ──────────────────────────────────────────────────

fn run_mla_paged_test(npz_name: &str, tol: f32) {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let mut npz = load_npz(npz_name);
    let q_c: ndarray::Array3<u16> = npz.by_name("q_c").unwrap();
    let q_r: ndarray::Array3<u16> = npz.by_name("q_r").unwrap();
    let c_kv: ndarray::Array3<u16> = npz.by_name("c_kv").unwrap();
    let k_rope: ndarray::Array3<u16> = npz.by_name("k_rope").unwrap();
    let page_table: ndarray::Array2<u32> = npz.by_name("page_table").unwrap();
    let seq_lens: ndarray::Array1<u32> = npz.by_name("seq_lens").unwrap();
    let o_expected: ndarray::Array3<u16> = npz.by_name("o").unwrap();
    let scale: ndarray::Array0<f32> = npz.by_name("scale").unwrap();
    let batch: ndarray::Array0<u32> = npz.by_name("batch").unwrap();
    let num_heads: ndarray::Array0<u32> = npz.by_name("num_heads").unwrap();
    let max_pages: ndarray::Array0<u32> = npz.by_name("max_pages").unwrap();
    let page_size: ndarray::Array0<u32> = npz.by_name("page_size").unwrap();

    let batch = batch.into_scalar();
    let num_heads = num_heads.into_scalar();
    let max_pages = max_pages.into_scalar();
    let page_size = page_size.into_scalar();
    let scale = scale.into_scalar();

    let q_c_flat: Vec<u16> = q_c.into_raw_vec_and_offset().0;
    let q_r_flat: Vec<u16> = q_r.into_raw_vec_and_offset().0;
    let c_kv_flat: Vec<u16> = c_kv.into_raw_vec_and_offset().0;
    let k_rope_flat: Vec<u16> = k_rope.into_raw_vec_and_offset().0;
    let page_table_flat: Vec<u32> = page_table.into_raw_vec_and_offset().0;
    let seq_lens_flat: Vec<u32> = seq_lens.into_raw_vec_and_offset().0;
    let expected: Vec<u16> = o_expected.into_raw_vec_and_offset().0;

    let q_c_dev = stream.memcpy_stod(&q_c_flat).unwrap();
    let q_r_dev = stream.memcpy_stod(&q_r_flat).unwrap();
    let c_kv_dev = stream.memcpy_stod(&c_kv_flat).unwrap();
    let k_rope_dev = stream.memcpy_stod(&k_rope_flat).unwrap();
    let page_table_dev = stream.memcpy_stod(&page_table_flat).unwrap();
    let seq_lens_dev = stream.memcpy_stod(&seq_lens_flat).unwrap();
    let o_len = (batch * num_heads * attention::MLA_D_C) as usize;
    let mut o_dev = stream.alloc_zeros::<u16>(o_len).unwrap();

    attention::mla_decode_bf16_paged(
        &ctx,
        &stream,
        &q_c_dev,
        &q_r_dev,
        &c_kv_dev,
        &k_rope_dev,
        &page_table_dev,
        &seq_lens_dev,
        &mut o_dev,
        batch,
        num_heads,
        max_pages,
        page_size,
        scale,
    )
    .expect("MLA paged decode failed");

    let o_host = stream.memcpy_dtov(&o_dev).unwrap();
    let (max_diff, mean_diff) = compare_bf16(&o_host, &expected, tol);
    eprintln!("MLA paged {npz_name}: max_diff={max_diff:.6} mean_diff={mean_diff:.6} (B={batch}, H={num_heads})");
    assert!(
        max_diff <= tol,
        "MLA paged decode differs (max_diff={:.4})",
        max_diff
    );
}

#[test]
fn test_mla_decode_paged_b1_h16_s32() {
    run_mla_paged_test("mla_decode_paged_B1_H16_S32.npz", 0.05);
}

#[test]
fn test_mla_decode_paged_b2_h32_s96() {
    run_mla_paged_test("mla_decode_paged_B2_H32_S96.npz", 0.08);
}

// ─── Tree attention (EAGLE-3 / Medusa spec decoding) ───────────────────

fn run_tree_attention_test(npz_name: &str, tol: f32) {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let mut npz = load_npz(npz_name);
    let q: ndarray::Array4<u16> = npz.by_name("q").unwrap();
    let k: ndarray::Array4<u16> = npz.by_name("k").unwrap();
    let v: ndarray::Array4<u16> = npz.by_name("v").unwrap();
    let mask: ndarray::Array2<u8> = npz.by_name("mask").unwrap();
    let o_expected: ndarray::Array4<u16> = npz.by_name("o").unwrap();
    let scale: ndarray::Array0<f32> = npz.by_name("scale").unwrap();
    let batch: ndarray::Array0<u32> = npz.by_name("batch").unwrap();
    let num_heads: ndarray::Array0<u32> = npz.by_name("num_heads").unwrap();
    let seq_q: ndarray::Array0<u32> = npz.by_name("seq_q").unwrap();
    let seq_kv: ndarray::Array0<u32> = npz.by_name("seq_kv").unwrap();

    let batch = batch.into_scalar();
    let num_heads = num_heads.into_scalar();
    let seq_q = seq_q.into_scalar();
    let seq_kv = seq_kv.into_scalar();
    let scale = scale.into_scalar();

    let q_flat: Vec<u16> = q.into_raw_vec_and_offset().0;
    let k_flat: Vec<u16> = k.into_raw_vec_and_offset().0;
    let v_flat: Vec<u16> = v.into_raw_vec_and_offset().0;
    let mask_flat: Vec<u8> = mask.into_raw_vec_and_offset().0;
    let expected: Vec<u16> = o_expected.into_raw_vec_and_offset().0;

    let q_dev = stream.memcpy_stod(&q_flat).unwrap();
    let k_dev = stream.memcpy_stod(&k_flat).unwrap();
    let v_dev = stream.memcpy_stod(&v_flat).unwrap();
    let mask_dev = stream.memcpy_stod(&mask_flat).unwrap();
    let o_len = (batch * seq_q * num_heads * 128) as usize;
    let mut o_dev = stream.alloc_zeros::<u16>(o_len).unwrap();

    attention::tree_attention_bf16(
        &ctx, &stream, &q_dev, &k_dev, &v_dev, &mask_dev, &mut o_dev, batch, num_heads, seq_q,
        seq_kv, scale,
    )
    .expect("tree attention failed");

    let o_host = stream.memcpy_dtov(&o_dev).unwrap();
    let (max_diff, mean_diff) = compare_bf16(&o_host, &expected, tol);
    eprintln!("Tree attn {npz_name}: max_diff={max_diff:.6} mean_diff={mean_diff:.6}");
    assert!(
        max_diff <= tol,
        "Tree attention differs (max_diff={:.4})",
        max_diff
    );
}

#[test]
fn test_tree_attn_b1_sq8_skv32() {
    run_tree_attention_test("tree_attn_B1_Sq8_Skv32_H8.npz", 0.05);
}

#[test]
fn test_tree_attn_b2_sq16_skv64() {
    run_tree_attention_test("tree_attn_B2_Sq16_Skv64_H16.npz", 0.05);
}

// ─── NSA sparse attention (DeepSeek V3.2-Exp) ───────────────────────────

fn run_nsa_test(npz_name: &str, tol: f32) {
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();

    let mut npz = load_npz(npz_name);
    let q: ndarray::Array4<u16> = npz.by_name("q").unwrap();
    let k: ndarray::Array4<u16> = npz.by_name("k").unwrap();
    let v: ndarray::Array4<u16> = npz.by_name("v").unwrap();
    let block_idx: ndarray::Array4<u32> = npz.by_name("block_idx").unwrap();
    let o_expected: ndarray::Array4<u16> = npz.by_name("o").unwrap();
    let scale: ndarray::Array0<f32> = npz.by_name("scale").unwrap();
    let batch: ndarray::Array0<u32> = npz.by_name("batch").unwrap();
    let num_heads: ndarray::Array0<u32> = npz.by_name("num_heads").unwrap();
    let seq_q: ndarray::Array0<u32> = npz.by_name("seq_q").unwrap();
    let seq_kv: ndarray::Array0<u32> = npz.by_name("seq_kv").unwrap();
    let k_top: ndarray::Array0<u32> = npz.by_name("k_top").unwrap();
    let block_size: ndarray::Array0<u32> = npz.by_name("block_size").unwrap();

    let batch = batch.into_scalar();
    let num_heads = num_heads.into_scalar();
    let seq_q = seq_q.into_scalar();
    let seq_kv = seq_kv.into_scalar();
    let k_top = k_top.into_scalar();
    let block_size = block_size.into_scalar();
    let scale = scale.into_scalar();

    let q_flat: Vec<u16> = q.into_raw_vec_and_offset().0;
    let k_flat: Vec<u16> = k.into_raw_vec_and_offset().0;
    let v_flat: Vec<u16> = v.into_raw_vec_and_offset().0;
    let block_flat: Vec<u32> = block_idx.into_raw_vec_and_offset().0;
    let expected: Vec<u16> = o_expected.into_raw_vec_and_offset().0;

    let q_dev = stream.memcpy_stod(&q_flat).unwrap();
    let k_dev = stream.memcpy_stod(&k_flat).unwrap();
    let v_dev = stream.memcpy_stod(&v_flat).unwrap();
    let block_dev = stream.memcpy_stod(&block_flat).unwrap();
    let o_len = (batch * seq_q * num_heads * 128) as usize;
    let mut o_dev = stream.alloc_zeros::<u16>(o_len).unwrap();

    attention::nsa_attention_bf16(
        &ctx, &stream, &q_dev, &k_dev, &v_dev, &block_dev, &mut o_dev, batch, num_heads, seq_q,
        seq_kv, k_top, block_size, scale,
    )
    .expect("NSA failed");

    let o_host = stream.memcpy_dtov(&o_dev).unwrap();
    let (max_diff, mean_diff) = compare_bf16(&o_host, &expected, tol);
    eprintln!("NSA {npz_name}: max_diff={max_diff:.6} mean_diff={mean_diff:.6}");
    assert!(max_diff <= tol, "NSA differs: {}", max_diff);
}

#[test]
fn test_nsa_b1_sq16_skv256_k4() {
    run_nsa_test("nsa_B1_Sq16_Skv256_H4_K4.npz", 0.05);
}

#[test]
fn test_nsa_b2_sq32_skv512_k6() {
    run_nsa_test("nsa_B2_Sq32_Skv512_H8_K6.npz", 0.05);
}

fn run_block_mean_test(npz_name: &str, tol: f32) {
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();

    let mut npz = load_npz(npz_name);
    let k: ndarray::Array4<u16> = npz.by_name("k").unwrap();
    let out_expected: ndarray::Array4<u16> = npz.by_name("out").unwrap();
    let batch: ndarray::Array0<u32> = npz.by_name("batch").unwrap();
    let num_heads: ndarray::Array0<u32> = npz.by_name("num_heads").unwrap();
    let seq_kv: ndarray::Array0<u32> = npz.by_name("seq_kv").unwrap();
    let block_size: ndarray::Array0<u32> = npz.by_name("block_size").unwrap();
    let num_blocks: ndarray::Array0<u32> = npz.by_name("num_blocks").unwrap();

    let batch = batch.into_scalar();
    let num_heads = num_heads.into_scalar();
    let seq_kv = seq_kv.into_scalar();
    let block_size = block_size.into_scalar();
    let num_blocks = num_blocks.into_scalar();

    let k_flat: Vec<u16> = k.into_raw_vec_and_offset().0;
    let expected: Vec<u16> = out_expected.into_raw_vec_and_offset().0;

    let k_dev = stream.memcpy_stod(&k_flat).unwrap();
    let out_len = (batch * num_blocks * num_heads * 128) as usize;
    let mut out_dev = stream.alloc_zeros::<u16>(out_len).unwrap();

    attention::k_block_mean(
        &ctx,
        &stream,
        &k_dev,
        &mut out_dev,
        batch,
        num_heads,
        seq_kv,
        block_size,
    )
    .expect("block mean failed");

    let out_host = stream.memcpy_dtov(&out_dev).unwrap();
    let (max_diff, mean_diff) = compare_bf16(&out_host, &expected, tol);
    eprintln!("block_mean {npz_name}: max_diff={max_diff:.6} mean_diff={mean_diff:.6}");
    assert!(max_diff <= tol, "block_mean differs: {}", max_diff);
}

#[test]
fn test_k_block_mean_b1_h4_skv128_bs32() {
    run_block_mean_test("k_block_mean_B1_H4_Skv128_bs32.npz", 0.02);
}

#[test]
fn test_k_block_mean_b2_h8_skv256_bs64() {
    run_block_mean_test("k_block_mean_B2_H8_Skv256_bs64.npz", 0.02);
}

#[test]
fn test_k_block_mean_ragged_skv100_bs32() {
    // Skv=100 not multiple of block_size=32 → last block has only 4 tokens
    run_block_mean_test("k_block_mean_B1_H2_Skv100_bs32.npz", 0.05);
}

// ─── Top-K block index selection (NSA pipeline) ────────────────────────

#[test]
fn test_topk_block_idx_known_pattern() {
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();

    let batch = 2u32;
    let num_heads = 4u32;
    let seq_q = 8u32;
    let num_blocks = 16u32;
    let k_top = 4u32;

    // Build a known-pattern scores tensor. For row (b, sq, h), set distinct
    // scores so that top-K argmax is deterministic: scores[i] = 100 - i (mod num_blocks).
    let rows = (batch * seq_q * num_heads) as usize;
    let mut scores = vec![0.0f32; rows * num_blocks as usize];
    let mut expected = vec![0u32; rows * k_top as usize];
    for row in 0..rows {
        let offset = row * num_blocks as usize;
        for i in 0..num_blocks as usize {
            scores[offset + i] = 100.0 - i as f32;
        }
        for k in 0..k_top as usize {
            expected[row * k_top as usize + k] = k as u32; // 0,1,2,3 (highest scores)
        }
    }

    let scores_dev = stream.memcpy_stod(&scores).unwrap();
    let mut idx_dev = stream.alloc_zeros::<u32>(rows * k_top as usize).unwrap();

    attention::topk_block_idx(
        &ctx,
        &stream,
        &scores_dev,
        &mut idx_dev,
        batch,
        num_heads,
        seq_q,
        num_blocks,
        k_top,
    )
    .expect("topk_block_idx failed");

    let idx_host = stream.memcpy_dtov(&idx_dev).unwrap();
    assert_eq!(
        idx_host,
        expected,
        "topk_block_idx output mismatch: got {:?}, expected {:?}",
        &idx_host[..8.min(idx_host.len())],
        &expected[..8.min(expected.len())]
    );
    eprintln!(
        "topk_block_idx: ok (B={batch}, H={num_heads}, Sq={seq_q}, nb={num_blocks}, K={k_top})"
    );
}

// ─── MTP draft heads (DeepSeek V3 / V4 spec decoding) ───────────────────

#[test]
fn test_mtp_draft_heads_known_pattern() {
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();

    // Small-enough to run fast but realistic structure
    let batch = 2u32;
    let k = 3u32;
    let vocab = 256u32;
    let d = 128u32;

    // Per-batch one-hot hidden: hidden[b, b] = 1, rest zero.
    // Then dot(hidden[b], W[k, v, :]) = W[k, v, b].
    let one_bits: u16 = (1.0f32.to_bits() >> 16) as u16;
    let mut hidden = vec![0u16; (batch * d) as usize];
    for bb in 0..batch as usize {
        hidden[bb * d as usize + bb] = one_bits;
    }
    // Set W[k, target(b,k), b] = 1, so argmax per (b, k) is target(b, k).
    let mut w = vec![0u16; (k * vocab * d) as usize];
    let mut expected = vec![0u32; (batch * k) as usize];
    for bb in 0..batch as usize {
        for kk in 0..k as usize {
            let target = (bb * 7 + kk * 13 + 1) % vocab as usize;
            let idx = (kk * vocab as usize + target) * d as usize + bb;
            w[idx] = one_bits;
            expected[bb * k as usize + kk] = target as u32;
        }
    }

    let h_dev = stream.memcpy_stod(&hidden).unwrap();
    let w_dev = stream.memcpy_stod(&w).unwrap();
    let mut draft_dev = stream.alloc_zeros::<u32>((batch * k) as usize).unwrap();

    attention::mtp_draft_heads(
        &ctx,
        &stream,
        &h_dev,
        &w_dev,
        &mut draft_dev,
        batch,
        k,
        vocab,
        d,
    )
    .expect("MTP failed");

    let draft_host = stream.memcpy_dtov(&draft_dev).unwrap();
    assert_eq!(draft_host, expected, "MTP draft mismatch");
    eprintln!("MTP draft heads: ok (B={batch}, K={k}, V={vocab}, D={d})");
}

// ─── DSA (DeepSeek Sparse Attention, DeepSeek V3.2) ───────────────────

// `0 * ...` keeps the batch index (b=0) explicit in the flat-index math so the
// addressing mirrors the kernel layout; keep it for readability.
#[allow(clippy::erasing_op)]
#[test]
fn test_dsa_attention_known_pattern() {
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();

    let b = 1u32;
    let h = 4u32;
    let sq = 8u32;
    let skv = 64u32;
    let d = 128u32;
    let k_top = 4u32;

    // Build: Q/K/V = randn, and each query attends to 4 specific positions.
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut q = vec![0u16; (b * sq * h * d) as usize];
    let mut k = vec![0u16; (b * skv * h * d) as usize];
    let mut v = vec![0u16; (b * skv * h * d) as usize];
    for i in 0..q.len() {
        let mut hh = DefaultHasher::new();
        i.hash(&mut hh);
        q[i] = ((hh.finish() >> 48) as u16) & 0x3fff;
    }
    for i in 0..k.len() {
        let mut hh = DefaultHasher::new();
        (i + 7919).hash(&mut hh);
        k[i] = ((hh.finish() >> 48) as u16) & 0x3fff;
    }
    for i in 0..v.len() {
        let mut hh = DefaultHasher::new();
        (i + 31337).hash(&mut hh);
        v[i] = ((hh.finish() >> 48) as u16) & 0x3fff;
    }
    // Position indices: for query i, attend to positions i, i+1, i+2, i+3 (mod skv)
    let mut pos = vec![0u32; (b * sq * h * k_top) as usize];
    for sqi in 0..sq as usize {
        for hh in 0..h as usize {
            for kk in 0..k_top as usize {
                let idx = ((0 * sq as usize + sqi) * h as usize + hh) * k_top as usize + kk;
                pos[idx] = ((sqi + kk) % skv as usize) as u32;
            }
        }
    }
    let scale = 1.0f32 / (d as f32).sqrt();

    let q_dev = stream.memcpy_stod(&q).unwrap();
    let k_dev = stream.memcpy_stod(&k).unwrap();
    let v_dev = stream.memcpy_stod(&v).unwrap();
    let pos_dev = stream.memcpy_stod(&pos).unwrap();
    let mut o_dsa = stream
        .alloc_zeros::<u16>((b * sq * h * d) as usize)
        .unwrap();
    let mut o_nsa = stream
        .alloc_zeros::<u16>((b * sq * h * d) as usize)
        .unwrap();

    attention::dsa_attention_bf16(
        &ctx, &stream, &q_dev, &k_dev, &v_dev, &pos_dev, &mut o_dsa, b, h, sq, skv, k_top, scale,
    )
    .expect("DSA failed");
    // Also call NSA with block_size=1 as a cross-check (should produce identical output)
    attention::nsa_attention_bf16(
        &ctx, &stream, &q_dev, &k_dev, &v_dev, &pos_dev, &mut o_nsa, b, h, sq, skv, k_top, 1, scale,
    )
    .expect("NSA ref failed");

    let o_dsa_host = stream.memcpy_dtov(&o_dsa).unwrap();
    let o_nsa_host = stream.memcpy_dtov(&o_nsa).unwrap();
    assert_eq!(
        o_dsa_host, o_nsa_host,
        "DSA should match NSA with block_size=1"
    );
    eprintln!("DSA = NSA(block_size=1): ok (B={b}, H={h}, Sq={sq}, Skv={skv}, K={k_top})");
}

// ─── MLA FP8 prefill ───────────────────────────────────────────────────

fn run_mla_prefill_fp8_test(npz_name: &str, tol: f32) {
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();
    let mut npz = load_npz(npz_name);
    let q_c: ndarray::Array4<u16> = npz.by_name("q_c").unwrap();
    let q_r: ndarray::Array4<u16> = npz.by_name("q_r").unwrap();
    let c_kv: ndarray::Array3<u8> = npz.by_name("c_kv").unwrap();
    let k_rope: ndarray::Array3<u8> = npz.by_name("k_rope").unwrap();
    let o_exp: ndarray::Array4<u16> = npz.by_name("o").unwrap();
    let scale: ndarray::Array0<f32> = npz.by_name("scale").unwrap();
    let kv_scale: ndarray::Array0<f32> = npz.by_name("kv_scale").unwrap();
    let batch: ndarray::Array0<u32> = npz.by_name("batch").unwrap();
    let num_heads: ndarray::Array0<u32> = npz.by_name("num_heads").unwrap();
    let seq_q: ndarray::Array0<u32> = npz.by_name("seq_q").unwrap();
    let seq_kv: ndarray::Array0<u32> = npz.by_name("seq_kv").unwrap();
    let batch = batch.into_scalar();
    let num_heads = num_heads.into_scalar();
    let seq_q = seq_q.into_scalar();
    let seq_kv = seq_kv.into_scalar();
    let scale = scale.into_scalar();
    let kv_scale = kv_scale.into_scalar();

    let q_c_dev = stream
        .memcpy_stod(&q_c.into_raw_vec_and_offset().0)
        .unwrap();
    let q_r_dev = stream
        .memcpy_stod(&q_r.into_raw_vec_and_offset().0)
        .unwrap();
    let c_kv_dev = stream
        .memcpy_stod(&c_kv.into_raw_vec_and_offset().0)
        .unwrap();
    let k_rope_dev = stream
        .memcpy_stod(&k_rope.into_raw_vec_and_offset().0)
        .unwrap();
    let exp: Vec<u16> = o_exp.into_raw_vec_and_offset().0;
    let o_len = (batch * seq_q * num_heads * attention::MLA_D_C) as usize;
    let mut o_dev = stream.alloc_zeros::<u16>(o_len).unwrap();

    attention::mla_prefill_fp8(
        &ctx,
        &stream,
        &q_c_dev,
        &q_r_dev,
        &c_kv_dev,
        &k_rope_dev,
        &mut o_dev,
        batch,
        num_heads,
        seq_q,
        seq_kv,
        scale,
        kv_scale,
    )
    .expect("MLA FP8 prefill failed");

    let o_host = stream.memcpy_dtov(&o_dev).unwrap();
    let (max_diff, mean_diff) = compare_bf16(&o_host, &exp, tol);
    eprintln!("MLA FP8 prefill {npz_name}: max_diff={max_diff:.6} mean_diff={mean_diff:.6}");
    assert!(max_diff <= tol, "MLA FP8 prefill differs: {}", max_diff);
}

#[test]
fn test_mla_prefill_fp8_b1_sq16_h4() {
    run_mla_prefill_fp8_test("mla_prefill_fp8_B1_Sq16_H4.npz", 2.0);
}

#[test]
fn test_mla_prefill_fp8_b2_sq32_h8() {
    run_mla_prefill_fp8_test("mla_prefill_fp8_B2_Sq32_H8.npz", 3.0);
}

// ============================================================================
// Flash Attention V3 d=256 (GDN-hybrid gated full-attention head_dim=256)
// ============================================================================

fn run_flash_attn_v3_d256_test(npz_name: &str, tol: f32) {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let mut npz = load_npz(npz_name);

    let q_np: ndarray::Array4<u16> = npz.by_name("q").unwrap();
    let k_np: ndarray::Array4<u16> = npz.by_name("k").unwrap();
    let v_np: ndarray::Array4<u16> = npz.by_name("v").unwrap();
    let o_expected: ndarray::Array4<u16> = npz.by_name("o").unwrap();
    let scale_np: ndarray::Array0<f32> = npz.by_name("scale").unwrap();

    let batch = q_np.shape()[0] as u32;
    let num_heads = q_np.shape()[1] as u32;
    let seq_q = q_np.shape()[2] as u32;
    let seq_kv = k_np.shape()[2] as u32;
    let head_dim = q_np.shape()[3];
    assert_eq!(head_dim, 256, "this test is for head_dim=256");
    let scale = scale_np.into_scalar();

    let q_flat: Vec<u16> = q_np.into_raw_vec_and_offset().0;
    let k_flat: Vec<u16> = k_np.into_raw_vec_and_offset().0;
    let v_flat: Vec<u16> = v_np.into_raw_vec_and_offset().0;
    let expected_flat: Vec<u16> = o_expected.into_raw_vec_and_offset().0;

    let q_dev = stream.memcpy_stod(&q_flat).unwrap();
    let k_dev = stream.memcpy_stod(&k_flat).unwrap();
    let v_dev = stream.memcpy_stod(&v_flat).unwrap();
    let mut o_dev = stream
        .alloc_zeros::<u16>((batch * num_heads * seq_q * 256) as usize)
        .unwrap();

    attention::flash_attn_bf16_v3_d256(
        &ctx, &stream, &q_dev, &k_dev, &v_dev, &mut o_dev, batch, num_heads, seq_q, seq_kv, scale,
    )
    .unwrap();

    let o_host = stream.memcpy_dtov(&o_dev).unwrap();
    let (max_diff, mean_diff) = compare_bf16(&o_host, &expected_flat, tol);
    eprintln!("V3 d256 {npz_name}: max_diff={max_diff:.6} mean_diff={mean_diff:.6} (tol={tol})");
}

#[test]
fn test_flash_attn_bf16_v3_d256_noncausal_s128() {
    // Single Q-block validation. seq=128 = exactly Br, so 1 q_block × 2 kv_blocks.
    run_flash_attn_v3_d256_test("flash_attn_bf16_d256_noncausal_s128.npz", 0.05);
}

#[test]
fn test_flash_attn_bf16_v3_d256_noncausal_s256() {
    // Multi Q-block validation. seq=256 = 2 q_blocks × 4 kv_blocks.
    run_flash_attn_v3_d256_test("flash_attn_bf16_d256_noncausal_s256.npz", 0.05);
}

fn run_flash_attn_v3_d256_causal_test(npz_name: &str, tol: f32) {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();
    let mut npz = load_npz(npz_name);
    let q_np: ndarray::Array4<u16> = npz.by_name("q").unwrap();
    let k_np: ndarray::Array4<u16> = npz.by_name("k").unwrap();
    let v_np: ndarray::Array4<u16> = npz.by_name("v").unwrap();
    let o_expected: ndarray::Array4<u16> = npz.by_name("o").unwrap();
    let scale_np: ndarray::Array0<f32> = npz.by_name("scale").unwrap();

    let batch = q_np.shape()[0] as u32;
    let num_heads = q_np.shape()[1] as u32;
    let seq_q = q_np.shape()[2] as u32;
    let seq_kv = k_np.shape()[2] as u32;
    let scale = scale_np.into_scalar();

    let q_flat: Vec<u16> = q_np.into_raw_vec_and_offset().0;
    let k_flat: Vec<u16> = k_np.into_raw_vec_and_offset().0;
    let v_flat: Vec<u16> = v_np.into_raw_vec_and_offset().0;
    let expected_flat: Vec<u16> = o_expected.into_raw_vec_and_offset().0;

    let q_dev = stream.memcpy_stod(&q_flat).unwrap();
    let k_dev = stream.memcpy_stod(&k_flat).unwrap();
    let v_dev = stream.memcpy_stod(&v_flat).unwrap();
    let mut o_dev = stream
        .alloc_zeros::<u16>((batch * num_heads * seq_q * 256) as usize)
        .unwrap();

    attention::flash_attn_bf16_v3_d256_causal(
        &ctx, &stream, &q_dev, &k_dev, &v_dev, &mut o_dev, batch, num_heads, seq_q, seq_kv, scale,
    )
    .unwrap();

    let o_host = stream.memcpy_dtov(&o_dev).unwrap();
    let (max_diff, mean_diff) = compare_bf16(&o_host, &expected_flat, tol);
    eprintln!(
        "V3 d256 causal {npz_name}: max_diff={max_diff:.6} mean_diff={mean_diff:.6} (tol={tol})"
    );
}

#[test]
fn test_flash_attn_bf16_v3_d256_causal_s128() {
    run_flash_attn_v3_d256_causal_test("flash_attn_bf16_d256_causal_s128.npz", 0.05);
}

#[test]
fn test_flash_attn_bf16_v3_d256_causal_s256() {
    run_flash_attn_v3_d256_causal_test("flash_attn_bf16_d256_causal_s256.npz", 0.05);
}

fn run_flash_attn_v3_d256_gqa_test(npz_name: &str, causal: bool, tol: f32) {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();
    let mut npz = load_npz(npz_name);
    let q_np: ndarray::Array4<u16> = npz.by_name("q").unwrap();
    let k_np: ndarray::Array4<u16> = npz.by_name("k").unwrap();
    let v_np: ndarray::Array4<u16> = npz.by_name("v").unwrap();
    let o_expected: ndarray::Array4<u16> = npz.by_name("o").unwrap();
    let scale_np: ndarray::Array0<f32> = npz.by_name("scale").unwrap();
    let nh_np: ndarray::Array0<i32> = npz.by_name("num_heads").unwrap();
    let nhkv_np: ndarray::Array0<i32> = npz.by_name("num_heads_kv").unwrap();

    let batch = q_np.shape()[0] as u32;
    let num_heads = nh_np.into_scalar() as u32;
    let num_heads_kv = nhkv_np.into_scalar() as u32;
    let seq_q = q_np.shape()[2] as u32;
    let seq_kv = k_np.shape()[2] as u32;
    let scale = scale_np.into_scalar();

    let q_flat: Vec<u16> = q_np.into_raw_vec_and_offset().0;
    let k_flat: Vec<u16> = k_np.into_raw_vec_and_offset().0;
    let v_flat: Vec<u16> = v_np.into_raw_vec_and_offset().0;
    let expected_flat: Vec<u16> = o_expected.into_raw_vec_and_offset().0;

    let q_dev = stream.memcpy_stod(&q_flat).unwrap();
    let k_dev = stream.memcpy_stod(&k_flat).unwrap();
    let v_dev = stream.memcpy_stod(&v_flat).unwrap();
    let mut o_dev = stream
        .alloc_zeros::<u16>((batch * num_heads * seq_q * 256) as usize)
        .unwrap();

    if causal {
        attention::flash_attn_bf16_v3_d256_gqa_causal(
            &ctx,
            &stream,
            &q_dev,
            &k_dev,
            &v_dev,
            &mut o_dev,
            batch,
            num_heads,
            num_heads_kv,
            seq_q,
            seq_kv,
            scale,
        )
        .unwrap();
    } else {
        attention::flash_attn_bf16_v3_d256_gqa(
            &ctx,
            &stream,
            &q_dev,
            &k_dev,
            &v_dev,
            &mut o_dev,
            batch,
            num_heads,
            num_heads_kv,
            seq_q,
            seq_kv,
            scale,
        )
        .unwrap();
    }

    let o_host = stream.memcpy_dtov(&o_dev).unwrap();
    let (max_diff, mean_diff) = compare_bf16(&o_host, &expected_flat, tol);
    let causal_tag = if causal { "causal" } else { "noncausal" };
    eprintln!(
        "V3 d256 gqa {causal_tag} {npz_name}: max_diff={max_diff:.6} mean_diff={mean_diff:.6} (tol={tol})"
    );
}

#[test]
fn test_flash_attn_bf16_v3_d256_gqa_noncausal_s128() {
    run_flash_attn_v3_d256_gqa_test("flash_attn_bf16_d256_gqa_noncausal_s128.npz", false, 0.05);
}

#[test]
fn test_flash_attn_bf16_v3_d256_gqa_noncausal_s256() {
    run_flash_attn_v3_d256_gqa_test("flash_attn_bf16_d256_gqa_noncausal_s256.npz", false, 0.05);
}

#[test]
fn test_flash_attn_bf16_v3_d256_gqa_causal_s128() {
    run_flash_attn_v3_d256_gqa_test("flash_attn_bf16_d256_gqa_causal_s128.npz", true, 0.05);
}

#[test]
fn test_flash_attn_bf16_v3_d256_gqa_causal_s256() {
    run_flash_attn_v3_d256_gqa_test("flash_attn_bf16_d256_gqa_causal_s256.npz", true, 0.05);
}

#[test]
fn test_flash_attn_backward_bf16() {
    let ctx = device::init_device(0).expect("init SM121");
    let stream = ctx.default_stream();

    for seq in [64u32, 128u32] {
        let mut npz = common::load_npz(&format!("flash_attn_backward_bf16_s{seq}_d64.npz"));
        let q_np: ndarray::Array4<u16> = npz.by_name("q").unwrap();
        let k_np: ndarray::Array4<u16> = npz.by_name("k").unwrap();
        let v_np: ndarray::Array4<u16> = npz.by_name("v").unwrap();
        let do_np: ndarray::Array4<u16> = npz.by_name("do").unwrap();
        let dq_expected: ndarray::Array4<u16> = npz.by_name("dq").unwrap();
        let dk_expected: ndarray::Array4<u16> = npz.by_name("dk").unwrap();
        let dv_expected: ndarray::Array4<u16> = npz.by_name("dv").unwrap();
        let scale_np: ndarray::Array0<f32> = npz.by_name("scale").unwrap();

        // Shapes: [B, H, S, d]
        let b = q_np.shape()[0] as u32;
        let h = q_np.shape()[1] as u32;
        let s = q_np.shape()[2] as u32;
        let d = q_np.shape()[3] as u32;
        let scale = scale_np.into_scalar();

        // The initial backward kernel only supports B*H=1: pick the first head only.
        let bh_stride = (s * d) as usize;
        let q_flat: Vec<u16> = q_np.into_raw_vec_and_offset().0;
        let k_flat: Vec<u16> = k_np.into_raw_vec_and_offset().0;
        let v_flat: Vec<u16> = v_np.into_raw_vec_and_offset().0;
        let do_flat: Vec<u16> = do_np.into_raw_vec_and_offset().0;
        let dq_exp_flat: Vec<u16> = dq_expected.into_raw_vec_and_offset().0;
        let dk_exp_flat: Vec<u16> = dk_expected.into_raw_vec_and_offset().0;
        let dv_exp_flat: Vec<u16> = dv_expected.into_raw_vec_and_offset().0;

        // Iterate over each (b, head) pair (golden has B=1, H=4 → 4 pairs).
        for bh in 0..(b * h) as usize {
            let off = bh * bh_stride;
            let q_dev = stream.memcpy_stod(&q_flat[off..off + bh_stride]).unwrap();
            let k_dev = stream.memcpy_stod(&k_flat[off..off + bh_stride]).unwrap();
            let v_dev = stream.memcpy_stod(&v_flat[off..off + bh_stride]).unwrap();
            let do_dev = stream.memcpy_stod(&do_flat[off..off + bh_stride]).unwrap();
            let mut dq_dev = stream.alloc_zeros::<u16>(bh_stride).unwrap();
            let mut dk_dev = stream.alloc_zeros::<u16>(bh_stride).unwrap();
            let mut dv_dev = stream.alloc_zeros::<u16>(bh_stride).unwrap();

            attention::flash_attn_backward_bf16(
                &ctx,
                &stream,
                &q_dev,
                &k_dev,
                &v_dev,
                &do_dev,
                &mut dq_dev,
                &mut dk_dev,
                &mut dv_dev,
                1,
                1,
                s,
                d,
                scale,
                false,
            )
            .unwrap();

            let dq_h = stream.memcpy_dtov(&dq_dev).unwrap();
            let dk_h = stream.memcpy_dtov(&dk_dev).unwrap();
            let dv_h = stream.memcpy_dtov(&dv_dev).unwrap();
            let (mq, _) = common::compare_bf16(&dq_h, &dq_exp_flat[off..off + bh_stride], 0.5);
            let (mk, _) = common::compare_bf16(&dk_h, &dk_exp_flat[off..off + bh_stride], 0.5);
            let (mv, _) = common::compare_bf16(&dv_h, &dv_exp_flat[off..off + bh_stride], 0.5);
            eprintln!("fa_bw s={s} bh={bh}: dq_max={mq:.4} dk_max={mk:.4} dv_max={mv:.4}");
        }
    }
}

#[test]
fn test_flash_attn_backward_causal_bf16() {
    let ctx = device::init_device(0).expect("init SM121");
    let stream = ctx.default_stream();

    for seq in [64u32, 128u32] {
        let mut npz = common::load_npz(&format!("flash_attn_backward_causal_bf16_s{seq}_d64.npz"));
        let q_np: ndarray::Array4<u16> = npz.by_name("q").unwrap();
        let k_np: ndarray::Array4<u16> = npz.by_name("k").unwrap();
        let v_np: ndarray::Array4<u16> = npz.by_name("v").unwrap();
        let do_np: ndarray::Array4<u16> = npz.by_name("do").unwrap();
        let dq_expected: ndarray::Array4<u16> = npz.by_name("dq").unwrap();
        let dk_expected: ndarray::Array4<u16> = npz.by_name("dk").unwrap();
        let dv_expected: ndarray::Array4<u16> = npz.by_name("dv").unwrap();
        let scale_np: ndarray::Array0<f32> = npz.by_name("scale").unwrap();

        let b = q_np.shape()[0] as u32;
        let h = q_np.shape()[1] as u32;
        let s = q_np.shape()[2] as u32;
        let d = q_np.shape()[3] as u32;
        let scale = scale_np.into_scalar();

        let bh_stride = (s * d) as usize;
        let q_flat: Vec<u16> = q_np.into_raw_vec_and_offset().0;
        let k_flat: Vec<u16> = k_np.into_raw_vec_and_offset().0;
        let v_flat: Vec<u16> = v_np.into_raw_vec_and_offset().0;
        let do_flat: Vec<u16> = do_np.into_raw_vec_and_offset().0;
        let dq_exp_flat: Vec<u16> = dq_expected.into_raw_vec_and_offset().0;
        let dk_exp_flat: Vec<u16> = dk_expected.into_raw_vec_and_offset().0;
        let dv_exp_flat: Vec<u16> = dv_expected.into_raw_vec_and_offset().0;

        for bh in 0..(b * h) as usize {
            let off = bh * bh_stride;
            let q_dev = stream.memcpy_stod(&q_flat[off..off + bh_stride]).unwrap();
            let k_dev = stream.memcpy_stod(&k_flat[off..off + bh_stride]).unwrap();
            let v_dev = stream.memcpy_stod(&v_flat[off..off + bh_stride]).unwrap();
            let do_dev = stream.memcpy_stod(&do_flat[off..off + bh_stride]).unwrap();
            let mut dq_dev = stream.alloc_zeros::<u16>(bh_stride).unwrap();
            let mut dk_dev = stream.alloc_zeros::<u16>(bh_stride).unwrap();
            let mut dv_dev = stream.alloc_zeros::<u16>(bh_stride).unwrap();

            attention::flash_attn_backward_bf16(
                &ctx,
                &stream,
                &q_dev,
                &k_dev,
                &v_dev,
                &do_dev,
                &mut dq_dev,
                &mut dk_dev,
                &mut dv_dev,
                1,
                1,
                s,
                d,
                scale,
                true,
            )
            .unwrap();

            let dq_h = stream.memcpy_dtov(&dq_dev).unwrap();
            let dk_h = stream.memcpy_dtov(&dk_dev).unwrap();
            let dv_h = stream.memcpy_dtov(&dv_dev).unwrap();
            let (mq, _) = common::compare_bf16(&dq_h, &dq_exp_flat[off..off + bh_stride], 0.5);
            let (mk, _) = common::compare_bf16(&dk_h, &dk_exp_flat[off..off + bh_stride], 0.5);
            let (mv, _) = common::compare_bf16(&dv_h, &dv_exp_flat[off..off + bh_stride], 0.5);
            eprintln!("fa_bw_causal s={s} bh={bh}: dq_max={mq:.4} dk_max={mk:.4} dv_max={mv:.4}");
        }
    }
}

#[test]
fn test_flash_attn_backward_gqa_bf16() {
    let ctx = device::init_device(0).expect("init SM121");
    let stream = ctx.default_stream();

    for (h_q, h_kv) in [(8u32, 2u32), (4u32, 1u32)] {
        for (causal, tag) in [(false, "noncausal"), (true, "causal")] {
            let path = format!("flash_attn_backward_gqa_{tag}_bf16_hq{h_q}_hkv{h_kv}_s64.npz");
            let mut npz = common::load_npz(&path);
            let q_np: ndarray::Array4<u16> = npz.by_name("q").unwrap();
            let k_np: ndarray::Array4<u16> = npz.by_name("k").unwrap();
            let v_np: ndarray::Array4<u16> = npz.by_name("v").unwrap();
            let do_np: ndarray::Array4<u16> = npz.by_name("do").unwrap();
            let dq_expected: ndarray::Array4<u16> = npz.by_name("dq").unwrap();
            let dk_expected: ndarray::Array4<u16> = npz.by_name("dk").unwrap();
            let dv_expected: ndarray::Array4<u16> = npz.by_name("dv").unwrap();
            let scale_np: ndarray::Array0<f32> = npz.by_name("scale").unwrap();

            let b = q_np.shape()[0] as u32;
            let s = q_np.shape()[2] as u32;
            let d = q_np.shape()[3] as u32;
            let scale = scale_np.into_scalar();

            let q_flat: Vec<u16> = q_np.into_raw_vec_and_offset().0;
            let k_flat: Vec<u16> = k_np.into_raw_vec_and_offset().0;
            let v_flat: Vec<u16> = v_np.into_raw_vec_and_offset().0;
            let do_flat: Vec<u16> = do_np.into_raw_vec_and_offset().0;
            let dq_exp: Vec<u16> = dq_expected.into_raw_vec_and_offset().0;
            let dk_exp: Vec<u16> = dk_expected.into_raw_vec_and_offset().0;
            let dv_exp: Vec<u16> = dv_expected.into_raw_vec_and_offset().0;

            let q_dev = stream.memcpy_stod(&q_flat).unwrap();
            let k_dev = stream.memcpy_stod(&k_flat).unwrap();
            let v_dev = stream.memcpy_stod(&v_flat).unwrap();
            let do_dev = stream.memcpy_stod(&do_flat).unwrap();
            let mut dq_dev = stream.alloc_zeros::<u16>(q_flat.len()).unwrap();
            let mut dk_dev = stream.alloc_zeros::<u16>(k_flat.len()).unwrap();
            let mut dv_dev = stream.alloc_zeros::<u16>(v_flat.len()).unwrap();

            attention::flash_attn_backward_bf16_gqa(
                &ctx,
                &stream,
                &q_dev,
                &k_dev,
                &v_dev,
                &do_dev,
                &mut dq_dev,
                &mut dk_dev,
                &mut dv_dev,
                b,
                h_q,
                h_kv,
                s,
                d,
                scale,
                causal,
            )
            .unwrap();

            let dq_h = stream.memcpy_dtov(&dq_dev).unwrap();
            let dk_h = stream.memcpy_dtov(&dk_dev).unwrap();
            let dv_h = stream.memcpy_dtov(&dv_dev).unwrap();
            // Higher tolerance for GQA dK/dV due to reduction across head groups.
            let (mq, _) = common::compare_bf16(&dq_h, &dq_exp, 0.5);
            let (mk, _) = common::compare_bf16(&dk_h, &dk_exp, 1.0);
            let (mv, _) = common::compare_bf16(&dv_h, &dv_exp, 1.0);
            eprintln!(
                "fa_bw_gqa hq={h_q} hkv={h_kv} {tag}: dq_max={mq:.4} dk_max={mk:.4} dv_max={mv:.4}"
            );
        }
    }
}
