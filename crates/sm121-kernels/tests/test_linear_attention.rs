mod common;

use common::{compare_bf16, load_npz};
use sm121_kernels::{device, linear_attention};

fn run_gdn_decode_test(npz_name: &str, tol_y: f32, tol_state: f32) {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let mut npz = load_npz(npz_name);
    let q: ndarray::Array3<u16> = npz.by_name("q").unwrap();
    let k: ndarray::Array3<u16> = npz.by_name("k").unwrap();
    let v: ndarray::Array3<u16> = npz.by_name("v").unwrap();
    let alpha: ndarray::Array2<f32> = npz.by_name("alpha").unwrap();
    let beta: ndarray::Array2<f32> = npz.by_name("beta").unwrap();
    let state_in: ndarray::Array4<f32> = npz.by_name("state_in").unwrap();
    let state_out_expected: ndarray::Array4<f32> = npz.by_name("state_out").unwrap();
    let y_expected: ndarray::Array3<u16> = npz.by_name("y").unwrap();
    let batch: ndarray::Array0<u32> = npz.by_name("batch").unwrap();
    let num_heads: ndarray::Array0<u32> = npz.by_name("num_heads").unwrap();

    let batch = batch.into_scalar();
    let num_heads = num_heads.into_scalar();

    let q_flat: Vec<u16> = q.into_raw_vec_and_offset().0;
    let k_flat: Vec<u16> = k.into_raw_vec_and_offset().0;
    let v_flat: Vec<u16> = v.into_raw_vec_and_offset().0;
    let alpha_flat: Vec<f32> = alpha.into_raw_vec_and_offset().0;
    let beta_flat: Vec<f32> = beta.into_raw_vec_and_offset().0;
    let state_flat: Vec<f32> = state_in.into_raw_vec_and_offset().0;
    let state_expected_flat: Vec<f32> = state_out_expected.into_raw_vec_and_offset().0;
    let y_expected_flat: Vec<u16> = y_expected.into_raw_vec_and_offset().0;

    let q_dev = stream.memcpy_stod(&q_flat).unwrap();
    let k_dev = stream.memcpy_stod(&k_flat).unwrap();
    let v_dev = stream.memcpy_stod(&v_flat).unwrap();
    let alpha_dev = stream.memcpy_stod(&alpha_flat).unwrap();
    let beta_dev = stream.memcpy_stod(&beta_flat).unwrap();
    // GDN state is stored FP16: upload f32 reference as f16 bits.
    let state_f16: Vec<u16> = state_flat
        .iter()
        .map(|&x| half::f16::from_f32(x).to_bits())
        .collect();
    let mut state_dev = stream.memcpy_stod(&state_f16).unwrap();
    let mut y_dev = stream.alloc_zeros::<u16>(q_flat.len()).unwrap();

    linear_attention::gdn_decode(
        &ctx,
        &stream,
        &q_dev,
        &k_dev,
        &v_dev,
        &alpha_dev,
        &beta_dev,
        &mut state_dev,
        &mut y_dev,
        batch,
        num_heads,
    )
    .expect("GDN decode failed");

    let y_host = stream.memcpy_dtov(&y_dev).unwrap();
    // GDN state is stored FP16: widen f16 bits to f32 for comparison.
    let state_host: Vec<f32> = stream
        .memcpy_dtov(&state_dev)
        .unwrap()
        .iter()
        .map(|&b| half::f16::from_bits(b).to_f32())
        .collect();

    let (max_y_diff, mean_y_diff) = compare_bf16(&y_host, &y_expected_flat, tol_y);
    eprintln!("GDN decode {npz_name}: y max_diff={max_y_diff:.6}, mean_diff={mean_y_diff:.6}");
    assert!(
        max_y_diff <= tol_y,
        "GDN y differs (max_diff={:.4})",
        max_y_diff
    );

    // State output comparison (FP32)
    let mut max_state_diff: f32 = 0.0;
    for (a, b) in state_host.iter().zip(state_expected_flat.iter()) {
        let d = (a - b).abs();
        if d > max_state_diff {
            max_state_diff = d;
        }
    }
    eprintln!("GDN decode {npz_name}: state max_diff={max_state_diff:.6}");
    assert!(
        max_state_diff <= tol_state,
        "GDN state differs (max_diff={:.4})",
        max_state_diff
    );
}

#[test]
fn test_gdn_decode_b1_h4() {
    // FP16 GDN state: loosen state tol from 0.05 to FP16-appropriate 0.2.
    run_gdn_decode_test("gdn_decode_B1_H4.npz", 0.15, 0.2);
}

#[test]
fn test_gdn_decode_b2_h8() {
    // FP16 GDN state: loosen state tol from 0.05 to FP16-appropriate 0.2.
    run_gdn_decode_test("gdn_decode_B2_H8.npz", 0.15, 0.2);
}

fn run_mamba2_decode_test(npz_name: &str, tol_y: f32, tol_state: f32) {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let mut npz = load_npz(npz_name);
    let x: ndarray::Array2<f32> = npz.by_name("x").unwrap();
    let delta: ndarray::Array2<f32> = npz.by_name("delta").unwrap();
    let a_log: ndarray::Array3<f32> = npz.by_name("A_log").unwrap();
    let b_proj: ndarray::Array3<f32> = npz.by_name("B").unwrap();
    let c_proj: ndarray::Array3<f32> = npz.by_name("C").unwrap();
    let h_in: ndarray::Array3<f32> = npz.by_name("h_in").unwrap();
    let h_out_expected: ndarray::Array3<f32> = npz.by_name("h_out").unwrap();
    let y_expected: ndarray::Array2<f32> = npz.by_name("y").unwrap();
    let batch: ndarray::Array0<u32> = npz.by_name("batch").unwrap();
    let num_heads: ndarray::Array0<u32> = npz.by_name("num_heads").unwrap();

    let batch = batch.into_scalar();
    let num_heads = num_heads.into_scalar();

    let x_flat: Vec<f32> = x.into_raw_vec_and_offset().0;
    let delta_flat: Vec<f32> = delta.into_raw_vec_and_offset().0;
    let a_flat: Vec<f32> = a_log.into_raw_vec_and_offset().0;
    let b_flat: Vec<f32> = b_proj.into_raw_vec_and_offset().0;
    let c_flat: Vec<f32> = c_proj.into_raw_vec_and_offset().0;
    let h_flat: Vec<f32> = h_in.into_raw_vec_and_offset().0;
    let h_expected_flat: Vec<f32> = h_out_expected.into_raw_vec_and_offset().0;
    let y_expected_flat: Vec<f32> = y_expected.into_raw_vec_and_offset().0;

    let x_dev = stream.memcpy_stod(&x_flat).unwrap();
    let delta_dev = stream.memcpy_stod(&delta_flat).unwrap();
    let a_dev = stream.memcpy_stod(&a_flat).unwrap();
    let b_dev = stream.memcpy_stod(&b_flat).unwrap();
    let c_dev = stream.memcpy_stod(&c_flat).unwrap();
    let mut h_dev = stream.memcpy_stod(&h_flat).unwrap();
    let mut y_dev = stream.alloc_zeros::<f32>(x_flat.len()).unwrap();

    linear_attention::mamba2_selective_scan_decode(
        &ctx, &stream, &x_dev, &delta_dev, &a_dev, &b_dev, &c_dev, &mut h_dev, &mut y_dev, batch,
        num_heads,
    )
    .expect("Mamba2 decode failed");

    let y_host = stream.memcpy_dtov(&y_dev).unwrap();
    let h_host = stream.memcpy_dtov(&h_dev).unwrap();

    let mut max_y_diff: f32 = 0.0;
    for (a, b) in y_host.iter().zip(y_expected_flat.iter()) {
        let d = (a - b).abs();
        if d > max_y_diff {
            max_y_diff = d;
        }
    }
    let mut max_state_diff: f32 = 0.0;
    for (a, b) in h_host.iter().zip(h_expected_flat.iter()) {
        let d = (a - b).abs();
        if d > max_state_diff {
            max_state_diff = d;
        }
    }
    eprintln!("Mamba2 {npz_name}: y max_diff={max_y_diff:.6} state max_diff={max_state_diff:.6}");
    assert!(max_y_diff <= tol_y, "Mamba2 y differs: {}", max_y_diff);
    assert!(
        max_state_diff <= tol_state,
        "Mamba2 state differs: {}",
        max_state_diff
    );
}

#[test]
fn test_mamba2_decode_b1_h4() {
    run_mamba2_decode_test("mamba2_decode_B1_H4.npz", 0.001, 0.0001);
}

#[test]
fn test_mamba2_decode_b2_h8() {
    run_mamba2_decode_test("mamba2_decode_B2_H8.npz", 0.001, 0.0001);
}

fn run_gdn_prefill_test(npz_name: &str, tol_y: f32, tol_state: f32) {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let mut npz = load_npz(npz_name);
    let q: ndarray::Array4<u16> = npz.by_name("q").unwrap();
    let k: ndarray::Array4<u16> = npz.by_name("k").unwrap();
    let v: ndarray::Array4<u16> = npz.by_name("v").unwrap();
    let alpha: ndarray::Array3<f32> = npz.by_name("alpha").unwrap();
    let beta: ndarray::Array3<f32> = npz.by_name("beta").unwrap();
    let state_in: ndarray::Array4<f32> = npz.by_name("state_in").unwrap();
    let state_out_expected: ndarray::Array4<f32> = npz.by_name("state_out").unwrap();
    let y_expected: ndarray::Array4<u16> = npz.by_name("y").unwrap();
    let batch: ndarray::Array0<u32> = npz.by_name("batch").unwrap();
    let num_heads: ndarray::Array0<u32> = npz.by_name("num_heads").unwrap();
    let seq_q: ndarray::Array0<u32> = npz.by_name("seq_q").unwrap();

    let batch = batch.into_scalar();
    let num_heads = num_heads.into_scalar();
    let seq_q = seq_q.into_scalar();

    let q_flat: Vec<u16> = q.into_raw_vec_and_offset().0;
    let k_flat: Vec<u16> = k.into_raw_vec_and_offset().0;
    let v_flat: Vec<u16> = v.into_raw_vec_and_offset().0;
    let alpha_flat: Vec<f32> = alpha.into_raw_vec_and_offset().0;
    let beta_flat: Vec<f32> = beta.into_raw_vec_and_offset().0;
    let state_flat: Vec<f32> = state_in.into_raw_vec_and_offset().0;
    let state_expected_flat: Vec<f32> = state_out_expected.into_raw_vec_and_offset().0;
    let y_expected_flat: Vec<u16> = y_expected.into_raw_vec_and_offset().0;

    let q_dev = stream.memcpy_stod(&q_flat).unwrap();
    let k_dev = stream.memcpy_stod(&k_flat).unwrap();
    let v_dev = stream.memcpy_stod(&v_flat).unwrap();
    let alpha_dev = stream.memcpy_stod(&alpha_flat).unwrap();
    let beta_dev = stream.memcpy_stod(&beta_flat).unwrap();
    // GDN state is stored FP16: upload f32 reference as f16 bits.
    let state_f16: Vec<u16> = state_flat
        .iter()
        .map(|&x| half::f16::from_f32(x).to_bits())
        .collect();
    let mut state_dev = stream.memcpy_stod(&state_f16).unwrap();
    let mut y_dev = stream.alloc_zeros::<u16>(q_flat.len()).unwrap();

    linear_attention::gdn_prefill(
        &ctx,
        &stream,
        &q_dev,
        &k_dev,
        &v_dev,
        &alpha_dev,
        &beta_dev,
        &mut state_dev,
        &mut y_dev,
        batch,
        num_heads,
        seq_q,
    )
    .expect("GDN prefill failed");

    let y_host = stream.memcpy_dtov(&y_dev).unwrap();
    // GDN state is stored FP16: widen f16 bits to f32 for comparison.
    let state_host: Vec<f32> = stream
        .memcpy_dtov(&state_dev)
        .unwrap()
        .iter()
        .map(|&b| half::f16::from_bits(b).to_f32())
        .collect();

    let (max_y_diff, mean_y_diff) = compare_bf16(&y_host, &y_expected_flat, tol_y);
    eprintln!("GDN prefill {npz_name}: y max_diff={max_y_diff:.6} mean_diff={mean_y_diff:.6}");
    assert!(max_y_diff <= tol_y, "GDN prefill y differs: {}", max_y_diff);

    let mut max_state_diff: f32 = 0.0;
    for (a, b) in state_host.iter().zip(state_expected_flat.iter()) {
        let d = (a - b).abs();
        if d > max_state_diff {
            max_state_diff = d;
        }
    }
    eprintln!("GDN prefill {npz_name}: state max_diff={max_state_diff:.6}");
    assert!(
        max_state_diff <= tol_state,
        "GDN prefill state differs: {}",
        max_state_diff
    );
}

#[test]
fn test_gdn_prefill_b1_h4_sq4() {
    // FP16 GDN state: loosen state tol from 0.05 to FP16-appropriate 0.2.
    // tol_y = 2.0: y reaches |y| ~ 230 where one BF16 ULP is 1.0, and the FP16
    // recurrent state shifts y by up to one representable step at that
    // magnitude (measured max diff 1.0; tolerance set to ~2x measured max,
    // matching the repo convention). The old 0.5 asserted sub-ULP agreement.
    run_gdn_prefill_test("gdn_prefill_B1_H4_Sq4.npz", 2.0, 0.2);
}

#[test]
fn test_gdn_prefill_b2_h4_sq8() {
    // tol_y = 4.0: |y| exceeds 256 here, where one BF16 ULP is 2.0 (measured max
    // diff 2.0; ~2x measured). tol_state = 1.0: the FP16 recurrent state reaches
    // magnitudes where its round-off vs the f32 reference accumulates to ~0.5
    // over the 8-step sequence (measured max 0.512; ~2x measured). This test
    // kept the pre-FP16-state tolerances when the state moved to FP16 and had
    // been failing on precision, not correctness (mean y diff 0.0016).
    run_gdn_prefill_test("gdn_prefill_B2_H4_Sq8.npz", 4.0, 1.0);
}

/// NVIDIA-review F3: validate the HF-correct GDN prefill kernel (gated behind
/// SPARK_GDN_HF) against its HF golden, AND confirm the alpha-decay actually
/// changes the output vs the default no-alpha golden (so the gate is real).
/// Must run single-threaded (the suite default `--test-threads=1`): the gate is
/// a process env var.
fn run_gdn_prefill_hf_test(hf_npz: &str, default_npz: &str, tol_y: f32) {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    // Load HF golden inputs (same inputs as the default golden; only the
    // reference y/state differ by the alpha-decay).
    let mut npz = load_npz(hf_npz);
    let q: ndarray::Array4<u16> = npz.by_name("q").unwrap();
    let k: ndarray::Array4<u16> = npz.by_name("k").unwrap();
    let v: ndarray::Array4<u16> = npz.by_name("v").unwrap();
    let alpha: ndarray::Array3<f32> = npz.by_name("alpha").unwrap();
    let beta: ndarray::Array3<f32> = npz.by_name("beta").unwrap();
    let state_in: ndarray::Array4<f32> = npz.by_name("state_in").unwrap();
    let y_hf: ndarray::Array4<u16> = npz.by_name("y").unwrap();
    let batch_a: ndarray::Array0<u32> = npz.by_name("batch").unwrap();
    let num_heads_a: ndarray::Array0<u32> = npz.by_name("num_heads").unwrap();
    let seq_q_a: ndarray::Array0<u32> = npz.by_name("seq_q").unwrap();
    let batch = batch_a.into_scalar();
    let num_heads = num_heads_a.into_scalar();
    let seq_q = seq_q_a.into_scalar();
    let y_default_arr: ndarray::Array4<u16> = load_npz(default_npz).by_name("y").unwrap();
    let y_default: Vec<u16> = y_default_arr.into_raw_vec_and_offset().0;

    let q_dev = stream.memcpy_stod(&q.into_raw_vec_and_offset().0).unwrap();
    let k_dev = stream.memcpy_stod(&k.into_raw_vec_and_offset().0).unwrap();
    let v_dev = stream.memcpy_stod(&v.into_raw_vec_and_offset().0).unwrap();
    let alpha_dev = stream
        .memcpy_stod(&alpha.into_raw_vec_and_offset().0)
        .unwrap();
    let beta_dev = stream
        .memcpy_stod(&beta.into_raw_vec_and_offset().0)
        .unwrap();
    let state_f16: Vec<u16> = state_in
        .into_raw_vec_and_offset()
        .0
        .iter()
        .map(|&x| half::f16::from_f32(x).to_bits())
        .collect();
    let mut state_dev = stream.memcpy_stod(&state_f16).unwrap();
    let n = q_dev.len();
    let mut y_dev = stream.alloc_zeros::<u16>(n).unwrap();

    // Enable the HF-correct kernel for this dispatch. SAFETY/ordering: process
    // env; the suite runs single-threaded, and we clear it immediately after.
    std::env::set_var("SPARK_GDN_HF", "1");
    let res = linear_attention::gdn_prefill(
        &ctx,
        &stream,
        &q_dev,
        &k_dev,
        &v_dev,
        &alpha_dev,
        &beta_dev,
        &mut state_dev,
        &mut y_dev,
        batch,
        num_heads,
        seq_q,
    );
    std::env::remove_var("SPARK_GDN_HF");
    res.expect("GDN prefill (HF) failed");

    let y_host = stream.memcpy_dtov(&y_dev).unwrap();
    let y_hf_flat: Vec<u16> = y_hf.into_raw_vec_and_offset().0;

    // (1) HF kernel matches the HF golden (within FP16-state precision).
    let (max_hf, _mean_hf) = compare_bf16(&y_host, &y_hf_flat, tol_y);
    eprintln!("GDN prefill HF {hf_npz}: y max_diff vs HF golden = {max_hf:.4}");
    assert!(
        max_hf <= tol_y,
        "HF kernel diverges from HF golden: {max_hf}"
    );

    // (2) HF kernel is meaningfully DIFFERENT from the default (no-alpha) golden
    // — proves the gate actually applies the alpha decay.
    let max_vs_default = y_host
        .iter()
        .zip(y_default.iter())
        .map(|(&a, &b)| (f32::from_bits((a as u32) << 16) - f32::from_bits((b as u32) << 16)).abs())
        .fold(0.0f32, f32::max);
    eprintln!("GDN prefill HF {hf_npz}: y max_diff vs default golden = {max_vs_default:.4} (expect >> tol)");
    assert!(
        max_vs_default > tol_y * 2.0,
        "HF kernel should differ from the no-alpha golden, got {max_vs_default}"
    );
}

#[test]
fn test_gdn_prefill_hf_b1_h4_sq4() {
    run_gdn_prefill_hf_test(
        "gdn_prefill_hf_B1_H4_Sq4.npz",
        "gdn_prefill_B1_H4_Sq4.npz",
        2.0,
    );
}

#[test]
fn test_gdn_prefill_hf_b2_h4_sq8() {
    run_gdn_prefill_hf_test(
        "gdn_prefill_hf_B2_H4_Sq8.npz",
        "gdn_prefill_B2_H4_Sq8.npz",
        4.0,
    );
}

fn run_mamba2_prefill_test(npz_name: &str, tol_y: f32, tol_state: f32) {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let mut npz = load_npz(npz_name);
    let x: ndarray::Array3<f32> = npz.by_name("x").unwrap();
    let delta: ndarray::Array3<f32> = npz.by_name("delta").unwrap();
    let a_log: ndarray::Array4<f32> = npz.by_name("A_log").unwrap();
    let b_proj: ndarray::Array4<f32> = npz.by_name("B").unwrap();
    let c_proj: ndarray::Array4<f32> = npz.by_name("C").unwrap();
    let h_in: ndarray::Array3<f32> = npz.by_name("h_in").unwrap();
    let h_out_expected: ndarray::Array3<f32> = npz.by_name("h_out").unwrap();
    let y_expected: ndarray::Array3<f32> = npz.by_name("y").unwrap();
    let batch: ndarray::Array0<u32> = npz.by_name("batch").unwrap();
    let num_heads: ndarray::Array0<u32> = npz.by_name("num_heads").unwrap();
    let seq_q: ndarray::Array0<u32> = npz.by_name("seq_q").unwrap();

    let batch = batch.into_scalar();
    let num_heads = num_heads.into_scalar();
    let seq_q = seq_q.into_scalar();

    let x_flat: Vec<f32> = x.into_raw_vec_and_offset().0;
    let delta_flat: Vec<f32> = delta.into_raw_vec_and_offset().0;
    let a_flat: Vec<f32> = a_log.into_raw_vec_and_offset().0;
    let b_flat: Vec<f32> = b_proj.into_raw_vec_and_offset().0;
    let c_flat: Vec<f32> = c_proj.into_raw_vec_and_offset().0;
    let h_flat: Vec<f32> = h_in.into_raw_vec_and_offset().0;
    let h_expected_flat: Vec<f32> = h_out_expected.into_raw_vec_and_offset().0;
    let y_expected_flat: Vec<f32> = y_expected.into_raw_vec_and_offset().0;

    let x_dev = stream.memcpy_stod(&x_flat).unwrap();
    let delta_dev = stream.memcpy_stod(&delta_flat).unwrap();
    let a_dev = stream.memcpy_stod(&a_flat).unwrap();
    let b_dev = stream.memcpy_stod(&b_flat).unwrap();
    let c_dev = stream.memcpy_stod(&c_flat).unwrap();
    let mut h_dev = stream.memcpy_stod(&h_flat).unwrap();
    let mut y_dev = stream.alloc_zeros::<f32>(x_flat.len()).unwrap();

    linear_attention::mamba2_selective_scan_prefill(
        &ctx, &stream, &x_dev, &delta_dev, &a_dev, &b_dev, &c_dev, &mut h_dev, &mut y_dev, batch,
        num_heads, seq_q,
    )
    .expect("Mamba2 prefill failed");

    let y_host = stream.memcpy_dtov(&y_dev).unwrap();
    let h_host = stream.memcpy_dtov(&h_dev).unwrap();

    let mut max_y_diff: f32 = 0.0;
    for (a, b) in y_host.iter().zip(y_expected_flat.iter()) {
        let d = (a - b).abs();
        if d > max_y_diff {
            max_y_diff = d;
        }
    }
    let mut max_state_diff: f32 = 0.0;
    for (a, b) in h_host.iter().zip(h_expected_flat.iter()) {
        let d = (a - b).abs();
        if d > max_state_diff {
            max_state_diff = d;
        }
    }
    eprintln!(
        "Mamba2 prefill {npz_name}: y max_diff={max_y_diff:.6} state max_diff={max_state_diff:.6}"
    );
    assert!(max_y_diff <= tol_y, "y differs: {}", max_y_diff);
    assert!(
        max_state_diff <= tol_state,
        "state differs: {}",
        max_state_diff
    );
}

#[test]
fn test_mamba2_prefill_b1_h4_sq4() {
    run_mamba2_prefill_test("mamba2_prefill_B1_H4_Sq4.npz", 0.001, 0.0001);
}

#[test]
fn test_mamba2_prefill_b2_h4_sq8() {
    run_mamba2_prefill_test("mamba2_prefill_B2_H4_Sq8.npz", 0.001, 0.0001);
}

fn run_gdn_decode_tma_test(npz_name: &str, tol_y: f32, tol_state: f32) {
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();

    let mut npz = load_npz(npz_name);
    let q: ndarray::Array3<u16> = npz.by_name("q").unwrap();
    let k: ndarray::Array3<u16> = npz.by_name("k").unwrap();
    let v: ndarray::Array3<u16> = npz.by_name("v").unwrap();
    let alpha: ndarray::Array2<f32> = npz.by_name("alpha").unwrap();
    let beta: ndarray::Array2<f32> = npz.by_name("beta").unwrap();
    let state_in: ndarray::Array4<f32> = npz.by_name("state_in").unwrap();
    let state_out_expected: ndarray::Array4<f32> = npz.by_name("state_out").unwrap();
    let y_expected: ndarray::Array3<u16> = npz.by_name("y").unwrap();
    let batch: ndarray::Array0<u32> = npz.by_name("batch").unwrap();
    let num_heads: ndarray::Array0<u32> = npz.by_name("num_heads").unwrap();
    let batch = batch.into_scalar();
    let num_heads = num_heads.into_scalar();

    let q_flat: Vec<u16> = q.into_raw_vec_and_offset().0;
    let k_flat: Vec<u16> = k.into_raw_vec_and_offset().0;
    let v_flat: Vec<u16> = v.into_raw_vec_and_offset().0;
    let alpha_flat: Vec<f32> = alpha.into_raw_vec_and_offset().0;
    let beta_flat: Vec<f32> = beta.into_raw_vec_and_offset().0;
    let state_flat: Vec<f32> = state_in.into_raw_vec_and_offset().0;
    let state_expected_flat: Vec<f32> = state_out_expected.into_raw_vec_and_offset().0;
    let y_expected_flat: Vec<u16> = y_expected.into_raw_vec_and_offset().0;

    let q_dev = stream.memcpy_stod(&q_flat).unwrap();
    let k_dev = stream.memcpy_stod(&k_flat).unwrap();
    let v_dev = stream.memcpy_stod(&v_flat).unwrap();
    let alpha_dev = stream.memcpy_stod(&alpha_flat).unwrap();
    let beta_dev = stream.memcpy_stod(&beta_flat).unwrap();
    let mut state_dev = stream.memcpy_stod(&state_flat).unwrap();
    let mut y_dev = stream.alloc_zeros::<u16>(q_flat.len()).unwrap();

    linear_attention::gdn_decode_tma(
        &ctx,
        &stream,
        &q_dev,
        &k_dev,
        &v_dev,
        &alpha_dev,
        &beta_dev,
        &mut state_dev,
        &mut y_dev,
        batch,
        num_heads,
    )
    .expect("GDN TMA failed");

    let y_host = stream.memcpy_dtov(&y_dev).unwrap();
    let state_host = stream.memcpy_dtov(&state_dev).unwrap();
    let (max_y, mean_y) = compare_bf16(&y_host, &y_expected_flat, tol_y);
    eprintln!("GDN TMA {npz_name}: y max={max_y:.6} mean={mean_y:.6}");
    assert!(max_y <= tol_y);
    let mut max_s: f32 = 0.0;
    for (a, b) in state_host.iter().zip(state_expected_flat.iter()) {
        let d = (a - b).abs();
        if d > max_s {
            max_s = d;
        }
    }
    eprintln!("GDN TMA {npz_name}: state max={max_s:.6}");
    assert!(max_s <= tol_state);
}

#[test]
fn test_gdn_decode_tma_b1_h4() {
    run_gdn_decode_tma_test("gdn_decode_B1_H4.npz", 0.1, 0.05);
}

#[test]
fn test_gdn_decode_tma_b2_h8() {
    run_gdn_decode_tma_test("gdn_decode_B2_H8.npz", 0.1, 0.05);
}
