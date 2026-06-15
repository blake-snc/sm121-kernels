//! Smoke test for TMA-accelerated MLA decode.
//!
//! This first-cut kernel pipes c_kv via TMA but keeps compute scalar;
//! it's a correctness proof that the TMA/SMEM layout is sound. Once we
//! layer MMA on top we keep this as the regression gate.

mod common;

use common::{compare_bf16, load_npz};
use sm121_kernels::{attention, device};

#[test]
fn test_mla_decode_tma_b1_h16_s32() {
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();

    let mut npz = load_npz("mla_decode_B1_H16_S32.npz");
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

    attention::mla_decode_bf16_tma(
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
    .expect("TMA MLA failed");

    let o_host = stream.memcpy_dtov(&o_dev).unwrap();
    let (max_diff, mean_diff) = compare_bf16(&o_host, &expected, 0.1);
    eprintln!("MLA TMA B1_H16_S32: max_diff={max_diff:.6} mean_diff={mean_diff:.6}");
    assert!(max_diff <= 0.1, "TMA MLA differs too much: {}", max_diff);
}

#[test]
fn test_mla_decode_tma_b2_h32_s128() {
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();

    let mut npz = load_npz("mla_decode_B2_H32_S128.npz");
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

    attention::mla_decode_bf16_tma(
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
    .expect("TMA MLA failed");

    let o_host = stream.memcpy_dtov(&o_dev).unwrap();
    let (max_diff, mean_diff) = compare_bf16(&o_host, &expected, 0.2);
    eprintln!("MLA TMA B2_H32_S128: max_diff={max_diff:.6} mean_diff={mean_diff:.6}");
    assert!(max_diff <= 0.2, "TMA MLA differs: {}", max_diff);
}
