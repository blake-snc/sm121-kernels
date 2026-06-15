mod common;

use common::load_npz;
use sm121_kernels::{device, kv_cache};

fn run_kv_write_test(npz_name: &str, tol_bytes: u8) {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let mut npz = load_npz(npz_name);
    let new_k: ndarray::Array3<u16> = npz.by_name("new_k").unwrap();
    let new_v: ndarray::Array3<u16> = npz.by_name("new_v").unwrap();
    let k_cache_in: ndarray::Array4<u8> = npz.by_name("k_cache_in").unwrap();
    let v_cache_in: ndarray::Array4<u8> = npz.by_name("v_cache_in").unwrap();
    let k_cache_out: ndarray::Array4<u8> = npz.by_name("k_cache_out").unwrap();
    let v_cache_out: ndarray::Array4<u8> = npz.by_name("v_cache_out").unwrap();
    let page_indices: ndarray::Array1<u32> = npz.by_name("page_indices").unwrap();
    let slot_in_page: ndarray::Array1<u32> = npz.by_name("slot_in_page").unwrap();
    let k_scale: ndarray::Array0<f32> = npz.by_name("k_scale").unwrap();
    let v_scale: ndarray::Array0<f32> = npz.by_name("v_scale").unwrap();
    let batch: ndarray::Array0<u32> = npz.by_name("batch").unwrap();
    let num_heads: ndarray::Array0<u32> = npz.by_name("num_heads").unwrap();
    let page_size: ndarray::Array0<u32> = npz.by_name("page_size").unwrap();

    let batch = batch.into_scalar();
    let num_heads = num_heads.into_scalar();
    let page_size = page_size.into_scalar();
    let k_scale = k_scale.into_scalar();
    let v_scale = v_scale.into_scalar();

    let new_k_flat: Vec<u16> = new_k.into_raw_vec_and_offset().0;
    let new_v_flat: Vec<u16> = new_v.into_raw_vec_and_offset().0;
    let k_cache_in_flat: Vec<u8> = k_cache_in.into_raw_vec_and_offset().0;
    let v_cache_in_flat: Vec<u8> = v_cache_in.into_raw_vec_and_offset().0;
    let k_cache_expected: Vec<u8> = k_cache_out.into_raw_vec_and_offset().0;
    let v_cache_expected: Vec<u8> = v_cache_out.into_raw_vec_and_offset().0;
    let page_indices_flat: Vec<u32> = page_indices.into_raw_vec_and_offset().0;
    let slot_in_page_flat: Vec<u32> = slot_in_page.into_raw_vec_and_offset().0;

    let new_k_dev = stream.memcpy_stod(&new_k_flat).unwrap();
    let new_v_dev = stream.memcpy_stod(&new_v_flat).unwrap();
    let mut k_cache_dev = stream.memcpy_stod(&k_cache_in_flat).unwrap();
    let mut v_cache_dev = stream.memcpy_stod(&v_cache_in_flat).unwrap();
    let page_indices_dev = stream.memcpy_stod(&page_indices_flat).unwrap();
    let slot_in_page_dev = stream.memcpy_stod(&slot_in_page_flat).unwrap();

    kv_cache::kv_cache_fp8_write(
        &ctx,
        &stream,
        &new_k_dev,
        &new_v_dev,
        &mut k_cache_dev,
        &mut v_cache_dev,
        &page_indices_dev,
        &slot_in_page_dev,
        batch,
        num_heads,
        page_size,
        k_scale,
        v_scale,
    )
    .expect("KV FP8 write failed");

    let k_cache_host = stream.memcpy_dtov(&k_cache_dev).unwrap();
    let v_cache_host = stream.memcpy_dtov(&v_cache_dev).unwrap();

    // Compare bytes; tolerance for minor rounding differences
    let mut k_diff_count = 0;
    let mut k_max_byte_diff: u8 = 0;
    for (a, b) in k_cache_host.iter().zip(k_cache_expected.iter()) {
        let d = a.abs_diff(*b);
        if d > 0 {
            k_diff_count += 1;
        }
        if d > k_max_byte_diff {
            k_max_byte_diff = d;
        }
    }
    let mut v_diff_count = 0;
    let mut v_max_byte_diff: u8 = 0;
    for (a, b) in v_cache_host.iter().zip(v_cache_expected.iter()) {
        let d = a.abs_diff(*b);
        if d > 0 {
            v_diff_count += 1;
        }
        if d > v_max_byte_diff {
            v_max_byte_diff = d;
        }
    }
    eprintln!(
        "KV FP8 write {npz_name}: k diffs={k_diff_count} (max byte={k_max_byte_diff}), v diffs={v_diff_count} (max byte={v_max_byte_diff})"
    );
    assert!(
        k_max_byte_diff <= tol_bytes,
        "K cache byte mismatch too large: {}",
        k_max_byte_diff
    );
    assert!(
        v_max_byte_diff <= tol_bytes,
        "V cache byte mismatch too large: {}",
        v_max_byte_diff
    );
}

#[test]
fn test_kv_fp8_write_b1_h4() {
    run_kv_write_test("kv_cache_fp8_write_B1_H4.npz", 1);
}

#[test]
fn test_kv_fp8_write_b2_h8() {
    run_kv_write_test("kv_cache_fp8_write_B2_H8.npz", 1);
}
