//! Correctness for `gemv_bf16_split_k_w16` vs the regular `gemv_bf16_split_k`.
//! Both are split-K with atomicAdd into an f32 accumulator, differing only in
//! the per-block work (8 vs 16 cols/thread). For the same X, B, K, N, and
//! `num_shards`, the f32 output is identical up to atomicAdd ordering — for
//! deterministic single-shard input it's bit-exact.

mod common;

use sm121_kernels::gemm::{gemv_bf16_split_k, gemv_bf16_split_k_w16};

fn bf16(x: f32) -> u16 {
    half::bf16::from_f32(x).to_bits()
}
fn unbf16(b: u16) -> f32 {
    half::bf16::from_bits(b).to_f32()
}

fn run_case(k: u32, n: u32, num_shards: u32, label: &str) {
    let ctx = sm121_kernels::device::init_device(0).expect("init");
    let stream = ctx.default_stream();

    let mut s = 0xBAAD_F00D_u64;
    let mut next = || -> f32 {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 33) as u32 as f32) / (u32::MAX as f32) * 2.0 - 1.0
    };
    let x: Vec<u16> = (0..k).map(|_| bf16(next() * 0.5)).collect();
    let b: Vec<u16> = (0..k as usize * n as usize)
        .map(|_| bf16(next() * 0.3))
        .collect();

    let x_d = stream.memcpy_stod(&x).unwrap();
    let b_d = stream.memcpy_stod(&b).unwrap();

    let mut out8 = stream.alloc_zeros::<f32>(n as usize).unwrap();
    let mut out16 = stream.alloc_zeros::<f32>(n as usize).unwrap();

    gemv_bf16_split_k(&ctx, &stream, &x_d, &b_d, &mut out8, n, k, num_shards).unwrap();
    gemv_bf16_split_k_w16(&ctx, &stream, &x_d, &b_d, &mut out16, n, k, num_shards).unwrap();
    stream.synchronize().unwrap();

    let a = stream.memcpy_dtov(&out8).unwrap();
    let c = stream.memcpy_dtov(&out16).unwrap();

    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    for i in 0..n as usize {
        let d = (a[i] - c[i]).abs();
        if d > max_abs {
            max_abs = d;
        }
        let denom = a[i].abs().max(1e-6);
        let r = d / denom;
        if r > max_rel {
            max_rel = r;
        }
    }
    eprintln!(
        "{label}: max_abs={max_abs:.3e}, max_rel={:.4}%",
        max_rel * 100.0
    );
    // f32 atomicAdd is order-sensitive at the ULP level; with num_shards=1
    // there's a single block-row per col so ordering is deterministic and
    // the two kernels should be bit-exact. Allow a tiny slack for multi-shard.
    let tol = if num_shards == 1 { 1e-5 } else { 5e-4 };
    assert!(max_rel < tol, "{label}: rel diff {max_rel} > tol {tol}");

    // Spot-check vs CPU reference on one column.
    let col = (n / 2) as usize;
    let mut acc = 0f64;
    for kk in 0..k as usize {
        acc += (unbf16(x[kk]) as f64) * (unbf16(b[kk * n as usize + col]) as f64);
    }
    let ref_val = acc as f32;
    let kern_val = c[col];
    let cpu_diff = (kern_val - ref_val).abs() / ref_val.abs().max(1e-6);
    eprintln!(
        "{label}: col {col} kernel={kern_val:+.4} cpu={ref_val:+.4} rel={:.4}%",
        cpu_diff * 100.0
    );
    assert!(cpu_diff < 0.01, "{label} col {col}: vs CPU rel={cpu_diff}");
}

#[test]
fn w16_small_n_single_shard() {
    run_case(512, 256, 1, "K=512 N=256 ns=1");
}
#[test]
fn w16_lm_head_shape_single_shard() {
    run_case(2816, 4096, 1, "K=2816 N=4096 ns=1");
}
#[test]
fn w16_large_n_multi_shard() {
    run_case(2816, 16384, 4, "K=2816 N=16384 ns=4");
}
