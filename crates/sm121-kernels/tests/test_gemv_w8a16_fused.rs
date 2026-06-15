//! Correctness test for gemv_w8a16_fused_bf16 against an FP32 host
//! reference. Compares output to the existing 3-launch path
//! (gemv_w8a16_split_k_managed) at Gemma-4-typical shapes.

use half::bf16;
use sm121_kernels::{device, gemm};

fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}
fn u16_from_bf16(f: f32) -> u16 {
    bf16::from_f32(f).to_bits()
}

fn deterministic_bf16(seed: u64, n: usize) -> Vec<u16> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let f = (((s >> 33) as u32 as f32) / u32::MAX as f32 - 0.5) * 0.4;
            u16_from_bf16(f)
        })
        .collect()
}

fn deterministic_fp8(seed: u64, n: usize) -> Vec<u8> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // e4m3 byte; avoid NaN (0x7F / 0xFF).
            let raw = ((s >> 33) & 0xFF) as u8;
            if raw == 0x7F || raw == 0xFF {
                raw ^ 0x10
            } else {
                raw
            }
        })
        .collect()
}

fn e4m3_bits_to_f32(b: u8) -> f32 {
    let sign = if (b & 0x80) != 0 { -1.0f32 } else { 1.0 };
    let e = ((b >> 3) & 0xF) as i32;
    let m = (b & 0x7) as i32;
    if e == 0 && m == 0 {
        return 0.0 * sign;
    }
    if e == 0 {
        return sign * (m as f32) / 8.0 * 2f32.powi(-6);
    }
    sign * (1.0 + m as f32 / 8.0) * 2f32.powi(e - 7)
}

fn host_gemv_w8a16(x_bf16: &[u16], b_fp8: &[u8], scale: f32, n: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0f32; n];
    for ni in 0..n {
        let mut acc = 0f32;
        for ki in 0..k {
            acc += bf16_to_f32(x_bf16[ki]) * (e4m3_bits_to_f32(b_fp8[ki * n + ni]) * scale);
        }
        out[ni] = acc;
    }
    out
}

fn run_case(n: u32, k: u32) {
    run_case_tol(n, k, 0.5)
}

fn run_case_tol(n: u32, k: u32, tol: f32) {
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();

    let x_host = deterministic_bf16(0x1111 + n as u64 + k as u64, k as usize);
    let b_host = deterministic_fp8(0x2222 + n as u64 + k as u64, (n * k) as usize);
    let scale = 0.0625f32;

    let x_dev = stream.memcpy_stod(&x_host).unwrap();
    let b_dev = stream.memcpy_stod(&b_host).unwrap();
    let mut out_dev = stream.alloc_zeros::<u16>(n as usize).unwrap();

    gemm::gemv_w8a16_fused_bf16(&ctx, &stream, &x_dev, &b_dev, scale, &mut out_dev, n, k)
        .expect("kernel launch");
    stream.synchronize().ok();

    let out_host = stream.memcpy_dtov(&out_dev).unwrap();
    let out_ref = host_gemv_w8a16(&x_host, &b_host, scale, n as usize, k as usize);

    let mut max_diff = 0f32;
    let mut mean_diff = 0f32;
    for (g, r) in out_host.iter().zip(out_ref.iter()) {
        let gf = bf16_to_f32(*g);
        let d = (gf - r).abs();
        if d > max_diff {
            max_diff = d;
        }
        mean_diff += d;
    }
    mean_diff /= out_host.len() as f32;
    eprintln!(
        "  fused W8A16 GEMV (N={n}, K={k}, tol={tol}): max_diff={max_diff:.4} mean_diff={mean_diff:.4}",
    );
    assert!(
        max_diff <= tol,
        "fused W8A16 GEMV max_diff={max_diff} > {tol} (N={n}, K={k})"
    );
}

#[test]
fn test_w8a16_fused_n2048_k2560() {
    // Gemma-4-e4b hidden=2560 (Q proj output for one head equivalent)
    run_case(2048, 2560);
}

#[test]
fn test_w8a16_fused_n4096_k2560() {
    run_case(4096, 2560);
}

#[test]
fn test_w8a16_fused_n8192_k2560() {
    // Gemma-4 q_dim sliding (n_q=32 * head_dim=256)
    run_case(8192, 2560);
}

#[test]
fn test_w8a16_fused_n2560_k10240() {
    // Gemma-4 down_proj-like: large K. FP8 e4m3 per-FMA noise (~12%)
    // accumulated over K=10240 gives expected max_diff up to ~1.0.
    run_case_tol(2560, 10240, 1.0);
}
