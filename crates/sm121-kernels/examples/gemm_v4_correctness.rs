//! v4 correctness gate vs v3 at M=128, N=128, K=32
//! (single CTA, single K-block — the minimal shape the current v4 supports).
//!
//! Generates random BF16 A [128,32] and B [32,128], runs both kernels,
//! checks element-wise that v4's output matches v3's within BF16 tolerance.

use std::sync::Arc;

use cudarc::driver::CudaStream;
use sm121_kernels::{device, gemm};

fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}
fn f32_to_bf16(f: f32) -> u16 {
    let bits = f.to_bits();
    // round-to-nearest-even
    let bias = 0x7FFF + ((bits >> 16) & 1);
    ((bits + bias) >> 16) as u16
}

fn run_one(
    ctx: &Arc<cudarc::driver::CudaContext>,
    stream: &Arc<CudaStream>,
    m: u32,
    n: u32,
    k: u32,
) -> bool {
    let mut rng_state: u32 = 0x12345678;
    let mut next_u32 = || {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 17;
        rng_state ^= rng_state << 5;
        rng_state
    };
    let mut rand_bf16 = || {
        let u = next_u32();
        let f = ((u & 0xFFFF) as f32 / 65536.0) - 0.5;
        f32_to_bf16(f)
    };

    let a_host: Vec<u16> = (0..m * k).map(|_| rand_bf16()).collect();
    let b_host: Vec<u16> = (0..k * n).map(|_| rand_bf16()).collect();

    let a_dev = stream.memcpy_stod(&a_host).unwrap();
    let b_dev = stream.memcpy_stod(&b_host).unwrap();
    let mut c_v3 = stream.alloc_zeros::<u16>((m * n) as usize).unwrap();
    let mut c_v4 = stream.alloc_zeros::<u16>((m * n) as usize).unwrap();

    gemm::gemm_bf16_mma_v3(ctx, stream, &a_dev, &b_dev, &mut c_v3, m, n, k).expect("v3");
    stream.synchronize().unwrap();
    gemm::gemm_bf16_mma_v4(ctx, stream, &a_dev, &b_dev, &mut c_v4, m, n, k).expect("v4");
    stream.synchronize().unwrap();

    let c_v3_host = stream.memcpy_dtov(&c_v3).unwrap();
    let c_v4_host = stream.memcpy_dtov(&c_v4).unwrap();

    let mut max_abs: f32 = 0.0;
    let mut mean_abs: f64 = 0.0;
    let mut zero_v4 = 0u32;
    for i in 0..(m * n) as usize {
        let v3 = bf16_to_f32(c_v3_host[i]);
        let v4 = bf16_to_f32(c_v4_host[i]);
        let d = (v3 - v4).abs();
        max_abs = max_abs.max(d);
        mean_abs += d as f64;
        if c_v4_host[i] == 0 {
            zero_v4 += 1;
        }
    }
    mean_abs /= (m * n) as f64;

    // Tolerance scales with K (more accumulations → more BF16 rounding error).
    let tol = 0.005 * (k as f32);
    let pass = max_abs <= tol && (zero_v4 as f32) < (m * n) as f32 * 0.5;
    let verdict = if pass { "PASS" } else { "FAIL" };
    println!("  M={m:>5} N={n:>5} K={k:>5}  max|Δ|={max_abs:>7.4}  mean|Δ|={mean_abs:>7.4}  zeros={zero_v4:>5}/{:<5}  tol={tol:.3}  {verdict}", m*n);
    pass
}

fn main() {
    println!("=== GEMM v4 correctness vs v3 across K-loop ===");
    let ctx = device::init_device(0).expect("SM121 init");
    let stream: Arc<CudaStream> = ctx.default_stream();

    let mut all_pass = true;
    for &(m, n, k) in &[
        (128u32, 128u32, 32u32), // Step 4 baseline (single K-block, single CTA)
        (128, 128, 64),          // 2 K-blocks (basic K-loop)
        (128, 128, 128),         // 4 K-blocks
        (128, 128, 512),         // 16 K-blocks
        (256, 128, 32),          // multi-CTA in M
        (128, 256, 32),          // multi-CTA in N
        (256, 256, 32),          // multi-CTA both axes
        (256, 256, 256),         // multi-CTA + K-loop
        (512, 8192, 4096),       // QKV-proj shape at M=512 (the cuBLAS comparison point)
    ] {
        if !run_one(&ctx, &stream, m, n, k) {
            all_pass = false;
        }
    }

    println!();
    if all_pass {
        println!("PASS: v4 matches v3 across full shape sweep");
    } else {
        println!("FAIL: at least one shape diverged");
        std::process::exit(1);
    }
}
