//! Scalar vs MMA-optimized dense FP8 1×128 block-scaled GEMM.
//!
//! Compares `gemm_fp8_block128` (scalar reference) and
//! `gemm_fp8_block128_mma` (MMA m16n8k16, vectorized B loads) at shapes
//! representative of DeepSeek V3 dense MLP layers.

use cudarc::driver::sys::{cuEventCreate, cuEventElapsedTime, cuEventRecord, cuEventSynchronize};
use half::bf16;
use sm121_kernels::{device, gemm};

fn random_bf16(n: usize, seed: u64) -> Vec<u16> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let f = (((state >> 33) as u32 as f32) / u32::MAX as f32 - 0.5) * 2.0;
            bf16::from_f32(f).to_bits()
        })
        .collect()
}

fn random_fp8(n: usize, seed: u64) -> Vec<u8> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let b = ((state >> 33) & 0xFF) as u8;
            if (b & 0x7F) == 0x7F {
                b ^ 0x10
            } else {
                b
            }
        })
        .collect()
}

unsafe fn time_fn<F: FnMut()>(
    stream_raw: cudarc::driver::sys::CUstream,
    mut f: F,
    iters: usize,
) -> f32 {
    let mut start = std::ptr::null_mut();
    let mut stop = std::ptr::null_mut();
    unsafe {
        cuEventCreate(&mut start, 0).result().unwrap();
        cuEventCreate(&mut stop, 0).result().unwrap();
        cuEventRecord(start, stream_raw).result().unwrap();
    }
    for _ in 0..iters {
        f();
    }
    let mut ms = 0f32;
    unsafe {
        cuEventRecord(stop, stream_raw).result().unwrap();
        cuEventSynchronize(stop).result().unwrap();
        cuEventElapsedTime(&mut ms, start, stop).result().unwrap();
    }
    ms / iters as f32
}

fn main() {
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();
    let stream_raw = stream.cu_stream();

    // DeepSeek V3 dense MLP shapes: hidden=7168, intermediate=18432
    // and the MoE expert dim is 2048.
    // Choose batch M values that exercise both compute-bound and BW-bound
    // regimes.
    let configs = [
        ("M=64 N=2048 K=4096", 64u32, 2048u32, 4096u32),
        ("M=128 N=4096 K=4096", 128, 4096, 4096),
        ("M=256 N=4096 K=2048", 256, 4096, 2048),
        ("M=512 N=2048 K=4096", 512, 2048, 4096),
        ("M=512 N=4096 K=4096", 512, 4096, 4096),
        ("DSv3-MLP-up M=512", 512, 18432, 7168),
    ];

    println!(
        "{:<26} {:>10} {:>10} {:>10} {:>12} {:>12}",
        "config", "scalar_us", "mma_us", "speedup", "scalar_TF", "mma_TF"
    );
    println!("{}", "-".repeat(86));

    for (name, m, n, k) in configs {
        let a = random_bf16((m * k) as usize, 0x1);
        let b = random_fp8((n * k) as usize, 0x2);
        let scales: Vec<f32> = (0..(n * (k / 128)))
            .map(|i| 0.5 + (i as f32 * 0.001))
            .collect();

        let a_dev = stream.memcpy_stod(&a).unwrap();
        let b_dev = stream.memcpy_stod(&b).unwrap();
        let s_dev = stream.memcpy_stod(&scales).unwrap();
        let mut c_scalar = stream.alloc_zeros::<u16>((m * n) as usize).unwrap();
        let mut c_mma = stream.alloc_zeros::<u16>((m * n) as usize).unwrap();

        // Warmup.
        for _ in 0..3 {
            gemm::gemm_fp8_block128(
                &ctx,
                &stream,
                &a_dev,
                &b_dev,
                &s_dev,
                &mut c_scalar,
                m,
                n,
                k,
            )
            .unwrap();
            gemm::gemm_fp8_block128_mma(&ctx, &stream, &a_dev, &b_dev, &s_dev, &mut c_mma, m, n, k)
                .unwrap();
        }
        stream.synchronize().ok();

        let iters = 30;
        let scalar_ms = unsafe {
            time_fn(
                stream_raw,
                || {
                    gemm::gemm_fp8_block128(
                        &ctx,
                        &stream,
                        &a_dev,
                        &b_dev,
                        &s_dev,
                        &mut c_scalar,
                        m,
                        n,
                        k,
                    )
                    .unwrap();
                },
                iters,
            )
        };
        let mma_ms = unsafe {
            time_fn(
                stream_raw,
                || {
                    gemm::gemm_fp8_block128_mma(
                        &ctx, &stream, &a_dev, &b_dev, &s_dev, &mut c_mma, m, n, k,
                    )
                    .unwrap();
                },
                iters,
            )
        };

        let scalar_us = scalar_ms * 1000.0;
        let mma_us = mma_ms * 1000.0;
        let speedup = scalar_ms / mma_ms;
        let flops = 2.0 * m as f64 * n as f64 * k as f64;
        let scalar_tf = (flops / (scalar_ms as f64 * 1e-3)) / 1e12;
        let mma_tf = (flops / (mma_ms as f64 * 1e-3)) / 1e12;

        println!(
            "{name:<26} {scalar_us:>10.1} {mma_us:>10.1} {speedup:>10.2}x \
                 {scalar_tf:>12.2} {mma_tf:>12.2}"
        );
    }
}
