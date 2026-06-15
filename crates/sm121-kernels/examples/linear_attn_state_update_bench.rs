//! Bench MMA state update kernel vs scalar baseline (chunk-prefill stage 4).
//!
//! Compares time for the V^T @ K computation only — the heaviest single GEMM
//! in linear-attention chunk-scan (~40% of per-chunk FLOPS).
use cudarc::driver::sys::{cuEventCreate, cuEventElapsedTime, cuEventRecord, cuEventSynchronize};
use half::bf16;
use sm121_kernels::{device, linear_attention};

fn random_bf16(n: usize) -> Vec<u16> {
    let mut s = 0xACE1u64;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let f = (((s >> 33) as u32 as f32) / u32::MAX as f32 - 0.5) * 0.5;
            bf16::from_f32(f).to_bits()
        })
        .collect()
}

unsafe fn time_fn<F: FnMut()>(s: cudarc::driver::sys::CUstream, mut f: F, iters: usize) -> f32 {
    let mut start = std::ptr::null_mut();
    let mut stop = std::ptr::null_mut();
    unsafe {
        cuEventCreate(&mut start, 0).result().unwrap();
        cuEventCreate(&mut stop, 0).result().unwrap();
        cuEventRecord(start, s).result().unwrap();
    }
    for _ in 0..iters {
        f();
    }
    let mut ms = 0f32;
    unsafe {
        cuEventRecord(stop, s).result().unwrap();
        cuEventSynchronize(stop).result().unwrap();
        cuEventElapsedTime(&mut ms, start, stop).result().unwrap();
    }
    ms / iters as f32
}

fn main() {
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();
    let raw = stream.cu_stream();

    let configs: &[(u32, u32)] = &[(1, 1), (1, 4), (1, 16), (2, 16), (4, 32)];

    println!("MMA state update kernel — V^T @ K + S_init, [128,128] FP32 output");
    println!("Per-chunk standalone (C=32, D=128); compare to scalar chunk-prefill total/chunk\n");
    println!("{:<14} {:>10} {:>10}", "B×H", "MMA(us)", "TFLOPS*");
    println!("{}", "-".repeat(40));

    for &(b, h) in configs {
        let bh = (b * h) as usize;
        let v = stream.memcpy_stod(&random_bf16(bh * 32 * 128)).unwrap();
        let k = stream.memcpy_stod(&random_bf16(bh * 32 * 128)).unwrap();
        let s_in = stream.alloc_zeros::<u16>(bh * 128 * 128).unwrap(); // FP16 state
        let mut s_out = stream.alloc_zeros::<u16>(bh * 128 * 128).unwrap();

        for _ in 0..3 {
            linear_attention::linear_attn_state_update_mma(
                &ctx, &stream, &v, &k, &s_in, &mut s_out, b, h,
            )
            .unwrap();
        }
        stream.synchronize().unwrap();

        let iters = 100;
        let ms = unsafe {
            time_fn(
                raw,
                || {
                    linear_attention::linear_attn_state_update_mma(
                        &ctx, &stream, &v, &k, &s_in, &mut s_out, b, h,
                    )
                    .unwrap();
                },
                iters,
            )
        };

        // FLOPs per CTA: 2 × M × N × K = 2 × 128 × 128 × 32 = 1.05M
        // Total FLOPs: bh CTAs × 1.05M = 1.05M × bh
        let flops = 2.0 * 128.0 * 128.0 * 32.0 * bh as f64;
        let tflops = flops / (ms as f64 * 1e-3) / 1e12;

        println!(
            "{:<14} {:>10.1} {:>10.2}",
            format!("{}×{}", b, h),
            ms * 1000.0,
            tflops
        );
    }

    println!("\n* TFLOPS computed as 2*M*N*K per CTA × #CTAs / time. Doesn't include the");
    println!("  S_init add (extra D*D adds per CTA, 16K ops, ~negligible).");
}
