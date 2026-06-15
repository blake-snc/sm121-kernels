//! Benchmark fused W8A16 GEMV (1 launch) vs the existing 3-launch path
//! (memset + split-K GEMV + cast) at Gemma-4-typical decode shapes.

use cudarc::driver::sys::{cuEventCreate, cuEventElapsedTime, cuEventRecord, cuEventSynchronize};
use sm121_kernels::{device, gemm};

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

    // Shapes representative of Gemma-4-e4b W8A16 GEMV calls per decode token:
    let configs: &[(&str, u32, u32)] = &[
        (
            "Q proj sliding (N=q_dim_sliding=8192, K=hidden=2560)",
            8192,
            2560,
        ),
        (
            "Q proj full (N=q_dim_full=16384, K=hidden=2560)",
            16384,
            2560,
        ),
        ("K/V proj sliding (N=512, K=2560)", 512, 2560),
        ("K/V proj full (N=1024, K=2560)", 1024, 2560),
        ("o_proj sliding (N=2560, K=8192)", 2560, 8192),
        ("o_proj full (N=2560, K=16384)", 2560, 16384),
        ("gate / up (N=10240, K=2560)", 10240, 2560),
        ("down (N=2560, K=10240)", 2560, 10240),
        ("lm_head fp8 (N=262144, K=2560)", 262144, 2560),
    ];

    println!(
        "{:<60} {:>10} {:>10} {:>10} {:>10}",
        "shape", "split_us", "w16_us", "fused_us", "best"
    );
    println!("{}", "-".repeat(108));

    for (name, n, k) in configs {
        let n = *n;
        let k = *k;
        let scale = 0.0625f32;

        // Allocate buffers.
        let x = stream.alloc_zeros::<u16>(k as usize).unwrap();
        let b = stream.alloc_zeros::<u8>((n * k) as usize).unwrap();
        let mut f32_scratch = stream.alloc_zeros::<f32>(n as usize).unwrap();
        let mut out_split = stream.alloc_zeros::<u16>(n as usize).unwrap();
        let mut out_w16 = stream.alloc_zeros::<u16>(n as usize).unwrap();
        let mut out_fused = stream.alloc_zeros::<u16>(n as usize).unwrap();

        let num_shards = (k / 1024).max(1).min(8);
        let w16_ok = (n & 15) == 0;
        let smem_ok = (k as usize) * 2 <= 99 * 1024;

        // Warmup.
        for _ in 0..3 {
            gemm::gemv_w8a16_split_k_managed(
                &ctx,
                &stream,
                &x,
                &b,
                scale,
                &mut f32_scratch,
                &mut out_split,
                n,
                k,
                num_shards,
            )
            .unwrap();
            if w16_ok {
                gemm::gemv_w8a16_split_k_w16_managed(
                    &ctx,
                    &stream,
                    &x,
                    &b,
                    scale,
                    &mut f32_scratch,
                    &mut out_w16,
                    n,
                    k,
                    num_shards,
                )
                .unwrap();
            }
            if smem_ok {
                gemm::gemv_w8a16_fused_bf16(&ctx, &stream, &x, &b, scale, &mut out_fused, n, k)
                    .unwrap();
            }
        }
        stream.synchronize().ok();

        let iters = 100;
        let split_ms = unsafe {
            time_fn(
                stream_raw,
                || {
                    gemm::gemv_w8a16_split_k_managed(
                        &ctx,
                        &stream,
                        &x,
                        &b,
                        scale,
                        &mut f32_scratch,
                        &mut out_split,
                        n,
                        k,
                        num_shards,
                    )
                    .unwrap();
                },
                iters,
            )
        };

        let w16_ms = if w16_ok {
            unsafe {
                time_fn(
                    stream_raw,
                    || {
                        gemm::gemv_w8a16_split_k_w16_managed(
                            &ctx,
                            &stream,
                            &x,
                            &b,
                            scale,
                            &mut f32_scratch,
                            &mut out_w16,
                            n,
                            k,
                            num_shards,
                        )
                        .unwrap();
                    },
                    iters,
                )
            }
        } else {
            f32::NAN
        };

        let fused_ms = if smem_ok {
            unsafe {
                time_fn(
                    stream_raw,
                    || {
                        gemm::gemv_w8a16_fused_bf16(
                            &ctx,
                            &stream,
                            &x,
                            &b,
                            scale,
                            &mut out_fused,
                            n,
                            k,
                        )
                        .unwrap();
                    },
                    iters,
                )
            }
        } else {
            f32::NAN
        };

        let split_us = split_ms * 1000.0;
        let w16_us_str = if w16_ms.is_nan() {
            "—".to_string()
        } else {
            format!("{:.1}", w16_ms * 1000.0)
        };
        let fused_us_str = if fused_ms.is_nan() {
            "—".to_string()
        } else {
            format!("{:.1}", fused_ms * 1000.0)
        };

        // Identify best.
        let mut best = ("split", split_ms);
        if !w16_ms.is_nan() && w16_ms < best.1 {
            best = ("w16", w16_ms);
        }
        if !fused_ms.is_nan() && fused_ms < best.1 {
            best = ("fused", fused_ms);
        }
        let best_str = format!("{}={:.2}x", best.0, split_ms / best.1);

        println!("{name:<60} {split_us:>10.1} {w16_us_str:>10} {fused_us_str:>10} {best_str:>10}");
    }
}
