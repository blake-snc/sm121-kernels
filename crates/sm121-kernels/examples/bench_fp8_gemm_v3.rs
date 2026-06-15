//! Standalone benchmark for FP8 GEMM v3 (and v3.5) to validate A2 PTX coercion
//! hypothesis on a less-optimized kernel than V12c.

use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::{CudaContext, CudaStream};
use sm121_kernels::gemm;

fn die<E: std::fmt::Debug>(e: E) -> anyhow::Error {
    anyhow::anyhow!("{e:?}")
}

fn bench(
    name: &str,
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    m: u32,
    n: u32,
    k: u32,
    mut f: impl FnMut(),
) {
    // Warmup
    for _ in 0..5 {
        f();
    }
    stream.synchronize().unwrap();

    let iters = 200;
    let mut times_us = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        f();
        stream.synchronize().unwrap();
        times_us.push(t0.elapsed().as_secs_f64() * 1.0e6);
    }
    times_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = times_us[iters / 2];
    let mean = times_us.iter().sum::<f64>() / iters as f64;
    let min = times_us[0];
    let max = times_us[iters - 1];

    let flops = 2.0 * m as f64 * n as f64 * k as f64;
    let tflops = flops / (median * 1.0e6);
    let _ = ctx;

    println!(
        "{name:40} M={m:5} N={n:5} K={k:5}  median={median:8.1} us  mean={mean:8.1} us  min={min:8.1} us  max={max:8.1} us  {tflops:6.2} TFLOPS",
    );
}

fn main() -> anyhow::Result<()> {
    let ctx = CudaContext::new(0).map_err(die)?;
    let stream = ctx.default_stream();

    println!("=== FP8 GEMM v3 / v3.5 baseline (for A2 hypothesis test) ===\n");

    for &(m, n, k) in &[
        (512u32, 512u32, 512u32),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
    ] {
        let a = stream.alloc_zeros::<u8>((m * k) as usize).map_err(die)?;
        let b = stream.alloc_zeros::<u8>((k * n) as usize).map_err(die)?;
        let mut c = stream.alloc_zeros::<u16>((m * n) as usize).map_err(die)?;

        bench("FP8 GEMM v3", &ctx, &stream, m, n, k, || {
            gemm::gemm_fp8_mma_v3(&ctx, &stream, &a, &b, &mut c, m, n, k).unwrap();
        });

        bench("FP8 GEMM v3.5", &ctx, &stream, m, n, k, || {
            gemm::gemm_fp8_mma_v3_5(&ctx, &stream, &a, &b, &mut c, m, n, k).unwrap();
        });

        bench("FP8 GEMM v1 (reference)", &ctx, &stream, m, n, k, || {
            gemm::gemm_fp8_mma(&ctx, &stream, &a, &b, &mut c, m, n, k).unwrap();
        });

        println!();
    }

    Ok(())
}
