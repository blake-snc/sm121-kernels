//! Quick scaling benchmark for current GEMM v1 to know where to focus v2.
use cudarc::driver::sys::{cuEventCreate, cuEventElapsedTime, cuEventRecord, cuEventSynchronize};
use sm121_kernels::{device, gemm};

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

    let configs: &[(u32, u32, u32)] = &[
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (1024, 4096, 4096), // typical attention head shape
        (4096, 4096, 1024),
    ];

    println!(
        "{:<18} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>7}",
        "shape", "v1(us)", "v2(us)", "v3(us)", "v1 TF", "v2 TF", "v3 TF", "v3/v1"
    );
    println!("{}", "-".repeat(85));

    for &(m, n, k) in configs {
        let a = stream.alloc_zeros::<u16>((m * k) as usize).unwrap();
        let b = stream.alloc_zeros::<u16>((k * n) as usize).unwrap();
        let mut c = stream.alloc_zeros::<u16>((m * n) as usize).unwrap();

        let v3_ok = n.is_multiple_of(128);

        for _ in 0..3 {
            gemm::gemm_bf16_mma(&ctx, &stream, &a, &b, &mut c, m, n, k).unwrap();
            gemm::gemm_bf16_mma_v2(&ctx, &stream, &a, &b, &mut c, m, n, k).unwrap();
            if v3_ok {
                gemm::gemm_bf16_mma_v3(&ctx, &stream, &a, &b, &mut c, m, n, k).unwrap();
            }
        }
        stream.synchronize().unwrap();

        let iters = if m * n * k > (1024u32.pow(3)) { 10 } else { 50 };
        let ms_v1 = unsafe {
            time_fn(
                raw,
                || {
                    gemm::gemm_bf16_mma(&ctx, &stream, &a, &b, &mut c, m, n, k).unwrap();
                },
                iters,
            )
        };
        let ms_v2 = unsafe {
            time_fn(
                raw,
                || {
                    gemm::gemm_bf16_mma_v2(&ctx, &stream, &a, &b, &mut c, m, n, k).unwrap();
                },
                iters,
            )
        };
        let ms_v3 = if v3_ok {
            unsafe {
                time_fn(
                    raw,
                    || {
                        gemm::gemm_bf16_mma_v3(&ctx, &stream, &a, &b, &mut c, m, n, k).unwrap();
                    },
                    iters,
                )
            }
        } else {
            f32::NAN
        };

        let flops = 2.0 * m as f64 * n as f64 * k as f64;
        let tf_v1 = flops / (ms_v1 as f64 * 1e-3) / 1e12;
        let tf_v2 = flops / (ms_v2 as f64 * 1e-3) / 1e12;
        let tf_v3 = flops / (ms_v3 as f64 * 1e-3) / 1e12;
        let speedup = ms_v1 / ms_v3;

        println!(
            "{:<18} {:>9.1} {:>9.1} {:>9.1} {:>9.2} {:>9.2} {:>9.2} {:>7.2}",
            format!("{}x{}x{}", m, n, k),
            ms_v1 * 1000.0,
            ms_v2 * 1000.0,
            ms_v3 * 1000.0,
            tf_v1,
            tf_v2,
            tf_v3,
            speedup
        );
    }
}
