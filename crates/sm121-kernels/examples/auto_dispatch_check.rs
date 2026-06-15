//! Verify auto-dispatch picks the right v1/v2 across shapes.
use cudarc::driver::sys::{cuEventCreate, cuEventElapsedTime, cuEventRecord, cuEventSynchronize};
use half::bf16;
use sm121_kernels::{device, moe};

fn random_bf16(n: usize, seed: u64) -> Vec<u16> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            bf16::from_f32((((state >> 33) as u32 as f32) / u32::MAX as f32 - 0.5) * 2.0).to_bits()
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

    let configs = [
        ("DSv3 N=1024", 16u32, 32u32, 1024u32, 1024u32),
        ("GDN-hybrid N=2048", 32, 64, 2048, 2048),
        ("Mixtral N=14336", 8, 64, 4096, 14336),
    ];

    println!(
        "{:<22} {:<10} {:<10} {:<10}",
        "shape", "v1_us", "v2_us", "auto_us"
    );
    println!("{}", "-".repeat(56));

    for (name, e, m, k, n) in configs {
        let total_tokens = e * m;
        let a = random_bf16((total_tokens * k) as usize, 1);
        let b: Vec<u8> = (0..(e * n * k) as usize)
            .map(|i| {
                let b = ((i.wrapping_mul(17) + 3) % 256) as u8;
                if (b & 0x7F) == 0x7F {
                    b ^ 0x10
                } else {
                    b
                }
            })
            .collect();
        let scales: Vec<f32> = vec![0.5; (e * n * (k / 128)) as usize];
        let off: Vec<u32> = (0..=e).map(|i| i * m).collect();
        let a_dev = stream.memcpy_stod(&a).unwrap();
        let b_dev = stream.memcpy_stod(&b).unwrap();
        let s_dev = stream.memcpy_stod(&scales).unwrap();
        let off_dev = stream.memcpy_stod(&off).unwrap();
        let mut c_dev = stream
            .alloc_zeros::<u16>((total_tokens * n) as usize)
            .unwrap();

        for _ in 0..3 {
            moe::gemm_fp8_block128_grouped_mma(
                &ctx, &stream, &a_dev, &b_dev, &s_dev, &mut c_dev, &off_dev, e, m, n, k,
            )
            .unwrap();
            moe::gemm_fp8_block128_grouped_mma_v2(
                &ctx, &stream, &a_dev, &b_dev, &s_dev, &mut c_dev, &off_dev, e, m, n, k,
            )
            .unwrap();
            moe::gemm_fp8_block128_grouped_mma_auto(
                &ctx, &stream, &a_dev, &b_dev, &s_dev, &mut c_dev, &off_dev, e, m, n, k,
            )
            .unwrap();
        }
        stream.synchronize().unwrap();

        let iters = 30;
        let t_v1 = unsafe {
            time_fn(
                raw,
                || {
                    moe::gemm_fp8_block128_grouped_mma(
                        &ctx, &stream, &a_dev, &b_dev, &s_dev, &mut c_dev, &off_dev, e, m, n, k,
                    )
                    .unwrap()
                },
                iters,
            )
        };
        let t_v2 = unsafe {
            time_fn(
                raw,
                || {
                    moe::gemm_fp8_block128_grouped_mma_v2(
                        &ctx, &stream, &a_dev, &b_dev, &s_dev, &mut c_dev, &off_dev, e, m, n, k,
                    )
                    .unwrap()
                },
                iters,
            )
        };
        let t_auto = unsafe {
            time_fn(
                raw,
                || {
                    moe::gemm_fp8_block128_grouped_mma_auto(
                        &ctx, &stream, &a_dev, &b_dev, &s_dev, &mut c_dev, &off_dev, e, m, n, k,
                    )
                    .unwrap()
                },
                iters,
            )
        };

        println!(
            "{:<22} {:<10.1} {:<10.1} {:<10.1}",
            name,
            t_v1 * 1000.0,
            t_v2 * 1000.0,
            t_auto * 1000.0
        );
    }
}
