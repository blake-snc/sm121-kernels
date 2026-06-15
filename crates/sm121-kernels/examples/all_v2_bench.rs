//! v1 vs v2 sweep for all 5 quantized grouped GEMMs at DSv3 + GDN-hybrid shapes.
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

    let shapes = [
        ("DSv3", 16u32, 32u32, 1024u32, 1024u32),
        ("GDN-hybrid", 32, 64, 2048, 2048),
    ];

    for (name, e, m, k, n) in shapes {
        let total_tokens = e * m;
        let flops = 2.0 * total_tokens as f64 * n as f64 * k as f64;
        let a = random_bf16((total_tokens * k) as usize, 1);
        let b_fp8: Vec<u8> = (0..(e * n * k) as usize)
            .map(|i| {
                let b = ((i.wrapping_mul(17) + 3) % 256) as u8;
                if (b & 0x7F) == 0x7F {
                    b ^ 0x10
                } else {
                    b
                }
            })
            .collect();
        let b_fp4: Vec<u8> = (0..(e * n * k / 2) as usize)
            .map(|i| ((i.wrapping_mul(37) + 11) % 256) as u8)
            .collect();
        let s_fp32_per_expert: Vec<f32> = vec![0.5; e as usize];
        let s_mxfp8: Vec<u8> = vec![127; (e * n * (k / 32)) as usize];
        let s_nvfp4_fp8: Vec<u8> = vec![0x38; (e * n * (k / 16)) as usize];
        let offsets: Vec<u32> = (0..=e).map(|i| i * m).collect();

        let a_dev = stream.memcpy_stod(&a).unwrap();
        let bfp8_dev = stream.memcpy_stod(&b_fp8).unwrap();
        let bfp4_dev = stream.memcpy_stod(&b_fp4).unwrap();
        let spe_dev = stream.memcpy_stod(&s_fp32_per_expert).unwrap();
        let smx8_dev = stream.memcpy_stod(&s_mxfp8).unwrap();
        let snv8_dev = stream.memcpy_stod(&s_nvfp4_fp8).unwrap();
        let off_dev = stream.memcpy_stod(&offsets).unwrap();
        let mut c_dev = stream
            .alloc_zeros::<u16>((total_tokens * n) as usize)
            .unwrap();

        let iters = 30;
        macro_rules! bench {
            ($name:expr, $call:expr) => {{
                for _ in 0..3 {
                    $call;
                }
                stream.synchronize().unwrap();
                let ms = unsafe { time_fn(stream_raw, || $call, iters) };
                let tflops = flops / (ms as f64 * 1e-3) / 1e12;
                println!("  {:<32} {:<8.1} {:<8.2}", $name, ms * 1000.0, tflops);
            }};
        }

        println!("\n=== {} (E{} M/exp={} K={} N={}) ===", name, e, m, k, n);
        println!("  {:<32} {:<8} {:<8}", "kernel", "us", "TFLOPS");

        bench!(
            "fp8_grouped_v1",
            moe::gemm_fp8_grouped_mma(
                &ctx, &stream, &a_dev, &bfp8_dev, &spe_dev, &mut c_dev, &off_dev, e, m, n, k
            )
            .unwrap()
        );
        bench!(
            "fp8_grouped_v2",
            moe::gemm_fp8_grouped_mma_v2(
                &ctx, &stream, &a_dev, &bfp8_dev, &spe_dev, &mut c_dev, &off_dev, e, m, n, k
            )
            .unwrap()
        );

        bench!(
            "mxfp8_v1",
            moe::gemm_mxfp8_grouped_mma(
                &ctx, &stream, &a_dev, &bfp8_dev, &smx8_dev, &mut c_dev, &off_dev, e, m, n, k
            )
            .unwrap()
        );
        bench!(
            "mxfp8_v2",
            moe::gemm_mxfp8_grouped_mma_v2(
                &ctx, &stream, &a_dev, &bfp8_dev, &smx8_dev, &mut c_dev, &off_dev, e, m, n, k
            )
            .unwrap()
        );

        bench!(
            "nvfp4_fp8scale_v1",
            moe::gemm_nvfp4_fp8scale_grouped_mma(
                &ctx, &stream, &a_dev, &bfp4_dev, &snv8_dev, &mut c_dev, &off_dev, e, m, n, k
            )
            .unwrap()
        );
        bench!(
            "nvfp4_fp8scale_v2",
            moe::gemm_nvfp4_fp8scale_grouped_mma_v2(
                &ctx, &stream, &a_dev, &bfp4_dev, &snv8_dev, &mut c_dev, &off_dev, e, m, n, k
            )
            .unwrap()
        );
    }
}
