//! Benchmark: NVFP4 grouped GEMM with FP32 scales vs FP8 E4M3 scales.
//! Same compute path (BF16 intermediate), only scale format differs.
//! Goal: measure the memory-traffic win from 4× smaller scales.
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
            let f = (((state >> 33) as u32 as f32) / u32::MAX as f32 - 0.5) * 2.0;
            bf16::from_f32(f).to_bits()
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

    let num_experts: u32 = 16;
    let tokens_per_expert: u32 = 32;
    let k: u32 = 1024;
    let n: u32 = 1024;
    let total_tokens = num_experts * tokens_per_expert;

    let a = random_bf16((total_tokens * k) as usize, 0x1111);
    let b_fp4: Vec<u8> = (0..(num_experts * n * k / 2) as usize)
        .map(|i| ((i.wrapping_mul(37) + 11) % 256) as u8)
        .collect();
    let scales_fp32: Vec<f32> = vec![1.0f32; (num_experts * n * (k / 16)) as usize];
    let scales_fp8: Vec<u8> = vec![0x38u8; (num_experts * n * (k / 16)) as usize]; // 1.0 in FP8

    let mut offsets = vec![0u32; num_experts as usize + 1];
    for i in 0..num_experts as usize {
        offsets[i + 1] = offsets[i] + tokens_per_expert;
    }

    let a_dev = stream.memcpy_stod(&a).unwrap();
    let b_dev = stream.memcpy_stod(&b_fp4).unwrap();
    let s32_dev = stream.memcpy_stod(&scales_fp32).unwrap();
    let s8_dev = stream.memcpy_stod(&scales_fp8).unwrap();
    let off_dev = stream.memcpy_stod(&offsets).unwrap();
    let mut c_dev = stream
        .alloc_zeros::<u16>((total_tokens * n) as usize)
        .unwrap();

    let iters = 50;

    // FP32-scale variant warmup + time
    for _ in 0..5 {
        moe::gemm_nvfp4_grouped_mma(
            &ctx,
            &stream,
            &a_dev,
            &b_dev,
            &s32_dev,
            &mut c_dev,
            &off_dev,
            num_experts,
            tokens_per_expert,
            n,
            k,
        )
        .unwrap();
    }
    stream.synchronize().unwrap();
    let t_fp32 = unsafe {
        time_fn(
            stream_raw,
            || {
                moe::gemm_nvfp4_grouped_mma(
                    &ctx,
                    &stream,
                    &a_dev,
                    &b_dev,
                    &s32_dev,
                    &mut c_dev,
                    &off_dev,
                    num_experts,
                    tokens_per_expert,
                    n,
                    k,
                )
                .unwrap();
            },
            iters,
        )
    };

    // FP8-scale variant warmup + time
    for _ in 0..5 {
        moe::gemm_nvfp4_fp8scale_grouped_mma(
            &ctx,
            &stream,
            &a_dev,
            &b_dev,
            &s8_dev,
            &mut c_dev,
            &off_dev,
            num_experts,
            tokens_per_expert,
            n,
            k,
        )
        .unwrap();
    }
    stream.synchronize().unwrap();
    let t_fp8 = unsafe {
        time_fn(
            stream_raw,
            || {
                moe::gemm_nvfp4_fp8scale_grouped_mma(
                    &ctx,
                    &stream,
                    &a_dev,
                    &b_dev,
                    &s8_dev,
                    &mut c_dev,
                    &off_dev,
                    num_experts,
                    tokens_per_expert,
                    n,
                    k,
                )
                .unwrap();
            },
            iters,
        )
    };

    let flops = 2.0 * total_tokens as f64 * n as f64 * k as f64;
    let t_fp32_tflops = flops / (t_fp32 as f64 * 1e-3) / 1e12;
    let t_fp8_tflops = flops / (t_fp8 as f64 * 1e-3) / 1e12;

    println!("Shape: E={num_experts} M/exp={tokens_per_expert} K={k} N={n}\n");
    println!("{:<38} {:<10} {:<10}", "variant", "μs", "TFLOPS");
    println!("{}", "-".repeat(60));
    println!(
        "{:<38} {:<10.1} {:<10.2}",
        "gemm_nvfp4_grouped_mma (FP32 scales)",
        t_fp32 * 1000.0,
        t_fp32_tflops
    );
    println!(
        "{:<38} {:<10.1} {:<10.2}",
        "gemm_nvfp4_fp8scale_grouped_mma",
        t_fp8 * 1000.0,
        t_fp8_tflops
    );
    println!(
        "\nSpeedup (FP8 scales / FP32 scales): {:.2}×",
        t_fp32 / t_fp8
    );
    println!("Scale memory reduction: 4× (1 byte vs 4 bytes per scale)");
}
