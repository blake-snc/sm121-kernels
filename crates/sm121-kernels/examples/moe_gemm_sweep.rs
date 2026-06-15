//! Benchmark sweep across all MoE grouped-GEMM variants: BF16, FP8 per-expert,
//! FP8 block-128 (DeepSeek V3), and MXFP8 (FlashInfer/SGLang).
//!
//! Reports μs per call and TFLOPS for a DeepSeek V3-representative shape.
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

fn random_fp8(n: usize, seed: u64) -> Vec<u8> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let b = ((state >> 33) & 0xFF) as u8;
            // Skip NaN patterns 0x7F / 0xFF.
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

    // DeepSeek V3-representative shape.
    let num_experts: u32 = 16;
    let tokens_per_expert: u32 = 32;
    let k: u32 = 1024;
    let n: u32 = 1024;
    let total_tokens = num_experts * tokens_per_expert;

    let a_bf16 = random_bf16((total_tokens * k) as usize, 0x1111);
    let b_bf16 = random_bf16((num_experts * k * n) as usize, 0x2222);
    let b_fp8 = random_fp8((num_experts * n * k) as usize, 0x3333);
    let scales_per_expert: Vec<f32> = vec![0.5f32; num_experts as usize];
    let scales_block128: Vec<f32> = vec![0.5f32; (num_experts * n * (k / 128)) as usize];
    let scales_mxfp8: Vec<u8> = vec![127u8; (num_experts * n * (k / 32)) as usize];
    // FP4 formats: B is nibble-packed (2 FP4 per byte) → half the bytes.
    let b_fp4: Vec<u8> = (0..(num_experts * n * k / 2) as usize)
        .map(|i| ((i.wrapping_mul(37) + 11) % 256) as u8)
        .collect();
    let scales_nvfp4: Vec<f32> = vec![1.0f32; (num_experts * n * (k / 16)) as usize];
    let scales_mxfp4: Vec<u8> = vec![127u8; (num_experts * n * (k / 32)) as usize];
    let offsets: Vec<u32> = (0..=num_experts).map(|e| e * tokens_per_expert).collect();

    let a_dev = stream.memcpy_stod(&a_bf16).unwrap();
    let b_bf16_dev = stream.memcpy_stod(&b_bf16).unwrap();
    let b_fp8_dev = stream.memcpy_stod(&b_fp8).unwrap();
    let sp_dev = stream.memcpy_stod(&scales_per_expert).unwrap();
    let sb_dev = stream.memcpy_stod(&scales_block128).unwrap();
    let sm_dev = stream.memcpy_stod(&scales_mxfp8).unwrap();
    let b_fp4_dev = stream.memcpy_stod(&b_fp4).unwrap();
    let snvfp4_dev = stream.memcpy_stod(&scales_nvfp4).unwrap();
    let smxfp4_dev = stream.memcpy_stod(&scales_mxfp4).unwrap();
    let off_dev = stream.memcpy_stod(&offsets).unwrap();
    let mut c_dev = stream
        .alloc_zeros::<u16>((total_tokens * n) as usize)
        .unwrap();

    let flops = 2.0 * total_tokens as f64 * n as f64 * k as f64;
    let iters = 30;

    println!("DeepSeek V3-shape: E{num_experts} M/exp={tokens_per_expert} K={k} N={n}\n");
    println!("{:<32} {:<12} {:<10}", "kernel", "μs", "TFLOPS");
    println!("{}", "-".repeat(56));

    // Warmup then time each variant.
    for _ in 0..3 {
        moe::gemm_bf16_grouped_mma(
            &ctx,
            &stream,
            &a_dev,
            &b_bf16_dev,
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
    let t_bf16 = unsafe {
        time_fn(
            stream_raw,
            || {
                moe::gemm_bf16_grouped_mma(
                    &ctx,
                    &stream,
                    &a_dev,
                    &b_bf16_dev,
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
    println!(
        "{:<32} {:<12.1} {:<10.2}",
        "gemm_bf16_grouped_mma",
        t_bf16 * 1000.0,
        flops / (t_bf16 as f64 * 1e-3) / 1e12
    );

    for _ in 0..3 {
        moe::gemm_fp8_grouped_mma(
            &ctx,
            &stream,
            &a_dev,
            &b_fp8_dev,
            &sp_dev,
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
                moe::gemm_fp8_grouped_mma(
                    &ctx,
                    &stream,
                    &a_dev,
                    &b_fp8_dev,
                    &sp_dev,
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
    println!(
        "{:<32} {:<12.1} {:<10.2}",
        "gemm_fp8_grouped_mma",
        t_fp8 * 1000.0,
        flops / (t_fp8 as f64 * 1e-3) / 1e12
    );

    for _ in 0..3 {
        moe::gemm_fp8_block128_grouped_mma(
            &ctx,
            &stream,
            &a_dev,
            &b_fp8_dev,
            &sb_dev,
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
    let t_b128 = unsafe {
        time_fn(
            stream_raw,
            || {
                moe::gemm_fp8_block128_grouped_mma(
                    &ctx,
                    &stream,
                    &a_dev,
                    &b_fp8_dev,
                    &sb_dev,
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
    println!(
        "{:<32} {:<12.1} {:<10.2}",
        "gemm_fp8_block128_grouped_mma",
        t_b128 * 1000.0,
        flops / (t_b128 as f64 * 1e-3) / 1e12
    );

    for _ in 0..3 {
        moe::gemm_mxfp8_grouped_mma(
            &ctx,
            &stream,
            &a_dev,
            &b_fp8_dev,
            &sm_dev,
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
    let t_mx = unsafe {
        time_fn(
            stream_raw,
            || {
                moe::gemm_mxfp8_grouped_mma(
                    &ctx,
                    &stream,
                    &a_dev,
                    &b_fp8_dev,
                    &sm_dev,
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
    println!(
        "{:<32} {:<12.1} {:<10.2}",
        "gemm_mxfp8_grouped_mma",
        t_mx * 1000.0,
        flops / (t_mx as f64 * 1e-3) / 1e12
    );

    for _ in 0..3 {
        moe::gemm_nvfp4_grouped_mma(
            &ctx,
            &stream,
            &a_dev,
            &b_fp4_dev,
            &snvfp4_dev,
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
    let t_nv = unsafe {
        time_fn(
            stream_raw,
            || {
                moe::gemm_nvfp4_grouped_mma(
                    &ctx,
                    &stream,
                    &a_dev,
                    &b_fp4_dev,
                    &snvfp4_dev,
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
    println!(
        "{:<32} {:<12.1} {:<10.2}",
        "gemm_nvfp4_grouped_mma",
        t_nv * 1000.0,
        flops / (t_nv as f64 * 1e-3) / 1e12
    );

    for _ in 0..3 {
        moe::gemm_mxfp4_grouped_mma(
            &ctx,
            &stream,
            &a_dev,
            &b_fp4_dev,
            &smxfp4_dev,
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
    let t_mx4 = unsafe {
        time_fn(
            stream_raw,
            || {
                moe::gemm_mxfp4_grouped_mma(
                    &ctx,
                    &stream,
                    &a_dev,
                    &b_fp4_dev,
                    &smxfp4_dev,
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
    println!(
        "{:<32} {:<12.1} {:<10.2}",
        "gemm_mxfp4_grouped_mma",
        t_mx4 * 1000.0,
        flops / (t_mx4 as f64 * 1e-3) / 1e12
    );

    println!("\nAll kernels use BF16 MMA internally; quantized formats decode to");
    println!("BF16 in SMEM with the appropriate per-block scale applied.");
}
