//! Compare scalar vs MMA FP8 block-128 grouped GEMM (DeepSeek V3 path).
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

    // Configs: (num_experts, tokens_per_expert, k, n) — k must be multiple of 128.
    let configs = [
        (8u32, 32u32, 128u32, 128u32),
        (8, 64, 512, 512),
        (16, 32, 1024, 1024), // DeepSeek V3-ish
        (32, 64, 2048, 2048), // Large MoE
    ];

    println!(
        "{:<18} {:<8} {:<12} {:<12} {:<12} {:<10}",
        "config", "M/exp", "scalar_us", "mma_us", "speedup", "mma_TFLOPS"
    );

    for (num_experts, tokens_per_expert, k, n) in configs {
        let total_tokens = num_experts * tokens_per_expert;
        let a = random_bf16((total_tokens * k) as usize, 0x1111);
        // Restrict FP8 range to avoid NaN bytes
        let b: Vec<u8> = (0..(num_experts * n * k) as usize)
            .map(|i| {
                let b = ((i.wrapping_mul(17) + 3) % 256) as u8;
                if (b & 0x7F) == 0x7F {
                    b ^ 0x10
                } else {
                    b
                }
            })
            .collect();
        let scales: Vec<f32> = vec![0.5f32; (num_experts * n * (k / 128)) as usize];

        let mut offsets = vec![0u32; num_experts as usize + 1];
        for i in 0..num_experts as usize {
            offsets[i + 1] = offsets[i] + tokens_per_expert;
        }

        let a_dev = stream.memcpy_stod(&a).unwrap();
        let b_dev = stream.memcpy_stod(&b).unwrap();
        let s_dev = stream.memcpy_stod(&scales).unwrap();
        let off_dev = stream.memcpy_stod(&offsets).unwrap();
        let mut c_dev = stream
            .alloc_zeros::<u16>((total_tokens * n) as usize)
            .unwrap();

        // Warmup
        for _ in 0..3 {
            moe::gemm_fp8_block128_grouped(
                &ctx,
                &stream,
                &a_dev,
                &b_dev,
                &s_dev,
                &mut c_dev,
                &off_dev,
                num_experts,
                tokens_per_expert,
                n,
                k,
            )
            .unwrap();
            moe::gemm_fp8_block128_grouped_mma(
                &ctx,
                &stream,
                &a_dev,
                &b_dev,
                &s_dev,
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

        let iters = 30;
        let scalar_ms = unsafe {
            time_fn(
                stream_raw,
                || {
                    moe::gemm_fp8_block128_grouped(
                        &ctx,
                        &stream,
                        &a_dev,
                        &b_dev,
                        &s_dev,
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
        let mma_ms = unsafe {
            time_fn(
                stream_raw,
                || {
                    moe::gemm_fp8_block128_grouped_mma(
                        &ctx,
                        &stream,
                        &a_dev,
                        &b_dev,
                        &s_dev,
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
        let mma_tflops = flops / (mma_ms as f64 * 1e-3) / 1e12;
        let speedup = scalar_ms / mma_ms;

        println!(
            "E{num_experts:>2} K{k:<4} N{n:<4}    {tokens_per_expert:<8} {:<12.1} {:<12.1} {:<12.2} {:<10.2}",
            scalar_ms * 1000.0, mma_ms * 1000.0, speedup, mma_tflops,
        );
    }
}
