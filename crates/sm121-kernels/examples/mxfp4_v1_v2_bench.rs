//! v1 vs v2 of `gemm_mxfp4_grouped_mma` (gpt-oss-120b path).
//! v2: vectorized FP4 loads (1 v2.b32 = 8 bytes = 16 FP4) + UE8M0 scale hoist.
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

    let configs = [
        ("DSv3-small", 16u32, 32u32, 1024u32, 1024u32),
        ("Mixtral-FFN", 8, 64, 4096, 14336),
        ("GDN-hybrid", 32, 64, 2048, 2048),
        ("LongK-test", 8, 32, 8192, 1024),
    ];

    println!(
        "{:<14} {:<14} {:<14} {:<10} {:<10}",
        "config", "v1_us", "v2_us", "speedup", "v2_TFLOPS"
    );
    println!("{}", "-".repeat(64));

    for (name, num_experts, tokens_per_expert, k, n) in configs {
        let total_tokens = num_experts * tokens_per_expert;
        let a = random_bf16((total_tokens * k) as usize, 0x1111);
        let b: Vec<u8> = (0..(num_experts * n * k / 2) as usize)
            .map(|i| ((i.wrapping_mul(37) + 11) % 256) as u8)
            .collect();
        let scales: Vec<u8> = vec![127; (num_experts * n * (k / 32)) as usize];
        let offsets: Vec<u32> = (0..=num_experts).map(|e| e * tokens_per_expert).collect();

        let a_dev = stream.memcpy_stod(&a).unwrap();
        let b_dev = stream.memcpy_stod(&b).unwrap();
        let s_dev = stream.memcpy_stod(&scales).unwrap();
        let off_dev = stream.memcpy_stod(&offsets).unwrap();
        let mut c_dev = stream
            .alloc_zeros::<u16>((total_tokens * n) as usize)
            .unwrap();

        for _ in 0..3 {
            moe::gemm_mxfp4_grouped_mma(
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
            moe::gemm_mxfp4_grouped_mma_v2(
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
        let t_v1 = unsafe {
            time_fn(
                stream_raw,
                || {
                    moe::gemm_mxfp4_grouped_mma(
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
        let t_v2 = unsafe {
            time_fn(
                stream_raw,
                || {
                    moe::gemm_mxfp4_grouped_mma_v2(
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
        let v2_tflops = flops / (t_v2 as f64 * 1e-3) / 1e12;
        let speedup = t_v1 / t_v2;

        println!(
            "{:<14} {:<14.1} {:<14.1} {:<10.2} {:<10.2}",
            name,
            t_v1 * 1000.0,
            t_v2 * 1000.0,
            speedup,
            v2_tflops
        );
    }
}
