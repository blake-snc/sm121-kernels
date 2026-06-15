//! v1 (BF16-MMA-on-FP8-KV hybrid) vs v2 (true FP8 MMA) MLA decode benchmark.
//! Measures the actual perf delta from halving QK MMA count.
use cudarc::driver::sys::{cuEventCreate, cuEventElapsedTime, cuEventRecord, cuEventSynchronize};
use half::bf16;
use sm121_kernels::{attention, device};

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

fn random_fp8(n: usize, seed: u64) -> Vec<u8> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let b = ((state >> 33) & 0xFF) as u8;
            if (b & 0x7F) == 0x7F {
                b ^ 0x10
            } else {
                b
            }
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
        ("Skv=128", 1u32, 16u32, 128u32),
        ("Skv=512", 1, 16, 512),
        ("Skv=2048", 1, 16, 2048),
    ];

    println!(
        "{:<14} {:<10} {:<10} {:<10} {:<10}",
        "config", "v1_us", "v2_us", "v3_us", "best"
    );
    println!("{}", "-".repeat(60));

    for (name, batch, num_heads, seq_kv) in configs {
        let q_bf16 = random_bf16((batch * num_heads * 512) as usize, 1);
        let q_r_bf16 = random_bf16((batch * num_heads * 64) as usize, 2);
        let q_fp8 = random_fp8((batch * num_heads * 512) as usize, 1);
        let q_r_fp8 = random_fp8((batch * num_heads * 64) as usize, 2);
        let c_kv = random_fp8((batch * seq_kv * 512) as usize, 3);
        let k_rope = random_fp8((batch * seq_kv * 64) as usize, 4);

        let q_bf16_dev = stream.memcpy_stod(&q_bf16).unwrap();
        let q_r_bf16_dev = stream.memcpy_stod(&q_r_bf16).unwrap();
        let q_fp8_dev = stream.memcpy_stod(&q_fp8).unwrap();
        let q_r_fp8_dev = stream.memcpy_stod(&q_r_fp8).unwrap();
        let c_kv_dev = stream.memcpy_stod(&c_kv).unwrap();
        let k_rope_dev = stream.memcpy_stod(&k_rope).unwrap();
        let mut o_dev = stream
            .alloc_zeros::<u16>((batch * num_heads * 512) as usize)
            .unwrap();

        let scale = 0.044194f32;
        let q_scale = 0.5f32;
        let kv_scale = 0.25f32;

        // Warmup
        for _ in 0..3 {
            attention::mla_decode_fp8kv_mma(
                &ctx,
                &stream,
                &q_bf16_dev,
                &q_r_bf16_dev,
                &c_kv_dev,
                &k_rope_dev,
                &mut o_dev,
                batch,
                num_heads,
                seq_kv,
                scale,
                kv_scale,
            )
            .unwrap();
            attention::fa_fp8_mla_decode_mma_v2(
                &ctx,
                &stream,
                &q_fp8_dev,
                &q_r_fp8_dev,
                &c_kv_dev,
                &k_rope_dev,
                &mut o_dev,
                batch,
                num_heads,
                seq_kv,
                scale,
                q_scale,
                kv_scale,
            )
            .unwrap();
        }
        stream.synchronize().unwrap();

        let iters = 50;
        let t_v1 = unsafe {
            time_fn(
                raw,
                || {
                    attention::mla_decode_fp8kv_mma(
                        &ctx,
                        &stream,
                        &q_bf16_dev,
                        &q_r_bf16_dev,
                        &c_kv_dev,
                        &k_rope_dev,
                        &mut o_dev,
                        batch,
                        num_heads,
                        seq_kv,
                        scale,
                        kv_scale,
                    )
                    .unwrap();
                },
                iters,
            )
        };

        let t_v2 = unsafe {
            time_fn(
                raw,
                || {
                    attention::fa_fp8_mla_decode_mma_v2(
                        &ctx,
                        &stream,
                        &q_fp8_dev,
                        &q_r_fp8_dev,
                        &c_kv_dev,
                        &k_rope_dev,
                        &mut o_dev,
                        batch,
                        num_heads,
                        seq_kv,
                        scale,
                        q_scale,
                        kv_scale,
                    )
                    .unwrap();
                },
                iters,
            )
        };

        for _ in 0..3 {
            attention::fa_fp8_mla_decode_mma_v3(
                &ctx,
                &stream,
                &q_fp8_dev,
                &q_r_fp8_dev,
                &c_kv_dev,
                &k_rope_dev,
                &mut o_dev,
                batch,
                num_heads,
                seq_kv,
                scale,
                q_scale,
                kv_scale,
            )
            .unwrap();
        }
        stream.synchronize().unwrap();
        let t_v3 = unsafe {
            time_fn(
                raw,
                || {
                    attention::fa_fp8_mla_decode_mma_v3(
                        &ctx,
                        &stream,
                        &q_fp8_dev,
                        &q_r_fp8_dev,
                        &c_kv_dev,
                        &k_rope_dev,
                        &mut o_dev,
                        batch,
                        num_heads,
                        seq_kv,
                        scale,
                        q_scale,
                        kv_scale,
                    )
                    .unwrap();
                },
                iters,
            )
        };

        let best = if t_v3 < t_v2 && t_v3 < t_v1 {
            "v3"
        } else if t_v2 < t_v1 {
            "v2"
        } else {
            "v1"
        };
        println!(
            "{:<14} {:<10.1} {:<10.1} {:<10.1} {:<10}",
            name,
            t_v1 * 1000.0,
            t_v2 * 1000.0,
            t_v3 * 1000.0,
            best
        );
    }
}
