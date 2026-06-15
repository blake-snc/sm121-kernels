//! Multi-shape MoE grouped-GEMM sweep across the full kernel family.
//!
//! Measures TFLOPS at 4 production-relevant shapes:
//!   - Tiny:    E=8, M/exp=8,  K=512,   N=512    (small batch, e.g. low-traffic)
//!   - Mixtral: E=8, M/exp=64, K=4096,  N=14336  (Mixtral-8x7B FFN dim)
//!   - DSv3:    E=16, M/exp=32, K=1024, N=1024   (DeepSeek V3 sub-tile)
//!   - GDN-hybrid:  E=32, M/exp=64, K=2048, N=2048   (GDN-hybrid MoE inner)
//!
//! Reports each kernel × shape, plus identifies the bottleneck pattern.

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

struct Shape {
    name: &'static str,
    num_experts: u32,
    tokens_per_expert: u32,
    k: u32,
    n: u32,
}

fn main() {
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();
    let stream_raw = stream.cu_stream();

    let shapes = [
        Shape {
            name: "Tiny",
            num_experts: 8,
            tokens_per_expert: 8,
            k: 512,
            n: 512,
        },
        Shape {
            name: "Mixtral",
            num_experts: 8,
            tokens_per_expert: 64,
            k: 4096,
            n: 14336,
        },
        Shape {
            name: "DSv3",
            num_experts: 16,
            tokens_per_expert: 32,
            k: 1024,
            n: 1024,
        },
        Shape {
            name: "GDN-hybrid",
            num_experts: 32,
            tokens_per_expert: 64,
            k: 2048,
            n: 2048,
        },
    ];

    let iters = 30;

    println!(
        "{:<10} | {:<22} | {:<8} | {:<8}",
        "shape", "kernel", "μs", "TFLOPS"
    );
    println!("{}", "-".repeat(60));

    for sh in &shapes {
        let total_tokens = sh.num_experts * sh.tokens_per_expert;
        let flops = 2.0 * total_tokens as f64 * sh.n as f64 * sh.k as f64;

        let a_bf16 = random_bf16((total_tokens * sh.k) as usize, 0x1111);
        let b_bf16 = random_bf16((sh.num_experts * sh.k * sh.n) as usize, 0x2222);
        let b_fp8 = random_fp8((sh.num_experts * sh.n * sh.k) as usize, 0x3333);
        let b_fp4: Vec<u8> = (0..(sh.num_experts * sh.n * sh.k / 2) as usize)
            .map(|i| ((i.wrapping_mul(37) + 11) % 256) as u8)
            .collect();
        let scales_fp8_per_expert: Vec<f32> = vec![0.5; sh.num_experts as usize];
        let scales_fp32_block128: Vec<f32> =
            vec![0.5; (sh.num_experts * sh.n * (sh.k / 128).max(1)) as usize];
        let scales_mxfp8: Vec<u8> = vec![127; (sh.num_experts * sh.n * (sh.k / 32)) as usize];
        let scales_nvfp4_fp32: Vec<f32> = vec![1.0; (sh.num_experts * sh.n * (sh.k / 16)) as usize];
        let scales_nvfp4_fp8: Vec<u8> = vec![0x38; (sh.num_experts * sh.n * (sh.k / 16)) as usize];
        let scales_mxfp4: Vec<u8> = vec![127; (sh.num_experts * sh.n * (sh.k / 32)) as usize];

        let offsets: Vec<u32> = (0..=sh.num_experts)
            .map(|e| e * sh.tokens_per_expert)
            .collect();

        let a_dev = stream.memcpy_stod(&a_bf16).unwrap();
        let b_bf16_dev = stream.memcpy_stod(&b_bf16).unwrap();
        let b_fp8_dev = stream.memcpy_stod(&b_fp8).unwrap();
        let b_fp4_dev = stream.memcpy_stod(&b_fp4).unwrap();
        let s_pe_dev = stream.memcpy_stod(&scales_fp8_per_expert).unwrap();
        let s_b128_dev = stream.memcpy_stod(&scales_fp32_block128).unwrap();
        let s_mx8_dev = stream.memcpy_stod(&scales_mxfp8).unwrap();
        let s_nv32_dev = stream.memcpy_stod(&scales_nvfp4_fp32).unwrap();
        let s_nv8_dev = stream.memcpy_stod(&scales_nvfp4_fp8).unwrap();
        let s_mx4_dev = stream.memcpy_stod(&scales_mxfp4).unwrap();
        let off_dev = stream.memcpy_stod(&offsets).unwrap();
        let mut c_dev = stream
            .alloc_zeros::<u16>((total_tokens * sh.n) as usize)
            .unwrap();

        macro_rules! bench {
            ($name:expr, $call:expr) => {{
                for _ in 0..3 {
                    $call;
                }
                stream.synchronize().unwrap();
                let ms = unsafe {
                    time_fn(
                        stream_raw,
                        || {
                            $call;
                        },
                        iters,
                    )
                };
                let tflops = flops / (ms as f64 * 1e-3) / 1e12;
                println!(
                    "{:<10} | {:<22} | {:<8.1} | {:<8.2}",
                    sh.name,
                    $name,
                    ms * 1000.0,
                    tflops
                );
            }};
        }

        bench!(
            "bf16_grouped_mma",
            moe::gemm_bf16_grouped_mma(
                &ctx,
                &stream,
                &a_dev,
                &b_bf16_dev,
                &mut c_dev,
                &off_dev,
                sh.num_experts,
                sh.tokens_per_expert,
                sh.n,
                sh.k
            )
            .unwrap()
        );

        bench!(
            "fp8_grouped_mma",
            moe::gemm_fp8_grouped_mma(
                &ctx,
                &stream,
                &a_dev,
                &b_fp8_dev,
                &s_pe_dev,
                &mut c_dev,
                &off_dev,
                sh.num_experts,
                sh.tokens_per_expert,
                sh.n,
                sh.k
            )
            .unwrap()
        );

        if sh.k % 128 == 0 {
            bench!(
                "fp8_block128_grouped_mma",
                moe::gemm_fp8_block128_grouped_mma(
                    &ctx,
                    &stream,
                    &a_dev,
                    &b_fp8_dev,
                    &s_b128_dev,
                    &mut c_dev,
                    &off_dev,
                    sh.num_experts,
                    sh.tokens_per_expert,
                    sh.n,
                    sh.k
                )
                .unwrap()
            );
        }

        bench!(
            "mxfp8_grouped_mma",
            moe::gemm_mxfp8_grouped_mma(
                &ctx,
                &stream,
                &a_dev,
                &b_fp8_dev,
                &s_mx8_dev,
                &mut c_dev,
                &off_dev,
                sh.num_experts,
                sh.tokens_per_expert,
                sh.n,
                sh.k
            )
            .unwrap()
        );

        bench!(
            "nvfp4_grouped_mma",
            moe::gemm_nvfp4_grouped_mma(
                &ctx,
                &stream,
                &a_dev,
                &b_fp4_dev,
                &s_nv32_dev,
                &mut c_dev,
                &off_dev,
                sh.num_experts,
                sh.tokens_per_expert,
                sh.n,
                sh.k
            )
            .unwrap()
        );

        bench!(
            "nvfp4_fp8scale_mma",
            moe::gemm_nvfp4_fp8scale_grouped_mma(
                &ctx,
                &stream,
                &a_dev,
                &b_fp4_dev,
                &s_nv8_dev,
                &mut c_dev,
                &off_dev,
                sh.num_experts,
                sh.tokens_per_expert,
                sh.n,
                sh.k
            )
            .unwrap()
        );

        bench!(
            "mxfp4_grouped_mma",
            moe::gemm_mxfp4_grouped_mma(
                &ctx,
                &stream,
                &a_dev,
                &b_fp4_dev,
                &s_mx4_dev,
                &mut c_dev,
                &off_dev,
                sh.num_experts,
                sh.tokens_per_expert,
                sh.n,
                sh.k
            )
            .unwrap()
        );

        println!();
    }
}
