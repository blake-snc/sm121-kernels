//! Diagnostic — kernel-level latency of gemm_w8a16_grouped_mma at
//! num_experts=1 vs gemm_bf16_mma_v3 at the dense decode shapes the batched
//! M=128 worker uses.
//!
//! Hypothesis: the grouped kernel uses 32×32 tiles with single-warp blocks
//! (~25% SM occupancy at our shapes), while BF16 v3 uses 128×128 tiles with
//! 8 warps (~50% occupancy). Latency-wise, grouped should be slower per call
//! despite reading half the bytes for B.

use anyhow::Result;
use sm121_kernels::{device, gemm, moe, quantization};

fn now_us() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs_f64()
        * 1e6
}

fn time_kernel<F: FnMut() -> Result<()>>(name: &str, warmup: usize, iters: usize, mut f: F) -> f64 {
    for _ in 0..warmup {
        f().unwrap();
    }
    let t0 = now_us();
    for _ in 0..iters {
        f().unwrap();
    }
    let t1 = now_us();
    let avg_us = (t1 - t0) / iters as f64;
    println!("  {name}: {avg_us:.1} us / call (mean over {iters})");
    avg_us
}

fn bench_shape(label: &str, m: u32, n: u32, k: u32) -> Result<()> {
    println!("\n=== {label}: M={m} K={k} N={n} ===");
    let ctx = device::init_device(0)?;
    let stream = ctx.default_stream();

    let a: Vec<u16> = (0..(m * k) as usize)
        .map(|i| (i as u16).wrapping_mul(257))
        .collect();
    let b: Vec<u16> = (0..(k * n) as usize)
        .map(|i| (i as u16).wrapping_mul(263))
        .collect();

    let a_dev = stream
        .memcpy_stod(&a)
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let b_dev = stream
        .memcpy_stod(&b)
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let scale = 0.001f32;
    let mut b_q = stream
        .alloc_zeros::<u8>((k * n) as usize)
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    quantization::quant_bf16_to_fp8_pertensor(&ctx, &stream, &b_dev, &mut b_q, k * n, scale)
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let mut a_q = stream
        .alloc_zeros::<u8>((m * k) as usize)
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    quantization::quant_bf16_to_fp8_pertensor(&ctx, &stream, &a_dev, &mut a_q, m * k, scale)
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let expert_offsets_dev = stream
        .memcpy_stod(&[0u32, m])
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;

    let mut c = stream
        .alloc_zeros::<u16>((m * n) as usize)
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;

    let warmup = 5;
    let iters = 30;

    let bf16_us = time_kernel("BF16  gemm_bf16_mma_v3", warmup, iters, || {
        gemm::gemm_bf16_mma_v3(&ctx, &stream, &a_dev, &b_dev, &mut c, m, n, k)
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        stream.synchronize().map_err(|e| anyhow::anyhow!("{e:?}"))?;
        Ok(())
    });

    // Compare v1 (128×64 tile, 4 warps in 2×2) — better
    // SM saturation at low M than v3 (128×128 tile).
    // v1 requires M%128, N%64, K%16.
    if m.is_multiple_of(128) && n.is_multiple_of(64) && k.is_multiple_of(16) {
        let bf16v1_us = time_kernel(
            "BF16  gemm_bf16_mma (v1, 128×64 tile)",
            warmup,
            iters,
            || {
                gemm::gemm_bf16_mma(&ctx, &stream, &a_dev, &b_dev, &mut c, m, n, k)
                    .map_err(|e| anyhow::anyhow!("{e:?}"))?;
                stream.synchronize().map_err(|e| anyhow::anyhow!("{e:?}"))?;
                Ok(())
            },
        );
        let v1_ratio = bf16v1_us / bf16_us;
        let v = if v1_ratio < 1.0 { "FASTER" } else { "SLOWER" };
        println!("  → BF16 v1 / BF16 v3 ratio: {v1_ratio:.2}× ({v})");
    }

    let w8a16_us = time_kernel("W8A16 gemm_w8a16_grouped_mma (NE=1)", warmup, iters, || {
        moe::gemm_w8a16_grouped_mma(
            &ctx,
            &stream,
            &a_dev,
            &b_q,
            scale,
            &mut c,
            &expert_offsets_dev,
            1,
            m,
            n,
            k,
        )
        .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        stream.synchronize().map_err(|e| anyhow::anyhow!("{e:?}"))?;
        Ok(())
    });

    // gemm_fp8_mma_v3 requires M%128, N%128, K%32 — skip if shape doesn't fit.
    if m.is_multiple_of(128) && n.is_multiple_of(128) && k.is_multiple_of(32) {
        let fp8v3_us = time_kernel(
            "FP8   gemm_fp8_mma_v3 (A pre-quanted)",
            warmup,
            iters,
            || {
                gemm::gemm_fp8_mma_v3(&ctx, &stream, &a_q, &b_q, &mut c, m, n, k)
                    .map_err(|e| anyhow::anyhow!("{e:?}"))?;
                stream.synchronize().map_err(|e| anyhow::anyhow!("{e:?}"))?;
                Ok(())
            },
        );
        // Account for on-the-fly quant of A once per call.
        let quant_us = time_kernel(
            "    + quant_bf16_to_fp8_pertensor(A)",
            warmup,
            iters,
            || {
                quantization::quant_bf16_to_fp8_pertensor(
                    &ctx,
                    &stream,
                    &a_dev,
                    &mut a_q,
                    m * k,
                    scale,
                )
                .map_err(|e| anyhow::anyhow!("{e:?}"))?;
                stream.synchronize().map_err(|e| anyhow::anyhow!("{e:?}"))?;
                Ok(())
            },
        );
        let total = fp8v3_us + quant_us;
        let ratio_fp8 = total / bf16_us;
        println!("  → FP8v3+quant_pertensor total: {total:.1} us, vs BF16: {ratio_fp8:.2}×");

        // Composite recipe: per-token quant + per-row scale.
        let mut row_scales = stream
            .alloc_zeros::<f32>(m as usize)
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        let quant_pt_us = time_kernel("    + quant_bf16_to_fp8_pertoken(A)", warmup, iters, || {
            quantization::quant_bf16_to_fp8_pertoken(
                &ctx,
                &stream,
                &a_dev,
                &mut a_q,
                &mut row_scales,
                m,
                k,
            )
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
            stream.synchronize().map_err(|e| anyhow::anyhow!("{e:?}"))?;
            Ok(())
        });
        let postscale_us = time_kernel("    + scale_bf16_rows_inplace(C)", warmup, iters, || {
            sm121_kernels::activation::scale_bf16_rows_inplace(
                &ctx,
                &stream,
                &mut c,
                &row_scales,
                m,
                n,
                scale,
            )
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
            stream.synchronize().map_err(|e| anyhow::anyhow!("{e:?}"))?;
            Ok(())
        });
        let total_pt = fp8v3_us + quant_pt_us + postscale_us;
        let ratio_pt = total_pt / bf16_us;
        println!(
            "  → FP8v3+quant_pertoken+postscale total: {total_pt:.1} us, vs BF16: {ratio_pt:.2}×"
        );
    } else {
        println!(
            "  (FP8 v3 skipped — shape constraints M%128={} N%128={} K%32={})",
            m % 128,
            n % 128,
            k % 32
        );
    }

    let ratio = w8a16_us / bf16_us;
    let verdict = if ratio < 1.0 { "FASTER" } else { "SLOWER" };
    println!("  → W8A16/BF16 ratio: {ratio:.2}× ({verdict})");

    // Dense W8A16 v3 — ONE kernel, no quant, no post-scale.
    if m.is_multiple_of(128) && n.is_multiple_of(128) && k.is_multiple_of(32) {
        let dense_us = time_kernel("DENSE gemm_w8a16_mma_v3", warmup, iters, || {
            gemm::gemm_w8a16_mma_v3(&ctx, &stream, &a_dev, &b_q, &mut c, m, n, k, scale)
                .map_err(|e| anyhow::anyhow!("{e:?}"))?;
            stream.synchronize().map_err(|e| anyhow::anyhow!("{e:?}"))?;
            Ok(())
        });
        let dense_ratio = dense_us / bf16_us;
        let v = if dense_ratio < 1.0 {
            "FASTER"
        } else {
            "SLOWER"
        };
        println!("  → DENSE W8A16 v3 / BF16 v3: {dense_ratio:.2}× ({v})");
    }
    Ok(())
}

fn main() -> Result<()> {
    println!("Kernel-level latency: W8A16 grouped (NE=1) vs BF16 v3");
    bench_shape("mlp.gate/up", 128, 12288, 4096)?;
    bench_shape("mlp.down", 128, 4096, 12288)?;
    bench_shape("lm_head", 128, 152064, 4096)?;
    bench_shape("FA qkv", 128, 4608, 4096)?;
    bench_shape("FA o_proj", 128, 4096, 4096)?;
    bench_shape("GDN in_proj", 128, 12352, 4096)?;
    bench_shape("GDN out_proj", 128, 4096, 2048)?;
    Ok(())
}
