//! End-to-end DeepSeek V3 MoE FP8 block-128 GEMM validation.
//!
//! Pipeline:
//!   1. Generate random BF16 weights [num_experts, N, K].
//!   2. Quantize to FP8 E4M3 + per-128-K FP32 scales via `quant_bf16_to_fp8_block128`.
//!   3. Run both paths:
//!      (a) `gemm_bf16_grouped_mma` on BF16 weights (reference)
//!      (b) `gemm_fp8_block128_grouped_mma` on FP8+scales (production path)
//!   4. Compare outputs — quantization error should be bounded.
//!
//! This verifies the quantize-then-matmul pipeline used in production on
//! DeepSeek V3 inference.

use half::bf16;
use sm121_kernels::{device, moe, quantization};

fn bf16_f32(x: &[u16]) -> Vec<f32> {
    x.iter().map(|b| bf16::from_bits(*b).to_f32()).collect()
}

#[test]
fn t_fp8_block128_end_to_end_quant_and_gemm() {
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();

    let num_experts: u32 = 4;
    let tokens_per_expert: u32 = 16;
    let n: u32 = 32;
    let k: u32 = 256; // 2 K-blocks of 128
    let total_tokens = num_experts * tokens_per_expert;

    // Deterministic random BF16 A and B (small range to stay in FP8 coverage).
    let mut state: u64 = 0xABCD_1234;
    let mut rnd = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) as u32 as f32 / u32::MAX as f32 - 0.5) * 2.0
    };

    let a_bf16: Vec<u16> = (0..(total_tokens * k))
        .map(|_| bf16::from_f32(rnd()).to_bits())
        .collect();
    // BF16 B in the shape grouped GEMM expects: [num_experts, K, N] for BF16 path.
    // BUT the block-128 kernel expects [num_experts, N, K]. We'll generate BF16
    // weights twice (different layouts) from the same underlying values.
    let b_flat_n_k: Vec<u16> = (0..(num_experts * n * k))
        .map(|_| bf16::from_f32(rnd()).to_bits())
        .collect();
    // Transpose to [num_experts, K, N] for BF16 grouped GEMM reference.
    let mut b_flat_k_n = vec![0u16; (num_experts * n * k) as usize];
    for e in 0..num_experts as usize {
        for nn in 0..n as usize {
            for kk in 0..k as usize {
                let src = e * n as usize * k as usize + nn * k as usize + kk;
                let dst = e * k as usize * n as usize + kk * n as usize + nn;
                b_flat_k_n[dst] = b_flat_n_k[src];
            }
        }
    }

    let offsets: Vec<u32> = (0..=num_experts).map(|e| e * tokens_per_expert).collect();

    let a_dev = stream.memcpy_stod(&a_bf16).unwrap();
    let b_nk_dev = stream.memcpy_stod(&b_flat_n_k).unwrap();
    let b_kn_dev = stream.memcpy_stod(&b_flat_k_n).unwrap();
    let off_dev = stream.memcpy_stod(&offsets).unwrap();

    // (a) BF16 grouped GEMM reference (K,N layout)
    let mut c_ref_dev = stream
        .alloc_zeros::<u16>((total_tokens * n) as usize)
        .unwrap();
    moe::gemm_bf16_grouped_mma(
        &ctx,
        &stream,
        &a_dev,
        &b_kn_dev,
        &mut c_ref_dev,
        &off_dev,
        num_experts,
        tokens_per_expert,
        n,
        k,
    )
    .unwrap();

    // (b) Quantize B_flat_n_k → FP8 block-128 + scales.
    //   quant_bf16_to_fp8_block128 takes [num_rows, hidden] where hidden is along last dim.
    //   B is [num_experts, N, K] — we pass num_rows=num_experts*N, hidden=K.
    let num_rows = num_experts * n;
    let mut b_fp8 = stream.alloc_zeros::<u8>((num_rows * k) as usize).unwrap();
    let mut b_scales = stream
        .alloc_zeros::<f32>((num_rows * k / 128) as usize)
        .unwrap();
    quantization::quant_bf16_to_fp8_block128(
        &ctx,
        &stream,
        &b_nk_dev,
        &mut b_fp8,
        &mut b_scales,
        num_rows,
        k,
    )
    .unwrap();

    // Run FP8 block-128 grouped MMA
    let mut c_fp8_dev = stream
        .alloc_zeros::<u16>((total_tokens * n) as usize)
        .unwrap();
    moe::gemm_fp8_block128_grouped_mma(
        &ctx,
        &stream,
        &a_dev,
        &b_fp8,
        &b_scales,
        &mut c_fp8_dev,
        &off_dev,
        num_experts,
        tokens_per_expert,
        n,
        k,
    )
    .unwrap();

    let c_ref = bf16_f32(&stream.memcpy_dtov(&c_ref_dev).unwrap());
    let c_fp8 = bf16_f32(&stream.memcpy_dtov(&c_fp8_dev).unwrap());

    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    let mut max_mag = 0f32;
    for (a, b) in c_ref.iter().zip(c_fp8.iter()) {
        let d = (a - b).abs();
        if d > max_abs {
            max_abs = d;
        }
        let m = a.abs().max(b.abs());
        if m > max_mag {
            max_mag = m;
        }
        if m > 2.0 {
            let rel = d / m;
            if rel > max_rel {
                max_rel = rel;
            }
        }
    }
    eprintln!("fp8_block128 e2e: max_abs={max_abs:.4} max_rel={max_rel:.4} max_mag={max_mag:.2}",);
    // FP8 E4M3 has ~3-bit mantissa. Per-block quantization of K=128 values
    // typically produces relative error ~5-8% on GEMM outputs at moderate
    // magnitudes. Accept up to 15% rel to cover worst-case accumulation.
    assert!(
        max_rel < 0.15,
        "e2e quant error too high: max_rel={max_rel}"
    );
}
