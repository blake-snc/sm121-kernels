//! End-to-end NVFP4 GEMM with FP8 E4M3 scales (modernized, CUTLASS-aligned path).
//!
//! Chain: BF16 weights → `quant_bf16_to_nvfp4` (FP8 scales) →
//!        `gemm_nvfp4_fp8scale_grouped_mma` → compare vs BF16 reference.
//!
//! This closes the quant→GEMM pipeline gap that the FP32-scale variant had.

use half::bf16;
use sm121_kernels::{device, moe, quantization};

fn bf16_f32(x: &[u16]) -> Vec<f32> {
    x.iter().map(|b| bf16::from_bits(*b).to_f32()).collect()
}

#[test]
fn t_nvfp4_fp8scale_v2_matches_v1() {
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();
    let num_experts = 4u32;
    let tokens_per_expert = 16u32;
    let n = 32u32;
    let k = 64u32;
    let total_tokens = num_experts * tokens_per_expert;

    let mut state: u64 = 0xACE1_2345;
    let mut rnd = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) as u32 as f32 / u32::MAX as f32 - 0.5) * 2.0
    };
    let a: Vec<u16> = (0..(total_tokens * k))
        .map(|_| bf16::from_f32(rnd()).to_bits())
        .collect();
    let b: Vec<u8> = (0..(num_experts * n * k / 2) as usize)
        .map(|i| ((i.wrapping_mul(53) + 7) % 256) as u8)
        .collect();
    let scales: Vec<u8> = (0..(num_experts * n * (k / 16)) as usize)
        .map(|i| ((i.wrapping_mul(13) + 5) % 256) as u8)
        .collect();
    let offsets: Vec<u32> = (0..=num_experts).map(|e| e * tokens_per_expert).collect();
    let a_dev = stream.memcpy_stod(&a).unwrap();
    let b_dev = stream.memcpy_stod(&b).unwrap();
    let s_dev = stream.memcpy_stod(&scales).unwrap();
    let off_dev = stream.memcpy_stod(&offsets).unwrap();
    let mut c_v1 = stream
        .alloc_zeros::<u16>((total_tokens * n) as usize)
        .unwrap();
    let mut c_v2 = stream
        .alloc_zeros::<u16>((total_tokens * n) as usize)
        .unwrap();
    moe::gemm_nvfp4_fp8scale_grouped_mma(
        &ctx,
        &stream,
        &a_dev,
        &b_dev,
        &s_dev,
        &mut c_v1,
        &off_dev,
        num_experts,
        tokens_per_expert,
        n,
        k,
    )
    .unwrap();
    moe::gemm_nvfp4_fp8scale_grouped_mma_v2(
        &ctx,
        &stream,
        &a_dev,
        &b_dev,
        &s_dev,
        &mut c_v2,
        &off_dev,
        num_experts,
        tokens_per_expert,
        n,
        k,
    )
    .unwrap();
    let v1 = bf16_f32(&stream.memcpy_dtov(&c_v1).unwrap());
    let v2 = bf16_f32(&stream.memcpy_dtov(&c_v2).unwrap());
    let mut max_diff = 0f32;
    for (a, b) in v1.iter().zip(v2.iter()) {
        if a.is_nan() || b.is_nan() {
            continue;
        }
        let d = (a - b).abs();
        if d > max_diff {
            max_diff = d;
        }
    }
    eprintln!("nvfp4_fp8scale v2 vs v1: max_diff={max_diff:.4}");
    assert!(
        max_diff < 0.5,
        "nvfp4_fp8scale v2 diverges from v1: {max_diff}"
    );
}

#[test]
fn t_nvfp4_fp8scale_uniform() {
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();
    let num_experts = 1u32;
    let tokens_per_expert = 4u32;
    let n = 32u32;
    let k = 64u32;
    let total_tokens = num_experts * tokens_per_expert;

    // A = 1.0, B all nibble 2 (+1.0) → 0x22. FP8 scale = 0x38 (FP8 E4M3 1.0).
    let a: Vec<u16> = vec![bf16::from_f32(1.0).to_bits(); (total_tokens * k) as usize];
    let b: Vec<u8> = vec![0x22u8; (num_experts * n * k / 2) as usize];
    let scales: Vec<u8> = vec![0x38u8; (num_experts * n * (k / 16)) as usize];
    let offsets: Vec<u32> = (0..=num_experts).map(|e| e * tokens_per_expert).collect();

    let a_dev = stream.memcpy_stod(&a).unwrap();
    let b_dev = stream.memcpy_stod(&b).unwrap();
    let s_dev = stream.memcpy_stod(&scales).unwrap();
    let off_dev = stream.memcpy_stod(&offsets).unwrap();
    let mut c_dev = stream
        .alloc_zeros::<u16>((total_tokens * n) as usize)
        .unwrap();

    moe::gemm_nvfp4_fp8scale_grouped_mma(
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

    let c = bf16_f32(&stream.memcpy_dtov(&c_dev).unwrap());
    let expected = k as f32;
    let mut max_diff = 0f32;
    for v in &c {
        let d = (v - expected).abs();
        if d > max_diff {
            max_diff = d;
        }
    }
    eprintln!("nvfp4_fp8scale uniform max_diff={max_diff:.4}");
    assert!(max_diff < 1.0, "uniform diverges: {max_diff}");
}

#[test]
fn t_nvfp4_fp8scale_end_to_end_quant_and_gemm() {
    // Chain quant → GEMM → compare to BF16 reference.
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();

    let num_experts: u32 = 4;
    let tokens_per_expert: u32 = 16;
    let n: u32 = 32;
    let k: u32 = 64;
    let total_tokens = num_experts * tokens_per_expert;

    let mut state: u64 = 0x31415926;
    let mut rnd = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) as u32 as f32 / u32::MAX as f32 - 0.5) * 2.0
    };

    let a_bf16: Vec<u16> = (0..(total_tokens * k))
        .map(|_| bf16::from_f32(rnd()).to_bits())
        .collect();
    let b_nk: Vec<u16> = (0..(num_experts * n * k))
        .map(|_| bf16::from_f32(rnd()).to_bits())
        .collect();
    // Transpose to [num_experts, K, N] for BF16 reference GEMM.
    let mut b_kn = vec![0u16; (num_experts * n * k) as usize];
    for e in 0..num_experts as usize {
        for nn in 0..n as usize {
            for kk in 0..k as usize {
                let src = e * n as usize * k as usize + nn * k as usize + kk;
                let dst = e * k as usize * n as usize + kk * n as usize + nn;
                b_kn[dst] = b_nk[src];
            }
        }
    }
    let offsets: Vec<u32> = (0..=num_experts).map(|e| e * tokens_per_expert).collect();

    let a_dev = stream.memcpy_stod(&a_bf16).unwrap();
    let b_nk_dev = stream.memcpy_stod(&b_nk).unwrap();
    let b_kn_dev = stream.memcpy_stod(&b_kn).unwrap();
    let off_dev = stream.memcpy_stod(&offsets).unwrap();

    // BF16 reference
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

    // Quantize B to NVFP4 (FP4 + FP8 E4M3 scales per 16 elements).
    let num_rows = num_experts * n;
    let mut b_fp4 = stream
        .alloc_zeros::<u8>((num_rows * k / 2) as usize)
        .unwrap();
    let mut b_scales = stream
        .alloc_zeros::<u8>((num_rows * k / 16) as usize)
        .unwrap();
    quantization::quant_bf16_to_nvfp4(
        &ctx,
        &stream,
        &b_nk_dev,
        &mut b_fp4,
        &mut b_scales,
        num_rows,
        k,
    )
    .unwrap();

    let mut c_nv_dev = stream
        .alloc_zeros::<u16>((total_tokens * n) as usize)
        .unwrap();
    moe::gemm_nvfp4_fp8scale_grouped_mma(
        &ctx,
        &stream,
        &a_dev,
        &b_fp4,
        &b_scales,
        &mut c_nv_dev,
        &off_dev,
        num_experts,
        tokens_per_expert,
        n,
        k,
    )
    .unwrap();

    let c_ref = bf16_f32(&stream.memcpy_dtov(&c_ref_dev).unwrap());
    let c_nv = bf16_f32(&stream.memcpy_dtov(&c_nv_dev).unwrap());

    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    let mut max_mag = 0f32;
    for (a, b) in c_ref.iter().zip(c_nv.iter()) {
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
    eprintln!("nvfp4_fp8scale e2e: max_abs={max_abs:.4} max_rel={max_rel:.4} max_mag={max_mag:.2}",);
    // NVFP4: 4-bit E2M1 with per-16-element FP8 scales — more fine-grained
    // than MXFP4 (per-32), so precision is between MXFP8 and MXFP4.
    assert!(
        max_rel < 0.30,
        "nvfp4_fp8scale e2e max_rel too high: {max_rel}"
    );
}
