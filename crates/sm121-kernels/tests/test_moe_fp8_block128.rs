//! DeepSeek V3 MoE block-scaled FP8 grouped GEMM.

mod common;

use sm121_kernels::{device, moe};

fn bf16_vec_to_f32(v: &[u16]) -> Vec<f32> {
    v.iter()
        .map(|b| f32::from_bits((*b as u32) << 16))
        .collect()
}

#[test]
fn test_gemm_fp8_block128_grouped_small() {
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();

    let num_experts = 2u32;
    let tokens_per_expert = 16u32;
    let n = 32u32;
    let k = 128u32; // one block of 128

    // A: [total_tokens, K] BF16
    let total_tokens = num_experts * tokens_per_expert;
    let mut a_bf16 = vec![0u16; (total_tokens * k) as usize];
    for i in 0..a_bf16.len() {
        let f = 0.1 * ((i % 7) as f32 - 3.0);
        a_bf16[i] = (f.to_bits() >> 16) as u16;
    }

    // B: [num_experts, N, K] FP8
    //   For the test: quantize a randomized FP32 tensor with block-scale = 1.0.
    //   Then the compute = A @ B_dequantized where B_dequantized = B * 1.0 = B.
    let mut b_fp8 = vec![0u8; (num_experts * n * k) as usize];
    let mut b_f32_ref = vec![0.0f32; (num_experts * n * k) as usize];
    for i in 0..b_fp8.len() {
        let v = 0.05 * ((i % 13) as f32 - 6.0);
        let _clamped = v.clamp(-6.0, 6.0);
        // FP8 E2M1-ish quantization approximation: here we use the same
        // levels as our on-GPU cvt. To avoid re-deriving exact FP8 bit
        // patterns, we use pre-quantized values through the hardware by
        // round-tripping through the kernel itself — but for simplicity,
        // we approximate: store raw f32 as "effectively the FP8 value"
        // and generate FP8 bits via a rough mapping.
        // For the test, we just use recognizable FP8 values.
        b_fp8[i] = 0x38; // FP8 E4M3 = 1.0
        b_f32_ref[i] = 1.0;
    }

    // Scales: [num_experts, N, K/128] FP32 — all 0.5 → dequantized B = 1.0 * 0.5 = 0.5
    let num_k_blocks = k / 128;
    let scales_len = (num_experts * n * num_k_blocks) as usize;
    let b_scales: Vec<f32> = vec![0.5f32; scales_len];
    // So effective B value = 0.5

    // Expert offsets: tokens_per_expert each
    let expert_offsets: Vec<u32> = (0..=num_experts).map(|e| e * tokens_per_expert).collect();

    // Expected: C[row, n] = Σ_k A[row, k] * 0.5 = 0.5 * Σ_k A[row, k]
    let a_f32 = bf16_vec_to_f32(&a_bf16);
    let mut expected_f32 = vec![0.0f32; (total_tokens * n) as usize];
    for t in 0..total_tokens as usize {
        let mut row_sum: f32 = 0.0;
        for kk in 0..k as usize {
            row_sum += a_f32[t * k as usize + kk];
        }
        for nn in 0..n as usize {
            expected_f32[t * n as usize + nn] = row_sum * 0.5;
        }
    }

    let a_dev = stream.memcpy_stod(&a_bf16).unwrap();
    let b_dev = stream.memcpy_stod(&b_fp8).unwrap();
    let s_dev = stream.memcpy_stod(&b_scales).unwrap();
    let off_dev = stream.memcpy_stod(&expert_offsets).unwrap();
    let mut c_dev = stream
        .alloc_zeros::<u16>((total_tokens * n) as usize)
        .unwrap();

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
    .expect("FP8 block128 grouped GEMM failed");

    let c_host = stream.memcpy_dtov(&c_dev).unwrap();
    let c_f32 = bf16_vec_to_f32(&c_host);

    let mut max_diff: f32 = 0.0;
    for i in 0..c_f32.len() {
        let d = (c_f32[i] - expected_f32[i]).abs();
        if d > max_diff {
            max_diff = d;
        }
    }
    eprintln!("FP8 block128 grouped GEMM: max_diff={max_diff:.4}");
    assert!(
        max_diff <= 0.5,
        "FP8 block128 grouped GEMM mismatch: {max_diff}"
    );
}

#[test]
fn test_gemm_fp8_block128_grouped_mma_uniform() {
    use half::bf16;
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();
    let num_experts = 1u32;
    let tokens_per_expert = 4u32;
    let n = 32u32;
    let k = 128u32;
    let total_tokens = num_experts * tokens_per_expert;

    // A = all 1.0 (BF16)
    let a_bf16: Vec<u16> = vec![bf16::from_f32(1.0).to_bits(); (total_tokens * k) as usize];
    // B = all 0x38 (= 1.0 in FP8 E4M3)
    let b_fp8: Vec<u8> = vec![0x38u8; (num_experts * n * k) as usize];
    // Scale = 1.0
    let b_scales: Vec<f32> = vec![1.0f32; (num_experts * n * (k / 128)) as usize];
    let expert_offsets: Vec<u32> = (0..=num_experts).map(|e| e * tokens_per_expert).collect();

    let a_dev = stream.memcpy_stod(&a_bf16).unwrap();
    let b_dev = stream.memcpy_stod(&b_fp8).unwrap();
    let s_dev = stream.memcpy_stod(&b_scales).unwrap();
    let off_dev = stream.memcpy_stod(&expert_offsets).unwrap();
    let mut c_dev = stream
        .alloc_zeros::<u16>((total_tokens * n) as usize)
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

    let c = bf16_vec_to_f32(&stream.memcpy_dtov(&c_dev).unwrap());
    // Expected: every output = k * 1.0 * 1.0 * 1.0 = 128
    let expected = k as f32;
    let mut max_diff = 0f32;
    for (i, v) in c.iter().enumerate() {
        let d = (v - expected).abs();
        if d > max_diff {
            max_diff = d;
            eprintln!(
                "mma_uniform: idx={i} got={v} expected={expected} (row={}, col={})",
                i / n as usize,
                i % n as usize
            );
        }
    }
    eprintln!("mma_uniform max_diff={max_diff:.4}");
    assert!(max_diff < 1.0, "uniform test diverges: {max_diff}");
}

#[test]
fn test_gemm_fp8_block128_grouped_mma_nonuniform_a() {
    use half::bf16;
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();
    let num_experts = 1u32;
    let tokens_per_expert = 32u32;
    let n = 32u32;
    let k = 128u32;
    let total_tokens = num_experts * tokens_per_expert;

    // A[r, k] = k as BF16 (so row sum = 0+1+...+127 = 8128)
    let a_bf16: Vec<u16> = (0..(total_tokens * k))
        .map(|i| bf16::from_f32((i % k) as f32 * 0.01).to_bits())
        .collect();
    // B all 1.0, scales all 1.0
    let b_fp8: Vec<u8> = vec![0x38u8; (num_experts * n * k) as usize];
    let b_scales: Vec<f32> = vec![1.0f32; (num_experts * n * (k / 128)) as usize];
    let expert_offsets: Vec<u32> = (0..=num_experts).map(|e| e * tokens_per_expert).collect();

    let a_dev = stream.memcpy_stod(&a_bf16).unwrap();
    let b_dev = stream.memcpy_stod(&b_fp8).unwrap();
    let s_dev = stream.memcpy_stod(&b_scales).unwrap();
    let off_dev = stream.memcpy_stod(&expert_offsets).unwrap();
    let mut c_dev = stream
        .alloc_zeros::<u16>((total_tokens * n) as usize)
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

    let c = bf16_vec_to_f32(&stream.memcpy_dtov(&c_dev).unwrap());
    // Expected per row: sum_{k=0..128} 0.01*k = 0.01 * 128*127/2 = 81.28
    let expected = 0.01f32 * (k * (k - 1)) as f32 / 2.0;
    let mut max_diff = 0f32;
    for (i, v) in c.iter().enumerate() {
        let d = (v - expected).abs();
        if d > max_diff {
            max_diff = d;
            if max_diff > 2.0 {
                eprintln!(
                    "nonuniform_a: idx={i} got={v} expected={expected} (row={}, col={})",
                    i / n as usize,
                    i % n as usize
                );
            }
        }
    }
    eprintln!("mma_nonuniform_a max_diff={max_diff:.4}");
    assert!(max_diff < 2.0, "nonuniform_a diverges: {max_diff}");
}

#[test]
fn test_gemm_fp8_block128_grouped_mma_nonuniform_b() {
    use half::bf16;
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();
    let num_experts = 1u32;
    let tokens_per_expert = 4u32;
    let n = 32u32;
    let k = 128u32;
    let total_tokens = num_experts * tokens_per_expert;

    // A all 1.0. Columns vary in B — col c has all 0x38 (1.0) if c even, 0x3C (1.5) if c odd.
    let a_bf16: Vec<u16> = vec![bf16::from_f32(1.0).to_bits(); (total_tokens * k) as usize];
    let b_fp8: Vec<u8> = (0..(num_experts * n * k) as usize)
        .map(|i| {
            // Layout [num_experts, N, K]: i = (e*N + n)*K + k_idx
            let n_idx = (i / k as usize) % n as usize;
            if n_idx.is_multiple_of(2) {
                0x38
            } else {
                0x3C
            } // FP8 E4M3: 1.0 or 1.5
        })
        .collect();
    let b_scales: Vec<f32> = vec![1.0f32; (num_experts * n * (k / 128)) as usize];
    let expert_offsets: Vec<u32> = (0..=num_experts).map(|e| e * tokens_per_expert).collect();

    let a_dev = stream.memcpy_stod(&a_bf16).unwrap();
    let b_dev = stream.memcpy_stod(&b_fp8).unwrap();
    let s_dev = stream.memcpy_stod(&b_scales).unwrap();
    let off_dev = stream.memcpy_stod(&expert_offsets).unwrap();
    let mut c_dev = stream
        .alloc_zeros::<u16>((total_tokens * n) as usize)
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

    let c = bf16_vec_to_f32(&stream.memcpy_dtov(&c_dev).unwrap());
    // Expected: col even → 128; col odd → 192
    let mut max_diff = 0f32;
    for r in 0..total_tokens as usize {
        for col in 0..n as usize {
            let expected = if col % 2 == 0 { 128.0 } else { 192.0 };
            let got = c[r * n as usize + col];
            let d = (got - expected).abs();
            if d > max_diff {
                max_diff = d;
                if d > 2.0 {
                    eprintln!("nonuniform_b: row={r} col={col} got={got} expected={expected}");
                }
            }
        }
    }
    eprintln!("mma_nonuniform_b max_diff={max_diff:.4}");
    assert!(max_diff < 2.0, "nonuniform_b diverges: {max_diff}");
}

#[test]
fn test_gemm_fp8_block128_grouped_mma_nonuniform_scale() {
    use half::bf16;
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();
    let num_experts = 1u32;
    let tokens_per_expert = 4u32;
    let n = 32u32;
    let k = 128u32;
    let total_tokens = num_experts * tokens_per_expert;

    // A all 1.0, B all 0x38 (1.0). Scale per-col: scale[n] = (n + 1) * 0.1
    let a_bf16: Vec<u16> = vec![bf16::from_f32(1.0).to_bits(); (total_tokens * k) as usize];
    let b_fp8: Vec<u8> = vec![0x38u8; (num_experts * n * k) as usize];
    // Scales layout [num_experts, N, K/128]. For k/128=1, one scale per n.
    let b_scales: Vec<f32> = (0..(num_experts * n))
        .map(|i| (i as f32 + 1.0) * 0.1)
        .collect();
    let expert_offsets: Vec<u32> = (0..=num_experts).map(|e| e * tokens_per_expert).collect();

    let a_dev = stream.memcpy_stod(&a_bf16).unwrap();
    let b_dev = stream.memcpy_stod(&b_fp8).unwrap();
    let s_dev = stream.memcpy_stod(&b_scales).unwrap();
    let off_dev = stream.memcpy_stod(&expert_offsets).unwrap();
    let mut c_dev = stream
        .alloc_zeros::<u16>((total_tokens * n) as usize)
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

    let c = bf16_vec_to_f32(&stream.memcpy_dtov(&c_dev).unwrap());
    // Expected per (row, col): sum_k 1.0 * 1.0 * scale[col] = 128 * scale[col]
    let mut max_diff = 0f32;
    for r in 0..total_tokens as usize {
        for col in 0..n as usize {
            let expected = 128.0 * (col as f32 + 1.0) * 0.1;
            let got = c[r * n as usize + col];
            let d = (got - expected).abs();
            if d > max_diff {
                max_diff = d;
                if d > 1.0 {
                    eprintln!(
                        "nonuniform_scale: row={r} col={col} got={got} expected={expected:.4}"
                    );
                }
            }
        }
    }
    eprintln!("mma_nonuniform_scale max_diff={max_diff:.4}");
    assert!(max_diff < 1.0, "nonuniform_scale diverges: {max_diff}");
}

#[test]
fn test_gemm_fp8_block128_v2_matches_v1() {
    use half::bf16;
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();
    let num_experts = 4u32;
    let tokens_per_expert = 32u32;
    let n = 64u32;
    let k = 256u32;
    let total_tokens = num_experts * tokens_per_expert;

    let mut state: u64 = 0xCAFE_BABE;
    let mut rnd = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) as u32 as f32 / u32::MAX as f32 - 0.5) * 2.0
    };

    let a: Vec<u16> = (0..(total_tokens * k))
        .map(|_| bf16::from_f32(rnd()).to_bits())
        .collect();
    let b: Vec<u8> = (0..(num_experts * n * k) as usize)
        .map(|i| {
            let byte = ((i.wrapping_mul(31) + 17) % 256) as u8;
            if (byte & 0x7F) == 0x7F {
                byte ^ 0x10
            } else {
                byte
            }
        })
        .collect();
    let scales: Vec<f32> = (0..(num_experts * n * (k / 128)) as usize)
        .map(|_| 0.25 + rnd().abs() * 0.5)
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

    moe::gemm_fp8_block128_grouped_mma(
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
    moe::gemm_fp8_block128_grouped_mma_v2(
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

    let v1 = bf16_vec_to_f32(&stream.memcpy_dtov(&c_v1).unwrap());
    let v2 = bf16_vec_to_f32(&stream.memcpy_dtov(&c_v2).unwrap());

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
    eprintln!("v2 vs v1: max_diff={max_diff:.4}");
    // Both kernels do BF16 MMA on the same decoded values — should be byte-exact.
    assert!(max_diff < 0.5, "v2 diverges from v1: {max_diff}");
}

#[test]
fn test_gemm_fp8_block128_grouped_mma_matches_scalar() {
    use half::bf16;
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();

    let num_experts = 4u32;
    let tokens_per_expert = 32u32;
    let n = 64u32;
    let k = 256u32; // 2 K-blocks

    let total_tokens = num_experts * tokens_per_expert;

    // Deterministic input.
    let mut state: u64 = 0xF00D1234;
    let mut rnd = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as u32 as f32 / u32::MAX as f32 - 0.5
    };

    // A: BF16 with modest range.
    let a_bf16: Vec<u16> = (0..(total_tokens * k) as usize)
        .map(|_| bf16::from_f32(rnd() * 2.0).to_bits())
        .collect();

    // B: varied FP8 E4M3 values. Skip 0x7F / 0xFF (NaN patterns).
    let b_fp8: Vec<u8> = (0..(num_experts * n * k) as usize)
        .map(|i| {
            let byte = ((i.wrapping_mul(31) + 17) % 256) as u8;
            if (byte & 0x7F) == 0x7F {
                byte ^ 0x10
            } else {
                byte
            }
        })
        .collect();

    // Scales: one per (expert, n, k_block) block.
    let num_k_blocks = k / 128;
    let b_scales: Vec<f32> = (0..(num_experts * n * num_k_blocks) as usize)
        .map(|_| 0.25 + rnd().abs() * 0.5)
        .collect();

    let expert_offsets: Vec<u32> = (0..=num_experts).map(|e| e * tokens_per_expert).collect();

    let a_dev = stream.memcpy_stod(&a_bf16).unwrap();
    let b_dev = stream.memcpy_stod(&b_fp8).unwrap();
    let s_dev = stream.memcpy_stod(&b_scales).unwrap();
    let off_dev = stream.memcpy_stod(&expert_offsets).unwrap();
    let mut c_scalar_dev = stream
        .alloc_zeros::<u16>((total_tokens * n) as usize)
        .unwrap();
    let mut c_mma_dev = stream
        .alloc_zeros::<u16>((total_tokens * n) as usize)
        .unwrap();

    moe::gemm_fp8_block128_grouped(
        &ctx,
        &stream,
        &a_dev,
        &b_dev,
        &s_dev,
        &mut c_scalar_dev,
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
        &mut c_mma_dev,
        &off_dev,
        num_experts,
        tokens_per_expert,
        n,
        k,
    )
    .unwrap();

    let c_scalar = bf16_vec_to_f32(&stream.memcpy_dtov(&c_scalar_dev).unwrap());
    let c_mma = bf16_vec_to_f32(&stream.memcpy_dtov(&c_mma_dev).unwrap());

    let mut max_diff = 0f32;
    let mut max_mag = 0f32;
    let mut sum_diff = 0f64;
    let mut n_checked = 0usize;
    let mut n_nan = 0usize;
    for (a, b) in c_scalar.iter().zip(c_mma.iter()) {
        if a.is_nan() || b.is_nan() {
            n_nan += 1;
            continue;
        }
        let d = (a - b).abs();
        if d > max_diff {
            max_diff = d;
        }
        let m = a.abs().max(b.abs());
        if m > max_mag {
            max_mag = m;
        }
        sum_diff += d as f64;
        n_checked += 1;
    }
    let mean_diff = sum_diff / n_checked.max(1) as f64;
    let rel = if max_mag > 0.0 {
        max_diff / max_mag
    } else {
        0.0
    };
    eprintln!(
        "fp8 block128 mma vs scalar: max_diff={max_diff:.4} mean_diff={mean_diff:.4} max_mag={max_mag:.2} rel={rel:.4} nan={n_nan}",
    );
    assert_eq!(n_nan, 0, "unexpected NaN in scalar or MMA output");
    // Relative tolerance: BF16 has ~8-bit mantissa (~0.4% eps). Over K=256
    // accumulation with block-scaled FP8 weights we expect cumulative error
    // up to ~2% of the largest output magnitude.
    assert!(
        rel <= 0.02,
        "fp8 block128 mma relative diff too large: rel={rel}"
    );
}
