//! Validate `gemm_bf16_grouped_mma` against the scalar `gemm_bf16_grouped`.
//!
//! Both kernels read the same [num_experts, K, N] weight layout and produce
//! [total_tokens, N] BF16 output. The MMA path uses tensor cores.
use half::bf16;
use sm121_kernels::{device, moe};

fn bf16_f32(x: &[u16]) -> Vec<f32> {
    x.iter().map(|b| bf16::from_bits(*b).to_f32()).collect()
}

fn make_bf16(v: &[f32]) -> Vec<u16> {
    v.iter().map(|f| bf16::from_f32(*f).to_bits()).collect()
}

#[test]
fn t_grouped_mma_matches_scalar() {
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();

    let num_experts: u32 = 4;
    let k: u32 = 64;
    let n: u32 = 64;
    // Tokens per expert: 10, 5, 32, 1 → deliberately irregular
    let counts = [10u32, 5, 32, 1];
    let total_tokens: u32 = counts.iter().sum();

    // Deterministic pseudo-random inputs.
    let mut state: u64 = 0xCAFE1234;
    let mut rnd = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) as u32 as f32) / u32::MAX as f32 - 0.5
    };

    let a: Vec<f32> = (0..(total_tokens * k)).map(|_| rnd() * 2.0).collect();
    let b: Vec<f32> = (0..(num_experts * k * n)).map(|_| rnd() * 2.0).collect();

    let a_bf16 = make_bf16(&a);
    let b_bf16 = make_bf16(&b);

    // Expert offsets prefix sum.
    let mut offsets = vec![0u32; num_experts as usize + 1];
    for (i, c) in counts.iter().enumerate() {
        offsets[i + 1] = offsets[i] + c;
    }

    let a_dev = stream.memcpy_stod(&a_bf16).unwrap();
    let b_dev = stream.memcpy_stod(&b_bf16).unwrap();
    let off_dev = stream.memcpy_stod(&offsets).unwrap();
    let mut c_scalar_dev = stream
        .alloc_zeros::<u16>((total_tokens * n) as usize)
        .unwrap();
    let mut c_mma_dev = stream
        .alloc_zeros::<u16>((total_tokens * n) as usize)
        .unwrap();

    let m_max = *counts.iter().max().unwrap();

    moe::gemm_bf16_grouped(
        &ctx,
        &stream,
        &a_dev,
        &b_dev,
        &mut c_scalar_dev,
        &off_dev,
        num_experts,
        m_max,
        n,
        k,
    )
    .unwrap();

    moe::gemm_bf16_grouped_mma(
        &ctx,
        &stream,
        &a_dev,
        &b_dev,
        &mut c_mma_dev,
        &off_dev,
        num_experts,
        m_max,
        n,
        k,
    )
    .unwrap();

    let c_scalar = bf16_f32(&stream.memcpy_dtov(&c_scalar_dev).unwrap());
    let c_mma = bf16_f32(&stream.memcpy_dtov(&c_mma_dev).unwrap());

    // Compare MMA vs scalar BF16 (both do FP32 accumulate, same round-to-bf16 at end —
    // difference should be small, within ~0.1 from reduction order differences).
    let mut max_diff = 0f32;
    let mut sum_diff = 0f64;
    let mut n_checked = 0usize;
    for (a, b) in c_scalar.iter().zip(c_mma.iter()) {
        let d = (a - b).abs();
        if d > max_diff {
            max_diff = d;
        }
        sum_diff += d as f64;
        n_checked += 1;
    }
    let mean_diff = sum_diff / n_checked as f64;
    eprintln!("grouped mma vs scalar: max_diff={max_diff:.4} mean_diff={mean_diff:.4}");
    assert!(
        max_diff <= 0.5,
        "grouped_mma diverges from scalar: max_diff={max_diff}"
    );
}

#[test]
fn t_grouped_mma_vs_naive_reference() {
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();

    let num_experts: u32 = 2;
    let k: u32 = 32;
    let n: u32 = 32;
    let counts = [16u32, 8];
    let total_tokens: u32 = counts.iter().sum();

    // Simple structured inputs.
    let a: Vec<f32> = (0..(total_tokens * k))
        .map(|i| ((i as i32 & 7) - 4) as f32 * 0.5)
        .collect();
    let b: Vec<f32> = (0..(num_experts * k * n))
        .map(|i| ((i as i32 & 15) - 8) as f32 * 0.25)
        .collect();

    let a_bf16 = make_bf16(&a);
    let b_bf16 = make_bf16(&b);

    let mut offsets = vec![0u32; num_experts as usize + 1];
    for (i, c) in counts.iter().enumerate() {
        offsets[i + 1] = offsets[i] + c;
    }

    let a_dev = stream.memcpy_stod(&a_bf16).unwrap();
    let b_dev = stream.memcpy_stod(&b_bf16).unwrap();
    let off_dev = stream.memcpy_stod(&offsets).unwrap();
    let mut c_dev = stream
        .alloc_zeros::<u16>((total_tokens * n) as usize)
        .unwrap();

    moe::gemm_bf16_grouped_mma(
        &ctx,
        &stream,
        &a_dev,
        &b_dev,
        &mut c_dev,
        &off_dev,
        num_experts,
        16,
        n,
        k,
    )
    .unwrap();

    let c = bf16_f32(&stream.memcpy_dtov(&c_dev).unwrap());

    // Reference: direct FP32 matmul per expert.
    let mut max_diff = 0f32;
    for e in 0..num_experts as usize {
        let row_start = offsets[e] as usize;
        let row_end = offsets[e + 1] as usize;
        let b_base = e * (k * n) as usize;
        for r in row_start..row_end {
            for col in 0..n as usize {
                let mut acc = 0f32;
                for kk in 0..k as usize {
                    let a_val = a[r * k as usize + kk];
                    let b_val = b[b_base + kk * n as usize + col];
                    acc += a_val * b_val;
                }
                let got = c[r * n as usize + col];
                let d = (got - acc).abs();
                if d > max_diff {
                    max_diff = d;
                }
            }
        }
    }
    eprintln!("grouped mma vs naive: max_diff={max_diff:.4}");
    assert!(max_diff <= 0.5, "grouped mma max_diff too high: {max_diff}");
}
