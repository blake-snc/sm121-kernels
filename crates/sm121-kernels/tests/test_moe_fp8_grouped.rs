//! Generic FP8 grouped GEMM with per-expert scalar scale.
use half::bf16;
use sm121_kernels::{device, moe};

fn bf16_f32(x: &[u16]) -> Vec<f32> {
    x.iter().map(|b| bf16::from_bits(*b).to_f32()).collect()
}

#[test]
fn t_fp8_grouped_v2_matches_v1() {
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();
    let num_experts = 4u32;
    let tokens_per_expert = 16u32;
    let n = 32u32;
    let k = 128u32;
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
    let scales: Vec<f32> = vec![0.5; num_experts as usize];
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
    moe::gemm_fp8_grouped_mma(
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
    moe::gemm_fp8_grouped_mma_v2(
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
    eprintln!("fp8_grouped v2 vs v1: max_diff={max_diff:.4}");
    assert!(
        max_diff < 0.5,
        "fp8_grouped v2 diverges from v1: {max_diff}"
    );
}

#[test]
fn t_fp8_grouped_uniform_scale_1() {
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();
    let num_experts = 1u32;
    let tokens_per_expert = 4u32;
    let n = 32u32;
    let k = 64u32;
    let total_tokens = num_experts * tokens_per_expert;

    let a: Vec<u16> = vec![bf16::from_f32(1.0).to_bits(); (total_tokens * k) as usize];
    let b: Vec<u8> = vec![0x38u8; (num_experts * n * k) as usize]; // FP8 1.0
    let scales: Vec<f32> = vec![1.0f32; num_experts as usize];
    let offsets: Vec<u32> = (0..=num_experts).map(|e| e * tokens_per_expert).collect();

    let a_dev = stream.memcpy_stod(&a).unwrap();
    let b_dev = stream.memcpy_stod(&b).unwrap();
    let s_dev = stream.memcpy_stod(&scales).unwrap();
    let off_dev = stream.memcpy_stod(&offsets).unwrap();
    let mut c_dev = stream
        .alloc_zeros::<u16>((total_tokens * n) as usize)
        .unwrap();

    moe::gemm_fp8_grouped_mma(
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
    eprintln!("fp8 grouped uniform max_diff={max_diff:.4}");
    assert!(max_diff < 1.0, "fp8 grouped uniform diverges: {max_diff}");
}

#[test]
fn t_fp8_grouped_per_expert_scale() {
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();
    let num_experts = 3u32;
    let tokens_per_expert = 4u32;
    let n = 32u32;
    let k = 64u32;
    let total_tokens = num_experts * tokens_per_expert;

    // A all 1.0, B all 0x38 (1.0 FP8). Expert scales: [0.5, 1.0, 2.0]
    // Expected per row in expert e: 64 * 1 * 1 * scale[e]
    let a: Vec<u16> = vec![bf16::from_f32(1.0).to_bits(); (total_tokens * k) as usize];
    let b: Vec<u8> = vec![0x38u8; (num_experts * n * k) as usize];
    let scales: Vec<f32> = vec![0.5f32, 1.0, 2.0];
    let offsets: Vec<u32> = (0..=num_experts).map(|e| e * tokens_per_expert).collect();

    let a_dev = stream.memcpy_stod(&a).unwrap();
    let b_dev = stream.memcpy_stod(&b).unwrap();
    let s_dev = stream.memcpy_stod(&scales).unwrap();
    let off_dev = stream.memcpy_stod(&offsets).unwrap();
    let mut c_dev = stream
        .alloc_zeros::<u16>((total_tokens * n) as usize)
        .unwrap();

    moe::gemm_fp8_grouped_mma(
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
    // For each row r, expert = r / tokens_per_expert. Output = k * scale[expert].
    let mut max_diff = 0f32;
    for r in 0..total_tokens as usize {
        let e = r / tokens_per_expert as usize;
        let expected = k as f32 * scales[e];
        for col in 0..n as usize {
            let v = c[r * n as usize + col];
            let d = (v - expected).abs();
            if d > max_diff {
                max_diff = d;
                if d > 1.0 {
                    eprintln!(
                        "fp8 grouped: row={r} col={col} expert={e} got={v} expected={expected}"
                    );
                }
            }
        }
    }
    eprintln!("fp8 grouped per-expert max_diff={max_diff:.4}");
    assert!(
        max_diff < 1.0,
        "fp8 grouped per-expert diverges: {max_diff}"
    );
}
