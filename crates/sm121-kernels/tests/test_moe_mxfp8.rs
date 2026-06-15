//! MXFP8 grouped GEMM: FP8 E4M3 weights with UE8M0 scales per 32-K-block.
use half::bf16;
use sm121_kernels::{device, moe};

fn bf16_f32(x: &[u16]) -> Vec<f32> {
    x.iter().map(|b| bf16::from_bits(*b).to_f32()).collect()
}

#[test]
fn t_mxfp8_v2_matches_v1() {
    use half::bf16;
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
    let scales: Vec<u8> = (0..(num_experts * n * (k / 32)) as usize)
        .map(|i| 125 + (i % 6) as u8)
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

    moe::gemm_mxfp8_grouped_mma(
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
    moe::gemm_mxfp8_grouped_mma_v2(
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
    eprintln!("mxfp8 v2 vs v1: max_diff={max_diff:.4}");
    assert!(max_diff < 0.5, "mxfp8 v2 diverges from v1: {max_diff}");
}

#[test]
fn t_mxfp8_uniform() {
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();
    let num_experts = 1u32;
    let tokens_per_expert = 4u32;
    let n = 32u32;
    let k = 128u32; // 4 MXFP8 blocks of 32 each
    let total_tokens = num_experts * tokens_per_expert;

    // A = 1.0, B = 0x38 (FP8 1.0), scale byte = 127 (→ 2^0 = 1.0)
    let a: Vec<u16> = vec![bf16::from_f32(1.0).to_bits(); (total_tokens * k) as usize];
    let b: Vec<u8> = vec![0x38u8; (num_experts * n * k) as usize];
    let scales: Vec<u8> = vec![127u8; (num_experts * n * (k / 32)) as usize];
    let offsets: Vec<u32> = (0..=num_experts).map(|e| e * tokens_per_expert).collect();

    let a_dev = stream.memcpy_stod(&a).unwrap();
    let b_dev = stream.memcpy_stod(&b).unwrap();
    let s_dev = stream.memcpy_stod(&scales).unwrap();
    let off_dev = stream.memcpy_stod(&offsets).unwrap();
    let mut c_dev = stream
        .alloc_zeros::<u16>((total_tokens * n) as usize)
        .unwrap();

    moe::gemm_mxfp8_grouped_mma(
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
    // Expected = K * 1 * 1 * 1 = 128
    let expected = k as f32;
    let mut max_diff = 0f32;
    for v in &c {
        let d = (v - expected).abs();
        if d > max_diff {
            max_diff = d;
        }
    }
    eprintln!("mxfp8 uniform max_diff={max_diff:.4}");
    assert!(max_diff < 1.0, "mxfp8 uniform diverges: {max_diff}");
}

#[test]
fn t_mxfp8_scale_doubling() {
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();
    let num_experts = 1u32;
    let tokens_per_expert = 4u32;
    let n = 32u32;
    let k = 128u32;
    let total_tokens = num_experts * tokens_per_expert;

    // A = 1.0, B = 0x38 (1.0 FP8). Scale byte = 128 → 2^1 = 2.0 per block.
    // 4 K-blocks × 32 elements × 1.0 FP8 × 2.0 scale = 4 * 32 * 2 = 256 per row.
    let a: Vec<u16> = vec![bf16::from_f32(1.0).to_bits(); (total_tokens * k) as usize];
    let b: Vec<u8> = vec![0x38u8; (num_experts * n * k) as usize];
    let scales: Vec<u8> = vec![128u8; (num_experts * n * (k / 32)) as usize];
    let offsets: Vec<u32> = (0..=num_experts).map(|e| e * tokens_per_expert).collect();

    let a_dev = stream.memcpy_stod(&a).unwrap();
    let b_dev = stream.memcpy_stod(&b).unwrap();
    let s_dev = stream.memcpy_stod(&scales).unwrap();
    let off_dev = stream.memcpy_stod(&offsets).unwrap();
    let mut c_dev = stream
        .alloc_zeros::<u16>((total_tokens * n) as usize)
        .unwrap();

    moe::gemm_mxfp8_grouped_mma(
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
    let expected = 2.0 * k as f32;
    let mut max_diff = 0f32;
    for v in &c {
        let d = (v - expected).abs();
        if d > max_diff {
            max_diff = d;
        }
    }
    eprintln!("mxfp8 scale_doubling max_diff={max_diff:.4} expected={expected}");
    assert!(max_diff < 2.0, "mxfp8 scale_doubling diverges: {max_diff}");
}

#[test]
fn t_mxfp8_per_block_varying_scale() {
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();
    let num_experts = 1u32;
    let tokens_per_expert = 4u32;
    let n = 32u32;
    let k = 128u32; // 4 K-blocks
    let total_tokens = num_experts * tokens_per_expert;

    // A = 1.0, B = 0x38 (1.0 FP8). Scale per K-block: [125, 126, 127, 128]
    // = [0.25, 0.5, 1.0, 2.0].
    // Per row, output = 32 * (0.25 + 0.5 + 1.0 + 2.0) = 32 * 3.75 = 120.
    let a: Vec<u16> = vec![bf16::from_f32(1.0).to_bits(); (total_tokens * k) as usize];
    let b: Vec<u8> = vec![0x38u8; (num_experts * n * k) as usize];
    // Scales layout [expert, n, k_block] — for each n, [125, 126, 127, 128].
    let num_k_blocks = k / 32;
    let mut scales = Vec::with_capacity((num_experts * n * num_k_blocks) as usize);
    for _ in 0..num_experts {
        for _ in 0..n {
            scales.extend_from_slice(&[125u8, 126, 127, 128]);
        }
    }
    let offsets: Vec<u32> = (0..=num_experts).map(|e| e * tokens_per_expert).collect();

    let a_dev = stream.memcpy_stod(&a).unwrap();
    let b_dev = stream.memcpy_stod(&b).unwrap();
    let s_dev = stream.memcpy_stod(&scales).unwrap();
    let off_dev = stream.memcpy_stod(&offsets).unwrap();
    let mut c_dev = stream
        .alloc_zeros::<u16>((total_tokens * n) as usize)
        .unwrap();

    moe::gemm_mxfp8_grouped_mma(
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
    let expected = 32.0 * (0.25 + 0.5 + 1.0 + 2.0); // = 120
    let mut max_diff = 0f32;
    for v in &c {
        let d = (v - expected).abs();
        if d > max_diff {
            max_diff = d;
        }
    }
    eprintln!("mxfp8 varying_scale max_diff={max_diff:.4} expected={expected}");
    assert!(max_diff < 1.0, "mxfp8 varying_scale diverges: {max_diff}");
}
