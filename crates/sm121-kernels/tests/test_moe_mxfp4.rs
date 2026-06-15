//! MXFP4 grouped GEMM: E2M1 weights with UE8M0 scales per 32-K-block. gpt-oss-120b path.
use half::bf16;
use sm121_kernels::{device, moe};

fn bf16_f32(x: &[u16]) -> Vec<f32> {
    x.iter().map(|b| bf16::from_bits(*b).to_f32()).collect()
}

fn decode_e2m1(n: u8) -> f32 {
    let lut = [0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let sign = if (n & 0x8) != 0 { -1.0 } else { 1.0 };
    sign * lut[(n & 0x7) as usize]
}

#[test]
fn t_mxfp4_uniform() {
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();
    let num_experts = 1u32;
    let tokens_per_expert = 4u32;
    let n = 32u32;
    let k = 128u32; // 4 K-blocks of 32
    let total_tokens = num_experts * tokens_per_expert;

    // A = 1.0, B nibble 2 (=+1.0) both sides → byte 0x22. Scale byte 127 → 2^0 = 1.0.
    let a: Vec<u16> = vec![bf16::from_f32(1.0).to_bits(); (total_tokens * k) as usize];
    let b: Vec<u8> = vec![0x22u8; (num_experts * n * k / 2) as usize];
    let scales: Vec<u8> = vec![127u8; (num_experts * n * (k / 32)) as usize];
    let offsets: Vec<u32> = (0..=num_experts).map(|e| e * tokens_per_expert).collect();

    let a_dev = stream.memcpy_stod(&a).unwrap();
    let b_dev = stream.memcpy_stod(&b).unwrap();
    let s_dev = stream.memcpy_stod(&scales).unwrap();
    let off_dev = stream.memcpy_stod(&offsets).unwrap();
    let mut c_dev = stream
        .alloc_zeros::<u16>((total_tokens * n) as usize)
        .unwrap();

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

    let c = bf16_f32(&stream.memcpy_dtov(&c_dev).unwrap());
    let expected = k as f32;
    let mut max_diff = 0f32;
    for v in &c {
        let d = (v - expected).abs();
        if d > max_diff {
            max_diff = d;
        }
    }
    eprintln!("mxfp4 uniform max_diff={max_diff:.4}");
    assert!(max_diff < 1.0, "mxfp4 uniform diverges: {max_diff}");
}

#[test]
fn t_mxfp4_scale_variation() {
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();
    let num_experts = 1u32;
    let tokens_per_expert = 4u32;
    let n = 32u32;
    let k = 128u32; // 4 K-blocks
    let total_tokens = num_experts * tokens_per_expert;

    // A = 1.0, B 0x22 (+1.0). Scales per K-block: [125, 126, 127, 128] = [0.25, 0.5, 1.0, 2.0]
    let a: Vec<u16> = vec![bf16::from_f32(1.0).to_bits(); (total_tokens * k) as usize];
    let b: Vec<u8> = vec![0x22u8; (num_experts * n * k / 2) as usize];
    let mut scales = Vec::new();
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

    let c = bf16_f32(&stream.memcpy_dtov(&c_dev).unwrap());
    // Per row: 32 * (0.25 + 0.5 + 1.0 + 2.0) = 32 * 3.75 = 120
    let expected = 32.0 * (0.25 + 0.5 + 1.0 + 2.0);
    let mut max_diff = 0f32;
    for v in &c {
        let d = (v - expected).abs();
        if d > max_diff {
            max_diff = d;
        }
    }
    eprintln!("mxfp4 scale_variation max_diff={max_diff:.4} expected={expected}");
    assert!(max_diff < 1.0, "mxfp4 scale_variation diverges: {max_diff}");
}

#[test]
fn t_mxfp4_v2_matches_v1() {
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
    let b: Vec<u8> = (0..(num_experts * n * k / 2) as usize)
        .map(|i| ((i.wrapping_mul(53) + 7) % 256) as u8)
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

    moe::gemm_mxfp4_grouped_mma(
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
    moe::gemm_mxfp4_grouped_mma_v2(
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
    eprintln!("mxfp4 v2 vs v1: max_diff={max_diff:.4}");
    assert!(max_diff < 0.5, "mxfp4 v2 diverges from v1: {max_diff}");
}

#[test]
fn t_mxfp4_reference_match() {
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();
    let num_experts = 2u32;
    let tokens_per_expert = 16u32;
    let n = 32u32;
    let k = 128u32;
    let total_tokens = num_experts * tokens_per_expert;

    let mut state: u64 = 0xDECAF123;
    let mut rnd = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as u32 as f32 / u32::MAX as f32 - 0.5
    };

    let a_f32: Vec<f32> = (0..(total_tokens * k)).map(|_| rnd() * 2.0).collect();
    let a: Vec<u16> = a_f32.iter().map(|f| bf16::from_f32(*f).to_bits()).collect();
    let b: Vec<u8> = (0..(num_experts * n * k / 2) as usize)
        .map(|i| ((i.wrapping_mul(53) + 7) % 256) as u8)
        .collect();
    // UE8M0 scales in range [125..130] → [0.25..4.0]
    let scales: Vec<u8> = (0..(num_experts * n * (k / 32)) as usize)
        .map(|i| 125 + (i % 6) as u8)
        .collect();
    let offsets: Vec<u32> = (0..=num_experts).map(|e| e * tokens_per_expert).collect();

    let a_dev = stream.memcpy_stod(&a).unwrap();
    let b_dev = stream.memcpy_stod(&b).unwrap();
    let s_dev = stream.memcpy_stod(&scales).unwrap();
    let off_dev = stream.memcpy_stod(&offsets).unwrap();
    let mut c_dev = stream
        .alloc_zeros::<u16>((total_tokens * n) as usize)
        .unwrap();

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

    let c = bf16_f32(&stream.memcpy_dtov(&c_dev).unwrap());

    // CPU reference
    let k_blocks = (k / 32) as usize;
    let n_usize = n as usize;
    let k_usize = k as usize;
    let mut max_rel = 0f32;
    let mut max_abs = 0f32;
    for r in 0..total_tokens as usize {
        let e = r / tokens_per_expert as usize;
        let a_row = &a_f32[r * k_usize..(r + 1) * k_usize];
        for col in 0..n_usize {
            let mut acc = 0f32;
            for kk in 0..k_usize {
                let byte_idx = e * n_usize * (k_usize / 2) + col * (k_usize / 2) + kk / 2;
                let nib = if kk & 1 == 0 {
                    b[byte_idx] & 0xF
                } else {
                    b[byte_idx] >> 4
                };
                let bv = decode_e2m1(nib);
                let scale_idx = e * n_usize * k_blocks + col * k_blocks + kk / 32;
                let sv = f32::from_bits((scales[scale_idx] as u32) << 23);
                acc += a_row[kk] * bv * sv;
            }
            let got = c[r * n_usize + col];
            let d = (got - acc).abs();
            if d > max_abs {
                max_abs = d;
            }
            let m = got.abs().max(acc.abs());
            if m > 2.0 {
                let rel = d / m;
                if rel > max_rel {
                    max_rel = rel;
                }
            }
        }
    }
    eprintln!("mxfp4 vs CPU ref: max_abs={max_abs:.4} max_rel={max_rel:.4}");
    assert!(max_abs < 5.0, "mxfp4 reference max_abs too high: {max_abs}");
    // Higher relative tolerance than NVFP4: UE8M0 power-of-2 scales amplify
    // BF16 rounding errors per K-block boundary.
    assert!(
        max_rel < 0.10,
        "mxfp4 reference max_rel too high: {max_rel}"
    );
}
