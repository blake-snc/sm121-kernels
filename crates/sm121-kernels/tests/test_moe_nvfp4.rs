//! NVFP4 grouped GEMM: BF16 A × NVFP4 nibble-packed B with per-16-K-block FP32 scales.
//!
//! E2M1 nibble encoding:
//!   0=0, 1=0.5, 2=1.0, 3=1.5, 4=2.0, 5=3.0, 6=4.0, 7=6.0 (negatives = sign bit 8 set).
use half::bf16;
use sm121_kernels::{device, moe};

fn bf16_f32(x: &[u16]) -> Vec<f32> {
    x.iter().map(|b| bf16::from_bits(*b).to_f32()).collect()
}

/// Decode a single E2M1 nibble to f32.
fn decode_e2m1(n: u8) -> f32 {
    let lut = [0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let sign = if (n & 0x8) != 0 { -1.0 } else { 1.0 };
    sign * lut[(n & 0x7) as usize]
}

#[test]
fn t_nvfp4_uniform() {
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();
    let num_experts = 1u32;
    let tokens_per_expert = 4u32;
    let n = 32u32;
    let k = 64u32; // 4 K-blocks of 16
    let total_tokens = num_experts * tokens_per_expert;

    // A = 1.0. B = all nibbles = 2 (+1.0). Packed: 0x22 per byte. Scale = 1.0.
    let a: Vec<u16> = vec![bf16::from_f32(1.0).to_bits(); (total_tokens * k) as usize];
    let b: Vec<u8> = vec![0x22u8; (num_experts * n * k / 2) as usize];
    let scales: Vec<f32> = vec![1.0f32; (num_experts * n * (k / 16)) as usize];
    let offsets: Vec<u32> = (0..=num_experts).map(|e| e * tokens_per_expert).collect();

    let a_dev = stream.memcpy_stod(&a).unwrap();
    let b_dev = stream.memcpy_stod(&b).unwrap();
    let s_dev = stream.memcpy_stod(&scales).unwrap();
    let off_dev = stream.memcpy_stod(&offsets).unwrap();
    let mut c_dev = stream
        .alloc_zeros::<u16>((total_tokens * n) as usize)
        .unwrap();

    moe::gemm_nvfp4_grouped_mma(
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
    let expected = k as f32; // K * 1 * 1 * 1
    let mut max_diff = 0f32;
    for v in &c {
        let d = (v - expected).abs();
        if d > max_diff {
            max_diff = d;
        }
    }
    eprintln!("nvfp4 uniform max_diff={max_diff:.4}");
    assert!(max_diff < 1.0, "nvfp4 uniform diverges: {max_diff}");
}

#[test]
fn t_nvfp4_alternating_values() {
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();
    let num_experts = 1u32;
    let tokens_per_expert = 4u32;
    let n = 32u32;
    let k = 32u32; // 2 K-blocks of 16
    let total_tokens = num_experts * tokens_per_expert;

    // A = 1.0. B nibble: low=3 (+1.5), high=4 (+2.0) → byte 0x43.
    // So even-K values = 1.5, odd-K values = 2.0.
    let a: Vec<u16> = vec![bf16::from_f32(1.0).to_bits(); (total_tokens * k) as usize];
    let b: Vec<u8> = vec![0x43u8; (num_experts * n * k / 2) as usize];
    let scales: Vec<f32> = vec![1.0f32; (num_experts * n * (k / 16)) as usize];
    let offsets: Vec<u32> = (0..=num_experts).map(|e| e * tokens_per_expert).collect();

    let a_dev = stream.memcpy_stod(&a).unwrap();
    let b_dev = stream.memcpy_stod(&b).unwrap();
    let s_dev = stream.memcpy_stod(&scales).unwrap();
    let off_dev = stream.memcpy_stod(&offsets).unwrap();
    let mut c_dev = stream
        .alloc_zeros::<u16>((total_tokens * n) as usize)
        .unwrap();

    moe::gemm_nvfp4_grouped_mma(
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
    // Per row: sum over K/2 bytes of (decode(low) + decode(high)) * A=1 * scale=1
    //   = (K/2) * (1.5 + 2.0) = (K/2) * 3.5
    let expected = (k / 2) as f32 * 3.5;
    let mut max_diff = 0f32;
    for v in &c {
        let d = (v - expected).abs();
        if d > max_diff {
            max_diff = d;
        }
    }
    eprintln!("nvfp4 alternating max_diff={max_diff:.4} expected={expected}");
    assert!(max_diff < 1.0, "nvfp4 alternating diverges: {max_diff}");
}

#[test]
fn t_nvfp4_varying_a() {
    // A varies along K, B and scales uniform. Tests A loads + FP4 decode + scale.
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();
    let num_experts = 1u32;
    let tokens_per_expert = 4u32;
    let n = 32u32;
    let k = 64u32;
    let total_tokens = num_experts * tokens_per_expert;

    // A[r, kk] = 0.25 * (kk % 4)  (repeating 0, 0.25, 0.5, 0.75)
    let a_f32: Vec<f32> = (0..(total_tokens * k))
        .map(|i| 0.25 * (i % 4) as f32)
        .collect();
    let a: Vec<u16> = a_f32.iter().map(|f| bf16::from_f32(*f).to_bits()).collect();
    // B = nibble 2 (+1.0) on both halves: byte 0x22
    let b: Vec<u8> = vec![0x22u8; (num_experts * n * k / 2) as usize];
    let scales: Vec<f32> = vec![1.0f32; (num_experts * n * (k / 16)) as usize];
    let offsets: Vec<u32> = (0..=num_experts).map(|e| e * tokens_per_expert).collect();

    let a_dev = stream.memcpy_stod(&a).unwrap();
    let b_dev = stream.memcpy_stod(&b).unwrap();
    let s_dev = stream.memcpy_stod(&scales).unwrap();
    let off_dev = stream.memcpy_stod(&offsets).unwrap();
    let mut c_dev = stream
        .alloc_zeros::<u16>((total_tokens * n) as usize)
        .unwrap();

    moe::gemm_nvfp4_grouped_mma(
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
    // Per row: sum over k of A[k] * 1 * 1 = 16 * (0 + 0.25 + 0.5 + 0.75) = 16 * 1.5 = 24
    let expected = (k / 4) as f32 * 1.5;
    let mut max_diff = 0f32;
    for v in &c {
        let d = (v - expected).abs();
        if d > max_diff {
            max_diff = d;
        }
    }
    eprintln!("nvfp4 varying_a max_diff={max_diff:.4} expected={expected}");
    assert!(max_diff < 1.0, "nvfp4 varying_a diverges: {max_diff}");
}

#[test]
fn t_nvfp4_reference_match() {
    // Compare against CPU reference with full decode pipeline.
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();
    let num_experts = 2u32;
    let tokens_per_expert = 16u32;
    let n = 32u32;
    let k = 64u32;
    let total_tokens = num_experts * tokens_per_expert;

    let mut state: u64 = 0xBEEF4321;
    let mut rnd = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as u32 as f32 / u32::MAX as f32 - 0.5
    };

    let a_f32: Vec<f32> = (0..(total_tokens * k)).map(|_| rnd() * 2.0).collect();
    let a: Vec<u16> = a_f32.iter().map(|f| bf16::from_f32(*f).to_bits()).collect();
    // Random nibbles (each byte = 2 nibbles).
    let b: Vec<u8> = (0..(num_experts * n * k / 2) as usize)
        .map(|i| ((i.wrapping_mul(37) + 11) % 256) as u8)
        .collect();
    let scales_f32: Vec<f32> = (0..(num_experts * n * (k / 16)) as usize)
        .map(|_| 0.5 + rnd().abs() * 0.5)
        .collect();
    let offsets: Vec<u32> = (0..=num_experts).map(|e| e * tokens_per_expert).collect();

    let a_dev = stream.memcpy_stod(&a).unwrap();
    let b_dev = stream.memcpy_stod(&b).unwrap();
    let s_dev = stream.memcpy_stod(&scales_f32).unwrap();
    let off_dev = stream.memcpy_stod(&offsets).unwrap();
    let mut c_dev = stream
        .alloc_zeros::<u16>((total_tokens * n) as usize)
        .unwrap();

    moe::gemm_nvfp4_grouped_mma(
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
    let k_blocks = (k / 16) as usize;
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
                let scale_idx = e * n_usize * k_blocks + col * k_blocks + kk / 16;
                let sv = scales_f32[scale_idx];
                acc += a_row[kk] * bv * sv;
            }
            let got = c[r * n_usize + col];
            let d = (got - acc).abs();
            if d > max_abs {
                max_abs = d;
            }
            // Relative only meaningful at magnitudes well above BF16 noise.
            let m = got.abs().max(acc.abs());
            if m > 2.0 {
                let rel = d / m;
                if rel > max_rel {
                    max_rel = rel;
                }
            }
        }
    }
    eprintln!("nvfp4 vs CPU ref: max_abs={max_abs:.4} max_rel={max_rel:.4}");
    // Absolute tolerance: BF16 rounding × K accumulation → expect sub-1.0 on small
    // outputs. Relative tolerance for larger outputs: ~2%.
    assert!(max_abs < 2.0, "nvfp4 reference max_abs too high: {max_abs}");
    assert!(
        max_rel < 0.05,
        "nvfp4 reference max_rel too high: {max_rel}"
    );
}
