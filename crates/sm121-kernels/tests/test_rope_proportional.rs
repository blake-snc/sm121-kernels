//! Validates `rope_proportional_bf16_pos_dev` against a host FP32 reference.
//!
//! Tests Gemma-4 full-attention RoPE: pair `(i, i+head_dim/2)`, rotation only
//! for `i in [0, rope_angles)`, identity (cos=1, sin=0) for `i in [rope_angles,
//! head_dim/2)`. Verifies math matches transformers'
//! `_compute_proportional_rope_parameters` + `apply_rotary_pos_emb`.

use half::bf16;
use sm121_kernels::{device, rope};

fn bf16_to_u16(v: f32) -> u16 {
    bf16::from_f32(v).to_bits()
}
fn u16_to_f32(b: u16) -> f32 {
    bf16::from_bits(b).to_f32()
}

/// Reference proportional RoPE on host FP32. Matches the validator's
/// `host_rope_proportional` in `examples/gemma4_validate.rs`.
fn apply_rope_proportional_reference(
    x: &[f32],
    pos: u32,
    heads: usize,
    head_dim: usize,
    rope_angles: usize,
    theta: f32,
) -> Vec<f32> {
    let half = head_dim / 2;
    let mut inv_freq = vec![0.0f32; half];
    for i in 0..rope_angles {
        inv_freq[i] = theta.powf(-(2.0 * i as f32) / (head_dim as f32));
    }
    let cos_v: Vec<f32> = inv_freq.iter().map(|&iv| (pos as f32 * iv).cos()).collect();
    let sin_v: Vec<f32> = inv_freq.iter().map(|&iv| (pos as f32 * iv).sin()).collect();
    let mut out = x.to_vec();
    for h in 0..heads {
        let base = h * head_dim;
        for i in 0..half {
            let lo = x[base + i];
            let hi = x[base + i + half];
            out[base + i] = lo * cos_v[i] - hi * sin_v[i];
            out[base + i + half] = lo * sin_v[i] + hi * cos_v[i];
        }
    }
    out
}

fn run_test(theta: f32, heads: u32, head_dim: u32, rope_angles: u32, positions: &[u32]) {
    let ctx = match device::init_device(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[skip] init_device: {e:?}");
            return;
        }
    };
    let stream = ctx.default_stream();

    let mut rng_state: u64 = 12345;
    let mut next_f = || -> f32 {
        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = ((rng_state >> 32) as u32) & 0x007fffff;
        f32::from_bits(0x3f800000 | bits) - 1.5
    };
    let n = (heads * head_dim) as usize;
    let x_f32: Vec<f32> = (0..n).map(|_| next_f()).collect();
    let x_round: Vec<f32> = x_f32.iter().map(|&v| bf16::from_f32(v).to_f32()).collect();
    let x_bf16: Vec<u16> = x_round.iter().map(|&v| bf16_to_u16(v)).collect();

    for &pos in positions {
        let mut x_dev = stream.memcpy_stod(&x_bf16).expect("htod");
        let mut pos_dev = stream.alloc_zeros::<u32>(1).expect("alloc pos");
        stream
            .memcpy_htod(&[pos][..], &mut pos_dev)
            .expect("htod pos");
        rope::rope_proportional_bf16_pos_dev(
            &ctx,
            &stream,
            &mut x_dev,
            &pos_dev,
            theta,
            heads,
            head_dim,
            rope_angles,
        )
        .expect("rope launch");
        stream.synchronize().ok();
        let got_bf16 = stream.memcpy_dtov(&x_dev).expect("dtoh");
        let got_f32: Vec<f32> = got_bf16.iter().map(|&b| u16_to_f32(b)).collect();

        let ref_f32 = apply_rope_proportional_reference(
            &x_round,
            pos,
            heads as usize,
            head_dim as usize,
            rope_angles as usize,
            theta,
        );

        let mut max_diff = 0.0f32;
        let mut sum_sq = 0.0f64;
        let mut sum_ref_sq = 0.0f64;
        for (g, r) in got_f32.iter().zip(ref_f32.iter()) {
            let d = (g - r).abs();
            if d > max_diff {
                max_diff = d;
            }
            sum_sq += (d as f64).powi(2);
            sum_ref_sq += (*r as f64).powi(2);
        }
        let rmse = (sum_sq / got_f32.len() as f64).sqrt() as f32;
        let ref_rms = (sum_ref_sq / got_f32.len() as f64).sqrt() as f32;
        let rel = rmse / ref_rms.max(1e-9);

        eprintln!(
            "  proportional theta={theta:.0} heads={heads} head_dim={head_dim} rope_angles={rope_angles} pos={pos}: max_diff={max_diff:.4} rmse={rmse:.4} rel={rel:.4}"
        );
        assert!(
            rel < 0.02,
            "Proportional RoPE mismatch at theta={theta} pos={pos}: rel={rel:.4}\n  got: {:?}\n  ref: {:?}",
            &got_f32[..8], &ref_f32[..8],
        );
    }
}

#[test]
fn test_rope_proportional_gemma4_full() {
    // Gemma-4-e4b full-attention: theta=1e6, head_dim=512, rope_angles=64
    // (= 0.25 * 512 / 2)
    run_test(1_000_000.0, 8, 512, 64, &[0, 1, 5, 13, 100]);
}

#[test]
fn test_rope_proportional_pos_zero_is_identity() {
    // pos=0: cos=1, sin=0 for every (rotated and non-rotated) index.
    // Output should equal input.
    run_test(1_000_000.0, 4, 512, 64, &[0]);
}

#[test]
fn test_rope_proportional_full_rotation() {
    // rope_angles == head_dim/2 → equivalent to rope_partial with rotary_dim==head_dim
    // (modulo the pair convention difference: partial uses (i, i+rotary_dim/2),
    // proportional uses (i, i+head_dim/2). When rotary_dim==head_dim, these are
    // the same pair convention.)
    run_test(10_000.0, 2, 256, 128, &[0, 7]);
}

#[test]
fn test_rope_proportional_no_rotation() {
    // rope_angles=0 → all entries pass through unchanged (cos=1, sin=0).
    run_test(1_000_000.0, 2, 64, 0, &[5]);
}
