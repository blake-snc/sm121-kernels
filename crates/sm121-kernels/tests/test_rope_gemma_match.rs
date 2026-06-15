//! Standalone RoPE comparison: sm121-kernels rope_partial_bf16_pos_dev vs
//! PyTorch's apply_rotary_pos_emb formula computed on host in FP32.
//!
//! Tests two configurations:
//! - Gemma sliding: theta=10000, head_dim=256, rotary_dim=256 (full)
//! - Gemma full attn: theta=1000000, head_dim=512, rotary_dim=128 (partial 0.25)
//!
//! Reference (PyTorch convention):
//!   rotate_half(x) = [-x[dim/2:], x[:dim/2]]
//!   x_rot = x * cos + rotate_half(x) * sin
//! where cos/sin shape [dim] with second half duplicating first half:
//!   inv_freq[i] = theta^(-2i/dim) for i in [0, dim/2)
//!   freqs[i] = pos * inv_freq[i]
//!   cos[i] = cos(freqs[i]), cos[i+dim/2] = cos[i]  (duplicated)
//!   sin[i] = sin(freqs[i]), sin[i+dim/2] = sin[i]

use half::bf16;
use sm121_kernels::{device, rope};

fn bf16_to_u16(v: f32) -> u16 {
    bf16::from_f32(v).to_bits()
}
fn u16_to_f32(b: u16) -> f32 {
    bf16::from_bits(b).to_f32()
}

/// Reference RoPE in FP32 host-side, matching PyTorch's apply_rotary_pos_emb.
fn apply_rope_reference(
    x: &[f32],
    pos: u32,
    heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    theta: f32,
) -> Vec<f32> {
    let half = rotary_dim / 2;
    // inv_freq[i] = theta^(-2i / rotary_dim), i in [0, half)
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| (theta).powf(-(2.0 * i as f32) / (rotary_dim as f32)))
        .collect();
    let cos_v: Vec<f32> = inv_freq.iter().map(|&iv| (pos as f32 * iv).cos()).collect();
    let sin_v: Vec<f32> = inv_freq.iter().map(|&iv| (pos as f32 * iv).sin()).collect();

    let mut out = x.to_vec();
    for h in 0..heads {
        let base = h * head_dim;
        // Apply to first `rotary_dim` of each head; trailing pass through.
        for i in 0..half {
            let lo = x[base + i];
            let hi = x[base + i + half];
            // result_lo = lo*cos[i] - hi*sin[i]
            // result_hi = lo*sin[i] + hi*cos[i]
            out[base + i] = lo * cos_v[i] - hi * sin_v[i];
            out[base + i + half] = lo * sin_v[i] + hi * cos_v[i];
        }
        // Trailing dims (rotary_dim..head_dim) pass through unchanged (already copied).
    }
    out
}

fn run_test(theta: f32, heads: u32, head_dim: u32, rotary_dim: u32, positions: &[u32]) {
    let ctx = match device::init_device(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[skip] init_device: {e:?}");
            return;
        }
    };
    let stream = ctx.default_stream();

    // Deterministic random input — small magnitudes so BF16 has decent precision.
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
    // Round through BF16 for fair comparison.
    let x_round: Vec<f32> = x_f32.iter().map(|&v| bf16::from_f32(v).to_f32()).collect();
    let x_bf16: Vec<u16> = x_round.iter().map(|&v| bf16_to_u16(v)).collect();

    for &pos in positions {
        // Run sm121-kernels rope_partial.
        let mut x_dev = stream.memcpy_stod(&x_bf16).expect("htod");
        let mut pos_dev = stream.alloc_zeros::<u32>(1).expect("alloc pos");
        stream
            .memcpy_htod(&[pos][..], &mut pos_dev)
            .expect("htod pos");
        rope::rope_partial_bf16_pos_dev(
            &ctx, &stream, &mut x_dev, &pos_dev, theta, heads, head_dim, rotary_dim,
        )
        .expect("rope launch");
        stream.synchronize().ok();
        let got_bf16 = stream.memcpy_dtov(&x_dev).expect("dtoh");
        let got_f32: Vec<f32> = got_bf16.iter().map(|&b| u16_to_f32(b)).collect();

        // Reference.
        let ref_f32 = apply_rope_reference(
            &x_round,
            pos,
            heads as usize,
            head_dim as usize,
            rotary_dim as usize,
            theta,
        );

        // Compare.
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

        let head_got: Vec<String> = got_f32.iter().take(6).map(|v| format!("{v:+.4}")).collect();
        let head_ref: Vec<String> = ref_f32.iter().take(6).map(|v| format!("{v:+.4}")).collect();
        eprintln!(
            "  RoPE theta={theta:.0} heads={heads} head_dim={head_dim} rotary_dim={rotary_dim} pos={pos}: max_diff={max_diff:.4} rmse={rmse:.4} rel={rel:.4}"
        );
        eprintln!("    got: [{}]", head_got.join(", "));
        eprintln!("    ref: [{}]", head_ref.join(", "));
        assert!(
            rel < 0.05,
            "RoPE mismatch at theta={theta} pos={pos}: rel={rel:.4}\n  got: {:?}\n  ref: {:?}",
            &got_f32[..8],
            &ref_f32[..8],
        );
    }
}

#[test]
fn test_rope_gemma_sliding_full_dim() {
    // Gemma sliding: theta=10000, head_dim=256, rotary_dim=256 (full)
    run_test(10000.0, 8, 256, 256, &[0, 1, 5, 13]);
}

#[test]
fn test_rope_gemma_full_attn_partial() {
    // Gemma full attn: theta=1000000, head_dim=512, rotary_dim=128 (partial 0.25)
    run_test(1000000.0, 8, 512, 128, &[0, 1, 5, 13]);
}

#[test]
fn test_rope_pos_zero_is_identity() {
    // pos=0: cos=1, sin=0, output should equal input.
    run_test(10000.0, 2, 256, 256, &[0]);
}
