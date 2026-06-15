//! Validate linear_attn_y_inter_mma against CPU reference.
//!
//! Computes Y_inter = Q @ S^T, where Q: [B, H, C=32, D=128] BF16 and
//! S: [B, H, D, D] FP16. Output Y_inter: [B, H, C, D] FP32.

use half::{bf16, f16};
use sm121_kernels::{device, linear_attention};

const C: usize = 32;
const D: usize = 128;

fn random_bf16(n: usize, seed: u64) -> Vec<u16> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let f = (((s >> 33) as u32 as f32) / u32::MAX as f32 - 0.5) * 0.5;
            bf16::from_f32(f).to_bits()
        })
        .collect()
}

fn random_f32(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((s >> 33) as u32 as f32) / u32::MAX as f32 - 0.5) * 0.5
        })
        .collect()
}

/// CPU reference: Y_inter[c, d_out] = Σ_d_in Q[c, d_in] * S[d_out, d_in].
/// The GPU now reads S as FP16 from global, then downcasts FP16→f32→BF16 in
/// SMEM. Model that: round through f16 then through bf16.
fn cpu_y_inter(q: &[u16], s: &[f32]) -> Vec<f32> {
    let mut y = vec![0f32; C * D];
    // S is stored FP16 in global, then staged BF16 in SMEM.
    let s_bf16: Vec<f32> = s
        .iter()
        .map(|&v| {
            let v_f16 = f16::from_f32(v).to_f32();
            bf16::from_f32(v_f16).to_f32()
        })
        .collect();
    for c in 0..C {
        for d_out in 0..D {
            let mut acc = 0f32;
            for d_in in 0..D {
                let q_val = bf16::from_bits(q[c * D + d_in]).to_f32();
                acc += q_val * s_bf16[d_out * D + d_in];
            }
            y[c * D + d_out] = acc;
        }
    }
    y
}

fn run_shape(batch: u32, num_heads: u32) {
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();

    let bh = (batch * num_heads) as usize;
    let q = random_bf16(bh * C * D, 0xA);
    let s = random_f32(bh * D * D, 0xB);

    let mut y_ref = vec![0f32; bh * C * D];
    for i in 0..bh {
        let q_slice = &q[i * C * D..(i + 1) * C * D];
        let s_slice = &s[i * D * D..(i + 1) * D * D];
        let y_chunk = cpu_y_inter(q_slice, s_slice);
        y_ref[i * C * D..(i + 1) * C * D].copy_from_slice(&y_chunk);
    }

    let s_f16: Vec<u16> = s.iter().map(|&x| f16::from_f32(x).to_bits()).collect();
    let q_dev = stream.memcpy_stod(&q).unwrap();
    let s_dev = stream.memcpy_stod(&s_f16).unwrap();
    let mut y_dev = stream.alloc_zeros::<f32>(bh * C * D).unwrap();

    linear_attention::linear_attn_y_inter_mma(
        &ctx, &stream, &q_dev, &s_dev, &mut y_dev, batch, num_heads,
    )
    .unwrap();

    let y_got = stream.memcpy_dtov(&y_dev).unwrap();

    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    let mut max_mag = 0f32;
    for (g, r) in y_got.iter().zip(y_ref.iter()) {
        let d = (g - r).abs();
        if d > max_abs {
            max_abs = d;
        }
        let m = g.abs().max(r.abs());
        if m > max_mag {
            max_mag = m;
        }
        if m > 0.01 {
            let rel = d / m;
            if rel > max_rel {
                max_rel = rel;
            }
        }
    }
    eprintln!(
        "B={batch} H={num_heads}: max_abs={max_abs:.4} max_rel={max_rel:.4} max_mag={max_mag:.2}"
    );
    assert!(max_rel < 0.05, "max_rel {max_rel} > 0.05");
}

#[test]
fn t_y_inter_b1_h1() {
    run_shape(1, 1);
}
#[test]
fn t_y_inter_b1_h4() {
    run_shape(1, 4);
}
#[test]
fn t_y_inter_b2_h8() {
    run_shape(2, 8);
}
