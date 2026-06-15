//! Validate linear_attn_state_update_mma against CPU reference.
//!
//! Computes S_new = S_init + V^T @ K, where V, K: [B, H, C=32, D=128] BF16
//! and S_init/S_new: [B, H, D, D] FP16 (FP32 accumulate).

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
            ((s >> 33) as u32 as f32) / u32::MAX as f32 - 0.5
        })
        .collect()
}

fn cpu_state_update(v: &[u16], k: &[u16], s_in: &[f32]) -> Vec<f32> {
    let mut s_out = s_in.to_vec();
    // S_out[d_out, d_in] += Σ_c V[c, d_out] * K[c, d_in]
    for c in 0..C {
        for d_out in 0..D {
            let v_val = bf16::from_bits(v[c * D + d_out]).to_f32();
            for d_in in 0..D {
                let k_val = bf16::from_bits(k[c * D + d_in]).to_f32();
                s_out[d_out * D + d_in] += v_val * k_val;
            }
        }
    }
    s_out
}

fn run_shape(batch: u32, num_heads: u32) {
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();

    let bh = (batch * num_heads) as usize;
    let v = random_bf16(bh * C * D, 0xA);
    let k = random_bf16(bh * C * D, 0xB);
    let s_in = random_f32(bh * D * D, 0xC);

    // CPU ref (per (b, h) independently)
    let mut s_ref = s_in.clone();
    for i in 0..bh {
        let v_slice = &v[i * C * D..(i + 1) * C * D];
        let k_slice = &k[i * C * D..(i + 1) * C * D];
        let s_in_slice = &s_in[i * D * D..(i + 1) * D * D];
        let s_out_slice = cpu_state_update(v_slice, k_slice, s_in_slice);
        s_ref[i * D * D..(i + 1) * D * D].copy_from_slice(&s_out_slice);
    }

    // State is FP16 now: store/read as f16 bits.
    let s_in_f16: Vec<u16> = s_in.iter().map(|&x| f16::from_f32(x).to_bits()).collect();
    let v_dev = stream.memcpy_stod(&v).unwrap();
    let k_dev = stream.memcpy_stod(&k).unwrap();
    let s_in_dev = stream.memcpy_stod(&s_in_f16).unwrap();
    let mut s_out_dev = stream.memcpy_stod(&s_in_f16).unwrap();

    linear_attention::linear_attn_state_update_mma(
        &ctx,
        &stream,
        &v_dev,
        &k_dev,
        &s_in_dev,
        &mut s_out_dev,
        batch,
        num_heads,
    )
    .unwrap();

    let s_got_bits = stream.memcpy_dtov(&s_out_dev).unwrap();
    let s_got: Vec<f32> = s_got_bits
        .iter()
        .map(|&b| f16::from_bits(b).to_f32())
        .collect();

    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    let mut max_mag = 0f32;
    for (g, r) in s_got.iter().zip(s_ref.iter()) {
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
    // FP16 state storage: loosen rel tolerance to ~1e-2 (FP16 has ~3 decimal digits).
    assert!(
        max_rel < 0.02,
        "max_rel {max_rel} > 0.02 (max_abs={max_abs}, max_mag={max_mag})"
    );
}

#[test]
fn t_state_update_b1_h1() {
    run_shape(1, 1);
}
#[test]
fn t_state_update_b1_h4() {
    run_shape(1, 4);
}
#[test]
fn t_state_update_b2_h8() {
    run_shape(2, 8);
}
