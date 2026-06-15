//! Validate MMA-MLA QK stage against Rust reference.
//!
//! Tests the QK-only entry point `mla_decode_bf16_mma_qk` which exercises
//! the m16n8k16 BF16 MMA pipeline for Q_c @ c_kv^T + Q_r @ k_rope^T with
//! 16 heads per CTA and SWIZZLE_128B SMEM loads via ldmatrix. Compares
//! scaled scores against a Rust BF16→F32 reference.

mod common;

use sm121_kernels::{attention, device};

const D_C: usize = 512;
const D_R: usize = 64;

fn bf16_bits(f: f32) -> u16 {
    half::bf16::from_f32(f).to_bits()
}

fn bf16_to_f32(b: u16) -> f32 {
    half::bf16::from_bits(b).to_f32()
}

/// Deterministic pseudo-random BF16 values in [-1, 1).
fn gen_bf16_values(n: usize, seed: u64) -> Vec<u16> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let v = (s >> 33) as u32;
            let f = ((v & 0xFFFF) as f32 / 32768.0) - 1.0;
            bf16_bits(f * 0.5)
        })
        .collect()
}

fn reference_scores(
    q_c: &[u16],
    q_r: &[u16],
    c_kv: &[u16],
    k_rope: &[u16],
    batch: u32,
    num_heads: u32,
    seq_kv: u32,
    scale: f32,
) -> Vec<f32> {
    let b = batch as usize;
    let h = num_heads as usize;
    let sk = seq_kv as usize;
    let mut out = vec![0.0f32; b * h * sk];

    for bi in 0..b {
        for hi in 0..h {
            for si in 0..sk {
                let mut acc = 0.0f32;
                // Q_c[bi, hi, :] @ c_kv[bi, si, :]
                let q_c_base = (bi * h + hi) * D_C;
                let c_kv_base = (bi * sk + si) * D_C;
                for d in 0..D_C {
                    acc += bf16_to_f32(q_c[q_c_base + d]) * bf16_to_f32(c_kv[c_kv_base + d]);
                }
                // Q_r[bi, hi, :] @ k_rope[bi, si, :]
                let q_r_base = (bi * h + hi) * D_R;
                let kr_base = (bi * sk + si) * D_R;
                for d in 0..D_R {
                    acc += bf16_to_f32(q_r[q_r_base + d]) * bf16_to_f32(k_rope[kr_base + d]);
                }
                out[(bi * h + hi) * sk + si] = acc * scale;
            }
        }
    }
    out
}

fn run_qk(batch: u32, num_heads: u32, seq_kv: u32, tol: f32) {
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();

    let q_c = gen_bf16_values((batch * num_heads) as usize * D_C, 0xDEADBEEF);
    let q_r = gen_bf16_values((batch * num_heads) as usize * D_R, 0xCAFEBABE);
    let c_kv = gen_bf16_values((batch * seq_kv) as usize * D_C, 0x12345678);
    let k_rope = gen_bf16_values((batch * seq_kv) as usize * D_R, 0x87654321);
    let scale = 1.0f32 / ((D_C + D_R) as f32).sqrt();

    let q_c_d = stream.memcpy_stod(&q_c).unwrap();
    let q_r_d = stream.memcpy_stod(&q_r).unwrap();
    let c_kv_d = stream.memcpy_stod(&c_kv).unwrap();
    let k_rope_d = stream.memcpy_stod(&k_rope).unwrap();

    let out_len = (batch * num_heads * seq_kv) as usize;
    let mut scores_d = stream.alloc_zeros::<f32>(out_len).unwrap();

    attention::mla_decode_bf16_mma_qk(
        &ctx,
        &stream,
        &q_c_d,
        &q_r_d,
        &c_kv_d,
        &k_rope_d,
        &mut scores_d,
        batch,
        num_heads,
        seq_kv,
        scale,
    )
    .expect("QK MMA launch");

    let got = stream.memcpy_dtov(&scores_d).unwrap();
    let expected = reference_scores(&q_c, &q_r, &c_kv, &k_rope, batch, num_heads, seq_kv, scale);

    let mut max_d: f32 = 0.0;
    let mut sum_d: f32 = 0.0;
    for i in 0..out_len {
        let d = (got[i] - expected[i]).abs();
        if d > max_d {
            max_d = d;
        }
        sum_d += d;
        if d > tol {
            panic!(
                "mismatch at i={i}: got={} expected={} diff={} (B={batch} H={num_heads} Skv={seq_kv})",
                got[i], expected[i], d
            );
        }
    }
    eprintln!(
        "MMA-QK B={batch} H={num_heads} Skv={seq_kv}: max_diff={max_d:.5} mean_diff={:.6}",
        sum_d / out_len as f32
    );
}

#[test]
fn t_mma_qk_b1_h16_skv8() {
    run_qk(1, 16, 8, 0.05);
}

#[test]
fn t_mma_qk_b1_h16_skv64() {
    run_qk(1, 16, 64, 0.05);
}

#[test]
fn t_mma_qk_b2_h32_skv128() {
    run_qk(2, 32, 128, 0.08);
}
