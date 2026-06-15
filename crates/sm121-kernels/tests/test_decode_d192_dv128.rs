//! Validate fa_bf16_decode_d192_dv128_gqa against an FP32-accumulated host
//! reference. Asymmetric K/V flash attention: Q,K have d_qk=192;
//! V,O have d_v=128.
//!
//! Used by DeepSeek-V3-style attention where K = concat(k_nope, k_rope)
//! is 128+64=192 and V keeps its 128-dim head.

use half::bf16;
use sm121_kernels::{attention, device};

fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

fn u16_from_bf16(f: f32) -> u16 {
    bf16::from_f32(f).to_bits()
}

fn deterministic_bf16(seed: u64, n: usize) -> Vec<u16> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let f = (((s >> 33) as u32 as f32) / u32::MAX as f32 - 0.5) * 0.4;
            u16_from_bf16(f)
        })
        .collect()
}

/// Host reference: full FP32 attention
///   for each q_head h:
///     scores[k] = (Q[h] · K[kv_head, k]) * scale
///     softmax
///     O[h, d_v] = sum_k softmax[k] * V[kv_head, k, d_v]
fn host_decode_asymmetric(
    q: &[u16],
    k: &[u16],
    v: &[u16],
    batch: usize,
    num_heads: usize,
    num_kv_heads: usize,
    seq_kv: usize,
    kv_stride: usize,
    d_qk: usize,
    d_v: usize,
    scale: f32,
) -> Vec<f32> {
    let mut o = vec![0f32; batch * num_heads * d_v];
    let q_per_kv = num_heads / num_kv_heads;
    for b in 0..batch {
        for h in 0..num_heads {
            let kv_h = h / q_per_kv;
            // Compute scores[k]
            let mut scores = vec![0f32; seq_kv];
            for k_idx in 0..seq_kv {
                let mut s = 0f32;
                for d in 0..d_qk {
                    let q_idx = ((b * num_heads + h) * d_qk) + d;
                    let k_offset =
                        ((b * num_kv_heads + kv_h) * kv_stride * d_qk) + (k_idx * d_qk) + d;
                    s += bf16_to_f32(q[q_idx]) * bf16_to_f32(k[k_offset]);
                }
                scores[k_idx] = s * scale;
            }
            // Softmax
            let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut exps = vec![0f32; seq_kv];
            let mut sum = 0f32;
            for k_idx in 0..seq_kv {
                let e = (scores[k_idx] - m).exp();
                exps[k_idx] = e;
                sum += e;
            }
            let inv_sum = 1.0 / sum;
            for k_idx in 0..seq_kv {
                exps[k_idx] *= inv_sum;
            }
            // O[h, d_v] = sum_k exps[k] * V[kv_h, k, d_v]
            for d in 0..d_v {
                let mut acc = 0f32;
                for k_idx in 0..seq_kv {
                    let v_offset =
                        ((b * num_kv_heads + kv_h) * kv_stride * d_v) + (k_idx * d_v) + d;
                    acc += exps[k_idx] * bf16_to_f32(v[v_offset]);
                }
                let o_idx = ((b * num_heads + h) * d_v) + d;
                o[o_idx] = acc;
            }
        }
    }
    o
}

fn run_case(
    batch: u32,
    num_heads: u32,
    num_kv_heads: u32,
    seq_kv: u32,
    kv_stride: u32,
    seed_q: u64,
    seed_k: u64,
    seed_v: u64,
) {
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();
    let d_qk = 192u32;
    let d_v = 128u32;

    let q_n = (batch * num_heads * d_qk) as usize;
    let k_n = (batch * num_kv_heads * kv_stride * d_qk) as usize;
    let v_n = (batch * num_kv_heads * kv_stride * d_v) as usize;
    let o_n = (batch * num_heads * d_v) as usize;

    let q_host = deterministic_bf16(seed_q, q_n);
    let k_host = deterministic_bf16(seed_k, k_n);
    let v_host = deterministic_bf16(seed_v, v_n);

    let q_dev = stream.memcpy_stod(&q_host).unwrap();
    let k_dev = stream.memcpy_stod(&k_host).unwrap();
    let v_dev = stream.memcpy_stod(&v_host).unwrap();
    let mut o_dev = stream.alloc_zeros::<u16>(o_n).unwrap();

    let scale = 1.0f32 / (d_qk as f32).sqrt();
    attention::flash_attn_bf16_decode_d192_dv128_gqa(
        &ctx,
        &stream,
        &q_dev,
        &k_dev,
        &v_dev,
        &mut o_dev,
        batch,
        num_heads,
        num_kv_heads,
        seq_kv,
        kv_stride,
        scale,
    )
    .expect("kernel launch");
    stream.synchronize().ok();

    let o_dev_host = stream.memcpy_dtov(&o_dev).unwrap();

    // Reference uses the seq_kv prefix of the cache only (matching kernel behaviour).
    // Only the first seq_kv * d positions per head matter.
    let o_ref = host_decode_asymmetric(
        &q_host,
        &k_host,
        &v_host,
        batch as usize,
        num_heads as usize,
        num_kv_heads as usize,
        seq_kv as usize,
        kv_stride as usize,
        d_qk as usize,
        d_v as usize,
        scale,
    );

    let mut max_diff = 0f32;
    let mut mean_diff = 0f32;
    for (got_bits, want) in o_dev_host.iter().zip(o_ref.iter()) {
        let g = bf16_to_f32(*got_bits);
        let d = (g - want).abs();
        if d > max_diff {
            max_diff = d;
        }
        mean_diff += d;
    }
    mean_diff /= o_dev_host.len() as f32;
    eprintln!(
        "  d_qk=192 d_v=128 GQA decode (B={batch}, H={num_heads}, KV={num_kv_heads}, \
         seq_kv={seq_kv}, kv_stride={kv_stride}): \
         max_diff={max_diff:.4} mean_diff={mean_diff:.4}",
    );
    assert!(
        max_diff <= 0.05,
        "asymmetric d=192/dv=128 GQA decode max_diff={max_diff} exceeds 0.05"
    );
}

#[test]
fn test_d192_dv128_b1_h4_kv1_skv8() {
    run_case(1, 4, 1, 8, 64, 0xA1, 0xA2, 0xA3);
}

#[test]
fn test_d192_dv128_b1_h8_kv1_skv32() {
    run_case(1, 8, 1, 32, 128, 0xB1, 0xB2, 0xB3);
}

#[test]
fn test_d192_dv128_b2_h16_kv4_skv64() {
    run_case(2, 16, 4, 64, 128, 0xC1, 0xC2, 0xC3);
}

#[test]
fn test_d192_dv128_b1_h32_kv4_skv1() {
    // Edge case: seq_kv=1 (first decode token).
    run_case(1, 32, 4, 1, 64, 0xD1, 0xD2, 0xD3);
}
