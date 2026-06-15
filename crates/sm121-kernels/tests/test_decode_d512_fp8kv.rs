//! Validate fa_bf16_fp8kv_decode_d512_gqa_pos_dev against an FP32 host
//! reference. d=512 full attention with FP8 e4m3 KV cache. Designed for
//! Gemma-4 full-attention layers (7 of 42).
//!
//! Same FP8 e4m3 quant model as test_decode_d256_fp8kv.

use half::bf16;
use sm121_kernels::{attention, device};

fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}
fn u16_from_bf16(f: f32) -> u16 {
    bf16::from_f32(f).to_bits()
}

fn f32_to_e4m3_bits(x: f32) -> u8 {
    if x.is_nan() {
        return 0x7F;
    }
    let xi = x.to_bits();
    let sign = ((xi >> 31) & 1) as u8;
    let mag = f32::from_bits(xi & 0x7fff_ffff);
    if mag >= 448.0 {
        return (sign << 7) | 0x7E;
    }
    if mag < 2f32.powi(-9) {
        return sign << 7;
    }
    let exp_f32 = ((xi >> 23) & 0xFF) as i32 - 127;
    let mant_f32 = xi & 0x007F_FFFF;
    let exp_e4m3 = exp_f32 + 7;
    if exp_e4m3 <= 0 {
        let shift = 1 - exp_e4m3;
        if shift > 24 {
            return sign << 7;
        }
        let m = (mant_f32 | 0x0080_0000) >> (16 + shift as u32);
        let m_lo = (m & 0x7) as u8;
        return (sign << 7) | m_lo;
    }
    let exp_bits = exp_e4m3.min(15) as u8;
    let m_round = mant_f32 >> 20;
    let extra = (mant_f32 >> 19) & 1;
    let mut m = m_round;
    if extra != 0 {
        m += 1;
    }
    let mut e_out = exp_bits as u32;
    if m == 8 {
        m = 0;
        e_out += 1;
    }
    if e_out > 15 {
        return (sign << 7) | 0x7E;
    }
    let bits = (e_out << 3) | m;
    (sign << 7) | bits as u8
}

fn e4m3_bits_to_f32(b: u8) -> f32 {
    let sign = if (b & 0x80) != 0 { -1.0f32 } else { 1.0 };
    let e = ((b >> 3) & 0xF) as i32;
    let m = (b & 0x7) as i32;
    if e == 0 && m == 0 {
        return 0.0 * sign;
    }
    if e == 0 {
        return sign * (m as f32) / 8.0 * 2f32.powi(-6);
    }
    sign * (1.0 + m as f32 / 8.0) * 2f32.powi(e - 7)
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

fn quantize_bf16_to_fp8_per_tensor(bf16_in: &[u16]) -> (Vec<u8>, f32) {
    let max_abs = bf16_in
        .iter()
        .map(|&b| bf16_to_f32(b).abs())
        .fold(0f32, f32::max);
    let scale = (max_abs / 448.0).max(1e-12);
    let inv_scale = 1.0 / scale;
    let bytes: Vec<u8> = bf16_in
        .iter()
        .map(|&b| f32_to_e4m3_bits(bf16_to_f32(b) * inv_scale))
        .collect();
    (bytes, scale)
}

fn host_attention_fp8kv_full(
    q_bf16: &[u16],
    k_fp8: &[u8],
    v_fp8: &[u8],
    batch: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    seq_kv: usize,
    kv_stride: usize,
    d: usize,
    scale: f32,
    kv_scale: f32,
) -> Vec<f32> {
    let mut o = vec![0f32; batch * num_q_heads * d];
    let q_per_kv = num_q_heads / num_kv_heads;
    for b in 0..batch {
        for h in 0..num_q_heads {
            let kv_h = h / q_per_kv;
            let mut scores = vec![0f32; seq_kv];
            for k_idx in 0..seq_kv {
                let mut s = 0f32;
                for dim in 0..d {
                    let q_idx = ((b * num_q_heads + h) * d) + dim;
                    let k_offset = ((b * num_kv_heads + kv_h) * kv_stride * d) + (k_idx * d) + dim;
                    let k_dq = e4m3_bits_to_f32(k_fp8[k_offset]) * kv_scale;
                    s += bf16_to_f32(q_bf16[q_idx]) * k_dq;
                }
                scores[k_idx] = s * scale;
            }
            let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut exps = vec![0f32; seq_kv];
            let mut sum = 0f32;
            for k_idx in 0..seq_kv {
                let e = (scores[k_idx] - m).exp();
                exps[k_idx] = e;
                sum += e;
            }
            let inv_sum = 1.0 / sum;
            for dim in 0..d {
                let mut acc = 0f32;
                for k_idx in 0..seq_kv {
                    let v_offset = ((b * num_kv_heads + kv_h) * kv_stride * d) + (k_idx * d) + dim;
                    let v_dq = e4m3_bits_to_f32(v_fp8[v_offset]) * kv_scale;
                    acc += exps[k_idx] * inv_sum * v_dq;
                }
                let o_idx = ((b * num_q_heads + h) * d) + dim;
                o[o_idx] = acc;
            }
        }
    }
    o
}

fn run_case(batch: u32, num_q_heads: u32, num_kv_heads: u32, q_pos: u32, kv_stride: u32) {
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();
    let d = 512u32;
    let seq_kv = q_pos + 1;

    let q_n = (batch * num_q_heads * d) as usize;
    let kv_n = (batch * num_kv_heads * kv_stride * d) as usize;

    let q_host = deterministic_bf16(0xCAFE_BABEu64.wrapping_mul(0x9E37) + q_pos as u64, q_n);
    let k_bf16 = deterministic_bf16(0xDEAD_BEEFu64 + q_pos as u64, kv_n);
    let v_bf16 = deterministic_bf16(0x1234_5678u64 + q_pos as u64, kv_n);

    let (k_fp8, k_scale) = quantize_bf16_to_fp8_per_tensor(&k_bf16);
    let (v_fp8, v_scale) = quantize_bf16_to_fp8_per_tensor(&v_bf16);
    let kv_scale = (k_scale + v_scale) * 0.5;

    let q_dev = stream.memcpy_stod(&q_host).unwrap();
    let k_dev = stream.memcpy_stod(&k_fp8).unwrap();
    let v_dev = stream.memcpy_stod(&v_fp8).unwrap();
    let mut o_dev = stream.alloc_zeros::<u16>(q_n).unwrap();
    let pos_dev = stream.memcpy_stod(&[q_pos]).unwrap();

    let scale = 1.0f32 / (d as f32).sqrt();
    attention::flash_attn_bf16_fp8kv_decode_d512_gqa_pos_dev(
        &ctx,
        &stream,
        &q_dev,
        &k_dev,
        &v_dev,
        &mut o_dev,
        batch,
        num_q_heads,
        num_kv_heads,
        kv_stride,
        &pos_dev,
        scale,
        kv_scale,
    )
    .expect("kernel launch");
    stream.synchronize().ok();

    let o_dev_host = stream.memcpy_dtov(&o_dev).unwrap();
    let o_ref = host_attention_fp8kv_full(
        &q_host,
        &k_fp8,
        &v_fp8,
        batch as usize,
        num_q_heads as usize,
        num_kv_heads as usize,
        seq_kv as usize,
        kv_stride as usize,
        d as usize,
        scale,
        kv_scale,
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
        "  d=512 fp8kv full GQA decode \
         (B={batch}, H={num_q_heads}, KV={num_kv_heads}, q_pos={q_pos}, \
         kv_stride={kv_stride}, kv_scale={kv_scale:.4}): \
         max_diff={max_diff:.4} mean_diff={mean_diff:.4}",
    );
    assert!(
        max_diff <= 0.15,
        "fp8kv d=512 GQA decode max_diff={max_diff} exceeds 0.15"
    );
}

#[test]
fn test_d512_fp8kv_b1_h4_kv1_pos7() {
    run_case(1, 4, 1, 7, 64);
}

#[test]
fn test_d512_fp8kv_b1_h32_kv2_pos31() {
    // Gemma-4-e4b full-attention dims (n_q=32, n_kv=2).
    run_case(1, 32, 2, 31, 128);
}

#[test]
fn test_d512_fp8kv_b2_h16_kv4_pos15() {
    run_case(2, 16, 4, 15, 64);
}
