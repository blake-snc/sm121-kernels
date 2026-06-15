//! Validate `fa_bf16_fp8kv_decode_d128` against a CPU reference.
//!
//! Pipeline:
//!   1. Generate BF16 Q + FP8 K/V test data + kv_scale.
//!   2. CPU reference: dequantize FP8 → FP32, run online softmax attention.
//!   3. GPU kernel run.
//!   4. Compare BF16 outputs.

use half::bf16;
use sm121_kernels::{attention, device};

fn bf16_f32(x: &[u16]) -> Vec<f32> {
    x.iter().map(|b| bf16::from_bits(*b).to_f32()).collect()
}

fn fp8_e4m3_to_f32(b: u8) -> f32 {
    // Use hardware-equivalent decode: cvt.rn.f16x2.e4m3x2 then cvt.f32.f16.
    // We approximate with the standard formula.
    let sign = (b >> 7) & 1;
    let exp = (b >> 3) & 0xF;
    let mant = b & 0x7;
    let s = if sign == 0 { 1.0f32 } else { -1.0 };
    if exp == 0 {
        s * (mant as f32) / 8.0 * (1.0 / 64.0)
    } else if exp == 0xF && mant == 0x7 {
        f32::NAN
    } else {
        let exp_val = (exp as i32) - 7;
        let pow = if exp_val >= 0 {
            (1u32 << exp_val) as f32
        } else {
            1.0 / (1u32 << -exp_val) as f32
        };
        s * (1.0 + (mant as f32) / 8.0) * pow
    }
}

#[test]
fn t_fp8kv_decode_d128_causal() {
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();

    let batch: u32 = 2;
    let num_heads: u32 = 4;
    let seq_kv: u32 = 64;
    let d: u32 = 128;
    let scale: f32 = 1.0 / (d as f32).sqrt();
    let kv_scale: f32 = 0.125;
    // Per-batch q_pos: batch 0 attends to first 16 positions, batch 1 to first 32.
    let q_pos_host: Vec<u32> = vec![15u32, 31u32];

    let mut state: u64 = 0xCAFE_DECA;
    let mut rnd = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) as u32 as f32 / u32::MAX as f32 - 0.5) * 2.0
    };

    let q_f32: Vec<f32> = (0..(batch * num_heads * d)).map(|_| rnd()).collect();
    let q_bf16: Vec<u16> = q_f32.iter().map(|f| bf16::from_f32(*f).to_bits()).collect();
    let kv_total = (batch * num_heads * seq_kv * d) as usize;
    let k_fp8: Vec<u8> = (0..kv_total)
        .map(|i| {
            let b = ((i.wrapping_mul(31) + 17) % 256) as u8;
            if (b & 0x7F) == 0x7F {
                b ^ 0x10
            } else {
                b
            }
        })
        .collect();
    let v_fp8: Vec<u8> = (0..kv_total)
        .map(|i| {
            let b = ((i.wrapping_mul(53) + 7) % 256) as u8;
            if (b & 0x7F) == 0x7F {
                b ^ 0x10
            } else {
                b
            }
        })
        .collect();

    let mut o_ref = vec![0f32; (batch * num_heads * d) as usize];
    let bs = num_heads as usize * d as usize;
    let hs = d as usize;
    let kv_bs = num_heads as usize * seq_kv as usize * d as usize;
    let kv_hs = seq_kv as usize * d as usize;

    for b in 0..batch as usize {
        let k_limit = (q_pos_host[b] as usize) + 1; // attend [0, q_pos] inclusive
        for h in 0..num_heads as usize {
            let q_off = b * bs + h * hs;
            let kv_off = b * kv_bs + h * kv_hs;
            let mut m_i = f32::NEG_INFINITY;
            let mut l_i = 0f32;
            let mut o_acc = vec![0f32; d as usize];
            for k in 0..k_limit.min(seq_kv as usize) {
                let mut sc = 0f32;
                for dd in 0..d as usize {
                    let qv = q_f32[q_off + dd];
                    let kv = fp8_e4m3_to_f32(k_fp8[kv_off + k * d as usize + dd]) * kv_scale;
                    sc += qv * kv;
                }
                sc *= scale;
                let m_new = m_i.max(sc);
                let alpha = (m_i - m_new).exp2();
                let p = (sc - m_new).exp2();
                l_i = l_i * alpha + p;
                for dd in 0..d as usize {
                    o_acc[dd] *= alpha;
                }
                for dd in 0..d as usize {
                    let vv = fp8_e4m3_to_f32(v_fp8[kv_off + k * d as usize + dd]) * kv_scale;
                    o_acc[dd] += p * vv;
                }
                m_i = m_new;
            }
            for dd in 0..d as usize {
                o_ref[q_off + dd] = o_acc[dd] / l_i;
            }
        }
    }

    let q_dev = stream.memcpy_stod(&q_bf16).unwrap();
    let k_dev = stream.memcpy_stod(&k_fp8).unwrap();
    let v_dev = stream.memcpy_stod(&v_fp8).unwrap();
    let qpos_dev = stream.memcpy_stod(&q_pos_host).unwrap();
    let mut o_dev = stream
        .alloc_zeros::<u16>((batch * num_heads * d) as usize)
        .unwrap();

    attention::fa_bf16_fp8kv_decode_d128_causal(
        &ctx, &stream, &q_dev, &k_dev, &v_dev, &mut o_dev, &qpos_dev, batch, num_heads, seq_kv,
        scale, kv_scale,
    )
    .unwrap();
    let o_got = bf16_f32(&stream.memcpy_dtov(&o_dev).unwrap());

    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    let mut max_mag = 0f32;
    for (g, r) in o_got.iter().zip(o_ref.iter()) {
        if g.is_nan() || r.is_nan() {
            continue;
        }
        let d = (g - r).abs();
        if d > max_abs {
            max_abs = d;
        }
        let m = g.abs().max(r.abs());
        if m > max_mag {
            max_mag = m;
        }
        if m > 0.5 {
            let rel = d / m;
            if rel > max_rel {
                max_rel = rel;
            }
        }
    }
    eprintln!(
        "fp8kv_decode_causal: max_abs={max_abs:.4} max_rel={max_rel:.4} max_mag={max_mag:.2}"
    );
    assert!(
        max_rel < 0.15,
        "fp8kv_decode_causal max_rel too high: {max_rel}"
    );
}

#[test]
fn t_fp8kv_decode_d128_gqa() {
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();

    let batch: u32 = 2;
    let num_q_heads: u32 = 8;
    let num_kv_heads: u32 = 2; // 4 query heads per KV head
    let seq_kv: u32 = 32;
    let d: u32 = 128;
    let scale: f32 = 1.0 / (d as f32).sqrt();
    let kv_scale: f32 = 0.125;
    let q_per_kv = num_q_heads / num_kv_heads;

    let mut state: u64 = 0xCAFE_F00D;
    let mut rnd = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) as u32 as f32 / u32::MAX as f32 - 0.5) * 2.0
    };

    let q_f32: Vec<f32> = (0..(batch * num_q_heads * d)).map(|_| rnd()).collect();
    let q_bf16: Vec<u16> = q_f32.iter().map(|f| bf16::from_f32(*f).to_bits()).collect();
    let kv_total = (batch * num_kv_heads * seq_kv * d) as usize;
    let k_fp8: Vec<u8> = (0..kv_total)
        .map(|i| {
            let b = ((i.wrapping_mul(31) + 17) % 256) as u8;
            if (b & 0x7F) == 0x7F {
                b ^ 0x10
            } else {
                b
            }
        })
        .collect();
    let v_fp8: Vec<u8> = (0..kv_total)
        .map(|i| {
            let b = ((i.wrapping_mul(53) + 7) % 256) as u8;
            if (b & 0x7F) == 0x7F {
                b ^ 0x10
            } else {
                b
            }
        })
        .collect();

    // CPU reference
    let mut o_ref = vec![0f32; (batch * num_q_heads * d) as usize];
    let q_bs = num_q_heads as usize * d as usize;
    let q_hs = d as usize;
    let kv_bs = num_kv_heads as usize * seq_kv as usize * d as usize;
    let kv_hs = seq_kv as usize * d as usize;

    for b in 0..batch as usize {
        for qh in 0..num_q_heads as usize {
            let kvh = qh / q_per_kv as usize;
            let q_off = b * q_bs + qh * q_hs;
            let kv_off = b * kv_bs + kvh * kv_hs;
            let mut m_i = f32::NEG_INFINITY;
            let mut l_i = 0f32;
            let mut o_acc = vec![0f32; d as usize];
            for k in 0..seq_kv as usize {
                let mut sc = 0f32;
                for dd in 0..d as usize {
                    let qv = q_f32[q_off + dd];
                    let kv = fp8_e4m3_to_f32(k_fp8[kv_off + k * d as usize + dd]) * kv_scale;
                    sc += qv * kv;
                }
                sc *= scale;
                let m_new = m_i.max(sc);
                let alpha = (m_i - m_new).exp2();
                let p = (sc - m_new).exp2();
                l_i = l_i * alpha + p;
                for dd in 0..d as usize {
                    o_acc[dd] *= alpha;
                }
                for dd in 0..d as usize {
                    let vv = fp8_e4m3_to_f32(v_fp8[kv_off + k * d as usize + dd]) * kv_scale;
                    o_acc[dd] += p * vv;
                }
                m_i = m_new;
            }
            for dd in 0..d as usize {
                o_ref[q_off + dd] = o_acc[dd] / l_i;
            }
        }
    }

    let q_dev = stream.memcpy_stod(&q_bf16).unwrap();
    let k_dev = stream.memcpy_stod(&k_fp8).unwrap();
    let v_dev = stream.memcpy_stod(&v_fp8).unwrap();
    let mut o_dev = stream
        .alloc_zeros::<u16>((batch * num_q_heads * d) as usize)
        .unwrap();

    attention::fa_bf16_fp8kv_decode_d128_gqa(
        &ctx,
        &stream,
        &q_dev,
        &k_dev,
        &v_dev,
        &mut o_dev,
        batch,
        num_q_heads,
        num_kv_heads,
        seq_kv,
        /*kv_stride=*/ seq_kv,
        scale,
        kv_scale,
    )
    .unwrap();
    let o_got = bf16_f32(&stream.memcpy_dtov(&o_dev).unwrap());

    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    let mut max_mag = 0f32;
    for (g, r) in o_got.iter().zip(o_ref.iter()) {
        if g.is_nan() || r.is_nan() {
            continue;
        }
        let d = (g - r).abs();
        if d > max_abs {
            max_abs = d;
        }
        let m = g.abs().max(r.abs());
        if m > max_mag {
            max_mag = m;
        }
        if m > 0.5 {
            let rel = d / m;
            if rel > max_rel {
                max_rel = rel;
            }
        }
    }
    eprintln!("fp8kv_decode_gqa: max_abs={max_abs:.4} max_rel={max_rel:.4} max_mag={max_mag:.2}");
    assert!(
        max_rel < 0.15,
        "fp8kv_decode_gqa max_rel too high: {max_rel}"
    );
}

#[test]
fn t_fp8kv_decode_d128_basic() {
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();

    let batch: u32 = 2;
    let num_heads: u32 = 4;
    let seq_kv: u32 = 64;
    let d: u32 = 128;
    let scale: f32 = 1.0 / (d as f32).sqrt();
    let kv_scale: f32 = 0.125;

    let mut state: u64 = 0xCAFE_DECA;
    let mut rnd = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) as u32 as f32 / u32::MAX as f32 - 0.5) * 2.0
    };

    // Q: BF16, modest range
    let q_f32: Vec<f32> = (0..(batch * num_heads * d)).map(|_| rnd()).collect();
    let q_bf16: Vec<u16> = q_f32.iter().map(|f| bf16::from_f32(*f).to_bits()).collect();

    // K, V: FP8, skip NaN bytes
    let kv_total = (batch * num_heads * seq_kv * d) as usize;
    let k_fp8: Vec<u8> = (0..kv_total)
        .map(|i| {
            let b = ((i.wrapping_mul(31) + 17) % 256) as u8;
            if (b & 0x7F) == 0x7F {
                b ^ 0x10
            } else {
                b
            }
        })
        .collect();
    let v_fp8: Vec<u8> = (0..kv_total)
        .map(|i| {
            let b = ((i.wrapping_mul(53) + 7) % 256) as u8;
            if (b & 0x7F) == 0x7F {
                b ^ 0x10
            } else {
                b
            }
        })
        .collect();

    // CPU reference
    let mut o_ref = vec![0f32; (batch * num_heads * d) as usize];
    let bs = num_heads as usize * d as usize;
    let hs = d as usize;
    let kv_bs = num_heads as usize * seq_kv as usize * d as usize;
    let kv_hs = seq_kv as usize * d as usize;

    for b in 0..batch as usize {
        for h in 0..num_heads as usize {
            // Q[b, h, :]
            let q_off = b * bs + h * hs;
            // KV base
            let kv_off = b * kv_bs + h * kv_hs;

            let mut m_i = f32::NEG_INFINITY;
            let mut l_i = 0f32;
            let mut o_acc = vec![0f32; d as usize];

            for k in 0..seq_kv as usize {
                // QK score
                let mut sc = 0f32;
                for dd in 0..d as usize {
                    let qv = q_f32[q_off + dd];
                    let kv = fp8_e4m3_to_f32(k_fp8[kv_off + k * d as usize + dd]) * kv_scale;
                    sc += qv * kv;
                }
                sc *= scale;

                let m_new = m_i.max(sc);
                let alpha = (m_i - m_new).exp2(); // CPU uses exp2 to match kernel's ex2.approx
                let p = (sc - m_new).exp2();
                l_i = l_i * alpha + p;
                for dd in 0..d as usize {
                    o_acc[dd] *= alpha;
                }
                for dd in 0..d as usize {
                    let vv = fp8_e4m3_to_f32(v_fp8[kv_off + k * d as usize + dd]) * kv_scale;
                    o_acc[dd] += p * vv;
                }
                m_i = m_new;
            }

            for dd in 0..d as usize {
                o_ref[q_off + dd] = o_acc[dd] / l_i;
            }
        }
    }

    let q_dev = stream.memcpy_stod(&q_bf16).unwrap();
    let k_dev = stream.memcpy_stod(&k_fp8).unwrap();
    let v_dev = stream.memcpy_stod(&v_fp8).unwrap();
    let mut o_dev = stream
        .alloc_zeros::<u16>((batch * num_heads * d) as usize)
        .unwrap();

    attention::fa_bf16_fp8kv_decode_d128(
        &ctx, &stream, &q_dev, &k_dev, &v_dev, &mut o_dev, batch, num_heads, seq_kv, scale,
        kv_scale,
    )
    .unwrap();

    let o_got = bf16_f32(&stream.memcpy_dtov(&o_dev).unwrap());

    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    let mut max_mag = 0f32;
    let mut n_nan = 0;
    for (g, r) in o_got.iter().zip(o_ref.iter()) {
        if g.is_nan() || r.is_nan() {
            n_nan += 1;
            continue;
        }
        let d = (g - r).abs();
        if d > max_abs {
            max_abs = d;
        }
        let m = g.abs().max(r.abs());
        if m > max_mag {
            max_mag = m;
        }
        if m > 0.5 {
            let rel = d / m;
            if rel > max_rel {
                max_rel = rel;
            }
        }
    }
    eprintln!(
        "fp8kv_decode: max_abs={max_abs:.4} max_rel={max_rel:.4} max_mag={max_mag:.2} nan={n_nan}"
    );
    assert_eq!(n_nan, 0, "unexpected NaN in output");
    // BF16 attention with FP8 KV dequant + ex2 (vs exp) approximations:
    // expect ~5-10% relative error on outputs at moderate magnitudes.
    assert!(max_rel < 0.15, "fp8kv_decode max_rel too high: {max_rel}");
}
