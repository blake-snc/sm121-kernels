//! Validate fa_bf16_decode_d512_gqa against PyTorch reference.

use half::bf16;
use sm121_kernels::{attention, device};

fn bf16_to_u16(v: f32) -> u16 {
    bf16::from_f32(v).to_bits()
}

fn u16_to_f32(b: u16) -> f32 {
    bf16::from_bits(b).to_f32()
}

fn reference_decode(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    num_heads: usize,
    num_kv_heads: usize,
    seq_kv: usize,
    d: usize,
    scale: f32,
) -> Vec<f32> {
    let q_per_kv = num_heads / num_kv_heads;
    let mut out = vec![0.0f32; num_heads * d];
    for h in 0..num_heads {
        let kv_h = h / q_per_kv;
        let mut scores = vec![0.0f32; seq_kv];
        for kk in 0..seq_kv {
            let mut s = 0.0f32;
            for dd in 0..d {
                s += q[h * d + dd] * k[(kv_h * seq_kv + kk) * d + dd];
            }
            scores[kk] = s * scale;
        }
        let m = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mut sum = 0.0f32;
        for kk in 0..seq_kv {
            scores[kk] = (scores[kk] - m).exp();
            sum += scores[kk];
        }
        for dd in 0..d {
            let mut o_d = 0.0f32;
            for kk in 0..seq_kv {
                o_d += scores[kk] / sum * v[(kv_h * seq_kv + kk) * d + dd];
            }
            out[h * d + dd] = o_d;
        }
    }
    out
}

fn run_test(seq_kv: u32) {
    let ctx = match device::init_device(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[skip] init_device: {e:?}");
            return;
        }
    };
    let stream = ctx.default_stream();

    let batch: u32 = 1;
    let num_heads: u32 = 8; // Gemma-4 full attention: 8 q heads
    let num_kv_heads: u32 = 2; //                        2 kv heads (4:1 GQA)
    let d: u32 = 512;
    let kv_stride = seq_kv.max(64);

    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    seq_kv.hash(&mut hasher);
    let mut rng_state = hasher.finish();
    let mut next_f = || -> f32 {
        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = ((rng_state >> 32) as u32) & 0x007fffff;
        let v = f32::from_bits(0x3f800000 | bits) - 1.5;
        v * 2.0
    };

    let q_size = (batch * num_heads * d) as usize;
    let kv_size = (batch * num_kv_heads * kv_stride * d) as usize;
    let mut q_f32 = vec![0.0f32; q_size];
    let mut k_f32 = vec![0.0f32; kv_size];
    let mut v_f32 = vec![0.0f32; kv_size];
    for x in &mut q_f32 {
        *x = next_f();
    }
    for x in &mut k_f32 {
        *x = next_f();
    }
    for x in &mut v_f32 {
        *x = next_f();
    }

    let to_bf16_round =
        |v: &[f32]| -> Vec<f32> { v.iter().map(|&x| bf16::from_f32(x).to_f32()).collect() };
    let q_round = to_bf16_round(&q_f32);
    let k_round = to_bf16_round(&k_f32);
    let v_round = to_bf16_round(&v_f32);

    let q_bf16: Vec<u16> = q_round.iter().map(|&x| bf16_to_u16(x)).collect();
    let k_bf16: Vec<u16> = k_round.iter().map(|&x| bf16_to_u16(x)).collect();
    let v_bf16: Vec<u16> = v_round.iter().map(|&x| bf16_to_u16(x)).collect();

    let q_dev = stream.memcpy_stod(&q_bf16).expect("htod q");
    let k_dev = stream.memcpy_stod(&k_bf16).expect("htod k");
    let v_dev = stream.memcpy_stod(&v_bf16).expect("htod v");
    let mut o_dev = stream.alloc_zeros::<u16>(q_size).expect("alloc o");

    let scale = 1.0f32 / (d as f32).sqrt();
    attention::flash_attn_bf16_decode_d512_gqa(
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

    let o_host_bf16 = stream.memcpy_dtov(&o_dev).expect("dtoh o");
    let o_host: Vec<f32> = o_host_bf16.iter().map(|&b| u16_to_f32(b)).collect();

    // Build compact reference KV.
    let mut k_ref = vec![0.0f32; (num_kv_heads * seq_kv * d) as usize];
    let mut v_ref = vec![0.0f32; (num_kv_heads * seq_kv * d) as usize];
    for h in 0..num_kv_heads as usize {
        for kk in 0..seq_kv as usize {
            for dd in 0..d as usize {
                let src_idx = h * (kv_stride * d) as usize + kk * d as usize + dd;
                let dst_idx = h * (seq_kv * d) as usize + kk * d as usize + dd;
                k_ref[dst_idx] = k_round[src_idx];
                v_ref[dst_idx] = v_round[src_idx];
            }
        }
    }

    let o_ref = reference_decode(
        &q_round,
        &k_ref,
        &v_ref,
        num_heads as usize,
        num_kv_heads as usize,
        seq_kv as usize,
        d as usize,
        scale,
    );

    let mut max_diff = 0.0f32;
    let mut mean_diff = 0.0f32;
    for (g, r) in o_host.iter().zip(o_ref.iter()) {
        let dif = (g - r).abs();
        max_diff = max_diff.max(dif);
        mean_diff += dif;
    }
    mean_diff /= o_host.len() as f32;
    eprintln!(
        "  d=512 GQA decode (seq_kv={seq_kv}): max_diff={max_diff:.4} mean_diff={mean_diff:.4}",
    );
    let tol = 0.05 * (seq_kv as f32).sqrt() / 16.0;
    let tol = tol.max(0.05);
    assert!(
        max_diff < tol,
        "d=512 GQA decode max_diff {max_diff:.4} > tol {tol:.4}"
    );
}

#[test]
fn test_d512_gqa_decode_64() {
    run_test(64);
}

#[test]
fn test_d512_gqa_decode_256() {
    run_test(256);
}
