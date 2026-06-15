//! Validate fa_bf16_decode_d256_gqa against PyTorch reference.
//!
//! Tests both modes:
//! - Full attention (sliding_window = 0)
//! - Sliding-window attention (sliding_window > 0)

use half::bf16;
use sm121_kernels::{attention, device};

fn bf16_to_u16(v: f32) -> u16 {
    bf16::from_f32(v).to_bits()
}

fn u16_to_f32(b: u16) -> f32 {
    bf16::from_bits(b).to_f32()
}

/// Single-query attention reference: O[h, d] = Σ_k softmax(scale * Q[h] · K[kv_head, k]) · V[kv_head, k, d]
fn reference_decode(
    q: &[f32], // [num_heads, D]
    k: &[f32], // [num_kv_heads, seq_kv, D]
    v: &[f32], // [num_kv_heads, seq_kv, D]
    num_heads: usize,
    num_kv_heads: usize,
    seq_kv: usize,
    d: usize,
    scale: f32,
    kv_min: usize, // SWA lower bound (attends k ∈ [kv_min, seq_kv))
) -> Vec<f32> {
    let q_per_kv = num_heads / num_kv_heads;
    let mut out = vec![0.0f32; num_heads * d];
    for h in 0..num_heads {
        let kv_h = h / q_per_kv;
        let mut scores = vec![f32::NEG_INFINITY; seq_kv];
        for kk in kv_min..seq_kv {
            let mut s = 0.0f32;
            for dd in 0..d {
                s += q[h * d + dd] * k[(kv_h * seq_kv + kk) * d + dd];
            }
            scores[kk] = s * scale;
        }
        let m = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mut sum = 0.0f32;
        let mut probs = vec![0.0f32; seq_kv];
        for kk in kv_min..seq_kv {
            let p = (scores[kk] - m).exp();
            probs[kk] = p;
            sum += p;
        }
        for dd in 0..d {
            let mut o_d = 0.0f32;
            for kk in kv_min..seq_kv {
                o_d += probs[kk] / sum * v[(kv_h * seq_kv + kk) * d + dd];
            }
            out[h * d + dd] = o_d;
        }
    }
    out
}

fn run_test(seq_kv: u32, sliding_window: u32, q_pos: u32) {
    let ctx = match device::init_device(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[skip] init_device: {e:?}");
            return;
        }
    };
    let stream = ctx.default_stream();

    let batch: u32 = 1;
    let num_heads: u32 = 8;
    let num_kv_heads: u32 = 2;
    let d: u32 = 256;
    let kv_stride = seq_kv.max(64);

    // Compute SWA lower bound to mirror kernel
    let kv_min = if sliding_window == 0 {
        0
    } else {
        (q_pos + 1).saturating_sub(sliding_window)
    };

    // Generate random Q, K, V — small magnitudes to stay in BF16 range.
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let seed = (seq_kv, sliding_window, q_pos);
    let mut hasher = DefaultHasher::new();
    seed.hash(&mut hasher);
    let mut rng_state = hasher.finish();
    let mut next_f = || -> f32 {
        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = ((rng_state >> 32) as u32) & 0x007fffff;
        let v = f32::from_bits(0x3f800000 | bits) - 1.5; // [-0.5, 0.5)
        v * 2.0 // [-1.0, 1.0)
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

    // Round through BF16 for parity (kernel reads BF16 inputs).
    let to_bf16_round =
        |v: &[f32]| -> Vec<f32> { v.iter().map(|&x| bf16::from_f32(x).to_f32()).collect() };
    let q_round = to_bf16_round(&q_f32);
    let k_round = to_bf16_round(&k_f32);
    let v_round = to_bf16_round(&v_f32);

    // Build BF16 device buffers. KV layout: [num_kv_heads, kv_stride, d].
    let q_bf16: Vec<u16> = q_round.iter().map(|&x| bf16_to_u16(x)).collect();
    let k_bf16: Vec<u16> = k_round.iter().map(|&x| bf16_to_u16(x)).collect();
    let v_bf16: Vec<u16> = v_round.iter().map(|&x| bf16_to_u16(x)).collect();

    let q_dev = stream.memcpy_stod(&q_bf16).expect("htod q");
    let k_dev = stream.memcpy_stod(&k_bf16).expect("htod k");
    let v_dev = stream.memcpy_stod(&v_bf16).expect("htod v");
    let mut o_dev = stream.alloc_zeros::<u16>(q_size).expect("alloc o");

    let scale = 1.0f32 / (d as f32).sqrt();
    attention::flash_attn_bf16_decode_d256_gqa(
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
        q_pos,
        sliding_window,
        scale,
    )
    .expect("kernel launch");
    stream.synchronize().ok();

    let o_host_bf16 = stream.memcpy_dtov(&o_dev).expect("dtoh o");
    let o_host: Vec<f32> = o_host_bf16.iter().map(|&b| u16_to_f32(b)).collect();

    // Compute reference using the BF16-rounded inputs for fair comparison.
    // Reshape K/V from [num_kv_heads, kv_stride, d] to "logical" [num_kv_heads, seq_kv, d].
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
        kv_min as usize,
    );

    // Compare with relaxed BF16 tolerance.
    let mut max_diff = 0.0f32;
    let mut mean_diff = 0.0f32;
    for (g, r) in o_host.iter().zip(o_ref.iter()) {
        let dif = (g - r).abs();
        max_diff = max_diff.max(dif);
        mean_diff += dif;
    }
    mean_diff /= o_host.len() as f32;
    eprintln!(
        "  d=256 GQA decode (seq_kv={seq_kv}, swa={sliding_window}, q_pos={q_pos}, kv_min={kv_min}): \
         max_diff={max_diff:.4} mean_diff={mean_diff:.4}",
    );
    // BF16 cumulative error is loose; tolerance scales with sqrt(seq_kv).
    let tol = 0.05 * (seq_kv as f32).sqrt() / 16.0;
    let tol = tol.max(0.05);
    assert!(
        max_diff < tol,
        "d=256 GQA decode max_diff {max_diff:.4} > tol {tol:.4}"
    );
}

#[test]
fn test_d256_gqa_decode_full_64() {
    run_test(64, 0, 63);
}

#[test]
fn test_d256_gqa_decode_full_256() {
    run_test(256, 0, 255);
}

#[test]
fn test_d256_gqa_decode_swa_window32() {
    // 64 KV positions, window=32, q_pos=63 → kv_min=32, attends k ∈ [32, 64)
    run_test(64, 32, 63);
}

#[test]
fn test_d256_gqa_decode_swa_window512_short() {
    // window > seq_kv → effectively full attention
    run_test(128, 512, 127);
}

#[test]
fn test_d256_gqa_decode_swa_window128_long() {
    // 256 KV, window=128, q_pos=255 → kv_min=128
    run_test(256, 128, 255);
}
