//! End-to-end DSA correctness test (indexer + sparse attention integration).
//!
//! Exercises the full DSA pipeline: indexer + top-k selection + sparse attention.
//! Compares against a scalar reference that runs the same algorithm CPU-side
//! using the same random inputs.
//!
//! Run:
//!     cargo test --release --test test_dsa_end_to_end -- --test-threads=1
//!
//! DSA is an experimental sparse-attention reference, gated behind the
//! `experimental` feature. Build/run with `--features experimental`.
#![cfg(feature = "experimental")]

mod common;

use sm121_kernels::attention::dsa::dsa_attention_bf16;
use sm121_kernels::device;

const H_IDX: usize = 32;
const D_IDX: usize = 128;
const D_ATTN: usize = 128;

fn f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    let rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16;
    rounded as u16
}
fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// Scalar reference: full DSA forward.
///   1. Compute indexer scores.
///   2. Top-k select per (b, s).
///   3. Sparse attention over the top-k positions.
#[allow(clippy::too_many_arguments)]
fn reference_dsa_attention(
    q_idx: &[u16],
    k_idx: &[u16],
    weights: &[f32],
    q_attn: &[u16],
    k_attn: &[u16],
    v_attn: &[u16],
    batch: usize,
    seq_q: usize,
    seq_kv_full: usize,
    num_heads: usize,
    num_kv_heads: usize,
    topk: usize,
    scale_attn: f32,
) -> Vec<u16> {
    let gqa = num_heads / num_kv_heads;

    // Stage 1: indexer scores [B, S, T_full]
    let mut idx_scores = vec![0f32; batch * seq_q * seq_kv_full];
    for b in 0..batch {
        for s in 0..seq_q {
            for t in 0..seq_kv_full {
                let mut score = 0.0f32;
                for h in 0..H_IDX {
                    let mut dot = 0.0f32;
                    for i in 0..D_IDX {
                        let qi = (((b * seq_q) + s) * H_IDX + h) * D_IDX + i;
                        let ki = ((b * seq_kv_full) + t) * D_IDX + i;
                        dot += bf16_to_f32(q_idx[qi]) * bf16_to_f32(k_idx[ki]);
                    }
                    let dot_relu = dot.max(0.0);
                    let wi = (b * seq_q + s) * H_IDX + h;
                    score += weights[wi] * dot_relu;
                }
                idx_scores[(b * seq_q + s) * seq_kv_full + t] = score;
            }
        }
    }

    // Stage 2: top-k selection
    let mut top_indices = vec![0i32; batch * seq_q * topk];
    for b in 0..batch {
        for s in 0..seq_q {
            let base = (b * seq_q + s) * seq_kv_full;
            let row = &idx_scores[base..base + seq_kv_full];
            let mut order: Vec<usize> = (0..seq_kv_full).collect();
            order.sort_unstable_by(|&i, &j| {
                row[j]
                    .partial_cmp(&row[i])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            for (i, pos) in order.iter().take(topk).enumerate() {
                top_indices[(b * seq_q + s) * topk + i] = *pos as i32;
            }
        }
    }

    // Stage 3: sparse attention. Per-(b, s, h): compute scores over selected
    // positions, softmax, weighted V sum.
    let mut out = vec![0u16; batch * seq_q * num_heads * D_ATTN];
    for b in 0..batch {
        for s in 0..seq_q {
            // Build the mask once per (b, s).
            let mut mask = vec![f32::NEG_INFINITY; seq_kv_full];
            for ti in 0..topk {
                let pos = top_indices[(b * seq_q + s) * topk + ti] as usize;
                mask[pos] = 0.0;
            }
            for h in 0..num_heads {
                let h_kv = h / gqa;
                let mut scores = vec![0f32; seq_kv_full];
                for t in 0..seq_kv_full {
                    let mut dot = 0.0f32;
                    for di in 0..D_ATTN {
                        let qi = ((b * seq_q + s) * num_heads + h) * D_ATTN + di;
                        let ki = ((b * seq_kv_full + t) * num_kv_heads + h_kv) * D_ATTN + di;
                        dot += bf16_to_f32(q_attn[qi]) * bf16_to_f32(k_attn[ki]);
                    }
                    scores[t] = dot * scale_attn + mask[t];
                }
                let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for sc in scores.iter_mut() {
                    *sc = (*sc - m).exp();
                    sum += *sc;
                }
                for sc in scores.iter_mut() {
                    *sc /= sum;
                }
                for di in 0..D_ATTN {
                    let mut acc = 0.0f32;
                    for t in 0..seq_kv_full {
                        let vi = ((b * seq_kv_full + t) * num_kv_heads + h_kv) * D_ATTN + di;
                        acc += scores[t] * bf16_to_f32(v_attn[vi]);
                    }
                    let oi = ((b * seq_q + s) * num_heads + h) * D_ATTN + di;
                    out[oi] = f32_to_bf16(acc);
                }
            }
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn run_dsa_e2e_test(
    batch: usize,
    seq_q: usize,
    seq_kv_full: usize,
    num_heads: usize,
    num_kv_heads: usize,
    topk: usize,
    tol: f32,
) {
    let ctx = device::init_device(0).expect("init_device");
    let stream = ctx.new_stream().expect("new_stream");

    // Random inputs
    let total_q_idx = batch * seq_q * H_IDX * D_IDX;
    let total_k_idx = batch * seq_kv_full * D_IDX;
    let total_w = batch * seq_q * H_IDX;
    let total_q_attn = batch * seq_q * num_heads * D_ATTN;
    let total_kv_attn = batch * seq_kv_full * num_kv_heads * D_ATTN;
    let total_idx_scores = batch * seq_q * seq_kv_full;
    let total_topk = batch * seq_q * topk;
    let total_o = batch * seq_q * num_heads * D_ATTN;

    let mut seed: u32 = 0x55555555;
    let mut next_u01 = || {
        seed = seed.wrapping_mul(48271).wrapping_add(1);
        ((seed >> 8) as f32) / (1u32 << 24) as f32
    };

    let mut q_idx_h = vec![0u16; total_q_idx];
    let mut k_idx_h = vec![0u16; total_k_idx];
    let mut w_h = vec![0f32; total_w];
    let mut q_attn_h = vec![0u16; total_q_attn];
    let mut k_attn_h = vec![0u16; total_kv_attn];
    let mut v_attn_h = vec![0u16; total_kv_attn];
    for v in &mut q_idx_h {
        *v = f32_to_bf16((next_u01() - 0.5) * 0.4);
    }
    for v in &mut k_idx_h {
        *v = f32_to_bf16((next_u01() - 0.5) * 0.4);
    }
    for v in &mut w_h {
        *v = (next_u01() - 0.5) * 0.2;
    }
    for v in &mut q_attn_h {
        *v = f32_to_bf16((next_u01() - 0.5) * 0.4);
    }
    for v in &mut k_attn_h {
        *v = f32_to_bf16((next_u01() - 0.5) * 0.4);
    }
    for v in &mut v_attn_h {
        *v = f32_to_bf16((next_u01() - 0.5) * 0.4);
    }

    let scale = 1.0 / (D_ATTN as f32).sqrt();

    // Reference
    let ref_out = reference_dsa_attention(
        &q_idx_h,
        &k_idx_h,
        &w_h,
        &q_attn_h,
        &k_attn_h,
        &v_attn_h,
        batch,
        seq_q,
        seq_kv_full,
        num_heads,
        num_kv_heads,
        topk,
        scale,
    );

    // GPU
    let q_idx_d = stream.memcpy_stod(&q_idx_h).expect("htod q_idx");
    let k_idx_d = stream.memcpy_stod(&k_idx_h).expect("htod k_idx");
    let w_d = stream.memcpy_stod(&w_h).expect("htod w");
    let q_attn_d = stream.memcpy_stod(&q_attn_h).expect("htod q_attn");
    let k_attn_d = stream.memcpy_stod(&k_attn_h).expect("htod k_attn");
    let v_attn_d = stream.memcpy_stod(&v_attn_h).expect("htod v_attn");
    let mut idx_scores_d = stream
        .alloc_zeros::<f32>(total_idx_scores)
        .expect("alloc idx_scores");
    let mut topk_d = stream.alloc_zeros::<i32>(total_topk).expect("alloc topk");
    let mut out_d = stream.alloc_zeros::<u16>(total_o).expect("alloc out");

    dsa_attention_bf16(
        &ctx,
        &stream,
        &q_idx_d,
        &k_idx_d,
        &w_d,
        &q_attn_d,
        &k_attn_d,
        &v_attn_d,
        &mut idx_scores_d,
        &mut topk_d,
        &mut out_d,
        batch as u32,
        seq_q as u32,
        seq_kv_full as u32,
        num_heads as u32,
        num_kv_heads as u32,
        topk as u32,
        scale,
    )
    .expect("dsa_attention_bf16");

    stream.synchronize().expect("sync");
    let out_host = stream.memcpy_dtov(&out_d).expect("dtoh out");

    // Compare
    let mut max_abs: f32 = 0.0;
    let mut sum_abs: f64 = 0.0;
    let mut worst_idx = 0usize;
    for (i, (&actual, &expected)) in out_host.iter().zip(ref_out.iter()).enumerate() {
        let a = bf16_to_f32(actual);
        let e = bf16_to_f32(expected);
        let d = (a - e).abs();
        sum_abs += d as f64;
        if d > max_abs {
            max_abs = d;
            worst_idx = i;
        }
    }
    let mean_abs = sum_abs as f32 / out_host.len() as f32;
    println!(
        "DSA E2E B={} S={} T={} H={}/{} topk={}: max_abs={:.4} mean_abs={:.5} worst_idx={}",
        batch, seq_q, seq_kv_full, num_heads, num_kv_heads, topk, max_abs, mean_abs, worst_idx,
    );
    assert!(max_abs <= tol, "max_abs {:.4} > tol {:.4}", max_abs, tol,);
}

#[test]
fn dsa_e2e_minimal() {
    // 1 batch, 1 query, T=128, MHA, topk=32.
    run_dsa_e2e_test(1, 1, 128, 4, 4, 32, 0.05);
}

#[test]
fn dsa_e2e_gqa() {
    // GQA 16/8.
    run_dsa_e2e_test(1, 1, 256, 16, 8, 64, 0.05);
}

#[test]
fn dsa_e2e_long_kv() {
    // Long context, modest topk.
    run_dsa_e2e_test(1, 1, 1024, 8, 4, 128, 0.05);
}
