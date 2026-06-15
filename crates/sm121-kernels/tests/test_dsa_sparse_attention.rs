//! Correctness test for the DSA sparse attention wrapper.
//!
//! Compares against a scalar reference that computes:
//!   1. Standard attention scores with -inf mask on non-selected positions.
//!   2. Softmax + weighted V sum.
//!
//! Run:
//!     cargo test --release --test test_dsa_sparse_attention -- --test-threads=1
//!
//! DSA is an experimental sparse-attention reference, gated behind the
//! `experimental` feature. Build/run with `--features experimental`.
#![cfg(feature = "experimental")]

mod common;

use sm121_kernels::attention::dsa::dsa_sparse_attention_bf16;
use sm121_kernels::device;

const D: usize = 128;

fn f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    let rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16;
    rounded as u16
}
fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// Scalar reference: standard attention with -inf mask on non-selected positions.
///
/// Inputs:
///   q [B, S, H, D] bf16
///   k [B, T_full, H_kv, D] bf16
///   v [B, T_full, H_kv, D] bf16
///   top_k_indices [B, S, topk] i32
fn reference_sparse_attention(
    q: &[u16],
    k: &[u16],
    v: &[u16],
    top_k_indices: &[i32],
    batch: usize,
    seq_q: usize,
    seq_kv_full: usize,
    num_heads: usize,
    num_kv_heads: usize,
    topk: usize,
    scale: f32,
) -> Vec<u16> {
    let gqa = num_heads / num_kv_heads;
    let mut out = vec![0u16; batch * seq_q * num_heads * D];
    for b in 0..batch {
        for s in 0..seq_q {
            // Build mask: T entries, -inf except for top-k positions which get 0.
            let mut mask = vec![f32::NEG_INFINITY; seq_kv_full];
            for ti in 0..topk {
                let pos = top_k_indices[(b * seq_q + s) * topk + ti] as usize;
                mask[pos] = 0.0;
            }
            for h in 0..num_heads {
                let h_kv = h / gqa;
                // Compute scores [T_full] = (q[h] · k[h_kv, t]) * scale + mask[t]
                let mut scores = vec![0f32; seq_kv_full];
                for t in 0..seq_kv_full {
                    let mut dot = 0.0f32;
                    for di in 0..D {
                        let q_idx = ((b * seq_q + s) * num_heads + h) * D + di;
                        let k_idx = ((b * seq_kv_full + t) * num_kv_heads + h_kv) * D + di;
                        dot += bf16_to_f32(q[q_idx]) * bf16_to_f32(k[k_idx]);
                    }
                    scores[t] = dot * scale + mask[t];
                }
                // Stable softmax (subtract max).
                let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for sc in scores.iter_mut() {
                    *sc = (*sc - m).exp();
                    sum += *sc;
                }
                for sc in scores.iter_mut() {
                    *sc /= sum;
                }
                // out[h, di] = Σ_t scores[t] * v[h_kv, t, di]
                for di in 0..D {
                    let mut acc = 0.0f32;
                    for t in 0..seq_kv_full {
                        let v_idx = ((b * seq_kv_full + t) * num_kv_heads + h_kv) * D + di;
                        acc += scores[t] * bf16_to_f32(v[v_idx]);
                    }
                    let dst = ((b * seq_q + s) * num_heads + h) * D + di;
                    out[dst] = f32_to_bf16(acc);
                }
            }
        }
    }
    out
}

fn run_sparse_attention_test(
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

    let total_q = batch * seq_q * num_heads * D;
    let total_kv = batch * seq_kv_full * num_kv_heads * D;
    let total_idx = batch * seq_q * topk;
    let total_o = batch * seq_q * num_heads * D;

    // Random inputs (deterministic LCG seed).
    let mut q_host = vec![0u16; total_q];
    let mut k_host = vec![0u16; total_kv];
    let mut v_host = vec![0u16; total_kv];
    let mut idx_host = vec![0i32; total_idx];
    let mut seed: u32 = 0xDEADBEEF;
    let mut next_u01 = || {
        seed = seed.wrapping_mul(48271).wrapping_add(1);
        ((seed >> 8) as f32) / (1u32 << 24) as f32
    };
    let mut next_norm = || (next_u01() - 0.5) * 0.4;
    for v in &mut q_host {
        *v = f32_to_bf16(next_norm());
    }
    for v in &mut k_host {
        *v = f32_to_bf16(next_norm());
    }
    for vv in &mut v_host {
        *vv = f32_to_bf16(next_norm());
    }
    // Top-k indices: pick `topk` random distinct positions in [0, seq_kv_full).
    // For the test, just take a deterministic stride that wraps.
    for b in 0..batch {
        for s in 0..seq_q {
            for i in 0..topk {
                let raw = (b * 7 + s * 11 + i * 13) % seq_kv_full;
                idx_host[(b * seq_q + s) * topk + i] = raw as i32;
            }
        }
    }

    let scale = 1.0 / (D as f32).sqrt();

    // Reference
    let ref_out = reference_sparse_attention(
        &q_host,
        &k_host,
        &v_host,
        &idx_host,
        batch,
        seq_q,
        seq_kv_full,
        num_heads,
        num_kv_heads,
        topk,
        scale,
    );

    // GPU
    let q_dev = stream.memcpy_stod(&q_host).expect("htod q");
    let k_dev = stream.memcpy_stod(&k_host).expect("htod k");
    let v_dev = stream.memcpy_stod(&v_host).expect("htod v");
    let idx_dev = stream.memcpy_stod(&idx_host).expect("htod idx");
    let mut out_dev = stream.alloc_zeros::<u16>(total_o).expect("alloc out");

    dsa_sparse_attention_bf16(
        &ctx,
        &stream,
        &q_dev,
        &k_dev,
        &v_dev,
        &idx_dev,
        &mut out_dev,
        batch as u32,
        seq_q as u32,
        seq_kv_full as u32,
        num_heads as u32,
        num_kv_heads as u32,
        topk as u32,
        scale,
    )
    .expect("dispatch");

    stream.synchronize().expect("sync");
    let out_host = stream.memcpy_dtov(&out_dev).expect("dtoh out");

    // Compare in FP32 space.
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
        "B={} S={} T={} H={}/{} topk={}: max_abs={:.4} mean_abs={:.5} worst_idx={}",
        batch, seq_q, seq_kv_full, num_heads, num_kv_heads, topk, max_abs, mean_abs, worst_idx,
    );
    assert!(max_abs <= tol, "max_abs {:.4} > tol {:.4}", max_abs, tol,);
}

#[test]
fn dsa_sparse_attn_small() {
    // 1 batch, 1 query, T_full=128, topk=32. MHA (no GQA), head_dim=128.
    run_sparse_attention_test(1, 1, 128, 4, 4, 32, 0.05);
}

#[test]
fn dsa_sparse_attn_gqa() {
    // GQA: 16 q heads, 8 kv heads.
    run_sparse_attention_test(1, 1, 256, 16, 8, 64, 0.05);
}

#[test]
fn dsa_sparse_attn_multibatch() {
    // Multi-batch + multi-query prefill.
    run_sparse_attention_test(2, 4, 256, 8, 4, 64, 0.05);
}

#[test]
fn dsa_sparse_attn_long_kv_small_topk() {
    // Long-context style: T_full=2048, topk=128 (heavy sparsity).
    run_sparse_attention_test(1, 1, 2048, 4, 4, 128, 0.05);
}
