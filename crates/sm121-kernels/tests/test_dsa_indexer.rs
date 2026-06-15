//! Correctness test for the DSA indexer score kernel.
//!
//! Computes the reference scalar-side from the same random inputs the GPU
//! kernel sees, compares element-wise with absolute tolerance.
//!
//! Run:
//!     cargo test --release --test test_dsa_indexer -- --test-threads=1
//!
//! DSA is an experimental sparse-attention reference, gated behind the
//! `experimental` feature. Build/run with `--features experimental`.
#![cfg(feature = "experimental")]

mod common;

use sm121_kernels::attention::dsa::dsa_indexer_score_bf16;
use sm121_kernels::device;

const H_IDX: usize = 32;
const D_IDX: usize = 128;

/// f32 → BF16 (round-to-nearest-even).
fn f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    let rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16;
    rounded as u16
}

fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// Scalar-side reference implementation of the indexer score:
///   score[b,s,t] = Σ_h weights[b,s,h] * ReLU( Σ_i q[b,s,h,i] * k[b,t,i] )
fn reference_scores(
    q: &[u16],
    k: &[u16],
    weights: &[f32],
    batch: usize,
    seq_q: usize,
    seq_kv: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * seq_q * seq_kv];
    for b in 0..batch {
        for s in 0..seq_q {
            for t in 0..seq_kv {
                let mut score = 0.0f32;
                for h in 0..H_IDX {
                    let mut dot = 0.0f32;
                    for i in 0..D_IDX {
                        let q_idx = (((b * seq_q) + s) * H_IDX + h) * D_IDX + i;
                        let k_idx = ((b * seq_kv) + t) * D_IDX + i;
                        let q_val = bf16_to_f32(q[q_idx]);
                        let k_val = bf16_to_f32(k[k_idx]);
                        dot += q_val * k_val;
                    }
                    let dot_relu = if dot > 0.0 { dot } else { 0.0 };
                    let w_idx = (b * seq_q + s) * H_IDX + h;
                    score += weights[w_idx] * dot_relu;
                }
                out[(b * seq_q + s) * seq_kv + t] = score;
            }
        }
    }
    out
}

fn run_indexer_test(batch: usize, seq_q: usize, seq_kv: usize, tol: f32) {
    let ctx = device::init_device(0).expect("init_device");
    let stream = ctx.new_stream().expect("new_stream");

    // Build random inputs. Use a deterministic seed via Lehmer LCG for portability.
    let total_q = batch * seq_q * H_IDX * D_IDX;
    let total_k = batch * seq_kv * D_IDX;
    let total_w = batch * seq_q * H_IDX;
    let total_o = batch * seq_q * seq_kv;
    let mut q_host = vec![0u16; total_q];
    let mut k_host = vec![0u16; total_k];
    let mut w_host = vec![0f32; total_w];
    let mut seed: u32 = 0x12345678;
    let mut next_u01 = || {
        seed = seed.wrapping_mul(48271).wrapping_add(1);
        ((seed >> 8) as f32) / (1u32 << 24) as f32 // ~ U[0, 1)
    };
    let mut next_norm = || (next_u01() - 0.5) * 0.4; // ~ N(0, 0.2)
    for v in &mut q_host {
        *v = f32_to_bf16(next_norm());
    }
    for v in &mut k_host {
        *v = f32_to_bf16(next_norm());
    }
    for v in &mut w_host {
        *v = (next_u01() - 0.5) * 0.2;
    }

    // Reference
    let ref_scores = reference_scores(&q_host, &k_host, &w_host, batch, seq_q, seq_kv);

    // GPU
    let q_dev = stream.memcpy_stod(&q_host).expect("htod q");
    let k_dev = stream.memcpy_stod(&k_host).expect("htod k");
    let w_dev = stream.memcpy_stod(&w_host).expect("htod w");
    let mut out_dev = stream.alloc_zeros::<f32>(total_o).expect("alloc out");

    dsa_indexer_score_bf16(
        &ctx,
        &stream,
        &q_dev,
        &k_dev,
        &w_dev,
        &mut out_dev,
        batch as u32,
        seq_q as u32,
        seq_kv as u32,
    )
    .expect("dispatch");

    stream.synchronize().expect("sync");
    let out_host = stream.memcpy_dtov(&out_dev).expect("dtoh out");

    // Compare
    let mut max_abs: f32 = 0.0;
    let mut sum_abs: f64 = 0.0;
    let mut worst_idx = 0usize;
    for (i, (&actual, &expected)) in out_host.iter().zip(ref_scores.iter()).enumerate() {
        let d = (actual - expected).abs();
        sum_abs += d as f64;
        if d > max_abs {
            max_abs = d;
            worst_idx = i;
        }
    }
    let mean_abs = sum_abs as f32 / out_host.len() as f32;
    println!(
        "B={} S={} T={}: max_abs={:.5} mean_abs={:.6} worst_idx={} actual={:.4} expected={:.4}",
        batch,
        seq_q,
        seq_kv,
        max_abs,
        mean_abs,
        worst_idx,
        out_host[worst_idx],
        ref_scores[worst_idx],
    );
    assert!(max_abs <= tol, "max_abs {:.5} > tol {:.5}", max_abs, tol,);
}

#[test]
fn dsa_indexer_small() {
    // Smallest realistic shape: B=1, S=1, T=128 (single decode token, short cache).
    // tol accounts for BF16 dot-product summation order differences.
    run_indexer_test(1, 1, 128, 0.05);
}

#[test]
fn dsa_indexer_prefill_short() {
    // Multi-query prefill, short KV.
    run_indexer_test(1, 4, 256, 0.05);
}

#[test]
fn dsa_indexer_prefill_long() {
    // Single query, long KV — exercises the t-tile loop.
    run_indexer_test(1, 1, 1024, 0.10);
}

#[test]
fn dsa_indexer_batched() {
    // Multi-batch, exercises batch indexing.
    run_indexer_test(2, 4, 256, 0.05);
}

#[test]
fn dsa_indexer_large_kv() {
    // Long KV — 64 t-tiles per query, exercises grid.y scaling and
    // the per-(b, s) Q-load consistency across many CTAs.
    run_indexer_test(1, 2, 8192, 0.10);
}
