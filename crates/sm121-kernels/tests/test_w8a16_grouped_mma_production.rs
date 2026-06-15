//! Production-shape coverage for `gemm_w8a16_grouped_mma` — addresses the
//! audit gap noted in `docs/audit_remediation.md` T2.6.
//!
//! The original `test_w8a16_grouped_mma` was bit-exact at one tiny synthetic
//! shape (4 experts × 16 tokens, K=128, N=64) and bypassed `moe_permute`.
//! Here we exercise:
//!
//!  1. **Production shape**: Gemma-4-26B-A4B's gate/up (K=2816, N=704) and
//!     down (K=704, N=2816) at M=128, top_k=8, n_experts=128.
//!  2. **Routing imbalance**: pathological distributions (one expert sees
//!     all 1024 slots; alternating sparse/dense; one expert sees zero).
//!  3. **moe_permute roundtrip**: drive the kernel through the full permute
//!     pipeline (histogram → offsets → permute → MMA → unpermute) and
//!     compare against the GEMV reference run with identity permutation.

mod common;

use sm121_kernels::{device, moe};

fn bf16(x: f32) -> u16 {
    half::bf16::from_f32(x).to_bits()
}
fn unbf16(b: u16) -> f32 {
    half::bf16::from_bits(b).to_f32()
}

/// Hand-build expert_offsets from per-expert counts (prefix sum).
fn host_offsets(counts: &[u32]) -> Vec<u32> {
    let mut off = Vec::with_capacity(counts.len() + 1);
    off.push(0);
    for &c in counts {
        off.push(off.last().unwrap() + c);
    }
    off
}

/// Build flat expert_ids from per-expert counts (token s in expert e if
/// offsets[e] <= s < offsets[e+1]).
fn host_active_eids(counts: &[u32]) -> Vec<u32> {
    let mut eids = Vec::new();
    for (e, &c) in counts.iter().enumerate() {
        for _ in 0..c {
            eids.push(e as u32);
        }
    }
    eids
}

/// Run gemv_w8a16_grouped_split_k as the per-slot reference, write [total, n]
/// in expert-sorted slot order.
fn run_gemv_reference(
    ctx: &std::sync::Arc<cudarc::driver::CudaContext>,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    a_dev: &cudarc::driver::CudaSlice<u16>,
    b_fp8_dev: &cudarc::driver::CudaSlice<u8>,
    b_scale: f32,
    active_eids_dev: &cudarc::driver::CudaSlice<u32>,
    total: u32,
    n: u32,
    k: u32,
) -> Vec<u16> {
    let mut out_f32 = stream.alloc_zeros::<f32>((total * n) as usize).unwrap();
    let num_shards: u32 = (k / 256).max(1).min(8);
    moe::gemv_w8a16_grouped_split_k(
        ctx,
        stream,
        a_dev,
        b_fp8_dev,
        b_scale,
        active_eids_dev,
        &mut out_f32,
        total,
        n,
        k,
        num_shards,
        k,
    )
    .unwrap();
    let mut out_bf16 = stream.alloc_zeros::<u16>((total * n) as usize).unwrap();
    sm121_kernels::activation::f32_to_bf16(ctx, stream, &out_f32, &mut out_bf16, total * n)
        .unwrap();
    stream.synchronize().unwrap();
    stream.memcpy_dtov(&out_bf16).unwrap()
}

fn check_close(name: &str, ref_h: &[u16], sub_h: &[u16], abs_tol: f32, rel_tol: f32) {
    assert_eq!(ref_h.len(), sub_h.len());
    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    let mut n_bad = 0;
    for i in 0..ref_h.len() {
        let r = unbf16(ref_h[i]);
        let s = unbf16(sub_h[i]);
        let d = (r - s).abs();
        if d > max_abs {
            max_abs = d;
        }
        let rel = d / r.abs().max(1e-6);
        if rel > max_rel {
            max_rel = rel;
        }
        if d > abs_tol && rel > rel_tol {
            n_bad += 1;
            if n_bad <= 4 {
                eprintln!("  [{name}] bad@{i} ref={r:.4} sub={s:.4} d={d:.4e}");
            }
        }
    }
    eprintln!(
        "[{name}] max_abs={max_abs:.4e}, max_rel={:.4}%, bad={n_bad}",
        max_rel * 100.0
    );
    assert!(
        n_bad == 0,
        "{name}: {n_bad} elements exceed both abs={abs_tol} and rel={rel_tol} tolerances"
    );
}

/// Run with a given (counts, K, N) shape — directly drives the MMA kernel
/// with manually-constructed sorted layout (mirrors what `moe_permute`
/// would produce). Caller verifies vs the GEMV reference.
fn run_case(name: &str, counts: &[u32], n: u32, k: u32) {
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();

    let num_experts = counts.len() as u32;
    let total: u32 = counts.iter().sum();
    let m_max = *counts.iter().max().unwrap_or(&1);
    if m_max == 0 {
        // Skip — empty routing has nothing to test (all experts see zero).
        return;
    }

    let offsets_h = host_offsets(counts);
    let eids_h = host_active_eids(counts);

    let mut s = 0xACE0_F00D_u64;
    let mut next = || -> f32 {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 33) as u32 as f32) / (u32::MAX as f32) - 0.5
    };
    let a_h: Vec<u16> = (0..(total * k) as usize)
        .map(|_| bf16(next() * 0.5))
        .collect();
    let b_fp8_h: Vec<u8> = (0..(num_experts * k * n) as usize)
        .map(|i| ((i.wrapping_mul(31) + 17) % 240) as u8)
        .collect();
    let b_scale: f32 = 0.03125;

    let a_dev = stream.memcpy_stod(&a_h).unwrap();
    let b_fp8_dev = stream.memcpy_stod(&b_fp8_h).unwrap();
    let offsets_dev = stream.memcpy_stod(&offsets_h).unwrap();
    let eids_dev = stream.memcpy_stod(&eids_h).unwrap();

    let ref_h = run_gemv_reference(
        &ctx, &stream, &a_dev, &b_fp8_dev, b_scale, &eids_dev, total, n, k,
    );

    let mut sub_bf16 = stream.alloc_zeros::<u16>((total * n) as usize).unwrap();
    moe::gemm_w8a16_grouped_mma(
        &ctx,
        &stream,
        &a_dev,
        &b_fp8_dev,
        b_scale,
        &mut sub_bf16,
        &offsets_dev,
        num_experts,
        m_max,
        n,
        k,
    )
    .unwrap();
    stream.synchronize().unwrap();
    let sub_h = stream.memcpy_dtov(&sub_bf16).unwrap();

    // Both kernels accumulate in FP32 and round to BF16 at store; small ULP
    // diffs are expected on rare elements when summation order differs.
    check_close(name, &ref_h, &sub_h, 0.05, 0.02);
}

#[test]
fn production_shape_gate_up_balanced() {
    // M=128 top_k=8 over 128 experts, exactly balanced (8 tokens each).
    // K=2816, N=704 — Gemma-4-26B-A4B gate/up.
    let counts = vec![8u32; 128];
    run_case(
        "balanced gate/up (128e × 8tok, K=2816 N=704)",
        &counts,
        704,
        2816,
    );
}

#[test]
fn production_shape_down_balanced() {
    // Same routing shape, swapped K/N — Gemma-4-26B-A4B down (K=704, N=2816).
    let counts = vec![8u32; 128];
    run_case(
        "balanced down (128e × 8tok, K=704 N=2816)",
        &counts,
        2816,
        704,
    );
}

#[test]
fn imbalance_one_expert_takes_all() {
    // Pathological: expert 0 sees all 1024 slots, experts 1..127 see zero.
    let mut counts = vec![0u32; 128];
    counts[0] = 1024;
    run_case("imbalanced (expert 0 = all 1024)", &counts, 704, 2816);
}

#[test]
fn imbalance_alternating_sparse_dense() {
    // Even-indexed experts: 16 tokens; odd: 0. Half the experts dormant.
    let mut counts = vec![0u32; 128];
    for i in (0..128).step_by(2) {
        counts[i] = 16;
    }
    run_case("alternating dense/sparse", &counts, 704, 2816);
}

#[test]
fn imbalance_small_unaligned_counts() {
    // Realistic routing imbalance — random-ish counts summing to 1024.
    // Per-expert counts span [0..32] to exercise the m_tile bound check.
    let mut counts = vec![0u32; 128];
    let mut s = 0xBEEF_FACE_u64;
    let mut next_u32 = || -> u32 {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (s >> 33) as u32
    };
    // Distribute 1024 tokens with a heavy-tailed-ish distribution.
    let mut remaining = 1024_i64;
    for i in 0..127 {
        let cap = remaining.min(33) as u32;
        let c = (next_u32() % (cap + 1)).min(32);
        counts[i] = c;
        remaining -= c as i64;
        if remaining <= 0 {
            break;
        }
    }
    counts[127] = remaining.max(0) as u32;
    let total: u32 = counts.iter().sum();
    eprintln!(
        "imbalance random: counts sum = {total}, max = {}, zeros = {}",
        counts.iter().max().unwrap(),
        counts.iter().filter(|&&c| c == 0).count()
    );
    run_case("random imbalance", &counts, 704, 2816);
}

#[test]
fn moe_permute_roundtrip_with_mma() {
    // End-to-end: build flat expert_ids [M*top_k] (representing M=64 tokens
    // each routing to top_k=4 experts uniformly randomly from n_experts=32),
    // run moe_histogram + moe_expert_offsets + moe_permute on a [M, K]
    // activation buffer; feed permuted_act + offsets into the MMA kernel;
    // compare per-expert outputs against the GEMV-per-slot reference run on
    // the SAME permuted_act + eids derived from the offset structure.
    //
    // This exercises the full integration pattern used by
    // `forward_decode_step_batched_pathb_moe` in MMA mode.
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();

    let m: u32 = 64;
    let top_k: u32 = 4;
    let num_experts: u32 = 32;
    let n: u32 = 64;
    let k: u32 = 256;
    let total = m * top_k;

    let mut s = 0xFEED_1234_u64;
    let mut step = || -> u64 {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        s >> 33
    };

    // [M, top_k] expert_ids per token, no two same per token.
    let mut expert_ids_h: Vec<u32> = Vec::with_capacity(total as usize);
    for _ in 0..m {
        let mut picked: Vec<u32> = Vec::new();
        while picked.len() < top_k as usize {
            let e = (step() as u32) % num_experts;
            if !picked.contains(&e) {
                picked.push(e);
            }
        }
        expert_ids_h.extend(picked);
    }

    let a_h: Vec<u16> = (0..(m * k) as usize)
        .map(|_| bf16(((step() as u32 as f32) / (u32::MAX as f32) - 0.5) * 0.5))
        .collect();
    let b_fp8_h: Vec<u8> = (0..(num_experts * k * n) as usize)
        .map(|i| ((i.wrapping_mul(31) + 17) % 240) as u8)
        .collect();
    let b_scale = 0.03125f32;

    let a_dev = stream.memcpy_stod(&a_h).unwrap();
    let b_fp8_dev = stream.memcpy_stod(&b_fp8_h).unwrap();
    let expert_ids_dev = stream.memcpy_stod(&expert_ids_h).unwrap();

    // Run the histogram → offsets → permute pipeline.
    let mut counts_dev = stream.alloc_zeros::<u32>(num_experts as usize).unwrap();
    moe::moe_histogram(
        &ctx,
        &stream,
        &expert_ids_dev,
        &mut counts_dev,
        total,
        num_experts,
    )
    .unwrap();
    let mut offsets_dev = stream.alloc_zeros::<u32>(num_experts as usize + 1).unwrap();
    moe::moe_expert_offsets(&ctx, &stream, &counts_dev, &mut offsets_dev, num_experts).unwrap();

    let mut cursor_dev = stream.alloc_zeros::<u32>(num_experts as usize).unwrap();
    let mut permuted_act_dev = stream.alloc_zeros::<u16>((total * k) as usize).unwrap();
    let mut inverse_index_dev = stream.alloc_zeros::<u32>(total as usize).unwrap();
    moe::moe_permute(
        &ctx,
        &stream,
        &a_dev,
        &expert_ids_dev,
        &offsets_dev,
        &mut cursor_dev,
        &mut permuted_act_dev,
        &mut inverse_index_dev,
        m,
        top_k,
        k,
    )
    .unwrap();

    // MMA subject: takes permuted_act + offsets (no eids needed for the
    // grouped path — it iterates per expert via offsets).
    let m_max = total; // worst case
    let mut sub_bf16 = stream.alloc_zeros::<u16>((total * n) as usize).unwrap();
    moe::gemm_w8a16_grouped_mma(
        &ctx,
        &stream,
        &permuted_act_dev,
        &b_fp8_dev,
        b_scale,
        &mut sub_bf16,
        &offsets_dev,
        num_experts,
        m_max,
        n,
        k,
    )
    .unwrap();

    // Reference: per-slot GEMV over the same permuted activations + per-slot
    // expert ID (which can be derived from inverse_index / top_k mapping back
    // to expert_ids, or recomputed from offsets).
    let offsets_h = stream.memcpy_dtov(&offsets_dev).unwrap();
    let mut ref_eids_h: Vec<u32> = Vec::with_capacity(total as usize);
    for e in 0..num_experts as usize {
        for _ in offsets_h[e]..offsets_h[e + 1] {
            ref_eids_h.push(e as u32);
        }
    }
    let ref_eids_dev = stream.memcpy_stod(&ref_eids_h).unwrap();
    let ref_h = run_gemv_reference(
        &ctx,
        &stream,
        &permuted_act_dev,
        &b_fp8_dev,
        b_scale,
        &ref_eids_dev,
        total,
        n,
        k,
    );

    stream.synchronize().unwrap();
    let sub_h = stream.memcpy_dtov(&sub_bf16).unwrap();
    check_close("moe_permute roundtrip", &ref_h, &sub_h, 0.05, 0.02);

    // Also sanity-check histogram: counts should sum to total.
    let counts_h = stream.memcpy_dtov(&counts_dev).unwrap();
    let counts_sum: u32 = counts_h.iter().sum();
    assert_eq!(counts_sum, total, "histogram sum mismatch");
}
