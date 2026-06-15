mod common;

use common::load_npz;
use sm121_kernels::{device, moe};

fn run_moe_test(npz_name: &str) {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let mut npz = load_npz(npz_name);

    let logits_np: ndarray::Array2<u16> = npz.by_name("logits").unwrap();
    let expert_ids_expected: ndarray::Array2<u32> = npz.by_name("expert_ids").unwrap();
    let weights_expected: ndarray::Array2<u16> = npz.by_name("weights").unwrap();
    let num_tokens: ndarray::Array0<u32> = npz.by_name("num_tokens").unwrap();
    let num_experts: ndarray::Array0<u32> = npz.by_name("num_experts").unwrap();
    let top_k: ndarray::Array0<u32> = npz.by_name("top_k").unwrap();

    let num_tokens = num_tokens.into_scalar();
    let num_experts = num_experts.into_scalar();
    let top_k = top_k.into_scalar();

    let logits_flat: Vec<u16> = logits_np.into_raw_vec_and_offset().0;
    let expected_ids: Vec<u32> = expert_ids_expected.into_raw_vec_and_offset().0;
    let expected_weights: Vec<u16> = weights_expected.into_raw_vec_and_offset().0;

    let logits_dev = stream.memcpy_stod(&logits_flat).unwrap();
    let mut ids_dev = stream
        .alloc_zeros::<u32>((num_tokens * top_k) as usize)
        .unwrap();
    let mut weights_dev = stream
        .alloc_zeros::<u16>((num_tokens * top_k) as usize)
        .unwrap();

    moe::moe_routing(
        &ctx,
        &stream,
        &logits_dev,
        &mut ids_dev,
        &mut weights_dev,
        num_tokens,
        num_experts,
        top_k,
    )
    .unwrap();

    let ids_host = stream.memcpy_dtov(&ids_dev).unwrap();
    let weights_host = stream.memcpy_dtov(&weights_dev).unwrap();

    let k = top_k as usize;
    for token in 0..num_tokens as usize {
        // Sort both actual and expected by expert_id so tie-breaking order doesn't matter
        let mut actual_pairs: Vec<(u32, u16)> = ids_host[token * k..(token + 1) * k]
            .iter()
            .zip(weights_host[token * k..(token + 1) * k].iter())
            .map(|(&id, &w)| (id, w))
            .collect();
        let mut expected_pairs: Vec<(u32, u16)> = expected_ids[token * k..(token + 1) * k]
            .iter()
            .zip(expected_weights[token * k..(token + 1) * k].iter())
            .map(|(&id, &w)| (id, w))
            .collect();
        actual_pairs.sort_by_key(|&(id, _)| id);
        expected_pairs.sort_by_key(|&(id, _)| id);

        // Check expert IDs match
        let actual_ids: Vec<u32> = actual_pairs.iter().map(|&(id, _)| id).collect();
        let expected_ids_sorted: Vec<u32> = expected_pairs.iter().map(|&(id, _)| id).collect();
        assert_eq!(
            actual_ids, expected_ids_sorted,
            "expert_ids set mismatch at token {token}"
        );

        // Check weights match per expert
        for (&(_, aw), &(_, ew)) in actual_pairs.iter().zip(expected_pairs.iter()) {
            let af = half::bf16::from_bits(aw).to_f32();
            let ef = half::bf16::from_bits(ew).to_f32();
            let diff = (af - ef).abs();
            assert!(
                diff < 0.1,
                "weight mismatch at token {token}: actual={af:.6} expected={ef:.6} diff={diff:.6}"
            );
        }

        // Check weights sum to ~1.0
        let mut weight_sum: f32 = 0.0;
        for rank in 0..k {
            let w = half::bf16::from_bits(weights_host[token * k + rank]).to_f32();
            assert!(
                (0.0..=1.0).contains(&w),
                "weight out of range at token {token} rank {rank}: {w:.6}"
            );
            weight_sum += w;
        }
        let sum_diff = (weight_sum - 1.0).abs();
        assert!(
            sum_diff < 0.05,
            "weight sum mismatch at token {token}: sum={weight_sum:.6}"
        );
    }

    eprintln!(
        "moe {npz_name}: all {num_tokens} tokens x top_{top_k} match ({num_experts} experts)"
    );
}

#[test]
fn test_moe_routing_t16_e8_k2() {
    run_moe_test("moe_routing_t16_e8_k2.npz");
}

#[test]
fn test_moe_routing_t32_e64_k2() {
    run_moe_test("moe_routing_t32_e64_k2.npz");
}

// Regression: top_k=6 (DSV2-Lite). Previously, the SMEM topk_val region
// (4 entries) overlapped the topk_idx region, corrupting expert IDs at slots
// 0..2 when top_k > 4. Fixed by expanding both regions to 8 entries.
#[test]
fn test_moe_routing_t8_e64_k6() {
    run_moe_test("moe_routing_t8_e64_k6.npz");
}

// Regression: top_k=8 (DeepSeek V3 ceiling). Locks the maximum supported top_k.
#[test]
fn test_moe_routing_t4_e64_k8() {
    run_moe_test("moe_routing_t4_e64_k8.npz");
}

// GPU-side compaction of histogram into active_experts list.
#[test]
fn test_moe_active_experts_compact() {
    use sm121_kernels::moe;
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    // Histogram with non-zero entries at experts {2, 5, 7, 11, 12}.
    let num_experts: u32 = 16;
    let mut hist = vec![0u32; num_experts as usize];
    hist[2] = 3;
    hist[5] = 1;
    hist[7] = 2;
    hist[11] = 4;
    hist[12] = 1;
    let expected_active: std::collections::HashSet<u32> =
        [2u32, 5, 7, 11, 12].iter().copied().collect();

    let hist_dev = stream.memcpy_stod(&hist).unwrap();
    let mut active_dev = stream.alloc_zeros::<u32>(num_experts as usize).unwrap();
    let mut count_dev = stream.alloc_zeros::<u32>(1).unwrap();

    moe::moe_active_experts_compact(
        &ctx,
        &stream,
        &hist_dev,
        &mut active_dev,
        &mut count_dev,
        num_experts,
    )
    .expect("compact");

    let count_host = stream.memcpy_dtov(&count_dev).unwrap();
    let active_host = stream.memcpy_dtov(&active_dev).unwrap();
    assert_eq!(count_host[0], 5, "should find 5 active experts");

    let active_set: std::collections::HashSet<u32> = active_host[..count_host[0] as usize]
        .iter()
        .copied()
        .collect();
    assert_eq!(active_set, expected_active, "active expert set must match");
    eprintln!(
        "moe_active_experts_compact: {} active out of {num_experts}, set={:?}",
        count_host[0],
        active_host[..count_host[0] as usize].to_vec()
    );
}

// Sparse grouped GEMM: routing some tokens to a SUBSET of experts and
// validating the sparse kernel produces identical output to the dense
// kernel run on the full expert set.
#[test]
fn test_grouped_mma_sparse_matches_dense() {
    use sm121_kernels::moe;
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let num_experts: u32 = 8;
    let n: u32 = 64;
    let k: u32 = 32;
    let m_max: u32 = 4;

    // Tokens go ONLY to experts {2, 5, 7}; rest are empty.
    // expert_offsets is the cumulative count: experts 0..2 have 0 each,
    // expert 2 has 2 tokens, 3..5 have 0, 5 has 1, 6 has 0, 7 has 3.
    let counts = [0u32, 0, 2, 0, 0, 1, 0, 3];
    let mut offsets = vec![0u32; num_experts as usize + 1];
    for i in 0..num_experts as usize {
        offsets[i + 1] = offsets[i] + counts[i];
    }
    let total_tokens = *offsets.last().unwrap();
    let active_set: Vec<u32> = counts
        .iter()
        .enumerate()
        .filter_map(|(i, &c)| if c > 0 { Some(i as u32) } else { None })
        .collect();
    let num_active = active_set.len() as u32;

    // Random A and B (deterministic via seed-y formulas).
    let a_host: Vec<u16> = (0..(total_tokens * k))
        .map(|i| {
            let v = ((i as f32) * 0.011).sin() * 0.5;
            half::bf16::from_f32(v).to_bits()
        })
        .collect();
    let b_host: Vec<u16> = (0..(num_experts * k * n))
        .map(|i| {
            let v = ((i as f32) * 0.0073).cos() * 0.4;
            half::bf16::from_f32(v).to_bits()
        })
        .collect();

    let a_dev = stream.memcpy_stod(&a_host).unwrap();
    let b_dev = stream.memcpy_stod(&b_host).unwrap();
    let off_dev = stream.memcpy_stod(&offsets).unwrap();
    let active_dev = stream.memcpy_stod(&active_set).unwrap();

    let mut c_dense = stream
        .alloc_zeros::<u16>((total_tokens * n) as usize)
        .unwrap();
    let mut c_sparse = stream
        .alloc_zeros::<u16>((total_tokens * n) as usize)
        .unwrap();

    moe::gemm_bf16_grouped_mma(
        &ctx,
        &stream,
        &a_dev,
        &b_dev,
        &mut c_dense,
        &off_dev,
        num_experts,
        m_max,
        n,
        k,
    )
    .expect("dense");
    moe::gemm_bf16_grouped_mma_sparse(
        &ctx,
        &stream,
        &a_dev,
        &b_dev,
        &mut c_sparse,
        &off_dev,
        &active_dev,
        num_active,
        m_max,
        n,
        k,
    )
    .expect("sparse");

    let h_dense = stream.memcpy_dtov(&c_dense).unwrap();
    let h_sparse = stream.memcpy_dtov(&c_sparse).unwrap();

    let mut max_diff = 0.0f32;
    for (a, b) in h_dense.iter().zip(h_sparse.iter()) {
        let fa = f32::from_bits((*a as u32) << 16);
        let fb = f32::from_bits((*b as u32) << 16);
        let d = (fa - fb).abs();
        if d > max_diff {
            max_diff = d;
        }
    }
    eprintln!("sparse vs dense grouped GEMM: total_tokens={total_tokens} active={num_active}/{num_experts} max_diff={max_diff:.5}");
    assert!(
        max_diff <= 0.001,
        "sparse must match dense exactly: {}",
        max_diff
    );
}

#[test]
fn test_gemv_bf16_grouped_split_k_broadcast() {
    // Path: gate/up step. x is broadcast across active experts (x_stride=0).
    // Verify that batched output matches per-expert serial split-K gemv.
    use sm121_kernels::{device, gemm};
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let n_routed: u32 = 8;
    let k: u32 = 1024;
    let n: u32 = 512;
    let active_eids: Vec<u32> = vec![3, 0, 6, 2];
    let num_active = active_eids.len() as u32;

    let stride_e = (k as usize) * (n as usize);

    // x [K]
    let x: Vec<u16> = (0..k)
        .map(|i| half::bf16::from_f32(((i as f32) * 0.0011 - 0.5).cos() * 0.7).to_bits())
        .collect();
    // B [n_routed, K, N]
    let mut b_stack: Vec<u16> = vec![0u16; (n_routed as usize) * stride_e];
    for e in 0..n_routed as usize {
        for i in 0..stride_e {
            let v = (((e * 19 + i) % 1009) as f32) * 0.0009 - 0.45;
            b_stack[e * stride_e + i] = half::bf16::from_f32(v).to_bits();
        }
    }

    let x_dev = stream.memcpy_stod(&x).unwrap();
    let b_dev = stream.memcpy_stod(&b_stack).unwrap();
    let eids_dev = stream.memcpy_stod(&active_eids).unwrap();
    let mut out_grouped = stream
        .alloc_zeros::<f32>((num_active as usize) * (n as usize))
        .unwrap();

    let shards = (k / 256).clamp(1, 16);
    sm121_kernels::moe::gemv_bf16_grouped_split_k(
        &ctx,
        &stream,
        &x_dev,
        &b_dev,
        &eids_dev,
        &mut out_grouped,
        num_active,
        n,
        k,
        shards,
        0,
    )
    .expect("grouped split_k broadcast");

    // Reference: per-expert serial split-K gemv on isolated weight slices.
    let mut max_diff = 0.0f32;
    let out_grouped_host = stream.memcpy_dtov(&out_grouped).unwrap();
    for (a_idx, &eid) in active_eids.iter().enumerate() {
        let off = (eid as usize) * stride_e;
        let iso: Vec<u16> = b_stack[off..off + stride_e].to_vec();
        let iso_dev = stream.memcpy_stod(&iso).unwrap();
        let mut ref_out = stream.alloc_zeros::<f32>(n as usize).unwrap();
        gemm::gemv_bf16_split_k(&ctx, &stream, &x_dev, &iso_dev, &mut ref_out, n, k, shards)
            .expect("ref gemv");
        let ref_host = stream.memcpy_dtov(&ref_out).unwrap();

        for j in 0..n as usize {
            let g = out_grouped_host[a_idx * (n as usize) + j];
            let r = ref_host[j];
            let d = (g - r).abs();
            if d > max_diff {
                max_diff = d;
            }
        }
    }
    eprintln!(
        "grouped split_k (broadcast x) num_active={num_active} k={k} n={n} max_diff={max_diff:.5}"
    );
    // Atomic-add ordering across SMs => tiny ulp drift even between launches of the same kernel.
    assert!(
        max_diff < 0.5,
        "grouped split_k broadcast diverged: {}",
        max_diff
    );
}

#[test]
fn test_gemv_bf16_grouped_split_k_per_expert_x() {
    // Path: down step. x_stride > 0 → each expert reads its own input row.
    use sm121_kernels::{device, gemm};
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let n_routed: u32 = 5;
    let k: u32 = 768;
    let n: u32 = 256;
    let active_eids: Vec<u32> = vec![1, 3, 0];
    let num_active = active_eids.len() as u32;

    let stride_e = (k as usize) * (n as usize);

    // X [num_active, K] — different content per expert
    let mut x_stacked: Vec<u16> = vec![0u16; (num_active as usize) * (k as usize)];
    for a in 0..num_active as usize {
        for i in 0..k as usize {
            let v = ((a * 13 + i) as f32 * 0.0007 - 0.5).sin();
            x_stacked[a * (k as usize) + i] = half::bf16::from_f32(v).to_bits();
        }
    }
    // B [n_routed, K, N]
    let mut b_stack: Vec<u16> = vec![0u16; (n_routed as usize) * stride_e];
    for e in 0..n_routed as usize {
        for i in 0..stride_e {
            let v = (((e * 7 + i) % 503) as f32) * 0.0013 - 0.3;
            b_stack[e * stride_e + i] = half::bf16::from_f32(v).to_bits();
        }
    }

    let x_dev = stream.memcpy_stod(&x_stacked).unwrap();
    let b_dev = stream.memcpy_stod(&b_stack).unwrap();
    let eids_dev = stream.memcpy_stod(&active_eids).unwrap();
    let mut out_grouped = stream
        .alloc_zeros::<f32>((num_active as usize) * (n as usize))
        .unwrap();

    let shards = (k / 256).clamp(1, 16);
    sm121_kernels::moe::gemv_bf16_grouped_split_k(
        &ctx,
        &stream,
        &x_dev,
        &b_dev,
        &eids_dev,
        &mut out_grouped,
        num_active,
        n,
        k,
        shards,
        k,
    )
    .expect("grouped split_k per-expert x");

    // Reference: per-expert serial split-K with the matching x slice + eid weight slice.
    let mut max_diff = 0.0f32;
    let out_grouped_host = stream.memcpy_dtov(&out_grouped).unwrap();
    for (a_idx, &eid) in active_eids.iter().enumerate() {
        let xi: Vec<u16> = x_stacked[a_idx * (k as usize)..(a_idx + 1) * (k as usize)].to_vec();
        let xi_dev = stream.memcpy_stod(&xi).unwrap();
        let off = (eid as usize) * stride_e;
        let iso: Vec<u16> = b_stack[off..off + stride_e].to_vec();
        let iso_dev = stream.memcpy_stod(&iso).unwrap();
        let mut ref_out = stream.alloc_zeros::<f32>(n as usize).unwrap();
        gemm::gemv_bf16_split_k(&ctx, &stream, &xi_dev, &iso_dev, &mut ref_out, n, k, shards)
            .expect("ref gemv");
        let ref_host = stream.memcpy_dtov(&ref_out).unwrap();

        for j in 0..n as usize {
            let g = out_grouped_host[a_idx * (n as usize) + j];
            let r = ref_host[j];
            let d = (g - r).abs();
            if d > max_diff {
                max_diff = d;
            }
        }
    }
    eprintln!(
        "grouped split_k (per-expert x) num_active={num_active} k={k} n={n} max_diff={max_diff:.5}"
    );
    assert!(
        max_diff < 0.5,
        "grouped split_k per-expert x diverged: {}",
        max_diff
    );
}

#[test]
fn test_moe_route_decode_full_matches_host() {
    // Cross-check the GPU-side full-softmax routing kernel against a host reference
    // matching `moe_route_host_decode` semantics in the dsv2_lite_chat example.
    use sm121_kernels::{device, moe};
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let n_routed: u32 = 64;
    let top_k: u32 = 6;
    let scaling: f32 = 1.0;

    // Deterministic logits with clear top-k separation (avoid f32 ulp ties).
    let logits_f: Vec<f32> = (0..n_routed as usize)
        .map(|i| ((i as f32) * 0.1) + ((i as f32 * 7.0).sin() * 0.5))
        .collect();
    let logits_bf16: Vec<u16> = logits_f
        .iter()
        .map(|&v| half::bf16::from_f32(v).to_bits())
        .collect();

    let logits_dev = stream.memcpy_stod(&logits_bf16).unwrap();
    let mut ids_dev = stream.alloc_zeros::<u32>(top_k as usize).unwrap();
    let mut weights_dev = stream.alloc_zeros::<f32>(top_k as usize).unwrap();

    moe::moe_route_decode_full(
        &ctx,
        &stream,
        &logits_dev,
        &mut ids_dev,
        &mut weights_dev,
        n_routed,
        top_k,
        scaling,
    )
    .expect("moe_route_decode_full");

    let ids_host = stream.memcpy_dtov(&ids_dev).unwrap();
    let weights_host = stream.memcpy_dtov(&weights_dev).unwrap();

    // Reference: full softmax, top-k by descending softmax prob.
    let mut probs: Vec<f32> = logits_f
        .iter()
        .map(|&v| half::bf16::from_f32(v).to_f32())
        .collect();
    let max = probs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in probs.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    for v in probs.iter_mut() {
        *v /= sum;
    }

    let mut indexed: Vec<(usize, f32)> = probs.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(top_k as usize);

    for i in 0..top_k as usize {
        let expected_id = indexed[i].0 as u32;
        let expected_w = indexed[i].1 * scaling;
        eprintln!(
            "slot {i}: gpu=({},{:.6}) host=({},{:.6})",
            ids_host[i], weights_host[i], expected_id, expected_w
        );
        assert_eq!(ids_host[i], expected_id, "expert ID mismatch at slot {i}");
        let diff = (weights_host[i] - expected_w).abs();
        // Tolerance: ex2.approx vs libm exp can differ by ~2 ulps in unnormalized,
        // then divide-by-sum amplifies. Sum is also ex2.approx-derived. Use 1e-3 abs.
        assert!(
            diff < 1e-3,
            "weight mismatch at slot {i}: gpu={} host={} diff={}",
            weights_host[i],
            expected_w,
            diff
        );
    }
}

#[test]
fn test_moe_route_decode_full_gdn_hybrid_n256_topk8() {
    // 35B-A3B MoE uses n_routed=256, top_k=8 (+ 1 shared expert).
    // Same algorithm as the n_routed=64 case but exercises the bumped SMEM
    // layout + 256-thread block.
    use sm121_kernels::{device, moe};
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let n_routed: u32 = 256;
    let top_k: u32 = 8;
    let scaling: f32 = 1.0;

    let logits_f: Vec<f32> = (0..n_routed as usize)
        .map(|i| ((i as f32) * 0.05) + ((i as f32 * 11.0).sin() * 0.7))
        .collect();
    let logits_bf16: Vec<u16> = logits_f
        .iter()
        .map(|&v| half::bf16::from_f32(v).to_bits())
        .collect();

    let logits_dev = stream.memcpy_stod(&logits_bf16).unwrap();
    let mut ids_dev = stream.alloc_zeros::<u32>(top_k as usize).unwrap();
    let mut weights_dev = stream.alloc_zeros::<f32>(top_k as usize).unwrap();

    moe::moe_route_decode_full(
        &ctx,
        &stream,
        &logits_dev,
        &mut ids_dev,
        &mut weights_dev,
        n_routed,
        top_k,
        scaling,
    )
    .expect("moe_route_decode_full n=256 k=8");

    let ids_host = stream.memcpy_dtov(&ids_dev).unwrap();
    let weights_host = stream.memcpy_dtov(&weights_dev).unwrap();

    // Reference
    let mut probs: Vec<f32> = logits_f
        .iter()
        .map(|&v| half::bf16::from_f32(v).to_f32())
        .collect();
    let max = probs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in probs.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    for v in probs.iter_mut() {
        *v /= sum;
    }

    let mut indexed: Vec<(usize, f32)> = probs.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(top_k as usize);

    for i in 0..top_k as usize {
        let expected_id = indexed[i].0 as u32;
        let expected_w = indexed[i].1 * scaling;
        eprintln!(
            "slot {i}: gpu=({},{:.6}) host=({},{:.6})",
            ids_host[i], weights_host[i], expected_id, expected_w
        );
        assert_eq!(ids_host[i], expected_id, "expert ID mismatch at slot {i}");
        let diff = (weights_host[i] - expected_w).abs();
        assert!(
            diff < 1e-3,
            "weight mismatch at slot {i}: gpu={} host={} diff={}",
            weights_host[i],
            expected_w,
            diff
        );
    }
}

#[test]
fn test_gemv_bf16_grouped_split_k_dual_matches_singles() {
    // Verify that the dual gemv produces gate AND up outputs matching two separate
    // single-output `gemv_bf16_grouped_split_k` calls.
    use sm121_kernels::{device, moe};
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let n_routed: u32 = 8;
    let k: u32 = 1024;
    let n: u32 = 512;
    let active_eids: Vec<u32> = vec![3, 0, 6, 2];
    let num_active = active_eids.len() as u32;
    let stride_e = (k as usize) * (n as usize);

    let x: Vec<u16> = (0..k)
        .map(|i| half::bf16::from_f32(((i as f32) * 0.0011 - 0.5).cos() * 0.7).to_bits())
        .collect();
    let mut b_gate: Vec<u16> = vec![0u16; (n_routed as usize) * stride_e];
    let mut b_up: Vec<u16> = vec![0u16; (n_routed as usize) * stride_e];
    for e in 0..n_routed as usize {
        for i in 0..stride_e {
            let g = (((e * 11 + i) % 503) as f32) * 0.0011 - 0.3;
            let u = (((e * 17 + i) % 401) as f32) * 0.0009 - 0.2;
            b_gate[e * stride_e + i] = half::bf16::from_f32(g).to_bits();
            b_up[e * stride_e + i] = half::bf16::from_f32(u).to_bits();
        }
    }

    let x_dev = stream.memcpy_stod(&x).unwrap();
    let bg_dev = stream.memcpy_stod(&b_gate).unwrap();
    let bu_dev = stream.memcpy_stod(&b_up).unwrap();
    let eids_dev = stream.memcpy_stod(&active_eids).unwrap();

    let need_out = (num_active as usize) * (n as usize);
    let mut out_g_dual = stream.alloc_zeros::<f32>(need_out).unwrap();
    let mut out_u_dual = stream.alloc_zeros::<f32>(need_out).unwrap();
    let mut out_g_single = stream.alloc_zeros::<f32>(need_out).unwrap();
    let mut out_u_single = stream.alloc_zeros::<f32>(need_out).unwrap();

    let shards = (k / 256).clamp(1, 16);
    moe::gemv_bf16_grouped_split_k_dual(
        &ctx,
        &stream,
        &x_dev,
        &bg_dev,
        &bu_dev,
        &eids_dev,
        &mut out_g_dual,
        &mut out_u_dual,
        num_active,
        n,
        k,
        shards,
    )
    .expect("dual");
    moe::gemv_bf16_grouped_split_k(
        &ctx,
        &stream,
        &x_dev,
        &bg_dev,
        &eids_dev,
        &mut out_g_single,
        num_active,
        n,
        k,
        shards,
        0,
    )
    .expect("single gate");
    moe::gemv_bf16_grouped_split_k(
        &ctx,
        &stream,
        &x_dev,
        &bu_dev,
        &eids_dev,
        &mut out_u_single,
        num_active,
        n,
        k,
        shards,
        0,
    )
    .expect("single up");

    let g_dual = stream.memcpy_dtov(&out_g_dual).unwrap();
    let u_dual = stream.memcpy_dtov(&out_u_dual).unwrap();
    let g_single = stream.memcpy_dtov(&out_g_single).unwrap();
    let u_single = stream.memcpy_dtov(&out_u_single).unwrap();

    let mut max_g_diff = 0.0f32;
    let mut max_u_diff = 0.0f32;
    for i in 0..need_out {
        let dg = (g_dual[i] - g_single[i]).abs();
        let du = (u_dual[i] - u_single[i]).abs();
        if dg > max_g_diff {
            max_g_diff = dg;
        }
        if du > max_u_diff {
            max_u_diff = du;
        }
    }
    eprintln!("dual vs singles: gate max_diff={max_g_diff:.5} up max_diff={max_u_diff:.5}");
    // Atomic-add ordering means partial sums can differ by tiny f32 ulps.
    assert!(max_g_diff < 0.5, "gate dual diverged: {max_g_diff}");
    assert!(max_u_diff < 0.5, "up dual diverged: {max_u_diff}");
}
