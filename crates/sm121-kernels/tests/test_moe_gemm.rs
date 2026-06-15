mod common;

use common::load_npz;
use sm121_kernels::{device, moe};

/// Convert a BF16-as-u16 slice to FP32 for numerical comparison.
fn bf16_vec_to_f32(v: &[u16]) -> Vec<f32> {
    v.iter()
        .map(|bits| f32::from_bits((*bits as u32) << 16))
        .collect()
}

/// End-to-end MoE routing pipeline test:
///   moe_histogram → moe_expert_offsets → moe_permute → gemm_bf16_grouped → moe_unpermute
///
/// Validation is order-INDEPENDENT: we use the kernel's own inverse_index
/// (which records, per permuted position, the original entry id) to rebuild
/// expected values. The specific within-expert permutation order is
/// non-deterministic (atomic cursor), so we don't compare to PyTorch argsort.
#[test]
fn test_moe_pipeline_bf16_t32_e8_k2_h128_n128() {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let num_tokens: u32 = 32;
    let num_experts: u32 = 8;
    let top_k: u32 = 2;
    let hidden: u32 = 128;
    let n_dim: u32 = 128;
    let total_entries = num_tokens * top_k;

    let mut npz = load_npz(&format!(
        "moe_grouped_gemm_t{num_tokens}_e{num_experts}_k{top_k}_h{hidden}_n{n_dim}.npz"
    ));
    let x_np: ndarray::Array2<u16> = npz.by_name("x").unwrap();
    let w_np: ndarray::Array3<u16> = npz.by_name("w").unwrap();
    let expert_ids_np: ndarray::Array1<u32> = npz.by_name("expert_ids").unwrap();
    let weights_np: ndarray::Array1<u16> = npz.by_name("weights").unwrap();
    let offsets_ref_np: ndarray::Array1<u32> = npz.by_name("offsets_ref").unwrap();
    let histogram_ref_np: ndarray::Array1<u32> = npz.by_name("histogram_ref").unwrap();
    let y_ref_np: ndarray::Array2<f32> = npz.by_name("y_ref").unwrap();

    let x: Vec<u16> = x_np.into_raw_vec_and_offset().0;
    let w: Vec<u16> = w_np.into_raw_vec_and_offset().0;
    let expert_ids: Vec<u32> = expert_ids_np.into_raw_vec_and_offset().0;
    let weights: Vec<u16> = weights_np.into_raw_vec_and_offset().0;
    let offsets_ref: Vec<u32> = offsets_ref_np.into_raw_vec_and_offset().0;
    let histogram_ref: Vec<u32> = histogram_ref_np.into_raw_vec_and_offset().0;
    let y_ref: Vec<f32> = y_ref_np.into_raw_vec_and_offset().0;

    // FP32 views of inputs for reference computation
    let _x_f32 = bf16_vec_to_f32(&x);
    let w_f32 = bf16_vec_to_f32(&w);
    let weights_f32 = bf16_vec_to_f32(&weights);

    let x_dev = stream.memcpy_stod(&x).unwrap();
    let w_dev = stream.memcpy_stod(&w).unwrap();
    let expert_ids_dev = stream.memcpy_stod(&expert_ids).unwrap();
    let weights_dev = stream.memcpy_stod(&weights).unwrap();

    // Histogram
    let mut histogram_dev = stream.alloc_zeros::<u32>(num_experts as usize).unwrap();
    moe::moe_histogram(
        &ctx,
        &stream,
        &expert_ids_dev,
        &mut histogram_dev,
        total_entries,
        num_experts,
    )
    .unwrap();
    let histogram_host = stream.memcpy_dtov(&histogram_dev).unwrap();
    assert_eq!(histogram_host, histogram_ref, "histogram mismatch");
    eprintln!("histogram: OK");

    // Offsets
    let mut offsets_dev = stream
        .alloc_zeros::<u32>((num_experts + 1) as usize)
        .unwrap();
    moe::moe_expert_offsets(&ctx, &stream, &histogram_dev, &mut offsets_dev, num_experts).unwrap();
    let offsets_host = stream.memcpy_dtov(&offsets_dev).unwrap();
    assert_eq!(offsets_host, offsets_ref, "offsets mismatch");
    eprintln!("offsets: OK");

    // Permute
    let mut cursor_dev = stream.alloc_zeros::<u32>(num_experts as usize).unwrap();
    let mut permuted_dev = stream
        .alloc_zeros::<u16>((total_entries * hidden) as usize)
        .unwrap();
    let mut inverse_dev = stream.alloc_zeros::<u32>(total_entries as usize).unwrap();
    moe::moe_permute(
        &ctx,
        &stream,
        &x_dev,
        &expert_ids_dev,
        &offsets_dev,
        &mut cursor_dev,
        &mut permuted_dev,
        &mut inverse_dev,
        num_tokens,
        top_k,
        hidden,
    )
    .unwrap();

    let permuted_host = stream.memcpy_dtov(&permuted_dev).unwrap();
    let inverse_host = stream.memcpy_dtov(&inverse_dev).unwrap();

    // 1. Each permuted row `dst` must equal x[inverse_index[dst] // top_k, :]
    // 2. inverse_index must be a permutation of [0..total_entries)
    // 3. expert_ids[inverse_index[dst]] must be monotone non-decreasing along dst
    //    (all entries for expert 0 come first, then expert 1, ...)
    let mut seen = vec![false; total_entries as usize];
    let mut last_expert = -1i64;
    for dst in 0..total_entries as usize {
        let entry_id = inverse_host[dst] as usize;
        assert!(
            entry_id < total_entries as usize,
            "inverse_index OOB at dst={dst}: got {entry_id}"
        );
        assert!(
            !seen[entry_id],
            "inverse_index not unique at dst={dst}: entry_id={entry_id} repeated"
        );
        seen[entry_id] = true;

        // Check expert ordering
        let this_expert = expert_ids[entry_id] as i64;
        assert!(this_expert >= last_expert,
            "expert_ids not non-decreasing at dst={dst}: prev_expert={last_expert}, this={this_expert}");
        last_expert = this_expert;

        // Check row contents match source token's row
        let src_token = entry_id / top_k as usize;
        for d in 0..hidden as usize {
            let actual = permuted_host[dst * hidden as usize + d];
            let expected = x[src_token * hidden as usize + d];
            assert_eq!(actual, expected,
                "permuted[{dst},{d}] = {actual:04x}, expected {expected:04x} (from token {src_token})");
        }
    }
    eprintln!("permute: OK ({total_entries} entries, all rows match source token)");

    // Grouped GEMM
    let m_max = total_entries;
    let mut c_permuted_dev = stream
        .alloc_zeros::<u16>((total_entries * n_dim) as usize)
        .unwrap();
    moe::gemm_bf16_grouped(
        &ctx,
        &stream,
        &permuted_dev,
        &w_dev,
        &mut c_permuted_dev,
        &offsets_dev,
        num_experts,
        m_max,
        n_dim,
        hidden,
    )
    .unwrap();

    let c_permuted_host = stream.memcpy_dtov(&c_permuted_dev).unwrap();

    // Verify grouped GEMM: for each permuted row dst, c[dst] should equal
    //   permuted_x[dst] @ w[expert_ids[inverse_index[dst]]]
    let c_permuted_f32 = bf16_vec_to_f32(&c_permuted_host);
    let permuted_f32 = bf16_vec_to_f32(&permuted_host);

    let mut gemm_max_diff: f32 = 0.0;
    let mut gemm_sum_diff: f64 = 0.0;
    let mut n_checked = 0usize;
    for dst in 0..total_entries as usize {
        let entry_id = inverse_host[dst] as usize;
        let expert = expert_ids[entry_id] as usize;
        let w_e_base = expert * (hidden as usize) * (n_dim as usize);
        let x_row_base = dst * hidden as usize;
        let c_row_base = dst * n_dim as usize;
        for n in 0..n_dim as usize {
            let mut acc = 0f32;
            for k in 0..hidden as usize {
                let x_val = permuted_f32[x_row_base + k];
                let w_val = w_f32[w_e_base + k * n_dim as usize + n];
                acc += x_val * w_val;
            }
            let expected = acc;
            let actual = c_permuted_f32[c_row_base + n];
            let d = (actual - expected).abs();
            if d > gemm_max_diff {
                gemm_max_diff = d;
            }
            gemm_sum_diff += d as f64;
            n_checked += 1;
        }
    }
    let gemm_mean_diff = gemm_sum_diff / n_checked as f64;
    eprintln!("grouped GEMM: max_diff={gemm_max_diff:.4} mean_diff={gemm_mean_diff:.4}");
    // BF16 GEMM with K=128 in FP32 accumulation: typical max_diff around 1-2
    assert!(
        gemm_max_diff <= 2.0,
        "grouped GEMM max_diff too high: {gemm_max_diff}"
    );

    // Unpermute with weights
    let mut y_dev = stream
        .alloc_zeros::<f32>((num_tokens * n_dim) as usize)
        .unwrap();
    moe::moe_unpermute(
        &ctx,
        &stream,
        &c_permuted_dev,
        &inverse_dev,
        &weights_dev,
        &mut y_dev,
        num_tokens,
        top_k,
        n_dim,
    )
    .unwrap();

    let y_host = stream.memcpy_dtov(&y_dev).unwrap();

    // Compute expected y: for each (token, k), find which dst it landed at,
    //   y[token] += weight[(token, k)] * c_permuted[dst]
    let mut y_expected = vec![0f32; (num_tokens * n_dim) as usize];
    // Invert inverse_index to get dst for each entry_id
    let mut forward_index = vec![0u32; total_entries as usize];
    for (dst, &entry) in inverse_host.iter().enumerate() {
        forward_index[entry as usize] = dst as u32;
    }
    for entry_id in 0..total_entries as usize {
        let token = entry_id / top_k as usize;
        let wt = weights_f32[entry_id];
        let dst = forward_index[entry_id] as usize;
        for n in 0..n_dim as usize {
            y_expected[token * n_dim as usize + n] += wt * c_permuted_f32[dst * n_dim as usize + n];
        }
    }

    let mut max_y_diff: f32 = 0.0;
    let mut sum_y_diff: f64 = 0.0;
    for (a, b) in y_host.iter().zip(y_expected.iter()) {
        let d = (a - b).abs();
        if d > max_y_diff {
            max_y_diff = d;
        }
        sum_y_diff += d as f64;
    }
    let mean_y_diff = sum_y_diff / y_host.len() as f64;
    eprintln!("unpermute+weight: max_diff={max_y_diff:.6} mean_diff={mean_y_diff:.6}");
    assert!(max_y_diff <= 0.1,
        "unpermute max_diff too high: {max_y_diff} (expected ~0 since we use kernel's own c_permuted)");

    // Also report y vs PyTorch reference (may diff by BF16 GEMM rounding)
    let mut max_ref_diff: f32 = 0.0;
    for (a, b) in y_host.iter().zip(y_ref.iter()) {
        let d = (a - b).abs();
        if d > max_ref_diff {
            max_ref_diff = d;
        }
    }
    eprintln!("end-to-end vs PyTorch: max_diff={max_ref_diff:.4}");
    assert!(
        max_ref_diff <= 3.0,
        "end-to-end diff vs PyTorch too high: {max_ref_diff}"
    );
}
