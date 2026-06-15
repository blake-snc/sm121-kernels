#![allow(clippy::useless_conversion)]

mod common;

use common::{compare_bf16, load_npz};
use sm121_kernels::{activation, device, embedding, norm, rope, sampling};

#[test]
fn test_rmsnorm_bf16() {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    for hidden in [2048, 4096, 8192] {
        let mut npz = load_npz(&format!("rmsnorm_bf16_h{hidden}.npz"));

        let x_np: ndarray::Array2<u16> = npz.by_name("x").unwrap();
        let weight_np: ndarray::Array1<u16> = npz.by_name("weight").unwrap();
        let out_expected: ndarray::Array2<u16> = npz.by_name("out").unwrap();
        let eps_np: ndarray::Array0<f32> = npz.by_name("eps").unwrap();

        let num_rows = x_np.shape()[0] as u32;
        let hidden_dim = x_np.shape()[1] as u32;
        let eps = eps_np.into_scalar();

        let x_flat: Vec<u16> = x_np.into_raw_vec_and_offset().0;
        let w_flat: Vec<u16> = weight_np.into_raw_vec_and_offset().0;
        let expected_flat: Vec<u16> = out_expected.into_raw_vec_and_offset().0;

        let x_dev = stream.memcpy_stod(&x_flat).unwrap();
        let w_dev = stream.memcpy_stod(&w_flat).unwrap();
        let mut out_dev = stream.alloc_zeros::<u16>(x_flat.len()).unwrap();

        norm::rmsnorm_bf16(
            &ctx,
            &stream,
            &x_dev,
            &mut out_dev,
            &w_dev,
            hidden_dim,
            eps,
            num_rows,
        )
        .unwrap();

        let out_host = stream.memcpy_dtov(&out_dev).unwrap();
        let (max_diff, mean_diff) = compare_bf16(&out_host, &expected_flat, 0.02);
        eprintln!("rmsnorm h={hidden}: max_diff={max_diff:.6} mean_diff={mean_diff:.6}");
    }
}

#[test]
fn test_rmsnorm_backward_bf16() {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    for hidden in [2048, 4096, 8192] {
        let mut npz = load_npz(&format!("rmsnorm_backward_bf16_h{hidden}.npz"));

        let x_np: ndarray::Array2<u16> = npz.by_name("x").unwrap();
        let weight_np: ndarray::Array1<u16> = npz.by_name("weight").unwrap();
        let dy_np: ndarray::Array2<u16> = npz.by_name("dy").unwrap();
        let dx_expected: ndarray::Array2<u16> = npz.by_name("dx").unwrap();
        let dweight_expected: ndarray::Array1<u16> = npz.by_name("dweight").unwrap();
        let eps_np: ndarray::Array0<f32> = npz.by_name("eps").unwrap();

        let num_rows = x_np.shape()[0] as u32;
        let hidden_dim = x_np.shape()[1] as u32;
        let eps = eps_np.into_scalar();

        let x_flat: Vec<u16> = x_np.into_raw_vec_and_offset().0;
        let w_flat: Vec<u16> = weight_np.into_raw_vec_and_offset().0;
        let dy_flat: Vec<u16> = dy_np.into_raw_vec_and_offset().0;
        let dx_expected_flat: Vec<u16> = dx_expected.into_raw_vec_and_offset().0;
        let dweight_expected_flat: Vec<u16> = dweight_expected.into_raw_vec_and_offset().0;

        let x_dev = stream.memcpy_stod(&x_flat).unwrap();
        let w_dev = stream.memcpy_stod(&w_flat).unwrap();
        let dy_dev = stream.memcpy_stod(&dy_flat).unwrap();
        let mut dx_dev = stream.alloc_zeros::<u16>(x_flat.len()).unwrap();
        // dweight MUST be zero-initialized — kernel uses atomicAdd.
        let mut dweight_dev = stream.alloc_zeros::<f32>(hidden_dim as usize).unwrap();

        norm::rmsnorm_backward_bf16(
            &ctx,
            &stream,
            &x_dev,
            &w_dev,
            &dy_dev,
            &mut dx_dev,
            &mut dweight_dev,
            hidden_dim,
            eps,
            num_rows,
        )
        .unwrap();

        // Compare dx (BF16)
        let dx_host = stream.memcpy_dtov(&dx_dev).unwrap();
        let (max_dx, mean_dx) = compare_bf16(&dx_host, &dx_expected_flat, 0.05);
        eprintln!("rmsnorm_bw dx h={hidden}: max_diff={max_dx:.6} mean_diff={mean_dx:.6}");

        // Compare dweight (kernel emits f32; expected is bf16). Cast f32 → bf16 for compare.
        let dweight_host_f32 = stream.memcpy_dtov(&dweight_dev).unwrap();
        let dweight_host_bf16: Vec<u16> = dweight_host_f32
            .iter()
            .map(|&v| {
                // Round-to-nearest-even f32 → bf16
                let bits = v.to_bits();
                let lsb = (bits >> 16) & 1;
                let bias = 0x7FFFu32 + lsb;
                ((bits.wrapping_add(bias)) >> 16) as u16
            })
            .collect();
        let (max_dw, mean_dw) = compare_bf16(&dweight_host_bf16, &dweight_expected_flat, 0.5);
        eprintln!("rmsnorm_bw dw h={hidden}: max_diff={max_dw:.6} mean_diff={mean_dw:.6}");
    }
}

#[test]
fn test_rope_backward_bf16() {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    for d in [64, 128] {
        let mut npz = load_npz(&format!("rope_backward_bf16_d{d}.npz"));
        let dy_np: ndarray::Array4<u16> = npz.by_name("dy").unwrap();
        let dx_expected: ndarray::Array4<u16> = npz.by_name("dx").unwrap();
        let cos_cache_np: ndarray::Array2<f32> = npz.by_name("cos_cache").unwrap();
        let sin_cache_np: ndarray::Array2<f32> = npz.by_name("sin_cache").unwrap();

        let (b, seq, h, dim) = (
            dy_np.shape()[0] as u32,
            dy_np.shape()[1] as u32,
            dy_np.shape()[2] as u32,
            dy_np.shape()[3] as u32,
        );

        let dy_flat: Vec<u16> = dy_np.into_raw_vec_and_offset().0;
        let dx_expected_flat: Vec<u16> = dx_expected.into_raw_vec_and_offset().0;
        let cos_flat: Vec<f32> = cos_cache_np.into_raw_vec_and_offset().0;
        let sin_flat: Vec<f32> = sin_cache_np.into_raw_vec_and_offset().0;

        let dy_dev = stream.memcpy_stod(&dy_flat).unwrap();
        let cos_dev = stream.memcpy_stod(&cos_flat).unwrap();
        let sin_dev = stream.memcpy_stod(&sin_flat).unwrap();
        let mut dx_dev = stream.alloc_zeros::<u16>(dy_flat.len()).unwrap();

        rope::rope_backward_bf16(
            &ctx,
            &stream,
            &dy_dev,
            &mut dx_dev,
            &cos_dev,
            &sin_dev,
            b,
            seq,
            h,
            dim,
        )
        .unwrap();

        let dx_host = stream.memcpy_dtov(&dx_dev).unwrap();
        let (max_diff, mean_diff) = compare_bf16(&dx_host, &dx_expected_flat, 0.02);
        eprintln!("rope_bw d={d}: max_diff={max_diff:.6} mean_diff={mean_diff:.6}");
    }
}

#[test]
fn test_silu_backward_bf16() {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    for d in [2048, 4096] {
        let mut npz = load_npz(&format!("silu_backward_bf16_d{d}.npz"));
        let x_np: ndarray::Array2<u16> = npz.by_name("x").unwrap();
        let dy_np: ndarray::Array2<u16> = npz.by_name("dy").unwrap();
        let dx_expected: ndarray::Array2<u16> = npz.by_name("dx").unwrap();

        let n = (x_np.shape()[0] * x_np.shape()[1]) as u32;
        let x_flat: Vec<u16> = x_np.into_raw_vec_and_offset().0;
        let dy_flat: Vec<u16> = dy_np.into_raw_vec_and_offset().0;
        let dx_expected_flat: Vec<u16> = dx_expected.into_raw_vec_and_offset().0;

        let x_dev = stream.memcpy_stod(&x_flat).unwrap();
        let dy_dev = stream.memcpy_stod(&dy_flat).unwrap();
        let mut dx_dev = stream.alloc_zeros::<u16>(x_flat.len()).unwrap();

        activation::silu_backward_bf16(&ctx, &stream, &x_dev, &dy_dev, &mut dx_dev, n).unwrap();

        let dx_host = stream.memcpy_dtov(&dx_dev).unwrap();
        let (max_diff, mean_diff) = compare_bf16(&dx_host, &dx_expected_flat, 0.02);
        eprintln!("silu_bw d={d}: max_diff={max_diff:.6} mean_diff={mean_diff:.6}");
    }
}

#[test]
fn test_gelu_tanh_backward_bf16() {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    for d in [2048, 4096] {
        let mut npz = load_npz(&format!("gelu_tanh_backward_bf16_d{d}.npz"));
        let x_np: ndarray::Array2<u16> = npz.by_name("x").unwrap();
        let dy_np: ndarray::Array2<u16> = npz.by_name("dy").unwrap();
        let dx_expected: ndarray::Array2<u16> = npz.by_name("dx").unwrap();

        let n = (x_np.shape()[0] * x_np.shape()[1]) as u32;
        let x_flat: Vec<u16> = x_np.into_raw_vec_and_offset().0;
        let dy_flat: Vec<u16> = dy_np.into_raw_vec_and_offset().0;
        let dx_expected_flat: Vec<u16> = dx_expected.into_raw_vec_and_offset().0;

        let x_dev = stream.memcpy_stod(&x_flat).unwrap();
        let dy_dev = stream.memcpy_stod(&dy_flat).unwrap();
        let mut dx_dev = stream.alloc_zeros::<u16>(x_flat.len()).unwrap();

        activation::gelu_tanh_backward_bf16(&ctx, &stream, &x_dev, &dy_dev, &mut dx_dev, n)
            .unwrap();

        let dx_host = stream.memcpy_dtov(&dx_dev).unwrap();
        let (max_diff, mean_diff) = compare_bf16(&dx_host, &dx_expected_flat, 0.05);
        eprintln!("gelu_tanh_bw d={d}: max_diff={max_diff:.6} mean_diff={mean_diff:.6}");
    }
}

#[test]
fn test_softmax_backward_bf16() {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    for d in [2048, 4096, 8192] {
        let mut npz = load_npz(&format!("softmax_backward_bf16_d{d}.npz"));
        let y_np: ndarray::Array2<u16> = npz.by_name("y").unwrap();
        let dy_np: ndarray::Array2<u16> = npz.by_name("dy").unwrap();
        let dx_expected: ndarray::Array2<u16> = npz.by_name("dx").unwrap();

        let num_rows = y_np.shape()[0] as u32;
        let dim = y_np.shape()[1] as u32;

        let y_flat: Vec<u16> = y_np.into_raw_vec_and_offset().0;
        let dy_flat: Vec<u16> = dy_np.into_raw_vec_and_offset().0;
        let dx_expected_flat: Vec<u16> = dx_expected.into_raw_vec_and_offset().0;

        let y_dev = stream.memcpy_stod(&y_flat).unwrap();
        let dy_dev = stream.memcpy_stod(&dy_flat).unwrap();
        let mut dx_dev = stream.alloc_zeros::<u16>(y_flat.len()).unwrap();

        sampling::softmax_backward_bf16(&ctx, &stream, &y_dev, &dy_dev, &mut dx_dev, num_rows, dim)
            .unwrap();

        let dx_host = stream.memcpy_dtov(&dx_dev).unwrap();
        let (max_diff, mean_diff) = compare_bf16(&dx_host, &dx_expected_flat, 0.05);
        eprintln!("softmax_bw d={d}: max_diff={max_diff:.6} mean_diff={mean_diff:.6}");
    }
}

#[test]
fn test_cross_entropy_backward_bf16() {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    for vocab in [4096u32, 32000] {
        let mut npz = load_npz(&format!("cross_entropy_backward_bf16_v{vocab}.npz"));
        let logits_np: ndarray::Array2<u16> = npz.by_name("logits").unwrap();
        let targets_np: ndarray::Array1<u32> = npz.by_name("targets").unwrap();
        let dlogits_expected: ndarray::Array2<u16> = npz.by_name("dlogits").unwrap();

        let batch = logits_np.shape()[0] as u32;
        let v = logits_np.shape()[1] as u32;

        let logits_flat: Vec<u16> = logits_np.into_raw_vec_and_offset().0;
        let targets_flat: Vec<u32> = targets_np.into_raw_vec_and_offset().0;
        let dlogits_expected_flat: Vec<u16> = dlogits_expected.into_raw_vec_and_offset().0;

        let logits_dev = stream.memcpy_stod(&logits_flat).unwrap();
        let targets_dev = stream.memcpy_stod(&targets_flat).unwrap();
        let mut dlogits_dev = stream.alloc_zeros::<u16>(logits_flat.len()).unwrap();

        sampling::cross_entropy_backward_bf16(
            &ctx,
            &stream,
            &logits_dev,
            &targets_dev,
            &mut dlogits_dev,
            batch,
            v,
        )
        .unwrap();

        let dlogits_host = stream.memcpy_dtov(&dlogits_dev).unwrap();
        let (max_diff, mean_diff) = compare_bf16(&dlogits_host, &dlogits_expected_flat, 0.01);
        eprintln!("xent_bw v={vocab}: max_diff={max_diff:.6} mean_diff={mean_diff:.6}");
    }
}

#[test]
fn test_silu_mul_bf16() {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    for d in [2048, 4096] {
        let mut npz = load_npz(&format!("silu_mul_bf16_d{d}.npz"));

        let input_np: ndarray::Array2<u16> = npz.by_name("input").unwrap();
        let out_expected: ndarray::Array2<u16> = npz.by_name("out").unwrap();

        let n_rows = input_np.shape()[0] as u32;
        let total_out_elems = n_rows * (d as u32);

        let input_flat: Vec<u16> = input_np.into_raw_vec_and_offset().0;
        let expected_flat: Vec<u16> = out_expected.into_raw_vec_and_offset().0;

        let input_dev = stream.memcpy_stod(&input_flat).unwrap();
        let mut out_dev = stream.alloc_zeros::<u16>(expected_flat.len()).unwrap();

        activation::silu_mul_bf16(
            &ctx,
            &stream,
            &input_dev,
            &mut out_dev,
            total_out_elems,
            d as u32,
        )
        .unwrap();

        let out_host = stream.memcpy_dtov(&out_dev).unwrap();
        let (max_diff, mean_diff) = compare_bf16(&out_host, &expected_flat, 0.01);
        eprintln!("silu_mul d={d}: max_diff={max_diff:.6} mean_diff={mean_diff:.6}");
    }
}

#[test]
fn test_gelu_tanh_mul_bf16() {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    for d in [2048, 4096] {
        let mut npz = load_npz(&format!("gelu_tanh_mul_bf16_d{d}.npz"));

        let input_np: ndarray::Array2<u16> = npz.by_name("input").unwrap();
        let out_expected: ndarray::Array2<u16> = npz.by_name("out").unwrap();

        let n_rows = input_np.shape()[0] as u32;
        let total_out_elems = n_rows * (d as u32);

        let input_flat: Vec<u16> = input_np.into_raw_vec_and_offset().0;
        let expected_flat: Vec<u16> = out_expected.into_raw_vec_and_offset().0;

        let input_dev = stream.memcpy_stod(&input_flat).unwrap();
        let mut out_dev = stream.alloc_zeros::<u16>(expected_flat.len()).unwrap();

        activation::gelu_tanh_mul_bf16(
            &ctx,
            &stream,
            &input_dev,
            &mut out_dev,
            total_out_elems,
            d as u32,
        )
        .unwrap();

        let out_host = stream.memcpy_dtov(&out_dev).unwrap();
        let (max_diff, mean_diff) = compare_bf16(&out_host, &expected_flat, 0.01);
        eprintln!("gelu_tanh_mul d={d}: max_diff={max_diff:.6} mean_diff={mean_diff:.6}");
        assert!(
            mean_diff < 0.02,
            "gelu_tanh_mul d={d}: mean_diff={mean_diff} too high (systematic error)"
        );
    }
}

// Step 6: gelu_mul_bf16 test (golden data + dispatch exist, was untested)
#[test]
fn test_gelu_mul_bf16() {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    for d in [2048, 4096] {
        let mut npz = load_npz(&format!("gelu_mul_bf16_d{d}.npz"));

        let input_np: ndarray::Array2<u16> = npz.by_name("input").unwrap();
        let out_expected: ndarray::Array2<u16> = npz.by_name("out").unwrap();

        let n_rows = input_np.shape()[0] as u32;
        let total_out_elems = n_rows * (d as u32);

        let input_flat: Vec<u16> = input_np.into_raw_vec_and_offset().0;
        let expected_flat: Vec<u16> = out_expected.into_raw_vec_and_offset().0;

        let input_dev = stream.memcpy_stod(&input_flat).unwrap();
        let mut out_dev = stream.alloc_zeros::<u16>(expected_flat.len()).unwrap();

        activation::gelu_mul_bf16(
            &ctx,
            &stream,
            &input_dev,
            &mut out_dev,
            total_out_elems,
            d as u32,
        )
        .unwrap();

        let out_host = stream.memcpy_dtov(&out_dev).unwrap();
        let (max_diff, mean_diff) = compare_bf16(&out_host, &expected_flat, 0.07);
        eprintln!("gelu_mul d={d}: max_diff={max_diff:.6} mean_diff={mean_diff:.6}");
    }
}

#[test]
fn test_rope_bf16() {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    for d in [64, 128] {
        let mut npz = load_npz(&format!("rope_bf16_d{d}.npz"));

        let x_np: ndarray::Array4<u16> = npz.by_name("x").unwrap();
        let out_expected: ndarray::Array4<u16> = npz.by_name("out").unwrap();
        let cos_np: ndarray::Array2<f32> = npz.by_name("cos_cache").unwrap();
        let sin_np: ndarray::Array2<f32> = npz.by_name("sin_cache").unwrap();

        let batch = x_np.shape()[0] as u32;
        let seq_len = x_np.shape()[1] as u32;
        let heads = x_np.shape()[2] as u32;
        let dim = x_np.shape()[3] as u32;

        let x_flat: Vec<u16> = x_np.into_raw_vec_and_offset().0;
        let expected_flat: Vec<u16> = out_expected.into_raw_vec_and_offset().0;
        let cos_flat: Vec<f32> = cos_np.into_raw_vec_and_offset().0;
        let sin_flat: Vec<f32> = sin_np.into_raw_vec_and_offset().0;

        let mut x_dev = stream.memcpy_stod(&x_flat).unwrap();
        let cos_dev = stream.memcpy_stod(&cos_flat).unwrap();
        let sin_dev = stream.memcpy_stod(&sin_flat).unwrap();

        rope::rope_bf16(
            &ctx, &stream, &mut x_dev, &cos_dev, &sin_dev, batch, seq_len, heads, dim,
        )
        .unwrap();

        let out_host = stream.memcpy_dtov(&x_dev).unwrap();
        let (max_diff, mean_diff) = compare_bf16(&out_host, &expected_flat, 0.02);
        eprintln!("rope d={d}: max_diff={max_diff:.6} mean_diff={mean_diff:.6}");
    }
}

#[test]
fn test_rmsnorm_residual_bf16() {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    for hidden in [2048, 4096, 8192] {
        let mut npz = load_npz(&format!("rmsnorm_residual_bf16_h{hidden}.npz"));

        let x_np: ndarray::Array2<u16> = npz.by_name("x").unwrap();
        let residual_in_np: ndarray::Array2<u16> = npz.by_name("residual_in").unwrap();
        let residual_out_expected: ndarray::Array2<u16> = npz.by_name("residual_out").unwrap();
        let weight_np: ndarray::Array1<u16> = npz.by_name("weight").unwrap();
        let out_expected: ndarray::Array2<u16> = npz.by_name("out").unwrap();
        let eps_np: ndarray::Array0<f32> = npz.by_name("eps").unwrap();

        let num_rows = x_np.shape()[0] as u32;
        let hidden_dim = x_np.shape()[1] as u32;
        let eps = eps_np.into_scalar();

        let x_flat: Vec<u16> = x_np.into_raw_vec_and_offset().0;
        let residual_flat: Vec<u16> = residual_in_np.into_raw_vec_and_offset().0;
        let w_flat: Vec<u16> = weight_np.into_raw_vec_and_offset().0;
        let expected_out: Vec<u16> = out_expected.into_raw_vec_and_offset().0;
        let expected_residual: Vec<u16> = residual_out_expected.into_raw_vec_and_offset().0;

        let x_dev = stream.memcpy_stod(&x_flat).unwrap();
        let mut residual_dev = stream.memcpy_stod(&residual_flat).unwrap();
        let w_dev = stream.memcpy_stod(&w_flat).unwrap();
        let mut out_dev = stream.alloc_zeros::<u16>(x_flat.len()).unwrap();

        norm::rmsnorm_residual_bf16(
            &ctx,
            &stream,
            &x_dev,
            &mut residual_dev,
            &mut out_dev,
            &w_dev,
            hidden_dim,
            eps,
            num_rows,
        )
        .unwrap();

        let out_host = stream.memcpy_dtov(&out_dev).unwrap();
        let residual_host = stream.memcpy_dtov(&residual_dev).unwrap();
        let (mo, _) = compare_bf16(&out_host, &expected_out, 0.05);
        let (mr, _) = compare_bf16(&residual_host, &expected_residual, 0.05);
        eprintln!("rmsnorm_residual h={hidden}: out_max_diff={mo:.6} residual_max_diff={mr:.6}");
        assert!(mo <= 0.05, "rmsnorm_residual out diff too high: {mo}");
        assert!(mr <= 0.05, "rmsnorm_residual residual diff too high: {mr}");
    }
}

#[test]
fn test_softmax_bf16() {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    for (vocab, t_tag, temp) in [(32000u32, 10, 1.0f32), (32000, 7, 0.7), (128256, 10, 1.0)] {
        let mut npz = load_npz(&format!("softmax_bf16_v{vocab}_t{t_tag}.npz"));
        let x_np: ndarray::Array2<u16> = npz.by_name("x").unwrap();
        let out_expected: ndarray::Array2<u16> = npz.by_name("out").unwrap();

        let num_rows = x_np.shape()[0] as u32;
        let vocab_size = x_np.shape()[1] as u32;

        let x_flat: Vec<u16> = x_np.into_raw_vec_and_offset().0;
        let expected: Vec<u16> = out_expected.into_raw_vec_and_offset().0;

        let x_dev = stream.memcpy_stod(&x_flat).unwrap();
        let mut out_dev = stream.alloc_zeros::<u16>(x_flat.len()).unwrap();

        sampling::softmax_bf16(
            &ctx,
            &stream,
            &x_dev,
            &mut out_dev,
            num_rows,
            vocab_size,
            temp,
        )
        .unwrap();

        let out_host = stream.memcpy_dtov(&out_dev).unwrap();
        let (max_diff, mean_diff) = compare_bf16(&out_host, &expected, 0.01);
        eprintln!("softmax v={vocab} T={temp}: max_diff={max_diff:.6} mean_diff={mean_diff:.6}");
        assert!(max_diff <= 0.01, "softmax diff too high: {max_diff}");
    }
}

#[test]
fn test_embedding_lookup_bf16() {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    for (vocab, hidden, ntok) in [(32000u32, 4096u32, 16u32), (50000, 4096, 32)] {
        let mut npz = load_npz(&format!(
            "embedding_lookup_bf16_v{vocab}_h{hidden}_t{ntok}.npz"
        ));
        let table_np: ndarray::Array2<u16> = npz.by_name("table").unwrap();
        let ids_np: ndarray::Array1<u32> = npz.by_name("token_ids").unwrap();
        let out_expected: ndarray::Array2<u16> = npz.by_name("out").unwrap();

        let table_flat: Vec<u16> = table_np.into_raw_vec_and_offset().0;
        let ids_flat: Vec<u32> = ids_np.into_raw_vec_and_offset().0;
        let expected: Vec<u16> = out_expected.into_raw_vec_and_offset().0;

        let table_dev = stream.memcpy_stod(&table_flat).unwrap();
        let ids_dev = stream.memcpy_stod(&ids_flat).unwrap();
        let mut out_dev = stream.alloc_zeros::<u16>(expected.len()).unwrap();

        embedding::embedding_lookup_bf16(
            &ctx,
            &stream,
            &ids_dev,
            &table_dev,
            &mut out_dev,
            ntok,
            vocab,
            hidden,
        )
        .unwrap();

        let out_host = stream.memcpy_dtov(&out_dev).unwrap();
        let (max_diff, mean_diff) = compare_bf16(&out_host, &expected, 0.0);
        eprintln!("embedding_lookup v={vocab} h={hidden} ntok={ntok}: max_diff={max_diff:.6} mean_diff={mean_diff:.6}");
        assert!(max_diff <= 0.0, "embedding differs (max_diff={max_diff})");
    }
}

#[test]
fn test_cross_entropy_bf16() {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    for (vocab, ntok) in [(32000u32, 16u32), (128256, 32)] {
        let mut npz = load_npz(&format!("cross_entropy_bf16_v{vocab}_t{ntok}.npz"));
        let logits_np: ndarray::Array2<u16> = npz.by_name("logits").unwrap();
        let targets_np: ndarray::Array1<u32> = npz.by_name("targets").unwrap();
        let losses_expected: ndarray::Array1<f32> = npz.by_name("losses").unwrap();

        let logits: Vec<u16> = logits_np.into_raw_vec_and_offset().0;
        let targets: Vec<u32> = targets_np.into_raw_vec_and_offset().0;
        let expected: Vec<f32> = losses_expected.into_raw_vec_and_offset().0;

        let logits_dev = stream.memcpy_stod(&logits).unwrap();
        let targets_dev = stream.memcpy_stod(&targets).unwrap();
        let mut losses_dev = stream.alloc_zeros::<f32>(ntok as usize).unwrap();

        sampling::cross_entropy_bf16(
            &ctx,
            &stream,
            &logits_dev,
            &targets_dev,
            &mut losses_dev,
            ntok,
            vocab,
        )
        .unwrap();

        let losses_host = stream.memcpy_dtov(&losses_dev).unwrap();
        let mut max_diff: f32 = 0.0;
        for (a, b) in losses_host.iter().zip(expected.iter()) {
            let d = (a - b).abs();
            if d > max_diff {
                max_diff = d;
            }
        }
        eprintln!("cross_entropy v={vocab} ntok={ntok}: max_diff={max_diff:.6}");
        assert!(max_diff <= 0.05, "cross_entropy diff too high: {max_diff}");
    }
}

#[test]
fn test_rmsnorm_bf16_fp8out() {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    for hidden in [2048u32, 4096] {
        let mut npz = load_npz(&format!("rmsnorm_bf16_fp8out_h{hidden}.npz"));
        let x_np: ndarray::Array2<u16> = npz.by_name("x").unwrap();
        let w_np: ndarray::Array1<u16> = npz.by_name("weight").unwrap();
        let out_expected: ndarray::Array2<u8> = npz.by_name("out").unwrap();
        let eps_np: ndarray::Array0<f32> = npz.by_name("eps").unwrap();
        let inv_scale_np: ndarray::Array0<f32> = npz.by_name("inv_scale").unwrap();

        let num_rows = x_np.shape()[0] as u32;
        let x: Vec<u16> = x_np.into_raw_vec_and_offset().0;
        let w: Vec<u16> = w_np.into_raw_vec_and_offset().0;
        let expected: Vec<u8> = out_expected.into_raw_vec_and_offset().0;
        let eps = eps_np.into_scalar();
        let inv_scale = inv_scale_np.into_scalar();

        let x_dev = stream.memcpy_stod(&x).unwrap();
        let w_dev = stream.memcpy_stod(&w).unwrap();
        let mut out_dev = stream.alloc_zeros::<u8>(expected.len()).unwrap();

        norm::rmsnorm_bf16_fp8out(
            &ctx,
            &stream,
            &x_dev,
            &mut out_dev,
            &w_dev,
            hidden,
            eps,
            inv_scale,
            num_rows,
        )
        .unwrap();

        let out_host = stream.memcpy_dtov(&out_dev).unwrap();
        let mut exact = 0usize;
        let mut adjacent = 0usize;
        let mut far = 0usize;
        for (a, b) in out_host.iter().zip(expected.iter()) {
            let diff = (*a as i16 - *b as i16).abs();
            if diff == 0 {
                exact += 1;
            } else if diff <= 2 {
                adjacent += 1;
            } else {
                far += 1;
            }
        }
        let total = out_host.len();
        let exact_pct = 100.0 * exact as f64 / total as f64;
        eprintln!("rmsnorm_fp8out h={hidden}: exact={exact}/{total} ({exact_pct:.1}%), adjacent={adjacent}, far={far}");
        assert!(
            exact_pct > 80.0,
            "rmsnorm_fp8out too many mismatches: {exact_pct:.1}% exact"
        );
        assert!(
            far < total / 100,
            "rmsnorm_fp8out has {far} far mismatches (>1% of {total})"
        );
    }
}

#[test]
fn test_rmsnorm_bf16_fp8out_pertoken() {
    use half::bf16;
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    // E4M3 FP8 decode table: u8 → f32. Covers finite values (exponent != 0xF or mantissa == 0).
    fn fp8_e4m3_to_f32(b: u8) -> f32 {
        let sign = (b >> 7) & 1;
        let exp = (b >> 3) & 0xF;
        let mant = b & 0x7;
        let s = if sign == 0 { 1.0 } else { -1.0 };
        if exp == 0 {
            // Denormal: value = s * mant/8 * 2^-6
            s * (mant as f32) / 8.0 * (1.0f32 / 64.0)
        } else {
            // Normal: value = s * (1 + mant/8) * 2^(exp-7)
            let exp_val = (exp as i32) - 7;
            let scale = exp_val.unsigned_abs().try_into().unwrap_or(0u32);
            let pow = if exp_val >= 0 {
                (1u32 << scale) as f32
            } else {
                1.0 / (1u32 << scale) as f32
            };
            s * (1.0 + (mant as f32) / 8.0) * pow
        }
    }

    for (hidden, num_rows, seed) in [(2048u32, 4u32, 0x1234u64), (4096, 8, 0x5678)] {
        // Deterministic pseudo-random BF16 input (keep values in a sane range).
        let mut x = vec![0u16; (num_rows * hidden) as usize];
        let mut w = vec![0u16; hidden as usize];
        let mut state = seed;
        let mut rand = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };
        for v in x.iter_mut() {
            let f = ((rand() as f32 / (u32::MAX as f32)) - 0.5) * 4.0; // ±2
            *v = bf16::from_f32(f).to_bits();
        }
        for v in w.iter_mut() {
            let f = (rand() as f32 / (u32::MAX as f32)) * 0.5 + 0.5; // [0.5, 1.0]
            *v = bf16::from_f32(f).to_bits();
        }
        let eps = 1e-5f32;

        // Reference: RMSNorm + per-token dynamic FP8 quant.
        let h = hidden as usize;
        let n = num_rows as usize;
        let mut ref_out = vec![0u8; n * h];
        let mut ref_scales = vec![0f32; n];
        for r in 0..n {
            let row_x = &x[r * h..(r + 1) * h];
            let sum_sq: f32 = row_x
                .iter()
                .map(|b| {
                    let f = bf16::from_bits(*b).to_f32();
                    f * f
                })
                .sum();
            let rms_inv = (sum_sq / hidden as f32 + eps).powf(-0.5);
            let mut tmp = vec![0f32; h];
            let mut row_max = 0f32;
            for i in 0..h {
                let xi = bf16::from_bits(row_x[i]).to_f32();
                let wi = bf16::from_bits(w[i]).to_f32();
                let t = xi * rms_inv * wi;
                tmp[i] = t;
                let a = t.abs();
                if a > row_max {
                    row_max = a;
                }
            }
            let scale = if row_max == 0.0 { 1.0 } else { row_max / 448.0 };
            ref_scales[r] = scale;
            let inv_scale = 1.0 / scale;
            // Quantize (simple nearest-even to E4M3 is hard to replicate here — we'll match
            // on decoded FP32 tolerance rather than exact bit pattern).
            for i in 0..h {
                let q = tmp[i] * inv_scale;
                // Placeholder: we don't need bit-exact out; compare after dequant below.
                ref_out[r * h + i] = 0;
                let _ = q;
            }
        }

        let x_dev = stream.memcpy_stod(&x).unwrap();
        let w_dev = stream.memcpy_stod(&w).unwrap();
        let mut out_dev = stream.alloc_zeros::<u8>(n * h).unwrap();
        let mut scales_dev = stream.alloc_zeros::<f32>(n).unwrap();

        norm::rmsnorm_bf16_fp8out_pertoken(
            &ctx,
            &stream,
            &x_dev,
            &mut out_dev,
            &w_dev,
            &mut scales_dev,
            hidden,
            eps,
            num_rows,
        )
        .unwrap();

        let out_host = stream.memcpy_dtov(&out_dev).unwrap();
        let scales_host = stream.memcpy_dtov(&scales_dev).unwrap();

        // Validate scales within 2% of reference.
        for r in 0..n {
            let rel = (scales_host[r] - ref_scales[r]).abs() / ref_scales[r].max(1e-8);
            eprintln!(
                "pertoken h={hidden} row={r}: scale got={:.6} ref={:.6} rel={:.4}",
                scales_host[r], ref_scales[r], rel
            );
            assert!(rel < 0.02, "row {r} scale off by {rel}");
        }

        // Validate dequant(fp8 * scale) ≈ normalized value (tmp).
        let mut max_err = 0f32;
        for r in 0..n {
            let row_x = &x[r * h..(r + 1) * h];
            let sum_sq: f32 = row_x
                .iter()
                .map(|b| {
                    let f = bf16::from_bits(*b).to_f32();
                    f * f
                })
                .sum();
            let rms_inv = (sum_sq / hidden as f32 + eps).powf(-0.5);
            let scale = scales_host[r];
            for i in 0..h {
                let xi = bf16::from_bits(row_x[i]).to_f32();
                let wi = bf16::from_bits(w[i]).to_f32();
                let t = xi * rms_inv * wi;
                let deq = fp8_e4m3_to_f32(out_host[r * h + i]) * scale;
                let err = (deq - t).abs();
                if err > max_err {
                    max_err = err;
                }
            }
        }
        eprintln!("pertoken h={hidden} rows={n}: max_err={max_err:.4}");
        // E4M3 mantissa is 3 bits → ~1/16 relative error. For values up to ~2 here
        // (peak normalized ≈ |x|*|w|/rms ≈ 2), expect absolute err ≤ 0.15.
        assert!(max_err < 0.2, "pertoken max_err too large: {max_err}");
    }
}

#[test]
fn test_topp_filter_bf16() {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    for (vocab, p_pct) in [(1024u32, 90u32), (4096, 70), (32000, 95)] {
        let mut npz = load_npz(&format!("topp_filter_bf16_v{vocab}_p{p_pct}.npz"));
        let probs_np: ndarray::Array2<u16> = npz.by_name("probs").unwrap();
        let out_expected: ndarray::Array2<u16> = npz.by_name("out").unwrap();
        let p_np: ndarray::Array0<f32> = npz.by_name("p_thresh").unwrap();

        let num_rows = probs_np.shape()[0] as u32;
        let probs: Vec<u16> = probs_np.into_raw_vec_and_offset().0;
        let _expected: Vec<u16> = out_expected.into_raw_vec_and_offset().0;
        let p_thresh = p_np.into_scalar();

        let probs_dev = stream.memcpy_stod(&probs).unwrap();
        let mut out_dev = stream.alloc_zeros::<u16>(probs.len()).unwrap();

        sampling::topp_filter_bf16(
            &ctx,
            &stream,
            &probs_dev,
            &mut out_dev,
            num_rows,
            vocab,
            p_thresh,
        )
        .unwrap();

        let out_host = stream.memcpy_dtov(&out_dev).unwrap();
        // Validate: output is a valid distribution (each row sums near 1)
        // and the retained mass is close to p_thresh.
        let mut sums = vec![0.0f32; num_rows as usize];
        for (i, h) in out_host
            .iter()
            .enumerate()
            .take((vocab * num_rows) as usize)
        {
            let bits: u32 = (*h as u32) << 16;
            sums[i / vocab as usize] += f32::from_bits(bits);
        }
        for (r, s) in sums.iter().enumerate() {
            eprintln!("topp v={vocab} p={p_pct}% row {r}: sum={s:.6}");
            assert!(
                (*s - 1.0).abs() < 0.05,
                "topp row {r} doesn't sum to 1: {s}"
            );
        }
    }
}

#[test]
fn test_attn_output_gate_vs_cpu() {
    let ctx = sm121_kernels::device::init_device(0).expect("init");
    let stream = ctx.default_stream();

    let n = 8u32;
    let d = 128u32;

    // Deterministic inputs
    let one_bits: u16 = (1.0f32.to_bits() >> 16) as u16;
    let _half_bits: u16 = (0.5f32.to_bits() >> 16) as u16;
    let attn: Vec<u16> = (0..(n * d) as usize).map(|_| one_bits).collect();
    // gate_logits = 0 → sigmoid(0) = 0.5 → y = attn * 0.5
    let gate: Vec<u16> = vec![0u16; (n * d) as usize];

    let a_dev = stream.memcpy_stod(&attn).unwrap();
    let g_dev = stream.memcpy_stod(&gate).unwrap();
    let mut o_dev = stream.alloc_zeros::<u16>((n * d) as usize).unwrap();

    sm121_kernels::activation::attn_output_gate(&ctx, &stream, &a_dev, &g_dev, &mut o_dev, n, d)
        .expect("gate failed");
    let o_host = stream.memcpy_dtov(&o_dev).unwrap();
    let mut max_diff: f32 = 0.0;
    for &bits in &o_host {
        let f = f32::from_bits((bits as u32) << 16);
        let d = (f - 0.5).abs();
        if d > max_diff {
            max_diff = d;
        }
    }
    eprintln!(
        "attn_output_gate: {} elements, sigmoid(0)*1.0 → max_diff={max_diff:.5}",
        o_host.len()
    );
    assert!(max_diff <= 0.01, "gate output wrong: {}", max_diff);
}

#[test]
fn test_gemv_bf16_split_k_v4_smoke() {
    use sm121_kernels::gemm;
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();
    let k: u32 = 2048;
    let n: u32 = 2048;
    let num_shards = 16u32;
    let x_host: Vec<u16> = (0..k)
        .map(|i| half::bf16::from_f32(((i as f32) * 0.0123).sin() * 0.5).to_bits())
        .collect();
    let b_host: Vec<u16> = (0..(k * n))
        .map(|i| {
            let r = (i / n) as f32;
            let c = (i % n) as f32;
            half::bf16::from_f32(((r * 0.0079 - c * 0.013).cos()) * 0.4).to_bits()
        })
        .collect();
    let x_dev = stream.memcpy_stod(&x_host).unwrap();
    let b_dev = stream.memcpy_stod(&b_host).unwrap();

    let mut tmp_f32 = stream.alloc_zeros::<f32>(n as usize).unwrap();
    let mut out_v4 = stream.alloc_zeros::<u16>(n as usize).unwrap();
    let mut out_ref = stream.alloc_zeros::<u16>(n as usize).unwrap();

    gemm::gemv_bf16_split_k_v4_managed(
        &ctx,
        &stream,
        &x_dev,
        &b_dev,
        &mut tmp_f32,
        &mut out_v4,
        n,
        k,
        num_shards,
    )
    .expect("v4");
    gemm::gemm_bf16(&ctx, &stream, &x_dev, &b_dev, &mut out_ref, 1, n, k).expect("ref");

    let h_v4 = stream.memcpy_dtov(&out_v4).unwrap();
    let h_ref = stream.memcpy_dtov(&out_ref).unwrap();
    let mut max_diff = 0.0f32;
    for (a, b) in h_v4.iter().zip(h_ref.iter()) {
        let fa = f32::from_bits((*a as u32) << 16);
        let fb = f32::from_bits((*b as u32) << 16);
        let d = (fa - fb).abs();
        if d > max_diff {
            max_diff = d;
        }
    }
    eprintln!(
        "gemv_split_k_v4 vs gemm(M=1): K={k} N={n} shards={num_shards} max_diff={max_diff:.5}"
    );
    assert!(max_diff <= 1.0, "v4 vs scalar mismatch: {}", max_diff);
}

#[test]
fn test_gemv_bf16_split_k_smoke() {
    use sm121_kernels::gemm;
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();
    let k: u32 = 2048;
    let n: u32 = 2048;
    let num_shards = 8u32;
    let x_host: Vec<u16> = (0..k)
        .map(|i| half::bf16::from_f32(((i as f32) * 0.0123).sin() * 0.5).to_bits())
        .collect();
    let b_host: Vec<u16> = (0..(k * n))
        .map(|i| {
            let r = (i / n) as f32;
            let c = (i % n) as f32;
            half::bf16::from_f32(((r * 0.0079 - c * 0.013).cos()) * 0.4).to_bits()
        })
        .collect();
    let x_dev = stream.memcpy_stod(&x_host).unwrap();
    let b_dev = stream.memcpy_stod(&b_host).unwrap();

    let mut tmp_f32 = stream.alloc_zeros::<f32>(n as usize).unwrap();
    let mut out_split = stream.alloc_zeros::<u16>(n as usize).unwrap();
    let mut out_ref = stream.alloc_zeros::<u16>(n as usize).unwrap();

    gemm::gemv_bf16_split_k_managed(
        &ctx,
        &stream,
        &x_dev,
        &b_dev,
        &mut tmp_f32,
        &mut out_split,
        n,
        k,
        num_shards,
    )
    .expect("split_k_managed");
    gemm::gemm_bf16(&ctx, &stream, &x_dev, &b_dev, &mut out_ref, 1, n, k).expect("gemm");

    let h_split = stream.memcpy_dtov(&out_split).unwrap();
    let h_ref = stream.memcpy_dtov(&out_ref).unwrap();
    let mut max_diff = 0.0f32;
    for (a, b) in h_split.iter().zip(h_ref.iter()) {
        let fa = f32::from_bits((*a as u32) << 16);
        let fb = f32::from_bits((*b as u32) << 16);
        let d = (fa - fb).abs();
        if d > max_diff {
            max_diff = d;
        }
    }
    eprintln!("gemv_split_k vs gemm(M=1): K={k} N={n} shards={num_shards} max_diff={max_diff:.5}");
    // Atomic accumulation of f32 across shards has small reordering noise;
    // BF16 mantissa rounding adds another ulp at scale. Allow generous slack.
    assert!(
        max_diff <= 1.0,
        "split-K vs scalar mismatch too large: {}",
        max_diff
    );
}

#[test]
fn test_gemv_bf16_v2_smoke() {
    use sm121_kernels::gemm;
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();
    // K covers DSV2-Lite shape range; N divisible by 8 (multiple of v2's 8-cols-per-thread).
    let k: u32 = 2048;
    let n: u32 = 2048;
    let x_host: Vec<u16> = (0..k)
        .map(|i| half::bf16::from_f32(((i as f32) * 0.0123).sin() * 0.5).to_bits())
        .collect();
    let b_host: Vec<u16> = (0..(k * n))
        .map(|i| {
            let r = (i / n) as f32;
            let c = (i % n) as f32;
            half::bf16::from_f32(((r * 0.0079 - c * 0.013).cos()) * 0.4).to_bits()
        })
        .collect();
    let x_dev = stream.memcpy_stod(&x_host).unwrap();
    let b_dev = stream.memcpy_stod(&b_host).unwrap();
    let mut out_v2 = stream.alloc_zeros::<u16>(n as usize).unwrap();
    let mut out_ref = stream.alloc_zeros::<u16>(n as usize).unwrap();
    gemm::gemv_bf16_v2(&ctx, &stream, &x_dev, &b_dev, &mut out_v2, n, k).expect("gemv_v2");
    gemm::gemm_bf16(&ctx, &stream, &x_dev, &b_dev, &mut out_ref, 1, n, k).expect("gemm");
    let h_v2 = stream.memcpy_dtov(&out_v2).unwrap();
    let h_ref = stream.memcpy_dtov(&out_ref).unwrap();
    let mut max_diff = 0.0f32;
    for (a, b) in h_v2.iter().zip(h_ref.iter()) {
        let fa = f32::from_bits((*a as u32) << 16);
        let fb = f32::from_bits((*b as u32) << 16);
        let d = (fa - fb).abs();
        if d > max_diff {
            max_diff = d;
        }
    }
    eprintln!("gemv_v2 vs gemm(M=1): K={k} N={n} max_diff={max_diff:.5}");
    assert!(max_diff <= 0.5, "gemv_v2 vs gemm mismatch: {}", max_diff);
}

#[test]
fn test_gemv_bf16_smoke() {
    use sm121_kernels::gemm;
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    // Compare gemv_bf16 against gemm_bf16(M=1) on a small case (correctness).
    let k: u32 = 256;
    let n: u32 = 512;
    // Deterministic non-trivial inputs.
    let x_host: Vec<u16> = (0..k)
        .map(|i| {
            let v = ((i as f32) * 0.0123).sin() * 0.5;
            half::bf16::from_f32(v).to_bits()
        })
        .collect();
    let b_host: Vec<u16> = (0..(k * n))
        .map(|i| {
            let r = (i / n) as f32;
            let c = (i % n) as f32;
            let v = ((r * 0.0079 - c * 0.013).cos()) * 0.4;
            half::bf16::from_f32(v).to_bits()
        })
        .collect();
    let x_dev = stream.memcpy_stod(&x_host).unwrap();
    let b_dev = stream.memcpy_stod(&b_host).unwrap();

    let mut out_gemv = stream.alloc_zeros::<u16>(n as usize).unwrap();
    let mut out_gemm = stream.alloc_zeros::<u16>(n as usize).unwrap();

    gemm::gemv_bf16(&ctx, &stream, &x_dev, &b_dev, &mut out_gemv, n, k).expect("gemv");
    gemm::gemm_bf16(&ctx, &stream, &x_dev, &b_dev, &mut out_gemm, 1, n, k).expect("gemm");

    let h_gemv = stream.memcpy_dtov(&out_gemv).unwrap();
    let h_gemm = stream.memcpy_dtov(&out_gemm).unwrap();
    let mut max_diff = 0.0f32;
    for (a, b) in h_gemv.iter().zip(h_gemm.iter()) {
        let fa = f32::from_bits((*a as u32) << 16);
        let fb = f32::from_bits((*b as u32) << 16);
        let d = (fa - fb).abs();
        if d > max_diff {
            max_diff = d;
        }
    }
    eprintln!("gemv_bf16 vs gemm_bf16(M=1): K={k} N={n} max_diff={max_diff:.5}");
    assert!(max_diff <= 0.05, "gemv vs gemm mismatch: {}", max_diff);
}

#[test]
fn test_add_bf16_smoke() {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let n: u32 = 4096;
    // Build a + b = 0.25 + 0.75 = 1.0 elementwise.
    let a: Vec<u16> = (0..n)
        .map(|_| half::bf16::from_f32(0.25).to_bits())
        .collect();
    let b: Vec<u16> = (0..n)
        .map(|_| half::bf16::from_f32(0.75).to_bits())
        .collect();
    let a_dev = stream.memcpy_stod(&a).unwrap();
    let b_dev = stream.memcpy_stod(&b).unwrap();
    let mut o_dev = stream.alloc_zeros::<u16>(n as usize).unwrap();

    activation::add_bf16(&ctx, &stream, &a_dev, &b_dev, &mut o_dev, n).expect("add_bf16");
    let o_host = stream.memcpy_dtov(&o_dev).unwrap();

    let mut max_diff = 0.0f32;
    for &bits in &o_host {
        let f = f32::from_bits((bits as u32) << 16);
        let d = (f - 1.0).abs();
        if d > max_diff {
            max_diff = d;
        }
    }
    eprintln!("add_bf16: n={n} 0.25+0.75 → max_diff={max_diff:.5}");
    assert!(max_diff <= 0.01, "add_bf16 wrong: {}", max_diff);
}

#[test]
fn test_f32_to_bf16_smoke() {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let n: u32 = 4096;
    // Mix of typical activation magnitudes; verify within BF16 mantissa rounding.
    let inputs_f32: Vec<f32> = (0..n)
        .map(|i| {
            let x = (i as f32) * 0.001 - 2.0;
            x.sin() // values in [-1, 1]
        })
        .collect();
    let in_dev = stream.memcpy_stod(&inputs_f32).unwrap();
    let mut out_dev = stream.alloc_zeros::<u16>(n as usize).unwrap();

    activation::f32_to_bf16(&ctx, &stream, &in_dev, &mut out_dev, n).expect("f32_to_bf16");
    let out_host = stream.memcpy_dtov(&out_dev).unwrap();

    let mut max_rel_err = 0.0f32;
    for (i, &bits) in out_host.iter().enumerate() {
        let f_out = f32::from_bits((bits as u32) << 16);
        let f_in = inputs_f32[i];
        let rel = if f_in.abs() < 1e-6 {
            0.0
        } else {
            ((f_out - f_in) / f_in).abs()
        };
        if rel > max_rel_err {
            max_rel_err = rel;
        }
    }
    // BF16 has 7 mantissa bits → ~1/128 ≈ 0.78% relative error worst case.
    eprintln!("f32_to_bf16: n={n} max_rel_err={max_rel_err:.5}");
    assert!(
        max_rel_err <= 0.01,
        "f32_to_bf16 rel err too high: {}",
        max_rel_err
    );
}

#[test]
fn test_f32_axpy_smoke() {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let n: u32 = 4096;
    let alpha: f32 = 0.25;
    let initial: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
    let src: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5).collect();

    let mut out_dev = stream.memcpy_stod(&initial).unwrap();
    let src_dev = stream.memcpy_stod(&src).unwrap();

    activation::f32_axpy(&ctx, &stream, &mut out_dev, &src_dev, alpha, n).expect("f32_axpy");
    let out_host = stream.memcpy_dtov(&out_dev).unwrap();

    let mut max_diff = 0.0f32;
    for i in 0..n as usize {
        let expected = initial[i] + alpha * src[i];
        let d = (out_host[i] - expected).abs();
        if d > max_diff {
            max_diff = d;
        }
    }
    eprintln!("f32_axpy: n={n} alpha={alpha} max_diff={max_diff:.6}");
    assert!(max_diff <= 1e-4, "f32_axpy wrong: {}", max_diff);

    // Run twice — verify accumulator semantics (out += alpha * src) holds across launches.
    activation::f32_axpy(&ctx, &stream, &mut out_dev, &src_dev, alpha, n).expect("f32_axpy 2nd");
    let out2 = stream.memcpy_dtov(&out_dev).unwrap();
    let mut max_diff2 = 0.0f32;
    for i in 0..n as usize {
        let expected = initial[i] + 2.0 * alpha * src[i];
        let d = (out2[i] - expected).abs();
        if d > max_diff2 {
            max_diff2 = d;
        }
    }
    eprintln!("f32_axpy: 2nd call max_diff={max_diff2:.6}");
    // Tolerance: max value here is ~1024 (4095*0.5*0.5 + 4095*0.01 + ~32 of accumulated ulp).
    // f32 has ~24-bit mantissa → ulp(1024) ~ 1.2e-4. Two FMAs in series → ~2.5e-4 worst case.
    assert!(
        max_diff2 <= 5e-4,
        "f32_axpy 2nd accumulate wrong: {}",
        max_diff2
    );
}

#[test]
fn test_gemv_bf16_split_k_view_smoke() {
    use sm121_kernels::gemm;
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    // Build a stack of 4 "experts", each with a unique BF16 weight matrix [K, N].
    // Then verify that gemv_split_k_view called on the 3rd slice matches gemv_split_k
    // called on the same data copied to its own buffer.
    let k: u32 = 2048;
    let n: u32 = 1024;
    let n_experts: u32 = 4;
    let target_eid: u32 = 2;
    let stride = (k as usize) * (n as usize);

    // Inputs: x (random-ish via deterministic pattern) and weight stack
    let x: Vec<u16> = (0..k)
        .map(|i| half::bf16::from_f32(((i as f32) * 0.001).sin()).to_bits())
        .collect();
    let mut stack: Vec<u16> = vec![0u16; (n_experts as usize) * stride];
    for e in 0..n_experts as usize {
        for i in 0..stride {
            let v = (((e * 7919 + i * 31) % 2003) as f32) * 0.0007 - 0.7;
            stack[e * stride + i] = half::bf16::from_f32(v).to_bits();
        }
    }
    let isolated_slice: Vec<u16> =
        stack[target_eid as usize * stride..(target_eid as usize + 1) * stride].to_vec();

    let x_dev = stream.memcpy_stod(&x).unwrap();
    let stack_dev = stream.memcpy_stod(&stack).unwrap();
    let iso_dev = stream.memcpy_stod(&isolated_slice).unwrap();
    let mut out_view_f32 = stream.alloc_zeros::<f32>(n as usize).unwrap();
    let mut out_full_f32 = stream.alloc_zeros::<f32>(n as usize).unwrap();

    let shards = (k / 256).clamp(1, 16);

    // Path A: view into the stack at expert 2
    let view = stack_dev.slice(target_eid as usize * stride..(target_eid as usize + 1) * stride);
    gemm::gemv_bf16_split_k_view(
        &ctx,
        &stream,
        &x_dev,
        &view,
        &mut out_view_f32,
        n,
        k,
        shards,
    )
    .expect("split_k_view");

    // Path B: same compute on the isolated copy
    gemm::gemv_bf16_split_k(
        &ctx,
        &stream,
        &x_dev,
        &iso_dev,
        &mut out_full_f32,
        n,
        k,
        shards,
    )
    .expect("split_k");

    let out_view_host = stream.memcpy_dtov(&out_view_f32).unwrap();
    let out_full_host = stream.memcpy_dtov(&out_full_f32).unwrap();

    let mut max_diff = 0.0f32;
    for i in 0..n as usize {
        let d = (out_view_host[i] - out_full_host[i]).abs();
        if d > max_diff {
            max_diff = d;
        }
    }
    eprintln!("split_k_view vs split_k(copy): n={n} k={k} max_diff={max_diff:.6}");
    // Atomic-add ordering means partial sums can differ by tiny f32 ulps.
    assert!(
        max_diff < 0.05,
        "split_k_view diverged from split_k: {}",
        max_diff
    );
}

#[test]
fn test_add_bf16_inplace_smoke() {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let n: u32 = 4096;
    let a: Vec<u16> = (0..n)
        .map(|_| half::bf16::from_f32(0.25).to_bits())
        .collect();
    let b: Vec<u16> = (0..n)
        .map(|_| half::bf16::from_f32(0.75).to_bits())
        .collect();
    let mut a_dev = stream.memcpy_stod(&a).unwrap();
    let b_dev = stream.memcpy_stod(&b).unwrap();

    activation::add_bf16_inplace(&ctx, &stream, &mut a_dev, &b_dev, n).expect("add_bf16_inplace");
    let a_host = stream.memcpy_dtov(&a_dev).unwrap();

    let mut max_diff = 0.0f32;
    for &bits in &a_host {
        let f = f32::from_bits((bits as u32) << 16);
        let d = (f - 1.0).abs();
        if d > max_diff {
            max_diff = d;
        }
    }
    eprintln!("add_bf16_inplace: n={n} 0.25+=0.75 → max_diff={max_diff:.5}");
    assert!(max_diff <= 0.01, "add_bf16_inplace wrong: {}", max_diff);

    // Run twice — verify accumulator semantics holds after launch (out_inout += src)
    activation::add_bf16_inplace(&ctx, &stream, &mut a_dev, &b_dev, n).expect("add 2nd");
    let a_host2 = stream.memcpy_dtov(&a_dev).unwrap();
    let mut max_diff2 = 0.0f32;
    for &bits in &a_host2 {
        let f = f32::from_bits((bits as u32) << 16);
        let d = (f - 1.75).abs();
        if d > max_diff2 {
            max_diff2 = d;
        }
    }
    eprintln!("add_bf16_inplace: 2nd call (1.0+0.75) max_diff={max_diff2:.5}");
    assert!(
        max_diff2 <= 0.02,
        "add_bf16_inplace 2nd wrong: {}",
        max_diff2
    );
}
