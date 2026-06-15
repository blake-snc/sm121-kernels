mod common;

use common::load_npz;
use sm121_kernels::{device, sampling};

fn run_topk_test(npz_name: &str) {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    let mut npz = load_npz(npz_name);

    let logits_np: ndarray::Array2<u16> = npz.by_name("logits").unwrap();
    let indices_expected: ndarray::Array1<u32> = npz.by_name("indices").unwrap();
    let values_expected: ndarray::Array1<u16> = npz.by_name("values").unwrap();
    let vocab_size: ndarray::Array0<u32> = npz.by_name("vocab_size").unwrap();
    let k_np: ndarray::Array0<u32> = npz.by_name("k").unwrap();
    let temperature: ndarray::Array0<f32> = npz.by_name("temperature").unwrap();

    let batch_size = logits_np.shape()[0] as u32;
    let vocab_size = vocab_size.into_scalar();
    let k = k_np.into_scalar();
    let temperature = temperature.into_scalar();

    let logits_flat: Vec<u16> = logits_np.into_raw_vec_and_offset().0;
    let expected_indices: Vec<u32> = indices_expected.into_raw_vec_and_offset().0;
    let expected_values: Vec<u16> = values_expected.into_raw_vec_and_offset().0;

    let logits_dev = stream.memcpy_stod(&logits_flat).unwrap();
    let mut indices_dev = stream
        .alloc_zeros::<u32>((batch_size * k) as usize)
        .unwrap();
    let mut values_dev = stream
        .alloc_zeros::<u16>((batch_size * k) as usize)
        .unwrap();

    sampling::topk_sampling(
        &ctx,
        &stream,
        &logits_dev,
        &mut indices_dev,
        &mut values_dev,
        batch_size,
        vocab_size,
        k,
        temperature,
    )
    .unwrap();

    let indices_host = stream.memcpy_dtov(&indices_dev).unwrap();
    let values_host = stream.memcpy_dtov(&values_dev).unwrap();

    for i in 0..batch_size as usize {
        assert_eq!(
            indices_host[i], expected_indices[i],
            "index mismatch at batch {i}: got {} expected {}",
            indices_host[i], expected_indices[i]
        );
    }

    for i in 0..batch_size as usize {
        let actual = half::bf16::from_bits(values_host[i]).to_f32();
        let expected = half::bf16::from_bits(expected_values[i]).to_f32();
        let diff = (actual - expected).abs();
        assert!(
            diff < 0.5,
            "value mismatch at batch {i}: got {actual:.4} expected {expected:.4} diff={diff:.6}"
        );
    }

    eprintln!("topk {npz_name}: all {batch_size} batch elements match (k={k})");
}

#[test]
fn test_topk_sampling_b4_v32000() {
    run_topk_test("topk_sampling_b4_v32000.npz");
}

#[test]
fn test_topk_sampling_b1_v128256() {
    run_topk_test("topk_sampling_b1_v128256.npz");
}

#[test]
fn test_argmax_f32_batched() {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    // Known-pattern logits: argmax is at index (b+1)*37 % vocab for each batch row
    let batch = 4u32;
    let vocab = 32000u32;
    let mut logits = vec![0.0f32; (batch * vocab) as usize];
    let mut expected = vec![0u32; batch as usize];
    for b in 0..batch as usize {
        let target = ((b + 1) * 37) % vocab as usize;
        logits[b * vocab as usize + target] = 999.0;
        expected[b] = target as u32;
    }

    let logits_dev = stream.memcpy_stod(&logits).unwrap();
    let mut tokens_dev = stream.alloc_zeros::<u32>(batch as usize).unwrap();

    sm121_kernels::sampling::argmax_f32(&ctx, &stream, &logits_dev, &mut tokens_dev, batch, vocab)
        .expect("argmax failed");

    let tokens_host = stream.memcpy_dtov(&tokens_dev).unwrap();
    assert_eq!(tokens_host, expected, "argmax output mismatch");
    eprintln!("argmax_f32 batched: ok (4 batches, V=32000)");
}
