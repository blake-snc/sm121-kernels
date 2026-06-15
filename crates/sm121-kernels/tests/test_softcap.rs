//! Validate softcap_bf16 against Python reference: cap * tanh(x / cap).

use half::bf16;
use sm121_kernels::{device, sampling};

fn bf16_to_u16(v: f32) -> u16 {
    bf16::from_f32(v).to_bits()
}

fn u16_to_f32(b: u16) -> f32 {
    bf16::from_bits(b).to_f32()
}

#[test]
fn test_softcap_bf16_basic() {
    let ctx = match device::init_device(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[skip] init_device: {e:?}");
            return;
        }
    };
    let stream = ctx.default_stream();

    let cap = 30.0f32;
    // Test points spanning saturation regimes
    let inputs: Vec<f32> = vec![
        0.0,   // → 0
        15.0,  // → cap * tanh(0.5) ≈ 30 * 0.4621 = 13.86
        30.0,  // → cap * tanh(1.0) ≈ 30 * 0.7616 = 22.85
        60.0,  // → cap * tanh(2.0) ≈ 30 * 0.9640 = 28.92
        100.0, // → cap * tanh(3.33) ≈ 30 * 0.9974 = 29.92
        300.0, // → cap * tanh(10) ≈ 30.0 (saturated)
        -15.0, -30.0, -60.0, -300.0, // → -30.0 (saturated)
    ];

    let _n = inputs.len() as u32;
    // Pad to even count if needed (kernel requires even).
    let mut input_bf16: Vec<u16> = inputs.iter().map(|&v| bf16_to_u16(v)).collect();
    if !input_bf16.len().is_multiple_of(2) {
        input_bf16.push(0);
    }
    let n_padded = input_bf16.len() as u32;

    let input_dev = stream.memcpy_stod(&input_bf16).expect("htod input");
    let mut out_dev = stream
        .alloc_zeros::<u16>(n_padded as usize)
        .expect("alloc out");

    sampling::softcap_bf16(&ctx, &stream, &input_dev, &mut out_dev, n_padded, cap)
        .expect("softcap launch");
    stream.synchronize().ok();

    let out_host = stream.memcpy_dtov(&out_dev).expect("dtoh");

    // Expected reference: cap * tanh(x / cap), computed in f32.
    for (i, &x) in inputs.iter().enumerate() {
        let expected = cap * (x / cap).tanh();
        let got = u16_to_f32(out_host[i]);
        let diff = (got - expected).abs();
        // BF16 has ~7 bits mantissa → relative tolerance ~0.5%, absolute tol scaled by cap.
        let tol = (expected.abs() * 0.02).max(0.2);
        eprintln!(
            "  in={x:+8.3}  expected={expected:+8.4}  got={got:+8.4}  diff={diff:.4} tol={tol:.4}"
        );
        assert!(
            diff <= tol,
            "softcap mismatch at input={x}: expected={expected}, got={got}, diff={diff}, tol={tol}"
        );
    }

    // Saturation: ±300 should be very close to ±cap.
    let sat_pos = u16_to_f32(out_host[5]);
    let sat_neg = u16_to_f32(out_host[9]);
    assert!(
        (sat_pos - cap).abs() < 0.1,
        "+300 should saturate to +30.0, got {sat_pos}"
    );
    assert!(
        (sat_neg + cap).abs() < 0.1,
        "-300 should saturate to -30.0, got {sat_neg}"
    );
}

#[test]
fn test_softcap_bf16_zero_passthrough() {
    let ctx = match device::init_device(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[skip] init_device: {e:?}");
            return;
        }
    };
    let stream = ctx.default_stream();

    let n: u32 = 256;
    let cap = 30.0f32;
    let input_bf16: Vec<u16> = (0..n).map(|_| 0u16).collect();
    let input_dev = stream.memcpy_stod(&input_bf16).unwrap();
    let mut out_dev = stream.alloc_zeros::<u16>(n as usize).unwrap();

    sampling::softcap_bf16(&ctx, &stream, &input_dev, &mut out_dev, n, cap).unwrap();
    stream.synchronize().ok();

    let out = stream.memcpy_dtov(&out_dev).unwrap();
    for (i, &v) in out.iter().enumerate() {
        assert_eq!(v, 0u16, "expected 0 at index {i}, got {v}");
    }
}
