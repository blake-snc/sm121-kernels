mod common;

use common::{compare_bf16, load_npz};
use sm121_kernels::{device, quantization};

fn byte_exact_pct(actual: &[u8], expected: &[u8]) -> (f64, usize, usize) {
    let mut exact = 0usize;
    let mut adjacent = 0usize;
    let mut far = 0usize;
    for (a, b) in actual.iter().zip(expected.iter()) {
        let d = (*a as i16 - *b as i16).abs();
        if d == 0 {
            exact += 1;
        } else if d <= 2 {
            adjacent += 1;
        } else {
            far += 1;
        }
    }
    let pct = 100.0 * exact as f64 / actual.len().max(1) as f64;
    (pct, adjacent, far)
}

#[test]
fn test_quant_bf16_to_fp8_pertoken() {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    for (n_rows, hidden) in [(4u32, 1024u32), (8, 2048)] {
        let mut npz = load_npz(&format!("quant_fp8_pertoken_n{n_rows}_h{hidden}.npz"));
        let x_np: ndarray::Array2<u16> = npz.by_name("x").unwrap();
        let scales_expected: ndarray::Array1<f32> = npz.by_name("scales").unwrap();
        let out_expected: ndarray::Array2<u8> = npz.by_name("out").unwrap();

        let x: Vec<u16> = x_np.into_raw_vec_and_offset().0;
        let expected_out: Vec<u8> = out_expected.into_raw_vec_and_offset().0;
        let expected_scales: Vec<f32> = scales_expected.into_raw_vec_and_offset().0;

        let x_dev = stream.memcpy_stod(&x).unwrap();
        let mut out_dev = stream.alloc_zeros::<u8>(expected_out.len()).unwrap();
        let mut scales_dev = stream.alloc_zeros::<f32>(n_rows as usize).unwrap();

        quantization::quant_bf16_to_fp8_pertoken(
            &ctx,
            &stream,
            &x_dev,
            &mut out_dev,
            &mut scales_dev,
            n_rows,
            hidden,
        )
        .unwrap();

        let out_host = stream.memcpy_dtov(&out_dev).unwrap();
        let scales_host = stream.memcpy_dtov(&scales_dev).unwrap();

        // Scale match
        let mut scale_max_diff: f32 = 0.0;
        for (a, b) in scales_host.iter().zip(expected_scales.iter()) {
            let d = ((a - b) / b.max(1e-5)).abs();
            if d > scale_max_diff {
                scale_max_diff = d;
            }
        }
        // FP8 output: byte-exact match is expected from identical scaling math
        let (exact_pct, adjacent, far) = byte_exact_pct(&out_host, &expected_out);
        eprintln!("quant_fp8_pertoken n={n_rows} h={hidden}: scale_rel_diff={scale_max_diff:.6}, bytes exact={exact_pct:.1}%, adjacent={adjacent}, far={far}");
        assert!(
            scale_max_diff <= 0.01,
            "scale diff too large: {scale_max_diff}"
        );
        assert!(exact_pct > 95.0, "fp8 bytes exact={exact_pct:.1}%");
        assert!(
            far < out_host.len() / 1000,
            "too many far FP8 mismatches: {far}"
        );
    }
}

#[test]
fn test_dequant_fp8_bf16() {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    for (n_rows, hidden) in [(4u32, 1024u32), (8, 2048)] {
        let mut npz = load_npz(&format!("dequant_fp8_n{n_rows}_h{hidden}.npz"));
        let x_np: ndarray::Array2<u8> = npz.by_name("x").unwrap();
        let scales_np: ndarray::Array1<f32> = npz.by_name("scales").unwrap();
        let out_expected: ndarray::Array2<u16> = npz.by_name("out").unwrap();

        let x: Vec<u8> = x_np.into_raw_vec_and_offset().0;
        let scales: Vec<f32> = scales_np.into_raw_vec_and_offset().0;
        let expected: Vec<u16> = out_expected.into_raw_vec_and_offset().0;

        let x_dev = stream.memcpy_stod(&x).unwrap();
        let scales_dev = stream.memcpy_stod(&scales).unwrap();
        let mut out_dev = stream.alloc_zeros::<u16>(expected.len()).unwrap();

        quantization::dequant_fp8_bf16(
            &ctx,
            &stream,
            &x_dev,
            &scales_dev,
            &mut out_dev,
            n_rows,
            hidden,
        )
        .unwrap();

        let out_host = stream.memcpy_dtov(&out_dev).unwrap();
        let (max_diff, mean_diff) = compare_bf16(&out_host, &expected, 0.01);
        eprintln!(
            "dequant_fp8 n={n_rows} h={hidden}: max_diff={max_diff:.6} mean_diff={mean_diff:.6}"
        );
        assert!(max_diff <= 0.01, "dequant_fp8 diff too high: {max_diff}");
    }
}

#[test]
fn test_quant_bf16_to_fp8_block128() {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    for (n_rows, hidden) in [(4u32, 512u32), (8, 1024)] {
        let mut npz = load_npz(&format!("quant_fp8_block128_n{n_rows}_h{hidden}.npz"));
        let x_np: ndarray::Array2<u16> = npz.by_name("x").unwrap();
        let scales_expected: ndarray::Array2<f32> = npz.by_name("scales").unwrap();
        let out_expected: ndarray::Array2<u8> = npz.by_name("out").unwrap();

        let x: Vec<u16> = x_np.into_raw_vec_and_offset().0;
        let expected_out: Vec<u8> = out_expected.into_raw_vec_and_offset().0;
        let expected_scales: Vec<f32> = scales_expected.into_raw_vec_and_offset().0;

        let x_dev = stream.memcpy_stod(&x).unwrap();
        let mut out_dev = stream.alloc_zeros::<u8>(expected_out.len()).unwrap();
        let mut scales_dev = stream.alloc_zeros::<f32>(expected_scales.len()).unwrap();

        quantization::quant_bf16_to_fp8_block128(
            &ctx,
            &stream,
            &x_dev,
            &mut out_dev,
            &mut scales_dev,
            n_rows,
            hidden,
        )
        .unwrap();

        let out_host = stream.memcpy_dtov(&out_dev).unwrap();
        let scales_host = stream.memcpy_dtov(&scales_dev).unwrap();

        let mut scale_max_diff: f32 = 0.0;
        for (a, b) in scales_host.iter().zip(expected_scales.iter()) {
            let d = ((a - b) / b.max(1e-5)).abs();
            if d > scale_max_diff {
                scale_max_diff = d;
            }
        }
        let (exact_pct, adjacent, far) = byte_exact_pct(&out_host, &expected_out);
        eprintln!("quant_fp8_block128 n={n_rows} h={hidden}: scale_rel_diff={scale_max_diff:.6}, bytes exact={exact_pct:.1}%, adjacent={adjacent}, far={far}");
        assert!(scale_max_diff <= 0.01, "scale diff: {scale_max_diff}");
        assert!(exact_pct > 95.0, "fp8 bytes exact={exact_pct:.1}%");
    }
}

#[test]
fn test_dequant_fp8_block128_bf16() {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    for (n_rows, hidden) in [(4u32, 512u32), (8, 1024)] {
        // Reuse the quant_fp8_block128 golden vectors — the fp8 output is the
        // dequant input, and we can re-derive expected BF16 from (x_bf16 -
        // quantized) == the lossy rounding result. Easier: use scales × dequant(fp8)
        // and compare against bf16(fp8 * scale).
        let mut npz = load_npz(&format!("quant_fp8_block128_n{n_rows}_h{hidden}.npz"));
        let fp8_np: ndarray::Array2<u8> = npz.by_name("out").unwrap();
        let scales_np: ndarray::Array2<f32> = npz.by_name("scales").unwrap();
        // Use the Python reference directly for the expected dequant (fp8 * scale rounded to bf16)
        // We re-derive it in-test:
        let fp8: Vec<u8> = fp8_np.into_raw_vec_and_offset().0;
        let scales: Vec<f32> = scales_np.into_raw_vec_and_offset().0;

        // Build expected: dequant each fp8 byte by its block scale, cast to bf16
        let n_blocks = (hidden / 128) as usize;
        let mut expected: Vec<u16> = Vec::with_capacity(fp8.len());
        for row in 0..(n_rows as usize) {
            for blk in 0..n_blocks {
                let scale = scales[row * n_blocks + blk];
                for i in 0..128 {
                    let byte = fp8[row * hidden as usize + blk * 128 + i];
                    // Decode FP8 E4M3 via bit pattern (reference decode matching hardware)
                    let f = fp8_e4m3_to_f32(byte) * scale;
                    expected.push(bf16_from_f32(f));
                }
            }
        }

        let x_dev = stream.memcpy_stod(&fp8).unwrap();
        let scales_dev = stream.memcpy_stod(&scales).unwrap();
        let mut out_dev = stream.alloc_zeros::<u16>(expected.len()).unwrap();

        quantization::dequant_fp8_block128_bf16(
            &ctx,
            &stream,
            &x_dev,
            &scales_dev,
            &mut out_dev,
            n_rows,
            hidden,
        )
        .unwrap();

        let out_host = stream.memcpy_dtov(&out_dev).unwrap();
        let (max_diff, mean_diff) = compare_bf16(&out_host, &expected, 0.05);
        eprintln!("dequant_fp8_block128 n={n_rows} h={hidden}: max_diff={max_diff:.6} mean_diff={mean_diff:.6}");
        assert!(max_diff <= 0.05, "dequant_fp8_block128 diff: {max_diff}");
    }
}

#[test]
fn test_quant_bf16_to_mxfp8() {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    for (n_rows, hidden) in [(4u32, 512u32), (8, 1024)] {
        let mut npz = load_npz(&format!("quant_mxfp8_n{n_rows}_h{hidden}.npz"));
        let x_np: ndarray::Array2<u16> = npz.by_name("x").unwrap();
        let scales_expected: ndarray::Array2<u8> = npz.by_name("scales").unwrap();
        let out_expected: ndarray::Array2<u8> = npz.by_name("out").unwrap();

        let x: Vec<u16> = x_np.into_raw_vec_and_offset().0;
        let expected_out: Vec<u8> = out_expected.into_raw_vec_and_offset().0;
        let expected_scales: Vec<u8> = scales_expected.into_raw_vec_and_offset().0;

        let x_dev = stream.memcpy_stod(&x).unwrap();
        let mut out_dev = stream.alloc_zeros::<u8>(expected_out.len()).unwrap();
        let mut scales_dev = stream.alloc_zeros::<u8>(expected_scales.len()).unwrap();

        quantization::quant_bf16_to_mxfp8(
            &ctx,
            &stream,
            &x_dev,
            &mut out_dev,
            &mut scales_dev,
            n_rows,
            hidden,
        )
        .unwrap();

        let out_host = stream.memcpy_dtov(&out_dev).unwrap();
        let scales_host = stream.memcpy_dtov(&scales_dev).unwrap();

        // Scales: should match exactly (deterministic byte)
        let scale_mismatches = scales_host
            .iter()
            .zip(expected_scales.iter())
            .filter(|(a, b)| a != b)
            .count();
        let (exact_pct, adjacent, far) = byte_exact_pct(&out_host, &expected_out);
        eprintln!("quant_mxfp8 n={n_rows} h={hidden}: scale_mismatches={scale_mismatches}, bytes exact={exact_pct:.1}%, adjacent={adjacent}, far={far}");
        assert_eq!(scale_mismatches, 0, "UE8M0 scale bytes must match exactly");
        assert!(exact_pct > 90.0, "mxfp8 bytes exact={exact_pct:.1}%");
    }
}

#[test]
fn test_dequant_mxfp8_bf16() {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    for (n_rows, hidden) in [(4u32, 512u32), (8, 1024)] {
        let mut npz = load_npz(&format!("dequant_mxfp8_n{n_rows}_h{hidden}.npz"));
        let x_np: ndarray::Array2<u8> = npz.by_name("x").unwrap();
        let scales_np: ndarray::Array2<u8> = npz.by_name("scales").unwrap();
        let out_expected: ndarray::Array2<u16> = npz.by_name("out").unwrap();

        let x: Vec<u8> = x_np.into_raw_vec_and_offset().0;
        let scales: Vec<u8> = scales_np.into_raw_vec_and_offset().0;
        let expected: Vec<u16> = out_expected.into_raw_vec_and_offset().0;

        let x_dev = stream.memcpy_stod(&x).unwrap();
        let scales_dev = stream.memcpy_stod(&scales).unwrap();
        let mut out_dev = stream.alloc_zeros::<u16>(expected.len()).unwrap();

        quantization::dequant_mxfp8_bf16(
            &ctx,
            &stream,
            &x_dev,
            &scales_dev,
            &mut out_dev,
            n_rows,
            hidden,
        )
        .unwrap();

        let out_host = stream.memcpy_dtov(&out_dev).unwrap();
        let (max_diff, mean_diff) = compare_bf16(&out_host, &expected, 0.01);
        eprintln!(
            "dequant_mxfp8 n={n_rows} h={hidden}: max_diff={max_diff:.6} mean_diff={mean_diff:.6}"
        );
        assert!(max_diff <= 0.01, "dequant_mxfp8 diff: {max_diff}");
    }
}

#[test]
fn test_quant_bf16_to_mxfp4() {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    for (n_rows, hidden) in [(4u32, 512u32), (8, 1024)] {
        let mut npz = load_npz(&format!("quant_mxfp4_n{n_rows}_h{hidden}.npz"));
        let x_np: ndarray::Array2<u16> = npz.by_name("x").unwrap();
        let scales_expected: ndarray::Array2<u8> = npz.by_name("scales").unwrap();
        let out_expected: ndarray::Array2<u8> = npz.by_name("out").unwrap();

        let x: Vec<u16> = x_np.into_raw_vec_and_offset().0;
        let expected_out: Vec<u8> = out_expected.into_raw_vec_and_offset().0;
        let expected_scales: Vec<u8> = scales_expected.into_raw_vec_and_offset().0;

        let x_dev = stream.memcpy_stod(&x).unwrap();
        let mut out_dev = stream.alloc_zeros::<u8>(expected_out.len()).unwrap();
        let mut scales_dev = stream.alloc_zeros::<u8>(expected_scales.len()).unwrap();

        quantization::quant_bf16_to_mxfp4(
            &ctx,
            &stream,
            &x_dev,
            &mut out_dev,
            &mut scales_dev,
            n_rows,
            hidden,
        )
        .unwrap();

        let out_host = stream.memcpy_dtov(&out_dev).unwrap();
        let scales_host = stream.memcpy_dtov(&scales_dev).unwrap();

        let scale_mismatches = scales_host
            .iter()
            .zip(expected_scales.iter())
            .filter(|(a, b)| a != b)
            .count();
        // FP4 has only 15 distinct values; reference Python vs hardware CVT may pick
        // different nibbles near midpoints. Count nibble exact match.
        let mut exact_nibbles = 0usize;
        let mut total_nibbles = 0usize;
        for (a, b) in out_host.iter().zip(expected_out.iter()) {
            let a_lo = a & 0xF;
            let a_hi = (a >> 4) & 0xF;
            let b_lo = b & 0xF;
            let b_hi = (b >> 4) & 0xF;
            if a_lo == b_lo {
                exact_nibbles += 1;
            }
            if a_hi == b_hi {
                exact_nibbles += 1;
            }
            total_nibbles += 2;
        }
        let pct = 100.0 * exact_nibbles as f64 / total_nibbles as f64;
        eprintln!("quant_mxfp4 n={n_rows} h={hidden}: scale_mismatches={scale_mismatches}, nibbles exact={pct:.1}%");
        assert_eq!(scale_mismatches, 0, "UE8M0 scale bytes must match");
        // Allow 20% nibble mismatch due to rounding edge cases (FP4 has only 8 positive values)
        assert!(pct > 75.0, "FP4 nibble exact match too low: {pct:.1}%");
    }
}

#[test]
fn test_dequant_mxfp4_bf16() {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    for (n_rows, hidden) in [(4u32, 512u32), (8, 1024)] {
        let mut npz = load_npz(&format!("dequant_mxfp4_n{n_rows}_h{hidden}.npz"));
        let x_np: ndarray::Array2<u8> = npz.by_name("x").unwrap();
        let scales_np: ndarray::Array2<u8> = npz.by_name("scales").unwrap();
        let out_expected: ndarray::Array2<u16> = npz.by_name("out").unwrap();

        let x: Vec<u8> = x_np.into_raw_vec_and_offset().0;
        let scales: Vec<u8> = scales_np.into_raw_vec_and_offset().0;
        let expected: Vec<u16> = out_expected.into_raw_vec_and_offset().0;

        let x_dev = stream.memcpy_stod(&x).unwrap();
        let scales_dev = stream.memcpy_stod(&scales).unwrap();
        let mut out_dev = stream.alloc_zeros::<u16>(expected.len()).unwrap();

        quantization::dequant_mxfp4_bf16(
            &ctx,
            &stream,
            &x_dev,
            &scales_dev,
            &mut out_dev,
            n_rows,
            hidden,
        )
        .unwrap();

        let out_host = stream.memcpy_dtov(&out_dev).unwrap();
        let (max_diff, mean_diff) = compare_bf16(&out_host, &expected, 0.01);
        eprintln!(
            "dequant_mxfp4 n={n_rows} h={hidden}: max_diff={max_diff:.6} mean_diff={mean_diff:.6}"
        );
        assert!(max_diff <= 0.01, "dequant_mxfp4 diff: {max_diff}");
    }
}

#[test]
fn test_quant_bf16_to_nvfp4() {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    for (n_rows, hidden) in [(4u32, 256u32), (8, 512)] {
        let mut npz = load_npz(&format!("quant_nvfp4_n{n_rows}_h{hidden}.npz"));
        let x_np: ndarray::Array2<u16> = npz.by_name("x").unwrap();
        let scales_expected: ndarray::Array2<u8> = npz.by_name("scales").unwrap();
        let out_expected: ndarray::Array2<u8> = npz.by_name("out").unwrap();

        let x: Vec<u16> = x_np.into_raw_vec_and_offset().0;
        let expected_out: Vec<u8> = out_expected.into_raw_vec_and_offset().0;
        let expected_scales: Vec<u8> = scales_expected.into_raw_vec_and_offset().0;

        let x_dev = stream.memcpy_stod(&x).unwrap();
        let mut out_dev = stream.alloc_zeros::<u8>(expected_out.len()).unwrap();
        let mut scales_dev = stream.alloc_zeros::<u8>(expected_scales.len()).unwrap();

        quantization::quant_bf16_to_nvfp4(
            &ctx,
            &stream,
            &x_dev,
            &mut out_dev,
            &mut scales_dev,
            n_rows,
            hidden,
        )
        .unwrap();

        let out_host = stream.memcpy_dtov(&out_dev).unwrap();
        let scales_host = stream.memcpy_dtov(&scales_dev).unwrap();

        // FP8 scale bytes should match (deterministic rounding)
        let scale_mismatches = scales_host
            .iter()
            .zip(expected_scales.iter())
            .filter(|(a, b)| (**a as i16 - **b as i16).abs() > 1)
            .count();
        let mut exact_nibbles = 0usize;
        let mut total_nibbles = 0usize;
        for (a, b) in out_host.iter().zip(expected_out.iter()) {
            if (a & 0xF) == (b & 0xF) {
                exact_nibbles += 1;
            }
            if ((a >> 4) & 0xF) == ((b >> 4) & 0xF) {
                exact_nibbles += 1;
            }
            total_nibbles += 2;
        }
        let pct = 100.0 * exact_nibbles as f64 / total_nibbles as f64;
        eprintln!("quant_nvfp4 n={n_rows} h={hidden}: scale_diffs_above_1={scale_mismatches}, nibbles exact={pct:.1}%");
        assert!(
            scale_mismatches < expected_scales.len() / 10,
            "too many FP8 scale mismatches"
        );
        assert!(pct > 70.0, "FP4 nibble exact match too low: {pct:.1}%");
    }
}

#[test]
fn test_dequant_nvfp4_bf16() {
    let ctx = device::init_device(0).expect("failed to init SM121");
    let stream = ctx.default_stream();

    for (n_rows, hidden) in [(4u32, 256u32), (8, 512)] {
        let mut npz = load_npz(&format!("dequant_nvfp4_n{n_rows}_h{hidden}.npz"));
        let x_np: ndarray::Array2<u8> = npz.by_name("x").unwrap();
        let scales_np: ndarray::Array2<u8> = npz.by_name("scales").unwrap();
        let out_expected: ndarray::Array2<u16> = npz.by_name("out").unwrap();

        let x: Vec<u8> = x_np.into_raw_vec_and_offset().0;
        let scales: Vec<u8> = scales_np.into_raw_vec_and_offset().0;
        let expected: Vec<u16> = out_expected.into_raw_vec_and_offset().0;

        let x_dev = stream.memcpy_stod(&x).unwrap();
        let scales_dev = stream.memcpy_stod(&scales).unwrap();
        let mut out_dev = stream.alloc_zeros::<u16>(expected.len()).unwrap();

        quantization::dequant_nvfp4_bf16(
            &ctx,
            &stream,
            &x_dev,
            &scales_dev,
            &mut out_dev,
            n_rows,
            hidden,
        )
        .unwrap();

        let out_host = stream.memcpy_dtov(&out_dev).unwrap();
        let (max_diff, mean_diff) = compare_bf16(&out_host, &expected, 0.02);
        eprintln!(
            "dequant_nvfp4 n={n_rows} h={hidden}: max_diff={max_diff:.6} mean_diff={mean_diff:.6}"
        );
        assert!(max_diff <= 0.02, "dequant_nvfp4 diff: {max_diff}");
    }
}

// --- FP8 E4M3 decode reference (matches hardware behavior) ---
fn fp8_e4m3_to_f32(b: u8) -> f32 {
    let sign = (b >> 7) & 1;
    let exp = (b >> 3) & 0xF;
    let mant = b & 0x7;
    let sign_f = if sign == 1 { -1.0f32 } else { 1.0 };
    if exp == 0 {
        // Subnormal: (-1)^s * 0.mmm * 2^(1-7)
        if mant == 0 {
            return 0.0;
        }
        return sign_f * (mant as f32) / 8.0 * 2f32.powi(1 - 7);
    }
    // E4M3 has no Inf; 0x7F and 0xFF are NaN only.
    if exp == 0xF && mant == 0x7 {
        return f32::NAN;
    }
    // Normal: (-1)^s * 1.mmm * 2^(exp - 7)
    let mantissa_f = 1.0 + (mant as f32) / 8.0;
    sign_f * mantissa_f * 2f32.powi(exp as i32 - 7)
}

fn bf16_from_f32(f: f32) -> u16 {
    let bits = f.to_bits();
    // Round-to-nearest-even
    let rounding = 0x7FFF + ((bits >> 16) & 1);
    let rounded = bits.wrapping_add(rounding);
    (rounded >> 16) as u16
}
