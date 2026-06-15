//! Shared test helpers for sm121-kernels integration tests.
#![allow(dead_code)]

/// Returns the workspace root (two levels above CARGO_MANIFEST_DIR).
pub fn project_root() -> std::path::PathBuf {
    let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest.parent().unwrap().parent().unwrap().to_path_buf()
}

/// Opens an .npz file from `tests/reference/data/`.
pub fn load_npz(name: &str) -> ndarray_npy::NpzReader<std::io::BufReader<std::fs::File>> {
    let path = project_root().join("tests/reference/data").join(name);
    ndarray_npy::NpzReader::new(std::io::BufReader::new(
        std::fs::File::open(&path)
            .unwrap_or_else(|e| panic!("failed to open {}: {e}", path.display())),
    ))
    .unwrap()
}

/// Compare two u16 slices interpreted as BF16 with absolute tolerance.
///
/// Panics on NaN (either side) or any element diff exceeding `tol`.
/// Returns (max_abs_diff, mean_abs_diff).
pub fn compare_bf16(actual: &[u16], expected: &[u16], tol: f32) -> (f32, f32) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    let mut max_diff: f32 = 0.0;
    let mut sum_diff: f32 = 0.0;
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let af = half::bf16::from_bits(a).to_f32();
        let ef = half::bf16::from_bits(e).to_f32();
        if af.is_nan() || ef.is_nan() {
            panic!("NaN detected at index {i}: actual={af} expected={ef}");
        }
        let diff = (af - ef).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        sum_diff += diff;
        if diff > tol {
            panic!(
                "mismatch at index {i}: actual={af:.6} expected={ef:.6} diff={diff:.6} > tol={tol}"
            );
        }
    }
    let mean_diff = sum_diff / actual.len() as f32;
    (max_diff, mean_diff)
}

#[allow(dead_code)]
/// Compare two u16 slices interpreted as BF16 using numpy-style allclose:
/// `pass if |a - e| <= atol + rtol * max(|e|, eps)`
///
/// Panics on NaN or any element failing the combined tolerance.
/// Returns (max_abs_diff, mean_abs_diff).
pub fn compare_bf16_allclose(actual: &[u16], expected: &[u16], atol: f32, rtol: f32) -> (f32, f32) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    let eps = 1e-8_f32;
    let mut max_diff: f32 = 0.0;
    let mut sum_diff: f32 = 0.0;
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let af = half::bf16::from_bits(a).to_f32();
        let ef = half::bf16::from_bits(e).to_f32();
        if af.is_nan() || ef.is_nan() {
            panic!("NaN detected at index {i}: actual={af} expected={ef}");
        }
        let diff = (af - ef).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        sum_diff += diff;
        let tol = atol + rtol * ef.abs().max(eps);
        if diff > tol {
            panic!(
                "allclose mismatch at index {i}: actual={af:.6} expected={ef:.6} diff={diff:.6} > tol={tol:.6} (atol={atol}, rtol={rtol})"
            );
        }
    }
    let mean_diff = sum_diff / actual.len() as f32;
    (max_diff, mean_diff)
}
