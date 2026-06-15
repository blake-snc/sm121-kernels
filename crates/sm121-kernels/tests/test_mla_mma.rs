//! Validate full MMA-MLA BF16 decode against scalar reference.

mod common;

use sm121_kernels::{attention, device};

const D_C: usize = 512;
const D_R: usize = 64;

fn bf16_bits(f: f32) -> u16 {
    half::bf16::from_f32(f).to_bits()
}

fn bf16_to_f32(b: u16) -> f32 {
    half::bf16::from_bits(b).to_f32()
}

fn gen_bf16_values(n: usize, seed: u64) -> Vec<u16> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let v = (s >> 33) as u32;
            let f = ((v & 0xFFFF) as f32 / 32768.0) - 1.0;
            bf16_bits(f * 0.5)
        })
        .collect()
}

fn compare_vs_scalar<F>(
    batch: u32,
    num_heads: u32,
    seq_kv: u32,
    tol: f32,
    kernel_name: &str,
    launch: F,
) where
    F: Fn(
        &std::sync::Arc<cudarc::driver::CudaContext>,
        &std::sync::Arc<cudarc::driver::CudaStream>,
        &cudarc::driver::CudaSlice<u16>,
        &cudarc::driver::CudaSlice<u16>,
        &cudarc::driver::CudaSlice<u16>,
        &cudarc::driver::CudaSlice<u16>,
        &mut cudarc::driver::CudaSlice<u16>,
    ),
{
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();

    let q_c = gen_bf16_values((batch * num_heads) as usize * D_C, 0xDEADBEEF);
    let q_r = gen_bf16_values((batch * num_heads) as usize * D_R, 0xCAFEBABE);
    let c_kv = gen_bf16_values((batch * seq_kv) as usize * D_C, 0x12345678);
    let k_rope = gen_bf16_values((batch * seq_kv) as usize * D_R, 0x87654321);
    let scale = 1.0f32 / ((D_C + D_R) as f32).sqrt();

    let q_c_d = stream.memcpy_stod(&q_c).unwrap();
    let q_r_d = stream.memcpy_stod(&q_r).unwrap();
    let c_kv_d = stream.memcpy_stod(&c_kv).unwrap();
    let k_rope_d = stream.memcpy_stod(&k_rope).unwrap();

    let o_len = (batch * num_heads) as usize * D_C;
    let mut o_scalar = stream.alloc_zeros::<u16>(o_len).unwrap();
    let mut o_kernel = stream.alloc_zeros::<u16>(o_len).unwrap();

    attention::mla_decode_bf16(
        &ctx,
        &stream,
        &q_c_d,
        &q_r_d,
        &c_kv_d,
        &k_rope_d,
        &mut o_scalar,
        batch,
        num_heads,
        seq_kv,
        scale,
    )
    .expect("scalar");

    launch(
        &ctx,
        &stream,
        &q_c_d,
        &q_r_d,
        &c_kv_d,
        &k_rope_d,
        &mut o_kernel,
    );

    let got = stream.memcpy_dtov(&o_kernel).unwrap();
    let ref_ = stream.memcpy_dtov(&o_scalar).unwrap();

    let mut max_d: f32 = 0.0;
    let mut sum_d: f32 = 0.0;
    let mut first_mismatch = None;
    for i in 0..o_len {
        let a = bf16_to_f32(got[i]);
        let e = bf16_to_f32(ref_[i]);
        let d = (a - e).abs();
        if d > max_d {
            max_d = d;
        }
        sum_d += d;
        if d > tol && first_mismatch.is_none() {
            first_mismatch = Some((i, a, e, d));
        }
    }
    eprintln!(
        "{kernel_name} B={batch} H={num_heads} Skv={seq_kv}: max_diff={max_d:.5} mean_diff={:.6}",
        sum_d / o_len as f32
    );
    if let Some((i, a, e, d)) = first_mismatch {
        panic!("{kernel_name} mismatch at i={i}: got={a:.6} expected={e:.6} diff={d:.6} > tol={tol} max_diff={max_d:.5}");
    }
}

fn run_mma(batch: u32, num_heads: u32, seq_kv: u32, tol: f32) {
    compare_vs_scalar(
        batch,
        num_heads,
        seq_kv,
        tol,
        "MMA",
        |ctx, stream, qc, qr, ckv, kr, o| {
            attention::mla_decode_bf16_mma(
                ctx,
                stream,
                qc,
                qr,
                ckv,
                kr,
                o,
                batch,
                num_heads,
                seq_kv,
                1.0 / ((D_C + D_R) as f32).sqrt(),
            )
            .unwrap();
        },
    );
}

fn run_mma_tma(batch: u32, num_heads: u32, seq_kv: u32, tol: f32) {
    compare_vs_scalar(
        batch,
        num_heads,
        seq_kv,
        tol,
        "MMA-TMA",
        |ctx, stream, qc, qr, ckv, kr, o| {
            attention::mla_decode_bf16_mma_tma(
                ctx,
                stream,
                qc,
                qr,
                ckv,
                kr,
                o,
                batch,
                num_heads,
                seq_kv,
                1.0 / ((D_C + D_R) as f32).sqrt(),
            )
            .unwrap();
        },
    );
}

#[test]
fn t_mma_b1_h16_skv8() {
    run_mma(1, 16, 8, 0.05);
}

#[test]
fn t_mma_b1_h16_skv64() {
    run_mma(1, 16, 64, 0.05);
}

#[test]
fn t_mma_b2_h32_skv128() {
    run_mma(2, 32, 128, 0.1);
}

#[test]
fn t_mma_tma_b1_h16_skv8() {
    run_mma_tma(1, 16, 8, 0.05);
}

#[test]
fn t_mma_tma_b1_h16_skv64() {
    run_mma_tma(1, 16, 64, 0.05);
}

#[test]
fn t_mma_tma_b2_h32_skv128() {
    run_mma_tma(2, 32, 128, 0.1);
}

fn run_mma_split(batch: u32, num_heads: u32, seq_kv: u32, num_splits: u32, tol: f32) {
    let tag = format!("MMA-Split{num_splits}");
    compare_vs_scalar(
        batch,
        num_heads,
        seq_kv,
        tol,
        &tag,
        |ctx, stream, qc, qr, ckv, kr, o| {
            attention::mla_decode_bf16_mma_split(
                ctx,
                stream,
                qc,
                qr,
                ckv,
                kr,
                o,
                batch,
                num_heads,
                seq_kv,
                num_splits,
                1.0 / ((D_C + D_R) as f32).sqrt(),
            )
            .unwrap();
        },
    );
}

#[test]
fn t_mma_split1_b1_h16_skv64() {
    run_mma_split(1, 16, 64, 1, 0.05);
}

#[test]
fn t_mma_split2_b1_h16_skv64() {
    run_mma_split(1, 16, 64, 2, 0.05);
}

#[test]
fn t_mma_split4_b1_h16_skv128() {
    run_mma_split(1, 16, 128, 4, 0.05);
}

#[test]
fn t_mma_split4_b2_h32_skv256() {
    run_mma_split(2, 32, 256, 4, 0.1);
}

fn run_auto(batch: u32, num_heads: u32, seq_kv: u32, tol: f32) {
    compare_vs_scalar(
        batch,
        num_heads,
        seq_kv,
        tol,
        "Auto",
        |ctx, stream, qc, qr, ckv, kr, o| {
            attention::mla_decode_bf16_auto(
                ctx,
                stream,
                qc,
                qr,
                ckv,
                kr,
                o,
                batch,
                num_heads,
                seq_kv,
                1.0 / ((D_C + D_R) as f32).sqrt(),
            )
            .unwrap();
        },
    );
}

#[test]
fn t_auto_b1h16_skv8() {
    run_auto(1, 16, 8, 0.05);
}
#[test]
fn t_auto_b1h16_skv64() {
    run_auto(1, 16, 64, 0.05);
}
#[test]
fn t_auto_b1h16_skv1024() {
    run_auto(1, 16, 1024, 0.1);
}
#[test]
fn t_auto_b2h32_skv256() {
    run_auto(2, 32, 256, 0.1);
}
#[test]
fn t_auto_b1h8_skv128() {
    run_auto(1, 8, 128, 0.05);
} // H%16 != 0 → scalar fallback

fn run_mma_tma_pq(batch: u32, num_heads: u32, seq_kv: u32, tol: f32) {
    compare_vs_scalar(
        batch,
        num_heads,
        seq_kv,
        tol,
        "MMA-TMA-PQr",
        |ctx, stream, qc, qr, ckv, kr, o| {
            attention::mla_decode_bf16_mma_tma_pq(
                ctx,
                stream,
                qc,
                qr,
                ckv,
                kr,
                o,
                batch,
                num_heads,
                seq_kv,
                1.0 / ((D_C + D_R) as f32).sqrt(),
            )
            .unwrap();
        },
    );
}

#[test]
fn t_mma_tma_pq_b1_h16_skv64() {
    run_mma_tma_pq(1, 16, 64, 0.05);
}
#[test]
fn t_mma_tma_pq_b2_h32_skv128() {
    run_mma_tma_pq(2, 32, 128, 0.1);
}
#[test]
fn t_mma_tma_pq_b2_h128_skv1024() {
    run_mma_tma_pq(2, 128, 1024, 0.1);
}

fn run_auto_override(batch: u32, num_heads: u32, seq_kv: u32, num_splits: Option<u32>, tol: f32) {
    let tag = format!("Auto-Override{num_splits:?}");
    compare_vs_scalar(
        batch,
        num_heads,
        seq_kv,
        tol,
        &tag,
        |ctx, stream, qc, qr, ckv, kr, o| {
            attention::mla_decode_bf16_auto_with_splits(
                ctx,
                stream,
                qc,
                qr,
                ckv,
                kr,
                o,
                batch,
                num_heads,
                seq_kv,
                1.0 / ((D_C + D_R) as f32).sqrt(),
                num_splits,
            )
            .unwrap();
        },
    );
}

#[test]
fn t_auto_ov_force_split4() {
    run_auto_override(1, 16, 128, Some(4), 0.05);
}
#[test]
fn t_auto_ov_force_nosplit() {
    run_auto_override(1, 16, 128, Some(1), 0.05);
}
#[test]
fn t_auto_ov_none_small() {
    run_auto_override(1, 16, 32, None, 0.05);
}
