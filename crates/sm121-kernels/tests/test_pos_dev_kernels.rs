//! Bit-exact validation of the position-from-device-pointer kernel variants
//! against their param-based siblings. The pos_dev kernels were written so
//! they can be replayed under a CUDA Graph; their math is otherwise identical
//! to the param variants, so a bit-exact match is the right correctness bar.
//!
//! Covers:
//!   * kv_append_strided_bf16_pos_dev   vs kv_append_strided_bf16
//!   * fa_bf16_decode_d256_gqa_pos_dev  vs fa_bf16_decode_d256_gqa
//!   * fa_bf16_decode_d512_gqa_pos_dev  vs fa_bf16_decode_d512_gqa

use half::bf16;
use sm121_kernels::{attention, device, kv_cache};
use std::hash::{Hash, Hasher};

fn deterministic_bf16_buf(seed: u64, n: usize) -> Vec<u16> {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    seed.hash(&mut hasher);
    let mut s = hasher.finish();
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = ((s >> 32) as u32) & 0x007fffff;
        let f = f32::from_bits(0x3f800000 | bits) - 1.5; // [-0.5, 0.5)
        out.push(bf16::from_f32(f * 2.0).to_bits()); // [-1.0, 1.0)
    }
    out
}

#[test]
fn kv_append_pos_dev_matches_param_variant() {
    let ctx = device::init_device(0).expect("init device");
    let stream = ctx.default_stream();

    let cases: &[(u32, u32, u32, u32)] = &[
        // (n_kv, head_dim, max_seq, position)
        (2, 256, 64, 0),
        (2, 256, 64, 5),
        (2, 256, 64, 63),
        (2, 512, 32, 17),
        (4, 128, 16, 9),
    ];

    for &(n_kv, head_dim, max_seq, pos) in cases {
        let src_n = (n_kv * head_dim) as usize;
        let dst_n = (n_kv * max_seq * head_dim) as usize;
        let src_k = deterministic_bf16_buf(0xA1, src_n);
        let src_v = deterministic_bf16_buf(0xA2, src_n);
        let init_k = deterministic_bf16_buf(0xB1, dst_n);
        let init_v = deterministic_bf16_buf(0xB2, dst_n);

        let src_k_dev = stream.memcpy_stod(&src_k).unwrap();
        let src_v_dev = stream.memcpy_stod(&src_v).unwrap();

        // Param variant.
        let mut dst_k_a = stream.memcpy_stod(&init_k).unwrap();
        let mut dst_v_a = stream.memcpy_stod(&init_v).unwrap();
        kv_cache::kv_append_strided_bf16(
            &ctx,
            &stream,
            &src_k_dev,
            &src_v_dev,
            &mut dst_k_a,
            &mut dst_v_a,
            pos,
            max_seq,
            n_kv,
            head_dim,
        )
        .unwrap();
        stream.synchronize().ok();
        let host_k_a = stream.memcpy_dtov(&dst_k_a).unwrap();
        let host_v_a = stream.memcpy_dtov(&dst_v_a).unwrap();

        // Pos_dev variant.
        let mut dst_k_b = stream.memcpy_stod(&init_k).unwrap();
        let mut dst_v_b = stream.memcpy_stod(&init_v).unwrap();
        let pos_dev = stream.memcpy_stod(&[pos]).unwrap();
        kv_cache::kv_append_strided_bf16_pos_dev(
            &ctx,
            &stream,
            &src_k_dev,
            &src_v_dev,
            &mut dst_k_b,
            &mut dst_v_b,
            &pos_dev,
            max_seq,
            n_kv,
            head_dim,
        )
        .unwrap();
        stream.synchronize().ok();
        let host_k_b = stream.memcpy_dtov(&dst_k_b).unwrap();
        let host_v_b = stream.memcpy_dtov(&dst_v_b).unwrap();

        assert_eq!(
            host_k_a, host_k_b,
            "K mismatch for (n_kv={n_kv}, head_dim={head_dim}, max_seq={max_seq}, pos={pos})"
        );
        assert_eq!(
            host_v_a, host_v_b,
            "V mismatch for (n_kv={n_kv}, head_dim={head_dim}, max_seq={max_seq}, pos={pos})"
        );
    }
}

#[test]
fn kv_append_fp8_pos_dev_matches_param_variant() {
    let ctx = device::init_device(0).expect("init device");
    let stream = ctx.default_stream();

    let cases: &[(u32, u32, u32, u32)] = &[
        (2, 256, 64, 0),
        (2, 256, 64, 5),
        (2, 256, 64, 63),
        (2, 512, 32, 17),
        (4, 128, 16, 9),
    ];

    for &(n_kv, head_dim, max_seq, pos) in cases {
        let src_n = (n_kv * head_dim) as usize;
        let dst_n = (n_kv * max_seq * head_dim) as usize;
        let src_k = deterministic_bf16_buf(0xC1, src_n);
        let src_v = deterministic_bf16_buf(0xC2, src_n);
        let init_k = vec![0u8; dst_n];
        let init_v = vec![0u8; dst_n];

        let src_k_dev = stream.memcpy_stod(&src_k).unwrap();
        let src_v_dev = stream.memcpy_stod(&src_v).unwrap();

        let k_scale = 0.07f32; // arbitrary; same for both kernel calls
        let v_scale = 0.05f32;

        // Param variant.
        let mut dst_k_a = stream.memcpy_stod(&init_k).unwrap();
        let mut dst_v_a = stream.memcpy_stod(&init_v).unwrap();
        kv_cache::kv_append_strided_fp8(
            &ctx,
            &stream,
            &src_k_dev,
            &src_v_dev,
            &mut dst_k_a,
            &mut dst_v_a,
            pos,
            max_seq,
            n_kv,
            head_dim,
            k_scale,
            v_scale,
        )
        .unwrap();
        stream.synchronize().ok();
        let host_k_a = stream.memcpy_dtov(&dst_k_a).unwrap();
        let host_v_a = stream.memcpy_dtov(&dst_v_a).unwrap();

        // Pos_dev variant.
        let mut dst_k_b = stream.memcpy_stod(&init_k).unwrap();
        let mut dst_v_b = stream.memcpy_stod(&init_v).unwrap();
        let pos_dev = stream.memcpy_stod(&[pos]).unwrap();
        kv_cache::kv_append_strided_fp8_pos_dev(
            &ctx,
            &stream,
            &src_k_dev,
            &src_v_dev,
            &mut dst_k_b,
            &mut dst_v_b,
            &pos_dev,
            max_seq,
            n_kv,
            head_dim,
            k_scale,
            v_scale,
        )
        .unwrap();
        stream.synchronize().ok();
        let host_k_b = stream.memcpy_dtov(&dst_k_b).unwrap();
        let host_v_b = stream.memcpy_dtov(&dst_v_b).unwrap();

        assert_eq!(
            host_k_a, host_k_b,
            "FP8 K mismatch (n_kv={n_kv}, head_dim={head_dim}, max_seq={max_seq}, pos={pos})"
        );
        assert_eq!(
            host_v_a, host_v_b,
            "FP8 V mismatch (n_kv={n_kv}, head_dim={head_dim}, max_seq={max_seq}, pos={pos})"
        );
    }
}

#[test]
fn flash_attn_d256_gqa_pos_dev_matches_param_variant() {
    let ctx = device::init_device(0).expect("init device");
    let stream = ctx.default_stream();

    let d: u32 = 256;
    let cases: &[(u32, u32, u32, u32, u32, u32, u32)] = &[
        // (batch, n_q, n_kv, kv_stride, q_pos, sliding_window, _unused)
        (1, 32, 2, 64, 0, 0, 0),
        (1, 32, 2, 64, 7, 0, 0),
        (1, 32, 2, 64, 31, 512, 0), // SWA bound but window > q_pos
        (1, 32, 2, 128, 65, 32, 0), // SWA bound clamps
        (2, 16, 4, 64, 17, 16, 0),
    ];

    for &(batch, n_q, n_kv, kv_stride, q_pos, sw, _) in cases {
        let q_n = (batch * n_q * d) as usize;
        let kv_n = (batch * n_kv * kv_stride * d) as usize;

        let q_buf = deterministic_bf16_buf(0xC1, q_n);
        let k_buf = deterministic_bf16_buf(0xC2, kv_n);
        let v_buf = deterministic_bf16_buf(0xC3, kv_n);

        let q_dev = stream.memcpy_stod(&q_buf).unwrap();
        let k_dev = stream.memcpy_stod(&k_buf).unwrap();
        let v_dev = stream.memcpy_stod(&v_buf).unwrap();

        let scale = 1.0f32 / (d as f32).sqrt();
        let seq_kv = q_pos + 1;

        let mut o_a = stream.alloc_zeros::<u16>(q_n).unwrap();
        attention::flash_attn_bf16_decode_d256_gqa(
            &ctx, &stream, &q_dev, &k_dev, &v_dev, &mut o_a, batch, n_q, n_kv, seq_kv, kv_stride,
            q_pos, sw, scale,
        )
        .unwrap();
        stream.synchronize().ok();
        let host_a = stream.memcpy_dtov(&o_a).unwrap();

        let pos_dev = stream.memcpy_stod(&[q_pos]).unwrap();
        let mut o_b = stream.alloc_zeros::<u16>(q_n).unwrap();
        attention::flash_attn_bf16_decode_d256_gqa_pos_dev(
            &ctx, &stream, &q_dev, &k_dev, &v_dev, &mut o_b, batch, n_q, n_kv, kv_stride, &pos_dev,
            sw, scale,
        )
        .unwrap();
        stream.synchronize().ok();
        let host_b = stream.memcpy_dtov(&o_b).unwrap();

        assert_eq!(host_a, host_b,
            "d256 attention output mismatch \
             (batch={batch}, n_q={n_q}, n_kv={n_kv}, kv_stride={kv_stride}, q_pos={q_pos}, sw={sw})");
    }
}

#[test]
fn flash_attn_d512_gqa_pos_dev_matches_param_variant() {
    let ctx = device::init_device(0).expect("init device");
    let stream = ctx.default_stream();

    let d: u32 = 512;
    let cases: &[(u32, u32, u32, u32, u32)] = &[
        // (batch, n_q, n_kv, kv_stride, q_pos)
        (1, 32, 2, 64, 0),
        (1, 32, 2, 64, 5),
        (1, 32, 2, 128, 63),
        (2, 16, 4, 64, 17),
    ];

    for &(batch, n_q, n_kv, kv_stride, q_pos) in cases {
        let q_n = (batch * n_q * d) as usize;
        let kv_n = (batch * n_kv * kv_stride * d) as usize;

        let q_buf = deterministic_bf16_buf(0xD1, q_n);
        let k_buf = deterministic_bf16_buf(0xD2, kv_n);
        let v_buf = deterministic_bf16_buf(0xD3, kv_n);

        let q_dev = stream.memcpy_stod(&q_buf).unwrap();
        let k_dev = stream.memcpy_stod(&k_buf).unwrap();
        let v_dev = stream.memcpy_stod(&v_buf).unwrap();

        let scale = 1.0f32 / (d as f32).sqrt();
        let seq_kv = q_pos + 1;

        let mut o_a = stream.alloc_zeros::<u16>(q_n).unwrap();
        attention::flash_attn_bf16_decode_d512_gqa(
            &ctx, &stream, &q_dev, &k_dev, &v_dev, &mut o_a, batch, n_q, n_kv, seq_kv, kv_stride,
            scale,
        )
        .unwrap();
        stream.synchronize().ok();
        let host_a = stream.memcpy_dtov(&o_a).unwrap();

        let pos_dev = stream.memcpy_stod(&[q_pos]).unwrap();
        let mut o_b = stream.alloc_zeros::<u16>(q_n).unwrap();
        attention::flash_attn_bf16_decode_d512_gqa_pos_dev(
            &ctx, &stream, &q_dev, &k_dev, &v_dev, &mut o_b, batch, n_q, n_kv, kv_stride, &pos_dev,
            scale,
        )
        .unwrap();
        stream.synchronize().ok();
        let host_b = stream.memcpy_dtov(&o_b).unwrap();

        assert_eq!(
            host_a, host_b,
            "d512 attention output mismatch \
             (batch={batch}, n_q={n_q}, n_kv={n_kv}, kv_stride={kv_stride}, q_pos={q_pos})"
        );
    }
}
