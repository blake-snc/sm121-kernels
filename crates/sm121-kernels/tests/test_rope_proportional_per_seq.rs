//! Validates `rope_proportional_bf16_per_seq` against the existing
//! `rope_proportional_bf16_pos_dev` kernel applied per-sequence with each
//! sequence's own position. Both kernels share the same math; the only
//! difference is where the position is read (broadcast vs per-seq lookup).
//! For matching inputs and identical positions, outputs must be bit-exact;
//! for HETEROGENEOUS positions, the per_seq kernel must produce
//! per-sequence results that match running the pos_dev kernel separately
//! on each sequence's slice with that sequence's position.

use half::bf16;
use sm121_kernels::{device, rope};

fn bf16(x: f32) -> u16 {
    bf16::from_f32(x).to_bits()
}

#[test]
fn per_seq_matches_pos_dev_homogeneous() {
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();

    // Gemma-4 full-attn shape: head_dim=512, rope_angles=64.
    let m: u32 = 4;
    let heads_per_seq: u32 = 16;
    let head_dim: u32 = 512;
    let rope_angles: u32 = 64;
    let theta = 1_000_000.0_f32;

    let mut s = 0xFEED_CAFE_u64;
    let mut next = || -> f32 {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 33) as u32 as f32) / (u32::MAX as f32) - 0.5
    };
    let n_elem = (m * heads_per_seq * head_dim) as usize;
    let host: Vec<u16> = (0..n_elem).map(|_| bf16(next())).collect();

    // Homogeneous positions (pos=10 for all 4 seqs).
    let positions = vec![10u32; m as usize];

    // Reference: rope_proportional_bf16_pos_dev called M times on per-seq slices.
    let mut ref_buf = stream.memcpy_stod(&host).unwrap();
    for s_idx in 0..m as usize {
        let off = s_idx * (heads_per_seq * head_dim) as usize;
        let len = (heads_per_seq * head_dim) as usize;
        // Copy out the slice, run pos_dev on a fresh CudaSlice, copy back.
        // (slice_mut + pos_dev variant would be cleaner but the dispatcher
        // takes CudaSlice not view; this is a test so we tolerate copies.)
        let slice_host = stream.memcpy_dtov(&ref_buf.slice(off..off + len)).unwrap();
        let mut slice_dev = stream.memcpy_stod(&slice_host).unwrap();
        let pos_dev = stream.memcpy_stod(&[positions[s_idx]]).unwrap();
        rope::rope_proportional_bf16_pos_dev(
            &ctx,
            &stream,
            &mut slice_dev,
            &pos_dev,
            theta,
            heads_per_seq,
            head_dim,
            rope_angles,
        )
        .unwrap();
        let out_host = stream.memcpy_dtov(&slice_dev).unwrap();
        let tmp = stream.memcpy_stod(&out_host).unwrap();
        let mut dst = ref_buf.slice_mut(off..off + len);
        stream.memcpy_dtod(&tmp, &mut dst).unwrap();
    }
    stream.synchronize().unwrap();
    let ref_out = stream.memcpy_dtov(&ref_buf).unwrap();

    // Subject: rope_proportional_bf16_per_seq with positions.
    let mut sub_buf = stream.memcpy_stod(&host).unwrap();
    let pos_dev_m = stream.memcpy_stod(&positions).unwrap();
    rope::rope_proportional_bf16_per_seq(
        &ctx,
        &stream,
        &mut sub_buf,
        &pos_dev_m,
        theta,
        m,
        heads_per_seq,
        head_dim,
        rope_angles,
    )
    .unwrap();
    stream.synchronize().unwrap();
    let sub_out = stream.memcpy_dtov(&sub_buf).unwrap();

    let mut mismatches = 0;
    for i in 0..n_elem {
        if ref_out[i] != sub_out[i] {
            mismatches += 1;
            if mismatches <= 4 {
                eprintln!(
                    "mismatch at {i}: ref=0x{:04x} sub=0x{:04x}",
                    ref_out[i], sub_out[i]
                );
            }
        }
    }
    assert_eq!(
        mismatches, 0,
        "{mismatches}/{n_elem} BF16 mismatches (homogeneous positions)"
    );
}

#[test]
fn per_seq_matches_pos_dev_heterogeneous() {
    let ctx = device::init_device(0).expect("init");
    let stream = ctx.default_stream();

    let m: u32 = 4;
    let heads_per_seq: u32 = 16;
    let head_dim: u32 = 512;
    let rope_angles: u32 = 64;
    let theta = 1_000_000.0_f32;

    let mut s = 0xC0DE_BEEF_u64;
    let mut next = || -> f32 {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 33) as u32 as f32) / (u32::MAX as f32) - 0.5
    };
    let n_elem = (m * heads_per_seq * head_dim) as usize;
    let host: Vec<u16> = (0..n_elem).map(|_| bf16(next())).collect();

    // HETEROGENEOUS positions — each seq at a distinct position.
    let positions = vec![0u32, 7u32, 31u32, 128u32];

    // Reference.
    let mut ref_buf = stream.memcpy_stod(&host).unwrap();
    for s_idx in 0..m as usize {
        let off = s_idx * (heads_per_seq * head_dim) as usize;
        let len = (heads_per_seq * head_dim) as usize;
        let slice_host = stream.memcpy_dtov(&ref_buf.slice(off..off + len)).unwrap();
        let mut slice_dev = stream.memcpy_stod(&slice_host).unwrap();
        let pos_dev = stream.memcpy_stod(&[positions[s_idx]]).unwrap();
        rope::rope_proportional_bf16_pos_dev(
            &ctx,
            &stream,
            &mut slice_dev,
            &pos_dev,
            theta,
            heads_per_seq,
            head_dim,
            rope_angles,
        )
        .unwrap();
        let out_host = stream.memcpy_dtov(&slice_dev).unwrap();
        let tmp = stream.memcpy_stod(&out_host).unwrap();
        let mut dst = ref_buf.slice_mut(off..off + len);
        stream.memcpy_dtod(&tmp, &mut dst).unwrap();
    }
    stream.synchronize().unwrap();
    let ref_out = stream.memcpy_dtov(&ref_buf).unwrap();

    // Subject.
    let mut sub_buf = stream.memcpy_stod(&host).unwrap();
    let pos_dev_m = stream.memcpy_stod(&positions).unwrap();
    rope::rope_proportional_bf16_per_seq(
        &ctx,
        &stream,
        &mut sub_buf,
        &pos_dev_m,
        theta,
        m,
        heads_per_seq,
        head_dim,
        rope_angles,
    )
    .unwrap();
    stream.synchronize().unwrap();
    let sub_out = stream.memcpy_dtov(&sub_buf).unwrap();

    let mut mismatches = 0;
    for i in 0..n_elem {
        if ref_out[i] != sub_out[i] {
            mismatches += 1;
            if mismatches <= 6 {
                let seq = i / (heads_per_seq * head_dim) as usize;
                let within = i % (heads_per_seq * head_dim) as usize;
                eprintln!(
                    "mismatch seq {seq} within={within}: ref=0x{:04x} sub=0x{:04x}",
                    ref_out[i], sub_out[i]
                );
            }
        }
    }
    // Bit-exact expectation: both kernels compute the same FP32 math, write
    // through the same cvt.rn.bf16.f32. The per_seq kernel just reads pos
    // from a different memory address. No reduction → no ordering noise.
    assert_eq!(
        mismatches, 0,
        "{mismatches}/{n_elem} BF16 mismatches (heterogeneous positions)"
    );
}
