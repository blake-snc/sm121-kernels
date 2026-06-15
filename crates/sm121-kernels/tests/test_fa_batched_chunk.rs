//! #437 step-1 foundation: validate that the EXISTING d256 GQA causal FA kernel
//! handles `batch=M` with `seq_q=C` block-diagonally — each batch element's
//! C-query chunk attends causally to ONLY its own KV at its own base position.
//!
//! This is the load-bearing claim for cross-prompt batched prefill (no new FA
//! kernel needed). Proof by self-reference: a single batch=2 call must produce,
//! for each batch element, BIT-IDENTICAL output to a batch=1 call on that
//! element's slice. If the kernel mixed batches or mishandled seq_q>1 per batch,
//! the halves would differ. Model-free + fast (pure kernel test).

use anyhow::{anyhow, Result};

use sm121_kernels::attention::flash_attn_bf16_v3_d256_gqa_causal_pos_dev;
use sm121_kernels::device;

#[inline]
fn f32_to_bf16(x: f32) -> u16 {
    (x.to_bits() >> 16) as u16
}

/// Deterministic pseudo-random bf16 fill in ~[-1, 1] via a small LCG.
fn fill(n: usize, seed: u64) -> Vec<u16> {
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((s >> 33) as u32) as f32 / (u32::MAX as f32); // [0,1)
            f32_to_bf16(u * 2.0 - 1.0)
        })
        .collect()
}

#[test]
fn fa_batch_m_seq_q_c_is_block_diagonal() -> Result<()> {
    let ctx = match device::init_device(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skip: no device ({e:?})");
            return Ok(());
        }
    };
    let stream = ctx.new_stream().map_err(|e| anyhow!("stream: {e:?}"))?;

    const D: usize = 256;
    let nh = 4u32;
    let nkv = 2u32;
    let c = 128u32; // seq_q per batch (the prefill chunk)
    let seq_kv = 512u32; // cache length (slab)
    let start_pos = 256u32; // base position of the chunk → query i at start_pos+i
    let scale = 1.0f32 / (D as f32).sqrt();

    let q_per_batch = nh as usize * c as usize * D;
    let kv_per_batch = nkv as usize * seq_kv as usize * D;

    // Two batch elements with DISTINCT data.
    let q0 = fill(q_per_batch, 1);
    let q1 = fill(q_per_batch, 2);
    let k0 = fill(kv_per_batch, 3);
    let k1 = fill(kv_per_batch, 4);
    let v0 = fill(kv_per_batch, 5);
    let v1 = fill(kv_per_batch, 6);

    // ---- batch=2 call: Q/K/V are [batch, heads, seq, D] = element0 || element1 ----
    let mut q2 = q0.clone();
    q2.extend_from_slice(&q1);
    let mut k2 = k0.clone();
    k2.extend_from_slice(&k1);
    let mut v2 = v0.clone();
    v2.extend_from_slice(&v1);

    let q2_d = stream.memcpy_stod(&q2).map_err(|e| anyhow!("q2: {e:?}"))?;
    let k2_d = stream.memcpy_stod(&k2).map_err(|e| anyhow!("k2: {e:?}"))?;
    let v2_d = stream.memcpy_stod(&v2).map_err(|e| anyhow!("v2: {e:?}"))?;
    let mut o2_d = stream
        .alloc_zeros::<u16>(2 * q_per_batch)
        .map_err(|e| anyhow!("o2: {e:?}"))?;
    // pos_ptr is per-batch (read at pos_ptr[blockIdx.z]); both chunks start at start_pos.
    let pos2_d = stream
        .memcpy_stod(&[start_pos, start_pos])
        .map_err(|e| anyhow!("pos2: {e:?}"))?;

    flash_attn_bf16_v3_d256_gqa_causal_pos_dev(
        &ctx, &stream, &q2_d, &k2_d, &v2_d, &mut o2_d, &pos2_d, 2, nh, nkv, c, seq_kv, scale,
    )
    .map_err(|e| anyhow!("FA batch=2: {e:?}"))?;
    let o2 = stream
        .memcpy_dtov(&o2_d)
        .map_err(|e| anyhow!("o2 dtoh: {e:?}"))?;

    // ---- per-batch reference: batch=1 call on each element ----
    let run_single = |q: &[u16], k: &[u16], v: &[u16]| -> Result<Vec<u16>> {
        let qd = stream.memcpy_stod(q).map_err(|e| anyhow!("q1: {e:?}"))?;
        let kd = stream.memcpy_stod(k).map_err(|e| anyhow!("k1: {e:?}"))?;
        let vd = stream.memcpy_stod(v).map_err(|e| anyhow!("v1: {e:?}"))?;
        let mut od = stream
            .alloc_zeros::<u16>(q_per_batch)
            .map_err(|e| anyhow!("o1: {e:?}"))?;
        let pd = stream
            .memcpy_stod(&[start_pos])
            .map_err(|e| anyhow!("pos1: {e:?}"))?;
        flash_attn_bf16_v3_d256_gqa_causal_pos_dev(
            &ctx, &stream, &qd, &kd, &vd, &mut od, &pd, 1, nh, nkv, c, seq_kv, scale,
        )
        .map_err(|e| anyhow!("FA batch=1: {e:?}"))?;
        stream
            .memcpy_dtov(&od)
            .map_err(|e| anyhow!("o1 dtoh: {e:?}"))
    };
    let ref0 = run_single(&q0, &k0, &v0)?;
    let ref1 = run_single(&q1, &k1, &v1)?;

    // batch=2 element0 == batch=1(element0), bit-identical; same for element1.
    let mism0 = o2[..q_per_batch]
        .iter()
        .zip(ref0.iter())
        .filter(|(a, b)| a != b)
        .count();
    let mism1 = o2[q_per_batch..]
        .iter()
        .zip(ref1.iter())
        .filter(|(a, b)| a != b)
        .count();
    eprintln!(
        "batch=M/seq_q=C FA: element0 mismatches={mism0}/{q_per_batch}, element1={mism1}/{q_per_batch}"
    );
    assert_eq!(mism0, 0, "batch=2 element0 must bit-match batch=1");
    assert_eq!(mism1, 0, "batch=2 element1 must bit-match batch=1");

    // Sanity: the two elements are NOT identical (distinct inputs → distinct out),
    // so the test isn't trivially passing on zeros.
    let cross = o2[..q_per_batch]
        .iter()
        .zip(o2[q_per_batch..].iter())
        .filter(|(a, b)| a != b)
        .count();
    assert!(cross > 0, "distinct inputs should give distinct outputs");
    eprintln!(
        "✓ FA kernel is block-diagonal at batch=M, seq_q=C — no new FA kernel needed for #437"
    );
    Ok(())
}
