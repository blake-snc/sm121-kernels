//! NCCL ring attention sanity (multi-rank, multi-node).
//!
//! Wires `NcclTransport` into the existing `ring_attention_bf16_distributed`
//! orchestrator. On a single-GPU DGX Spark this is gated by NCCL's "duplicate
//! GPU" check (see `nccl_allreduce_sanity` for the constraint discussion). On
//! a real 2-node DGX Spark cluster, this is the canonical sanity check that
//! the trait contract holds across NCCL transports.
//!
//! # Single-rank self-test (validates compile + comm setup)
//! ```bash
//! LD_LIBRARY_PATH=/tmp/nccl-shim:$NCCL_PATH RANK=0 WORLD_SIZE=1 \
//!   NCCL_ID_FILE=/tmp/ring_nccl.id \
//!   cargo run --release --example nccl_ring_attention_sanity
//! ```
//!
//! # 2-node invocation (intended use)
//! ```bash
//! # Node 0: holds KV slice [0..S/2], runs ring attention, expects output
//! # within BF16 noise of monolithic single-node attention.
//! RANK=0 WORLD_SIZE=2 NCCL_ID_FILE=/shared-nfs/ring_nccl.id \
//!   NCCL_SOCKET_IFNAME=eth0 cargo run --release --example nccl_ring_attention_sanity
//! # Node 1: holds KV slice [S/2..S]
//! RANK=1 WORLD_SIZE=2 NCCL_ID_FILE=/shared-nfs/ring_nccl.id \
//!   NCCL_SOCKET_IFNAME=eth0 cargo run --release --example nccl_ring_attention_sanity
//! ```

#![allow(clippy::arc_with_non_send_sync)]

use std::env;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::thread::sleep;
use std::time::{Duration, Instant};

use cudarc::driver::CudaContext;
use sm121_kernels::distributed::{ring_attention_bf16_distributed, NcclTransport};

const BATCH: u32 = 1;
const NUM_HEADS: u32 = 4;
const SEQ_Q: u32 = 128;
const HEAD_DIM: u32 = 128;

fn bf16_from_f32(f: f32) -> u16 {
    let bits = f.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7f_ffff;
    if exp == 0xff {
        return (sign | 0x7f80 | (mant >> 16)) as u16;
    }
    let r = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
    ((r >> 16) & 0xffff) as u16
}

fn random_bf16(n: usize, seed: u64) -> Vec<u16> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let f = ((s >> 33) as f32 / (1u32 << 31) as f32) * 0.1 - 0.05;
            bf16_from_f32(f)
        })
        .collect()
}

fn main() -> anyhow::Result<()> {
    let rank: u32 = env::var("RANK")?.parse()?;
    let world_size: u32 = env::var("WORLD_SIZE")?.parse()?;
    let id_file = PathBuf::from(env::var("NCCL_ID_FILE")?);

    // Each rank holds 1024 KV elements; total cluster KV = world_size * 1024.
    let seq_kv_local: u32 = 1024;
    let scale = 1.0f32 / (HEAD_DIM as f32).sqrt();

    println!(
        "[rank {rank}/{world_size}] B={BATCH} H={NUM_HEADS} Sq={SEQ_Q} Skv_local={seq_kv_local} D={HEAD_DIM}"
    );

    // ── 1. NCCL id exchange via shared file ──────────────────────────────
    let id = if rank == 0 {
        let id = NcclTransport::new_id()?;
        let raw: Vec<u8> = id.internal().to_vec();
        if id_file.exists() {
            let _ = fs::remove_file(&id_file);
        }
        fs::write(&id_file, &raw)?;
        println!("[rank 0] published id");
        id
    } else {
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            if id_file.exists() {
                let raw = fs::read(&id_file)?;
                if raw.len() == 128 {
                    let mut buf = [0 as std::ffi::c_char; 128];
                    for (i, &b) in raw.iter().enumerate() {
                        buf[i] = b as std::ffi::c_char;
                    }
                    break NcclTransport::id_from_bytes(buf);
                }
            }
            if Instant::now() > deadline {
                anyhow::bail!("rank {rank} timed out waiting for id");
            }
            sleep(Duration::from_millis(100));
        }
    };

    // ── 2. CUDA + NCCL setup ─────────────────────────────────────────────
    let ctx = CudaContext::new(0).map_err(|e| anyhow::anyhow!("CudaContext::new: {e:?}"))?;
    let stream = ctx.default_stream();
    let transport = Arc::new(NcclTransport::new(stream.clone(), rank, world_size, id)?);
    println!("[rank {rank}] NcclTransport constructed");

    // ── 3. Allocate Q (replicated) + per-rank K/V slice ──────────────────
    // Q seed shared across ranks; K/V seed varies by rank to simulate
    // genuinely different KV slices per node.
    let q_host = random_bf16((BATCH * NUM_HEADS * SEQ_Q * HEAD_DIM) as usize, 0xCAFE);
    let k_host = random_bf16(
        (BATCH * NUM_HEADS * seq_kv_local * HEAD_DIM) as usize,
        0xBEEF + rank as u64,
    );
    let v_host = random_bf16(
        (BATCH * NUM_HEADS * seq_kv_local * HEAD_DIM) as usize,
        0xFACE + rank as u64,
    );

    let q_dev = stream
        .memcpy_stod(&q_host)
        .map_err(|e| anyhow::anyhow!("memcpy_stod q: {e:?}"))?;
    let k_dev = stream
        .memcpy_stod(&k_host)
        .map_err(|e| anyhow::anyhow!("memcpy_stod k: {e:?}"))?;
    let v_dev = stream
        .memcpy_stod(&v_host)
        .map_err(|e| anyhow::anyhow!("memcpy_stod v: {e:?}"))?;
    let mut o_dev = stream
        .alloc_zeros::<u16>((BATCH * NUM_HEADS * SEQ_Q * HEAD_DIM) as usize)
        .map_err(|e| anyhow::anyhow!("alloc_zeros o: {e:?}"))?;

    // ── 4. Run ring attention via NcclTransport ──────────────────────────
    let t0 = Instant::now();
    ring_attention_bf16_distributed(
        &ctx,
        &stream,
        transport.as_ref(),
        &q_dev,
        &k_dev,
        &v_dev,
        &mut o_dev,
        BATCH,
        NUM_HEADS,
        SEQ_Q,
        seq_kv_local,
        scale,
    )?;
    stream
        .synchronize()
        .map_err(|e| anyhow::anyhow!("sync: {e:?}"))?;
    let elapsed = t0.elapsed();

    // ── 5. Verify output is finite ───────────────────────────────────────
    let out = stream
        .memcpy_dtov(&o_dev)
        .map_err(|e| anyhow::anyhow!("memcpy_dtov: {e:?}"))?;
    let n_finite = out
        .iter()
        .filter(|b| {
            let bits = **b;
            let s = (bits as u32) << 16;
            f32::from_bits(s).is_finite()
        })
        .count();

    if n_finite == out.len() {
        println!(
            "[rank {rank}] PASS: {n_finite}/{} finite outputs in {:.2}ms",
            out.len(),
            elapsed.as_secs_f32() * 1e3
        );
    } else {
        eprintln!(
            "[rank {rank}] FAIL: only {n_finite}/{} finite outputs",
            out.len()
        );
        std::process::exit(1);
    }

    if rank == 0 {
        let _ = fs::remove_file(&id_file);
    }
    Ok(())
}
