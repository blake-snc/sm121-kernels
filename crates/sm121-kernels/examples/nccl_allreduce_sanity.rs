//! NCCL AllReduce SUM sanity test.
//!
//! Validates that `NcclTransport` can construct a Comm and run a basic
//! collective. Falls back to single-node multi-process when only one DGX Spark
//! is reachable (`NCCL_SOCKET_IFNAME=lo`).
//!
//! # Hardware constraints
//!
//! NCCL refuses to colocate 2 ranks on 1 GPU ("Duplicate GPU detected" — hard
//! abort at init.cc:737). On a single-GPU DGX Spark (GB10), the only
//! single-machine sanity available is `WORLD_SIZE=1` self-test, which
//! validates dlopen + Comm construction + AllReduce execution path. True 2-rank
//! AllReduce requires either:
//!   - 2 physical DGX Sparks (intended target — set NCCL_SOCKET_IFNAME and
//!     point both ranks at the network interface), or
//!   - A multi-GPU machine (RTX 6000 Ada × 2, etc.).
//!
//! # Single-rank self-test
//! ```bash
//! LD_LIBRARY_PATH=/tmp/nccl-shim:$NCCL_PATH RANK=0 WORLD_SIZE=1 \
//!   NCCL_ID_FILE=/tmp/nccl.id cargo run --release --example nccl_allreduce_sanity
//! ```
//!
//! # 2-node invocation (intended production use)
//! ```bash
//! # Node 0:
//! RANK=0 WORLD_SIZE=2 NCCL_ID_FILE=/shared-nfs/nccl.id \
//!   NCCL_SOCKET_IFNAME=eth0 cargo run --release --example nccl_allreduce_sanity
//! # Node 1:
//! RANK=1 WORLD_SIZE=2 NCCL_ID_FILE=/shared-nfs/nccl.id \
//!   NCCL_SOCKET_IFNAME=eth0 cargo run --release --example nccl_allreduce_sanity
//! ```
//!
//! Expected output (each rank): `[N, N, ..., N]` where N = sum(1..=WORLD_SIZE).
//! For WORLD_SIZE=1: `[1, 1, ..., 1]`. For WORLD_SIZE=2: `[3, 3, ..., 3]`.

#![allow(clippy::arc_with_non_send_sync)]

use std::env;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::thread::sleep;
use std::time::{Duration, Instant};

use cudarc::driver::CudaContext;
use sm121_kernels::distributed::NcclTransport;

const N: usize = 1024;

fn main() -> anyhow::Result<()> {
    let rank: u32 = env::var("RANK")
        .map_err(|_| anyhow::anyhow!("RANK env var required"))?
        .parse()?;
    let world_size: u32 = env::var("WORLD_SIZE")
        .map_err(|_| anyhow::anyhow!("WORLD_SIZE env var required"))?
        .parse()?;
    let id_file = PathBuf::from(
        env::var("NCCL_ID_FILE").map_err(|_| anyhow::anyhow!("NCCL_ID_FILE env var required"))?,
    );

    println!("[rank {rank}/{world_size}] starting allreduce sanity ({N} f32 elements)");

    // ── 1. Get / publish the unique NCCL id ────────────────────────────
    let id = if rank == 0 {
        let id = NcclTransport::new_id()?;
        let bytes = id.internal();
        // c_char layout differs across platforms; serialize as raw bytes.
        let raw: Vec<u8> = bytes.to_vec();
        if id_file.exists() {
            let _ = fs::remove_file(&id_file);
        }
        fs::write(&id_file, &raw)?;
        println!("[rank 0] wrote 128-byte id to {}", id_file.display());
        id
    } else {
        // Wait for rank 0 to publish (up to 30s).
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            if id_file.exists() {
                let raw = fs::read(&id_file)?;
                if raw.len() == 128 {
                    let mut buf = [0 as std::ffi::c_char; 128];
                    for (i, &b) in raw.iter().enumerate() {
                        buf[i] = b as std::ffi::c_char;
                    }
                    println!("[rank {rank}] read 128-byte id from {}", id_file.display());
                    break NcclTransport::id_from_bytes(buf);
                }
            }
            if Instant::now() > deadline {
                anyhow::bail!("rank {rank} timed out waiting for {}", id_file.display());
            }
            sleep(Duration::from_millis(100));
        }
    };

    // ── 2. CUDA setup ─────────────────────────────────────────────────
    let ctx = CudaContext::new(0).map_err(|e| anyhow::anyhow!("CudaContext::new: {e:?}"))?;
    let stream = ctx.default_stream();

    // ── 3. Construct NcclTransport ────────────────────────────────────
    let transport = Arc::new(NcclTransport::new(stream.clone(), rank, world_size, id)?);
    println!("[rank {rank}] NcclTransport constructed");

    // ── 4. Allocate buffers, fill with rank+1, AllReduce SUM ──────────
    let send_host: Vec<f32> = vec![(rank + 1) as f32; N];
    let send_dev = stream
        .memcpy_stod(&send_host)
        .map_err(|e| anyhow::anyhow!("memcpy_stod: {e:?}"))?;
    let mut recv_dev = stream
        .alloc_zeros::<f32>(N)
        .map_err(|e| anyhow::anyhow!("alloc_zeros: {e:?}"))?;

    transport.all_reduce_sum_f32(&send_dev, &mut recv_dev)?;

    stream
        .synchronize()
        .map_err(|e| anyhow::anyhow!("stream sync: {e:?}"))?;

    // ── 5. Verify ──────────────────────────────────────────────────────
    let recv_host = stream
        .memcpy_dtov(&recv_dev)
        .map_err(|e| anyhow::anyhow!("memcpy_dtov: {e:?}"))?;

    // Sum of 1..=world_size
    let expected: f32 = (1..=world_size).map(|r| r as f32).sum();
    let mut errors = 0usize;
    for (i, &v) in recv_host.iter().enumerate() {
        if (v - expected).abs() > 1e-5 {
            if errors < 5 {
                eprintln!("[rank {rank}] MISMATCH at idx {i}: got {v} expected {expected}");
            }
            errors += 1;
        }
    }

    if errors == 0 {
        println!("[rank {rank}] PASS: AllReduce SUM = {expected} for all {N} elements");
    } else {
        eprintln!("[rank {rank}] FAIL: {errors} mismatches");
        std::process::exit(1);
    }

    // Rank 0 cleans up the shared id file.
    if rank == 0 {
        let _ = fs::remove_file(&id_file);
    }

    Ok(())
}
