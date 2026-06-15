# SM120/SM121 Architecture Guide for Kernel Developers

## Overview

SM120 (RTX 5090) and SM121 (DGX Spark GB10) are NVIDIA's consumer Blackwell GPUs. They share the Blackwell architecture generation with datacenter Blackwell (SM100/B200) but are different silicon — the SM120 core is the consumer design, which is why tcgen05/TMEM are absent. They lack several datacenter-exclusive features.

## What SM120 Has

| Feature | Instruction | Notes |
|---------|-------------|-------|
| BF16 MMA | `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` | SM80-era, 4 regs A, 2 regs B, 4 regs C/D |
| FP8 MMA | `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` | Unscaled FP8 MMA, 2x K vs BF16 (see `ptx/common/mma.ptxh`) |
| MXFP4 MMA | `mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m1.e2m1.f32.ue8m0` | MXFP4 (e2m1 + UE8M0 block scale). True NVFP4 (e2m1 with FP8 E4M3 per-16 scales) is implemented in software in `ptx/moe/gemm_nvfp4_fp8scale_grouped_mma.ptx`; hardware port pending. |
| TMA | `cp.async.bulk.tensor.{2d,3d,4d,5d}` | Global→shared, no multicast |
| CpAsync | `cp.async.cg.shared.global` | 16B granularity, register-bypass |
| Shared memory | 99 KB per thread block (100 KB/SM) | Less than SM80 (163 KB/block, 192 KB/SM) and SM100 (227 KB/block, 256 KB/SM) |
| Registers | 255 per thread | Architectural max (optimization_journey.md says 255 correctly) |
| Warp size | 32 threads | Standard |
| Max threads/block | 1024 | Standard |

## What SM120 Does NOT Have

| Feature | Available on | Impact |
|---------|-------------|--------|
| tcgen05 | SM100 only | No 256-bit MMA, no FA4 |
| WGMMA (`wgmma.mma_async`) | SM90a Hopper only | Replaced by tcgen05 on SM100; absent here |
| TMEM | SM100 only | No tensor memory, limits FA4/SM100 techniques |
| TMA multicast | SM90a/SM100 | No broadcast across SMs |
| Clusters | SM90a/SM100 | Only 1x1x1 cluster shape |
| Async GEMM | SM100 only | No overlapped GEMM-softmax |

## MMA Register Layouts

### BF16 m16n8k16

```
A-operand (4 regs): ldmatrix.x4 from SMEM
  Threads 0-15:  rows 0-15
  Threads 16-31: rows 0-15 (second 8x8 tile pair)
  Each thread: 4 x b32 regs = 8 bf16 elements

B-operand (2 regs): ldmatrix.x2 from SMEM
  8 rows, transposed layout
  Each thread: 2 x b32 regs = 4 bf16 elements

C/D-operand (4 regs): f32 accumulators
  Thread (groupID, tidInGroup) covers:
    row0 = groupID, row1 = groupID + 8
    col0 = tidInGroup * 2, col1 = tidInGroup * 2 + 1
```

### FP8 m16n8k32

```
A-operand (4 regs): {upper_k0, lower_k0, upper_k16, lower_k16}
  K is split into two halves, matching the PTX ISA m16n8k32 A fragment layout

B-operand (2 regs): K-split (b0→K first half, b1→K second half)
  This is the documented PTX ISA m16n8k32 B fragment layout (see the
  PTX ISA "Matrix Fragments for mma.m16n8k32" section), not a quirk.

Loading B: sB_f32[gid, tip] — index ordering follows the PTX ISA fragment layout
```

## SMEM Swizzle Patterns

### SWIZZLE_128B (used by TMA kernels)

```
swizzled_byte = byte_offset XOR ((row & 7) * 16)
```

16-byte granularity XOR swizzle eliminates bank conflicts for 128-byte rows.

### Non-swizzled (used by CpAsync kernels)

Standard row-major layout with 256-byte row stride (D=128 × 2 bytes).

## Key Optimization Techniques

### 1. TMA with Warp Specialization (V11, 30 TFLOPS BF16)
- 1 DMA warp handles TMA loads
- 4 MMA warps compute while DMA loads next tile
- mbarrier phase-based synchronization
- Double-buffered K/V (K0/K1, V0/V1)

### 2. VT-GMEM Layout (V12c, 100 TFLOPS FP8)
- Pre-transpose V in global memory: [D, B*H*Skv] instead of [B*H*Skv, D]
- Eliminates per-thread 4-load transpose in PV GEMM
- Single `ld.b32` per B-fragment instead of gather+shuffle

### 3. CpAsync All-Threads-Load (V3, 20 TFLOPS BF16)
- All 256 threads cooperatively load K/V
- `bar.sync` between load and compute phases
- No warp specialization overhead
- Natural fit for paged KV (per-row address computation)

### 4. Online Softmax with Fused Scale (V11)
- `scale_log2e = scale * LOG2E` precomputed
- `P = exp2(S * scale_log2e + neg_max_scaled)` — single FMA + exp2
- Eliminates 32 explicit FMUL per KV iteration

## Performance Reference (SM121a, B=1 H=16 S=2048 D=128)

> Shape note: this table is at **H=16** — the shape-matched cell against the CUTLASS
> CuTe DSL reference (see `cutlass_comparison.md`), so the vs-cuDNN / vs-CUTLASS columns
> are apples-to-apples. At the benchmark's default **H=32** shape the FP8 V12c kernel
> reads **~108 TFLOPS** (and BF16 V21 climbs with occupancy — see `optimization_journey.md`);
> do not compare the H=16 figures below against an H=32 headline number.

| Kernel | Dtype | TFLOPS | vs cuDNN | vs CUTLASS CuTe DSL |
|--------|-------|--------|----------|---------------------|
| V12c VT-GMEM | FP8 | 100 | +47% | +123% |
| V11 TMA | BF16 | 30 | — | -55% |
| V3 CpAsync | BF16 | 20 | — | -70% |

## Files

- PTX kernels: `ptx/attention/`, `ptx/gemm/`, `ptx/elementwise/`
- Shared headers: `ptx/common/` (mma.ptxh, convert.ptxh, reduction.ptxh)
- Rust dispatch: `crates/sm121-kernels/src/`
- Tests: `crates/sm121-kernels/tests/`
- SASS analysis: `scripts/sass_analysis.sh` (from SASS disassembly of the compiled cubins)
