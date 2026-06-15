# sm121-kernels Implementation Guide

Step-by-step instructions for building the library on the DGX Spark. Each phase is self-contained with exact commands and gate criteria.

## Prerequisites

Run before starting any phase:

```bash
# Verify hardware and toolchain
bash scripts/verify_hardware.sh

# Initialize git repo (if not done)
cd /path/to/sm121-kernels
git init
git add -A
git commit -m "initial scaffold"
```

---

## Stage 0: Foundation (Week 1)

**Goal**: Validate the full pipeline from PTX source to kernel execution.

### Step 0.1: Create build.rs

Copy the `build.rs` code from `docs/design.md` Section 1 into `crates/sm121-kernels/build.rs`.

Test that it compiles:
```bash
# Should succeed even with no PTX files (empty registry)
cargo check
```

### Step 0.2: Write core Rust modules

Copy from `docs/design.md`:
- Section 1 (module.rs code) -> `crates/sm121-kernels/src/module.rs`
- Already scaffolded: `error.rs`, `device.rs`, `lib.rs`

Verify compilation:
```bash
cargo check
```

### Step 0.3: Create common PTX headers

Create these files from `docs/design.md` Section 2:
- `ptx/common/macros.ptxh`
- `ptx/common/reduction.ptxh`
- `ptx/common/smem_swizzle.ptxh`
- `ptx/common/convert.ptxh`
- `ptx/common/mbarrier_helpers.ptxh`

### Step 0.4: Write vector_add PTX kernel

Copy from `docs/design.md` Section 3 -> `ptx/test/vector_add.ptx`

Test assembly manually:
```bash
cpp -P -I ptx/common/ ptx/test/vector_add.ptx -o /tmp/vector_add.preprocessed.ptx
ptxas --gpu-name sm_121a -O3 --warn-on-spills -o /tmp/vector_add.cubin /tmp/vector_add.preprocessed.ptx
echo "ptxas succeeded"
cuobjdump -sass /tmp/vector_add.cubin | head -40
```

### Step 0.5: Build and test

```bash
cargo build
```

This should:
1. build.rs discovers `ptx/test/vector_add.ptx`
2. Preprocesses it via cpp
3. Assembles via ptxas -> cubin
4. Generates `embedded_kernels.rs` with `vector_add` entry
5. Compiles the Rust crate with embedded cubin

### Step 0.6: Write integration test

Copy from `docs/design.md` Section 3 -> `tests/integration/test_vector_add.rs`

Run:
```bash
cargo test test_vector_add
```

### Gate

- [x] `ptxas` successfully assembles vector_add.ptx targeting sm_121a
- [x] `cargo build` completes without errors
- [x] `cargo test test_vector_add` passes: kernel loads, launches, produces correct output

**Commit**: `git commit -am "phase 0: build system + vector_add pipeline validation"`

---

## Stage 1: Elementwise Kernels (Weeks 2-3)

**Goal**: Implement all 5 elementwise kernels, pass correctness tests.

### Step 1.1: Generate golden test vectors

```bash
python tests/reference/generate_golden.py
ls tests/reference/data/
# Should see: rmsnorm_bf16_*.npz, silu_mul_*.npz, rope_*.npz
```

### Step 1.2: RMSNorm kernel

1. Write `ptx/elementwise/rmsnorm_bf16.ptx` following design doc Section 6.1
   - Key: warp butterfly reduction, cross-warp shared memory reduction, rsqrt
   - Test assembly: `ptxas --gpu-name sm_121a -O3 ptx/elementwise/rmsnorm_bf16.ptx`
2. Write `crates/sm121-kernels/src/norm/mod.rs` dispatch wrapper
3. Uncomment `pub mod norm;` in `lib.rs`
4. Write `tests/integration/test_norm.rs` loading `rmsnorm_bf16_*.npz`
5. Run: `cargo test test_rmsnorm`

### Step 1.3: RoPE kernel

1. Write `ptx/elementwise/rope_bf16.ptx` following design doc Section 6.2
   - Key: cos/sin from precomputed cache, rotation formula
2. Write `crates/sm121-kernels/src/rope/mod.rs`
3. Write `tests/integration/test_rope.rs`
4. Run: `cargo test test_rope`

### Step 1.4: Activation fusion kernels

1. Write `ptx/elementwise/silu_mul_bf16.ptx` following design doc Section 6.3
   - Key: vectorized 128-bit loads, SIGMOID_F32 macro, fused multiply
2. Write `ptx/elementwise/gelu_mul_bf16.ptx`
   - Key: erf approximation or polynomial
3. Write `ptx/elementwise/gelu_tanh_mul_bf16.ptx`
   - Key: tanh via exp, 0.044715 coefficient
4. Write `crates/sm121-kernels/src/activation/mod.rs` dispatching all 3
5. Write `tests/integration/test_activation.rs`
6. Run: `cargo test test_activation`

### Step 1.5: C API for elementwise

1. Write `crates/sm121-kernels/src/ffi/types.rs` and `crates/sm121-kernels/src/ffi/mod.rs`
   - `spark_rmsnorm`, `spark_rope`, `spark_activation`
2. Uncomment `pub mod ffi;` in `lib.rs`
3. Build: `cargo build --features c-api`

### Gate

- [x] All 5 elementwise kernels (RMSNorm, RoPE, SiLU*mul, GeLU*mul, GeLU-tanh*mul) pass integration tests
- [x] Tolerance: BF16 max_abs_diff < 1e-2 vs PyTorch reference
- [x] `compute-sanitizer --tool memcheck cargo test` reports no errors

**Commit**: `git commit -am "phase 1: elementwise kernels (rmsnorm, rope, activations)"`

---

## Stage 2: GEMM Kernels (Weeks 4-6)

**Goal**: Implement GEMM for FP8, NVFP4, and W4A16. Achieve >50% theoretical throughput on FP8.

### Step 2.1: GEMM FP8

1. Write `ptx/gemm/gemm_fp8_128x128x64.ptx` following design doc Section 5.1
   - Start with single-stage pipeline, then add 3-stage
   - Key: `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`
   - Key: `cp.async.cg.shared.global` for async loads, `cp.async.commit_group`/`wait_group` for pipelining
   - Shared memory swizzling via SWIZZLE_ADDR macro
2. Write `crates/sm121-kernels/src/gemm/mod.rs` and `launch.rs`
3. Write `tests/integration/test_gemm.rs`
4. Run: `cargo test test_gemm_fp8`

**Debugging tips**:
- Use `cuobjdump -sass` to verify MMA instructions are emitted
- Use `ncu --set full` to check occupancy, register count, memory throughput
- If register spills: reduce warp tile size or add `.maxnreg` directive

### Step 2.2: GEMM NVFP4

1. Write `ptx/gemm/gemm_nvfp4_128x128x128.ptx` following design doc Section 5.2
   - Requires `.target sm_121a` and CUDA 13+ ptxas
   - Key: `mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32`
   - Block scale factors in `ue8m0` format
2. Add dispatch in `gemm/mod.rs`
3. Test: `cargo test test_gemm_nvfp4`

**If ptxas rejects the instruction**: Check CUDA toolkit version. NVFP4 block-scaled MMA on sm_121a requires CUDA 13.x. If unavailable, defer this kernel and document the requirement.

### Step 2.3: GEMM W4A16

1. Write `ptx/gemm/gemm_w4a16_128x128x64.ptx` following design doc Section 5.3
   - Key: 4-bit unpack + dequantize to BF16 in registers/SMEM
   - Then standard `mma.sync.aligned.m16n8k16` BF16 path
2. Test: `cargo test test_gemm_w4a16`

### Step 2.4: Benchmarks

1. Write `benches/common.rs` from design doc Section 9
2. Write `benches/gemm.rs` benchmarking all GEMM variants
3. Run: `cargo bench gemm`
4. Compare FP8 TFLOPS against cuBLAS:
   ```bash
   # Quick cuBLAS baseline (if available via Python)
   python -c "
   import torch
   a = torch.randn(4096, 4096, dtype=torch.float8_e4m3fn, device='cuda')
   b = torch.randn(4096, 4096, dtype=torch.float8_e4m3fn, device='cuda')
   # warmup
   for _ in range(10): c = torch._scaled_mm(a, b, scale_a=torch.tensor(1.0), scale_b=torch.tensor(1.0))
   torch.cuda.synchronize()
   import time
   start = time.time()
   for _ in range(100): c = torch._scaled_mm(a, b, scale_a=torch.tensor(1.0), scale_b=torch.tensor(1.0))
   torch.cuda.synchronize()
   elapsed = (time.time() - start) / 100
   flops = 2 * 4096**3
   print(f'cuBLAS FP8 GEMM: {flops / elapsed / 1e12:.2f} TFLOPS')
   "
   ```

### Gate

- [x] FP8 GEMM produces correct results vs PyTorch reference
- [x] FP8 GEMM achieves >50% of cuBLAS throughput (or theoretical MMA peak)
- [x] NVFP4 and W4A16 pass correctness (if CUDA version supports them)

**Commit**: `git commit -am "phase 2: GEMM kernels (fp8, nvfp4, w4a16)"`

---

## Stage 3: Flash Attention (Weeks 6-9)

**Goal**: Implement flash attention for BF16 and FP8 with causal/non-causal support.

### Step 3.1: FA BF16 non-causal

This is the most complex kernel. Follow design doc Section 4 carefully.

1. Write `ptx/attention/fa_bf16_m128_d128.ptx`
   - Start with a simplified version: no pipelining, single K/V buffer
   - Get correctness first, optimize second
   - Key sections from design doc:
     - 4.4.1: Kernel header + thread setup
     - 4.4.4: Load Q to registers via ldmatrix.x4
     - 4.4.5: S = Q @ K^T MMA loop
     - 4.4.6: Online softmax
     - 4.4.7: O += P @ V MMA
     - 4.4.10: Epilogue (write O, LSE)
2. Write `crates/sm121-kernels/src/attention/mod.rs` and `launch.rs`
3. Test:
   ```bash
   python tests/reference/generate_golden.py  # if not already done
   cargo test test_flash_attention_bf16_noncausal
   ```

**Debugging strategy**:
- First test with tiny sizes: seq=16, d=128, 1 head, 1 batch
- Print intermediate values by writing to a debug output buffer
- Compare S matrix (QK^T) against PyTorch before softmax
- Compare P matrix (after softmax) against PyTorch
- Compare final O against PyTorch
- Increase sizes gradually: 64, 128, 256, 512, 1024, 2048

### Step 3.2: FA BF16 causal

1. Copy `fa_bf16_m128_d128.ptx` to `fa_bf16_m128_d128_causal.ptx`
2. Add causal masking from design doc 4.4.8
3. Add early termination: skip KV blocks entirely when all positions are masked
4. Test: `cargo test test_flash_attention_bf16_causal`

### Step 3.3: Add K double-buffering

Optimize the BF16 kernels:
1. Double-buffer K in SMEM: while computing QK^T with K_buf[i], load next K into K_buf[1-i]
2. Overlap V load with softmax computation
3. Re-run benchmarks to measure improvement

### Step 3.4: FA FP8 variants

1. Write `ptx/attention/fa_fp8_m128_d128.ptx` and `fa_fp8_m128_d128_causal.ptx`
   - Key change: `mma.sync.aligned.m16n8k32` instead of m16n8k16
   - Bc doubles to 128 (FP8 elements are 1 byte)
   - 4 warps (128 threads), WARP_Q=32
2. Test: `cargo test test_flash_attention_fp8`

### Step 3.5: Variable sequence length

Add `cu_seqlens_q` and `cu_seqlens_k` support:
1. Load sequence boundaries from cumulative length arrays
2. Clamp loads to avoid OOB
3. Early-exit blocks beyond actual sequence length
4. Test with ragged batches

### Step 3.6: Benchmarks

```bash
cargo bench attention
```

Compare against PyTorch SDPA:
```python
import torch
from torch.nn.functional import scaled_dot_product_attention

q = torch.randn(1, 32, 2048, 128, dtype=torch.bfloat16, device='cuda')
k = torch.randn(1, 32, 2048, 128, dtype=torch.bfloat16, device='cuda')
v = torch.randn(1, 32, 2048, 128, dtype=torch.bfloat16, device='cuda')

# warmup
for _ in range(10):
    o = scaled_dot_product_attention(q, k, v, is_causal=True)
torch.cuda.synchronize()

import time
start = time.time()
for _ in range(100):
    o = scaled_dot_product_attention(q, k, v, is_causal=True)
torch.cuda.synchronize()
elapsed = (time.time() - start) / 100
flops = 4 * 1 * 32 * 2048 * 2048 * 128
print(f"PyTorch SDPA: {flops / elapsed / 1e12:.2f} TFLOPS ({elapsed*1000:.2f} ms)")
```

### Gate

- [x] FA BF16 causal matches PyTorch reference within tolerance (1e-2)
- [x] FA BF16 causal achieves >80% of theoretical MMA throughput for seq >= 1024
- [x] FA FP8 variants pass correctness
- [x] Variable sequence length works correctly

**Commit**: `git commit -am "phase 3: flash attention (bf16, fp8, causal, varlen)"`

---

## Stage 4: MoE + Sampling + Polish (Weeks 9-11)

### Step 4.1: Top-k sampling

1. Write `ptx/elementwise/topk_sampling.ptx`
2. Write `crates/sm121-kernels/src/sampling/mod.rs`
3. Test with various k values (1, 5, 10, 50)

### Step 4.2: MoE routing

1. Write `ptx/elementwise/moe_routing.ptx`
2. Write `crates/sm121-kernels/src/moe/mod.rs`
3. Test with typical configs: 8 experts top-2, 64 experts top-8

### Step 4.3: Complete C API

Add remaining functions to `ffi/mod.rs`:
- `spark_gemm`
- `spark_flash_attention`
- `spark_moe_routing`
- `spark_topk_sample`

### Step 4.4: Full benchmark suite

```bash
cargo bench
```

Write a summary table comparing all kernels against baseline (PyTorch/cuBLAS).

### Step 4.5: Documentation

- README.md with usage examples
- API documentation via `cargo doc`
- C API example (`examples/c_api_demo.c`)
- Rust examples (`examples/basic_gemm.rs`, `examples/flash_attention.rs`)

### Gate

- [x] All kernels have passing tests
- [x] Full benchmark results documented
- [x] C API compiles and works from C example

**Commit**: `git commit -am "phase 4: moe routing, top-k sampling, full benchmarks"`

---

## Stage 5: Release (Weeks 11-12)

### Step 5.1: Generate C header

```bash
cbindgen --crate sm121-kernels --output include/sm121_kernels.h
```

### Step 5.2: Prepare for crates.io

- Verify `Cargo.toml` metadata (description, repository, keywords)
- Run `cargo publish --dry-run`

### Step 5.3: Performance writeup

Document:
- Methodology (CUDA events, warmup, iterations)
- Results table: kernel, config, TFLOPS, GB/s, % peak
- Comparison vs PyTorch/cuBLAS
- SM121-specific insights and optimizations

### Step 5.4: vLLM plugin (stretch)

If time permits, create a Python package wrapping the C API:
- `pip install sm121-kernels`
- Exposes kernels as drop-in replacements for vLLM's default paths
- Uses ctypes or PyO3 to call the C API

**Commit**: `git commit -am "phase 5: release prep, cbindgen header, docs"`

---

## Troubleshooting

### ptxas rejects an instruction

```
ptxas error   : Feature not supported on target architecture
```

Check the `.target` directive in your PTX file. Some instructions require `sm_121a` (architecture-accelerated) rather than `sm_121`. Block-scaled FP4 MMA is the most common case.

### Register spills

```
ptxas warning : Function uses N registers, spilling to local memory
```

Reduce register pressure:
1. Lower the warp tile size (fewer accumulator registers)
2. Add `.maxnreg N` directive after `.entry`
3. Increase warp count to reduce per-warp work
4. Move some data from registers to shared memory

### cubin load fails

If `dev.load_ptx(Ptx::Image(...))` fails, verify:
1. The cubin was compiled for the correct architecture (`sm_121a`)
2. The CUDA driver version supports sm_121 (CUDA 13+)
3. The cubin bytes are not corrupted (check file size in `embedded_kernels.rs`)

### Wrong results

Debugging PTX kernels:
1. Add a debug output buffer parameter to the kernel
2. Write intermediate values (S matrix rows, softmax row_max, etc.) to the buffer
3. Load the buffer back to host and print
4. Compare step-by-step against PyTorch reference
5. Use `compute-sanitizer --tool memcheck` to catch OOB access
6. Use `compute-sanitizer --tool initcheck` to find uninitialized memory reads

### Bank conflicts

Profile with ncu:
```bash
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld cargo bench -- <kernel>
```

If conflicts > 0, review SMEM swizzling. The XOR pattern `addr ^ ((row & 7) << 4)` should eliminate conflicts for 128-byte row strides. Adjust the shift amount for different row strides.
