# Kernel Inventory

Hand-written SM121a PTX kernels in sm121-kernels, organized by category.

Last updated: 2026-06-02

## Status legend (authoritative)
- **Production** — maintained, benchmarked hot path.
- **Production (reference)** — correct + tested, kept for model coverage; not the perf-tuned hot path.
- **Experimental** — superseded optimization generations kept for benchmarking/history. Gated
  behind the `experimental` cargo feature; not part of the default public API surface.
- **Archived** — earlier generation retained only as reference; same gate as Experimental.
- **Reverted** — assembled but proven incorrect/unsafe, NOT built into the default crate
  (e.g. NVFP4 GEMV decode was measured faster per-kernel but flips greedy argmax at tol=0;
  not shipped as a decode path).

## Codegen note
The three production FA families are now generated from cpp-`#ifdef` templates, not maintained
as separate hand files: `ptx/attention/templates/{fa_bf16_v3_d256, fa_bf16_v21, fa_fp8_v12c_vt}.ptx.in`
+ `.variants` manifests (21 variants total). `build.rs` expands them; the original hand-PTX is
frozen in `ptx/attention/archive/` as the cubin-identity reference (proven byte-identical via
`scripts/check_codegen_identity.sh`, gated in CI). The `swa`/`softcap` v21 variants are not
templated (non-additive mask/softmax) and remain hand files.

## Measured perf (corrected, 2026-06-02 benchmark — supersedes older inline numbers below)
- FP8 `fa_fp8_v12c_vt`: **~100-115 TFLOPS** (to our knowledge, the fastest open-source FP8-input
  SM120 FA we measured on GB10/SM121 at these shapes as of 2026-06; 2.2x over the CUTLASS FP8 path
  at ~45 — note that CUTLASS FP8 path is impaired by issue #3044).
- BF16 `fa_bf16_v21_streaming_p`: **occupancy-sensitive, scales strongly with batch×heads×seq.**
  At the small benchmark default (B=1 H=32 / B=2 H=16) it reads ~12 TFLOPS @ S=1024 and
  ~35 @ S=2048. With more work in flight (B=2, H=32) the same kernel measures **~48 @
  S=2048, ~69 @ S=4096, and ~75 TFLOPS @ S=8192** (non-causal dense, `4·B·H·S²·D`, CUDA
  events; `~` because GB10 has no clock lock). V21 is the long-context BF16 kernel; the low
  single-shape numbers are occupancy, not the kernel's ceiling.
- BF16 `fa_bf16_v22_db` (NEW, double-buffered v3): best short-context BF16 (~21 TFLOPS @ S=1024),
  but it does **not** scale to long context (~24 TFLOPS at S=4096–8192); use V21 there.
- BF16 vs the CUTLASS CuTe-DSL reference (67.3 TFLOPS @ B=1 H=16 S=2048): the ~0.45× ratio is
  a **small-shape** figure. At long context V21's ~75 TFLOPS lands within ~13% of upstream
  flash-attention's FA4 forward (86.9 TFLOPS dense / 74.6 causal at S=8192 H32 B2, measured on
  this same GB10), so the BF16 gap is occupancy at short sequences, not a structural ceiling of
  hand-written PTX. v22 double-buffering remains the best short-context variant.

## Flash Attention — BF16

| Kernel | Dims | Features | Status |
|--------|------|----------|--------|
| `fa_bf16_d128.ptx` | d=128 | Legacy CpAsync baseline | Experimental |
| `fa_bf16_d128_causal.ptx` | d=128 | + causal | Experimental |
| `fa_bf16_v3_d128.ptx` | d=128 | CpAsync 3-stage pipeline | Archived |
| `fa_bf16_v3_d128_causal.ptx` | d=128 | + causal | Archived |
| `fa_bf16_v3_paged_kv.ptx` | d=128 | Paged KV cache | Production |
| `fa_bf16_v3_paged_kv_causal.ptx` | d=128 | + causal | Production |
| `fa_bf16_v3_split_kv.ptx` | d=128 | FlashDecoding split-K | Production |
| `fa_bf16_v3_split_kv_causal.ptx` | d=128 | + causal | Production |
| `fa_bf16_v3_split_paged_kv.ptx` | d=128 | split-K + paged | Production |
| `fa_bf16_v3_split_paged_kv_causal.ptx` | d=128 | + causal | Production |
| `fa_bf16_v11_*.ptx` | d=128 | TMA warp-spec baseline (8 variants) | Archived |
| `fa_bf16_v3_d128.ptx` | d=128 | CpAsync all-warps base for v22 | Experimental |
| `fa_bf16_v22_db.ptx` | d=128 | **v3 + double-buffered async pipeline** — best short-context BF16 (~21 TFLOPS @ S=1024) | Production |
| `fa_bf16_v21_streaming_p.ptx` | d=128 | **Production BF16** (~31 TFLOPS @ S=2048; see corrected perf above) | Production |
| `fa_bf16_v21_causal.ptx` | d=128 | V21 + causal | Production |
| `fa_bf16_v21_gqa.ptx` | d=128 | V21 + GQA | Production |
| `fa_bf16_v21_gqa_causal.ptx` | d=128 | V21 + GQA + causal | Production |
| `fa_bf16_v21_varlen.ptx` | d=128 | V21 + cu_seqlens | Production |
| `fa_bf16_v21_varlen_causal.ptx` | d=128 | V21 + varlen + causal | Production |
| `fa_bf16_v21_varlen_gqa.ptx` | d=128 | V21 + varlen + GQA | Production |
| `fa_bf16_v21_varlen_gqa_causal.ptx` | d=128 | full triple | Production |
| `fa_bf16_v21_softcap_causal.ptx` | d=128 | **Gemma 2/3 global** — softcap + causal | Production |
| `fa_bf16_v21_swa.ptx` | d=128 | **Gemma 2/3 local, Mistral** — sliding window | Production |
| `fa_bf16_v21_swa_softcap.ptx` | d=128 | Gemma 2 local — SWA + softcap | Production |
| `fa_bf16_mla_decode.ptx` | D_C=512,D_R=64 | **DeepSeek V3 MLA decode** | Production (reference) |
| `fa_bf16_mla_prefill.ptx` | D_C=512,D_R=64 | **DeepSeek V3 MLA prefill** (causal) | Production (reference) |
| `fa_bf16_mla_decode_paged.ptx` | D_C=512,D_R=64 | MLA decode + paged KV cache (vLLM/SGLang) | Production (reference) |
| `fa_bf16_tree.ptx` | d=128 | **EAGLE-3 / Medusa tree attention** (custom mask) | Production (reference) |

## Flash Attention — FP8 (E4M3)

| Kernel | Features | Status |
|--------|----------|--------|
| `fa_fp8_d128.ptx` | d=128 baseline | Archived |
| `fa_fp8_d128_causal.ptx` | + causal | Archived |
| `fa_fp8_v11_tma.ptx` | TMA + warp-spec | Archived |
| `fa_fp8_v12a_transpose.ptx` | Transposed V | Archived |
| `fa_fp8_v12c_vt.ptx` | **Production FP8 — ~108 TFLOPS** at the H=32 default shape (~100-115 across shapes; to our knowledge the fastest open-source FP8-input SM120 FA we measured on GB10/SM121 as of 2026-06) | Production |
| `fa_fp8_v12c_vt_causal.ptx` | + causal | Production |
| `fa_fp8_v12c_vt_gqa.ptx` | + GQA | Production |
| `fa_fp8_v12c_vt_gqa_causal.ptx` | + GQA + causal | Production |
| `fa_fp8_v12c_vt_varlen.ptx` | + varlen | Production |
| `fa_fp8_v12c_vt_varlen_causal.ptx` | + varlen + causal | Production |
| `fa_fp8_v12c_vt_varlen_gqa.ptx` | + varlen + GQA | Production |
| `fa_fp8_v12c_vt_varlen_gqa_causal.ptx` | full triple | Production |
| `fa_fp8_v3_paged_kv*.ptx` | Paged KV (2 variants) | Production |
| `fa_fp8_v3_split_paged_kv*.ptx` | split-K + paged (2 variants) | Production |
| `fa_fp8_mla_decode.ptx` | **DeepSeek V3 MLA FP8 KV decode** | Production (reference) |

## Linear attention (Qwen3-Next, Nemotron-H)

| Kernel | Purpose | Status |
|--------|---------|--------|
| `gdn_decode.ptx` | Gated DeltaNet decode (Qwen3-Next 80B) | Production (reference) |
| `gdn_decode_tma.ptx` | GDN decode with TMA bulk state load/store — replaces the scalar kernel's un-coalesced per-element state reads | Production |
| `gdn_prefill.ptx` | Gated DeltaNet prefill (sequential reference) | Production (reference) |
| `gdn_decode_backward_bf16.ptx` | Backward for the single-token GDN recurrent state update | Production (reference) |
| `linear_attn_chunk_prefill.ptx` | Chunk-parallel linear-attention prefill: Y_intra + Y_inter + state update, host threads state chunk-to-chunk | Production |
| `linear_attn_state_update_mma.ptx` | MMA-accelerated chunk state update `S_new = S_init + V^T @ K` | Production |
| `linear_attn_y_inter_mma.ptx` | MMA-accelerated inter-chunk output `Y_inter = Q @ S^T` (with state_update, ~80% of per-chunk FLOPs in MMA) | Production |
| `conv1d_backward_dx_bf16.ptx` | Backward (dx) for the depthwise causal conv1d (k=4) in GDN layers | Production (reference) |
| `conv1d_backward_dw_bf16.ptx` | Backward (dW) for the depthwise causal conv1d (k=4) | Production (reference) |
| `mamba2_selective_scan_decode.ptx` | Mamba2 SSM decode (Nemotron-H) | Production (reference) |
| `mamba2_selective_scan_prefill.ptx` | Mamba2 selective scan prefill (sequential reference, state in per-thread registers) | Production (reference) |

## KV cache utilities

| Kernel | Purpose |
|--------|---------|
| `kv_cache_fp8_write.ptx` | Fused BF16→FP8 paged KV cache write |
| `kv_cache_nvfp4_write.ptx` | Fused BF16→NVFP4 quantize + paged KV write (E2M1 packed 2/byte, FP8 E4M3 scale per 16-element block) |
| `kv_append_bf16.ptx` | Single-tile BF16 append at a device-resident position — replaces a dtod memcpy, CUDA-Graph-capturable |
| `kv_append_multihead_bf16_pos_dev.ptx` | Single-token per-head K/V append, all KV heads in one launch, position from device pointer |
| `kv_append_strided_bf16.ptx` | Strided K+V append into a head-major cache — one launch replaces 2·n_kv dtod memcpys |
| `kv_append_strided_bf16_pos_dev.ptx` | Position-from-device-pointer variant (replayable under a CUDA Graph) |
| `kv_append_strided_fp8.ptx` | Fused BF16→FP8 quantize + strided append into a head-major FP8 KV cache |
| `kv_append_strided_fp8_pos_dev.ptx` | Position-from-device-pointer variant of the FP8 append |
| `paged_kv_gather_bf16.ptx` | Materialize paged KV into contiguous per-sequence layout (for the backward orchestrator) |
| `paged_kv_scatter_atomic_bf16.ptx` | Inverse of the gather: scatter dK/dV from contiguous layout back into paged layout |

## GEMM

| Kernel | Format | Notes |
|--------|--------|-------|
| `gemm_bf16_*.ptx` | BF16 MMA m16n8k16 | Multiple variants |
| `gemm_fp8_*.ptx` | FP8 E4M3 MMA m16n8k32 | Per-tensor scale |
| `gemm_nvfp4_*.ptx` | NVFP4 block-scaled MMA | 16-block scales |
| `gemm_w4a16_*.ptx` | W4A16 quantized weights | Register dequant |
| `gemv_bf16_split_k_w16.ptx` | Wide-block BF16 split-K GEMV | 16 cols/thread, 2048 cols/block — for large-N lm_head in BF16-only mode |

## MoE (Grouped GEMM pipeline)

| Kernel | Purpose |
|--------|---------|
| `moe_histogram.ptx` | Per-expert token count (atomics) |
| `moe_expert_offsets.ptx` | Prefix sum over histogram → offsets table |
| `moe_permute.ptx` | Atomic cursor scatter; records inverse_index |
| `moe_unpermute.ptx` | Gather+weight with FP32 atomic add |
| `gemm_bf16_grouped.ptx` | MoE grouped GEMM (reference scalar) |
| `gemm_bf16_grouped_sparse.ptx` | Sparse variant of `gemm_bf16_grouped` — expert_id indirected via `active_experts`, grid z = non-empty experts only |
| `gemm_bf16_grouped_mma.ptx` | MoE grouped GEMM with MMA m16n8k16 — 32×32 tile, ldmatrix |
| `gemm_bf16_grouped_mma_sparse.ptx` | Sparse variant of `gemm_bf16_grouped_mma` — skips empty-expert CTAs at batch=1 / top_k ≪ num_experts |
| `gemm_fp8_grouped_mma.ptx` | Grouped GEMM with FP8 weights + per-expert scalar scale (applied at epilogue) |
| `gemm_fp8_grouped_mma_v2.ptx` | v2: vectorized B loads (16 FP8 per `v4.b32`) |
| `gemm_fp8_block128_grouped_mma.ptx` | Grouped GEMM with DeepSeek-V3 1×128 block-scaled FP8 weights — FP8→BF16 decode + scale on SMEM staging, pure-BF16 MMA inner loop |
| `gemm_fp8_block128_grouped_mma_v2.ptx` | v2: vectorized B loads + per-block scale hoisted into a register at K-block boundary |
| `gemm_mxfp8_grouped_mma.ptx` | Grouped GEMM with MXFP8 weights (E4M3 values + UE8M0 scale per 32-element K-block) |
| `gemm_mxfp8_grouped_mma_v2.ptx` | v2: vectorized B loads + hoisted UE8M0 scale |
| `gemm_mxfp4_grouped_mma.ptx` | Grouped GEMM with MXFP4 weights (E2M1 nibble-packed + UE8M0 block scales) — gpt-oss-120b path |
| `gemm_mxfp4_grouped_mma_v2.ptx` | v2: contiguous 16-element FP4 loads via `ld.global.b64` + hoisted scale |
| `gemm_nvfp4_grouped_mma.ptx` | Grouped GEMM with NVFP4 weights (E2M1 + FP32 scale per 16-K block), SMEM LUT decode — GLM-5 |
| `gemm_nvfp4_fp8scale_grouped_mma.ptx` | NVFP4 grouped GEMM with FP8 E4M3 block scales (NVIDIA/CUTLASS convention; matches `quant_bf16_to_nvfp4` output) |
| `gemm_nvfp4_fp8scale_grouped_mma_v2.ptx` | v2: vectorized B loads + FP8 scale hoisted at K-block boundary |
| `gemm_w8a16_grouped_mma.ptx` | MoE grouped GEMM with FP8 weights + per-tensor scale; MMA. Used for Gemma-4-26B-A4B batched serving (71.4× at M=128 — in-stack scaling figure vs single-user M=1) |
| `gemv_w8a16_grouped_split_k.ptx` | Per-slot grouped W8A16 GEMV (M×top_k active slots) — alternative when MMA tile padding is too high |
| `gemv_bf16_grouped_split_k.ptx` | BF16 sibling — used by DSV2-Lite |
| `gemv_bf16_grouped_split_k_dual.ptx` | Dual M=1 split-K GEMV: gate AND up weight stacks in one launch (shared x, K, N, expert IDs) |
| `broadcast_top_k_bf16.ptx` | Replicate `[M, hidden]` → `[M*top_k, hidden]` for cross-seq grouped GEMV |
| `moe_active_experts_compact.ptx` | Build the compact non-empty expert-ID list from the histogram on-device (drops the host roundtrip for sparse dispatch) |
| `moe_device_counts.ptx` | Expert-parallel: per-destination-device token counts for NCCL alltoall sizing |
| `moe_device_permute.ptx` | Pack activations grouped by target device via an arbitrary expert→device table (alltoall send buffer) |
| `moe_routing.ptx` | softmax-over-top-k routing (single launch, batched over M tokens) |
| `moe_route_decode_full.ptx` | Full-softmax variant (GDN-hybrid family) |
| `moe_apply_per_expert_scale_bf16.ptx` | Multiply top_k_w by per-expert scale |

## Quantization

| Kernel | Format |
|--------|--------|
| `quant_bf16_to_fp8_pertoken.ptx` | FP8 E4M3 per-token scale |
| `dequant_fp8_bf16.ptx` | FP8 → BF16 |
| `quant_bf16_to_fp8_block128.ptx` | DSv3 1×128 block-scaled FP8 |
| `dequant_fp8_block128_bf16.ptx` | Inverse |
| `quant_bf16_to_mxfp8.ptx` | MXFP8 32-element blocks + UE8M0 scales |
| `dequant_mxfp8_bf16.ptx` | Inverse |
| `quant_bf16_to_mxfp4.ptx` | MXFP4 32-block + UE8M0 (sm_121f) |
| `dequant_mxfp4_bf16.ptx` | Inverse |
| `quant_bf16_to_nvfp4.ptx` | NVFP4 16-block + FP8 scales |
| `dequant_nvfp4_bf16.ptx` | Inverse |

## Elementwise / Serving primitives

| Kernel | Purpose |
|--------|---------|
| `rope_bf16.ptx` | RoPE rotation (vectorized) |
| `rope_partial_bf16_pos_dev.ptx` / `_per_seq.ptx` | Position from device pointer (single / per-seq M) — continuous batching |
| `rope_proportional_bf16_pos_dev.ptx` / `_per_seq.ptx` | Gemma-4 full-attn proportional RoPE (single / per-seq) |
| `bf16_to_f32.ptx` / `f32_to_bf16.ptx` | Elementwise dtype casts |
| `f32_weighted_sum_to_bf16.ptx` | Reduce f32 rows × bf16 weights → bf16 |
| `add_bf16.ptx` | Elementwise BF16 add (residual streams) |
| `silu_mul_bf16.ptx` | Fused SiLU × gate |
| `gelu_mul_bf16.ptx` | Fused GeLU × gate |
| `gelu_tanh_mul_bf16.ptx` | GeLU-tanh × gate |
| `rmsnorm_bf16.ptx` | RMSNorm |
| `rmsnorm_residual_bf16.ptx` | Fused residual add + RMSNorm |
| `rmsnorm_bf16_fp8out.ptx` | RMSNorm with FP8 output |
| `softmax_bf16.ptx` | Standalone row-wise softmax |
| `embedding_lookup_bf16.ptx` | Token embedding gather |
| `cross_entropy_bf16.ptx` | Cross-entropy (inference diagnostic) |
| `topk_sampling.ptx` | Top-K sampling |
| `topp_filter_bf16.ptx` | Top-p (nucleus) filter |
| `argmax_f32.ptx` | Greedy argmax sampling |

## Model coverage

| Model | Kernels used |
|-------|--------------|
| **DeepSeek V3 / V3.1 / R1** | MLA decode + prefill (BF16, FP8), MoE grouped GEMM, FP8 block-scaled quant, RoPE, RMSNorm |
| **GDN-hybrid MoE** | V21 attention (BF16/FP8), MoE pipeline, grouped GEMM, RMSNorm+residual, SiLU-mul |
| **Llama 4 Scout / Maverick** | V21 attention, MoE grouped GEMM, GeLU-mul |
| **Gemma 2 / Gemma 3** | V21 SWA, V21 softcap_causal, V21 swa_softcap, RoPE, RMSNorm |
| **Mistral / Mixtral** | V21 SWA, grouped GEMM, SiLU-mul |
| **GLM-4.5 / GLM-5** | MLA decode + prefill, MoE grouped GEMM, quantization (NVFP4/FP8) |
| **gpt-oss-120b** | MXFP4 quant/dequant, MoE pipeline, BF16 attention |
| **Gemma-4-26B-A4B** (validated, 71.4× at M=128 batched — in-stack scaling figure vs single-user M=1) | MoE grouped MMA W8A16, moe_histogram/permute/unpermute, broadcast_top_k_bf16, per-seq RoPE (partial + proportional), shared-dense+routed-MoE FFN |

## Gaps (future work, tracked in plan)

- FP8 MLA prefill variant
- NSA / DSA / MoBA sparse attention
- Gated DeltaNet prefill (chunk-based parallel, O(L))  — current prefill is O(L) sequential
- Mamba2 prefill (chunk scan)
- MTP draft heads (DeepSeek V3)
- Medusa draft heads (can be expressed as grouped GEMM today)
- MMA-optimized MLA / GDN / Mamba2 (currently scalar reference)
- BF16 GEMM reaching the FA-demonstrated ceiling (warp-specialized+TMA v4 shipped but regresses at large square — v3 ≈ 49 TFLOPS at 4096³ is the current best; FA proves ~75 is reachable on the same cores, so ~1.5× headroom remains)
- Ring attention (multi-DGX Spark clustering)
- NVFP4 KV cache write (FP8 KV write + standalone NVFP4 quant already landed)
- FP8 KV cache consume integration into attention kernels
