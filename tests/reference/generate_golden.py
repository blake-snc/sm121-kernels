#!/usr/bin/env python3
"""Generate golden test vectors for sm121-kernels integration tests.

Run on the DGX Spark (or any CUDA device) with PyTorch installed:
    python tests/reference/generate_golden.py

Output: tests/reference/data/*.npz
"""

import torch
import sys
import numpy as np
from pathlib import Path

OUT = Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)


def bf16_to_np(t):
    """Convert BF16 tensor to numpy uint16 (raw BF16 bits)."""
    return t.cpu().view(torch.uint16).numpy()


def np_to_bf16(arr):
    """Convert numpy uint16 (raw BF16 bits) to BF16 tensor."""
    return torch.from_numpy(arr).view(torch.bfloat16)


def gen_rmsnorm():
    torch.manual_seed(42)
    np.random.seed(42)
    for hidden in [2048, 4096, 8192]:
        x = torch.randn(32, hidden, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(hidden, dtype=torch.bfloat16, device="cuda")
        eps = 1e-5
        rms = torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
        out = (x.float() / rms * w.float()).to(torch.bfloat16)
        np.savez(
            OUT / f"rmsnorm_bf16_h{hidden}.npz",
            x=bf16_to_np(x),
            weight=bf16_to_np(w),
            out=bf16_to_np(out),
            eps=np.float32(eps),
        )


def gen_rmsnorm_backward():
    """RMSNorm backward via PyTorch autograd.

    Forward:  y = x * rsqrt(mean(x^2) + eps) * weight
    Backward (per row, hidden_dim D):
      r = rsqrt(mean(x^2) + eps)
      g = dy * weight
      mean_gx = mean(g * x)
      dx = r * (g - x * r^2 * mean_gx)
      dweight = sum_over_rows(dy * x * r)
    """
    torch.manual_seed(42)
    np.random.seed(42)
    for hidden in [2048, 4096, 8192]:
        x = torch.randn(32, hidden, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        w = torch.randn(hidden, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        eps = 1e-5

        # Compute forward in FP32, cast back to BF16 (matches our kernel semantics).
        x_f = x.float()
        rms = torch.sqrt(x_f.pow(2).mean(-1, keepdim=True) + eps)
        y = (x_f / rms * w.float()).to(torch.bfloat16)

        # Random upstream gradient.
        dy = torch.randn(32, hidden, dtype=torch.bfloat16, device="cuda")

        # Backward in FP32 to match kernel's accumulation precision.
        dy_f = dy.float()
        w_f = w.float()
        x_f = x.float()
        r = torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + eps)  # [32,1]
        g = dy_f * w_f                                              # [32,H]
        mean_gx = (g * x_f).mean(-1, keepdim=True)                  # [32,1]
        dx = r * (g - x_f * r.pow(2) * mean_gx)                     # [32,H]
        dweight = (dy_f * x_f * r).sum(0)                           # [H]

        # Cast outputs to BF16 (kernel writes BF16 dx, BF16 dweight).
        dx_bf = dx.to(torch.bfloat16)
        dweight_bf = dweight.to(torch.bfloat16)

        np.savez(
            OUT / f"rmsnorm_backward_bf16_h{hidden}.npz",
            x=bf16_to_np(x),
            weight=bf16_to_np(w),
            dy=bf16_to_np(dy),
            dx=bf16_to_np(dx_bf),
            dweight=bf16_to_np(dweight_bf),
            eps=np.float32(eps),
        )


def gen_rmsnorm_residual():
    """Fused residual + RMSNorm: residual += x; out = rmsnorm(residual) * weight."""
    torch.manual_seed(42)
    np.random.seed(42)
    for hidden in [2048, 4096, 8192]:
        x = torch.randn(32, hidden, dtype=torch.bfloat16, device="cuda")
        residual_in = torch.randn(32, hidden, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(hidden, dtype=torch.bfloat16, device="cuda")
        eps = 1e-5
        # Operation: residual := x + residual; out = rmsnorm(residual) * weight
        residual_out = (x.float() + residual_in.float()).to(torch.bfloat16)
        rms = torch.sqrt(residual_out.float().pow(2).mean(-1, keepdim=True) + eps)
        out = (residual_out.float() / rms * w.float()).to(torch.bfloat16)
        np.savez(
            OUT / f"rmsnorm_residual_bf16_h{hidden}.npz",
            x=bf16_to_np(x),
            residual_in=bf16_to_np(residual_in),
            residual_out=bf16_to_np(residual_out),
            weight=bf16_to_np(w),
            out=bf16_to_np(out),
            eps=np.float32(eps),
        )


def gen_embedding_lookup():
    torch.manual_seed(42)
    np.random.seed(42)
    # Smaller configs to keep golden-vector npz files compact — kernel is trivially scalable.
    for (vocab, hidden, n_tok) in [(32000, 4096, 16), (50000, 4096, 32)]:
        table = torch.randn(vocab, hidden, dtype=torch.bfloat16, device="cuda") * 0.02
        token_ids = torch.randint(0, vocab, (n_tok,), dtype=torch.int32, device="cuda")
        out = table[token_ids.long()]
        np.savez(
            OUT / f"embedding_lookup_bf16_v{vocab}_h{hidden}_t{n_tok}.npz",
            table=bf16_to_np(table),
            token_ids=token_ids.cpu().numpy().astype(np.uint32),
            out=bf16_to_np(out),
        )


def gen_cross_entropy():
    torch.manual_seed(42)
    np.random.seed(42)
    for (vocab, n_tok) in [(32000, 16), (128256, 32)]:
        logits = torch.randn(n_tok, vocab, dtype=torch.bfloat16, device="cuda")
        targets = torch.randint(0, vocab, (n_tok,), dtype=torch.int32, device="cuda")
        # Reference: torch cross-entropy in F32 (natural log)
        logf = logits.float()
        log_sum_exp = torch.logsumexp(logf, dim=-1)
        target_logits = logf.gather(-1, targets.long().unsqueeze(-1)).squeeze(-1)
        losses = (log_sum_exp - target_logits).float().cpu().numpy().astype(np.float32)
        np.savez(
            OUT / f"cross_entropy_bf16_v{vocab}_t{n_tok}.npz",
            logits=bf16_to_np(logits),
            targets=targets.cpu().numpy().astype(np.uint32),
            losses=losses,
        )


def gen_rmsnorm_fp8out():
    torch.manual_seed(42)
    np.random.seed(42)
    for hidden in [2048, 4096]:
        x = torch.randn(16, hidden, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(hidden, dtype=torch.bfloat16, device="cuda")
        eps = 1e-5
        inv_scale = 0.1   # simulates fp8_scale = 10
        rms = torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
        tmp = (x.float() / rms) * w.float() * inv_scale
        # Round to FP8 e4m3 via torch's FP8 support
        fp8_max = 448.0
        tmp_clamped = torch.clamp(tmp, -fp8_max, fp8_max)
        out_fp8 = tmp_clamped.to(torch.float8_e4m3fn)
        np.savez(
            OUT / f"rmsnorm_bf16_fp8out_h{hidden}.npz",
            x=bf16_to_np(x),
            weight=bf16_to_np(w),
            out=out_fp8.view(torch.uint8).cpu().numpy(),
            eps=np.float32(eps),
            inv_scale=np.float32(inv_scale),
        )


def gen_topp_filter():
    torch.manual_seed(42)
    np.random.seed(42)
    for (vocab, p) in [(1024, 0.9), (4096, 0.7), (32000, 0.95)]:
        # Generate a realistic distribution: softmax of random logits
        logits = torch.randn(4, vocab, dtype=torch.bfloat16, device="cuda")
        probs = torch.softmax(logits.float(), dim=-1).to(torch.bfloat16)

        # Reference top-p filter (sort-based)
        p_f32 = probs.float()
        sorted_probs, sorted_idx = torch.sort(p_f32, dim=-1, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        # Threshold: keep indices up to and including the one that crosses p
        keep_mask_sorted = cumsum - sorted_probs < p
        keep_mask = torch.zeros_like(p_f32, dtype=torch.bool)
        keep_mask.scatter_(-1, sorted_idx, keep_mask_sorted)
        filtered = torch.where(keep_mask, p_f32, torch.zeros_like(p_f32))
        filtered = filtered / filtered.sum(-1, keepdim=True).clamp(min=1e-8)
        out = filtered.to(torch.bfloat16)

        np.savez(
            OUT / f"topp_filter_bf16_v{vocab}_p{int(p*100)}.npz",
            probs=bf16_to_np(probs),
            out=bf16_to_np(out),
            p_thresh=np.float32(p),
        )


def gen_moe_grouped_gemm():
    """MoE grouped GEMM + permute/unpermute end-to-end golden vectors."""
    # Small config: 32 tokens, 8 experts, top_k=2, hidden=64, K=128, N=128
    num_tokens, num_experts, top_k = 32, 8, 2
    k_dim, n_dim = 128, 128
    torch.manual_seed(42)
    x = torch.randn(num_tokens, k_dim, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(num_experts, k_dim, n_dim, dtype=torch.bfloat16, device="cuda") * 0.1

    # Assign each (token, k) slot to a random expert
    expert_ids_flat = torch.randint(
        0, num_experts, (num_tokens * top_k,), dtype=torch.int32, device="cuda"
    )
    # Random routing weights that sum to ~1 per token
    weights_flat = torch.rand(num_tokens * top_k, dtype=torch.float32, device="cuda")
    weights_view = weights_flat.view(num_tokens, top_k)
    weights_view /= weights_view.sum(-1, keepdim=True)
    weights_bf16 = weights_flat.to(torch.bfloat16)

    # Reference: permute activations
    histogram = torch.bincount(expert_ids_flat, minlength=num_experts).to(torch.int32)
    offsets = torch.cat([
        torch.zeros(1, dtype=torch.int32, device="cuda"),
        histogram.cumsum(0).to(torch.int32),
    ])
    # Stable sort by expert_id to get the expected permutation
    entry_idx = torch.arange(num_tokens * top_k, dtype=torch.int32, device="cuda")
    sort_key = expert_ids_flat.long() * (num_tokens * top_k) + entry_idx.long()
    perm = torch.argsort(sort_key)
    # Reference permuted activations: each permuted row comes from entry perm[i],
    # and the source token for entry e is e / top_k.
    src_tokens = perm // top_k
    x_permuted = x[src_tokens]
    # Reference inverse_index
    inv_index = perm.to(torch.int32)

    # Reference grouped GEMM per expert
    c_permuted = torch.zeros(num_tokens * top_k, n_dim, dtype=torch.bfloat16, device="cuda")
    for e in range(num_experts):
        lo = offsets[e].item()
        hi = offsets[e + 1].item()
        if hi > lo:
            c_permuted[lo:hi] = (x_permuted[lo:hi].float() @ w[e].float()).to(torch.bfloat16)

    # Reference unpermute: apply weights and gather back
    y_ref_f32 = torch.zeros(num_tokens, n_dim, dtype=torch.float32, device="cuda")
    for dst in range(num_tokens * top_k):
        src_entry = perm[dst].item()
        token = src_entry // top_k
        wt = weights_flat[src_entry].item()
        y_ref_f32[token] += wt * c_permuted[dst].float()

    np.savez(
        OUT / f"moe_grouped_gemm_t{num_tokens}_e{num_experts}_k{top_k}_h{k_dim}_n{n_dim}.npz",
        x=bf16_to_np(x),
        w=bf16_to_np(w),
        expert_ids=expert_ids_flat.cpu().numpy().astype(np.uint32),
        weights=bf16_to_np(weights_bf16),
        offsets_ref=offsets.cpu().numpy().astype(np.uint32),
        histogram_ref=histogram.cpu().numpy().astype(np.uint32),
        x_permuted_ref=bf16_to_np(x_permuted),
        inverse_index_ref=inv_index.cpu().numpy().astype(np.uint32),
        c_permuted_ref=bf16_to_np(c_permuted),
        y_ref=y_ref_f32.cpu().numpy().astype(np.float32),
    )


def gen_quant_fp8_pertoken():
    """Dynamic per-token FP8 E4M3 quantization golden vectors."""
    torch.manual_seed(42)
    np.random.seed(42)
    for (n_rows, hidden) in [(4, 1024), (8, 2048)]:
        x = torch.randn(n_rows, hidden, dtype=torch.bfloat16, device="cuda") * 2.0
        # Reference: abs_max -> scale -> quantize
        xf = x.float()
        abs_max = xf.abs().max(-1, keepdim=True).values
        scale = (abs_max / 448.0).clamp(min=1e-5)
        inv_scale = 1.0 / scale
        fp8_ref = torch.clamp(xf * inv_scale, -448.0, 448.0).to(torch.float8_e4m3fn)
        np.savez(
            OUT / f"quant_fp8_pertoken_n{n_rows}_h{hidden}.npz",
            x=bf16_to_np(x),
            scales=scale.squeeze(-1).cpu().numpy().astype(np.float32),
            out=fp8_ref.view(torch.uint8).cpu().numpy(),
        )


def gen_dequant_fp8():
    """FP8 per-token dequantization."""
    torch.manual_seed(42)
    np.random.seed(42)
    for (n_rows, hidden) in [(4, 1024), (8, 2048)]:
        fp8 = torch.randn(n_rows, hidden, dtype=torch.bfloat16, device="cuda").to(torch.float8_e4m3fn)
        scales = torch.rand(n_rows, dtype=torch.float32, device="cuda") * 0.5 + 0.01
        out = (fp8.float() * scales.unsqueeze(-1)).to(torch.bfloat16)
        np.savez(
            OUT / f"dequant_fp8_n{n_rows}_h{hidden}.npz",
            x=fp8.view(torch.uint8).cpu().numpy(),
            scales=scales.cpu().numpy().astype(np.float32),
            out=bf16_to_np(out),
        )


def gen_quant_fp8_block128():
    """DeepSeek V3-style 1x128 block-scaled FP8 quant."""
    torch.manual_seed(42)
    np.random.seed(42)
    for (n_rows, hidden) in [(4, 512), (8, 1024)]:
        x = torch.randn(n_rows, hidden, dtype=torch.bfloat16, device="cuda") * 2.0
        n_blocks = hidden // 128
        # Reshape to [rows, blocks, 128]
        xf = x.float().view(n_rows, n_blocks, 128)
        abs_max = xf.abs().amax(-1, keepdim=True)
        scale = (abs_max / 448.0).clamp(min=1e-5)
        inv_scale = 1.0 / scale
        fp8_ref = torch.clamp(xf * inv_scale, -448.0, 448.0).to(torch.float8_e4m3fn)
        np.savez(
            OUT / f"quant_fp8_block128_n{n_rows}_h{hidden}.npz",
            x=bf16_to_np(x),
            scales=scale.squeeze(-1).cpu().numpy().astype(np.float32),  # [n_rows, n_blocks]
            out=fp8_ref.view(-1, hidden).view(torch.uint8).cpu().numpy(),
        )


def gen_quant_mxfp8():
    """OCP MX microscaling FP8 (32-block, UE8M0)."""
    torch.manual_seed(42)
    np.random.seed(42)
    for (n_rows, hidden) in [(4, 512), (8, 1024)]:
        x = torch.randn(n_rows, hidden, dtype=torch.bfloat16, device="cuda") * 2.0
        n_blocks = hidden // 32
        xf = x.float().view(n_rows, n_blocks, 32)
        abs_max = xf.abs().amax(-1, keepdim=True)
        # Extract FP32 biased exp from abs_max
        abs_max_safe = abs_max.clamp(min=1e-38)
        max_bits = abs_max_safe.view(torch.int32)
        biased_exp = (max_bits >> 23) & 0xFF
        # UE8M0 scale byte: max(biased_exp - 8, 1) in [1, 254]
        scale_byte = (biased_exp - 8).clamp(min=1, max=254).to(torch.int32)
        # Zero-abs_max path: byte = 127 (scale = 2^0 = 1)
        scale_byte = torch.where(biased_exp == 0, torch.tensor(127, device="cuda"), scale_byte)
        inv_scale_exp = 254 - scale_byte
        inv_scale = (inv_scale_exp << 23).view(torch.float32)
        fp8_ref = torch.clamp(xf * inv_scale, -448.0, 448.0).to(torch.float8_e4m3fn)
        np.savez(
            OUT / f"quant_mxfp8_n{n_rows}_h{hidden}.npz",
            x=bf16_to_np(x),
            scales=scale_byte.view(n_rows, n_blocks).squeeze().to(torch.uint8).cpu().numpy(),
            out=fp8_ref.view(-1, hidden).view(torch.uint8).cpu().numpy(),
        )


def fp4_e2m1_to_f32(byte_val):
    """Decode one FP4 E2M1 nibble (4 bits) to FP32. Matches hardware."""
    sign = (byte_val >> 3) & 1
    exp = (byte_val >> 1) & 0x3
    mant = byte_val & 0x1
    sign_f = -1.0 if sign else 1.0
    if exp == 0:
        # Subnormal: (-1)^s * 0.m * 2^(1-1) = 0.m
        return sign_f * (mant * 0.5)
    # Normal: (-1)^s * 1.m * 2^(exp - 1)
    return sign_f * (1.0 + mant * 0.5) * (2.0 ** (exp - 1))


def f32_to_fp4_e2m1(v):
    """Round FP32 to FP4 E2M1, saturating at ±6.0. Returns a 4-bit nibble."""
    import math
    if math.isnan(v) or math.isinf(v):
        return 0x7 if v < 0 else 0x7  # saturate at +6 (nibble 0b0111 = 7)
    if v == 0.0:
        return 0
    sign = 0x8 if v < 0 else 0
    a = abs(v)
    if a >= 6.0:
        return sign | 0x7
    # Normalize to [1, 2): exp is power-of-2 floor
    exp = math.floor(math.log2(a))
    exp = max(-1, min(exp, 2))  # valid normal range: exp ∈ [-1, 2]
    biased_exp = exp + 1
    if biased_exp <= 0:
        # Subnormal range: val < 1.0
        # Subnormal value: 0.m * 2^0 = 0.5 * m
        # Closest subnormal: round(a * 2) which is 0 or 1
        mant = round(a * 2)
        if mant == 0:
            return 0
        return sign | 0x1   # 0.1 * 2^0 = 0.5
    normalized = a / (2.0 ** exp)
    mant_bit = 1 if round((normalized - 1.0) * 2) >= 1 else 0
    return sign | (biased_exp << 1) | mant_bit


def gen_quant_mxfp4():
    """OCP MX microscaling FP4 E2M1 quantization (32-block, UE8M0)."""
    torch.manual_seed(42)
    np.random.seed(42)
    for (n_rows, hidden) in [(4, 512), (8, 1024)]:
        x = torch.randn(n_rows, hidden, dtype=torch.bfloat16, device="cuda") * 2.0
        n_blocks = hidden // 32
        xf = x.float().view(n_rows, n_blocks, 32)
        abs_max = xf.abs().amax(-1, keepdim=True)
        abs_max_safe = abs_max.clamp(min=1e-38)
        max_bits = abs_max_safe.view(torch.int32)
        biased_exp = (max_bits >> 23) & 0xFF
        # FP4 E2M1 emax = 2, so scale_byte = biased_exp - 2
        scale_byte = (biased_exp - 2).clamp(min=1, max=254).to(torch.int32)
        scale_byte = torch.where(biased_exp == 0, torch.tensor(127, device="cuda"), scale_byte)
        inv_scale_exp = 254 - scale_byte
        inv_scale = (inv_scale_exp << 23).view(torch.float32)
        scaled = xf * inv_scale

        # Quantize to FP4 via Python reference (on CPU for simplicity)
        scaled_cpu = scaled.cpu().numpy()
        packed = np.zeros((n_rows, hidden // 2), dtype=np.uint8)
        for r in range(n_rows):
            for b in range(n_blocks):
                for i in range(16):
                    lo = f32_to_fp4_e2m1(float(scaled_cpu[r, b, 2*i]))
                    hi = f32_to_fp4_e2m1(float(scaled_cpu[r, b, 2*i + 1]))
                    packed[r, b*16 + i] = (hi << 4) | (lo & 0xF)

        np.savez(
            OUT / f"quant_mxfp4_n{n_rows}_h{hidden}.npz",
            x=bf16_to_np(x),
            scales=scale_byte.view(n_rows, n_blocks).squeeze().to(torch.uint8).cpu().numpy(),
            out=packed,
        )


def gen_dequant_mxfp4():
    torch.manual_seed(42)
    np.random.seed(42)
    for (n_rows, hidden) in [(4, 512), (8, 1024)]:
        n_blocks = hidden // 32
        packed = np.random.randint(0, 256, (n_rows, hidden // 2), dtype=np.uint8)
        scale_bytes = np.random.randint(120, 140, (n_rows, n_blocks), dtype=np.uint8)
        out_f32 = np.zeros((n_rows, hidden), dtype=np.float32)
        for r in range(n_rows):
            for b in range(n_blocks):
                scale = 2.0 ** (int(scale_bytes[r, b]) - 127)
                for i in range(16):
                    byte = int(packed[r, b*16 + i])
                    lo = byte & 0xF
                    hi = (byte >> 4) & 0xF
                    out_f32[r, b*32 + 2*i] = fp4_e2m1_to_f32(lo) * scale
                    out_f32[r, b*32 + 2*i + 1] = fp4_e2m1_to_f32(hi) * scale
        out_bf16 = torch.from_numpy(out_f32).to(torch.bfloat16)
        np.savez(
            OUT / f"dequant_mxfp4_n{n_rows}_h{hidden}.npz",
            x=packed,
            scales=scale_bytes,
            out=bf16_to_np(out_bf16),
        )


def gen_quant_nvfp4():
    """NVFP4: 16-block with FP8 E4M3 per-block scales."""
    torch.manual_seed(42)
    np.random.seed(42)
    for (n_rows, hidden) in [(4, 256), (8, 512)]:
        x = torch.randn(n_rows, hidden, dtype=torch.bfloat16, device="cuda") * 2.0
        n_blocks = hidden // 16
        xf = x.float().view(n_rows, n_blocks, 16)
        abs_max = xf.abs().amax(-1, keepdim=True).clamp(min=1e-5)
        scale_f32 = (abs_max / 6.0).clamp(min=1e-5)
        # Encode scale as FP8 E4M3, re-decode for exact consistency
        scale_fp8 = scale_f32.to(torch.float8_e4m3fn)
        scale_decoded = scale_fp8.float().clamp(min=1e-5)
        inv_scale = 1.0 / scale_decoded
        scaled = xf * inv_scale
        scaled_cpu = scaled.cpu().numpy()
        packed = np.zeros((n_rows, hidden // 2), dtype=np.uint8)
        for r in range(n_rows):
            for b in range(n_blocks):
                for i in range(8):
                    lo = f32_to_fp4_e2m1(float(scaled_cpu[r, b, 2*i]))
                    hi = f32_to_fp4_e2m1(float(scaled_cpu[r, b, 2*i + 1]))
                    packed[r, b*8 + i] = (hi << 4) | (lo & 0xF)
        np.savez(
            OUT / f"quant_nvfp4_n{n_rows}_h{hidden}.npz",
            x=bf16_to_np(x),
            scales=scale_fp8.view(n_rows, n_blocks).squeeze().view(torch.uint8).cpu().numpy(),
            out=packed,
        )


def gen_dequant_nvfp4():
    torch.manual_seed(42)
    np.random.seed(42)
    for (n_rows, hidden) in [(4, 256), (8, 512)]:
        n_blocks = hidden // 16
        packed = np.random.randint(0, 256, (n_rows, hidden // 2), dtype=np.uint8)
        scale_bytes_u8 = np.random.randint(40, 80, (n_rows, n_blocks), dtype=np.uint8)
        # Decode FP8 scales
        scale_tensor = torch.from_numpy(scale_bytes_u8).view(torch.float8_e4m3fn).float().cuda()
        out_f32 = np.zeros((n_rows, hidden), dtype=np.float32)
        for r in range(n_rows):
            for b in range(n_blocks):
                scale = float(scale_tensor[r, b].cpu())
                for i in range(8):
                    byte = int(packed[r, b*8 + i])
                    lo = byte & 0xF
                    hi = (byte >> 4) & 0xF
                    out_f32[r, b*16 + 2*i] = fp4_e2m1_to_f32(lo) * scale
                    out_f32[r, b*16 + 2*i + 1] = fp4_e2m1_to_f32(hi) * scale
        out_bf16 = torch.from_numpy(out_f32).to(torch.bfloat16)
        np.savez(
            OUT / f"dequant_nvfp4_n{n_rows}_h{hidden}.npz",
            x=packed,
            scales=scale_bytes_u8,
            out=bf16_to_np(out_bf16),
        )


def gen_dequant_mxfp8():
    torch.manual_seed(42)
    np.random.seed(42)
    for (n_rows, hidden) in [(4, 512), (8, 1024)]:
        n_blocks = hidden // 32
        # Generate random FP8 values + random UE8M0 scale bytes
        fp8 = torch.randn(n_rows, hidden, dtype=torch.bfloat16, device="cuda").to(torch.float8_e4m3fn)
        scale_bytes = torch.randint(120, 140, (n_rows, n_blocks), dtype=torch.uint8, device="cuda")
        # Decode: scale = 2^(byte - 127)
        scale_bits = (scale_bytes.to(torch.int32) << 23).view(torch.float32)
        fp8_f32 = fp8.float().view(n_rows, n_blocks, 32)
        out = (fp8_f32 * scale_bits.unsqueeze(-1)).to(torch.bfloat16).view(n_rows, hidden)
        np.savez(
            OUT / f"dequant_mxfp8_n{n_rows}_h{hidden}.npz",
            x=fp8.view(torch.uint8).cpu().numpy(),
            scales=scale_bytes.cpu().numpy(),
            out=bf16_to_np(out),
        )


def gen_softmax():
    """Temperature-scaled softmax over vocab dim (rightmost)."""
    torch.manual_seed(42)
    np.random.seed(42)
    for vocab, temp in [(32000, 1.0), (32000, 0.7), (128256, 1.0)]:
        x = torch.randn(4, vocab, dtype=torch.bfloat16, device="cuda")
        # softmax((x - max) / temp), numerically stable
        x_scaled = x.float() / temp
        x_max = x_scaled.max(-1, keepdim=True).values
        probs = torch.exp(x_scaled - x_max)
        probs = probs / probs.sum(-1, keepdim=True)
        out = probs.to(torch.bfloat16)
        np.savez(
            OUT / f"softmax_bf16_v{vocab}_t{int(temp*10)}.npz",
            x=bf16_to_np(x),
            out=bf16_to_np(out),
            temperature=np.float32(temp),
        )


def gen_silu_mul():
    torch.manual_seed(42)
    np.random.seed(42)
    for d in [2048, 4096]:
        inp = torch.randn(32, 2 * d, dtype=torch.bfloat16, device="cuda")
        gate = inp[..., :d].float()
        up = inp[..., d:].float()
        out = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
        np.savez(
            OUT / f"silu_mul_bf16_d{d}.npz",
            input=bf16_to_np(inp),
            out=bf16_to_np(out),
        )


def gen_gelu_mul():
    torch.manual_seed(42)
    np.random.seed(42)
    for d in [2048, 4096]:
        inp = torch.randn(32, 2 * d, dtype=torch.bfloat16, device="cuda")
        gate = inp[..., :d].float()
        up = inp[..., d:].float()
        out = (torch.nn.functional.gelu(gate) * up).to(torch.bfloat16)
        np.savez(
            OUT / f"gelu_mul_bf16_d{d}.npz",
            input=bf16_to_np(inp),
            out=bf16_to_np(out),
        )


def gen_gelu_tanh_mul():
    torch.manual_seed(42)
    np.random.seed(42)
    for d in [2048, 4096]:
        inp = torch.randn(32, 2 * d, dtype=torch.bfloat16, device="cuda")
        gate = inp[..., :d].float()
        up = inp[..., d:].float()
        out = (torch.nn.functional.gelu(gate, approximate="tanh") * up).to(
            torch.bfloat16
        )
        np.savez(
            OUT / f"gelu_tanh_mul_bf16_d{d}.npz",
            input=bf16_to_np(inp),
            out=bf16_to_np(out),
        )


def gen_rope():
    torch.manual_seed(42)
    np.random.seed(42)
    for d in [64, 128]:
        seq, heads = 512, 32
        x = torch.randn(1, seq, heads, d, dtype=torch.bfloat16, device="cuda")
        pos = torch.arange(seq, device="cuda").float()
        dim_pairs = torch.arange(0, d, 2, device="cuda").float()
        theta = 1.0 / (10000.0 ** (dim_pairs / d))
        angles = pos.unsqueeze(1) * theta.unsqueeze(0)
        cos_cache = torch.cos(angles)
        sin_cache = torch.sin(angles)

        x_f = x.float()
        x1 = x_f[..., 0::2]
        x2 = x_f[..., 1::2]
        out_f = torch.zeros_like(x_f)
        out_f[..., 0::2] = x1 * cos_cache.view(1, seq, 1, -1) - x2 * sin_cache.view(
            1, seq, 1, -1
        )
        out_f[..., 1::2] = x1 * sin_cache.view(1, seq, 1, -1) + x2 * cos_cache.view(
            1, seq, 1, -1
        )
        out = out_f.to(torch.bfloat16)

        np.savez(
            OUT / f"rope_bf16_d{d}.npz",
            x=bf16_to_np(x),
            out=bf16_to_np(out),
            cos_cache=cos_cache.cpu().numpy(),
            sin_cache=sin_cache.cpu().numpy(),
        )


def gen_rope_backward():
    """RoPE backward: with rotation by angle, backward is rotation by -angle.
    Forward (interleaved):
      out[2i]   = x[2i]*cos - x[2i+1]*sin
      out[2i+1] = x[2i]*sin + x[2i+1]*cos
    Backward:
      dx[2i]   = dy[2i]*cos + dy[2i+1]*sin
      dx[2i+1] = -dy[2i]*sin + dy[2i+1]*cos
    """
    torch.manual_seed(42)
    np.random.seed(42)
    for d in [64, 128]:
        seq, heads = 512, 32
        x = torch.randn(1, seq, heads, d, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        pos = torch.arange(seq, device="cuda").float()
        dim_pairs = torch.arange(0, d, 2, device="cuda").float()
        theta = 1.0 / (10000.0 ** (dim_pairs / d))
        angles = pos.unsqueeze(1) * theta.unsqueeze(0)
        cos_cache = torch.cos(angles)
        sin_cache = torch.sin(angles)

        x_f = x.float()
        x1 = x_f[..., 0::2]
        x2 = x_f[..., 1::2]
        out_f = torch.zeros_like(x_f)
        out_f[..., 0::2] = x1 * cos_cache.view(1, seq, 1, -1) - x2 * sin_cache.view(1, seq, 1, -1)
        out_f[..., 1::2] = x1 * sin_cache.view(1, seq, 1, -1) + x2 * cos_cache.view(1, seq, 1, -1)
        out = out_f.to(torch.bfloat16)

        # Random upstream gradient
        dy = torch.randn(1, seq, heads, d, dtype=torch.bfloat16, device="cuda")
        dy_f = dy.float()
        dy1 = dy_f[..., 0::2]
        dy2 = dy_f[..., 1::2]
        dx_f = torch.zeros_like(dy_f)
        dx_f[..., 0::2] = dy1 * cos_cache.view(1, seq, 1, -1) + dy2 * sin_cache.view(1, seq, 1, -1)
        dx_f[..., 1::2] = -dy1 * sin_cache.view(1, seq, 1, -1) + dy2 * cos_cache.view(1, seq, 1, -1)
        dx = dx_f.to(torch.bfloat16)

        np.savez(
            OUT / f"rope_backward_bf16_d{d}.npz",
            dy=bf16_to_np(dy),
            dx=bf16_to_np(dx),
            cos_cache=cos_cache.cpu().numpy(),
            sin_cache=sin_cache.cpu().numpy(),
        )


def gen_silu_backward():
    """SiLU backward: dx = dy * (sigmoid(x) + x*sigmoid(x)*(1-sigmoid(x))).
    Equivalent: sigmoid(x) * (1 + x * (1 - sigmoid(x))).
    """
    torch.manual_seed(42)
    np.random.seed(42)
    for d in [2048, 4096]:
        x = torch.randn(32, d, dtype=torch.bfloat16, device="cuda")
        dy = torch.randn(32, d, dtype=torch.bfloat16, device="cuda")
        x_f = x.float()
        dy_f = dy.float()
        sig = torch.sigmoid(x_f)
        dsilu = sig * (1.0 + x_f * (1.0 - sig))
        dx = (dy_f * dsilu).to(torch.bfloat16)
        np.savez(
            OUT / f"silu_backward_bf16_d{d}.npz",
            x=bf16_to_np(x),
            dy=bf16_to_np(dy),
            dx=bf16_to_np(dx),
        )


def gen_gelu_tanh_backward():
    """GeLU-tanh (PyTorch tanh approx) backward via analytical derivative.
    y = x * 0.5 * (1 + tanh(k*(x + 0.044715*x^3)))   k = sqrt(2/pi)
    inner = k*(x + 0.044715*x^3)
    t = tanh(inner)
    dy/dx = 0.5*(1+t) + 0.5*x*(1-t^2)*k*(1+0.134145*x^2)
    """
    torch.manual_seed(42)
    np.random.seed(42)
    import math
    K = math.sqrt(2.0 / math.pi)
    for d in [2048, 4096]:
        x = torch.randn(32, d, dtype=torch.bfloat16, device="cuda")
        dy = torch.randn(32, d, dtype=torch.bfloat16, device="cuda")
        x_f = x.float()
        dy_f = dy.float()
        inner = K * (x_f + 0.044715 * x_f.pow(3))
        t = torch.tanh(inner)
        dgelu = 0.5 * (1.0 + t) + 0.5 * x_f * (1.0 - t.pow(2)) * K * (1.0 + 0.134145 * x_f.pow(2))
        dx = (dy_f * dgelu).to(torch.bfloat16)
        np.savez(
            OUT / f"gelu_tanh_backward_bf16_d{d}.npz",
            x=bf16_to_np(x),
            dy=bf16_to_np(dy),
            dx=bf16_to_np(dx),
        )


def gen_softmax_backward():
    """Softmax backward via PyTorch autograd.
    Forward: y = softmax(x, dim=-1)
    Backward: dx[i] = y[i] * (dy[i] - sum_j(dy[j] * y[j]))
    """
    torch.manual_seed(42)
    np.random.seed(42)
    for d in [2048, 4096, 8192]:
        x = torch.randn(32, d, dtype=torch.bfloat16, device="cuda")
        dy = torch.randn(32, d, dtype=torch.bfloat16, device="cuda")
        x_f = x.float()
        # Compute softmax in FP32 to match kernel.
        y = torch.softmax(x_f, dim=-1)
        # Backward
        dy_f = dy.float()
        sum_term = (dy_f * y).sum(-1, keepdim=True)
        dx = (y * (dy_f - sum_term)).to(torch.bfloat16)
        # Save y as well so the kernel can take y as input (skip recomputation).
        y_bf = y.to(torch.bfloat16)
        np.savez(
            OUT / f"softmax_backward_bf16_d{d}.npz",
            y=bf16_to_np(y_bf),
            dy=bf16_to_np(dy),
            dx=bf16_to_np(dx),
        )


def gen_cross_entropy_backward():
    """Cross-entropy backward.
    Forward (per row): loss = -log(softmax(logits)[target])
    Backward: dlogits[i] = (softmax(logits)[i] - 1[i==target]) / batch_size
    """
    torch.manual_seed(42)
    np.random.seed(42)
    for vocab in [4096, 32000]:
        batch = 64
        logits = torch.randn(batch, vocab, dtype=torch.bfloat16, device="cuda")
        targets = torch.randint(0, vocab, (batch,), dtype=torch.int64, device="cuda")
        logits_f = logits.float()
        # Softmax in FP32
        probs = torch.softmax(logits_f, dim=-1)
        # dlogits[b, i] = (probs[b, i] - 1[i==targets[b]]) / batch
        dlogits_f = probs.clone()
        dlogits_f[torch.arange(batch, device="cuda"), targets] -= 1.0
        dlogits_f /= batch
        dlogits = dlogits_f.to(torch.bfloat16)
        np.savez(
            OUT / f"cross_entropy_backward_bf16_v{vocab}.npz",
            logits=bf16_to_np(logits),
            targets=targets.cpu().numpy().astype(np.uint32),
            dlogits=bf16_to_np(dlogits),
            batch_size=np.uint32(batch),
        )


def gen_fp8_w8a16_gemm_backward():
    """FP8 W8A16 GEMM backward reference. Forward C = A_bf16 @ (B_fp8 * scale_B).
    Backward dA via dequant-then-BF16. Tolerance is wider since FP8 quant error
    propagates."""
    torch.manual_seed(42)
    np.random.seed(42)
    for (m, n, k) in [(64, 128, 128), (128, 64, 256)]:
        a = torch.randn(m, k, dtype=torch.bfloat16, device="cpu")
        b_bf16 = torch.randn(k, n, dtype=torch.bfloat16, device="cpu") * 0.5
        # Quantize B to FP8 e4m3 with per-tensor scale.
        b_max = b_bf16.float().abs().max().item()
        scale_b = max(b_max / 448.0, 1e-12)
        b_fp8_f32 = (b_bf16.float() / scale_b).clamp(-448.0, 448.0)
        # Round-to-nearest-even via torch's e4m3 cast
        b_fp8 = b_fp8_f32.to(torch.float8_e4m3fn) if hasattr(torch, "float8_e4m3fn") else None
        if b_fp8 is None:
            # Skip if PyTorch lacks FP8 — just use BF16 reference for golden values.
            b_fp8_bytes = b_fp8_f32.round().clamp(-127, 127).to(torch.int8).to(torch.uint8)
            b_dequant = (b_fp8_bytes.to(torch.int8).float()) * scale_b
        else:
            b_fp8_bytes = b_fp8.view(torch.uint8)
            b_dequant = b_fp8.float() * scale_b
        c = (a.float() @ b_dequant).to(torch.bfloat16)
        dc = torch.randn(m, n, dtype=torch.bfloat16, device="cpu")
        # Backward in fp32 using the dequantized B
        da = (dc.float() @ b_dequant.T).to(torch.bfloat16)
        np.savez(
            OUT / f"gemm_fp8_w8a16_backward_m{m}_n{n}_k{k}.npz",
            a=bf16_to_np(a),
            b_fp8=b_fp8_bytes.cpu().numpy().astype(np.uint8),
            scale_b=np.float32(scale_b),
            dc=bf16_to_np(dc),
            da=bf16_to_np(da),
        )


def gen_flash_attention_backward_gqa():
    """FA backward with GQA (BF16) reference. H_q > H_kv; head-grouping
    aggregates dK/dV across Q heads sharing the same KV head."""
    torch.manual_seed(42)
    np.random.seed(42)
    for (h_q, h_kv) in [(8, 2), (4, 1)]:
        for causal in [False, True]:
            seq = 64
            B, d = 1, 64
            scale = 1.0 / (d ** 0.5)
            gen_dev = "cpu"
            q = torch.randn(B, h_q, seq, d, dtype=torch.bfloat16, device=gen_dev)
            k = torch.randn(B, h_kv, seq, d, dtype=torch.bfloat16, device=gen_dev)
            v = torch.randn(B, h_kv, seq, d, dtype=torch.bfloat16, device=gen_dev)
            do = torch.randn(B, h_q, seq, d, dtype=torch.bfloat16, device=gen_dev)

            q_f, k_f, v_f, do_f = q.float(), k.float(), v.float(), do.float()
            # Expand K/V to H_q heads (PyTorch repeat_interleave is the GQA convention)
            group = h_q // h_kv
            k_exp = k_f.repeat_interleave(group, dim=1)
            v_exp = v_f.repeat_interleave(group, dim=1)

            attn = torch.matmul(q_f, k_exp.transpose(-2, -1)) * scale
            if causal:
                mask = torch.triu(torch.ones(seq, seq, device=gen_dev), diagonal=1).bool()
                attn = attn.masked_fill(mask, float("-inf"))
            p = torch.softmax(attn, dim=-1)

            dv_exp_grad = torch.matmul(p.transpose(-2, -1), do_f)
            dp = torch.matmul(do_f, v_exp.transpose(-2, -1))
            d_term = (dp * p).sum(-1, keepdim=True)
            ds = p * (dp - d_term)
            dq_f = torch.matmul(ds, k_exp) * scale
            dk_exp_grad = torch.matmul(ds.transpose(-2, -1), q_f) * scale

            # Contract dK/dV back to H_kv: sum over each group of `group` heads.
            dk_f = dk_exp_grad.view(B, h_kv, group, seq, d).sum(dim=2)
            dv_f = dv_exp_grad.view(B, h_kv, group, seq, d).sum(dim=2)

            tag = "causal" if causal else "noncausal"
            np.savez(
                OUT / f"flash_attn_backward_gqa_{tag}_bf16_hq{h_q}_hkv{h_kv}_s{seq}.npz",
                q=bf16_to_np(q),
                k=bf16_to_np(k),
                v=bf16_to_np(v),
                do=bf16_to_np(do),
                dq=bf16_to_np(dq_f.to(torch.bfloat16)),
                dk=bf16_to_np(dk_f.to(torch.bfloat16)),
                dv=bf16_to_np(dv_f.to(torch.bfloat16)),
                scale=np.float32(scale),
            )


def gen_flash_attention_backward_causal():
    """FA backward (CAUSAL, BF16) reference. Same algorithm as non-causal
    plus n > m mask before softmax."""
    torch.manual_seed(42)
    np.random.seed(42)
    for seq in [64, 128]:
        B, H, d = 1, 4, 64
        scale = 1.0 / (d ** 0.5)
        gen_dev = "cpu"
        q = torch.randn(B, H, seq, d, dtype=torch.bfloat16, device=gen_dev)
        k = torch.randn(B, H, seq, d, dtype=torch.bfloat16, device=gen_dev)
        v = torch.randn(B, H, seq, d, dtype=torch.bfloat16, device=gen_dev)
        do = torch.randn(B, H, seq, d, dtype=torch.bfloat16, device=gen_dev)

        q_f, k_f, v_f, do_f = q.float(), k.float(), v.float(), do.float()

        attn = torch.matmul(q_f, k_f.transpose(-2, -1)) * scale
        # Causal mask
        mask = torch.triu(torch.ones(seq, seq, device=gen_dev), diagonal=1).bool()
        attn = attn.masked_fill(mask, float("-inf"))
        p = torch.softmax(attn, dim=-1)

        dv_f = torch.matmul(p.transpose(-2, -1), do_f)
        dp_f = torch.matmul(do_f, v_f.transpose(-2, -1))
        d_term = (dp_f * p).sum(-1, keepdim=True)
        ds = p * (dp_f - d_term)
        dq_f = torch.matmul(ds, k_f) * scale
        dk_f = torch.matmul(ds.transpose(-2, -1), q_f) * scale

        np.savez(
            OUT / f"flash_attn_backward_causal_bf16_s{seq}_d{d}.npz",
            q=bf16_to_np(q),
            k=bf16_to_np(k),
            v=bf16_to_np(v),
            do=bf16_to_np(do),
            dq=bf16_to_np(dq_f.to(torch.bfloat16)),
            dk=bf16_to_np(dk_f.to(torch.bfloat16)),
            dv=bf16_to_np(dv_f.to(torch.bfloat16)),
            scale=np.float32(scale),
        )


def gen_flash_attention_backward():
    """FA backward (non-causal, BF16) via PyTorch autograd.

    Forward: O = softmax(Q @ K^T * scale) @ V
    Inputs/outputs all [B, H, S, d] BF16.

    Saves: q, k, v, do, dq, dk, dv (no need to save O — kernel
    recomputes it for D = rowsum(dO*O) anyway, but autograd reference
    just needs the gradients).
    """
    torch.manual_seed(42)
    np.random.seed(42)
    for seq in [64, 128]:
        B, H, d = 1, 4, 64
        scale = 1.0 / (d ** 0.5)
        # CPU generation (FA backward goldens are small; avoids GPU OOM from
        # DGX Spark page-cache fragmentation issues during long sessions).
        gen_dev = "cpu"
        q = torch.randn(B, H, seq, d, dtype=torch.bfloat16, device=gen_dev, requires_grad=True)
        k = torch.randn(B, H, seq, d, dtype=torch.bfloat16, device=gen_dev, requires_grad=True)
        v = torch.randn(B, H, seq, d, dtype=torch.bfloat16, device=gen_dev, requires_grad=True)

        # Forward in fp32 to match kernel
        q_f = q.float()
        k_f = k.float()
        v_f = v.float()
        attn = torch.matmul(q_f, k_f.transpose(-2, -1)) * scale
        p = torch.softmax(attn, dim=-1)
        o = torch.matmul(p, v_f)

        # Random upstream gradient
        do = torch.randn(B, H, seq, d, dtype=torch.bfloat16, device=gen_dev)
        do_f = do.float()

        # Backward (analytical, in fp32)
        # dV = P^T @ dO
        dv_f = torch.matmul(p.transpose(-2, -1), do_f)
        # dP = dO @ V^T
        dp_f = torch.matmul(do_f, v_f.transpose(-2, -1))
        # dS = P * (dP - rowsum(dP * P))  (= P * (dP - D))
        d_term = (dp_f * p).sum(-1, keepdim=True)
        ds = p * (dp_f - d_term)
        # dQ = dS @ K * scale; dK = dS^T @ Q * scale
        dq_f = torch.matmul(ds, k_f) * scale
        dk_f = torch.matmul(ds.transpose(-2, -1), q_f) * scale

        dq = dq_f.to(torch.bfloat16)
        dk = dk_f.to(torch.bfloat16)
        dv = dv_f.to(torch.bfloat16)

        np.savez(
            OUT / f"flash_attn_backward_bf16_s{seq}_d{d}.npz",
            q=bf16_to_np(q),
            k=bf16_to_np(k),
            v=bf16_to_np(v),
            do=bf16_to_np(do),
            dq=bf16_to_np(dq),
            dk=bf16_to_np(dk),
            dv=bf16_to_np(dv),
            scale=np.float32(scale),
        )


def gen_gemm_backward():
    """GEMM backward via PyTorch autograd.
    Forward: C = A @ B, A:[M,K], B:[K,N], C:[M,N]
    Backward:
      dA[M,K] = dC[M,N] @ B^T[N,K]
      dB[K,N] = A^T[K,M] @ dC[M,N]
    """
    torch.manual_seed(42)
    np.random.seed(42)
    for (m, n, k) in [(64, 128, 128), (128, 64, 256), (256, 128, 64)]:
        a = torch.randn(m, k, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        b = torch.randn(k, n, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        a_f = a.float()
        b_f = b.float()
        c = (a_f @ b_f).to(torch.bfloat16)
        dc = torch.randn(m, n, dtype=torch.bfloat16, device="cuda")
        dc_f = dc.float()
        # Backward in fp32, cast to BF16
        da = (dc_f @ b_f.T).to(torch.bfloat16)
        db = (a_f.T @ dc_f).to(torch.bfloat16)
        np.savez(
            OUT / f"gemm_backward_bf16_m{m}_n{n}_k{k}.npz",
            a=bf16_to_np(a),
            b=bf16_to_np(b),
            dc=bf16_to_np(dc),
            da=bf16_to_np(da),
            db=bf16_to_np(db),
        )


def gen_flash_attention():
    torch.manual_seed(42)
    np.random.seed(42)
    for dtype_name, dtype in [("bf16", torch.bfloat16)]:
        for causal in [True, False]:
            for seq in [256, 1024]:
                B, H, d = 2, 8, 128
                q = torch.randn(B, H, seq, d, dtype=dtype, device="cuda")
                k = torch.randn(B, H, seq, d, dtype=dtype, device="cuda")
                v = torch.randn(B, H, seq, d, dtype=dtype, device="cuda")
                scale = 1.0 / (d**0.5)
                attn = (
                    torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
                )
                if causal:
                    mask = torch.triu(
                        torch.ones(seq, seq, device="cuda"), diagonal=1
                    ).bool()
                    attn.masked_fill_(mask, float("-inf"))
                attn = torch.softmax(attn, dim=-1)
                o = torch.matmul(attn, v.float()).to(dtype)
                tag = f"{'causal' if causal else 'noncausal'}_s{seq}"
                np.savez(
                    OUT / f"flash_attn_{dtype_name}_{tag}.npz",
                    q=bf16_to_np(q),
                    k=bf16_to_np(k),
                    v=bf16_to_np(v),
                    o=bf16_to_np(o),
                    scale=np.float32(scale),
                )


def gen_flash_attention_v3():
    """Generate V3-specific golden vectors (seq divisible by 128)."""
    for causal in [True, False]:
        B, H, seq, d = 2, 8, 128, 128
        torch.manual_seed(42)
        q = torch.randn(B, H, seq, d, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(B, H, seq, d, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(B, H, seq, d, dtype=torch.bfloat16, device="cuda")
        scale = 1.0 / (d**0.5)
        attn = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
        if causal:
            mask = torch.triu(
                torch.ones(seq, seq, device="cuda"), diagonal=1
            ).bool()
            attn.masked_fill_(mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1)
        o = torch.matmul(attn, v.float()).to(torch.bfloat16)
        tag = f"{'causal' if causal else 'noncausal'}_s{seq}"
        np.savez(
            OUT / f"flash_attn_bf16_{tag}.npz",
            q=bf16_to_np(q),
            k=bf16_to_np(k),
            v=bf16_to_np(v),
            o=bf16_to_np(o),
            scale=np.float32(scale),
        )


def gen_flash_attention_v3_d256():
    """Generate V3 d=256 golden vectors for GDN-hybrid gated full-attention layers.

    Tests the head_dim=256 BF16 attention kernel built for GDN-hybrid's 10/40
    full-attention layers. Br=128, Bc=64, 8 warps. seq must be multiple of 128.
    """
    for causal in [False, True]:
        # Single Q-block (seq=128) for first-correctness validation, then a
        # multi-block case (seq=256) to exercise the KV loop properly.
        for seq in [128, 256]:
            B, H, d = 2, 4, 256
            torch.manual_seed(42 + seq)
            q = torch.randn(B, H, seq, d, dtype=torch.bfloat16, device="cuda")
            k = torch.randn(B, H, seq, d, dtype=torch.bfloat16, device="cuda")
            v = torch.randn(B, H, seq, d, dtype=torch.bfloat16, device="cuda")
            scale = 1.0 / (d**0.5)
            attn = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
            if causal:
                mask = torch.triu(
                    torch.ones(seq, seq, device="cuda"), diagonal=1
                ).bool()
                attn.masked_fill_(mask, float("-inf"))
            attn = torch.softmax(attn, dim=-1)
            o = torch.matmul(attn, v.float()).to(torch.bfloat16)
            tag = f"{'causal' if causal else 'noncausal'}_s{seq}"
            np.savez(
                OUT / f"flash_attn_bf16_d256_{tag}.npz",
                q=bf16_to_np(q),
                k=bf16_to_np(k),
                v=bf16_to_np(v),
                o=bf16_to_np(o),
                scale=np.float32(scale),
            )


def gen_flash_attention_v3_d256_gqa():
    """GQA golden vectors for GDN-hybrid's gated full attention: 16 Q heads, 2 KV heads.

    K/V are stored once per KV head and shared across Q heads in the same group.
    For the test we replicate K/V to match Q's head count when computing reference.
    """
    H_q, H_kv, d = 16, 2, 256
    ratio = H_q // H_kv  # 8
    for causal in [False, True]:
        for seq in [128, 256]:
            B = 1
            torch.manual_seed(101 + seq + (10 if causal else 0))
            q = torch.randn(B, H_q, seq, d, dtype=torch.bfloat16, device="cuda")
            k_kv = torch.randn(B, H_kv, seq, d, dtype=torch.bfloat16, device="cuda")
            v_kv = torch.randn(B, H_kv, seq, d, dtype=torch.bfloat16, device="cuda")
            # Expand K/V to match Q for the reference computation
            k_expanded = k_kv.repeat_interleave(ratio, dim=1)
            v_expanded = v_kv.repeat_interleave(ratio, dim=1)
            scale = 1.0 / (d**0.5)
            attn = torch.matmul(q.float(), k_expanded.float().transpose(-2, -1)) * scale
            if causal:
                mask = torch.triu(
                    torch.ones(seq, seq, device="cuda"), diagonal=1
                ).bool()
                attn.masked_fill_(mask, float("-inf"))
            attn = torch.softmax(attn, dim=-1)
            o = torch.matmul(attn, v_expanded.float()).to(torch.bfloat16)
            tag = f"{'causal' if causal else 'noncausal'}_s{seq}"
            np.savez(
                OUT / f"flash_attn_bf16_d256_gqa_{tag}.npz",
                q=bf16_to_np(q),
                k=bf16_to_np(k_kv),
                v=bf16_to_np(v_kv),
                o=bf16_to_np(o),
                scale=np.float32(scale),
                num_heads=np.int32(H_q),
                num_heads_kv=np.int32(H_kv),
            )


def gen_flash_attention_fp8():
    """Generate FP8 flash attention golden vectors (Q,K,V in e4m3, O in bf16)."""
    for causal in [False, True]:
        for seq in [256, 1024]:
            B, H, d = 2, 8, 128
            torch.manual_seed(42 + seq)
            # Generate in float32, clamp to FP8 range, convert to e4m3
            q = torch.randn(B, H, seq, d, device="cuda").clamp(-448, 448).to(torch.float8_e4m3fn)
            k = torch.randn(B, H, seq, d, device="cuda").clamp(-448, 448).to(torch.float8_e4m3fn)
            v = torch.randn(B, H, seq, d, device="cuda").clamp(-448, 448).to(torch.float8_e4m3fn)
            scale = 1.0 / (d**0.5)
            # Reference computation in float32
            attn = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
            if causal:
                mask = torch.triu(
                    torch.ones(seq, seq, device="cuda"), diagonal=1
                ).bool()
                attn.masked_fill_(mask, float("-inf"))
            attn = torch.softmax(attn, dim=-1)
            o = torch.matmul(attn, v.float()).to(torch.bfloat16)
            tag = f"{'causal' if causal else 'noncausal'}_s{seq}"
            np.savez(
                OUT / f"flash_attn_fp8_{tag}.npz",
                q=fp8_to_np(q),
                k=fp8_to_np(k),
                v=fp8_to_np(v),
                o=bf16_to_np(o),
                scale=np.float32(scale),
            )


def gen_flash_attention_softcap_causal():
    """Softcap + causal (Gemma 2 global layer variant)."""
    for seq in [256, 1024]:
        B, H, d = 2, 8, 128
        torch.manual_seed(1000 + seq)
        q = torch.randn(B, H, seq, d, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(B, H, seq, d, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(B, H, seq, d, dtype=torch.bfloat16, device="cuda")
        scale = 1.0 / (d**0.5)
        softcap = 50.0
        attn = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
        attn = softcap * torch.tanh(attn / softcap)
        mask = torch.triu(torch.ones(seq, seq, device="cuda"), diagonal=1).bool()
        attn.masked_fill_(mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1)
        o = torch.matmul(attn, v.float()).to(torch.bfloat16)
        tag = f"softcap_causal_s{seq}"
        np.savez(
            OUT / f"flash_attn_bf16_{tag}.npz",
            q=bf16_to_np(q), k=bf16_to_np(k), v=bf16_to_np(v), o=bf16_to_np(o),
            scale=np.float32(scale), softcap=np.float32(softcap),
        )


def gen_flash_attention_swa():
    """Sliding window attention (causal). Window is smaller than seq to exercise masking."""
    for seq, window in [(256, 64), (1024, 256)]:
        B, H, d = 2, 8, 128
        torch.manual_seed(2000 + seq)
        q = torch.randn(B, H, seq, d, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(B, H, seq, d, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(B, H, seq, d, dtype=torch.bfloat16, device="cuda")
        scale = 1.0 / (d**0.5)
        attn = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
        # q_idx - k_idx in [0, window-1] keeps; else mask
        idx = torch.arange(seq, device="cuda")
        gap = idx[:, None] - idx[None, :]
        keep = (gap >= 0) & (gap < window)
        attn.masked_fill_(~keep, float("-inf"))
        attn = torch.softmax(attn, dim=-1)
        o = torch.matmul(attn, v.float()).to(torch.bfloat16)
        tag = f"swa_w{window}_s{seq}"
        np.savez(
            OUT / f"flash_attn_bf16_{tag}.npz",
            q=bf16_to_np(q), k=bf16_to_np(k), v=bf16_to_np(v), o=bf16_to_np(o),
            scale=np.float32(scale), window=np.uint32(window),
        )


def gen_flash_attention_swa_softcap():
    """SWA + softcap (Gemma 2 local layer variant)."""
    for seq, window in [(256, 64), (1024, 256)]:
        B, H, d = 2, 8, 128
        torch.manual_seed(3000 + seq)
        q = torch.randn(B, H, seq, d, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(B, H, seq, d, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(B, H, seq, d, dtype=torch.bfloat16, device="cuda")
        scale = 1.0 / (d**0.5)
        softcap = 50.0
        attn = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
        attn = softcap * torch.tanh(attn / softcap)
        idx = torch.arange(seq, device="cuda")
        gap = idx[:, None] - idx[None, :]
        keep = (gap >= 0) & (gap < window)
        attn.masked_fill_(~keep, float("-inf"))
        attn = torch.softmax(attn, dim=-1)
        o = torch.matmul(attn, v.float()).to(torch.bfloat16)
        tag = f"swa_softcap_w{window}_s{seq}"
        np.savez(
            OUT / f"flash_attn_bf16_{tag}.npz",
            q=bf16_to_np(q), k=bf16_to_np(k), v=bf16_to_np(v), o=bf16_to_np(o),
            scale=np.float32(scale), window=np.uint32(window), softcap=np.float32(softcap),
        )


def gen_mla_prefill():
    """MLA prefill (causal, seq_q > 1). DeepSeek V3 dims: D_C=512, D_R=64."""
    D_C, D_R = 512, 64
    for B, H, Sq in [(1, 8, 32), (2, 16, 64)]:
        Skv = Sq
        torch.manual_seed(6000 + Sq)
        q_c = torch.randn(B, Sq, H, D_C, dtype=torch.bfloat16, device="cuda")
        q_r = torch.randn(B, Sq, H, D_R, dtype=torch.bfloat16, device="cuda")
        c_kv = torch.randn(B, Skv, D_C, dtype=torch.bfloat16, device="cuda")
        k_rope = torch.randn(B, Skv, D_R, dtype=torch.bfloat16, device="cuda")
        scale = 1.0 / ((D_C + D_R) ** 0.5)

        # scores[b,q,h,s] = q_c[b,q,h]·c_kv[b,s] + q_r[b,q,h]·k_rope[b,s]
        scores_c = torch.einsum("bqhd,bsd->bqhs", q_c.float(), c_kv.float())
        scores_r = torch.einsum("bqhr,bsr->bqhs", q_r.float(), k_rope.float())
        scores = (scores_c + scores_r) * scale

        # Causal mask: mask s > q
        idx = torch.arange(Skv, device="cuda")
        causal = (idx[None, :] > idx[:, None]).view(1, Sq, 1, Skv)
        scores = scores.masked_fill(causal, float("-inf"))

        p = torch.softmax(scores, dim=-1)
        o = torch.einsum("bqhs,bsd->bqhd", p, c_kv.float()).to(torch.bfloat16)

        np.savez(
            OUT / f"mla_prefill_B{B}_Sq{Sq}_H{H}.npz",
            q_c=bf16_to_np(q_c), q_r=bf16_to_np(q_r),
            c_kv=bf16_to_np(c_kv), k_rope=bf16_to_np(k_rope),
            o=bf16_to_np(o),
            scale=np.float32(scale),
            batch=np.uint32(B), num_heads=np.uint32(H),
            seq_q=np.uint32(Sq), seq_kv=np.uint32(Skv),
        )


def gen_mamba2_decode():
    """Mamba2 selective scan decode reference."""
    D_state = 128
    for B, H in [(1, 4), (2, 8)]:
        torch.manual_seed(12000 + B * 100 + H)
        x = torch.randn(B, H, dtype=torch.float32, device="cuda")
        delta = (torch.rand(B, H, dtype=torch.float32, device="cuda") * 0.1 + 0.01)  # small positive
        A_log = -torch.rand(B, H, D_state, dtype=torch.float32, device="cuda") * 0.5  # negative → decay<1
        B_proj = torch.randn(B, H, D_state, dtype=torch.float32, device="cuda")
        C_proj = torch.randn(B, H, D_state, dtype=torch.float32, device="cuda")
        h_in = torch.randn(B, H, D_state, dtype=torch.float32, device="cuda") * 0.1

        # Reference: h[i] = h[i] * exp(A_log[i] * delta) + B[i] * x * delta
        decay = torch.exp(A_log * delta[:, :, None])
        h_out = h_in * decay + B_proj * x[:, :, None] * delta[:, :, None]
        y = (C_proj * h_out).sum(dim=-1)

        np.savez(
            OUT / f"mamba2_decode_B{B}_H{H}.npz",
            x=x.cpu().numpy(), delta=delta.cpu().numpy(),
            a=A_log.cpu().numpy(), B=B_proj.cpu().numpy(), C=C_proj.cpu().numpy(),
            h_in=h_in.cpu().numpy(), h_out=h_out.cpu().numpy(), y=y.cpu().numpy(),
            batch=np.uint32(B), num_heads=np.uint32(H),
        )


def gen_mamba2_prefill():
    """Mamba2 prefill sequential reference."""
    D_state = 128
    for B, H, Sq in [(1, 4, 4), (2, 4, 8)]:
        torch.manual_seed(14000 + B * 100 + Sq)
        x = torch.randn(B, Sq, H, dtype=torch.float32, device="cuda")
        delta = torch.rand(B, Sq, H, dtype=torch.float32, device="cuda") * 0.1 + 0.01
        A_log = -torch.rand(B, Sq, H, D_state, dtype=torch.float32, device="cuda") * 0.5
        B_proj = torch.randn(B, Sq, H, D_state, dtype=torch.float32, device="cuda")
        C_proj = torch.randn(B, Sq, H, D_state, dtype=torch.float32, device="cuda")
        h_in = torch.randn(B, H, D_state, dtype=torch.float32, device="cuda") * 0.1
        h = h_in.clone()

        y = torch.zeros(B, Sq, H, dtype=torch.float32, device="cuda")
        for t in range(Sq):
            decay = torch.exp(A_log[:, t] * delta[:, t, :, None])
            h = h * decay + B_proj[:, t] * x[:, t, :, None] * delta[:, t, :, None]
            y[:, t] = (C_proj[:, t] * h).sum(dim=-1)

        np.savez(
            OUT / f"mamba2_prefill_B{B}_H{H}_Sq{Sq}.npz",
            x=x.cpu().numpy(), delta=delta.cpu().numpy(),
            a=A_log.cpu().numpy(), B=B_proj.cpu().numpy(), C=C_proj.cpu().numpy(),
            h_in=h_in.cpu().numpy(), h_out=h.cpu().numpy(), y=y.cpu().numpy(),
            batch=np.uint32(B), num_heads=np.uint32(H), seq_q=np.uint32(Sq),
        )


def gen_gdn_prefill():
    """GDN prefill sequential reference: run decode recurrence over seq_q tokens."""
    D = 128
    for B, H, Sq in [(1, 4, 4), (2, 4, 8)]:
        torch.manual_seed(13000 + B * 100 + Sq)
        q = torch.randn(B, Sq, H, D, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(B, Sq, H, D, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(B, Sq, H, D, dtype=torch.bfloat16, device="cuda")
        alpha = torch.rand(B, Sq, H, dtype=torch.float32, device="cuda") * 0.2 + 0.8
        beta = torch.rand(B, Sq, H, dtype=torch.float32, device="cuda") * 0.5
        state_in = torch.randn(B, H, D, D, dtype=torch.float32, device="cuda") * 0.01
        state = state_in.clone()

        q_f = q.float(); k_f = k.float(); v_f = v.float()
        y = torch.zeros(B, Sq, H, D, dtype=torch.float32, device="cuda")
        for t in range(Sq):
            temp = torch.einsum("bhij,bhj->bhi", state, k_f[:, t])
            diff = v_f[:, t] - temp
            state = alpha[:, t, :, None, None] * state \
                    + beta[:, t, :, None, None] * diff[:, :, :, None] * k_f[:, t, :, None, :]
            y[:, t] = torch.einsum("bhij,bhj->bhi", state, q_f[:, t])
        y_bf16 = y.to(torch.bfloat16)

        np.savez(
            OUT / f"gdn_prefill_B{B}_H{H}_Sq{Sq}.npz",
            q=bf16_to_np(q), k=bf16_to_np(k), v=bf16_to_np(v),
            alpha=alpha.cpu().numpy(), beta=beta.cpu().numpy(),
            state_in=state_in.cpu().numpy(),
            state_out=state.cpu().numpy(),
            y=bf16_to_np(y_bf16),
            batch=np.uint32(B), num_heads=np.uint32(H), seq_q=np.uint32(Sq),
        )


def gen_gdn_prefill_hf():
    """GDN prefill, HF-correct recurrence: the S.k dot is taken against the
    ALPHA-DECAYED state (S_t = alpha*S_{t-1} decayed first), matching HF
    transformers' GDN-hybrid recurrent_gated_delta_rule and the gdn_decode kernel.
    Same inputs as gen_gdn_prefill; only the alpha decay on `temp` differs."""
    torch.manual_seed(42); np.random.seed(42)
    D = 128
    for B, H, Sq in [(1, 4, 4), (2, 4, 8)]:
        torch.manual_seed(13000 + B * 100 + Sq)
        q = torch.randn(B, Sq, H, D, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(B, Sq, H, D, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(B, Sq, H, D, dtype=torch.bfloat16, device="cuda")
        alpha = torch.rand(B, Sq, H, dtype=torch.float32, device="cuda") * 0.2 + 0.8
        beta = torch.rand(B, Sq, H, dtype=torch.float32, device="cuda") * 0.5
        state_in = torch.randn(B, H, D, D, dtype=torch.float32, device="cuda") * 0.01
        state = state_in.clone()

        q_f = q.float(); k_f = k.float(); v_f = v.float()
        y = torch.zeros(B, Sq, H, D, dtype=torch.float32, device="cuda")
        for t in range(Sq):
            temp = torch.einsum("bhij,bhj->bhi", state, k_f[:, t])
            temp = temp * alpha[:, t, :, None]  # HF alpha-decay (the only difference)
            diff = v_f[:, t] - temp
            state = alpha[:, t, :, None, None] * state \
                    + beta[:, t, :, None, None] * diff[:, :, :, None] * k_f[:, t, :, None, :]
            y[:, t] = torch.einsum("bhij,bhj->bhi", state, q_f[:, t])
        y_bf16 = y.to(torch.bfloat16)

        np.savez(
            OUT / f"gdn_prefill_hf_B{B}_H{H}_Sq{Sq}.npz",
            q=bf16_to_np(q), k=bf16_to_np(k), v=bf16_to_np(v),
            alpha=alpha.cpu().numpy(), beta=beta.cpu().numpy(),
            state_in=state_in.cpu().numpy(),
            state_out=state.cpu().numpy(),
            y=bf16_to_np(y_bf16),
            batch=np.uint32(B), num_heads=np.uint32(H), seq_q=np.uint32(Sq),
        )


def gen_kv_cache_fp8_write():
    """FP8 KV cache write: bf16 → fp8 e4m3 with per-tensor scale."""
    D = 128
    for B, H, page_size, num_pages in [(1, 4, 16, 2), (2, 8, 16, 4)]:
        torch.manual_seed(11000 + B * 100 + H)
        new_k = torch.randn(B, H, D, dtype=torch.bfloat16, device="cuda") * 2.0
        new_v = torch.randn(B, H, D, dtype=torch.bfloat16, device="cuda") * 2.0
        # Initial cache state: zeros
        k_cache = np.zeros((num_pages, page_size, H, D), dtype=np.uint8)
        v_cache = np.zeros((num_pages, page_size, H, D), dtype=np.uint8)
        # Destination: batch b writes to page b, slot b (simple)
        page_indices = np.array([b % num_pages for b in range(B)], dtype=np.uint32)
        slot_in_page = np.array([b % page_size for b in range(B)], dtype=np.uint32)
        k_scale = 4.0
        v_scale = 4.0

        # Reference: clamp, divide by scale, to fp8 e4m3
        k_cache_ref = k_cache.copy()
        v_cache_ref = v_cache.copy()
        for b in range(B):
            pg = int(page_indices[b])
            sl = int(slot_in_page[b])
            k_vals = (new_k[b].float() / k_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
            v_vals = (new_v[b].float() / v_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
            # Reshape to [H, D] → view as u8
            k_cache_ref[pg, sl, :, :] = k_vals.view(torch.uint8).cpu().numpy()
            v_cache_ref[pg, sl, :, :] = v_vals.view(torch.uint8).cpu().numpy()

        np.savez(
            OUT / f"kv_cache_fp8_write_B{B}_H{H}.npz",
            new_k=bf16_to_np(new_k), new_v=bf16_to_np(new_v),
            k_cache_in=k_cache, v_cache_in=v_cache,
            k_cache_out=k_cache_ref, v_cache_out=v_cache_ref,
            page_indices=page_indices, slot_in_page=slot_in_page,
            k_scale=np.float32(k_scale), v_scale=np.float32(v_scale),
            batch=np.uint32(B), num_heads=np.uint32(H),
            page_size=np.uint32(page_size), num_pages=np.uint32(num_pages),
        )


def gen_gdn_decode():
    """Gated DeltaNet decode reference (Qwen3-Next linear attention)."""
    D = 128
    for B, H in [(1, 4), (2, 8)]:
        torch.manual_seed(10000 + B * 100 + H)
        q = torch.randn(B, H, D, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(B, H, D, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(B, H, D, dtype=torch.bfloat16, device="cuda")
        alpha = torch.rand(B, H, dtype=torch.float32, device="cuda") * 0.2 + 0.8  # [0.8, 1.0]
        beta = torch.rand(B, H, dtype=torch.float32, device="cuda") * 0.5         # [0, 0.5]
        # Initial state: small random
        state_in = torch.randn(B, H, D, D, dtype=torch.float32, device="cuda") * 0.01

        # Reference computation
        q_f = q.float()
        k_f = k.float()
        v_f = v.float()
        # temp[b, h, i] = sum_j state_in[b, h, i, j] * k[b, h, j]
        temp = torch.einsum("bhij,bhj->bhi", state_in, k_f)
        # HF gated delta rule: the S.k dot is against the ALPHA-DECAYED state.
        temp = temp * alpha[:, :, None]
        diff = v_f - temp
        # new_state[b, h, i, j] = alpha * state_in[b, h, i, j] + beta * diff[b, h, i] * k[b, h, j]
        new_state = alpha[:, :, None, None] * state_in \
                    + beta[:, :, None, None] * diff[:, :, :, None] * k_f[:, :, None, :]
        # y[b, h, i] = sum_j new_state[b, h, i, j] * q[b, h, j]
        y = torch.einsum("bhij,bhj->bhi", new_state, q_f).to(torch.bfloat16)

        np.savez(
            OUT / f"gdn_decode_B{B}_H{H}.npz",
            q=bf16_to_np(q), k=bf16_to_np(k), v=bf16_to_np(v),
            alpha=alpha.cpu().numpy(), beta=beta.cpu().numpy(),
            state_in=state_in.cpu().numpy(),
            state_out=new_state.cpu().numpy(),
            y=bf16_to_np(y),
            batch=np.uint32(B), num_heads=np.uint32(H),
        )


def gen_k_block_mean():
    """Block-mean of K over fixed-size blocks (for NSA / MoBA block scoring)."""
    D = 128
    for B, H, Skv, block_size in [(1, 4, 128, 32), (2, 8, 256, 64), (1, 2, 100, 32)]:
        torch.manual_seed(16000 + B * 100 + Skv)
        k = torch.randn(B, Skv, H, D, dtype=torch.bfloat16, device="cuda")

        num_blocks = (Skv + block_size - 1) // block_size
        out_ref = torch.zeros(B, num_blocks, H, D, dtype=torch.float32, device="cuda")
        for block in range(num_blocks):
            start = block * block_size
            end = min(start + block_size, Skv)
            out_ref[:, block] = k[:, start:end].float().mean(dim=1)
        out_bf16 = out_ref.to(torch.bfloat16)

        np.savez(
            OUT / f"k_block_mean_B{B}_H{H}_Skv{Skv}_bs{block_size}.npz",
            k=bf16_to_np(k),
            out=bf16_to_np(out_bf16),
            batch=np.uint32(B), num_heads=np.uint32(H),
            seq_kv=np.uint32(Skv), block_size=np.uint32(block_size),
            num_blocks=np.uint32(num_blocks),
        )


def gen_nsa_attention():
    """NSA sparse attention: attend only to selected KV blocks per query."""
    for B, H, Sq, Skv, D, block_size, k_top in [
        (1, 4, 16, 256, 128, 32, 4),
        (2, 8, 32, 512, 128, 64, 6),
    ]:
        torch.manual_seed(15000 + Sq)
        q = torch.randn(B, Sq, H, D, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(B, Skv, H, D, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(B, Skv, H, D, dtype=torch.bfloat16, device="cuda")
        scale = 1.0 / (D ** 0.5)

        num_blocks = Skv // block_size
        # Simple block selection: query i picks blocks [i % num_blocks, ..., (i+k_top-1) % num_blocks]
        block_idx = torch.zeros(B, Sq, H, k_top, dtype=torch.int64, device="cuda")
        for sq in range(Sq):
            for kk in range(k_top):
                block_idx[:, sq, :, kk] = (sq + kk) % num_blocks

        # Reference: compute attention only over selected blocks
        o_ref = torch.zeros(B, Sq, H, D, dtype=torch.float32, device="cuda")
        for b in range(B):
            for sq in range(Sq):
                for h in range(H):
                    # Gather selected positions
                    positions = []
                    for kk in range(k_top):
                        bi = int(block_idx[b, sq, h, kk].item())
                        start = bi * block_size
                        end = min(start + block_size, Skv)
                        positions.extend(range(start, end))
                    # Attention over selected positions
                    q_val = q[b, sq, h].float()  # [D]
                    k_sel = k[b, positions, h].float()  # [N, D]
                    v_sel = v[b, positions, h].float()  # [N, D]
                    scores = (k_sel @ q_val) * scale   # [N]
                    probs = torch.softmax(scores, dim=-1)
                    o_ref[b, sq, h] = probs @ v_sel

        o_bf16 = o_ref.to(torch.bfloat16)
        block_idx_u32 = block_idx.to(torch.int64).to(torch.uint32)

        np.savez(
            OUT / f"nsa_B{B}_Sq{Sq}_Skv{Skv}_H{H}_K{k_top}.npz",
            q=bf16_to_np(q), k=bf16_to_np(k), v=bf16_to_np(v),
            block_idx=block_idx_u32.cpu().numpy(),
            o=bf16_to_np(o_bf16),
            scale=np.float32(scale),
            batch=np.uint32(B), num_heads=np.uint32(H),
            seq_q=np.uint32(Sq), seq_kv=np.uint32(Skv),
            k_top=np.uint32(k_top), block_size=np.uint32(block_size),
        )


def gen_tree_attention():
    """Tree attention reference for EAGLE-3 / Medusa speculative decoding.
    Mask is an explicit [Sq, Skv] bool matrix; simulates a 4-ancestor draft tree."""
    for B, H, Sq, Skv, D in [(1, 8, 8, 32, 128), (2, 16, 16, 64, 128)]:
        torch.manual_seed(9000 + Sq)
        q = torch.randn(B, Sq, H, D, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(B, Skv, H, D, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(B, Skv, H, D, dtype=torch.bfloat16, device="cuda")
        scale = 1.0 / (D ** 0.5)

        # Build a tree-style mask: q_i attends to base context [0..Skv-Sq)
        # plus a subset of previous draft positions (binary tree chain)
        base_ctx = Skv - Sq
        mask = torch.zeros(Sq, Skv, dtype=torch.uint8, device="cuda")
        mask[:, :base_ctx] = 1  # All queries attend to base context
        for i in range(Sq):
            # Draft token i attends to itself and ancestor chain i/2, i/4, ...
            idx = i
            while idx >= 0:
                mask[i, base_ctx + idx] = 1
                if idx == 0:
                    break
                idx = (idx - 1) // 2

        # Reference attention
        attn = torch.einsum("bqhd,bkhd->bqhk", q.float(), k.float()) * scale
        mask_bool = mask.bool().view(1, Sq, 1, Skv)
        attn = attn.masked_fill(~mask_bool, float("-inf"))
        p = torch.softmax(attn, dim=-1)
        o = torch.einsum("bqhk,bkhd->bqhd", p, v.float()).to(torch.bfloat16)

        np.savez(
            OUT / f"tree_attn_B{B}_Sq{Sq}_Skv{Skv}_H{H}.npz",
            q=bf16_to_np(q), k=bf16_to_np(k), v=bf16_to_np(v),
            mask=mask.cpu().numpy(), o=bf16_to_np(o),
            scale=np.float32(scale),
            batch=np.uint32(B), num_heads=np.uint32(H),
            seq_q=np.uint32(Sq), seq_kv=np.uint32(Skv),
        )


def gen_mla_decode_paged():
    """Paged MLA decode: KV cache stored in pages of fixed size."""
    D_C, D_R = 512, 64
    page_size = 16
    for B, H, S in [(1, 16, 32), (2, 32, 96)]:  # S must be a multiple of page_size for simplicity
        torch.manual_seed(8000 + S)
        q_c = torch.randn(B, H, D_C, dtype=torch.bfloat16, device="cuda")
        q_r = torch.randn(B, H, D_R, dtype=torch.bfloat16, device="cuda")
        # Per-batch sequence lengths (varying)
        seq_lens = torch.tensor([S, S - 17][:B], dtype=torch.uint32, device="cuda") if B > 1 else torch.tensor([S], dtype=torch.uint32, device="cuda")
        # Number of pages per batch (rounded up); use a fixed max for storage
        pages_per_batch = (S + page_size - 1) // page_size
        max_pages = pages_per_batch
        total_pages = B * pages_per_batch  # one disjoint set per batch (simplification)

        # Page pool: random KV; we'll address it via page table
        c_kv_pool = torch.randn(total_pages, page_size, D_C, dtype=torch.bfloat16, device="cuda")
        k_rope_pool = torch.randn(total_pages, page_size, D_R, dtype=torch.bfloat16, device="cuda")

        # Page table: batch b owns physical pages [b*pages_per_batch, (b+1)*pages_per_batch)
        page_table = torch.arange(total_pages, dtype=torch.int64, device="cuda").to(torch.uint32).view(B, pages_per_batch)

        scale = 1.0 / ((D_C + D_R) ** 0.5)

        # Build dense [B, S, D] view via page table for reference
        c_kv_dense = torch.zeros(B, S, D_C, dtype=torch.float32, device="cuda")
        k_rope_dense = torch.zeros(B, S, D_R, dtype=torch.float32, device="cuda")
        for b in range(B):
            slen = int(seq_lens[b].item())
            for s in range(slen):
                p = int(page_table[b, s // page_size].item())
                w = s % page_size
                c_kv_dense[b, s] = c_kv_pool[p, w].float()
                k_rope_dense[b, s] = k_rope_pool[p, w].float()

        # Reference attention with per-batch seq_len mask
        scores_c = torch.einsum("bhd,bsd->bhs", q_c.float(), c_kv_dense)
        scores_r = torch.einsum("bhr,bsr->bhs", q_r.float(), k_rope_dense)
        scores = (scores_c + scores_r) * scale
        # Mask positions >= seq_lens[b] to -inf
        idx = torch.arange(S, device="cuda")
        valid = idx[None, :] < seq_lens.long().to(idx.device)[:, None]    # [B, S]
        scores = scores.masked_fill(~valid[:, None, :], float("-inf"))
        p = torch.softmax(scores, dim=-1)
        o = torch.einsum("bhs,bsd->bhd", p, c_kv_dense).to(torch.bfloat16)

        np.savez(
            OUT / f"mla_decode_paged_B{B}_H{H}_S{S}.npz",
            q_c=bf16_to_np(q_c), q_r=bf16_to_np(q_r),
            c_kv=bf16_to_np(c_kv_pool), k_rope=bf16_to_np(k_rope_pool),
            page_table=page_table.cpu().numpy(),
            seq_lens=seq_lens.cpu().numpy(),
            o=bf16_to_np(o),
            scale=np.float32(scale),
            batch=np.uint32(B), num_heads=np.uint32(H),
            max_pages=np.uint32(max_pages), page_size=np.uint32(page_size),
        )


def gen_mla_prefill_fp8():
    """MLA FP8 KV prefill reference (causal)."""
    D_C, D_R = 512, 64
    for B, H, Sq in [(1, 4, 16), (2, 8, 32)]:
        torch.manual_seed(20000 + Sq)
        q_c = torch.randn(B, Sq, H, D_C, dtype=torch.bfloat16, device="cuda")
        q_r = torch.randn(B, Sq, H, D_R, dtype=torch.bfloat16, device="cuda")
        kv_scale = 4.0
        c_kv_f32 = torch.randn(B, Sq, D_C, device="cuda") * kv_scale
        k_rope_f32 = torch.randn(B, Sq, D_R, device="cuda") * kv_scale
        c_kv_fp8 = c_kv_f32.clamp(-448, 448).to(torch.float8_e4m3fn)
        k_rope_fp8 = k_rope_f32.clamp(-448, 448).to(torch.float8_e4m3fn)
        scale = 1.0 / ((D_C + D_R) ** 0.5)

        c_kv_deq = c_kv_fp8.float()
        k_rope_deq = k_rope_fp8.float()
        scores_c = torch.einsum("bqhd,bsd->bqhs", q_c.float(), c_kv_deq)
        scores_r = torch.einsum("bqhr,bsr->bqhs", q_r.float(), k_rope_deq)
        scores = (scores_c + scores_r) * scale
        idx = torch.arange(Sq, device="cuda")
        causal = (idx[None, :] > idx[:, None]).view(1, Sq, 1, Sq)
        scores = scores.masked_fill(causal, float("-inf"))
        p = torch.softmax(scores, dim=-1)
        o = torch.einsum("bqhs,bsd->bqhd", p, c_kv_deq).to(torch.bfloat16)

        np.savez(
            OUT / f"mla_prefill_fp8_B{B}_Sq{Sq}_H{H}.npz",
            q_c=bf16_to_np(q_c), q_r=bf16_to_np(q_r),
            c_kv=fp8_to_np(c_kv_fp8), k_rope=fp8_to_np(k_rope_fp8),
            o=bf16_to_np(o),
            scale=np.float32(scale), kv_scale=np.float32(1.0),
            batch=np.uint32(B), num_heads=np.uint32(H),
            seq_q=np.uint32(Sq), seq_kv=np.uint32(Sq),
        )


def gen_mla_decode_fp8():
    """MLA FP8 KV decode reference. c_kv/k_rope are FP8 E4M3 with per-tensor scale."""
    D_C, D_R = 512, 64
    for B, H, S in [(1, 16, 32), (2, 32, 128)]:
        torch.manual_seed(7000 + S)
        q_c = torch.randn(B, H, D_C, dtype=torch.bfloat16, device="cuda")
        q_r = torch.randn(B, H, D_R, dtype=torch.bfloat16, device="cuda")
        # FP8 E4M3 max is 448
        kv_scale = 4.0  # map [-448,448] / scale → reasonable range
        c_kv_f32 = torch.randn(B, S, D_C, device="cuda") * kv_scale
        k_rope_f32 = torch.randn(B, S, D_R, device="cuda") * kv_scale
        c_kv_fp8 = c_kv_f32.clamp(-448, 448).to(torch.float8_e4m3fn)
        k_rope_fp8 = k_rope_f32.clamp(-448, 448).to(torch.float8_e4m3fn)

        scale = 1.0 / ((D_C + D_R) ** 0.5)
        # Reference in fp32 from dequantized fp8
        c_kv_deq = c_kv_fp8.float()
        k_rope_deq = k_rope_fp8.float()
        scores_c = torch.einsum("bhd,bsd->bhs", q_c.float(), c_kv_deq)
        scores_r = torch.einsum("bhr,bsr->bhs", q_r.float(), k_rope_deq)
        scores = (scores_c + scores_r) * scale
        p = torch.softmax(scores, dim=-1)
        o = torch.einsum("bhs,bsd->bhd", p, c_kv_deq).to(torch.bfloat16)

        np.savez(
            OUT / f"mla_decode_fp8_B{B}_H{H}_S{S}.npz",
            q_c=bf16_to_np(q_c), q_r=bf16_to_np(q_r),
            c_kv=fp8_to_np(c_kv_fp8), k_rope=fp8_to_np(k_rope_fp8),
            o=bf16_to_np(o),
            scale=np.float32(scale),
            kv_scale=np.float32(1.0),  # kernel expects scale=1 since we pass raw bytes
            batch=np.uint32(B), num_heads=np.uint32(H), seq_kv=np.uint32(S),
        )


def gen_mla_decode():
    """MLA decode (seq_q=1) reference. DeepSeek V3 dims: D_C=512, D_R=64."""
    D_C, D_R = 512, 64
    for B, H, S in [(1, 16, 32), (2, 32, 128)]:
        torch.manual_seed(5000 + S)
        q_c = torch.randn(B, H, D_C, dtype=torch.bfloat16, device="cuda")
        q_r = torch.randn(B, H, D_R, dtype=torch.bfloat16, device="cuda")
        c_kv = torch.randn(B, S, D_C, dtype=torch.bfloat16, device="cuda")
        k_rope = torch.randn(B, S, D_R, dtype=torch.bfloat16, device="cuda")
        scale = 1.0 / ((D_C + D_R) ** 0.5)

        # scores[b,h,s] = q_c[b,h] · c_kv[b,s] + q_r[b,h] · k_rope[b,s]
        scores_c = torch.einsum("bhd,bsd->bhs", q_c.float(), c_kv.float())
        scores_r = torch.einsum("bhr,bsr->bhs", q_r.float(), k_rope.float())
        scores = (scores_c + scores_r) * scale
        p = torch.softmax(scores, dim=-1)
        # Output: sum_s p[b,h,s] * c_kv[b,s] = [B, H, D_C]
        o = torch.einsum("bhs,bsd->bhd", p, c_kv.float()).to(torch.bfloat16)

        np.savez(
            OUT / f"mla_decode_B{B}_H{H}_S{S}.npz",
            q_c=bf16_to_np(q_c), q_r=bf16_to_np(q_r),
            c_kv=bf16_to_np(c_kv), k_rope=bf16_to_np(k_rope),
            o=bf16_to_np(o),
            scale=np.float32(scale),
            batch=np.uint32(B), num_heads=np.uint32(H), seq_kv=np.uint32(S),
        )


def gen_gemm():
    torch.manual_seed(42)
    np.random.seed(42)
    for m, n, k in [(128, 128, 128), (512, 512, 512), (1024, 4096, 4096)]:
        a = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(k, n, dtype=torch.bfloat16, device="cuda")
        c = torch.matmul(a.float(), b.float()).to(torch.bfloat16)
        np.savez(
            OUT / f"gemm_bf16_{m}x{n}x{k}.npz",
            a=bf16_to_np(a),
            b=bf16_to_np(b),
            c=bf16_to_np(c),
        )


def fp8_to_np(t):
    """Convert FP8 (e4m3fn) tensor to numpy uint8 (raw bits)."""
    return t.cpu().view(torch.uint8).numpy()


def gen_gemm_fp8():
    torch.manual_seed(42)
    np.random.seed(42)
    for m, n, k in [(128, 128, 128), (512, 512, 512)]:
        # Generate random data in FP32, clamp to FP8 range, convert
        a_f32 = torch.randn(m, k, device="cuda").clamp(-448, 448)
        b_f32 = torch.randn(k, n, device="cuda").clamp(-448, 448)
        a_fp8 = a_f32.to(torch.float8_e4m3fn)
        b_fp8 = b_f32.to(torch.float8_e4m3fn)
        # Reference: FP32 matmul of the actual FP8 values, output as BF16
        c = torch.matmul(a_fp8.float(), b_fp8.float()).to(torch.bfloat16)
        np.savez(
            OUT / f"gemm_fp8_{m}x{n}x{k}.npz",
            a=fp8_to_np(a_fp8),
            b=fp8_to_np(b_fp8),
            c=bf16_to_np(c),
        )


def e2m1_quantize_block(values, block_size=64):
    """Quantize FP32 values to MXFP4 (e2m1 + UE8M0 block scale).

    Args:
        values: 1D FP32 array of length divisible by block_size
        block_size: number of elements per scale block (64 for our kernel)

    Returns:
        nibbles: uint8 array of 4-bit e2m1 codes
        scales: uint8 array of UE8M0 scale bytes
    """
    # E2M1 representable values (unsigned, then apply sign)
    e2m1_table = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)

    n = len(values)
    assert n % block_size == 0
    num_blocks = n // block_size

    nibbles = np.zeros(n, dtype=np.uint8)
    scales = np.zeros(num_blocks, dtype=np.uint8)

    for blk in range(num_blocks):
        start = blk * block_size
        block = values[start : start + block_size].astype(np.float32)
        max_abs = np.max(np.abs(block))

        if max_abs == 0:
            scales[blk] = 127  # scale = 1.0
            # all nibbles stay 0
            continue

        # Choose UE8M0 scale: 2^(ue8m0 - 127) such that max_abs / scale <= 6.0
        log2_scale = np.ceil(np.log2(max_abs / 6.0))
        ue8m0 = int(log2_scale) + 127
        ue8m0 = max(0, min(255, ue8m0))
        scales[blk] = ue8m0

        scale_val = 2.0 ** (ue8m0 - 127)

        for i in range(block_size):
            val = block[i] / scale_val
            sign = 0
            if val < 0:
                sign = 1
                val = -val

            # Find nearest e2m1 value
            best_idx = 0
            best_dist = abs(val - e2m1_table[0])
            for j in range(1, len(e2m1_table)):
                dist = abs(val - e2m1_table[j])
                if dist < best_dist:
                    best_dist = dist
                    best_idx = j
            nibbles[start + i] = (sign << 3) | best_idx

    return nibbles, scales


def pack_nibbles_k(matrix_nibbles, rows, cols):
    """Pack e2m1 nibbles into bytes along the K (column) dimension for A,
    or along the K (row) dimension for B.

    For A [M, K]: pairs consecutive K-elements within each row.
      A_packed[m, k_pair] = nibble[m, 2*k_pair] | (nibble[m, 2*k_pair+1] << 4)
      Result shape: [M, K/2]

    For B [K, N]: pairs consecutive K-rows for each column.
      B_packed[k_pair, n] = nibble[2*k_pair, n] | (nibble[2*k_pair+1, n] << 4)
      Result shape: [K/2, N]
    """
    assert cols % 2 == 0
    packed = np.zeros((rows, cols // 2), dtype=np.uint8)
    for r in range(rows):
        for c in range(0, cols, 2):
            lo = matrix_nibbles[r, c]
            hi = matrix_nibbles[r, c + 1]
            packed[r, c // 2] = (hi << 4) | lo
    return packed


def e2m1_to_float(nibble):
    """Convert a single e2m1 nibble (0-15) to float."""
    e2m1_table = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    sign = (nibble >> 3) & 1
    idx = nibble & 0x7
    val = e2m1_table[idx]
    return -val if sign else val


def gen_gemm_nvfp4():
    """Generate golden vectors for NVFP4 block-scaled GEMM.

    Layout:
      A: [M, K/2] u8 (nibble-packed along K, row-major)
      B: [K/2, N] u8 (nibble-packed along K, each byte = 2 K-elements for 1 N-col)
      scale_a: [ceil(M/16), K/64] u8 (UE8M0, one per MMA A-fragment)
      scale_b: [K/64, ceil(N/8)] u8 (UE8M0, one per MMA B-fragment)
      C: [M, N] u16 (bf16 output)
    """
    for m, n, k in [(32, 32, 64), (32, 32, 128)]:
        torch.manual_seed(42)
        a_f32 = torch.randn(m, k, device="cuda").clamp(-4.0, 4.0).cpu().numpy()
        b_f32 = torch.randn(k, n, device="cuda").clamp(-4.0, 4.0).cpu().numpy()

        # Quantize A: blocks of 64 along K, one scale per (16 rows, 64 K-elements)
        m_groups = (m + 15) // 16
        k_steps = k // 64
        a_nibbles = np.zeros((m, k), dtype=np.uint8)
        scale_a = np.zeros((m_groups, k_steps), dtype=np.uint8)

        for mg in range(m_groups):
            for ks in range(k_steps):
                row_start = mg * 16
                row_end = min(row_start + 16, m)
                col_start = ks * 64
                col_end = col_start + 64
                block = a_f32[row_start:row_end, col_start:col_end].flatten()
                nibs, scales = e2m1_quantize_block(block, block_size=len(block))
                a_nibbles[row_start:row_end, col_start:col_end] = nibs.reshape(
                    row_end - row_start, 64
                )
                scale_a[mg, ks] = scales[0]

        # Quantize B: blocks of 64 along K, one scale per (64 K-elements, 8 N-cols)
        n_groups = (n + 7) // 8
        b_nibbles = np.zeros((k, n), dtype=np.uint8)
        scale_b = np.zeros((k_steps, n_groups), dtype=np.uint8)

        for ks in range(k_steps):
            for ng in range(n_groups):
                row_start = ks * 64
                row_end = row_start + 64
                col_start = ng * 8
                col_end = min(col_start + 8, n)
                block = b_f32[row_start:row_end, col_start:col_end].flatten()
                nibs, scales = e2m1_quantize_block(block, block_size=len(block))
                b_nibbles[row_start:row_end, col_start:col_end] = nibs.reshape(
                    row_end - row_start, col_end - col_start
                )
                scale_b[ks, ng] = scales[0]

        # Pack nibbles: A along K (natural row-major), B along K (pair K-rows)
        a_packed = pack_nibbles_k(a_nibbles, m, k)  # [M, K/2]

        # B packing: pair consecutive K-rows for each N-column
        # B_packed[k_pair, n] = nibble[2*k_pair, n] | (nibble[2*k_pair+1, n] << 4)
        b_packed = np.zeros((k // 2, n), dtype=np.uint8)
        for kp in range(k // 2):
            for j in range(n):
                lo = b_nibbles[2 * kp, j]
                hi = b_nibbles[2 * kp + 1, j]
                b_packed[kp, j] = (hi << 4) | lo

        # Reference matmul: dequantize → FP32 matmul → BF16
        a_deq = np.zeros((m, k), dtype=np.float32)
        for mg in range(m_groups):
            for ks in range(k_steps):
                s = 2.0 ** (float(scale_a[mg, ks]) - 127.0)
                for i in range(mg * 16, min((mg + 1) * 16, m)):
                    for j in range(ks * 64, (ks + 1) * 64):
                        a_deq[i, j] = e2m1_to_float(a_nibbles[i, j]) * s

        b_deq = np.zeros((k, n), dtype=np.float32)
        for ks in range(k_steps):
            for ng in range(n_groups):
                s = 2.0 ** (float(scale_b[ks, ng]) - 127.0)
                for i in range(ks * 64, (ks + 1) * 64):
                    for j in range(ng * 8, min((ng + 1) * 8, n)):
                        b_deq[i, j] = e2m1_to_float(b_nibbles[i, j]) * s

        c_f32 = a_deq @ b_deq
        c_bf16 = torch.from_numpy(c_f32).to(torch.bfloat16)

        np.savez(
            OUT / f"gemm_nvfp4_{m}x{n}x{k}.npz",
            a=a_packed,
            b=b_packed,
            scale_a=scale_a.flatten(),
            scale_b=scale_b.flatten(),
            c=bf16_to_np(c_bf16),
        )


def gen_gemm_w4a16():
    """Generate golden vectors for W4A16 dequant GEMM.

    Layout:
      A: [M, K] u16 (bf16 activations)
      W: [K, N/2] u8 (INT4 weights, nibble-packed along N)
      scales: [N] u16 (bf16 per-column scale)
      zeros: [N] u16 (bf16 per-column zero point)
      C: [M, N] u16 (bf16 output)
    """
    for m, n, k in [(128, 128, 128), (128, 128, 256)]:
        torch.manual_seed(42)
        a = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")

        # Generate weights and quantization params
        w_f32 = torch.randn(k, n, device="cuda")
        w_min = w_f32.min(dim=0).values  # [N]
        w_max = w_f32.max(dim=0).values  # [N]

        # Per-column symmetric quantization to [0, 15]
        scales_f = (w_max - w_min) / 15.0
        scales_f = scales_f.clamp(min=1e-8)  # avoid div by zero
        zeros_f = w_min / scales_f  # zero point in quantized space
        zeros_f = zeros_f.round()

        # Quantize
        w_q = ((w_f32 / scales_f.unsqueeze(0)) - zeros_f.unsqueeze(0)).round()
        w_q = w_q.clamp(0, 15).to(torch.int32)

        # Pack nibbles along N dimension: w_packed[k, n_pair] = w_q[k, 2*n_pair] | (w_q[k, 2*n_pair+1] << 4)
        w_q_np = w_q.cpu().numpy().astype(np.uint8)
        w_packed = np.zeros((k, n // 2), dtype=np.uint8)
        for ki in range(k):
            for ni in range(0, n, 2):
                lo = w_q_np[ki, ni]
                hi = w_q_np[ki, ni + 1]
                w_packed[ki, ni // 2] = (hi << 4) | lo

        # Convert scales/zeros to bf16 first, then use bf16 values for reference
        # (matches what the kernel receives)
        scales_bf16 = scales_f.to(torch.bfloat16)
        zeros_bf16 = zeros_f.to(torch.bfloat16)

        # Reference: dequant w_q using bf16 scales/zeros, matmul, output BF16
        w_deq = (w_q.float() + zeros_bf16.float().unsqueeze(0)) * scales_bf16.float().unsqueeze(0)
        c = torch.matmul(a.float(), w_deq).to(torch.bfloat16)

        np.savez(
            OUT / f"gemm_w4a16_{m}x{n}x{k}.npz",
            a=bf16_to_np(a),
            w=w_packed,
            scales=bf16_to_np(scales_bf16),
            zeros=bf16_to_np(zeros_bf16),
            c=bf16_to_np(c),
        )


def gen_topk_sampling():
    """Generate top-k sampling golden vectors (k=1 greedy decoding)."""
    for batch, vocab in [(4, 32000), (1, 128256)]:
        torch.manual_seed(42 + vocab)
        logits = torch.randn(batch, vocab, dtype=torch.bfloat16, device="cuda")
        temperature = 1.0
        k = 1
        # Reference: argmax
        scaled = logits.float() / temperature
        indices = scaled.argmax(dim=-1)  # [batch]
        values = torch.gather(scaled, 1, indices.unsqueeze(1)).squeeze(1)  # [batch]
        values_bf16 = values.to(torch.bfloat16)
        np.savez(
            OUT / f"topk_sampling_b{batch}_v{vocab}.npz",
            logits=bf16_to_np(logits),
            indices=indices.cpu().numpy().astype(np.uint32),
            values=bf16_to_np(values_bf16),
            vocab_size=np.uint32(vocab),
            k=np.uint32(k),
            temperature=np.float32(temperature),
        )


def gen_moe_routing():
    """Generate MoE routing golden vectors."""
    # Includes top_k=6 (DSV2-Lite) and top_k=8 (DeepSeek V3) to cover the cases
    # that earlier exposed an SMEM-overlap bug at top_k > 4.
    for num_tokens, num_experts, top_k in [(16, 8, 2), (32, 64, 2), (8, 64, 6), (4, 64, 8)]:
        torch.manual_seed(42 + num_experts)
        logits = torch.randn(num_tokens, num_experts, dtype=torch.bfloat16, device="cuda")
        logits_f = logits.float()
        # Top-k selection
        topk_vals, topk_ids = torch.topk(logits_f, top_k, dim=-1)  # [num_tokens, top_k]
        # Softmax over selected experts
        weights = torch.softmax(topk_vals, dim=-1).to(torch.bfloat16)
        np.savez(
            OUT / f"moe_routing_t{num_tokens}_e{num_experts}_k{top_k}.npz",
            logits=bf16_to_np(logits),
            expert_ids=topk_ids.cpu().numpy().astype(np.uint32),
            weights=bf16_to_np(weights),
            num_tokens=np.uint32(num_tokens),
            num_experts=np.uint32(num_experts),
            top_k=np.uint32(top_k),
        )


def gen_flash_attention_varlen():
    """Generate varlen flash attention golden vectors.
    Layout: Q,K,V = [total_tokens, num_heads, d], cu_seqlens = [batch+1]."""
    torch.manual_seed(42)
    H, d = 4, 128
    # Two sequences of different lengths
    seq_lens = [32, 64]
    batch = len(seq_lens)
    total_q = sum(seq_lens)
    cu_seqlens_q = np.array([0] + list(np.cumsum(seq_lens)), dtype=np.uint32)
    cu_seqlens_k = cu_seqlens_q.copy()  # same lengths for Q and K

    # Generate packed Q, K, V: [total_tokens, num_heads, d]
    q = torch.randn(total_q, H, d, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(total_q, H, d, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(total_q, H, d, dtype=torch.bfloat16, device="cuda")
    scale = 1.0 / (d**0.5)

    # Compute reference output per sequence
    o_list = []
    for b in range(batch):
        s = cu_seqlens_q[b]
        e = cu_seqlens_q[b + 1]
        seq = e - s
        # Extract per-head Q, K, V for this batch: [seq, H, d] -> [H, seq, d]
        q_b = q[s:e].permute(1, 0, 2).float()  # [H, seq, d]
        k_b = k[s:e].permute(1, 0, 2).float()
        v_b = v[s:e].permute(1, 0, 2).float()
        attn = torch.matmul(q_b, k_b.transpose(-2, -1)) * scale  # [H, seq, seq]
        attn = torch.softmax(attn, dim=-1)
        o_b = torch.matmul(attn, v_b).permute(1, 0, 2).to(torch.bfloat16)  # [seq, H, d]
        o_list.append(o_b)
    o = torch.cat(o_list, dim=0)  # [total_q, H, d]

    np.savez(
        OUT / "flash_attn_bf16_varlen.npz",
        q=bf16_to_np(q.reshape(-1).unsqueeze(0)).squeeze(0),  # flat u16
        k=bf16_to_np(k.reshape(-1).unsqueeze(0)).squeeze(0),
        v=bf16_to_np(v.reshape(-1).unsqueeze(0)).squeeze(0),
        o=bf16_to_np(o.reshape(-1).unsqueeze(0)).squeeze(0),
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        num_heads=np.uint32(H),
        max_seqlen_q=np.uint32(max(seq_lens)),
        scale=np.float32(scale),
        total_q=np.uint32(total_q),
    )


def gen_flash_attention_v11_varlen():
    """Generate V11 varlen non-causal golden vectors (head-major layout).

    HEAD-MAJOR layout: Q,K,V,O = [num_heads, total_tokens, D].
    Full (non-causal) cross-attention between Q and KV per batch element.
    seq_q may differ from seq_kv.

    Two configs:
      flash_attn_bf16_v11_varlen_a.npz: B=2, H=4, seq_lens_q=[64,128], seq_lens_k=[128,64]
      flash_attn_bf16_v11_varlen_b.npz: B=2, H=4, seq_lens_q=[128,256], seq_lens_k=[256,128]
    """
    torch.manual_seed(43)

    def _run(seq_lens_q, seq_lens_k, H=4, d=128, name=""):
        batch = len(seq_lens_q)
        total_q = sum(seq_lens_q)
        total_kv = sum(seq_lens_k)
        cu_q = np.array([0] + list(np.cumsum(seq_lens_q)), dtype=np.uint32)
        cu_k = np.array([0] + list(np.cumsum(seq_lens_k)), dtype=np.uint32)
        scale = 1.0 / (d ** 0.5)

        q_hm = torch.randn(H, total_q, d, dtype=torch.bfloat16, device="cuda")
        k_hm = torch.randn(H, total_kv, d, dtype=torch.bfloat16, device="cuda")
        v_hm = torch.randn(H, total_kv, d, dtype=torch.bfloat16, device="cuda")

        o_hm = torch.zeros(H, total_q, d, dtype=torch.bfloat16, device="cuda")
        for b in range(batch):
            sq_s, sq_e = int(cu_q[b]), int(cu_q[b + 1])
            sk_s, sk_e = int(cu_k[b]), int(cu_k[b + 1])
            qb = q_hm[:, sq_s:sq_e, :].float()   # [H, sq, d]
            kb = k_hm[:, sk_s:sk_e, :].float()   # [H, sk, d]
            vb = v_hm[:, sk_s:sk_e, :].float()   # [H, sk, d]
            attn = torch.softmax(torch.matmul(qb, kb.transpose(-2, -1)) * scale, dim=-1)
            o_hm[:, sq_s:sq_e, :] = torch.matmul(attn, vb).to(torch.bfloat16)

        np.savez(
            OUT / f"flash_attn_bf16_v11_varlen_{name}.npz",
            q=bf16_to_np(q_hm.reshape(-1).unsqueeze(0)).squeeze(0),
            k=bf16_to_np(k_hm.reshape(-1).unsqueeze(0)).squeeze(0),
            v=bf16_to_np(v_hm.reshape(-1).unsqueeze(0)).squeeze(0),
            o=bf16_to_np(o_hm.reshape(-1).unsqueeze(0)).squeeze(0),
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            num_heads=np.uint32(H),
            batch=np.uint32(batch),
            max_seqlen_q=np.uint32(max(seq_lens_q)),
            total_q=np.uint32(total_q),
            total_kv=np.uint32(total_kv),
            scale=np.float32(scale),
        )

    _run([64, 128], [128, 64], name="a")    # cross-attention: different Q/KV lengths
    _run([128, 256], [256, 128], name="b")  # larger cross-attention


def gen_flash_attention_v11_varlen_causal():
    """Generate V11 varlen causal golden vectors.

    HEAD-MAJOR layout: Q,K,V,O = [num_heads, total_tokens, D].
    TMA descriptor covers [num_heads * total_tokens, D] (flat 2D view).
    Causal masking applied within each batch element's sequence.

    Two configs saved:
      flash_attn_bf16_v11_varlen_causal_a.npz: B=2, H=4, seq_lens=[64, 128]
      flash_attn_bf16_v11_varlen_causal_b.npz: B=2, H=4, seq_lens=[128, 256]
    """
    torch.manual_seed(42)

    def _run(seq_lens, H=4, d=128, name=""):
        batch = len(seq_lens)
        total = sum(seq_lens)
        cu_q = np.array([0] + list(np.cumsum(seq_lens)), dtype=np.uint32)
        cu_k = cu_q.copy()
        scale = 1.0 / (d ** 0.5)

        # HEAD-MAJOR layout: [H, total, D]
        q_hm = torch.randn(H, total, d, dtype=torch.bfloat16, device="cuda")
        k_hm = torch.randn(H, total, d, dtype=torch.bfloat16, device="cuda")
        v_hm = torch.randn(H, total, d, dtype=torch.bfloat16, device="cuda")

        # Compute reference output per batch element, causal masked
        o_parts = []  # each part: [H, seq, d]
        for b in range(batch):
            s, e = int(cu_q[b]), int(cu_q[b + 1])
            seq = e - s
            # q_hm[:, s:e, :] is [H, seq, d]
            qb = q_hm[:, s:e, :].float()  # [H, seq, d]
            kb = k_hm[:, s:e, :].float()
            vb = v_hm[:, s:e, :].float()
            # S = qb @ kb^T * scale  [H, seq, seq]
            attn = torch.matmul(qb, kb.transpose(-2, -1)) * scale
            # Causal mask: zero out upper triangle (j > i)
            causal_mask = torch.triu(
                torch.ones(seq, seq, device="cuda"), diagonal=1
            ).bool()
            attn.masked_fill_(causal_mask.unsqueeze(0), float("-inf"))
            attn = torch.softmax(attn, dim=-1)
            o_parts.append(torch.matmul(attn, vb).to(torch.bfloat16))  # [H, seq, d]

        # Reconstruct head-major output: [H, total, d]
        o_hm = torch.zeros(H, total, d, dtype=torch.bfloat16, device="cuda")
        for b in range(batch):
            s, e = int(cu_q[b]), int(cu_q[b + 1])
            o_hm[:, s:e, :] = o_parts[b]

        np.savez(
            OUT / f"flash_attn_bf16_v11_varlen_causal_{name}.npz",
            q=bf16_to_np(q_hm.reshape(-1).unsqueeze(0)).squeeze(0),
            k=bf16_to_np(k_hm.reshape(-1).unsqueeze(0)).squeeze(0),
            v=bf16_to_np(v_hm.reshape(-1).unsqueeze(0)).squeeze(0),
            o=bf16_to_np(o_hm.reshape(-1).unsqueeze(0)).squeeze(0),
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            num_heads=np.uint32(H),
            batch=np.uint32(batch),
            max_seqlen_q=np.uint32(max(seq_lens)),
            total_q=np.uint32(total),
            total_kv=np.uint32(total),
            scale=np.float32(scale),
        )

    _run([64, 128], name="a")   # B=2, total_q=192
    _run([128, 256], name="b")  # B=2, total_q=384


def gen_flash_attention_v11_gqa():
    """Generate V11 GQA fixed-length golden vectors.

    Fixed-length GQA (non-causal and causal variants).
    Q: [batch, num_heads_q, seq_q, D]  — stored flat in head-indexed order matching kernel layout
    K: [batch, num_heads_kv, seq_kv, D]
    V: [batch, num_heads_kv, seq_kv, D]

    Configs: B=2, H_q=8, H_kv=2 (GQA ratio 4), seq=256
    """
    torch.manual_seed(44)
    B, H_q, H_kv, seq_q, seq_kv, d = 2, 8, 2, 256, 256, 128
    scale = 1.0 / (d ** 0.5)

    # Kernel layout: batch-major [B, H_q, seq_q, D] for Q/O, [B, H_kv, seq_kv, D] for K/V
    q = torch.randn(B, H_q, seq_q, d, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(B, H_kv, seq_kv, d, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(B, H_kv, seq_kv, d, dtype=torch.bfloat16, device="cuda")

    gqa_groups = H_q // H_kv  # 4

    def compute_gqa(causal):
        o = torch.zeros_like(q)
        for b in range(B):
            for h_q in range(H_q):
                h_kv = h_q // gqa_groups
                qbh = q[b, h_q, :, :].float()           # [seq_q, d]
                kbh = k[b, h_kv, :, :].float()          # [seq_kv, d]
                vbh = v[b, h_kv, :, :].float()          # [seq_kv, d]
                attn = torch.matmul(qbh, kbh.t()) * scale  # [seq_q, seq_kv]
                if causal:
                    mask = torch.triu(torch.ones(seq_q, seq_kv, device="cuda"), diagonal=1).bool()
                    attn.masked_fill_(mask, float("-inf"))
                attn = torch.softmax(attn, dim=-1)
                o[b, h_q, :, :] = torch.matmul(attn, vbh).to(torch.bfloat16)
        return o

    for causal in [False, True]:
        o = compute_gqa(causal)
        tag = "causal" if causal else "noncausal"
        np.savez(
            OUT / f"flash_attn_bf16_v11_gqa_{tag}.npz",
            q=bf16_to_np(q.reshape(-1)).flatten(),
            k=bf16_to_np(k.reshape(-1)).flatten(),
            v=bf16_to_np(v.reshape(-1)).flatten(),
            o=bf16_to_np(o.reshape(-1)).flatten(),
            num_heads_q=np.uint32(H_q),
            num_heads_kv=np.uint32(H_kv),
            batch=np.uint32(B),
            seq_q=np.uint32(seq_q),
            seq_kv=np.uint32(seq_kv),
            scale=np.float32(scale),
        )


def gen_flash_attention_v11_varlen_gqa():
    """Generate V11 varlen GQA golden vectors (non-causal and causal).

    HEAD-MAJOR varlen layout:
      Q, O: [H_q, total_q, D]
      K, V: [H_kv, total_kv, D]
    TMA descriptors: Q over [H_q * total_q, D], K/V over [H_kv * total_kv, D].

    Config: B=2, H_q=8, H_kv=2, seq_lens=[128, 192]
    """
    torch.manual_seed(45)
    H_q, H_kv, d = 8, 2, 128
    seq_lens = [128, 192]
    batch = len(seq_lens)
    total = sum(seq_lens)
    cu = np.array([0] + list(np.cumsum(seq_lens)), dtype=np.uint32)
    scale = 1.0 / (d ** 0.5)
    gqa_groups = H_q // H_kv

    # Head-major layout
    q_hm = torch.randn(H_q, total, d, dtype=torch.bfloat16, device="cuda")
    k_hm = torch.randn(H_kv, total, d, dtype=torch.bfloat16, device="cuda")
    v_hm = torch.randn(H_kv, total, d, dtype=torch.bfloat16, device="cuda")

    def compute(causal):
        o_hm = torch.zeros(H_q, total, d, dtype=torch.bfloat16, device="cuda")
        for b in range(batch):
            s, e = int(cu[b]), int(cu[b + 1])
            seq = e - s
            for h_q in range(H_q):
                h_kv = h_q // gqa_groups
                qbh = q_hm[h_q, s:e, :].float()     # [seq, d]
                kbh = k_hm[h_kv, s:e, :].float()    # [seq, d]
                vbh = v_hm[h_kv, s:e, :].float()    # [seq, d]
                attn = torch.matmul(qbh, kbh.t()) * scale  # [seq, seq]
                if causal:
                    mask = torch.triu(torch.ones(seq, seq, device="cuda"), diagonal=1).bool()
                    attn.masked_fill_(mask, float("-inf"))
                attn = torch.softmax(attn, dim=-1)
                o_hm[h_q, s:e, :] = torch.matmul(attn, vbh).to(torch.bfloat16)
        return o_hm

    for causal in [False, True]:
        o_hm = compute(causal)
        tag = "causal" if causal else "noncausal"
        np.savez(
            OUT / f"flash_attn_bf16_v11_varlen_gqa_{tag}.npz",
            q=bf16_to_np(q_hm.reshape(-1)).flatten(),
            k=bf16_to_np(k_hm.reshape(-1)).flatten(),
            v=bf16_to_np(v_hm.reshape(-1)).flatten(),
            o=bf16_to_np(o_hm.reshape(-1)).flatten(),
            cu_seqlens_q=cu,
            cu_seqlens_k=cu,
            num_heads_q=np.uint32(H_q),
            num_heads_kv=np.uint32(H_kv),
            batch=np.uint32(batch),
            max_seqlen_q=np.uint32(max(seq_lens)),
            total_q=np.uint32(total),
            total_kv=np.uint32(total),
            scale=np.float32(scale),
        )


def gen_flash_attention_fp8_varlen():
    """Generate FP8 varlen non-causal flash attention golden vectors (head-major layout).

    HEAD-MAJOR layout: Q,K,V,O = [num_heads, total_tokens, D].
    Full (non-causal) cross-attention. seq_q may differ from seq_kv.

    Two configs:
      flash_attn_fp8_varlen_a.npz: B=2, H=4, seq_lens_q=[64,128], seq_lens_k=[128,64]
      flash_attn_fp8_varlen_b.npz: B=2, H=4, seq_lens_q=[128,256], seq_lens_k=[256,128]
    """
    torch.manual_seed(45)

    def _run(seq_lens_q, seq_lens_k, H=4, d=128, name=""):
        batch = len(seq_lens_q)
        total_q = sum(seq_lens_q)
        total_kv = sum(seq_lens_k)
        cu_q = np.array([0] + list(np.cumsum(seq_lens_q)), dtype=np.uint32)
        cu_k = np.array([0] + list(np.cumsum(seq_lens_k)), dtype=np.uint32)
        scale = 1.0 / (d ** 0.5)

        q_hm = torch.randn(H, total_q, d, device="cuda").clamp(-448, 448).to(torch.float8_e4m3fn)
        k_hm = torch.randn(H, total_kv, d, device="cuda").clamp(-448, 448).to(torch.float8_e4m3fn)
        v_hm = torch.randn(H, total_kv, d, device="cuda").clamp(-448, 448).to(torch.float8_e4m3fn)

        o_hm = torch.zeros(H, total_q, d, dtype=torch.bfloat16, device="cuda")
        for b in range(batch):
            sq_s, sq_e = int(cu_q[b]), int(cu_q[b + 1])
            sk_s, sk_e = int(cu_k[b]), int(cu_k[b + 1])
            qb = q_hm[:, sq_s:sq_e, :].float()
            kb = k_hm[:, sk_s:sk_e, :].float()
            vb = v_hm[:, sk_s:sk_e, :].float()
            attn = torch.softmax(torch.matmul(qb, kb.transpose(-2, -1)) * scale, dim=-1)
            o_hm[:, sq_s:sq_e, :] = torch.matmul(attn, vb).to(torch.bfloat16)

        np.savez(
            OUT / f"flash_attn_fp8_varlen_{name}.npz",
            q=fp8_to_np(q_hm.reshape(-1)),
            k=fp8_to_np(k_hm.reshape(-1)),
            v=fp8_to_np(v_hm.reshape(-1)),
            o=bf16_to_np(o_hm.reshape(-1).unsqueeze(0)).squeeze(0),
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            num_heads=np.uint32(H),
            batch=np.uint32(batch),
            max_seqlen_q=np.uint32(max(seq_lens_q)),
            total_q=np.uint32(total_q),
            total_kv=np.uint32(total_kv),
            scale=np.float32(scale),
        )

    _run([64, 128], [128, 64], name="a")
    _run([128, 256], [256, 128], name="b")


def gen_flash_attention_fp8_varlen_causal():
    """Generate FP8 varlen causal flash attention golden vectors (head-major layout).

    HEAD-MAJOR layout: Q,K,O = [num_heads, total_tokens, D], V = [num_heads, total_tokens, D].
    Causal masking applied per batch element.

    Two configs:
      flash_attn_fp8_varlen_causal_a.npz: B=2, H=4, seq_lens=[64, 128]
      flash_attn_fp8_varlen_causal_b.npz: B=2, H=4, seq_lens=[128, 256]
    """
    torch.manual_seed(44)

    def _run(seq_lens, H=4, d=128, name=""):
        batch = len(seq_lens)
        total = sum(seq_lens)
        cu_q = np.array([0] + list(np.cumsum(seq_lens)), dtype=np.uint32)
        cu_k = cu_q.copy()
        scale = 1.0 / (d ** 0.5)

        # HEAD-MAJOR layout: [H, total, D], FP8 e4m3
        q_hm = torch.randn(H, total, d, device="cuda").clamp(-448, 448).to(torch.float8_e4m3fn)
        k_hm = torch.randn(H, total, d, device="cuda").clamp(-448, 448).to(torch.float8_e4m3fn)
        v_hm = torch.randn(H, total, d, device="cuda").clamp(-448, 448).to(torch.float8_e4m3fn)

        o_hm = torch.zeros(H, total, d, dtype=torch.bfloat16, device="cuda")
        for b in range(batch):
            s, e = int(cu_q[b]), int(cu_q[b + 1])
            seq = e - s
            qb = q_hm[:, s:e, :].float()
            kb = k_hm[:, s:e, :].float()
            vb = v_hm[:, s:e, :].float()
            attn = torch.matmul(qb, kb.transpose(-2, -1)) * scale
            causal_mask = torch.triu(
                torch.ones(seq, seq, device="cuda"), diagonal=1
            ).bool()
            attn.masked_fill_(causal_mask.unsqueeze(0), float("-inf"))
            attn = torch.softmax(attn, dim=-1)
            o_hm[:, s:e, :] = torch.matmul(attn, vb).to(torch.bfloat16)

        np.savez(
            OUT / f"flash_attn_fp8_varlen_causal_{name}.npz",
            q=fp8_to_np(q_hm.reshape(-1)),
            k=fp8_to_np(k_hm.reshape(-1)),
            v=fp8_to_np(v_hm.reshape(-1)),
            o=bf16_to_np(o_hm.reshape(-1).unsqueeze(0)).squeeze(0),
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            num_heads=np.uint32(H),
            batch=np.uint32(batch),
            max_seqlen_q=np.uint32(max(seq_lens)),
            total_q=np.uint32(total),
            total_kv=np.uint32(total),
            scale=np.float32(scale),
        )

    _run([64, 128], name="a")
    _run([128, 256], name="b")


def gen_flash_attention_fp8_gqa():
    """FP8 GQA flash attention: Q has H_q heads, KV has H_kv heads.
    configs: B=2, H_q=8, H_kv=2, seq=[256,1024], causal=[False,True]
    """
    for causal in [False, True]:
        for seq in [256, 1024]:
            B, H_q, H_kv, d = 2, 8, 2, 128
            torch.manual_seed(42 + seq)
            q = torch.randn(B, H_q, seq, d, device="cuda").clamp(-448, 448).to(torch.float8_e4m3fn)
            k = torch.randn(B, H_kv, seq, d, device="cuda").clamp(-448, 448).to(torch.float8_e4m3fn)
            v = torch.randn(B, H_kv, seq, d, device="cuda").clamp(-448, 448).to(torch.float8_e4m3fn)
            scale = 1.0 / (d**0.5)
            # Reference: expand k,v to H_q heads
            groups = H_q // H_kv
            k_exp = k.repeat_interleave(groups, dim=1)  # [B, H_q, seq, d]
            v_exp = v.repeat_interleave(groups, dim=1)
            attn = torch.matmul(q.float(), k_exp.float().transpose(-2, -1)) * scale
            if causal:
                mask = torch.triu(torch.ones(seq, seq, device="cuda"), diagonal=1).bool()
                attn.masked_fill_(mask, float("-inf"))
            attn = torch.softmax(attn, dim=-1)
            o = torch.matmul(attn, v_exp.float()).to(torch.bfloat16)
            tag = f"{'causal' if causal else 'noncausal'}_s{seq}"
            np.savez(
                OUT / f"flash_attn_fp8_gqa_{tag}.npz",
                q=fp8_to_np(q), k=fp8_to_np(k), v=fp8_to_np(v),
                o=bf16_to_np(o),
                scale=np.float32(scale),
            )


def gen_flash_attention_fp8_varlen_gqa():
    """FP8 varlen GQA (causal and non-causal).
    H_q=8, H_kv=2, B=2
    a: seq_lens=[64,128] (causal), b: seq_lens=[128,256] (causal)
    c: seq_lens_q=[64,128], seq_lens_k=[128,64] (non-causal cross-attention)
    """
    torch.manual_seed(46)
    H_q, H_kv, d = 8, 2, 128
    scale = 1.0 / (d**0.5)
    groups = H_q // H_kv

    def _run_causal(seq_lens, name):
        batch = len(seq_lens)
        total = sum(seq_lens)
        cu_q = np.array([0] + list(np.cumsum(seq_lens)), dtype=np.uint32)
        cu_k = cu_q.copy()
        q_hm = torch.randn(H_q, total, d, device="cuda").clamp(-448, 448).to(torch.float8_e4m3fn)
        k_hm = torch.randn(H_kv, total, d, device="cuda").clamp(-448, 448).to(torch.float8_e4m3fn)
        v_hm = torch.randn(H_kv, total, d, device="cuda").clamp(-448, 448).to(torch.float8_e4m3fn)
        k_exp = k_hm.repeat_interleave(groups, dim=0)  # [H_q, total, d]
        v_exp = v_hm.repeat_interleave(groups, dim=0)
        o_hm = torch.zeros(H_q, total, d, dtype=torch.bfloat16, device="cuda")
        for b in range(batch):
            s, e = int(cu_q[b]), int(cu_q[b+1])
            seq = e - s
            qb = q_hm[:, s:e, :].float()
            kb = k_exp[:, s:e, :].float()
            vb = v_exp[:, s:e, :].float()
            attn = torch.matmul(qb, kb.transpose(-2, -1)) * scale
            causal_mask = torch.triu(torch.ones(seq, seq, device="cuda"), diagonal=1).bool()
            attn.masked_fill_(causal_mask.unsqueeze(0), float("-inf"))
            attn = torch.softmax(attn, dim=-1)
            o_hm[:, s:e, :] = torch.matmul(attn, vb).to(torch.bfloat16)
        np.savez(
            OUT / f"flash_attn_fp8_varlen_gqa_causal_{name}.npz",
            q=fp8_to_np(q_hm.reshape(-1)),
            k=fp8_to_np(k_hm.reshape(-1)),
            v=fp8_to_np(v_hm.reshape(-1)),
            o=bf16_to_np(o_hm.reshape(-1).unsqueeze(0)).squeeze(0),
            cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            num_heads_q=np.uint32(H_q), num_heads_kv=np.uint32(H_kv),
            batch=np.uint32(batch), max_seqlen_q=np.uint32(max(seq_lens)),
            total_q=np.uint32(total), total_kv=np.uint32(total), scale=np.float32(scale),
        )

    def _run_noncausal(seq_lens_q, seq_lens_k, name):
        batch = len(seq_lens_q)
        total_q = sum(seq_lens_q)
        total_kv = sum(seq_lens_k)
        cu_q = np.array([0] + list(np.cumsum(seq_lens_q)), dtype=np.uint32)
        cu_k = np.array([0] + list(np.cumsum(seq_lens_k)), dtype=np.uint32)
        q_hm = torch.randn(H_q, total_q, d, device="cuda").clamp(-448, 448).to(torch.float8_e4m3fn)
        k_hm = torch.randn(H_kv, total_kv, d, device="cuda").clamp(-448, 448).to(torch.float8_e4m3fn)
        v_hm = torch.randn(H_kv, total_kv, d, device="cuda").clamp(-448, 448).to(torch.float8_e4m3fn)
        k_exp = k_hm.repeat_interleave(groups, dim=0)
        v_exp = v_hm.repeat_interleave(groups, dim=0)
        o_hm = torch.zeros(H_q, total_q, d, dtype=torch.bfloat16, device="cuda")
        for b in range(batch):
            sq_s, sq_e = int(cu_q[b]), int(cu_q[b+1])
            sk_s, sk_e = int(cu_k[b]), int(cu_k[b+1])
            qb = q_hm[:, sq_s:sq_e, :].float()
            kb = k_exp[:, sk_s:sk_e, :].float()
            vb = v_exp[:, sk_s:sk_e, :].float()
            attn = torch.softmax(torch.matmul(qb, kb.transpose(-2, -1)) * scale, dim=-1)
            o_hm[:, sq_s:sq_e, :] = torch.matmul(attn, vb).to(torch.bfloat16)
        np.savez(
            OUT / f"flash_attn_fp8_varlen_gqa_{name}.npz",
            q=fp8_to_np(q_hm.reshape(-1)),
            k=fp8_to_np(k_hm.reshape(-1)),
            v=fp8_to_np(v_hm.reshape(-1)),
            o=bf16_to_np(o_hm.reshape(-1).unsqueeze(0)).squeeze(0),
            cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            num_heads_q=np.uint32(H_q), num_heads_kv=np.uint32(H_kv),
            batch=np.uint32(batch), max_seqlen_q=np.uint32(max(seq_lens_q)),
            total_q=np.uint32(total_q), total_kv=np.uint32(total_kv), scale=np.float32(scale),
        )

    _run_causal([64, 128], name="a")
    _run_causal([128, 256], name="b")
    _run_noncausal([64, 128], [128, 64], name="a")
    _run_noncausal([128, 256], [256, 128], name="b")


def gen_adamw():
    """Multi-step AdamW reference. After N steps with random grads, capture
    final (weight, master, m, v) for the kernel to match.

    We deliberately step several times with non-trivial state so a per-step
    bug accumulates and gets caught — single-step matches are too easy.
    """
    torch.manual_seed(0xADA)
    # Cover small (alignment edge cases) and medium sizes.
    for n in [256, 4096, 16384]:
        # Initial state.
        master_init = torch.randn(n, dtype=torch.float32, device="cuda") * 0.1
        weight_init = master_init.to(torch.bfloat16)
        m_init = torch.zeros(n, dtype=torch.float32, device="cuda")
        v_init = torch.zeros(n, dtype=torch.float32, device="cuda")

        # Optimizer hyperparameters.
        lr = 1e-3
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        weight_decay = 0.01
        n_steps = 8

        # Pre-generate gradients per step (BF16 → FP32 in-kernel).
        grads_bf16 = []
        for _ in range(n_steps):
            g = torch.randn(n, dtype=torch.bfloat16, device="cuda") * 0.05
            grads_bf16.append(g)

        # PyTorch reference: replicate the kernel's update math exactly so the
        # comparison is apples-to-apples. (torch.optim.AdamW uses native FP32
        # parameters; here we emulate the BF16 weight + FP32 master pattern.)
        master = master_init.clone()
        m = m_init.clone()
        v = v_init.clone()
        for step, g_bf16 in enumerate(grads_bf16, start=1):
            g = g_bf16.to(torch.float32)
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * g * g
            bc1 = 1 - beta1 ** step
            bc2 = 1 - beta2 ** step
            m_hat = m / bc1
            v_hat = v / bc2
            update = m_hat / (torch.sqrt(v_hat) + eps) + weight_decay * master
            master = master - lr * update
        weight_final = master.to(torch.bfloat16)

        np.savez(
            OUT / f"adamw_n{n}.npz",
            master_init=master_init.cpu().numpy(),
            weight_init=bf16_to_np(weight_init),
            m_init=m_init.cpu().numpy(),
            v_init=v_init.cpu().numpy(),
            grads=np.stack([bf16_to_np(g) for g in grads_bf16], axis=0),
            master_final=master.cpu().numpy(),
            weight_final=bf16_to_np(weight_final),
            m_final=m.cpu().numpy(),
            v_final=v.cpu().numpy(),
            lr=np.float32(lr),
            beta1=np.float32(beta1),
            beta2=np.float32(beta2),
            eps=np.float32(eps),
            weight_decay=np.float32(weight_decay),
            n_steps=np.int32(n_steps),
        )


def gen_qat_fakequant():
    """FP8 e4m3 fakequant golden via PyTorch's bit-exact float8_e4m3fn dtype.

    Saves per-tensor scale, BF16 input x, expected BF16 output y, and a
    second pair (in-range and out-of-range) for the STE backward.
    """
    torch.manual_seed(0xFA8E)
    n = 4096
    # Magnitudes that produce a mix of in-range and out-of-range when divided
    # by `scale`. With scale=0.01, x in [-2, 2] gives x/scale in [-200, 200],
    # comfortably inside e4m3's [-448, 448] range, so all elements quantize
    # without saturation — exercises the round-trip path.
    x = (torch.randn(n, dtype=torch.bfloat16, device="cuda") * 1.0)
    scale = 0.01
    # PyTorch fakequant: scale * round_to_e4m3(x / scale)
    x_scaled = x.to(torch.float32) / scale
    q = x_scaled.to(torch.float8_e4m3fn).to(torch.float32)
    y = (q * scale).to(torch.bfloat16)

    # STE backward: dy is in-range elements only.
    dy = torch.randn(n, dtype=torch.bfloat16, device="cuda") * 0.1
    in_range = (x_scaled.abs() <= 448.0)
    dx = torch.where(in_range, dy.to(torch.float32), torch.zeros_like(dy.to(torch.float32))).to(
        torch.bfloat16
    )

    np.savez(
        OUT / "qat_fakequant_fp8_e4m3.npz",
        x=bf16_to_np(x),
        y=bf16_to_np(y),
        dy=bf16_to_np(dy),
        dx=bf16_to_np(dx),
        scale=np.float32(scale),
    )


def gen_fa_backward_varlen():
    """FA backward varlen golden via PyTorch autograd on packed sequences.

    Layout: q/k/v/do/dq/dk/dv all [total_tokens, num_heads, d].
    cu_seqlens [batch+1] u32 holds cumulative offsets.

    We run the per-sequence backward through torch.nn.functional.
    scaled_dot_product_attention (which dispatches to flash on CUDA) and
    accumulate the dq/dk/dv from each sequence into the packed output.
    """
    torch.manual_seed(0xFA21)
    head_dim = 128
    num_heads = 4
    seqlens = [37, 64, 19, 128]  # mix of short / power-of-2 / odd
    total = sum(seqlens)
    cu = np.cumsum([0] + seqlens, dtype=np.uint32)

    q = torch.randn(total, num_heads, head_dim, dtype=torch.bfloat16, device="cuda") * 0.05
    k = torch.randn(total, num_heads, head_dim, dtype=torch.bfloat16, device="cuda") * 0.05
    v = torch.randn(total, num_heads, head_dim, dtype=torch.bfloat16, device="cuda") * 0.05
    do = torch.randn(total, num_heads, head_dim, dtype=torch.bfloat16, device="cuda") * 0.05

    q.requires_grad_()
    k.requires_grad_()
    v.requires_grad_()

    scale = 1.0 / (head_dim ** 0.5)

    o_packed = torch.zeros_like(q)
    for i in range(len(seqlens)):
        s, e = int(cu[i]), int(cu[i+1])
        # Per-sequence: [S, H, D] -> [1, H, S, D] for SDPA.
        q_i = q[s:e].permute(1, 0, 2).unsqueeze(0)
        k_i = k[s:e].permute(1, 0, 2).unsqueeze(0)
        v_i = v[s:e].permute(1, 0, 2).unsqueeze(0)
        scores = torch.matmul(q_i.float(), k_i.float().transpose(-2, -1)) * scale
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v_i.float())
        o_packed[s:e] = out.squeeze(0).permute(1, 0, 2).to(torch.bfloat16)

    # Backward via autograd on the packed loss.
    loss = (o_packed.float() * do.float()).sum()
    loss.backward()

    np.savez(
        OUT / "fa_backward_varlen_d128.npz",
        q=bf16_to_np(q.detach()),
        k=bf16_to_np(k.detach()),
        v=bf16_to_np(v.detach()),
        do=bf16_to_np(do),
        dq=bf16_to_np(q.grad.to(torch.bfloat16)),
        dk=bf16_to_np(k.grad.to(torch.bfloat16)),
        dv=bf16_to_np(v.grad.to(torch.bfloat16)),
        cu_seqlens=cu,
        num_heads=np.int32(num_heads),
        head_dim=np.int32(head_dim),
        scale=np.float32(scale),
    )


def gen_fa_backward_paged():
    """FA backward against paged KV cache. Builds a non-overlapping page
    table (no shared pages — training case) and runs PyTorch autograd.
    """
    torch.manual_seed(0xFA22)
    batch = 2
    num_heads = 4
    num_kv_heads = 4  # non-GQA for this test
    head_dim = 128
    seq = 256                 # seq_q = seq_kv (training case)
    page_size = 64
    pages_per_seq = (seq + page_size - 1) // page_size
    s_kv = pages_per_seq * page_size  # padded
    num_pages_total = batch * pages_per_seq

    # Page table: each batch gets pages_per_seq distinct physical pages.
    # No sharing => safe for non-atomic scatter.
    page_table = np.arange(batch * pages_per_seq, dtype=np.uint32).reshape(batch, pages_per_seq)
    max_pages = pages_per_seq

    # Build "logical" K/V in [B, H_kv, S_kv, D] for autograd, then materialize
    # to paged layout.
    q = torch.randn(batch, num_heads, seq, head_dim, dtype=torch.bfloat16, device="cuda") * 0.05
    k = torch.randn(batch, num_kv_heads, s_kv, head_dim, dtype=torch.bfloat16, device="cuda") * 0.05
    v = torch.randn(batch, num_kv_heads, s_kv, head_dim, dtype=torch.bfloat16, device="cuda") * 0.05
    do = torch.randn(batch, num_heads, seq, head_dim, dtype=torch.bfloat16, device="cuda") * 0.05

    q.requires_grad_(); k.requires_grad_(); v.requires_grad_()

    scale = 1.0 / (head_dim ** 0.5)
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v.float()).to(torch.bfloat16)
    loss = (out.float() * do.float()).sum()
    loss.backward()

    # Materialize K/V to paged: [num_pages, page_size, H_kv, D].
    k_paged = torch.zeros(num_pages_total, page_size, num_kv_heads, head_dim,
                          dtype=torch.bfloat16, device="cuda")
    v_paged = torch.zeros_like(k_paged)
    dk_paged = torch.zeros_like(k_paged)
    dv_paged = torch.zeros_like(k_paged)
    for b in range(batch):
        for p in range(pages_per_seq):
            phys = int(page_table[b, p])
            slot_start = p * page_size
            slot_end = slot_start + page_size
            # k[b, h, slot_start:slot_end, d] -> k_paged[phys, slot_in_page, h, d]
            k_paged[phys] = k.detach()[b, :, slot_start:slot_end, :].permute(1, 0, 2)
            v_paged[phys] = v.detach()[b, :, slot_start:slot_end, :].permute(1, 0, 2)
            dk_paged[phys] = k.grad[b, :, slot_start:slot_end, :].permute(1, 0, 2).to(torch.bfloat16)
            dv_paged[phys] = v.grad[b, :, slot_start:slot_end, :].permute(1, 0, 2).to(torch.bfloat16)

    np.savez(
        OUT / "fa_backward_paged_d128.npz",
        q=bf16_to_np(q.detach()),
        k_paged=bf16_to_np(k_paged),
        v_paged=bf16_to_np(v_paged),
        do=bf16_to_np(do),
        dq=bf16_to_np(q.grad.to(torch.bfloat16)),
        dk_paged=bf16_to_np(dk_paged),
        dv_paged=bf16_to_np(dv_paged),
        page_table=page_table,
        batch=np.int32(batch),
        num_heads=np.int32(num_heads),
        num_kv_heads=np.int32(num_kv_heads),
        seq=np.int32(seq),
        head_dim=np.int32(head_dim),
        page_size=np.int32(page_size),
        max_pages=np.int32(max_pages),
        scale=np.float32(scale),
    )


def gen_fa_backward_fp8kv():
    """FA backward with FP8-quantized K/V (per-tensor scale).

    The kernel implements `dequant(K_fp8) -> K_bf16` then runs the standard
    BF16 backward; we mirror that on the reference side so the golden is
    apples-to-apples.
    """
    torch.manual_seed(0xFA8F)
    batch = 2
    num_heads = 4
    seq = 128
    head_dim = 128
    scale = 1.0 / (head_dim ** 0.5)
    kv_scale = 1.0 / 64.0  # kv_scale * fp8 -> bf16 magnitude ~ 0.05 range

    q = torch.randn(batch, num_heads, seq, head_dim, dtype=torch.bfloat16, device="cuda") * 0.05
    do = torch.randn(batch, num_heads, seq, head_dim, dtype=torch.bfloat16, device="cuda") * 0.05

    # Generate K, V in FP32 -> quantize to FP8 -> dequantize back to BF16 for the
    # ref so the comparison is apples-to-apples (the kernel does the same
    # round-trip via dequant_fp8_bf16_pertensor). Keep magnitudes low enough
    # that no element overflows e4m3's [-448, 448] range (otherwise PyTorch
    # to(float8_e4m3fn) produces NaN for the FN variant; CUDA satfinite
    # would clamp instead — we want the references to agree).
    k_full = torch.randn(batch, num_heads, seq, head_dim, dtype=torch.float32, device="cuda") * 0.5
    v_full = torch.randn(batch, num_heads, seq, head_dim, dtype=torch.float32, device="cuda") * 0.5
    # Clamp to the dequant range so e4m3 round-trips without NaN.
    k_full = k_full.clamp(-(448 * kv_scale), 448 * kv_scale)
    v_full = v_full.clamp(-(448 * kv_scale), 448 * kv_scale)
    k_fp8 = (k_full / kv_scale).to(torch.float8_e4m3fn)
    v_fp8 = (v_full / kv_scale).to(torch.float8_e4m3fn)
    k_dequant = (k_fp8.to(torch.float32) * kv_scale).to(torch.bfloat16)
    v_dequant = (v_fp8.to(torch.float32) * kv_scale).to(torch.bfloat16)

    q.requires_grad_()
    k_dequant.requires_grad_()
    v_dequant.requires_grad_()

    scores = torch.matmul(q.float(), k_dequant.float().transpose(-2, -1)) * scale
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v_dequant.float()).to(torch.bfloat16)
    loss = (out.float() * do.float()).sum()
    loss.backward()

    # Pack FP8 to u8 for storage.
    k_fp8_u8 = k_fp8.view(torch.uint8).cpu().numpy()
    v_fp8_u8 = v_fp8.view(torch.uint8).cpu().numpy()

    np.savez(
        OUT / "fa_backward_fp8kv_d128.npz",
        q=bf16_to_np(q.detach()),
        k_fp8=k_fp8_u8,
        v_fp8=v_fp8_u8,
        do=bf16_to_np(do),
        dq=bf16_to_np(q.grad.to(torch.bfloat16)),
        dk=bf16_to_np(k_dequant.grad.to(torch.bfloat16)),
        dv=bf16_to_np(v_dequant.grad.to(torch.bfloat16)),
        batch=np.int32(batch),
        num_heads=np.int32(num_heads),
        seq=np.int32(seq),
        head_dim=np.int32(head_dim),
        scale=np.float32(scale),
        kv_scale=np.float32(kv_scale),
    )


if __name__ == "__main__":
    # Seed every RNG up front so a full run is bit-reproducible. Each gen_*
    # function additionally re-seeds at its own top, so regenerating a single
    # vector in isolation produces the same bytes as a full run.
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    print("Generating golden test vectors...")
    # Call EVERY gen_* generator defined in this module, in source order, so a
    # generator can never be silently orphaned: a fresh `python generate_golden.py`
    # must produce the golden for every kept test. (Each gen_* re-seeds at its top.)
    import types as _types

    _this = sys.modules[__name__]
    _gens = [
        v
        for k, v in vars(_this).items()
        if k.startswith("gen_") and isinstance(v, _types.FunctionType)
    ]
    _gens.sort(key=lambda f: f.__code__.co_firstlineno)
    for _fn in _gens:
        _fn()
        print(f"  {_fn.__name__[4:]} done")
    print(f"All golden vectors saved to {OUT}")
