use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};

use crate::error::{Result, SparkError};
use crate::module;

/// Launch MoE routing kernel.
///
/// Performs top-k expert selection with softmax gating weights.
///
/// `logits`: [num_tokens, num_experts] BF16 expert logits
/// `expert_ids`: [num_tokens, top_k] u32 output expert IDs
/// `weights`: [num_tokens, top_k] BF16 output gating weights (softmax over selected)
#[allow(clippy::too_many_arguments)]
pub fn moe_routing(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    logits: &CudaSlice<u16>,
    expert_ids: &mut CudaSlice<u32>,
    weights: &mut CudaSlice<u16>,
    num_tokens: u32,
    num_experts: u32,
    top_k: u32,
) -> Result<()> {
    if num_tokens == 0 {
        return Err(SparkError::InvalidArgument("num_tokens must be > 0".into()));
    }
    if num_experts == 0 {
        return Err(SparkError::InvalidArgument(
            "num_experts must be > 0".into(),
        ));
    }
    if top_k == 0 {
        return Err(SparkError::InvalidArgument("top_k must be > 0".into()));
    }
    if top_k > num_experts {
        return Err(SparkError::InvalidArgument(format!(
            "top_k ({top_k}) must be <= num_experts ({num_experts})"
        )));
    }
    // Kernel SMEM layout: 256 f32 logits + 8 f32 topk_val + 8 u32 topk_idx.
    // top_k > 8 corrupts the topk_idx slots; num_experts > 256 overruns the
    // logits array into topk_val/topk_idx (this was the Gemma-4-26B-A4B bug
    // when the kernel only had 64 logit slots).
    if top_k > 8 {
        return Err(SparkError::InvalidArgument(format!(
            "top_k ({top_k}) > 8 not supported by moe_routing kernel"
        )));
    }
    if num_experts > 256 {
        return Err(SparkError::InvalidArgument(format!(
            "num_experts ({num_experts}) > 256 not supported by moe_routing kernel"
        )));
    }
    if logits.len() < num_tokens as usize * num_experts as usize {
        return Err(SparkError::InvalidArgument(format!(
            "logits buffer too small: {} < {}",
            logits.len(),
            num_tokens as usize * num_experts as usize
        )));
    }
    if expert_ids.len() < num_tokens as usize * top_k as usize {
        return Err(SparkError::InvalidArgument(format!(
            "expert_ids buffer too small: {} < {}",
            expert_ids.len(),
            num_tokens as usize * top_k as usize
        )));
    }
    if weights.len() < num_tokens as usize * top_k as usize {
        return Err(SparkError::InvalidArgument(format!(
            "weights buffer too small: {} < {}",
            weights.len(),
            num_tokens as usize * top_k as usize
        )));
    }
    let func = module::load_kernel(ctx, "moe_routing", "moe_routing")?;

    let cfg = LaunchConfig {
        grid_dim: (num_tokens, 1, 1),
        block_dim: (32, 1, 1), // 1 warp
        shared_mem_bytes: 0,
    };

    unsafe {
        stream
            .launch_builder(&func)
            .arg(logits)
            .arg(expert_ids)
            .arg(weights)
            .arg(&num_tokens)
            .arg(&num_experts)
            .arg(&top_k)
            .launch(cfg)?;
    }

    Ok(())
}

/// Gemma-4 MoE router post-step: `top_k_w[t, j] *= per_expert_scale[ top_k_idx[t, j] ]`.
///
/// Multiplies each top-k routing weight by the corresponding expert's
/// learned per-expert scale. Used inside the Gemma-4-26B-A4B MoE block.
///
/// `top_k_w`: `[num_tokens, top_k]` BF16 (in/out)
/// `top_k_idx`: `[num_tokens, top_k]` u32 (in)
/// `per_expert_scale`: `[num_experts]` BF16 (in)
pub fn moe_apply_per_expert_scale_bf16(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    top_k_w: &mut CudaSlice<u16>,
    top_k_idx: &CudaSlice<u32>,
    per_expert_scale: &CudaSlice<u16>,
    num_tokens: u32,
    top_k: u32,
) -> Result<()> {
    if num_tokens == 0 || top_k == 0 {
        return Err(SparkError::InvalidArgument(
            "num_tokens and top_k must be > 0".into(),
        ));
    }
    let n_entries = num_tokens * top_k;
    if top_k_w.len() < n_entries as usize || top_k_idx.len() < n_entries as usize {
        return Err(SparkError::InvalidArgument(format!(
            "top_k_w / top_k_idx too small (need {n_entries})"
        )));
    }
    let func = module::load_kernel(
        ctx,
        "moe_apply_per_expert_scale_bf16",
        "moe_apply_per_expert_scale_bf16",
    )?;
    let threads = 256u32;
    let blocks = n_entries.div_ceil(threads).max(1);
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(top_k_w)
            .arg(top_k_idx)
            .arg(per_expert_scale)
            .arg(&n_entries)
            .launch(cfg)?;
    }
    Ok(())
}

/// Count tokens routed to each expert.
///
/// `expert_ids`: [total_entries] u32 (where total_entries = num_tokens * top_k)
/// `histogram`: [num_experts] u32 (output, caller must zero-initialize)
pub fn moe_histogram(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    expert_ids: &CudaSlice<u32>,
    histogram: &mut CudaSlice<u32>,
    num_entries: u32,
    num_experts: u32,
) -> Result<()> {
    if num_entries == 0 || num_experts == 0 {
        return Err(SparkError::InvalidArgument(
            "num_entries and num_experts must be > 0".into(),
        ));
    }
    if expert_ids.len() < num_entries as usize || histogram.len() < num_experts as usize {
        return Err(SparkError::InvalidArgument("buffers too small".into()));
    }
    let func = module::load_kernel(ctx, "moe_histogram", "moe_histogram")?;
    let grid_x = num_entries.div_ceil(256);
    let cfg = LaunchConfig {
        grid_dim: (grid_x, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(expert_ids)
            .arg(histogram)
            .arg(&num_entries)
            .arg(&num_experts)
            .launch(cfg)?;
    }
    Ok(())
}

/// Multi-node expert-parallel helper: count tokens per target device.
///
/// For each entry i, computes `device = expert_ids[i] / experts_per_node` and
/// atomically increments `device_counts[device]`. These counts are the
/// per-device send sizes for an NCCL alltoall in expert-parallel MoE.
///
/// `expert_ids`: [total_entries] u32
/// `device_counts`: [num_devices] u32 (caller zero-initializes)
pub fn moe_device_counts(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    expert_ids: &CudaSlice<u32>,
    device_counts: &mut CudaSlice<u32>,
    total_entries: u32,
    experts_per_node: u32,
) -> Result<()> {
    if total_entries == 0 || experts_per_node == 0 {
        return Err(SparkError::InvalidArgument(
            "total_entries and experts_per_node must be > 0".into(),
        ));
    }
    if expert_ids.len() < total_entries as usize {
        return Err(SparkError::InvalidArgument("expert_ids too small".into()));
    }
    let func = module::load_kernel(ctx, "moe_device_counts", "moe_device_counts")?;
    let grid_x = total_entries.div_ceil(256);
    let cfg = LaunchConfig {
        grid_dim: (grid_x, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(expert_ids)
            .arg(device_counts)
            .arg(&total_entries)
            .arg(&experts_per_node)
            .launch(cfg)?;
    }
    Ok(())
}

/// Multi-node helper: convert per-device token counts to exclusive-prefix-sum
/// device offsets (= NCCL alltoall send displacements). Equivalent to
/// `moe_expert_offsets` with num_experts = num_devices; provided for clarity
/// in distributed MoE dispatch code.
pub fn moe_device_offsets(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    device_counts: &CudaSlice<u32>,
    device_offsets: &mut CudaSlice<u32>,
    num_devices: u32,
) -> Result<()> {
    moe_expert_offsets(ctx, stream, device_counts, device_offsets, num_devices)
}

/// Convert expert histogram to exclusive-prefix-sum offsets.
///
/// `histogram`: [num_experts] u32
/// `offsets`: [num_experts+1] u32, `offsets[0] = 0`, `offsets[e+1] = offsets[e] + histogram[e]`
pub fn moe_expert_offsets(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    histogram: &CudaSlice<u32>,
    offsets: &mut CudaSlice<u32>,
    num_experts: u32,
) -> Result<()> {
    if num_experts == 0 {
        return Err(SparkError::InvalidArgument(
            "num_experts must be > 0".into(),
        ));
    }
    if histogram.len() < num_experts as usize || offsets.len() < num_experts as usize + 1 {
        return Err(SparkError::InvalidArgument("buffers too small".into()));
    }
    let func = module::load_kernel(ctx, "moe_expert_offsets", "moe_expert_offsets")?;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(histogram)
            .arg(offsets)
            .arg(&num_experts)
            .launch(cfg)?;
    }
    Ok(())
}

/// Permute activations into expert-sorted order.
///
/// Produces `permuted_activations[dst] = activations[src_token]` where `dst`
/// is assigned per-expert via atomic cursor bump, and `inverse_index[dst] =
/// original entry index (token*top_k + k)`.
///
/// `activations`: [num_tokens, hidden] BF16
/// `expert_ids`: [num_tokens * top_k] u32
/// `offsets`: [num_experts+1] u32 (from `moe_expert_offsets`)
/// `cursor`: [num_experts] u32 scratch (caller zero-initializes)
/// `permuted_activations`: [num_tokens * top_k, hidden] BF16 output
/// `inverse_index`: [num_tokens * top_k] u32 output
#[allow(clippy::too_many_arguments)]
pub fn moe_permute(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    activations: &CudaSlice<u16>,
    expert_ids: &CudaSlice<u32>,
    offsets: &CudaSlice<u32>,
    cursor: &mut CudaSlice<u32>,
    permuted_activations: &mut CudaSlice<u16>,
    inverse_index: &mut CudaSlice<u32>,
    num_tokens: u32,
    top_k: u32,
    hidden: u32,
) -> Result<()> {
    if num_tokens == 0 || top_k == 0 || hidden == 0 {
        return Err(SparkError::InvalidArgument(
            "num_tokens, top_k, hidden must be > 0".into(),
        ));
    }
    let total_entries = num_tokens * top_k;
    if activations.len() < (num_tokens * hidden) as usize {
        return Err(SparkError::InvalidArgument("activations too small".into()));
    }
    if expert_ids.len() < total_entries as usize {
        return Err(SparkError::InvalidArgument("expert_ids too small".into()));
    }
    if permuted_activations.len() < (total_entries * hidden) as usize {
        return Err(SparkError::InvalidArgument(
            "permuted_activations too small".into(),
        ));
    }
    if inverse_index.len() < total_entries as usize {
        return Err(SparkError::InvalidArgument(
            "inverse_index too small".into(),
        ));
    }

    let func = module::load_kernel(ctx, "moe_permute", "moe_permute")?;
    let cfg = LaunchConfig {
        grid_dim: (total_entries, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(activations)
            .arg(expert_ids)
            .arg(offsets)
            .arg(cursor)
            .arg(permuted_activations)
            .arg(inverse_index)
            .arg(&top_k)
            .arg(&hidden)
            .launch(cfg)?;
    }
    Ok(())
}

/// Pack activations grouped by *target device* (not by expert). Handles the
/// general case where `expert_to_device[num_experts]` is an arbitrary table —
/// e.g. non-uniform experts-per-node or dynamic expert placement. For the
/// common contiguous case (experts `[d*E, (d+1)*E)` → device `d`),
/// `moe_permute` already yields a device-grouped buffer; this kernel is the
/// general fallback.
///
/// Arguments:
///  - `activations`: `[num_tokens, hidden]` BF16
///  - `expert_ids`: `[num_tokens * top_k]` u32
///  - `expert_to_device`: `[num_experts]` u32
///  - `device_offsets`: `[num_devices+1]` u32 (prefix sum of device_counts)
///  - `device_cursor`: `[num_devices]` u32 scratch (caller zero-initializes)
///  - `permuted`: `[num_tokens * top_k, hidden]` BF16 output
///  - `inverse_index`: `[num_tokens * top_k]` u32 output (feed into `moe_unpermute`
///    after the alltoall return trip)
#[allow(clippy::too_many_arguments)]
pub fn moe_device_permute(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    activations: &CudaSlice<u16>,
    expert_ids: &CudaSlice<u32>,
    expert_to_device: &CudaSlice<u32>,
    device_offsets: &CudaSlice<u32>,
    device_cursor: &mut CudaSlice<u32>,
    permuted: &mut CudaSlice<u16>,
    inverse_index: &mut CudaSlice<u32>,
    num_tokens: u32,
    top_k: u32,
    hidden: u32,
) -> Result<()> {
    if num_tokens == 0 || top_k == 0 || hidden == 0 {
        return Err(SparkError::InvalidArgument(
            "num_tokens, top_k, hidden must be > 0".into(),
        ));
    }
    let total_entries = num_tokens * top_k;
    if activations.len() < (num_tokens * hidden) as usize {
        return Err(SparkError::InvalidArgument("activations too small".into()));
    }
    if expert_ids.len() < total_entries as usize {
        return Err(SparkError::InvalidArgument("expert_ids too small".into()));
    }
    if permuted.len() < (total_entries * hidden) as usize {
        return Err(SparkError::InvalidArgument("permuted too small".into()));
    }
    if inverse_index.len() < total_entries as usize {
        return Err(SparkError::InvalidArgument(
            "inverse_index too small".into(),
        ));
    }

    let func = module::load_kernel(ctx, "moe_device_permute", "moe_device_permute")?;
    let cfg = LaunchConfig {
        grid_dim: (total_entries, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(activations)
            .arg(expert_ids)
            .arg(expert_to_device)
            .arg(device_offsets)
            .arg(device_cursor)
            .arg(permuted)
            .arg(inverse_index)
            .arg(&top_k)
            .arg(&hidden)
            .launch(cfg)?;
    }
    Ok(())
}

/// Unpermute expert outputs back to token order, applying gating weights.
///
/// Accumulates into an FP32 buffer (must be zeroed by caller). Caller can
/// cast to BF16 as a separate step if desired.
///
/// `permuted_out`: [num_tokens * top_k, hidden] BF16
/// `inverse_index`: [num_tokens * top_k] u32 (from moe_permute)
/// `weights`: [num_tokens * top_k] BF16 (from moe_routing, flattened)
/// `out`: [num_tokens, hidden] F32 (output, zeroed)
#[allow(clippy::too_many_arguments)]
pub fn moe_unpermute(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    permuted_out: &CudaSlice<u16>,
    inverse_index: &CudaSlice<u32>,
    weights: &CudaSlice<u16>,
    out: &mut CudaSlice<f32>,
    num_tokens: u32,
    top_k: u32,
    hidden: u32,
) -> Result<()> {
    let total_entries = num_tokens * top_k;
    if permuted_out.len() < (total_entries * hidden) as usize {
        return Err(SparkError::InvalidArgument("permuted_out too small".into()));
    }
    if inverse_index.len() < total_entries as usize {
        return Err(SparkError::InvalidArgument(
            "inverse_index too small".into(),
        ));
    }
    if weights.len() < total_entries as usize {
        return Err(SparkError::InvalidArgument("weights too small".into()));
    }
    if out.len() < (num_tokens * hidden) as usize {
        return Err(SparkError::InvalidArgument("out too small".into()));
    }
    let func = module::load_kernel(ctx, "moe_unpermute", "moe_unpermute")?;
    let cfg = LaunchConfig {
        grid_dim: (total_entries, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(permuted_out)
            .arg(inverse_index)
            .arg(weights)
            .arg(out)
            .arg(&top_k)
            .arg(&hidden)
            .launch(cfg)?;
    }
    Ok(())
}

/// MoE grouped BF16 GEMM: compute `Y_e = X_e @ W_e` for all experts in one launch.
///
/// `a_permuted`: [total_tokens, K] BF16 (expert-sorted input activations)
/// `b_stacked`: [num_experts, K, N] BF16 (stacked weight matrices)
/// `c_permuted`: [total_tokens, N] BF16 (expert-sorted output; caller must
///               unpermute if needed via moe_unpermute)
/// `expert_offsets`: [num_experts+1] u32 (per-expert cumulative token counts)
/// `m_max`: upper bound on tokens-per-expert for grid sizing (typically set to
///          num_tokens*top_k / num_experts rounded up, or pessimistically to
///          total tokens when experts are imbalanced)
#[allow(clippy::too_many_arguments)]
pub fn gemm_bf16_grouped(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    a_permuted: &CudaSlice<u16>,
    b_stacked: &CudaSlice<u16>,
    c_permuted: &mut CudaSlice<u16>,
    expert_offsets: &CudaSlice<u32>,
    num_experts: u32,
    m_max: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    if num_experts == 0 || m_max == 0 || n == 0 || k == 0 {
        return Err(SparkError::InvalidArgument(
            "num_experts, m_max, n, k must be > 0".into(),
        ));
    }
    if k & 15 != 0 {
        return Err(SparkError::InvalidArgument(
            "k must be multiple of 16".into(),
        ));
    }
    if expert_offsets.len() < num_experts as usize + 1 {
        return Err(SparkError::InvalidArgument(
            "expert_offsets too small".into(),
        ));
    }
    if b_stacked.len() < (num_experts * k * n) as usize {
        return Err(SparkError::InvalidArgument("b_stacked too small".into()));
    }

    let func = module::load_kernel(ctx, "gemm_bf16_grouped", "gemm_bf16_grouped")?;
    // Simple reference implementation: BM=BN=32, 32 threads per CTA.
    let grid_n = n.div_ceil(32);
    let grid_m = m_max.div_ceil(32);
    let cfg = LaunchConfig {
        grid_dim: (grid_n, grid_m, num_experts),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(a_permuted)
            .arg(b_stacked)
            .arg(c_permuted)
            .arg(expert_offsets)
            .arg(&n)
            .arg(&k)
            .launch(cfg)?;
    }
    Ok(())
}

/// v2 of MXFP8 grouped GEMM: vectorized B loads + UE8M0 scale hoist.
/// Same signature as `gemm_mxfp8_grouped_mma`; byte-exact replacement.
#[allow(clippy::too_many_arguments)]
pub fn gemm_mxfp8_grouped_mma_v2(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    a_permuted: &CudaSlice<u16>,
    b_stacked: &CudaSlice<u8>,
    b_scales: &CudaSlice<u8>,
    c_permuted: &mut CudaSlice<u16>,
    expert_offsets: &CudaSlice<u32>,
    num_experts: u32,
    m_max: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    if num_experts == 0 || m_max == 0 || n == 0 || k == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    if n & 31 != 0 {
        return Err(SparkError::InvalidArgument(
            "n must be multiple of 32".into(),
        ));
    }
    if k & 31 != 0 {
        return Err(SparkError::InvalidArgument(
            "k must be multiple of 32".into(),
        ));
    }
    if expert_offsets.len() < num_experts as usize + 1 {
        return Err(SparkError::InvalidArgument(
            "expert_offsets too small".into(),
        ));
    }
    if b_stacked.len() < (num_experts * k * n) as usize {
        return Err(SparkError::InvalidArgument("b_stacked too small".into()));
    }
    let scales_need = (num_experts * n * (k / 32)) as usize;
    if b_scales.len() < scales_need {
        return Err(SparkError::InvalidArgument(format!(
            "b_scales too small: need {scales_need}"
        )));
    }
    let func = module::load_kernel(
        ctx,
        "gemm_mxfp8_grouped_mma_v2",
        "gemm_mxfp8_grouped_mma_v2",
    )?;
    let cfg = LaunchConfig {
        grid_dim: (n.div_ceil(32), m_max.div_ceil(32), num_experts),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(a_permuted)
            .arg(b_stacked)
            .arg(b_scales)
            .arg(c_permuted)
            .arg(expert_offsets)
            .arg(&n)
            .arg(&k)
            .launch(cfg)?;
    }
    Ok(())
}

/// v2 of generic per-expert FP8 grouped GEMM: vectorized B loads.
/// Same signature as `gemm_fp8_grouped_mma`; byte-exact replacement.
#[allow(clippy::too_many_arguments)]
pub fn gemm_fp8_grouped_mma_v2(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    a_permuted: &CudaSlice<u16>,
    b_stacked: &CudaSlice<u8>,
    b_scales: &CudaSlice<f32>,
    c_permuted: &mut CudaSlice<u16>,
    expert_offsets: &CudaSlice<u32>,
    num_experts: u32,
    m_max: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    if num_experts == 0 || m_max == 0 || n == 0 || k == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    if n & 31 != 0 {
        return Err(SparkError::InvalidArgument(
            "n must be multiple of 32".into(),
        ));
    }
    if k & 15 != 0 {
        return Err(SparkError::InvalidArgument(
            "k must be multiple of 16".into(),
        ));
    }
    if expert_offsets.len() < num_experts as usize + 1 {
        return Err(SparkError::InvalidArgument(
            "expert_offsets too small".into(),
        ));
    }
    if b_stacked.len() < (num_experts * k * n) as usize {
        return Err(SparkError::InvalidArgument("b_stacked too small".into()));
    }
    if b_scales.len() < num_experts as usize {
        return Err(SparkError::InvalidArgument("b_scales too small".into()));
    }
    let func = module::load_kernel(ctx, "gemm_fp8_grouped_mma_v2", "gemm_fp8_grouped_mma_v2")?;
    let cfg = LaunchConfig {
        grid_dim: (n.div_ceil(32), m_max.div_ceil(32), num_experts),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(a_permuted)
            .arg(b_stacked)
            .arg(b_scales)
            .arg(c_permuted)
            .arg(expert_offsets)
            .arg(&n)
            .arg(&k)
            .launch(cfg)?;
    }
    Ok(())
}

/// v2 of NVFP4 FP8-scale grouped GEMM: vectorized B loads (1 v2.b32 = 16 FP4)
/// + FP8 scale hoist. Same signature as `gemm_nvfp4_fp8scale_grouped_mma`.
#[allow(clippy::too_many_arguments)]
pub fn gemm_nvfp4_fp8scale_grouped_mma_v2(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    a_permuted: &CudaSlice<u16>,
    b_stacked: &CudaSlice<u8>,
    b_scales: &CudaSlice<u8>,
    c_permuted: &mut CudaSlice<u16>,
    expert_offsets: &CudaSlice<u32>,
    num_experts: u32,
    m_max: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    if num_experts == 0 || m_max == 0 || n == 0 || k == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    if n & 31 != 0 {
        return Err(SparkError::InvalidArgument(
            "n must be multiple of 32".into(),
        ));
    }
    if k & 15 != 0 {
        return Err(SparkError::InvalidArgument(
            "k must be multiple of 16".into(),
        ));
    }
    if expert_offsets.len() < num_experts as usize + 1 {
        return Err(SparkError::InvalidArgument(
            "expert_offsets too small".into(),
        ));
    }
    let b_bytes = (num_experts * n * k / 2) as usize;
    if b_stacked.len() < b_bytes {
        return Err(SparkError::InvalidArgument("b_stacked too small".into()));
    }
    let scales_need = (num_experts * n * (k / 16)) as usize;
    if b_scales.len() < scales_need {
        return Err(SparkError::InvalidArgument("b_scales too small".into()));
    }
    let func = module::load_kernel(
        ctx,
        "gemm_nvfp4_fp8scale_grouped_mma_v2",
        "gemm_nvfp4_fp8scale_grouped_mma_v2",
    )?;
    let cfg = LaunchConfig {
        grid_dim: (n.div_ceil(32), m_max.div_ceil(32), num_experts),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(a_permuted)
            .arg(b_stacked)
            .arg(b_scales)
            .arg(c_permuted)
            .arg(expert_offsets)
            .arg(&n)
            .arg(&k)
            .launch(cfg)?;
    }
    Ok(())
}

/// Modernized NVFP4 grouped GEMM matching CUTLASS/NVIDIA convention:
/// FP8 E4M3 scales per 16-K-block (4× less scale memory vs the FP32-scale
/// variant). Directly chains with `quant_bf16_to_nvfp4` which outputs FP8
/// scales. Decode path same as NVFP4: 16-entry BF16 LUT for FP4 values,
/// FP8 scales decoded via cvt.rn.f16x2.e4m3x2 into FP32 at block boundary.
///
/// Layout:
///  - `a_permuted`: [total_tokens, K] BF16
///  - `b_stacked`:  [num_experts, N, K/2] u8 (nibble-packed E2M1)
///  - `b_scales`:   [num_experts, N, K/16] u8 (FP8 E4M3)
///  - `c_permuted`: [total_tokens, N] BF16
///
/// Constraints: N multiple of 32, K multiple of 16.
#[allow(clippy::too_many_arguments)]
pub fn gemm_nvfp4_fp8scale_grouped_mma(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    a_permuted: &CudaSlice<u16>,
    b_stacked: &CudaSlice<u8>,
    b_scales: &CudaSlice<u8>,
    c_permuted: &mut CudaSlice<u16>,
    expert_offsets: &CudaSlice<u32>,
    num_experts: u32,
    m_max: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    if num_experts == 0 || m_max == 0 || n == 0 || k == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    if n & 31 != 0 {
        return Err(SparkError::InvalidArgument(
            "n must be multiple of 32".into(),
        ));
    }
    if k & 15 != 0 {
        return Err(SparkError::InvalidArgument(
            "k must be multiple of 16".into(),
        ));
    }
    if expert_offsets.len() < num_experts as usize + 1 {
        return Err(SparkError::InvalidArgument(
            "expert_offsets too small".into(),
        ));
    }
    let b_bytes = (num_experts * n * k / 2) as usize;
    if b_stacked.len() < b_bytes {
        return Err(SparkError::InvalidArgument("b_stacked too small".into()));
    }
    let scales_need = (num_experts * n * (k / 16)) as usize;
    if b_scales.len() < scales_need {
        return Err(SparkError::InvalidArgument(format!(
            "b_scales too small: need {scales_need}"
        )));
    }

    let func = module::load_kernel(
        ctx,
        "gemm_nvfp4_fp8scale_grouped_mma",
        "gemm_nvfp4_fp8scale_grouped_mma",
    )?;
    let grid_n = n.div_ceil(32);
    let grid_m = m_max.div_ceil(32);
    let cfg = LaunchConfig {
        grid_dim: (grid_n, grid_m, num_experts),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(a_permuted)
            .arg(b_stacked)
            .arg(b_scales)
            .arg(c_permuted)
            .arg(expert_offsets)
            .arg(&n)
            .arg(&k)
            .launch(cfg)?;
    }
    Ok(())
}

/// v2 of MXFP4 grouped GEMM with vectorized B loads (gpt-oss-120b hot path).
/// Each thread handles 1 col × 16 K-rows = 8 packed FP4 bytes per inner iter
/// (1 ld.global.v2.b32 = 16 FP4). UE8M0 scale hoisted into a register at K-block
/// boundary. Same signature as `gemm_mxfp4_grouped_mma`; byte-exact equivalent.
#[allow(clippy::too_many_arguments)]
pub fn gemm_mxfp4_grouped_mma_v2(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    a_permuted: &CudaSlice<u16>,
    b_stacked: &CudaSlice<u8>,
    b_scales: &CudaSlice<u8>,
    c_permuted: &mut CudaSlice<u16>,
    expert_offsets: &CudaSlice<u32>,
    num_experts: u32,
    m_max: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    if num_experts == 0 || m_max == 0 || n == 0 || k == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    if n & 31 != 0 {
        return Err(SparkError::InvalidArgument(
            "n must be multiple of 32".into(),
        ));
    }
    if k & 31 != 0 {
        return Err(SparkError::InvalidArgument(
            "k must be multiple of 32".into(),
        ));
    }
    if expert_offsets.len() < num_experts as usize + 1 {
        return Err(SparkError::InvalidArgument(
            "expert_offsets too small".into(),
        ));
    }
    let b_bytes = (num_experts * n * k / 2) as usize;
    if b_stacked.len() < b_bytes {
        return Err(SparkError::InvalidArgument("b_stacked too small".into()));
    }
    let scales_need = (num_experts * n * (k / 32)) as usize;
    if b_scales.len() < scales_need {
        return Err(SparkError::InvalidArgument(format!(
            "b_scales too small: need {scales_need}"
        )));
    }

    let func = module::load_kernel(
        ctx,
        "gemm_mxfp4_grouped_mma_v2",
        "gemm_mxfp4_grouped_mma_v2",
    )?;
    let grid_n = n.div_ceil(32);
    let grid_m = m_max.div_ceil(32);
    let cfg = LaunchConfig {
        grid_dim: (grid_n, grid_m, num_experts),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(a_permuted)
            .arg(b_stacked)
            .arg(b_scales)
            .arg(c_permuted)
            .arg(expert_offsets)
            .arg(&n)
            .arg(&k)
            .launch(cfg)?;
    }
    Ok(())
}

/// MXFP4 grouped GEMM using BF16 MMA (gpt-oss-120b path). Same 4-bit E2M1
/// nibble-packed B as NVFP4 but with UE8M0 scales per 32-K-block (instead of
/// FP32 per 16). FP4 decoded via 16-entry SMEM LUT; UE8M0 byte `e` decoded as
/// `2^(e-127)` by shifting into FP32 exponent field.
///
/// Layout:
///  - `a_permuted`: [total_tokens, K] BF16
///  - `b_stacked`:  [num_experts, N, K/2] u8 (nibble-packed E2M1)
///  - `b_scales`:   [num_experts, N, K/32] u8 (UE8M0)
///  - `c_permuted`: [total_tokens, N] BF16
///
/// Constraints: N multiple of 32, K multiple of 32.
#[allow(clippy::too_many_arguments)]
pub fn gemm_mxfp4_grouped_mma(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    a_permuted: &CudaSlice<u16>,
    b_stacked: &CudaSlice<u8>,
    b_scales: &CudaSlice<u8>,
    c_permuted: &mut CudaSlice<u16>,
    expert_offsets: &CudaSlice<u32>,
    num_experts: u32,
    m_max: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    if num_experts == 0 || m_max == 0 || n == 0 || k == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    if n & 31 != 0 {
        return Err(SparkError::InvalidArgument(
            "n must be multiple of 32".into(),
        ));
    }
    if k & 31 != 0 {
        return Err(SparkError::InvalidArgument(
            "k must be multiple of 32".into(),
        ));
    }
    if expert_offsets.len() < num_experts as usize + 1 {
        return Err(SparkError::InvalidArgument(
            "expert_offsets too small".into(),
        ));
    }
    let b_bytes = (num_experts * n * k / 2) as usize;
    if b_stacked.len() < b_bytes {
        return Err(SparkError::InvalidArgument("b_stacked too small".into()));
    }
    let scales_need = (num_experts * n * (k / 32)) as usize;
    if b_scales.len() < scales_need {
        return Err(SparkError::InvalidArgument(format!(
            "b_scales too small: need {scales_need}"
        )));
    }

    let func = module::load_kernel(ctx, "gemm_mxfp4_grouped_mma", "gemm_mxfp4_grouped_mma")?;
    let grid_n = n.div_ceil(32);
    let grid_m = m_max.div_ceil(32);
    let cfg = LaunchConfig {
        grid_dim: (grid_n, grid_m, num_experts),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(a_permuted)
            .arg(b_stacked)
            .arg(b_scales)
            .arg(c_permuted)
            .arg(expert_offsets)
            .arg(&n)
            .arg(&k)
            .launch(cfg)?;
    }
    Ok(())
}

/// NVFP4 grouped GEMM using BF16 MMA (GLM-5 path). BF16 A × NVFP4 B with
/// FP32 scale per 16-K-block per column. NVFP4 values are 4-bit E2M1 nibbles
/// (2 per byte); decoded via a 16-entry BF16 LUT in SMEM.
///
/// Layout:
///  - `a_permuted`: [total_tokens, K] BF16
///  - `b_stacked`:  [num_experts, N, K/2] u8 (nibble-packed E2M1; low nibble = even K, high = odd K)
///  - `b_scales`:   [num_experts, N, K/16] FP32
///  - `c_permuted`: [total_tokens, N] BF16
///
/// Constraints: N multiple of 32, K multiple of 16.
#[allow(clippy::too_many_arguments)]
pub fn gemm_nvfp4_grouped_mma(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    a_permuted: &CudaSlice<u16>,
    b_stacked: &CudaSlice<u8>,
    b_scales: &CudaSlice<f32>,
    c_permuted: &mut CudaSlice<u16>,
    expert_offsets: &CudaSlice<u32>,
    num_experts: u32,
    m_max: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    if num_experts == 0 || m_max == 0 || n == 0 || k == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    if n & 31 != 0 {
        return Err(SparkError::InvalidArgument(
            "n must be multiple of 32".into(),
        ));
    }
    if k & 15 != 0 {
        return Err(SparkError::InvalidArgument(
            "k must be multiple of 16".into(),
        ));
    }
    if expert_offsets.len() < num_experts as usize + 1 {
        return Err(SparkError::InvalidArgument(
            "expert_offsets too small".into(),
        ));
    }
    let b_bytes = (num_experts * n * k / 2) as usize;
    if b_stacked.len() < b_bytes {
        return Err(SparkError::InvalidArgument("b_stacked too small".into()));
    }
    let scales_need = (num_experts * n * (k / 16)) as usize;
    if b_scales.len() < scales_need {
        return Err(SparkError::InvalidArgument(format!(
            "b_scales too small: need {scales_need}"
        )));
    }

    let func = module::load_kernel(ctx, "gemm_nvfp4_grouped_mma", "gemm_nvfp4_grouped_mma")?;
    let grid_n = n.div_ceil(32);
    let grid_m = m_max.div_ceil(32);
    let cfg = LaunchConfig {
        grid_dim: (grid_n, grid_m, num_experts),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(a_permuted)
            .arg(b_stacked)
            .arg(b_scales)
            .arg(c_permuted)
            .arg(expert_offsets)
            .arg(&n)
            .arg(&k)
            .launch(cfg)?;
    }
    Ok(())
}

/// Generic FP8 grouped GEMM with per-expert scalar scale (BF16 A × FP8 E4M3 B).
/// Simpler than `gemm_fp8_block128_grouped_mma`: one FP32 scale per expert,
/// applied at epilogue. Covers MoE models with per-expert FP8 quantization but
/// without DeepSeek's per-column-per-K-block scaling.
///
/// Layout:
///  - `a_permuted`: [total_tokens, K] BF16
///  - `b_stacked`:  [num_experts, N, K] FP8 E4M3
///  - `b_scales`:   [num_experts] FP32 (one scalar per expert)
///  - `c_permuted`: [total_tokens, N] BF16
///
/// Constraints: N multiple of 32, K multiple of 16.
#[allow(clippy::too_many_arguments)]
pub fn gemm_fp8_grouped_mma(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    a_permuted: &CudaSlice<u16>,
    b_stacked: &CudaSlice<u8>,
    b_scales: &CudaSlice<f32>,
    c_permuted: &mut CudaSlice<u16>,
    expert_offsets: &CudaSlice<u32>,
    num_experts: u32,
    m_max: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    if num_experts == 0 || m_max == 0 || n == 0 || k == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    if n & 31 != 0 {
        return Err(SparkError::InvalidArgument(
            "n must be multiple of 32".into(),
        ));
    }
    if k & 15 != 0 {
        return Err(SparkError::InvalidArgument(
            "k must be multiple of 16".into(),
        ));
    }
    if expert_offsets.len() < num_experts as usize + 1 {
        return Err(SparkError::InvalidArgument(
            "expert_offsets too small".into(),
        ));
    }
    if b_stacked.len() < (num_experts * k * n) as usize {
        return Err(SparkError::InvalidArgument("b_stacked too small".into()));
    }
    if b_scales.len() < num_experts as usize {
        return Err(SparkError::InvalidArgument("b_scales too small".into()));
    }

    let func = module::load_kernel(ctx, "gemm_fp8_grouped_mma", "gemm_fp8_grouped_mma")?;
    let grid_n = n.div_ceil(32);
    let grid_m = m_max.div_ceil(32);
    let cfg = LaunchConfig {
        grid_dim: (grid_n, grid_m, num_experts),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(a_permuted)
            .arg(b_stacked)
            .arg(b_scales)
            .arg(c_permuted)
            .arg(expert_offsets)
            .arg(&n)
            .arg(&k)
            .launch(cfg)?;
    }
    Ok(())
}

/// MXFP8 grouped GEMM using BF16 MMA: BF16 A × FP8 E4M3 B with per-32-K-block
/// UE8M0 scales (FlashInfer/SGLang MXFP8 MoE format). Scale byte `e` is decoded
/// as 2^(e - 127) by shifting into the FP32 exponent field.
///
/// Layout:
///  - `a_permuted`: [total_tokens, K] BF16
///  - `b_stacked`:  [num_experts, N, K] FP8 E4M3
///  - `b_scales`:   [num_experts, N, K/32] u8 (UE8M0)
///  - `c_permuted`: [total_tokens, N] BF16
///
/// Constraints: N multiple of 32, K multiple of 32.
#[allow(clippy::too_many_arguments)]
pub fn gemm_mxfp8_grouped_mma(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    a_permuted: &CudaSlice<u16>,
    b_stacked: &CudaSlice<u8>,
    b_scales: &CudaSlice<u8>,
    c_permuted: &mut CudaSlice<u16>,
    expert_offsets: &CudaSlice<u32>,
    num_experts: u32,
    m_max: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    if num_experts == 0 || m_max == 0 || n == 0 || k == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    if n & 31 != 0 {
        return Err(SparkError::InvalidArgument(
            "n must be multiple of 32".into(),
        ));
    }
    if k & 31 != 0 {
        return Err(SparkError::InvalidArgument(
            "k must be multiple of 32".into(),
        ));
    }
    if expert_offsets.len() < num_experts as usize + 1 {
        return Err(SparkError::InvalidArgument(
            "expert_offsets too small".into(),
        ));
    }
    if b_stacked.len() < (num_experts * k * n) as usize {
        return Err(SparkError::InvalidArgument("b_stacked too small".into()));
    }
    let scales_need = (num_experts * n * (k / 32)) as usize;
    if b_scales.len() < scales_need {
        return Err(SparkError::InvalidArgument(format!(
            "b_scales too small: need {scales_need}"
        )));
    }

    let func = module::load_kernel(ctx, "gemm_mxfp8_grouped_mma", "gemm_mxfp8_grouped_mma")?;
    let grid_n = n.div_ceil(32);
    let grid_m = m_max.div_ceil(32);
    let cfg = LaunchConfig {
        grid_dim: (grid_n, grid_m, num_experts),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(a_permuted)
            .arg(b_stacked)
            .arg(b_scales)
            .arg(c_permuted)
            .arg(expert_offsets)
            .arg(&n)
            .arg(&k)
            .launch(cfg)?;
    }
    Ok(())
}

/// Compact a histogram into a list of active expert IDs, all device-side.
///
/// `histogram`: `[num_experts]` u32 — counts per expert from `moe_histogram`.
/// `active_experts_out`: `[num_experts]` u32 — first `num_active_out[0]` entries are
///   the IDs of experts with non-zero count (unsorted; output order is atomic).
/// `num_active_out`: `[1]` u32 — count of non-empty experts.
///
/// Single warp, single block. Used to feed sparse grouped-GEMM variants without
/// a host roundtrip on the histogram.
pub fn moe_active_experts_compact(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    histogram: &CudaSlice<u32>,
    active_experts_out: &mut CudaSlice<u32>,
    num_active_out: &mut CudaSlice<u32>,
    num_experts: u32,
) -> Result<()> {
    if num_experts == 0 {
        return Err(SparkError::InvalidArgument(
            "num_experts must be > 0".into(),
        ));
    }
    if histogram.len() < num_experts as usize {
        return Err(SparkError::InvalidArgument("histogram too small".into()));
    }
    if active_experts_out.len() < num_experts as usize {
        return Err(SparkError::InvalidArgument(
            "active_experts_out too small".into(),
        ));
    }
    if num_active_out.is_empty() {
        return Err(SparkError::InvalidArgument(
            "num_active_out must hold at least 1 element".into(),
        ));
    }
    let func = module::load_kernel(
        ctx,
        "moe_active_experts_compact",
        "moe_active_experts_compact",
    )?;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(histogram)
            .arg(active_experts_out)
            .arg(num_active_out)
            .arg(&num_experts)
            .launch(cfg)?;
    }
    Ok(())
}

/// Sparse variant of `gemm_bf16_grouped_mma`. Indirects the per-CTA expert_id
/// through `active_experts[ctaid.z]`, dropping launches for experts with no
/// tokens. Grid z-dim is `num_active`, not `num_experts`.
#[allow(clippy::too_many_arguments)]
pub fn gemm_bf16_grouped_mma_sparse(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    a_permuted: &CudaSlice<u16>,
    b_stacked: &CudaSlice<u16>,
    c_permuted: &mut CudaSlice<u16>,
    expert_offsets: &CudaSlice<u32>,
    active_experts: &CudaSlice<u32>,
    num_active: u32,
    m_max: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    if num_active == 0 || m_max == 0 || n == 0 || k == 0 {
        return Err(SparkError::InvalidArgument(
            "num_active, m_max, n, k must be > 0".into(),
        ));
    }
    if n & 31 != 0 {
        return Err(SparkError::InvalidArgument(
            "n must be multiple of 32".into(),
        ));
    }
    if k & 15 != 0 {
        return Err(SparkError::InvalidArgument(
            "k must be multiple of 16".into(),
        ));
    }
    if active_experts.len() < num_active as usize {
        return Err(SparkError::InvalidArgument(
            "active_experts too small".into(),
        ));
    }
    let func = module::load_kernel(
        ctx,
        "gemm_bf16_grouped_mma_sparse",
        "gemm_bf16_grouped_mma_sparse",
    )?;
    let grid_n = n.div_ceil(32);
    let grid_m = m_max.div_ceil(32);
    let cfg = LaunchConfig {
        grid_dim: (grid_n, grid_m, num_active),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(a_permuted)
            .arg(b_stacked)
            .arg(c_permuted)
            .arg(expert_offsets)
            .arg(active_experts)
            .arg(&n)
            .arg(&k)
            .launch(cfg)?;
    }
    Ok(())
}

/// Sparse variant of scalar `gemm_bf16_grouped`. Falls back here when the MMA
/// constraints (n%32, k%16) are not met.
#[allow(clippy::too_many_arguments)]
pub fn gemm_bf16_grouped_sparse(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    a_permuted: &CudaSlice<u16>,
    b_stacked: &CudaSlice<u16>,
    c_permuted: &mut CudaSlice<u16>,
    expert_offsets: &CudaSlice<u32>,
    active_experts: &CudaSlice<u32>,
    num_active: u32,
    m_max: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    if num_active == 0 || m_max == 0 || n == 0 || k == 0 {
        return Err(SparkError::InvalidArgument(
            "num_active, m_max, n, k must be > 0".into(),
        ));
    }
    if active_experts.len() < num_active as usize {
        return Err(SparkError::InvalidArgument(
            "active_experts too small".into(),
        ));
    }
    let func = module::load_kernel(ctx, "gemm_bf16_grouped_sparse", "gemm_bf16_grouped_sparse")?;
    let grid_n = n.div_ceil(32);
    let grid_m = m_max.div_ceil(32);
    let cfg = LaunchConfig {
        grid_dim: (grid_n, grid_m, num_active),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(a_permuted)
            .arg(b_stacked)
            .arg(c_permuted)
            .arg(expert_offsets)
            .arg(active_experts)
            .arg(&n)
            .arg(&k)
            .launch(cfg)?;
    }
    Ok(())
}

/// Auto-dispatch for sparse BF16 grouped GEMM. MMA path if constraints are met.
#[allow(clippy::too_many_arguments)]
pub fn gemm_bf16_grouped_sparse_auto(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    a: &CudaSlice<u16>,
    b: &CudaSlice<u16>,
    c: &mut CudaSlice<u16>,
    expert_offsets: &CudaSlice<u32>,
    active_experts: &CudaSlice<u32>,
    num_active: u32,
    m_max: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    if n & 31 == 0 && k & 15 == 0 {
        gemm_bf16_grouped_mma_sparse(
            ctx,
            stream,
            a,
            b,
            c,
            expert_offsets,
            active_experts,
            num_active,
            m_max,
            n,
            k,
        )
    } else {
        gemm_bf16_grouped_sparse(
            ctx,
            stream,
            a,
            b,
            c,
            expert_offsets,
            active_experts,
            num_active,
            m_max,
            n,
            k,
        )
    }
}

/// Auto-dispatch for BF16 grouped GEMM. Picks the MMA variant when dimensions
/// allow (N multiple of 32, K multiple of 16); otherwise falls back to scalar.
/// MMA delivers 3.5-15× over scalar on typical MoE shapes.
#[allow(clippy::too_many_arguments)]
pub fn gemm_bf16_grouped_auto(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    a_permuted: &CudaSlice<u16>,
    b_stacked: &CudaSlice<u16>,
    c_permuted: &mut CudaSlice<u16>,
    expert_offsets: &CudaSlice<u32>,
    num_experts: u32,
    m_max: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    if n & 31 == 0 && k & 15 == 0 {
        gemm_bf16_grouped_mma(
            ctx,
            stream,
            a_permuted,
            b_stacked,
            c_permuted,
            expert_offsets,
            num_experts,
            m_max,
            n,
            k,
        )
    } else {
        gemm_bf16_grouped(
            ctx,
            stream,
            a_permuted,
            b_stacked,
            c_permuted,
            expert_offsets,
            num_experts,
            m_max,
            n,
            k,
        )
    }
}

/// Auto-pick v1 or v2 based on N. v2's vectorized layout regresses at very large
/// N (≥8192) due to register pressure / pipeline limits; v1's distributed
/// per-thread loads do better there. Crossover empirically at N ≈ 4096.
/// Returns the kernel actually chosen for caller logging if desired.
fn pick_v1_or_v2(n: u32) -> bool {
    n < 4096
}

/// Auto FP8 block-128 grouped GEMM: v2 for typical MoE, v1 for very-wide N.
#[allow(clippy::too_many_arguments)]
pub fn gemm_fp8_block128_grouped_mma_auto(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    a: &CudaSlice<u16>,
    b: &CudaSlice<u8>,
    s: &CudaSlice<f32>,
    c: &mut CudaSlice<u16>,
    off: &CudaSlice<u32>,
    num_experts: u32,
    m_max: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    if pick_v1_or_v2(n) {
        gemm_fp8_block128_grouped_mma_v2(ctx, stream, a, b, s, c, off, num_experts, m_max, n, k)
    } else {
        gemm_fp8_block128_grouped_mma(ctx, stream, a, b, s, c, off, num_experts, m_max, n, k)
    }
}

/// Auto MXFP8 grouped GEMM.
#[allow(clippy::too_many_arguments)]
pub fn gemm_mxfp8_grouped_mma_auto(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    a: &CudaSlice<u16>,
    b: &CudaSlice<u8>,
    s: &CudaSlice<u8>,
    c: &mut CudaSlice<u16>,
    off: &CudaSlice<u32>,
    num_experts: u32,
    m_max: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    if pick_v1_or_v2(n) {
        gemm_mxfp8_grouped_mma_v2(ctx, stream, a, b, s, c, off, num_experts, m_max, n, k)
    } else {
        gemm_mxfp8_grouped_mma(ctx, stream, a, b, s, c, off, num_experts, m_max, n, k)
    }
}

/// Auto generic FP8 grouped GEMM.
#[allow(clippy::too_many_arguments)]
pub fn gemm_fp8_grouped_mma_auto(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    a: &CudaSlice<u16>,
    b: &CudaSlice<u8>,
    s: &CudaSlice<f32>,
    c: &mut CudaSlice<u16>,
    off: &CudaSlice<u32>,
    num_experts: u32,
    m_max: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    if pick_v1_or_v2(n) {
        gemm_fp8_grouped_mma_v2(ctx, stream, a, b, s, c, off, num_experts, m_max, n, k)
    } else {
        gemm_fp8_grouped_mma(ctx, stream, a, b, s, c, off, num_experts, m_max, n, k)
    }
}

/// Auto MXFP4 grouped GEMM.
#[allow(clippy::too_many_arguments)]
pub fn gemm_mxfp4_grouped_mma_auto(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    a: &CudaSlice<u16>,
    b: &CudaSlice<u8>,
    s: &CudaSlice<u8>,
    c: &mut CudaSlice<u16>,
    off: &CudaSlice<u32>,
    num_experts: u32,
    m_max: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    if pick_v1_or_v2(n) {
        gemm_mxfp4_grouped_mma_v2(ctx, stream, a, b, s, c, off, num_experts, m_max, n, k)
    } else {
        gemm_mxfp4_grouped_mma(ctx, stream, a, b, s, c, off, num_experts, m_max, n, k)
    }
}

/// Auto NVFP4 FP8-scale grouped GEMM.
#[allow(clippy::too_many_arguments)]
pub fn gemm_nvfp4_fp8scale_grouped_mma_auto(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    a: &CudaSlice<u16>,
    b: &CudaSlice<u8>,
    s: &CudaSlice<u8>,
    c: &mut CudaSlice<u16>,
    off: &CudaSlice<u32>,
    num_experts: u32,
    m_max: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    if pick_v1_or_v2(n) {
        gemm_nvfp4_fp8scale_grouped_mma_v2(ctx, stream, a, b, s, c, off, num_experts, m_max, n, k)
    } else {
        gemm_nvfp4_fp8scale_grouped_mma(ctx, stream, a, b, s, c, off, num_experts, m_max, n, k)
    }
}

/// Dispatch summary (signature-matched pairs where possible):
///
/// | Format | Scalar | MMA | Auto |
/// |--------|--------|-----|------|
/// | BF16 | `gemm_bf16_grouped` | `gemm_bf16_grouped_mma` | `gemm_bf16_grouped_auto` |
/// | FP8 block-128 | `gemm_fp8_block128_grouped` | `gemm_fp8_block128_grouped_mma` | `gemm_fp8_block128_grouped_auto` |
/// | FP8 per-expert | *(none)* | `gemm_fp8_grouped_mma` | *(use MMA directly)* |
/// | MXFP8 32-block | *(none)* | `gemm_mxfp8_grouped_mma` | *(use MMA directly)* |
///
/// Auto-dispatch for FP8 block-128 grouped GEMM (DeepSeek V3 hot path).
/// Picks MMA when N is multiple of 32 (K is always multiple of 128 in this format).
#[allow(clippy::too_many_arguments)]
pub fn gemm_fp8_block128_grouped_auto(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    a_permuted: &CudaSlice<u16>,
    b_stacked: &CudaSlice<u8>,
    b_scales: &CudaSlice<f32>,
    c_permuted: &mut CudaSlice<u16>,
    expert_offsets: &CudaSlice<u32>,
    num_experts: u32,
    m_max: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    if n & 31 == 0 {
        gemm_fp8_block128_grouped_mma(
            ctx,
            stream,
            a_permuted,
            b_stacked,
            b_scales,
            c_permuted,
            expert_offsets,
            num_experts,
            m_max,
            n,
            k,
        )
    } else {
        gemm_fp8_block128_grouped(
            ctx,
            stream,
            a_permuted,
            b_stacked,
            b_scales,
            c_permuted,
            expert_offsets,
            num_experts,
            m_max,
            n,
            k,
        )
    }
}

/// BF16 grouped GEMM using BF16 MMA (m16n8k16). Same signature/layout as
/// `gemm_bf16_grouped` but uses tensor cores for ~8-16× throughput on large
/// experts. Tile: 32×32 per CTA, 1 warp. Per-expert dispatch via z-grid.
///
/// `a_permuted`: [total_tokens, K] BF16 (expert-sorted)
/// `b_stacked`:  [num_experts, K, N] BF16 (same layout as `gemm_bf16_grouped`)
/// `c_permuted`: [total_tokens, N] BF16 output
/// `expert_offsets`: [num_experts+1] u32
/// `m_max`: upper bound on tokens-per-expert for grid sizing
///
/// Constraints: N multiple of 32, K multiple of 16.
#[allow(clippy::too_many_arguments)]
pub fn gemm_bf16_grouped_mma(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    a_permuted: &CudaSlice<u16>,
    b_stacked: &CudaSlice<u16>,
    c_permuted: &mut CudaSlice<u16>,
    expert_offsets: &CudaSlice<u32>,
    num_experts: u32,
    m_max: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    if num_experts == 0 || m_max == 0 || n == 0 || k == 0 {
        return Err(SparkError::InvalidArgument(
            "num_experts, m_max, n, k must be > 0".into(),
        ));
    }
    if n & 31 != 0 {
        return Err(SparkError::InvalidArgument(
            "n must be multiple of 32 for MMA grouped GEMM".into(),
        ));
    }
    if k & 15 != 0 {
        return Err(SparkError::InvalidArgument(
            "k must be multiple of 16 for MMA grouped GEMM".into(),
        ));
    }
    if expert_offsets.len() < num_experts as usize + 1 {
        return Err(SparkError::InvalidArgument(
            "expert_offsets too small".into(),
        ));
    }
    if b_stacked.len() < (num_experts * k * n) as usize {
        return Err(SparkError::InvalidArgument("b_stacked too small".into()));
    }

    let func = module::load_kernel(ctx, "gemm_bf16_grouped_mma", "gemm_bf16_grouped_mma")?;
    let grid_n = n.div_ceil(32);
    let grid_m = m_max.div_ceil(32);
    let cfg = LaunchConfig {
        grid_dim: (grid_n, grid_m, num_experts),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(a_permuted)
            .arg(b_stacked)
            .arg(c_permuted)
            .arg(expert_offsets)
            .arg(&n)
            .arg(&k)
            .launch(cfg)?;
    }
    Ok(())
}

/// MoE grouped W8A16 GEMM with MMA (BF16 act × FP8 e4m3 weight × per-tensor scale).
///
/// Same per-expert tile dispatch + padding pattern as `gemm_bf16_grouped_mma`;
/// the B-load section reads FP8 (1 byte per elem) and dequants to BF16 inline
/// before the SMEM stage that ldmatrix consumes. The per-tensor `b_scale` is
/// multiplied into the FP32 accumulators just before the BF16 cast — one
/// multiply per output element instead of per K-step.
///
/// `a_permuted`: [total_tokens, K] BF16 (tokens pre-permuted by expert)
/// `b_stacked_fp8`: [num_experts, K, N] u8 (FP8 e4m3)
/// `b_scale`: f32 (per-tensor)
/// `c_permuted`: [total_tokens, N] BF16
/// `expert_offsets`: [num_experts+1] u32
///
/// Constraints: N divisible by 32, K divisible by 16. Per-expert token count
/// arbitrary (handled via bounds checks).
#[allow(clippy::too_many_arguments)]
pub fn gemm_w8a16_grouped_mma(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    a_permuted: &CudaSlice<u16>,
    b_stacked_fp8: &CudaSlice<u8>,
    b_scale: f32,
    c_permuted: &mut CudaSlice<u16>,
    expert_offsets: &CudaSlice<u32>,
    num_experts: u32,
    m_max: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    if num_experts == 0 || m_max == 0 || n == 0 || k == 0 {
        return Err(SparkError::InvalidArgument(
            "num_experts, m_max, n, k must be > 0".into(),
        ));
    }
    if n & 31 != 0 {
        return Err(SparkError::InvalidArgument(
            "n must be multiple of 32 for W8A16 MMA grouped GEMM".into(),
        ));
    }
    if k & 15 != 0 {
        return Err(SparkError::InvalidArgument(
            "k must be multiple of 16 for W8A16 MMA grouped GEMM".into(),
        ));
    }
    if expert_offsets.len() < num_experts as usize + 1 {
        return Err(SparkError::InvalidArgument(
            "expert_offsets too small".into(),
        ));
    }
    if b_stacked_fp8.len() < (num_experts * k * n) as usize {
        return Err(SparkError::InvalidArgument(
            "b_stacked_fp8 too small".into(),
        ));
    }
    if !(b_scale > 0.0 && b_scale.is_finite()) {
        return Err(SparkError::InvalidArgument(format!(
            "b_scale must be > 0 and finite; got {b_scale}"
        )));
    }
    let func = module::load_kernel(ctx, "gemm_w8a16_grouped_mma", "gemm_w8a16_grouped_mma")?;
    let grid_n = n.div_ceil(32);
    let grid_m = m_max.div_ceil(32);
    let cfg = LaunchConfig {
        grid_dim: (grid_n, grid_m, num_experts),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(a_permuted)
            .arg(b_stacked_fp8)
            .arg(c_permuted)
            .arg(expert_offsets)
            .arg(&b_scale)
            .arg(&n)
            .arg(&k)
            .launch(cfg)?;
    }
    Ok(())
}

/// Grouped GEMM with DeepSeek V3 block-scaled FP8 weights (DSv3 1×128 format).
///
///   Y_e[row, n] = Σ_k A_e[row, k] * (B_e[n, k] * scale_e[n, k / 128])
///
/// `a_permuted`: [total_tokens, K] BF16 (tokens pre-permuted by expert)
/// `b_stacked`:  [num_experts, N, K] FP8 E4M3 (1 byte per element)
/// `b_scales`:   [num_experts, N, K/128] FP32 (one scale per 1×128 weight block)
/// `c_permuted`: [total_tokens, N] BF16 (pre-permuted output)
/// `expert_offsets`: [num_experts + 1] u32 token range per expert
///
/// Constraint: K must be divisible by 128 (the block size).
#[allow(clippy::too_many_arguments)]
pub fn gemm_fp8_block128_grouped(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    a_permuted: &CudaSlice<u16>,
    b_stacked: &CudaSlice<u8>,
    b_scales: &CudaSlice<f32>,
    c_permuted: &mut CudaSlice<u16>,
    expert_offsets: &CudaSlice<u32>,
    num_experts: u32,
    m_max: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    if num_experts == 0 || m_max == 0 || n == 0 || k == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    if !k.is_multiple_of(128) {
        return Err(SparkError::InvalidArgument(
            "k must be divisible by 128 (DSv3 block size)".into(),
        ));
    }
    if expert_offsets.len() < num_experts as usize + 1 {
        return Err(SparkError::InvalidArgument(
            "expert_offsets too small".into(),
        ));
    }
    if b_stacked.len() < (num_experts * k * n) as usize {
        return Err(SparkError::InvalidArgument("b_stacked too small".into()));
    }
    let scales_need = (num_experts * n * (k / 128)) as usize;
    if b_scales.len() < scales_need {
        return Err(SparkError::InvalidArgument(format!(
            "b_scales too small: need {scales_need}"
        )));
    }

    let func = module::load_kernel(
        ctx,
        "gemm_fp8_block128_grouped",
        "gemm_fp8_block128_grouped",
    )?;
    let grid_n = n.div_ceil(32);
    let grid_m = m_max.div_ceil(32);
    let cfg = LaunchConfig {
        grid_dim: (grid_n, grid_m, num_experts),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(a_permuted)
            .arg(b_stacked)
            .arg(b_scales)
            .arg(c_permuted)
            .arg(expert_offsets)
            .arg(&n)
            .arg(&k)
            .launch(cfg)?;
    }
    Ok(())
}

/// v2 of FP8 block-128 grouped GEMM with vectorized B loads. Each thread now
/// handles 1 col × 16 K-rows contiguous (1 v4.b32 = 16 FP8 per inner iter)
/// instead of 16 cols × 1 K-row scattered. 16× fewer global memory transactions
/// for the B tensor. Per-block scale hoisted into a register at K-block boundary.
///
/// Same signature/layout as `gemm_fp8_block128_grouped_mma`. Use when K is large
/// enough that scattered loads were the bottleneck (GDN-hybrid MoE scale and up).
#[allow(clippy::too_many_arguments)]
pub fn gemm_fp8_block128_grouped_mma_v2(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    a_permuted: &CudaSlice<u16>,
    b_stacked: &CudaSlice<u8>,
    b_scales: &CudaSlice<f32>,
    c_permuted: &mut CudaSlice<u16>,
    expert_offsets: &CudaSlice<u32>,
    num_experts: u32,
    m_max: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    if num_experts == 0 || m_max == 0 || n == 0 || k == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    if n & 31 != 0 {
        return Err(SparkError::InvalidArgument(
            "n must be multiple of 32".into(),
        ));
    }
    if !k.is_multiple_of(128) {
        return Err(SparkError::InvalidArgument(
            "k must be divisible by 128".into(),
        ));
    }
    if expert_offsets.len() < num_experts as usize + 1 {
        return Err(SparkError::InvalidArgument(
            "expert_offsets too small".into(),
        ));
    }
    if b_stacked.len() < (num_experts * k * n) as usize {
        return Err(SparkError::InvalidArgument("b_stacked too small".into()));
    }
    let scales_need = (num_experts * n * (k / 128)) as usize;
    if b_scales.len() < scales_need {
        return Err(SparkError::InvalidArgument(format!(
            "b_scales too small: need {scales_need}"
        )));
    }

    let func = module::load_kernel(
        ctx,
        "gemm_fp8_block128_grouped_mma_v2",
        "gemm_fp8_block128_grouped_mma_v2",
    )?;
    let grid_n = n.div_ceil(32);
    let grid_m = m_max.div_ceil(32);
    let cfg = LaunchConfig {
        grid_dim: (grid_n, grid_m, num_experts),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(a_permuted)
            .arg(b_stacked)
            .arg(b_scales)
            .arg(c_permuted)
            .arg(expert_offsets)
            .arg(&n)
            .arg(&k)
            .launch(cfg)?;
    }
    Ok(())
}

/// BF16×FP8 block-scaled grouped GEMM using BF16 MMA (DeepSeek V3 hot path).
/// Drop-in for `gemm_fp8_block128_grouped` with identical layout + semantics.
/// Decodes FP8 B → BF16 with per-column per-128-K scale pre-applied during
/// SMEM staging, then runs pure BF16 m16n8k16 MMA.
///
/// Constraints: N multiple of 32, K multiple of 128.
#[allow(clippy::too_many_arguments)]
pub fn gemm_fp8_block128_grouped_mma(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    a_permuted: &CudaSlice<u16>,
    b_stacked: &CudaSlice<u8>,
    b_scales: &CudaSlice<f32>,
    c_permuted: &mut CudaSlice<u16>,
    expert_offsets: &CudaSlice<u32>,
    num_experts: u32,
    m_max: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    if num_experts == 0 || m_max == 0 || n == 0 || k == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    if n & 31 != 0 {
        return Err(SparkError::InvalidArgument(
            "n must be multiple of 32".into(),
        ));
    }
    if !k.is_multiple_of(128) {
        return Err(SparkError::InvalidArgument(
            "k must be divisible by 128".into(),
        ));
    }
    if expert_offsets.len() < num_experts as usize + 1 {
        return Err(SparkError::InvalidArgument(
            "expert_offsets too small".into(),
        ));
    }
    if b_stacked.len() < (num_experts * k * n) as usize {
        return Err(SparkError::InvalidArgument("b_stacked too small".into()));
    }
    let scales_need = (num_experts * n * (k / 128)) as usize;
    if b_scales.len() < scales_need {
        return Err(SparkError::InvalidArgument(format!(
            "b_scales too small: need {scales_need}"
        )));
    }

    let func = module::load_kernel(
        ctx,
        "gemm_fp8_block128_grouped_mma",
        "gemm_fp8_block128_grouped_mma",
    )?;
    let grid_n = n.div_ceil(32);
    let grid_m = m_max.div_ceil(32);
    let cfg = LaunchConfig {
        grid_dim: (grid_n, grid_m, num_experts),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(a_permuted)
            .arg(b_stacked)
            .arg(b_scales)
            .arg(c_permuted)
            .arg(expert_offsets)
            .arg(&n)
            .arg(&k)
            .launch(cfg)?;
    }
    Ok(())
}

/// Batched M=1 split-K GEMV across active experts (MoE decode hot path).
///
/// For each `active_idx` in `[0, num_active)`, computes:
///   `out_f32[active_idx, :] += x_e @ B[active_eids[active_idx], :, :]`
///
/// where:
/// - `x` is `[K]` BF16 if `x_stride == 0` (broadcast across experts), else
///   `[num_active, x_stride]` (per-expert input row).
/// - `b_stacked` is `[n_routed, K, N]` BF16 (full weight stack).
/// - `active_eids` is `[num_active]` u32 (compact list of active expert IDs).
/// - `out_f32` is `[num_active, N]` f32 (caller-zeroed; atomic-accumulated).
///
/// Constraints: `n` must be a multiple of 8.
///
/// Why this kernel: at top_k=6 the per-expert serial split-K GEMV under-fills
/// the SM array (32 blocks each, drains between launches). Batching all 6
/// experts in one launch yields 192 blocks at moe_inter=1408 (≈4/SM), beating
/// both serial GEMV and the dense 64-expert grouped GEMM (which iterates idle
/// expert blocks). `x_stride == 0` covers the gate/up step (shared input);
/// `x_stride == K` covers the down step (per-expert silu_mul output).
#[allow(clippy::too_many_arguments)]
/// W8A16 batched M=1 split-K grouped GEMV: same as `gemv_bf16_grouped_split_k`
/// but the weight stack is FP8 e4m3 (1 byte/elem) with a single per-tensor
/// dequant scale shared across all experts. Halves HBM bandwidth on the heavy
/// MoE expert weight reads.
#[allow(clippy::too_many_arguments)]
pub fn gemv_w8a16_grouped_split_k(
    ctx: &std::sync::Arc<CudaContext>,
    stream: &std::sync::Arc<CudaStream>,
    x: &CudaSlice<u16>,
    b_stacked_fp8: &CudaSlice<u8>,
    b_scale: f32,
    active_eids: &CudaSlice<u32>,
    out_f32: &mut CudaSlice<f32>,
    num_active: u32,
    n: u32,
    k: u32,
    num_shards: u32,
    x_stride: u32,
) -> Result<()> {
    if num_active == 0 || n == 0 || k == 0 || num_shards == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    if n & 7 != 0 {
        return Err(SparkError::InvalidArgument(format!(
            "gemv_w8a16_grouped_split_k: N must be divisible by 8, got {n}"
        )));
    }
    if active_eids.len() < num_active as usize {
        return Err(SparkError::InvalidArgument("active_eids too small".into()));
    }
    if out_f32.len() < (num_active as usize) * (n as usize) {
        return Err(SparkError::InvalidArgument("out_f32 too small".into()));
    }
    let k_shard = k.div_ceil(num_shards);
    let smem_bytes = (k_shard as usize) * 2;
    if smem_bytes > 96 * 1024 {
        return Err(SparkError::InvalidArgument(format!(
            "gemv_w8a16_grouped_split_k: k_shard={k_shard} exceeds SMEM budget"
        )));
    }
    if x_stride == 0 {
        if x.len() < k as usize {
            return Err(SparkError::InvalidArgument(format!(
                "x too small for broadcast: {} < {k}",
                x.len()
            )));
        }
    } else {
        let need = (num_active as usize - 1) * (x_stride as usize) + (k as usize);
        if x.len() < need {
            return Err(SparkError::InvalidArgument(format!(
                "x too small for per-expert: {} < {need}",
                x.len()
            )));
        }
    }

    let func = module::load_kernel(
        ctx,
        "gemv_w8a16_grouped_split_k",
        "gemv_w8a16_grouped_split_k",
    )?;
    let threads = 128u32;
    let cols_per_block = threads * 8;
    let n_blocks = n.div_ceil(cols_per_block);
    let cfg = LaunchConfig {
        grid_dim: (n_blocks, num_shards, num_active),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: smem_bytes as u32,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(x)
            .arg(b_stacked_fp8)
            .arg(active_eids)
            .arg(out_f32)
            .arg(&b_scale)
            .arg(&n)
            .arg(&k)
            .arg(&k_shard)
            .arg(&x_stride)
            .launch(cfg)?;
    }
    Ok(())
}

/// Grouped split-K BF16 GEMV for MoE decode: for each of `num_active` selected
/// experts, computes `out = x * b_stacked[expert]` (M=1) where the expert is
/// chosen by `active_eids` and `x_stride` indexes the per-expert input row. K is
/// split across `num_shards` partial sums; N must be divisible by 8.
pub fn gemv_bf16_grouped_split_k(
    ctx: &std::sync::Arc<CudaContext>,
    stream: &std::sync::Arc<CudaStream>,
    x: &CudaSlice<u16>,
    b_stacked: &CudaSlice<u16>,
    active_eids: &CudaSlice<u32>,
    out_f32: &mut CudaSlice<f32>,
    num_active: u32,
    n: u32,
    k: u32,
    num_shards: u32,
    x_stride: u32,
) -> Result<()> {
    if num_active == 0 || n == 0 || k == 0 || num_shards == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    if n & 7 != 0 {
        return Err(SparkError::InvalidArgument(format!(
            "gemv_bf16_grouped_split_k: N must be divisible by 8, got {n}"
        )));
    }
    if active_eids.len() < num_active as usize {
        return Err(SparkError::InvalidArgument("active_eids too small".into()));
    }
    if out_f32.len() < (num_active as usize) * (n as usize) {
        return Err(SparkError::InvalidArgument("out_f32 too small".into()));
    }
    let k_shard = k.div_ceil(num_shards);
    let smem_bytes = (k_shard as usize) * 2;
    if smem_bytes > 96 * 1024 {
        return Err(SparkError::InvalidArgument(format!(
            "gemv_bf16_grouped_split_k: k_shard={k_shard} exceeds SMEM budget"
        )));
    }
    if x_stride == 0 {
        if x.len() < k as usize {
            return Err(SparkError::InvalidArgument(format!(
                "x too small for broadcast: {} < {k}",
                x.len()
            )));
        }
    } else {
        let need = (num_active as usize - 1) * (x_stride as usize) + (k as usize);
        if x.len() < need {
            return Err(SparkError::InvalidArgument(format!(
                "x too small for per-expert: {} < {need}",
                x.len()
            )));
        }
    }

    let func = module::load_kernel(
        ctx,
        "gemv_bf16_grouped_split_k",
        "gemv_bf16_grouped_split_k",
    )?;
    let threads = 128u32;
    let cols_per_block = threads * 8;
    let n_blocks = n.div_ceil(cols_per_block);
    let cfg = LaunchConfig {
        grid_dim: (n_blocks, num_shards, num_active),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: smem_bytes as u32,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(x)
            .arg(b_stacked)
            .arg(active_eids)
            .arg(out_f32)
            .arg(&n)
            .arg(&k)
            .arg(&k_shard)
            .arg(&x_stride)
            .launch(cfg)?;
    }
    Ok(())
}

/// GPU-side MoE routing for batch=1 decode with FULL-SOFTMAX-OVER-ALL-EXPERTS
/// semantics (norm_topk_prob=False, what DSV2-Lite-Chat uses).
///
/// Replaces (`moe_routing` GPU + host dtoh + host softmax + host htod) per
/// MoE layer with a single GPU kernel, eliminating one dtoh sync per layer.
/// At 26 MoE layers/token on DSV2-Lite, this removes 26 sync points.
///
/// Inputs:
/// - `logits`: `[n_routed]` BF16 (single-token router logits)
/// - `routed_scaling_factor`: scalar applied to weights when norm_topk_prob=False
///
/// Outputs:
/// - `expert_ids`: `[top_k]` u32 — descending order of softmax probability
/// - `weights`: `[top_k]` f32 — softmax(logits)[expert_ids[i]] × scale
///
/// Constraints: `n_routed` ≤ 256, `top_k` ≤ 16.
///
/// Sized for: DSV2-Lite-Chat (n_routed=64, top_k=6), 35B-A3B MoE
/// (n_routed=256, top_k=8). Block uses 256 threads regardless of n_routed
/// — extras pad with -inf for the serial argmax.
pub fn moe_route_decode_full(
    ctx: &std::sync::Arc<CudaContext>,
    stream: &std::sync::Arc<CudaStream>,
    logits: &CudaSlice<u16>,
    expert_ids: &mut CudaSlice<u32>,
    weights: &mut CudaSlice<f32>,
    n_routed: u32,
    top_k: u32,
    routed_scaling_factor: f32,
) -> Result<()> {
    if n_routed == 0 || top_k == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    if n_routed > 256 {
        return Err(SparkError::InvalidArgument(format!(
            "moe_route_decode_full: n_routed={n_routed} exceeds 256"
        )));
    }
    if top_k > 16 {
        return Err(SparkError::InvalidArgument(format!(
            "moe_route_decode_full: top_k={top_k} exceeds 16"
        )));
    }
    if logits.len() < n_routed as usize {
        return Err(SparkError::InvalidArgument("logits too small".into()));
    }
    if expert_ids.len() < top_k as usize || weights.len() < top_k as usize {
        return Err(SparkError::InvalidArgument(
            "output buffers too small".into(),
        ));
    }
    let func = module::load_kernel(ctx, "moe_route_decode_full", "moe_route_decode_full")?;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(logits)
            .arg(expert_ids)
            .arg(weights)
            .arg(&n_routed)
            .arg(&top_k)
            .arg(&routed_scaling_factor)
            .launch(cfg)?;
    }
    Ok(())
}

/// DUAL batched M=1 split-K GEMV: processes gate and up weight stacks in one
/// launch. Output is two separate F32 buffers per active expert.
///
/// For each `active_idx` in `[0, num_active)`:
///   `out_gate_f32[active_idx, :] += x @ B_gate[active_eids[active_idx], :, :]`
///   `out_up_f32[active_idx, :]   += x @ B_up[active_eids[active_idx], :, :]`
///
/// Shares the X-shard SMEM load between the two outputs (16 FMAs/K-iter
/// instead of 8) and saves 1 zero+1 cast pair per MoE layer vs two separate
/// `gemv_bf16_grouped_split_k` calls.
///
/// Constraints: `n` must be a multiple of 8. `b_gate` and `b_up` MUST have
/// matching `[n_routed, K, N]` shape.
#[allow(clippy::too_many_arguments)]
pub fn gemv_bf16_grouped_split_k_dual(
    ctx: &std::sync::Arc<CudaContext>,
    stream: &std::sync::Arc<CudaStream>,
    x: &CudaSlice<u16>,
    b_gate: &CudaSlice<u16>,
    b_up: &CudaSlice<u16>,
    active_eids: &CudaSlice<u32>,
    out_gate_f32: &mut CudaSlice<f32>,
    out_up_f32: &mut CudaSlice<f32>,
    num_active: u32,
    n: u32,
    k: u32,
    num_shards: u32,
) -> Result<()> {
    if num_active == 0 || n == 0 || k == 0 || num_shards == 0 {
        return Err(SparkError::InvalidArgument("dims must be > 0".into()));
    }
    if n & 7 != 0 {
        return Err(SparkError::InvalidArgument(format!(
            "gemv_bf16_grouped_split_k_dual: N must be divisible by 8, got {n}"
        )));
    }
    if active_eids.len() < num_active as usize {
        return Err(SparkError::InvalidArgument("active_eids too small".into()));
    }
    let need_out = (num_active as usize) * (n as usize);
    if out_gate_f32.len() < need_out || out_up_f32.len() < need_out {
        return Err(SparkError::InvalidArgument("out_*_f32 too small".into()));
    }
    if x.len() < k as usize {
        return Err(SparkError::InvalidArgument("x too small".into()));
    }
    let k_shard = k.div_ceil(num_shards);
    let smem_bytes = (k_shard as usize) * 2;
    if smem_bytes > 96 * 1024 {
        return Err(SparkError::InvalidArgument(format!(
            "gemv_bf16_grouped_split_k_dual: k_shard={k_shard} exceeds SMEM budget"
        )));
    }

    let func = module::load_kernel(
        ctx,
        "gemv_bf16_grouped_split_k_dual",
        "gemv_bf16_grouped_split_k_dual",
    )?;
    let threads = 128u32;
    let cols_per_block = threads * 8;
    let n_blocks = n.div_ceil(cols_per_block);
    let cfg = LaunchConfig {
        grid_dim: (n_blocks, num_shards, num_active),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: smem_bytes as u32,
    };
    unsafe {
        stream
            .launch_builder(&func)
            .arg(x)
            .arg(b_gate)
            .arg(b_up)
            .arg(active_eids)
            .arg(out_gate_f32)
            .arg(out_up_f32)
            .arg(&n)
            .arg(&k)
            .arg(&k_shard)
            .launch(cfg)?;
    }
    Ok(())
}
