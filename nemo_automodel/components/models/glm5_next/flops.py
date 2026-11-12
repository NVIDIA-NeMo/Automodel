# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend-independent training FLOPs for the GLM-5.3 text backbone."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nemo_automodel.components.models.glm5_next.config import Glm5NextTextConfig


def glm5_next_flops(config: Glm5NextTextConfig, gbs: int = 1, seq_len: int | None = None) -> float:
    """Count useful forward/backward FLOPs for unpadded, independent sequences.

    Trainable matrix products cost six FLOPs per MAC (forward, input gradient,
    weight gradient). KDA uses the recurrent equation's three contractions per
    head, or ``18 * heads * head_dim**2`` training FLOPs per token. Extra chunk
    algorithm work is excluded, as are activation recomputation and CP replicas.
    DSA uses expanded QK/value widths, independent of the backend's equivalent
    latent absorption. Its KPool indexer runs without gradients, so projections
    and causal pool scoring cost two FLOPs per MAC; masked future pools and
    padded sparse slots are excluded.

    Includes dense/MoE SwiGLU, selected and shared experts, router, language-model
    head, mHC coefficient projections and stream contractions, and causal short
    convolutions. Norms, activations, softmax, top-k, communication and optimizer
    work are excluded. The checkpoint's unused MTP head and vision tower are not
    counted. Use ``benchmark.flops_scope: text`` for the composite model.

    For the released 45-layer configuration with ``index_kpool=4``, this fixed
    convention gives 104.177535648 GFLOPs/token at 4096 tokens. A packed batch
    requires summing this formula over its actual document lengths; treating a
    pack of short documents as one long sequence overcounts sparse/indexer work.

    Args:
        config: Native text configuration, including effective layer patterns.
        gbs: Number of independent sequences in the global optimizer step.
        seq_len: Valid tokens in each sequence; defaults to the context limit.

    Returns:
        Global training FLOPs, before division by device count or elapsed time.

    Raises:
        ValueError: If the global batch size or sequence length is not positive.
    """
    if seq_len is None:
        seq_len = config.max_position_embeddings
    if gbs < 1 or seq_len < 1:
        raise ValueError(f"GLM-5.3 FLOPs require positive gbs and seq_len, got {gbs=} and {seq_len=}.")

    hidden = config.hidden_size
    heads = config.num_attention_heads
    kda_heads = config.linear_num_heads
    kda_dim = config.linear_head_dim
    kda_width = kda_heads * kda_dim
    kda_layers = config.layer_types.count("linear_attention")
    dsa_layers = config.layer_types.count("deepseek_sparse_attention")
    dense_layers = config.mlp_layer_types.count("dense")
    moe_layers = config.mlp_layer_types.count("sparse")

    kda_projections = 4 * hidden * kda_width + 2 * (hidden * kda_dim + kda_dim * kda_width) + hidden * kda_heads
    kda_recurrence = 3 * kda_heads * kda_dim**2
    dsa_projections = (
        hidden * config.q_lora_rank
        + config.q_lora_rank * heads * config.qk_nope_head_dim
        + hidden * config.kv_lora_rank
        + config.kv_lora_rank * heads * (config.qk_nope_head_dim + config.v_head_dim)
        + heads * config.v_head_dim * hidden
    )
    dense_mlp = 3 * hidden * config.intermediate_size
    moe_mlp = 3 * hidden * config.moe_intermediate_size * (config.num_experts_per_tok + config.n_shared_experts)
    router = hidden * config.n_routed_experts
    streams = config.hc_mult
    mhc = 2 * hidden * (streams**2 * (streams + 2) + streams + streams**2)
    trainable_macs = (
        kda_layers * (kda_projections + kda_recurrence)
        + dsa_layers * dsa_projections
        + dense_layers * dense_mlp
        + moe_layers * (moe_mlp + router)
        + config.num_hidden_layers * mhc
        + hidden * config.vocab_size
    )

    # Count only real causal convolution taps, including short documents.
    width = min(seq_len, config.linear_conv_kernel_dim)
    conv_taps = width * (width + 1) // 2 + (seq_len - width) * width
    trainable_macs = seq_len * trainable_macs + kda_layers * 3 * kda_width * conv_taps

    # For query j-1: completed pools=floor(j/p); the incomplete tail has j%p keys.
    pool_width = config.index_kpool
    complete_pools, remainder = divmod(seq_len, pool_width)
    visible_pools = pool_width * complete_pools * (complete_pools - 1) // 2 + complete_pools * (remainder + 1)
    selected_pools = min(complete_pools, config.index_topk // pool_width)
    selected_pool_edges = pool_width * selected_pools * (selected_pools - 1) // 2 + selected_pools * (
        seq_len - pool_width * selected_pools + 1
    )
    selected_keys = pool_width * selected_pool_edges
    if config.index_kpool_always_select_tail:
        selected_keys += complete_pools * pool_width * (pool_width - 1) // 2 + remainder * (remainder + 1) // 2
    trainable_macs += dsa_layers * heads * (config.qk_nope_head_dim + config.v_head_dim) * selected_keys

    # wq_b, wk, compression gate, weights projection; all execute under no_grad.
    indexer_projections = (
        config.q_lora_rank * config.index_n_heads * config.index_head_dim
        + 2 * hidden * config.index_head_dim
        + hidden * config.index_n_heads
    )
    indexer_macs = dsa_layers * (
        seq_len * indexer_projections + config.index_n_heads * (config.index_head_dim + 1) * visible_pools
    )
    return float(gbs * (6 * trainable_macs + 2 * indexer_macs))
