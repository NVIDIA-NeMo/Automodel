# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Useful model FLOPs for GLM MoE DSA (GLM-5.x) training."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig


def _causal_capped_pairs(length: int, cap: int) -> int:
    """Sum over one-based query positions of ``min(position, cap)``."""
    capped = min(length, cap)
    return capped * (capped + 1) // 2 + max(length - cap, 0) * cap


def _layer_is_moe(config: GlmMoeDsaConfig, layer_idx: int) -> bool:
    """Mirror ``Block.__init__``: ``mlp_layer_types`` wins, else the first ``first_k_dense_replace`` layers are dense."""
    mlp_layer_types = getattr(config, "mlp_layer_types", None)
    if mlp_layer_types is not None:
        return mlp_layer_types[layer_idx] == "sparse"
    return layer_idx >= getattr(config, "first_k_dense_replace", 0)


def _layer_has_indexer(config: GlmMoeDsaConfig, layer_idx: int) -> bool:
    """Mirror ``Block.__init__``: only IndexShare ``"shared"`` layers own no indexer."""
    indexer_types = getattr(config, "indexer_types", None)
    return indexer_types is None or indexer_types[layer_idx] != "shared"


def model_flops(config: GlmMoeDsaConfig, *, gbs: int = 1, seq_len: int | None = None) -> float:
    """Count useful GEMM FLOPs for one optimizer step of full-parameter GLM MoE DSA training.

    The MLA projections, the dense MLPs, the active routed and shared experts, the router
    and the LM head count forward and backward (6 FLOPs per MAC). Sparse attention counts
    the unabsorbed MLA form over the keys a causal query can select: one-based position
    ``i`` attends to ``min(i, index_topk)`` keys. The indexer exists only on
    ``indexer_types == "full"`` layers and receives no gradient (its top-k selection is
    discrete and its scores feed nothing else), so its projections and causal score GEMM
    count forward only (2 FLOPs per MAC); IndexShare ``"shared"`` layers reuse the previous
    selection at no indexer cost. The AutoModel GLM model has no MTP module, so
    ``num_nextn_predict_layers`` adds nothing.

    Counts assume one unpadded document per sequence. Activation recomputation, the extra
    latent-width work of the absorbed sparse kernels, norms, RoPE, softmax, the top-k
    selection, communication and optimizer updates are excluded, as in the other MFU
    formulas.

    Args:
        config: GLM MoE DSA configuration.
        gbs: Sequences per global optimizer step.
        seq_len: Tokens per sequence; defaults to ``max_position_embeddings``.

    Returns:
        Model FLOPs across all devices for one optimizer step.

    Raises:
        ValueError: ``gbs``, the sequence length or ``index_topk`` is not positive.
    """
    length = config.max_position_embeddings if seq_len is None else seq_len
    if length <= 0 or gbs <= 0:
        raise ValueError("GLM MoE DSA FLOPs require positive seq_len and gbs")
    topk = int(config.index_topk)
    if topk <= 0:
        raise ValueError("GLM MoE DSA FLOPs require a positive index_topk")

    hidden = config.hidden_size
    heads = config.num_attention_heads
    qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
    # q_a, q_b, kv_a (latent plus the shared rope key), kv_b and o projections.
    attention_params = (
        hidden * config.q_lora_rank
        + config.q_lora_rank * heads * qk_head_dim
        + hidden * (config.kv_lora_rank + config.qk_rope_head_dim)
        + config.kv_lora_rank * heads * (config.qk_nope_head_dim + config.v_head_dim)
        + heads * config.v_head_dim * hidden
    )
    dense_ffn_params = 3 * hidden * config.intermediate_size
    moe_ffn_params = 3 * hidden * config.moe_intermediate_size * (config.num_experts_per_tok + config.n_shared_experts)
    router_params = hidden * config.n_routed_experts
    # Indexer wq_b, wk and weights_proj.
    indexer_params = (
        config.q_lora_rank * config.index_n_heads * config.index_head_dim
        + hidden * config.index_head_dim
        + hidden * config.index_n_heads
    )

    per_token = hidden * config.vocab_size  # LM head
    indexer_layers = 0
    for layer_idx in range(config.num_hidden_layers):
        per_token += attention_params
        per_token += moe_ffn_params + router_params if _layer_is_moe(config, layer_idx) else dense_ffn_params
        indexer_layers += int(_layer_has_indexer(config, layer_idx))

    attention_macs = (
        config.num_hidden_layers * _causal_capped_pairs(length, topk) * heads * (qk_head_dim + config.v_head_dim)
    )
    trainable_macs = length * per_token + attention_macs
    indexer_macs = indexer_layers * (
        length * indexer_params + config.index_n_heads * config.index_head_dim * length * (length + 1) // 2
    )
    return float(gbs * (6 * trainable_macs + 2 * indexer_macs))
