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

"""Useful model FLOPs for DeepSeek V4 text training."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nemo_automodel.components.models.deepseek_v4.config import DeepseekV4Config


def _compressed_pairs(length: int, ratio: int, cap: int) -> int:
    """Sum min(floor(position / ratio), cap) over one-based query positions."""
    prefix = min(length, ratio * cap)
    groups, remainder = divmod(prefix, ratio)
    return ratio * groups * (groups - 1) // 2 + groups * (remainder + 1) + (length - prefix) * cap


def model_flops(config: DeepseekV4Config, *, gbs: int = 1, seq_len: int | None = None) -> float:
    """Count useful GEMM FLOPs for text-only full-parameter training without MTP.

    Projections, active routed/shared experts, hyper-connection coefficient and
    residual mixing, and causal attention count forward and backward (6 per
    MAC). The hard-top-k indexer has no gradient and counts forward only (2 per
    MAC), including its separate overlapping compressor. Ratio-4 attention
    caps compressed keys at index_topk; other compressed layers use all causal
    pooled keys. Every layer also uses causal sliding-window attention.

    Counts assume one unpadded document per sequence. Activation recomputation,
    masked-out dense work, communication, optimizer updates, pooling, softmax,
    norms, RoPE, and elementwise operations are excluded, as for other MFU
    formulas. Compressor projections run at every token, but have no backward
    pass when the sequence is too short to produce a complete pooled window.

    Args:
        config: DeepSeek V4 text configuration with MTP disabled.
        gbs: Number of sequences in the global optimizer step.
        seq_len: Tokens per sequence; defaults to max_position_embeddings.

    Returns:
        Model FLOPs across all devices for one optimizer step.

    Raises:
        ValueError: Vision, MTP, or invalid sequence/compression settings prevent
            this text-only accounting.
    """
    length = config.max_position_embeddings if seq_len is None else seq_len
    if length <= 0 or gbs <= 0:
        raise ValueError("DeepSeek V4 FLOPs require positive seq_len and gbs")
    if config.vision_n_layers or config.num_nextn_predict_layers:
        raise ValueError("DeepSeek V4 FLOPs require vision_n_layers=0 and num_nextn_predict_layers=0")
    layers = config.num_hidden_layers
    ratios = config.compress_ratios or [0] * layers
    if len(ratios) < layers or any(ratio < 0 for ratio in ratios[:layers]):
        raise ValueError("DeepSeek V4 compress_ratios must cover every layer with nonnegative ratios")
    if config.index_topk <= 0 or config.sliding_window <= 0:
        raise ValueError("DeepSeek V4 FLOPs require positive index_topk and sliding_window")

    hidden = config.hidden_size
    heads = config.num_attention_heads
    dim = config.head_dim
    hc = config.hc_mult
    # Q-LoRA, one shared KV projection, grouped O-LoRA, and its output projection.
    attention_params = (
        hidden * config.q_lora_rank
        + config.q_lora_rank * heads * dim
        + hidden * dim
        + heads * dim * config.o_lora_rank
        + config.o_groups * config.o_lora_rank * hidden
    )
    expert_params = 3 * hidden * config.moe_intermediate_size * (config.num_experts_per_tok + config.n_shared_experts)
    # Hash-routing layers still compute a full router GEMM for expert weights.
    router_params = hidden * config.n_routed_experts
    hc_params = 2 * (hc * (hc + 2)) * (hc * hidden)
    residual_macs = 2 * hc * hc * hidden
    per_token = layers * (attention_params + expert_params + router_params + hc_params + residual_macs)
    per_token += hc * hc * hidden + hidden * config.vocab_size  # HyperHead and LM head.

    window = min(length, config.sliding_window)
    local_pairs = window * (window + 1) // 2 + (length - window) * window
    attention_pairs = layers * local_pairs
    indexer_macs = 0
    compressor_flops = 0
    for ratio in ratios[:layers]:
        if ratio == 0:
            continue
        overlap = 2 if ratio == 4 else 1
        # The training cache disconnects the empty pool when length < ratio.
        compressor_macs = length * 2 * hidden * overlap * dim  # wkv and wgate.
        compressor_flops += (6 if length >= ratio else 2) * compressor_macs
        cap = config.index_topk if ratio == 4 else length // ratio
        attention_pairs += _compressed_pairs(length, ratio, cap)
        if ratio == 4:
            index_dim = config.index_head_dim
            index_heads = config.index_n_heads
            indexer_macs += length * (
                4 * hidden * index_dim + config.q_lora_rank * index_heads * index_dim + hidden * index_heads
            )
            indexer_macs += index_heads * index_dim * _compressed_pairs(length, ratio, length // ratio)

    trainable_macs = length * per_token + 2 * heads * dim * attention_pairs
    return float(gbs * (6 * trainable_macs + compressor_flops + 2 * indexer_macs))
