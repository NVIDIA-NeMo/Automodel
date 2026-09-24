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

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nemo_automodel.components.models.mimo_v2_flash.config import MiMoV2FlashConfig


def mimo_v2_flops(config: MiMoV2FlashConfig, gbs: int = 1, seq_len: int | None = None) -> float:
    """Count MiMo-V2-Flash / V2.6 text-backbone training model FLOPs.

    Counts six FLOPs per MAC (forward, input gradient and weight gradient):
    Q/K/V/O projections, causal QK/PV attention, dense SwiGLU, top-k routed
    and shared SwiGLU experts, router and vocabulary projection. Full and SWA
    layers have independent Q/KV head counts and QK/V dimensions. A sliding
    window includes the current token, matching TE's (window - 1, 0) mask.

    This is the nominal model-work convention: the router is included even
    with benchmark-only fake routing, and the vocabulary projection is counted
    at every input position even if fused CE elides ignored labels. Elementwise
    operations (norms, RoPE, softmax, sinks, activations, residuals and routing
    weights), embedding lookups, communication, optimizer work, recomputation
    and backend padding are excluded. Vision/audio encoders and MTP are not
    part of this text-backbone formula.

    Args:
        config: MiMo text configuration, including a MiMoV2Config subclass.
        gbs: Global number of sequences per training step, before sharding.
        seq_len: Tokens per unpadded sequence, defaulting to max_position_embeddings.
            Packed inputs with shorter constituent sequences need their actual
            sequence lengths; treating the whole pack as one sequence overcounts attention.

    Returns:
        Nominal forward-plus-backward FLOPs across all devices for one step.
    """
    if seq_len is None:
        seq_len = config.max_position_embeddings
    if gbs <= 0 or seq_len <= 0:
        raise ValueError(f"gbs and seq_len must be positive, got gbs={gbs}, seq_len={seq_len}")

    hidden = config.hidden_size
    tokens = gbs * seq_len
    linear_macs_per_token = hidden * config.vocab_size
    attention_macs_per_sequence = 0

    for layer_idx in range(config.num_hidden_layers):
        is_swa = config.hybrid_layer_pattern[layer_idx] == 1
        if is_swa:
            heads = config.swa_num_attention_heads
            kv_heads = config.swa_num_key_value_heads
            qk_dim = config.swa_head_dim
            v_dim = config.swa_v_head_dim
            if config.sliding_window is None or config.sliding_window <= 0:
                raise ValueError("MiMo sliding_window must be positive for sliding-attention layers")
            window = min(seq_len, config.sliding_window)
        else:
            heads = config.num_attention_heads
            kv_heads = config.num_key_value_heads
            qk_dim = config.head_dim
            v_dim = config.v_head_dim
            window = seq_len

        # Q, K, V and O, including GQA's reduced K/V projections. Fusing QKV
        # changes the layout, not its mathematical work.
        linear_macs_per_token += hidden * (heads * qk_dim + kv_heads * qk_dim + kv_heads * v_dim + heads * v_dim)
        # Sum min(query_position + 1, window), including the causal ramp.
        pairs = window * (window + 1) // 2 + (seq_len - window) * window
        attention_macs_per_sequence += pairs * heads * (qk_dim + v_dim)

        is_moe = config.n_routed_experts is not None and bool(config.moe_layer_freq[layer_idx])
        if is_moe:
            active_experts = config.num_experts_per_tok + (config.n_shared_experts or 0)
            linear_macs_per_token += 3 * hidden * config.moe_intermediate_size * active_experts
            linear_macs_per_token += hidden * config.n_routed_experts
        else:
            linear_macs_per_token += 3 * hidden * config.intermediate_size

    return float(6 * (tokens * linear_macs_per_token + gbs * attention_macs_per_sequence))
