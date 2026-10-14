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

"""Useful text-backbone training FLOPs for MiMo V2 models."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nemo_automodel.components.models.mimo_v2_flash.config import MiMoV2FlashConfig


def model_flops(config: "MiMoV2FlashConfig", *, gbs: int = 1, seq_len: int | None = None) -> float:
    """Count full-parameter training GEMMs for full-length text sequences.

    Count forward, input-gradient and weight-gradient GEMMs (six FLOPs per
    multiply-accumulate), including dense/shared/routed MLPs, router and LM
    head. Attention uses each layer's actual QK/V dimensions and causal or
    sliding-window visible key pairs. Embedding lookups, normalization,
    softmax/sinks, elementwise work, activation recomputation, communication
    and optimizer updates are excluded. Vision/audio encoders and MTP are
    excluded. The benchmark applies its existing PEFT scaling separately.

    Args:
        config: Effective MiMo text-backbone configuration with MTP disabled.
        gbs: Number of full-length sequences in the global optimizer step.
        seq_len: Tokens per sequence; defaults to max_position_embeddings.

    Returns:
        Useful training FLOPs across all devices for one optimizer step.

    Raises:
        ValueError: The sequence, batch or layer pattern cannot be represented.
    """
    length = config.max_position_embeddings if seq_len is None else seq_len
    if length <= 0 or gbs <= 0:
        raise ValueError("MiMo FLOPs require positive seq_len and gbs")
    if getattr(config, "num_nextn_predict_layers", 0):
        raise ValueError("MiMo text FLOPs require num_nextn_predict_layers=0")
    layers = config.num_hidden_layers
    if len(config.hybrid_layer_pattern) < layers or len(config.moe_layer_freq) < layers:
        raise ValueError("MiMo layer patterns must cover num_hidden_layers")

    hidden = config.hidden_size
    parameters = hidden * config.vocab_size
    attention = 0
    for layer in range(layers):
        sliding = config.hybrid_layer_pattern[layer] == 1
        if sliding:
            heads = config.swa_num_attention_heads
            kv_heads = config.swa_num_key_value_heads
            qk_dim = config.swa_head_dim
            v_dim = config.swa_v_head_dim
            window = config.sliding_window
            if window is None or window <= 0:
                raise ValueError("MiMo sliding attention requires a positive sliding_window")
            prefix = min(length, window)
            pairs = prefix * (prefix + 1) // 2 + (length - prefix) * window
        else:
            heads = config.num_attention_heads
            kv_heads = config.num_key_value_heads
            qk_dim = config.head_dim
            v_dim = config.v_head_dim
            pairs = length * (length + 1) // 2
        parameters += hidden * (heads + kv_heads) * (qk_dim + v_dim)
        attention += 6 * heads * (qk_dim + v_dim) * pairs

        if config.n_routed_experts is not None and config.moe_layer_freq[layer]:
            active_experts = config.num_experts_per_tok + (config.n_shared_experts or 0)
            parameters += 3 * hidden * config.moe_intermediate_size * active_experts
            parameters += hidden * config.n_routed_experts
        else:
            parameters += 3 * hidden * config.intermediate_size

    return float(gbs * (6 * length * parameters + attention))
