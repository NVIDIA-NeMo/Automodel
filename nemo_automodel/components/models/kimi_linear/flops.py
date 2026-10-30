# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Architecture FLOPs for Kimi Linear 48B A3B training."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nemo_automodel.components.models.kimi_linear.config import KimiLinear48BConfig


def model_flops(config: KimiLinear48BConfig, *, gbs: int = 1, seq_len: int | None = None) -> float:
    """Estimate training work for independent, fully supervised sequences.

    Counts linear projections, active SwiGLU experts, routers, the LM head,
    causal MLA attention (using the conventional S**2 / 2 approximation),
    KDA convolutions and the chunkwise delta-rule kernel. KDA's forward kernel
    count per head is 6*S*d**2 + 3*S*C*d + S*C**2, with C=64. Forward plus
    backward is estimated as three times forward work (six times MACs).
    Norms, activations, embedding lookup, optimizer work and activation
    recomputation are excluded. Packed document boundaries are unavailable
    through this interface: seq_len is treated as one document, so packed MLA
    work is overestimated. This is an architecture estimate, not executed FLOPs.

    Args:
        config: Complete model configuration before pipeline partitioning.
        gbs: Number of independent sequences in the global batch.
        seq_len: Tokens per sequence; defaults to max_position_embeddings.

    Returns:
        Estimated forward and backward FLOPs across the global batch.
    """
    length = config.max_position_embeddings if seq_len is None else seq_len
    if gbs < 0 or length < 0:
        raise ValueError("gbs and seq_len must be nonnegative")
    hidden = config.hidden_size
    kda_layers = sum(config.is_kda_layer(i) for i in range(config.num_hidden_layers))
    mla_layers = config.num_hidden_layers - kda_layers
    moe_layers = sum(
        config.is_moe and i >= config.first_k_dense_replace and i % config.moe_layer_freq == 0
        for i in range(config.num_hidden_layers)
    )
    dense_layers = config.num_hidden_layers - moe_layers
    # SwiGLU gate/up/down projections, counting only activated routed experts.
    weights = dense_layers * 3 * hidden * config.intermediate_size
    if moe_layers:
        weights += (
            moe_layers
            * hidden
            * (
                3 * config.moe_intermediate_size * (config.num_experts_per_token + config.num_shared_experts)
                + config.num_experts
            )
        )
    weights += hidden * config.vocab_size
    kernel_macs = 0.0
    if kda_layers:
        linear = config.linear_attn_config
        heads, dim = linear["num_heads"], linear["head_dim"]
        projection = heads * dim
        # Q/K/V/O, low-rank forget/output gates, and beta.
        weights += kda_layers * (4 * hidden * projection + 2 * (hidden + projection) * dim + hidden * heads)
        convolution = 3 * projection * linear["short_conv_kernel_size"]
        chunk = 64
        delta_rule = heads * (6 * dim**2 + 3 * chunk * dim + chunk**2) / 2
        kernel_macs += kda_layers * length * (convolution + delta_rule)
    if mla_layers:
        # KimiMLAAttention currently supports direct Q projection only.
        if config.q_lora_rank is not None:
            raise ValueError("Kimi Linear FLOPs expects q_lora_rank=None, like KimiMLAAttention")
        heads = config.num_attention_heads
        qk = config.qk_nope_head_dim + config.qk_rope_head_dim
        weights += mla_layers * (
            hidden * heads * (qk + config.v_head_dim)
            + hidden * (config.kv_lora_rank + config.qk_rope_head_dim)
            + config.kv_lora_rank * heads * (config.qk_nope_head_dim + config.v_head_dim)
        )
        kernel_macs += mla_layers * heads * length**2 * (qk + config.v_head_dim) / 2
    return float(6 * gbs * (length * weights + kernel_macs))
