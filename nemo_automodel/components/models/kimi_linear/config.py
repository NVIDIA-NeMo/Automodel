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

"""Configuration for Moonshot Kimi Linear checkpoints."""

from __future__ import annotations

from typing import Any

from transformers.configuration_utils import PretrainedConfig


class KimiLinearConfig(PretrainedConfig):
    """HF-compatible configuration for Kimi Linear causal LM checkpoints."""

    model_type = "kimi_linear"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        model_type: str = "kimi_linear",
        vocab_size: int = 163840,
        hidden_size: int = 4096,
        head_dim: int | None = None,
        intermediate_size: int = 11008,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int | None = None,
        hidden_act: str = "silu",
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        rope_theta: float = 10000.0,
        rope_scaling: dict[str, Any] | None = None,
        tie_word_embeddings: bool = False,
        moe_intermediate_size: int | None = None,
        moe_renormalize: bool = True,
        moe_router_activation_func: str = "sigmoid",
        num_experts: int | None = None,
        num_experts_per_token: int | None = None,
        num_shared_experts: int = 0,
        routed_scaling_factor: float = 1.0,
        first_k_dense_replace: int = 0,
        moe_layer_freq: int = 1,
        use_grouped_topk: bool = True,
        num_expert_group: int = 1,
        topk_group: int = 1,
        q_lora_rank: int | None = None,
        kv_lora_rank: int | None = None,
        qk_nope_head_dim: int | None = None,
        qk_rope_head_dim: int | None = None,
        v_head_dim: int | None = None,
        mla_use_nope: bool | None = False,
        num_nextn_predict_layers: int = 0,
        linear_attn_config: dict[str, Any] | None = None,
        kda_mode: str = "chunk",
        kda_unpad_inputs: bool = True,
        kda_use_fused_gate: bool = True,
        kda_use_qk_l2norm_in_kernel: bool = True,
        **kwargs: Any,
    ) -> None:
        self.model_type = model_type
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_attention_heads
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_attention_heads if num_key_value_heads is None else num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.mla_use_nope = mla_use_nope

        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.moe_renormalize = moe_renormalize
        self.num_shared_experts = num_shared_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.moe_router_activation_func = moe_router_activation_func
        if self.moe_router_activation_func not in ("softmax", "sigmoid"):
            raise ValueError("moe_router_activation_func must be 'softmax' or 'sigmoid'.")
        self.moe_intermediate_size = moe_intermediate_size
        self.first_k_dense_replace = first_k_dense_replace
        self.moe_layer_freq = moe_layer_freq
        self.use_grouped_topk = use_grouped_topk
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.num_nextn_predict_layers = num_nextn_predict_layers
        if kda_mode not in ("chunk", "fused_recurrent"):
            raise ValueError("kda_mode must be 'chunk' or 'fused_recurrent'.")
        self.kda_mode = kda_mode
        self.kda_unpad_inputs = kda_unpad_inputs
        self.kda_use_fused_gate = kda_use_fused_gate
        self.kda_use_qk_l2norm_in_kernel = kda_use_qk_l2norm_in_kernel

        if linear_attn_config is not None:
            if linear_attn_config.get("kda_layers") is None or linear_attn_config.get("full_attn_layers") is None:
                raise ValueError("linear_attn_config must define kda_layers and full_attn_layers.")
        self.linear_attn_config = linear_attn_config

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def is_mla(self) -> bool:
        """Return whether full-attention layers use Kimi MLA projection fields."""
        return (
            self.q_lora_rank is not None
            or self.kv_lora_rank is not None
            or self.qk_nope_head_dim is not None
            or self.qk_rope_head_dim is not None
            or self.v_head_dim is not None
            or self.mla_use_nope is True
        )

    @property
    def is_moe(self) -> bool:
        """Return whether the checkpoint config declares routed experts."""
        return self.num_experts is not None

    @property
    def is_linear_attn(self) -> bool:
        """Return whether any decoder layer uses Kimi Delta Attention."""
        return not (
            self.linear_attn_config is None
            or (
                isinstance(self.linear_attn_config, dict)
                and self.linear_attn_config.get("kda_layers") is not None
                and len(self.linear_attn_config["kda_layers"]) == 0
            )
        )

    def is_kda_layer(self, layer_idx: int) -> bool:
        """Return whether a zero-based layer index is configured as KDA."""
        return self.linear_attn_config is not None and (layer_idx + 1) in self.linear_attn_config["kda_layers"]
