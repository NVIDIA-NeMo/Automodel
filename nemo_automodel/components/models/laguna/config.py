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

from typing import Any

from transformers.configuration_utils import PretrainedConfig


def _normalize_layer_list(name: str, value: list[Any] | None, num_hidden_layers: int) -> list[Any] | None:
    """Return a per-layer list sized for ``num_hidden_layers``.

    ``from_pretrained(..., num_hidden_layers=N)`` is useful for first-layer
    parity checks against a large checkpoint whose JSON still carries full-length
    per-layer lists.  Longer lists are clipped; shorter lists remain an error.
    """
    if value is None:
        return None
    value = list(value)
    if len(value) < num_hidden_layers:
        raise ValueError(f"{name} must have at least {num_hidden_layers} entries, got {len(value)}")
    return value[:num_hidden_layers]


_PRE_INIT_OVERRIDE_FIELDS = {
    "num_hidden_layers",
    "layer_types",
    "num_attention_heads_per_layer",
    "gating_types",
    "mlp_layer_types",
    "mlp_only_layers",
    "rope_parameters",
    "swa_rope_parameters",
    "partial_rotary_factor",
}


class LagunaConfig(PretrainedConfig):
    """Configuration for Poolside Laguna causal language models."""

    model_type = "laguna"
    keys_to_ignore_at_inference = ["past_key_values"]
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any], **kwargs):
        config_dict = dict(config_dict)
        for key in list(kwargs):
            if key in _PRE_INIT_OVERRIDE_FIELDS:
                config_dict[key] = kwargs.pop(key)
        return super().from_dict(config_dict, **kwargs)

    def __init__(
        self,
        vocab_size: int = 100352,
        hidden_size: int = 2048,
        intermediate_size: int = 8192,
        num_hidden_layers: int = 48,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        qkv_bias: bool = False,
        attention_bias: bool = False,
        gating: bool | str = True,
        gating_types: list[str] | None = None,
        hidden_act: str = "silu",
        max_position_embeddings: int = 4096,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        rope_parameters: dict[str, Any] | None = None,
        partial_rotary_factor: float | None = None,
        attention_dropout: float = 0.0,
        sliding_window: int | None = None,
        layer_types: list[str] | None = None,
        num_attention_heads_per_layer: list[int] | None = None,
        swa_attention_sink_enabled: bool = False,
        swa_rope_parameters: dict[str, Any] | None = None,
        num_experts: int = 256,
        num_experts_per_tok: int = 16,
        moe_intermediate_size: int = 1024,
        shared_expert_intermediate_size: int = 1024,
        norm_topk_prob: bool = True,
        decoder_sparse_step: int = 1,
        mlp_only_layers: list[int] | None = None,
        mlp_layer_types: list[str] | None = None,
        router_aux_loss_coef: float = 0.001,
        moe_routed_scaling_factor: float = 1.0,
        moe_apply_router_weight_on_input: bool = False,
        moe_router_logit_softcapping: float = 0.0,
        output_router_logits: bool = False,
        torch_dtype: str = "bfloat16",
        **kwargs,
    ):
        if rope_parameters is None:
            rope_parameters = {"rope_type": "default", "rope_theta": 500000.0}
        if swa_rope_parameters is None and isinstance(rope_parameters, dict):
            swa_rope_parameters = rope_parameters.get("sliding_attention")

        if partial_rotary_factor is not None:
            if isinstance(rope_parameters, dict) and "partial_rotary_factor" not in rope_parameters:
                rope_parameters = {**rope_parameters, "partial_rotary_factor": partial_rotary_factor}
            if isinstance(swa_rope_parameters, dict) and "partial_rotary_factor" not in swa_rope_parameters:
                swa_rope_parameters = {**swa_rope_parameters, "partial_rotary_factor": partial_rotary_factor}

        if layer_types is None:
            layer_types = ["full_attention"] * num_hidden_layers
        layer_types = _normalize_layer_list("layer_types", layer_types, num_hidden_layers)

        num_attention_heads_per_layer = _normalize_layer_list(
            "num_attention_heads_per_layer",
            num_attention_heads_per_layer,
            num_hidden_layers,
        )
        gating_types = _normalize_layer_list("gating_types", gating_types, num_hidden_layers)
        mlp_layer_types = _normalize_layer_list("mlp_layer_types", mlp_layer_types, num_hidden_layers)

        if mlp_only_layers is None:
            if mlp_layer_types is None:
                mlp_only_layers = [0]
            else:
                mlp_only_layers = [idx for idx, layer_type in enumerate(mlp_layer_types) if layer_type == "dense"]
        else:
            mlp_only_layers = [idx for idx in mlp_only_layers if idx < num_hidden_layers]

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.qkv_bias = qkv_bias
        self.attention_bias = attention_bias
        self.gating = gating
        self.gating_types = gating_types
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_parameters = rope_parameters
        self.partial_rotary_factor = partial_rotary_factor
        self.attention_dropout = attention_dropout
        self.sliding_window = sliding_window
        self.layer_types = layer_types
        self.num_attention_heads_per_layer = num_attention_heads_per_layer
        self.swa_attention_sink_enabled = swa_attention_sink_enabled
        self.swa_rope_parameters = swa_rope_parameters
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.norm_topk_prob = norm_topk_prob
        self.decoder_sparse_step = decoder_sparse_step
        self.mlp_only_layers = mlp_only_layers
        self.mlp_layer_types = mlp_layer_types
        self.router_aux_loss_coef = router_aux_loss_coef
        self.moe_routed_scaling_factor = moe_routed_scaling_factor
        self.moe_apply_router_weight_on_input = moe_apply_router_weight_on_input
        self.moe_router_logit_softcapping = moe_router_logit_softcapping
        self.output_router_logits = output_router_logits
        self.torch_dtype = torch_dtype

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            use_cache=use_cache,
            **kwargs,
        )


__all__ = ["LagunaConfig"]
