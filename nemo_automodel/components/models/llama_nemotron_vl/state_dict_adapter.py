# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0.

import re
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from nemo_automodel.components.models.common.bidirectional import EncoderStateDictAdapter

if TYPE_CHECKING:
    from nemo_automodel.components.models.llama_nemotron_vl.model import LlamaNemotronVLModel


def _get_config_value(config, name: str, default=None):
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _pop_to(state_dict: dict, source: str, target: str) -> None:
    value = state_dict.pop(source, None)
    if value is not None:
        state_dict[target] = value


def _cat_from(state_dict: dict, sources: tuple[str, ...], target: str) -> None:
    values = [state_dict.pop(source, None) for source in sources]
    if all(value is not None for value in values):
        state_dict[target] = torch.cat(values, dim=0)
        return
    for source, value in zip(sources, values, strict=True):
        if value is not None:
            state_dict[source] = value


def _filter_converted_pairs(pairs: list[tuple[str, Any]], exclude_key_regex: str | None) -> list[tuple[str, Any]]:
    if exclude_key_regex is None:
        return pairs
    return [(key, value) for key, value in pairs if not re.match(exclude_key_regex, key)]


def _get_siglip_state_prefix(model: nn.Module) -> str:
    """Return the state-dict prefix for the active SigLIP layout."""
    vision_model = getattr(model, "vision_model", None)
    if getattr(getattr(vision_model, "vision_model", None), "encoder", None) is not None:
        return "vision_model.vision_model"
    if getattr(vision_model, "encoder", None) is not None:
        return "vision_model"
    return "vision_model.vision_model"


class LlamaNemotronVLEncoderStateDictAdapter(EncoderStateDictAdapter):
    """HF-compatible state-dict adapter for optimized LlamaNemotron VL retrieval checkpoints."""

    def __init__(self, model: "LlamaNemotronVLModel"):
        super().__init__()
        self.llm_config = model.config.llm_config
        self.vision_config = model.config.vision_config
        self.use_te_fused_llama_mlp = getattr(model, "_nemo_use_te_fused_llama_mlp", False)
        self.use_te_fused_llama_qkv = getattr(model, "_nemo_use_te_fused_llama_qkv", False)
        self.use_te_fused_siglip_layer = getattr(model, "_nemo_use_te_fused_siglip_layer", False)
        self.vision_state_prefix = _get_siglip_state_prefix(model)

    def to_hf(self, state_dict, **kwargs):
        hf_state_dict = {}
        for fqn, tensor in state_dict.items():
            for key, value in self.convert_single_tensor_to_hf(fqn, tensor, **kwargs):
                hf_state_dict[key] = value
        return hf_state_dict

    def from_hf(self, hf_state_dict, device_mesh=None, **kwargs):
        optimized_state_dict = dict(hf_state_dict)
        self._hf_to_optimized(optimized_state_dict)
        return super().from_hf(optimized_state_dict, device_mesh=device_mesh, **kwargs)

    def convert_single_tensor_to_hf(self, fqn: str, tensor: Any, **kwargs) -> list[tuple[str, Any]]:
        key = self._strip_model_prefix(fqn)
        if key is None:
            return []

        converted = self._optimized_single_tensor_to_hf(key, tensor)
        return _filter_converted_pairs(converted, kwargs.get("exclude_key_regex"))

    def _llama_sizes(self) -> tuple[int, int, int, int]:
        hidden_size = _get_config_value(self.llm_config, "hidden_size")
        num_attention_heads = _get_config_value(self.llm_config, "num_attention_heads")
        num_key_value_heads = _get_config_value(self.llm_config, "num_key_value_heads", num_attention_heads)
        head_dim = _get_config_value(self.llm_config, "head_dim", hidden_size // num_attention_heads)
        intermediate_size = _get_config_value(self.llm_config, "intermediate_size")
        return (
            num_attention_heads * head_dim,
            num_key_value_heads * head_dim,
            num_key_value_heads * head_dim,
            intermediate_size,
        )

    def _optimized_single_tensor_to_hf(self, key: str, tensor: Any) -> list[tuple[str, Any]]:
        q_size, k_size, v_size, intermediate_size = self._llama_sizes()
        layer_match = re.match(
            r"^language_model\.layers\.(\d+)\.(self_attn\.pre_attention_qkv|mlp)\.fused\.(.+)$",
            key,
        )
        if layer_match:
            layer_idx, fused_module, suffix = layer_match.groups()
            prefix = f"language_model.layers.{layer_idx}."
            if self.use_te_fused_llama_qkv and fused_module == "self_attn.pre_attention_qkv":
                if suffix == "layer_norm_weight":
                    return [(f"{prefix}input_layernorm.weight", tensor)]
                if suffix == "weight":
                    q, k, v = tensor.split((q_size, k_size, v_size), dim=0)
                    return [
                        (f"{prefix}self_attn.q_proj.weight", q),
                        (f"{prefix}self_attn.k_proj.weight", k),
                        (f"{prefix}self_attn.v_proj.weight", v),
                    ]
                if suffix == "bias":
                    q, k, v = tensor.split((q_size, k_size, v_size), dim=0)
                    return [
                        (f"{prefix}self_attn.q_proj.bias", q),
                        (f"{prefix}self_attn.k_proj.bias", k),
                        (f"{prefix}self_attn.v_proj.bias", v),
                    ]
            if self.use_te_fused_llama_mlp and fused_module == "mlp":
                if suffix == "layer_norm_weight":
                    return [(f"{prefix}post_attention_layernorm.weight", tensor)]
                if suffix == "fc1_weight":
                    gate, up = tensor.split((intermediate_size, intermediate_size), dim=0)
                    return [(f"{prefix}mlp.gate_proj.weight", gate), (f"{prefix}mlp.up_proj.weight", up)]
                if suffix == "fc1_bias":
                    gate, up = tensor.split((intermediate_size, intermediate_size), dim=0)
                    return [(f"{prefix}mlp.gate_proj.bias", gate), (f"{prefix}mlp.up_proj.bias", up)]
                if suffix == "fc2_weight":
                    return [(f"{prefix}mlp.down_proj.weight", tensor)]
                if suffix == "fc2_bias":
                    return [(f"{prefix}mlp.down_proj.bias", tensor)]

        vision_match = re.match(r"^(vision_model(?:\.vision_model)?)\.encoder\.layers\.(\d+)\.mlp\.fused\.(.+)$", key)
        if self.use_te_fused_siglip_layer and vision_match:
            vision_prefix, layer_idx, suffix = vision_match.groups()
            prefix = f"{vision_prefix}.encoder.layers.{layer_idx}."
            hidden_size = _get_config_value(self.vision_config, "hidden_size")
            if suffix == "self_attention.layernorm_qkv.layer_norm_weight":
                return [(f"{prefix}layer_norm1.weight", tensor)]
            if suffix == "self_attention.layernorm_qkv.layer_norm_bias":
                return [(f"{prefix}layer_norm1.bias", tensor)]
            if suffix == "self_attention.layernorm_qkv.weight":
                q, k, v = tensor.split((hidden_size, hidden_size, hidden_size), dim=0)
                return [
                    (f"{prefix}self_attn.q_proj.weight", q),
                    (f"{prefix}self_attn.k_proj.weight", k),
                    (f"{prefix}self_attn.v_proj.weight", v),
                ]
            if suffix == "self_attention.layernorm_qkv.bias":
                q, k, v = tensor.split((hidden_size, hidden_size, hidden_size), dim=0)
                return [
                    (f"{prefix}self_attn.q_proj.bias", q),
                    (f"{prefix}self_attn.k_proj.bias", k),
                    (f"{prefix}self_attn.v_proj.bias", v),
                ]
            if suffix == "self_attention.proj.weight":
                return [(f"{prefix}self_attn.out_proj.weight", tensor)]
            if suffix == "self_attention.proj.bias":
                return [(f"{prefix}self_attn.out_proj.bias", tensor)]
            if suffix == "layernorm_mlp.layer_norm_weight":
                return [(f"{prefix}layer_norm2.weight", tensor)]
            if suffix == "layernorm_mlp.layer_norm_bias":
                return [(f"{prefix}layer_norm2.bias", tensor)]
            if suffix == "layernorm_mlp.fc1_weight":
                return [(f"{prefix}mlp.fc1.weight", tensor)]
            if suffix == "layernorm_mlp.fc1_bias":
                return [(f"{prefix}mlp.fc1.bias", tensor)]
            if suffix == "layernorm_mlp.fc2_weight":
                return [(f"{prefix}mlp.fc2.weight", tensor)]
            if suffix == "layernorm_mlp.fc2_bias":
                return [(f"{prefix}mlp.fc2.bias", tensor)]

        return [(key, tensor)]

    def _hf_to_optimized(self, state_dict: dict) -> None:
        _, _, _, intermediate_size = self._llama_sizes()
        for layer_idx in range(_get_config_value(self.llm_config, "num_hidden_layers", 0)):
            prefix = f"language_model.layers.{layer_idx}."
            if self.use_te_fused_llama_qkv:
                fused_prefix = f"{prefix}self_attn.pre_attention_qkv.fused."
                _pop_to(state_dict, f"{prefix}input_layernorm.weight", f"{fused_prefix}layer_norm_weight")
                _cat_from(
                    state_dict,
                    (
                        f"{prefix}self_attn.q_proj.weight",
                        f"{prefix}self_attn.k_proj.weight",
                        f"{prefix}self_attn.v_proj.weight",
                    ),
                    f"{fused_prefix}weight",
                )
                _cat_from(
                    state_dict,
                    (
                        f"{prefix}self_attn.q_proj.bias",
                        f"{prefix}self_attn.k_proj.bias",
                        f"{prefix}self_attn.v_proj.bias",
                    ),
                    f"{fused_prefix}bias",
                )
            if self.use_te_fused_llama_mlp:
                fused_prefix = f"{prefix}mlp.fused."
                _pop_to(state_dict, f"{prefix}post_attention_layernorm.weight", f"{fused_prefix}layer_norm_weight")
                _cat_from(
                    state_dict,
                    (f"{prefix}mlp.gate_proj.weight", f"{prefix}mlp.up_proj.weight"),
                    f"{fused_prefix}fc1_weight",
                )
                _cat_from(
                    state_dict,
                    (f"{prefix}mlp.gate_proj.bias", f"{prefix}mlp.up_proj.bias"),
                    f"{fused_prefix}fc1_bias",
                )
                if f"{fused_prefix}fc1_weight" in state_dict:
                    assert state_dict[f"{fused_prefix}fc1_weight"].shape[0] == 2 * intermediate_size
                _pop_to(state_dict, f"{prefix}mlp.down_proj.weight", f"{fused_prefix}fc2_weight")
                _pop_to(state_dict, f"{prefix}mlp.down_proj.bias", f"{fused_prefix}fc2_bias")

        if self.use_te_fused_siglip_layer:
            self._siglip_hf_to_optimized(state_dict)

    def _siglip_hf_to_optimized(self, state_dict: dict) -> None:
        for layer_idx in range(_get_config_value(self.vision_config, "num_hidden_layers", 0)):
            prefix = f"{self.vision_state_prefix}.encoder.layers.{layer_idx}."
            fused_prefix = f"{prefix}mlp.fused."
            _pop_to(
                state_dict,
                f"{prefix}layer_norm1.weight",
                f"{fused_prefix}self_attention.layernorm_qkv.layer_norm_weight",
            )
            _pop_to(
                state_dict,
                f"{prefix}layer_norm1.bias",
                f"{fused_prefix}self_attention.layernorm_qkv.layer_norm_bias",
            )
            _cat_from(
                state_dict,
                (
                    f"{prefix}self_attn.q_proj.weight",
                    f"{prefix}self_attn.k_proj.weight",
                    f"{prefix}self_attn.v_proj.weight",
                ),
                f"{fused_prefix}self_attention.layernorm_qkv.weight",
            )
            _cat_from(
                state_dict,
                (
                    f"{prefix}self_attn.q_proj.bias",
                    f"{prefix}self_attn.k_proj.bias",
                    f"{prefix}self_attn.v_proj.bias",
                ),
                f"{fused_prefix}self_attention.layernorm_qkv.bias",
            )
            _pop_to(state_dict, f"{prefix}self_attn.out_proj.weight", f"{fused_prefix}self_attention.proj.weight")
            _pop_to(state_dict, f"{prefix}self_attn.out_proj.bias", f"{fused_prefix}self_attention.proj.bias")
            _pop_to(state_dict, f"{prefix}layer_norm2.weight", f"{fused_prefix}layernorm_mlp.layer_norm_weight")
            _pop_to(state_dict, f"{prefix}layer_norm2.bias", f"{fused_prefix}layernorm_mlp.layer_norm_bias")
            _pop_to(state_dict, f"{prefix}mlp.fc1.weight", f"{fused_prefix}layernorm_mlp.fc1_weight")
            _pop_to(state_dict, f"{prefix}mlp.fc1.bias", f"{fused_prefix}layernorm_mlp.fc1_bias")
            _pop_to(state_dict, f"{prefix}mlp.fc2.weight", f"{fused_prefix}layernorm_mlp.fc2_weight")
            _pop_to(state_dict, f"{prefix}mlp.fc2.bias", f"{fused_prefix}layernorm_mlp.fc2_bias")


__all__ = [
    "LlamaNemotronVLEncoderStateDictAdapter",
]
