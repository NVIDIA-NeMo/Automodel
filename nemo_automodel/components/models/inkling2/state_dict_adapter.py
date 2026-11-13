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

"""State-dict adapter between native Inkling 2 modules and the released checkpoint layout.

The released checkpoint (``thinkingmachines/Inkling-2-Small``) uses ``model.llm.*`` and
``model.visual.*`` names. The key correspondence follows the Transformers ``inkling2``
checkpoint conversion (huggingface/transformers commit 109a7d19bf667168837f32929e00b791e195d14d,
``src/transformers/conversion_mapping.py``, Apache-2.0). Native names equal the Transformers
names except for the routed experts, which use AutoModel's grouped-expert parameters:

* ``mlp.experts.gate_and_up_projs`` ``[experts, hidden, 2 * intermediate]`` is the transpose of
  ``mlp.experts.w13_weight`` ``[experts, 2 * intermediate, hidden]``. The released gate/up row
  interleaving ``[g0, u0, g1, u1, ...]`` is kept (Transformers de-interleaves it instead).
* ``mlp.experts.down_projs`` ``[experts, intermediate, hidden]`` is the transpose of
  ``mlp.experts.w2_weight`` ``[experts, hidden, intermediate]``.

Quantized experts (NVFP4 layers with ``*_amax`` companions, MXFP8 layers stored as
``float8_e4m3fn``) are dequantized on load. ``to_hf`` writes unquantized experts in the
released ``w13_weight`` / ``w2_weight`` layout.
"""

from __future__ import annotations

import re
from typing import Any

import torch
from torch.distributed.device_mesh import DeviceMesh

from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter
from nemo_automodel.components.moe import state_dict_utils

from .configuration import Inkling2TextConfig
from .quantization import dequantize_mxfp8_experts, dequantize_nvfp4_experts

_RAW_TEXT = "model.llm."
_NATIVE_TEXT = "model.language_model."
_RAW_VISION = "model.visual."
_NATIVE_VISION = "model.vision_tower."

# Whole-key renames outside the decoder layers (raw -> native).
_TEXT_GLOBAL_KEYS = {
    "model.llm.embed.weight": "model.language_model.embed_tokens.weight",
    "model.llm.embed_norm.weight": "model.language_model.embed_tokens.embed_norm.weight",
    "model.llm.norm.weight": "model.language_model.norm.weight",
    "model.llm.unembed.weight": "lm_head.weight",
    "model.llm.attn_res.output_query": "model.language_model.final_input.query",
}

# Decoder-layer suffix renames (raw -> native), applied after ``layers.<i>.``.
_LAYER_SUFFIXES = {
    "attn.wq_du.weight": "self_attn.q_proj.weight",
    "attn.wk_dv.weight": "self_attn.k_proj.weight",
    "attn.wv_dv.weight": "self_attn.v_proj.weight",
    "attn.wr_du.weight": "self_attn.r_proj.weight",
    "attn.wo_ud.weight": "self_attn.o_proj.weight",
    "attn.gate_attn.weight": "self_attn.gate_proj.weight",
    "attn.attn_sink_proj.weight": "self_attn.sink_proj.weight",
    "attn.q_l2_clip.scale": "self_attn.q_l2_clip.scale",
    "attn.rel_l2_clip.scale": "self_attn.rel_l2_clip.scale",
    "attn.rel_logits_proj.proj": "self_attn.rel_logits_proj.proj",
    # Sliding layers: static depthwise convolutions.
    "attn.k_sconv.weight": "self_attn.k_sconv.conv1d.weight",
    "attn.v_sconv.weight": "self_attn.v_sconv.conv1d.weight",
    # Full layers: dynamic convolutions.
    "attn.k_sconv.static_w": "self_attn.k_sconv.static_weight",
    "attn.k_sconv.weight_proj": "self_attn.k_sconv.weight_proj.weight",
    "attn.v_sconv.static_w": "self_attn.v_sconv.static_weight",
    "attn.v_sconv.weight_proj": "self_attn.v_sconv.weight_proj.weight",
    "attn_sconv.weight": "post_attention_convolution.conv1d.weight",
    "mlp_sconv.weight": "post_mlp_convolution.conv1d.weight",
    "attn_norm.weight": "input_layernorm.weight",
    "q_input_norm.weight": "q_input_layernorm.weight",
    "k_input_norm.weight": "k_input_layernorm.weight",
    "v_input_norm.weight": "v_input_layernorm.weight",
    "mlp_norm.weight": "post_attention_layernorm.weight",
    "mlp.gate.weight": "mlp.gate.weight",
    "mlp.gate.bias": "mlp.gate.e_score_correction_bias",
    "mlp.experts.w13_weight": "mlp.experts.gate_and_up_projs",
    "mlp.experts.w2_weight": "mlp.experts.down_projs",
}
_NATIVE_LAYER_SUFFIXES = {native: raw for raw, native in _LAYER_SUFFIXES.items()}

# Whole-key vision renames (raw -> native).
_VISION_GLOBAL_KEYS = {
    "model.visual.backbone.embeddings.class_embedding": "model.vision_tower.embeddings.class_embedding",
    "model.visual.backbone.embeddings.cls_position_embedding": "model.vision_tower.embeddings.cls_position_embedding",
    "model.visual.backbone.embeddings.patch_embedding.weight": "model.vision_tower.embeddings.patch_embedding.weight",
    "model.visual.backbone.embeddings.position_embedding": "model.vision_tower.embeddings.position_embedding.weight",
    "model.visual.backbone.ln_pre.weight": "model.vision_tower.pre_layernorm.weight",
    "model.visual.backbone.ln_pre.bias": "model.vision_tower.pre_layernorm.bias",
    "model.visual.proj_in.weight": "model.vision_tower.merger.linear_1.weight",
    "model.visual.proj_in.bias": "model.vision_tower.merger.linear_1.bias",
    "model.visual.proj_out.weight": "model.vision_tower.merger.linear_2.weight",
    "model.visual.proj_out.bias": "model.vision_tower.merger.linear_2.bias",
}

# Vision encoder-layer suffix renames (raw -> native), applied after ``layers.<i>.``.
_VISION_LAYER_SUFFIXES = {
    "ln_1.weight": "layernorm_before.weight",
    "ln_1.bias": "layernorm_before.bias",
    "ln_2.weight": "layernorm_after.weight",
    "ln_2.bias": "layernorm_after.bias",
    "ls_1.gamma": "lambda_1",
    "ls_2.gamma": "lambda_2",
    "mlp.c_fc.weight": "mlp.fc1.weight",
    "mlp.c_fc.bias": "mlp.fc1.bias",
    "mlp.c_proj.weight": "mlp.fc2.weight",
    "mlp.c_proj.bias": "mlp.fc2.bias",
    **{
        f"self_attn.{projection}.{kind}": f"self_attn.{projection}.{kind}"
        for projection in ("q_proj", "k_proj", "v_proj", "out_proj")
        for kind in ("weight", "bias")
    },
}

_RAW_LAYER_RE = re.compile(r"model\.llm\.layers\.(\d+)\.(.+)")
_NATIVE_LAYER_RE = re.compile(r"model\.language_model\.layers\.(\d+)\.(.+)")
_RAW_ATTN_QUERY_RE = re.compile(r"model\.llm\.attn_res\.attn_queries\.(\d+)")
_RAW_MLP_QUERY_RE = re.compile(r"model\.llm\.attn_res\.mlp_queries\.(\d+)")
_RAW_VISION_LAYER_RE = re.compile(r"model\.visual\.backbone\.layers\.(\d+)\.(.+)")
_NATIVE_VISION_LAYER_RE = re.compile(r"model\.vision_tower\.layers\.(\d+)\.(.+)")
_RAW_EXPERT_RE = re.compile(r"(model\.llm\.layers\.\d+\.mlp\.experts\.w(?:13|2)_weight)(.*)")
_EXPERT_COMPANION_SUFFIXES = ("", "_scale", "_amax", "_outer_scale_denominator")
_NATIVE_EXPERT_SUFFIXES = (".mlp.experts.gate_and_up_projs", ".mlp.experts.down_projs")


def raw_to_native_key(key: str) -> str:
    """Map a released-checkpoint key to its native AutoModel key.

    Args:
        key: Released key, for example ``model.llm.layers.0.attn.wq_du.weight``.

    Returns:
        The native key, for example ``model.language_model.layers.0.self_attn.q_proj.weight``.

    Raises:
        KeyError: If ``key`` is not an Inkling 2 checkpoint key. Quantization companions
            (``*_weight_scale`` and similar) are not keys of their own and also raise.
    """
    if key in _TEXT_GLOBAL_KEYS:
        return _TEXT_GLOBAL_KEYS[key]
    if key in _VISION_GLOBAL_KEYS:
        return _VISION_GLOBAL_KEYS[key]
    if match := _RAW_ATTN_QUERY_RE.fullmatch(key):
        # attn_queries.<i> belongs to layer i + 1: layer 0 has no attention-input mixing.
        return f"{_NATIVE_TEXT}layers.{int(match.group(1)) + 1}.attention_input.query"
    if match := _RAW_MLP_QUERY_RE.fullmatch(key):
        return f"{_NATIVE_TEXT}layers.{match.group(1)}.mlp_input.query"
    if (match := _RAW_LAYER_RE.fullmatch(key)) and match.group(2) in _LAYER_SUFFIXES:
        return f"{_NATIVE_TEXT}layers.{match.group(1)}.{_LAYER_SUFFIXES[match.group(2)]}"
    if (match := _RAW_VISION_LAYER_RE.fullmatch(key)) and match.group(2) in _VISION_LAYER_SUFFIXES:
        return f"{_NATIVE_VISION}layers.{match.group(1)}.{_VISION_LAYER_SUFFIXES[match.group(2)]}"
    raise KeyError(f"Unknown Inkling 2 checkpoint key: {key}")


_NATIVE_TEXT_GLOBAL_KEYS = {native: raw for raw, native in _TEXT_GLOBAL_KEYS.items()}
_NATIVE_VISION_GLOBAL_KEYS = {native: raw for raw, native in _VISION_GLOBAL_KEYS.items()}
_NATIVE_VISION_LAYER_SUFFIXES = {native: raw for raw, native in _VISION_LAYER_SUFFIXES.items()}


def native_to_raw_key(key: str) -> str:
    """Map a native AutoModel key to its released-checkpoint key; inverse of :func:`raw_to_native_key`.

    Args:
        key: Native key.

    Returns:
        The released key.

    Raises:
        KeyError: If ``key`` is not a native Inkling 2 parameter or buffer name.
    """
    if key in _NATIVE_TEXT_GLOBAL_KEYS:
        return _NATIVE_TEXT_GLOBAL_KEYS[key]
    if key in _NATIVE_VISION_GLOBAL_KEYS:
        return _NATIVE_VISION_GLOBAL_KEYS[key]
    if match := _NATIVE_LAYER_RE.fullmatch(key):
        layer, suffix = int(match.group(1)), match.group(2)
        if suffix == "attention_input.query" and layer > 0:
            return f"{_RAW_TEXT}attn_res.attn_queries.{layer - 1}"
        if suffix == "mlp_input.query":
            return f"{_RAW_TEXT}attn_res.mlp_queries.{layer}"
        if suffix in _NATIVE_LAYER_SUFFIXES:
            return f"{_RAW_TEXT}layers.{layer}.{_NATIVE_LAYER_SUFFIXES[suffix]}"
    if (match := _NATIVE_VISION_LAYER_RE.fullmatch(key)) and match.group(2) in _NATIVE_VISION_LAYER_SUFFIXES:
        return f"{_RAW_VISION}backbone.layers.{match.group(1)}.{_NATIVE_VISION_LAYER_SUFFIXES[match.group(2)]}"
    raise KeyError(f"Unknown native Inkling 2 key: {key}")


def _dequantize_experts(parts: dict[str, torch.Tensor], raw_key: str, dtype: torch.dtype) -> torch.Tensor:
    """Return expert weights in the released ``[experts, out, in]`` layout and ``dtype``.

    Args:
        parts: The expert weight under ``""`` and its quantization companions under
            ``"_scale"``, ``"_amax"`` and ``"_outer_scale_denominator"``, already narrowed to
            the local experts. NVFP4: ``uint8 [experts, out, in / 2]``, ``float8_e4m3fn``
            scales ``[experts, out, in / 16]``, FP32 ``amax`` ``[experts]`` and denominator ``[1]``.
            MXFP8: ``float8_e4m3fn [experts, out, in]`` with ``uint8`` scales ``[experts, out, in / 32]``.
            Unquantized: a floating tensor ``[experts, out, in]`` alone.
        raw_key: Released weight key, for error messages.
        dtype: Output dtype; quantized weights are decoded straight into it (one rounding).

    Returns:
        Tensor of shape ``[experts, out, in]`` in ``dtype``.

    Raises:
        ValueError: If the companions do not describe a supported format.
    """
    weight = parts[""]
    companions = set(parts) - {""}
    if companions == {"_scale", "_amax", "_outer_scale_denominator"}:
        return dequantize_nvfp4_experts(
            weight, parts["_scale"], parts["_amax"], parts["_outer_scale_denominator"], dtype=dtype
        )
    if companions == {"_scale"} and weight.dtype == torch.float8_e4m3fn:
        return dequantize_mxfp8_experts(weight, parts["_scale"], dtype=dtype)
    if not companions and weight.is_floating_point() and weight.dtype != torch.float8_e4m3fn:
        return weight.to(dtype)
    raise ValueError(f"Unsupported expert format for {raw_key}: dtype={weight.dtype}, companions={sorted(companions)}")


class Inkling2StateDictAdapter(StateDictAdapter):
    """Convert Inkling 2 weights between native AutoModel and released checkpoint layouts."""

    def __init__(self, config: Inkling2TextConfig, *, dtype: torch.dtype) -> None:
        """Create the adapter.

        Args:
            config: Text configuration (number of routed experts).
            dtype: Storage dtype of the native expert weights.
        """
        self.config = config
        self.dtype = dtype

    def to_hf(
        self,
        state_dict: dict[str, Any],
        exclude_key_regex: str | None = None,
        quantization: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Convert a native state dict to released checkpoint keys and layouts.

        Expert matrices become transposed views; every other tensor keeps its storage.

        Args:
            state_dict: Native state dict. Expert tensors have shape ``[experts, hidden, 2 *
                intermediate]`` (interleaved gate/up) and ``[experts, intermediate, hidden]``.
            exclude_key_regex: Optional regex of released keys to drop.
            quantization: Unused; experts are written unquantized.
            **kwargs: Unused checkpointer arguments.

        Returns:
            Released-layout state dict with ``w13_weight`` ``[experts, 2 * intermediate, hidden]``
            and ``w2_weight`` ``[experts, hidden, intermediate]``.
        """
        del quantization, kwargs
        checkpoint_state: dict[str, Any] = {}
        for fqn, tensor in state_dict.items():
            for key, value in self.convert_single_tensor_to_hf(fqn, tensor, exclude_key_regex=exclude_key_regex):
                checkpoint_state[key] = value
        return checkpoint_state

    def convert_single_tensor_to_hf(self, fqn: str, tensor: Any, **kwargs: Any) -> list[tuple[str, Any]]:
        """Convert one native tensor to the released layout.

        Args:
            fqn: Native key.
            tensor: Tensor of the native layout; expert tensors are ``[experts, in, out]``.
            **kwargs: ``exclude_key_regex`` optionally drops matching released keys.

        Returns:
            One ``(released_key, tensor)`` pair, or none when excluded. Expert tensors are
            transposed views of shape ``[experts, out, in]``.
        """
        raw_key = native_to_raw_key(fqn)
        exclude_key_regex = kwargs.get("exclude_key_regex")
        if exclude_key_regex and re.match(exclude_key_regex, raw_key):
            return []
        if fqn.endswith(_NATIVE_EXPERT_SUFFIXES):
            tensor = tensor.transpose(-1, -2)
        return [(raw_key, tensor)]

    def from_hf(
        self,
        hf_state_dict: dict[str, Any],
        device_mesh: DeviceMesh | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Convert a released-layout state dict to native keys and layouts.

        Args:
            hf_state_dict: Released keys mapped to tensors. Expert weights may be BF16/FP32, NVFP4
                or MXFP8 with their companions; with ``device_mesh`` they may hold all experts
                or already be expert-sharded DTensors.
            device_mesh: Optional mesh; only this rank's experts are kept and wrapped as DTensors.
            **kwargs: Unused checkpointer arguments.

        Returns:
            Native state dict. Experts are ``[local_experts, hidden, 2 * intermediate]`` and
            ``[local_experts, intermediate, hidden]`` in the adapter dtype.

        Raises:
            KeyError: For keys that are not Inkling 2 checkpoint keys.
            ValueError: For unsupported expert formats.
        """
        del kwargs
        n_experts = self.config.n_routed_experts
        start_expert, end_expert, rank = 0, n_experts, None
        if device_mesh is not None:
            start_expert, end_expert = state_dict_utils.get_expert_range_for_rank_from_mesh(device_mesh, n_experts)
            rank = (
                state_dict_utils.get_submesh(device_mesh, ("ep",)).get_rank()
                if "ep" in device_mesh.mesh_dim_names
                else device_mesh.get_rank()
            )

        state_dict: dict[str, Any] = {}
        expert_parts: dict[str, dict[str, torch.Tensor]] = {}
        for key, value in hf_state_dict.items():
            if (match := _RAW_EXPERT_RE.fullmatch(key)) and match.group(2) in _EXPERT_COMPANION_SUFFIXES:
                expert_parts.setdefault(match.group(1), {})[match.group(2)] = value
                continue
            state_dict[raw_to_native_key(key)] = value

        for raw_key, parts in expert_parts.items():
            native_key = raw_to_native_key(raw_key)
            if "" not in parts:
                raise ValueError(f"Expert quantization companions without {raw_key}")
            if state_dict_utils.is_dtensor(parts[""]):
                if len(parts) > 1:
                    raise ValueError(f"Sharded quantized experts are not supported for {raw_key}")
                state_dict[native_key] = parts[""].transpose(-1, -2)
                continue
            local_parts = {
                suffix: tensor if suffix == "_outer_scale_denominator" else tensor[start_expert:end_expert]
                for suffix, tensor in parts.items()
            }
            weight = _dequantize_experts(local_parts, raw_key, self.dtype).transpose(-1, -2).contiguous()
            if device_mesh is not None:
                weight = state_dict_utils.create_dtensor_from_local(weight, device_mesh, rank)
            state_dict[native_key] = weight
        return state_dict


__all__ = ["Inkling2StateDictAdapter", "native_to_raw_key", "raw_to_native_key"]
