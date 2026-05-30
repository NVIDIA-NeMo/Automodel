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

"""State-dict adapter for the MiniMax M3 (text) backbone.

Converts between the released HF checkpoint layout and the native AutoModel
layout:

* ``block_sparse_moe.{gate,e_score_correction_bias}`` -> ``mlp.gate.*``
* ``block_sparse_moe.experts.{e}.{w1,w3,w2}`` -> grouped ``mlp.experts.*``
  (gate/up/down) via ``MoESplitExpertsStateDictMixin``
* ``block_sparse_moe.shared_experts.*`` -> ``shared_experts.*`` (a sibling of
  ``mlp`` on the decoder block)
* dense (non-MoE) layers keep ``mlp.{gate,up,down}_proj.*`` unchanged

MXFP8 weights (FP8 e4m3 + ``*_scale_inv`` stored as e8m0/uint8, block ``[1,32]``
along the input dim) are dequantized to ``dtype`` on load (Q2 decision: train in
BF16). Stage 1 drops the sparse-attention index branch (``self_attn.index_*``)
and MTP (``mtp.*``) tensors; those are wired in Stages 2 and 4.
"""

import re
from typing import Any, Optional

import torch
from torch.distributed.device_mesh import DeviceMesh

from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.moe.layers import MoEConfig
from nemo_automodel.components.moe.state_dict_mixin import MoESplitExpertsStateDictMixin

# MXFP8 block layout from config.json: weight_block_size = [1, 32] (1 row, 32 cols).
MXFP8_BLOCK_SIZE = 32


def dequantize_mxfp8(
    weight: torch.Tensor,
    scale_inv: torch.Tensor,
    *,
    block_size: int = MXFP8_BLOCK_SIZE,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize an MXFP8 weight to ``dtype``.

    ``weight`` is FP8 ``e4m3`` of shape ``[out, in]``; ``scale_inv`` holds e8m0
    (uint8) exponents of shape ``[out, in / block_size]`` where the dequant
    scale for block ``b`` is ``2 ** (scale_inv[:, b] - 127)`` (the MX e8m0
    convention; confirmed against the sglang reference).
    """
    w = weight.to(torch.float32)
    scale = torch.exp2(scale_inv.to(torch.float32) - 127.0)
    scale = scale.repeat_interleave(block_size, dim=1)
    if scale.shape[1] != w.shape[1]:
        scale = scale[:, : w.shape[1]]
    return (w * scale).to(dtype)


class MiniMaxM3StateDictAdapter(MoESplitExpertsStateDictMixin, StateDictAdapter):
    """Convert MiniMax M3 HF checkpoints to/from the native grouped-expert format."""

    def __init__(
        self,
        config: Any,
        moe_config: MoEConfig,
        backend: BackendConfig,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.config = config
        self.moe_config = moe_config
        self.backend = backend
        self.dtype = dtype
        self._uses_model_prefix = True

    @property
    def _expert_path_segment(self) -> str:
        return "mlp.experts"

    def _dequantize(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        scale_inv_keys = []
        for key, weight in state_dict.items():
            scale_key = key + "_scale_inv"
            if key.endswith(".weight") and scale_key in state_dict:
                state_dict[key] = dequantize_mxfp8(weight, state_dict[scale_key], dtype=self.dtype)
                scale_inv_keys.append(scale_key)
        for key in scale_inv_keys:
            state_dict.pop(key, None)
        return state_dict

    @property
    def _mtp_enabled(self) -> bool:
        return int(getattr(self.config, "num_mtp_modules", 0) or 0) > 0

    def _hf_key_to_native(self, key: str) -> str:
        key = key.replace(".block_sparse_moe.gate.weight", ".mlp.gate.weight")
        key = key.replace(".block_sparse_moe.e_score_correction_bias", ".mlp.gate.e_score_correction_bias")
        key = key.replace(".block_sparse_moe.shared_experts.", ".shared_experts.")
        key = re.sub(r"\.block_sparse_moe\.experts\.(\d+)\.w1\.weight$", r".mlp.experts.\1.gate_proj.weight", key)
        key = re.sub(r"\.block_sparse_moe\.experts\.(\d+)\.w3\.weight$", r".mlp.experts.\1.up_proj.weight", key)
        key = re.sub(r"\.block_sparse_moe\.experts\.(\d+)\.w2\.weight$", r".mlp.experts.\1.down_proj.weight", key)
        # Sparse-attention index branch lives under an ``indexer`` submodule.
        key = key.replace(".self_attn.index_", ".self_attn.indexer.index_")
        return key

    def _native_key_to_hf(self, key: str) -> str:
        key = re.sub(r"\.mlp\.experts\.(\d+)\.gate_proj\.weight$", r".block_sparse_moe.experts.\1.w1.weight", key)
        key = re.sub(r"\.mlp\.experts\.(\d+)\.up_proj\.weight$", r".block_sparse_moe.experts.\1.w3.weight", key)
        key = re.sub(r"\.mlp\.experts\.(\d+)\.down_proj\.weight$", r".block_sparse_moe.experts.\1.w2.weight", key)
        key = key.replace(".mlp.gate.weight", ".block_sparse_moe.gate.weight")
        key = key.replace(".mlp.gate.e_score_correction_bias", ".block_sparse_moe.e_score_correction_bias")
        key = key.replace(".shared_experts.", ".block_sparse_moe.shared_experts.")
        key = key.replace(".self_attn.indexer.index_", ".self_attn.index_")
        return key

    def from_hf(
        self,
        hf_state_dict: dict[str, Any],
        device_mesh: Optional["DeviceMesh"] = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Convert an HF checkpoint to native format (operates in-place to limit peak memory)."""
        for key in hf_state_dict.keys():
            if ".block_sparse_moe.experts." in key and key.endswith(".weight"):
                self._uses_model_prefix = key.startswith("model.")
                break

        # MTP tensors are converted separately (the transformer_layer is a full
        # decoder block); dropped entirely when the model has no MTP module.
        mtp_keys = {k: hf_state_dict.pop(k) for k in list(hf_state_dict) if ".mtp." in k}

        self._dequantize(hf_state_dict)
        for key in list(hf_state_dict.keys()):
            new_key = self._hf_key_to_native(key)
            if new_key != key:
                hf_state_dict[new_key] = hf_state_dict.pop(key)
        native = self._from_hf_w_merged_experts(hf_state_dict, device_mesh)

        if mtp_keys and self._mtp_enabled:
            native.update(self._mtp_from_hf(mtp_keys, device_mesh))
        return native

    def _mtp_from_hf(self, mtp_keys: dict[str, Any], device_mesh: Optional["DeviceMesh"] = None) -> dict[str, Any]:
        """Convert MTP tensors: the transformer_layer reuses the full text from_hf
        (as a fake 1-layer model, so expert-merge / index / dequant all apply); the
        enorm/hnorm/eh_proj/final_layernorm fusion tensors pass through (eh_proj is FP8)."""
        pattern = re.compile(r"(?P<pfx>.*?)mtp\.layers\.(?P<d>\d+)\.(?P<rest>.+)")
        tl_hf: dict[str, Any] = {}
        passthrough: dict[str, Any] = {}
        for key, value in mtp_keys.items():
            m = pattern.match(key)
            depth, rest = m.group("d"), m.group("rest")
            if rest.startswith("transformer_layer."):
                tl_hf[f"model.layers.{depth}.{rest[len('transformer_layer.') :]}"] = value
            else:
                passthrough[key] = value

        self._dequantize(passthrough)  # eh_proj is MXFP8
        native = dict(passthrough)
        for key, value in self.from_hf(tl_hf, device_mesh).items():
            m = re.match(r"model\.layers\.(\d+)\.(.+)", key)
            native[f"model.mtp.layers.{m.group(1)}.transformer_layer.{m.group(2)}"] = value
        return native

    def to_hf(
        self,
        state_dict: dict[str, Any],
        exclude_key_regex: Optional[str] = None,
        quantization: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        hf_state_dict = {}
        for fqn, tensor in state_dict.items():
            for key, value in self.convert_single_tensor_to_hf(
                fqn, tensor, exclude_key_regex=exclude_key_regex, **kwargs
            ):
                hf_state_dict[key] = value
        return hf_state_dict

    def convert_single_tensor_to_hf(self, fqn: str, tensor: Any, **kwargs) -> list[tuple[str, Any]]:
        if ".mtp." in fqn:
            return self._mtp_tensor_to_hf(fqn, tensor, **kwargs)

        exclude_key_regex = kwargs.get("exclude_key_regex", None)

        expert_result = self._convert_single_merged_expert_to_hf_split_experts(fqn, tensor, **kwargs)
        if expert_result is not None:
            result = [(self._native_key_to_hf(k), v) for k, v in expert_result]
        else:
            result = [(self._native_key_to_hf(fqn), tensor)]

        if exclude_key_regex:
            result = [(k, v) for k, v in result if not re.match(exclude_key_regex, k)]
        return result

    def _mtp_tensor_to_hf(self, fqn: str, tensor: Any, **kwargs) -> list[tuple[str, Any]]:
        m = re.match(r"(?P<head>.*?)mtp\.layers\.(?P<d>\d+)\.(?P<rest>.+)", fqn)
        head, depth, rest = m.group("head"), m.group("d"), m.group("rest")
        if rest.startswith("transformer_layer."):
            suffix = rest[len("transformer_layer.") :]
            converted = self.convert_single_tensor_to_hf(f"model.layers.{depth}.{suffix}", tensor, **kwargs)
            tl_prefix = f"{head}mtp.layers.{depth}.transformer_layer."
            return [(k.replace(f"model.layers.{depth}.", tl_prefix, 1), v) for k, v in converted]
        exclude_key_regex = kwargs.get("exclude_key_regex", None)
        if exclude_key_regex and re.match(exclude_key_regex, fqn):
            return []
        return [(fqn, tensor)]


class MiniMaxM3VLStateDictAdapter(StateDictAdapter):
    """VLM adapter: splits the M3 VL checkpoint into text / vision / projector parts.

    The released checkpoint stores the language backbone under
    ``language_model.model.*`` / ``language_model.lm_head`` and the vision side
    under ``vision_tower.vision_model.*`` with the projector / patch-merger at
    top level (``multi_modal_projector.*`` / ``patch_merge_mlp.*``).  The native
    VLM keeps the text model at ``model.*`` / ``lm_head`` and nests the projector
    / merger under ``vision_tower.*``.  Text tensors are delegated to
    :class:`MiniMaxM3StateDictAdapter` (block_sparse_moe -> mlp, index branch,
    MXFP8 dequant, grouped experts); vision tensors are BF16 and pass through.
    """

    def __init__(self, config: Any, moe_config: MoEConfig, backend: BackendConfig, dtype: torch.dtype = torch.bfloat16):
        self.config = config
        self.text_adapter = MiniMaxM3StateDictAdapter(config.text_config, moe_config, backend, dtype=dtype)

    @staticmethod
    def _map_non_text_from_hf(key: str) -> str | None:
        if ".mtp." in key:
            return None  # MTP (Stage 4)
        if key.startswith("multi_modal_projector.") or key.startswith("patch_merge_mlp."):
            return "vision_tower." + key
        return key  # vision_tower.vision_model.* passes through

    @staticmethod
    def _map_non_text_to_hf(key: str) -> str:
        if key.startswith("vision_tower.multi_modal_projector.") or key.startswith("vision_tower.patch_merge_mlp."):
            return key[len("vision_tower.") :]
        return key

    def from_hf(self, hf_state_dict: dict[str, Any], device_mesh=None, **kwargs) -> dict[str, Any]:
        # Native text keys are model.* / lm_head.* (self.model + self.lm_head);
        # the checkpoint's language_model. prefix is stripped here and re-added in to_hf.
        text_hf: dict[str, Any] = {}
        native: dict[str, Any] = {}
        for key, value in hf_state_dict.items():
            if key.startswith("language_model."):
                text_hf[key[len("language_model.") :]] = value
                continue
            mapped = self._map_non_text_from_hf(key)
            if mapped is not None:
                native[mapped] = value

        native.update(self.text_adapter.from_hf(text_hf, device_mesh=device_mesh, **kwargs))
        return native

    def to_hf(self, state_dict: dict[str, Any], exclude_key_regex=None, quantization: bool = False, **kwargs):
        hf_state_dict: dict[str, Any] = {}
        for key, tensor in state_dict.items():
            for hf_key, hf_tensor in self.convert_single_tensor_to_hf(
                key, tensor, exclude_key_regex=exclude_key_regex, quantization=quantization, **kwargs
            ):
                hf_state_dict[hf_key] = hf_tensor
        return hf_state_dict

    def convert_single_tensor_to_hf(self, fqn: str, tensor: Any, **kwargs) -> list[tuple[str, Any]]:
        exclude_key_regex = kwargs.get("exclude_key_regex", None)
        if fqn.startswith("vision_tower."):
            hf_key = self._map_non_text_to_hf(fqn)
            if exclude_key_regex and re.match(exclude_key_regex, hf_key):
                return []
            return [(hf_key, tensor)]
        # Text backbone (model.* / lm_head.*): delegate, then re-add language_model. prefix.
        converted = self.text_adapter.convert_single_tensor_to_hf(fqn, tensor, **kwargs)
        return [("language_model." + k, v) for k, v in converted]
