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

    @staticmethod
    def _is_unsupported_stage1_key(key: str) -> bool:
        """Sparse-attention index branch (Stage 2) and MTP (Stage 4) tensors."""
        return ".self_attn.index_" in key or ".mtp." in key

    def _hf_key_to_native(self, key: str) -> str:
        key = key.replace(".block_sparse_moe.gate.weight", ".mlp.gate.weight")
        key = key.replace(".block_sparse_moe.e_score_correction_bias", ".mlp.gate.e_score_correction_bias")
        key = key.replace(".block_sparse_moe.shared_experts.", ".shared_experts.")
        key = re.sub(r"\.block_sparse_moe\.experts\.(\d+)\.w1\.weight$", r".mlp.experts.\1.gate_proj.weight", key)
        key = re.sub(r"\.block_sparse_moe\.experts\.(\d+)\.w3\.weight$", r".mlp.experts.\1.up_proj.weight", key)
        key = re.sub(r"\.block_sparse_moe\.experts\.(\d+)\.w2\.weight$", r".mlp.experts.\1.down_proj.weight", key)
        return key

    def _native_key_to_hf(self, key: str) -> str:
        key = re.sub(r"\.mlp\.experts\.(\d+)\.gate_proj\.weight$", r".block_sparse_moe.experts.\1.w1.weight", key)
        key = re.sub(r"\.mlp\.experts\.(\d+)\.up_proj\.weight$", r".block_sparse_moe.experts.\1.w3.weight", key)
        key = re.sub(r"\.mlp\.experts\.(\d+)\.down_proj\.weight$", r".block_sparse_moe.experts.\1.w2.weight", key)
        key = key.replace(".mlp.gate.weight", ".block_sparse_moe.gate.weight")
        key = key.replace(".mlp.gate.e_score_correction_bias", ".block_sparse_moe.e_score_correction_bias")
        key = key.replace(".shared_experts.", ".block_sparse_moe.shared_experts.")
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

        # Stage 1: drop tensors for components not yet implemented (index branch / MTP).
        for key in list(hf_state_dict.keys()):
            if self._is_unsupported_stage1_key(key):
                hf_state_dict.pop(key, None)

        self._dequantize(hf_state_dict)
        for key in list(hf_state_dict.keys()):
            new_key = self._hf_key_to_native(key)
            if new_key != key:
                hf_state_dict[new_key] = hf_state_dict.pop(key)
        return self._from_hf_w_merged_experts(hf_state_dict, device_mesh)

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
        exclude_key_regex = kwargs.get("exclude_key_regex", None)

        expert_result = self._convert_single_merged_expert_to_hf_split_experts(fqn, tensor, **kwargs)
        if expert_result is not None:
            result = [(self._native_key_to_hf(k), v) for k, v in expert_result]
        else:
            result = [(self._native_key_to_hf(fqn), tensor)]

        if exclude_key_regex:
            result = [(k, v) for k, v in result if not re.match(exclude_key_regex, k)]
        return result
