# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""State-dict adapter for Qwen3.5 dense (non-MoE) models.

Native parameter names equal the HF checkpoint names, including the fp32
GatedDeltaNet decay-gate parameters (``linear_attn.A_log`` / ``linear_attn.dt_bias``),
so the adapter only maps the Megatron-style MTP module keys to the HF MTP layout.
Checkpoint loads copy into the model's existing fp32 parameters and the HF export
pins them to F32 from ``_keep_in_fp32_modules_strict``.
"""

from __future__ import annotations

from typing import Any

from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter

_MTP_HF_TO_NATIVE = {
    "mtp.fc.weight": "mtp.layers.0.eh_proj.weight",
    "mtp.pre_fc_norm_embedding.weight": "mtp.layers.0.enorm.weight",
    "mtp.pre_fc_norm_hidden.weight": "mtp.layers.0.hnorm.weight",
    "mtp.norm.weight": "mtp.layers.0.final_layernorm.weight",
}
_MTP_NATIVE_TO_HF = {v: k for k, v in _MTP_HF_TO_NATIVE.items()}


def map_qwen3_5_mtp_from_hf_key(key: str) -> str:
    """Map HF Qwen3.5 MTP keys to Automodel's Megatron-style MTP module."""
    return _MTP_HF_TO_NATIVE.get(key, key)


def map_qwen3_5_mtp_to_hf_key(key: str) -> str:
    """Map Automodel Qwen3.5 MTP keys back to HF checkpoint keys."""
    return _MTP_NATIVE_TO_HF.get(key, key)


class Qwen3_5DenseStateDictAdapter(StateDictAdapter):
    """Map the MTP module keys between the native and HF layouts."""

    _supports_low_memory_dcp_load = True

    def to_hf(self, state_dict: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        return {map_qwen3_5_mtp_to_hf_key(key): value for key, value in state_dict.items()}

    def from_hf(
        self,
        hf_state_dict: dict[str, Any],
        device_mesh: Any | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del device_mesh, kwargs
        return {map_qwen3_5_mtp_from_hf_key(key): value for key, value in hf_state_dict.items()}

    def convert_single_tensor_to_hf(self, fqn: str, tensor: Any, **kwargs: Any) -> list[tuple[str, Any]]:
        return [(map_qwen3_5_mtp_to_hf_key(fqn), tensor)]
