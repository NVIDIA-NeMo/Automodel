# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""State-dict adapter for the linear-memory Titans model.

The native parameter names already match a flat HuggingFace layout 1:1, so the
conversion is an identity remap. Its real job is the precision contract: the
decay-gate params ``A_log`` / ``dt_bias`` are intrinsically fp32 (``A_log`` is
exponentiated), so they are upcast to fp32 on load and exported as ``F32``.
"""

from __future__ import annotations

import re
from typing import Any, Optional

import torch
from torch.distributed.device_mesh import DeviceMesh

from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter

# Intrinsically-fp32 decay-gate params (per neural-memory head).
_FP32_PARAM_NAMES = ("A_log", "dt_bias")


def _is_fp32_param_key(key: str) -> bool:
    return key.endswith(_FP32_PARAM_NAMES) and ".memory." in key


def _maybe_upcast(key: str, tensor: Any) -> Any:
    """Cast intrinsically-fp32 params to fp32; leave everything else untouched."""
    if not _is_fp32_param_key(key):
        return tensor
    if getattr(tensor, "dtype", None) == torch.float32:
        return tensor
    is_fp = getattr(tensor, "is_floating_point", None)
    if callable(is_fp) and is_fp():
        return tensor.to(dtype=torch.float32)
    return tensor


class TitansStateDictAdapter(StateDictAdapter):
    """Identity HF<->native remap that preserves the fp32 decay-gate contract."""

    def __init__(self, config: Any, dtype: torch.dtype = torch.float32):
        self.config = config
        self.dtype = dtype

    def to_hf(self, state_dict: dict[str, Any], exclude_key_regex: Optional[str] = None, **kwargs) -> dict[str, Any]:
        hf_state_dict: dict[str, Any] = {}
        for fqn, tensor in state_dict.items():
            for key, value in self.convert_single_tensor_to_hf(fqn, tensor, exclude_key_regex=exclude_key_regex):
                hf_state_dict[key] = value
        return hf_state_dict

    def from_hf(
        self, hf_state_dict: dict[str, Any], device_mesh: Optional["DeviceMesh"] = None, **kwargs
    ) -> dict[str, Any]:
        return {key: _maybe_upcast(key, value) for key, value in hf_state_dict.items()}

    def convert_single_tensor_to_hf(self, fqn: str, tensor: Any, **kwargs) -> list[tuple[str, Any]]:
        exclude_key_regex = kwargs.get("exclude_key_regex", None)
        if exclude_key_regex and re.match(exclude_key_regex, fqn):
            return []
        return [(fqn, _maybe_upcast(fqn, tensor))]

    def forced_hf_dtype_mapping(self, state_dict: dict[str, Any]) -> dict[str, str]:
        """Force ``F32`` export dtype for the intrinsically-fp32 decay-gate params."""
        forced: dict[str, str] = {}
        for key, tensor in state_dict.items():
            if not _is_fp32_param_key(key):
                continue
            is_fp = getattr(tensor, "is_floating_point", None)
            if callable(is_fp) and is_fp():
                forced[key] = "F32"
        return forced
