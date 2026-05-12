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

Qwen3.5 dense uses HF's GatedDeltaNet linear-attention layers. For FSDP
compatibility (mixed-dtype: bf16 + fp32 ``A_log``), ``patch_hf_model`` in
``cp_linear_attn`` moves ``A_log`` from ``mod._parameters`` into a
``_fp32_params`` submodule and patches ``__getattr__`` to redirect
``mod.A_log`` reads. After patching, the model's state_dict contains keys of
the form ``...linear_attn._fp32_params.A_log`` instead of the original
``...linear_attn.A_log``.

This adapter renames keys at save/load boundaries so that on-disk checkpoints
match the original HF Qwen3.5 layout (bare ``A_log``) and are directly
loadable via ``transformers.AutoModelForImageTextToText.from_pretrained``.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter

_FP32_PARAMS_TO_BARE = re.compile(r"(\.linear_attn)\._fp32_params\.")
_BARE_FP32_PARAM_NAMES = ("A_log",)


def _strip_fp32_prefix(key: str) -> str:
    return _FP32_PARAMS_TO_BARE.sub(r"\1.", key)


def _route_to_fp32_holder(key: str) -> str:
    if not key.endswith(_BARE_FP32_PARAM_NAMES):
        return key
    if "._fp32_params." in key:
        return key
    if ".linear_attn." not in key:
        return key
    head, tail = key.rsplit(".linear_attn.", 1)
    return f"{head}.linear_attn._fp32_params.{tail}"


class Qwen3_5DenseStateDictAdapter(StateDictAdapter):
    """Adapter that hides the ``_fp32_params`` wrapping in saved checkpoints."""

    def to_hf(self, state_dict: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        return {_strip_fp32_prefix(k): v for k, v in state_dict.items()}

    def from_hf(
        self,
        hf_state_dict: dict[str, Any],
        device_mesh: Optional[Any] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return {_route_to_fp32_holder(k): v for k, v in hf_state_dict.items()}

    def convert_single_tensor_to_hf(self, fqn: str, tensor: Any, **kwargs: Any) -> list[tuple[str, Any]]:
        return [(_strip_fp32_prefix(fqn), tensor)]
