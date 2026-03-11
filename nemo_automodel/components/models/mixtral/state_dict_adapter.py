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

"""State-dict adapter for Mixtral MoE checkpoints.

Converts between HuggingFace per-expert format::

    model.layers.{L}.block_sparse_moe.experts.{E}.w1.weight   (gate_proj)
    model.layers.{L}.block_sparse_moe.experts.{E}.w3.weight   (up_proj)
    model.layers.{L}.block_sparse_moe.experts.{E}.w2.weight   (down_proj)

and the native grouped-expert format::

    model.layers.{L}.mlp.experts.gate_and_up_projs   # [n_experts, 2*inter, dim]
    model.layers.{L}.mlp.experts.down_projs           # [n_experts, dim, inter]
"""

import logging
import re
from typing import Any, Optional

import torch
from torch.distributed.device_mesh import DeviceMesh

from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.state_dict_mixin import MoESplitExpertsStateDictMixin

logger = logging.getLogger(__name__)

# HF Mixtral uses ``block_sparse_moe`` for the MoE sub-module and
# ``w1`` / ``w3`` / ``w2`` instead of the more common
# ``gate_proj`` / ``up_proj`` / ``down_proj`` names.
_HF_MOE_RE = re.compile(
    r"^(model\.layers\.\d+)\."
    r"block_sparse_moe\.experts\.(\d+)\."
    r"(w1|w2|w3)\.weight$"
)

_PROJ_MAP = {"w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"}

# Reverse: native key → HF block_sparse_moe key
_NATIVE_MOE_RE = re.compile(
    r"^(model\.layers\.\d+)\."
    r"mlp\.(experts\.(?:gate_and_up_projs|down_projs)|gate\.weight)$"
)

_HF_GATE_RE = re.compile(r"^(model\.layers\.\d+)\.block_sparse_moe\.gate\.weight$")


class MixtralStateDictAdapter(MoESplitExpertsStateDictMixin, StateDictAdapter):
    """Converts between HF Mixtral checkpoints and native grouped-expert format."""

    def __init__(
        self,
        config: Any,
        moe_config: MoEConfig,
        backend: BackendConfig,
        dtype: torch.dtype = torch.float32,
    ):
        self.config = config
        self.moe_config = moe_config
        self.backend = backend
        self.dtype = dtype
        self._uses_model_prefix = True

    # ------------------------------------------------------------------
    # native → HF
    # ------------------------------------------------------------------
    def to_hf(
        self,
        state_dict: dict[str, Any],
        exclude_key_regex: Optional[str] = None,
        quantization: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        hf_state_dict: dict[str, Any] = {}
        for fqn, tensor in state_dict.items():
            for key, value in self.convert_single_tensor_to_hf(
                fqn, tensor, exclude_key_regex=exclude_key_regex, quantization=quantization, **kwargs
            ):
                hf_state_dict[key] = value
        return hf_state_dict

    def convert_single_tensor_to_hf(self, fqn: str, tensor: Any, **kwargs) -> list[tuple[str, Any]]:
        exclude_key_regex = kwargs.get("exclude_key_regex", None)

        expert_result = self._convert_single_merged_expert_to_hf_split_experts(fqn, tensor, **kwargs)
        if expert_result is not None:
            result = expert_result
        else:
            # Rename native MLP gate path to HF block_sparse_moe path
            native_gate = fqn.replace(".mlp.gate.weight", ".block_sparse_moe.gate.weight")
            if native_gate != fqn:
                result = [(native_gate, tensor)]
            else:
                result = [(fqn, tensor)]

        if exclude_key_regex:
            result = [(k, v) for k, v in result if not re.match(exclude_key_regex, k)]

        return result

    # ------------------------------------------------------------------
    # HF → native
    # ------------------------------------------------------------------
    def from_hf(
        self,
        hf_state_dict: dict[str, Any],
        device_mesh: Optional[DeviceMesh] = None,
        **kwargs,
    ) -> dict[str, Any]:
        # Detect model prefix
        for key in hf_state_dict:
            if ".block_sparse_moe.experts." in key and key.endswith(".weight"):
                self._uses_model_prefix = key.startswith("model.")
                break

        # Rename HF Mixtral expert keys (w1/w2/w3) to the canonical
        # gate_proj/up_proj/down_proj names expected by the mixin.
        # Also rename ``block_sparse_moe`` → ``mlp``.
        remapped: dict[str, Any] = {}
        for key, value in hf_state_dict.items():
            m = _HF_MOE_RE.match(key)
            if m:
                layer_prefix, expert_id, proj = m.groups()
                canonical = _PROJ_MAP[proj]
                new_key = f"{layer_prefix}.mlp.experts.{expert_id}.{canonical}.weight"
                remapped[new_key] = value
                continue

            gm = _HF_GATE_RE.match(key)
            if gm:
                layer_prefix = gm.group(1)
                remapped[f"{layer_prefix}.mlp.gate.weight"] = value
                continue

            remapped[key] = value

        return self._from_hf_w_merged_experts(remapped, device_mesh)
