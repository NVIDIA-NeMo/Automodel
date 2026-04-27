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

"""State dict conversion between HuggingFace HYV3 format and Automodel native format.

Key differences between HF and native formats:

HF format (tencent/Hy3-preview):
  model.layers.{L}.mlp.experts.gate_up_proj          # [n_experts, 2*moe_inter, hidden]
  model.layers.{L}.mlp.experts.down_proj             # [n_experts, hidden, moe_inter]
  model.layers.{L}.mlp.e_score_correction_bias        # [n_experts]  (on MoE module)
  model.layers.{L}.mlp.gate.weight                   # [n_experts, hidden]
  model.layers.{L}.mlp.shared_experts.{gate,up,down}_proj.weight  # separate MLPs

Native format (Automodel GroupedExperts):
  model.layers.{L}.mlp.experts.gate_and_up_projs     # [n_local, hidden, 2*moe_inter]  (transposed)
  model.layers.{L}.mlp.experts.down_projs            # [n_local, moe_inter, hidden]    (transposed)
  model.layers.{L}.mlp.gate.e_score_correction_bias  # [n_local]  (moved to Gate module)
  model.layers.{L}.mlp.gate.weight                   # [n_experts, hidden]  (unchanged)
  model.layers.{L}.mlp.shared_experts.{gate,up,down}_proj.weight  # unchanged

All other keys (attention, norms, embeddings, lm_head) are identical between formats.
Expert parallelism: in from_hf each rank loads only its slice [start:end] of the
first (expert) dimension of gate_up_proj / down_proj.
"""

import logging
import re
from typing import Any, Optional

import torch
from torch.distributed.device_mesh import DeviceMesh

from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.state_dict_utils import get_expert_range_for_rank_from_mesh

logger = logging.getLogger(__name__)

# Regex patterns used for key detection
_EXPERTS_GATE_UP = re.compile(r"(.*\.mlp)\.experts\.gate_up_proj$")
_EXPERTS_DOWN = re.compile(r"(.*\.mlp)\.experts\.down_proj$")
_E_SCORE_BIAS_HF = re.compile(r"(.*\.mlp)\.e_score_correction_bias$")

_EXPERTS_GATE_AND_UP = re.compile(r"(.*\.mlp)\.experts\.gate_and_up_projs$")
_EXPERTS_DOWN_NATIVE = re.compile(r"(.*\.mlp)\.experts\.down_projs$")
_E_SCORE_BIAS_NATIVE = re.compile(r"(.*\.mlp)\.gate\.e_score_correction_bias$")


class HYV3StateDictAdapter(StateDictAdapter):
    """Converts between HF HYV3 checkpoints and Automodel native format.

    The HF format uses pre-grouped (batched) expert tensors but with the last
    two dimensions transposed relative to Automodel's GroupedExperts convention.
    The e_score_correction_bias lives on the MoE module in HF but on the Gate
    sub-module in native format.
    """

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

    # ------------------------------------------------------------------
    # HF → native
    # ------------------------------------------------------------------

    def from_hf(
        self,
        hf_state_dict: dict[str, Any],
        device_mesh: Optional[DeviceMesh] = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Convert HF checkpoint to native Automodel format.

        For expert parallelism, only the slice owned by the current rank is
        retained from the batched expert tensors.
        """
        n_experts = self.moe_config.n_routed_experts

        if device_mesh is not None:
            start_expert, end_expert = get_expert_range_for_rank_from_mesh(device_mesh, n_experts)
        else:
            start_expert, end_expert = 0, n_experts

        native: dict[str, Any] = {}

        for key, value in hf_state_dict.items():
            # Skip MTP (multi-token prediction) layers — not used during SFT.
            # These appear as layers beyond num_hidden_layers in the checkpoint.
            if self._is_mtp_key(key):
                continue

            m = _EXPERTS_GATE_UP.match(key)
            if m:
                # HF: [n_experts, 2*moe_inter, hidden] → native: [n_local, hidden, 2*moe_inter]
                sliced = value[start_expert:end_expert]
                native_key = m.group(1) + ".experts.gate_and_up_projs"
                native[native_key] = sliced.transpose(1, 2).contiguous()
                continue

            m = _EXPERTS_DOWN.match(key)
            if m:
                # HF: [n_experts, hidden, moe_inter] → native: [n_local, moe_inter, hidden]
                sliced = value[start_expert:end_expert]
                native_key = m.group(1) + ".experts.down_projs"
                native[native_key] = sliced.transpose(1, 2).contiguous()
                continue

            m = _E_SCORE_BIAS_HF.match(key)
            if m:
                # Move from MoE module level to Gate sub-module.
                native_key = m.group(1) + ".gate.e_score_correction_bias"
                native[native_key] = value
                continue

            # All other keys (attn, norms, dense mlp, shared experts, lm_head) pass through.
            native[key] = value

        return native

    # ------------------------------------------------------------------
    # native → HF
    # ------------------------------------------------------------------

    def to_hf(
        self,
        state_dict: dict[str, Any],
        exclude_key_regex: Optional[str] = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Convert native Automodel state dict back to HF format."""
        hf: dict[str, Any] = {}

        for key, value in state_dict.items():
            converted_pairs = self.convert_single_tensor_to_hf(key, value, exclude_key_regex=exclude_key_regex)
            for k, v in converted_pairs:
                hf[k] = v

        return hf

    def convert_single_tensor_to_hf(self, fqn: str, tensor: Any, **kwargs) -> list[tuple[str, Any]]:
        """Convert a single native tensor to its HF equivalent."""
        exclude_key_regex = kwargs.get("exclude_key_regex", None)

        m = _EXPERTS_GATE_AND_UP.match(fqn)
        if m:
            # native: [n_local, hidden, 2*moe_inter] → HF: [n_local, 2*moe_inter, hidden]
            hf_key = m.group(1) + ".experts.gate_up_proj"
            result = [(hf_key, tensor.transpose(1, 2).contiguous())]
        else:
            m = _EXPERTS_DOWN_NATIVE.match(fqn)
            if m:
                # native: [n_local, moe_inter, hidden] → HF: [n_local, hidden, moe_inter]
                hf_key = m.group(1) + ".experts.down_proj"
                result = [(hf_key, tensor.transpose(1, 2).contiguous())]
            else:
                m = _E_SCORE_BIAS_NATIVE.match(fqn)
                if m:
                    hf_key = m.group(1) + ".e_score_correction_bias"
                    result = [(hf_key, tensor)]
                else:
                    result = [(fqn, tensor)]

        if exclude_key_regex:
            result = [(k, v) for k, v in result if not re.match(exclude_key_regex, k)]

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_mtp_key(self, key: str) -> bool:
        """Return True if key belongs to a multi-token prediction (MTP) layer.

        HYV3 checkpoints include num_nextn_predict_layers extra layers beyond
        the main num_hidden_layers. These are stored at layer indices >= num_hidden_layers
        and are not used during standard SFT.
        """
        num_hidden = getattr(self.config, "num_hidden_layers", 80)
        m = re.match(r"(?:model\.)?layers\.(\d+)\.", key)
        if m and int(m.group(1)) >= num_hidden:
            return True
        return False
