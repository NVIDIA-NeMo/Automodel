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

"""State dict conversion between the on-disk tencent/Hy-MT2-30B-A3B HF
checkpoint and Automodel's native (grouped-experts) format.

The on-disk key layout is identical to tencent/Hy3-preview because both
share ``model_type: "hy_v3"`` and ``architectures: ["HYV3ForCausalLM"]``:

  model.layers.{L}.mlp.expert_bias                                       # [n_experts]
  model.layers.{L}.mlp.router.gate.weight                                # [n_experts, hidden]
  model.layers.{L}.mlp.experts.{E}.gate_proj.weight                      # [moe_inter, hidden]
  model.layers.{L}.mlp.experts.{E}.up_proj.weight                        # [moe_inter, hidden]
  model.layers.{L}.mlp.experts.{E}.down_proj.weight                      # [hidden, moe_inter]
  model.layers.{L}.mlp.shared_mlp.{gate,up,down}_proj.weight             # shared expert

Automodel native:

  model.layers.{L}.mlp.gate.e_score_correction_bias                      # [n_local]
  model.layers.{L}.mlp.gate.weight                                       # [n_experts, hidden]
  model.layers.{L}.mlp.experts.gate_and_up_projs                         # grouped
  model.layers.{L}.mlp.experts.down_projs                                # grouped
  model.layers.{L}.mlp.shared_experts.{gate,up,down}_proj.weight

This adapter handles three on-disk-specific renames plus per-expert
split/merge (via ``MoESplitExpertsStateDictMixin``). It is functionally a
clone of ``HYV3StateDictAdapter``; kept separate so future Hy-MT2-only
key changes (e.g. an MTP / aux-head extension that Hy-MT2 ships but
Hy3-preview does not) can be added here without affecting Hy3-preview.
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


_NATIVE_TO_HF_RENAMES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\.mlp\.gate\.e_score_correction_bias$"), ".mlp.expert_bias"),
    (re.compile(r"\.mlp\.gate\.weight$"), ".mlp.router.gate.weight"),
    (re.compile(r"\.mlp\.shared_experts\."), ".mlp.shared_mlp."),
)
_HF_TO_NATIVE_RENAMES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\.mlp\.expert_bias$"), ".mlp.gate.e_score_correction_bias"),
    (re.compile(r"\.mlp\.router\.gate\.weight$"), ".mlp.gate.weight"),
    (re.compile(r"\.mlp\.shared_mlp\."), ".mlp.shared_experts."),
)


class HyMT2StateDictAdapter(MoESplitExpertsStateDictMixin, StateDictAdapter):
    """Bridges Automodel native (grouped experts) and on-disk Hy-MT2 HF format."""

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

    def to_hf(
        self,
        state_dict: dict[str, Any],
        exclude_key_regex: Optional[str] = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Native -> on-disk Hy-MT2 HF: per-expert split + name renames."""
        hf_split: dict[str, Any] = self._to_hf_w_split_experts(state_dict)

        out: dict[str, Any] = {}
        for k, v in hf_split.items():
            new_k = k
            for pat, repl in _NATIVE_TO_HF_RENAMES:
                new_k, n = pat.subn(repl, new_k)
                if n:
                    break
            if exclude_key_regex and re.match(exclude_key_regex, new_k):
                continue
            out[new_k] = v
        return out

    def from_hf(
        self,
        hf_state_dict: dict[str, Any],
        device_mesh: Optional[DeviceMesh] = None,
        **kwargs,
    ) -> dict[str, Any]:
        """On-disk Hy-MT2 HF -> native: filter MTP, rename, then merge experts."""
        renamed: dict[str, Any] = {}
        for k, v in hf_state_dict.items():
            if self._is_mtp_key(k):
                continue
            new_k = k
            for pat, repl in _HF_TO_NATIVE_RENAMES:
                new_k, n = pat.subn(repl, new_k)
                if n:
                    break
            renamed[new_k] = v

        return self._from_hf_w_merged_experts(renamed, device_mesh)

    def convert_single_tensor_to_hf(
        self,
        fqn: str,
        tensor: Any,
        **kwargs,
    ) -> list[tuple[str, Any]]:
        """Per-tensor variant of ``to_hf`` for streaming-save code paths."""
        exclude_key_regex = kwargs.get("exclude_key_regex", None)

        expert_split = self._convert_single_merged_expert_to_hf_split_experts(fqn, tensor, **kwargs)
        if expert_split is not None:
            pairs = expert_split
        else:
            pairs = [(fqn, tensor)]

        out: list[tuple[str, Any]] = []
        for k, v in pairs:
            new_k = k
            for pat, repl in _NATIVE_TO_HF_RENAMES:
                new_k, n = pat.subn(repl, new_k)
                if n:
                    break
            if exclude_key_regex and re.match(exclude_key_regex, new_k):
                continue
            out.append((new_k, v))
        return out

    def _is_mtp_key(self, key: str) -> bool:
        """Return True if *key* belongs to an MTP layer (index >= num_hidden_layers).

        Hy-MT2-30B-A3B does not appear to ship MTP layers in its public
        checkpoint, but the filter is kept as a defensive no-op so the
        adapter remains symmetric with ``HYV3StateDictAdapter``.
        """
        num_hidden = getattr(self.config, "num_hidden_layers", 48)
        m = re.match(r"(?:model\.)?layers\.(\d+)\.", key)
        return bool(m and int(m.group(1)) >= num_hidden)
