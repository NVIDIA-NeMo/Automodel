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

"""State dict conversion between the on-disk tencent/Hy3-preview HF checkpoint
and Automodel's native (grouped-experts) format.

On-disk HF format (what tencent/Hy3-preview safetensors actually contain):
  model.layers.{L}.mlp.expert_bias                                       # [n_experts]
  model.layers.{L}.mlp.router.gate.weight                                # [n_experts, hidden]
  model.layers.{L}.mlp.experts.{E}.gate_proj.weight                      # [moe_inter, hidden]
  model.layers.{L}.mlp.experts.{E}.up_proj.weight                        # [moe_inter, hidden]
  model.layers.{L}.mlp.experts.{E}.down_proj.weight                      # [hidden, moe_inter]
  model.layers.{L}.mlp.shared_mlp.{gate,up,down}_proj.weight             # [moe_inter, hidden] / [hidden, moe_inter]

Automodel native format (matches the rest of the MoE stack):
  model.layers.{L}.mlp.gate.e_score_correction_bias                      # [n_local]  (on Gate, not MoE)
  model.layers.{L}.mlp.gate.weight                                       # [n_experts, hidden]
  model.layers.{L}.mlp.experts.gate_and_up_projs                         # [n_local, hidden, 2*moe_inter]
  model.layers.{L}.mlp.experts.down_projs                                # [n_local, moe_inter, hidden]
  model.layers.{L}.mlp.shared_experts.{gate,up,down}_proj.weight         # unchanged shapes

Differences (vs. every other Automodel MoE adapter):
  1. Per-expert split tensors -> grouped (handled by MoESplitExpertsStateDictMixin).
  2. Three HYV3-specific name renames: expert_bias <-> gate.e_score_correction_bias,
     router.gate.weight <-> gate.weight, shared_mlp.* <-> shared_experts.*.
  3. MTP layers (indices >= num_hidden_layers) on disk must be filtered out on load.

Why the renames live in the adapter rather than in the storage reader's key_mapping:
nemo_automodel/components/checkpoint/checkpointing.py:507 deliberately passes
``reader_key_mapping=None`` when a model has a state_dict_adapter (to avoid
double-translation). So the adapter's ``to_hf`` / ``from_hf`` must produce keys
that match the actual on-disk strings.
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


# Pre-compiled HYV3-specific name renames (anchored to end-of-key for safety).
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


class HYV3StateDictAdapter(MoESplitExpertsStateDictMixin, StateDictAdapter):
    """Bridges Automodel native (grouped experts) and tencent/Hy3-preview on-disk HF.

    Inherits the per-expert split/merge logic from ``MoESplitExpertsStateDictMixin``;
    only the three HYV3-specific name renames + MTP-layer filtering live here.
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
    # Native -> on-disk HF
    # ------------------------------------------------------------------

    def to_hf(
        self,
        state_dict: dict[str, Any],
        exclude_key_regex: Optional[str] = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Convert native state dict back to the on-disk Tencent format.

        Steps:
          1. Split grouped expert tensors into per-expert HF keys (mixin).
          2. Apply HYV3 name renames (gate.e_score_correction_bias -> expert_bias,
             gate.weight -> router.gate.weight, shared_experts. -> shared_mlp.).
        """
        # Step 1: per-expert split via the mixin. Pass-through for non-expert keys.
        hf_split: dict[str, Any] = self._to_hf_w_split_experts(state_dict)

        # Step 2: rename native -> on-disk Tencent.
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

    # ------------------------------------------------------------------
    # On-disk HF -> native
    # ------------------------------------------------------------------

    def from_hf(
        self,
        hf_state_dict: dict[str, Any],
        device_mesh: Optional[DeviceMesh] = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Convert the on-disk Tencent state dict to native format.

        Steps:
          1. Drop MTP (multi-token prediction) layer keys.
          2. Apply HYV3 name renames (on-disk -> native HF naming).
          3. Recover the EP device_mesh if the caller passed None: the standard
             DCP path (checkpointing.py) calls us with ``moe_mesh=None`` on the
             assumption that the state dict is already EP-local, but the mixin's
             validator/merger needs the mesh to know which expert subset to
             expect. Any per-expert tensor that survived ``to_hf`` is a DTensor
             we can introspect for the mesh.
          4. Merge per-expert split tensors into grouped form via the mixin.
        """
        # Step 1 + 2: filter MTP, then rename in a single pass to avoid copying twice.
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

        # Step 3: recover device_mesh from any expert-shard DTensor in the dict.
        if device_mesh is None:
            device_mesh = self._infer_ep_mesh(renamed)

        # Step 4: per-expert merge + EP slicing via the mixin.
        return self._from_hf_w_merged_experts(renamed, device_mesh)

    # ------------------------------------------------------------------
    # Single-tensor variant required by the abstract base class.
    # ------------------------------------------------------------------

    def convert_single_tensor_to_hf(
        self,
        fqn: str,
        tensor: Any,
        **kwargs,
    ) -> list[tuple[str, Any]]:
        """Per-tensor variant of ``to_hf`` (used by save paths that stream tensors).

        Mirrors ``to_hf`` but operating on one (fqn, tensor) at a time:
          1. Try the mixin's per-expert split. Returns multiple (key, tensor) pairs
             when *fqn* names a grouped expert tensor; otherwise returns ``None``.
          2. Apply HYV3 name renames to whichever key set we end up with.
        """
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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_mtp_key(self, key: str) -> bool:
        """Return True if *key* belongs to an MTP layer (index >= num_hidden_layers)."""
        num_hidden = getattr(self.config, "num_hidden_layers", 80)
        m = re.match(r"(?:model\.)?layers\.(\d+)\.", key)
        return bool(m and int(m.group(1)) >= num_hidden)

    def _infer_ep_mesh(self, state_dict: dict[str, Any]) -> Optional[DeviceMesh]:
        """Return the EP device mesh inferred from any per-expert DTensor in *state_dict*.

        The standard DCP load path produces per-expert split DTensors (via
        ``to_hf``) whose ``device_mesh`` carries the ``ep`` dimension we need
        to drive ``_from_hf_w_merged_experts``. We probe the dict for any
        DTensor expert tensor and reuse its mesh.
        """
        from torch.distributed.tensor import DTensor

        expert_segment = self._expert_path_segment
        for k, v in state_dict.items():
            if not isinstance(v, DTensor):
                continue
            if f".{expert_segment}." not in k:
                continue
            mesh = v.device_mesh
            if mesh is not None and "ep" in (mesh.mesh_dim_names or ()):
                return mesh
        return None
