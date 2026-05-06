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


class NemotronV3StateDictAdapter(MoESplitExpertsStateDictMixin, StateDictAdapter):
    """State dict adapter for NemotronV3 models.

    Converts between HuggingFace checkpoint format and internal NeMo format.

    HF format uses 'backbone' prefix:
        - backbone.embed_tokens.weight
        - backbone.layers.{}.norm.weight
        - backbone.layers.{}.mixer.* (mamba/attention/moe components)
        - backbone.norm_f.weight
        - lm_head.weight

    Internal format uses 'model' prefix:
        - model.embed_tokens.weight
        - model.layers.{}.norm.weight
        - model.layers.{}.mixer.* (mamba/attention/moe components)
        - model.norm.weight
        - lm_head.weight

    For MoE layers:
        - HF: Split per-expert weights (experts.{}.up_proj.weight, experts.{}.down_proj.weight)
        - Internal: Merged expert weights (experts.gate_and_up_projs, experts.down_projs)

    NemotronV3 uses ReLU² activation (non-gated), so gate_and_up_projs has
    shape [n_experts, dim, inter_dim] instead of [n_experts, dim, 2*inter_dim].

    Note: NemotronV3 uses 'mixer' instead of 'mlp' in layer paths.
    """

    def __init__(
        self,
        config,
        moe_config: MoEConfig,
        backend: BackendConfig,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.config = config
        self.moe_config = moe_config
        self.backend = backend
        self.dtype = dtype
        self._uses_model_prefix = True

        # Mapping for expert weights (HF split → internal merged)
        self.from_hf_map = {
            "model.layers.{}.mixer.experts.{}.up_proj.weight": "model.layers.{}.mixer.experts.gate_and_up_projs",
            "model.layers.{}.mixer.experts.{}.down_proj.weight": "model.layers.{}.mixer.experts.down_projs",
        }

    @property
    def _hf_prefix(self) -> str:
        """NemotronV3 HF format uses 'backbone.' prefix."""
        return "backbone."

    @property
    def _expert_path_segment(self) -> str:
        """NemotronV3 uses 'mixer.experts' instead of 'mlp.experts'."""
        return "mixer.experts"

    def to_hf(self, state_dict: dict[str, Any], exclude_key_regex: Optional[str] = None, **kwargs) -> dict[str, Any]:
        """Convert from internal model state dict to HuggingFace format.

        Args:
            state_dict: Internal format state dict
            exclude_key_regex: Optional regex pattern to exclude keys
            **kwargs: Additional arguments

        Returns:
            HuggingFace format state dict
        """
        hf_state_dict = {}
        for fqn in list(state_dict.keys()):
            tensor = state_dict.pop(fqn)
            converted_tensors = self.convert_single_tensor_to_hf(
                fqn, tensor, exclude_key_regex=exclude_key_regex, **kwargs
            )
            for key, value in converted_tensors:
                hf_state_dict[key] = value

        return hf_state_dict

    def from_hf(
        self,
        hf_state_dict: dict[str, Any],
        device_mesh: Optional["DeviceMesh"] = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Convert HF checkpoint to internal format.

        - Rename backbone → model
        - Rename norm_f → norm
        - Aggregate per-expert weights into grouped tensors
        - If device_mesh is provided, only load experts needed for the current rank
        - Process MTP keys (``mtp.layers.{i}.*``) separately, reusing the
          same MoE expert-merge logic for the MoE sublayer of each MTP depth.

        Args:
            hf_state_dict: HuggingFace format state dict
            device_mesh: Optional device mesh for distributed expert loading
            **kwargs: Additional arguments

        Returns:
            Internal format state dict
        """
        # Separate MTP keys; they live in their own top-level namespace and
        # are not subject to the backbone/model rename.
        mtp_state_dict: dict[str, Any] = {}
        backbone_state_dict: dict[str, Any] = {}
        for key in list(hf_state_dict.keys()):
            value = hf_state_dict.pop(key)
            if key.startswith("mtp."):
                mtp_state_dict[key] = value
            else:
                backbone_state_dict[key] = value

        # Detect if HF checkpoint uses 'backbone' or 'model' prefix. Only
        # look at backbone keys; MTP keys never carry a backbone/model prefix.
        for key in backbone_state_dict.keys():
            if ".mixer.experts." in key:
                self._uses_model_prefix = not key.startswith("backbone.")
                break

        # First, rename backbone → model and norm_f → norm
        renamed_state_dict = {}
        for key in list(backbone_state_dict.keys()):
            value = backbone_state_dict.pop(key)
            new_key = key
            if new_key.startswith("backbone."):
                new_key = "model." + new_key[len("backbone.") :]
            if new_key == "model.norm_f.weight":
                new_key = "model.norm.weight"
            # HF uses 'embeddings' but internal uses 'embed_tokens'
            if new_key == "model.embeddings.weight":
                new_key = "model.embed_tokens.weight"

            renamed_state_dict[new_key] = value

        # Then merge experts using the mixin method
        merged = self._from_hf_w_merged_experts(renamed_state_dict, device_mesh)

        # Process MTP keys: strip the ``mtp.`` namespace, reuse the standard
        # expert-merge logic, then re-add ``mtp.`` to the merged output keys.
        # ``apply_ep`` (``components/moe/parallelizer.py:76``) iterates both
        # ``model.backbone.layers`` and ``model.mtp.layers``, so MTP MoE
        # experts are EP-sharded the same way as backbone experts and the
        # standard ``device_mesh``-aware merge produces matching DTensors.
        if mtp_state_dict:
            stripped: dict[str, Any] = {}
            for key, value in mtp_state_dict.items():
                stripped_key = key[len("mtp.") :] if key.startswith("mtp.") else key
                stripped[stripped_key] = value
            merged_mtp = self._from_hf_w_merged_experts(stripped, device_mesh)
            for key, value in merged_mtp.items():
                merged[f"mtp.{key}"] = value

        return merged

    def convert_single_tensor_to_hf(self, fqn: str, tensor: Any, **kwargs) -> list[tuple[str, Any]]:
        """Convert a single tensor from internal format to HuggingFace format.

        Args:
            fqn: Fully qualified name of the tensor in internal format
            tensor: The tensor to convert
            **kwargs: Additional arguments for conversion

        Returns:
            List of (fqn, tensor) tuples in HuggingFace format
        """
        exclude_key_regex = kwargs.get("exclude_key_regex", None)

        # MTP keys live in their own ``mtp.*`` namespace. Merged-experts
        # tensors are split with prefix ``mtp.`` rather than ``backbone.``;
        # all other MTP keys pass through unchanged.
        if fqn.startswith("mtp."):
            expert_split = self._mtp_convert_merged_expert_to_hf_split(fqn, tensor)
            if expert_split is not None:
                result = expert_split
            else:
                result = [(fqn, tensor)]
            if exclude_key_regex:
                result = [(k, v) for k, v in result if not re.match(exclude_key_regex, k)]
            return result

        # Try to convert merged expert weights to split experts
        expert_result = self._convert_single_merged_expert_to_hf_split_experts(fqn, tensor, **kwargs)
        if expert_result is not None:
            result = expert_result
        else:
            # Standard conversion: just rename keys
            new_fqn = fqn

            # Rename model → backbone
            if new_fqn.startswith("model."):
                new_fqn = "backbone." + new_fqn[len("model.") :]

            # Rename norm → norm_f
            if new_fqn == "backbone.norm.weight":
                new_fqn = "backbone.norm_f.weight"

            # Internal uses 'embed_tokens' but HF uses 'embeddings'
            if new_fqn == "backbone.embed_tokens.weight":
                new_fqn = "backbone.embeddings.weight"

            result = [(new_fqn, tensor)]

        if exclude_key_regex:
            result = [(k, v) for k, v in result if not re.match(exclude_key_regex, k)]

        return result

    def _mtp_convert_merged_expert_to_hf_split(self, fqn: str, tensor: Any) -> Optional[list[tuple[str, Any]]]:
        """Split a merged MTP MoE expert tensor into per-expert HF keys.

        Mirrors :meth:`MoESplitExpertsStateDictMixin._convert_single_merged_expert_to_hf_split_experts`
        but emits keys under the ``mtp.`` namespace instead of ``backbone.``.
        Returns ``None`` if ``fqn`` does not refer to a merged MTP expert
        tensor.

        Sharding note: backbone MoE experts are EP-sharded by ``apply_ep``
        (``components/moe/parallelizer.py:76``), which only iterates
        ``model.backbone.layers``. MTP MoE experts are never visited by EP,
        so their tensors arrive here as plain tensors or as DTensors sharded
        only along DP. ``_split_experts_weights`` requires either a plain
        tensor or a DTensor with ``ep`` mesh dim, so we materialize
        non-EP-sharded DTensors via ``full_tensor()`` first.
        """
        from nemo_automodel.components.moe.state_dict_utils import is_dtensor

        expert_segment = self._expert_path_segment
        n_experts = self.moe_config.n_routed_experts
        inter_dim = self.moe_config.moe_inter_dim

        def _materialize(t):
            if not is_dtensor(t):
                return t
            mesh = t.device_mesh
            if "ep" in mesh.mesh_dim_names:
                return t
            return t.full_tensor()

        if f".{expert_segment}.gate_and_up_projs" in fqn and fqn.endswith(".gate_and_up_projs"):
            layer_num = re.search(r"layers\.(\d+)", fqn).group(1)
            tensor = _materialize(tensor)
            splits = self._split_experts_weights(tensor, n_experts)
            result: list[tuple[str, Any]] = []
            for i, w in enumerate(splits):
                expert_id = self._last_expert_ids[i]
                base = f"mtp.layers.{layer_num}.{expert_segment}.{expert_id}"
                if self._is_gated_moe:
                    w_gate = w[:, :inter_dim].transpose(0, 1).contiguous()
                    w_up = w[:, inter_dim:].transpose(0, 1).contiguous()
                    result.append((f"{base}.gate_proj.weight", w_gate))
                    result.append((f"{base}.up_proj.weight", w_up))
                else:
                    w_up = w.transpose(0, 1).contiguous()
                    result.append((f"{base}.up_proj.weight", w_up))
            return result

        if (
            f".{expert_segment}.down_projs" in fqn
            and fqn.endswith(".down_projs")
            and tensor.ndim == 3
            and tensor.shape[1] == inter_dim
        ):
            layer_num = re.search(r"layers\.(\d+)", fqn).group(1)
            tensor = _materialize(tensor)
            splits = self._split_experts_weights(tensor, n_experts)
            result = []
            for i, w in enumerate(splits):
                expert_id = self._last_expert_ids[i]
                result.append(
                    (
                        f"mtp.layers.{layer_num}.{expert_segment}.{expert_id}.down_proj.weight",
                        w.transpose(0, 1).contiguous(),
                    )
                )
            return result

        return None
