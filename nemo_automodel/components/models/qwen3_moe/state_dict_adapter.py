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
from nemo_automodel.components.moe import state_dict_utils
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.state_dict_mixin import MoESplitExpertsStateDictMixin

logger = logging.getLogger(__name__)

# Native LoRA suffixes for grouped MoE expert tensors
_LORA_EXPERT_SUFFIXES = ("lora_gate_and_up_A", "lora_gate_and_up_B", "lora_down_A", "lora_down_B")


class Qwen3MoeStateDictAdapter(MoESplitExpertsStateDictMixin, StateDictAdapter):
    """Converts between HF Qwen3-MoE checkpoints and our grouped-experts native format.

    Qwen3-MoE HF experts use grouped keys:
      model.layers.{L}.mlp.experts.gate_up_proj  # [n_experts, 2*moe_inter_dim, dim]
      model.layers.{L}.mlp.experts.down_proj     # [n_experts, dim, moe_inter_dim]

    Our native format groups them into:
      model.layers.{L}.mlp.experts.gate_and_up_projs  # [n_experts, dim, 2*moe_inter_dim]
      model.layers.{L}.mlp.experts.down_projs         # [n_experts, moe_inter_dim, dim]

    Some checkpoints, including the source checkpoint for
    ``Qwen/Qwen3-30B-A3B``, store split per-expert tensors.  The load path
    selects that conversion from checkpoint metadata.  The non-legacy save path
    emits grouped HF expert tensors that the installed Qwen3-MoE implementation
    can reload directly.
    """

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

    def to_hf(
        self, state_dict: dict[str, Any], exclude_key_regex: Optional[str] = None, quantization: bool = False, **kwargs
    ) -> dict[str, Any]:
        hf_state_dict = {}
        for fqn, tensor in state_dict.items():
            converted_tensors = self.convert_single_tensor_to_hf(
                fqn, tensor, exclude_key_regex=exclude_key_regex, quantization=quantization, **kwargs
            )
            for key, value in converted_tensors:
                hf_state_dict[key] = value

        return hf_state_dict

    def convert_single_tensor_to_hf(self, fqn: str, tensor: Any, **kwargs) -> list[tuple[str, Any]]:
        """Convert a single tensor from native format to HuggingFace format.

        Loads whose checkpoint metadata uses split expert keys and
        ``v4_compatible=True`` saves use split per-expert keys via the parent
        mixin. Non-legacy saves emit full expert weights as grouped HF tensors
        and LoRA expert tensors in PEFT ParamWrapper format so that
        ``PeftModel.from_pretrained()`` can load them directly.

        Args:
            fqn: Fully qualified name of the tensor in native format
            tensor: The tensor to convert
            **kwargs: Additional arguments for conversion

        Returns:
            List of (fqn, tensor) tuples in HuggingFace format
        """
        exclude_key_regex = kwargs.get("exclude_key_regex", None)
        source_checkpoint_uses_split_experts = kwargs.get("source_checkpoint", False)
        v4_compatible = kwargs.get("v4_compatible", False)

        # Check if this is a LoRA expert tensor eligible for ParamWrapper conversion
        if not source_checkpoint_uses_split_experts and not v4_compatible:
            expert_segment = self._expert_path_segment
            for suffix in _LORA_EXPERT_SUFFIXES:
                if fqn.endswith(f".{suffix}") and f".{expert_segment}.{suffix}" in fqn:
                    result = self._convert_lora_to_paramwrapper(fqn, tensor)
                    if exclude_key_regex:
                        result = [(k, v) for k, v in result if not re.match(exclude_key_regex, k)]
                    return result

            grouped_result = self._convert_grouped_expert_to_hf(fqn, tensor)
            if grouped_result is not None:
                result = grouped_result
                if exclude_key_regex:
                    result = [(k, v) for k, v in result if not re.match(exclude_key_regex, k)]
                return result

        # Source loads, legacy split-export mode, or non-expert keys: fall through to parent mixin.
        expert_result = self._convert_single_merged_expert_to_hf_split_experts(fqn, tensor, **kwargs)
        if expert_result is not None:
            result = expert_result
        else:
            result = [(fqn, tensor)]

        if exclude_key_regex:
            result = [(k, v) for k, v in result if not re.match(exclude_key_regex, k)]

        return result

    def _convert_grouped_expert_to_hf(self, fqn: str, tensor: Any) -> list[tuple[str, Any]] | None:
        """Convert native grouped expert tensors to grouped HF expert keys."""
        expert_segment = self._expert_path_segment
        if f".{expert_segment}.gate_and_up_projs" in fqn and fqn.endswith(".gate_and_up_projs"):
            return [(fqn.replace(".gate_and_up_projs", ".gate_up_proj"), tensor.transpose(1, 2))]
        if f".{expert_segment}.down_projs" in fqn and fqn.endswith(".down_projs"):
            return [(fqn.replace(".down_projs", ".down_proj"), tensor.transpose(1, 2))]
        return None

    def _convert_lora_to_paramwrapper(self, fqn: str, tensor: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
        """Convert a single grouped MoE LoRA tensor to PEFT ParamWrapper format.

        ParamWrapper format stores fused 3-D expert LoRA parameters as 2-D
        tensors with the expert dimension folded into the rank dimension.

        Shape mapping (automodel native -> ParamWrapper):

        down_proj (outer wrapper, NO ``base_layer`` prefix — processed first alphabetically):
          - ``lora_down_B``  (E, r, H) -> ``lora_A.weight``  (r*E, H)  reshape
          - ``lora_down_A``  (E, I, r) -> ``lora_B.weight``  (I, r*E)  permute+reshape

        gate_up_proj (inner wrapper, HAS ``base_layer.`` prefix):
          - ``lora_gate_and_up_B``  (E, r, 2*I) -> ``base_layer.lora_A.weight``  (r*E, 2*I)  reshape
          - ``lora_gate_and_up_A``  (E, H, r)   -> ``base_layer.lora_B.weight``  (H, r*E)    permute+reshape

        Returns:
            List containing one ``(fqn, tensor)`` tuple in ParamWrapper format.
        """
        match = re.search(r"(.*)layers\.(\d+)\.", fqn)
        if not match:
            return [(fqn, tensor)]

        prefix = match.group(1)
        layer_num = match.group(2)
        expert_segment = self._expert_path_segment
        suffix = fqn.rsplit(".", 1)[-1]

        # PEFT ParamWrapper nesting: target_parameters are sorted alphabetically
        # and wrapped in order. The FIRST wrapped becomes the OUTER ParamWrapper.
        # "down_proj" < "gate_up_proj", so down_proj is outer (no base_layer prefix)
        # and gate_up_proj is inner (has base_layer prefix).
        if suffix == "lora_gate_and_up_B":
            # (E, r, 2*I) -> (r*E, 2*I)
            out = tensor.reshape(-1, tensor.shape[2]).contiguous()
            pw_suffix = "base_layer.lora_A.weight"
        elif suffix == "lora_gate_and_up_A":
            # (E, H, r) -> permute(1,2,0) -> (H, r, E) -> (H, r*E)
            out = tensor.permute(1, 2, 0).contiguous().reshape(tensor.shape[1], -1)
            pw_suffix = "base_layer.lora_B.weight"
        elif suffix == "lora_down_B":
            # (E, r, H) -> (r*E, H)
            out = tensor.reshape(-1, tensor.shape[2]).contiguous()
            pw_suffix = "lora_A.weight"
        elif suffix == "lora_down_A":
            # (E, I, r) -> permute(1,2,0) -> (I, r, E) -> (I, r*E)
            out = tensor.permute(1, 2, 0).contiguous().reshape(tensor.shape[1], -1)
            pw_suffix = "lora_B.weight"
        else:
            return [(fqn, tensor)]

        out_fqn = f"{prefix}layers.{layer_num}.{expert_segment}.{pw_suffix}"
        return [(out_fqn, out)]

    def from_hf(
        self,
        hf_state_dict: dict[str, Any],
        device_mesh: Optional["DeviceMesh"] = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Convert HF checkpoint to native format, handling ParamWrapper LoRA keys.

        Before delegating to the parent ``_from_hf_w_merged_experts`` (which
        handles legacy per-expert LoRA format), this method scans for
        ParamWrapper-format LoRA keys and converts them back to the native
        grouped format expected by ``GroupedExpertsLoRA``.
        """
        # Detect whether HF checkpoints use the "model." prefix.
        for key in hf_state_dict.keys():
            if ".mlp.experts." in key and (
                key.endswith(".weight") or key.endswith(".gate_up_proj") or key.endswith(".down_proj")
            ):
                self._uses_model_prefix = key.startswith("model.")
                break

        # Convert any ParamWrapper-format LoRA keys to native grouped format
        hf_state_dict = self._convert_paramwrapper_to_native(hf_state_dict)

        grouped_state_dict, remaining_state_dict = self._convert_grouped_hf_experts_to_native(
            hf_state_dict, device_mesh
        )
        state_dict = self._from_hf_w_merged_experts(remaining_state_dict, device_mesh)
        state_dict.update(grouped_state_dict)
        return state_dict

    def _convert_grouped_hf_experts_to_native(
        self,
        hf_state_dict: dict[str, Any],
        device_mesh: Optional["DeviceMesh"] = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Convert grouped HF expert tensors to native keys."""
        n_experts = self.moe_config.n_routed_experts
        start_expert, end_expert, rank = 0, n_experts, None
        ep_shard_rank, ep_shard_size = 0, 1
        if device_mesh is not None:
            start_expert, end_expert = state_dict_utils.get_expert_range_for_rank_from_mesh(device_mesh, n_experts)
            rank = (
                state_dict_utils.get_submesh(device_mesh, ("ep",)).get_rank()
                if "ep" in device_mesh.mesh_dim_names
                else device_mesh.get_rank()
            )
            if "ep_shard" in device_mesh.mesh_dim_names:
                ep_shard_sub = state_dict_utils.get_submesh(device_mesh, ("ep_shard",))
                if ep_shard_sub.size() > 1:
                    ep_shard_rank = ep_shard_sub.get_local_rank()
                    ep_shard_size = ep_shard_sub.size()

        grouped_state_dict: dict[str, Any] = {}
        remaining_state_dict: dict[str, Any] = {}
        pattern = re.compile(
            r"(?P<prefix>(?:model\.)?)layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<which>gate_up_proj|down_proj)$"
        )

        for key, value in hf_state_dict.items():
            match = pattern.match(key)
            if match is None:
                remaining_state_dict[key] = value
                continue

            native_key = (
                f"{match.group('prefix') or ''}layers.{match.group('layer')}.mlp.experts."
                f"{'gate_and_up_projs' if match.group('which') == 'gate_up_proj' else 'down_projs'}"
            )
            if state_dict_utils.is_dtensor(value):
                grouped_state_dict[native_key] = value.transpose(1, 2)
            else:
                local_tensor = value[start_expert:end_expert].transpose(1, 2).to(self.dtype)
                if ep_shard_size > 1:
                    if local_tensor.shape[1] % ep_shard_size != 0:
                        raise ValueError(
                            f"{native_key} dim 1 ({local_tensor.shape[1]}) is not divisible by "
                            f"ep_shard_size={ep_shard_size}"
                        )
                    chunk = local_tensor.shape[1] // ep_shard_size
                    local_tensor = local_tensor[:, ep_shard_rank * chunk : (ep_shard_rank + 1) * chunk, :]
                grouped_state_dict[native_key] = state_dict_utils.create_dtensor_from_local(
                    local_tensor, device_mesh, rank
                )

        return grouped_state_dict, remaining_state_dict

    def _convert_paramwrapper_to_native(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert PEFT ParamWrapper LoRA keys to native grouped MoE LoRA format.

        This is the reverse of ``_convert_lora_to_paramwrapper``.  It detects
        ParamWrapper-format keys and converts them back to the 3-D grouped
        tensors expected by GroupedExpertsLoRA.

        Reverse transforms (down_proj is outer, gate_up_proj is inner):
          - ``experts.lora_A.weight``            (r*E, H)   -> (E, r, H)    = lora_down_B
          - ``experts.lora_B.weight``            (I, r*E)   -> (E, I, r)    = lora_down_A
          - ``experts.base_layer.lora_A.weight`` (r*E, 2*I) -> (E, r, 2*I)  = lora_gate_and_up_B
          - ``experts.base_layer.lora_B.weight`` (H, r*E)   -> (E, H, r)    = lora_gate_and_up_A
        """
        expert_segment = re.escape(self._expert_path_segment)
        n_experts = self.moe_config.n_routed_experts

        # Detect ParamWrapper keys
        pw_pattern = re.compile(
            rf"(?P<prefix>.*)layers\.(?P<layer>\d+)\.{expert_segment}\."
            rf"(?P<pw_suffix>(?:base_layer\.)?lora_[AB]\.weight)$"
        )

        consumed_keys: set[str] = set()
        new_entries: dict[str, torch.Tensor] = {}

        for key, tensor in state_dict.items():
            m = pw_pattern.match(key)
            if m is None:
                continue

            pw_suffix = m.group("pw_suffix")
            # Preserve the full prefix from the input key (e.g. "base_model.model.model.")
            # so downstream prefix stripping (_drop_outer_prefix) works correctly.
            prefix = m.group("prefix")
            layer_num = m.group("layer")
            base_key = f"{prefix}layers.{layer_num}.{self._expert_path_segment}"

            # down_proj is outer (no base_layer), gate_up_proj is inner (base_layer)
            if pw_suffix == "lora_A.weight":
                # (r*E, H) -> (E, r, H) = lora_down_B
                r = tensor.shape[0] // n_experts
                out = tensor.reshape(n_experts, r, tensor.shape[1]).contiguous()
                new_entries[f"{base_key}.lora_down_B"] = out

            elif pw_suffix == "lora_B.weight":
                # (I, r*E) -> reshape (I, r, E) -> permute(2,0,1) -> (E, I, r) = lora_down_A
                r = tensor.shape[1] // n_experts
                out = tensor.reshape(tensor.shape[0], r, n_experts).permute(2, 0, 1).contiguous()
                new_entries[f"{base_key}.lora_down_A"] = out

            elif pw_suffix == "base_layer.lora_A.weight":
                # (r*E, 2*I) -> (E, r, 2*I) = lora_gate_and_up_B
                r = tensor.shape[0] // n_experts
                out = tensor.reshape(n_experts, r, tensor.shape[1]).contiguous()
                new_entries[f"{base_key}.lora_gate_and_up_B"] = out

            elif pw_suffix == "base_layer.lora_B.weight":
                # (H, r*E) -> reshape (H, r, E) -> permute(2,0,1) -> (E, H, r) = lora_gate_and_up_A
                r = tensor.shape[1] // n_experts
                out = tensor.reshape(tensor.shape[0], r, n_experts).permute(2, 0, 1).contiguous()
                new_entries[f"{base_key}.lora_gate_and_up_A"] = out

            else:
                continue

            consumed_keys.add(key)

        if not consumed_keys:
            return state_dict

        result = {k: v for k, v in state_dict.items() if k not in consumed_keys}
        result.update(new_entries)
        return result
