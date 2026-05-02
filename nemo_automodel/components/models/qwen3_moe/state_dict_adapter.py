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

# Native LoRA suffixes for grouped MoE expert tensors
_LORA_EXPERT_SUFFIXES = ("lora_gate_and_up_A", "lora_gate_and_up_B", "lora_down_A", "lora_down_B")


class Qwen3MoeStateDictAdapter(MoESplitExpertsStateDictMixin, StateDictAdapter):
    """Converts between HF Qwen3-MoE checkpoints and our grouped-experts native format.

    Qwen3-MoE HF experts use keys:
      model.layers.{L}.mlp.experts.{E}.gate_proj.weight
      model.layers.{L}.mlp.experts.{E}.up_proj.weight
      model.layers.{L}.mlp.experts.{E}.down_proj.weight

    Our native format groups them into:
      model.layers.{L}.mlp.experts.gate_and_up_projs  # [n_experts, dim, 2*moe_inter_dim]
      model.layers.{L}.mlp.experts.down_projs         # [n_experts, moe_inter_dim, dim]
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

        When ``v4_compatible=False`` (the default), LoRA expert tensors are
        emitted in PEFT v0.18+ ParamWrapper format so that
        ``PeftModel.from_pretrained()`` can load them directly.  When
        ``v4_compatible=True``, the legacy per-expert split is used instead
        (via the parent mixin).

        Args:
            fqn: Fully qualified name of the tensor in native format
            tensor: The tensor to convert
            **kwargs: Additional arguments for conversion

        Returns:
            List of (fqn, tensor) tuples in HuggingFace format
        """
        exclude_key_regex = kwargs.get("exclude_key_regex", None)
        v4_compatible = kwargs.get("v4_compatible", False)

        # Check if this is a LoRA expert tensor eligible for ParamWrapper conversion
        if not v4_compatible:
            expert_segment = self._expert_path_segment
            for suffix in _LORA_EXPERT_SUFFIXES:
                if fqn.endswith(f".{suffix}") and f".{expert_segment}.{suffix}" in fqn:
                    result = self._convert_lora_to_paramwrapper(fqn, tensor)
                    if exclude_key_regex:
                        result = [(k, v) for k, v in result if not re.match(exclude_key_regex, k)]
                    return result

        # Non-LoRA keys or legacy mode: fall through to parent mixin
        expert_result = self._convert_single_merged_expert_to_hf_split_experts(fqn, tensor, **kwargs)
        if expert_result is not None:
            result = expert_result
        else:
            result = [(fqn, tensor)]

        if exclude_key_regex:
            result = [(k, v) for k, v in result if not re.match(exclude_key_regex, k)]

        return result

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
        # Detect whether HF checkpoints use the "model." prefix
        for key in hf_state_dict.keys():
            if ".mlp.experts." in key and key.endswith(".weight"):
                self._uses_model_prefix = key.startswith("model.")
                break

        # Convert any ParamWrapper-format LoRA keys to native grouped format
        hf_state_dict = self._convert_paramwrapper_to_native(hf_state_dict)

        return self._from_hf_w_merged_experts(hf_state_dict, device_mesh)

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
