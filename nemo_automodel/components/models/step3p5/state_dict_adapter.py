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

"""State dict adapter for Step3p5 model.

Step3p5 uses grouped MoELinear weights with shape [n_exp, out, in], different from
the standard per-expert format. This adapter handles conversion between:

HF Format (Step3p5):
    model.layers.{L}.moe.gate_proj.weight    # [n_exp, inter, dim]
    model.layers.{L}.moe.up_proj.weight      # [n_exp, inter, dim]
    model.layers.{L}.moe.down_proj.weight    # [n_exp, dim, inter]
    model.layers.{L}.moe.gate.weight         # [n_exp, dim] (router)
    model.layers.{L}.moe.router_bias         # [n_exp] (router bias, optional)
    model.layers.{L}.share_expert.*.weight   # Shared expert

Native Format (Automodel):
    model.layers.{L}.moe.experts.gate_and_up_projs  # [n_exp, dim, 2*inter]
    model.layers.{L}.moe.experts.down_projs         # [n_exp, inter, dim]
    model.layers.{L}.moe.gate.weight                # [n_exp, dim]
    model.layers.{L}.moe.gate.bias                  # [n_exp]
    model.layers.{L}.share_expert.*.weight

Note: Router gate weights and shared expert weights pass through with the same key names.
Only the expert MLP weights (gate_proj, up_proj, down_proj) need transformation.
"""

import logging
import re
from typing import Any, Optional

import torch
from torch.distributed.device_mesh import DeviceMesh

from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.moe.layers import MoEConfig
from nemo_automodel.components.moe.state_dict_utils import (
    create_dtensor_from_local,
    get_expert_range_for_rank_from_mesh,
    get_submesh,
    is_dtensor,
)

logger = logging.getLogger(__name__)


def _create_dtensor_from_local_or_reference(
    local_tensor: torch.Tensor,
    reference_dtensor: Optional["torch.Tensor"],
    device_mesh: Optional["DeviceMesh"] = None,
    rank: Optional[int] = None,
) -> torch.Tensor:
    """Create a DTensor from a local tensor.

    Prefers using reference_dtensor's mesh/placements if available (for preserving
    DTensor structure from DCP-loaded tensors). Falls back to creating a new DTensor
    using device_mesh if reference is not a DTensor.

    Args:
        local_tensor: Local portion of the tensor after transformation
        reference_dtensor: Optional DTensor to copy mesh/placements from
        device_mesh: Device mesh for EP (used if reference is not DTensor)
        rank: Current rank for device placement

    Returns:
        DTensor if mesh is available, otherwise local_tensor
    """
    from torch.distributed._tensor import DTensor

    if reference_dtensor is not None and is_dtensor(reference_dtensor):
        # Use the exact same mesh and placements from the reference
        return DTensor.from_local(local_tensor, reference_dtensor.device_mesh, reference_dtensor.placements)
    elif device_mesh is not None:
        # Create DTensor using the provided mesh
        return create_dtensor_from_local(local_tensor, device_mesh, rank)
    else:
        # No mesh available, return regular tensor
        return local_tensor


class Step3p5StateDictAdapter(StateDictAdapter):
    """Converts between HF Step3p5 checkpoints and Automodel grouped-experts native format.

    Step3p5 HF uses grouped MoELinear with shape [n_experts, out_features, in_features]:
        model.layers.{L}.moe.gate_proj.weight  # [n_exp, inter, dim]
        model.layers.{L}.moe.up_proj.weight    # [n_exp, inter, dim]
        model.layers.{L}.moe.down_proj.weight  # [n_exp, dim, inter]

    Automodel native format uses:
        model.layers.{L}.moe.experts.gate_and_up_projs  # [n_exp, dim, 2*inter]
        model.layers.{L}.moe.experts.down_projs         # [n_exp, inter, dim]
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

    @property
    def _hf_prefix(self) -> str:
        """Prefix for HuggingFace format keys."""
        return "model." if self._uses_model_prefix else ""

    def to_hf(
        self,
        state_dict: dict[str, Any],
        exclude_key_regex: Optional[str] = None,
        quantization: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        """Convert from native model state dict to HuggingFace format."""
        hf_state_dict = {}

        for fqn, tensor in state_dict.items():
            converted_tensors = self.convert_single_tensor_to_hf(
                fqn, tensor, exclude_key_regex=exclude_key_regex, quantization=quantization, **kwargs
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
        """Convert HF checkpoint to native format.

        Handles Step3p5's grouped MoELinear format:
        - [n_exp, inter, dim] gate_proj/up_proj -> [n_exp, dim, 2*inter] gate_and_up_projs
        - [n_exp, dim, inter] down_proj -> [n_exp, inter, dim] down_projs
        """
        # Detect prefix
        for key in hf_state_dict.keys():
            if ".moe.gate_proj.weight" in key or ".moe.up_proj.weight" in key:
                self._uses_model_prefix = key.startswith("model.")
                break

        n_experts = self.moe_config.n_routed_experts

        if device_mesh is not None:
            start_expert, end_expert = get_expert_range_for_rank_from_mesh(device_mesh, n_experts)
            rank = (
                get_submesh(device_mesh, ("ep",)).get_rank()
                if "ep" in device_mesh.mesh_dim_names
                else device_mesh.get_rank()
            )
        else:
            start_expert, end_expert = 0, n_experts
            rank = None

        state_dict: dict[str, Any] = {}
        # Track gate_proj and up_proj tensors to merge them
        pending_gate_up: dict[str, dict[str, torch.Tensor]] = {}

        # Pattern for Step3p5 grouped MoE expert weights (gate_proj, up_proj, down_proj)
        # Note: This pattern does NOT match router gate (moe.gate.weight/bias) or shared experts
        moe_pattern = re.compile(
            r"(?P<prefix>(?:model\.)?(?:language_model\.)?)layers\.(?P<layer>\d+)\.moe\.(?P<proj>gate_proj|up_proj|down_proj)\.weight"
        )

        # Pattern for router gate weights that need expert slicing for EP
        # HF Step3p5 uses moe.gate.weight and moe.router_bias (not moe.gate.bias)
        router_gate_pattern = re.compile(
            r"(?P<prefix>(?:model\.)?(?:language_model\.)?)layers\.(?P<layer>\d+)\.moe\.(?P<param>gate\.weight|gate\.bias|router_bias|gate\.e_score_correction_bias)"
        )

        for key, value in hf_state_dict.items():
            m = moe_pattern.match(key)
            router_m = router_gate_pattern.match(key)

            if m:
                prefix = m.group("prefix") or ""
                layer_num = m.group("layer")
                proj = m.group("proj")

                # For expert parallelism with regular tensors, slice the experts dimension.
                # If the value is already a DTensor (from DCP loading), it's already sharded
                # and we should not slice it again.
                if device_mesh is not None and not is_dtensor(value):
                    value = value[start_expert:end_expert]

                if proj in ("gate_proj", "up_proj"):
                    # Collect gate_proj and up_proj to merge
                    layer_key = f"{prefix}layers.{layer_num}"
                    if layer_key not in pending_gate_up:
                        pending_gate_up[layer_key] = {}

                    pending_gate_up[layer_key][proj] = value

                    # Check if we have both gate_proj and up_proj
                    if "gate_proj" in pending_gate_up[layer_key] and "up_proj" in pending_gate_up[layer_key]:
                        gate_weight = pending_gate_up[layer_key]["gate_proj"]  # [n_exp, inter, dim]
                        up_weight = pending_gate_up[layer_key]["up_proj"]  # [n_exp, inter, dim]

                        # Keep reference to original DTensor for mesh/placements (if available)
                        reference_dtensor = gate_weight if is_dtensor(gate_weight) else None

                        # Extract local tensors if DTensor for transformation
                        gate_local = gate_weight.to_local() if is_dtensor(gate_weight) else gate_weight
                        up_local = up_weight.to_local() if is_dtensor(up_weight) else up_weight

                        # Transpose: [n_exp, inter, dim] -> [n_exp, dim, inter]
                        gate_t = gate_local.transpose(1, 2)
                        up_t = up_local.transpose(1, 2)

                        # Concatenate: [n_exp, dim, 2*inter]
                        merged = torch.cat([gate_t, up_t], dim=-1).to(self.dtype)

                        native_key = f"{prefix}layers.{layer_num}.moe.experts.gate_and_up_projs"
                        # Create DTensor using reference or device_mesh
                        state_dict[native_key] = _create_dtensor_from_local_or_reference(
                            merged, reference_dtensor, device_mesh, rank
                        )

                        # Clean up
                        del pending_gate_up[layer_key]

                elif proj == "down_proj":
                    # down_proj: [n_exp, dim, inter] -> [n_exp, inter, dim]
                    # Keep reference for DTensor recreation (if available)
                    reference_dtensor = value if is_dtensor(value) else None

                    down_local = value.to_local() if is_dtensor(value) else value
                    down_t = down_local.transpose(1, 2).to(self.dtype)

                    native_key = f"{prefix}layers.{layer_num}.moe.experts.down_projs"
                    # Create DTensor using reference or device_mesh
                    state_dict[native_key] = _create_dtensor_from_local_or_reference(
                        down_t, reference_dtensor, device_mesh, rank
                    )

            elif router_m:
                # Router gate weight/bias - handle key mapping
                # Note: Router weights are NOT sliced for EP - they are replicated
                # because all ranks need full routing information
                param = router_m.group("param")
                prefix = router_m.group("prefix") or ""
                layer_num = router_m.group("layer")

                # Map HF router_bias to native gate.bias
                if param == "router_bias":
                    native_key = f"{prefix}layers.{layer_num}.moe.gate.bias"
                else:
                    native_key = key

                state_dict[native_key] = value

            else:
                # Non-MoE weights pass through unchanged (embeddings, norms, attention, shared experts, etc.)
                state_dict[key] = value

        return state_dict

    def convert_single_tensor_to_hf(
        self,
        fqn: str,
        tensor: Any,
        **kwargs,
    ) -> list[tuple[str, Any]]:
        """Convert a single tensor from native format to HuggingFace format.

        Args:
            fqn: Fully qualified name of the tensor in native format
            tensor: The tensor to convert
            **kwargs: Additional arguments for conversion

        Returns:
            List of (fqn, tensor) tuples in HuggingFace format
        """
        exclude_key_regex = kwargs.get("exclude_key_regex", None)

        result = self._convert_native_to_hf(fqn, tensor)
        if result is None:
            result = [(fqn, tensor)]

        if exclude_key_regex:
            result = [(k, v) for k, v in result if not re.match(exclude_key_regex, k)]

        return result

    def _convert_native_to_hf(
        self,
        fqn: str,
        tensor: torch.Tensor,
    ) -> list[tuple[str, torch.Tensor]] | None:
        """Convert native format expert tensors to HF Step3p5 format.

        Native: gate_and_up_projs [n_exp, dim, 2*inter] -> HF: gate_proj, up_proj [n_exp, inter, dim]
        Native: down_projs [n_exp, inter, dim] -> HF: down_proj [n_exp, dim, inter]

        Preserves DTensor structure when input is a DTensor.
        """
        from torch.distributed._tensor import DTensor

        inter_dim = self.moe_config.moe_inter_dim
        prefix = self._hf_prefix

        # Handle gate_and_up_projs
        if ".moe.experts.gate_and_up_projs" in fqn:
            layer_match = re.search(r"layers\.(\d+)", fqn)
            if layer_match is None:
                return None

            layer_num = layer_match.group(1)

            # Check if input is DTensor to preserve structure
            tensor_is_dtensor = is_dtensor(tensor)
            if tensor_is_dtensor:
                device_mesh = tensor.device_mesh
                placements = tensor.placements
                local_tensor = tensor.to_local()
            else:
                local_tensor = tensor

            # Split gate and up: [n_exp, dim, 2*inter] -> [n_exp, dim, inter] each
            gate_t = local_tensor[:, :, :inter_dim]  # [n_exp, dim, inter]
            up_t = local_tensor[:, :, inter_dim:]  # [n_exp, dim, inter]

            # Transpose: [n_exp, dim, inter] -> [n_exp, inter, dim]
            gate_weight = gate_t.transpose(1, 2).contiguous()
            up_weight = up_t.transpose(1, 2).contiguous()

            # Wrap in DTensor if original was DTensor
            if tensor_is_dtensor:
                gate_weight = DTensor.from_local(gate_weight, device_mesh, placements)
                up_weight = DTensor.from_local(up_weight, device_mesh, placements)

            return [
                (f"{prefix}layers.{layer_num}.moe.gate_proj.weight", gate_weight),
                (f"{prefix}layers.{layer_num}.moe.up_proj.weight", up_weight),
            ]

        # Handle down_projs
        if ".moe.experts.down_projs" in fqn:
            layer_match = re.search(r"layers\.(\d+)", fqn)
            if layer_match is None:
                return None

            layer_num = layer_match.group(1)

            # Check if input is DTensor to preserve structure
            tensor_is_dtensor = is_dtensor(tensor)
            if tensor_is_dtensor:
                device_mesh = tensor.device_mesh
                placements = tensor.placements
                local_tensor = tensor.to_local()
            else:
                local_tensor = tensor

            # Transpose: [n_exp, inter, dim] -> [n_exp, dim, inter]
            down_weight = local_tensor.transpose(1, 2).contiguous()

            # Wrap in DTensor if original was DTensor
            if tensor_is_dtensor:
                down_weight = DTensor.from_local(down_weight, device_mesh, placements)

            return [
                (f"{prefix}layers.{layer_num}.moe.down_proj.weight", down_weight),
            ]

        # Handle moe.gate.bias -> router_bias mapping for to_hf
        if ".moe.gate.bias" in fqn:
            layer_match = re.search(r"layers\.(\d+)", fqn)
            if layer_match is None:
                return None
            layer_num = layer_match.group(1)
            return [(f"{prefix}layers.{layer_num}.moe.router_bias", tensor)]

        # Router gate weight and shared expert weights pass through unchanged
        # These keys include: moe.gate.weight, share_expert.*
        return None
