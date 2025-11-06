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

"""State dict adapter for Llama model with combined projections.

This module provides conversion between HuggingFace format (separate Q/K/V and gate/up projections)
and the custom format with combined projections (qkv_proj and gate_up_proj).
"""

import logging
import re
from typing import Any, Optional

import torch
from torch.distributed.tensor import DTensor
from transformers import LlamaConfig

from nemo_automodel.components.moe.state_dict_utils import is_dtensor

logger = logging.getLogger(__name__)


def _safe_split(tensor: torch.Tensor, split_sizes: list[int], dim: int = 0) -> list[torch.Tensor]:
    """Split tensor handling both regular tensors and DTensors without triggering redistribution.

    For DTensors, extracts local shard, splits it, and rewraps each piece as DTensor.
    For regular tensors, performs normal split.

    Args:
        tensor: Tensor to split (can be DTensor or regular tensor)
        split_sizes: Split sizes computed based on global/local tensor size
        dim: Dimension to split along
    """
    # Check if this is a DTensor using proper type checking
    if is_dtensor(tensor):
        # DTensor: work on local shard to avoid redistribution/OOM
        # Use public API to extract local tensor
        local_tensor = tensor.to_local()
        local_size = local_tensor.shape[dim]
        global_size = sum(split_sizes)

        # Scale split sizes to local tensor size
        local_split_sizes = [(size * local_size) // global_size for size in split_sizes]

        # Verify splits sum exactly to local size (should not need rounding)
        if sum(local_split_sizes) != local_size:
            raise RuntimeError(
                f"Split size mismatch: local_split_sizes={local_split_sizes} sum to {sum(local_split_sizes)}, "
                f"but local tensor size is {local_size}. This indicates an incorrect split calculation."
            )

        local_splits = local_tensor.split(local_split_sizes, dim=dim)

        # Rewrap each split as DTensor with same placement/device_mesh
        return [
            DTensor.from_local(
                local_split,
                device_mesh=tensor.device_mesh,
                placements=tensor.placements,
                run_check=False,
            )
            for local_split in local_splits
        ]
    else:
        # Regular tensor: normal split
        return list(tensor.split(split_sizes, dim=dim))


def _safe_concat(tensors: list[torch.Tensor], dim: int = 0) -> torch.Tensor:
    """Concatenate tensors handling both regular tensors and DTensors without triggering redistribution.

    For DTensors, extracts local shards, concatenates them, and rewraps as DTensor.
    For regular tensors, performs normal concat.

    Args:
        tensors: List of tensors to concatenate (all DTensors or all regular tensors)
        dim: Dimension to concatenate along
    """
    if not tensors:
        raise ValueError("Cannot concatenate empty list of tensors")

    # Check if first tensor is DTensor using proper type checking (assume all are same type)
    if is_dtensor(tensors[0]):
        # DTensor: work on local shards to avoid redistribution/OOM
        # Use public API to extract local tensors from all DTensors
        local_tensors = [t.to_local() for t in tensors]

        # Concatenate local tensors
        local_concat = torch.cat(local_tensors, dim=dim)

        # Rewrap as DTensor with same placement/device_mesh as first tensor
        return DTensor.from_local(
            local_concat,
            device_mesh=tensors[0].device_mesh,
            placements=tensors[0].placements,
            run_check=False,
        )
    else:
        # Regular tensors: normal concat
        return torch.cat(tensors, dim=dim)


# -from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter
# LlamaStateDictAdapter should be a subclass of StateDictAdapter, but import-linter complains about it.
class LlamaStateDictAdapter:
    """Adapter for converting between HF and custom Llama formats.

    Handles conversion of:
    - Separate q_proj, k_proj, v_proj <-> Combined qkv_proj
    - Separate gate_proj, up_proj <-> Combined gate_up_proj
    """

    def __init__(self, config: LlamaConfig):
        """Initialize the adapter.

        Args:
            config: LlamaConfig from transformers
        """
        self.config = config
        self._uses_model_prefix = True

    def from_hf(
        self,
        hf_state_dict: dict[str, Any],
        **kwargs,
    ) -> dict[str, Any]:
        """Convert HuggingFace state dict to custom format.

        Converts separate Q/K/V and gate/up projections to combined projections.

        Args:
            hf_state_dict: State dict from HuggingFace model

        Returns:
            State dict in custom format
        """
        # Determine if model prefix is used
        for key in hf_state_dict.keys():
            if "layers" in key:
                self._uses_model_prefix = key.startswith("model.")
                break

        custom_state_dict = {}
        processed_keys = set()

        # Process each layer
        num_layers = self.config.num_hidden_layers
        for layer_idx in range(num_layers):
            prefix = f"model.layers.{layer_idx}" if self._uses_model_prefix else f"layers.{layer_idx}"

            # Handle attention projections - combine Q, K, V into single qkv_proj
            q_weight_key = f"{prefix}.self_attn.q_proj.weight"
            k_weight_key = f"{prefix}.self_attn.k_proj.weight"
            v_weight_key = f"{prefix}.self_attn.v_proj.weight"

            if q_weight_key in hf_state_dict:
                q_weight = hf_state_dict[q_weight_key]
                k_weight = hf_state_dict[k_weight_key]
                v_weight = hf_state_dict[v_weight_key]

                # Concatenate along output dimension (use _safe_concat to avoid DTensor all-gather)
                qkv_weight = _safe_concat([q_weight, k_weight, v_weight], dim=0)
                custom_state_dict[f"{prefix}.self_attn.qkv_proj.weight"] = qkv_weight

                processed_keys.update([q_weight_key, k_weight_key, v_weight_key])

                # Handle biases if present
                q_bias_key = f"{prefix}.self_attn.q_proj.bias"
                if q_bias_key in hf_state_dict:
                    k_bias_key = f"{prefix}.self_attn.k_proj.bias"
                    v_bias_key = f"{prefix}.self_attn.v_proj.bias"

                    q_bias = hf_state_dict[q_bias_key]
                    k_bias = hf_state_dict[k_bias_key]
                    v_bias = hf_state_dict[v_bias_key]

                    qkv_bias = _safe_concat([q_bias, k_bias, v_bias], dim=0)
                    custom_state_dict[f"{prefix}.self_attn.qkv_proj.bias"] = qkv_bias

                    processed_keys.update([q_bias_key, k_bias_key, v_bias_key])

            # Handle MLP projections - combine gate and up into single gate_up_proj
            gate_weight_key = f"{prefix}.mlp.gate_proj.weight"
            up_weight_key = f"{prefix}.mlp.up_proj.weight"

            if gate_weight_key in hf_state_dict:
                gate_weight = hf_state_dict[gate_weight_key]
                up_weight = hf_state_dict[up_weight_key]

                # Concatenate along output dimension (use _safe_concat to avoid DTensor all-gather)
                gate_up_weight = _safe_concat([gate_weight, up_weight], dim=0)
                custom_state_dict[f"{prefix}.mlp.gate_up_proj.weight"] = gate_up_weight

                processed_keys.update([gate_weight_key, up_weight_key])

                # Handle biases if present
                gate_bias_key = f"{prefix}.mlp.gate_proj.bias"
                if gate_bias_key in hf_state_dict:
                    up_bias_key = f"{prefix}.mlp.up_proj.bias"

                    gate_bias = hf_state_dict[gate_bias_key]
                    up_bias = hf_state_dict[up_bias_key]

                    gate_up_bias = _safe_concat([gate_bias, up_bias], dim=0)
                    custom_state_dict[f"{prefix}.mlp.gate_up_proj.bias"] = gate_up_bias

                    processed_keys.update([gate_bias_key, up_bias_key])

        # Copy all other weights that weren't processed
        for key, value in hf_state_dict.items():
            if key not in processed_keys:
                custom_state_dict[key] = value

        return custom_state_dict

    def to_hf(
        self,
        state_dict: dict[str, Any],
        exclude_key_regex: Optional[str] = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Convert custom state dict to HuggingFace format.

        Splits combined qkv_proj and gate_up_proj back to separate projections.
        Handles both full (unsharded) and TP-sharded tensors by computing
        split sizes dynamically based on actual tensor dimensions.

        Args:
            state_dict: State dict from custom model (can be TP-sharded DTensors)
            exclude_key_regex: Optional regex pattern to exclude keys

        Returns:
            State dict in HuggingFace format
        """
        hf_state_dict = {}
        processed_keys = set()

        # Determine if model prefix is used
        for key in state_dict.keys():
            if "layers" in key:
                self._uses_model_prefix = key.startswith("model.")
                break

        # Process each layer
        num_layers = self.config.num_hidden_layers
        head_dim = getattr(self.config, "head_dim", self.config.hidden_size // self.config.num_attention_heads)
        q_size = self.config.num_attention_heads * head_dim
        kv_size = self.config.num_key_value_heads * head_dim

        for layer_idx in range(num_layers):
            prefix = f"model.layers.{layer_idx}" if self._uses_model_prefix else f"layers.{layer_idx}"

            # Handle attention projections - split qkv_proj into separate Q, K, V
            qkv_weight_key = f"{prefix}.self_attn.qkv_proj.weight"

            if qkv_weight_key in state_dict:
                qkv_weight = state_dict[qkv_weight_key]

                # Split along output dimension
                # Handle TP sharding: compute local split sizes based on actual tensor size
                qkv_actual_size = qkv_weight.shape[0]
                total_size = q_size + 2 * kv_size
                local_q_size = (q_size * qkv_actual_size) // total_size
                local_kv_size = (kv_size * qkv_actual_size) // total_size
                # Use safe_split to avoid DTensor redistribution/OOM
                q_weight, k_weight, v_weight = _safe_split(
                    qkv_weight, [local_q_size, local_kv_size, local_kv_size], dim=0
                )

                hf_state_dict[f"{prefix}.self_attn.q_proj.weight"] = q_weight
                hf_state_dict[f"{prefix}.self_attn.k_proj.weight"] = k_weight
                hf_state_dict[f"{prefix}.self_attn.v_proj.weight"] = v_weight

                processed_keys.add(qkv_weight_key)

                # Handle biases if present
                qkv_bias_key = f"{prefix}.self_attn.qkv_proj.bias"
                if qkv_bias_key in state_dict:
                    qkv_bias = state_dict[qkv_bias_key]

                    # Handle TP sharding for biases
                    qkv_bias_size = qkv_bias.shape[0]
                    total_size = q_size + 2 * kv_size
                    local_q_size = (q_size * qkv_bias_size) // total_size
                    local_kv_size = (kv_size * qkv_bias_size) // total_size
                    # Use safe_split to avoid DTensor redistribution/OOM
                    q_bias, k_bias, v_bias = _safe_split(qkv_bias, [local_q_size, local_kv_size, local_kv_size], dim=0)

                    hf_state_dict[f"{prefix}.self_attn.q_proj.bias"] = q_bias
                    hf_state_dict[f"{prefix}.self_attn.k_proj.bias"] = k_bias
                    hf_state_dict[f"{prefix}.self_attn.v_proj.bias"] = v_bias

                    processed_keys.add(qkv_bias_key)

            # Handle MLP projections - split gate_up_proj into separate gate and up
            gate_up_weight_key = f"{prefix}.mlp.gate_up_proj.weight"

            if gate_up_weight_key in state_dict:
                gate_up_weight = state_dict[gate_up_weight_key]

                # Split along output dimension
                # Handle TP sharding: compute local split sizes based on actual tensor size
                gate_up_actual_size = gate_up_weight.shape[0]
                local_intermediate_size = gate_up_actual_size // 2
                # Use safe_split to avoid DTensor redistribution/OOM
                gate_weight, up_weight = _safe_split(
                    gate_up_weight, [local_intermediate_size, local_intermediate_size], dim=0
                )

                hf_state_dict[f"{prefix}.mlp.gate_proj.weight"] = gate_weight
                hf_state_dict[f"{prefix}.mlp.up_proj.weight"] = up_weight

                processed_keys.add(gate_up_weight_key)

                # Handle biases if present
                gate_up_bias_key = f"{prefix}.mlp.gate_up_proj.bias"
                if gate_up_bias_key in state_dict:
                    gate_up_bias = state_dict[gate_up_bias_key]

                    # Handle TP sharding for biases
                    gate_up_bias_size = gate_up_bias.shape[0]
                    local_intermediate_size = gate_up_bias_size // 2
                    # Use safe_split to avoid DTensor redistribution/OOM
                    gate_bias, up_bias = _safe_split(
                        gate_up_bias, [local_intermediate_size, local_intermediate_size], dim=0
                    )

                    hf_state_dict[f"{prefix}.mlp.gate_proj.bias"] = gate_bias
                    hf_state_dict[f"{prefix}.mlp.up_proj.bias"] = up_bias

                    processed_keys.add(gate_up_bias_key)

        # Copy all other weights that weren't processed
        for key, value in state_dict.items():
            if key not in processed_keys:
                hf_state_dict[key] = value

        # Apply exclusion regex if provided
        if exclude_key_regex:
            hf_state_dict = {k: v for k, v in hf_state_dict.items() if not re.match(exclude_key_regex, k)}

        return hf_state_dict
