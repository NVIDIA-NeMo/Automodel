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

"""Generic state dict adapter for models with combined projections.

This module provides a unified state dict converter that handles:
- Separate q_proj, k_proj, v_proj <-> Combined qkv_proj
- Separate gate_proj, up_proj <-> Combined gate_up_proj
- Tied weights (lm_head <-> embed_tokens)

Works with any transformer model (Llama, Qwen2, etc.) that uses these projection patterns.
"""

import logging
import re
from typing import Any, Optional

import torch
from torch.distributed.tensor import DTensor

logger = logging.getLogger(__name__)


def _is_dtensor(tensor: torch.Tensor) -> bool:
    """Check if tensor is a DTensor without importing DTensor directly."""
    return isinstance(tensor, DTensor)


def _safe_split(tensor: torch.Tensor, split_sizes: list[int], dim: int = 0) -> list[torch.Tensor]:
    """Split tensor handling both regular tensors and DTensors without triggering redistribution.

    For DTensors, extracts local shard, splits it, and rewraps each piece as DTensor.
    For regular tensors, performs normal split.

    Args:
        tensor: Tensor to split (can be DTensor or regular tensor)
        split_sizes: Split sizes computed based on global/local tensor size
        dim: Dimension to split along
    """
    if _is_dtensor(tensor):
        # DTensor: work on local shard to avoid redistribution/OOM
        local_tensor = tensor.to_local()
        local_size = local_tensor.shape[dim]
        global_size = sum(split_sizes)

        # Scale split sizes to local tensor size
        local_split_sizes = [(size * local_size) // global_size for size in split_sizes]

        # Verify splits sum exactly to local size
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

    if _is_dtensor(tensors[0]):
        # DTensor: work on local shards to avoid redistribution/OOM
        local_tensors = [t.to_local() for t in tensors]
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


class CombinedProjectionStateDictAdapter:
    """Generic adapter for converting between HF and combined-projection formats.

    Handles conversion of:
    - Separate q_proj, k_proj, v_proj <-> Combined qkv_proj
    - Separate gate_proj, up_proj <-> Combined gate_up_proj
    - Tied weights (lm_head <-> embed_tokens) for loading HF checkpoints

    Works with any transformer model config that has:
    - num_hidden_layers
    - num_attention_heads
    - num_key_value_heads
    - hidden_size

    Args:
        config: Model config (LlamaConfig, Qwen2Config, etc.)

    Example:
        # For Llama
        from transformers import LlamaConfig
        adapter = CombinedProjectionStateDictAdapter(LlamaConfig.from_pretrained("meta-llama/Llama-3-8B"))

        # For Qwen2
        from transformers import Qwen2Config
        adapter = CombinedProjectionStateDictAdapter(Qwen2Config.from_pretrained("Qwen/Qwen2.5-7B"))
    """

    def __init__(self, config):
        """Initialize the adapter with model config."""
        self.config = config
        self._uses_model_prefix = True

        # Extract config parameters
        self.num_layers = config.num_hidden_layers
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.hidden_size = config.hidden_size
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

        # Compute projection sizes
        self.q_size = self.num_attention_heads * self.head_dim
        self.kv_size = self.num_key_value_heads * self.head_dim

    def from_hf(self, hf_state_dict: dict[str, Any], **kwargs) -> dict[str, Any]:
        """Convert HuggingFace state dict to combined-projection format.

        Converts separate Q/K/V and gate/up projections to combined projections.
        Also handles tied weights (lm_head <-> embed_tokens) by copying embed_tokens
        to lm_head if lm_head is missing (common in HF Qwen2 and Llama checkpoints).

        Args:
            hf_state_dict: State dict from HuggingFace model

        Returns:
            State dict in combined-projection format
        """
        # Determine if model prefix is used
        for key in hf_state_dict.keys():
            if "layers" in key:
                self._uses_model_prefix = key.startswith("model.")
                break

        custom_state_dict = {}
        processed_keys = set()

        # Process each layer
        for layer_idx in range(self.num_layers):
            prefix = f"model.layers.{layer_idx}" if self._uses_model_prefix else f"layers.{layer_idx}"

            # Combine Q, K, V into qkv_proj
            q_weight_key = f"{prefix}.self_attn.q_proj.weight"
            k_weight_key = f"{prefix}.self_attn.k_proj.weight"
            v_weight_key = f"{prefix}.self_attn.v_proj.weight"

            if q_weight_key in hf_state_dict:
                q_weight = hf_state_dict[q_weight_key]
                k_weight = hf_state_dict[k_weight_key]
                v_weight = hf_state_dict[v_weight_key]

                # Concatenate along output dimension
                qkv_weight = _safe_concat([q_weight, k_weight, v_weight], dim=0)
                custom_state_dict[f"{prefix}.self_attn.qkv_proj.weight"] = qkv_weight
                processed_keys.update([q_weight_key, k_weight_key, v_weight_key])

                # Handle biases if present
                q_bias_key = f"{prefix}.self_attn.q_proj.bias"
                if q_bias_key in hf_state_dict:
                    k_bias_key = f"{prefix}.self_attn.k_proj.bias"
                    v_bias_key = f"{prefix}.self_attn.v_proj.bias"

                    qkv_bias = _safe_concat(
                        [hf_state_dict[q_bias_key], hf_state_dict[k_bias_key], hf_state_dict[v_bias_key]], dim=0
                    )
                    custom_state_dict[f"{prefix}.self_attn.qkv_proj.bias"] = qkv_bias
                    processed_keys.update([q_bias_key, k_bias_key, v_bias_key])

            # Combine gate and up into gate_up_proj
            gate_weight_key = f"{prefix}.mlp.gate_proj.weight"
            up_weight_key = f"{prefix}.mlp.up_proj.weight"

            if gate_weight_key in hf_state_dict:
                gate_weight = hf_state_dict[gate_weight_key]
                up_weight = hf_state_dict[up_weight_key]

                # Concatenate along output dimension
                gate_up_weight = _safe_concat([gate_weight, up_weight], dim=0)
                custom_state_dict[f"{prefix}.mlp.gate_up_proj.weight"] = gate_up_weight
                processed_keys.update([gate_weight_key, up_weight_key])

                # Handle biases if present
                gate_bias_key = f"{prefix}.mlp.gate_proj.bias"
                if gate_bias_key in hf_state_dict:
                    up_bias_key = f"{prefix}.mlp.up_proj.bias"

                    gate_up_bias = _safe_concat([hf_state_dict[gate_bias_key], hf_state_dict[up_bias_key]], dim=0)
                    custom_state_dict[f"{prefix}.mlp.gate_up_proj.bias"] = gate_up_bias
                    processed_keys.update([gate_bias_key, up_bias_key])

        # Copy all other weights that weren't processed
        for key, value in hf_state_dict.items():
            if key not in processed_keys:
                custom_state_dict[key] = value

        # Handle tied weights: if lm_head.weight is missing but embed_tokens exists, tie them
        # This is common in Qwen2 and Llama where lm_head shares weights with embeddings
        # Only do this if config specifies tie_word_embeddings=True
        if getattr(self.config, "tie_word_embeddings", True):
            embed_key = "model.embed_tokens.weight" if self._uses_model_prefix else "embed_tokens.weight"
            lm_head_key = "lm_head.weight"

            if lm_head_key not in custom_state_dict and embed_key in custom_state_dict:
                logger.info(f"Tying lm_head.weight to {embed_key} (HuggingFace checkpoint has tied weights)")
                custom_state_dict[lm_head_key] = custom_state_dict[embed_key]

        return custom_state_dict

    def to_hf(
        self,
        state_dict: dict[str, Any],
        exclude_key_regex: Optional[str] = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Convert combined-projection state dict to HuggingFace format.

        Splits combined qkv_proj and gate_up_proj back to separate projections.
        Handles both full (unsharded) and TP-sharded tensors.

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
        for layer_idx in range(self.num_layers):
            prefix = f"model.layers.{layer_idx}" if self._uses_model_prefix else f"layers.{layer_idx}"

            # Split qkv_proj into separate Q, K, V
            qkv_weight_key = f"{prefix}.self_attn.qkv_proj.weight"

            if qkv_weight_key in state_dict:
                qkv_weight = state_dict[qkv_weight_key]

                # Compute local split sizes based on actual tensor size (handles TP sharding)
                qkv_actual_size = qkv_weight.shape[0]
                total_size = self.q_size + 2 * self.kv_size
                local_q_size = (self.q_size * qkv_actual_size) // total_size
                local_kv_size = (self.kv_size * qkv_actual_size) // total_size

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
                    qkv_bias_size = qkv_bias.shape[0]
                    local_q_size = (self.q_size * qkv_bias_size) // total_size
                    local_kv_size = (self.kv_size * qkv_bias_size) // total_size

                    q_bias, k_bias, v_bias = _safe_split(qkv_bias, [local_q_size, local_kv_size, local_kv_size], dim=0)

                    hf_state_dict[f"{prefix}.self_attn.q_proj.bias"] = q_bias
                    hf_state_dict[f"{prefix}.self_attn.k_proj.bias"] = k_bias
                    hf_state_dict[f"{prefix}.self_attn.v_proj.bias"] = v_bias
                    processed_keys.add(qkv_bias_key)

            # Split gate_up_proj into separate gate and up
            gate_up_weight_key = f"{prefix}.mlp.gate_up_proj.weight"

            if gate_up_weight_key in state_dict:
                gate_up_weight = state_dict[gate_up_weight_key]

                # Compute local split sizes
                gate_up_actual_size = gate_up_weight.shape[0]
                local_intermediate_size = gate_up_actual_size // 2

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
                    gate_up_bias_size = gate_up_bias.shape[0]
                    local_intermediate_size = gate_up_bias_size // 2

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
