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

logger = logging.getLogger(__name__)


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

        This method supports a memory-efficient mode:
        - If ``inplace=True`` is passed, it will **mutate** the input dict and
          eagerly ``pop()`` the source tensors (e.g., q/k/v) after merging them.
          This reduces peak GPU memory during checkpoint loading because the
          unmerged tensors can be freed earlier.

        Args:
            hf_state_dict: State dict from HuggingFace model
            inplace: If True, mutate ``hf_state_dict`` in-place. Defaults to False.

        Returns:
            State dict in combined-projection format
        """
        inplace = bool(kwargs.get("inplace", False))
        # If the caller needs to preserve the original HF dict (common in tests),
        # operate on a shallow copy. This duplicates only the Python dict, not tensors.
        state_dict: dict[str, Any] = hf_state_dict if inplace else dict(hf_state_dict)

        # Determine if model prefix is used
        for key in state_dict.keys():
            if "layers" in key:
                self._uses_model_prefix = key.startswith("model.")
                break

        # Process each layer and pop source tensors as soon as they've been merged.
        for layer_idx in range(self.num_layers):
            prefix = f"model.layers.{layer_idx}" if self._uses_model_prefix else f"layers.{layer_idx}"

            # Combine Q, K, V into qkv_proj
            q_weight_key = f"{prefix}.self_attn.q_proj.weight"
            if q_weight_key in state_dict:
                k_weight_key = f"{prefix}.self_attn.k_proj.weight"
                v_weight_key = f"{prefix}.self_attn.v_proj.weight"

                q_weight = state_dict[q_weight_key]
                k_weight = state_dict[k_weight_key]
                v_weight = state_dict[v_weight_key]

                state_dict[f"{prefix}.self_attn.qkv_proj.weight"] = torch.cat([q_weight, k_weight, v_weight], dim=0)

                # Drop references ASAP to reduce peak memory in inplace mode.
                state_dict.pop(q_weight_key, None)
                state_dict.pop(k_weight_key, None)
                state_dict.pop(v_weight_key, None)
                del q_weight, k_weight, v_weight

                # Handle biases if present
                q_bias_key = f"{prefix}.self_attn.q_proj.bias"
                if q_bias_key in state_dict:
                    k_bias_key = f"{prefix}.self_attn.k_proj.bias"
                    v_bias_key = f"{prefix}.self_attn.v_proj.bias"

                    q_bias = state_dict[q_bias_key]
                    k_bias = state_dict[k_bias_key]
                    v_bias = state_dict[v_bias_key]
                    state_dict[f"{prefix}.self_attn.qkv_proj.bias"] = torch.cat([q_bias, k_bias, v_bias], dim=0)

                    state_dict.pop(q_bias_key, None)
                    state_dict.pop(k_bias_key, None)
                    state_dict.pop(v_bias_key, None)
                    del q_bias, k_bias, v_bias

            # Combine gate and up into gate_up_proj
            gate_weight_key = f"{prefix}.mlp.gate_proj.weight"
            if gate_weight_key in state_dict:
                up_weight_key = f"{prefix}.mlp.up_proj.weight"
                gate_weight = state_dict[gate_weight_key]
                up_weight = state_dict[up_weight_key]

                state_dict[f"{prefix}.mlp.gate_up_proj.weight"] = torch.cat([gate_weight, up_weight], dim=0)

                state_dict.pop(gate_weight_key, None)
                state_dict.pop(up_weight_key, None)
                del gate_weight, up_weight

                # Handle biases if present
                gate_bias_key = f"{prefix}.mlp.gate_proj.bias"
                if gate_bias_key in state_dict:
                    up_bias_key = f"{prefix}.mlp.up_proj.bias"
                    gate_bias = state_dict[gate_bias_key]
                    up_bias = state_dict[up_bias_key]

                    state_dict[f"{prefix}.mlp.gate_up_proj.bias"] = torch.cat([gate_bias, up_bias], dim=0)

                    state_dict.pop(gate_bias_key, None)
                    state_dict.pop(up_bias_key, None)
                    del gate_bias, up_bias

        # Handle tied weights: if lm_head.weight is missing but embed_tokens exists, tie them
        # This is common in Qwen2 and Llama where lm_head shares weights with embeddings
        # Only do this if config specifies tie_word_embeddings=True
        if getattr(self.config, "tie_word_embeddings", True):
            embed_key = "model.embed_tokens.weight" if self._uses_model_prefix else "embed_tokens.weight"
            lm_head_key = "lm_head.weight"

            if lm_head_key not in state_dict and embed_key in state_dict:
                logger.info(f"Tying lm_head.weight to {embed_key} (HuggingFace checkpoint has tied weights)")
                state_dict[lm_head_key] = state_dict[embed_key]

        return state_dict

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
        inplace = bool(kwargs.get("inplace", False))
        if inplace:
            # Mutate the input dict in-place to reduce peak memory in conversion paths
            # (e.g., checkpoint saving) by dropping the combined tensors as soon as
            # the split tensors are created.
            hf_state_dict = state_dict
        else:
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

                q_weight, k_weight, v_weight = qkv_weight.split([local_q_size, local_kv_size, local_kv_size], dim=0)

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

                    q_bias, k_bias, v_bias = qkv_bias.split([local_q_size, local_kv_size, local_kv_size], dim=0)

                    hf_state_dict[f"{prefix}.self_attn.q_proj.bias"] = q_bias
                    hf_state_dict[f"{prefix}.self_attn.k_proj.bias"] = k_bias
                    hf_state_dict[f"{prefix}.self_attn.v_proj.bias"] = v_bias
                    processed_keys.add(qkv_bias_key)

                if inplace:
                    # Drop combined tensor keys eagerly to reduce peak memory.
                    state_dict.pop(qkv_weight_key, None)
                    state_dict.pop(qkv_bias_key, None)
                    del qkv_weight, q_weight, k_weight, v_weight
                    if qkv_bias_key in processed_keys:
                        # Only delete locals if we actually created them
                        del qkv_bias, q_bias, k_bias, v_bias

            # Split gate_up_proj into separate gate and up
            gate_up_weight_key = f"{prefix}.mlp.gate_up_proj.weight"

            if gate_up_weight_key in state_dict:
                gate_up_weight = state_dict[gate_up_weight_key]

                # Compute local split sizes
                gate_up_actual_size = gate_up_weight.shape[0]
                local_intermediate_size = gate_up_actual_size // 2

                gate_weight, up_weight = gate_up_weight.split([local_intermediate_size, local_intermediate_size], dim=0)

                hf_state_dict[f"{prefix}.mlp.gate_proj.weight"] = gate_weight
                hf_state_dict[f"{prefix}.mlp.up_proj.weight"] = up_weight
                processed_keys.add(gate_up_weight_key)

                # Handle biases if present
                gate_up_bias_key = f"{prefix}.mlp.gate_up_proj.bias"
                if gate_up_bias_key in state_dict:
                    gate_up_bias = state_dict[gate_up_bias_key]
                    gate_up_bias_size = gate_up_bias.shape[0]
                    local_intermediate_size = gate_up_bias_size // 2

                    gate_bias, up_bias = gate_up_bias.split([local_intermediate_size, local_intermediate_size], dim=0)

                    hf_state_dict[f"{prefix}.mlp.gate_proj.bias"] = gate_bias
                    hf_state_dict[f"{prefix}.mlp.up_proj.bias"] = up_bias
                    processed_keys.add(gate_up_bias_key)

                if inplace:
                    # Drop combined tensor keys eagerly to reduce peak memory.
                    state_dict.pop(gate_up_weight_key, None)
                    state_dict.pop(gate_up_bias_key, None)
                    del gate_up_weight, gate_weight, up_weight
                    if gate_up_bias_key in processed_keys:
                        del gate_up_bias, gate_bias, up_bias

        if not inplace:
            # Copy all other weights that weren't processed
            for key, value in state_dict.items():
                if key not in processed_keys:
                    hf_state_dict[key] = value

        # Apply exclusion regex if provided
        if exclude_key_regex:
            if inplace:
                for k in list(hf_state_dict.keys()):
                    if re.match(exclude_key_regex, k):
                        hf_state_dict.pop(k, None)
            else:
                hf_state_dict = {k: v for k, v in hf_state_dict.items() if not re.match(exclude_key_regex, k)}

        return hf_state_dict
