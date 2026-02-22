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
        self.group_size = self.num_attention_heads // self.num_key_value_heads

    def _interleave_qkv(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Interleave Q, K, V by KV-head groups for TP-correct ColwiseParallel sharding.

        Layout: [Q_group_0 | K_0 | V_0 | Q_group_1 | K_1 | V_1 | ...]
        where each group has (group_size * head_dim) Q rows, head_dim K rows, head_dim V rows.
        """
        rest = q.shape[1:]
        q = q.reshape(self.num_key_value_heads, self.group_size * self.head_dim, *rest)
        k = k.reshape(self.num_key_value_heads, self.head_dim, *rest)
        v = v.reshape(self.num_key_value_heads, self.head_dim, *rest)
        return torch.cat([q, k, v], dim=1).reshape(-1, *rest)

    def _deinterleave_qkv(self, qkv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """De-interleave QKV from KV-head-grouped layout back to separate Q, K, V.

        Works for both full (unsharded) and TP-sharded tensors — the group structure
        is the same regardless of how many groups this rank holds.
        """
        rest = qkv.shape[1:]
        group_width = (self.group_size + 2) * self.head_dim
        qkv = qkv.reshape(-1, group_width, *rest)
        q, k, v = qkv.split([self.group_size * self.head_dim, self.head_dim, self.head_dim], dim=1)
        return q.reshape(-1, *rest), k.reshape(-1, *rest), v.reshape(-1, *rest)

    def _interleave_gate_up(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        """Interleave gate and up row-by-row for TP-correct ColwiseParallel sharding."""
        return torch.stack([gate, up], dim=1).reshape(-1, *gate.shape[1:])

    def _deinterleave_gate_up(self, gate_up: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """De-interleave gate/up from row-interleaved layout."""
        rest = gate_up.shape[1:]
        gate_up = gate_up.reshape(-1, 2, *rest)
        return gate_up[:, 0].contiguous(), gate_up[:, 1].contiguous()

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

                qkv_weight = self._interleave_qkv(q_weight, k_weight, v_weight)
                custom_state_dict[f"{prefix}.self_attn.qkv_proj.weight"] = qkv_weight
                processed_keys.update([q_weight_key, k_weight_key, v_weight_key])

                # Handle biases if present
                q_bias_key = f"{prefix}.self_attn.q_proj.bias"
                if q_bias_key in hf_state_dict:
                    k_bias_key = f"{prefix}.self_attn.k_proj.bias"
                    v_bias_key = f"{prefix}.self_attn.v_proj.bias"

                    qkv_bias = self._interleave_qkv(
                        hf_state_dict[q_bias_key], hf_state_dict[k_bias_key], hf_state_dict[v_bias_key]
                    )
                    custom_state_dict[f"{prefix}.self_attn.qkv_proj.bias"] = qkv_bias
                    processed_keys.update([q_bias_key, k_bias_key, v_bias_key])

            # Combine gate and up into gate_up_proj
            gate_weight_key = f"{prefix}.mlp.gate_proj.weight"
            up_weight_key = f"{prefix}.mlp.up_proj.weight"

            if gate_weight_key in hf_state_dict:
                gate_weight = hf_state_dict[gate_weight_key]
                up_weight = hf_state_dict[up_weight_key]

                gate_up_weight = self._interleave_gate_up(gate_weight, up_weight)
                custom_state_dict[f"{prefix}.mlp.gate_up_proj.weight"] = gate_up_weight
                processed_keys.update([gate_weight_key, up_weight_key])

                # Handle biases if present
                gate_bias_key = f"{prefix}.mlp.gate_proj.bias"
                if gate_bias_key in hf_state_dict:
                    up_bias_key = f"{prefix}.mlp.up_proj.bias"

                    gate_up_bias = self._interleave_gate_up(hf_state_dict[gate_bias_key], hf_state_dict[up_bias_key])
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
                q_weight, k_weight, v_weight = self._deinterleave_qkv(state_dict[qkv_weight_key])

                hf_state_dict[f"{prefix}.self_attn.q_proj.weight"] = q_weight
                hf_state_dict[f"{prefix}.self_attn.k_proj.weight"] = k_weight
                hf_state_dict[f"{prefix}.self_attn.v_proj.weight"] = v_weight
                processed_keys.add(qkv_weight_key)

                # Handle biases if present
                qkv_bias_key = f"{prefix}.self_attn.qkv_proj.bias"
                if qkv_bias_key in state_dict:
                    q_bias, k_bias, v_bias = self._deinterleave_qkv(state_dict[qkv_bias_key])

                    hf_state_dict[f"{prefix}.self_attn.q_proj.bias"] = q_bias
                    hf_state_dict[f"{prefix}.self_attn.k_proj.bias"] = k_bias
                    hf_state_dict[f"{prefix}.self_attn.v_proj.bias"] = v_bias
                    processed_keys.add(qkv_bias_key)

            # Split gate_up_proj into separate gate and up
            gate_up_weight_key = f"{prefix}.mlp.gate_up_proj.weight"

            if gate_up_weight_key in state_dict:
                gate_weight, up_weight = self._deinterleave_gate_up(state_dict[gate_up_weight_key])

                hf_state_dict[f"{prefix}.mlp.gate_proj.weight"] = gate_weight
                hf_state_dict[f"{prefix}.mlp.up_proj.weight"] = up_weight
                processed_keys.add(gate_up_weight_key)

                # Handle biases if present
                gate_up_bias_key = f"{prefix}.mlp.gate_up_proj.bias"
                if gate_up_bias_key in state_dict:
                    gate_bias, up_bias = self._deinterleave_gate_up(state_dict[gate_up_bias_key])

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

    def _split_remaining_combined_projection_keys(self, hf_state_dict: dict[str, Any]) -> None:
        """Split any remaining combined-projection keys in-place.

        Handles LoRA adapter weights (lora_A, lora_B), DoRA magnitude vectors,
        and any base weight/bias keys that weren't caught by the layer-indexed loop
        (e.g., keys with a ``base_model.model.`` prefix from PEFT saving).

        For keys containing ``.self_attn.qkv_proj.``:
          - ``lora_A`` weights (input dimension) are duplicated to q/k/v projections.
          - All other weights (lora_B, magnitude, weight, bias) are split along dim 0
            using the Q/KV size ratio.

        For keys containing ``.mlp.gate_up_proj.``:
          - ``lora_A`` weights are duplicated to gate/up projections.
          - All other weights are split in half along dim 0.

        Args:
            hf_state_dict: State dict to modify in-place.
        """
        combined_qkv_keys = [k for k in hf_state_dict if ".self_attn.qkv_proj." in k]
        for key in combined_qkv_keys:
            value = hf_state_dict.pop(key)
            pre, suffix = key.split(".self_attn.qkv_proj.", 1)

            if "lora_A" in suffix:
                # Input-dimension LoRA weight: identical for all projections
                hf_state_dict[f"{pre}.self_attn.q_proj.{suffix}"] = value
                hf_state_dict[f"{pre}.self_attn.k_proj.{suffix}"] = value.clone()
                hf_state_dict[f"{pre}.self_attn.v_proj.{suffix}"] = value.clone()
            else:
                # Output-dimension weight (lora_B, magnitude, base weight/bias): de-interleave
                q_val, k_val, v_val = self._deinterleave_qkv(value)
                hf_state_dict[f"{pre}.self_attn.q_proj.{suffix}"] = q_val
                hf_state_dict[f"{pre}.self_attn.k_proj.{suffix}"] = k_val
                hf_state_dict[f"{pre}.self_attn.v_proj.{suffix}"] = v_val

        combined_gate_up_keys = [k for k in hf_state_dict if ".mlp.gate_up_proj." in k]
        for key in combined_gate_up_keys:
            value = hf_state_dict.pop(key)
            pre, suffix = key.split(".mlp.gate_up_proj.", 1)

            if "lora_A" in suffix:
                hf_state_dict[f"{pre}.mlp.gate_proj.{suffix}"] = value
                hf_state_dict[f"{pre}.mlp.up_proj.{suffix}"] = value.clone()
            else:
                # Output-dimension weight: de-interleave
                gate_val, up_val = self._deinterleave_gate_up(value)
                hf_state_dict[f"{pre}.mlp.gate_proj.{suffix}"] = gate_val
                hf_state_dict[f"{pre}.mlp.up_proj.{suffix}"] = up_val

    def _recombine_split_projection_keys(self, state_dict: dict[str, Any]) -> None:
        """Recombine split projection LoRA/DoRA keys back to combined format.

        This is the reverse of ``_split_remaining_combined_projection_keys``.
        It handles LoRA adapter weights and DoRA magnitude vectors that were
        split for HF-PEFT compatibility during ``to_hf()`` and need to be
        recombined when loading back into a model with combined projections.

        For keys containing ``.self_attn.q_proj.<suffix>``:
          - ``lora_A`` weights (which were duplicated during split) are
            deduplicated — we take the ``q_proj`` version.
          - All other weights (``lora_B``, magnitude, etc.) are concatenated
            along dim 0 in Q, K, V order.

        For keys containing ``.mlp.gate_proj.<suffix>``:
          - ``lora_A`` weights are deduplicated — we take the ``gate_proj`` version.
          - All other weights are concatenated along dim 0 in gate, up order.

        Keys that end with ``.weight`` or ``.bias`` directly on the projection
        (e.g., ``q_proj.weight``) are skipped because those are already handled
        by the layer-indexed loop in ``from_hf``.

        Args:
            state_dict: State dict to modify in-place.
        """
        # --- QKV recombination ---
        # Find q_proj keys that are NOT base weight/bias (already handled by layer loop)
        q_keys = [
            k
            for k in list(state_dict.keys())
            if ".self_attn.q_proj." in k
            and not k.endswith(".self_attn.q_proj.weight")
            and not k.endswith(".self_attn.q_proj.bias")
        ]

        for q_key in q_keys:
            k_key = q_key.replace(".self_attn.q_proj.", ".self_attn.k_proj.")
            v_key = q_key.replace(".self_attn.q_proj.", ".self_attn.v_proj.")

            if k_key not in state_dict or v_key not in state_dict:
                continue

            combined_key = q_key.replace(".self_attn.q_proj.", ".self_attn.qkv_proj.")

            q_val = state_dict.pop(q_key)
            k_val = state_dict.pop(k_key)
            v_val = state_dict.pop(v_key)

            if "lora_A" in q_key:
                # lora_A weights were duplicated during split — just take one
                state_dict[combined_key] = q_val
            else:
                # lora_B, magnitude, etc. — interleave by KV-head groups
                state_dict[combined_key] = self._interleave_qkv(q_val, k_val, v_val)

        # --- gate_up recombination ---
        gate_keys = [
            k
            for k in list(state_dict.keys())
            if ".mlp.gate_proj." in k
            and not k.endswith(".mlp.gate_proj.weight")
            and not k.endswith(".mlp.gate_proj.bias")
        ]

        for gate_key in gate_keys:
            up_key = gate_key.replace(".mlp.gate_proj.", ".mlp.up_proj.")

            if up_key not in state_dict:
                continue

            combined_key = gate_key.replace(".mlp.gate_proj.", ".mlp.gate_up_proj.")

            gate_val = state_dict.pop(gate_key)
            up_val = state_dict.pop(up_key)

            if "lora_A" in gate_key:
                # lora_A weights were duplicated during split — just take one
                state_dict[combined_key] = gate_val
            else:
                # lora_B, magnitude, etc. — interleave row-by-row
                state_dict[combined_key] = self._interleave_gate_up(gate_val, up_val)
