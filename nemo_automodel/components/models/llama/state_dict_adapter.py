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

"""State dict adapter for Llama model.

The PyTorch backend uses HuggingFace projection names directly. The QuACK MLP
backend interleaves gate/up rows in ``fc1`` and renames ``down_proj`` to ``fc2``;
the adapter converts that layout without changing the HuggingFace checkpoint
contract.
"""

import logging
import re
from typing import Any, Optional

import torch
from transformers import LlamaConfig

logger = logging.getLogger(__name__)


class LlamaStateDictAdapter:
    """State dict adapter for Llama models.

    Uses HuggingFace projection names directly for the PyTorch backend and
    converts the fused, interleaved QuACK MLP weight layout when requested.

    Example:
        from transformers import LlamaConfig

        config = LlamaConfig.from_pretrained("meta-llama/Llama-3-8B")
        adapter = LlamaStateDictAdapter(config)

        # Convert HF checkpoint to custom format
        custom_state_dict = adapter.from_hf(hf_state_dict)

        # Convert custom checkpoint back to HF format
        hf_state_dict = adapter.to_hf(custom_state_dict)
    """

    def __init__(self, config: LlamaConfig, mlp_backend: str = "torch"):
        """Initialize adapter with Llama config."""
        self.config = config
        self.mlp_backend = mlp_backend

    def from_hf(self, hf_state_dict: dict[str, Any], **kwargs) -> dict[str, Any]:
        # HF keys match model keys directly.
        # Only need to handle tied lm_head weights.
        custom_state_dict = dict(hf_state_dict)
        if self.mlp_backend == "quack":
            layer_ids = {
                match.group(1)
                for key in custom_state_dict
                if (match := re.match(r"model\.layers\.(\d+)\.mlp\.gate_proj\.weight$", key))
            }
            for layer_id in layer_ids:
                prefix = f"model.layers.{layer_id}.mlp"
                gate = custom_state_dict.pop(f"{prefix}.gate_proj.weight")
                up = custom_state_dict.pop(f"{prefix}.up_proj.weight")
                custom_state_dict[f"{prefix}.fc1.weight"] = torch.stack((gate, up), dim=1).flatten(0, 1)
                custom_state_dict[f"{prefix}.fc2.weight"] = custom_state_dict.pop(f"{prefix}.down_proj.weight")
        # Default False to match __init__/tie_weights (config always carries the flag).
        if getattr(self.config, "tie_word_embeddings", False):
            embed_key = "model.embed_tokens.weight"
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
        hf_state_dict = dict(state_dict)
        if self.mlp_backend == "quack":
            layer_ids = {
                match.group(1)
                for key in hf_state_dict
                if (match := re.match(r"model\.layers\.(\d+)\.mlp\.fc1\.weight$", key))
            }
            for layer_id in layer_ids:
                prefix = f"model.layers.{layer_id}.mlp"
                gate_up = hf_state_dict.pop(f"{prefix}.fc1.weight")
                hf_state_dict[f"{prefix}.gate_proj.weight"] = gate_up[::2]
                hf_state_dict[f"{prefix}.up_proj.weight"] = gate_up[1::2]
                hf_state_dict[f"{prefix}.down_proj.weight"] = hf_state_dict.pop(f"{prefix}.fc2.weight")
        if exclude_key_regex is not None:
            hf_state_dict = {k: v for k, v in hf_state_dict.items() if not re.search(exclude_key_regex, k)}
        return hf_state_dict
