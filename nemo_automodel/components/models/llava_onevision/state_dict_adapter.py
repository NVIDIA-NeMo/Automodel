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
"""State dict adapter for LLaVA-OneVision-1.5."""

import re
from typing import Any, Optional

from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter


class LlavaOneVisionStateDictAdapter(StateDictAdapter):
    """Converts between HF LLaVA-OneVision checkpoints and NeMo format.

    HF checkpoint key patterns:
      model.visual.{...}          -> Rice ViT weights
      model.language_model.{...}  -> Qwen3 LLM weights
      lm_head.{...}               -> Language model head

    NeMo model key patterns:
      model.vision_tower.{...}    -> Rice ViT weights
      model.language_model.{...}  -> Qwen3 LLM weights
      lm_head.{...}               -> Language model head

    The adapter primarily handles key renaming to bridge HF and NeMo formats.
    """

    def __init__(
        self,
        config: Any,
        **kwargs,
    ):
        self.config = config
        self._uses_model_prefix = True

    def to_hf(
        self,
        state_dict: dict[str, Any],
        exclude_key_regex: Optional[str] = None,
        quantization: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        """Rename NeMo keys to HF keys. Tensors passed through as-is."""
        hf_state_dict = {}
        exclude_pattern = re.compile(exclude_key_regex) if exclude_key_regex else None

        for fqn, tensor in state_dict.items():
            if exclude_pattern and exclude_pattern.match(fqn):
                continue

            hf_key = self._nemo_to_hf_key(fqn)
            hf_state_dict[hf_key] = tensor

        return hf_state_dict

    def from_hf(
        self,
        hf_state_dict: dict[str, Any],
        **kwargs,
    ) -> dict[str, Any]:
        """Rename HF keys to NeMo keys.

        This adapter only performs key renaming. Tensor shapes should match
        since we use the same architecture as the HF implementation.
        """
        self._uses_model_prefix = any(key.startswith("model.") for key in hf_state_dict)

        nemo_state_dict = {}
        for hf_key, tensor in hf_state_dict.items():
            nemo_key = self._hf_to_nemo_key(hf_key)
            nemo_state_dict[nemo_key] = tensor

        return nemo_state_dict

    def _hf_to_nemo_key(self, hf_key: str) -> str:
        """Convert HF checkpoint key to NeMo key."""
        # Handle visual tower
        if hf_key.startswith("model.visual."):
            return hf_key.replace("model.visual.", "model.vision_tower.")

        # Handle language model
        if hf_key.startswith("model.language_model."):
            return hf_key  # Already matches

        # Handle lm_head
        if hf_key.startswith("lm_head."):
            return hf_key  # Already matches

        # Handle top-level model keys
        if hf_key.startswith("model."):
            return "model." + hf_key

        return hf_key

    def _nemo_to_hf_key(self, nemo_key: str) -> str:
        """Convert NeMo key to HF checkpoint key."""
        # Handle visual tower
        if nemo_key.startswith("model.vision_tower."):
            return nemo_key.replace("model.vision_tower.", "model.visual.")

        # Handle language model
        if nemo_key.startswith("model.language_model."):
            return nemo_key  # Already matches

        # Handle lm_head
        if nemo_key.startswith("lm_head."):
            return nemo_key  # Already matches

        return nemo_key
