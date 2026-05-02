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

"""State dict adapter for Qwen2 model.

The model uses separate q/k/v and gate/up projections that match HuggingFace key
names exactly, so the adapter is a passthrough (only tied-weight handling in
from_hf).
"""

import logging
import re
from typing import Any, Optional

from transformers import Qwen2Config

logger = logging.getLogger(__name__)


class Qwen2StateDictAdapter:
    """State dict adapter for Qwen2 models.

    Uses separate projections that match HuggingFace key names exactly, so
    from_hf / to_hf are simple passthroughs (only tied-weight handling in
    from_hf).

    Example:
        from transformers import Qwen2Config

        config = Qwen2Config.from_pretrained("Qwen/Qwen2.5-7B")
        adapter = Qwen2StateDictAdapter(config)

        # Convert HF checkpoint to custom format
        custom_state_dict = adapter.from_hf(hf_state_dict)

        # Convert custom checkpoint back to HF format
        hf_state_dict = adapter.to_hf(custom_state_dict)
    """

    def __init__(self, config: Qwen2Config):
        """Initialize adapter with Qwen2 config."""
        self.config = config

    def from_hf(self, hf_state_dict: dict[str, Any], **kwargs) -> dict[str, Any]:
        # HF keys match model keys directly.
        # Only need to handle tied lm_head weights.
        custom_state_dict = dict(hf_state_dict)
        if getattr(self.config, "tie_word_embeddings", True):
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
        # Model keys are already in HF format.
        if exclude_key_regex is not None:
            return {k: v for k, v in state_dict.items() if not re.search(exclude_key_regex, k)}
        return dict(state_dict)
