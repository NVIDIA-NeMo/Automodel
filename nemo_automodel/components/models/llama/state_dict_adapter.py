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

Uses the generic CombinedProjectionStateDictAdapter from common/.
"""

import logging
from typing import Any, Optional

from transformers import LlamaConfig

from nemo_automodel.components.models.common.combined_projection.state_dict_adapter import (
    CombinedProjectionStateDictAdapter,
)

logger = logging.getLogger(__name__)


class LlamaStateDictAdapter(CombinedProjectionStateDictAdapter):
    """State dict adapter for Llama models.

    Inherits from the generic CombinedProjectionStateDictAdapter,
    providing a clean interface specific to Llama.

    When ``config.use_combined_projections`` is ``False`` the model uses
    separate ``q_proj`` / ``k_proj`` / ``v_proj`` / ``gate_proj`` / ``up_proj``
    that match HuggingFace key names exactly, so the combining / splitting steps
    are skipped and the state dict is passed through as-is (only tied-weight
    handling is applied).

    Example:
        from transformers import LlamaConfig

        config = LlamaConfig.from_pretrained("meta-llama/Llama-3-8B")
        adapter = LlamaStateDictAdapter(config)

        # Convert HF checkpoint to custom format
        custom_state_dict = adapter.from_hf(hf_state_dict)

        # Convert custom checkpoint back to HF format
        hf_state_dict = adapter.to_hf(custom_state_dict)
    """

    def __init__(self, config: LlamaConfig):
        """Initialize adapter with Llama config."""
        super().__init__(config)
        self._use_combined = getattr(config, "use_combined_projections", True)

    def from_hf(self, hf_state_dict: dict[str, Any], **kwargs) -> dict[str, Any]:
        if self._use_combined:
            return super().from_hf(hf_state_dict, **kwargs)
        # Separate-projection mode: HF keys match model keys directly.
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
        if self._use_combined:
            return super().to_hf(state_dict, exclude_key_regex=exclude_key_regex, **kwargs)
        # Separate-projection mode: model keys are already in HF format.
        if exclude_key_regex is not None:
            import re
            return {k: v for k, v in state_dict.items() if not re.search(exclude_key_regex, k)}
        return dict(state_dict)
