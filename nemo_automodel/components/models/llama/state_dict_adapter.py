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

from transformers import LlamaConfig

from nemo_automodel.components.models.common.combined_projection.state_dict_adapter import (
    CombinedProjectionStateDictAdapter,
)


class LlamaStateDictAdapter(CombinedProjectionStateDictAdapter):
    """State dict adapter for Llama models.

    Inherits from the generic CombinedProjectionStateDictAdapter,
    providing a clean interface specific to Llama.

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
