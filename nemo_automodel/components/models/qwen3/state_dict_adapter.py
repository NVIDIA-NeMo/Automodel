# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""State dict adapter for Qwen3 dense model.

Because Qwen3Attention uses separate q_proj / k_proj / v_proj and the MLP uses
separate gate_proj / up_proj, the native key names already match the HF checkpoint
format exactly -- no weight reshaping or renaming is required.

This adapter is therefore a pass-through: from_hf and to_hf both return the
input unchanged.  It exists so callers that expect a state_dict_adapter attribute
still work without special-casing.
"""

from typing import Any

from transformers import Qwen3Config


class Qwen3StateDictAdapter:
    """Identity (pass-through) state dict adapter for Qwen3 dense models.

    Native format == HF format, so no conversion is needed.
    """

    def __init__(self, config: Qwen3Config):
        self.config = config

    def from_hf(self, hf_state_dict: dict[str, Any], **kwargs) -> dict[str, Any]:
        return hf_state_dict

    def to_hf(self, state_dict: dict[str, Any], **kwargs) -> dict[str, Any]:
        return state_dict

    def convert_single_tensor_to_hf(self, fqn: str, tensor: Any, **kwargs) -> list[tuple[str, Any]]:
        return [(fqn, tensor)]
