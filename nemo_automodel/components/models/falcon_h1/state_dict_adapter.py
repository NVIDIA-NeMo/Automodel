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

"""State dict adapter for Falcon-H1.

Converts between HuggingFace's checkpoint key naming and the keys this
implementation produces. The two layouts agree on every key except the
final layernorm, which HF calls 'final_layernorm' and we call 'norm'.
"""

from typing import Any

from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter

# One-to-one key renames. Left = HF, right = ours.
_HF_TO_LOCAL = {
    "model.final_layernorm.weight": "model.norm.weight",
}

# Reverse mapping for to_hf
_LOCAL_TO_HF = {v: k for k, v in _HF_TO_LOCAL.items()}


class FalconH1StateDictAdapter(StateDictAdapter):
    """Maps Falcon-H1 HF checkpoint keys to/from this implementation's keys."""

    def __init__(self, config, **kwargs):
        self.config = config

    def from_hf(self, hf_state_dict: dict[str, Any], **kwargs) -> dict[str, Any]:
        """Translate an HF state dict into local keys."""
        return {_HF_TO_LOCAL.get(k, k): v for k, v in hf_state_dict.items()}

    def to_hf(self, state_dict: dict[str, Any], **kwargs) -> dict[str, Any]:
        """Translate a local state dict into HF keys."""
        return {_LOCAL_TO_HF.get(k, k): v for k, v in state_dict.items()}

    def convert_single_tensor_to_hf(self, fqn: str, tensor: Any, **kwargs) -> list[tuple[str, Any]]:
        """Convert one native tensor to HF format (1:1, single rename)."""
        return [(_LOCAL_TO_HF.get(fqn, fqn), tensor)]
