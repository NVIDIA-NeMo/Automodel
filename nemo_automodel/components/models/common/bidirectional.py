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

"""
Bidirectional model state dict adapter utilities.

This module provides the EncoderStateDictAdapter for converting between
encoder and HuggingFace state dict formats.
"""

from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter


class EncoderStateDictAdapter(StateDictAdapter):
    """Identity adapter for EncoderModel state dicts.

    Internal and HF formats both use ``model.`` prefix. The adapter filters
    to ``model.``-prefixed keys (including PEFT-wrapped variants) and passes
    them through unchanged.
    """

    _PEFT_PREFIX = "base_model.model."

    def __init__(self):
        self._uses_model_prefix = True

    def to_hf(self, state_dict, **kwargs):
        return {
            k: v for k, v in state_dict.items() if k.startswith("model.") or k.startswith(self._PEFT_PREFIX + "model.")
        }

    def from_hf(self, hf_state_dict, device_mesh=None, **kwargs):
        return {
            k: v
            for k, v in hf_state_dict.items()
            if k.startswith("model.") or k.startswith(self._PEFT_PREFIX + "model.")
        }

    def convert_single_tensor_to_hf(self, fqn, tensor, **kwargs):
        if fqn.startswith("model."):
            return [(fqn, tensor)]
        return []


__all__ = [
    "EncoderStateDictAdapter",
]
