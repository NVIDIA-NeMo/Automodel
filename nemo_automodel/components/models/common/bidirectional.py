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

from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter


class EncoderStateDictAdapter(StateDictAdapter):
    """Adapter for EncoderModel state dicts.

    Internal format uses a ``model.`` prefix on all keys.  HF format does not.
    This adapter strips or adds the ``model.`` prefix as needed, including
    for PEFT-wrapped keys (``base_model.model.model.X`` <-> ``base_model.model.X``).
    """

    _PEFT_PREFIX = "base_model.model."

    def __init__(self):
        self._uses_model_prefix = True

    _MODEL_PREFIX = "model."
    _PEFT_MODEL_PREFIX = _PEFT_PREFIX + _MODEL_PREFIX

    def _strip_model_prefix(self, key):
        if key.startswith(self._PEFT_MODEL_PREFIX):
            return self._PEFT_PREFIX + key[len(self._PEFT_MODEL_PREFIX):]
        if key.startswith(self._MODEL_PREFIX):
            return key[len(self._MODEL_PREFIX):]
        return None

    def _add_model_prefix(self, key):
        if key.startswith(self._PEFT_PREFIX):
            return self._PEFT_MODEL_PREFIX + key[len(self._PEFT_PREFIX):]
        return self._MODEL_PREFIX + key

    def to_hf(self, state_dict, **kwargs):
        hf_state_dict = {}
        for key, value in state_dict.items():
            new_key = self._strip_model_prefix(key)
            if new_key is not None:
                hf_state_dict[new_key] = value
        return hf_state_dict

    def from_hf(self, hf_state_dict, device_mesh=None, **kwargs):
        return {self._add_model_prefix(key): value for key, value in hf_state_dict.items()}

    def convert_single_tensor_to_hf(self, fqn, tensor, **kwargs):
        new_fqn = self._strip_model_prefix(fqn)
        if new_fqn is not None:
            return [(new_fqn, tensor)]
        return []


__all__ = [
    "EncoderStateDictAdapter",
]
