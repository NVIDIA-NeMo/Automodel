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

import torch

from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter


class EncoderStateDictAdapter(StateDictAdapter):
    """Adapter for encoder model state dicts.

    Internal format uses a ``model.`` prefix on all keys.  HF format does not.
    This adapter strips or adds the ``model.`` prefix as needed, including
    for PEFT-wrapped keys (``base_model.model.model.X`` <-> ``base_model.model.X``).
    """

    _PEFT_PREFIX = "base_model.model."

    def __init__(self, backbone_adapter: StateDictAdapter | None = None) -> None:
        self._uses_model_prefix = True
        self.backbone_adapter = backbone_adapter

    _MODEL_PREFIX = "model."
    _PEFT_MODEL_PREFIX = _PEFT_PREFIX + _MODEL_PREFIX

    def _strip_model_prefix(self, key):
        if key.startswith(self._PEFT_MODEL_PREFIX):
            return self._PEFT_PREFIX + key[len(self._PEFT_MODEL_PREFIX) :]
        if key.startswith(self._MODEL_PREFIX):
            return key[len(self._MODEL_PREFIX) :]
        return None

    def _add_model_prefix(self, key):
        if key.startswith(self._PEFT_PREFIX):
            return self._PEFT_MODEL_PREFIX + key[len(self._PEFT_PREFIX) :]
        return self._MODEL_PREFIX + key

    def to_hf(self, state_dict, **kwargs):
        """Remove the wrapper prefix, then apply the backbone conversion.

        Args:
            state_dict: Native tensors in the wrapped model's documented layouts and distributed placements.
            **kwargs: Options passed to the backbone adapter.

        Returns:
            HF tensors in the backbone adapter's documented layouts, or unchanged tensors without an adapter.
        """
        hf_state_dict = {}
        for key, value in state_dict.items():
            new_key = self._strip_model_prefix(key)
            if new_key is not None:
                hf_state_dict[new_key] = value
        if self.backbone_adapter is not None:
            return self.backbone_adapter.to_hf(hf_state_dict, **kwargs)
        return hf_state_dict

    def from_hf(self, hf_state_dict, device_mesh=None, **kwargs):
        """Apply the backbone conversion before restoring the wrapper prefix.

        Args:
            hf_state_dict: HF tensors in the backbone adapter's documented layouts.
            device_mesh: Mesh forwarded to the backbone adapter without modification.
            **kwargs: Options passed to the backbone adapter.

        Returns:
            Native tensors in the wrapped model's layouts and distributed placements.
        """
        if self.backbone_adapter is not None:
            hf_state_dict = self.backbone_adapter.from_hf(hf_state_dict, device_mesh=device_mesh, **kwargs)
        return {self._add_model_prefix(key): value for key, value in hf_state_dict.items()}

    def forced_hf_dtype_mapping(self, state_dict: dict[str, torch.Tensor]) -> dict[str, str]:
        """Preserve backbone export dtype requirements through the encoder wrapper.

        Args:
            state_dict: HF parameter names and tensors in their unchanged model-specific layouts.

        Returns:
            HF parameter names mapped to safetensors dtype names required by the backbone adapter.
        """
        forced_mapping = getattr(self.backbone_adapter, "forced_hf_dtype_mapping", None)
        return forced_mapping(state_dict) if callable(forced_mapping) else {}

    def convert_single_tensor_to_hf(self, fqn, tensor, **kwargs):
        """Convert a wrapped parameter using the backbone's per-tensor contract.

        Args:
            fqn: Wrapped native parameter name.
            tensor: Tensor in its native model-specific shape, dtype, device and distributed placement.
            **kwargs: Options passed to the backbone adapter.

        Returns:
            HF name/tensor pairs following the backbone adapter's layouts; without an adapter the tensor aliases input.
        """
        new_fqn = self._strip_model_prefix(fqn)
        if new_fqn is not None:
            if self.backbone_adapter is not None:
                return self.backbone_adapter.convert_single_tensor_to_hf(new_fqn, tensor, **kwargs)
            return [(new_fqn, tensor)]
        return []


__all__ = [
    "EncoderStateDictAdapter",
]
