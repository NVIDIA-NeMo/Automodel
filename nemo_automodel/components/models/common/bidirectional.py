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

from typing import Any, Optional

from torch.distributed.device_mesh import DeviceMesh

from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter


class EncoderStateDictAdapter(StateDictAdapter):
    """Adapter for converting EncoderModel state dict to/from HuggingFace format.

    On save, maps ``lm_q.`` prefixed keys to ``model.`` prefixed keys.
    On load, maps ``model.`` prefixed keys back to ``lm_q.`` prefixed keys.
    PEFT-prefixed keys (``base_model.model.``) are handled transparently.
    """

    _PEFT_PREFIX = "base_model.model."

    def __init__(self):
        self._uses_model_prefix = True

    @staticmethod
    def _swap_key(key: str, src: str, dst: str, peft_prefix: str) -> Optional[str]:
        """Return *key* with *src* prefix replaced by *dst*, handling an optional PEFT wrapper.

        Returns ``None`` when *key* doesn't match *src* (bare or PEFT-wrapped).
        """
        if key.startswith(src):
            return dst + key[len(src) :]
        peft_src = peft_prefix + src
        if key.startswith(peft_src):
            return peft_prefix + dst + key[len(peft_src) :]
        return None

    def to_hf(self, state_dict: dict[str, Any], **kwargs) -> dict[str, Any]:
        """Convert encoder state dict to HF format (lm_q -> model)."""
        hf_state_dict = {}
        for key, value in state_dict.items():
            new_key = self._swap_key(key, "lm_q.", "model.", self._PEFT_PREFIX)
            if new_key is not None:
                hf_state_dict[new_key] = value
        return hf_state_dict

    def from_hf(
        self,
        hf_state_dict: dict[str, Any],
        device_mesh: Optional["DeviceMesh"] = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Convert HF state dict to encoder format (model -> lm_q)."""
        encoder_state_dict = {}
        for key, value in hf_state_dict.items():
            q_key = self._swap_key(key, "model.", "lm_q.", self._PEFT_PREFIX)
            if q_key is not None:
                encoder_state_dict[q_key] = value
        return encoder_state_dict

    def convert_single_tensor_to_hf(self, fqn: str, tensor: Any, **kwargs) -> list[tuple[str, Any]]:
        """Convert a single tensor from encoder to HF format. Skips non-lm_q tensors."""
        if fqn.startswith("lm_q."):
            return [("model." + fqn[len("lm_q.") :], tensor)]
        return []


__all__ = [
    "EncoderStateDictAdapter",
]
