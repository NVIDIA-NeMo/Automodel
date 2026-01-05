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

from typing import Any, Optional

from torch.distributed.device_mesh import DeviceMesh

from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter


class BiencoderStateDictAdapter(StateDictAdapter):
    """Adapter for converting BiencoderModel state dict to single encoder format.

    This adapter extracts only the query encoder (lm_q) state dict and converts
    the "lm_q." prefix to "model." prefix, making it compatible with standard
    HuggingFace model format.
    """

    def __init__(self):
        """Initialize the adapter."""
        self._uses_model_prefix = True

    def to_hf(self, state_dict: dict[str, Any], **kwargs) -> dict[str, Any]:
        """Convert from biencoder state dict to HuggingFace format.

        Filters to only lm_q keys and converts "lm_q." prefix to "model." prefix.

        Args:
            state_dict: The biencoder model state dict

        Returns:
            The converted HuggingFace format state dict with only query encoder
        """
        hf_state_dict = {}

        for key, value in state_dict.items():
            if key.startswith("lm_q."):
                new_key = key.replace("lm_q.", "model.")
                hf_state_dict[new_key] = value
            elif key.startswith("linear_pooler."):
                hf_state_dict[key] = value

        return hf_state_dict

    def from_hf(
        self,
        hf_state_dict: dict[str, Any],
        device_mesh: Optional["DeviceMesh"] = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Convert HuggingFace state dict to biencoder format.

        Converts "model." prefix to "lm_q." prefix for loading into biencoder.

        Args:
            hf_state_dict: The HuggingFace format state dict
            device_mesh: Optional device mesh (not used in this adapter)

        Returns:
            The converted biencoder format state dict
        """
        biencoder_state_dict = {}

        for key, value in hf_state_dict.items():
            if key.startswith("model."):
                new_key_q = key.replace("model.", "lm_q.")
                biencoder_state_dict[new_key_q] = value
                new_key_p = key.replace("model.", "lm_p.")
                biencoder_state_dict[new_key_p] = value
            elif key.startswith("linear_pooler."):
                biencoder_state_dict[key] = value

        return biencoder_state_dict

    def convert_single_tensor_to_hf(self, fqn: str, tensor: Any, **kwargs) -> list[tuple[str, Any]]:
        """Convert a single tensor from biencoder format to HuggingFace format.

        Args:
            fqn: Fully qualified name of the tensor in biencoder format
            tensor: The tensor to convert
            **kwargs: Additional arguments (unused)

        Returns:
            List of (fqn, tensor) tuples in HuggingFace format.
            Returns empty list if tensor is not part of lm_q.
        """
        if fqn.startswith("lm_q."):
            new_fqn = fqn.replace("lm_q.", "model.")
            return [(new_fqn, tensor)]
        if fqn.startswith("linear_pooler."):
            return [(fqn, tensor)]

        # Skip tensors that are not part of lm_q
        return []
