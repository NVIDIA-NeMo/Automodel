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
Bidirectional model factory utilities.

This module provides factory functions for creating bidirectional variants
of transformer models by disabling causal masking. This enables models
to attend to all tokens bidirectionally, which is useful for embedding
and retrieval tasks.
"""

from typing import Any, Optional, Type

from torch.distributed.device_mesh import DeviceMesh
from transformers import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig

from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter


def make_bidirectional(model: PreTrainedModel) -> PreTrainedModel:
    """
    Disable causal masking on any transformer model.

    This function iterates through the model's layers and sets is_causal=False
    on all attention modules, allowing tokens to attend bidirectionally.

    Args:
        model: A pretrained transformer model with attention layers.

    Returns:
        The same model with causal masking disabled on all attention layers.

    Example:
        >>> from transformers import LlamaModel
        >>> model = LlamaModel.from_pretrained("meta-llama/Llama-3.2-1B")
        >>> model = make_bidirectional(model)
    """
    # Handle models with 'layers' attribute (Llama, Qwen, Mistral, etc.)
    layers = getattr(model, "layers", None)
    if layers is not None:
        for layer in layers:
            if hasattr(layer, "self_attn"):
                layer.self_attn.is_causal = False

    # Handle models with 'encoder.layer' attribute (BERT-style)
    encoder = getattr(model, "encoder", None)
    if encoder is not None:
        encoder_layers = getattr(encoder, "layer", None)
        if encoder_layers is not None:
            for layer in encoder_layers:
                attention = getattr(layer, "attention", None)
                if attention is not None and hasattr(attention, "self"):
                    attention.self.is_causal = False

    return model


def create_bidirectional_model_class(
    base_model_class: Type[PreTrainedModel],
    base_config_class: Type[PretrainedConfig],
    model_type_suffix: str = "_bidirec",
) -> Type[PreTrainedModel]:
    """
    Factory to create a bidirectional variant of any transformer model.

    This function dynamically creates a new class that inherits from the base
    model class but disables causal masking in the __init__ method.

    Args:
        base_model_class: The base model class (e.g., LlamaModel, Qwen2Model).
        base_config_class: The config class for the base model.
        model_type_suffix: Suffix to append to the model_type for the config.

    Returns:
        A new class that is a bidirectional variant of the base model.

    Example:
        >>> from transformers.models.llama.modeling_llama import LlamaModel
        >>> from transformers.models.llama.configuration_llama import LlamaConfig
        >>> LlamaBidirectionalModel = create_bidirectional_model_class(
        ...     LlamaModel, LlamaConfig
        ... )
    """
    # Get the original model type from the config
    original_model_type = getattr(base_config_class, "model_type", "unknown")

    # Create the bidirectional config class
    class BidirectionalConfig(base_config_class):
        model_type = f"{original_model_type}{model_type_suffix}"

        def __init__(self, pooling: str = "avg", temperature: float = 1.0, **kwargs):
            self.pooling = pooling
            self.temperature = temperature
            super().__init__(**kwargs)

    # Create the bidirectional model class
    class BidirectionalModel(base_model_class):
        config_class = BidirectionalConfig

        def __init__(self, config):
            super().__init__(config)
            # Disable causal attention for all layers after initialization
            make_bidirectional(self)

    # Set meaningful class names
    base_name = base_model_class.__name__
    BidirectionalConfig.__name__ = f"{base_name.replace('Model', '')}BidirectionalConfig"
    BidirectionalConfig.__qualname__ = BidirectionalConfig.__name__
    BidirectionalModel.__name__ = f"{base_name.replace('Model', '')}BidirectionalModel"
    BidirectionalModel.__qualname__ = BidirectionalModel.__name__

    # Attach config class reference
    BidirectionalModel.config_class = BidirectionalConfig

    return BidirectionalModel, BidirectionalConfig


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


__all__ = [
    "make_bidirectional",
    "create_bidirectional_model_class",
    "BiencoderStateDictAdapter",
]
