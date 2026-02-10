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

from typing import Any, Optional, Tuple, Type

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
) -> Tuple[Type[PreTrainedModel], Type[PretrainedConfig]]:
    """
    Factory to create a bidirectional variant of any transformer model.

    This function dynamically creates a new class that inherits from the base
    model class but disables causal masking in the __init__ method.

    Args:
        base_model_class: The base model class (e.g., LlamaModel, Qwen2Model).
        base_config_class: The config class for the base model.
        model_type_suffix: Suffix to append to the model_type for the config.

    Returns:
        A tuple of (BidirectionalModel, BidirectionalConfig) classes.

    Example:
        >>> from transformers.models.llama.modeling_llama import LlamaModel
        >>> from transformers.models.llama.configuration_llama import LlamaConfig
        >>> LlamaBidirectionalModel, LlamaBidirectionalConfig = create_bidirectional_model_class(
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
    """Adapter for converting BiencoderModel state dict to/from single-encoder HF format.

    Extracts only the query encoder (lm_q) on save, mapping ``lm_q.`` to ``model.``.
    On load, fans ``model.`` keys back out to both ``lm_q.`` and ``lm_p.``.
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
            return dst + key[len(src):]
        peft_src = peft_prefix + src
        if key.startswith(peft_src):
            return peft_prefix + dst + key[len(peft_src):]
        return None

    @staticmethod
    def _is_pooler_key(key: str, peft_prefix: str) -> bool:
        return key.startswith("linear_pooler.") or key.startswith(peft_prefix + "linear_pooler.")

    def to_hf(self, state_dict: dict[str, Any], **kwargs) -> dict[str, Any]:
        """Convert biencoder state dict to HF format (lm_q -> model)."""
        hf_state_dict = {}
        for key, value in state_dict.items():
            new_key = self._swap_key(key, "lm_q.", "model.", self._PEFT_PREFIX)
            if new_key is not None:
                hf_state_dict[new_key] = value
            elif self._is_pooler_key(key, self._PEFT_PREFIX):
                hf_state_dict[key] = value
        return hf_state_dict

    def from_hf(
        self,
        hf_state_dict: dict[str, Any],
        device_mesh: Optional["DeviceMesh"] = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Convert HF state dict to biencoder format (model -> lm_q + lm_p)."""
        biencoder_state_dict = {}
        for key, value in hf_state_dict.items():
            q_key = self._swap_key(key, "model.", "lm_q.", self._PEFT_PREFIX)
            if q_key is not None:
                p_key = self._swap_key(key, "model.", "lm_p.", self._PEFT_PREFIX)
                biencoder_state_dict[q_key] = value
                biencoder_state_dict[p_key] = value
            elif self._is_pooler_key(key, self._PEFT_PREFIX):
                biencoder_state_dict[key] = value
        return biencoder_state_dict

    def convert_single_tensor_to_hf(self, fqn: str, tensor: Any, **kwargs) -> list[tuple[str, Any]]:
        """Convert a single tensor from biencoder to HF format. Skips non-lm_q tensors."""
        if fqn.startswith("lm_q."):
            return [("model." + fqn[len("lm_q."):], tensor)]
        if fqn.startswith("linear_pooler."):
            return [(fqn, tensor)]
        return []


__all__ = [
    "make_bidirectional",
    "create_bidirectional_model_class",
    "BiencoderStateDictAdapter",
]
