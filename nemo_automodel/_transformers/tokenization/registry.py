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

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, Type, Union

from nemo_automodel._transformers.tokenization.nemo_auto_tokenizer import AutoTokenizerWithBosEosEnforced


logger = logging.getLogger(__name__)


@dataclass
class _TokenizerRegistry:
    """
    Registry for custom tokenizer implementations.

    Maps model types (from config) to tokenizer classes or factory functions.
    """

    # Maps model_type -> tokenizer class or factory function
    model_type_to_tokenizer: Dict[str, Union[Type, Callable]] = field(default_factory=dict)

    # Default tokenizer class when no custom implementation is found
    default_tokenizer_cls: Type = AutoTokenizerWithBosEosEnforced

    def register(self, model_type: str, tokenizer_cls: Union[Type, Callable]) -> None:
        """
        Register a custom tokenizer for a specific model type.

        Args:
            model_type: The model type string (e.g., "mistral", "llama")
            tokenizer_cls: The tokenizer class or factory function
        """
        self.model_type_to_tokenizer[model_type] = tokenizer_cls
        logger.debug(f"Registered tokenizer {tokenizer_cls} for model type '{model_type}'")

    def get_tokenizer_cls(self, model_type: str) -> Union[Type, Callable]:
        """
        Get the tokenizer class for a given model type.

        Args:
            model_type: The model type string

        Returns:
            The registered tokenizer class, or the default if not found
        """
        return self.model_type_to_tokenizer.get(model_type, self.default_tokenizer_cls)

    def has_custom_tokenizer(self, model_type: str) -> bool:
        """Check if a custom tokenizer is registered for the given model type."""
        return model_type in self.model_type_to_tokenizer


# Global tokenizer registry
TokenizerRegistry = _TokenizerRegistry()


def _register_default_tokenizers():
    """Register default custom tokenizer implementations."""
    try:
        from nemo_automodel._transformers.tokenization.tokenization_mistral_common import MistralCommonBackend

        # Register for Mistral model types
        TokenizerRegistry.register("mistral", MistralCommonBackend)
        TokenizerRegistry.register("pixtral", MistralCommonBackend)
        TokenizerRegistry.register("mistral3", MistralCommonBackend)
    except ImportError:
        logger.debug("MistralCommonBackend not available, skipping registration")


# Register defaults on module load
_register_default_tokenizers()


__all__ = [
    "TokenizerRegistry",
]

