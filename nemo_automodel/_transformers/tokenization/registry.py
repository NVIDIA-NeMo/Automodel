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

import importlib
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Type, Union

logger = logging.getLogger(__name__)

TokenizerImpl = Union[Type[Any], Callable[..., Any], str]

_DEFAULT_TOKENIZER_IMPL: TokenizerImpl = (
    "nemo_automodel._transformers.tokenization.nemo_auto_tokenizer:NeMoAutoTokenizerWithBosEosEnforced"
)


def _resolve_tokenizer_impl(tokenizer_impl: TokenizerImpl) -> Union[Type[Any], Callable[..., Any]]:
    """
    Resolve a tokenizer implementation.

    Supports:
    - concrete classes / callables
    - import strings of the form "some.module:SomeClass"
    """
    if isinstance(tokenizer_impl, str):
        module_name, sep, attr_name = tokenizer_impl.partition(":")
        if not sep or not module_name or not attr_name:
            raise ValueError(
                f"Invalid tokenizer import path {tokenizer_impl!r}. Expected format: 'some.module:SomeClass'"
            )
        module = importlib.import_module(module_name)
        return getattr(module, attr_name)
    return tokenizer_impl


@dataclass
class _TokenizerRegistry:
    """
    Registry for custom tokenizer implementations.

    Maps model types (from config) to tokenizer classes or factory functions.
    """

    # Maps model_type -> tokenizer implementation (class/callable/import-string)
    model_type_to_tokenizer: Dict[str, TokenizerImpl] = field(default_factory=dict)

    # Default tokenizer when no custom implementation is found
    default_tokenizer: TokenizerImpl = _DEFAULT_TOKENIZER_IMPL

    def register(self, model_type: str, tokenizer_cls: TokenizerImpl) -> None:
        """
        Register a custom tokenizer for a specific model type.

        Args:
            model_type: The model type string (e.g., "mistral", "llama")
            tokenizer_cls: The tokenizer class or factory function (or import-string)
        """
        self.model_type_to_tokenizer[model_type] = tokenizer_cls
        logger.debug(f"Registered tokenizer {tokenizer_cls} for model type '{model_type}'")

    def get_custom_tokenizer_cls(self, model_type: str) -> Union[Type[Any], Callable[..., Any], None]:
        """
        Resolve the custom tokenizer for `model_type` if one is registered and importable.
        Returns None if no custom tokenizer is registered (or if it cannot be imported).
        """
        tokenizer_impl = self.model_type_to_tokenizer.get(model_type)
        if tokenizer_impl is None:
            return None
        try:
            return _resolve_tokenizer_impl(tokenizer_impl)
        except Exception as e:
            logger.debug(f"Custom tokenizer for model type '{model_type}' could not be imported: {e}")
            # Avoid repeatedly trying a missing optional dependency.
            self.model_type_to_tokenizer.pop(model_type, None)
            return None

    def get_tokenizer_cls(self, model_type: str) -> Union[Type[Any], Callable[..., Any]]:
        """
        Get the tokenizer implementation for a given model type.

        If a custom tokenizer is registered and importable, returns it; otherwise returns the default.
        """
        custom = self.get_custom_tokenizer_cls(model_type)
        if custom is not None:
            return custom
        return _resolve_tokenizer_impl(self.default_tokenizer)

    def has_custom_tokenizer(self, model_type: str) -> bool:
        """Check if a custom tokenizer is registered for the given model type."""
        return model_type in self.model_type_to_tokenizer


# Global tokenizer registry
TokenizerRegistry = _TokenizerRegistry()


def _register_default_tokenizers():
    """Register default custom tokenizer implementations."""
    # Register for Mistral model types (resolved lazily to keep import time low)
    mistral_common = "nemo_automodel._transformers.tokenization.tokenization_mistral_common:MistralCommonBackend"
    TokenizerRegistry.register("mistral", mistral_common)
    TokenizerRegistry.register("pixtral", mistral_common)
    TokenizerRegistry.register("mistral3", mistral_common)


# Register defaults on module load
_register_default_tokenizers()


__all__ = [
    "TokenizerRegistry",
]
