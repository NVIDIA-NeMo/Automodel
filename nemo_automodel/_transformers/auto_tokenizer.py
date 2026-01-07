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
from typing import Callable, Optional, Type, Union

from transformers import AutoConfig, AutoTokenizer

from nemo_automodel._transformers.tokenization.nemo_auto_tokenizer import NeMoAutoTokenizerWithBosEosEnforced
from nemo_automodel._transformers.tokenization.registry import TokenizerRegistry

logger = logging.getLogger(__name__)


def _get_model_type(pretrained_model_name_or_path: str, trust_remote_code: bool = False) -> Optional[str]:
    """
    Determine the model type from the config.

    Args:
        pretrained_model_name_or_path: Model identifier or path
        trust_remote_code: Whether to trust remote code

    Returns:
        The model_type string, or None if it cannot be determined
    """
    try:
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=trust_remote_code)
        return getattr(config, "model_type", None)
    except Exception as e:
        logger.debug(f"Could not load config to determine model type: {e}")
        return None

class NeMoAutoTokenizer(AutoTokenizer):
    """
    Auto tokenizer class that dispatches to appropriate tokenizer implementations.

    Similar to HuggingFace's AutoTokenizer, but with a custom registry for specialized
    tokenizer implementations.

    The dispatch logic is:
    1. If a custom tokenizer is registered for the model type, use it
    2. Otherwise, fall back to AutoTokenizerWithBosEosEnforced

    Example:
        >>> # Will use MistralCommonBackend if available for Mistral models
        >>> tokenizer = NeMoAutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

        >>> # Force using HF AutoTokenizer with BOS/EOS enforcement
        >>> tokenizer = NeMoAutoTokenizer.from_pretrained("gpt2", force_default=True)
    """

    # Make registry accessible at class level
    _registry = TokenizerRegistry

    def __init__(self):
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to be instantiated using the "
            f"`{self.__class__.__name__}.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @classmethod
    def register(cls, model_type: str, tokenizer_cls: Union[Type, Callable]) -> None:
        """
        Register a custom tokenizer for a specific model type.

        Args:
            model_type: The model type string (e.g., "mistral", "llama")
            tokenizer_cls: The tokenizer class or factory function
        """
        cls._registry.register(model_type, tokenizer_cls)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *args,
        force_default: bool = False,
        force_hf: bool = False,
        trust_remote_code: bool = False,
        **kwargs,
    ):
        """
        Load a tokenizer from a pretrained model.

        Args:
            pretrained_model_name_or_path: Model identifier or path
            force_default: If True, always use NeMoAutoTokenizerWithBosEosEnforced
            force_hf: If True, return the raw HF AutoTokenizer without any wrapping
            trust_remote_code: Whether to trust remote code when loading config
            **kwargs: Additional arguments passed to the tokenizer's from_pretrained

        Returns:
            A tokenizer instance appropriate for the model type
        """
        # If force_hf, just use the base HF AutoTokenizer
        if force_hf:
            return super().from_pretrained(
                pretrained_model_name_or_path, *args, trust_remote_code=trust_remote_code, **kwargs
            )

        # Try to determine model type from config
        model_type = _get_model_type(pretrained_model_name_or_path, trust_remote_code=trust_remote_code)

        if model_type and cls._registry.has_custom_tokenizer(model_type):
            tokenizer_cls = cls._registry.get_tokenizer_cls(model_type)
            logger.info(f"Using custom tokenizer {tokenizer_cls.__name__} for model type '{model_type}'")
            return tokenizer_cls.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

        # Fall back to default BOS/EOS enforced tokenizer
        return NeMoAutoTokenizerWithBosEosEnforced.from_pretrained(
            pretrained_model_name_or_path, *args, trust_remote_code=trust_remote_code, **kwargs
        )

__all__ = [
    "NeMoAutoTokenizer",
    "NeMoAutoTokenizerWithBosEosEnforced",
    "TokenizerRegistry",
]
