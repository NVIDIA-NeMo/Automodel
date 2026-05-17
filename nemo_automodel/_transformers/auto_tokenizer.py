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
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=trust_remote_code)
        return getattr(config, "model_type", None)
    except Exception as e:
        logger.debug(f"Could not load config to determine model type: {e}")
        return None


def _get_tokenizer_registry():
    # Import lazily to avoid pulling in optional/custom backends (and transformers)
    # when users only do `from nemo_automodel import NeMoAutoTokenizer`.
    from nemo_automodel._transformers.tokenization.registry import TokenizerRegistry

    return TokenizerRegistry


class NeMoAutoTokenizer:
    """
    Auto tokenizer class that dispatches to appropriate tokenizer implementations.

    Similar to HuggingFace's AutoTokenizer, but with a custom registry for specialized
    tokenizer implementations.

    The dispatch logic is:
    1. If a custom tokenizer is registered for the model type, use it
    2. Otherwise, fall back to NeMoAutoTokenizerWithBosEosEnforced

    Example:
        >>> # Will use MistralCommonBackend if available for Mistral models
        >>> tokenizer = NeMoAutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

        >>> # Force using HF AutoTokenizer with BOS/EOS enforcement
        >>> tokenizer = NeMoAutoTokenizer.from_pretrained("gpt2", force_default=True)
    """

    # Make registry accessible at class level
    _registry = None

    @classmethod
    def register(cls, model_type: str, tokenizer_cls: Union[Type, Callable]) -> None:
        """
        Register a custom tokenizer for a specific model type.

        Args:
            model_type: The model type string (e.g., "mistral", "llama")
            tokenizer_cls: The tokenizer class or factory function
        """
        _get_tokenizer_registry().register(model_type, tokenizer_cls)

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
            from transformers import AutoTokenizer

            return AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path, *args, trust_remote_code=trust_remote_code, **kwargs
            )

        # Try to determine model type from config
        model_type = _get_model_type(pretrained_model_name_or_path, trust_remote_code=trust_remote_code)

        registry = _get_tokenizer_registry()

        if not force_default and model_type:
            tokenizer_cls = registry.get_custom_tokenizer_cls(model_type)
            if tokenizer_cls is not None:
                logger.info(f"Using custom tokenizer {tokenizer_cls.__name__} for model type '{model_type}'")
                tokenizer = tokenizer_cls.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
                from nemo_automodel._transformers.tokenization.nemo_auto_tokenizer import _ensure_pad_token_id

                _ensure_pad_token_id(tokenizer, pretrained_model_name_or_path)
                return tokenizer

        # Fall back to default BOS/EOS enforced tokenizer
        from nemo_automodel._transformers.tokenization.nemo_auto_tokenizer import NeMoAutoTokenizerWithBosEosEnforced

        return NeMoAutoTokenizerWithBosEosEnforced.from_pretrained(
            pretrained_model_name_or_path, *args, trust_remote_code=trust_remote_code, **kwargs
        )


# ── Recipe helpers (model-name / trust_remote_code / tokenizer-from-cfg) ──
# These used to live in recipes/llm/train_ft.py. They're tokenizer/model-config
# utilities used by the recipe and by external callers; they belong next to
# NeMoAutoTokenizer.


def _get_model_name(cfg_model):
    """Pull ``pretrained_model_name_or_path`` from a recipe-style model config.

    Accepts either a top-level ``pretrained_model_name_or_path`` field, or a
    nested ``config`` (string or sub-config) carrying the same field. Returns
    ``None`` if absent.
    """
    if cfg_model.get("pretrained_model_name_or_path", None) is not None:
        return cfg_model.pretrained_model_name_or_path
    elif cfg_model.get("config", None) is not None:
        if isinstance(cfg_model.config, str):
            return cfg_model.config
        return cfg_model.config.get("pretrained_model_name_or_path", None)
    else:
        return None


def compute_trust_remote_code_from_model(cfg_model):
    """Compute ``trust_remote_code`` from a recipe-style model config.

    Reads ``cfg_model.trust_remote_code`` or ``cfg_model.config.trust_remote_code``
    if present; otherwise falls back to
    :func:`nemo_automodel.components.utils.model_utils.resolve_trust_remote_code`
    against the model name.
    """
    from nemo_automodel.components.utils.model_utils import resolve_trust_remote_code

    if hasattr(cfg_model, "trust_remote_code"):
        return getattr(cfg_model, "trust_remote_code")
    elif hasattr(cfg_model, "config") and hasattr(cfg_model.config, "trust_remote_code"):
        return getattr(cfg_model.config, "trust_remote_code")
    return resolve_trust_remote_code(_get_model_name(cfg_model))


def _build_tokenizer(cfg_model, cfg_ds):
    """Build a tokenizer from the recipe's model + dataset config.

    Resolution order:
      1. If ``cfg_ds.tokenizer`` is absent and ``cfg_model`` provides a model name,
         load via :class:`NeMoAutoTokenizer.from_pretrained`.
      2. Else if ``cfg_ds.tokenizer`` is a plain config (no ``_target_``), forward
         its kwargs to ``NeMoAutoTokenizer.from_pretrained``.
      3. Else if ``cfg_ds.tokenizer`` has ``_target_``, instantiate the configured
         class.

    Returns ``(kwargs_for_dataset, tokenizer)`` where ``kwargs_for_dataset`` is
    a dict ready to merge into the dataset constructor — when the dataset's
    target accepts a ``tokenizer`` argument, the tokenizer is included.
    """
    import inspect as _inspect

    trust_remote_code = compute_trust_remote_code_from_model(cfg_model)
    if "tokenizer" not in cfg_ds and _get_model_name(cfg_model) is not None:
        logger.info("Using model config to instantiate tokenizer")
        tokenizer = NeMoAutoTokenizer.from_pretrained(_get_model_name(cfg_model), trust_remote_code=trust_remote_code)
    elif cfg_ds.get("tokenizer", None) is None:
        tokenizer = None
    elif "_target_" not in cfg_ds.tokenizer:
        tokenizer_dict = cfg_ds.tokenizer.to_dict()
        trust_remote_code = tokenizer_dict.pop("trust_remote_code", trust_remote_code)
        tokenizer = NeMoAutoTokenizer.from_pretrained(**tokenizer_dict, trust_remote_code=trust_remote_code)
    else:
        trust_remote_code = cfg_ds.tokenizer.to_dict().pop("trust_remote_code", trust_remote_code)
        tokenizer = cfg_ds.tokenizer.instantiate(trust_remote_code=trust_remote_code)

    # Pass tokenizer into the dataset constructor only if it accepts it.
    kwargs = {}
    if tokenizer is not None and callable(cfg_ds._target_):
        try:
            sig = _inspect.signature(cfg_ds._target_)
            if "tokenizer" in sig.parameters:
                kwargs["tokenizer"] = tokenizer
        except (ValueError, TypeError):
            pass
    return kwargs, tokenizer


__all__ = [
    "NeMoAutoTokenizer",
    "NeMoAutoTokenizerWithBosEosEnforced",
    "TokenizerRegistry",
    "_build_tokenizer",
    "_get_model_name",
    "compute_trust_remote_code_from_model",
]


def __getattr__(name: str):
    if name == "TokenizerRegistry":
        return _get_tokenizer_registry()
    if name == "NeMoAutoTokenizerWithBosEosEnforced":
        from nemo_automodel._transformers.tokenization.nemo_auto_tokenizer import NeMoAutoTokenizerWithBosEosEnforced

        return NeMoAutoTokenizerWithBosEosEnforced
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
