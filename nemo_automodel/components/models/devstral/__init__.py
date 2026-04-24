# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
"""FP8-native Mistral3 / Ministral3 custom model registration.

Importing this module installs a resolver hook on
``_resolve_custom_model_cls_for_config`` (nemo_automodel/_transformers/
model_init.py) that routes FP8-native Mistral3 / Ministral3 configs to one
of two classes depending on whether the user asked for a text-only or a
full-VLM flow:

  * ``NeMoAutoModelForCausalLM.from_pretrained`` → ``Mistral3FP8ForCausalLM``
    (text-only; vision modules dropped). Layout auto-detected.
  * ``NeMoAutoModelForImageTextToText.from_pretrained`` →
    ``Mistral3FP8VLMForConditionalGeneration`` (full VLM; vision_tower +
    multi_modal_projector preserved).

The model registry is **not** overwritten, so non-FP8 Mistral3 VLMs keep
the stock ``mistral4.model`` path and there is no regression for other users.

Covers:
  * mistralai/Devstral-Small-2-24B-Instruct-2512 (VLM, language_model. prefix)
  * mistralai/Devstral-2-123B-Instruct-2512     (dense, no prefix)
  * dawn-ridge-medium-3p5-128b                  (VLM, model.language_model. infix)
"""

from __future__ import annotations

import logging

from nemo_automodel._transformers import model_init as _mi
from nemo_automodel.components.models.devstral.model import (
    Mistral3FP8ForCausalLM,
    Mistral3FP8VLMForConditionalGeneration,
)

logger = logging.getLogger(__name__)


def _is_image_text_to_text_entry(entry_cls) -> bool:
    """Return True if *entry_cls* is a NeMoAuto image-text-to-text flavor.

    We inspect by name rather than issubclass to avoid a circular import
    against nemo_automodel._transformers.auto_model from this package.
    """
    if entry_cls is None:
        return False
    for c in getattr(entry_cls, "__mro__", (entry_cls,)):
        name = getattr(c, "__name__", "")
        if "ImageTextToText" in name:
            return True
    return False


def _install_resolver_hook() -> None:
    """Prepend an FP8-Mistral3 check to _resolve_custom_model_cls_for_config."""
    if getattr(_mi._resolve_custom_model_cls_for_config, "_devstral_hook_installed", False):
        return
    _orig_resolve = _mi._resolve_custom_model_cls_for_config

    def _patched(config, *, entry_cls=None):
        try:
            if _is_image_text_to_text_entry(entry_cls):
                if Mistral3FP8VLMForConditionalGeneration.supports_config(config):
                    logger.info(
                        "Mistral3 FP8 resolver claiming config %s for %s (VLM)",
                        type(config).__name__,
                        Mistral3FP8VLMForConditionalGeneration.__name__,
                    )
                    return Mistral3FP8VLMForConditionalGeneration
            elif Mistral3FP8ForCausalLM.supports_config(config):
                logger.info(
                    "Mistral3 FP8 resolver claiming config %s for %s (text)",
                    type(config).__name__,
                    Mistral3FP8ForCausalLM.__name__,
                )
                return Mistral3FP8ForCausalLM
        except Exception:  # pragma: no cover - defensive
            logger.debug("Mistral3 FP8 resolver raised", exc_info=True)
        return _orig_resolve(config, entry_cls=entry_cls)

    _patched._devstral_hook_installed = True  # type: ignore[attr-defined]
    _mi._resolve_custom_model_cls_for_config = _patched


_install_resolver_hook()


__all__ = [
    "Mistral3FP8ForCausalLM",
    "Mistral3FP8VLMForConditionalGeneration",
]
