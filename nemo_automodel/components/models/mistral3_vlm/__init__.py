# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
"""FP8-native Mistral3 VLM custom model registration (dawn-ridge 128B).

Importing this module installs a resolver hook on
``_resolve_custom_model_cls_for_config`` (nemo_automodel/_transformers/
model_init.py) that routes FP8-native Mistral3 VLM configs to
``Mistral3FP8VLMForConditionalGeneration``. The model registry is **not**
overwritten, so non-FP8 Mistral3 VLMs keep the stock ``mistral4.model``
path and there is no regression for other users.
"""

from __future__ import annotations

import logging

from nemo_automodel._transformers import model_init as _mi
from nemo_automodel.components.models.mistral3_vlm.model import (
    Mistral3FP8VLMForConditionalGeneration,
)

logger = logging.getLogger(__name__)


def _install_resolver_hook() -> None:
    """Prepend an FP8-Mistral3 VLM check to _resolve_custom_model_cls_for_config."""
    if getattr(_mi._resolve_custom_model_cls_for_config, "_mistral3_vlm_hook_installed", False):
        return
    _orig_resolve = _mi._resolve_custom_model_cls_for_config

    def _patched(config):
        try:
            if Mistral3FP8VLMForConditionalGeneration.supports_config(config):
                logger.info(
                    "Mistral3 VLM FP8 resolver claiming config %s for %s",
                    type(config).__name__,
                    Mistral3FP8VLMForConditionalGeneration.__name__,
                )
                return Mistral3FP8VLMForConditionalGeneration
        except Exception:  # pragma: no cover - defensive
            logger.debug("Mistral3 VLM FP8 resolver raised", exc_info=True)
        return _orig_resolve(config)

    _patched._mistral3_vlm_hook_installed = True  # type: ignore[attr-defined]
    _mi._resolve_custom_model_cls_for_config = _patched


_install_resolver_hook()


__all__ = [
    "Mistral3FP8VLMForConditionalGeneration",
]
