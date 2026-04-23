# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
"""Devstral model wrappers with streaming FP8 dequantize load support.

Importing this module installs a tiny resolver hook that prepends a check
for FP8-native Devstral configs to
``_resolve_custom_model_cls_for_config`` (nemo_automodel/_transformers/
model_init.py). When the check matches, our streaming-FP8 custom class is
returned; otherwise the original resolver runs unchanged, preserving the
standard mistral4/Ministral3 paths.

This avoids overwriting the model registry (which has only one class per
architecture name) and therefore does not regress non-FP8 users.
"""

from __future__ import annotations

import logging

from nemo_automodel._transformers import model_init as _mi
from nemo_automodel.components.models.devstral.model import (
    Devstral24BFP8TextForCausalLM,
    Devstral123BFP8ForCausalLM,
)

logger = logging.getLogger(__name__)


_FP8_CLASSES = (Devstral24BFP8TextForCausalLM, Devstral123BFP8ForCausalLM)


def _install_resolver_hook():
    """Prepend an FP8-Devstral check to _resolve_custom_model_cls_for_config."""
    if getattr(_mi._resolve_custom_model_cls_for_config, "_devstral_hook_installed", False):
        return
    _orig_resolve = _mi._resolve_custom_model_cls_for_config

    def _patched(config):
        for cls in _FP8_CLASSES:
            try:
                if cls.supports_config(config):
                    logger.info(
                        "Devstral FP8 resolver claiming config %s for %s",
                        type(config).__name__,
                        cls.__name__,
                    )
                    return cls
            except Exception:  # pragma: no cover — defensive
                logger.debug("supports_config raised for %s", cls.__name__, exc_info=True)
        return _orig_resolve(config)

    _patched._devstral_hook_installed = True  # type: ignore[attr-defined]
    _mi._resolve_custom_model_cls_for_config = _patched


_install_resolver_hook()


__all__ = [
    "Devstral24BFP8TextForCausalLM",
    "Devstral123BFP8ForCausalLM",
]
