# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
"""FP8-native Mistral3 VLM custom model registration (dawn-ridge 128B + ministral3).

Importing this module installs a resolver hook on
``_resolve_custom_model_cls_for_config`` (nemo_automodel/_transformers/
model_init.py) that routes FP8-native Mistral3 VLM configs (any ministral3
text backbone with ``quant_method == 'fp8'`` -- e.g. dawn-ridge-128B,
``Devstral-Small-2-24B-Instruct-2512``, ``Ministral-3-3B-Instruct-2512``) to
``Mistral3FP8VLMForConditionalGeneration``. The model registry is **not**
overwritten, so non-FP8 Mistral3 VLMs keep the stock ``mistral4.model`` path
and there is no regression for other users.

The hook is installed eagerly from ``_transformers/registry.py`` (which
imports this package) so the routing does not depend on any other module
being imported first. To keep that eager install cheap, the heavy
``model.py`` (transformers Mistral3) import is deferred: ``supports_config``
is duplicated here as a lightweight, torch-free predicate and the concrete
model class is imported only when a config is actually claimed.
"""

from __future__ import annotations

import logging

from nemo_automodel._transformers import model_init as _mi

logger = logging.getLogger(__name__)


def _config_is_fp8_ministral3_vlm(config) -> bool:
    """Lightweight, import-free predicate mirroring ``Mistral3FP8VLMForConditionalGeneration.supports_config``.

    Claims an outer ``Mistral3Config`` VLM whose text backbone is ministral3
    and whose ``quantization_config`` requests FP8. Kept here (rather than
    calling the model classmethod) so the resolver hook can be installed
    without importing the heavy ``model.py`` module.
    """
    text_config = getattr(config, "text_config", None)
    if text_config is None or getattr(text_config, "model_type", None) != "ministral3":
        return False
    qc = getattr(config, "quantization_config", None)
    if qc is None:
        return False
    method = qc.get("quant_method") if isinstance(qc, dict) else getattr(qc, "quant_method", None)
    return method == "fp8"


def _install_resolver_hook() -> None:
    """Prepend an FP8-Mistral3 VLM check to ``_resolve_custom_model_cls_for_config``."""
    if getattr(_mi._resolve_custom_model_cls_for_config, "_mistral3_vlm_hook_installed", False):
        return
    _orig_resolve = _mi._resolve_custom_model_cls_for_config

    def _patched(config):
        try:
            if _config_is_fp8_ministral3_vlm(config):
                from nemo_automodel.components.models.mistral3_vlm.model import (
                    Mistral3FP8VLMForConditionalGeneration,
                )

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


def __getattr__(name: str):
    """Lazily expose ``Mistral3FP8VLMForConditionalGeneration`` without eager heavy import."""
    if name == "Mistral3FP8VLMForConditionalGeneration":
        from nemo_automodel.components.models.mistral3_vlm.model import (
            Mistral3FP8VLMForConditionalGeneration,
        )

        return Mistral3FP8VLMForConditionalGeneration
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Mistral3FP8VLMForConditionalGeneration",
]
