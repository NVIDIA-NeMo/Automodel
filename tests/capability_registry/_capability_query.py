# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Pure query side of the capability registry.

Each NeMo custom model class declares its parallelism support via a nested
``ModelCapabilities`` dataclass, and optionally a ``get_capabilities(config)``
classmethod that returns config-specialised values (e.g. dense vs MoE
variants of the same class). The query path prefers ``get_capabilities``
when present and falls back to the dataclass defaults otherwise::

    class Gemma4ForConditionalGeneration(...):
        @dataclass(frozen=True)
        class ModelCapabilities:
            supports_tp: bool = False
            supports_cp: bool = False
            supports_pp: bool = False
            supports_ep: bool = False

        @classmethod
        def get_capabilities(cls, config) -> "ModelCapabilities":
            ...
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any

from transformers import AutoConfig

from nemo_automodel._transformers.registry import ModelRegistry

logger = logging.getLogger(__name__)


# Capability flags exposed by the registry. Each name maps to a field of the
# model class's nested ``ModelCapabilities`` dataclass.
CAPABILITIES: tuple[str, ...] = (
    "supports_tp",
    "supports_cp",
    "supports_pp",
    "supports_ep",
)


def query_capabilities(model_id: str, *, trust_remote_code: bool = False) -> dict[str, bool]:
    """Return the declared ``ModelCapabilities`` for ``model_id``.

    The lookup prefers ``model_cls.get_capabilities(config)`` so that classes
    serving multiple variants (e.g. dense vs MoE Gemma4) can specialise their
    flags by config. If the model class doesn't provide one, the bare
    ``ModelCapabilities`` defaults are used.

    Args:
        model_id: HuggingFace model id (e.g. ``"google/gemma-4-..."``).
        trust_remote_code: Forwarded to ``AutoConfig.from_pretrained``.

    Returns:
        Dict mapping each capability name (``"supports_tp"``, ``"supports_cp"``,
        ``"supports_pp"``, ``"supports_ep"``) to its declared boolean value.

    Raises:
        ValueError: If no NeMo custom class is registered for the model's
            architecture, or the registered class does not declare
            ``ModelCapabilities``.
    """
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    model_cls = _resolve_nemo_class(config)
    caps_cls = _get_capabilities_class(model_cls)
    caps_instance = _resolve_capabilities_instance(model_cls, caps_cls, config)
    return _capabilities_to_dict(caps_instance)


def _resolve_nemo_class(config: Any) -> type:
    """Return the NeMo custom model class for ``config.architectures[0]``."""
    architectures = getattr(config, "architectures", None) or []
    for arch in architectures:
        try:
            cls = ModelRegistry.resolve_custom_model_cls(arch, config)
        except Exception as exc:  # noqa: BLE001 - any failure is a query miss
            logger.info("resolve_custom_model_cls(%s) raised %s", arch, exc)
            cls = None
        if cls is not None:
            return cls
    raise ValueError(
        f"No NeMo custom model class registered for architectures={architectures}. "
        f"Capability querying requires a registered class that declares ModelCapabilities."
    )


def _get_capabilities_class(model_cls: type) -> type:
    """Return the nested ``ModelCapabilities`` dataclass on ``model_cls``."""
    caps_cls = getattr(model_cls, "ModelCapabilities", None)
    if caps_cls is None:
        raise ValueError(
            f"Model class {model_cls.__name__} does not declare a nested ModelCapabilities dataclass. "
            f"Add one with fields {CAPABILITIES} to register its parallelism support contract."
        )
    if not dataclasses.is_dataclass(caps_cls):
        raise ValueError(f"{model_cls.__name__}.ModelCapabilities must be a @dataclass; got {type(caps_cls).__name__}.")
    return caps_cls


def _resolve_capabilities_instance(model_cls: type, caps_cls: type, config: Any):
    """Pick the config-specialised capabilities when available, else defaults.

    If ``model_cls.get_capabilities`` is defined, it is called with the model's
    config — this lets a single class (e.g. ``Gemma4ForConditionalGeneration``)
    expose different flags for dense vs MoE variants. Otherwise the dataclass's
    default-constructed instance is returned.
    """
    get_caps = getattr(model_cls, "get_capabilities", None)
    if callable(get_caps):
        instance = get_caps(config)
        if not isinstance(instance, caps_cls):
            raise ValueError(
                f"{model_cls.__name__}.get_capabilities returned {type(instance).__name__}, "
                f"expected {caps_cls.__qualname__}."
            )
        return instance
    return caps_cls()


def _capabilities_to_dict(instance: Any) -> dict[str, bool]:
    """Read the canonical fields off a ``ModelCapabilities`` instance into a dict."""
    out: dict[str, bool] = {}
    for name in CAPABILITIES:
        if not hasattr(instance, name):
            raise ValueError(f"{type(instance).__qualname__} is missing required capability field '{name}'.")
        out[name] = bool(getattr(instance, name))
    return out
