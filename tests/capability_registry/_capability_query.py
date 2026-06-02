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

Each NeMo custom model class declares its parallelism / packing support
via a nested ``ModelCapabilities`` dataclass, e.g.::

    class Gemma4ForConditionalGeneration(...):
        @dataclass(frozen=True)
        class ModelCapabilities:
            supports_tp: bool = False
            supports_cp: bool = False
            supports_pp: bool = False
            supports_ep: bool = False
            supports_packing: bool = False

This module looks up the registered NeMo class for a given HF model id and
returns the declared capability values. Validation that the declared values
actually hold is intentionally out of scope here -- standardized tests live
under ``tests/capability_registry/standardized_tests/``.
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
    "supports_packing",
)


def query_capabilities(model_id: str, *, trust_remote_code: bool = False) -> dict[str, bool]:
    """Return the declared ``ModelCapabilities`` for ``model_id``.

    Args:
        model_id: HuggingFace model id (e.g. ``"google/gemma-4-..."``).
        trust_remote_code: Forwarded to ``AutoConfig.from_pretrained``.

    Returns:
        Dict mapping each capability name (``"supports_tp"``, ``"supports_cp"``,
        ``"supports_pp"``, ``"supports_ep"``, ``"supports_packing"``) to its
        declared boolean value.

    Raises:
        ValueError: If no NeMo custom class is registered for the model's
            architecture, or the registered class does not declare
            ``ModelCapabilities``.
    """
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    model_cls = _resolve_nemo_class(config)
    caps_cls = _get_capabilities_class(model_cls)
    return _capabilities_to_dict(caps_cls)


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


def _capabilities_to_dict(caps_cls: type) -> dict[str, bool]:
    """Instantiate the (defaults-only) capabilities dataclass and return its fields as a dict."""
    instance = caps_cls()  # all fields have defaults -> no args required
    out: dict[str, bool] = {}
    for name in CAPABILITIES:
        if not hasattr(instance, name):
            raise ValueError(f"{caps_cls.__qualname__} is missing required capability field '{name}'.")
        out[name] = bool(getattr(instance, name))
    return out
