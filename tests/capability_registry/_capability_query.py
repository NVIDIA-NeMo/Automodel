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

"""Pure-query side of the capability registry.

Given a model id, return the ``supports_*`` flags without loading weights or
requiring a GPU. The model is constructed on the meta device (zero memory)
purely so the :class:`nemo_automodel._transformers.capabilities.ModelSupports`
introspection layer has an instance to inspect.
"""

from __future__ import annotations

import logging

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from nemo_automodel._transformers.capabilities import ModelSupports
from nemo_automodel._transformers.registry import ModelRegistry

logger = logging.getLogger(__name__)


# Capabilities exposed by the registry. The CLI iterates this list and the
# corresponding ``supports_*`` property is read off the ModelSupports descriptor.
CAPABILITIES: tuple[str, ...] = ("tp", "cp", "pp", "ep")


def query_capabilities(model_id: str, *, trust_remote_code: bool = False) -> dict[str, bool]:
    """Return a ``{capability: bool}`` dict for ``model_id``.

    Args:
        model_id: HuggingFace model id (e.g. ``"meta-llama/Llama-3.1-8B"``).
        trust_remote_code: Forwarded to ``AutoConfig`` / ``AutoModelForCausalLM``.

    Returns:
        Dict like ``{"tp": True, "cp": True, "pp": True, "ep": False}``.
    """
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)

    # Prefer a NeMo custom model class if one is registered for this architecture;
    # otherwise fall back to the HF class. Either way the model is instantiated
    # on the meta device so no real memory is allocated.
    model = _instantiate_on_meta(config, trust_remote_code=trust_remote_code)

    supports = ModelSupports(model, mesh=None)
    return {name: bool(getattr(supports, f"supports_{name}")) for name in CAPABILITIES}


def _instantiate_on_meta(config, *, trust_remote_code: bool) -> torch.nn.Module:
    """Build the model on the meta device for cheap introspection.

    First tries the registered NeMo custom model class (so we exercise the
    NeMo code path that owns capability flags); falls back to the HF class.
    """
    custom_cls = None
    architectures = getattr(config, "architectures", None) or []
    for arch in architectures:
        try:
            custom_cls = ModelRegistry.resolve_custom_model_cls(arch, config)
        except Exception as exc:  # noqa: BLE001 - any failure falls back to HF.
            logger.info("resolve_custom_model_cls(%s) raised %s; falling back to HF.", arch, exc)
            custom_cls = None
        if custom_cls is not None:
            break

    with torch.device("meta"):
        if custom_cls is not None:
            try:
                return custom_cls(config)
            except Exception as exc:  # noqa: BLE001 - any failure falls back to HF.
                logger.info(
                    "NeMo custom model class %s failed to construct on meta device (%s); "
                    "falling back to HF AutoModelForCausalLM for introspection only.",
                    custom_cls.__name__,
                    exc,
                )
        return AutoModelForCausalLM.from_config(config, trust_remote_code=trust_remote_code)
