# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""Adapter contract for the FlowMatching pipeline.

This package owns the *contract* -- :class:`ModelAdapter` and
:class:`FlowMatchingContext` -- which belongs to the flow-matching algorithm
rather than to any one model.

The per-model implementations live with their model, under
``nemo_automodel._diffusers.models.<arch>.adapter``:

===================  ==============================================
``adapter_type``     module
===================  ==============================================
``hunyuan``          ``diffusers.models.hunyuan.adapter``
``simple``           ``diffusers.models.wan.adapter`` (Wan)
``flux``             ``diffusers.models.flux.adapter``
``flux2``            ``diffusers.models.flux2.adapter``
``qwen_image``       ``diffusers.models.qwen_image.adapter``
``qwen_image_edit``  ``diffusers.models.qwen_image_edit.adapter``
``ltx2``             ``diffusers.models.ltx2.adapter``
===================  ==============================================

Use ``create_adapter(adapter_type)`` from
:mod:`nemo_automodel.components.flow_matching.pipeline` rather than importing a
concrete adapter directly.  The adapter classes remain importable from here for
backwards compatibility; they resolve lazily so that touching this package does
not pull in every model's code.
"""

import importlib
from typing import Any

from .base import FlowMatchingContext, ModelAdapter

# Adapter class name -> model package holding it.
_RELOCATED_ADAPTERS = {
    "FluxAdapter": "flux",
    "Flux2Adapter": "flux2",
    "HunyuanAdapter": "hunyuan",
    "LTX2Adapter": "ltx2",
    "QwenImageAdapter": "qwen_image",
    "QwenImageEditAdapter": "qwen_image_edit",
    "SimpleAdapter": "wan",
}

__all__ = [
    "FlowMatchingContext",
    "ModelAdapter",
    *sorted(_RELOCATED_ADAPTERS),
]


def __getattr__(name: str) -> Any:
    """Resolve a relocated adapter class on first access."""
    package = _RELOCATED_ADAPTERS.get(name)
    if package is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f"nemo_automodel._diffusers.models.{package}.adapter")
    attr = getattr(module, name)
    globals()[name] = attr
    return attr


def __dir__() -> list[str]:
    return sorted(__all__)
