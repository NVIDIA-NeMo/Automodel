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

"""Compatibility patches for legacy v4-style transformer models.

Submodules are imported on first attribute access: ``import nemo_automodel`` installs the
``layer_types`` import hook (stdlib only) and must not load ``rotary`` and, through it,
``torch.distributed.tensor``.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nemo_automodel._transformers.v4_patches.kv_sharing import (
        install_kv_sharing_holder,
        should_install_kv_sharing_holder,
    )
    from nemo_automodel._transformers.v4_patches.layer_types import (
        install_layer_types_patch_hook,
        patch_allowed_layer_types,
    )
    from nemo_automodel._transformers.v4_patches.rotary import (
        fix_rotary_embeddings,
        should_fix_rotary_embeddings,
    )

_EXPORTS = {
    "install_kv_sharing_holder": "kv_sharing",
    "should_install_kv_sharing_holder": "kv_sharing",
    "install_layer_types_patch_hook": "layer_types",
    "patch_allowed_layer_types": "layer_types",
    "fix_rotary_embeddings": "rotary",
    "should_fix_rotary_embeddings": "rotary",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        return getattr(importlib.import_module(f"{__name__}.{_EXPORTS[name]}"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
