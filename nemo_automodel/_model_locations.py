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
"""Single source of truth for where model sub-packages live.

Model implementations are split by the upstream HuggingFace package they are
written against:

* :mod:`nemo_automodel._transformers.models` -- models built on ``transformers``
  (the overwhelming majority: dense/MoE LLMs, VLMs, omni and dLLM backbones).
* :mod:`nemo_automodel._diffusers.models` -- models built on ``diffusers``.

Three call sites need to agree on that split -- the two ``models/__init__.py``
"unknown model" error messages and the import aliases in
:mod:`nemo_automodel` -- so the table lives here.

This module must stay import-light (stdlib only): it is imported from
``nemo_automodel/__init__.py`` and must not drag in ``torch`` or
``transformers``.
"""

from __future__ import annotations

TRANSFORMERS_MODELS_PACKAGE = "nemo_automodel._transformers.models"
DIFFUSERS_MODELS_PACKAGE = "nemo_automodel._diffusers.models"

# Model sub-packages that live under ``_diffusers.models`` rather than the
# ``_transformers.models`` default. Keep in sync with the directory contents;
# ``tests/unit_tests/test_model_locations.py`` asserts they match.
DIFFUSERS_MODELS: frozenset[str] = frozenset({"qwen_image_edit"})


def models_package_for(name: str) -> str:
    """Return the canonical package holding the model sub-package ``name``.

    Args:
        name: A direct model sub-package name, e.g. ``"llama"``.

    Returns:
        The dotted package path that ``name`` lives under.
    """
    return DIFFUSERS_MODELS_PACKAGE if name in DIFFUSERS_MODELS else TRANSFORMERS_MODELS_PACKAGE


def resolve_model_module(suffix: str) -> str:
    """Map a legacy ``models``-relative module path to its canonical location.

    Args:
        suffix: Path relative to the old flat ``models`` package. May be empty
            (the package root), a bare model name (``"llama"``), or a deeper
            path (``"llama.model"``). Non-model members such as ``"gpt2"``,
            ``"common.utils"`` and ``"deprecation"`` resolve to the
            transformers-side package.

    Returns:
        The fully-qualified module path under ``_transformers`` or
        ``_diffusers``.
    """
    if not suffix:
        return TRANSFORMERS_MODELS_PACKAGE
    head = suffix.split(".", 1)[0]
    return f"{models_package_for(head)}.{suffix}"
