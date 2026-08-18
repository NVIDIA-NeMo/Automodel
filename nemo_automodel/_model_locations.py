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

Model implementations are organized by bridge and task ownership:

* :mod:`nemo_automodel._transformers.models` -- models trained through the
  HuggingFace ``transformers`` bridge: dense/MoE LLMs, VLMs, omni and dLLM
  backbones. These subclass ``PreTrainedModel``.
* :mod:`nemo_automodel._diffusers.models` -- models trained through the
  flow-matching / diffusion bridge. These are ``ModelAdapter`` implementations
  that drive an upstream ``diffusers`` pipeline; most do not import
  ``diffusers`` themselves, because the pipeline is constructed for them by
  :class:`NeMoAutoDiffusionPipeline`.
* :mod:`nemo_automodel.retrieval.models` -- embedding and retrieval model
  families. They use the transformers bridge, but live with the task-level
  retrieval APIs instead of being mixed into the generic bridge internals.

Note this is a *domain* split, not "which upstream package does the file
import". ``qwen_image_edit`` is the only diffusers-side package that imports
``diffusers`` directly; the rest operate on tensors handed to them by the
pipeline. The reliable invariant is the one asserted in
``tests/unit_tests/test_model_locations.py``: diffusers-side packages expose a
``ModelAdapter``, transformers-side packages do not.

Import aliases and bridge-specific error messages need to agree on that split,
so the routing tables live here.

This module must stay import-light (stdlib only): it is imported from
``nemo_automodel/__init__.py`` and must not drag in ``torch`` or
``transformers``.
"""

from __future__ import annotations

DIFFUSERS_MODELS_PACKAGE = "nemo_automodel._diffusers.models"
RETRIEVAL_MODELS_PACKAGE = "nemo_automodel.retrieval.models"
TRANSFORMERS_MODELS_PACKAGE = "nemo_automodel._transformers.models"

# Where each model sub-package lives today. Keep in sync with the directory
# contents; ``tests/unit_tests/test_model_locations.py`` asserts they match.
#
# ``wan`` holds ``SimpleAdapter``: Wan is the only user of the generic "simple"
# transformer interface, and ``adapter_type: "simple"`` remains its config key.
DIFFUSERS_MODELS: frozenset[str] = frozenset(
    {
        "flux",
        "flux2",
        "hunyuan",
        "ltx2",
        "qwen_image",
        "qwen_image_edit",
        "wan",
    }
)

RETRIEVAL_MODELS: frozenset[str] = frozenset(
    {
        "llama_bidirectional",
        "llama_nemotron_vl",
        "ministral_bidirectional",
    }
)

# Model names that were *actually* reachable under the pre-split
# ``nemo_automodel.components.models`` namespace and now live on the diffusers
# side. Deliberately narrower than ``DIFFUSERS_MODELS``: the flow-matching
# adapters (flux, wan, ...) were never under ``components.models``, so aliasing
# them there would invent a history that never existed and would swallow the
# "unknown model" error for those names.
_LEGACY_DIFFUSERS_MODELS: frozenset[str] = frozenset({"qwen_image_edit"})

# Retrieval-only helpers historically nested below the shared ``common`` model
# package. Their parent package remains transformers-owned, so these need exact
# routing rather than model-family routing.
_RELOCATED_MODEL_MODULES: dict[str, str] = {
    "common.bidirectional": "nemo_automodel.retrieval.state_dict_adapter",
    "common.inbatch_neg_utils": "nemo_automodel.retrieval.inbatch_negatives",
}


def models_package_for(name: str) -> str:
    """Return the canonical package holding the model sub-package ``name``.

    Args:
        name: A direct model sub-package name, e.g. ``"llama"``.

    Returns:
        The dotted package path that ``name`` lives under.
    """
    if name in RETRIEVAL_MODELS:
        return RETRIEVAL_MODELS_PACKAGE
    if name in DIFFUSERS_MODELS:
        return DIFFUSERS_MODELS_PACKAGE
    return TRANSFORMERS_MODELS_PACKAGE


def resolve_model_module(suffix: str, *, legacy: bool = False) -> str:
    """Map a flat ``models``-relative module path to its canonical location.

    Args:
        suffix: Path relative to the flat ``models`` package. May be empty
            (the package root), a bare model name (``"llama"``), or a deeper
            path (``"llama.model"``). Non-model members such as ``"gpt2"``,
            ``"common.utils"`` and ``"deprecation"`` resolve to the
            transformers-side package.
        legacy: Resolve for the deprecated ``nemo_automodel.components.models``
            namespace, which only ever contained ``qwen_image_edit`` on the
            diffusers side. Names that never lived there fall through to the
            transformers-side package so that the "unknown model" error still
            fires instead of silently resolving.

    Returns:
        The fully-qualified canonical module path.
    """
    if not suffix:
        return TRANSFORMERS_MODELS_PACKAGE
    if suffix in _RELOCATED_MODEL_MODULES:
        return _RELOCATED_MODEL_MODULES[suffix]
    head = suffix.split(".", 1)[0]
    if head in RETRIEVAL_MODELS:
        return f"{RETRIEVAL_MODELS_PACKAGE}.{suffix}"
    diffusers_names = _LEGACY_DIFFUSERS_MODELS if legacy else DIFFUSERS_MODELS
    package = DIFFUSERS_MODELS_PACKAGE if head in diffusers_names else TRANSFORMERS_MODELS_PACKAGE
    return f"{package}.{suffix}"
