# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import importlib
import importlib.abc
import importlib.machinery
import sys
import warnings
from types import ModuleType
from typing import Any

# Pydantic v2 emits UnsupportedFieldAttributeWarning for Field(repr=...) /
# Field(frozen=...) used inside 3.12-style `type` aliases in third-party libs.
# Suppress early so any later import that triggers pydantic schema generation
# (e.g. transformers, huggingface_hub) won't emit these warnings.
try:
    from pydantic.warnings import UnsupportedFieldAttributeWarning

    warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)
except ImportError:
    pass

from .package_info import __package_name__, __version__

# Keep the base package import lightweight.
# Heavy dependencies (e.g., torch/transformers) are intentionally imported lazily
# via __getattr__ so importing tokenizers doesn't pull in the full training stack.

_SUBMODULES = {"recipes", "shared", "components", "models"}

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "NeMoAutoModelForCausalLM": ("nemo_automodel._transformers.auto_model", "NeMoAutoModelForCausalLM"),
    "NeMoAutoModelForImageTextToText": ("nemo_automodel._transformers.auto_model", "NeMoAutoModelForImageTextToText"),
    "NeMoAutoModelForMultimodalLM": ("nemo_automodel._transformers.auto_model", "NeMoAutoModelForMultimodalLM"),
    "NeMoAutoModelForSeq2SeqLM": ("nemo_automodel._transformers.auto_model", "NeMoAutoModelForSeq2SeqLM"),
    "NeMoAutoModelForSequenceClassification": (
        "nemo_automodel._transformers.auto_model",
        "NeMoAutoModelForSequenceClassification",
    ),
    "NeMoAutoModelForTokenClassification": (
        "nemo_automodel._transformers.auto_model",
        "NeMoAutoModelForTokenClassification",
    ),
    "NeMoAutoModelForTextToWaveform": ("nemo_automodel._transformers.auto_model", "NeMoAutoModelForTextToWaveform"),
    "NeMoAutoModelBiEncoder": ("nemo_automodel._transformers.auto_model", "NeMoAutoModelBiEncoder"),
    "NeMoAutoModelCrossEncoder": ("nemo_automodel._transformers.auto_model", "NeMoAutoModelCrossEncoder"),
    "NeMoAutoTokenizer": ("nemo_automodel._transformers.auto_tokenizer", "NeMoAutoTokenizer"),
    "NeMoAutoDiffusionPipeline": ("nemo_automodel._diffusers.auto_diffusion_pipeline", "NeMoAutoDiffusionPipeline"),
    "ModelCapabilities": ("nemo_automodel._transformers.model_capabilities", "ModelCapabilities"),
    "query_capabilities": ("nemo_automodel._transformers.model_capabilities", "query_capabilities"),
}

__all__ = sorted([*_SUBMODULES, "__version__", "__package_name__", *_LAZY_ATTRS.keys()])


# ---------------------------------------------------------------------------
# Model import aliases.
#
# Model implementations are split by the upstream HuggingFace package they are
# written against: ``nemo_automodel._transformers.models`` (transformers) and
# ``nemo_automodel._diffusers.models`` (diffusers). Two legacy flat paths are
# kept working on top of that split:
#
#   nemo_automodel.models.*             -- supported public alias (no warning)
#   nemo_automodel.components.models.*  -- deprecated pre-split location
#
# Implemented as a meta-path finder so it works regardless of whether physical
# directories are shipped in the installation. The actual import of the
# canonical module happens inside exec_module so that _load_unlocked's
# pop-and-set pattern on sys.modules picks up the replacement correctly.
# ---------------------------------------------------------------------------

from ._model_locations import resolve_model_module as _resolve_model_module  # noqa: E402

_MODELS_ALIAS = "nemo_automodel.models"
_LEGACY_MODELS_ALIAS = "nemo_automodel.components.models"


class _AliasLoader(importlib.abc.Loader):
    """Loader that replaces the placeholder module in sys.modules with the
    canonical module during exec_module."""

    def __init__(self, real_name, deprecated_name=None):
        self._real_name = real_name
        self._deprecated_name = deprecated_name

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        if self._deprecated_name is not None:
            warnings.warn(
                f"'{self._deprecated_name}' has moved to '{self._real_name}'. "
                f"The old path still works but will be removed in a future release; "
                f"update imports and YAML '_target_' values accordingly.",
                DeprecationWarning,
                stacklevel=2,
            )
        real = importlib.import_module(self._real_name)
        sys.modules[module.__name__] = real


class _ModelsAliasFinder(importlib.abc.MetaPathFinder):
    """Redirect the legacy flat model paths to their canonical packages.

    Installed at the *front* of ``sys.meta_path`` so it intercepts before the
    default ``PathFinder``.  All import paths resolve to the exact same module
    objects, avoiding duplication.
    """

    _ALIASES = (
        (_MODELS_ALIAS, False),
        (_LEGACY_MODELS_ALIAS, True),
    )

    def find_spec(self, fullname, path, target=None):
        for alias, deprecated in self._ALIASES:
            if fullname == alias:
                suffix = ""
            elif fullname.startswith(alias + "."):
                suffix = fullname[len(alias) + 1 :]
            else:
                continue
            real_name = _resolve_model_module(suffix)
            return importlib.machinery.ModuleSpec(
                fullname,
                _AliasLoader(real_name, fullname if deprecated else None),
                is_package=not suffix,
            )
        return None


sys.meta_path.insert(0, _ModelsAliasFinder())


# ---------------------------------------------------------------------------
# Register a lightweight import hook that widens ``ALLOWED_LAYER_TYPES`` the
# moment ``transformers.configuration_utils`` is loaded. The hook module imports
# only stdlib + logging, so it does NOT force a transformers import at
# ``import nemo_automodel`` time — preserving the lightweight-import promise.
# ---------------------------------------------------------------------------
try:
    from nemo_automodel._transformers.v4_patches.layer_types import (
        install_layer_types_patch_hook as _install_layer_types_patch_hook,
    )

    _install_layer_types_patch_hook()
except Exception:
    # Never let a hook failure break ``import nemo_automodel``.
    pass


def __getattr__(name: str) -> ModuleType | Any:
    """
    Lazily import and cache selected submodules / exported symbols when accessed.

    Raises:
        AttributeError if the name isn't in __all__.
    """
    if name in _SUBMODULES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    if name in _LAZY_ATTRS:
        module_name, attr_name = _LAZY_ATTRS[name]
        module = importlib.import_module(module_name)
        attr = getattr(module, attr_name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """
    Expose the names of all available submodules for auto-completion.
    """
    return sorted(__all__)
