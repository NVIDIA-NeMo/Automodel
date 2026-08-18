# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Retrieval and embedding models built on NeMo Automodel bridges.

The package owns task-level retrieval APIs and concrete embedding model
families. Bridge machinery remains in :mod:`nemo_automodel._transformers`,
while reusable training primitives remain in :mod:`nemo_automodel.components`.

Imports are lazy so ``import nemo_automodel.retrieval`` does not eagerly load
the task implementation, exporter, or any concrete model family.
"""

import importlib
from types import ModuleType
from typing import Any

_SUBMODULES = {
    "auto_model",
    "inbatch_negatives",
    "modeling",
    "models",
    "sentence_transformer_export",
    "state_dict_adapter",
}

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "NeMoAutoModelBiEncoder": ("nemo_automodel.retrieval.auto_model", "NeMoAutoModelBiEncoder"),
    "NeMoAutoModelCrossEncoder": ("nemo_automodel.retrieval.auto_model", "NeMoAutoModelCrossEncoder"),
    "BiEncoderModel": ("nemo_automodel.retrieval.modeling", "BiEncoderModel"),
    "CrossEncoderModel": ("nemo_automodel.retrieval.modeling", "CrossEncoderModel"),
    "RetrieverStudentWithProjection": (
        "nemo_automodel.retrieval.modeling",
        "RetrieverStudentWithProjection",
    ),
    "RetrieverTeacherEmbeddingEncoder": (
        "nemo_automodel.retrieval.modeling",
        "RetrieverTeacherEmbeddingEncoder",
    ),
    "EncoderStateDictAdapter": ("nemo_automodel.retrieval.state_dict_adapter", "EncoderStateDictAdapter"),
    "SentenceTransformerExportConfig": (
        "nemo_automodel.retrieval.sentence_transformer_export",
        "SentenceTransformerExportConfig",
    ),
}

__all__ = sorted([*_SUBMODULES, *_LAZY_ATTRS])


def __getattr__(name: str) -> ModuleType | Any:
    """Lazily resolve retrieval submodules and public symbols."""
    if name in _SUBMODULES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    if name in _LAZY_ATTRS:
        module_name, attr_name = _LAZY_ATTRS[name]
        attr = getattr(importlib.import_module(module_name), attr_name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return public retrieval symbols for interactive discovery."""
    return sorted(__all__)
