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

"""Public exports for the model registry package."""

from nemo_automodel._transformers.registry.base import _BaseModelRegistry
from nemo_automodel._transformers.registry.model_package_spec import ModelPackageSpec

_MODEL_REGISTRY_EXPORTS = {
    "MODEL_PACKAGE_SPECS",
    "RETRIEVAL_MODEL_PACKAGE_SPECS",
    "ModelRegistry",
    "RetrievalModelRegistry",
    "make_registry",
    "make_retrieval_registry",
}


def __getattr__(name: str):
    """Lazily expose model-specific registry globals without import cycles."""
    if name not in _MODEL_REGISTRY_EXPORTS:
        raise AttributeError(name)
    from nemo_automodel._transformers.registry import model_registry

    return getattr(model_registry, name)


__all__ = [
    "MODEL_PACKAGE_SPECS",
    "RETRIEVAL_MODEL_PACKAGE_SPECS",
    "ModelRegistry",
    "RetrievalModelRegistry",
    "ModelPackageSpec",
    "_BaseModelRegistry",
    "make_registry",
    "make_retrieval_registry",
]
