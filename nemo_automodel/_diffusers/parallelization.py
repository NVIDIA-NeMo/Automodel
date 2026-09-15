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

"""Binds model-owned ``ParallelSpec`` declarations onto the diffusers transformers NeMo AutoModel trains."""

from __future__ import annotations

import re

from torch import nn

from nemo_automodel.components.models import declared_parallel_spec

_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def diffusers_family(class_name: str) -> str:
    """Model package under ``components/models`` that declares the contract for a diffusers transformer class.

    The package is the snake_case stem before ``Transformer``: ``WanTransformer3DModel`` -> ``wan``,
    ``HunyuanVideo15Transformer3DModel`` -> ``hunyuan_video15``, ``LTX2VideoTransformer3DModel`` ->
    ``ltx2_video``, ``QwenImageTransformer2DModel`` -> ``qwen_image``.
    """
    return _CAMEL_BOUNDARY.sub("_", class_name.split("Transformer", 1)[0]).lower()


def attach_parallel_spec(module: nn.Module) -> nn.Module:
    """Bind the ``ParallelSpec`` declared for a diffusers transformer onto ``module``'s class.

    Mirrors the HF bridge: ``module.__class__`` becomes a dynamic subclass carrying
    ``parallel_spec``, so ``query_parallel_spec`` finds it without touching the upstream class.
    Subclasses inherit through the MRO; modules without a declaration are returned unchanged.
    """
    cls = type(module)
    if hasattr(cls, "parallel_spec"):
        return module
    for base in cls.__mro__:
        spec = declared_parallel_spec(diffusers_family(base.__name__), base.__name__)
        if spec is not None:
            namespace = {"parallel_spec": spec, "__module__": cls.__module__, "__qualname__": cls.__qualname__}
            module.__class__ = type(cls.__name__, (cls,), namespace)
            break
    return module
