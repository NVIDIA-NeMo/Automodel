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

"""Runtime backend configuration owned by GLM-5.3-Flash."""

import copy
import dataclasses
from dataclasses import dataclass
from typing import Literal

from nemo_automodel.components.models.common import BackendConfig


@dataclass(kw_only=True)
class Glm5NextBackendConfig(BackendConfig):
    """Select mHC Sinkhorn independently of the sparse-attention backend.

    Attributes:
        sinkhorn: ``"torch"`` preserves eager FP32 normalization. ``"tilelang"``
            requires CUDA and the optional TileKernels package; an unavailable
            kernel raises instead of silently falling back to PyTorch.
        indexer: Score generation for KPool selection. The cuDNN option keeps
            BF16 projections and PyTorch top-k, and requires CUDA and cuDNN DSA.
    """

    sinkhorn: Literal["torch", "tilelang"] = "torch"
    indexer: Literal["torch", "cudnn"] = "torch"

    @classmethod
    def from_backend(cls, backend: BackendConfig) -> "Glm5NextBackendConfig":
        """Return a private GLM config carrying ``backend``'s values.

        The model accepts the shared ``BackendConfig``, which has no ``sinkhorn`` or ``indexer``
        field; those take their ``"torch"`` defaults here, once, so layers read them directly.
        Every shared field is copied by value, and ``__post_init__`` re-validates it.

        Args:
            backend: A shared ``BackendConfig`` or a ``Glm5NextBackendConfig``.

        Returns:
            A new ``Glm5NextBackendConfig``; a ``Glm5NextBackendConfig`` input is shallow-copied.
        """
        if isinstance(backend, cls):
            return copy.copy(backend)
        return cls(**{f.name: getattr(backend, f.name) for f in dataclasses.fields(BackendConfig) if f.init})

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.indexer not in ("torch", "cudnn"):
            raise ValueError(f"Unsupported indexer={self.indexer!r}; expected 'torch' or 'cudnn'.")
        if self.sinkhorn not in ("torch", "tilelang"):
            raise ValueError(f"Unsupported sinkhorn={self.sinkhorn!r}; expected 'torch' or 'tilelang'.")
