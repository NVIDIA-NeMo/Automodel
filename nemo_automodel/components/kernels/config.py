# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Declarative Hub kernel settings for native models."""

from __future__ import annotations

from dataclasses import dataclass

from nemo_automodel.components.kernels.hub import HUB_FLASH_ATTN2


@dataclass(kw_only=True)
class HubKernelConfig:
    """Hub kernel settings for attention resolution and native-model kernelize.

    Layer replacements (RMSNorm, MLP, Linear, activations, RoPE) should use
    ``use_kernels=True`` on ``NeMoAutoModel.from_pretrained`` / ``from_config``
    so Transformers' ``hub_kernels.kernelize`` owns the mapping. This config
    focuses on attention repo selection for native MLA/GQA factories and optional
    ``backend.hub_kernels`` overrides in recipes.

    Attributes:
        attn_repo: Hub repo id for flash attention (e.g.
            ``kernels-community/flash-attn2``). When set on ``BackendConfig``,
            overrides a top-level ``attn_implementation="hub"`` alias.
        attn_version: Kernel major version branch passed to ``get_kernel``.
        kernelize_layers: When True and ``use_kernels=True`` on a custom model,
            apply ``kernelize`` after construction. Ignored on the HF load path
            (Transformers kernelizes during ``from_pretrained``).
    """

    attn_repo: str | None = HUB_FLASH_ATTN2
    attn_version: int = 1
    kernelize_layers: bool = True
