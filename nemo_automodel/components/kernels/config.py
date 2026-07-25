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
    """Hub kernel settings for native ``BackendConfig.attn="hub"``.

    Attributes:
        attn_repo: Hub repo id for flash attention (default
            ``kernels-community/flash-attn2``).
        attn_version: Kernel major version passed to ``kernels.get_kernel``.
        kernelize_layers: When ``True`` and ``use_kernels=True`` on a custom
            model, apply Transformers ``kernelize`` after construction.
    """

    attn_repo: str | None = HUB_FLASH_ATTN2
    attn_version: int = 1
    kernelize_layers: bool = True
