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

"""Tiny Hy4-flash configuration with every distinguishing feature enabled (CPU-sized)."""

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.hy_v4_flash.config import HY4FlashConfig, HY4FlashTextConfig

from .tiny_kwargs import TINY_TEXT_KWARGS

CPU_BACKEND = BackendConfig(
    attn="sdpa", linear="torch", rms_norm="torch", experts="torch", dispatcher="torch", rope_fusion=False
)


def tiny_text_config(**overrides: object) -> HY4FlashTextConfig:
    """Six layers: sparse, sparse, swa, swa, swa, sparse; dense MLP first; Engram on layers 1 and 3 (ids 2, 4)."""
    return HY4FlashTextConfig(**{**TINY_TEXT_KWARGS, **overrides})


def tiny_config(**overrides: object) -> HY4FlashConfig:
    """Composite config around :func:`tiny_text_config`."""
    return HY4FlashConfig(text_config=tiny_text_config(**overrides).to_dict())
