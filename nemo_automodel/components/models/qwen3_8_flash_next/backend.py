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

"""Runtime backend configuration owned by Qwen3.8-Flash-Next."""

from dataclasses import dataclass
from typing import Literal

from nemo_automodel.components.models.common import BackendConfig


@dataclass(kw_only=True)
class Qwen3_8_FlashNextBackendConfig(BackendConfig):
    """Configure this model's optional QSA compressed-block selector.

    qsa_topk="torch" retains the default selector. "deepselect" requires the
    optional CUDA extension and exactly 512 selected compressed blocks.
    Attention, linear and MoE backend settings are inherited unchanged.
    """

    qsa_topk: Literal["torch", "deepselect"] = "torch"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.qsa_topk not in ("torch", "deepselect"):
            raise ValueError(f"Unsupported QSA top-k backend: {self.qsa_topk!r}")
