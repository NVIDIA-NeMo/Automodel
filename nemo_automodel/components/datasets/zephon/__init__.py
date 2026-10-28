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

"""Zephon data backend for AutoModel recipes."""

from typing import TYPE_CHECKING

from nemo_automodel.components.datasets.zephon.config import (
    ZephonDataloaderConfig,
    ZephonEncodedTextSourceConfig,
    ZephonInstructionSourceConfig,
    ZephonTextSourceConfig,
)

if TYPE_CHECKING:
    from nemo_automodel.components.datasets.zephon.vlm_config import ZephonVlmDataloaderConfig, ZephonVlmSourceConfig

__all__ = [
    "ZephonDataloaderConfig",
    "ZephonEncodedTextSourceConfig",
    "ZephonInstructionSourceConfig",
    "ZephonTextSourceConfig",
    "ZephonVlmDataloaderConfig",
    "ZephonVlmSourceConfig",
]


def __getattr__(name: str) -> object:
    """Load image configuration only when a VLM caller requests it."""
    if name in ("ZephonVlmDataloaderConfig", "ZephonVlmSourceConfig"):
        from nemo_automodel.components.datasets.zephon.vlm_config import (
            ZephonVlmDataloaderConfig,
            ZephonVlmSourceConfig,
        )

        return {"ZephonVlmDataloaderConfig": ZephonVlmDataloaderConfig, "ZephonVlmSourceConfig": ZephonVlmSourceConfig}[
            name
        ]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
