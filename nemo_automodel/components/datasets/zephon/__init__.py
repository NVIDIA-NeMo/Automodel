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

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nemo_automodel.components.datasets.zephon.config import (
        ZephonDataloaderConfig,
        ZephonPretokenizedTextDatasetConfig,
        ZephonTextDatasetConfig,
    )

__all__ = [
    "ZephonDataloaderConfig",
    "ZephonPretokenizedTextDatasetConfig",
    "ZephonTextDatasetConfig",
]


def __getattr__(name: str) -> type:
    """Load configuration only when requested, keeping worker helpers lightweight."""
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(".config", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose the public configuration classes before loading them."""
    return sorted(__all__)
