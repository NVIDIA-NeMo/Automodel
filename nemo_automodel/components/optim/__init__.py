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

import importlib as _importlib

from .dion import build_dion_optimizer, is_dion_optimizer
from .optimizer import (
    OPTIMIZER_CONFIG_REGISTRY,
    AdamConfig,
    AdamWConfig,
    Dion2Config,
    DionConfig,
    FlashAdamWConfig,
    FusedAdamConfig,
    LRSchedulerConfig,
    MuonConfig,
    NorMuonConfig,
    OptimizerConfig,
    OptimizerFromFactoryConfig,
    ParamGroupOverride,
    build_optimizer,
    build_optimizer_config,
)
from .scheduler import OptimizerParamScheduler

__all__ = [
    "OPTIMIZER_CONFIG_REGISTRY",
    "AdamConfig",
    "AdamWConfig",
    "Dion2Config",
    "DionConfig",
    "FlashAdamWConfig",
    "FusedAdamConfig",
    "LRSchedulerConfig",
    "MuonConfig",
    "NorMuonConfig",
    "OptimizerConfig",
    "OptimizerFromFactoryConfig",
    "ParamGroupOverride",
    "OptimizerParamScheduler",
    "build_optimizer",
    "build_optimizer_config",
    "build_dion_optimizer",
    "is_dion_optimizer",
]

_LAZY_ATTRS = {
    "resolve_storage_dtype": (".precision_warnings", "resolve_storage_dtype"),
    "warn_if_torch_adam_with_bf16_params": (".precision_warnings", "warn_if_torch_adam_with_bf16_params"),
}

__all__ += sorted(_LAZY_ATTRS.keys())


def __getattr__(name: str) -> object:
    """Load an exported component symbol on first access."""
    if name in _LAZY_ATTRS:
        module_path, attr_name = _LAZY_ATTRS[name]
        module = _importlib.import_module(module_path, __name__)
        attr = getattr(module, attr_name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return the component's exported symbols."""
    return sorted(__all__)
