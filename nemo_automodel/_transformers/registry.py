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


# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/registry.py

import importlib
import logging
import pkgutil
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Type, Union

import torch.nn as nn

logger = logging.getLogger(__name__)

MODELING_PATH = ["nemo_automodel.components.models"]


@dataclass
class _ModelRegistry:
    # Keyed by model_arch
    modeling_path: List[str] = field(default_factory=list)
    model_arch_name_to_cls: Dict[str, Union[Type[nn.Module], str]] = field(default_factory=dict)

    def __post_init__(self):
        for modeling_path in self.modeling_path:
            self._mapping_model_arch_name_to_cls(modeling_path)

    @property
    def supported_models(self) -> Dict[str, Type[nn.Module]]:
        return self.model_arch_name_to_cls.keys()

    def get_model_cls_from_model_arch(self, model_arch: str) -> Type[nn.Module]:
        return self.model_arch_name_to_cls[model_arch]

    def register_modeling_path(self, path: str) -> None:
        """Add a new modeling path and register models from it."""
        if path not in self.modeling_path:
            self.modeling_path.append(path)
            self._mapping_model_arch_name_to_cls(path)

    def _mapping_model_arch_name_to_cls(self, modeling_path: str):
        package = importlib.import_module(modeling_path)
        for _, name, ispkg in pkgutil.walk_packages(package.__path__, modeling_path + "."):
            if not ispkg:
                try:
                    module = importlib.import_module(name)
                except Exception as e:
                    logger.warning(f"Ignore import error when loading {name}. {e}")
                    continue
                if hasattr(module, "ModelClass"):
                    entry = module.ModelClass
                    if isinstance(entry, list):
                        for tmp in entry:
                            assert tmp.__name__ not in self.model_arch_name_to_cls, (
                                f"Duplicated model implementation for {tmp.__name__}"
                            )
                            self.model_arch_name_to_cls[tmp.__name__] = tmp
                    else:
                        assert entry.__name__ not in self.model_arch_name_to_cls, (
                            f"Duplicated model implementation for {entry.__name__}"
                        )
                        self.model_arch_name_to_cls[entry.__name__] = entry


@lru_cache
def get_registry():
    return _ModelRegistry(modeling_path=MODELING_PATH)


ModelRegistry = get_registry()

# Currently applying manual overrides for specific architecture name mappings
from nemo_automodel.components.models.qwen3_omni_moe.model import (
    Qwen3OmniMoeThinkerForConditionalGeneration,
)

ModelRegistry.model_arch_name_to_cls["Qwen3OmniMoeForConditionalGeneration"] = (
    Qwen3OmniMoeThinkerForConditionalGeneration
)
