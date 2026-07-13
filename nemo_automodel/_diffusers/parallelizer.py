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

"""Lazy dispatcher for Diffusers model-owned parallelizers."""

from __future__ import annotations

from importlib import import_module

from torch import nn

from nemo_automodel.components.distributed.parallelizer import ParallelizationStrategy

_STRATEGY_FACTORIES: dict[str, tuple[str, str]] = {
    "WanTransformer3DModel": ("nemo_automodel._diffusers.wan.parallelizer", "get_parallelization_strategy"),
    "HunyuanVideo15Transformer3DModel": (
        "nemo_automodel._diffusers.hunyuan_video.parallelizer",
        "get_parallelization_strategy",
    ),
}


def get_parallelization_strategy(model: nn.Module) -> ParallelizationStrategy | None:
    """Return the matching model-owned strategy without loading unrelated ones."""
    strategy_factory = _STRATEGY_FACTORIES.get(type(model).__name__)
    if strategy_factory is None:
        return None

    module_path, factory_name = strategy_factory
    factory = getattr(import_module(module_path), factory_name)
    return factory()
