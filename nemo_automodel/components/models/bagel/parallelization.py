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

"""Model-owned distributed parallelization for BAGEL."""

import logging

from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    checkpoint_wrapper,
)

from nemo_automodel.components.distributed import ModelParallelizer

logger = logging.getLogger(__name__)

_FULL_LAYER_CONTAINERS = (
    "model.language_model.model.layers",
    "model.vit_model.vision_model.encoder.layers",
)


def _module_by_path(module: nn.Module, path: str) -> nn.Module | None:
    value = module
    for part in path.split("."):
        value = getattr(value, part, None)
        if value is None:
            return None
    return value


def _apply_full_layer_checkpointing(model: nn.Module) -> None:
    wrapped_count = 0
    for path in _FULL_LAYER_CONTAINERS:
        container = _module_by_path(model, path)
        if not isinstance(container, (nn.ModuleList, nn.ModuleDict)):
            logger.warning("BAGEL activation checkpointing skipped missing layer container %s", path)
            continue
        items = container.items() if isinstance(container, nn.ModuleDict) else enumerate(container)
        for key, layer in list(items):
            if hasattr(layer, "_checkpoint_wrapped_module"):
                continue
            container[key] = checkpoint_wrapper(layer, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
            wrapped_count += 1
    logger.info("Applied BAGEL full-layer activation checkpointing to %d layers", wrapped_count)


class BagelModelParallelizer(ModelParallelizer):
    """Apply BAGEL whole-layer checkpointing before the generic FSDP2 flow."""

    def _apply(self, model: nn.Module, *args, **kwargs) -> nn.Module:
        activation_checkpointing = kwargs.get("activation_checkpointing", False)
        selective = (
            isinstance(activation_checkpointing, str)
            and activation_checkpointing.lower().replace("-", "_") == "selective"
        )
        if activation_checkpointing and not selective:
            _apply_full_layer_checkpointing(model)
            kwargs["activation_checkpointing"] = False
        return super()._apply(model, *args, **kwargs)


PARALLELIZER = BagelModelParallelizer()

__all__ = ["PARALLELIZER"]


HF_PARALLELIZER = ModelParallelizer(
    layer_group_paths={
        "language": ("model.language_model.model.layers",),
        "vision": ("model.vit_model.vision_model.encoder.layers",),
    }
)
