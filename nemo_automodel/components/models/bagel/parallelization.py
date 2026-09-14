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

"""BAGEL-owned activation checkpointing: whole Qwen2 decoder and SigLIP encoder layers."""

from __future__ import annotations

import logging

from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl, checkpoint_wrapper

logger = logging.getLogger(__name__)

_BAGEL_FULL_LAYER_CHECKPOINT_MODULE_LISTS = (
    "model.language_model.model.layers",
    "model.vit_model.vision_model.encoder.layers",
)


def _get_module_by_fqn(module: nn.Module, fqn: str) -> nn.Module | None:
    obj = module
    for part in fqn.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def _is_checkpoint_wrapped(module: nn.Module) -> bool:
    return hasattr(module, "_checkpoint_wrapped_module")


def apply_bagel_full_layer_activation_checkpointing(model: nn.Module) -> bool:
    """Apply native BAGEL-style activation checkpointing to whole logical layers."""
    wrapped_count = 0
    for fqn in _BAGEL_FULL_LAYER_CHECKPOINT_MODULE_LISTS:
        container = _get_module_by_fqn(model, fqn)
        if container is None:
            logger.warning("BAGEL activation checkpointing skipped missing module list %s", fqn)
            continue
        if not isinstance(container, (nn.ModuleList, nn.ModuleDict)):
            logger.warning(
                "BAGEL activation checkpointing expected %s to be a module list, got %s",
                fqn,
                type(container),
            )
            continue

        items = container.items() if isinstance(container, nn.ModuleDict) else enumerate(container)
        for key, layer in list(items):
            if _is_checkpoint_wrapped(layer):
                continue
            container[key] = checkpoint_wrapper(layer, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
            wrapped_count += 1

    logger.info("Applied BAGEL full-layer activation checkpointing to %d layers", wrapped_count)
    return wrapped_count > 0
