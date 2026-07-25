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

"""Qwen-Image activation checkpointing boundaries."""

from __future__ import annotations

import logging

from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    checkpoint_wrapper,
)

logger = logging.getLogger(__name__)


def checkpoint_qwen_image_transformer_blocks(model: nn.Module) -> int:
    """Checkpoint all Diffusers Qwen-Image transformer blocks.

    Diffusers Qwen-Image blocks have joint ``attn``, ``img_mlp``, and
    ``txt_mlp`` paths. AutoModel's generic submodule checkpointing does not
    cover both MLP streams, so the model-owned boundary is each complete block.

    Args:
        model: A ``QwenImageTransformer2DModel`` exposing ``transformer_blocks``.

    Returns:
        The number of checkpoint-wrapped transformer blocks.
    """
    blocks = getattr(model, "transformer_blocks", None)
    if blocks is None:
        raise AttributeError("QwenImageTransformer2DModel does not expose `transformer_blocks`.")
    for index, block in enumerate(blocks):
        blocks[index] = checkpoint_wrapper(
            block,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
    logger.info(
        "Qwen-Image activation checkpointing enabled for %d full blocks.",
        len(blocks),
    )
    return len(blocks)
