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

"""FSDP2 strategy for the diffusers ``QwenImageTransformer2DModel`` (Qwen-Image and Qwen-Image-Edit)."""

from __future__ import annotations

import logging

from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    CheckpointWrapper,
    checkpoint_wrapper,
)

from nemo_automodel.components.distributed.parallel_spec import ParallelSpec
from nemo_automodel.components.distributed.parallelizer import DefaultParallelizationStrategy

logger = logging.getLogger(__name__)


def _unwrap_qwen_checkpointed_block(block: nn.Module) -> nn.Module:
    """Return the Qwen transformer block held by a checkpoint wrapper."""
    if isinstance(block, CheckpointWrapper):
        return block._checkpoint_wrapped_module
    return block


def _validate_qwen_transformer_blocks(model: nn.Module) -> nn.ModuleList:
    """Validate the upstream Qwen dual-stream block structure.

    Args:
        model: Upstream Qwen image transformer whose ``transformer_blocks``
            container must hold dual-stream image/text blocks.

    Returns:
        The model's ``transformer_blocks`` ModuleList. Returned modules alias
        the model-owned blocks.
    """
    blocks = getattr(model, "transformer_blocks", None)
    if not isinstance(blocks, nn.ModuleList) or not blocks:
        raise TypeError("Qwen image FSDP2 requires a non-empty transformer_blocks nn.ModuleList")

    required_branches = ("attn", "img_mlp", "txt_mlp")
    for index, wrapped_block in enumerate(blocks):
        block = _unwrap_qwen_checkpointed_block(wrapped_block)
        missing = [name for name in required_branches if not isinstance(getattr(block, name, None), nn.Module)]
        if missing:
            raise TypeError(
                f"Qwen transformer_blocks[{index}] is missing required dual-stream modules: {', '.join(missing)}"
            )
    return blocks


def _apply_qwen_block_activation_checkpointing(model: nn.Module) -> None:
    """Checkpoint complete Qwen blocks, including both image and text MLPs.

    Args:
        model: Upstream Qwen image transformer. Each block consumes image
            hidden states with shape [batch, image_tokens, hidden], text hidden
            states with shape [batch, text_tokens, hidden], a text mask with
            shape [batch, text_tokens], timestep embeddings with shape [batch,
            hidden], and rotary tensors whose leading layout is owned by
            Diffusers. Its block outputs preserve the image/text layouts.
    """
    blocks = _validate_qwen_transformer_blocks(model)
    wrapped_count = 0
    for index, block in enumerate(blocks):
        if isinstance(block, CheckpointWrapper):
            continue
        blocks[index] = checkpoint_wrapper(block, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
        wrapped_count += 1
    logger.info("Applied whole-block activation checkpointing to %d Qwen image transformer blocks", wrapped_count)


class QwenImageEditParallelizationStrategy(DefaultParallelizationStrategy):
    """Shard upstream Qwen image transformer blocks as complete FSDP2 units.

    Applies whole-block activation checkpointing that covers the attention,
    image-MLP, and text-MLP branches of each dual-stream block, then delegates
    TP planning and FSDP2 sharding to the default strategy.
    """

    def parallelize(self, model: nn.Module, *args, **kwargs) -> nn.Module:
        """Apply Qwen block checkpointing followed by the standard FSDP2 flow.

        Args:
            model: Upstream Qwen image transformer. Tensor layouts are unchanged
                by sharding; each dual-stream block consumes image tensors of
                shape [batch, image_tokens, hidden] and text tensors of shape
                [batch, text_tokens, hidden].
            *args: Positional arguments forwarded to the default strategy.
            **kwargs: Keyword arguments accepted by
                :meth:`DefaultParallelizationStrategy.parallelize`.

        Returns:
            The same upstream model with its parameters represented by FSDP2
            DTensors on distributed runs. Global tensor shapes and upstream
            Diffusers state-dict keys are preserved.
        """
        _validate_qwen_transformer_blocks(model)
        activation_checkpointing = kwargs.get("activation_checkpointing", False)
        selective_checkpointing = (
            isinstance(activation_checkpointing, str)
            and activation_checkpointing.lower().replace("-", "_") == "selective"
        )
        if activation_checkpointing and not selective_checkpointing:
            _apply_qwen_block_activation_checkpointing(model)
            kwargs["activation_checkpointing"] = False

        return super().parallelize(model, *args, **kwargs)


class QwenImageTransformer2DModel:
    """Contract for the diffusers ``QwenImageTransformer2DModel``; bound by the diffusion pipeline before sharding."""

    parallel_spec: ParallelSpec = ParallelSpec(strategy=QwenImageEditParallelizationStrategy())
