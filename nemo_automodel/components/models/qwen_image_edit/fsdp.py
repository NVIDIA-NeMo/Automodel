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

"""Model-owned FSDP2 and activation-checkpoint strategy for Qwen image edit."""

from __future__ import annotations

import logging
from typing import Any, cast

import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    CheckpointWrapper,
    checkpoint_wrapper,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy, OffloadPolicy
from torch.distributed.tensor.parallel import ParallelStyle

from nemo_automodel.components.distributed.config import (
    ActivationCheckpointingMode,
    ActivationCheckpointingScope,
)
from nemo_automodel.components.distributed.parallelizer import (
    PARALLELIZATION_STRATEGIES,
    DefaultParallelizationStrategy,
    register_parallel_strategy,
)

logger = logging.getLogger(__name__)

_MODEL_CLASS_NAME = "QwenImageTransformer2DModel"


def _unwrap_checkpointed_block(block: nn.Module) -> nn.Module:
    """Return the Qwen transformer block held by a checkpoint wrapper."""
    if isinstance(block, CheckpointWrapper):
        return block._checkpoint_wrapped_module
    return block


def _validate_transformer_blocks(model: nn.Module) -> nn.ModuleList:
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
        block = _unwrap_checkpointed_block(wrapped_block)
        missing = [name for name in required_branches if not isinstance(getattr(block, name, None), nn.Module)]
        if missing:
            raise TypeError(
                f"Qwen transformer_blocks[{index}] is missing required dual-stream modules: {', '.join(missing)}"
            )
    return blocks


def _apply_block_activation_checkpointing(model: nn.Module) -> None:
    """Checkpoint complete Qwen blocks, including both image and text MLPs.

    Args:
        model: Upstream Qwen image transformer. Each block consumes image
            hidden states with shape [batch, image_tokens, hidden], text hidden
            states with shape [batch, text_tokens, hidden], a text mask with
            shape [batch, text_tokens], timestep embeddings with shape [batch,
            hidden], and rotary tensors whose leading layout is owned by
            Diffusers. Its block outputs preserve the image/text layouts.
    """
    blocks = _validate_transformer_blocks(model)
    wrapped_count = 0
    for index, block in enumerate(blocks):
        if isinstance(block, CheckpointWrapper):
            continue
        blocks[index] = checkpoint_wrapper(block, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
        wrapped_count += 1
    logger.info("Applied whole-block activation checkpointing to %d Qwen image transformer blocks", wrapped_count)


class QwenImageEditParallelizationStrategy(DefaultParallelizationStrategy):
    """Shard upstream Qwen transformer blocks as complete FSDP2 units."""

    def parallelize(
        self,
        model: nn.Module,
        device_mesh: DeviceMesh,
        mp_policy: MixedPrecisionPolicy | None = None,
        offload_policy: OffloadPolicy | None = None,
        sequence_parallel: bool = False,
        activation_checkpointing: ActivationCheckpointingMode = False,
        tp_shard_plan: dict[str, ParallelStyle] | str | None = None,
        dp_replicate_mesh_name: str = "dp_replicate",
        dp_shard_cp_mesh_name: str = "dp_shard_cp",
        tp_mesh_name: str = "tp",
        enable_async_tensor_parallel: bool = False,
        enable_compile: bool = False,
        enable_fsdp2_prefetch: bool = True,
        fsdp2_backward_prefetch_depth: int = 2,
        fsdp2_forward_prefetch_depth: int = 1,
        reshard_after_forward: bool | None = None,
        activation_checkpointing_scope: ActivationCheckpointingScope | None = "all",
        fully_shard_fn: Any = None,
    ) -> nn.Module:
        """Apply model-owned checkpointing followed by the standard FSDP2 flow.

        Args:
            model: Upstream Qwen image transformer. Tensor layouts are unchanged
                by sharding; each dual-stream block consumes image tensors of
                shape [batch, image_tokens, hidden] and text tensors of shape
                [batch, text_tokens, hidden].
            device_mesh: Device mesh containing the data-parallel and tensor-
                parallel axes used by the standard strategy.
            mp_policy: Optional FSDP2 mixed-precision policy.
            offload_policy: Optional FSDP2 parameter offload policy.
            sequence_parallel: Whether to apply the standard sequence-parallel plan.
            activation_checkpointing: ``True`` for whole-block checkpointing,
                ``"selective"`` for the repository's selective policy, or
                ``False`` to disable recomputation.
            tp_shard_plan: Optional tensor-parallel module plan.
            dp_replicate_mesh_name: Replicated-data-parallel mesh axis name.
            dp_shard_cp_mesh_name: Flattened FSDP/context-parallel mesh axis name.
            tp_mesh_name: Tensor-parallel mesh axis name.
            enable_async_tensor_parallel: Whether to enable asynchronous TP.
            enable_compile: Whether per-layer compilation is enabled.
            enable_fsdp2_prefetch: Whether to configure FSDP2 prefetch chains.
            fsdp2_backward_prefetch_depth: Backward prefetch chain depth.
            fsdp2_forward_prefetch_depth: Forward prefetch chain depth.
            reshard_after_forward: Optional per-block reshard override.
            activation_checkpointing_scope: Repository activation-checkpoint scope.
            fully_shard_fn: Optional FSDP2 sharding callable supplied by the
                distributed manager.

        Returns:
            The same upstream model with its parameters represented by FSDP2
            DTensors on distributed runs. Global tensor shapes and upstream
            Diffusers state-dict keys are preserved.
        """
        _validate_transformer_blocks(model)
        selective_checkpointing = (
            isinstance(activation_checkpointing, str)
            and activation_checkpointing.lower().replace("-", "_") == "selective"
        )
        if activation_checkpointing and not selective_checkpointing:
            _apply_block_activation_checkpointing(model)

        # The default strategy handles the string value at runtime, although
        # its legacy public annotation still declares this argument as bool.
        delegated_activation_checkpointing = cast(bool, activation_checkpointing) if selective_checkpointing else False

        return super().parallelize(
            model=model,
            device_mesh=device_mesh,
            mp_policy=mp_policy,
            offload_policy=offload_policy,
            sequence_parallel=sequence_parallel,
            activation_checkpointing=delegated_activation_checkpointing,
            tp_shard_plan=tp_shard_plan,
            dp_replicate_mesh_name=dp_replicate_mesh_name,
            dp_shard_cp_mesh_name=dp_shard_cp_mesh_name,
            tp_mesh_name=tp_mesh_name,
            enable_async_tensor_parallel=enable_async_tensor_parallel,
            enable_compile=enable_compile,
            enable_fsdp2_prefetch=enable_fsdp2_prefetch,
            fsdp2_backward_prefetch_depth=fsdp2_backward_prefetch_depth,
            fsdp2_forward_prefetch_depth=fsdp2_forward_prefetch_depth,
            reshard_after_forward=reshard_after_forward,
            activation_checkpointing_scope=activation_checkpointing_scope,
            fully_shard_fn=fully_shard_fn,
        )


def register_qwen_image_edit_parallel_strategy() -> None:
    """Register the Qwen image transformer strategy exactly once.

    Raises:
        RuntimeError: If another strategy already owns the upstream Diffusers
            Qwen transformer class name.
    """
    existing = PARALLELIZATION_STRATEGIES.get(_MODEL_CLASS_NAME)
    if isinstance(existing, QwenImageEditParallelizationStrategy):
        return
    if existing is not None:
        raise RuntimeError(
            f"Parallel strategy {_MODEL_CLASS_NAME!r} is already registered by {type(existing).__name__}"
        )
    register_parallel_strategy(name=_MODEL_CLASS_NAME)(QwenImageEditParallelizationStrategy)
