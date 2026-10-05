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

"""Diffusers-owned adapters for model parallelization sidecars."""

import logging
from collections.abc import Callable
from typing import Dict, Union

import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    CheckpointWrapper,
    checkpoint_wrapper,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy, OffloadPolicy, fully_shard
from torch.distributed.tensor.parallel import ColwiseParallel, ParallelStyle, RowwiseParallel, parallelize_module

from nemo_automodel.components.distributed import ModelParallelizer
from nemo_automodel.components.distributed.mesh_utils import get_fsdp_dp_mesh
from nemo_automodel.components.distributed.parallelizer import apply_fsdp2_sharding_recursively

logger = logging.getLogger(__name__)


class WanModelParallelizer(ModelParallelizer):
    """Apply Wan-specific tensor parallelism, checkpointing, and FSDP."""

    def _apply(
        self,
        model: nn.Module,
        device_mesh: DeviceMesh,
        mp_policy: MixedPrecisionPolicy | None = None,
        offload_policy: OffloadPolicy | None = None,
        sequence_parallel: bool = False,
        activation_checkpointing: bool = False,
        tp_shard_plan: Union[Dict[str, ParallelStyle], str] | None = None,
        dp_replicate_mesh_name: str = "dp_replicate",
        dp_shard_cp_mesh_name: str = "dp_shard_cp",
        tp_mesh_name: str = "tp",
        reapply_trainability: Callable[[nn.Module], None] | None = None,
        **kwargs,
    ) -> nn.Module:
        """Parallelize a Wan transformer."""
        del sequence_parallel, tp_shard_plan
        tp_mesh = device_mesh[tp_mesh_name]
        dp_mesh = get_fsdp_dp_mesh(device_mesh, dp_replicate_mesh_name, dp_shard_cp_mesh_name)

        if tp_mesh.size() > 1:
            try:
                condition = getattr(model, "condition_embedder", None)
                if condition is not None and hasattr(condition, "text_embedder"):
                    condition.text_embedder = parallelize_module(
                        condition.text_embedder,
                        tp_mesh,
                        {"linear_1": ColwiseParallel(), "linear_2": RowwiseParallel()},
                    )
                if condition is not None and hasattr(condition, "time_embedder"):
                    condition.time_embedder = parallelize_module(
                        condition.time_embedder,
                        tp_mesh,
                        {"linear_1": ColwiseParallel(), "linear_2": RowwiseParallel()},
                    )
                if condition is not None and hasattr(condition, "time_proj"):
                    condition.time_proj = parallelize_module(condition.time_proj, tp_mesh, {"": ColwiseParallel()})
            except Exception as error:
                logger.warning("Wan strategy: failed to TP condition embedders: %s", error)

            try:
                for block in getattr(model, "blocks", ()):
                    if hasattr(block, "ffn"):
                        block.ffn = parallelize_module(
                            block.ffn,
                            tp_mesh,
                            {"net.0.proj": ColwiseParallel(), "net.2": RowwiseParallel()},
                        )
                if hasattr(model, "proj_out"):
                    model.proj_out = parallelize_module(model.proj_out, tp_mesh, {"": RowwiseParallel()})
            except Exception as error:
                logger.warning("Wan strategy: failed to TP blocks/proj_out: %s", error)

        if activation_checkpointing and hasattr(model, "blocks"):
            for index in range(len(model.blocks)):
                model.blocks[index] = checkpoint_wrapper(
                    model.blocks[index],
                    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                )

        if mp_policy is None:
            mp_policy = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                output_dtype=torch.float32,
                cast_forward_inputs=False,
            )
        if reapply_trainability is not None:
            reapply_trainability(model)
        apply_fsdp2_sharding_recursively(
            model,
            dp_mesh,
            mp_policy,
            offload_policy,
            kwargs.get("enable_fsdp2_prefetch", True),
            kwargs.get("fsdp2_backward_prefetch_depth", 2),
            kwargs.get("fsdp2_forward_prefetch_depth", 1),
        )
        return fully_shard(
            model,
            mesh=dp_mesh,
            mp_policy=mp_policy,
            offload_policy=offload_policy,
            reshard_after_forward=False,
        )


class HunyuanModelParallelizer(ModelParallelizer):
    """Apply whole-block checkpointing and FSDP to Hunyuan-style models."""

    def _apply(
        self,
        model: nn.Module,
        device_mesh: DeviceMesh,
        mp_policy: MixedPrecisionPolicy | None = None,
        offload_policy: OffloadPolicy | None = None,
        sequence_parallel: bool = False,
        activation_checkpointing: bool = True,
        tp_shard_plan: Union[Dict[str, ParallelStyle], str] | None = None,
        dp_replicate_mesh_name: str = "dp_replicate",
        dp_shard_cp_mesh_name: str = "dp_shard_cp",
        tp_mesh_name: str = "tp",
        reapply_trainability: Callable[[nn.Module], None] | None = None,
        **kwargs,
    ) -> nn.Module:
        """Parallelize a Hunyuan-style transformer."""
        del sequence_parallel, tp_shard_plan, tp_mesh_name
        dp_mesh = get_fsdp_dp_mesh(device_mesh, dp_replicate_mesh_name, dp_shard_cp_mesh_name)
        if mp_policy is None:
            mp_policy = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                output_dtype=torch.bfloat16,
                cast_forward_inputs=False,
            )
        if activation_checkpointing:
            for index in range(len(model.transformer_blocks)):
                model.transformer_blocks[index] = checkpoint_wrapper(
                    model.transformer_blocks[index],
                    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                )
        if reapply_trainability is not None:
            reapply_trainability(model)
        apply_fsdp2_sharding_recursively(
            model,
            dp_mesh,
            mp_policy,
            offload_policy,
            kwargs.get("enable_fsdp2_prefetch", True),
            kwargs.get("fsdp2_backward_prefetch_depth", 2),
            kwargs.get("fsdp2_forward_prefetch_depth", 1),
        )
        return fully_shard(
            model,
            mesh=dp_mesh,
            mp_policy=mp_policy,
            offload_policy=offload_policy,
            reshard_after_forward=False,
        )


class LTX2ModelParallelizer(HunyuanModelParallelizer):
    """Apply Hunyuan-style whole-block checkpointing to LTX-2."""


def _validate_qwen_transformer_blocks(model: nn.Module) -> nn.ModuleList:
    blocks = getattr(model, "transformer_blocks", None)
    if not isinstance(blocks, nn.ModuleList) or not blocks:
        raise TypeError("Qwen image FSDP2 requires a non-empty transformer_blocks nn.ModuleList")
    for index, wrapped_block in enumerate(blocks):
        block = (
            wrapped_block._checkpoint_wrapped_module if isinstance(wrapped_block, CheckpointWrapper) else wrapped_block
        )
        missing = [
            name for name in ("attn", "img_mlp", "txt_mlp") if not isinstance(getattr(block, name, None), nn.Module)
        ]
        if missing:
            raise TypeError(
                f"Qwen transformer_blocks[{index}] is missing required dual-stream modules: {', '.join(missing)}"
            )
    return blocks


def _apply_qwen_block_activation_checkpointing(model: nn.Module) -> None:
    blocks = _validate_qwen_transformer_blocks(model)
    wrapped_count = 0
    for index, block in enumerate(blocks):
        if isinstance(block, CheckpointWrapper):
            continue
        blocks[index] = checkpoint_wrapper(block, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
        wrapped_count += 1
    logger.info("Applied whole-block activation checkpointing to %d Qwen image transformer blocks", wrapped_count)


class QwenImageEditModelParallelizer(ModelParallelizer):
    """Shard Qwen image transformer blocks as complete FSDP2 units."""

    def _apply(self, model: nn.Module, *args, **kwargs) -> nn.Module:
        _validate_qwen_transformer_blocks(model)
        activation_checkpointing = kwargs.get("activation_checkpointing", False)
        selective = (
            isinstance(activation_checkpointing, str)
            and activation_checkpointing.lower().replace("-", "_") == "selective"
        )
        if activation_checkpointing and not selective:
            _apply_qwen_block_activation_checkpointing(model)
            kwargs["activation_checkpointing"] = False
        return super()._apply(model, *args, **kwargs)


_PARALLELIZERS: dict[str, ModelParallelizer] = {
    "HunyuanVideo15Transformer3DModel": HunyuanModelParallelizer(),
    "LTX2VideoTransformer3DModel": LTX2ModelParallelizer(),
    "QwenImageTransformer2DModel": QwenImageEditModelParallelizer(),
    "WanTransformer3DModel": WanModelParallelizer(),
}


def attach_parallelizer(model: nn.Module) -> None:
    """Attach the adapter-owned sidecar for a supported Diffusers model."""
    parallelizer = _PARALLELIZERS.get(type(model).__name__)
    if parallelizer is not None:
        type(model).parallelizer = parallelizer


__all__ = ["attach_parallelizer"]
