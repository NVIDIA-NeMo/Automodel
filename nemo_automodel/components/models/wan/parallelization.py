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

"""FSDP2/TP strategy for the diffusers ``WanTransformer3DModel``."""

from __future__ import annotations

import logging
from collections.abc import Callable

import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl, checkpoint_wrapper
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy, OffloadPolicy, fully_shard
from torch.distributed.tensor.parallel import ColwiseParallel, ParallelStyle, RowwiseParallel, parallelize_module

from nemo_automodel.components.distributed.mesh_utils import get_fsdp_dp_mesh
from nemo_automodel.components.distributed.parallel_spec import ParallelSpec
from nemo_automodel.components.distributed.parallelizer import ParallelizationStrategy, apply_fsdp2_sharding_recursively

logger = logging.getLogger(__name__)


class WanParallelizationStrategy(ParallelizationStrategy):
    """Parallelization strategy for Wan-style transformer modules used in Diffusers.

    Applies TP to condition embedders, FFN projections in each block, and final projection,
    then applies FSDP sharding similarly to other strategies.
    """

    def parallelize(
        self,
        model: nn.Module,
        device_mesh: DeviceMesh,
        mp_policy: MixedPrecisionPolicy | None = None,
        offload_policy: OffloadPolicy | None = None,
        sequence_parallel: bool = False,
        activation_checkpointing: bool = False,
        tp_shard_plan: dict[str, ParallelStyle] | str | None = None,
        dp_replicate_mesh_name: str = "dp_replicate",
        dp_shard_cp_mesh_name: str = "dp_shard_cp",
        tp_mesh_name: str = "tp",
        reapply_trainability: Callable[[nn.Module], None] | None = None,
        **kwargs,
    ) -> nn.Module:
        # Not using custom tp_shard_plan; apply Wan-specific plan
        tp_mesh = device_mesh[tp_mesh_name]
        dp_mesh = get_fsdp_dp_mesh(device_mesh, dp_replicate_mesh_name, dp_shard_cp_mesh_name)

        # Apply TP only when TP group size > 1
        if tp_mesh.size() > 1:
            # Condition embedders if present
            try:
                if hasattr(model, "condition_embedder"):
                    cond = model.condition_embedder
                    if hasattr(cond, "text_embedder"):
                        cond.text_embedder = parallelize_module(
                            cond.text_embedder,
                            tp_mesh,
                            {
                                "linear_1": ColwiseParallel(),
                                "linear_2": RowwiseParallel(),
                            },
                        )
                    if hasattr(cond, "time_embedder"):
                        cond.time_embedder = parallelize_module(
                            cond.time_embedder,
                            tp_mesh,
                            {
                                "linear_1": ColwiseParallel(),
                                "linear_2": RowwiseParallel(),
                            },
                        )
                    if hasattr(cond, "time_proj"):
                        cond.time_proj = parallelize_module(
                            cond.time_proj,
                            tp_mesh,
                            {"": ColwiseParallel()},
                        )
            except Exception as e:
                logger.warning(f"Wan strategy: failed to TP condition embedders: {e}")

            # Blocks FFN and final projection
            try:
                if hasattr(model, "blocks"):
                    for block in model.blocks:
                        if hasattr(block, "ffn"):
                            block.ffn = parallelize_module(
                                block.ffn,
                                tp_mesh,
                                {
                                    "net.0.proj": ColwiseParallel(),
                                    "net.2": RowwiseParallel(),
                                },
                            )
                if hasattr(model, "proj_out"):
                    model.proj_out = parallelize_module(model.proj_out, tp_mesh, {"": RowwiseParallel()})
            except Exception as e:
                logger.warning(f"Wan strategy: failed to TP blocks/proj_out: {e}")

        # Activation checkpointing wraps every WanTransformerBlock so its
        # forward activations are recomputed on backward instead of being
        # held in memory. Critical for Wan2.2-A14B (14B params, ~30k-token
        # video sequence) — without this, fp32 layer-norm casts in the block
        # forward will OOM even on 8x80GB H100.
        if activation_checkpointing and hasattr(model, "blocks"):
            for idx in range(len(model.blocks)):
                model.blocks[idx] = checkpoint_wrapper(
                    model.blocks[idx],
                    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                )

        # Mixed precision default like Default strategy
        if not mp_policy:
            mp_policy = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                output_dtype=torch.float32,
            )

        if reapply_trainability is not None:
            reapply_trainability(model)

        # Apply FSDP sharding recursively and to root
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


class WanTransformer3DModel:
    """Contract for the diffusers ``WanTransformer3DModel``; bound by the diffusion pipeline before sharding."""

    parallel_spec: ParallelSpec = ParallelSpec(strategy=WanParallelizationStrategy())
