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

"""Wan-specific FSDP2 parallelization."""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy, OffloadPolicy
from torch.distributed.tensor.parallel import ColwiseParallel, ParallelStyle, RowwiseParallel

from nemo_automodel.components.distributed import parallelizer
from nemo_automodel.components.distributed.parallelizer import ParallelizationStrategy

logger = logging.getLogger(__name__)


class WanParallelizationStrategy(ParallelizationStrategy):
    """Parallelize Wan transformer modules used by Diffusers."""

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
        **kwargs: Any,
    ) -> nn.Module:
        """Apply Wan-specific tensor parallelism and FSDP2 sharding."""
        del sequence_parallel, tp_shard_plan
        tp_mesh = device_mesh[tp_mesh_name]
        dp_mesh = parallelizer.get_fsdp_dp_mesh(device_mesh, dp_replicate_mesh_name, dp_shard_cp_mesh_name)

        if tp_mesh.size() > 1:
            try:
                if hasattr(model, "condition_embedder"):
                    cond = model.condition_embedder
                    if hasattr(cond, "text_embedder"):
                        cond.text_embedder = parallelizer.parallelize_module(
                            cond.text_embedder,
                            tp_mesh,
                            {"linear_1": ColwiseParallel(), "linear_2": RowwiseParallel()},
                        )
                    if hasattr(cond, "time_embedder"):
                        cond.time_embedder = parallelizer.parallelize_module(
                            cond.time_embedder,
                            tp_mesh,
                            {"linear_1": ColwiseParallel(), "linear_2": RowwiseParallel()},
                        )
                    if hasattr(cond, "time_proj"):
                        cond.time_proj = parallelizer.parallelize_module(
                            cond.time_proj,
                            tp_mesh,
                            {"": ColwiseParallel()},
                        )
            except Exception as error:
                logger.warning("Wan strategy: failed to TP condition embedders: %s", error)

            try:
                if hasattr(model, "blocks"):
                    for block in model.blocks:
                        if hasattr(block, "ffn"):
                            block.ffn = parallelizer.parallelize_module(
                                block.ffn,
                                tp_mesh,
                                {"net.0.proj": ColwiseParallel(), "net.2": RowwiseParallel()},
                            )
                if hasattr(model, "proj_out"):
                    model.proj_out = parallelizer.parallelize_module(model.proj_out, tp_mesh, {"": RowwiseParallel()})
            except Exception as error:
                logger.warning("Wan strategy: failed to TP blocks/proj_out: %s", error)

        if activation_checkpointing and hasattr(model, "blocks"):
            for index in range(len(model.blocks)):
                model.blocks[index] = parallelizer.checkpoint_wrapper(
                    model.blocks[index],
                    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                )

        if not mp_policy:
            mp_policy = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                output_dtype=torch.float32,
            )

        parallelizer.apply_fsdp2_sharding_recursively(
            model,
            dp_mesh,
            mp_policy,
            offload_policy,
            kwargs.get("enable_fsdp2_prefetch", True),
            kwargs.get("fsdp2_backward_prefetch_depth", 2),
            kwargs.get("fsdp2_forward_prefetch_depth", 1),
        )
        return parallelizer.fully_shard(
            model,
            mesh=dp_mesh,
            mp_policy=mp_policy,
            offload_policy=offload_policy,
            reshard_after_forward=False,
        )


_STRATEGY = WanParallelizationStrategy()


def get_parallelization_strategy() -> ParallelizationStrategy:
    """Return Wan's lazily requested strategy instance."""
    return _STRATEGY
