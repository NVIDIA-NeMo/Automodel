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

"""Hunyuan Video-specific FSDP2 parallelization."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy, OffloadPolicy
from torch.distributed.tensor.parallel import ParallelStyle

from nemo_automodel.components.distributed import parallelizer
from nemo_automodel.components.distributed.parallelizer import ParallelizationStrategy


class HunyuanParallelizationStrategy(ParallelizationStrategy):
    """Parallelize HunyuanVideo transformer modules used by Diffusers."""

    def parallelize(
        self,
        model: nn.Module,
        device_mesh: DeviceMesh,
        mp_policy: MixedPrecisionPolicy | None = None,
        offload_policy: OffloadPolicy | None = None,
        sequence_parallel: bool = False,
        activation_checkpointing: bool = True,
        tp_shard_plan: dict[str, ParallelStyle] | str | None = None,
        dp_replicate_mesh_name: str = "dp_replicate",
        dp_shard_cp_mesh_name: str = "dp_shard_cp",
        tp_mesh_name: str = "tp",
        **kwargs: Any,
    ) -> nn.Module:
        """Apply HunyuanVideo activation checkpointing and FSDP2 sharding."""
        del sequence_parallel, tp_shard_plan, tp_mesh_name
        dp_mesh = parallelizer.get_fsdp_dp_mesh(device_mesh, dp_replicate_mesh_name, dp_shard_cp_mesh_name)

        if not mp_policy:
            mp_policy = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                output_dtype=torch.bfloat16,
            )
        if activation_checkpointing:
            for index in range(len(model.transformer_blocks)):
                model.transformer_blocks[index] = parallelizer.checkpoint_wrapper(
                    model.transformer_blocks[index],
                    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
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


_STRATEGY = HunyuanParallelizationStrategy()


def get_parallelization_strategy() -> ParallelizationStrategy:
    """Return Hunyuan Video's lazily requested strategy instance."""
    return _STRATEGY
