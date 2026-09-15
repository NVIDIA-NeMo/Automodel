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

"""FSDP2 strategy for the diffusers ``HunyuanVideo15Transformer3DModel``."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl, checkpoint_wrapper
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy, OffloadPolicy, fully_shard
from torch.distributed.tensor.parallel import ParallelStyle

from nemo_automodel.components.distributed.mesh_utils import get_fsdp_dp_mesh
from nemo_automodel.components.distributed.parallel_spec import ParallelSpec
from nemo_automodel.components.distributed.parallelizer import ParallelizationStrategy, apply_fsdp2_sharding_recursively


class HunyuanParallelizationStrategy(ParallelizationStrategy):
    """Parallelization strategy for Hunyuan-style transformer modules used in HunyuanVideo."""

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
        reapply_trainability: Callable[[nn.Module], None] | None = None,
        **kwargs,
    ) -> nn.Module:
        dp_mesh = get_fsdp_dp_mesh(device_mesh, dp_replicate_mesh_name, dp_shard_cp_mesh_name)

        # Mixed precision default like Default strategy
        if not mp_policy:
            mp_policy = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                output_dtype=torch.bfloat16,
            )
        # Apply activation checkpointing to transformer blocks if requested
        if activation_checkpointing:
            for idx in range(len(model.transformer_blocks)):
                model.transformer_blocks[idx] = checkpoint_wrapper(
                    model.transformer_blocks[idx],
                    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
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


class HunyuanVideo15Transformer3DModel:
    """Contract for the diffusers ``HunyuanVideo15Transformer3DModel``; bound by the diffusion pipeline before sharding."""

    parallel_spec: ParallelSpec = ParallelSpec(strategy=HunyuanParallelizationStrategy())
