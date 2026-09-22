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

"""Model-owned FSDP and activation-checkpoint policy for Wan-Animate-2."""

from typing import Any

from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl, checkpoint_wrapper
from torch.distributed.device_mesh import DeviceMesh

from nemo_automodel.components.distributed.activation_checkpointing import is_selective_activation_checkpointing
from nemo_automodel.components.distributed.parallelizer import DefaultParallelizationStrategy
from nemo_automodel.components.models.wan_animate2.interleaved import install_forward_origin


class WanAnimate2ParallelizationStrategy(DefaultParallelizationStrategy):
    """Checkpoint both streams together before the default per-block sharding."""

    def parallelize(self, model: nn.Module, device_mesh: DeviceMesh, **kwargs: Any) -> nn.Module:
        """Install the training forward and wrap whole blocks before FSDP.

        Args:
            model: Upstream transformer. Blocks consume generation/reference
                tensors [batch, tokens, hidden] and return the same layouts.
            device_mesh: Data-parallel device mesh; TP, CP, and PP must be one.
            **kwargs: Arguments required by the shared parallelization interface.

        Returns:
            The same model with checkpointed blocks and FSDP2 parameters.
        """
        for name in (kwargs.get("tp_mesh_name", "tp"), "cp", "pp"):
            if name in device_mesh.mesh_dim_names and device_mesh[name].size() > 1:
                raise ValueError(f"Wan-Animate-2 does not support {name} parallelism; set {name}_size=1")
        checkpointing = kwargs.pop("activation_checkpointing", False)
        if is_selective_activation_checkpointing(checkpointing):
            raise ValueError(
                "Wan-Animate-2 does not support selective activation checkpointing; "
                "set activation_checkpointing=true for full checkpointing or false to disable it."
            )
        install_forward_origin(model)
        if checkpointing:
            for index, block in enumerate(model.blocks):
                model.blocks[index] = checkpoint_wrapper(block, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
        # The cache is local to the whole block; submodule checkpointing could
        # replay attention after that cache has changed. Do not double-wrap it.
        return super().parallelize(model, device_mesh, activation_checkpointing=False, **kwargs)
