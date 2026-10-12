# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
import torch
import torch.nn as nn
from torch.distributed.tensor import (
    DeviceMesh,
    DTensor,
    Replicate,
    Shard,
    distribute_tensor,
)
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    ParallelStyle,
    RowwiseParallel,
    SequenceParallel,
)

from nemo_automodel.components.distributed.tp_replicas import mark_tp_replica_gradient_reduction
from nemo_automodel.shared.tp_linear import TPLinear as TPLinear

__all__ = ["ReplicatedWithGradAllReduce", "translate_to_lora"]


class ReplicatedWithGradAllReduce(ParallelStyle):
    """Keep parameters local while sum-reducing their partial TP gradients."""

    def _apply(self, module: torch.nn.Module, device_mesh: DeviceMesh) -> torch.nn.Module:
        """Mark a replicated module whose ranks compute disjoint gradient contributions."""
        mark_tp_replica_gradient_reduction(module, "sum")
        return module


def _distribute_param(_module, name, device_mesh, src_data_rank, placements):
    param = getattr(_module, name)
    dist_param = nn.Parameter(
        distribute_tensor(param, device_mesh, placements, src_data_rank=src_data_rank),
        requires_grad=param.requires_grad,
    )
    assert dist_param.requires_grad == param.requires_grad
    _module.register_parameter(name, dist_param)


class ColwiseParallelLora(ColwiseParallel):
    """Column-wise tensor parallel style for LoRA-aware modules."""

    def _partition_linear_fn(self, name, module, device_mesh):
        # colwise shard weight/bias to Shard(0), weight be Shard(0)
        # means Colwise as Linear is input * weight^T + bias, where
        # weight would become Shard(1)
        def _get_module_and_name(module, name):
            if name.endswith("lora_A.weight"):
                assert hasattr(module, "lora_A"), f"lora_A not found in {module}"
                return module.lora_A, "weight"
            elif name.endswith("lora_B.weight"):
                assert hasattr(module, "lora_B"), f"lora_B not found in {module}"
                return module.lora_B, "weight"
            else:
                return module, name

        for name, param in module.named_parameters():
            _module, _name = _get_module_and_name(module, name)
            _distribute_param(_module, _name, device_mesh, self.src_data_rank, [Shard(0)])

        # Register forward hook on lora_A to all-gather its low rank output
        def lora_a_output_hook(module, input, output):
            if isinstance(output, DTensor):
                if any(isinstance(p, Shard) for p in output.placements):
                    output = output.redistribute(device_mesh=output.device_mesh, placements=[Replicate()])
            return output

        if hasattr(module, "lora_A"):
            module.lora_A.register_forward_hook(lora_a_output_hook)
            # lora_A/lora_B are nn.Linear sub-modules whose weights are now DTensors.
            # Convert to TPLinear so torch.compile sees matmul-based forward and
            # avoids the aten.view recursion in the backward.
            module.lora_A.__class__ = TPLinear
        if hasattr(module, "lora_B"):
            module.lora_B.__class__ = TPLinear

        # Plain nn.Linear (not a LinearLoRA subclass): same conversion needed.
        # LinearLoRA already uses _dtensor_linear (matmul) in its own forward.
        if type(module) is nn.Linear:
            module.__class__ = TPLinear

    def _partition_embedding_fn(self, name, module, device_mesh):
        # colwise shard embedding.weight is straight forward as Shard(1)
        for name, param in module.named_parameters():
            _distribute_param(module, name, device_mesh, self.src_data_rank, [Shard(1)])


class RowwiseParallelLora(RowwiseParallel):
    """Row-wise tensor parallel style for LoRA-aware modules."""

    def _partition_linear_fn(self, name, module, device_mesh):
        # Rowwise shard weight to Shard(1), bias to Replicate(), weight be Shard(1)
        # means Rowwise as nn.Linear is input * weight^T + bias, where
        # weight would become Shard(0)
        #
        # distribute_module iterates named_modules() and calls this fn for every
        # submodule (lora_A, lora_B, ...) in addition to the root module.  The root
        # call already sets each submodule's weight to the correct placement, so
        # subsequent submodule calls would clash (e.g. trying to re-distribute
        # lora_B.weight from Replicate() -> Shard(1)).  Return early if the weight
        # is already a DTensor -- the parent call handled it.
        if isinstance(getattr(module, "weight", None), DTensor):
            return
        _distribute_param(module, "weight", device_mesh, self.src_data_rank, [Shard(1)])
        if getattr(module, "bias", None) is not None:
            _distribute_param(module, "bias", device_mesh, self.src_data_rank, [Replicate()])
        if hasattr(module, "lora_A"):
            _distribute_param(module.lora_A, "weight", device_mesh, self.src_data_rank, [Shard(1)])
            _distribute_param(module.lora_B, "weight", device_mesh, self.src_data_rank, [Shard(1)])
            module.lora_A.__class__ = TPLinear
            module.lora_B.__class__ = TPLinear
        # Plain nn.Linear: convert to TPLinear for compile safety.
        # LinearLoRA subclasses are already handled by their own _dtensor_linear forward.
        if type(module) is nn.Linear:
            module.__class__ = TPLinear
        if hasattr(module, "lora_magnitude"):
            _distribute_param(module, "lora_magnitude", device_mesh, self.src_data_rank, [Replicate()])

    def _partition_embedding_fn(self, name, module, device_mesh):
        # rowwise shard embedding.weight is Shard(0)
        for name, param in module.named_parameters():
            _distribute_param(module, name, device_mesh, self.src_data_rank, [Shard(0)])


class SequenceParallelLora(SequenceParallel):
    """Sequence parallel style that replicates LoRA module parameters."""

    def _replicate_module_fn(self, name: str, module: nn.Module, device_mesh: DeviceMesh):
        for p_name, param in module.named_parameters():
            # simple replication with fixed ones_ init from LayerNorm/RMSNorm, which allow
            # us to simply just use from_local
            replicated_param = torch.nn.Parameter(
                DTensor.from_local(param, device_mesh, [Replicate()], run_check=False),
                requires_grad=param.requires_grad,
            )
            module.register_parameter(p_name, replicated_param)


def translate_to_lora(plan):
    """Mutate a tensor-parallel plan to the matching LoRA-aware style."""
    CLS_MAP = {
        ColwiseParallel: ColwiseParallelLora,
        RowwiseParallel: RowwiseParallelLora,
        SequenceParallel: SequenceParallelLora,
    }
    plan.__class__ = CLS_MAP.get(type(plan), plan.__class__)
    return plan
