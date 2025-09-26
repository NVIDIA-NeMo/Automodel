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

import torch.nn as nn
from torch.distributed.tensor import (
    Shard,
    distribute_tensor,
)
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
)

def _distribute_param(_module, name, device_mesh, src_data_rank, placements):
    param = getattr(_module, name)
    dist_param = nn.Parameter(
        distribute_tensor(param, device_mesh, placements, src_data_rank=src_data_rank),
        requires_grad=param.requires_grad,
    )
    assert dist_param.requires_grad == param.requires_grad
    _module.register_parameter(name, dist_param)



class ColwiseParallelLora(ColwiseParallel):
    def _partition_linear_fn(self, name, module, device_mesh):
        # colwise shard weight/bias to Shard(0), weight be Shard(0)
        # means Colwise as Linear is input * weight^T + bias, where
        # weight would become Shard(1)
        def _get_module_and_name(module, name):
            if name.endswith("lora_A.weight"):
                return module.lora_A, "weight"
            elif name.endswith("lora_B.weight") and hasattr(module, "lora_B"):
                return module.lora_B, "weight"
            else:
                return module, name

        for name, param in module.named_parameters():
            _module, _name = _get_module_and_name(module, name)
            _distribute_param(_module, _name, device_mesh, self.src_data_rank, [Shard(0)])


class RowwiseParallelLora(RowwiseParallel):
    def _partition_linear_fn(self, name, module, device_mesh):
        # Rowwise shard weight to Shard(1), bias to Replicate(), weight be Shard(1)
        # means Rowwise as nn.Linear is input * weight^T + bias, where
        # weight would become Shard(0)
        _distribute_param(module, "weight", device_mesh, self.src_data_rank, [Shard(1)])
        if getattr(module, "bias", None) is not None:
            _distribute_param(module, "bias", device_mesh, self.src_data_rank, [Replicate()])
        if hasattr(module, "lora_A"):
            _distribute_param(module.lora_A, "weight", device_mesh, self.src_data_rank, [Shard(1)])
            _distribute_param(module.lora_B, "weight", device_mesh, self.src_data_rank, [Shard(1)])


def translate_to_lora(plan):
    if isinstance(plan, ColwiseParallel):
        plan.__class__ = ColwiseParallelLora
    elif isinstance(plan, RowwiseParallel):
        plan.__class__ = RowwiseParallelLora
    return plan
