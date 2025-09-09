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


class ColwiseParallelLora(ColwiseParallel):
    def _partition_linear_fn(self, name, module, device_mesh):
        # colwise shard weight/bias to Shard(0), weight be Shard(0)
        # means Colwise as Linear is input * weight^T + bias, where
        # weight would become Shard(1)
        def _distribute_param(_module, name):
            param = getattr(_module, name)
            dist_param = nn.Parameter(
                distribute_tensor(param, device_mesh, [Shard(0)], src_data_rank=self.src_data_rank)
            )
            _module.register_parameter(name, dist_param)

        for name, param in module.named_parameters():
            if name.endswith("lora_A.weight"):
                _distribute_param(module.lora_A, "weight")
            elif name.endswith("lora_B.weight"):
                _distribute_param(module.lora_B, "weight")
            else:
                _distribute_param(module, name)


class RowwiseParallelLora(RowwiseParallel):
    def _partition_linear_fn(self, name, module, device_mesh):
        # Rowwise shard weight to Shard(1), bias to Replicate(), weight be Shard(1)
        # means Rowwise as nn.Linear is input * weight^T + bias, where
        # weight would become Shard(0)
        super()._partition_linear_fn(name, module, device_mesh)
        if hasattr(module, "lora_A"):
            super()._partition_linear_fn(name, module.lora_A, device_mesh)
            super()._partition_linear_fn(name, module.lora_B, device_mesh)


def translate_to_lora(plan):
    if isinstance(plan, ColwiseParallel):
        plan.__class__ = ColwiseParallelLora
    elif isinstance(plan, RowwiseParallel):
        plan.__class__ = RowwiseParallelLora
    return plan
