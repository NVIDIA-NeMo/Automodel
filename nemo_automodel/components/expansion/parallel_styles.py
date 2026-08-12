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

"""Tensor-parallel styles that also distribute an expansion weight.

A stock ``ColwiseParallel`` / ``RowwiseParallel`` distributes the parameters it knows
about, which does not include the expansion weight an :class:`ExpandedLinear` carries.
Left undistributed it would stay replicated on every rank while its base weight is
sharded, and the two would not compose.

This mirrors ``nemo_automodel.components.distributed.parallel_styles``' LoRA-aware styles
and their ``translate_to_lora`` entry point, with one simplification: the expansion weight
has the *same shape* as the base weight, so it takes the same placement. LoRA's low-rank
factors have different shapes and need their own handling.
"""

from __future__ import annotations

from typing import Union

import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Placement, Replicate, Shard, distribute_tensor
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

__all__ = ["ColwiseParallelExpanded", "RowwiseParallelExpanded", "translate_to_expanded"]


def _distribute(module: nn.Module, name: str, mesh: DeviceMesh, placement: Placement) -> None:
    """Replace one parameter with its distributed counterpart, if it has one."""
    param = getattr(module, name, None)
    if param is None or isinstance(param, DTensor):
        return
    setattr(
        module,
        name,
        nn.Parameter(distribute_tensor(param.data, mesh, [placement]), requires_grad=param.requires_grad),
    )


def _partition(module: nn.Module, mesh: DeviceMesh, weight_placement: Placement, bias_placement: Placement) -> None:
    """Distribute a linear's weight, bias and expansion weight."""
    # ``distribute_module`` invokes the partition function for every submodule, so the
    # expansion gets its own call after the parent already handled it. Returning early on
    # an already-distributed weight avoids re-distributing it, which would clash.
    if isinstance(getattr(module, "weight", None), DTensor):
        return
    _distribute(module, "weight", mesh, weight_placement)
    _distribute(module, "bias", mesh, bias_placement)
    expansion = getattr(module, "expansion", None)
    if expansion is not None:
        _distribute(expansion, "weight", mesh, weight_placement)


class ColwiseParallelExpanded(ColwiseParallel):
    """``ColwiseParallel`` that also distributes an :class:`ExpandedLinear`'s expansion weight."""

    def _partition_linear_fn(self, name: str, module: nn.Module, device_mesh: DeviceMesh) -> None:
        """Shard weight, bias and expansion weight on the output dimension.

        Args:
            name: Module name, unused; kept for the base-class signature.
            module: The linear being partitioned.
            device_mesh: Mesh to distribute over.
        """
        _partition(module, device_mesh, Shard(0), Shard(0))


class RowwiseParallelExpanded(RowwiseParallel):
    """``RowwiseParallel`` that also distributes an :class:`ExpandedLinear`'s expansion weight."""

    def _partition_linear_fn(self, name: str, module: nn.Module, device_mesh: DeviceMesh) -> None:
        """Shard weight and expansion weight on the input dimension; replicate the bias.

        Args:
            name: Module name, unused; kept for the base-class signature.
            module: The linear being partitioned.
            device_mesh: Mesh to distribute over.
        """
        _partition(module, device_mesh, Shard(1), Replicate())


def translate_to_expanded(plan: Union[ColwiseParallel, RowwiseParallel]) -> Union[ColwiseParallel, RowwiseParallel]:
    """Mutate a tensor-parallel style in place to its expansion-aware equivalent.

    Args:
        plan: A parallel style from a model's TP plan. Styles with no expansion-aware
            equivalent are returned unchanged.

    Returns:
        The same object, retyped where an equivalent exists.
    """
    CLS_MAP = {
        ColwiseParallel: ColwiseParallelExpanded,
        RowwiseParallel: RowwiseParallelExpanded,
    }
    plan.__class__ = CLS_MAP.get(type(plan), plan.__class__)
    return plan
