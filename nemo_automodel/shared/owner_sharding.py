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

"""Contract for parameters physically partitioned by model-owned collectives."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from numbers import Real

import torch
from torch.distributed.tensor import DTensor


@dataclass(frozen=True)
class OwnerShardedParameterSpec:
    """Describe a rank-local parameter shard managed outside DTensor/FSDP.

    Args:
        process_group: Group whose ranks own disjoint pieces of the logical
            parameter. The same group supplies the global gradient-norm
            reduction and the rank identity used by distributed checkpoints.
            ``None`` is valid only for a non-distributed, single-rank owner.
        gradient_divisor: Explicit factor applied to this parameter's gradient
            before clipping. Model-owned autograd collectives can accumulate a
            different replication factor than ordinary FSDP parameters, so the
            model must state that factor rather than relying on a model-name
            branch in the training loop.
        optimizer_state_namespace: Stable, versioned DCP key namespace. Save
            and resume must use the same value and owner topology.
    """

    process_group: torch.distributed.ProcessGroup | None
    gradient_divisor: float
    optimizer_state_namespace: str

    def __post_init__(self) -> None:
        """Reject ambiguous scaling and checkpoint identities eagerly."""
        divisor = self.gradient_divisor
        if isinstance(divisor, bool) or not isinstance(divisor, Real):
            raise TypeError("gradient_divisor must be a real number")
        if not math.isfinite(float(divisor)) or float(divisor) <= 0:
            raise ValueError("gradient_divisor must be finite and greater than zero")
        namespace = self.optimizer_state_namespace
        if not isinstance(namespace, str):
            raise TypeError("optimizer_state_namespace must be a string")
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", namespace) is None:
            raise ValueError("optimizer_state_namespace must be a non-empty identifier containing only [A-Za-z0-9_-]")


def get_owner_sharded_parameter_spec(parameter: torch.Tensor) -> OwnerShardedParameterSpec | None:
    """Return and type-check the owner-sharded contract attached to ``parameter``.

    Args:
        parameter: Tensor of shape ``[...]``, with arbitrary rank and axis
            order, that may carry ``_nemo_owner_sharded_spec`` metadata.

    Returns:
        The attached owner-sharding specification, or ``None`` when the tensor
        is managed by the ordinary distributed-parameter paths.

    Raises:
        TypeError: If the marker exists but is not an
            :class:`OwnerShardedParameterSpec`.
        RuntimeError: If a DTensor also carries the owner-sharded marker,
            because the parameter would have two incompatible sharding owners.
    """
    spec = getattr(parameter, "_nemo_owner_sharded_spec", None)
    if spec is not None and not isinstance(spec, OwnerShardedParameterSpec):
        raise TypeError("_nemo_owner_sharded_spec must be an OwnerShardedParameterSpec")
    if spec is not None and isinstance(parameter, DTensor):
        raise RuntimeError("A DTensor cannot also use a model-owned sharding specification")
    return spec
