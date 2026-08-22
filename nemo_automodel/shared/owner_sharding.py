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

"""Contracts for parameters partitioned by model-owned collectives."""

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


@dataclass(frozen=True)
class ModelOwnedDTensorSpec:
    """Describe a logical DTensor whose local shard is consumed by model code.

    The parameter is a real, globally shaped DTensor, so model and optimizer
    checkpoints use PyTorch's native DTensor resharding.  Some models route
    activations directly to the rank that owns a row, however, instead of
    letting FSDP materialize the full parameter for forward.  Such a parameter
    must be excluded from FSDP ownership and may need an explicit gradient
    divisor because its model-owned backward collective has different averaging
    semantics from FSDP.

    Args:
        process_group: Group whose rank order matches the DTensor shard mesh.
        gradient_divisor: Explicit factor applied to the DTensor gradient before
            clipping.
        legacy_optimizer_state_namespace: Optional versioned namespace used by
            checkpoints written before the parameter became a DTensor.  It is
            read-only compatibility metadata; new checkpoints always use the
            ordinary logical DTensor FQN.
    """

    process_group: torch.distributed.ProcessGroup
    gradient_divisor: float
    legacy_optimizer_state_namespace: str | None = None

    def __post_init__(self) -> None:
        """Reject ambiguous scaling and legacy checkpoint identities eagerly."""
        divisor = self.gradient_divisor
        if isinstance(divisor, bool) or not isinstance(divisor, Real):
            raise TypeError("gradient_divisor must be a real number")
        if not math.isfinite(float(divisor)) or float(divisor) <= 0:
            raise ValueError("gradient_divisor must be finite and greater than zero")
        namespace = self.legacy_optimizer_state_namespace
        if namespace is not None:
            if not isinstance(namespace, str):
                raise TypeError("legacy_optimizer_state_namespace must be a string or None")
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", namespace) is None:
                raise ValueError(
                    "legacy_optimizer_state_namespace must be a non-empty identifier containing only [A-Za-z0-9_-]"
                )


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


def get_model_owned_dtensor_spec(parameter: torch.Tensor) -> ModelOwnedDTensorSpec | None:
    """Return and type-check the model-owned contract attached to a DTensor.

    Args:
        parameter: Parameter that may be a globally shaped DTensor whose local
            shard is consumed by model-owned communication.

    Returns:
        The attached specification, or ``None`` when FSDP owns the DTensor.

    Raises:
        TypeError: If the marker has the wrong type or is attached to a plain
            tensor.
        RuntimeError: If both owner-sharding contracts are attached.
    """
    spec = getattr(parameter, "_nemo_model_owned_dtensor_spec", None)
    if spec is not None and not isinstance(spec, ModelOwnedDTensorSpec):
        raise TypeError("_nemo_model_owned_dtensor_spec must be a ModelOwnedDTensorSpec")
    if spec is not None and not isinstance(parameter, DTensor):
        raise TypeError("ModelOwnedDTensorSpec may only be attached to a DTensor")
    if spec is not None and getattr(parameter, "_nemo_owner_sharded_spec", None) is not None:
        raise RuntimeError("A parameter cannot use both owner-sharding contracts")
    return spec
