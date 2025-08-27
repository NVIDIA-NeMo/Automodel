# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

from typing import Optional

import torch
from torch.distributed._tensor import DTensor
from torch.distributed._tensor.placement_types import Replicate, Shard
from torch.distributed.device_mesh import DeviceMesh


def is_dtensor(tensor: torch.Tensor) -> bool:
    """Check if a tensor is a DTensor."""
    return isinstance(tensor, DTensor)


def get_expert_slice_for_rank(experts_tensor: torch.Tensor, n_experts: int) -> tuple[torch.Tensor, int, int]:
    """
    Get the slice of experts present on the current rank for a DTensor.

    For non-DTensors, returns the full tensor with start_expert=0, end_expert=n_experts.
    For DTensors sharded along the expert dimension (dim=0), returns only the local experts.

    Args:
        experts_tensor: Input tensor containing expert weights [n_experts, ...]
        n_experts: Total number of experts across all ranks

    Returns:
        tuple of (local_tensor, start_expert_id, end_expert_id)
        - local_tensor: The local portion of the tensor
        - start_expert_id: Global ID of the first expert on this rank
        - end_expert_id: Global ID after the last expert on this rank (exclusive)
    """
    if not is_dtensor(experts_tensor):
        return experts_tensor, 0, n_experts

    dtensor = experts_tensor
    local_tensor = dtensor.to_local()

    device_mesh = dtensor.device_mesh
    ep_mesh = device_mesh["ep"] if "ep" in device_mesh.mesh_dim_names else device_mesh
    current_rank = ep_mesh.get_local_rank()

    placement = dtensor.placements[0]  # Assume single device mesh for now
    if isinstance(placement, Shard) and placement.dim == 0:
        # Tensor is sharded along expert dimension
        world_size = ep_mesh.size()

        # Calculate expert range for this rank
        experts_per_rank = n_experts // world_size
        remainder = n_experts % world_size

        if current_rank < remainder:
            # First `remainder` ranks get one extra expert
            experts_on_rank = experts_per_rank + 1
            start_expert = current_rank * experts_on_rank
        else:
            # Remaining ranks get standard number of experts
            experts_on_rank = experts_per_rank
            start_expert = remainder * (experts_per_rank + 1) + (current_rank - remainder) * experts_per_rank

        end_expert = start_expert + experts_on_rank
        return local_tensor, start_expert, end_expert
    elif isinstance(placement, Replicate):
        # Tensor is replicated - all ranks have full data
        return local_tensor, 0, n_experts
    else:
        # Other sharding patterns - assume full range for now
        # Could be extended to handle sharding along other dimensions
        return local_tensor, 0, n_experts


def split_experts_weights_dtensor_aware(weight: torch.Tensor, n_experts: int) -> tuple[list[torch.Tensor], list[int]]:
    """
    Split expert weights, handling both regular tensors and DTensors.

    For DTensors, only splits the experts present on the current rank.

    Args:
        weight: Expert weights tensor [n_experts, ...] (regular tensor or DTensor)
        n_experts: Total number of experts across all ranks

    Returns:
        tuple of (split_weights, expert_ids)
        - split_weights: List of individual expert weight tensors
        - expert_ids: List of global expert IDs corresponding to split_weights
    """
    local_tensor, start_expert, end_expert = get_expert_slice_for_rank(weight, n_experts)
    local_n_experts = end_expert - start_expert
    if local_tensor.shape[0] != local_n_experts:
        raise ValueError(
            f"Expected local tensor first dimension to be {local_n_experts} "
            f"(experts {start_expert}:{end_expert}), got {local_tensor.shape[0]}"
        )

    split_weights = []
    expert_ids = []
    for i in range(local_n_experts):
        expert_weight = local_tensor[i]  # Shape: [...] (expert dimension removed)
        global_expert_id = start_expert + i

        split_weights.append(expert_weight)
        expert_ids.append(global_expert_id)

    return split_weights, expert_ids


def validate_dtensor_expert_sharding(tensor: torch.Tensor, expected_experts: int, tensor_name: str = "tensor") -> bool:
    """
    Validate that a DTensor is properly sharded for expert parallelism.

    Args:
        tensor: Tensor to validate
        expected_experts: Expected total number of experts
        tensor_name: Name for error messages

    Returns:
        True if valid, raises ValueError if invalid
    """
    if not is_dtensor(tensor):
        if tensor.shape[0] != expected_experts:
            raise ValueError(f"{tensor_name} has shape {tensor.shape[0]} experts, expected {expected_experts}")
        return True

    dtensor = tensor

    if dtensor.shape[0] != expected_experts:
        raise ValueError(f"{tensor_name} global shape has {dtensor.shape[0]} experts, expected {expected_experts}")

    placement = dtensor.placements[0] if dtensor.placements else None

    if isinstance(placement, Shard) and placement.dim == 0:
        return True
    elif isinstance(placement, Replicate):
        return True
    else:
        raise ValueError(
            f"{tensor_name} has unsupported DTensor placement: {placement}. "
            f"Expected Shard(dim=0) or Replicate for expert parallelism."
        )


def create_dtensor_from_local(
    local_tensor: torch.Tensor, device_mesh: Optional["DeviceMesh"], rank: Optional[int] = None
) -> torch.Tensor:
    """
    Create a DTensor from a local tensor for expert parallelism.

    Args:
        local_tensor: Local portion of the tensor on this rank
        device_mesh: Device mesh for DTensor creation
        rank: Current rank (for device placement)

    Returns:
        DTensor if device_mesh is provided and DTensor is available, otherwise local_tensor
    """
    if device_mesh is None:
        return local_tensor

    if rank is not None and torch.cuda.is_available():
        local_tensor = local_tensor.to(f"cuda:{torch.cuda.current_device()}")

    dtensor = DTensor.from_local(local_tensor, device_mesh, [Shard(0)])
    return dtensor


def get_expert_range_for_rank_from_mesh(device_mesh: Optional["DeviceMesh"], n_experts: int) -> tuple[int, int]:
    """
    Get the range of experts that should be loaded for the current rank.

    Args:
        device_mesh: Device mesh for expert parallelism
        n_experts: Total number of experts

    Returns:
        Tuple of (start_expert_id, end_expert_id) for this rank
    """
    if device_mesh is None:
        return 0, n_experts

    ep_mesh = device_mesh["ep"] if "ep" in device_mesh.mesh_dim_names else device_mesh
    world_size = ep_mesh.size()
    rank = ep_mesh.get_local_rank()

    experts_per_rank = n_experts // world_size
    remainder = n_experts % world_size

    if rank < remainder:
        experts_per_rank += 1
        start_expert = rank * experts_per_rank
    else:
        start_expert = rank * experts_per_rank + remainder

    end_expert = start_expert + experts_per_rank
    return start_expert, end_expert


def should_load_expert_for_rank(expert_id: int, device_mesh: Optional["DeviceMesh"], n_experts: int) -> bool:
    """
    Check if a specific expert should be loaded on the current rank.

    Args:
        expert_id: The expert ID to check
        device_mesh: Device mesh for expert parallelism
        n_experts: Total number of experts

    Returns:
        True if this expert should be loaded on the current rank
    """
    start_expert, end_expert = get_expert_range_for_rank_from_mesh(device_mesh, n_experts)
    return start_expert <= expert_id < end_expert
