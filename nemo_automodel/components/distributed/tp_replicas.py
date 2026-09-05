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

"""Synchronization helpers for parameters replicated across tensor parallel ranks."""

from collections.abc import Iterator
from itertools import chain
from typing import Literal

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate

_TP_REPLICA_GRAD_REDUCTION_ATTR = "_nemo_tp_replica_grad_reduction"
_MAX_FLAT_BUFFER_BYTES = 256 * 1024 * 1024


def _mark_tp_replica_gradient_reduction(
    module: torch.nn.Module,
    reduction: Literal["mean", "sum"],
) -> None:
    """Mark direct parameters of a module with their TP-replica reduction semantic.

    Args:
        module: Module whose direct parameters are replicated across TP ranks.
        reduction: ``"mean"`` for redundant full computation or ``"sum"`` for
            disjoint partial contributions.
    """
    setattr(module, _TP_REPLICA_GRAD_REDUCTION_ATTR, reduction)


def _get_tp_mesh(device_mesh: DeviceMesh | None, tp_axis_name: str) -> DeviceMesh | None:
    """Return a non-trivial TP submesh when one is present."""
    if not isinstance(device_mesh, DeviceMesh) or tp_axis_name not in (device_mesh.mesh_dim_names or ()):
        return None
    tp_mesh = device_mesh[tp_axis_name]
    return tp_mesh if tp_mesh.size() > 1 else None


def _is_tp_replicated(
    tensor: torch.Tensor,
    tp_group_ranks: tuple[int, ...],
    current_rank: int,
    tp_axis_name: str,
) -> bool:
    """Return whether rank-local tensor storage is replicated across TP peers.

    A DTensor may live on a DP-only submesh even though physical copies exist for
    every TP coordinate. Conversely, an MoE mesh may fold the TP ranks into axes
    named ``ep`` or ``ep_shard``. Comparing mesh coordinates and placements keeps
    both cases explicit instead of assuming every unnamed TP dimension is a copy.

    Args:
        tensor: Tensor or DTensor of arbitrary global and rank-local shape.
        tp_group_ranks: Ordered global ranks in the current TP group.
        current_rank: Current global rank.
        tp_axis_name: Name of the tensor-parallel mesh dimension.

    Returns:
        Whether every TP peer represented in the tensor mesh differs only along
        dimensions carrying ``Replicate`` placements.
    """
    if not isinstance(tensor, DTensor):
        return True
    mesh_names = tensor.device_mesh.mesh_dim_names or ()
    if tp_axis_name in mesh_names:
        tp_dim = list(mesh_names).index(tp_axis_name)
        return isinstance(tensor.placements[tp_dim], Replicate)

    mesh_ranks = tensor.device_mesh.mesh
    current_coordinate = (mesh_ranks == current_rank).nonzero(as_tuple=False)
    if current_coordinate.numel() == 0:
        raise RuntimeError(f"Current rank {current_rank} is absent from a local DTensor mesh")
    current_coordinate = current_coordinate[0].tolist()
    for peer_rank in tp_group_ranks:
        peer_coordinate = (mesh_ranks == peer_rank).nonzero(as_tuple=False)
        if peer_coordinate.numel() == 0:
            continue
        for mesh_dim, (current_index, peer_index) in enumerate(zip(current_coordinate, peer_coordinate[0].tolist())):
            if current_index != peer_index and not isinstance(tensor.placements[mesh_dim], Replicate):
                return False
    return True


def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Return rank-local storage for a Tensor or DTensor."""
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _iter_unique_parameters(
    model_parts: list[torch.nn.Module],
) -> Iterator[tuple[torch.nn.Parameter, Literal["mean", "sum"]]]:
    """Yield parameters once, together with their module-owned reduction semantic."""
    seen: dict[int, Literal["mean", "sum"]] = {}
    for model_part in model_parts:
        for module in model_part.modules():
            reduction = getattr(module, _TP_REPLICA_GRAD_REDUCTION_ATTR, "mean")
            if reduction not in ("mean", "sum"):
                raise ValueError(f"Unsupported TP replica gradient reduction: {reduction!r}")
            for parameter in module.parameters(recurse=False):
                parameter_id = id(parameter)
                prior_reduction = seen.get(parameter_id)
                if prior_reduction is not None:
                    if prior_reduction != reduction:
                        raise RuntimeError(
                            "A shared parameter has conflicting TP replica gradient reductions: "
                            f"{prior_reduction!r} and {reduction!r}"
                        )
                    continue
                seen[parameter_id] = reduction
                yield parameter, reduction


def _iter_unique_buffers(model_parts: list[torch.nn.Module]) -> Iterator[torch.Tensor]:
    """Yield model buffers once in deterministic module traversal order."""
    seen: set[int] = set()
    for model_part in model_parts:
        for buffer in model_part.buffers():
            buffer_id = id(buffer)
            if buffer_id in seen:
                continue
            seen.add(buffer_id)
            yield buffer


@torch.no_grad()
def _broadcast_tp_replicas(
    model_parts: list[torch.nn.Module],
    device_mesh: DeviceMesh | None,
    tp_axis_name: str = "tp",
) -> int:
    """Broadcast replicated parameters and buffers from the first TP rank.

    Intended TP shards are excluded by their DTensor placement. The collective
    operates on each tensor's rank-local storage, whose shape is identical within
    a TP group even when another mesh axis shards the global tensor.

    Args:
        model_parts: Local pipeline-stage modules containing tensors of arbitrary
            shape. Shared tensors are visited once.
        device_mesh: Root mesh containing the tensor-parallel axis.
        tp_axis_name: Name of the tensor-parallel mesh dimension.

    Returns:
        Number of rank-local tensors broadcast on this rank.
    """
    tp_mesh = _get_tp_mesh(device_mesh, tp_axis_name)
    if tp_mesh is None:
        return 0

    group = tp_mesh.get_group()
    source_rank = dist.get_global_rank(group, 0)
    tp_group_ranks = tuple(dist.get_process_group_ranks(group))
    current_rank = dist.get_rank()
    placement_cache: dict[tuple[int, tuple[object, ...]], bool] = {}

    def is_replicated(tensor: torch.Tensor) -> bool:
        if not isinstance(tensor, DTensor):
            return True
        cache_key = (id(tensor.device_mesh), tuple(tensor.placements))
        if cache_key not in placement_cache:
            placement_cache[cache_key] = _is_tp_replicated(
                tensor,
                tp_group_ranks,
                current_rank,
                tp_axis_name,
            )
        return placement_cache[cache_key]

    synchronized = 0
    parameters = (parameter for parameter, _ in _iter_unique_parameters(model_parts))
    for tensor in chain(parameters, _iter_unique_buffers(model_parts)):
        if not is_replicated(tensor):
            continue
        local = _local_tensor(tensor)
        if local.is_meta or local.numel() == 0:
            continue
        if local.device.type == tp_mesh.device_type:
            dist.broadcast(local, src=source_rank, group=group)
        else:
            communication_tensor = local.to(device=tp_mesh.device_type)
            dist.broadcast(communication_tensor, src=source_rank, group=group)
            local.copy_(communication_tensor.to(device=local.device))
        synchronized += 1
    return synchronized


def _gradient_chunks(gradients: list[torch.Tensor]) -> Iterator[list[torch.Tensor]]:
    """Split same-device, same-dtype gradients into bounded communication buffers."""
    chunk: list[torch.Tensor] = []
    chunk_bytes = 0
    for gradient in gradients:
        gradient_bytes = gradient.numel() * gradient.element_size()
        if chunk and chunk_bytes + gradient_bytes > _MAX_FLAT_BUFFER_BYTES:
            yield chunk
            chunk = []
            chunk_bytes = 0
        chunk.append(gradient)
        chunk_bytes += gradient_bytes
    if chunk:
        yield chunk


@torch.no_grad()
def _synchronize_tp_replica_gradients(
    model_parts: list[torch.nn.Module],
    device_mesh: DeviceMesh | None,
    tp_axis_name: str = "tp",
) -> int:
    """Reduce TP-replicated gradients once at the optimizer boundary.

    Full-computation replicas are mean-reduced so each logical parameter has one
    optimizer update. Modules explicitly marked as producing disjoint partial
    contributions are sum-reduced. Gradients are flattened into bounded buffers
    by reduction, device, and dtype before communication. DTensor gradients are
    reduced through rank-local storage while retaining their global shape and
    placements. Parameters sharded or partial on the TP axis are untouched.

    Args:
        model_parts: Local pipeline-stage modules whose gradients have arbitrary
            shapes and have completed accumulation for this optimizer update.
        device_mesh: Root mesh containing the tensor-parallel axis.
        tp_axis_name: Name of the tensor-parallel mesh dimension.

    Returns:
        Number of rank-local gradients synchronized on this rank.

    Raises:
        RuntimeError: If gradient presence differs across TP replicas, or if a
            replicated parameter has a sparse gradient that cannot be flattened
            without changing its representation.
    """
    tp_mesh = _get_tp_mesh(device_mesh, tp_axis_name)
    if tp_mesh is None:
        return 0

    group = tp_mesh.get_group()
    tp_group_ranks = tuple(dist.get_process_group_ranks(group))
    current_rank = dist.get_rank()
    placement_cache: dict[tuple[int, tuple[object, ...]], bool] = {}
    replicated_parameters = []
    for parameter, reduction in _iter_unique_parameters(model_parts):
        if not parameter.requires_grad:
            continue
        if isinstance(parameter, DTensor):
            cache_key = (id(parameter.device_mesh), tuple(parameter.placements))
            if cache_key not in placement_cache:
                placement_cache[cache_key] = _is_tp_replicated(
                    parameter,
                    tp_group_ranks,
                    current_rank,
                    tp_axis_name,
                )
            if not placement_cache[cache_key]:
                continue
        replicated_parameters.append((parameter, reduction))
    if not replicated_parameters:
        return 0

    tp_size = tp_mesh.size()
    gradient_presence = torch.tensor(
        [parameter.grad is not None for parameter, _ in replicated_parameters],
        dtype=torch.int32,
        device=tp_mesh.device_type,
    )
    dist.all_reduce(gradient_presence, op=dist.ReduceOp.SUM, group=group)
    presence_counts = gradient_presence.cpu().tolist()
    partially_present = sum(0 < count < tp_size for count in presence_counts)
    if partially_present:
        raise RuntimeError(
            f"Gradient presence differs across TP replicas for {partially_present} parameter(s); "
            "all TP ranks must execute the same trainable graph"
        )

    grouped_gradients: dict[tuple[str, torch.device, torch.dtype], list[torch.Tensor]] = {}
    for (parameter, reduction), presence_count in zip(replicated_parameters, presence_counts):
        if presence_count == 0:
            continue
        assert parameter.grad is not None
        gradient = _local_tensor(parameter.grad)
        if gradient.is_sparse:
            raise RuntimeError("Sparse gradients are not supported for tensor-parallel replicated parameters")
        grouped_gradients.setdefault((reduction, gradient.device, gradient.dtype), []).append(gradient)

    synchronized = 0
    for (reduction, _, _), gradients in grouped_gradients.items():
        for chunk in _gradient_chunks(gradients):
            flat_gradient = torch.cat([gradient.detach().reshape(-1) for gradient in chunk])
            communication_tensor = (
                flat_gradient
                if flat_gradient.device.type == tp_mesh.device_type
                else flat_gradient.to(device=tp_mesh.device_type)
            )
            dist.all_reduce(communication_tensor, op=dist.ReduceOp.SUM, group=group)
            if reduction == "mean":
                communication_tensor.div_(tp_size)
            if communication_tensor is not flat_gradient:
                flat_gradient.copy_(communication_tensor.to(device=flat_gradient.device))

            offset = 0
            for gradient in chunk:
                next_offset = offset + gradient.numel()
                gradient.copy_(flat_gradient[offset:next_offset].view_as(gradient))
                offset = next_offset
            synchronized += len(chunk)
    return synchronized
