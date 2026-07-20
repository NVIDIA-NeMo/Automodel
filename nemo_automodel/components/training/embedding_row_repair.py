# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Repair near-zero input-embedding rows before optimizer construction."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

from nemo_automodel.shared.tied_weights import (
    get_input_embeddings_weight_and_name,
    get_lm_head_weight_and_name,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbeddingRowRepairReport:
    """Summary of a completed input-embedding repair."""

    input_embedding_name: str
    output_embedding_name: str | None
    repaired_row_ids: tuple[int, ...]
    min_norm_before: float
    target_norm: float | None


@dataclass
class EmbeddingRowRepairConfig:
    """Configuration for detecting and repairing damaged input embeddings.

    A row is repaired when its L2 norm is non-finite or no larger than
    ``min_norm``. Its replacement uses the corresponding output-embedding
    direction, scaled to the RMS norm of the checkpoint's healthy input rows.
    This preserves a token-specific direction while restoring the magnitude
    expected by the pretrained model.

    The operation runs once after base-checkpoint loading and model sharding,
    before optimizer construction. It supports regular tensors and DTensors
    whose two matrix dimensions are sharded or replicated.
    """

    enabled: bool = True
    min_norm: float = 1.0e-6
    max_rows: int = 256

    def __post_init__(self) -> None:
        if not math.isfinite(self.min_norm) or self.min_norm < 0:
            raise ValueError(f"embedding_row_repair.min_norm must be finite and non-negative, got {self.min_norm}")
        if self.max_rows <= 0:
            raise ValueError(f"embedding_row_repair.max_rows must be positive, got {self.max_rows}")

    def apply(self, model: nn.Module) -> EmbeddingRowRepairReport | None:
        """Repair damaged input-embedding rows in ``model`` when enabled."""
        if not self.enabled:
            return None
        return repair_input_embedding_rows(model, min_norm=self.min_norm, max_rows=self.max_rows)


@dataclass(frozen=True)
class _WeightView:
    local: torch.Tensor
    global_shape: tuple[int, ...]
    local_shape: tuple[int, ...]
    global_offset: tuple[int, ...]
    mesh: object | None
    placements: tuple[object, ...]


def _weight_view(weight: torch.Tensor, name: str) -> _WeightView:
    if weight.ndim != 2:
        raise ValueError(f"{name} must be a 2-D embedding matrix, got shape={tuple(weight.shape)}")
    if not weight.is_floating_point():
        raise ValueError(f"{name} must have a floating-point dtype, got {weight.dtype}")

    if not isinstance(weight, DTensor):
        shape = tuple(weight.shape)
        return _WeightView(
            local=weight,
            global_shape=shape,
            local_shape=shape,
            global_offset=(0, 0),
            mesh=None,
            placements=(),
        )

    placements = tuple(weight.placements)
    for placement in placements:
        if isinstance(placement, Partial):
            raise ValueError(f"{name} has unsupported Partial placement: {placements}")
        if isinstance(placement, Shard) and placement.dim not in (0, 1, -1, -2):
            raise ValueError(f"{name} has unsupported placement for a matrix: {placement}")
        if not isinstance(placement, (Replicate, Shard)):
            raise ValueError(f"{name} has unsupported DTensor placement: {placement}")

    global_shape = tuple(weight.shape)
    local_shape, global_offset = compute_local_shape_and_global_offset(global_shape, weight.device_mesh, placements)
    local = weight.to_local()
    if tuple(local.shape) != tuple(local_shape):
        raise RuntimeError(
            f"{name} local shape mismatch: DTensor metadata says {local_shape}, tensor has {tuple(local.shape)}"
        )
    return _WeightView(
        local=local,
        global_shape=global_shape,
        local_shape=tuple(local_shape),
        global_offset=tuple(global_offset),
        mesh=weight.device_mesh,
        placements=placements,
    )


def _full_row_squared_norms(view: _WeightView) -> torch.Tensor:
    local = view.local.detach().to(torch.float32)
    row_squared_norms = local.square().sum(dim=1, dtype=torch.float64)
    if view.mesh is None:
        return row_squared_norms

    for mesh_dim, placement in enumerate(view.placements):
        if not isinstance(placement, Shard):
            continue
        shard_dim = placement.dim % len(view.global_shape)
        if shard_dim == 1:
            dist.all_reduce(row_squared_norms, op=dist.ReduceOp.SUM, group=view.mesh.get_group(mesh_dim=mesh_dim))
    return row_squared_norms


def _global_ids(view: _WeightView, local_mask: torch.Tensor) -> list[int]:
    local_ids = torch.nonzero(local_mask, as_tuple=False).flatten().cpu().tolist()
    return [view.global_offset[0] + int(local_id) for local_id in local_ids]


def _gather_unique_ids(local_ids: list[int]) -> tuple[int, ...]:
    if not dist.is_initialized():
        return tuple(sorted(set(local_ids)))

    gathered: list[list[int] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local_ids)
    return tuple(sorted({row_id for rank_ids in gathered if rank_ids is not None for row_id in rank_ids}))


def _global_min(value: torch.Tensor) -> float:
    value = value.detach().to(dtype=torch.float64).clone()
    if dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.MIN)
    return float(value.item())


def _healthy_rms_norm(row_squared_norms: torch.Tensor, healthy_mask: torch.Tensor) -> float:
    stats = torch.stack(
        (
            row_squared_norms[healthy_mask].sum(dtype=torch.float64),
            healthy_mask.sum().to(dtype=torch.float64),
        )
    )
    if dist.is_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    if stats[1] == 0:
        raise ValueError("Cannot repair input embeddings because the checkpoint has no healthy input rows")
    return math.sqrt(float((stats[0] / stats[1]).item()))


def _check_matching_layout(input_view: _WeightView, output_view: _WeightView) -> None:
    if input_view.global_shape != output_view.global_shape:
        raise ValueError(
            "Input and output embedding shapes must match for row repair, got "
            f"{input_view.global_shape} and {output_view.global_shape}"
        )
    if input_view.local_shape != output_view.local_shape or input_view.global_offset != output_view.global_offset:
        raise ValueError(
            "Input and output embeddings must have matching local shards for row repair, got "
            f"input shape/offset={input_view.local_shape}/{input_view.global_offset} and "
            f"output shape/offset={output_view.local_shape}/{output_view.global_offset}"
        )


@torch.no_grad()
def repair_input_embedding_rows(
    model: nn.Module,
    *,
    min_norm: float = 1.0e-6,
    max_rows: int = 256,
) -> EmbeddingRowRepairReport:
    """Detect and repair near-zero input-embedding rows.

    Args:
        model: Loaded and optionally DTensor-sharded causal language model.
        min_norm: Inclusive L2-norm threshold identifying a damaged row.
        max_rows: Safety bound; abort instead of mutating a broadly damaged checkpoint.

    Returns:
        A report containing every repaired global vocabulary row ID.
    """
    model = getattr(model, "module", model)
    input_weight, input_name = get_input_embeddings_weight_and_name(model)
    if input_weight is None or input_name is None:
        raise ValueError("Could not locate the model's input embedding weight")

    input_view = _weight_view(input_weight, input_name)
    input_squared_norms = _full_row_squared_norms(input_view)
    input_norms = input_squared_norms.sqrt()
    damaged_local = ~torch.isfinite(input_norms) | (input_norms <= min_norm)
    damaged_ids = _gather_unique_ids(_global_ids(input_view, damaged_local))
    min_norm_before = _global_min(input_norms.min())

    if not damaged_ids:
        logger.info(
            "Input embedding %s has no rows at or below norm %.3e (minimum %.3e).",
            input_name,
            min_norm,
            min_norm_before,
        )
        return EmbeddingRowRepairReport(
            input_embedding_name=input_name,
            output_embedding_name=None,
            repaired_row_ids=(),
            min_norm_before=min_norm_before,
            target_norm=None,
        )
    if len(damaged_ids) > max_rows:
        raise ValueError(
            f"Refusing to repair {len(damaged_ids)} input embedding rows; configured max_rows={max_rows}. "
            "This likely indicates a mismatched or corrupted checkpoint."
        )

    output_weight, output_name = get_lm_head_weight_and_name(model)
    if output_weight is None or output_name is None:
        raise ValueError("Could not locate a separate output embedding weight to repair damaged input rows")
    output_view = _weight_view(output_weight, output_name)
    _check_matching_layout(input_view, output_view)

    output_norms = _full_row_squared_norms(output_view).sqrt()
    unusable_source_local = damaged_local & (~torch.isfinite(output_norms) | (output_norms <= min_norm))
    unusable_source_ids = _gather_unique_ids(_global_ids(output_view, unusable_source_local))
    if unusable_source_ids:
        raise ValueError(
            "Cannot repair input embedding rows because the corresponding output rows are also damaged: "
            f"{list(unusable_source_ids)}"
        )

    healthy_local = torch.isfinite(input_norms) & (input_norms > min_norm)
    target_norm = _healthy_rms_norm(input_squared_norms, healthy_local)
    if not math.isfinite(target_norm) or target_norm <= min_norm:
        raise ValueError(f"Computed invalid healthy input embedding RMS norm: {target_norm}")

    replacement_scale = target_norm / output_norms[damaged_local]
    replacement = output_view.local.detach()[damaged_local].to(torch.float32)
    replacement.mul_(replacement_scale.to(torch.float32).unsqueeze(1))
    damaged_local_ids = torch.nonzero(damaged_local, as_tuple=False).flatten()
    input_view.local.index_copy_(0, damaged_local_ids, replacement.to(input_view.local.dtype))

    repaired_norms = _full_row_squared_norms(input_view).sqrt()
    failed_local = damaged_local & (~torch.isfinite(repaired_norms) | (repaired_norms <= min_norm))
    failed_ids = _gather_unique_ids(_global_ids(input_view, failed_local))
    if failed_ids:
        raise RuntimeError(f"Input embedding row repair did not produce healthy rows: {list(failed_ids)}")

    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        logger.warning(
            "Repaired %d input embedding row(s) in %s from %s: ids=%s, minimum_before=%.3e, target_norm=%.3e",
            len(damaged_ids),
            input_name,
            output_name,
            list(damaged_ids),
            min_norm_before,
            target_norm,
        )

    return EmbeddingRowRepairReport(
        input_embedding_name=input_name,
        output_embedding_name=output_name,
        repaired_row_ids=damaged_ids,
        min_norm_before=min_norm_before,
        target_norm=target_norm,
    )
