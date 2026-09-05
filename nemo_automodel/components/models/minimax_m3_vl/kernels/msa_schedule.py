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

"""Forward-schedule adaptation and CTA scheduling for MSA SM100 backward.

Forward CSR work items become fixed task rows -- 8 compact queries x 16 main heads of one (index head, key
block) bucket, i.e. one 128-row M tile -- without expanding or sorting the canonical q2k edges.
"""

from dataclasses import dataclass

import torch

_BLOCK_SIZE = 128
_NUM_INDEX_HEADS = 4
_QUERY_CHUNK = 8
# Below this many task rows the 4-row CTA walk fills the machine better.
_ROWS_PER_CTA_SWITCH = 2400


@dataclass(frozen=True, slots=True)
class _MSABackwardSchedule:
    """Forward-derived int32 execution metadata for MSA backward.

    Built from canonical document-local q2k during the same autograd forward, so it
    must be saved with ``ctx.save_for_backward``. ``scheduler_metadata`` columns are
    ``(index_head, row_linear, q_begin, q_count, document_ordinal,
    document_local_kblock)``, valid only up to ``work_count``;
    :func:`_check_schedule` lists the per-field shapes.
    """

    row_ptr: torch.Tensor
    q_indices: torch.Tensor
    scheduler_metadata: torch.Tensor
    work_count: torch.Tensor
    cu_seqlens: torch.Tensor
    document_workspace_starts: torch.Tensor


def _check_schedule(schedule: _MSABackwardSchedule) -> None:
    """Reject dtype and shape violations of :class:`_MSABackwardSchedule`.

    Expected extents below are exact when non-negative, lower bounds when negative.
    """
    documents = max(schedule.cu_seqlens.numel() - 1, 0)
    contract = (
        ("row_ptr", "[4, rows + 1]", (_NUM_INDEX_HEADS, -2)),
        ("q_indices", "[4, edge_capacity]", (_NUM_INDEX_HEADS, -1)),
        ("scheduler_metadata", "[work_capacity, 6]", (-1, 6)),
        ("work_count", "[1]", (1,)),
        ("cu_seqlens", "[documents + 1]", (-2,)),
        ("document_workspace_starts", "[documents]", (documents,)),
    )
    for name, layout, expected in contract:
        tensor = getattr(schedule, name)
        if tensor.dtype != torch.int32:
            raise TypeError(f"{name} must be int32, got {tensor.dtype}")
        shape = tuple(tensor.shape)
        if len(shape) != len(expected) or any(
            extent != bound if bound >= 0 else extent < -bound for extent, bound in zip(shape, expected)
        ):
            raise ValueError(f"{name} must have shape {layout}, got {shape}")


def _build_backward_tasks(
    schedule: _MSABackwardSchedule,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert forward CSR work items into fixed-width task rows.

    Split points are multiples of eight, so every work item splits into whole tiles.
    Returns int32 ``task_meta [num_tasks, 4]`` -- ``(batch=0, index_head,
    workspace_global_kblock, valid_queries)`` -- plus ``-1``-padded ``task_qrows`` and
    ``task_qpos``, both ``[num_tasks, 8]``: compact Q/dO/O/LSE/dQ rows, and query
    positions for the causal predicate only.
    """
    _check_schedule(schedule)
    device = schedule.row_ptr.device
    work_capacity = schedule.scheduler_metadata.shape[0]

    work_ids = torch.arange(work_capacity, dtype=torch.int32, device=device)
    q_counts = torch.where(work_ids < schedule.work_count[0], schedule.scheduler_metadata[:, 3], 0)
    tasks_per_work = torch.div(q_counts + _QUERY_CHUNK - 1, _QUERY_CHUNK, rounding_mode="floor")
    probe = torch.stack((schedule.work_count[0], tasks_per_work.sum(dtype=torch.int32)))
    num_work_items, num_tasks = probe.tolist()  # The one accepted task-count host synchronization.
    if num_tasks <= 0:
        return (
            torch.empty((0, 4), dtype=torch.int32, device=device),
            torch.empty((0, _QUERY_CHUNK), dtype=torch.int32, device=device),
            torch.empty((0, _QUERY_CHUNK), dtype=torch.int32, device=device),
        )

    work_ids = work_ids[:num_work_items]
    tasks_per_work = tasks_per_work[:num_work_items]
    work_task_offsets = tasks_per_work.cumsum(0, dtype=torch.int32) - tasks_per_work
    task_work_ids = torch.repeat_interleave(work_ids, tasks_per_work, output_size=num_tasks)
    task_work = schedule.scheduler_metadata.index_select(0, task_work_ids)
    task_in_work = torch.arange(num_tasks, dtype=torch.int32, device=device) - work_task_offsets.index_select(
        0, task_work_ids
    )
    task_valid = (task_work[:, 3] - task_in_work * _QUERY_CHUNK).clamp(max=_QUERY_CHUNK)

    task_heads = task_work[:, 0]
    row_offsets = task_heads * schedule.row_ptr.shape[1] + task_work[:, 1]
    csr_row_starts = schedule.row_ptr.reshape(-1).index_select(0, row_offsets)
    task_edge_starts = csr_row_starts + task_work[:, 2] + task_in_work * _QUERY_CHUNK

    slots = torch.arange(_QUERY_CHUNK, dtype=torch.int32, device=device).view(1, -1)
    valid_slots = slots < task_valid.view(-1, 1)
    edge_indices = task_heads.view(-1, 1) * schedule.q_indices.shape[1] + task_edge_starts.view(-1, 1) + slots
    edge_indices = edge_indices.clamp(min=0, max=schedule.q_indices.numel() - 1)
    query_local = schedule.q_indices.reshape(-1).index_select(0, edge_indices.reshape(-1)).view(num_tasks, -1)

    document_ordinals = task_work[:, 4]
    compact_document_starts = schedule.cu_seqlens[:-1].index_select(0, document_ordinals)
    workspace_document_starts = schedule.document_workspace_starts.index_select(0, document_ordinals)
    task_qrows = torch.where(valid_slots, compact_document_starts.view(-1, 1) + query_local, -1)
    task_qpos = torch.where(valid_slots, workspace_document_starts.view(-1, 1) + query_local, -1)

    global_kblocks = torch.div(workspace_document_starts, _BLOCK_SIZE, rounding_mode="floor") + task_work[:, 5]
    task_meta = torch.stack(
        (torch.zeros(num_tasks, dtype=torch.int32, device=device), task_heads, global_kblocks, task_valid),
        dim=-1,
    )
    return task_meta, task_qrows, task_qpos


def _chunk_map(num_rows: int, rows_per_cta: int, num_sms: int) -> tuple[int, int, int]:
    """Split task rows into whole waves plus a balanced final wave.

    Returns ``(num_full_ctas, tail_rows, grid_ctas)``, which with
    ``_cta_row_interval`` covers every task row exactly once.
    """
    num_chunks = -(-num_rows // rows_per_cta)
    # Only complete chunks may form full waves; a partial one makes rows_left
    # negative and silently drops the last rows_per_cta - 1 rows.
    num_full = min((num_chunks // num_sms) * num_sms, num_rows // rows_per_cta)
    rows_left = num_rows - num_full * rows_per_cta
    if rows_left <= 0:
        return num_full, 1, num_full
    tail_rows = max(1, -(-rows_left // num_sms))
    if tail_rows < 3:
        # Measured: for such short tails the per-CTA prologue eats the gain.
        return 0, rows_per_cta, num_chunks
    return num_full, tail_rows, num_full + -(-rows_left // tail_rows)


def _select_rows_per_cta(num_rows: int) -> int:
    """Return the full-wave CTA walk length: 4 for small backward passes, else 8."""
    return 4 if num_rows <= _ROWS_PER_CTA_SWITCH else 8


def _cta_row_interval(
    bidx: int,
    num_rows: int,
    rows_per_cta: int,
    num_full_ctas: int,
    tail_rows: int,
) -> tuple[int, int]:
    """Return ``(row_lo, row_hi)`` for one CTA.

    Host mirror of the interval arithmetic at the top of the ``msa_backward_sm100``
    device kernel; keep the two in sync.
    """
    if bidx >= num_full_ctas:
        row_lo = num_full_ctas * rows_per_cta + (bidx - num_full_ctas) * tail_rows
        return row_lo, min(row_lo + tail_rows, num_rows)
    row_lo = bidx * rows_per_cta
    return row_lo, min(row_lo + rows_per_cta, num_rows)
