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

"""CuTe-free contract of the MSA backward task build.

What ``_msa`` saves from the forward and what the CPU tests need without importing CuTe: the
forward-derived schedule and its shape check, the CTA-walk rule that ``msa_task_build_sm100``
mirrors on the device, and the grid bound the main kernel is launched with.
"""

from dataclasses import dataclass

import torch

_BLOCK_SIZE = 128
_NUM_INDEX_HEADS = 4
_QUERY_CHUNK = 8
_ROWS_PER_CTA_SMALL = 4
_ROWS_PER_CTA_SWITCH = 2400
# 64, not 32: a CTA flushes dK/dV once per bucket run it walks, so the flush count is
# ~= buckets + CTAs.  At s4096 there are only 128 buckets against 881 CTAs, i.e. 7.7 flushes
# per bucket -- nearly all of the flush work is CTA-boundary splitting, and every split
# re-does the whole flush (dK/dV T2R + quad transpose + FP32 atomics).  Halving the CTA count
# halves that redundancy.  64 is where it stops paying: 96 and 128 cut splits further but lose
# more to the shrinking wave count (128 is +3.6% at s4096, which runs only 1.3 waves).
_ROWS_PER_CTA_LARGE = 64


@dataclass(frozen=True, slots=True)
class _MSABackwardSchedule:
    """Forward-derived int32 metadata; save with ``ctx.save_for_backward``.

    ``scheduler_metadata`` columns are
    ``(index_head, row_linear, q_begin, q_count, document_ordinal,
    document_local_kblock)``, valid only up to ``work_count``.
    """

    row_ptr: torch.Tensor
    q_indices: torch.Tensor
    scheduler_metadata: torch.Tensor
    work_count: torch.Tensor
    cu_seqlens: torch.Tensor
    document_workspace_starts: torch.Tensor


def _check_schedule(schedule: _MSABackwardSchedule) -> None:
    """Validate the int32 schedule shapes."""
    documents = max(schedule.cu_seqlens.numel() - 1, 0)
    row_shape = schedule.row_ptr.shape
    edge_shape = schedule.q_indices.shape
    work_shape = schedule.scheduler_metadata.shape
    contract = (
        ("row_ptr", "[4, rows + 1]", len(row_shape) == 2 and row_shape[0] == _NUM_INDEX_HEADS and row_shape[1] >= 2),
        (
            "q_indices",
            "[4, edge_capacity]",
            len(edge_shape) == 2 and edge_shape[0] == _NUM_INDEX_HEADS and edge_shape[1] >= 1,
        ),
        (
            "scheduler_metadata",
            "[work_capacity, 6]",
            len(work_shape) == 2 and work_shape[0] >= 1 and work_shape[1] == 6,
        ),
        ("work_count", "[1]", schedule.work_count.shape == (1,)),
        ("cu_seqlens", "[documents + 1]", schedule.cu_seqlens.ndim == 1 and schedule.cu_seqlens.numel() >= 2),
        ("document_workspace_starts", "[documents]", schedule.document_workspace_starts.shape == (documents,)),
    )
    for name, layout, valid_shape in contract:
        tensor = getattr(schedule, name)
        if tensor.dtype != torch.int32:
            raise TypeError(f"{name} must be int32, got {tensor.dtype}")
        if not valid_shape:
            raise ValueError(f"{name} must have shape {layout}, got {tuple(tensor.shape)}")


def _chunk_map(num_rows: int, rows_per_cta: int, num_sms: int) -> tuple[int, int, int]:
    """Return ``(num_full_ctas, tail_rows, grid_ctas)`` for a walk covering every row once.

    The tables kernel of ``msa_task_build_sm100`` mirrors this rule on the device.
    """
    num_chunks = -(-num_rows // rows_per_cta)
    # A partial chunk cannot join a full wave: that would silently drop tail rows.
    num_full = min((num_chunks // num_sms) * num_sms, num_rows // rows_per_cta)
    rows_left = num_rows - num_full * rows_per_cta
    if rows_left <= 0:
        return num_full, 1, num_full
    tail_rows = -(-rows_left // num_sms)
    if tail_rows < 3:
        # Measured: for such short tails the per-CTA prologue eats the gain.
        return 0, rows_per_cta, num_chunks
    return num_full, tail_rows, num_full + -(-rows_left // tail_rows)


def _select_rows_per_cta(num_rows: int) -> int:
    """Select the CTA walk length; ``msa_task_build_sm100`` mirrors this rule on the device."""
    return _ROWS_PER_CTA_SMALL if num_rows <= _ROWS_PER_CTA_SWITCH else _ROWS_PER_CTA_LARGE


def _grid_launch_bound(capacity: int, num_sms: int) -> int:
    """Bound every count up to capacity: full CTAs plus at most one tail CTA per SM.

    ``_chunk_map(n, r, s)[2] <= n // r + s`` for every count, so the bound takes the larger of the
    small and the large walk regime over ``[0, capacity]``.
    """
    return max(min(capacity, _ROWS_PER_CTA_SWITCH) // _ROWS_PER_CTA_SMALL, capacity // _ROWS_PER_CTA_LARGE) + num_sms
