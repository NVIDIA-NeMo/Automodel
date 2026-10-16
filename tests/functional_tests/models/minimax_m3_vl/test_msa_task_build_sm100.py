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

"""The device-built MSA backward task tables against their Torch definition, on real forward schedules.

The Torch reference below is the definition of the tables: eight-query task rows in forward work-item
order, then the locality order. The device build keeps the task count and the CTA walk on the device,
so nothing downstream can notice a table that drifts from this definition or a descriptor that
disagrees with the host rule; this pins both, including work counts on either side of the rows/CTA
switch.
"""

from typing import Any

import pytest
import torch

from nemo_automodel.components.models.minimax_m3_vl import msa, msa_bindings
from nemo_automodel.components.models.minimax_m3_vl.kernels import msa_schedule as sched
from nemo_automodel.components.models.minimax_m3_vl.kernels import sm_count
from nemo_automodel.shared.import_utils import UnavailableError

_BLOCK, _KV_HEADS, _TOPK, _QUERY_HEADS = 128, 4, 16, 64


def _unavailable() -> str | None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        return "requires an SM100 GPU"
    try:
        msa_bindings.kernels()
    except UnavailableError:
        return "requires uv sync --extra msa"
    return None


_SKIP_REASON = _unavailable()
pytestmark = pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or "SM100 available")

if _SKIP_REASON is None:
    # Both modules import cutlass at module scope, which only the msa extra provides. Importing them
    # unconditionally would make this file a collection error on a runner that should skip it.
    from nemo_automodel.components.models.minimax_m3_vl.kernels import msa_task_build_sm100 as tb


def _packed_schedule(lengths: tuple[int, ...]) -> tuple[sched.MSABackwardSchedule, int, int]:
    """Build the forward CSR schedule for one packed row of documents; return it with its shapes."""
    device = torch.device("cuda", torch.cuda.current_device())
    num_tokens = sum(lengths)
    documents = torch.zeros(1, num_tokens, dtype=torch.int64, device=device)
    position = 0
    for document, length in enumerate(lengths, start=1):
        documents[0, position : position + length] = document
        position += length
    microbatch = msa.MSAMicrobatch.from_document_map(documents, forced_blocks=(0, 1))
    # Fully causal top-k support: every query selects every earlier key block of its own document.
    support_rows = [
        torch.where(
            torch.arange(_TOPK, device=device)[None, :] <= torch.arange(length, device=device)[:, None] // _BLOCK,
            torch.arange(_TOPK, device=device)[None, :],
            -1,
        )
        for length in lengths
    ]
    support = torch.cat(support_rows).expand(_KV_HEADS, -1, -1).to(torch.int32).contiguous()
    workspace_rows = microbatch.workspace_size
    row_ptr, q_indices, forward_schedule = msa_bindings.kernels().build_k2q_csr(
        support,
        microbatch.cu_seqlens,
        microbatch.cu_seqlens,
        _BLOCK,
        total_k=num_tokens,
        max_seqlen_q=microbatch.max_seqlen,
        max_seqlen_k=microbatch.max_seqlen,
        total_rows=workspace_rows // _BLOCK,
        qhead_per_kv=_QUERY_HEADS // _KV_HEADS,
        return_schedule=True,
    )
    schedule = sched.MSABackwardSchedule(
        row_ptr=row_ptr,
        q_indices=q_indices,
        scheduler_metadata=forward_schedule.scheduler_metadata,
        work_count=forward_schedule.work_count,
        cu_seqlens=microbatch.cu_seqlens,
        document_workspace_starts=microbatch.document_workspace_starts,
    )
    return schedule, num_tokens, workspace_rows


def _reference_tasks(schedule: sched.MSABackwardSchedule) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The Torch definition of the task tables, in forward work-item order.

    Splits every work item of ``schedule.scheduler_metadata`` at multiples of eight queries and
    synchronizes once for the exact task count. Returns int32 ``task_meta [num_tasks, 4]`` holding
    ``(batch=0, index_head, workspace_kblock, valid_queries)`` and ``task_qrows`` / ``task_qpos``
    ``[num_tasks, 8]`` holding the compact tensor rows / workspace positions of the task's queries,
    padded with -1.
    """
    device = schedule.row_ptr.device
    work_capacity = schedule.scheduler_metadata.shape[0]
    chunk = sched.QUERY_CHUNK

    work_ids = torch.arange(work_capacity, dtype=torch.int32, device=device)
    q_counts = torch.where(work_ids < schedule.work_count[0], schedule.scheduler_metadata[:, 3], 0)
    tasks_per_work = torch.div(q_counts + chunk - 1, chunk, rounding_mode="floor")
    probe = torch.stack((schedule.work_count[0], tasks_per_work.sum(dtype=torch.int32)))
    num_work_items, num_tasks = probe.tolist()
    if num_tasks <= 0:
        return (
            torch.empty((0, 4), dtype=torch.int32, device=device),
            torch.empty((0, chunk), dtype=torch.int32, device=device),
            torch.empty((0, chunk), dtype=torch.int32, device=device),
        )

    work_ids = work_ids[:num_work_items]
    tasks_per_work = tasks_per_work[:num_work_items]
    work_task_offsets = tasks_per_work.cumsum(0, dtype=torch.int32) - tasks_per_work
    task_work_ids = torch.repeat_interleave(work_ids, tasks_per_work, output_size=num_tasks)
    task_work = schedule.scheduler_metadata.index_select(0, task_work_ids)
    task_query_offsets = (
        torch.arange(num_tasks, dtype=torch.int32, device=device) - work_task_offsets.index_select(0, task_work_ids)
    ) * chunk
    task_valid = (task_work[:, 3] - task_query_offsets).clamp(max=chunk)

    task_heads = task_work[:, 0]
    row_offsets = task_heads * schedule.row_ptr.shape[1] + task_work[:, 1]
    csr_row_starts = schedule.row_ptr.reshape(-1).index_select(0, row_offsets)
    task_edge_starts = csr_row_starts + task_work[:, 2] + task_query_offsets

    slots = torch.arange(chunk, dtype=torch.int32, device=device).view(1, -1)
    valid_slots = slots < task_valid.view(-1, 1)
    edge_indices = task_heads.view(-1, 1) * schedule.q_indices.shape[1] + task_edge_starts.view(-1, 1) + slots
    edge_indices = edge_indices.clamp(min=0, max=schedule.q_indices.numel() - 1)
    query_local = schedule.q_indices.reshape(-1).index_select(0, edge_indices.reshape(-1)).view(num_tasks, -1)

    document_ordinals = task_work[:, 4]
    compact_document_starts = schedule.cu_seqlens[:-1].index_select(0, document_ordinals)
    workspace_document_starts = schedule.document_workspace_starts.index_select(0, document_ordinals)
    task_qrows = torch.where(valid_slots, compact_document_starts.view(-1, 1) + query_local, -1)
    task_qpos = torch.where(valid_slots, workspace_document_starts.view(-1, 1) + query_local, -1)

    global_kblocks = torch.div(workspace_document_starts, sched.BLOCK_SIZE, rounding_mode="floor") + task_work[:, 5]
    task_meta = torch.stack(
        (torch.zeros(num_tasks, dtype=torch.int32, device=device), task_heads, global_kblocks, task_valid), dim=-1
    )
    return task_meta, task_qrows, task_qpos


def _reference_order(
    task_meta: torch.Tensor, task_qrows: torch.Tensor, task_qpos: torch.Tensor, num_tokens: int, workspace_rows: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reorder reference rows [num_tasks, ...] by (index_head, locality window, key block, first query)."""
    window = tb._LOCALITY_WINDOW
    if task_meta.shape[0] <= 1:
        return task_meta, task_qrows, task_qpos
    head = task_meta[:, 1].to(torch.int64)
    kblock = task_meta[:, 2].to(torch.int64)
    qmin = task_qrows[:, 0].to(torch.int64)
    num_windows = (num_tokens + window - 1) // window
    num_kblocks = workspace_rows // sched.BLOCK_SIZE
    key = ((head * num_windows + qmin // window) * num_kblocks + kblock) * window + qmin % window
    order = torch.argsort(key)
    return task_meta.index_select(0, order), task_qrows.index_select(0, order), task_qpos.index_select(0, order)


def _build(schedule: sched.MSABackwardSchedule, num_tokens: int, workspace_rows: int, num_sms: int) -> Any:
    """Run the device build into freshly allocated scratch and table buffers of the required sizes."""
    scratch_words, table_words = tb.task_build_storage(schedule, num_tokens, workspace_rows)
    device = schedule.row_ptr.device
    scratch = torch.empty(scratch_words, dtype=torch.int32, device=device)
    tables = torch.empty(table_words, dtype=torch.int32, device=device)
    return tb.build_backward_tasks(
        schedule, num_tokens, workspace_rows, num_sms=num_sms, scratch=scratch, tables=tables
    )


def _check_build(schedule: sched.MSABackwardSchedule, num_tokens: int, workspace_rows: int, num_sms: int) -> int:
    """Assert the device tables, descriptor and launch bound for one schedule; return the task count."""
    expected = _reference_order(*_reference_tasks(schedule), num_tokens, workspace_rows)
    num_tasks = int(expected[0].shape[0])
    tables = _build(schedule, num_tokens, workspace_rows, num_sms)
    for actual, reference in zip((tables.task_meta, tables.task_qrows, tables.task_qpos), expected, strict=True):
        assert torch.equal(actual[:num_tasks], reference)
    desc = tables.desc.tolist()
    walk_rows = sched.rows_per_cta(num_tasks)
    assert desc[: tb.DESC_GRID_CTAS + 1] == [num_tasks, walk_rows, *sched.chunk_map(num_tasks, walk_rows, num_sms)]
    assert desc[tb.DESC_NUM_TASKS] == num_tasks and desc[tb.DESC_FLAGS] == 0
    assert tables.grid_launch >= desc[tb.DESC_GRID_CTAS]
    return num_tasks


@pytest.mark.parametrize("num_sms", (8, None))
@pytest.mark.parametrize("lengths", ((127, 129, 1, 2, 3, 5, 128), (4093,)))
def test_device_task_tables_match_the_torch_reference(lengths: tuple[int, ...], num_sms: int | None) -> None:
    schedule, num_tokens, workspace_rows = _packed_schedule(lengths)
    resolved = num_sms or sm_count(schedule.row_ptr.device)
    assert _check_build(schedule, num_tokens, workspace_rows, resolved) > 0


def test_truncated_work_counts_keep_the_tables_and_the_walk_exact() -> None:
    # 4096 tokens put the task count above ROWS_PER_CTA_SWITCH, so truncating the work count walks
    # the build through both rows/CTA regimes as well as the over/half/zero edges.
    schedule, num_tokens, workspace_rows = _packed_schedule((4096,))
    device = schedule.row_ptr.device
    num_sms = sm_count(device)
    work_count = int(schedule.work_count[0])
    # Padding rows past the count are live storage for the over-count case; zero them so the extra
    # work items contribute no tasks and the comparison stays meaningful.
    metadata = schedule.scheduler_metadata.clone()
    metadata[work_count:] = 0
    tasks_per_work = (metadata[:work_count, 3] + sched.QUERY_CHUNK - 1) // sched.QUERY_CHUNK
    prefix = tasks_per_work.cumsum(0).tolist()
    assert prefix[-1] > sched.ROWS_PER_CTA_SWITCH, prefix[-1]
    switch = sched.ROWS_PER_CTA_SWITCH
    counts = {
        "over_count": int(metadata.shape[0]) + 7,
        "zero_count": 0,
        "switch_below": max(w + 1 for w, tasks in enumerate(prefix) if tasks <= switch),
        "switch_above": min(w + 1 for w, tasks in enumerate(prefix) if tasks > switch),
    }
    observed = {}
    for label, count in counts.items():
        truncated = sched.MSABackwardSchedule(
            row_ptr=schedule.row_ptr,
            q_indices=schedule.q_indices,
            scheduler_metadata=metadata,
            work_count=torch.tensor([count], dtype=torch.int32, device=device),
            cu_seqlens=schedule.cu_seqlens,
            document_workspace_starts=schedule.document_workspace_starts,
        )
        observed[label] = _check_build(truncated, num_tokens, workspace_rows, num_sms)
    assert observed["zero_count"] == 0
    assert observed["switch_below"] <= switch < observed["switch_above"]
    assert observed["over_count"] == prefix[-1]


def test_too_many_locality_bins_raise_before_any_launch(monkeypatch: pytest.MonkeyPatch) -> None:
    # Past MAX_BINS the single-CTA bin scan and its scratch stop being cheap. The build refuses
    # instead of degrading, and it refuses before compiling or launching anything.
    schedule, num_tokens, workspace_rows = _packed_schedule((128,))
    monkeypatch.setattr(tb, "MAX_BINS", 1)
    monkeypatch.setattr(tb, "_compile", lambda: pytest.fail("compiled after rejecting the schedule"))
    with pytest.raises(ValueError, match="locality bins"):
        _build(schedule, num_tokens, workspace_rows, 8)
