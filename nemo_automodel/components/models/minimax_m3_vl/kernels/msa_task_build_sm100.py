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

"""Build locality-ordered backward task tables and their CTA walk.

The fused path uses four launches: work scan, segment keys, bin scan, table scatter.

Forward buckets have ascending queries and contiguous (head, query window, key block)
segments, so ``offsets[segment] + task - first_tasks[segment]`` reproduces the eager locality sort.
Noncontiguous segments keep source order (descriptor flag 2). Capacity overflow
disables the CTA walk (flag 1). Exact task counts and the walk stay on the device;
the host launches a grid bound and surplus CTAs take empty intervals.
"""

import os
from dataclasses import dataclass
from typing import Any

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings import driver as cuda
from cutlass import Int32
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream

from nemo_automodel.components.models.minimax_m3_vl.kernels.msa_schedule import (
    _BLOCK_SIZE,
    _NUM_INDEX_HEADS,
    _QUERY_CHUNK,
    _ROWS_PER_CTA_LARGE,
    _ROWS_PER_CTA_SMALL,
    _ROWS_PER_CTA_SWITCH,
    _build_backward_tasks,
    _check_schedule,
    _chunk_map,
    _MSABackwardSchedule,
    _rows_per_cta_override,
    _select_rows_per_cta,
)

THREADS = 256
SCAN_THREADS = 1024
DESC_WORDS = 8
MAX_BINS = 1 << 20
# descriptor words (int32[8])
DESC_NUM_TASK_ROWS = 0
DESC_ROWS_PER_CTA = 1
DESC_NUM_FULL_CTAS = 2
DESC_TAIL_ROWS = 3
DESC_GRID_CTAS = 4
DESC_NUM_WORK = 5
DESC_NUM_TASKS = 6
DESC_FLAGS = 7
FLAG_CAPACITY_OVERFLOW = 1
FLAG_NONCONTIGUOUS_SEGMENTS = 2
BIN_COUNTS = 0
BIN_FIRST_TASKS = 1
BIN_LAST_TASKS = 2
BIN_OFFSETS = 3
_INT32_MAX = 2**31 - 1

# Zero disables locality sorting; a positive window keeps nearby Q/dO/dQ rows together.
_TASK_ORDER_CHUNK = int(os.environ.get("MSA_M3_TASK_ORDER_CHUNK", "2048"))
_TASK_BUILD = os.environ.get("MSA_M3_TASK_BUILD", "fused")
if _TASK_BUILD not in ("fused", "torch"):
    raise ValueError(f"MSA_M3_TASK_BUILD must be 'fused' or 'torch', got {_TASK_BUILD!r}")

_COMPILE_CACHE: dict[tuple[Any, ...], Any] = {}


@dataclass(frozen=True, slots=True)
class _MSABackwardTaskTables:
    """Task tables in the locality order plus the CTA-walk descriptor the main kernel reads.

    The tables hold ``capacity`` rows; rows past the task count are never written nor read (the
    descriptor's exact count bounds every CTA interval). ``grid_launch >= desc[4]``.
    """

    task_meta: torch.Tensor  # [capacity, 4] int32
    task_qrows: torch.Tensor  # [capacity, 8] int32
    task_qpos: torch.Tensor  # [capacity, 8] int32
    desc: torch.Tensor  # [8] int32 on the device (see DESC_*)
    capacity: int
    grid_launch: int  # CTAs to launch for the main kernel

    def device_counts(self) -> dict[str, int]:
        """Read the descriptor (synchronizes the stream); for tests and diagnostics."""
        words = self.desc.tolist()
        names = (
            "num_task_rows",
            "rows_per_cta",
            "num_full_ctas",
            "tail_rows",
            "grid_ctas",
            "num_work",
            "num_tasks",
            "flags",
        )
        return dict(zip(names, words))


def _task_capacity(schedule: _MSABackwardSchedule) -> int:
    """Upper bound of the task count from the schedule shapes (disjoint work items)."""
    return _NUM_INDEX_HEADS * schedule.q_indices.shape[1] // _QUERY_CHUNK + schedule.scheduler_metadata.shape[0]


def _task_build_sizes(
    schedule: _MSABackwardSchedule, num_tokens: int, workspace_rows: int, chunk: int
) -> tuple[int, int, int, int]:
    """``(capacity, bins, scratch_words, table_words)`` of the fused build for this schedule (host shape math only)."""
    capacity = _task_capacity(schedule)
    window = max(chunk, 1)  # Storage may be queried before the fused eligibility check.
    num_chunks = (num_tokens + window - 1) // window
    num_kblocks = workspace_rows // _BLOCK_SIZE
    bins = _NUM_INDEX_HEADS * num_chunks * max(num_kblocks, 1)
    work_capacity = int(schedule.scheduler_metadata.shape[0])
    return capacity, bins, DESC_WORDS + work_capacity + capacity + 4 * bins, capacity * (4 + 2 * _QUERY_CHUNK)


def _grid_launch_bound(capacity: int, rows_forced: int, num_sms: int) -> int:
    """Bound every count up to capacity: full CTAs plus at most one tail CTA per SM."""
    if rows_forced > 0:
        return capacity // rows_forced + num_sms
    return max(min(capacity, _ROWS_PER_CTA_SWITCH) // _ROWS_PER_CTA_SMALL, capacity // _ROWS_PER_CTA_LARGE) + num_sms


@cute.jit
def _decode_task(
    mWorkMeta: cute.Tensor,  # [work_capacity, 6] int32 scheduler_metadata
    mTaskEnds: cute.Tensor,  # [work_capacity] int32 inclusive prefix of tasks per work item
    mRowPtr: cute.Tensor,  # [4, rows + 1] int32
    num_work: Int32,
    task: Int32,
):
    """Forward-order task ``task`` -> (index_head, document, local kblock, valid, first edge)."""
    # w = first work item whose inclusive prefix exceeds task (lower_bound of task + 1)
    lo = Int32(0)
    hi = num_work
    while lo < hi:
        mid = (lo + hi) // 2
        if mTaskEnds[mid] <= task:
            lo = mid + 1
        else:
            hi = mid
    w = lo
    head = mWorkMeta[w, 0]
    row_linear = mWorkMeta[w, 1]
    q_begin = mWorkMeta[w, 2]
    q_count = mWorkMeta[w, 3]
    doc = mWorkMeta[w, 4]
    kblock_local = mWorkMeta[w, 5]
    first_task = mTaskEnds[w] - ((q_count + Int32(_QUERY_CHUNK - 1)) // Int32(_QUERY_CHUNK))
    query_offset = (task - first_task) * Int32(_QUERY_CHUNK)
    valid = q_count - query_offset
    if valid > Int32(_QUERY_CHUNK):
        valid = Int32(_QUERY_CHUNK)
    edge_start = mRowPtr[head, row_linear] + q_begin + query_offset
    return head, doc, kblock_local, valid, edge_start


@cute.jit
def _warp_inclusive_scan(value: Int32, lane: Int32) -> Int32:
    """Inclusive prefix sum across the 32 lanes of a warp (shfl.up; clamp 0 = plain up-shuffle)."""
    acc = value
    for s in cutlass.range_constexpr(5):
        other = cute.arch.shuffle_sync_up(acc, 1 << s, mask_and_clamp=0)
        if lane >= Int32(1 << s):
            acc = acc + other
    return acc


@cute.jit
def _cta_exclusive_scan(value: Int32, tidx: Int32, with_total: cutlass.Constexpr = False):
    """Exclusive prefix across SCAN_THREADS; all threads must participate."""
    lane = tidx % Int32(32)
    warp = tidx // Int32(32)
    incl = _warp_inclusive_scan(value, lane)
    warp_totals = cute.make_tensor(cute.arch.alloc_smem(Int32, 33), cute.make_layout(33))
    if lane == Int32(31):
        warp_totals[warp] = incl
    cute.arch.sync_threads()
    if warp == Int32(0):
        mine = warp_totals[lane]
        scanned = _warp_inclusive_scan(mine, lane)
        warp_totals[lane] = scanned - mine
        if cutlass.const_expr(with_total):
            if lane == Int32(31):
                warp_totals[32] = scanned
    cute.arch.sync_threads()
    offset = warp_totals[warp] + (incl - value)
    if cutlass.const_expr(with_total):
        return offset, warp_totals[32]
    return offset


class _MSATaskBuildSm100:
    """One JIT owns the four build stages and the interpretation of segment scratch."""

    @cute.jit
    def __call__(
        self,
        mWorkMeta: cute.Tensor,  # [work_capacity, 6] int32
        mWorkCount: cute.Tensor,  # [1] int32
        mTaskEnds: cute.Tensor,  # [work_capacity] int32 scratch
        mRowPtr: cute.Tensor,  # [4, rows + 1] int32
        mQIdx: cute.Tensor,  # [4, edge_capacity] int32
        mCuSeqlens: cute.Tensor,  # [documents + 1] int32
        mDocStarts: cute.Tensor,  # [documents] int32 workspace starts
        mTaskSegments: cute.Tensor,  # [capacity] int32 scratch: segment of each source task
        mBins: cute.Tensor,  # [4, bins] int32 scratch: counts, first tasks, last tasks, offsets
        mTaskMeta: cute.Tensor,  # [capacity, 4] int32 out
        mQRows: cute.Tensor,  # [capacity, 8] int32 out
        mQPos: cute.Tensor,  # [capacity, 8] int32 out
        mDesc: cute.Tensor,  # [8] int32 out
        num_windows: Int32,
        num_kblocks: Int32,
        locality_window: Int32,
        rows_forced: Int32,  # > 0: forced walk length (MSA_M3_ROWS_PER_CTA); 0: the size rule below
        num_sms: Int32,
        stream: cuda.CUstream,
    ):
        capacity = cute.size(mTaskSegments)
        self.scan_work_kernel(mWorkMeta, mWorkCount, mTaskEnds, mBins, mDesc).launch(
            grid=[1, 1, 1], block=[SCAN_THREADS, 1, 1], stream=stream
        )
        self.assign_segments_kernel(
            mWorkMeta,
            mTaskEnds,
            mRowPtr,
            mQIdx,
            mCuSeqlens,
            mDocStarts,
            mTaskSegments,
            mBins,
            mDesc,
            num_windows,
            num_kblocks,
            locality_window,
        ).launch(grid=[cute.ceil_div(capacity, THREADS), 1, 1], block=[THREADS, 1, 1], stream=stream)
        self.scan_bins_kernel(mBins, mDesc).launch(grid=[1, 1, 1], block=[SCAN_THREADS, 1, 1], stream=stream)
        self.scatter_tables_kernel(
            mWorkMeta,
            mTaskEnds,
            mRowPtr,
            mQIdx,
            mCuSeqlens,
            mDocStarts,
            mTaskSegments,
            mBins,
            mTaskMeta,
            mQRows,
            mQPos,
            mDesc,
            rows_forced,
            num_sms,
        ).launch(grid=[cute.ceil_div(capacity, THREADS), 1, 1], block=[THREADS, 1, 1], stream=stream)

    @cute.kernel
    def scan_work_kernel(
        self,
        mWorkMeta: cute.Tensor,
        mWorkCount: cute.Tensor,
        mTaskEnds: cute.Tensor,
        mBins: cute.Tensor,
        mDesc: cute.Tensor,
    ):
        """Inclusive task offsets, exact counts and cleared segment bins."""
        tidx, _, _ = cute.arch.thread_idx()
        n = Int32(cute.size(mTaskEnds))
        num_work = mWorkCount[0]
        if num_work > n:
            num_work = n
        if num_work < Int32(0):
            num_work = Int32(0)
        per = (num_work + Int32(SCAN_THREADS - 1)) // Int32(SCAN_THREADS)
        lo = tidx * per
        hi = lo + per
        if hi > num_work:
            hi = num_work
        local = Int32(0)
        w = lo
        while w < hi:
            local = local + (mWorkMeta[w, 3] + Int32(_QUERY_CHUNK - 1)) // Int32(_QUERY_CHUNK)
            w = w + 1
        running, total = _cta_exclusive_scan(local, tidx, with_total=True)
        w = lo
        while w < hi:
            running = running + (mWorkMeta[w, 3] + Int32(_QUERY_CHUNK - 1)) // Int32(_QUERY_CHUNK)
            mTaskEnds[w] = running
            w = w + 1
        if tidx == Int32(0):
            mDesc[DESC_NUM_WORK] = num_work
            mDesc[DESC_NUM_TASKS] = total
            mDesc[DESC_FLAGS] = Int32(0)
        bins = Int32(cute.size(mBins, mode=[1]))
        b = tidx
        while b < bins:
            mBins[BIN_COUNTS, b] = Int32(0)
            mBins[BIN_FIRST_TASKS, b] = Int32(_INT32_MAX)
            mBins[BIN_LAST_TASKS, b] = Int32(-1)
            b = b + Int32(SCAN_THREADS)

    @cute.kernel
    def assign_segments_kernel(
        self,
        mWorkMeta: cute.Tensor,
        mTaskEnds: cute.Tensor,
        mRowPtr: cute.Tensor,
        mQIdx: cute.Tensor,
        mCuSeqlens: cute.Tensor,
        mDocStarts: cute.Tensor,
        mTaskSegments: cute.Tensor,
        mBins: cute.Tensor,
        mDesc: cute.Tensor,
        num_windows: Int32,
        num_kblocks: Int32,
        locality_window: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        task = bidx * Int32(THREADS) + tidx
        num_work = mDesc[DESC_NUM_WORK]
        num_tasks = mDesc[DESC_NUM_TASKS]
        capacity = Int32(cute.size(mTaskSegments))
        if num_tasks > capacity:
            num_tasks = capacity
        if task < num_tasks:
            head, doc, kblock_local, _valid, edge_start = _decode_task(mWorkMeta, mTaskEnds, mRowPtr, num_work, task)
            # slot 0 is always valid and queries ascend inside a bucket: the task's first query
            first_query = mCuSeqlens[doc] + mQIdx[head, edge_start]
            kblock = mDocStarts[doc] // Int32(_BLOCK_SIZE) + kblock_local
            segment = (head * num_windows + first_query // locality_window) * num_kblocks + kblock
            mTaskSegments[task] = segment
            cute.arch.atomic_add(mBins.iterator + cute.crd2idx((Int32(BIN_COUNTS), segment), mBins.layout), Int32(1))
            cute.arch.atomic_min(mBins.iterator + cute.crd2idx((Int32(BIN_FIRST_TASKS), segment), mBins.layout), task)
            cute.arch.atomic_max(mBins.iterator + cute.crd2idx((Int32(BIN_LAST_TASKS), segment), mBins.layout), task)

    @cute.kernel
    def scan_bins_kernel(
        self,
        mBins: cute.Tensor,
        mDesc: cute.Tensor,
    ):
        """Exclusive bin offsets; flag segments that are not contiguous source runs."""
        tidx, _, _ = cute.arch.thread_idx()
        n = Int32(cute.size(mBins, mode=[1]))
        per = (n + Int32(SCAN_THREADS - 1)) // Int32(SCAN_THREADS)
        lo = tidx * per
        hi = lo + per
        if hi > n:
            hi = n
        local = Int32(0)
        bad = Int32(0)
        b = lo
        while b < hi:
            count = mBins[BIN_COUNTS, b]
            local = local + count
            if count > Int32(0):
                if mBins[BIN_LAST_TASKS, b] - mBins[BIN_FIRST_TASKS, b] + Int32(1) != count:
                    bad = Int32(1)
            b = b + 1
        running = _cta_exclusive_scan(local, tidx)
        b = lo
        while b < hi:
            mBins[BIN_OFFSETS, b] = running
            running = running + mBins[BIN_COUNTS, b]
            b = b + 1
        if bad > Int32(0):
            cute.arch.atomic_or(mDesc.iterator + DESC_FLAGS, Int32(FLAG_NONCONTIGUOUS_SEGMENTS))

    @cute.kernel
    def scatter_tables_kernel(
        self,
        mWorkMeta: cute.Tensor,
        mTaskEnds: cute.Tensor,
        mRowPtr: cute.Tensor,
        mQIdx: cute.Tensor,
        mCuSeqlens: cute.Tensor,
        mDocStarts: cute.Tensor,
        mTaskSegments: cute.Tensor,
        mBins: cute.Tensor,
        mTaskMeta: cute.Tensor,
        mQRows: cute.Tensor,
        mQPos: cute.Tensor,
        mDesc: cute.Tensor,
        rows_forced: Int32,
        num_sms: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        task = bidx * Int32(THREADS) + tidx
        num_work = mDesc[DESC_NUM_WORK]
        total = mDesc[DESC_NUM_TASKS]
        flags = mDesc[DESC_FLAGS]
        capacity = Int32(cute.size(mTaskSegments))
        num_tasks = total
        if num_tasks > capacity:
            num_tasks = capacity
        if task < num_tasks:
            segment = mTaskSegments[task]
            output_row = task  # source (bucket-major) order when the segments are not contiguous runs
            if (flags & Int32(FLAG_NONCONTIGUOUS_SEGMENTS)) == Int32(0):
                output_row = mBins[BIN_OFFSETS, segment] + (task - mBins[BIN_FIRST_TASKS, segment])
            head, doc, kblock_local, valid, edge_start = _decode_task(mWorkMeta, mTaskEnds, mRowPtr, num_work, task)
            compact_start = mCuSeqlens[doc]
            workspace_start = mDocStarts[doc]
            mTaskMeta[output_row, 0] = Int32(0)
            mTaskMeta[output_row, 1] = head
            mTaskMeta[output_row, 2] = workspace_start // Int32(_BLOCK_SIZE) + kblock_local
            mTaskMeta[output_row, 3] = valid
            for s in cutlass.range_constexpr(_QUERY_CHUNK):
                qrow = Int32(-1)
                qpos = Int32(-1)
                if Int32(s) < valid:
                    query_local = mQIdx[head, edge_start + Int32(s)]
                    qrow = compact_start + query_local
                    qpos = workspace_start + query_local
                mQRows[output_row, s] = qrow
                mQPos[output_row, s] = qpos
        if task == Int32(0):
            # device mirror of msa_schedule._select_rows_per_cta(num_rows) / _chunk_map(num_rows, rows_per_cta, num_sms)
            num_rows = num_tasks
            overflow = Int32(0)
            if total > capacity:
                overflow = Int32(FLAG_CAPACITY_OVERFLOW)
                num_rows = Int32(0)  # contract violation: no CTA walks, zero gradients, flag set
            rows_per_cta = rows_forced
            if rows_per_cta <= Int32(0):
                rows_per_cta = Int32(_ROWS_PER_CTA_LARGE)
                if num_rows <= Int32(_ROWS_PER_CTA_SWITCH):
                    rows_per_cta = Int32(_ROWS_PER_CTA_SMALL)
            num_cta_chunks = (num_rows + rows_per_cta - Int32(1)) // rows_per_cta
            num_full = (num_cta_chunks // num_sms) * num_sms
            whole = num_rows // rows_per_cta
            if num_full > whole:
                num_full = whole
            rows_left = num_rows - num_full * rows_per_cta
            tail = Int32(1)
            grid = num_full
            if rows_left > Int32(0):
                tail = (rows_left + num_sms - Int32(1)) // num_sms
                if tail < Int32(3):
                    num_full = Int32(0)
                    tail = rows_per_cta
                    grid = num_cta_chunks
                else:
                    grid = num_full + (rows_left + tail - Int32(1)) // tail
            mDesc[DESC_NUM_TASK_ROWS] = num_rows
            mDesc[DESC_ROWS_PER_CTA] = rows_per_cta
            mDesc[DESC_NUM_FULL_CTAS] = num_full
            mDesc[DESC_TAIL_ROWS] = tail
            mDesc[DESC_GRID_CTAS] = grid
            mDesc[DESC_FLAGS] = flags | overflow


def _compile(device: torch.device) -> Any:
    key = ("minimax-m3-msa-task-build-sm100", torch.cuda.get_device_capability(device))
    if key not in _COMPILE_CACHE:
        n_work = cute.sym_int32(symbol="work_capacity")
        n_rows = cute.sym_int32(symbol="rows_plus_one")
        n_edges = cute.sym_int32(symbol="edge_capacity")
        n_docs = cute.sym_int32(symbol="documents")
        n_docs1 = cute.sym_int32(symbol="documents_plus_one")
        n_cap = cute.sym_int32(symbol="capacity")
        n_bins = cute.sym_int32(symbol="bins")

        def tensor(shape, align=4):
            return make_fake_compact_tensor(
                Int32, shape, stride_order=tuple(reversed(range(len(shape)))), assumed_align=align
            )

        task_rows = tensor((n_cap, _QUERY_CHUNK), align=16)
        tensors = (
            tensor((n_work, 6)),
            tensor((1,)),
            tensor((n_work,)),
            tensor((_NUM_INDEX_HEADS, n_rows)),
            tensor((_NUM_INDEX_HEADS, n_edges)),
            tensor((n_docs1,)),
            tensor((n_docs,)),
            tensor((n_cap,)),
            tensor((4, n_bins)),
            tensor((n_cap, 4), align=16),
            task_rows,
            task_rows,
            tensor((DESC_WORDS,)),
        )
        # num_windows, num_kblocks, locality_window, rows_forced, num_sms
        scalars = (Int32(0), Int32(0), Int32(0), Int32(0), Int32(0))
        _COMPILE_CACHE[key] = cute.compile(
            _MSATaskBuildSm100(),
            *tensors,
            *scalars,
            make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    return _COMPILE_CACHE[key]


def compile_task_build(device: torch.device) -> None:
    """Precompile the fused build; no-op when the Torch implementation is selected."""
    if _TASK_BUILD == "fused":
        _compile(device)


def _fused_task_build_applies(schedule: _MSABackwardSchedule, bins: int, chunk: int) -> bool:
    """Whether a validated schedule can use the fused implementation."""
    if chunk <= 0 or bins > MAX_BINS:
        return False
    device = schedule.row_ptr.device
    if device.type != "cuda" or any(
        t.device != device
        for t in (
            schedule.q_indices,
            schedule.scheduler_metadata,
            schedule.work_count,
            schedule.cu_seqlens,
            schedule.document_workspace_starts,
        )
    ):
        return False
    return True


def _build_backward_tasks_fused(
    schedule: _MSABackwardSchedule,
    num_tokens: int,
    workspace_rows: int,
    chunk: int,
    *,
    num_sms: int,
    scratch: torch.Tensor | None = None,
    tables: torch.Tensor | None = None,
) -> _MSABackwardTaskTables | None:
    """Enqueue four launches without host sync; return None for the torch fallback.

    The descriptor stays at scratch offset zero to preserve its 16-byte alignment.
    """
    _check_schedule(schedule)
    capacity, bins, scratch_words, table_words = _task_build_sizes(schedule, num_tokens, workspace_rows, chunk)
    if not _fused_task_build_applies(schedule, bins, chunk):
        return None
    device = schedule.row_ptr.device
    num_chunks = (num_tokens + chunk - 1) // chunk
    num_kblocks = workspace_rows // _BLOCK_SIZE
    meta = schedule.scheduler_metadata.contiguous()
    work_capacity = int(meta.shape[0])
    exe = _compile(device)
    row_ptr = schedule.row_ptr.contiguous()
    q_idx = schedule.q_indices.contiguous()
    cu = schedule.cu_seqlens.contiguous()
    dws = schedule.document_workspace_starts.contiguous()
    work_count = schedule.work_count.contiguous()
    # scratch: descriptor | task ends | task segments | bins; tables: meta | qrows | qpos.
    if scratch is None or scratch.numel() < scratch_words:
        scratch = torch.empty(scratch_words, dtype=torch.int32, device=device)
    desc = scratch[:DESC_WORDS]
    segments_start = DESC_WORDS + work_capacity
    bins_start = segments_start + capacity
    task_ends = scratch[DESC_WORDS:segments_start]
    task_segments = scratch[segments_start:bins_start]
    segment_bins = scratch[bins_start:scratch_words].view(4, bins)
    if tables is None or tables.numel() < table_words:
        tables = torch.empty(table_words, dtype=torch.int32, device=device)
    task_meta = tables[: 4 * capacity].view(capacity, 4)
    task_qrows = tables[4 * capacity : (4 + _QUERY_CHUNK) * capacity].view(capacity, _QUERY_CHUNK)
    task_qpos = tables[(4 + _QUERY_CHUNK) * capacity : table_words].view(capacity, _QUERY_CHUNK)
    forced = _rows_per_cta_override()
    tensors = (
        meta,
        work_count,
        task_ends,
        row_ptr,
        q_idx,
        cu,
        dws,
        task_segments,
        segment_bins,
        task_meta,
        task_qrows,
        task_qpos,
        desc,
    )
    scalars = (Int32(num_chunks), Int32(num_kblocks), Int32(chunk), Int32(forced), Int32(num_sms))
    exe(*tensors, *scalars)
    return _MSABackwardTaskTables(
        task_meta, task_qrows, task_qpos, desc, capacity, _grid_launch_bound(capacity, forced, num_sms)
    )


def _host_task_tables(
    task_meta: torch.Tensor, task_qrows: torch.Tensor, task_qpos: torch.Tensor, num_sms: int
) -> _MSABackwardTaskTables:
    """Wrap eager tables with the exact CTA walk, uploading the descriptor once."""
    num_tasks = int(task_meta.shape[0])
    rows_per_cta = _select_rows_per_cta(num_tasks)
    num_full, tail, grid = _chunk_map(num_tasks, rows_per_cta, num_sms)
    desc = torch.tensor([num_tasks, rows_per_cta, num_full, tail, grid, 0, num_tasks, 0], dtype=torch.int32).to(
        task_meta.device
    )
    return _MSABackwardTaskTables(
        task_meta.contiguous(), task_qrows.contiguous(), task_qpos.contiguous(), desc, num_tasks, grid
    )


def _order_tasks_for_locality(
    task_meta: torch.Tensor,
    task_qrows: torch.Tensor,
    task_qpos: torch.Tensor,
    num_tokens: int,
    workspace_rows: int,
    chunk: int = _TASK_ORDER_CHUNK,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reorder task rows by (index_head, query_chunk, key_block, first query); no host sync."""
    num_tasks = int(task_meta.shape[0])
    if chunk <= 0 or num_tasks <= 1:
        return task_meta, task_qrows, task_qpos
    head = task_meta[:, 1].to(torch.int64)
    kblock = task_meta[:, 2].to(torch.int64)
    qmin = task_qrows[:, 0].to(torch.int64)
    num_chunks = (num_tokens + chunk - 1) // chunk
    num_kblocks = workspace_rows // _BLOCK_SIZE
    key = ((head * num_chunks + qmin // chunk) * num_kblocks + kblock) * chunk + qmin % chunk
    order = torch.argsort(key)
    return (
        task_meta.index_select(0, order),
        task_qrows.index_select(0, order),
        task_qpos.index_select(0, order),
    )


def task_build_storage(schedule: _MSABackwardSchedule, num_tokens: int, workspace_rows: int) -> tuple[int, int]:
    """Required scratch/table int32 words; the caller owns and may reuse both buffers."""
    if _TASK_BUILD == "torch":
        return 0, 0
    return _task_build_sizes(schedule, num_tokens, workspace_rows, _TASK_ORDER_CHUNK)[2:]


def build_backward_tasks(
    schedule: _MSABackwardSchedule,
    num_tokens: int,
    workspace_rows: int,
    *,
    num_sms: int,
    scratch: torch.Tensor | None = None,
    tables: torch.Tensor | None = None,
) -> _MSABackwardTaskTables:
    """Build ordered tables and the CTA walk, using Torch when fused is disabled or ineligible.

    The fused path leaves counts on device. Buffers are optional, contiguous int32 storage
    on the schedule device; undersized buffers are replaced. See ``task_build_storage``.
    """
    if _TASK_BUILD == "fused":
        built = _build_backward_tasks_fused(
            schedule,
            num_tokens,
            workspace_rows,
            _TASK_ORDER_CHUNK,
            num_sms=num_sms,
            scratch=scratch,
            tables=tables,
        )
        if built is not None:
            return built
    task_meta, task_qrows, task_qpos = _build_backward_tasks(schedule)
    return _host_task_tables(
        *_order_tasks_for_locality(task_meta, task_qrows, task_qpos, num_tokens, workspace_rows, _TASK_ORDER_CHUNK),
        num_sms,
    )
