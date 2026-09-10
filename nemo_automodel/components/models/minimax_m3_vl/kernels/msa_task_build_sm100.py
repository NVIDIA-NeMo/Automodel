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

"""Build locality-ordered backward task tables and their CTA walk on the device.

One JIT issues four launches: work scan, segment keys, bin scan, table scatter. Forward buckets have
ascending queries and contiguous (head, query window, key block) segments, so
``offsets[segment] + task - first_tasks[segment]`` reproduces the Torch reference order kept in
``tests/functional_tests/models/minimax_m3_vl/test_msa_task_build_sm100.py``. Noncontiguous segments
keep source order (descriptor flag 2). Capacity overflow disables the CTA walk (flag 1). Exact task
counts and the walk stay on the device; the host launches a grid bound and surplus CTAs take empty
intervals.
"""

from dataclasses import dataclass
from typing import Any

import torch

from nemo_automodel.components.models.minimax_m3_vl.kernels import require_cute_dsl, sm_capability
from nemo_automodel.components.models.minimax_m3_vl.kernels.msa_schedule import (
    _BLOCK_SIZE,
    _NUM_INDEX_HEADS,
    _QUERY_CHUNK,
    _ROWS_PER_CTA_LARGE,
    _ROWS_PER_CTA_SMALL,
    _ROWS_PER_CTA_SWITCH,
    _check_schedule,
    _grid_launch_bound,
    _MSABackwardSchedule,
)

# Bind the CuTe DSL only after proving it is importable, so a host without the msa extra sees
# UnavailableError here instead of ModuleNotFoundError from the imports below.
require_cute_dsl()

import cutlass
import cutlass.cute as cute
from cuda.bindings import driver as cuda
from cutlass import Int32
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream

THREADS = 256
SCAN_THREADS = 1024
# The two single-CTA scans stage one tile of int32 in SMEM at a time (32 KiB) so their
# global traffic is striped -- i.e. coalesced -- while the scan itself keeps the blocked
# partition it needs.  SCAN_PER is the constexpr slot count per thread: unrolling the
# striped passes is what puts more than one load per thread in flight, which is what the
# kernels are actually short of on a single SM.
SCAN_TILE = 8192
SCAN_PER = SCAN_TILE // SCAN_THREADS
DESC_WORDS = 8
MAX_BINS = 1 << 22
# `assign_segments` already resolves every task through `_decode_task`; `scatter_tables` used to
# repeat the identical binary search for the same task index.  Three int32 of scratch per task
# carry the rest of the decode across instead.  Only three, because `head` and `kblock` are
# already inside the segment id that scatter_tables reads anyway --
# `segment = (head * num_windows + query_window) * num_kblocks + kblock` inverts exactly -- and
# the write side is what limits this trade: at five words assign_segments gave back most of what
# scatter_tables saved.
DEC_WORDS = 3
DEC_VALID, DEC_EDGE, DEC_DOC = 0, 1, 2
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

# Locality window in queries: a wave of CTAs stays inside one, so its Q/dO/dQ rows are hit in L2.
# 512, not 2048: the window is the reuse distance of one CTA's own Q/dO/dQ fetch stream, and
# shrinking it raises the L2 read hit rate 73.0% -> 81.1% and cuts DRAM read 389.8 -> 259.5 MB at
# s4096.  Measured -1.27 / -1.22 / -0.41 / -1.01 % on the four M3 cases (interleaved forward and
# reverse passes, disjoint value ranges at s4096).  The curve has an interior optimum: 256 costs
# +6.78% at s8192 because the extra dK/dV segment flushes push Q/dO back out of L2, and 4096 costs
# +1.44% at s32768.  The optimum is an absolute query count -- it does not scale with T or with
# rows/CTA.  Bins are 4 * ceil(T/window) * (W/128), so a smaller window needs a larger MAX_BINS to
# hold the supported token count; the two constants move together.
_LOCALITY_WINDOW = 512

_COMPILE_CACHE: dict[tuple[Any, ...], Any] = {}


@dataclass(frozen=True, slots=True)
class _MSABackwardTaskTables:
    """Task tables in the locality order plus the CTA-walk descriptor the main kernel reads.

    The tables hold ``capacity`` rows; rows past the task count are never written nor read (the
    descriptor's exact count bounds every CTA interval). ``grid_launch >= desc[DESC_GRID_CTAS]``.
    """

    task_meta: torch.Tensor  # [capacity, 4] int32
    task_qrows: torch.Tensor  # [capacity, 8] int32
    task_qpos: torch.Tensor  # [capacity, 8] int32
    desc: torch.Tensor  # [8] int32 on the device (see DESC_*)
    grid_launch: int  # CTAs to launch for the main kernel


def _task_capacity(schedule: _MSABackwardSchedule) -> int:
    """Upper bound of the task count from the schedule shapes (disjoint work items)."""
    return _NUM_INDEX_HEADS * schedule.q_indices.shape[1] // _QUERY_CHUNK + schedule.scheduler_metadata.shape[0]


def _task_build_sizes(
    schedule: _MSABackwardSchedule, num_tokens: int, workspace_rows: int
) -> tuple[int, int, int, int]:
    """``(capacity, bins, scratch_words, table_words)`` of the build for this schedule (host shape math only)."""
    capacity = _task_capacity(schedule)
    num_windows = (num_tokens + _LOCALITY_WINDOW - 1) // _LOCALITY_WINDOW
    num_kblocks = workspace_rows // _BLOCK_SIZE
    bins = _NUM_INDEX_HEADS * num_windows * max(num_kblocks, 1)
    work_capacity = int(schedule.scheduler_metadata.shape[0])
    scratch_words = DESC_WORDS + work_capacity + capacity + 4 * bins + DEC_WORDS * capacity
    return capacity, bins, scratch_words, capacity * (4 + 2 * _QUERY_CHUNK)


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
        mTaskDecode: cute.Tensor,  # [capacity, DEC_WORDS] int32 scratch: assign_segments' decode
        mTaskMeta: cute.Tensor,  # [capacity, 4] int32 out
        mQRows: cute.Tensor,  # [capacity, 8] int32 out
        mQPos: cute.Tensor,  # [capacity, 8] int32 out
        mDesc: cute.Tensor,  # [8] int32 out
        num_windows: Int32,
        num_kblocks: Int32,
        locality_window: Int32,
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
            mTaskDecode,
            mDesc,
            num_windows,
            num_kblocks,
            locality_window,
        ).launch(grid=[cute.ceil_div(capacity, THREADS), 1, 1], block=[THREADS, 1, 1], stream=stream)
        self.scan_bins_kernel(mBins, mDesc).launch(grid=[1, 1, 1], block=[SCAN_THREADS, 1, 1], stream=stream)
        self.scatter_tables_kernel(
            mQIdx,
            mCuSeqlens,
            mDocStarts,
            mTaskSegments,
            mBins,
            mTaskDecode,
            mTaskMeta,
            mQRows,
            mQPos,
            mDesc,
            num_windows,
            num_kblocks,
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
        # A prefix sum wants a blocked partition (thread t owns a contiguous run), but reading a
        # blocked partition straight from global gives each warp 32 distinct sectors per load,
        # reads the column twice, and leaves one load per thread in flight -- and this is a single
        # CTA, i.e. one SM's share of L2.  Two coalesced shapes replace it:
        #   * up to one SMEM tile, scan a CTA-wide round at a time and carry between rounds.  Every
        #     access is striped, and each element is read once.
        #   * beyond that, stage a tile in SMEM through unrolled striped passes and keep the
        #     blocked scan inside SMEM, so one CTA scan is amortised over SCAN_TILE elements
        #     instead of over SCAN_THREADS.
        # Both carry the running total the same way, so the tables are bit-identical.
        total = Int32(0)
        if num_work <= Int32(SCAN_TILE):
            base = Int32(0)
            while base < num_work:
                w = base + tidx
                v = Int32(0)
                if w < num_work:
                    v = (mWorkMeta[w, 3] + Int32(_QUERY_CHUNK - 1)) // Int32(_QUERY_CHUNK)
                off, round_total = _cta_exclusive_scan(v, tidx, with_total=True)
                if w < num_work:
                    mTaskEnds[w] = total + off + v
                total = total + round_total
                base = base + Int32(SCAN_THREADS)
        else:
            sVal = cute.make_tensor(cute.arch.alloc_smem(Int32, SCAN_TILE), cute.make_layout(SCAN_TILE))
            c0 = Int32(0)
            while c0 < num_work:
                m = num_work - c0
                if m > Int32(SCAN_TILE):
                    m = Int32(SCAN_TILE)
                for u in cutlass.range_constexpr(SCAN_PER):
                    i = tidx + Int32(u * SCAN_THREADS)
                    if i < m:
                        sVal[i] = (mWorkMeta[c0 + i, 3] + Int32(_QUERY_CHUNK - 1)) // Int32(_QUERY_CHUNK)
                cute.arch.sync_threads()
                per = (m + Int32(SCAN_THREADS - 1)) // Int32(SCAN_THREADS)
                lo = tidx * per
                hi = lo + per
                if hi > m:
                    hi = m
                local = Int32(0)
                w = lo
                while w < hi:
                    local = local + sVal[w]
                    w = w + 1
                off, chunk_total = _cta_exclusive_scan(local, tidx, with_total=True)
                running = off + total
                w = lo
                while w < hi:
                    running = running + sVal[w]
                    sVal[w] = running
                    w = w + 1
                cute.arch.sync_threads()
                for u in cutlass.range_constexpr(SCAN_PER):
                    i = tidx + Int32(u * SCAN_THREADS)
                    if i < m:
                        mTaskEnds[c0 + i] = sVal[i]
                cute.arch.sync_threads()
                total = total + chunk_total
                c0 = c0 + Int32(SCAN_TILE)
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
        mTaskDecode: cute.Tensor,
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
            head, doc, kblock_local, valid, edge_start = _decode_task(mWorkMeta, mTaskEnds, mRowPtr, num_work, task)
            # slot 0 is always valid and queries ascend inside a bucket: the task's first query
            first_query = mCuSeqlens[doc] + mQIdx[head, edge_start]
            kblock = mDocStarts[doc] // Int32(_BLOCK_SIZE) + kblock_local
            segment = (head * num_windows + first_query // locality_window) * num_kblocks + kblock
            mTaskSegments[task] = segment
            # Hand the decode to scatter_tables rather than make it redo the binary search: the
            # search is ~log2(num_work) dependent global loads, and both kernels are indexed by
            # the same forward-order `task`.
            mTaskDecode[task, DEC_VALID] = valid
            mTaskDecode[task, DEC_EDGE] = edge_start
            mTaskDecode[task, DEC_DOC] = doc
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
        # Same two coalesced shapes as scan_work, and the same reason to need them: `bins` is
        # (index heads) * (tokens / window) * (workspace rows / 128), so it grows quadratically
        # with the sequence and this single CTA was scanning 65 536 entries at s32768 with one
        # outstanding load per thread.  The contiguity check reads first/last unconditionally so a
        # slot's three loads are independent -- empty bins carry INT32_MAX / -1 and the comparison
        # is discarded, so the flag is unchanged.
        bad = Int32(0)
        if n <= Int32(SCAN_TILE):
            carry = Int32(0)
            base = Int32(0)
            while base < n:
                b = base + tidx
                count = Int32(0)
                if b < n:
                    count = mBins[BIN_COUNTS, b]
                    last = mBins[BIN_LAST_TASKS, b]
                    first = mBins[BIN_FIRST_TASKS, b]
                    if count > Int32(0):
                        if last - first + Int32(1) != count:
                            bad = Int32(1)
                off, round_total = _cta_exclusive_scan(count, tidx, with_total=True)
                if b < n:
                    mBins[BIN_OFFSETS, b] = carry + off
                carry = carry + round_total
                base = base + Int32(SCAN_THREADS)
        else:
            sCount = cute.make_tensor(cute.arch.alloc_smem(Int32, SCAN_TILE), cute.make_layout(SCAN_TILE))
            carry = Int32(0)
            c0 = Int32(0)
            while c0 < n:
                m = n - c0
                if m > Int32(SCAN_TILE):
                    m = Int32(SCAN_TILE)
                for u in cutlass.range_constexpr(SCAN_PER):
                    i = tidx + Int32(u * SCAN_THREADS)
                    if i < m:
                        count = mBins[BIN_COUNTS, c0 + i]
                        last = mBins[BIN_LAST_TASKS, c0 + i]
                        first = mBins[BIN_FIRST_TASKS, c0 + i]
                        sCount[i] = count
                        if count > Int32(0):
                            if last - first + Int32(1) != count:
                                bad = Int32(1)
                cute.arch.sync_threads()
                per = (m + Int32(SCAN_THREADS - 1)) // Int32(SCAN_THREADS)
                lo = tidx * per
                hi = lo + per
                if hi > m:
                    hi = m
                local = Int32(0)
                b = lo
                while b < hi:
                    local = local + sCount[b]
                    b = b + 1
                off, chunk_total = _cta_exclusive_scan(local, tidx, with_total=True)
                running = off + carry
                b = lo
                while b < hi:
                    count = sCount[b]
                    sCount[b] = running
                    running = running + count
                    b = b + 1
                cute.arch.sync_threads()
                for u in cutlass.range_constexpr(SCAN_PER):
                    i = tidx + Int32(u * SCAN_THREADS)
                    if i < m:
                        mBins[BIN_OFFSETS, c0 + i] = sCount[i]
                cute.arch.sync_threads()
                carry = carry + chunk_total
                c0 = c0 + Int32(SCAN_TILE)
        if bad > Int32(0):
            cute.arch.atomic_or(mDesc.iterator + DESC_FLAGS, Int32(FLAG_NONCONTIGUOUS_SEGMENTS))

    @cute.kernel
    def scatter_tables_kernel(
        self,
        mQIdx: cute.Tensor,
        mCuSeqlens: cute.Tensor,
        mDocStarts: cute.Tensor,
        mTaskSegments: cute.Tensor,
        mBins: cute.Tensor,
        mTaskDecode: cute.Tensor,
        mTaskMeta: cute.Tensor,
        mQRows: cute.Tensor,
        mQPos: cute.Tensor,
        mDesc: cute.Tensor,
        num_windows: Int32,
        num_kblocks: Int32,
        num_sms: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        task = bidx * Int32(THREADS) + tidx
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
            # assign_segments already ran _decode_task for this same task index.  Read the three
            # words it left instead of repeating the binary search, which is ~log2(num_work)
            # DEPENDENT global loads -- the reason this kernel cost 6.4 us at s4096 to write 2 MB.
            valid = mTaskDecode[task, DEC_VALID]
            edge_start = mTaskDecode[task, DEC_EDGE]
            doc = mTaskDecode[task, DEC_DOC]
            # head and kblock come back out of the segment id, so they cost no scratch traffic
            kblock = segment - (segment // num_kblocks) * num_kblocks
            head = (segment // num_kblocks) // num_windows
            compact_start = mCuSeqlens[doc]
            workspace_start = mDocStarts[doc]
            mTaskMeta[output_row, 0] = Int32(0)
            mTaskMeta[output_row, 1] = head
            mTaskMeta[output_row, 2] = kblock
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
    key = ("minimax-m3-msa-task-build-sm100", sm_capability(device))
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
            tensor((n_cap, DEC_WORDS)),
            tensor((n_cap, 4), align=16),
            task_rows,
            task_rows,
            tensor((DESC_WORDS,)),
        )
        # num_windows, num_kblocks, locality_window, num_sms
        scalars = (Int32(0), Int32(0), Int32(0), Int32(0))
        _COMPILE_CACHE[key] = cute.compile(
            _MSATaskBuildSm100(),
            *tensors,
            *scalars,
            make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    return _COMPILE_CACHE[key]


def task_build_storage(schedule: _MSABackwardSchedule, num_tokens: int, workspace_rows: int) -> tuple[int, int]:
    """Required scratch/table int32 words; the caller owns and may reuse both buffers."""
    return _task_build_sizes(schedule, num_tokens, workspace_rows)[2:]


def build_backward_tasks(
    schedule: _MSABackwardSchedule,
    num_tokens: int,
    workspace_rows: int,
    *,
    num_sms: int,
    scratch: torch.Tensor | None = None,
    tables: torch.Tensor | None = None,
) -> _MSABackwardTaskTables:
    """Enqueue the four build launches without a host sync; return the tables and the CTA walk.

    Args:
        schedule: Forward-derived int32 schedule (see ``_MSABackwardSchedule``), all on one CUDA device.
        num_tokens: Compact token count ``T`` of the backward call.
        workspace_rows: Aligned K/V workspace length ``W``, a multiple of 128.
        num_sms: Streaming multiprocessors of the device; sizes the CTA walk.
        scratch: Optional int32 buffer of at least ``task_build_storage(...)[0]`` words on the schedule
            device, carved as ``descriptor | task ends | task segments | bins``; replaced when missing
            or undersized. The descriptor stays at offset zero to keep its 16-byte alignment.
        tables: Optional int32 buffer of at least ``task_build_storage(...)[1]`` words on the schedule
            device, carved as ``task_meta [capacity, 4] | task_qrows [capacity, 8] | task_qpos
            [capacity, 8]``; replaced when missing or undersized.

    Returns:
        The three task tables (views into ``tables``), the device descriptor (a view into
        ``scratch``) and the grid bound to launch the main kernel with.

    Raises:
        ValueError: If the locality bins exceed ``MAX_BINS``, which happens past roughly 250k tokens.
    """
    _check_schedule(schedule)
    capacity, bins, scratch_words, table_words = _task_build_sizes(schedule, num_tokens, workspace_rows)
    if bins > MAX_BINS:
        raise ValueError(
            f"MiniMax M3 MSA backward supports at most {MAX_BINS} locality bins per microbatch, got {bins} "
            f"for {num_tokens} tokens and {workspace_rows} workspace rows."
        )
    device = schedule.row_ptr.device
    num_windows = (num_tokens + _LOCALITY_WINDOW - 1) // _LOCALITY_WINDOW
    num_kblocks = workspace_rows // _BLOCK_SIZE
    meta = schedule.scheduler_metadata.contiguous()
    work_capacity = int(meta.shape[0])
    exe = _compile(device)
    row_ptr = schedule.row_ptr.contiguous()
    q_idx = schedule.q_indices.contiguous()
    cu = schedule.cu_seqlens.contiguous()
    dws = schedule.document_workspace_starts.contiguous()
    work_count = schedule.work_count.contiguous()
    if scratch is None or scratch.numel() < scratch_words:
        scratch = torch.empty(scratch_words, dtype=torch.int32, device=device)
    segments_start = DESC_WORDS + work_capacity
    bins_start = segments_start + capacity
    decode_start = bins_start + 4 * bins
    if tables is None or tables.numel() < table_words:
        tables = torch.empty(table_words, dtype=torch.int32, device=device)
    # One ``as_strided`` per region instead of a slice (+ a reshape for the 2-D ones): the carve
    # runs on every backward call and the slice-plus-view pair costs about twice a single
    # ``as_strided``.  Offsets are ABSOLUTE storage offsets, so the caller's own offset has to be
    # added -- ``scratch``/``tables`` are normally windows into the one per-call allocation.
    s0, t0 = scratch.storage_offset(), tables.storage_offset()
    desc = scratch.as_strided((DESC_WORDS,), (1,), s0)
    task_ends = scratch.as_strided((work_capacity,), (1,), s0 + DESC_WORDS)
    task_segments = scratch.as_strided((capacity,), (1,), s0 + segments_start)
    segment_bins = scratch.as_strided((4, bins), (bins, 1), s0 + bins_start)
    task_decode = scratch.as_strided((capacity, DEC_WORDS), (DEC_WORDS, 1), s0 + decode_start)
    task_meta = tables.as_strided((capacity, 4), (4, 1), t0)
    task_qrows = tables.as_strided((capacity, _QUERY_CHUNK), (_QUERY_CHUNK, 1), t0 + 4 * capacity)
    task_qpos = tables.as_strided((capacity, _QUERY_CHUNK), (_QUERY_CHUNK, 1), t0 + (4 + _QUERY_CHUNK) * capacity)
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
        task_decode,
        task_meta,
        task_qrows,
        task_qpos,
        desc,
    )
    scalars = (Int32(num_windows), Int32(num_kblocks), Int32(_LOCALITY_WINDOW), Int32(num_sms))
    exe(*tensors, *scalars)
    return _MSABackwardTaskTables(task_meta, task_qrows, task_qpos, desc, _grid_launch_bound(capacity, num_sms))
