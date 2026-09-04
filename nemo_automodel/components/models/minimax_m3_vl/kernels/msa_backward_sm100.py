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

"""MiniMax M3 main-attention backward kernel for MSA on SM100.

KV-parallel: each CTA walks task rows (8 queries x 16 main heads = one 128-row
tile) bucketed by (batch, index_head, key_block), with K/V TMA-resident per
bucket and Q/dO cp.async-gathered per tile. One mma warp issues all five tcgen05
GEMMs transposed (S^T, dP^T, dV, dK, dQ^T) over four 128-column TMEM
allocations; dV/dK accumulate per bucket segment and dQ per tile, all flushed
with fp32 atomics.
"""

import math
from typing import Any

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cuda.bindings import driver as cuda
from cutlass import Float32, Int32
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.nvgpu.common import OperandMajorMode
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.utils import LayoutEnum

from nemo_automodel.components.models.minimax_m3_vl.kernels.msa_backward_preprocess_sm100 import (
    _run_msa_backward_preprocess,
)
from nemo_automodel.components.models.minimax_m3_vl.kernels.msa_schedule import (
    _build_backward_tasks,
    _chunk_map,
    _MSABackwardSchedule,
    _select_rows_per_cta,
)

BLOCK_SIZE = 128
TILE_M = 128  # keys per tile (= one key block)
TILE_N = 128  # folded (q_slot, head) rows per tile
HEAD_DIM = 128
NUM_Q_HEADS = 64
NUM_KV_HEADS = 4
NUM_INDEX_HEADS = 4
MAIN_HEADS_PER_INDEX = NUM_Q_HEADS // NUM_INDEX_HEADS
QUERY_CHUNK = TILE_N // MAIN_HEADS_PER_INDEX

_COMPILE_CACHE = {}


class _MSABackwardSm100Kernel:
    arch = 100

    def __init__(self) -> None:
        self.main_per_index = MAIN_HEADS_PER_INDEX
        self.query_chunk = QUERY_CHUNK
        self.index_heads_per_kv = NUM_INDEX_HEADS // NUM_KV_HEADS
        # Always causal for M3: the non-causal predicate branches were folded away.
        self.causal = True

        self.acc_dtype = Float32
        self.q_stage = 2
        self.do_stage = 2
        # Double buffered: gather publishes tile t+1 while tile t is consumed.
        self.row_stage = 2

        # warp roles (16 warps / 512 threads)
        self.gather_warp_id = (0, 1, 2, 3)
        self.compute_warp_id = (4, 5, 6, 7)
        self.reduce_warp_id = (8, 9, 10, 11)
        self.mma_warp_id = 12
        self.load_warp_id = 13
        self.empty_warp_id = (14, 15)
        self.num_gather_warps = 4
        self.num_compute_warps = 4
        self.num_reduce_warps = 4
        self.threads_per_warp = 32
        self.threads_per_cta = 512

        # register budget: 4*32*(48 + R_compute + R_reduce) + 32*192 <= 512*128
        self.num_regs_gather = 48
        self.num_regs_compute = 184
        # Measured: reduce=184 deadlocks in setmaxnreg.inc on B200 though the sum fits.
        self.num_regs_reduce = 168
        self.num_regs_mma = 48
        self.num_regs_load = 48
        self.num_regs_empty = 24
        # compute processes S^T/dP^T in 32-column chunks to bound registers
        self.compute_chunk_cols = 32
        self.num_compute_chunks = TILE_N // self.compute_chunk_cols
        # mma hand-off stages of the 4-chunk loop: chunk sizes (2, 1, 1)
        self.chunk_stage_starts = (0, 2, 3)

        self.num_tmem_alloc_cols = 512
        self.tmem_S_offset = 0
        self.tmem_dPdQ_offset = 128
        self.tmem_dV_offset = 256
        self.tmem_dK_offset = 384

        # named barriers
        # TMEM allocator handoff: compute (allocator warp group) + reduce + mma
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.threads_per_warp * (self.num_compute_warps + self.num_reduce_warps + 1),
        )
        self.compute_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2, num_threads=self.num_compute_warps * self.threads_per_warp
        )
        self.gather_sync_barrier = pipeline.NamedBarrier(
            barrier_id=3, num_threads=self.num_gather_warps * self.threads_per_warp
        )
        self.reduce_sync_barrier = pipeline.NamedBarrier(
            barrier_id=4, num_threads=self.num_reduce_warps * self.threads_per_warp
        )
        # Orders reduce's dQ^T T2R (cols 128..256) before mma overwrites them with dP^T.
        self.t2r_dQ_done_barrier = pipeline.NamedBarrier(
            barrier_id=5, num_threads=(self.num_reduce_warps + 1) * self.threads_per_warp
        )
        # TMEM dealloc must follow every T2R reader's final read.
        self.tmem_dealloc_barrier = pipeline.NamedBarrier(
            barrier_id=6,
            num_threads=(self.num_compute_warps + self.num_reduce_warps) * self.threads_per_warp,
        )
        self.buffer_align_bytes = 1024

        # pipeline stage counts
        self.load_mma_KV_stage = 1
        self.gather_mma_QdO_stage = self.q_stage
        self.gather_row_stage = self.row_stage
        self.mma_compute_S_stage = 1
        self.mma_compute_dP_stage = 1
        self.mma_reduce_dQ_stage = 1
        # per-chunk hand-off, so G3/G4 trail the softmax loop instead of waiting.
        self.compute_mma_chunk_stage = len(self.chunk_stage_starts)
        self.mma_reduce_dKV_stage = 1

    # ---- host-side entry ----
    @cute.jit
    def __call__(
        self,
        # Head-major views built by _run_msa_backward; T, W, num_tasks are dynamic.
        mQ: cute.Tensor,  # [1, Hq, T, D] view of [T, Hq, D]
        mK: cute.Tensor,  # [1, Hkv, W, D] view of [W, Hkv, D]
        mV: cute.Tensor,  # [1, Hkv, W, D] view of [W, Hkv, D]
        mdO: cute.Tensor,  # [1, Hq, T, D] view of [T, Hq, D]
        mLSE: cute.Tensor,  # [1, Hq, T] fp32 view of [T, Hq]
        mDelta: cute.Tensor,  # [1, Hq, T] fp32 view of [T, Hq]
        mTaskMeta: cute.Tensor,  # [num_tasks, 4] int32
        mTaskQRows: cute.Tensor,  # [num_tasks, 8] int32, compact Q/dO/dQ rows
        mTaskQPos: cute.Tensor,  # [num_tasks, 8] int32, aligned causal positions
        mdQ: cute.Tensor,  # [1, Hq, T, D] fp32 view of [T, Hq, D]
        mdK: cute.Tensor,  # [1, Hkv, W, D] fp32 view of [W, Hkv, D]
        mdV: cute.Tensor,  # [1, Hkv, W, D] fp32 view of [W, Hkv, D]
        num_task_rows: Int32,
        rows_per_cta: Int32,
        n_full_ctas: Int32,
        tail_rows: Int32,
        grid_ctas: Int32,
        softmax_scale: Float32,
        # Keep the stream last: with --enable-tvm-ffi it is the TVM FFI environment stream.
        stream: cuda.CUstream,
    ):
        if cutlass.const_expr(
            not (mQ.element_type == mK.element_type == mV.element_type == mdO.element_type == cutlass.BFloat16)
        ):
            raise TypeError("q/k/v/dO must be BF16")
        self.element_dtype = mQ.element_type

        cta_group = tcgen05.CtaGroup.ONE
        dt = self.element_dtype
        f32 = self.acc_dtype
        tiler = (TILE_M, TILE_N, HEAD_DIM)  # (128,128,128); all five GEMMs share it

        # G1 S^T = K . Q^T
        mma_S = sm100_utils.make_trivial_tiled_mma(
            dt, dt, OperandMajorMode.K, OperandMajorMode.K, f32, cta_group, tiler[:2]
        )
        # G2 dP^T = V . dO^T
        mma_dP = sm100_utils.make_trivial_tiled_mma(
            dt, dt, OperandMajorMode.K, OperandMajorMode.K, f32, cta_group, tiler[:2]
        )
        # G3 dV += P^T . dO  (M=key, N=d, K=q); A = P^T in TMEM, B = dO MN-major
        mma_dV = sm100_utils.make_trivial_tiled_mma(
            dt,
            dt,
            OperandMajorMode.K,
            OperandMajorMode.MN,
            f32,
            cta_group,
            tiler[:2],
            a_source=tcgen05.OperandSource.TMEM,
        )
        # G4 dK += dS^T . Q   (M=key, N=d, K=q); A = dS^T in TMEM, B = Q MN-major
        mma_dK = sm100_utils.make_trivial_tiled_mma(
            dt,
            dt,
            OperandMajorMode.K,
            OperandMajorMode.MN,
            f32,
            cta_group,
            tiler[:2],
            a_source=tcgen05.OperandSource.TMEM,
        )
        # G5 dQ^T = K^T . dS^T  (M=d, N=q, K=key)
        mma_dQ = sm100_utils.make_trivial_tiled_mma(
            dt, dt, OperandMajorMode.MN, OperandMajorMode.MN, f32, cta_group, tiler[:2]
        )
        cluster_layout_vmnk = cute.make_layout(((1), (1, 1, 1)), stride=((0), (0, 0, 0)))

        # SMEM layouts: every big operand buffer is a 128x128 bf16 tile.
        sK_layout = sm100_utils.make_smem_layout_a(mma_S, tiler, dt, self.load_mma_KV_stage)
        sQ_layout = sm100_utils.make_smem_layout_b(mma_S, tiler, dt, self.q_stage)
        sV_layout = sm100_utils.make_smem_layout_a(mma_dP, tiler, dt, self.load_mma_KV_stage)
        sdO_layout = sm100_utils.make_smem_layout_b(mma_dP, tiler, dt, self.do_stage)
        # P / dS shared buffer: canonical B operand layout of G3/G4 (K-major).
        sPdS_layout = sm100_utils.make_smem_layout_b(mma_dV, tiler, dt, 1)
        # store-side view for the compute warps' StMatrix/vector stores
        sPdS_store_layout = sm100_utils.make_smem_layout_epi(dt, LayoutEnum.ROW_MAJOR, (TILE_M, TILE_N), 1)
        # MN-major B views of dO / Q for G3 / G4 alias the K-major bytes of G1 / G2.
        sdOb_layout = sm100_utils.make_smem_layout_b(mma_dV, tiler, dt, self.do_stage)
        sQb_layout = sm100_utils.make_smem_layout_b(mma_dK, tiler, dt, self.q_stage)
        # TMEM-resident A operand view for P^T / dS^T (bf16 over the acc columns).
        tP_layout = cute.slice_(sm100_utils.make_smem_layout_a(mma_dV, tiler, dt, 1), (None, None, None, 0))
        sKt_layout = sm100_utils.make_smem_layout_a(mma_dQ, tiler, dt, self.load_mma_KV_stage)
        # MN-major B view of dS for G5
        sPdSn_layout = sm100_utils.make_smem_layout_b(mma_dQ, tiler, dt, 1)

        sLSE_layout = cute.make_layout((TILE_N, self.row_stage))
        sDelta_layout = cute.make_layout((TILE_N, self.row_stage))
        sQRows_layout = cute.make_layout((self.query_chunk, self.row_stage))
        sQPos_layout = cute.make_layout((self.query_chunk, self.row_stage))

        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(cta_group)

        # gmem as (S, D, (h, B)) so local_tile picks the key block for (kv_head, batch).
        mK_v = cute.make_tensor(
            mK.iterator,
            cute.make_layout(
                (mK.shape[2], mK.shape[3], (mK.shape[1], mK.shape[0])),
                stride=(mK.stride[2], mK.stride[3], (mK.stride[1], mK.stride[0])),
            ),
        )
        mV_v = cute.make_tensor(
            mV.iterator,
            cute.make_layout(
                (mV.shape[2], mV.shape[3], (mV.shape[1], mV.shape[0])),
                stride=(mV.stride[2], mV.stride[3], (mV.stride[1], mV.stride[0])),
            ),
        )

        sK_layout_single = cute.select(sK_layout, mode=[0, 1, 2])
        tma_atom_K, tma_tensor_K = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op, mK_v, sK_layout_single, tiler, mma_S, cluster_layout_vmnk.shape
        )
        sV_layout_single = cute.select(sV_layout, mode=[0, 1, 2])
        tma_atom_V, tma_tensor_V = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op, mV_v, sV_layout_single, tiler, mma_dP, cluster_layout_vmnk.shape
        )

        self.tma_copy_KV_bytes = cute.size_in_bytes(dt, sK_layout_single) + cute.size_in_bytes(dt, sV_layout_single)

        _max_smem_bytes = 232448

        @cute.struct
        class SharedStorage:
            load_mma_KV_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.load_mma_KV_stage * 2]
            gather_mma_QdO_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.gather_mma_QdO_stage * 2]
            gather_row_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.gather_row_stage * 2]
            mma_compute_S_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.mma_compute_S_stage * 2]
            mma_compute_dP_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.mma_compute_dP_stage * 2]
            mma_reduce_dQ_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.mma_reduce_dQ_stage * 2]
            compute_mma_chunk_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.compute_mma_chunk_stage * 2]
            mma_reduce_dKV_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.mma_reduce_dKV_stage * 2]
            tmem_holding_buf: cutlass.Int32
            sLSE: cute.struct.Align[cute.struct.MemRange[self.acc_dtype, cute.cosize(sLSE_layout)], 128]
            sDelta: cute.struct.Align[cute.struct.MemRange[self.acc_dtype, cute.cosize(sDelta_layout)], 128]
            sQRows: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, cute.cosize(sQRows_layout)], 128]
            sQPos: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, cute.cosize(sQPos_layout)], 128]
            sK: cute.struct.Align[
                cute.struct.MemRange[self.element_dtype, cute.cosize(sK_layout)],
                self.buffer_align_bytes,
            ]
            sV: cute.struct.Align[
                cute.struct.MemRange[self.element_dtype, cute.cosize(sV_layout)],
                self.buffer_align_bytes,
            ]
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.element_dtype, cute.cosize(sQ_layout)],
                self.buffer_align_bytes,
            ]
            sdO: cute.struct.Align[
                cute.struct.MemRange[self.element_dtype, cute.cosize(sdO_layout)],
                self.buffer_align_bytes,
            ]
            sPdS: cute.struct.Align[
                cute.struct.MemRange[self.element_dtype, cute.cosize(sPdS_layout)],
                self.buffer_align_bytes,
            ]

        assert SharedStorage.size_in_bytes() <= _max_smem_bytes, (
            f"SharedStorage {SharedStorage.size_in_bytes()} bytes exceeds {_max_smem_bytes}"
        )
        self.shared_storage = SharedStorage

        LOG2_E = Float32(math.log2(math.e))

        self.kernel(
            mma_S,
            mma_dP,
            mma_dV,
            mma_dK,
            mma_dQ,
            tma_atom_K,
            tma_tensor_K,
            tma_atom_V,
            tma_tensor_V,
            mQ,
            mdO,
            mLSE,
            mDelta,
            mTaskMeta,
            mTaskQRows,
            mTaskQPos,
            mdQ,
            mdK,
            mdV,
            num_task_rows,
            rows_per_cta,
            n_full_ctas,
            tail_rows,
            softmax_scale * LOG2_E,
            LOG2_E,
            sK_layout,
            sV_layout,
            sQ_layout,
            sdO_layout,
            sPdS_layout,
            sPdS_store_layout,
            sdOb_layout,
            sQb_layout,
            sKt_layout,
            sPdSn_layout,
            tP_layout,
            sLSE_layout,
            sDelta_layout,
            sQRows_layout,
            sQPos_layout,
        ).launch(
            grid=[grid_ctas, 1, 1],
            block=[self.threads_per_cta, 1, 1],
            cluster=[1, 1, 1],
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

    # ---- helpers shared by device functions ----
    @cute.jit
    def _task_fields(self, mTaskMeta: cute.Tensor, row: Int32):
        b = mTaskMeta[row, 0]
        index_head = mTaskMeta[row, 1]
        kb = mTaskMeta[row, 2]
        valid = mTaskMeta[row, 3]
        return b, index_head, kb, valid

    @cute.jit
    def _same_bucket(self, mTaskMeta: cute.Tensor, row_a: Int32, row_b: Int32) -> cutlass.Boolean:
        same = cutlass.Boolean(True)
        for c in cutlass.range_constexpr(3):
            same = same & (mTaskMeta[row_a, c] == mTaskMeta[row_b, c])
        return same

    @cute.jit
    def get_tmem_tensors(
        self,
        mma_S: cute.TiledMma,
        mma_dP: cute.TiledMma,
        mma_dQ: cute.TiledMma,
        mma_dV: cute.TiledMma,
        mma_dK: cute.TiledMma,
        tmem_ptr_base: cute.Pointer,
    ):
        tStS_shape = mma_S.partition_shape_C((TILE_M, TILE_N))
        tStS = mma_S.make_fragment_C(tStS_shape)
        tStS = cute.make_tensor(tmem_ptr_base + self.tmem_S_offset, tStS.layout)

        tdPtdP_shape = mma_dP.partition_shape_C((TILE_M, TILE_N))
        tdPtdP = mma_dP.make_fragment_C(tdPtdP_shape)
        tdPtdP = cute.make_tensor(tmem_ptr_base + self.tmem_dPdQ_offset, tdPtdP.layout)

        tdQtdQ_shape = mma_dQ.partition_shape_C((TILE_M, TILE_N))
        tdQtdQ = mma_dQ.make_fragment_C(tdQtdQ_shape)
        tdQtdQ = cute.make_tensor(tmem_ptr_base + self.tmem_dPdQ_offset, tdQtdQ.layout)

        tdVtdV_shape = mma_dV.partition_shape_C((TILE_M, TILE_N))
        tdVtdV = mma_dV.make_fragment_C(tdVtdV_shape)
        tdVtdV = cute.make_tensor(tmem_ptr_base + self.tmem_dV_offset, tdVtdV.layout)

        tdKtdK_shape = mma_dK.partition_shape_C((TILE_M, TILE_N))
        tdKtdK = mma_dK.make_fragment_C(tdKtdK_shape)
        tdKtdK = cute.make_tensor(tmem_ptr_base + self.tmem_dK_offset, tdKtdK.layout)

        return tStS, tdPtdP, tdQtdQ, tdVtdV, tdKtdK

    # ---- device kernel ----
    @cute.kernel
    def kernel(
        self,
        mma_S: cute.TiledMma,
        mma_dP: cute.TiledMma,
        mma_dV: cute.TiledMma,
        mma_dK: cute.TiledMma,
        mma_dQ: cute.TiledMma,
        tma_atom_K: cute.CopyAtom,
        tma_tensor_K: cute.Tensor,
        tma_atom_V: cute.CopyAtom,
        tma_tensor_V: cute.Tensor,
        mQ: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mDelta: cute.Tensor,
        mTaskMeta: cute.Tensor,
        mTaskQRows: cute.Tensor,
        mTaskQPos: cute.Tensor,
        mdQ: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        num_task_rows: Int32,
        rows_per_cta: Int32,
        n_full_ctas: Int32,
        tail_rows: Int32,
        scale_log2e: Float32,
        log2_e: Float32,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sQ_layout: cute.ComposedLayout,
        sdO_layout: cute.ComposedLayout,
        sPdS_layout: cute.ComposedLayout,
        sPdS_store_layout: cute.ComposedLayout,
        sdOb_layout: cute.ComposedLayout,
        sQb_layout: cute.ComposedLayout,
        sKt_layout: cute.ComposedLayout,
        sPdSn_layout: cute.ComposedLayout,
        tP_layout: cute.ComposedLayout,
        sLSE_layout: cute.Layout,
        sDelta_layout: cute.Layout,
        sQRows_layout: cute.Layout,
        sQPos_layout: cute.Layout,
    ):
        bidx, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        if warp_idx == self.load_warp_id:
            cpasync.prefetch_descriptor(tma_atom_K)
            cpasync.prefetch_descriptor(tma_atom_V)

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # CTA -> contiguous task-row interval (final wave split into tail_rows
        # pieces). A split bucket stays correct: each CTA reloads K/V and atomically
        # accumulates dK/dV.
        row_lo = bidx * rows_per_cta
        row_hi = cutlass.min(row_lo + rows_per_cta, num_task_rows)
        if bidx >= n_full_ctas:
            row_lo = n_full_ctas * rows_per_cta + (bidx - n_full_ctas) * tail_rows
            row_hi = cutlass.min(row_lo + tail_rows, num_task_rows)

        load_mma_KV_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.load_mma_KV_mbar_ptr.data_ptr(),
            num_stages=self.load_mma_KV_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=self.tma_copy_KV_bytes,
            defer_sync=True,
        )
        gather_mma_QdO_pipeline = pipeline.PipelineAsyncUmma.create(
            barrier_storage=storage.gather_mma_QdO_mbar_ptr.data_ptr(),
            num_stages=self.gather_mma_QdO_stage,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.num_gather_warps * self.threads_per_warp
            ),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            defer_sync=True,
        )
        gather_row_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.gather_row_mbar_ptr.data_ptr(),
            num_stages=self.gather_row_stage,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.num_gather_warps * self.threads_per_warp
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                (self.num_compute_warps + self.num_reduce_warps) * self.threads_per_warp,
            ),
            defer_sync=True,
        )
        mma_compute_S_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.mma_compute_S_mbar_ptr.data_ptr(),
            num_stages=self.mma_compute_S_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.num_compute_warps * self.threads_per_warp
            ),
            defer_sync=True,
        )
        mma_compute_dP_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.mma_compute_dP_mbar_ptr.data_ptr(),
            num_stages=self.mma_compute_dP_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.num_compute_warps * self.threads_per_warp
            ),
            defer_sync=True,
        )
        mma_reduce_dQ_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.mma_reduce_dQ_mbar_ptr.data_ptr(),
            num_stages=self.mma_reduce_dQ_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.num_reduce_warps * self.threads_per_warp
            ),
            defer_sync=True,
        )
        compute_mma_chunk_pipeline = pipeline.PipelineAsyncUmma.create(
            barrier_storage=storage.compute_mma_chunk_mbar_ptr.data_ptr(),
            num_stages=self.compute_mma_chunk_stage,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.num_compute_warps * self.threads_per_warp
            ),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            defer_sync=True,
        )
        mma_reduce_dKV_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.mma_reduce_dKV_mbar_ptr.data_ptr(),
            num_stages=self.mma_reduce_dKV_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.num_reduce_warps * self.threads_per_warp
            ),
            defer_sync=True,
        )

        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.compute_warp_id[0],
        )

        pipeline.pipeline_init_arrive(is_relaxed=True)

        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sdO = storage.sdO.get_tensor(sdO_layout.outer, swizzle=sdO_layout.inner)
        sPdS = storage.sPdS.get_tensor(sPdS_layout.outer, swizzle=sPdS_layout.inner)

        sPdS_store = cute.make_tensor(cute.recast_ptr(sPdS.iterator, sPdS_store_layout.inner), sPdS_store_layout.outer)
        sdOb = cute.make_tensor(cute.recast_ptr(sdO.iterator, sdOb_layout.inner), sdOb_layout.outer)
        sQb = cute.make_tensor(cute.recast_ptr(sQ.iterator, sQb_layout.inner), sQb_layout.outer)
        sKt = cute.make_tensor(cute.recast_ptr(sK.iterator, sKt_layout.inner), sKt_layout.outer)
        sPdSn = cute.make_tensor(cute.recast_ptr(sPdS.iterator, sPdSn_layout.inner), sPdSn_layout.outer)

        sLSE = storage.sLSE.get_tensor(sLSE_layout)
        sDelta = storage.sDelta.get_tensor(sDelta_layout)
        sQRows = storage.sQRows.get_tensor(sQRows_layout)
        sQPos = storage.sQPos.get_tensor(sQPos_layout)

        pipeline.pipeline_init_wait()

        if warp_idx == self.load_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_load)
            self.load_kv(
                mma_S,
                mma_dP,
                tma_atom_K,
                tma_tensor_K,
                tma_atom_V,
                tma_tensor_V,
                sK,
                sV,
                mTaskMeta,
                row_lo,
                row_hi,
                load_mma_KV_pipeline,
            )

        elif warp_idx in self.gather_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_gather)
            self.gather_rows(
                mQ,
                mdO,
                mLSE,
                mDelta,
                mTaskMeta,
                mTaskQRows,
                mTaskQPos,
                sQ,
                sdO,
                sLSE,
                sDelta,
                sQRows,
                sQPos,
                row_lo,
                row_hi,
                log2_e,
                gather_mma_QdO_pipeline,
                gather_row_pipeline,
            )

        elif warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_mma)
            tmem.wait_for_alloc()
            tmem_ptr_base = tmem.retrieve_ptr(self.acc_dtype)
            tStS, tdPtdP, tdQtdQ, tdVtdV, tdKtdK = self.get_tmem_tensors(
                mma_S, mma_dP, mma_dQ, mma_dV, mma_dK, tmem_ptr_base
            )
            self.mma(
                mma_S,
                mma_dP,
                mma_dV,
                mma_dK,
                mma_dQ,
                sK,
                sV,
                sQ,
                sdO,
                sPdS,
                sdOb,
                sQb,
                sKt,
                sPdSn,
                tStS,
                tdPtdP,
                tdQtdQ,
                tdVtdV,
                tdKtdK,
                tmem_ptr_base,
                tP_layout,
                mTaskMeta,
                row_lo,
                row_hi,
                (
                    load_mma_KV_pipeline,
                    gather_mma_QdO_pipeline,
                    mma_compute_S_pipeline,
                    mma_compute_dP_pipeline,
                    mma_reduce_dQ_pipeline,
                    compute_mma_chunk_pipeline,
                    mma_reduce_dKV_pipeline,
                ),
            )

        elif warp_idx in self.compute_warp_id:
            cute.arch.setmaxregister_increase(self.num_regs_compute)
            if warp_idx == self.compute_warp_id[0]:
                tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr_base = tmem.retrieve_ptr(self.acc_dtype)
            tStS, tdPtdP, tdQtdQ, tdVtdV, tdKtdK = self.get_tmem_tensors(
                mma_S, mma_dP, mma_dQ, mma_dV, mma_dK, tmem_ptr_base
            )
            self.compute(
                tStS,
                tdPtdP,
                sPdS_store,
                sLSE,
                sDelta,
                sQPos,
                mTaskMeta,
                row_lo,
                row_hi,
                scale_log2e,
                log2_e,
                (
                    mma_compute_S_pipeline,
                    mma_compute_dP_pipeline,
                    compute_mma_chunk_pipeline,
                    gather_row_pipeline,
                ),
            )
            if warp_idx == self.compute_warp_id[0]:
                self.tmem_dealloc_barrier.arrive_and_wait()
                cute.arch.dealloc_tmem(tmem_ptr_base, self.num_tmem_alloc_cols)
            else:
                self.tmem_dealloc_barrier.arrive()

        elif warp_idx in self.reduce_warp_id:
            cute.arch.setmaxregister_increase(self.num_regs_reduce)
            tmem.wait_for_alloc()
            tmem_ptr_base = tmem.retrieve_ptr(self.acc_dtype)
            tStS, tdPtdP, tdQtdQ, tdVtdV, tdKtdK = self.get_tmem_tensors(
                mma_S, mma_dP, mma_dQ, mma_dV, mma_dK, tmem_ptr_base
            )
            self.reduce(
                tdQtdQ,
                tdVtdV,
                tdKtdK,
                mdQ,
                mdK,
                mdV,
                sQRows,
                mTaskMeta,
                row_lo,
                row_hi,
                (
                    mma_reduce_dQ_pipeline,
                    mma_reduce_dKV_pipeline,
                    gather_row_pipeline,
                ),
            )
            self.tmem_dealloc_barrier.arrive()

        else:
            cute.arch.setmaxregister_decrease(self.num_regs_empty)

    # ---- load warp: TMA K/V per bucket segment ----
    @cute.jit
    def load_kv(
        self,
        mma_S: cute.TiledMma,
        mma_dP: cute.TiledMma,
        tma_atom_K: cute.CopyAtom,
        tma_tensor_K: cute.Tensor,
        tma_atom_V: cute.CopyAtom,
        tma_tensor_V: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        mTaskMeta: cute.Tensor,
        row_lo: Int32,
        row_hi: Int32,
        load_mma_KV_pipeline,
    ):
        producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.load_mma_KV_stage)

        tiler = (TILE_M, TILE_N, HEAD_DIM)
        thr_mma_S = mma_S.get_slice(0)
        thr_mma_dP = mma_dP.get_slice(0)

        row = row_lo
        while row < row_hi:
            is_seg_start = cutlass.Boolean(row == row_lo)
            if row > row_lo:
                is_seg_start = ~self._same_bucket(mTaskMeta, row, row - 1)
            if is_seg_start:
                b, index_head, kb, _valid = self._task_fields(mTaskMeta, row)
                kv_head = index_head // Int32(self.index_heads_per_kv)
                # (bM, bK, RestM, RestK) at the dynamic (kv_head, batch)
                gK = cute.local_tile(
                    tma_tensor_K,
                    cute.select(tiler, mode=[0, 2]),
                    (None, None, (kv_head, b)),
                )
                gV = cute.local_tile(
                    tma_tensor_V,
                    cute.select(tiler, mode=[0, 2]),
                    (None, None, (kv_head, b)),
                )
                tKgK = thr_mma_S.partition_A(gK)
                tKsK, tKgK_mkl = cpasync.tma_partition(
                    tma_atom_K,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sK, 0, 3),
                    cute.group_modes(tKgK, 0, 3),
                )
                tVgV = thr_mma_dP.partition_A(gV)
                tVsV, tVgV_mkl = cpasync.tma_partition(
                    tma_atom_V,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sV, 0, 3),
                    cute.group_modes(tVgV, 0, 3),
                )
                load_mma_KV_pipeline.producer_acquire(producer_state)
                tma_barrier = load_mma_KV_pipeline.producer_get_barrier(producer_state)
                cute.copy(
                    tma_atom_K,
                    tKgK_mkl[None, kb, 0],
                    tKsK[None, producer_state.index],
                    tma_bar_ptr=tma_barrier,
                )
                cute.copy(
                    tma_atom_V,
                    tVgV_mkl[None, kb, 0],
                    tVsV[None, producer_state.index],
                    tma_bar_ptr=tma_barrier,
                )
                producer_state.advance()
            row += 1

    # ---- gather warps: Q/dO row gather + per-row scalars, per tile ----
    @cute.jit
    def gather_rows(
        self,
        mQ: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mDelta: cute.Tensor,
        mTaskMeta: cute.Tensor,
        mTaskQRows: cute.Tensor,
        mTaskQPos: cute.Tensor,
        sQ: cute.Tensor,
        sdO: cute.Tensor,
        sLSE: cute.Tensor,
        sDelta: cute.Tensor,
        sQRows: cute.Tensor,
        sQPos: cute.Tensor,
        row_lo: Int32,
        row_hi: Int32,
        log2_e: Float32,
        gather_mma_QdO_pipeline,
        gather_row_pipeline,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        local_tidx = tidx % self.threads_per_warp
        local_warp = tidx // self.threads_per_warp  # 0..3 (gather warps are 0..3)

        async_copy_atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            self.element_dtype,
            num_bits_per_copy=128,
        )
        thr_layout = cute.make_layout((16,))
        val_layout = cute.make_layout((8,))
        async_tiled_copy = cute.make_tiled_copy_tv(async_copy_atom, thr_layout, val_layout)
        async_thr_copy = async_tiled_copy.get_slice(local_tidx % 16)

        qdo_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.gather_mma_QdO_stage)
        row_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.gather_row_stage)

        mpp = Int32(self.main_per_index)
        row = row_lo
        while row < row_hi:
            b, index_head, _kb, valid = self._task_fields(mTaskMeta, row)

            # Lane l < 2 holds the compact row for this warp's q slot.
            qrow_reg = Int32(0)
            if local_tidx < Int32(self.query_chunk // self.num_gather_warps):
                qrow_reg = mTaskQRows[row, local_warp * Int32(self.query_chunk // self.num_gather_warps) + local_tidx]

            gather_mma_QdO_pipeline.producer_acquire(qdo_state)
            gather_row_pipeline.producer_acquire(row_state)

            sQ_slice = sQ[(None, None), 0, (None, None), qdo_state.index % self.q_stage]
            sQ_slice = cute.composition(sQ_slice, cute.make_layout((TILE_N, HEAD_DIM)))
            sdO_slice = sdO[(None, None), 0, (None, None), qdo_state.index % self.do_stage]
            sdO_slice = cute.composition(sdO_slice, cute.make_layout((TILE_N, HEAD_DIM)))

            # each warp owns rows [32*warp, +32); lanes 0-15 / 16-31 take two rows
            for i in cutlass.range_constexpr(16):
                r = local_warp * 32 + i * 2 + local_tidx // 16
                q_slot = Int32(r) // mpp
                head_off = Int32(r) - q_slot * mpp
                slot_in_warp = q_slot - local_warp * Int32(self.query_chunk // self.num_gather_warps)
                q_row = cute.arch.shuffle_sync(qrow_reg, slot_in_warp)
                head = index_head * mpp + head_off

                tile_sQ = sQ_slice[r, None]
                tile_sdO = sdO_slice[r, None]
                if q_slot < valid:
                    gQ_row = mQ[b, head, q_row, None]
                    tQs = async_thr_copy.partition_D(tile_sQ)
                    tQg = async_thr_copy.partition_S(gQ_row)
                    cute.copy(async_copy_atom, tQg, tQs)
                    gdO_row = mdO[b, head, q_row, None]
                    tOs = async_thr_copy.partition_D(tile_sdO)
                    tOg = async_thr_copy.partition_S(gdO_row)
                    cute.copy(async_copy_atom, tOg, tOs)
                else:
                    zfill = cute.flat_divide(tile_sQ, (8,))
                    zfill[None, local_tidx % 16].fill(0.0)
                    zfillo = cute.flat_divide(tile_sdO, (8,))
                    zfillo[None, local_tidx % 16].fill(0.0)

            # per-row scalars: the row pipeline's mbarrier release orders them
            r = local_warp * 32 + local_tidx
            q_slot = Int32(r) // mpp
            head_off = Int32(r) - q_slot * mpp
            # sLSE holds the prefolded -lse*log2e; invalid slots carry -inf, so
            # exp2 is exactly +0 there and P/dS vanish without a validity compare.
            if q_slot < valid:
                q_row2 = mTaskQRows[row, q_slot]
                head2 = index_head * mpp + head_off
                sLSE[r, row_state.index] = Float32(0.0) - mLSE[b, head2, q_row2] * log2_e
                sDelta[r, row_state.index] = mDelta[b, head2, q_row2]
            else:
                sLSE[r, row_state.index] = Float32(float("-inf"))
                sDelta[r, row_state.index] = Float32(0.0)
            if local_warp == 0 and local_tidx < Int32(self.query_chunk):
                qrow = Int32(-1)
                qpos = Int32(-1)
                if local_tidx < valid:
                    qrow = mTaskQRows[row, local_tidx]
                    qpos = mTaskQPos[row, local_tidx]
                sQRows[local_tidx, row_state.index] = qrow
                sQPos[local_tidx, row_state.index] = qpos

            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.fence_view_async_shared()
            self.gather_sync_barrier.arrive_and_wait()
            gather_mma_QdO_pipeline.producer_commit(qdo_state)
            qdo_state.advance()
            gather_row_pipeline.producer_commit(row_state)
            row_state.advance()

            row += 1

    # ---- mma warp ----
    @cute.jit
    def mma(
        self,
        mma_S: cute.TiledMma,
        mma_dP: cute.TiledMma,
        mma_dV: cute.TiledMma,
        mma_dK: cute.TiledMma,
        mma_dQ: cute.TiledMma,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sQ: cute.Tensor,
        sdO: cute.Tensor,
        sPdS: cute.Tensor,
        sdOb: cute.Tensor,
        sQb: cute.Tensor,
        sKt: cute.Tensor,
        sPdSn: cute.Tensor,
        tStS: cute.Tensor,
        tdPtdP: cute.Tensor,
        tdQtdQ: cute.Tensor,
        tdVtdV: cute.Tensor,
        tdKtdK: cute.Tensor,
        tmem_ptr_base: cute.Pointer,
        tP_layout: cute.ComposedLayout,
        mTaskMeta: cute.Tensor,
        row_lo: Int32,
        row_hi: Int32,
        pipelines,
    ):
        (
            load_mma_KV_pipeline,
            gather_mma_QdO_pipeline,
            mma_compute_S_pipeline,
            mma_compute_dP_pipeline,
            mma_reduce_dQ_pipeline,
            compute_mma_chunk_pipeline,
            mma_reduce_dKV_pipeline,
        ) = pipelines

        # TMEM-resident A operands: P^T over the S columns, dS^T over the dP ones.
        # The column offset must be applied to the *fragment* iterator in bf16 units:
        # a recast_ptr(base + off) view silently drops it in the TS-gemm lowering.
        tP_base = cute.make_tensor(tmem_ptr_base, tP_layout.outer)
        col_units = self.acc_dtype.width // self.element_dtype.width

        tSrK = mma_S.make_fragment_A(sK)
        tSrQ = mma_S.make_fragment_B(sQ)
        tdPrV = mma_dP.make_fragment_A(sV)
        tdPrdO = mma_dP.make_fragment_B(sdO)
        tdVrP0 = mma_dV.make_fragment_A(tP_base)
        tdVrP = cute.make_tensor(tdVrP0.iterator + col_units * self.tmem_S_offset, tdVrP0.layout)
        tdVrdOb = mma_dV.make_fragment_B(sdOb)
        tdKrdS0 = mma_dK.make_fragment_A(tP_base)
        tdKrdS = cute.make_tensor(tdKrdS0.iterator + col_units * self.tmem_dPdQ_offset, tdKrdS0.layout)
        tdKrQb = mma_dK.make_fragment_B(sQb)
        tdQrKt = mma_dQ.make_fragment_A(sKt)
        tdQrdSn = mma_dQ.make_fragment_B(sPdSn)

        NCHUNK = self.num_compute_chunks
        # k-blocks (MMA K=16) per 32-column compute chunk
        KK_PER_CHUNK = cute.size(tdVrP, mode=[2]) // NCHUNK

        kv_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.load_mma_KV_stage)
        qdo_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.gather_mma_QdO_stage)
        s_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.mma_compute_S_stage)
        dp_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.mma_compute_dP_stage)
        dq_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.mma_reduce_dQ_stage)
        chunk_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.compute_mma_chunk_stage)
        chunk_rel_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.compute_mma_chunk_stage)
        dkv_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.mma_reduce_dKV_stage)

        # held-slot protocol for TMEM cols [128,256): dP(t) is committed after G2,
        # then re-acquired (on the compute warps' release) before dQ overwrites it.
        mma_compute_dP_pipeline.producer_acquire(dp_state)
        # held-slot for TMEM cols [0,128): S(t), then S(t+1).
        mma_compute_S_pipeline.producer_acquire(s_state)

        is_first_tile = True
        seg_first = True
        row = row_lo
        while row < row_hi:
            if seg_first:
                load_mma_KV_pipeline.consumer_wait(kv_state)

            gather_mma_QdO_pipeline.consumer_wait(qdo_state)

            # ---- G1: S^T = K . Q^T (held S slot) ----
            mma_S.set(tcgen05.Field.ACCUMULATE, False)
            for kk in cutlass.range(0, cute.size(tSrK, mode=[2]), unroll=4):
                cute.gemm(
                    mma_S,
                    tStS,
                    tSrK[None, None, kk, kv_state.index],
                    tSrQ[None, None, kk, qdo_state.index % self.q_stage],
                    tStS,
                )
                mma_S.set(tcgen05.Field.ACCUMULATE, True)
            mma_compute_S_pipeline.producer_commit(s_state)
            s_state.advance()

            # ---- G2: dP^T = V . dO^T (cols 128..256) ----
            # reduce must have drained the previous dQ^T T2R (same columns)
            if not is_first_tile:
                self.t2r_dQ_done_barrier.arrive_and_wait()
            mma_dP.set(tcgen05.Field.ACCUMULATE, False)
            for kk in cutlass.range(0, cute.size(tdPrV, mode=[2]), unroll=4):
                cute.gemm(
                    mma_dP,
                    tdPtdP,
                    tdPrV[None, None, kk, kv_state.index],
                    tdPrdO[None, None, kk, qdo_state.index % self.do_stage],
                    tdPtdP,
                )
                mma_dP.set(tcgen05.Field.ACCUMULATE, True)
            mma_compute_dP_pipeline.producer_commit(dp_state)
            dp_state.advance()

            if seg_first:
                # G1/G2 never touch the dV/dK columns, so the acquire sits here and
                # the flush of segment s-1 overlaps segment s.
                mma_reduce_dKV_pipeline.producer_acquire(dkv_state)

            # ---- G3/G4 per chunk: dV += P^T[:,c].dO[c,:], dK += dS^T[:,c].Q[c,:] ----
            mma_dV.set(tcgen05.Field.ACCUMULATE, not seg_first)
            mma_dK.set(tcgen05.Field.ACCUMULATE, not seg_first)
            STARTS = self.chunk_stage_starts
            NSTAGE = len(STARTS)
            for st in cutlass.range_constexpr(NSTAGE):
                c_lo = STARTS[st]
                c_hi = STARTS[st + 1] if st + 1 < NSTAGE else NCHUNK
                compute_mma_chunk_pipeline.consumer_wait(chunk_state)
                chunk_state.advance()
                for j in cutlass.range_constexpr((c_hi - c_lo) * KK_PER_CHUNK):
                    kk = c_lo * KK_PER_CHUNK + j
                    cute.gemm(
                        mma_dV,
                        tdVtdV,
                        tdVrP[None, None, kk],
                        tdVrdOb[None, None, kk, qdo_state.index % self.do_stage],
                        tdVtdV,
                    )
                    mma_dV.set(tcgen05.Field.ACCUMULATE, True)
                for j in cutlass.range_constexpr((c_hi - c_lo) * KK_PER_CHUNK):
                    kk = c_lo * KK_PER_CHUNK + j
                    cute.gemm(
                        mma_dK,
                        tdKtdK,
                        tdKrdS[None, None, kk],
                        tdKrQb[None, None, kk, qdo_state.index % self.q_stage],
                        tdKtdK,
                    )
                    mma_dK.set(tcgen05.Field.ACCUMULATE, True)

            # Q/dO consumed after G4 (G1..G4 read them; G5 does not).
            gather_mma_QdO_pipeline.consumer_release(qdo_state)
            qdo_state.advance()

            # ---- G5: dQ^T = K^T . dS^T (cols 128..256, aliases dP^T/dS^T) ----
            # dP and the previous dQ generation are released; G4's dS^T reads
            # precede G5 in the tensor core's issue order.
            mma_compute_dP_pipeline.producer_acquire(dp_state)
            mma_reduce_dQ_pipeline.producer_acquire(dq_state)
            mma_dQ.set(tcgen05.Field.ACCUMULATE, False)
            for kk in cutlass.range(0, cute.size(tdQrKt, mode=[2]), unroll=2):
                cute.gemm(
                    mma_dQ,
                    tdQtdQ,
                    tdQrKt[None, None, kk, kv_state.index],
                    tdQrdSn[None, None, kk, 0],
                    tdQtdQ,
                )
                mma_dQ.set(tcgen05.Field.ACCUMULATE, True)
            mma_reduce_dQ_pipeline.producer_commit(dq_state)
            dq_state.advance()

            # release every chunk stage: the tcgen05.commit fires once G5 has drained,
            # so compute may only then overwrite sPdS / the packed TMEM columns.
            for st in cutlass.range_constexpr(len(self.chunk_stage_starts)):
                compute_mma_chunk_pipeline.consumer_release(chunk_rel_state)
                chunk_rel_state.advance()

            # compute has finished reading S(t): the columns are free for S(t+1)
            mma_compute_S_pipeline.producer_acquire(s_state)

            is_seg_end = cutlass.Boolean(row + 1 >= row_hi)
            if row + 1 < row_hi:
                is_seg_end = ~self._same_bucket(mTaskMeta, row + 1, row)
            if is_seg_end:
                load_mma_KV_pipeline.consumer_release(kv_state)
                kv_state.advance()
                mma_reduce_dKV_pipeline.producer_commit(dkv_state)
                dkv_state.advance()
                seg_first = True
            else:
                seg_first = False

            is_first_tile = False
            row += 1

        # Balance the reduce warps' final t2r_dQ_done arrive (none if the interval is empty).
        if row_lo < row_hi:
            self.t2r_dQ_done_barrier.arrive_and_wait()

    # ---- compute warps: softmax / dS ----
    @cute.jit
    def compute(
        self,
        tStS: cute.Tensor,
        tdPtdP: cute.Tensor,
        sPdS_store: cute.Tensor,
        sLSE: cute.Tensor,
        sDelta: cute.Tensor,
        sQPos: cute.Tensor,
        mTaskMeta: cute.Tensor,
        row_lo: Int32,
        row_hi: Int32,
        scale_log2e: Float32,
        log2_e: Float32,
        pipelines,
    ):
        (
            mma_compute_S_pipeline,
            mma_compute_dP_pipeline,
            compute_mma_chunk_pipeline,
            gather_row_pipeline,
        ) = pipelines

        tidx, _, _ = cute.arch.thread_idx()
        dp_idx = (tidx - self.compute_warp_id[0] * self.threads_per_warp) % 128

        s_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.mma_compute_S_stage)
        dp_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.mma_compute_dP_stage)
        chunk_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.compute_mma_chunk_stage)
        # separate producer state: the mma warp releases every stage together after G5
        chunk_acq_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.compute_mma_chunk_stage)
        row_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.gather_row_stage)

        CHUNK = self.compute_chunk_cols
        NCHUNK = self.num_compute_chunks
        HALF = CHUNK // 2  # packed bf16 pairs: 16 f32-typed TMEM columns per chunk

        # per-chunk TMEM views: columns [c*CHUNK, (c+1)*CHUNK)
        chunk_shape = (cute.make_layout((TILE_M, CHUNK)), 1, 1)
        tS_chunk_layout = cute.composition(tStS, chunk_shape).layout
        tdP_chunk_layout = cute.composition(tdPtdP, chunk_shape).layout

        tmem_load_atom = cute.make_copy_atom(tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(CHUNK)), self.acc_dtype)

        cS_chunk = cute.make_identity_tensor((TILE_M, CHUNK))
        sPdS_div = cute.flat_divide(sPdS_store[None, None, 0], (TILE_M, CHUNK))

        tS0 = cute.make_tensor(tStS.iterator, tS_chunk_layout)
        tS0_v = tS0[(None, None), 0, 0]
        tiled_t2r = tcgen05.make_tmem_copy(tmem_load_atom, tS0_v)
        thr_t2r = tiled_t2r.get_slice(dp_idx)
        tTR_cS = thr_t2r.partition_D(cS_chunk)

        tTR_tS = []
        tTR_tdP = []
        for c in cutlass.range_constexpr(NCHUNK):
            tSc = cute.make_tensor(tStS.iterator + c * CHUNK, tS_chunk_layout)
            tdPc = cute.make_tensor(tdPtdP.iterator + c * CHUNK, tdP_chunk_layout)
            tTR_tS.append(thr_t2r.partition_S(tSc[(None, None), 0, 0]))
            tTR_tdP.append(thr_t2r.partition_S(tdPc[(None, None), 0, 0]))

        # P^T / dS^T publication: chunk c's 32 bf16 values pack into the 16 f32-typed
        # columns [c*HALF, (c+1)*HALF) that this thread has already drained.
        p_chunk_shape = (cute.make_layout((TILE_M, HALF)), 1, 1)
        tP_chunk_layout = cute.composition(tStS, p_chunk_shape).layout
        tmem_store_atom = cute.make_copy_atom(tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(HALF)), self.acc_dtype)
        tP0 = cute.make_tensor(tStS.iterator, tP_chunk_layout)
        tiled_r2t = tcgen05.make_tmem_copy(tmem_store_atom, tP0[(None, None), 0, 0])
        thr_r2t = tiled_r2t.get_slice(dp_idx)
        cP_chunk = cute.make_identity_tensor((TILE_M, HALF))
        tRT_cP = thr_r2t.partition_S(cP_chunk)
        tRT_tP = []
        tRT_tdS = []
        for c in cutlass.range_constexpr(NCHUNK):
            tPc = cute.make_tensor(tStS.iterator + c * HALF, tP_chunk_layout)
            tdSc = cute.make_tensor(tdPtdP.iterator + c * HALF, tP_chunk_layout)
            tRT_tP.append(thr_r2t.partition_D(tPc[(None, None), 0, 0]))
            tRT_tdS.append(thr_r2t.partition_D(tdSc[(None, None), 0, 0]))

        # rmem->smem store of dS chunks for G5 (thread dp_idx owns one key row)
        smem_store_atom = sm100_utils.get_smem_store_op(
            LayoutEnum.ROW_MAJOR, self.element_dtype, self.acc_dtype, tiled_t2r
        )
        tiled_r2s = cute.make_tiled_copy_D(smem_store_atom, tiled_t2r)
        thr_r2s = tiled_r2s.get_slice(dp_idx)
        tRS_sPdS = []
        for c in cutlass.range_constexpr(NCHUNK):
            tRS_sPdS.append(thr_r2s.partition_D(sPdS_div[None, None, 0, c]))

        mpp = Int32(self.main_per_index)
        softmax_scale = scale_log2e / log2_e

        row = row_lo
        while row < row_hi:
            _b, _index_head, kb, valid = self._task_fields(mTaskMeta, row)
            key_base = kb * Int32(BLOCK_SIZE)
            gather_row_pipeline.consumer_wait(row_state)
            ridx = row_state.index

            # S^T(t) lands first; the dP wait is deferred to chunk 0 to overlap G2.
            mma_compute_S_pipeline.consumer_wait(s_state)
            # all chunk stages are free once the previous tile's G3/G4/G5 have drained
            for st in cutlass.range_constexpr(len(self.chunk_stage_starts)):
                compute_mma_chunk_pipeline.producer_acquire(chunk_acq_state)
                chunk_acq_state.advance()

            # causal predicate row: the thread's datapath row is its key row
            key_pos_thr = key_base + tTR_cS[0][0]

            for c in cutlass.range_constexpr(NCHUNK):
                rLse = cute.make_rmem_tensor(tTR_cS.shape, self.acc_dtype)
                rDelta = cute.make_rmem_tensor(tTR_cS.shape, self.acc_dtype)
                for i in cutlass.range_constexpr(cute.size(tTR_cS)):
                    n_col = Int32(c * CHUNK) + tTR_cS[i][1]
                    rLse[i] = sLSE[n_col, ridx]
                    rDelta[i] = sDelta[n_col, ridx]
                if cutlass.const_expr(self.causal):
                    rQpos = cute.make_rmem_tensor(tTR_cS.shape, cutlass.Int32)
                    for i in cutlass.range_constexpr(cute.size(tTR_cS)):
                        n_col = Int32(c * CHUNK) + tTR_cS[i][1]
                        rQpos[i] = sQPos[n_col // mpp, ridx]

                # ---- S / dP chunk -> registers (chunk 0 defers its dP load) ----
                tTR_rS = cute.make_rmem_tensor(tTR_cS.shape, self.acc_dtype)
                tTR_rdP = cute.make_rmem_tensor(tTR_cS.shape, self.acc_dtype)
                cute.copy(tiled_t2r, tTR_tS[c], tTR_rS)
                if cutlass.const_expr(c > 0):
                    cute.copy(tiled_t2r, tTR_tdP[c], tTR_rdP)

                if cutlass.const_expr(c > 0 and c in self.chunk_stage_starts):
                    # publish the previous stage; its store latency hides behind this T2R
                    cute.arch.fence_view_async_tmem_store()
                    cute.arch.fence_proxy("async.shared", space="cta")
                    compute_mma_chunk_pipeline.producer_commit(chunk_state)
                    chunk_state.advance()

                # ---- S chunk -> P chunk (branchless, vectorized) ----
                # invalid slots carry lse = -inf (exp2 -> +0); causal masks qpos = -1
                v = tTR_rS.load() * scale_log2e + rLse.load()
                if cutlass.const_expr(self.causal):
                    cond = rQpos.load() >= key_pos_thr
                    p = cute.where(cond, cute.math.exp2(v, fastmath=True), Float32(0.0))
                else:
                    p = cute.math.exp2(v, fastmath=True)
                tTR_rS.store(p)
                # P^T chunk -> TMEM (A operand of G3)
                rP_f16 = self.quantize(tTR_rS, 4)
                rP_words = cute.make_rmem_tensor(tRT_cP.shape, self.acc_dtype)
                rP_view = cute.recast_tensor(rP_words, self.element_dtype)
                for i in cutlass.range_constexpr(cute.size(rP_f16)):
                    rP_view[i] = rP_f16[i]
                cute.copy(tiled_r2t, rP_words, tRT_tP[c])

                # ---- dP chunk -> dS chunk ----
                if cutlass.const_expr(c == 0):
                    mma_compute_dP_pipeline.consumer_wait(dp_state)
                    cute.copy(tiled_t2r, tTR_tdP[c], tTR_rdP)
                ds = tTR_rS.load() * (tTR_rdP.load() - rDelta.load())
                tTR_rdP.store(ds)
                # softmax scale folded into dS: dQ/dK writeouts need no scale
                rdS_f16 = self.quantize(tTR_rdP, 4, softmax_scale)
                # dS^T chunk -> TMEM (A operand of G4)
                rdS_words = cute.make_rmem_tensor(tRT_cP.shape, self.acc_dtype)
                rdS_view = cute.recast_tensor(rdS_words, self.element_dtype)
                for i in cutlass.range_constexpr(cute.size(rdS_f16)):
                    rdS_view[i] = rdS_f16[i]
                cute.copy(tiled_r2t, rdS_words, tRT_tdS[c])
                # dS chunk -> SMEM (B operand of G5)
                tRS_rdS = tiled_r2s.retile(rdS_f16)
                cute.copy(tiled_r2s, tRS_rdS, tRS_sPdS[c])

            # publish the last stage after this thread's own fences
            cute.arch.fence_view_async_tmem_store()
            cute.arch.fence_proxy("async.shared", space="cta")
            compute_mma_chunk_pipeline.producer_commit(chunk_state)
            chunk_state.advance()

            cute.arch.fence_view_async_tmem_load()
            mma_compute_S_pipeline.consumer_release(s_state)
            s_state.advance()
            mma_compute_dP_pipeline.consumer_release(dp_state)
            dp_state.advance()
            gather_row_pipeline.consumer_release(row_state)
            row_state.advance()

            row += 1

    # ---- reduce warps: dQ per tile; dV/dK per segment ----
    @cute.jit
    def reduce(
        self,
        tdQtdQ: cute.Tensor,
        tdVtdV: cute.Tensor,
        tdKtdK: cute.Tensor,
        mdQ: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        sQRows: cute.Tensor,
        mTaskMeta: cute.Tensor,
        row_lo: Int32,
        row_hi: Int32,
        pipelines,
    ):
        (
            mma_reduce_dQ_pipeline,
            mma_reduce_dKV_pipeline,
            gather_row_pipeline,
        ) = pipelines

        tidx, _, _ = cute.arch.thread_idx()
        dp_idx = (tidx - self.reduce_warp_id[0] * self.threads_per_warp) % 128

        dq_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.mma_reduce_dQ_stage)
        dkv_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.mma_reduce_dKV_stage)
        row_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.gather_row_stage)

        tdQtdQ_v = tdQtdQ[(None, None), 0, 0]
        tdVtdV_v = tdVtdV[(None, None), 0, 0]
        tdKtdK_v = tdKtdK[(None, None), 0, 0]

        tmem_load_atom = cute.make_copy_atom(tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), self.acc_dtype)
        tiled_t2r = tcgen05.make_tmem_copy(tmem_load_atom, tdQtdQ_v)
        thr_t2r = tiled_t2r.get_slice(dp_idx)

        cAcc = cute.make_identity_tensor((TILE_M, TILE_N))
        tTR_cAcc = thr_t2r.partition_D(cAcc)
        tTR_tdQ = thr_t2r.partition_S(tdQtdQ_v)
        tTR_tdV = thr_t2r.partition_S(tdVtdV_v)
        tTR_tdK = thr_t2r.partition_S(tdKtdK_v)
        tTR_r = cute.make_rmem_tensor(tTR_cAcc.shape, self.acc_dtype)

        mpp = Int32(self.main_per_index)
        row = row_lo
        while row < row_hi:
            b, index_head, kb, valid = self._task_fields(mTaskMeta, row)
            gather_row_pipeline.consumer_wait(row_state)
            ridx = row_state.index
            # Release the row slot before the dQ atomics below (measured: holding it
            # through red_atomic throttled gth_acq by ~1.5 us/tile).
            rQRow = cute.make_rmem_tensor((self.query_chunk,), cutlass.Int32)
            for qs in cutlass.range_constexpr(self.query_chunk):
                rQRow[qs] = sQRows[qs, ridx]
            gather_row_pipeline.consumer_release(row_state)
            row_state.advance()

            # ---- dQ^T of this tile ----
            mma_reduce_dQ_pipeline.consumer_wait(dq_state)
            cute.copy(tiled_t2r, tTR_tdQ, tTR_r)
            cute.arch.fence_view_async_tmem_load()
            # T2R done: unblock the mma warp's next dP write before the slow atomics
            self.t2r_dQ_done_barrier.arrive()
            mma_reduce_dQ_pipeline.consumer_release(dq_state)
            dq_state.advance()
            # coalesced scatter: the 128 dp lanes span head_dim -> 128B RED runs.
            # Keep the per-element crd2idx; hoisting a base pointer out of the loop
            # grew instruction count and stack usage, as ptxas already optimizes it.
            for i in cutlass.range_constexpr(cute.size(tTR_r)):
                n_col = tTR_cAcc[i][1]
                d_row = tTR_cAcc[i][0]
                q_slot = n_col // mpp
                head_off = n_col - q_slot * mpp
                if q_slot < valid:
                    q_row = rQRow[q_slot]
                    head = index_head * mpp + head_off
                    dq_ptr = mdQ.iterator + cute.crd2idx((b, head, q_row, d_row), mdQ.layout)
                    cute.arch.atomic_add(dq_ptr.llvm_ptr, tTR_r[i])

            # ---- segment flush: dV^T / dK^T ----
            is_seg_end = cutlass.Boolean(row + 1 >= row_hi)
            if row + 1 < row_hi:
                is_seg_end = ~self._same_bucket(mTaskMeta, row + 1, row)
            if is_seg_end:
                kv_head = index_head // Int32(self.index_heads_per_kv)
                key_base = kb * Int32(BLOCK_SIZE)
                mma_reduce_dKV_pipeline.consumer_wait(dkv_state)

                # dV / dK land as (M = key lanes, N = d columns). The RED path is
                # sector-bound, so each 4x4 block is transposed inside the quad first:
                # one warp instruction then writes 8 rows x 64 B = 16 full sectors
                # instead of 32 half-filled ones.
                row_in_tile = tTR_cAcc[0][0]
                q4 = row_in_tile % 4
                quad_row = key_base + row_in_tile - q4
                d_off = q4 * 4
                tTR_r_flat = cute.make_tensor(tTR_r.iterator, cute.make_layout(cute.size(tTR_r)))
                tTR_r4 = cute.logical_divide(tTR_r_flat, cute.make_layout(4))
                # lane-bit predicates, 4-wide so cute.where can select vectors
                sel0 = cute.make_rmem_tensor((4,), cutlass.Int32)
                sel1 = cute.make_rmem_tensor((4,), cutlass.Int32)
                for e in cutlass.range_constexpr(4):
                    sel0[e] = q4 % 2
                    sel1[e] = q4 // 2
                c0 = sel0.load() != 0
                c1 = sel1.load() != 0

                cute.copy(tiled_t2r, tTR_tdV, tTR_r)
                cute.arch.fence_view_async_tmem_load()
                dv_rows = [mdV.iterator + cute.crd2idx((b, kv_head, quad_row + k, 0), mdV.layout) for k in range(4)]
                for m in cutlass.range_constexpr(cute.size(tTR_r) // 16):
                    blk = [tTR_r4[None, 4 * m + k] for k in range(4)]
                    self._quad_transpose4(blk, c0, c1)
                    for k in cutlass.range_constexpr(4):
                        dv_ptr = dv_rows[k] + (16 * m) + d_off
                        cute.arch.atomic_add(dv_ptr.llvm_ptr, blk[k].load())

                cute.copy(tiled_t2r, tTR_tdK, tTR_r)
                cute.arch.fence_view_async_tmem_load()
                # both accumulators are in registers: free the TMEM columns first
                mma_reduce_dKV_pipeline.consumer_release(dkv_state)
                dkv_state.advance()
                dk_rows = [mdK.iterator + cute.crd2idx((b, kv_head, quad_row + k, 0), mdK.layout) for k in range(4)]
                for m in cutlass.range_constexpr(cute.size(tTR_r) // 16):
                    blk = [tTR_r4[None, 4 * m + k] for k in range(4)]
                    self._quad_transpose4(blk, c0, c1)
                    for k in cutlass.range_constexpr(4):
                        dk_ptr = dk_rows[k] + (16 * m) + d_off
                        cute.arch.atomic_add(dk_ptr.llvm_ptr, blk[k].load())

                self.reduce_sync_barrier.arrive_and_wait()

            row += 1

    @cute.jit
    def quantize(
        self,
        input: cute.Tensor,
        frg_cnt: Int32,
        scale: Float32 | None = None,
    ):
        output = cute.make_rmem_tensor(input.shape, self.element_dtype)
        frg_tile = cute.size(input) // frg_cnt
        t_frg = cute.logical_divide(input, cute.make_layout(frg_cnt))
        output_frg = cute.make_tensor(output.iterator, t_frg.layout)
        for i in cutlass.range(frg_tile, unroll_full=True):
            frg_vec = t_frg[None, i].load()
            if cutlass.const_expr(scale is not None):
                frg_vec = frg_vec * scale
            output_frg[None, i].store(frg_vec.to(self.element_dtype))
        return output

    def _quad_transpose4(self, blk, c0, c1):
        """In-place 4x4 transpose of four F32x4 fragments across a quad's 4 lanes.

        Lane q holds element (row q, block k) before and (row k, block q) after.
        ``c0``/``c1`` are 4-wide lane-bit vectors, so each butterfly stage's slot
        choice lowers to FSEL rather than a branch.
        """
        tmp = cute.make_rmem_tensor((4,), self.acc_dtype)
        recv = cute.make_rmem_tensor((4,), self.acc_dtype)
        # stage 1: lane bit 0 <-> slot bit 0 (pairs (0,1) and (2,3))
        for j1 in range(2):
            s0, s1 = blk[2 * j1], blk[2 * j1 + 1]
            tmp.store(cute.where(c0, s0.load(), s1.load()))
            for e in range(4):
                recv[e] = cute.arch.shuffle_sync_bfly(tmp[e], 1)
            s0.store(cute.where(c0, recv.load(), s0.load()))
            s1.store(cute.where(c0, s1.load(), recv.load()))
        # stage 2: lane bit 1 <-> slot bit 1 (pairs (0,2) and (1,3))
        for k0 in range(2):
            s0, s1 = blk[k0], blk[2 + k0]
            tmp.store(cute.where(c1, s0.load(), s1.load()))
            for e in range(4):
                recv[e] = cute.arch.shuffle_sync_bfly(tmp[e], 2)
            s0.store(cute.where(c1, recv.load(), s0.load()))
            s1.store(cute.where(c1, s1.load(), recv.load()))


_NUM_SMS: dict[int, int] = {}


def _num_sms(device: torch.device) -> int:
    device_index = torch.cuda.current_device() if device.index is None else device.index
    if device_index not in _NUM_SMS:
        _NUM_SMS[device_index] = torch.cuda.get_device_properties(device_index).multi_processor_count
    return _NUM_SMS[device_index]


def _validate_inputs(
    q: torch.Tensor,
    k_aligned: torch.Tensor,
    v_aligned: torch.Tensor,
    grad_out: torch.Tensor,
    lse: torch.Tensor,
    out: torch.Tensor,
    task_meta: torch.Tensor,
    task_qrows: torch.Tensor,
    task_qpos: torch.Tensor,
    softmax_scale: float,
) -> None:
    """Check device, dtype, 16-byte alignment, and THD shapes before launch."""
    tensors = (q, k_aligned, v_aligned, grad_out, lse, out, task_meta, task_qrows, task_qpos)
    if q.device.type != "cuda":
        raise ValueError("MiniMax M3 MSA backward requires CUDA tensors")
    if any(tensor.device != q.device for tensor in tensors):
        raise ValueError("all MiniMax M3 MSA backward tensors must be on one CUDA device")
    if torch.cuda.get_device_capability(q.device) != (10, 0):
        raise NotImplementedError("MiniMax M3 MSA backward requires an SM100 CUDA device")
    misaligned = [
        name
        for name, tensor in zip(
            ("q", "k_aligned", "v_aligned", "grad_out", "lse", "out", "task_meta", "task_qrows", "task_qpos"),
            tensors,
            strict=True,
        )
        if tensor.data_ptr() % 16 != 0
    ]
    if misaligned:
        raise ValueError(
            "MiniMax M3 MSA backward requires 16-byte-aligned storage for its compiled tensor ABI; "
            f"misaligned tensors={misaligned}."
        )

    if q.dtype != torch.bfloat16:
        raise TypeError(f"q must be BF16, got {q.dtype}")
    for name, tensor in (
        ("k_aligned", k_aligned),
        ("v_aligned", v_aligned),
        ("grad_out", grad_out),
        ("out", out),
    ):
        if tensor.dtype != torch.bfloat16:
            raise TypeError(f"{name} must be BF16, got {tensor.dtype}")
    if lse.dtype != torch.float32:
        raise TypeError(f"lse must be FP32, got {lse.dtype}")
    for name, tensor in (
        ("task_meta", task_meta),
        ("task_qrows", task_qrows),
        ("task_qpos", task_qpos),
    ):
        if tensor.dtype != torch.int32:
            raise TypeError(f"{name} must be int32, got {tensor.dtype}")

    if q.ndim != 3 or q.shape[1] != NUM_Q_HEADS or q.shape[2] != HEAD_DIM:
        raise ValueError(f"q must have shape [T, {NUM_Q_HEADS}, {HEAD_DIM}], got {tuple(q.shape)}")
    if q.shape[0] <= 0:
        raise ValueError("q must contain at least one compact token")
    for name, tensor in (("k_aligned", k_aligned), ("v_aligned", v_aligned)):
        if tensor.ndim != 3 or tensor.shape[1] != NUM_KV_HEADS or tensor.shape[2] != HEAD_DIM:
            raise ValueError(f"{name} must have shape [W, {NUM_KV_HEADS}, {HEAD_DIM}], got {tuple(tensor.shape)}")
    if k_aligned.shape != v_aligned.shape:
        raise ValueError("k_aligned and v_aligned must have identical shapes")
    if k_aligned.shape[0] <= 0 or k_aligned.shape[0] % BLOCK_SIZE != 0:
        raise ValueError(f"aligned K/V workspace length must be a positive multiple of {BLOCK_SIZE}")
    if grad_out.shape != q.shape or out.shape != q.shape:
        raise ValueError("grad_out and out must have the same shape as q")
    if lse.shape != q.shape[:2]:
        raise ValueError(f"lse must have shape {tuple(q.shape[:2])}, got {tuple(lse.shape)}")

    if task_meta.ndim != 2 or task_meta.shape[1] != 4:
        raise ValueError(f"task_meta must have shape [num_tasks, 4], got {tuple(task_meta.shape)}")
    expected_task_shape = (task_meta.shape[0], QUERY_CHUNK)
    if task_qrows.shape != expected_task_shape:
        raise ValueError(f"task_qrows must have shape {expected_task_shape}, got {tuple(task_qrows.shape)}")
    if task_qpos.shape != expected_task_shape:
        raise ValueError(f"task_qpos must have shape {expected_task_shape}, got {tuple(task_qpos.shape)}")
    if not math.isfinite(softmax_scale) or softmax_scale <= 0.0:
        raise ValueError(f"softmax_scale must be finite and positive, got {softmax_scale}")


def _compile_backward() -> Any:
    """Compile the backward once with dynamic token, workspace, and task counts.

    The fake tensors describe the head-major views ``_run_msa_backward`` builds, so
    every stride except the size-1 batch stride is static; ``stride_order[i]`` is the
    rank of mode ``i``, ``0`` innermost. Returns an executable taking the kernel's
    positional arguments minus the trailing stream.
    """
    num_tokens = cute.sym_int32(symbol="num_tokens")
    workspace_rows = cute.sym_int32(divisibility=BLOCK_SIZE, symbol="workspace_rows")
    num_tasks = cute.sym_int32(symbol="num_tasks")

    def rows_by_head(dtype: type[cutlass.Numeric], heads: int, rows: Any) -> Any:
        # (1, heads, rows, D):(heads*D*rows, D, heads*D, 1)
        return make_fake_compact_tensor(dtype, (1, heads, rows, HEAD_DIM), stride_order=(3, 1, 2, 0), assumed_align=16)

    def row_stats() -> Any:
        # (1, Hq, T):(Hq*T, 1, Hq)
        return make_fake_compact_tensor(Float32, (1, NUM_Q_HEADS, num_tokens), stride_order=(2, 0, 1), assumed_align=16)

    def tasks(width: int) -> Any:
        return make_fake_compact_tensor(Int32, (num_tasks, width), stride_order=(1, 0), assumed_align=16)

    fake_args = (
        rows_by_head(cutlass.BFloat16, NUM_Q_HEADS, num_tokens),
        rows_by_head(cutlass.BFloat16, NUM_KV_HEADS, workspace_rows),
        rows_by_head(cutlass.BFloat16, NUM_KV_HEADS, workspace_rows),
        rows_by_head(cutlass.BFloat16, NUM_Q_HEADS, num_tokens),
        row_stats(),
        row_stats(),
        tasks(4),
        tasks(QUERY_CHUNK),
        tasks(QUERY_CHUNK),
        rows_by_head(Float32, NUM_Q_HEADS, num_tokens),
        rows_by_head(Float32, NUM_KV_HEADS, workspace_rows),
        rows_by_head(Float32, NUM_KV_HEADS, workspace_rows),
        Int32(1),
        Int32(1),
        Int32(1),
        Int32(1),
        Int32(1),
        Float32(1.0),
        make_fake_stream(use_tvm_ffi_env_stream=True),
    )
    return cute.compile(_MSABackwardSm100Kernel(), *fake_args, options="--enable-tvm-ffi")


def _run_msa_backward(
    q: torch.Tensor,
    k_aligned: torch.Tensor,
    v_aligned: torch.Tensor,
    grad_out: torch.Tensor,
    lse: torch.Tensor,
    out: torch.Tensor,
    schedule: _MSABackwardSchedule,
    *,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the SM100 main-attention backward on THD-contract tensors.

    ``q``/``grad_out``/``out`` are BF16 ``[T, 64, 128]`` (compact tokens, heads,
    head_dim), ``lse`` is FP32 ``[T, 64]``, and ``k_aligned``/``v_aligned`` are BF16
    ``[W, 4, 128]`` with ``W`` a multiple of 128. Every kernel operand must have
    contiguous, 16-byte-aligned storage. Returns BF16
    ``(dq, dk_aligned, dv_aligned)`` in those same layouts, zero outside the support.
    The kernel's head-major operands are strided views built here, so no transposed
    copies are made, and one executable serves every ``T``, ``W``, and task count.
    """
    task_meta, task_qrows, task_qpos = _build_backward_tasks(schedule)
    _validate_inputs(
        q,
        k_aligned,
        v_aligned,
        grad_out,
        lse,
        out,
        task_meta,
        task_qrows,
        task_qpos,
        softmax_scale,
    )

    q_c = q.detach().contiguous()
    k_c = k_aligned.detach().contiguous()
    v_c = v_aligned.detach().contiguous()
    grad_out_c = grad_out.detach().contiguous()
    lse_c = lse.detach().contiguous()
    out_c = out.detach().contiguous()
    task_meta_c = task_meta.detach().contiguous()
    task_qrows_c = task_qrows.detach().contiguous()
    task_qpos_c = task_qpos.detach().contiguous()

    num_task_rows = int(task_meta_c.shape[0])
    if num_task_rows == 0:
        return torch.zeros_like(q_c), torch.zeros_like(k_c), torch.zeros_like(v_c)

    delta = _run_msa_backward_preprocess(out_c, grad_out_c)

    num_dq = q_c.numel()
    num_dk = k_c.numel()
    num_dv = v_c.numel()
    grad_pool = torch.zeros(num_dq + num_dk + num_dv, dtype=torch.float32, device=q.device)
    dq = grad_pool[:num_dq].view(q_c.shape)
    dk = grad_pool[num_dq : num_dq + num_dk].view(k_c.shape)
    dv = grad_pool[num_dq + num_dk :].view(v_c.shape)

    # Head-major kernel views; only the size-1 batch mode is added.
    q_v, grad_out_v, dq_v, k_v, v_v, dk_v, dv_v = (
        tensor.unsqueeze(0).transpose(1, 2) for tensor in (q_c, grad_out_c, dq, k_c, v_c, dk, dv)
    )
    lse_v = lse_c.unsqueeze(0).transpose(1, 2)
    delta_v = delta.unsqueeze(0).transpose(1, 2)

    rows_per_cta = _select_rows_per_cta(num_task_rows)
    num_full_ctas, tail_rows, grid_ctas = _chunk_map(num_task_rows, rows_per_cta, _num_sms(q.device))
    key = ("minimax-m3-msa-backward-sm100", torch.cuda.get_device_capability(q.device), q_c.dtype)
    if key not in _COMPILE_CACHE:
        _COMPILE_CACHE[key] = _compile_backward()
    _COMPILE_CACHE[key](
        q_v,
        k_v,
        v_v,
        grad_out_v,
        lse_v,
        delta_v,
        task_meta_c,
        task_qrows_c,
        task_qpos_c,
        dq_v,
        dk_v,
        dv_v,
        Int32(num_task_rows),
        Int32(rows_per_cta),
        Int32(num_full_ctas),
        Int32(tail_rows),
        Int32(grid_ctas),
        Float32(softmax_scale),
    )

    grad_pool_bf16 = grad_pool.to(dtype=torch.bfloat16)
    dq_out = grad_pool_bf16[:num_dq].view(q_c.shape)
    dk_out = grad_pool_bf16[num_dq : num_dq + num_dk].view(k_c.shape)
    dv_out = grad_pool_bf16[num_dq + num_dk :].view(v_c.shape)
    return dq_out, dk_out, dv_out
