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
bucket and Q/dO TMA-loaded per tile (one load warp, 8-row gathers). One mma warp
issues all five tcgen05 GEMMs transposed (S^T, dP^T, dV, dK, dQ^T) over four
128-column TMEM allocations; dV/dK accumulate per bucket segment and are flushed
with fp32 vector atomics, dQ^T per tile with packed 16-bit atomics (fp16 by
default, ``MSA_M3_DQ_ACCUM=bf16``) into a head-pair-interleaved pool that
``msa_grad_finalize_sm100`` casts to the bf16 gradient. The task tables come
from ``msa_task_build_sm100`` (``MSA_M3_TASK_BUILD=torch`` restores the eager
chain).
"""

import math
import os
from typing import Any

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cuda.bindings import driver as cuda
from cutlass import Float32, Int32
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.nvgpu.common import OperandMajorMode
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.utils import LayoutEnum

from nemo_automodel.components.models.minimax_m3_vl.kernels.msa_backward_preprocess_sm100 import (
    _run_msa_backward_preprocess,
    preprocess_executable,
)
from nemo_automodel.components.models.minimax_m3_vl.kernels.msa_grad_finalize_sm100 import (
    grad_finalize_executable,
    run_grad_finalize,
)
from nemo_automodel.components.models.minimax_m3_vl.kernels.msa_schedule import _MSABackwardSchedule
from nemo_automodel.components.models.minimax_m3_vl.kernels.msa_task_build_sm100 import (
    DESC_WORDS,
    build_backward_tasks,
    compile_task_build,
    task_build_storage,
)

BLOCK_SIZE = 128
TILE_M = 128  # keys per tile (= one key block)
TILE_N = 128  # folded (q_slot, head) rows per tile
HEAD_DIM = 128
NUM_Q_HEADS = 64
NUM_KV_HEADS = 4
# Packed 16-bit dQ atomics: fp16 has 10 mantissa bits but saturates at 65504, bf16 keeps the FP32 range.
_DQ_ACCUM = os.environ.get("MSA_M3_DQ_ACCUM", "fp16")
if _DQ_ACCUM not in ("fp16", "bf16"):
    raise ValueError(f"MSA_M3_DQ_ACCUM must be 'fp16' or 'bf16', got {_DQ_ACCUM!r}")
_DQ_ACCUM_TORCH_DTYPE = {"fp16": torch.float16, "bf16": torch.bfloat16}[_DQ_ACCUM]
_DQ_ACCUM_CUTLASS_DTYPE = {"fp16": cutlass.Float16, "bf16": cutlass.BFloat16}[_DQ_ACCUM]
NUM_INDEX_HEADS = 4
MAIN_HEADS_PER_INDEX = NUM_Q_HEADS // NUM_INDEX_HEADS
QUERY_CHUNK = TILE_N // MAIN_HEADS_PER_INDEX

_COMPILE_CACHE = {}


@dsl_user_op
def _pack_f16x2(lo: Float32, hi: Float32, *, loc=None, ip=None) -> cutlass.Uint32:
    """Round two FP32 values to one f16x2 word, low logical element in the low half."""
    return cutlass.Uint32(
        llvm.inline_asm(
            T.i32(),
            [Float32(lo).ir_value(loc=loc, ip=ip), Float32(hi).ir_value(loc=loc, ip=ip)],
            "cvt.rn.f16x2.f32 $0, $2, $1;",
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _pack_bf16x2(lo: Float32, hi: Float32, *, loc=None, ip=None) -> cutlass.Uint32:
    """Round two FP32 values to one bf16x2 word, low logical element in the low half."""
    return cutlass.Uint32(
        llvm.inline_asm(
            T.i32(),
            [Float32(lo).ir_value(loc=loc, ip=ip), Float32(hi).ir_value(loc=loc, ip=ip)],
            "cvt.rn.bf16x2.f32 $0, $2, $1;",
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _l2_policy_evict_last(*, loc=None, ip=None) -> cutlass.Uint64:
    """64-bit L2 cache policy: keep the whole line set as evict-last (fraction 1.0)."""
    return cutlass.Uint64(
        llvm.inline_asm(
            T.i64(),
            [],
            "createpolicy.fractional.L2::evict_last.b64 $0, 1.0;",
            "=l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


def _red_add_16bitx2_hint(kind: str, destination: cute.Pointer, word, policy, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [
            destination.toint(loc=loc, ip=ip).ir_value(),
            cutlass.Uint32(word).ir_value(loc=loc, ip=ip),
            cutlass.Uint64(policy).ir_value(loc=loc, ip=ip),
        ],
        f"red.global.add.noftz.L2::cache_hint.{kind} [$0], $1, $2;",
        "l,r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _red_add_f16x2_hint(
    destination: cute.Pointer, word: cutlass.Uint32, policy: cutlass.Uint64, *, loc=None, ip=None
) -> None:
    _red_add_16bitx2_hint("f16x2", destination, word, policy, loc=loc, ip=ip)


@dsl_user_op
def _red_add_bf16x2_hint(
    destination: cute.Pointer, word: cutlass.Uint32, policy: cutlass.Uint64, *, loc=None, ip=None
) -> None:
    _red_add_16bitx2_hint("bf16x2", destination, word, policy, loc=loc, ip=ip)


class _MSABackwardSm100Kernel:
    def __init__(self) -> None:
        self.main_per_index = MAIN_HEADS_PER_INDEX
        self.query_chunk = QUERY_CHUNK
        self.index_heads_per_kv = NUM_INDEX_HEADS // NUM_KV_HEADS

        self.acc_dtype = Float32
        self.q_stage = 2
        self.do_stage = 2
        # Double buffered: gather publishes tile t+1 while tile t is consumed.
        self.row_stage = 2

        # warp roles (16 warps / 512 threads); warps 0-3 and 15 are idle
        self.compute_warp_id = (4, 5, 6, 7)
        self.reduce_warp_id = (8, 9, 10, 11)
        self.mma_warp_id = 12
        self.load_warp_id = 13
        self.scalar_warp_id = 14
        self.num_compute_warps = 4
        self.num_reduce_warps = 4
        self.threads_per_warp = 32
        self.threads_per_cta = 512

        # register budget: 4*32*(R_compute + R_reduce) + 32*(3*48 + 5*24) <= 512*128
        self.num_regs_compute = 184
        self.num_regs_reduce = 168  # 184 deadlocks in setmaxnreg.inc on B200 though the budget fits
        self.num_regs_mma = 48
        self.num_regs_load = 48
        self.num_regs_scalar = 48
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
        mdQ: cute.Tensor,  # [1, T, Hq/2, D, 2] fp16 head-pair pool: (t, hp, d, e) = dQ[t, 2*hp + e, d]
        mdK: cute.Tensor,  # [1, Hkv, W, D] fp32 view of [W, Hkv, D]
        mdV: cute.Tensor,  # [1, Hkv, W, D] fp32 view of [W, Hkv, D]
        mDesc: cute.Tensor,  # [8] int32 CTA-walk descriptor (msa_task_build_sm100.DESC_*)
        grid_launch: Int32,  # CTAs launched: >= desc[4]; the surplus ones get an empty interval
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

        # One TMA box = the (16 heads, 64 d) K-major SW128 sub-block of sQ/sdO for one query slot
        # and d half, so G1-G4's MMA descriptors read it unchanged; a slot with row -1 is out of
        # bounds and TMA zero-fills it.
        sQbox_layout = cute.make_composed_layout(
            sQ_layout.inner, 0, cute.make_layout((MAIN_HEADS_PER_INDEX, 64), stride=(64, 1))
        )
        mQ_hdt = cute.make_tensor(
            mQ.iterator,
            cute.make_layout(
                (mQ.shape[1], mQ.shape[3], (mQ.shape[2], mQ.shape[0])),
                stride=(mQ.stride[1], mQ.stride[3], (mQ.stride[2], mQ.stride[0])),
            ),
        )
        mdO_hdt = cute.make_tensor(
            mdO.iterator,
            cute.make_layout(
                (mdO.shape[1], mdO.shape[3], (mdO.shape[2], mdO.shape[0])),
                stride=(mdO.stride[1], mdO.stride[3], (mdO.stride[2], mdO.stride[0])),
            ),
        )
        qdo_box_tiler = (MAIN_HEADS_PER_INDEX, 64)
        tma_atom_Q, tma_tensor_Q = cpasync.make_tiled_tma_atom(tma_load_op, mQ_hdt, sQbox_layout, qdo_box_tiler)
        tma_atom_dO, tma_tensor_dO = cpasync.make_tiled_tma_atom(tma_load_op, mdO_hdt, sQbox_layout, qdo_box_tiler)
        sQ_layout_single = cute.select(sQ_layout, mode=[0, 1, 2])
        sdO_layout_single = cute.select(sdO_layout, mode=[0, 1, 2])
        self.tma_copy_QdO_bytes = cute.size_in_bytes(dt, sQ_layout_single) + cute.size_in_bytes(dt, sdO_layout_single)

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
            tma_atom_Q,
            tma_tensor_Q,
            tma_atom_dO,
            tma_tensor_dO,
            mLSE,
            mDelta,
            mTaskMeta,
            mTaskQRows,
            mTaskQPos,
            mdQ,
            mdK,
            mdV,
            mDesc,
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
            grid=[grid_launch, 1, 1],
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
        tma_atom_Q: cute.CopyAtom,
        tma_tensor_Q: cute.Tensor,
        tma_atom_dO: cute.CopyAtom,
        tma_tensor_dO: cute.Tensor,
        mLSE: cute.Tensor,
        mDelta: cute.Tensor,
        mTaskMeta: cute.Tensor,
        mTaskQRows: cute.Tensor,
        mTaskQPos: cute.Tensor,
        mdQ: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        mDesc: cute.Tensor,
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
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        if warp_idx == self.load_warp_id:
            cpasync.prefetch_descriptor(tma_atom_K)
            cpasync.prefetch_descriptor(tma_atom_V)
            cpasync.prefetch_descriptor(tma_atom_Q)
            cpasync.prefetch_descriptor(tma_atom_dO)

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # CTA -> contiguous task-row interval, read from the descriptor the task build wrote on the
        # device. A split bucket stays correct: each CTA reloads K/V and atomically accumulates
        # dK/dV. The grid is an upper bound, so CTAs beyond desc[4] get an empty interval.
        num_task_rows = cute.arch.make_warp_uniform(mDesc[0])
        rows_per_cta = cute.arch.make_warp_uniform(mDesc[1])
        n_full_ctas = cute.arch.make_warp_uniform(mDesc[2])
        tail_rows = cute.arch.make_warp_uniform(mDesc[3])
        grid_ctas = cute.arch.make_warp_uniform(mDesc[4])
        row_lo = bidx * rows_per_cta
        row_hi = cutlass.min(row_lo + rows_per_cta, num_task_rows)
        if bidx >= n_full_ctas:
            row_lo = n_full_ctas * rows_per_cta + (bidx - n_full_ctas) * tail_rows
            row_hi = cutlass.min(row_lo + tail_rows, num_task_rows)
        if bidx >= grid_ctas:
            row_lo = num_task_rows
            row_hi = num_task_rows

        load_mma_KV_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.load_mma_KV_mbar_ptr.data_ptr(),
            num_stages=self.load_mma_KV_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=self.tma_copy_KV_bytes,
            defer_sync=True,
        )
        gather_mma_QdO_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.gather_mma_QdO_mbar_ptr.data_ptr(),
            num_stages=self.gather_mma_QdO_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=self.tma_copy_QdO_bytes,
            defer_sync=True,
        )
        gather_row_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.gather_row_mbar_ptr.data_ptr(),
            num_stages=self.gather_row_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.threads_per_warp),
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
            self.load_kv_qdo(
                mma_S,
                mma_dP,
                tma_atom_K,
                tma_tensor_K,
                tma_atom_V,
                tma_tensor_V,
                tma_atom_Q,
                tma_tensor_Q,
                tma_atom_dO,
                tma_tensor_dO,
                sK,
                sV,
                sQ,
                sdO,
                mTaskMeta,
                mTaskQRows,
                row_lo,
                row_hi,
                load_mma_KV_pipeline,
                gather_mma_QdO_pipeline,
            )

        elif warp_idx == self.scalar_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_scalar)
            self.gather_scalars(
                mLSE,
                mDelta,
                mTaskMeta,
                mTaskQRows,
                mTaskQPos,
                sLSE,
                sDelta,
                sQRows,
                sQPos,
                row_lo,
                row_hi,
                log2_e,
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

    # ---- load warp: TMA K/V per bucket segment, TMA Q/dO boxes per task ----
    @cute.jit
    def load_kv_qdo(
        self,
        mma_S: cute.TiledMma,
        mma_dP: cute.TiledMma,
        tma_atom_K: cute.CopyAtom,
        tma_tensor_K: cute.Tensor,
        tma_atom_V: cute.CopyAtom,
        tma_tensor_V: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_tensor_Q: cute.Tensor,
        tma_atom_dO: cute.CopyAtom,
        tma_tensor_dO: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sQ: cute.Tensor,
        sdO: cute.Tensor,
        mTaskMeta: cute.Tensor,
        mTaskQRows: cute.Tensor,
        row_lo: Int32,
        row_hi: Int32,
        load_mma_KV_pipeline,
        gather_mma_QdO_pipeline,
    ):
        producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.load_mma_KV_stage)
        qdo_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.gather_mma_QdO_stage)

        tiler = (TILE_M, TILE_N, HEAD_DIM)
        thr_mma_S = mma_S.get_slice(0)
        thr_mma_dP = mma_dP.get_slice(0)

        # (box = (16 heads, 64 d), slot, d half, stage) views of sQ / sdO -- the bytes G1-G4 read.
        boxes_layout = cute.make_layout(
            (MAIN_HEADS_PER_INDEX, 64, self.query_chunk, 2, self.q_stage),
            stride=(64, 1, MAIN_HEADS_PER_INDEX * 64, 64 * TILE_N, TILE_N * HEAD_DIM),
        )
        sQ_boxes = cute.make_tensor(sQ.iterator, boxes_layout)
        sdO_boxes = cute.make_tensor(sdO.iterator, boxes_layout)
        gQ = cute.local_tile(tma_tensor_Q, (MAIN_HEADS_PER_INDEX, 64), (None, None, None))
        gdO = cute.local_tile(tma_tensor_dO, (MAIN_HEADS_PER_INDEX, 64), (None, None, None))
        tQsQ, tQgQ = cpasync.tma_partition(
            tma_atom_Q, 0, cute.make_layout(1), cute.group_modes(sQ_boxes, 0, 2), cute.group_modes(gQ, 0, 2)
        )
        tOsO, tOgO = cpasync.tma_partition(
            tma_atom_dO, 0, cute.make_layout(1), cute.group_modes(sdO_boxes, 0, 2), cute.group_modes(gdO, 0, 2)
        )

        row = row_lo
        while row < row_hi:
            b, index_head, kb, _valid = self._task_fields(mTaskMeta, row)
            is_seg_start = cutlass.Boolean(row == row_lo)
            if row > row_lo:
                is_seg_start = ~self._same_bucket(mTaskMeta, row, row - 1)
            if is_seg_start:
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

            # 8 slots x 2 d halves x 2 tensors, one transaction count; a row -1 slot lands as zeros.
            gather_mma_QdO_pipeline.producer_acquire(qdo_state)
            qdo_barrier = gather_mma_QdO_pipeline.producer_get_barrier(qdo_state)
            stage = qdo_state.index
            for slot in cutlass.range_constexpr(self.query_chunk):
                q_row = mTaskQRows[row, slot]
                for half in cutlass.range_constexpr(2):
                    cute.copy(
                        tma_atom_Q,
                        tQgQ[None, index_head, half, (q_row, b)],
                        tQsQ[None, slot, half, stage],
                        tma_bar_ptr=qdo_barrier,
                    )
                    cute.copy(
                        tma_atom_dO,
                        tOgO[None, index_head, half, (q_row, b)],
                        tOsO[None, slot, half, stage],
                        tma_bar_ptr=qdo_barrier,
                    )
            qdo_state.advance()
            row += 1

    # ---- scalar warp: per-row LSE / Delta / q rows / q positions, per tile ----
    @cute.jit
    def gather_scalars(
        self,
        mLSE: cute.Tensor,
        mDelta: cute.Tensor,
        mTaskMeta: cute.Tensor,
        mTaskQRows: cute.Tensor,
        mTaskQPos: cute.Tensor,
        sLSE: cute.Tensor,
        sDelta: cute.Tensor,
        sQRows: cute.Tensor,
        sQPos: cute.Tensor,
        row_lo: Int32,
        row_hi: Int32,
        log2_e: Float32,
        gather_row_pipeline,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        lane = tidx % self.threads_per_warp
        # lane -> (query slot, 4-head group): 8 slots x 4 groups x 4 heads = 128 folded rows
        slot = lane // 4
        part = lane - slot * 4
        row_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.gather_row_stage)

        mpp = Int32(self.main_per_index)
        row = row_lo
        while row < row_hi:
            b, index_head, _kb, valid = self._task_fields(mTaskMeta, row)
            gather_row_pipeline.producer_acquire(row_state)
            ridx = row_state.index
            # sLSE holds -lse*log2e; invalid slots carry -inf, so exp2 is +0 and P/dS vanish there.
            if slot < valid:
                q_row = mTaskQRows[row, slot]
                for k in cutlass.range_constexpr(4):
                    head = index_head * mpp + Int32(part * 4 + k)
                    r = slot * mpp + Int32(part * 4 + k)
                    sLSE[r, ridx] = Float32(0.0) - mLSE[b, head, q_row] * log2_e
                    sDelta[r, ridx] = mDelta[b, head, q_row]
            else:
                for k in cutlass.range_constexpr(4):
                    r = slot * mpp + Int32(part * 4 + k)
                    sLSE[r, ridx] = Float32(float("-inf"))
                    sDelta[r, ridx] = Float32(0.0)
            if lane < Int32(self.query_chunk):
                qrow = Int32(-1)
                qpos = Int32(-1)
                if lane < valid:
                    qrow = mTaskQRows[row, lane]
                    qpos = mTaskQPos[row, lane]
                sQRows[lane, ridx] = qrow
                sQPos[lane, ridx] = qpos
            cute.arch.fence_view_async_shared()
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

        # TMEM-resident A operands: P^T over the S columns, dS^T over the dP ones. Apply the column
        # offset to the *fragment* iterator in bf16 units: a recast_ptr(base + off) view drops it.
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

        # held slot for TMEM cols [128,256): dP(t) commits after G2, re-acquired before dQ takes it.
        mma_compute_dP_pipeline.producer_acquire(dp_state)
        # held slot for TMEM cols [0,128): S(t), then S(t+1).
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
                    tSrQ[None, None, kk, qdo_state.index],
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
                    tdPrdO[None, None, kk, qdo_state.index],
                    tdPtdP,
                )
                mma_dP.set(tcgen05.Field.ACCUMULATE, True)
            mma_compute_dP_pipeline.producer_commit(dp_state)
            dp_state.advance()

            if seg_first:
                # G1/G2 never touch the dV/dK columns, so segment s-1's flush overlaps segment s.
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
                        tdVrdOb[None, None, kk, qdo_state.index],
                        tdVtdV,
                    )
                    mma_dV.set(tcgen05.Field.ACCUMULATE, True)
                for j in cutlass.range_constexpr((c_hi - c_lo) * KK_PER_CHUNK):
                    kk = c_lo * KK_PER_CHUNK + j
                    cute.gemm(
                        mma_dK,
                        tdKtdK,
                        tdKrdS[None, None, kk],
                        tdKrQb[None, None, kk, qdo_state.index],
                        tdKtdK,
                    )
                    mma_dK.set(tcgen05.Field.ACCUMULATE, True)

            # Q/dO consumed after G4 (G1..G4 read them; G5 does not).
            gather_mma_QdO_pipeline.consumer_release(qdo_state)
            qdo_state.advance()

            # ---- G5: dQ^T = K^T . dS^T (cols 128..256, aliases dP^T/dS^T) ----
            # dP and the previous dQ are released; G4's dS^T reads precede G5 in issue order.
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

            # release every chunk stage: tcgen05.commit fires once G5 drained, freeing sPdS / TMEM.
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
        dp_idx = tidx - self.compute_warp_id[0] * self.threads_per_warp

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

        # P^T / dS^T publication: chunk c's 32 bf16 values pack into the 16 f32 columns it drained.
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
                rQpos = cute.make_rmem_tensor(tTR_cS.shape, cutlass.Int32)
                for i in cutlass.range_constexpr(cute.size(tTR_cS)):
                    n_col = Int32(c * CHUNK) + tTR_cS[i][1]
                    rLse[i] = sLSE[n_col, ridx]
                    rDelta[i] = sDelta[n_col, ridx]
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
                cond = rQpos.load() >= key_pos_thr
                tTR_rS.store(cute.where(cond, cute.math.exp2(v, fastmath=True), Float32(0.0)))
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
        dp_idx = tidx - self.reduce_warp_id[0] * self.threads_per_warp

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
            # Release the row slot before the dQ atomics: holding it across them stalls the gather.
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
            # Packed 16-bit atomics into the (T, Hq/2, D, 2) dQ pool: column pair (2j, 2j+1) of this
            # lane = heads (2m, 2m+1) of one query slot at this lane's d, adjacent in the pool -> one
            # 4-byte RED per pair, no lane exchange. The L2 evict-last hint keeps the pool lines
            # resident for the read-modify-writes of later tasks.
            policy = _l2_policy_evict_last()
            pk = cute.make_rmem_tensor((8,), cutlass.Uint32)
            PAIRS_PER_SLOT = self.main_per_index // 2
            hp0 = index_head * Int32(PAIRS_PER_SLOT)
            d_row = tTR_cAcc[0][0]
            for g in cutlass.range_constexpr(cute.size(tTR_r) // (2 * PAIRS_PER_SLOT)):
                for pp in cutlass.range_constexpr(PAIRS_PER_SLOT):
                    e = 2 * (g * PAIRS_PER_SLOT + pp)
                    if cutlass.const_expr(mdQ.element_type == cutlass.Float16):
                        pk[pp] = _pack_f16x2(tTR_r[e], tTR_r[e + 1])
                    else:
                        pk[pp] = _pack_bf16x2(tTR_r[e], tTR_r[e + 1])
                n_col0 = tTR_cAcc[2 * g * PAIRS_PER_SLOT][1]
                q_slot = n_col0 // mpp
                if q_slot < valid:
                    q_row = rQRow[q_slot]
                    base = mdQ.iterator + cute.crd2idx((b, q_row, hp0, d_row, Int32(0)), mdQ.layout)
                    for pp in cutlass.range_constexpr(PAIRS_PER_SLOT):
                        hoff = tTR_cAcc[2 * (g * PAIRS_PER_SLOT + pp)][1] - n_col0
                        dq_ptr = base + (hoff // 2) * Int32(2 * HEAD_DIM)
                        if cutlass.const_expr(mdQ.element_type == cutlass.Float16):
                            _red_add_f16x2_hint(dq_ptr, pk[pp], policy)
                        else:
                            _red_add_bf16x2_hint(dq_ptr, pk[pp], policy)

            # ---- segment flush: dV^T / dK^T ----
            is_seg_end = cutlass.Boolean(row + 1 >= row_hi)
            if row + 1 < row_hi:
                is_seg_end = ~self._same_bucket(mTaskMeta, row + 1, row)
            if is_seg_end:
                kv_head = index_head // Int32(self.index_heads_per_kv)
                key_base = kb * Int32(BLOCK_SIZE)
                mma_reduce_dKV_pipeline.consumer_wait(dkv_state)

                # dV / dK land as (M = key lanes, N = d columns). The RED path is sector-bound,
                # so each 4x4 block is transposed inside the quad first: one warp instruction
                # then writes 8 rows x 64 B = 16 full sectors instead of 32 half-filled ones.
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

        Lane q holds element (row q, block k) before and (row k, block q) after. ``c0``/``c1``
        are 4-wide lane-bit vectors, so each stage's slot choice lowers to FSEL, not a branch.
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
    softmax_scale: float,
) -> None:
    """Check device, dtype, 16-byte alignment, and THD shapes before launch."""
    names = ("q", "k_aligned", "v_aligned", "grad_out", "lse", "out")
    tensors = (q, k_aligned, v_aligned, grad_out, lse, out)
    if q.device.type != "cuda":
        raise ValueError("MiniMax M3 MSA backward requires CUDA tensors")
    if any(tensor.device != q.device for tensor in tensors):
        raise ValueError("all MiniMax M3 MSA backward tensors must be on one CUDA device")
    if torch.cuda.get_device_capability(q.device) != (10, 0):
        raise NotImplementedError("MiniMax M3 MSA backward requires an SM100 CUDA device")
    misaligned = [name for name, tensor in zip(names, tensors, strict=True) if tensor.data_ptr() % 16 != 0]
    if misaligned:
        raise ValueError(
            "MiniMax M3 MSA backward requires 16-byte-aligned storage for its compiled tensor ABI; "
            f"misaligned tensors={misaligned}."
        )

    for name, tensor in zip(names, tensors, strict=True):
        if name != "lse" and tensor.dtype != torch.bfloat16:
            raise TypeError(f"{name} must be BF16, got {tensor.dtype}")
    if lse.dtype != torch.float32:
        raise TypeError(f"lse must be FP32, got {lse.dtype}")

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
    if not math.isfinite(softmax_scale) or softmax_scale <= 0.0:
        raise ValueError(f"softmax_scale must be finite and positive, got {softmax_scale}")


def _compile_backward() -> Any:
    """Compile the backward once with dynamic token, workspace, and task counts.

    The fake tensors describe the head-major views ``_run_msa_backward`` builds;
    ``stride_order[i]`` is the rank of mode ``i``, ``0`` innermost. Returns an executable
    taking the kernel's positional arguments minus the trailing stream.
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
        make_fake_compact_tensor(
            _DQ_ACCUM_CUTLASS_DTYPE,
            (1, num_tokens, NUM_Q_HEADS // 2, HEAD_DIM, 2),
            stride_order=(4, 3, 2, 1, 0),
            assumed_align=16,
        ),  # 16-bit dQ pool
        rows_by_head(Float32, NUM_KV_HEADS, workspace_rows),
        rows_by_head(Float32, NUM_KV_HEADS, workspace_rows),
        make_fake_compact_tensor(Int32, (DESC_WORDS,), stride_order=(0,), assumed_align=16),  # CTA-walk descriptor
        Int32(1),
        Float32(1.0),
        make_fake_stream(use_tvm_ffi_env_stream=True),
    )
    return cute.compile(_MSABackwardSm100Kernel(), *fake_args, options="--enable-tvm-ffi")


def _round_up(n: int, m: int = 256) -> int:
    return (n + m - 1) // m * m


def _alloc_call_buffers(q_c: torch.Tensor, k_c: torch.Tensor, v_c: torch.Tensor, schedule: _MSABackwardSchedule):
    """One internal allocation per call (freed when the plan dies), carved into: the 16-bit dQ
    head-pair pool ``[T, Hq/2, D, 2]`` + the FP32 dK/dV pool (cleared by ``zero()``), the FP32
    ``delta [T, 64]``, and the int32 scratch / tables of the fused task build. One block of one
    size per call keeps the caching allocator from splitting and re-growing."""
    num_dk, num_dv = k_c.numel(), v_c.numel()
    dq_bytes = q_c.numel() * _DQ_ACCUM_TORCH_DTYPE.itemsize  # multiple of 256: keeps the FP32 pool 16-byte aligned
    pool_bytes = dq_bytes + (num_dk + num_dv) * 4
    delta_off = _round_up(pool_bytes)
    delta_bytes = q_c.shape[0] * NUM_Q_HEADS * 4
    scratch_off = _round_up(delta_off + delta_bytes)
    scratch_words, table_words = task_build_storage(schedule, int(q_c.shape[0]), int(k_c.shape[0]))
    tables_off = _round_up(scratch_off + scratch_words * 4)
    total = _round_up(tables_off + table_words * 4)
    raw = torch.empty(total, dtype=torch.uint8, device=q_c.device)
    pool = raw[:pool_bytes]
    dq_pool = pool[:dq_bytes].view(_DQ_ACCUM_TORCH_DTYPE).view(q_c.shape[0], NUM_Q_HEADS // 2, HEAD_DIM, 2)
    grad_pool = pool[dq_bytes:].view(torch.float32)
    delta = raw[delta_off : delta_off + delta_bytes].view(torch.float32).view(q_c.shape[0], NUM_Q_HEADS)
    scratch = raw[scratch_off : scratch_off + scratch_words * 4].view(torch.int32) if scratch_words else None
    tables = raw[tables_off : tables_off + table_words * 4].view(torch.int32) if table_words else None
    return (
        pool,
        dq_pool,
        grad_pool,
        grad_pool[:num_dk].view(k_c.shape),
        grad_pool[num_dk:].view(v_c.shape),
        delta,
        scratch,
        tables,
    )


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
    contiguous, 16-byte-aligned storage. Returns BF16 ``(dq, dk_aligned, dv_aligned)`` in
    those same layouts, zero outside the support. The kernel's head-major operands are
    strided views built here, so no transposed copies are made.

    The Delta preprocess and the pool clear are issued first (they depend only on the
    inputs) so the GPU is busy while the host issues the task build.
    """
    plan = _plan_msa_backward(
        q, k_aligned, v_aligned, grad_out, lse, out, schedule, softmax_scale=softmax_scale, build_tasks_now=False
    )
    plan.preprocess()
    plan.zero()
    plan.build_tasks()
    plan.launch_main()
    plan.cast()
    return plan.outputs()


class _MSABackwardPlan:
    """One backward call minus its launches: validated inputs, one internal buffer (dQ/dK/dV
    pools, delta, task scratch and tables) and the compiled executables (nothing compiles in the
    methods). ``zero`` and ``build_tasks`` must both run before ``launch_main``; ``build_tasks``
    sets ``tables``, whose ``desc`` carries the exact task count and the CTA walk that
    ``device_counts()`` reads back. A schedule without tasks yields zero gradients through the
    same launches. Methods are bound on access, so the plan holds no reference cycle and its
    buffers return to the caching allocator as soon as it is dropped."""

    def __init__(
        self,
        *,
        exe,
        args_before_tasks,
        args_after_tasks,
        out_c,
        grad_out_c,
        delta,
        pool,
        dq_pool,
        dq_bf16,
        grad_pool,
        grad_pool_bf16,
        num_dk,
        k_shape,
        v_shape,
        schedule,
        num_tokens,
        workspace_rows,
        num_sms,
        scratch,
        tables_buf,
        softmax_scale,
    ):
        self._exe = exe
        self._softmax_scale = float(softmax_scale)
        self._args_before_tasks = args_before_tasks
        self._args_after_tasks = args_after_tasks
        self._out_c, self._grad_out_c, self._delta = out_c, grad_out_c, delta
        self._pool, self._dq_pool, self._dq_bf16 = pool, dq_pool, dq_bf16
        self.grad_pool, self._grad_pool_bf16 = grad_pool, grad_pool_bf16
        self._num_dk, self._k_shape, self._v_shape = num_dk, k_shape, v_shape
        self._schedule, self._num_tokens, self._workspace_rows, self._num_sms = (
            schedule,
            num_tokens,
            workspace_rows,
            num_sms,
        )
        self._scratch, self._tables_buf = scratch, tables_buf
        self.tables = None

    def build_tasks(self):
        self.tables = build_backward_tasks(
            self._schedule,
            self._num_tokens,
            self._workspace_rows,
            num_sms=self._num_sms,
            scratch=self._scratch,
            tables=self._tables_buf,
        )

    def device_counts(self):
        return self.tables.device_counts()

    def preprocess(self):
        _run_msa_backward_preprocess(self._out_c, self._grad_out_c, self._delta)

    def zero(self):
        self._pool.zero_()

    def launch_main(self):
        tables = self.tables
        if tables is None:
            raise RuntimeError("build_tasks() must run before launch_main()")
        if tables.grid_launch <= 0:
            return  # no task can exist: the cleared pools already hold the zero gradients
        self._exe(
            *self._args_before_tasks,
            tables.task_meta,
            tables.task_qrows,
            tables.task_qpos,
            *self._args_after_tasks,
            tables.desc,
            Int32(tables.grid_launch),
            Float32(self._softmax_scale),
        )

    def cast(self):
        run_grad_finalize(self._dq_pool, self._dq_bf16, self.grad_pool, self._grad_pool_bf16)

    def outputs(self):
        return (
            self._dq_bf16,
            self._grad_pool_bf16[: self._num_dk].view(self._k_shape),
            self._grad_pool_bf16[self._num_dk :].view(self._v_shape),
        )


def _plan_msa_backward(q, k_aligned, v_aligned, grad_out, lse, out, schedule, *, softmax_scale, build_tasks_now=True):
    """Build a :class:`_MSABackwardPlan` (validation, one internal buffer, compiled executables);
    ``build_tasks_now=False`` leaves the task build to the caller."""
    _validate_inputs(q, k_aligned, v_aligned, grad_out, lse, out, softmax_scale=softmax_scale)
    q_c, k_c, v_c = q.detach().contiguous(), k_aligned.detach().contiguous(), v_aligned.detach().contiguous()
    grad_out_c, lse_c, out_c = grad_out.detach().contiguous(), lse.detach().contiguous(), out.detach().contiguous()
    device = q.device
    num_sms = _num_sms(device)
    num_tokens, workspace_rows = int(q_c.shape[0]), int(k_c.shape[0])
    num_dk = k_c.numel()
    pool, dq_pool, grad_pool, dk, dv, delta, scratch, tables_buf = _alloc_call_buffers(q_c, k_c, v_c, schedule)
    dq_bf16 = torch.empty_like(q_c)
    grad_pool_bf16 = torch.empty_like(grad_pool, dtype=torch.bfloat16)
    q_v, grad_out_v, k_v, v_v, dk_v, dv_v = (
        t.unsqueeze(0).transpose(1, 2) for t in (q_c, grad_out_c, k_c, v_c, dk, dv)
    )
    dq_v = dq_pool.unsqueeze(0)
    lse_v = lse_c.unsqueeze(0).transpose(1, 2)
    delta_v = delta.unsqueeze(0).transpose(1, 2)
    key = ("minimax-m3-msa-backward-sm100", torch.cuda.get_device_capability(device), q_c.dtype, _DQ_ACCUM)
    if key not in _COMPILE_CACHE:
        _COMPILE_CACHE[key] = _compile_backward()
    exe = _COMPILE_CACHE[key]
    preprocess_executable(device, out_c.dtype)
    grad_finalize_executable(device, dq_pool.dtype, interleaved=True)
    compile_task_build(device)
    plan = _MSABackwardPlan(
        exe=exe,
        args_before_tasks=(q_v, k_v, v_v, grad_out_v, lse_v, delta_v),
        args_after_tasks=(dq_v, dk_v, dv_v),
        out_c=out_c,
        grad_out_c=grad_out_c,
        delta=delta,
        pool=pool,
        dq_pool=dq_pool,
        dq_bf16=dq_bf16,
        grad_pool=grad_pool,
        grad_pool_bf16=grad_pool_bf16,
        num_dk=num_dk,
        k_shape=k_c.shape,
        v_shape=v_c.shape,
        schedule=schedule,
        num_tokens=num_tokens,
        workspace_rows=workspace_rows,
        num_sms=num_sms,
        scratch=scratch,
        tables_buf=tables_buf,
        softmax_scale=softmax_scale,
    )
    if build_tasks_now:
        plan.build_tasks()
    return plan
