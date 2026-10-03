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

"""Gradient finalize for the MiniMax M3 MSA SM100 backward (one launch).

* dQ: the main kernel accumulates dQ with packed 16-bit atomics (``DQ_ACCUM_DTYPE``) either into
  a head-pair interleaved pool ``[T, Hq/2, D, 2]`` -- de-interleaved here: pool row
  ``(t, hp)`` of 256 elements becomes rows ``2hp`` and ``2hp + 1`` of the BF16 ``[T, Hq, D]``
  gradient (16-byte loads, 8-byte stores) -- or into a plain ``[T, Hq, D]`` pool, which is
  cast in place (16-byte loads and stores).  One warp per 256-element pool row.
* dK/dV: the FP32 pool is cast to BF16 (2048 elements per CTA, 16-byte loads/stores).

Grid ``[max(dq_blocks, kv_blocks), 2]``: ``blockIdx.y == 0`` does 8 dQ rows, ``1`` does one
2048-element dK/dV chunk; both roles are predicated on their own extents.
"""

from functools import lru_cache
from typing import Any

import torch

from nemo_automodel.components.models.minimax_m3_vl.kernels import require_cute_dsl
from nemo_automodel.components.models.minimax_m3_vl.kernels.msa_schedule import HEAD_DIM

# Bind the CuTe DSL only after proving it is importable, so a host without the msa extra sees
# UnavailableError here instead of ModuleNotFoundError from the imports below.
require_cute_dsl()

import cutlass
import cutlass.cute as cute
from cuda.bindings import driver as cuda
from cutlass import Float32, Int32
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream

POOL_ROW = 2 * HEAD_DIM  # (d, e) pairs of one head pair
NUM_THREADS = 256
DQ_ROWS_PER_CTA = NUM_THREADS // 32  # one warp per pool row
KV_PER_CTA = NUM_THREADS * 8  # 8 fp32 per thread


class _MSAGradFinalizeSm100:
    def __init__(self, interleaved: bool):
        self.interleaved = interleaved  # pool row = (d, e) pairs of one head pair (else 256 plain elements)
        # 16 values per thread on the interleaved path so each of its two per-e stores is 128-bit
        self.vals_per_thread = 16 if interleaved else 8
        self.rows_per_cta = NUM_THREADS // (POOL_ROW // self.vals_per_thread)

    @cute.jit
    def __call__(
        self,
        mDQPool: cute.Tensor,  # [R, 256] fp16/bf16
        mDQOut: cute.Tensor,  # [R, 256] bf16 (interleaved: columns [128e, 128e + 128) = head 2hp + e)
        mKVPool: cute.Tensor,  # [Nkv / 2048, 2048] fp32
        mKVOut: cute.Tensor,  # [Nkv / 2048, 2048] bf16
        num_dq_rows: Int32,
        num_kv_blocks: Int32,
        stream: cuda.CUstream,
    ):
        in_copy = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mDQPool.element_type, num_bits_per_copy=128)
        # A pool row holds (d, e) pairs, so one head's 8 consecutive outputs come from 16 consecutive
        # pool elements.  Giving the interleaved path 16 lanes per row (16 values each) instead of 32
        # (8 each) makes both per-e stores 128-bit; at 8 values per thread they were 64-bit, i.e. the
        # store side ran at half the load side's width.
        rows = self.rows_per_cta
        lanes = POOL_ROW // self.vals_per_thread
        thr_layout = cute.make_ordered_layout((rows, lanes), order=(1, 0))
        tiled_in = cute.make_tiled_copy_tv(in_copy, thr_layout, cute.make_layout((1, self.vals_per_thread)))
        out_copy = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=128)
        tiled_out = cute.make_tiled_copy_tv(
            out_copy, thr_layout, cute.make_layout((1, self.vals_per_thread // (2 if self.interleaved else 1)))
        )
        kv_copy = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=128)
        kv_thr = cute.make_layout((1, NUM_THREADS))
        tiled_kv_in = cute.make_tiled_copy_tv(kv_copy, kv_thr, cute.make_layout((1, 8)))
        kv_out_copy = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=128)
        tiled_kv_out = cute.make_tiled_copy_tv(kv_out_copy, kv_thr, cute.make_layout((1, 8)))
        dq_blocks = cute.ceil_div(num_dq_rows, self.rows_per_cta)
        blocks = dq_blocks
        if num_kv_blocks > blocks:
            blocks = num_kv_blocks
        self.kernel(
            mDQPool, mDQOut, mKVPool, mKVOut, num_dq_rows, num_kv_blocks, tiled_in, tiled_out, tiled_kv_in, tiled_kv_out
        ).launch(grid=[blocks, 2, 1], block=[NUM_THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(
        self,
        mDQPool: cute.Tensor,
        mDQOut: cute.Tensor,
        mKVPool: cute.Tensor,
        mKVOut: cute.Tensor,
        num_dq_rows: Int32,
        num_kv_blocks: Int32,
        tiled_in: cute.TiledCopy,
        tiled_out: cute.TiledCopy,
        tiled_kv_in: cute.TiledCopy,
        tiled_kv_out: cute.TiledCopy,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, role, _ = cute.arch.block_idx()
        if role == 0:
            if bidx * Int32(self.rows_per_cta) < num_dq_rows:
                gIn = cute.local_tile(mDQPool, (self.rows_per_cta, POOL_ROW), (bidx, 0))
                thr_in = tiled_in.get_slice(tidx)
                tIn = thr_in.partition_S(gIn)
                frag = cute.make_rmem_tensor_like(tIn)
                row = bidx * Int32(self.rows_per_cta) + tidx // Int32(POOL_ROW // self.vals_per_thread)
                if row < num_dq_rows:
                    cute.copy(tiled_in, tIn, frag)
                    thr_out = tiled_out.get_slice(tidx)
                    if cutlass.const_expr(self.interleaved):
                        # frag[j] = pool[row, 8c + j] = (d = 4c + j // 2, e = j % 2)
                        frag_flat = cute.make_tensor(frag.iterator, cute.make_layout(cute.size(frag)))
                        frag2 = cute.logical_divide(frag_flat, cute.make_layout(2))  # (e, d-local)
                        for e in cutlass.range_constexpr(2):
                            gOut = cute.local_tile(mDQOut, (self.rows_per_cta, HEAD_DIM), (bidx, e))
                            tOut = thr_out.partition_D(gOut)
                            packed = cute.make_rmem_tensor_like(tOut)
                            packed_flat = cute.make_tensor(packed.iterator, cute.make_layout(cute.size(packed)))
                            packed_flat.store(frag2[e, None].load().to(cutlass.BFloat16))
                            cute.copy(tiled_out, packed, tOut)
                    else:
                        gOut = cute.local_tile(mDQOut, (self.rows_per_cta, POOL_ROW), (bidx, 0))
                        tOut = thr_out.partition_D(gOut)
                        packed = cute.make_rmem_tensor_like(tOut)
                        packed.store(frag.load().to(cutlass.BFloat16))
                        cute.copy(tiled_out, packed, tOut)
        else:
            if bidx < num_kv_blocks:
                gKV = cute.local_tile(mKVPool, (1, KV_PER_CTA), (bidx, 0))
                gKVOut = cute.local_tile(mKVOut, (1, KV_PER_CTA), (bidx, 0))
                thr_kv_in = tiled_kv_in.get_slice(tidx)
                thr_kv_out = tiled_kv_out.get_slice(tidx)
                tKV = thr_kv_in.partition_S(gKV)
                tKVOut = thr_kv_out.partition_D(gKVOut)
                frag = cute.make_rmem_tensor_like(tKV)
                cute.copy(tiled_kv_in, tKV, frag)
                out = cute.make_rmem_tensor_like(tKVOut)
                out.store(frag.load().to(cutlass.BFloat16))
                cute.copy(tiled_kv_out, out, tKVOut)


@lru_cache(maxsize=None)
def _compile(dq_dtype: torch.dtype, interleaved: bool) -> Any:
    """Compile the finalize once per dQ pool dtype and layout."""
    in_dtype = {torch.float16: cutlass.Float16, torch.bfloat16: cutlass.BFloat16}[dq_dtype]
    n_rows = cute.sym_int32(symbol="dq_rows")
    n_kv = cute.sym_int32(symbol="kv_blocks")
    fake_pool = make_fake_compact_tensor(in_dtype, (n_rows, POOL_ROW), stride_order=(1, 0), assumed_align=16)
    fake_out = make_fake_compact_tensor(cutlass.BFloat16, (n_rows, POOL_ROW), stride_order=(1, 0), assumed_align=16)
    fake_kv = make_fake_compact_tensor(Float32, (n_kv, KV_PER_CTA), stride_order=(1, 0), assumed_align=16)
    fake_kv_out = make_fake_compact_tensor(cutlass.BFloat16, (n_kv, KV_PER_CTA), stride_order=(1, 0), assumed_align=16)
    return cute.compile(
        _MSAGradFinalizeSm100(interleaved),
        fake_pool,
        fake_out,
        fake_kv,
        fake_kv_out,
        Int32(0),
        Int32(0),
        make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def run_grad_finalize(dq_pool: torch.Tensor, dq_out: torch.Tensor, kv_pool: torch.Tensor, kv_out: torch.Tensor) -> None:
    """Cast the accumulation pools to the BF16 gradients in one launch.

    Args:
        dq_pool: Contiguous 16-bit ``[T, Hq/2, D, 2]`` head-pair pool, or a plain ``[T, Hq, D]`` pool.
        dq_out: Contiguous BF16 ``[T, Hq, D]`` written in place.
        kv_pool: Contiguous FP32 ``[N]`` dK/dV pool, ``N`` a multiple of 2048.
        kv_out: Contiguous BF16 ``[N]`` written in place.
    """
    num_dq_rows = dq_pool.numel() // POOL_ROW
    num_kv_blocks = kv_pool.numel() // KV_PER_CTA
    _compile(dq_pool.dtype, dq_pool.dim() == 4)(
        dq_pool.view(num_dq_rows, POOL_ROW),
        dq_out.view(num_dq_rows, POOL_ROW),
        kv_pool.view(num_kv_blocks, KV_PER_CTA),
        kv_out.view(num_kv_blocks, KV_PER_CTA),
        Int32(num_dq_rows),
        Int32(num_kv_blocks),
    )
