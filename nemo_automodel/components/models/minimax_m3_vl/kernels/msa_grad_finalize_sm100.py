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

* dQ: the main kernel accumulates dQ with packed 16-bit atomics (fp16 or bf16) either into
  a head-pair interleaved pool ``[T, Hq/2, D, 2]`` -- de-interleaved here: pool row
  ``(t, hp)`` of 256 elements becomes rows ``2hp`` and ``2hp + 1`` of the BF16 ``[T, Hq, D]``
  gradient (16-byte loads, 8-byte stores) -- or into a plain ``[T, Hq, D]`` pool, which is
  cast in place (16-byte loads and stores).  One warp per 256-element pool row.
* dK/dV: the FP32 pool is cast to BF16 (2048 elements per CTA, 16-byte loads/stores).

Grid ``[max(dq_blocks, kv_blocks), 2]``: ``blockIdx.y == 0`` does 8 dQ rows, ``1`` does one
2048-element dK/dV chunk; both roles are predicated on their own extents.
"""

from typing import Any

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings import driver as cuda
from cutlass import Float32, Int32
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream

HEAD_DIM = 128
POOL_ROW = 2 * HEAD_DIM  # (d, e) pairs of one head pair
NUM_THREADS = 256
DQ_ROWS_PER_CTA = NUM_THREADS // 32  # one warp per pool row
KV_PER_CTA = NUM_THREADS * 8  # 8 fp32 per thread

_COMPILE_CACHE: dict[tuple[Any, ...], Any] = {}


class _MSAGradFinalizeSm100:
    def __init__(self, interleaved: bool):
        self.interleaved = interleaved  # pool row = (d, e) pairs of one head pair (else 256 plain elements)

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
        in_bits = mDQPool.element_type.width
        in_copy = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mDQPool.element_type, num_bits_per_copy=128)
        # lane c of warp r reads pool[r, 8c : 8c + 8] (16 B) and writes out[r, e, 4c : 4c + 4] (8 B)
        thr_layout = cute.make_ordered_layout((DQ_ROWS_PER_CTA, 32), order=(1, 0))
        tiled_in = cute.make_tiled_copy_tv(in_copy, thr_layout, cute.make_layout((1, 128 // in_bits)))
        if cutlass.const_expr(self.interleaved):
            out_copy = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=64)
            tiled_out = cute.make_tiled_copy_tv(out_copy, thr_layout, cute.make_layout((1, 4)))
        else:
            out_copy = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=128)
            tiled_out = cute.make_tiled_copy_tv(out_copy, thr_layout, cute.make_layout((1, 8)))
        kv_copy = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=128)
        kv_thr = cute.make_layout((1, NUM_THREADS))
        tiled_kv_in = cute.make_tiled_copy_tv(kv_copy, kv_thr, cute.make_layout((1, 8)))
        kv_out_copy = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=128)
        tiled_kv_out = cute.make_tiled_copy_tv(kv_out_copy, kv_thr, cute.make_layout((1, 8)))
        dq_blocks = cute.ceil_div(num_dq_rows, DQ_ROWS_PER_CTA)
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
            if bidx * Int32(DQ_ROWS_PER_CTA) < num_dq_rows:
                gIn = cute.local_tile(mDQPool, (DQ_ROWS_PER_CTA, POOL_ROW), (bidx, 0))
                thr_in = tiled_in.get_slice(tidx)
                tIn = thr_in.partition_S(gIn)
                frag = cute.make_rmem_tensor_like(tIn)
                row = bidx * Int32(DQ_ROWS_PER_CTA) + tidx // 32
                if row < num_dq_rows:
                    cute.copy(tiled_in, tIn, frag)
                    thr_out = tiled_out.get_slice(tidx)
                    if cutlass.const_expr(self.interleaved):
                        # frag[j] = pool[row, 8c + j] = (d = 4c + j // 2, e = j % 2)
                        frag_flat = cute.make_tensor(frag.iterator, cute.make_layout(cute.size(frag)))
                        frag2 = cute.logical_divide(frag_flat, cute.make_layout(2))  # (e, d-local)
                        for e in cutlass.range_constexpr(2):
                            gOut = cute.local_tile(mDQOut, (DQ_ROWS_PER_CTA, HEAD_DIM), (bidx, e))
                            tOut = thr_out.partition_D(gOut)
                            packed = cute.make_rmem_tensor_like(tOut)
                            packed_flat = cute.make_tensor(packed.iterator, cute.make_layout(cute.size(packed)))
                            packed_flat.store(frag2[e, None].load().to(cutlass.BFloat16))
                            cute.copy(tiled_out, packed, tOut)
                    else:
                        gOut = cute.local_tile(mDQOut, (DQ_ROWS_PER_CTA, POOL_ROW), (bidx, 0))
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


def grad_finalize_executable(device: torch.device, dq_dtype: torch.dtype, interleaved: bool) -> Any:
    """Compile (once per device capability, pool dtype and layout) and return the finalize executable."""
    key = ("minimax-m3-msa-grad-finalize-sm100", torch.cuda.get_device_capability(device), dq_dtype, interleaved)
    if key not in _COMPILE_CACHE:
        in_dtype = {torch.float16: cutlass.Float16, torch.bfloat16: cutlass.BFloat16}[dq_dtype]
        n_rows = cute.sym_int32(symbol="dq_rows")
        n_kv = cute.sym_int32(symbol="kv_blocks")
        fake_pool = make_fake_compact_tensor(in_dtype, (n_rows, POOL_ROW), stride_order=(1, 0), assumed_align=16)
        fake_out = make_fake_compact_tensor(cutlass.BFloat16, (n_rows, POOL_ROW), stride_order=(1, 0), assumed_align=16)
        fake_kv = make_fake_compact_tensor(Float32, (n_kv, KV_PER_CTA), stride_order=(1, 0), assumed_align=16)
        fake_kv_out = make_fake_compact_tensor(
            cutlass.BFloat16, (n_kv, KV_PER_CTA), stride_order=(1, 0), assumed_align=16
        )
        _COMPILE_CACHE[key] = cute.compile(
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
    return _COMPILE_CACHE[key]


def run_grad_finalize(dq_pool: torch.Tensor, dq_out: torch.Tensor, kv_pool: torch.Tensor, kv_out: torch.Tensor) -> None:
    """``dq_pool`` ``[T, Hq/2, D, 2]`` or ``[T, Hq, D]`` (fp16/bf16) -> ``dq_out [T, Hq, D]`` bf16; ``kv_pool [N]`` fp32 -> ``kv_out [N]`` bf16."""
    interleaved = dq_pool.dim() == 4
    if interleaved:
        T, half_heads, dim, two = dq_pool.shape
        assert dim == HEAD_DIM and two == 2 and dq_out.shape == (T, 2 * half_heads, dim)
    else:
        assert dq_out.shape == dq_pool.shape and dq_pool.shape[-1] == HEAD_DIM
    assert dq_out.dtype == torch.bfloat16 and dq_pool.numel() % POOL_ROW == 0
    assert dq_pool.is_contiguous() and dq_out.is_contiguous() and kv_pool.is_contiguous() and kv_out.is_contiguous()
    assert kv_pool.dtype == torch.float32 and kv_out.dtype == torch.bfloat16 and kv_pool.numel() == kv_out.numel()
    assert kv_pool.numel() % KV_PER_CTA == 0, kv_pool.numel()
    num_dq_rows = dq_pool.numel() // POOL_ROW
    num_kv_blocks = kv_pool.numel() // KV_PER_CTA
    grad_finalize_executable(dq_pool.device, dq_pool.dtype, interleaved)(
        dq_pool.view(num_dq_rows, POOL_ROW),
        dq_out.view(num_dq_rows, POOL_ROW),
        kv_pool.view(num_kv_blocks, KV_PER_CTA),
        kv_out.view(num_kv_blocks, KV_PER_CTA),
        Int32(num_dq_rows),
        Int32(num_kv_blocks),
    )
