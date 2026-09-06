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

"""Delta preprocess for the MiniMax M3 MSA SM100 backward.

Computes ``delta[t, h] = sum_d out[t, h, d] * grad_out[t, h, d]`` in FP32 from
BF16 inputs, which the backward subtracts from ``dP`` to form ``dS``. One CTA of
256 threads reduces a ``[128, 128]`` tile of one head; rows past ``T`` are
predicated off.

Adapted from FlashAttention's CuTe DSL backward preprocess
(``flash_attn/cute/flash_bwd_preprocess.py``, Copyright (c) 2025, Jay Shah,
Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao;
BSD-3-Clause), specialized to the fixed ``[T, 64, 128]`` MSA layout.
"""

from typing import Any

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings import driver as cuda
from cutlass import Float32
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream

HEAD_DIM = 128
NUM_Q_HEADS = 64
TILE_M = 128
NUM_THREADS = 256
COPY_BITS = 128
COPY_ELEMS = COPY_BITS // cutlass.BFloat16.width  # 8 BF16 elements per thread copy
THREADS_PER_ROW = HEAD_DIM // COPY_ELEMS  # 16 lanes cover one 128-element row

_COMPILE_CACHE: dict[tuple[Any, ...], Any] = {}


# Adapted from https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/flash_bwd_preprocess.py
class _MSABackwardPreprocessSm100:
    """Row-wise ``sum(O * dO)`` over head_dim for the THD contract."""

    @cute.jit
    def __call__(
        self,
        mO: cute.Tensor,  # [T, 64, 128] bf16
        mdO: cute.Tensor,  # [T, 64, 128] bf16
        mDelta: cute.Tensor,  # [T, 64] fp32
        # Keep the stream last: with --enable-tvm-ffi it is the TVM FFI environment stream.
        stream: cuda.CUstream,
    ):
        if cutlass.const_expr(mO.element_type != cutlass.BFloat16 or mdO.element_type != cutlass.BFloat16):
            raise TypeError("O and dO must be BFloat16")
        if cutlass.const_expr(mDelta.element_type != Float32):
            raise TypeError("delta must be Float32")

        copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=COPY_BITS)
        # Thread (r, c) -> row r, columns [8c, 8c + 8): a row is 16 consecutive lanes.
        thr_layout = cute.make_ordered_layout((NUM_THREADS // THREADS_PER_ROW, THREADS_PER_ROW), order=(1, 0))
        val_layout = cute.make_layout((1, COPY_ELEMS))
        tiled_copy = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)

        num_row_blocks = cute.ceil_div(mO.shape[0], TILE_M)
        self.kernel(mO, mdO, mDelta, tiled_copy).launch(
            grid=[num_row_blocks, NUM_Q_HEADS, 1],
            block=[NUM_THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mO: cute.Tensor,
        mdO: cute.Tensor,
        mDelta: cute.Tensor,
        tiled_copy: cute.TiledCopy,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        m_block, head, _ = cute.arch.block_idx()
        row_limit = mO.shape[0] - m_block * TILE_M

        mO_head = mO[None, head, None]
        mdO_head = mdO[None, head, None]
        mDelta_head = mDelta[None, head]

        tile_shape = (TILE_M, HEAD_DIM)
        gO = cute.local_tile(mO_head, tile_shape, (m_block, 0))
        gdO = cute.local_tile(mdO_head, tile_shape, (m_block, 0))
        gDelta = cute.local_tile(mDelta_head, (TILE_M,), (m_block,))

        thr_copy = tiled_copy.get_slice(tidx)
        # (CPY_ATOM, CPY_M, CPY_K) = ((8), 8, 1): 8 row iterations of 8 elements.
        tOgO = thr_copy.partition_S(gO)
        tOgdO = thr_copy.partition_S(gdO)
        tile_coords = cute.make_identity_tensor(tile_shape)
        tOcO = thr_copy.partition_S(tile_coords)
        # Thread 0's coordinates are compile-time constants, so shifting the limit
        # by this thread's row offset keeps the predicate cheap.
        t0OcO = tiled_copy.get_slice(0).partition_S(tile_coords)
        row_offset = tOcO[0, 0, 0][0]

        tOrO = cute.make_rmem_tensor_like(tOgO)
        tOrdO = cute.make_rmem_tensor_like(tOgdO)
        for m in cutlass.range_constexpr(cute.size(tOrO, mode=[1])):
            if t0OcO[0, m, 0][0] < row_limit - row_offset:
                cute.copy(tiled_copy, tOgO[None, m, None], tOrO[None, m, None])
                cute.copy(tiled_copy, tOgdO[None, m, None], tOrdO[None, m, None])

        partial = (tOrO.load().to(Float32) * tOrdO.load().to(Float32)).reduce(
            cute.ReductionOp.ADD, init_val=0.0, reduction_profile=(0, None, 1)
        )
        row_sums = cute.make_rmem_tensor(cute.size(tOrO, mode=[1]), Float32)
        row_sums.store(partial)
        # a butterfly over offsets 1, 2, 4, 8 gives every lane the full row sum
        for m in cutlass.range_constexpr(cute.size(row_sums)):
            value = row_sums[m]
            for shift in cutlass.range_constexpr(4):
                value = value + cute.arch.shuffle_sync_bfly(value, offset=1 << shift)
            row_sums[m] = value

        if tOcO[0, 0, 0][1] == 0:
            for m in cutlass.range_constexpr(cute.size(row_sums)):
                row = tOcO[0, m, 0][0]
                if row < row_limit:
                    gDelta[row] = row_sums[m]


def _run_msa_backward_preprocess(
    out: torch.Tensor, grad_out: torch.Tensor, delta: torch.Tensor | None = None
) -> torch.Tensor:
    """Reduce contiguous, 16-byte-aligned BF16 rows to FP32 ``[T, 64]`` delta (into ``delta`` when given)."""
    if out.device.type != "cuda" or grad_out.device != out.device:
        raise ValueError("MiniMax M3 MSA delta preprocess requires CUDA tensors on one device")
    if out.dtype != torch.bfloat16 or grad_out.dtype != torch.bfloat16:
        raise ValueError(f"out and grad_out must be BF16, got {out.dtype} and {grad_out.dtype}")
    if out.ndim != 3 or out.shape[1] != NUM_Q_HEADS or out.shape[2] != HEAD_DIM or out.shape[0] <= 0:
        raise ValueError(f"out must have shape [T, {NUM_Q_HEADS}, {HEAD_DIM}] with T > 0, got {tuple(out.shape)}")
    if grad_out.shape != out.shape:
        raise ValueError(f"grad_out must match out, got {tuple(grad_out.shape)} vs {tuple(out.shape)}")
    if not out.is_contiguous() or not grad_out.is_contiguous():
        raise ValueError("out and grad_out must be contiguous")
    if out.data_ptr() % 16 != 0 or grad_out.data_ptr() % 16 != 0:
        raise ValueError("out and grad_out must have 16-byte-aligned storage")

    if delta is None:
        delta = torch.empty((out.shape[0], NUM_Q_HEADS), dtype=torch.float32, device=out.device)
    elif delta.shape != (out.shape[0], NUM_Q_HEADS) or delta.dtype != torch.float32 or not delta.is_contiguous():
        raise ValueError(
            f"delta must be a contiguous FP32 [T, {NUM_Q_HEADS}] tensor, got {tuple(delta.shape)} {delta.dtype}"
        )
    preprocess_executable(out.device, out.dtype)(out, grad_out, delta)
    return delta


def preprocess_executable(device: torch.device, dtype: torch.dtype) -> Any:
    """Compile (once per device capability and dtype) and return the delta executable."""
    key = ("minimax-m3-msa-backward-preprocess-sm100", torch.cuda.get_device_capability(device), dtype)
    if key not in _COMPILE_CACHE:
        num_tokens = cute.sym_int32(symbol="num_tokens")
        # stride_order[i] is the rank of mode i, 0 = innermost: row-major THD.
        fake_rows = make_fake_compact_tensor(
            cutlass.BFloat16, (num_tokens, NUM_Q_HEADS, HEAD_DIM), stride_order=(2, 1, 0), assumed_align=16
        )
        fake_delta = make_fake_compact_tensor(Float32, (num_tokens, NUM_Q_HEADS), stride_order=(1, 0), assumed_align=16)
        _COMPILE_CACHE[key] = cute.compile(
            _MSABackwardPreprocessSm100(),
            fake_rows,
            fake_rows,
            fake_delta,
            make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    return _COMPILE_CACHE[key]
