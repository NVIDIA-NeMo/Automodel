# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Exact block classifications from QSA bitmaps; no dense token mask or CPU sync."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Uint32
from cutlass.cute.runtime import from_dlpack
from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch


@cute.kernel
def _classify(bits: cute.Tensor, kinds: cute.Tensor, sq: cutlass.Constexpr, sk: cutlass.Constexpr):
    """Classify exact 128-by-64 query/key tiles.

    Args:
        bits: Int32 CUDA [batch, sq, ceil(sk/32)] route bitmap.
        kinds: Writable int32 [batch, ceil(sq/128), ceil(sk/64)]:
            0 absent, 1 partial, 2 full.
        sq: Query length.
        sk: Key/value length.
    """
    qb, kb, bi = cute.arch.block_idx()
    lane = cute.arch.thread_idx()[0]
    any_bit = Uint32(0)
    all_bit = Uint32(0xFFFFFFFF)
    for r in cutlass.range_constexpr(4):
        qi = qb * 128 + lane + r * 32
        for w in cutlass.range_constexpr(2):
            wi = kb * 2 + w
            word = Uint32(0)
            if qi < sq and wi < bits.shape[2]:
                word = Uint32(bits[bi, qi, wi])
            any_bit = any_bit | word
            all_bit = all_bit & word
    for shift in cutlass.range_constexpr(5):
        delta = 1 << shift
        any_bit = any_bit | cute.arch.shuffle_sync_bfly(any_bit, delta)
        all_bit = all_bit & cute.arch.shuffle_sync_bfly(all_bit, delta)
    if lane == 0:
        kind = Int32(0)
        if any_bit != 0:
            kind = Int32(1)
            if all_bit == Uint32(0xFFFFFFFF):
                kind = Int32(2)
        kinds[bi, qb, kb] = kind


@cute.kernel
def _compact(
    kinds: cute.Tensor, mc: cute.Tensor, mi: cute.Tensor, fc: cute.Tensor, fi: cute.Tensor, transpose: cutlass.Constexpr
):
    """Compact tile IDs without reading any device scalar on the host.

    Args:
        kinds: Int32 CUDA [batch, query_blocks, key_blocks].
        mc: Writable masked counts [batch, 1, rows].
        mi: Writable masked indices [batch, 1, rows, columns]; only count entries
            per row are initialized or consumed by FlashAttention.
        fc: Writable full counts with mc's layout.
        fi: Writable full indices with mi's layout, likewise count-bounded.
        transpose: Reverse query/key block axes for backward metadata.
    """
    bx, bi, _ = cute.arch.block_idx()
    ti = cute.arch.thread_idx()[0]
    row = bx * 32 + ti
    if row < mc.shape[2]:
        p = Int32(0)
        f = Int32(0)
        for col in cutlass.range(mi.shape[3]):
            kind = Int32(0)
            if cutlass.const_expr(transpose):
                kind = kinds[bi, col, row]
            else:
                kind = kinds[bi, row, col]
            if kind == 1:
                mi[bi, 0, row, p] = col
                p += 1
            if kind == 2:
                fi[bi, 0, row, f] = col
                f += 1
        mc[bi, 0, row] = p
        fc[bi, 0, row] = f


@cute.jit
def _build(
    bits: cute.Tensor,
    kinds: cute.Tensor,
    fmc: cute.Tensor,
    fmi: cute.Tensor,
    ffc: cute.Tensor,
    ffi: cute.Tensor,
    bmc: cute.Tensor,
    bmi: cute.Tensor,
    bfc: cute.Tensor,
    bfi: cute.Tensor,
    sq: cutlass.Constexpr,
    sk: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    """Launch classification and forward/reverse compaction on one stream.

    Args:
        bits: Int32 CUDA [batch, sq, ceil(sk/32)].
        kinds: Scratch int32 [batch, query_blocks, key_blocks].
        fmc: Forward masked counts [batch, 1, query_blocks].
        fmi: Forward masked indices [batch, 1, query_blocks, key_blocks].
        ffc: Forward full counts with fmc's layout.
        ffi: Forward full indices with fmi's layout.
        bmc: Reverse masked counts [batch, 1, key_blocks].
        bmi: Reverse masked indices [batch, 1, key_blocks, query_blocks].
        bfc: Reverse full counts with bmc's layout.
        bfi: Reverse full indices with bmi's layout.
        sq: Query sequence length.
        sk: Key/value sequence length.
        stream: Current CUDA stream. All scratch/output tensors share its device.
    """
    _classify(bits, kinds, sq, sk).launch(
        grid=(kinds.shape[1], kinds.shape[2], kinds.shape[0]), block=(32, 1, 1), stream=stream
    )
    _compact(kinds, fmc, fmi, ffc, ffi, False).launch(
        grid=((fmc.shape[2] + 31) // 32, kinds.shape[0], 1), block=(32, 1, 1), stream=stream
    )
    _compact(kinds, bmc, bmi, bfc, bfi, True).launch(
        grid=((bmc.shape[2] + 31) // 32, kinds.shape[0], 1), block=(32, 1, 1), stream=stream
    )


# Process-local FIFO of compiled callables only; retain at most 32 shapes.
_CACHE = {}


def build_block_metadata(
    bits: torch.Tensor, sq: int, sk: int
) -> tuple[BlockSparseTensorsTorch, BlockSparseTensorsTorch]:
    """Classify and compact 128-query by 64-key tiles on the current CUDA stream.

    Args:
        bits: Contiguous int32 CUDA [batch, sq, ceil(sk/32)] route bitmap.
        sq: Query sequence length.
        sk: Key/value sequence length.

    Returns:
        Forward and transposed-backward block records. Each contains masked/full
        counts [batch, 1, rows] and masked/full indices [batch, 1, rows, columns].
        Forward rows/columns are ceil(sq/128), ceil(sk/64); backward reverses them.
        All tensors are independently allocated int32 CUDA storage.
    """
    b = bits.shape[0]
    nq = (sq + 127) // 128
    nk = (sk + 63) // 64

    def empty(*shape):
        return torch.empty(shape, device=bits.device, dtype=torch.int32)

    kinds = empty(b, nq, nk)

    def allocate(rows, cols):
        return [empty(b, 1, rows), empty(b, 1, rows, cols), empty(b, 1, rows), empty(b, 1, rows, cols)]

    fwd = allocate(nq, nk)
    bwd = allocate(nk, nq)
    ts = [bits, kinds, *fwd, *bwd]
    cs = [from_dlpack(t, assumed_align=4, enable_tvm_ffi=True) for t in ts]
    stream = cuda.CUstream(torch.cuda.current_stream(bits.device).cuda_stream)
    key = (b, sq, sk, bits.device.index)
    if key not in _CACHE:
        if len(_CACHE) >= 32:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[key] = cute.compile(_build, *cs, sq, sk, stream, options="--enable-tvm-ffi --gpu-arch sm_90a")
    _CACHE[key](*cs, stream)
    return BlockSparseTensorsTorch(*fwd, block_size=(128, 64)), BlockSparseTensorsTorch(*bwd, block_size=(128, 64))
