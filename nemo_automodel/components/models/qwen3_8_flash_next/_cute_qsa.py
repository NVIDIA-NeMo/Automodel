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

"""SM90 route bitmaps and first-order FlashAttention CuTe autograd.

This optional implementation is loaded only through safe_import in cute_qsa.
FlashAttention owns the attention arithmetic; this model owns route-set masks.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op
from flash_attn.cute import utils
from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch
from flash_attn.cute.interface import _flash_attn_bwd, _flash_attn_fwd
from torch.autograd.function import once_differentiable

from nemo_automodel.components.models.qwen3_8_flash_next._cute_qsa_metadata import build_block_metadata


@dsl_user_op
def _atomic_or_shared(ptr: cute.Pointer, value: Uint32, *, loc=None, ip=None):
    """Atomically OR one uint32 value into shared-memory storage at ptr."""
    llvm.inline_asm(
        T.i32(),
        [Int32(ptr.toint()).ir_value(loc=loc, ip=ip), value.ir_value(loc=loc, ip=ip)],
        "atom.shared.or.b32 $0, [$1], $2;",
        "=r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cute.kernel
def _route_bits_kernel(routes: cute.Tensor, bits: cute.Tensor, kv_len: cutlass.Constexpr):
    """Build an exact bitmap for each query row.

    Args:
        routes: Contiguous signed CUDA [batch, queries, routes].
        bits: Output int32 CUDA [batch, queries, ceil(kv_len/32)], fully written.
        kv_len: Physical key/value sequence length.
    """
    qi, bi, _ = cute.arch.block_idx()
    ti, _, _ = cute.arch.thread_idx()
    words = cute.size(bits.shape[2])
    smem = cutlass.utils.SmemAllocator()
    shared = smem.allocate_tensor(Uint32, cute.make_layout(words), 16)
    for wi in cutlass.range(ti, words, 128):
        shared[wi] = Uint32(0)
    cute.arch.sync_threads()
    for ri in cutlass.range(ti, routes.shape[2], 128):
        token = routes[bi, qi, ri]
        if token >= 0 and token < kv_len:
            token32 = Int32(token)
            _atomic_or_shared(shared.iterator + token32 // 32, Uint32(1) << (token32 % 32))
    cute.arch.sync_threads()
    for wi in cutlass.range(ti, words, 128):
        bits[bi, qi, wi] = Int32(shared[wi])


@cute.jit
def _route_bits(routes: cute.Tensor, bits: cute.Tensor, kv_len: cutlass.Constexpr, stream: cuda.CUstream):
    """Launch route deduplication on the supplied stream.

    Args:
        routes: Contiguous signed CUDA [batch, queries, routes].
        bits: Writable int32 CUDA [batch, queries, ceil(kv_len/32)].
        kv_len: Physical key/value sequence length.
        stream: Current PyTorch CUDA stream.
    """
    _route_bits_kernel(routes, bits, kv_len).launch(
        grid=(routes.shape[1], routes.shape[0], 1),
        block=(128, 1, 1),
        stream=stream,
    )


# Process-local FIFO of compiled callables only, never tensors. Bounded to 32 shapes.
_BIT_CACHE = {}


def routes_to_bits(routes: torch.Tensor, kv_len: int) -> torch.Tensor:
    """Deduplicate routes without host readback.

    Args:
        routes: CUDA signed IDs [batch, query_sequence, routes], arbitrary strides.
        kv_len: Physical key/value sequence length.

    Returns:
        Independent int32 CUDA bitmap [batch, query_sequence, ceil(kv_len/32)].
    """
    routes = routes.contiguous()
    bits = torch.empty((*routes.shape[:2], (kv_len + 31) // 32), device=routes.device, dtype=torch.int32)
    cr = from_dlpack(routes.detach(), assumed_align=4, enable_tvm_ffi=True)
    cb = from_dlpack(bits, assumed_align=16, enable_tvm_ffi=True)
    stream = cuda.CUstream(torch.cuda.current_stream(routes.device).cuda_stream)
    key = (tuple(routes.shape), routes.dtype, kv_len, routes.device.index)
    if key not in _BIT_CACHE:
        if len(_BIT_CACHE) >= 32:
            _BIT_CACHE.pop(next(iter(_BIT_CACHE)))
        _BIT_CACHE[key] = cute.compile(
            _route_bits, cr, cb, kv_len, stream, options="--enable-tvm-ffi --gpu-arch sm_90a"
        )
    _BIT_CACHE[key](cr, cb, stream)
    bits.__leading_dim__ = 2
    bits.__assumed_align__ = 16
    return bits


@cute.jit
def _qsa_mask(batch, head, m_idx, n_idx, seqlen_info, aux_tensors):
    """Evaluate FlashAttention's coordinate-SSA mask callback.

    Args:
        batch: Scalar SSA batch coordinate, indexed at [0].
        head: Scalar SSA head coordinate; all heads share routing.
        m_idx: Scalar SSA query coordinate, indexed at [0].
        n_idx: Scalar SSA physical key coordinate, indexed at [0].
        seqlen_info: FlashAttention query/key extent metadata.
        aux_tensors: One int32 bitmap [batch, queries, ceil(keys/32)].

    Returns:
        Boolean scalar SSA indicating exact route-set membership.
    """
    bits = aux_tensors[0]
    mi = cutlass.min(m_idx[0], seqlen_info.seqlen_q - 1)
    ni = cutlass.min(n_idx[0], seqlen_info.seqlen_k - 1)
    word = Uint32(bits[batch[0], mi, ni // 32])
    selected = ((word >> (ni % 32)) & Uint32(1)) != Uint32(0)
    valid = selected & (m_idx[0] < seqlen_info.seqlen_q) & (n_idx[0] < seqlen_info.seqlen_k)
    return utils.scalar_to_ssa(valid, cutlass.Boolean)


class _QsaSparse(torch.autograd.Function):
    """Own saved attention state and block metadata through first-order backward."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bits: torch.Tensor,
        scale: float,
        fwd_meta: BlockSparseTensorsTorch,
        bwd_meta: BlockSparseTensorsTorch,
    ) -> torch.Tensor:
        """Save independently owned bitmap/metadata and run sparse attention.

        Args:
            ctx: Autograd state; retains inputs and metadata until backward.
            q: BF16 CUDA [batch, queries, query_heads, 256].
            k: BF16 CUDA [batch, keys, kv_heads, 256].
            v: BF16 CUDA tensor with k's layout.
            bits: Int32 CUDA [batch, queries, ceil(keys/32)].
            scale: QK score multiplier.
            fwd_meta: Block counts [batch, 1, query_blocks] and indices
                [batch, 1, query_blocks, key_blocks], separately for full/masked.
            bwd_meta: Same metadata with query/key block axes transposed.

        Returns:
            Independent BF16 CUDA [batch, queries, query_heads, 256].
        """
        out, lse, _p, _m = _flash_attn_fwd(
            q,
            k,
            v,
            softmax_scale=scale,
            mask_mod=_qsa_mask,
            aux_tensors=[bits],
            block_sparse_tensors=fwd_meta,
            tile_mn=(128, 64),
            pack_gqa=False,
            return_lse=True,
        )
        ctx.save_for_backward(q, k, v, bits, out, lse)
        ctx.scale = scale
        ctx.bwd_meta = bwd_meta
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx: torch.autograd.function.FunctionCtx, dout: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        """Differentiate Q/K/V with the saved exact routes.

        Args:
            ctx: Saved Q/K/V, bitmap, output [batch, queries, query_heads, 256],
                FP32 LSE [batch, query_heads, queries], and reverse block metadata.
            dout: BF16 CUDA [batch, queries, query_heads, 256] output gradient.

        Returns:
            Gradients matching Q/K/V shapes, dtypes and devices, followed by four
            None entries for nondifferentiable bitmap, scale and metadata inputs.
        """
        q, k, v, bits, out, lse = ctx.saved_tensors
        bits.__leading_dim__ = 2
        bits.__assumed_align__ = 16
        with torch.cuda.device(q.device):
            dq, dk, dv = _flash_attn_bwd(
                q,
                k,
                v,
                out,
                dout,
                lse,
                softmax_scale=ctx.scale,
                mask_mod=_qsa_mask,
                aux_tensors=[bits],
                block_sparse_tensors=ctx.bwd_meta,
            )
        return dq, dk, dv, None, None, None, None


def qsa_cutedsl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    selected_token_ids: torch.Tensor,
    *,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Execute the validated contract of cute_qsa.cute_sparse_gqa_attention.

    Args:
        query: BF16 CUDA [batch, query_sequence, query_heads, 256].
        key: BF16 CUDA [batch, kv_sequence, kv_heads, 256].
        value: BF16 CUDA tensor with key's layout.
        selected_token_ids: Signed CUDA [batch, query_sequence, routes].
        softmax_scale: Validated finite positive scale.

    Returns:
        Independent BF16 CUDA [batch, query_sequence, query_heads, 256].
    """
    bits = routes_to_bits(selected_token_ids, key.shape[1])
    fwd_meta, bwd_meta = build_block_metadata(bits, query.shape[1], key.shape[1])
    scale = query.shape[-1] ** -0.5 if softmax_scale is None else float(softmax_scale)
    return _QsaSparse.apply(query, key, value, bits, scale, fwd_meta, bwd_meta)
