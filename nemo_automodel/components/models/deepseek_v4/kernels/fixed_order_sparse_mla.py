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

"""Bounded, fixed-order reduction of the existing TileLang backward's partials.

Each partial has one writer. Stable key sorting preserves query/head/slot order;
32-entry reduction trees and sequential query chunks define the accumulation
order independently of GPU scheduling. No floating-point atomics are used.
"""

import torch

from nemo_automodel.components.models.deepseek_v4.kernels import tilelang_sparse_mla_bwd as backward
from nemo_automodel.components.models.deepseek_v4.kernels._tilelang import T, tilelang


@tilelang.jit
def _reduce_kv(D: int):
    """Build a fixed-tree reducer reusable across batch, edge and KV lengths."""
    B = T.dynamic("batch")
    E = T.dynamic("edges")
    N = T.dynamic("keys")

    @T.prim_func
    def kernel(
        partials: T.Tensor([B, E, D], T.float32),
        order: T.Tensor([B, E], T.int64),
        offsets: T.Tensor([B, N + 1], T.int64),
        result: T.Tensor([B, N, D], T.float32),
    ):
        """Add FP32 [B, E, D] partials into FP32 [B, N, D] result in place.

        Int64 order [B, E] stably orders query/slot edges by key. Int64
        offsets [B, N + 1] delimit each key's valid edges; invalid edges
        are excluded. Inputs are read-only; result accumulates one chunk.
        """
        with T.Kernel(N, B, T.ceildiv(D, 128), threads=128) as (key, batch, feature):
            start = offsets[batch, key]
            end = offsets[batch, key + 1]
            if end > start:
                values = T.alloc_fragment([32, 128], T.float32)
                reduced = T.alloc_fragment([128], T.float32)
                total = T.alloc_fragment([128], T.float32)
                T.clear(total)
                for chunk in T.serial(T.ceildiv(end - start, 32)):
                    for row, dim in T.Parallel(32, 128):
                        position = start + chunk * 32 + row
                        values[row, dim] = T.if_then_else(
                            (position < end) & (feature * 128 + dim < D),
                            partials[batch, order[batch, position], feature * 128 + dim],
                            0,
                        )
                    T.reduce_sum(values, reduced, 0)
                    for dim in T.Parallel(128):
                        total[dim] += reduced[dim]
                for dim in T.Parallel(128):
                    if feature * 128 + dim < D:
                        result[batch, key, feature * 128 + dim] += total[dim]

    return kernel


@tilelang.jit
def _reduce_sink(H: int):
    """Build a fixed-tree sink reducer over flattened batch/query rows."""
    R = T.dynamic("rows")

    @T.prim_func
    def kernel(
        partials: T.Tensor([R, H], T.float32),
        result: T.Tensor([H], T.float32),
    ):
        """Add FP32 [R, H] partials into FP32 [H] result in place.

        R flattens batch and query rows of the current chunk. Partials are
        read-only; result accumulates their fixed-tree, fixed-order sum.
        """
        with T.Kernel(T.ceildiv(H, 16), threads=128) as block:
            values = T.alloc_fragment([32, 16], T.float32)
            reduced = T.alloc_fragment([16], T.float32)
            total = T.alloc_fragment([16], T.float32)
            T.clear(total)
            for chunk in T.serial(T.ceildiv(R, 32)):
                for row, head in T.Parallel(32, 16):
                    values[row, head] = T.if_then_else(
                        (chunk * 32 + row < R) & (block * 16 + head < H),
                        partials[chunk * 32 + row, block * 16 + head],
                        0,
                    )
                T.reduce_sum(values, reduced, 0)
                for head in T.Parallel(16):
                    total[head] += reduced[head]
            for head in T.Parallel(16):
                if block * 16 + head < H:
                    result[block * 16 + head] += total[head]

    return kernel


def fixed_order_backward(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    o: torch.Tensor,
    do: torch.Tensor,
    topk_idxs: torch.Tensor,
    lse: torch.Tensor,
    *,
    sm_scale: float | None = None,
    workspace_bytes: int = 2 * 1024**3,
    return_dkv_accum_dtype: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculate sparse MQA gradients without unordered accumulation.

    Args:
        q: Contiguous CUDA BF16 queries [batch, sequence, heads, head_dim].
        kv: Contiguous CUDA BF16 keys/values [batch, kv_sequence, head_dim].
        attn_sink: CUDA FP32 sink logits [heads].
        o: BF16 forward output [batch, sequence, heads, head_dim].
        do: BF16 incoming gradient [batch, sequence, heads, head_dim].
        topk_idxs: Integer indices [batch, sequence, slots]; invalid indices
            are masked, including duplicates of the -1 sentinel.
        lse: FP32 saved base-two log normalizers [batch, sequence, heads].
        sm_scale: Score scale, default head_dim**-0.5.
        workspace_bytes: Positive partial-KV workspace cap. Sorting, delta,
            sink partials, contiguous input chunks and output buffers are extra.
            Query chunk size depends only on this cap and tensor shapes.
        return_dkv_accum_dtype: Return KV gradient before its final BF16 cast.

    Returns:
        Independent query gradient [batch, sequence, heads, head_dim] in BF16,
        KV gradient [batch, kv_sequence, head_dim] in BF16 or FP32, and sink
        gradient [heads] in FP32. Inputs are never mutated.
    """
    batch, sequence, heads, dim = q.shape
    keys = kv.shape[1]
    if heads not in (16, 32, 64):
        raise ValueError("fixed-order backward requires 16, 32 or 64 local heads; use head chunking")
    if keys <= 0 or sequence <= 0:
        raise ValueError("fixed-order backward requires nonempty query and KV axes")
    if q.dtype != torch.bfloat16 or kv.dtype != torch.bfloat16 or not q.is_cuda:
        raise ValueError("fixed-order backward requires CUDA BF16 queries and KV")
    slots = (topk_idxs.shape[-1] + 31) // 32 * 32
    bytes_per_query = batch * slots * dim * 4
    if slots == 0 or workspace_bytes < bytes_per_query:
        raise ValueError("fixed-order workspace cannot hold one query's partial KV gradients")
    # Preserve the established chunk partition and floating-point addition order.
    # Dynamic kernel lengths let main chunks and tails share compiled code.
    capacity = min(sequence, workspace_bytes // bytes_per_query)
    chunk_size = 1 << (capacity.bit_length() - 1)
    dq = torch.empty_like(q)
    dkv = torch.zeros_like(kv, dtype=torch.float32)
    dsink = torch.zeros_like(attn_sink, dtype=torch.float32)
    boundaries = torch.arange(keys + 1, device=q.device).expand(batch, -1).contiguous()
    for start in range(0, sequence, chunk_size):
        stop = min(start + chunk_size, sequence)
        count = stop - start
        indices = topk_idxs[:, start:stop]
        if indices.dtype != torch.int32:
            # Mask before narrowing: out-of-range int64 values can wrap onto
            # valid keys. Native int32 model plans retain their existing path.
            indices = torch.where((indices >= 0) & (indices < keys), indices, -1).to(torch.int32)
        indices = indices.contiguous()
        if indices.shape[-1] != slots:
            indices = torch.nn.functional.pad(indices, (0, slots - indices.shape[-1]), value=-1)
        valid = ((indices >= 0) & (indices < keys)).to(torch.int32)
        safe_indices = indices.clamp(0, keys - 1)
        # Invalid entries sort to the final sentinel and are never read by reduction.
        sorted_keys, order = torch.sort(indices.masked_fill(valid == 0, keys).reshape(batch, -1), stable=True)
        offsets = torch.searchsorted(sorted_keys, boundaries)
        partials = torch.empty((batch, count, 1, slots, dim), dtype=torch.float32, device=q.device)
        sink_partials = torch.empty((batch, count, heads), dtype=torch.float32, device=q.device)
        output = o[:, start:stop].contiguous()
        incoming = do[:, start:stop].contiguous()
        delta = backward.preprocess(None, None, heads, dim)(output, incoming)
        kernel = backward.bwd(
            None,
            None,
            None,
            heads,
            dim,
            slots,
            sm_scale,
            threads=backward._bwd_threads_for_heads(heads),
            deterministic=True,
        )
        dq[:, start:stop] = kernel(
            q[:, start:stop].contiguous(),
            kv,
            incoming,
            attn_sink,
            safe_indices,
            valid,
            lse[:, start:stop].contiguous(),
            delta,
            partials,
            sink_partials,
        )
        _reduce_kv(dim)(
            partials.view(batch, count * slots, dim),
            order,
            offsets,
            dkv,
        )
        _reduce_sink(heads)(sink_partials.view(batch * count, heads), dsink)
        # Release the large buffer before allocating the next chunk's partials.
        del partials, sink_partials, sorted_keys, order, offsets, delta
    return dq, dkv if return_dkv_accum_dtype else dkv.to(kv.dtype), dsink
