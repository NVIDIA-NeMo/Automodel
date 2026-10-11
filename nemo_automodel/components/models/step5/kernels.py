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

"""Discrete CSA kernels; attention gradients remain in the shared fused backend."""

import torch

from nemo_automodel.shared.import_utils import safe_import

_HAVE_TRITON, triton = safe_import("triton")
_HAVE_TL, tl = safe_import("triton.language")

if _HAVE_TRITON and _HAVE_TL:
    # Packed boundaries change both counts; keep them as runtime scalars.
    @triton.jit(do_not_specialize=["regions", "words"])
    def _pack_regions(ids, bitmap, total, slots: tl.constexpr, regions, words, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        region = tl.load(ids + offsets, offsets < total, other=regions)
        word = (offsets // slots) * words + region // 32
        bit = 1 << (region % 32)
        tl.atomic_or(bitmap + word, bit, mask=(offsets < total) & (region < regions), sem="relaxed")

    @triton.jit
    def _score_regions(
        q,
        w,
        k,
        out,
        query_start,
        queries,
        regions,
        q_batch,
        q_token,
        q_group,
        q_head,
        w_batch,
        w_token,
        w_group,
        k_batch,
        k_region,
        out_batch,
        out_group,
        out_token,
        GROUPS: tl.constexpr,
        HEADS: tl.constexpr,
        DIM: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_Q: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        row = tl.program_id(0) * BLOCK_Q + tl.arange(0, BLOCK_Q)
        col = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
        batch, group = tl.program_id(2) // GROUPS, tl.program_id(2) % GROUPS
        dim = tl.arange(0, BLOCK_D)
        keys = tl.load(
            k + batch * k_batch + col[None, :] * k_region + dim[:, None],
            (col[None, :] < regions) & (dim[:, None] < DIM),
            other=0,
        )
        scores = tl.zeros((BLOCK_Q, BLOCK_K), tl.float32)
        for head in tl.static_range(HEADS):
            query = tl.load(
                q
                + batch * q_batch
                + (query_start + row[:, None]) * q_token
                + group * q_group
                + head * q_head
                + dim[None, :],
                (row[:, None] < queries) & (dim[None, :] < DIM),
                other=0,
            )
            weight = tl.load(
                w + batch * w_batch + (query_start + row) * w_token + group * w_group + head, row < queries, other=0
            ).to(tl.float32)
            dot = tl.dot(query, keys, out_dtype=tl.float32)
            scores += tl.maximum(dot, 0) * weight[:, None]
        tl.store(
            out + batch * out_batch + group * out_group + row[:, None] * out_token + col[None, :],
            scores,
            (row[:, None] < queries) & (col[None, :] < regions),
        )


def pack_region_bitmap(ids: torch.Tensor, regions: int) -> torch.Tensor:
    """Pack selected regions directly without int64 scatter workspaces.

    Args:
        ids: Sorted CUDA int32 tensor [batch, groups, queries, slots]; ids at
            or above regions are padding. Repeated valid ids are allowed.
        regions: Total number of key regions.

    Returns:
        CUDA int32 bitmap [batch, groups, queries, ceil(regions / 32)].
    """
    if not (_HAVE_TRITON and _HAVE_TL):
        raise ImportError("Step5 CUDA selection requires Triton")
    words = (regions + 31) // 32
    bitmap = torch.zeros((*ids.shape[:-1], words), device=ids.device, dtype=torch.int32)
    _pack_regions[(triton.cdiv(ids.numel(), 256),)](
        ids.contiguous(),
        bitmap,
        ids.numel(),
        ids.shape[-1],
        regions,
        words,
        BLOCK=256,
    )
    return bitmap


def indexer_score_tile(
    query: torch.Tensor, weights: torch.Tensor, summary: torch.Tensor, start: int, stop: int
) -> torch.Tensor:
    """Fuse weighted-ReLU scoring into one bounded FP32 output tile.

    Args:
        query: E4M3-rounded BF16 tensor [batch, queries, groups, heads, dim].
        weights: FP32 signed head weights [batch, queries, groups, heads].
        summary: E4M3-rounded BF16 region summaries [batch, regions, dim].
        start: First query included in this tile.
        stop: Exclusive last query included in this tile.

    Returns:
        FP32 scores [batch, groups, stop - start, regions]. No gradients are
        defined: these scores drive discrete top-k selection only.
    """
    batch, _, groups, heads, dim = query.shape
    count, regions = stop - start, summary.shape[1]
    scores = torch.zeros(batch, groups, count, regions, device=query.device, dtype=torch.float32)
    if query.is_cuda:
        if not (_HAVE_TRITON and _HAVE_TL):
            raise ImportError("Step5 CUDA selection requires Triton")
        _score_regions[(triton.cdiv(count, 32), triton.cdiv(regions, 128), batch * groups)](
            query,
            weights,
            summary,
            scores,
            start,
            count,
            regions,
            *query.stride()[:4],
            *weights.stride()[:3],
            *summary.stride()[:2],
            *scores.stride()[:3],
            GROUPS=groups,
            HEADS=heads,
            DIM=dim,
            BLOCK_D=triton.next_power_of_2(dim),
            BLOCK_Q=32,
            BLOCK_K=128,
        )
    else:
        for head in range(heads):
            q = query[:, start:stop, :, head].permute(0, 2, 1, 3).reshape(batch, groups * count, dim)
            dots = torch.bmm(q.float(), summary.float().transpose(1, 2)).view(batch, groups, count, regions)
            scores.add_(dots.relu_() * weights[:, start:stop, :, head].transpose(1, 2).unsqueeze(-1))
    return scores
