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

"""Fused attention with compact CSA selection and block-level mask metadata."""

import torch
from torch.nn import functional as F
from torch.nn.attention.flex_attention import BlockMask

from nemo_automodel.components.attention.flex_attention import FlexAttention
from nemo_automodel.components.models.step5.kernels import pack_region_bitmap
from nemo_automodel.components.models.step5.sparse import RegionSelection


def fused_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    positions: torch.Tensor,
    key_positions: torch.Tensor,
    key_padding: torch.Tensor,
    documents: torch.Tensor,
    *,
    query_start: int,
    window: int | None,
    selection: RegionSelection | None,
) -> torch.Tensor:
    """Run the shared compiled FlexAttention kernel without a dense token mask.

    Args:
        query: CUDA tensor [batch, local_queries, query_heads, head_dim].
        key: CUDA tensor [batch, global_keys, kv_heads, head_dim].
        value: CUDA tensor with the same shape and dtype as key.
        positions: Document-local positions [batch, local_queries].
        key_positions: Document-local positions [batch, global_keys].
        key_padding: Boolean tensor [batch, global_keys], true for padding.
        documents: Document ids [batch, global_keys].
        query_start: Global offset of the contiguous query shard.
        window: Sliding window size, or None for CSA layers.
        selection: Compact sorted CSA region ids, or None for sliding attention.

    Returns:
        Attention output [batch, local_queries, query_heads, head_dim].
    """
    batch, queries, heads, dim = query.shape
    keys, kv_heads = key.shape[1:3]
    heads_per_group = heads // kv_heads
    query_documents = documents[:, query_start : query_start + queries]
    ids = selection.region_ids if selection is not None else None
    token_regions = selection.token_regions if selection is not None else None
    bitmap = None
    if ids is not None:
        regions = int(token_regions.max().item()) + 1
        words = (regions + 31) // 32
        if batch * kv_heads * queries * words * 4 > 128 * 1024**2:
            raise ValueError("Step5 Flex comparison bitmap exceeds 128 MiB; use BackendConfig(attn='fa4')")
        bitmap = pack_region_bitmap(ids, regions)

    def mask_mod(b: torch.Tensor, h: torch.Tensor, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        safe_q = q.clamp(max=queries - 1)
        safe_k = k.clamp(max=keys - 1)
        distance = positions[b, safe_q] - key_positions[b, safe_k]
        keep = (
            (q < queries)
            & (k < keys)
            & (distance >= 0)
            & ~key_padding[b, safe_k]
            & (query_documents[b, safe_q] == documents[b, safe_k])
        )
        if window is not None:
            keep = keep & (distance < window)
        if bitmap is not None:
            target = token_regions[b, safe_k]
            word = bitmap[b, h // heads_per_group, safe_q, target // 32]
            keep = keep & (((word >> (target % 32)) & 1) != 0)
        return keep

    # Coarse metadata is O((Q / 128) * (K / 128)), shared across heads and
    # batches. The fine predicate runs inside the fused attention kernel.
    block = 128
    q_blocks = (queries + block - 1) // block
    k_blocks = (keys + block - 1) // block
    rows = torch.arange(q_blocks, device=query.device)
    last = ((query_start + (rows + 1) * block - 1) // block).clamp(max=k_blocks - 1)
    first = (
        torch.zeros_like(last) if window is None else ((query_start + rows * block - window + 1) // block).clamp(min=0)
    )
    width = k_blocks if window is None else min(k_blocks, (window + 2 * block - 2) // block + 1)
    indices = first[:, None] + torch.arange(width, device=query.device)
    counts = (last - first + 1).clamp(min=0, max=width)
    indices = indices.clamp(max=k_blocks - 1).to(torch.int32)[None, None]
    # Construct reverse metadata analytically. from_kv_blocks otherwise expands
    # to a dense block grid, and assumes every row has K_blocks storage slots.
    columns = torch.arange(k_blocks, device=query.device)
    q_first = ((columns * block - query_start) // block).clamp(min=0)
    q_last = (
        torch.full_like(columns, q_blocks - 1)
        if window is None
        else ((((columns + 1) * block + window - 2 - query_start) // block).clamp(max=q_blocks - 1))
    )
    q_width = q_blocks if window is None else min(q_blocks, (window + 2 * block - 2) // block + 1)
    q_counts = (q_last - q_first + 1).clamp(min=0, max=q_width)
    q_indices = (q_first[:, None] + torch.arange(q_width, device=query.device)).clamp(max=q_blocks - 1)
    block_mask = BlockMask(
        seq_lengths=(queries, keys),
        kv_num_blocks=counts.to(torch.int32)[None, None],
        kv_indices=indices,
        full_kv_num_blocks=None,
        full_kv_indices=None,
        q_num_blocks=q_counts.to(torch.int32)[None, None],
        q_indices=q_indices.to(torch.int32)[None, None],
        full_q_num_blocks=None,
        full_q_indices=None,
        BLOCK_SIZE=(block, block),
        mask_mod=mask_mod,
    )
    # FlexAttention needs power-of-two head dimensions. Preserve Step-5's
    # original scale when zero-padding 192 to 256, then discard padded outputs.
    padded = 1 << (dim - 1).bit_length()
    q = F.pad(query.transpose(1, 2), (0, padded - dim))
    k = F.pad(key.transpose(1, 2), (0, padded - dim))
    v = F.pad(value.transpose(1, 2), (0, padded - dim))
    out = FlexAttention.flex_attn(q, k, v, block_mask=block_mask, scale=dim**-0.5, enable_gqa=True)
    return out[..., :dim].transpose(1, 2)
