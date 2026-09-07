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

"""Block selection for MSA: which key blocks of its own document each query attends to.

``select_blocks_reference`` states the rule over plain tensors and returns the canonical support
directly: int32 ``[index_heads, tokens, topk_blocks]`` of document-local block ids padded with -1.
That tensor is the sole semantic seam between selection and the rest of MSA -- ``build_k2q_csr``
and ``sparse_atten_func`` consume it unchanged.
"""

import torch
import torch.nn.functional as F

# Cap the fp32 score tile. Scoring every query against every key block at once materializes a
# [heads, tokens, workspace_rows] tensor -- 6 GiB at 16k packed tokens -- so the key-block dimension
# is tiled and each tile is reduced to per-block scores immediately. Blocks are independent, so the
# tiling is exact.
_SCORE_TILE_BYTES = 256 * 1024**2


@torch.no_grad()
def select_blocks_reference(
    index_q: torch.Tensor,
    aligned_index_k: torch.Tensor,
    query_positions: torch.Tensor,
    document_starts: torch.Tensor,
    *,
    block_size: int,
    topk_blocks: int,
    init_blocks: int,
    local_blocks: int,
    score_type: str,
) -> torch.Tensor:
    """Select each query's top-k key blocks within its own document, in Torch.

    The score for (query ``i``, key ``j``) is ``(index_q[i] . index_k[j]) * dim**-0.5``; keys are
    grouped into blocks of ``block_size`` and reduced per block (``max`` or ``lse``). A query may
    only pick blocks of its own document, up to and including its own block; the first
    ``init_blocks`` of the document and the query's own block (when ``local_blocks``) are forced in,
    and the rest of the budget goes to the highest-scoring candidates. Ties resolve as
    ``torch.topk`` orders them.

    The rule is stated in workspace block coordinates and the document's first block is subtracted
    at the end, because the aligned workspace is what makes every document's blocks start on a
    block boundary.

    Args:
        index_q: Index queries [tokens, index_heads, index_dim], post norm and RoPE.
        aligned_index_k: Shared index key [1, workspace_rows, 1, index_dim], scattered to workspace
            rows so block ``b`` spans rows ``[b * block_size, (b + 1) * block_size)``.
        query_positions: Workspace row of each query [tokens], int64.
        document_starts: Workspace row where each query's document starts [tokens], int64.
        block_size: Keys per block; must divide ``workspace_rows``.
        topk_blocks: Blocks each query keeps.
        init_blocks: Leading blocks of the document forced in.
        local_blocks: Non-zero forces the query's own block in.
        score_type: ``"max"`` or ``"lse"`` block reduction.

    Returns:
        Document-local block ids [index_heads, tokens, topk_blocks], int32, padded with -1.
    """
    num_tokens, index_heads, index_dim = index_q.shape
    workspace_rows = aligned_index_k.shape[1]
    device = index_q.device
    num_blocks = workspace_rows // block_size
    negative_infinity = float("-inf")

    queries = index_q.permute(1, 0, 2).float()
    keys = aligned_index_k.reshape(workspace_rows, index_dim).float()
    rows = torch.arange(workspace_rows, device=device)
    block_score = torch.empty((index_heads, num_tokens, num_blocks), dtype=torch.float32, device=device)
    tile_blocks = max(1, _SCORE_TILE_BYTES // (index_heads * num_tokens * block_size * 4))
    for start in range(0, num_blocks, tile_blocks):
        stop = min(start + tile_blocks, num_blocks)
        first_row, last_row = start * block_size, stop * block_size
        scores = torch.matmul(queries, keys[first_row:last_row].t())
        # In place: scaling after the matmul is what an untiled pass would round, and a fresh
        # tensor here would double the tile's peak.
        scores.mul_(index_dim**-0.5)
        scores.masked_fill_(rows[first_row:last_row] > query_positions[:, None], negative_infinity)
        tile = scores.view(index_heads, num_tokens, stop - start, block_size)
        block_score[:, :, start:stop] = tile.logsumexp(-1) if score_type == "lse" else tile.amax(-1)

    blocks = torch.arange(num_blocks, device=device)
    first_block = document_starts // block_size
    own_block = query_positions // block_size
    candidate = (blocks >= first_block[:, None]) & (blocks <= own_block[:, None])
    forced = blocks < (first_block + init_blocks)[:, None]
    if local_blocks > 0:
        forced = forced | (blocks == own_block[:, None])
    block_score.masked_fill_(~candidate, negative_infinity)
    block_score.masked_fill_(forced & candidate, float("inf"))
    if num_blocks < topk_blocks:
        block_score = F.pad(block_score, (0, topk_blocks - num_blocks), value=negative_infinity)
    values, indices = block_score.topk(topk_blocks, dim=-1)
    selected = torch.where(values == negative_infinity, -1, indices - first_block[None, :, None])
    return selected.to(torch.int32).contiguous()
