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

"""The Torch definition of MSA block selection, the oracle ``select_blocks`` is checked against.

Nothing dispatches to this, so it is free to be the slow, obvious statement of the rule: it scatters
the keys into workspace rows and scores every query against every key block at once. That costs
``index_heads * tokens * workspace_rows`` fp32 values, about 1.3 GiB at the largest fixture here, so
it is sized for tests and not for a real microbatch.
"""

import torch
import torch.nn.functional as F

from nemo_automodel.components.models.minimax_m3_vl.msa import MSAMicrobatch

_NEGATIVE_INFINITY = float("-inf")


@torch.no_grad()
def block_scores_reference(
    microbatch: MSAMicrobatch,
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    *,
    block_size: int,
    init_blocks: int,
    local_blocks: int,
    score_type: str,
) -> torch.Tensor:
    """Rank every key block of the workspace for every query, in Torch.

    The score for (query ``i``, key ``j``) is ``(index_q[i] . index_k[j]) * index_dim**-0.5``; keys
    are grouped into blocks of ``block_size`` and reduced per block (``max`` or ``lse``). Blocks a
    query may not pick -- another document's, or its own document's future -- score ``-inf``, and
    blocks forced in score ``+inf``: the leading ``init_blocks`` of the document, and the query's own
    block when ``local_blocks`` is non-zero.

    Args:
        microbatch: The packed microbatch the tokens came from.
        index_q: Index queries [tokens, index_heads, index_dim], post norm and RoPE.
        index_k: Shared index key [tokens, 1, index_dim], post norm and RoPE.
        block_size: Keys per block.
        init_blocks: Leading blocks of the document forced in.
        local_blocks: Non-zero forces the query's own block in.
        score_type: ``"max"`` or ``"lse"`` block reduction.

    Returns:
        Ranking scores [index_heads, tokens, workspace_blocks], fp32, in workspace block
        coordinates: block ``b`` spans workspace rows ``[b * block_size, (b + 1) * block_size)``.
    """
    num_tokens, index_heads, index_dim = index_q.shape
    query_positions = microbatch.workspace_positions
    document_starts = query_positions - microbatch.document_positions
    if index_k.dim() != 3 or index_k.shape[0] != num_tokens or index_k.shape[1] != 1:
        raise ValueError(f"index_k must have shape [{num_tokens}, 1, index_dim], got {tuple(index_k.shape)}")
    workspace_rows = microbatch.workspace_size
    keys = index_k.new_zeros((workspace_rows, index_dim)).index_copy_(
        0, query_positions, index_k.reshape(num_tokens, index_dim)
    )

    scores = torch.matmul(index_q.permute(1, 0, 2).float(), keys.float().t()) * index_dim**-0.5
    rows = torch.arange(workspace_rows, device=index_q.device)
    scores.masked_fill_(rows > query_positions[:, None], _NEGATIVE_INFINITY)
    tiles = scores.view(index_heads, num_tokens, workspace_rows // block_size, block_size)
    ranked = tiles.logsumexp(-1) if score_type == "lse" else tiles.amax(-1)

    blocks = torch.arange(ranked.shape[-1], device=index_q.device)
    first_block, own_block = document_starts // block_size, query_positions // block_size
    candidate = (blocks >= first_block[:, None]) & (blocks <= own_block[:, None])
    forced = blocks < (first_block + init_blocks)[:, None]
    if local_blocks > 0:
        forced = forced | (blocks == own_block[:, None])
    ranked.masked_fill_(~candidate, _NEGATIVE_INFINITY)
    ranked.masked_fill_(forced & candidate, float("inf"))
    return ranked


def select_blocks_reference(
    microbatch: MSAMicrobatch,
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    *,
    block_size: int,
    topk_blocks: int,
    init_blocks: int,
    local_blocks: int,
    score_type: str,
) -> torch.Tensor:
    """Keep each query's ``topk_blocks`` highest-ranked blocks, as document-local ids.

    This is the definition ``MSAMicrobatch.select_blocks`` is checked against. It ranks
    in workspace block coordinates and subtracts the document's first block at the end, where the
    production rule works in document-local coordinates throughout; the two agreeing is what the
    SM100 comparison tests, and is the one deliberate duplication of this rule. Ties resolve as
    ``torch.topk`` orders them.

    Args:
        microbatch, index_q, index_k, block_size, init_blocks, local_blocks, score_type: As for
            ``block_scores_reference``.
        topk_blocks: Blocks each query keeps.

    Returns:
        Document-local block ids [index_heads, tokens, topk_blocks], int32, padded with -1.
    """
    ranked = block_scores_reference(
        microbatch,
        index_q,
        index_k,
        block_size=block_size,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        score_type=score_type,
    )
    if ranked.shape[-1] < topk_blocks:
        ranked = F.pad(ranked, (0, topk_blocks - ranked.shape[-1]), value=_NEGATIVE_INFINITY)
    values, indices = ranked.topk(topk_blocks, dim=-1)
    first_block = ((microbatch.workspace_positions - microbatch.document_positions) // block_size)[None, :, None]
    return torch.where(values == _NEGATIVE_INFINITY, -1, indices - first_block).to(torch.int32).contiguous()


def select_blocks_reference_for(
    indexer: torch.nn.Module, microbatch: MSAMicrobatch, index_q: torch.Tensor, index_k: torch.Tensor
) -> torch.Tensor:
    """Run the reference with the five selection values ``indexer`` carries."""
    return select_blocks_reference(
        microbatch,
        index_q,
        index_k,
        block_size=indexer.block_size,
        topk_blocks=indexer.topk_blocks,
        init_blocks=indexer.init_blocks,
        local_blocks=indexer.local_blocks,
        score_type=indexer.score_type,
    )
