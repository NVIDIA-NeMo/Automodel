# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Distributed in-batch negative utilities for bi-encoder contrastive training.

Architecture-agnostic helpers used by the bi-encoder trainer to expand the
negative pool with passages gathered across DP ranks. Backbones (Llama,
Ministral3, Qwen3, ...) do not import these directly; the trainer wires them
in around ``BiEncoderModel.encode``.
"""

from typing import Optional

import torch
import torch.distributed as dist


def dist_gather_tensor(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """All-gather ``t`` along dim 0 across the default process group.

    The local-rank slice is replaced with the original ``t`` so that gradients
    flow back only to the local portion of the gathered tensor (other ranks'
    slices are detached). Returns ``t`` unchanged when distributed is not
    available, not initialized, or world size is 1.
    """
    if t is None:
        return None
    if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() <= 1:
        return t
    t = t.contiguous()
    gathered = [torch.empty_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, t)
    gathered[dist.get_rank()] = t
    return torch.cat(gathered, dim=0)


def mask_gathered_passages_same_doc_as_positive(
    scores: torch.Tensor,
    passage_doc_ids: torch.Tensor,
    train_n_passages: int,
    rank: int,
    local_batch_size: int,
) -> None:
    """In-place mask passages sharing a doc id with this row's positive.

    After all-gather, each query's positive sits at column ``i * train_n_passages``
    of the gathered passage tensor. For each local query row, set scores to
    ``finfo(dtype).min`` on any other column whose ``passage_doc_ids`` matches
    the positive's id, so duplicates of the positive elsewhere in the global
    batch are not treated as negatives. The true positive column is left intact.

    Args:
        scores: ``[local_batch_size, B_global * train_n_passages]`` (already
            sliced to the local rank's query rows).
        passage_doc_ids: ``[B_global * train_n_passages]`` int64 doc ids for
            every gathered passage.
        train_n_passages: Number of passages per query (1 positive + negatives).
        rank: Caller's DP rank.
        local_batch_size: Number of queries per rank.
    """
    device = scores.device
    n = passage_doc_ids.shape[0]
    mask_val = torch.finfo(scores.dtype).min
    g = torch.arange(rank * local_batch_size, (rank + 1) * local_batch_size, device=device)
    pos_cols = g * train_n_passages
    pos_ids = passage_doc_ids[pos_cols].unsqueeze(1)
    j = torch.arange(n, device=device, dtype=torch.long).unsqueeze(0)
    same_id = passage_doc_ids.unsqueeze(0) == pos_ids
    not_pos_column = j != pos_cols.unsqueeze(1)
    scores.masked_fill_(same_id & not_pos_column, mask_val)
