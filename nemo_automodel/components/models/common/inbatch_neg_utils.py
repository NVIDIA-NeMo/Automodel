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
import torch.distributed.nn.functional as dist_nn_func


def _all_gather_tensor(t: torch.Tensor, preserve_grad: bool = False) -> torch.Tensor:
    """All-gather ``t`` along dim 0, preserving autograd only when needed."""
    if preserve_grad and t.requires_grad:
        return torch.cat(dist_nn_func.all_gather(t), dim=0)

    gathered = [torch.empty_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, t)
    if t.requires_grad:
        gathered[dist.get_rank()] = t
    return torch.cat(gathered, dim=0)


def dist_gather_tensor(t: Optional[torch.Tensor], preserve_grad: bool = False) -> Optional[torch.Tensor]:
    """All-gather ``t`` along dim 0 across the default process group.

    When ``preserve_grad`` is true, tensors that require gradients use an
    autograd-aware gather so distributed in-batch-negative losses can send
    passage gradients back to the owning rank. Otherwise, remote slices are
    detached and only the local slice keeps gradient flow. Non-gradient tensors,
    such as masks or IDs, always use a regular detached gather.
    Returns ``t`` unchanged when distributed is not available, not initialized,
    or world size is 1.
    """
    if t is None:
        return None
    if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() <= 1:
        return t
    t = t.contiguous()
    return _all_gather_tensor(t, preserve_grad=preserve_grad)


def dist_gather_tensor_with_dim1_padding(
    t: Optional[torch.Tensor],
    padding_value: int | float | bool = 0,
    preserve_grad: bool = False,
) -> Optional[torch.Tensor]:
    """All-gather ``t`` after padding dim 1 to the maximum length across ranks."""
    if t is None:
        return None
    if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() <= 1:
        return t
    local_shape = torch.tensor(t.shape, device=t.device, dtype=torch.long)
    shapes = [torch.empty_like(local_shape) for _ in range(dist.get_world_size())]
    dist.all_gather(shapes, local_shape)
    max_dim1 = max(int(shape[1].item()) for shape in shapes)
    if t.shape[1] < max_dim1:
        pad_shape = list(t.shape)
        pad_shape[1] = max_dim1 - t.shape[1]
        padding = t.new_full(pad_shape, padding_value)
        t = torch.cat([t, padding], dim=1)
    t = t.contiguous()
    return _all_gather_tensor(t, preserve_grad=preserve_grad)


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
