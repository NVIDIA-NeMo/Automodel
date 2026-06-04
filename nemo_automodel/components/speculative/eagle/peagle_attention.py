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

"""FlexAttention mask for P-EAGLE parallel-group prediction (arXiv:2602.01469).

COD sampling packs all prediction depths of a sequence into one flat axis of
length ``total_sampled``. This mask lets each packed element attend to (a) the
depth-0 causal context of its own document and (b) earlier depths in its own
sampling chain (same anchor), while blocking cross-document and acausal leakage.
"""

from __future__ import annotations

import torch


def create_peagle_mask_mod(
    anchor_pos: torch.Tensor,
    depth: torch.Tensor,
    lengths: torch.Tensor,
    total_seq_len: int,
):
    """Return a ``mask_mod`` for ``torch.nn.attention.flex_attention.create_block_mask``.

    Args:
        anchor_pos: ``[total_sampled]`` chain anchor of each packed element.
        depth: ``[total_sampled]`` prediction depth of each packed element.
        lengths: ``[num_docs]`` token length per packed document (prevents
            cross-document attention).
        total_seq_len: Padded length of the original (depth-0) sequence axis.
    """
    # Per-original-position document id; pad the tail with -1 so padded keys
    # never match a real query's document.
    document_ids = torch.repeat_interleave(
        torch.arange(lengths.shape[0], device=lengths.device, dtype=torch.long), lengths
    )
    document_ids = torch.cat(
        [
            document_ids,
            -1 * torch.ones(total_seq_len - document_ids.shape[0], device=lengths.device, dtype=torch.long),
        ]
    ).contiguous()

    def peagle_mask_mod(_b, _h, q_idx, kv_idx):
        q_anchor_pos = anchor_pos[q_idx]
        kv_anchor_pos = anchor_pos[kv_idx]
        q_depth = depth[q_idx]
        kv_depth = depth[kv_idx]

        same_document = document_ids[q_anchor_pos] == document_ids[kv_anchor_pos]
        is_not_padding = document_ids[q_anchor_pos] != -1
        same_rollout = q_anchor_pos == kv_anchor_pos
        kv_depth0 = kv_depth == 0
        in_depth_order = q_depth >= kv_depth
        is_anchor_causal = q_anchor_pos >= kv_anchor_pos

        return is_not_padding & same_document & ((kv_depth0 & is_anchor_causal) | (same_rollout & in_depth_order))

    return peagle_mask_mod
