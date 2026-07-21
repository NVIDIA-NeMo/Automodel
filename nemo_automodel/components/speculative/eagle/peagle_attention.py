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

"""Flex-attention mask for P-EAGLE parallel-group prediction.

P-EAGLE flattens all COD-sampled depths into one sequence and runs a *single*
attention forward over it. The cross-depth visibility pattern is not plain
causal: an element may attend to (a) any earlier-position element at depth 0
(the committed real context) and (b) earlier-or-equal depths *of its own
rollout* (the masked multi-token-prediction chain it belongs to). Documents are
isolated so packed / padded rows never attend across each other.

This is a verbatim port of speculators' ``create_peagle_mask_mod``
(https://github.com/vllm-project/speculators/pull/480) so training reproduces
exactly what vLLM's parallel-drafting runtime sees at inference.
"""

from __future__ import annotations

import torch


def create_peagle_mask_mod(
    anchor_pos: torch.Tensor,  # [total_sampled]
    depth: torch.Tensor,  # [total_sampled]
    lengths: torch.Tensor,  # [num_documents]
    total_seq_len: int,
):
    """Build a ``flex_attention`` ``mask_mod`` for P-EAGLE parallel groups.

    Each query attends only to previous elements in the same sampling chain /
    rollout, plus the causal depth-0 context in the same document. ``lengths``
    drives a document-id map so padded positions (ids ``-1``) and cross-document
    pairs are excluded.

    Example (one document of length 6, COD sampling):
        Round 1 positions: [0, 1, 3, 4]; Round 2: [0, 3]; Round 3: [0]
        anchor_pos: [0,1,2,3,4,5, 0,1,3,4, 0,3, 0]
        depth:      [0,0,0,0,0,0, 1,1,1,1, 2,2, 3]

    Args:
        anchor_pos: Chain-start position in the original sequence per element.
        depth: COD round per element.
        lengths: Valid (unpadded) length of each document in the flat sequence.
        total_seq_len: Combined padded length of the original sequence(s).

    Returns:
        A ``mask_mod`` callable compatible with ``create_block_mask``.
    """
    # Document ids over the *original* positions; pad the tail with -1.
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
