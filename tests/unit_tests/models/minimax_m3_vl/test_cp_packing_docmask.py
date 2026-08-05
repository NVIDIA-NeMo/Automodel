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

"""CPU unit tests for MiniMax M3 CP packed-sequence (per-document) masking.

The CP sparse forward derives a per-token document id from packed position ids
(reset to 0 per document) and adds a same-document term to the FlexAttention
mask so packed multi-document sequences do not attend across document boundaries.
These tests cover the document-id derivation and the same-document + causal
predicate (a block-diagonal mask), without needing FlexAttention/GPU/a process
group.
"""

import torch

from nemo_automodel.components.models.minimax_m3_vl.cp_sparse_attn import cp_document_ids


def test_single_sequence_is_all_zero_docid():
    """A single (non-packed) sequence -> doc id all zeros -> same_doc is a no-op."""
    pos = torch.arange(16).unsqueeze(0)  # [1, 16], no interior resets
    assert torch.equal(cp_document_ids(pos), torch.zeros(1, 16, dtype=torch.long))


def test_packed_docids_increment_per_document():
    """Position ids resetting per document yield contiguous 0-based document ids."""
    # docs of length 3, 2, 4 -> positions [0,1,2, 0,1, 0,1,2,3]
    pos = torch.tensor([[0, 1, 2, 0, 1, 0, 1, 2, 3]])
    expected = torch.tensor([[0, 0, 0, 1, 1, 2, 2, 2, 2]])
    assert torch.equal(cp_document_ids(pos), expected)


def test_trailing_cp_pad_opens_spurious_doc_but_is_isolated():
    """A trailing cp-pad (position 0) starts an extra doc id beyond the real docs."""
    # one doc of length 5 (positions 0..4) + 1 cp-pad at position 0
    pos = torch.tensor([[0, 1, 2, 3, 4, 0]])
    doc = cp_document_ids(pos)
    assert doc.tolist() == [[0, 0, 0, 0, 0, 1]]
    # the pad (doc 1) shares no document with the real tokens (doc 0)
    assert doc[0, -1] != doc[0, 0]


def test_same_doc_plus_causal_is_block_diagonal():
    """same_doc & causal reproduces a per-document (block-diagonal) causal mask."""
    pos = torch.tensor([[0, 1, 2, 0, 1]])  # docs: [0,1,2], [3,4]
    doc = cp_document_ids(pos)[0]  # [5]
    t = pos.shape[1]
    q = torch.arange(t)
    same_doc = doc[:, None] == doc[None, :]  # [T, T]
    causal = q[:, None] >= q[None, :]  # slot-causal
    allowed = same_doc & causal

    # Reference: build the expected block-diagonal causal mask directly.
    seq_lens = [3, 2]
    expected = torch.zeros(t, t, dtype=torch.bool)
    start = 0
    for n in seq_lens:
        block = torch.tril(torch.ones(n, n, dtype=torch.bool))
        expected[start : start + n, start : start + n] = block
        start += n
    assert torch.equal(allowed, expected)
    # no token attends across the doc boundary (e.g. query 3 must not see key 0)
    assert not allowed[3, 0] and not allowed[3, 2]
    assert allowed[3, 3] and allowed[4, 3]


def test_multi_batch_independent_docids():
    pos = torch.tensor([[0, 1, 0, 1], [0, 1, 2, 3]])  # batch row 0 packed (2 docs), row 1 single
    doc = cp_document_ids(pos)
    assert doc.tolist() == [[0, 0, 1, 1], [0, 0, 0, 0]]
