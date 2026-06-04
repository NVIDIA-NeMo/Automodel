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

"""Tests for the DFlash block attention mask.

The KV axis is ``[context (S) | block_0 | block_1 | ...]`` and the Q axis is
``[block_0 | block_1 | ...]``. The mask must enforce:
  1. a block sees context strictly before its anchor,
  2. intra-block attention is bidirectional,
  3. different blocks are mutually invisible,
  4. invalid blocks see nothing.
"""

from __future__ import annotations

import torch

from nemo_automodel.components.speculative.dflash.core import create_dflash_sdpa_mask


def _build(anchors, keep, S, block_size):
    anchor_positions = torch.tensor(anchors, dtype=torch.long)
    block_keep_mask = torch.tensor(keep, dtype=torch.bool)
    mask = create_dflash_sdpa_mask(anchor_positions, block_keep_mask, S, block_size, torch.device("cpu"))
    return mask[:, 0]  # drop the broadcast head dim -> (B, Q_LEN, KV_LEN)


def test_context_visible_strictly_before_anchor():
    S, bs = 6, 2
    # one sample, two blocks, anchors at 3 and 5, both valid
    mask = _build([[3, 5]], [[True, True]], S, bs)[0]
    # block 0 (rows 0,1) -> context columns < 3 are visible, >= 3 invisible
    for q in (0, 1):
        assert mask[q, :3].all()
        assert not mask[q, 3:S].any()
    # block 1 (rows 2,3) -> context columns < 5 visible
    for q in (2, 3):
        assert mask[q, :5].all()
        assert not mask[q, 5:S].any()


def test_intra_block_bidirectional_and_blocks_isolated():
    S, bs = 6, 2
    mask = _build([[3, 5]], [[True, True]], S, bs)[0]
    # draft KV columns: S + block*bs + pos
    b0 = [S + 0, S + 1]
    b1 = [S + 2, S + 3]
    # block 0 queries (rows 0,1) attend to both block-0 draft cols (bidirectional)
    for q in (0, 1):
        assert mask[q, b0[0]] and mask[q, b0[1]]
        assert not mask[q, b1[0]] and not mask[q, b1[1]]  # block 1 invisible
    # block 1 queries (rows 2,3) attend only to block-1 draft cols
    for q in (2, 3):
        assert mask[q, b1[0]] and mask[q, b1[1]]
        assert not mask[q, b0[0]] and not mask[q, b0[1]]


def test_invalid_block_sees_nothing():
    S, bs = 6, 2
    # block 1 invalid -> its rows are entirely masked out
    mask = _build([[3, 5]], [[True, False]], S, bs)[0]
    assert mask[0].any() and mask[1].any()  # valid block 0 sees something
    assert not mask[2].any() and not mask[3].any()  # invalid block 1 sees nothing


def test_batch_independent_anchors():
    S, bs = 6, 2
    mask = _build([[2, 4], [1, 5]], [[True, True], [True, True]], S, bs)
    # sample 0, block 0 anchor=2 -> context < 2
    assert mask[0, 0, :2].all() and not mask[0, 0, 2:S].any()
    # sample 1, block 0 anchor=1 -> context < 1
    assert mask[1, 0, :1].all() and not mask[1, 0, 1:S].any()
