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

"""CPU unit tests for the MiniMax M3 sparse-block selection under context parallelism.

The CP FlexAttention path selects key blocks for a rank's *local* queries against
the *global* key sequence, using each local query's global position. The
correctness property -- a silent-failure trap, since a wrong selection trains
without errors -- is that this must produce exactly the same blocks the eager
square selection produces for those query rows. These tests verify that against
``select_sparse_blocks`` without needing FlexAttention, a GPU, or a process group.
"""

import pytest
import torch

from nemo_automodel.components.models.minimax_m3_vl.layers import (
    build_block_sparse_attn_mask,
    select_sparse_blocks,
)


def _rand_idx(seqlen, h_idx=4, dim=16, bsz=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    idx_q = torch.randn(bsz, seqlen, h_idx, dim, generator=g)
    idx_k = torch.randn(bsz, seqlen, 1, dim, generator=g)
    return idx_q, idx_k


@pytest.mark.parametrize("block_size,topk", [(8, 2), (8, 4), (16, 2)])
def test_cp_local_queries_match_eager_rows(block_size, topk):
    """select_sparse_blocks for local query rows (with q_positions) == the eager
    full-sequence selection restricted to those rows."""
    seqlen = 48
    idx_q, idx_k = _rand_idx(seqlen)

    full = select_sparse_blocks(
        idx_q, idx_k, block_size=block_size, topk_blocks=topk, init_blocks=0, local_blocks=1
    )  # [B, H_idx, T, num_blocks]

    # Simulate a CP shard: an arbitrary, non-contiguous subset of query rows.
    rows = torch.tensor([0, 3, 7, 8, 23, 24, 47])
    sub = select_sparse_blocks(
        idx_q[:, rows],
        idx_k,  # global keys (full sequence) -- the CP path gathers these
        block_size=block_size,
        topk_blocks=topk,
        init_blocks=0,
        local_blocks=1,
        q_positions=rows,
    )

    assert sub.shape == (full.shape[0], full.shape[1], rows.numel(), full.shape[3])
    assert torch.equal(sub, full[:, :, rows, :]), "CP local-query selection diverged from eager rows"


def test_forced_local_block_always_selected():
    """The query's own (diagonal) block is force-included (local_blocks=1)."""
    seqlen, block_size = 32, 8
    idx_q, idx_k = _rand_idx(seqlen, seed=3)
    sel = select_sparse_blocks(idx_q, idx_k, block_size=block_size, topk_blocks=1, init_blocks=0, local_blocks=1)
    num_blocks = seqlen // block_size
    for q in range(seqlen):
        cur_block = q // block_size
        # the current block must be selected for every (b, h) at query q
        assert sel[:, :, q, cur_block].all(), f"local block {cur_block} not forced for query {q}"
        # no future block may be selected (causal)
        if cur_block + 1 < num_blocks:
            assert not sel[:, :, q, cur_block + 1 :].any(), f"future block selected for query {q}"


def test_topk_ge_numblocks_degenerates_to_causal():
    """With topk >= num_blocks, selection is all causal blocks (degenerate dense causal)."""
    seqlen, block_size = 32, 8
    num_blocks = seqlen // block_size  # 4
    idx_q, idx_k = _rand_idx(seqlen, seed=5)
    sel = select_sparse_blocks(
        idx_q, idx_k, block_size=block_size, topk_blocks=num_blocks, init_blocks=0, local_blocks=1
    )
    blk = torch.arange(num_blocks)
    for q in range(seqlen):
        cur_block = q // block_size
        expected = blk <= cur_block  # all causal blocks selected
        assert torch.equal(sel[0, 0, q], expected), f"query {q} not degenerate-causal"


def test_eager_mask_consistent_with_selection():
    """build_block_sparse_attn_mask's boolean keep pattern matches select_sparse_blocks
    expanded to keys and intersected with token-level causal."""
    seqlen, block_size, topk, num_q_heads = 24, 8, 2, 8
    idx_q, idx_k = _rand_idx(seqlen, h_idx=4, seed=7)
    keep = build_block_sparse_attn_mask(
        idx_q,
        idx_k,
        block_size=block_size,
        topk_blocks=topk,
        init_blocks=0,
        local_blocks=1,
        num_q_heads=num_q_heads,
    )  # [B, num_q_heads, T, T] bool (True == attend)
    assert keep.dtype == torch.bool
    assert keep.shape == (idx_q.shape[0], num_q_heads, seqlen, seqlen)

    sel = select_sparse_blocks(idx_q, idx_k, block_size=block_size, topk_blocks=topk, init_blocks=0, local_blocks=1)
    causal = torch.tril(torch.ones(seqlen, seqlen, dtype=torch.bool))
    key_sel = sel.repeat_interleave(block_size, dim=-1)[..., :seqlen] & causal[None, None]
    rep = num_q_heads // sel.shape[1]
    expected_attend = key_sel.repeat_interleave(rep, dim=1)  # [B, num_q_heads, T, T]

    assert torch.equal(keep, expected_attend)
    # every query attends to at least its own position (causal diagonal)
    diag = torch.arange(seqlen)
    assert keep[:, :, diag, diag].all()
