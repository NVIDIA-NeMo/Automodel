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

from nemo_automodel.components.models.minimax_m3_vl import layers as layers_module
from nemo_automodel.components.models.minimax_m3_vl.layers import (
    MiniMaxM3Indexer,
    _select_sparse_block_indices,
    build_block_sparse_attn_mask,
    select_sparse_blocks,
)
from nemo_automodel.components.models.minimax_m3_vl.msa_plan import _MSAPackedLayout


def _rand_idx(seqlen, h_idx=4, dim=16, bsz=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    idx_q = torch.randn(bsz, seqlen, h_idx, dim, generator=g)
    idx_k = torch.randn(bsz, seqlen, 1, dim, generator=g)
    return idx_q, idx_k


def _brute_document_local_selection(
    idx_q,
    idx_k,
    *,
    block_size,
    topk,
    init_blocks,
    local_blocks,
    q_positions,
    q_doc_starts,
):
    """Independent per-query LSE reference for fixed-width document-local selection.

    Args:
        idx_q: Index queries of shape [batch, query_tokens, index_heads, dim].
        idx_k: Index keys of shape [batch, key_tokens, 1, dim] on the aligned key axis.
        block_size: Key rows per block.
        topk: Fixed number of block slots per query.
        init_blocks: Blocks forced from the query document's first block.
        local_blocks: A positive value forces the query's current block.
        q_positions: Key-axis position of each query, shape [query_tokens].
        q_doc_starts: Key-axis document start of each query, shape [query_tokens].
    Returns:
        Dict keyed by ``(batch, index_head, query_row)`` holding the set of
        selected key-axis block ids.
    """
    bsz, tq, h_idx, dim = idx_q.shape
    tk = idx_k.shape[1]
    scale = dim**-0.5
    selected = {}
    for b in range(bsz):
        for h in range(h_idx):
            for qi in range(tq):
                pos = int(q_positions[qi])
                lo = int(q_doc_starts[qi]) // block_size
                cur = pos // block_size
                scores = {}
                for blk in range(lo, cur + 1):
                    keys = [j for j in range(blk * block_size, min((blk + 1) * block_size, tk)) if j <= pos]
                    values = torch.tensor(
                        [(idx_q[b, qi, h].float() @ idx_k[b, j, 0].float()).item() * scale for j in keys]
                    )
                    scores[blk] = torch.logsumexp(values, 0).item()
                forced = {cur} if local_blocks > 0 else set()
                forced |= {blk for blk in range(lo, lo + init_blocks) if blk in scores}
                chosen = set(forced)
                ranked = sorted((blk for blk in scores if blk not in forced), key=lambda blk: scores[blk], reverse=True)
                for blk in ranked:
                    if len(chosen) >= min(topk, len(scores)):
                        break
                    chosen.add(blk)
                selected[(b, h, qi)] = chosen
    return selected


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


def test_fixed_width_lse_indices_match_document_local_reference(monkeypatch):
    """Fixed-width support is causal, document-local, rebased, and padded with -1."""
    seqlen, block_size, topk = 40, 4, 3
    idx_q, idx_k = _rand_idx(seqlen, h_idx=2, dim=8, bsz=1, seed=43)
    q_positions = torch.arange(seqlen)
    document_start = 24
    q_doc_starts = torch.where(
        q_positions < document_start,
        torch.zeros_like(q_positions),
        torch.full_like(q_positions, document_start),
    )
    monkeypatch.setattr(layers_module, "_SELECT_SCORE_BUDGET_BYTES", 512)
    selected = _select_sparse_block_indices(
        idx_q,
        idx_k,
        block_size=block_size,
        topk_blocks=topk,
        init_blocks=1,
        local_blocks=1,
        score_type="lse",
        q_positions=q_positions,
        q_doc_starts=q_doc_starts,
    )
    reference = _brute_document_local_selection(
        idx_q,
        idx_k,
        block_size=block_size,
        topk=topk,
        init_blocks=1,
        local_blocks=1,
        q_positions=q_positions,
        q_doc_starts=q_doc_starts,
    )

    assert selected.shape == (1, 2, seqlen, topk)
    assert selected.dtype == torch.int64
    for head in range(2):
        for query in range(seqlen):
            row = selected[0, head, query]
            valid = row[row >= 0]
            start_block = int(q_doc_starts[query]) // block_size
            assert set((valid + start_block).tolist()) == reference[(0, head, query)]
            assert (row[valid.numel() :] == -1).all()


def test_indexer_flat_projection_matches_project_then_pack(sparse_text_config, backend):
    """Changing the token axis preserves the indexer's projected values."""
    indexer = MiniMaxM3Indexer(sparse_text_config, dict(sparse_text_config.sparse_attention_config), backend)
    layout = _MSAPackedLayout.build(torch.tensor([[1, 1, 0, 2, 2], [7, 7, 7, 0, 0]]))
    generator = torch.Generator().manual_seed(19)
    hidden = torch.randn(
        2,
        5,
        sparse_text_config.hidden_size,
        generator=generator,
        dtype=indexer.index_q_proj.weight.dtype,
    )
    half_dim = indexer.index_head_dim // 2
    frequencies = torch.cat((torch.ones(2, 5, half_dim), torch.zeros(2, 5, half_dim)), dim=-1)

    full_q, full_k = indexer._project_qk(hidden, freqs_cis=frequencies, cp_size=1, cp_rank=0)
    flat_q, flat_k = indexer._project_qk(
        layout.pack(hidden),
        freqs_cis=layout.pack(frequencies),
        cp_size=1,
        cp_rank=0,
    )

    torch.testing.assert_close(flat_q, layout.pack(full_q), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(flat_k, layout.pack(full_k), rtol=1e-6, atol=1e-6)


def test_indexer_emits_document_local_q2k(sparse_text_config, backend):
    """The indexer adapts compact documents to canonical local int32 support."""
    sparse_cfg = dict(
        sparse_text_config.sparse_attention_config,
        sparse_block_size=4,
        sparse_topk_blocks=2,
        sparse_init_block=0,
        sparse_local_block=1,
    )
    indexer = MiniMaxM3Indexer(sparse_text_config, sparse_cfg, backend)
    doc_ids = torch.cat(
        (
            torch.ones(14, dtype=torch.int64),
            torch.zeros(1, dtype=torch.int64),
            torch.full((10,), 2, dtype=torch.int64),
        )
    ).unsqueeze(0)
    layout = _MSAPackedLayout.build(doc_ids)
    index_q = torch.zeros(24, indexer.num_index_heads, indexer.index_head_dim)
    index_q[..., 0] = 1
    index_k = torch.zeros(24, 1, indexer.index_head_dim)
    index_k[:, 0, 0] = torch.tensor([9.0] * 4 + [2.0] * 4 + [5.0] * 4 + [-1.0] * 2 + [1.0] * 4 + [8.0] * 4 + [-1.0] * 2)

    q2k = indexer._select_msa_blocks(index_q, index_k, layout=layout)

    assert q2k.shape == (indexer.num_index_heads, 24, 2)
    assert q2k.dtype == torch.int32
    assert q2k.is_contiguous()
    assert q2k[:, 0, 0].eq(0).all() and q2k[:, 0, 1].eq(-1).all()
    assert set(q2k[0, 13].tolist()) == {0, 3}
    assert set(q2k[0, 23].tolist()) == {1, 2}


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        pytest.param({"block_size": 0}, "block_size must be positive", id="block-size"),
        pytest.param({"topk_blocks": 0}, "topk_blocks must be positive", id="topk"),
        pytest.param({"score_type": "mean"}, "score_type", id="score-type"),
        pytest.param({"q_positions": torch.arange(7)}, "q_positions", id="query-positions"),
        pytest.param({"q_doc_starts": torch.arange(7)}, "q_doc_starts", id="document-starts"),
    ],
)
def test_fixed_width_indices_reject_invalid_contract(overrides, match):
    idx_q, idx_k = _rand_idx(8, h_idx=2, dim=8, bsz=1, seed=31)
    kwargs = dict(block_size=4, topk_blocks=2, init_blocks=0, local_blocks=1)
    kwargs.update(overrides)

    with pytest.raises(ValueError, match=match):
        _select_sparse_block_indices(idx_q, idx_k, **kwargs)
