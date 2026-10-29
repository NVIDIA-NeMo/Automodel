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

import math
from collections.abc import Iterator, Sequence
from dataclasses import fields, replace
from types import SimpleNamespace

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.step5 import fa4, sparse
from nemo_automodel.components.models.step5.configuration import Step5BackendConfig
from nemo_automodel.components.models.step5.fa4 import (
    _all_padding_attention,
    _BackwardBlockPartition,
    _provider_heads,
    _split_backward_full_blocks,
    _split_forward_full_blocks,
)
from nemo_automodel.components.models.step5.model import Step5ForCausalLM
from nemo_automodel.components.models.step5.sparse import (
    CausalDocumentLayout,
    RegionMetadata,
    Step5SparseIndexer,
    _summarize_region_payload,
    _summary_ranges,
    gather_region_payload,
    region_metadata,
)
from tests.unit_tests.models.step5.test_model import tiny_config


def _indexer(*, topk: int = 2, region_size: int = 2) -> Step5SparseIndexer:
    cfg = tiny_config(
        sparse_config=dict(
            proxy_dim=16,
            sparse_indexer_rope_dim=4,
            sparse_indexer_num_heads=4,
            num_provider_groups=2,
            topk=topk,
            region_block_size=region_size,
        )
    )
    torch.manual_seed(329)
    module = Step5SparseIndexer(cfg, 3, dtype=torch.float32)
    with torch.no_grad():
        module.q_norm.weight.zero_()
        module.k_norm.weight.fill_(1)
        module.k_norm.bias.zero_()
    return module


@torch.no_grad()
def _oracle(
    module: Step5SparseIndexer,
    hidden: torch.Tensor,
    positions: torch.Tensor,
    keys: torch.Tensor,
    logits: torch.Tensor,
    global_positions: torch.Tensor,
    padding: torch.Tensor,
    *,
    offset: int,
) -> torch.Tensor:
    """Select by a Python document scan and stable full sort.

    Args:
        module: Indexer whose projection weights define the scoring problem.
        hidden: FP32 local states [batch, queries, hidden].
        positions: Local integer positions [batch, queries].
        keys: Projected global keys [batch, keys, proxy_dim].
        logits: Global compression logits, shaped like keys.
        global_positions: Integer document-local positions [batch, keys].
        padding: Boolean global padding mask [batch, keys].
        offset: First physical global token owned by hidden.

    Returns:
        Int32 selected IDs [batch, groups, queries, budget + 1].
    """
    cfg = module.config
    batch, queries = hidden.shape[:2]
    groups = cfg.num_provider_groups
    heads = cfg.sparse_indexer_num_heads // groups
    q = module.q_proj(hidden).reshape(batch, queries, groups * heads, cfg.proxy_dim)
    q = module.rotary_emb(module.q_norm(q), positions).reshape(batch, queries, groups, heads, cfg.proxy_dim)
    q = q.to(torch.float8_e4m3fn).float()
    weights = module.w_proj(hidden).reshape(batch, queries, groups, heads) * heads**-0.5
    layouts = []
    for b in range(batch):
        region_docs, region_tokens, token_ids = [], [], []
        document = 0
        previous = None
        for token, pos in enumerate(global_positions[b].tolist()):
            if pos == 0 and not bool(padding[b, token]):
                document += 1
            identity = (document, pos // cfg.region_block_size)
            if identity != previous:
                region_docs.append(document)
                region_tokens.append({})
                previous = identity
            region_tokens[-1][pos % cfg.region_block_size] = token
            token_ids.append(len(region_tokens) - 1)
        summaries = []
        for slots in region_tokens:
            ts = list(slots.values())
            z = logits[b, ts].float().masked_fill(padding[b, ts, None], -torch.inf)
            probabilities = (z - z.amax()).exp().nan_to_num(0)
            summary = (probabilities * keys[b, ts].float()).sum(0) / probabilities.sum(0).clamp_min(1e-20)
            summaries.append(summary.to(torch.bfloat16).to(torch.float8_e4m3fn).float())
        layouts.append((region_docs, token_ids, torch.stack(summaries)))
    count = max(len(x[0]) for x in layouts)
    budget = min(cfg.topk, count)
    output = torch.full((batch, groups, queries, budget + 1), count, dtype=torch.int32)
    for b, (docs, token_ids, summary) in enumerate(layouts):
        for local in range(queries):
            current = token_ids[offset + local]
            candidates = [i for i in range(current) if docs[i] == docs[current]]
            for group in range(groups):
                # Head-wise FP32 accumulation matches the reference's specified
                # score, while selection uses an independent stable Python sort.
                scores = torch.zeros(len(summary))
                for head in range(heads):
                    dots = torch.mv(summary, q[b, local, group, head]).relu()
                    scores += dots * weights[b, local, group, head]
                chosen = sorted(candidates, key=lambda i: (-float(scores[i]), i))[:budget]
                chosen = sorted(chosen + [current])
                output[b, group, local, : len(chosen)] = torch.tensor(chosen)
    return output


@pytest.mark.parametrize("offset,length", [(0, 35), (3, 13), (17, 18)])
@pytest.mark.parametrize("tied", [False, True])
def test_packed_selection_matches_independent_stable_sort(offset: int, length: int, tied: bool) -> None:
    module = _indexer()
    positions = torch.cat([torch.arange(n) for n in (3, 11, 1, 20)])[None]
    hidden = torch.randn(1, 35, 32)
    if tied:
        with torch.no_grad():
            module.w_proj.weight.zero_()
    padding = torch.zeros_like(positions, dtype=torch.bool)
    keys, logits = module.project_keys(hidden, positions)
    local = hidden[:, offset : offset + length]
    local_positions = positions[:, offset : offset + length]
    meta = region_metadata(positions, padding, query_start=offset, queries=length, region_size=2, topk=2)
    actual = module(local, local_positions, keys, logits, positions, padding, query_start=offset, metadata=meta)
    expected = _oracle(module, local, local_positions, keys, logits, positions, padding, offset=offset)
    torch.testing.assert_close(actual.region_ids, expected, rtol=0, atol=0)


def test_dense_prefix_skips_projection_scoring_and_preserves_partial_documents() -> None:
    module = _indexer()
    positions = torch.cat([torch.arange(n) for n in (1, 5, 3, 6)])[None]
    padding = torch.zeros_like(positions, dtype=torch.bool)
    hidden = torch.randn(1, positions.shape[1], 32)
    meta = region_metadata(positions, padding, query_start=0, queries=positions.shape[1], region_size=2, topk=2)
    assert not meta.needs_projection
    assert not meta.score_tiles

    keys, logits = module.project_keys(hidden, positions)
    calls = []
    for projection in (module.q_proj, module.q_norm, module.w_proj):
        projection.register_forward_pre_hook(lambda module, args: calls.append(args[0].shape[1]))
    actual = module(hidden, positions, keys, logits, positions, padding)
    assert calls == [0, 0, 0]  # FSDP hooks remain ordered; every projection has zero tokens.
    # Uneven documents own separate partial regions, even when their physical
    # starts do not align with the region size.
    assert actual.region_ids[0, 0, 0].tolist() == [0, 9, 9]
    assert actual.region_ids[0, 0, 5].tolist() == [1, 2, 3]
    assert actual.region_ids[0, 0, 8].tolist() == [4, 5, 9]
    assert actual.region_ids[0, 0, 14].tolist() == [6, 7, 8]
    assert actual.region_ids.stride(1) == 0
    assert actual.all_regions_selected


def test_global_projection_decision_is_identical_for_asymmetric_cp_shards() -> None:
    positions = torch.cat((torch.arange(4), torch.arange(20)))[None]
    padding = torch.zeros_like(positions, dtype=torch.bool)
    shards = [
        region_metadata(positions, padding, query_start=i * 4, queries=4, region_size=2, topk=2) for i in range(6)
    ]
    assert all(x.needs_projection for x in shards)
    assert not shards[0].score_tiles and not shards[1].score_tiles
    assert shards[-1].score_tiles
    assert all(x.region_count == 12 for x in shards)


@pytest.mark.parametrize("offset,length", [(4090, 30), (8176, 16)])
def test_real_region_budget_and_eight_k_document(offset: int, length: int) -> None:
    module = _indexer(topk=512, region_size=8)
    positions = torch.arange(8192)[None]
    padding = torch.zeros_like(positions, dtype=torch.bool)
    hidden = torch.randn(1, 8192, 32)
    keys, logits = module.project_keys(hidden, positions)
    local, local_positions = hidden[:, offset : offset + length], positions[:, offset : offset + length]
    meta = region_metadata(positions, padding, query_start=offset, queries=length, region_size=8, topk=512)
    actual = module(local, local_positions, keys, logits, positions, padding, query_start=offset, metadata=meta)
    expected = _oracle(module, local, local_positions, keys, logits, positions, padding, offset=offset)
    torch.testing.assert_close(actual.region_ids, expected, rtol=0, atol=0)
    if offset == 4090:
        assert not meta.sparse_queries[0, :14].any()  # positions 4090 through 4103
        assert meta.sparse_queries[0, 14:].all()
    assert meta.needs_projection


def test_partial_regions_and_padding_do_not_create_cross_document_routes() -> None:
    module = _indexer()
    positions = torch.tensor([[0, 1, 2, 0, 1, 2, 3, 4, 5, 6, 0, 0]])
    padding = torch.tensor([[False] * 10 + [True, True]])
    hidden = torch.randn(1, 12, 32)
    keys, logits = module.project_keys(hidden, positions)
    actual = module(hidden, positions, keys, logits, positions, padding)
    expected = _oracle(module, hidden, positions, keys, logits, positions, padding, offset=0)
    torch.testing.assert_close(actual.region_ids, expected, rtol=0, atol=0)


def _native_mask(layout: CausalDocumentLayout, queries: int, keys: int) -> torch.Tensor:
    """Expand only these tiny CPU fixtures using bottom-right causal lengths."""
    mask = torch.zeros(queries, keys, dtype=torch.bool)
    q_bounds = layout.cu_seqlens_q.tolist()
    k_bounds = layout.cu_seqlens_k.tolist()
    for q0, q1, k0, k1 in zip(q_bounds, q_bounds[1:], k_bounds, k_bounds[1:]):
        for query in range(q1 - q0):
            for key in range(k1 - k0):
                mask[q0 + query, layout.key_start + k0 + key] = key <= query + (k1 - k0) - (q1 - q0)
    if layout.padding_key is not None:
        mask[layout.valid_queries :, layout.padding_key] = True
    return mask


def _oracle_causal_documents(
    positions: torch.Tensor, start: int, queries: int, padding: torch.Tensor | None = None
) -> torch.Tensor:
    """Build a mask from document resets and absolute positions, without cu data."""
    if padding is None:
        padding = torch.zeros_like(positions, dtype=torch.bool)
    docs = []
    document = -1
    for position, padded in zip(positions[0].tolist(), padding[0].tolist()):
        if position == 0 and not padded:
            document += 1
        docs.append(document)
    mask = torch.zeros(queries, positions.shape[1], dtype=torch.bool)
    for query in range(queries):
        physical = start + query
        for key in range(positions.shape[1]):
            mask[query, key] = (
                not padding[0, key] and docs[key] == docs[physical] and positions[0, key] <= positions[0, physical]
            )
    return mask


@pytest.mark.parametrize(
    "lengths,start,queries",
    [((9,), 2, 4), ((9, 3, 7), 5, 10), ((9, 3, 7), 9, 3), ((3, 5, 4), 0, 12), ((1, 1, 2), 1, 2)],
)
def test_native_document_layout_matches_independent_mask(lengths: tuple[int, ...], start: int, queries: int) -> None:
    positions = torch.cat([torch.arange(n) for n in lengths])[None]
    meta = region_metadata(
        positions,
        torch.zeros_like(positions, dtype=torch.bool),
        query_start=start,
        queries=queries,
        region_size=8,
        topk=512,
    )
    layout = meta.causal_layout
    assert layout is not None
    assert layout.key_stop == start + queries
    assert layout.key_start == start - int(positions[0, start])
    assert layout.cu_seqlens_q.dtype == layout.cu_seqlens_k.dtype == torch.int32
    assert int(layout.cu_seqlens_q[-1]) == queries
    assert int(layout.cu_seqlens_k[-1]) == layout.key_stop - layout.key_start
    assert int(torch.diff(layout.cu_seqlens_q).max()) == layout.max_seqlen_q
    assert int(torch.diff(layout.cu_seqlens_k).max()) == layout.max_seqlen_k
    torch.testing.assert_close(
        _native_mask(layout, queries, positions.shape[1]), _oracle_causal_documents(positions, start, queries)
    )


def test_native_layout_rejects_padding_in_required_keys_but_allows_future_padding() -> None:
    positions = torch.arange(9)[None]
    for padded_key, eligible in ((0, False), (3, False), (8, True)):
        padding = torch.zeros_like(positions, dtype=torch.bool)
        padding[0, padded_key] = True
        meta = region_metadata(positions, padding, query_start=2, queries=4, region_size=8, topk=512)
        assert (meta.causal_layout is not None) == eligible


@pytest.mark.parametrize("positions", [[[0, 1, 3, 4]], [[0, 1, 1, 2]], [[2, 3, 4, 5]]])
def test_native_layout_falls_back_for_noncontiguous_document_positions(positions: list[list[int]]) -> None:
    positions = torch.tensor(positions)
    meta = region_metadata(
        positions, torch.zeros_like(positions, dtype=torch.bool), query_start=0, queries=4, region_size=8, topk=512
    )
    assert meta.causal_layout is None


def test_native_layout_falls_back_for_multiple_batches_and_sparse_queries() -> None:
    positions = torch.arange(12)[None].expand(2, -1)
    meta = region_metadata(
        positions, torch.zeros_like(positions, dtype=torch.bool), query_start=0, queries=12, region_size=2, topk=2
    )
    assert meta.causal_layout is None
    meta = region_metadata(
        positions[:1],
        torch.zeros_like(positions[:1], dtype=torch.bool),
        query_start=5,
        queries=7,
        region_size=2,
        topk=2,
    )
    assert meta.score_tiles and meta.causal_layout is None


def test_document_mask_oracle_detects_wrong_bottom_right_alignment() -> None:
    positions = torch.arange(9)[None]
    meta = region_metadata(
        positions, torch.zeros_like(positions, dtype=torch.bool), query_start=2, queries=4, region_size=8, topk=512
    )
    layout = meta.causal_layout
    expected = _oracle_causal_documents(positions, 2, 4)
    # Keeping the future tail shifts bottom-right causality three tokens right.
    future = replace(layout, key_stop=9, cu_seqlens_k=torch.tensor([0, 9], dtype=torch.int32), max_seqlen_k=9)
    # Dropping the document prefix instead loses keys before the CP boundary.
    missing = replace(layout, key_start=2, cu_seqlens_k=torch.tensor([0, 4], dtype=torch.int32), max_seqlen_k=4)
    for incorrect in (future, missing):
        with pytest.raises(AssertionError):
            torch.testing.assert_close(_native_mask(incorrect, 4, 9), expected)


@pytest.mark.parametrize("lengths,start,padded", [((9, 7), 0, 3), ((9, 7), 5, 3), ((9, 7), 11, 3), ((3, 1), 3, 1)])
def test_native_layout_trims_only_trailing_padding_and_keeps_singleton_value(
    lengths: tuple[int, ...], start: int, padded: int
) -> None:
    positions = torch.cat([torch.arange(n) for n in lengths] + [torch.zeros(padded, dtype=torch.long)])[None]
    padding = torch.arange(positions.shape[1])[None] >= sum(lengths)
    queries = positions.shape[1] - start
    meta = region_metadata(positions, padding, query_start=start, queries=queries, region_size=8, topk=512)
    layout = meta.causal_layout
    assert layout is not None
    assert layout.valid_queries == sum(lengths) - start
    assert layout.key_stop == sum(lengths)
    assert layout.padding_key == sum(lengths[:-1])
    expected = _oracle_causal_documents(positions, start, queries, padding)
    torch.testing.assert_close(_native_mask(layout, queries, positions.shape[1]), expected)
    # Neither zero outputs nor the first document's V preserve pad semantics.
    incorrect = _native_mask(layout, queries, positions.shape[1])
    incorrect[layout.valid_queries :] = False
    with pytest.raises(AssertionError):
        torch.testing.assert_close(incorrect, expected)
    wrong_document = replace(layout, padding_key=0)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(_native_mask(wrong_document, queries, positions.shape[1]), expected)


@pytest.mark.parametrize(
    "positions,padding,start,topk",
    [
        ([0, 1, 0, 2], [False, False, True, False], 0, 512),
        ([0, 1, 2, 3], [False, False, False, True], 0, 512),
        ([0, 1, 0, 0], [False, False, True, True], 2, 512),
        ([0, 0, 0, 0], [True, True, True, True], 0, 512),
        (list(range(9)) + [0, 0], [False] * 9 + [True] * 2, 0, 2),
    ],
)
def test_native_padding_layout_rejects_holes_nonzero_positions_empty_queries_and_sparse_tail(
    positions: list[int], padding: list[bool], start: int, topk: int
) -> None:
    meta = region_metadata(
        torch.tensor([positions]),
        torch.tensor([padding]),
        query_start=start,
        queries=len(positions) - start,
        region_size=2,
        topk=topk,
    )
    assert meta.causal_layout is None


@pytest.mark.parametrize(
    "lengths,start,queries",
    [
        (((9, 3, 7), (3, 5, 11)), 5, 16),
        (((3,), (9, 7)), 5, 16),
        (((3, 5), (4,)), 11, 10),
        (((9, 3, 7), (3, 5, 11)), 3, 9),
    ],
)
def test_b2_layouts_match_independent_masks_and_single_batch_fields(
    lengths: tuple[tuple[int, ...], tuple[int, ...]], start: int, queries: int
) -> None:
    """Each row has different boundaries and may be a proven all-padding shard."""
    keys = 21
    positions = torch.stack(
        [
            torch.cat([torch.arange(n) for n in row] + [torch.zeros(keys - sum(row), dtype=torch.long)])
            for row in lengths
        ]
    )
    padding = torch.stack([torch.arange(keys) >= sum(row) for row in lengths])
    metadata = region_metadata(positions, padding, query_start=start, queries=queries, region_size=8, topk=512)
    assert metadata.causal_layout is None and metadata.all_padding_key is None
    assert metadata.causal_batch_layouts is not None and len(metadata.causal_batch_layouts) == 2
    for row, layout in enumerate(metadata.causal_batch_layouts):
        singleton = region_metadata(
            positions[row : row + 1],
            padding[row : row + 1],
            query_start=start,
            queries=queries,
            region_size=8,
            topk=512,
        )
        assert singleton.causal_batch_layouts is None
        expected = _oracle_causal_documents(positions[row : row + 1], start, queries, padding[row : row + 1])
        torch.testing.assert_close(_native_mask(layout, queries, keys), expected, rtol=0, atol=0)
        if singleton.all_padding_key is not None:
            assert layout.valid_queries == 0
            assert layout.padding_key == singleton.all_padding_key
        else:
            old = singleton.causal_layout
            assert old is not None
            for field in (
                "key_start",
                "key_stop",
                "max_seqlen_q",
                "max_seqlen_k",
                "valid_queries",
                "padding_key",
                "leading_padding",
            ):
                assert vars(layout)[field] == vars(old)[field]
            torch.testing.assert_close(layout.cu_seqlens_q, old.cu_seqlens_q, rtol=0, atol=0)
            torch.testing.assert_close(layout.cu_seqlens_k, old.cu_seqlens_k, rtol=0, atol=0)


@pytest.mark.parametrize(
    "defect", ["internal_padding", "duplicate_position", "nonzero_padding", "no_valid_key", "sparse"]
)
def test_b2_layout_falls_back_when_either_row_is_unproven(defect: str) -> None:
    positions = torch.cat((torch.arange(24), torch.zeros(8, dtype=torch.long)))[None].expand(2, -1).clone()
    padding = torch.arange(32)[None].expand(2, -1) >= 24
    topk = 512
    if defect == "internal_padding":
        padding[1, 10] = True
    elif defect == "duplicate_position":
        positions[1, 10] = 9
    elif defect == "nonzero_padding":
        positions[1, 24:] = 1
    elif defect == "no_valid_key":
        positions[1] = 0
        padding[1] = True
    else:
        topk = 1
        positions[0, :24] = torch.arange(24) % 4
    metadata = region_metadata(positions, padding, query_start=8, queries=24, region_size=8, topk=topk)
    assert metadata.causal_batch_layouts is None
    assert metadata.causal_layout is None


def test_b2_layout_uses_one_metadata_host_transfer_and_leaves_swa_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    original = torch.Tensor.cpu

    def observed_cpu(tensor: torch.Tensor, memory_format: torch.memory_format = torch.preserve_format) -> torch.Tensor:
        """Observe the existing host transfer.

        Args:
            tensor: Tensor of arbitrary metadata shape.
            memory_format: Unchanged torch transfer option.

        Returns:
            CPU tensor with the same shape and values.
        """
        calls.append(tensor.shape)
        return original(tensor, memory_format=memory_format)

    monkeypatch.setattr(torch.Tensor, "cpu", observed_cpu)
    positions = torch.arange(17)[None].expand(2, -1)
    metadata = region_metadata(
        positions,
        torch.zeros_like(positions, dtype=torch.bool),
        query_start=3,
        queries=11,
        region_size=8,
        topk=512,
        window=5,
    )
    assert len(calls) == 1
    assert metadata.causal_batch_layouts is not None
    assert metadata.window_layout is None
    three = positions[:1].expand(3, -1)
    metadata = region_metadata(
        three, torch.zeros_like(three, dtype=torch.bool), query_start=3, queries=11, region_size=8, topk=512
    )
    assert metadata.causal_batch_layouts is None and metadata.causal_layout is None


@pytest.mark.parametrize("lengths", [(3,), (9, 3), (13, 4096), (1, 1, 7), (19, 1), (4096,)])
@pytest.mark.parametrize("shard_offset", [0, 3])
def test_all_padding_key_matches_independent_document_scan(lengths: tuple[int, ...], shard_offset: int) -> None:
    positions = torch.cat([torch.arange(n) for n in lengths] + [torch.zeros(19, dtype=torch.long)])[None]
    valid = sum(lengths)
    padding = torch.arange(positions.shape[1])[None] >= valid
    metadata = region_metadata(
        positions, padding, query_start=valid + shard_offset, queries=19 - shard_offset, region_size=8, topk=512
    )
    assert metadata.all_padding_key == sum(lengths[:-1])
    assert metadata.causal_layout is None
    assert not metadata.score_tiles
    # Scan the independently described documents: a position-zero padding query
    # keeps the final document ID and can see exactly that document's first key.
    docs = [i for i, n in enumerate(lengths) for _ in range(n)] + [len(lengths) - 1] * 19
    for query in range(valid + shard_offset, valid + 19):
        allowed = [key for key in range(valid) if docs[key] == docs[query] and positions[0, key] <= positions[0, query]]
        assert allowed == [metadata.all_padding_key]


@pytest.mark.parametrize(
    "positions,padding,start",
    [
        ([0, 0, 0, 0], [True, True, True, True], 2),
        ([0, 1, 0, 2, 0, 0], [False, False, True, False, True, True], 4),
        ([0, 2, 3, 0, 0], [False, False, False, True, True], 3),
        ([5, 6, 7, 0, 0], [False, False, False, True, True], 3),
        ([0, 1, 2, 1, 1], [False, False, False, True, True], 3),
        ([0, 1, 2, 0, 0], [False, False, False, True, True], 2),
    ],
)
def test_all_padding_key_falls_back_for_unproven_layout(positions: list[int], padding: list[bool], start: int) -> None:
    metadata = region_metadata(
        torch.tensor([positions]),
        torch.tensor([padding]),
        query_start=start,
        queries=len(positions) - start,
        region_size=8,
        topk=512,
    )
    assert metadata.all_padding_key is None


def test_all_padding_key_rejects_sparse_final_document_and_multiple_batches() -> None:
    # Resetting to pad position0 creates another region after a4097-token doc.
    # Those padded queries use learned selection, which may drop its first key.
    positions = torch.cat((torch.arange(4097), torch.zeros(31, dtype=torch.long)))[None]
    padding = torch.arange(positions.shape[1])[None] >= 4097
    metadata = region_metadata(positions, padding, query_start=4097, queries=31, region_size=8, topk=512)
    assert metadata.score_tiles
    assert metadata.all_padding_key is None
    positions = torch.tensor([[0, 1, 0, 0]]).expand(2, -1)
    padding = torch.tensor([[False, False, True, True]]).expand(2, -1)
    metadata = region_metadata(positions, padding, query_start=2, queries=2, region_size=8, topk=512)
    assert metadata.all_padding_key is None


def test_indexer_preserves_proven_all_padding_key() -> None:
    config = tiny_config(
        sparse_config=dict(
            proxy_dim=16,
            sparse_indexer_rope_dim=4,
            sparse_indexer_num_heads=4,
            num_provider_groups=2,
            topk=512,
            region_block_size=8,
        )
    )
    module = Step5SparseIndexer(config, 3, dtype=torch.float32)
    positions = torch.tensor([[0, 1, 2, 0, 1, 0, 0]])
    padding = torch.tensor([[False, False, False, False, False, True, True]])
    metadata = region_metadata(positions, padding, query_start=5, queries=2, region_size=8, topk=512)
    selected = module(
        torch.zeros(1, 2, config.hidden_size),
        positions[:, 5:],
        None,
        None,
        positions,
        padding,
        query_start=5,
        metadata=metadata,
    )
    assert selected.all_regions_selected
    assert selected.all_padding_key == 3


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16])
@pytest.mark.parametrize("nonfinite_qk", [False, True])
def test_singleton_value_preserves_bits_and_explicit_zero_gradients(dtype: torch.dtype, nonfinite_qk: bool) -> None:
    torch.manual_seed(33)
    query = torch.randn(1, 9, 8, 3, dtype=dtype, requires_grad=True)
    # Exercise the exact noncontiguous stack/unbind K/V storage layout.
    kv = torch.randn(1, 23, 2, 2, 3, dtype=dtype, requires_grad=True)
    with torch.no_grad():
        kv[0, 13, 1, 0, 0] = -0.0
        if nonfinite_qk:
            query.fill_(torch.nan)
            kv[:, :, 0].fill_(torch.nan)
    key, value = kv.unbind(dim=2)
    key.retain_grad()
    output = _all_padding_attention(query, key, value, 13)
    expected = value[:, 13:14].repeat_interleave(4, dim=2).expand_as(output).contiguous()
    assert torch.equal(output.view(torch.uint8), expected.view(torch.uint8))
    upstream = torch.arange(output.numel()).reshape_as(output).remainder(5).to(dtype)
    output.backward(upstream)
    assert query.grad is not None and torch.equal(query.grad, torch.zeros_like(query))
    assert key.grad is not None and torch.equal(key.grad, torch.zeros_like(key))
    wanted = torch.zeros_like(kv)
    wanted[:, 13, 1] = upstream.reshape(1, 9, 2, 4, 3).sum((1, 3))
    assert torch.equal(kv.grad, wanted)


def test_singleton_key_negative_control_detects_wrong_document() -> None:
    query = torch.zeros(1, 3, 4, 2)
    key = torch.zeros(1, 7, 2, 2)
    value = torch.arange(28).reshape(1, 7, 2, 2).float()
    correct = _all_padding_attention(query, key, value, 3)
    wrong = _all_padding_attention(query, key, value, 0)
    assert not torch.equal(correct, wrong)


def test_indexer_preserves_proven_b2_prefix_and_all_padding_rows() -> None:
    config = tiny_config(
        sparse_config=dict(
            proxy_dim=16,
            sparse_indexer_rope_dim=4,
            sparse_indexer_num_heads=4,
            num_provider_groups=2,
            topk=512,
            region_block_size=8,
        )
    )
    module = Step5SparseIndexer(config, 3, dtype=torch.float32)
    positions = torch.tensor([[0, 1, 2, 0, 1, 0, 0], [0, 1, 0, 0, 0, 0, 0]])
    padding = torch.tensor([[False] * 5 + [True] * 2, [False] * 2 + [True] * 5])
    metadata = region_metadata(positions, padding, query_start=3, queries=4, region_size=8, topk=512)
    assert metadata.causal_batch_layouts is not None
    assert [layout.valid_queries for layout in metadata.causal_batch_layouts] == [2, 0]
    selected = module(
        torch.zeros(2, 4, config.hidden_size),
        positions[:, 3:],
        None,
        None,
        positions,
        padding,
        query_start=3,
        metadata=metadata,
    )
    assert selected.all_regions_selected
    assert selected.causal_batch_layouts is metadata.causal_batch_layouts
    assert selected.causal_layout is None and selected.all_padding_key is None


def _metadata(
    positions: torch.Tensor,
    padding: torch.Tensor,
    *,
    enabled: bool,
    start: int = 0,
    queries: int | None = None,
    topk: int = 512,
) -> RegionMetadata:
    """Prepare the shared CPU region layout.

    Args:
        positions: Integer positions [batch, global_keys].
        padding: Boolean padding flags with the same layout.
        enabled: Whether to request the optional forward proof.
        start: Physical global offset of the local query shard.
        queries: Local query count, or the complete key length.
        topk: Selected historical region budget.

    Returns:
        RegionMetadata with the tensor layouts documented on that dataclass.
    """
    return region_metadata(
        positions,
        padding,
        query_start=start,
        queries=positions.shape[1] if queries is None else queries,
        region_size=8,
        topk=topk,
        csa_forward_full_blocks=enabled,
    )


def _assert_existing_fields_equal(first: RegionMetadata, second: RegionMetadata) -> None:
    """Compare old fields of two shared region layouts.

    Args:
        first: RegionMetadata with global [batch, keys] token/document tensors,
            local [batch, queries] positions, [batch, queries, slots] route
            templates and optional compact region/document layouts.
        second: Metadata with the same per-field layouts and dtypes.
    """

    def equal(a: object, b: object) -> None:
        if isinstance(a, torch.Tensor):
            assert isinstance(b, torch.Tensor)
            assert a.dtype == b.dtype and a.shape == b.shape
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        elif isinstance(a, tuple):
            assert isinstance(b, tuple) and len(a) == len(b)
            for x, y in zip(a, b, strict=True):
                equal(x, y)
        elif a is None or isinstance(a, (bool, int)):
            assert a == b
        else:
            assert type(a) is type(b)
            for field in fields(a):
                equal(a.__dict__[field.name], b.__dict__[field.name])

    for field in fields(first):
        if field.name not in ("forward_full_block_prefix", "canonical_region_size"):
            equal(first.__dict__[field.name], second.__dict__[field.name])


def _route_block_lists(length: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Build CPU int32 lists [2,1,Qtiles] / [2,1,Qtiles,Ktiles].

    A valid synthetic router chooses the most recent 512 completed regions,
    plus the current region. Each region contains eight tokens. This is an
    independent legal routing fixture, not a model score implementation.
    """
    rows, columns = (length + 127) // 128, (length + 111) // 112
    counts = torch.zeros(2, 1, rows, dtype=torch.int32)
    indices = torch.full((2, 1, rows, columns), columns, dtype=torch.int32)
    for row in range(rows):
        start, stop = row * 128, min((row + 1) * 128, length)
        first_allowed_key = max(0, start // 8 - 512) * 8
        blocks = list(range(first_allowed_key // 112, (stop + 111) // 112))
        counts[:, :, row] = len(blocks)
        indices[:, :, row, : len(blocks)] = torch.tensor(blocks, dtype=torch.int32)
    return counts, indices


def _assert_pair_oracle(
    counts: torch.Tensor,
    indices: torch.Tensor,
    partial_counts: torch.Tensor,
    partial_indices: torch.Tensor,
    full_counts: torch.Tensor,
    *,
    length: int,
    document_reset: int | None = None,
) -> None:
    """Check block partitioning against explicit token-pair semantics.

    Args:
        counts: Original int32 counts [providers, 1, query_tiles].
        indices: Sorted original indices [providers, 1, query_tiles, key_tiles].
        partial_counts: Remaining partial counts with the counts layout.
        partial_indices: Remaining partial indices with the indices layout.
        full_counts: Full counts with the counts layout; indices aliases its list.
        length: Actual token count, excluding tile padding.
        document_reset: Optional physical boundary making cross-document pairs
            invalid independently of the candidate metadata.
    """
    for provider in range(counts.shape[0]):
        for row in range(counts.shape[-1]):
            old = indices[provider, 0, row, : int(counts[provider, 0, row])].tolist()
            partial = partial_indices[provider, 0, row, : int(partial_counts[provider, 0, row])].tolist()
            full = indices[provider, 0, row, : int(full_counts[provider, 0, row])].tolist()
            assert old == sorted(set(old)) and all(block >= 0 for block in old)
            assert len(set(partial + full)) == len(old)
            assert set(partial + full) == set(old)
            assert list(reversed(partial)) + list(reversed(full)) == list(reversed(old))
            q = torch.arange(row * 128, min((row + 1) * 128, length))[:, None]
            for block in full:
                k = torch.arange(block * 112, min((block + 1) * 112, length))[None]
                # Explicit independent causal / selected-region / document mask.
                allowed = (k <= q) & (k // 8 >= (q // 8 - 512).clamp_min(0))
                if document_reset is not None:
                    allowed &= (q >= document_reset) == (k >= document_reset)
                assert allowed.all(), (row, block)


@pytest.mark.parametrize("length", [4096, 4104, 4224, 8192])
def test_single_document_proof_and_prior_metadata_unchanged(length: int) -> None:
    positions = torch.arange(length)[None]
    padding = torch.zeros_like(positions, dtype=torch.bool)
    old = _metadata(positions, padding, enabled=False)
    candidate = _metadata(positions, padding, enabled=True)
    assert old.forward_full_block_prefix == 0
    assert candidate.forward_full_block_prefix == min(length, 4104)
    _assert_existing_fields_equal(old, candidate)
    if length <= 4104:
        assert candidate.causal_layout is not None  # Existing native path stays first.


@pytest.mark.parametrize(
    "case",
    [
        "packed",
        "batch2",
        "cp_first",
        "cp_second",
        "trailing_pad",
        "internal_pad",
        "repeated",
        "missing",
        "nonzero_start",
        "negative",
    ],
)
def test_unproven_layout_preserves_old_metadata(case: str) -> None:
    positions = torch.arange(512)[None]
    padding = torch.zeros_like(positions, dtype=torch.bool)
    start, queries = 0, 512
    if case == "packed":
        positions[:, 256:] -= 256
    elif case == "batch2":
        positions = positions.expand(2, -1)
        padding = padding.expand(2, -1)
    elif case == "cp_first":
        queries = 256
    elif case == "cp_second":
        start, queries = 256, 256
    elif case == "trailing_pad":
        padding[:, 479:] = True
        positions[:, 479:] = 0
    elif case == "internal_pad":
        padding[:, 111] = True
        positions[:, 111] = 0
    elif case == "repeated":
        positions[:, 111] = 110
    elif case == "missing":
        positions[:, 111:] += 1
    elif case == "nonzero_start":
        positions += 8
    elif case == "negative":
        positions[:, 0] = -1
    old = _metadata(positions, padding, enabled=False, start=start, queries=queries, topk=2)
    candidate = _metadata(positions, padding, enabled=True, start=start, queries=queries, topk=2)
    assert candidate.forward_full_block_prefix == 0
    _assert_existing_fields_equal(old, candidate)


@pytest.mark.parametrize("length", [4096, 8192])
def test_full_blocks_match_independent_pairs_union_and_order(length: int) -> None:
    counts, indices = _route_block_lists(length)
    saved_counts, saved_indices = counts.clone(), indices.clone()
    partial_counts, partial_indices, full_counts = _split_forward_full_blocks.__wrapped__(
        counts, indices, prefix_length=min(length, 4104)
    )
    assert full_counts.sum() > 0
    assert int(full_counts[0, 0, 31]) == 35
    if length > 4104:
        assert torch.equal(full_counts[:, :, 32:], torch.zeros_like(full_counts[:, :, 32:]))
    _assert_pair_oracle(counts, indices, partial_counts, partial_indices, full_counts, length=length)
    assert torch.equal(counts, saved_counts) and torch.equal(indices, saved_indices)


@pytest.mark.parametrize("missing", [0, 2])
def test_discontinuous_low_block_prefix_is_not_marked_full(missing: int) -> None:
    counts, indices = _route_block_lists(4096)
    # Sorted unique nonnegative indices can still have a hole in the low prefix.
    indices[:, :, 4, missing:-1] = indices[:, :, 4, missing + 1 :].clone()
    counts[:, :, 4] -= 1
    _, _, full_counts = _split_forward_full_blocks.__wrapped__(counts, indices, prefix_length=4096)
    assert torch.equal(full_counts[:, :, 4], torch.zeros_like(full_counts[:, :, 4]))


@pytest.mark.parametrize("mutation", ["boundary", "document", "order"])
def test_independent_oracle_detects_deliberately_wrong_full_blocks(mutation: str) -> None:
    counts, indices = _route_block_lists(8192)
    prefix = 4224 if mutation == "boundary" else 4104
    partial_counts, partial_indices, full_counts = _split_forward_full_blocks.__wrapped__(
        counts, indices, prefix_length=prefix
    )
    if mutation == "order":
        partial_indices = partial_indices.clone()
        n = int(partial_counts[0, 0, 40])
        partial_indices[0, 0, 40, :n] = partial_indices[0, 0, 40, :n].flip(0)
    with pytest.raises(AssertionError):
        _assert_pair_oracle(
            counts,
            indices,
            partial_counts,
            partial_indices,
            full_counts,
            length=8192,
            document_reset=128 if mutation == "document" else None,
        )


def test_integer_split_has_one_full_graph() -> None:
    counts, indices = _route_block_lists(8192)
    expected = _split_forward_full_blocks.__wrapped__(counts, indices, prefix_length=4104)
    compiled = torch.compile(_split_forward_full_blocks.__wrapped__, backend="eager", fullgraph=True)
    actual = compiled(counts, indices, prefix_length=4104)
    for left, right in zip(actual, expected, strict=True):
        assert torch.equal(left, right)


@pytest.mark.parametrize("topk,length", [(512, 16), (2, 32)])
def test_indexer_carries_shared_proof_without_changing_routes(topk: int, length: int) -> None:
    torch.manual_seed(533)
    config = tiny_config(
        sparse_config=dict(
            proxy_dim=16,
            sparse_indexer_rope_dim=4,
            sparse_indexer_num_heads=4,
            num_provider_groups=2,
            topk=topk,
            region_block_size=8,
        )
    )
    indexer = Step5SparseIndexer(config, 3, dtype=torch.float32)
    positions = torch.arange(length)[None]
    padding = torch.zeros_like(positions, dtype=torch.bool)
    hidden = torch.randn(1, length, config.hidden_size)
    with torch.no_grad():
        keys, logits = indexer.project_keys(hidden, positions)
        selections = []
        for enabled in (False, True):
            meta = _metadata(positions, padding, enabled=enabled, topk=topk)
            selections.append(indexer(hidden, positions, keys, logits, positions, padding, metadata=meta))
    assert selections[0].forward_full_block_prefix == 0
    assert selections[1].forward_full_block_prefix == min(length, (topk + 1) * 8)
    assert torch.equal(selections[0].region_ids, selections[1].region_ids)


@pytest.mark.parametrize("value", [1, "true", None])
def test_backend_requires_strict_bool(value: int | str | None) -> None:
    with pytest.raises(TypeError, match="csa_forward_full_blocks"):
        Step5BackendConfig(attn="fa4", csa_forward_full_blocks=value)


def test_backend_requires_fa4_and_preserves_default_model_behavior() -> None:
    with pytest.raises(ValueError, match="requires attn=fa4"):
        Step5BackendConfig(attn="sdpa", csa_forward_full_blocks=True)
    for backend in (
        BackendConfig(attn="sdpa", linear="torch", dispatcher="torch", experts="torch_mm"),
        Step5BackendConfig(attn="fa4", linear="torch", dispatcher="torch", experts="torch_mm"),
        Step5BackendConfig(
            attn="fa4", linear="torch", dispatcher="torch", experts="torch_mm", csa_forward_full_blocks=True
        ),
    ):
        model = Step5ForCausalLM(tiny_config(), backend=backend)
        expected = isinstance(backend, Step5BackendConfig) and backend.csa_forward_full_blocks
        assert model.model.csa_forward_full_blocks == expected


def test_empty_and_short_counts_do_not_read_outside_allocated_indices() -> None:
    counts, indices = _route_block_lists(4096)
    counts[:, :, 8] = 0
    indices[:, :, 8] = indices.shape[-1]
    counts[:, :, 16] = 1
    indices[:, :, 16, 1:] = indices.shape[-1]
    partial_counts, partial_indices, full_counts = _split_forward_full_blocks.__wrapped__(
        counts, indices, prefix_length=4096
    )
    assert torch.equal(full_counts[:, :, (8, 16)], torch.zeros_like(full_counts[:, :, (8, 16)]))
    assert torch.equal(partial_counts[:, :, (8, 16)], counts[:, :, (8, 16)])
    _assert_pair_oracle(counts, indices, partial_counts, partial_indices, full_counts, length=4096)


def _lists(length: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return independent legal recency-route reverse lists [2,1,Ktiles,Qtiles]."""
    nq, nk = (length + 127) // 128, (length + 95) // 96
    counts = torch.zeros((2, 1, nk), dtype=torch.int32)
    ids = torch.full((2, 1, nk, nq), nq, dtype=torch.int32)
    for kblock in range(nk):
        chosen = []
        for qblock in range(nq):
            first_q, last_q = qblock * 128, min(length, (qblock + 1) * 128) - 1
            first_k, last_k = kblock * 96, min(length, (kblock + 1) * 96) - 1
            if first_k <= last_q and last_k >= max(0, first_q // 8 - 512) * 8:
                chosen.append(qblock)
        counts[:, :, kblock] = len(chosen)
        ids[:, :, kblock, : len(chosen)] = torch.tensor(chosen, dtype=torch.int32)
    return counts, ids


def _oracle_backward_full_blocks(
    counts: torch.Tensor,
    indices: torch.Tensor,
    split: _BackwardBlockPartition,
    *,
    length: int,
    document_reset: int | None = None,
) -> None:
    """Check partition union/order and all token pairs of every full rectangle.

    Args:
        counts: Original int32 [providers, 1, key_tiles] counts.
        indices: Original int32 [providers, 1, key_tiles, query_tiles] IDs.
        split: Partial/full counts and IDs in the same respective layouts.
        length: Unpadded document token count.
        document_reset: Independent hypothetical packed boundary for negative control.
    """
    for provider in range(counts.shape[0]):
        for kblock in range(counts.shape[-1]):
            old = indices[provider, 0, kblock, : int(counts[provider, 0, kblock])].tolist()
            partial = split.partial_indices[
                provider, 0, kblock, : int(split.partial_counts[provider, 0, kblock])
            ].tolist()
            full = split.full_indices[provider, 0, kblock, : int(split.full_counts[provider, 0, kblock])].tolist()
            assert old == sorted(set(old)) and all(v >= 0 for v in old)
            assert sorted(partial + full) == old
            assert partial == [v for v in old if v not in full]
            assert full == sorted(set(full))
            k = torch.arange(kblock * 96, min((kblock + 1) * 96, length))[None]
            for qblock in full:
                q = torch.arange(qblock * 128, min((qblock + 1) * 128, length))[:, None]
                allowed = (k <= q) & (k // 8 >= (q // 8 - 512).clamp_min(0))
                if document_reset is not None:
                    allowed &= (q >= document_reset) == (k >= document_reset)
                assert allowed.all(), (provider, kblock, qblock)


@pytest.mark.parametrize("length", [256, 4096, 4104, 4224, 8192])
def test_reverse_partition_has_exact_union_and_valid_full_pairs(length: int) -> None:
    counts, indices = _lists(length)
    old_counts, old_ids = counts.clone(), indices.clone()
    result = _split_backward_full_blocks.__wrapped__(counts, indices, prefix_length=min(length, 4104))
    assert result.full_counts.sum() > 0
    _oracle_backward_full_blocks(counts, indices, result, length=length)
    assert torch.equal(counts, old_counts) and torch.equal(indices, old_ids)
    if length == 8192:
        assert int(result.full_counts[0, 0].sum()) == 651
        assert torch.all(result.full_indices[result.full_counts[..., None] > torch.arange(indices.shape[-1])] < 32)


@pytest.mark.parametrize("mutation", ["boundary", "document", "partial_order", "full_order"])
def test_wrong_boundary_document_or_order_is_rejected(mutation: str) -> None:
    counts, ids = _lists(8192)
    result = _split_backward_full_blocks.__wrapped__(
        counts, ids, prefix_length=4224 if mutation == "boundary" else 4104
    )
    if mutation == "partial_order":
        result.partial_indices[0, 0, 0, :2] = result.partial_indices[0, 0, 0, :2].flip(0)
    if mutation == "full_order":
        result.full_indices[0, 0, 0, :2] = result.full_indices[0, 0, 0, :2].flip(0)
    with pytest.raises(AssertionError):
        _oracle_backward_full_blocks(
            counts, ids, result, length=8192, document_reset=128 if mutation == "document" else None
        )


def test_holes_empty_and_short_rows_preserve_all_entries() -> None:
    counts, ids = _lists(8192)
    counts[:, :, 0] = 0
    counts[:, :, 1] = 1
    row = ids[0, 0, 2, : int(counts[0, 0, 2])].tolist()
    row.remove(5)
    counts[0, 0, 2] = len(row)
    ids[0, 0, 2, : len(row)] = torch.tensor(row, dtype=torch.int32)
    result = _split_backward_full_blocks.__wrapped__(counts, ids, prefix_length=4104)
    _oracle_backward_full_blocks(counts, ids, result, length=8192)
    assert result.full_counts[0, 0, 0] == 0


def test_full_graph_has_no_host_tensor_dependency() -> None:
    counts, ids = _lists(512)
    expected = _split_backward_full_blocks.__wrapped__(counts, ids, prefix_length=512)
    compiled = torch.compile(_split_backward_full_blocks.__wrapped__, backend="eager", fullgraph=True)
    actual = compiled(counts, ids, prefix_length=512)
    for a, b in zip(actual, expected, strict=True):
        assert torch.equal(a, b)


@pytest.mark.parametrize(
    "case", ["single", "packed", "batch2", "cp_first", "cp_second", "pad", "internal_pad", "repeat", "missing"]
)
def test_backward_proof_is_independent_and_conservative(case: str) -> None:
    positions = torch.arange(512)[None]
    padding = torch.zeros_like(positions, dtype=torch.bool)
    start, queries = 0, 512
    if case == "packed":
        positions[:, 256:] -= 256
    elif case == "batch2":
        positions, padding = positions.expand(2, -1), padding.expand(2, -1)
    elif case == "cp_first":
        queries = 256
    elif case == "cp_second":
        start, queries = 256, 256
    elif case in ("pad", "internal_pad"):
        pos = 500 if case == "pad" else 111
        padding[:, pos:] = case == "pad"
        padding[:, pos] = True
        positions[padding] = 0
    elif case == "repeat":
        positions[:, 111] = 110
    elif case == "missing":
        positions[:, 111:] += 1
    old = region_metadata(positions, padding, query_start=start, queries=queries, region_size=8, topk=32)
    new = region_metadata(
        positions, padding, query_start=start, queries=queries, region_size=8, topk=32, csa_backward_full_blocks=True
    )
    assert new.backward_full_block_prefix == (264 if case == "single" else 0)
    assert new.forward_full_block_prefix == 0
    assert new.canonical_region_size == (8 if case == "single" else 0)
    _assert_existing_fields_equal(old, replace(new, backward_full_block_prefix=0, canonical_region_size=0))


@pytest.mark.parametrize("value", [1, "true", None])
def test_flag_is_strict_bool(value: int | str | None) -> None:
    with pytest.raises(TypeError, match="csa_backward_full_blocks"):
        Step5BackendConfig(attn="fa4", csa_backward_full_blocks=value)


def test_wrong_backend_is_rejected_and_default_stays_off() -> None:
    assert not Step5BackendConfig(attn="fa4").csa_backward_full_blocks
    with pytest.raises(ValueError, match="requires attn=fa4"):
        Step5BackendConfig(attn="sdpa", csa_backward_full_blocks=True)


@pytest.fixture
def scalar_cute(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Interpret CuTe scalar operations on CPU; this does not validate CUDA compilation."""

    def identity(value: object) -> object:
        return value

    def to_bool(value: object, dtype: type) -> bool:
        return bool(value)

    modules = {
        "flash_attn.cute.interface": SimpleNamespace(flash_attn_func=identity),
        "flash_attn.cute.block_sparsity": SimpleNamespace(BlockSparseTensorsTorch=object),
        "cutlass": SimpleNamespace(
            const_expr=identity, Int32=int, Boolean=bool, min=min, max=max, range_constexpr=range
        ),
        "cutlass.cute": SimpleNamespace(jit=identity),
        "flash_attn.cute.utils": SimpleNamespace(scalar_to_ssa=to_bool),
    }

    def safe_import(name: str) -> tuple[bool, object]:
        return True, modules[name]

    fa4._load_fa4.cache_clear()
    monkeypatch.setattr(fa4, "safe_import", safe_import)
    yield
    fa4._load_fa4.cache_clear()


@pytest.mark.parametrize("region_size", [3, 8])
@pytest.mark.parametrize("bitmap", [False, True])
@pytest.mark.parametrize("head_layout", [False, True])
def test_actual_mask_matches_independent_membership_and_bounds(
    scalar_cute: None, region_size: int, bitmap: bool, head_layout: bool
) -> None:
    length, groups, slots = region_size * 7, 2, 3
    positions = torch.arange(length)[None]
    documents = torch.ones_like(positions)
    padding = torch.zeros_like(positions, dtype=torch.bool)
    token_regions = positions // region_size
    ids = torch.full((1, groups, length, slots), 7, dtype=torch.int32)
    routes: list[list[set[int]]] = [[], []]
    for group in range(groups):
        for query in range(length):
            current = query // region_size
            history = list(range(current))
            selected = history[:2] if group == 0 else history[-2:]
            selected = sorted(set(selected + [current]))
            routes[group].append(set(selected))
            ids[0, group, query, : len(selected)] = torch.tensor(selected)
    membership = ids
    if bitmap:
        membership = torch.zeros((1, groups, length, 1), dtype=torch.int32)
        for group in range(groups):
            for query in range(length):
                membership[0, group, query, 0] = sum(1 << v for v in routes[group][query])
    aux = [positions, positions, documents, documents, padding, membership, token_regions]
    seq = SimpleNamespace(seqlen_q=length, seqlen_k=length)
    masks = [
        fa4._load_fa4(
            None, bitmap, slots, groups, heads_per_group=16 if head_layout else 0, canonical_region_size=width
        )[2]
        for width in (0, region_size)
    ]
    for group in range(groups):
        batch, head = ([0], [group * 16 + 15]) if head_layout else ([group], [15])
        for query in range(length + 2):
            for key in range(length + 2):
                expected = (
                    query < length
                    and key < length
                    and key <= query
                    and key // region_size in routes[group][min(query, length - 1)]
                )
                assert all(mask(batch, head, [query], [key], seq, aux) == expected for mask in masks)


@pytest.mark.parametrize("defect", ["region_size", "document", "padding"])
def test_scalar_oracle_rejects_wrong_specialization(scalar_cute: None, defect: str) -> None:
    length = 24
    positions = torch.arange(length)[None]
    docs = torch.ones_like(positions)
    padding = torch.zeros_like(positions, dtype=torch.bool)
    ids = torch.tensor([0, 2, 3], dtype=torch.int32)[None, None, None].expand(1, 1, length, 3)
    if defect == "document":
        docs[:, 8:] += 1
    if defect == "padding":
        padding[:, 0] = True
    aux = [positions, positions, docs, docs, padding, ids, positions // 8]
    seq = SimpleNamespace(seqlen_q=length, seqlen_k=length)
    mask = fa4._load_fa4(None, False, 3, 1, canonical_region_size=3 if defect == "region_size" else 8)[2]

    def assert_correct() -> None:
        for query in range(length):
            for key in range(length):
                expected = (
                    key <= query and not padding[0, key] and docs[0, query] == docs[0, key] and key // 8 in (0, 2, 3)
                )
                assert mask([0], [0], [query], [key], seq, aux) == bool(expected)

    with pytest.raises(AssertionError):
        assert_correct()


@pytest.mark.parametrize("region_size", [3, 8])
def test_proof_uses_actual_region_width_and_remains_opt_in(region_size: int) -> None:
    positions = torch.arange(79)[None]
    padding = torch.zeros_like(positions, dtype=torch.bool)
    kwargs = dict(query_start=0, queries=79, region_size=region_size, topk=2)
    assert region_metadata(positions, padding, **kwargs).canonical_region_size == 0
    assert (
        region_metadata(positions, padding, csa_forward_full_blocks=True, **kwargs).canonical_region_size == region_size
    )
    assert (
        region_metadata(positions, padding, csa_backward_full_blocks=True, **kwargs).canonical_region_size
        == region_size
    )


@pytest.mark.parametrize("batch,groups,heads_per_group", [(1, 4, 16), (2, 3, 2), (2, 1, 1)])
@pytest.mark.parametrize("indices", [False, True])
def test_provider_heads_preserve_distinct_batch_and_group_routes(
    batch: int, groups: int, heads_per_group: int, indices: bool
) -> None:
    shape = (batch * groups, 1, 5, 7) if indices else (batch * groups, 1, 5)
    backing = torch.arange(2 * math.prod(shape), dtype=torch.int32).view(*shape, 2)
    tensor = backing[..., 1]
    original = tensor.clone()
    result = _provider_heads(tensor, batch=batch, heads_per_group=heads_per_group)
    assert result.shape == (batch, groups * heads_per_group, *shape[2:])
    assert result.dtype == torch.int32 and result.is_contiguous()
    for b in range(batch):
        for h in range(groups * heads_per_group):
            assert torch.equal(result[b, h], original[b * groups + h // heads_per_group, 0])
    assert torch.equal(tensor, original)
    result.fill_(-1)
    assert torch.equal(tensor, original), "Converted metadata must not alias its provider source"


def _positions(lengths: list[int], size: int) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.cat([torch.arange(n) for n in lengths])
    padding = torch.arange(size) >= len(values)
    return torch.cat((values, torch.zeros(size - len(values), dtype=torch.long))), padding


def _legacy_summary(
    keys: torch.Tensor, logits: torch.Tensor, positions: torch.Tensor, padding: torch.Tensor, meta: RegionMetadata
) -> torch.Tensor:
    """Preserve the unoptimized source's scatter and arithmetic as a control."""
    batch, _, dim = keys.shape
    rows = torch.arange(batch, device=keys.device)[:, None]
    packed = keys.new_zeros(batch, meta.region_count, 8, dim)
    z = torch.full_like(packed, -torch.inf)
    packed[rows, meta.token_regions, positions % 8] = keys
    z[rows, meta.token_regions, positions % 8] = logits.masked_fill(padding[..., None], -torch.inf)
    shift = z.float().amax((-1, -2), keepdim=True)
    probabilities = (z.float() - shift).exp().nan_to_num(0)
    result = (probabilities * packed.float()).sum(2) / probabilities.sum(2).clamp_min(1e-20)
    return result.to(torch.bfloat16).to(torch.float8_e4m3fn).to(torch.bfloat16)


@pytest.mark.parametrize(
    "rows,size",
    [
        ([[257, *([1] * 260), 257]], 790),
        ([[43, 3], [1, 53]], 80),
        ([[79], [1, 2, 3]], 80),
        ([[3, 61, 9], [69, 1]], 80),
    ],
)
@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"))],
)
@pytest.mark.runtime_budget(15, reason="Cold CUDA integer gather compilation for packed region shapes")
def test_region_payload_matches_original_on_all_required_intervals(
    rows: list[list[int]], size: int, device: str
) -> None:
    pairs = [_positions(lengths, size) for lengths in rows]
    positions = torch.stack([p[0] for p in pairs]).to(device)
    padding = torch.stack([p[1] for p in pairs]).to(device)
    meta = region_metadata(positions, padding, query_start=0, queries=size, region_size=8, topk=4)
    assert meta.summary_ranges and meta.region_sources is not None and meta.region_source_valid is not None
    assert meta.region_sources.dtype == torch.int32
    torch.manual_seed(129)
    keys = torch.randn(len(rows), size, 16, dtype=torch.bfloat16, device=device)
    logits = torch.randn_like(keys) * 3
    control = _legacy_summary(keys, logits, positions, padding, meta)
    for b, first, stop in meta.summary_ranges:
        actual = _summarize_region_payload(
            *gather_region_payload(
                keys[b : b + 1],
                logits[b : b + 1],
                meta.region_sources[b : b + 1, first:stop],
                meta.region_source_valid[b : b + 1, first:stop],
            )
        )
        torch.testing.assert_close(actual, control[b : b + 1, first:stop], rtol=0, atol=0)
    if size == 790:
        assert len(meta.summary_ranges) == 2  # The intervening 260 short docs need no summaries.
        assert meta.summary_ranges[0][2] < meta.summary_ranges[1][1]


def test_summary_range_merge_and_score_tile_mapping() -> None:
    tiles = [(1, 256, 384, 20, 30), (0, 0, 128, 0, 11), (0, 128, 256, 7, 14), (1, 0, 128, 2, 6)]
    ranges, indices = _summary_ranges(tiles)
    assert ranges == ((0, 0, 14), (1, 2, 6), (1, 20, 30))
    assert indices == (2, 0, 0, 1)


@pytest.mark.parametrize(
    "irregular", ["internal_padding", "duplicate_position", "negative_position", "nonzero_padding"]
)
def test_irregular_layout_keeps_original_scatter(irregular: str) -> None:
    positions, padding = _positions([79], 80)
    if irregular == "internal_padding":
        padding[7] = True
        positions[7] = 0
    elif irregular == "duplicate_position":
        positions[7] = 6
    elif irregular == "negative_position":
        positions[7] = -1
    else:
        positions[-1] = 13
    meta = region_metadata(positions[None], padding[None], query_start=0, queries=80, region_size=8, topk=4)
    assert meta.score_tiles
    assert meta.region_sources is None and meta.region_source_valid is None
    assert meta.summary_ranges == meta.score_range_indices == ()


def test_prefix_and_asymmetric_cp_ranks_do_not_allocate_source_maps() -> None:
    positions = torch.arange(8192)[None]
    padding = torch.zeros_like(positions, dtype=torch.bool)
    first = region_metadata(positions, padding, query_start=0, queries=4096, region_size=8, topk=512)
    last = region_metadata(positions, padding, query_start=4096, queries=4096, region_size=8, topk=512)
    assert first.needs_projection and last.needs_projection
    assert first.region_sources is None and first.summary_ranges == ()
    assert last.region_sources is not None and last.summary_ranges == ((0, 0, 1023),)


def test_final_short_document_collision_cannot_change_legal_history() -> None:
    positions, padding = _positions([43, 3], 80)
    meta = region_metadata(positions[None], padding[None], query_start=0, queries=80, region_size=8, topk=4)
    colliding = int(meta.token_regions[0, 43])
    assert bool((meta.token_regions[0, 46:] == colliding).all())
    candidates = (
        (torch.arange(meta.region_count)[None] < meta.current[0, :, None])
        & (meta.region_documents[0, None] == meta.query_documents[0, :, None])
        & meta.sparse_queries[0, :, None]
    )
    assert not candidates[:, colliding].any()
    assert all(not (b == 0 and first <= colliding < stop) for b, first, stop in meta.summary_ranges)


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"))],
)
def test_double_rounding_analytical_negative_control(device: str) -> None:
    keys = torch.zeros(1, 1, 8, 16, dtype=torch.bfloat16, device=device)
    logits = torch.full_like(keys, -torch.inf)
    keys[:, :, 0] = 1.0
    keys[:, :, 1] = 1.125
    logits[:, :, 0] = 0
    logits[:, :, 1] = 0.0078125
    actual = _summarize_region_payload(keys, logits)
    assert torch.equal(actual, torch.ones_like(actual))
    p = (logits.float() - logits.float().amax((-1, -2), keepdim=True)).exp()
    raw = (p * keys.float()).sum(2) / p.sum(2)
    wrong = raw.to(torch.float8_e4m3fn).to(torch.bfloat16)
    assert torch.equal(wrong, torch.full_like(wrong, 1.125))


def test_wrong_source_slot_negative_control() -> None:
    positions, padding = _positions([79], 80)
    meta = region_metadata(positions[None], padding[None], query_start=0, queries=80, region_size=8, topk=4)
    assert meta.region_sources is not None and meta.region_source_valid is not None
    keys = torch.arange(80).reshape(1, 80, 1).expand(1, 80, 16).to(torch.bfloat16)
    logits = torch.zeros_like(keys)
    sources = meta.region_sources[:, :5]
    valid = meta.region_source_valid[:, :5]
    correct = _summarize_region_payload(*gather_region_payload(keys, logits, sources, valid))
    wrong_sources = sources.clone()
    wrong_sources[:, 0] += 8
    wrong = _summarize_region_payload(*gather_region_payload(keys, logits, wrong_sources, valid))
    assert not torch.equal(correct, wrong)
    assert torch.equal(correct[:, 1:], wrong[:, 1:])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.runtime_budget(20, reason="Cold CUDA indexer and gather compilation with complete 4K CP score comparison")
def test_real_indexer_dimensions_full_scores_and_sets_match_original(
    batch: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = tiny_config(
        num_attention_groups=4,
        sparse_config=dict(
            proxy_dim=256,
            sparse_indexer_rope_dim=32,
            sparse_indexer_num_heads=16,
            num_provider_groups=4,
            topk=512,
            region_block_size=8,
        ),
    )
    torch.manual_seed(450)
    module = Step5SparseIndexer(cfg, 3, dtype=torch.bfloat16).cuda()
    hidden = torch.randn(batch, 8192, 32, dtype=torch.bfloat16, device="cuda")
    positions = torch.arange(8192, device="cuda")[None].expand(batch, -1).clone()
    padding = torch.zeros_like(positions, dtype=torch.bool)
    if batch == 2:
        positions[1] = 0
        padding[1] = True
    keys, logits = module.project_keys(hidden, positions)
    meta = region_metadata(positions, padding, query_start=4096, queries=4096, region_size=8, topk=512)
    control_meta = replace(
        meta, summary_ranges=(), score_range_indices=(), region_sources=None, region_source_valid=None
    )
    scores: list[torch.Tensor] = []
    score_fn = sparse.indexer_score_tile

    def capture(q: torch.Tensor, w: torch.Tensor, k: torch.Tensor, start: int, stop: int) -> torch.Tensor:
        result = score_fn(q, w, k, start, stop)
        scores.append(result.clone())
        return result

    monkeypatch.setattr(sparse, "indexer_score_tile", capture)
    original = module(
        hidden[:, 4096:], positions[:, 4096:], keys, logits, positions, padding, query_start=4096, metadata=control_meta
    )
    controls = scores.copy()
    scores.clear()
    actual = module(
        hidden[:, 4096:], positions[:, 4096:], keys, logits, positions, padding, query_start=4096, metadata=meta
    )
    assert len(controls) == len(scores) > 0
    for expected, observed in zip(controls, scores):
        torch.testing.assert_close(observed, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual.region_ids, original.region_ids, rtol=0, atol=0)
    assert meta.sparse_queries[0, 8:].all()  # Document positions 4104 through8191 really drop history.


def _native(layout: CausalDocumentLayout, queries: int, keys: int, window: int) -> torch.Tensor:
    result = torch.zeros(queries, keys, dtype=torch.bool)
    qb, kb = layout.cu_seqlens_q.tolist(), layout.cu_seqlens_k.tolist()
    for q0, q1, k0, k1 in zip(qb, qb[1:], kb, kb[1:]):
        distance = torch.arange(q1 - q0)[:, None] + k1 - k0 - q1 + q0 - torch.arange(k1 - k0)[None]
        result[q0:q1, layout.key_start + k0 : layout.key_start + k1] = (distance >= 0) & (distance < window)
    return result


def _oracle_window_documents(positions: torch.Tensor, start: int, queries: int, window: int) -> torch.Tensor:
    documents = (positions == 0).long().cumsum(-1)
    distance = positions[0, start : start + queries, None] - positions[0, None]
    return (documents[0, start : start + queries, None] == documents[0, None]) & (distance >= 0) & (distance < window)


@pytest.mark.parametrize(
    "lengths,start,queries,window",
    [
        ((11,), 0, 11, 4),
        ((11,), 3, 4, 4),
        ((11,), 7, 4, 4),
        ((9, 3, 7), 5, 10, 4),
        ((9, 3, 7), 9, 3, 4),
        ((1, 1, 2), 1, 2, 1),
        ((4096,), 0, 4096, 512),
        ((8192,), 4096, 4096, 512),
        ((4097, 4095), 4096, 4096, 512),
        ((2053, 3000, 3139), 4096, 4096, 512),
    ],
)
def test_window_layout_matches_independent_documents_and_causality(
    lengths: tuple[int, ...], start: int, queries: int, window: int
) -> None:
    positions = torch.cat([torch.arange(n) for n in lengths])[None]
    metadata = region_metadata(
        positions,
        torch.zeros_like(positions, dtype=torch.bool),
        query_start=start,
        queries=queries,
        region_size=8,
        topk=512,
        window=window,
    )
    layout = metadata.window_layout
    assert layout is not None
    assert layout.valid_queries == queries and layout.padding_key is None
    assert layout.key_start + layout.leading_padding >= max(0, start - window + 1)
    assert layout.key_stop == start + queries
    torch.testing.assert_close(
        _native(layout, queries, positions.shape[-1], window),
        _oracle_window_documents(positions, start, queries, window),
    )
    # Existing halo alignment padding changes only local storage coordinates.
    first_key = max(0, start - window + 1)
    left_padding = max(first_key % 112, layout.leading_padding)
    halo_query_start = start - first_key + left_padding
    offset = layout.key_stop - queries - halo_query_start
    assert layout.key_start - offset >= 0
    assert layout.key_stop - offset == halo_query_start + queries


@pytest.mark.parametrize("padded", [0, 6, 10])
def test_window_layout_padding_falls_back_only_when_required(padded: int) -> None:
    positions = torch.arange(12)[None]
    padding = torch.zeros_like(positions, dtype=torch.bool)
    padding[0, padded] = True
    metadata = region_metadata(positions, padding, query_start=7, queries=3, region_size=2, topk=2, window=3)
    assert (metadata.window_layout is None) == (padded == 6)


@pytest.mark.parametrize(
    "positions,padding,start",
    [
        ([0, 1, 2, 0], [False, False, False, True], 3),
        ([0, 1, 3, 4], [False] * 4, 0),
        ([0, 1, 1, 2], [False] * 4, 0),
        ([0, 0, 0, 0], [True] * 4, 0),
    ],
)
def test_window_layout_conservatively_rejects_padding_and_position_holes(
    positions: list[int], padding: list[bool], start: int
) -> None:
    metadata = region_metadata(
        torch.tensor([positions]),
        torch.tensor([padding]),
        query_start=start,
        queries=len(positions) - start,
        region_size=2,
        topk=2,
        window=3,
    )
    assert metadata.window_layout is None


def test_window_layout_is_independent_of_csa_sparsity_and_has_explicit_batch_guard() -> None:
    positions = torch.arange(12)[None]
    arguments = dict(query_start=5, queries=7, region_size=2, topk=2, window=3)
    metadata = region_metadata(positions, torch.zeros_like(positions, dtype=torch.bool), **arguments)
    assert metadata.score_tiles and metadata.causal_layout is None and metadata.window_layout is not None
    metadata = region_metadata(positions.expand(2, -1), torch.zeros(2, 12, dtype=torch.bool), **arguments)
    assert metadata.window_layout is None


def test_window_oracle_detects_wrong_alignment_and_unbounded_window() -> None:
    positions = torch.arange(15)[None]
    metadata = region_metadata(
        positions,
        torch.zeros_like(positions, dtype=torch.bool),
        query_start=5,
        queries=7,
        region_size=2,
        topk=2,
        window=3,
    )
    layout = metadata.window_layout
    expected = _oracle_window_documents(positions, 5, 7, 3)
    bad = replace(layout, key_stop=15, cu_seqlens_k=torch.tensor([0, 15 - layout.key_start], dtype=torch.int32))
    assert not torch.equal(_native(bad, 7, 15, 3), expected)
    assert not torch.equal(_native(layout, 7, 15, 99), expected)


def test_window_backend_opt_in_defaults_and_type() -> None:
    assert not Step5BackendConfig().swa_varlen
    assert Step5BackendConfig(swa_varlen=True).swa_varlen
    with pytest.raises(TypeError, match="swa_varlen"):
        Step5BackendConfig(swa_varlen=1)


@pytest.mark.parametrize("start,queries", [(511, 37), (512, 37), (2048, 37), (4096, 37)])
def test_window_dummy_alignment_proof_without_extra_transport(start: int, queries: int) -> None:
    """All extra keys needed to align native K tiles are outside every Q window."""
    positions = torch.cat((torch.arange(1031), torch.arange(7161)))[None]
    metadata = region_metadata(
        positions,
        torch.zeros_like(positions, dtype=torch.bool),
        query_start=start,
        queries=queries,
        region_size=8,
        topk=512,
        window=512,
    )
    layout = metadata.window_layout
    assert layout is not None
    document_start = start - int(positions[0, start])
    dummy = max(0, int(positions[0, start]) - 511) % 96
    assert layout.leading_padding == dummy
    assert (layout.key_start - document_start) % 96 == 0
    native = _native(layout, queries, positions.shape[1], 512)
    torch.testing.assert_close(native, _oracle_window_documents(positions, start, queries, 512))
    assert not native[:, layout.key_start : layout.key_start + dummy].any()
    # The existing halo construction already pads and concatenates once. A
    # larger zero prefix changes storage only, never the communicated payload.
    first_key = max(0, start - 511)
    local_padding = max(first_key % 112, dummy)
    assert local_padding <= 111
    local_query_start = start - first_key + local_padding
    physical_offset = layout.key_stop - queries - local_query_start
    assert layout.key_start - physical_offset >= 0
    assert layout.key_stop - physical_offset == local_query_start + queries
    if dummy:
        wrong_cu = layout.cu_seqlens_k.clone()
        wrong_cu[1:] -= dummy
        wrong = replace(layout, cu_seqlens_k=wrong_cu)
        assert not torch.equal(_native(wrong, queries, positions.shape[1], 512), native)


@pytest.mark.parametrize("start", [0, 4096])
def test_window_trailing_padding_preserves_valid_native_prefix(start: int) -> None:
    positions = torch.cat(
        (torch.arange(2051), torch.arange(4096), torch.arange(2038), torch.zeros(7, dtype=torch.long))
    )[None]
    padding = torch.arange(8192)[None] >= 8185
    metadata = region_metadata(
        positions, padding, query_start=start, queries=8192 - start, region_size=8, topk=512, window=512
    )
    layout = metadata.window_layout
    assert layout is not None and layout.valid_queries == 8185 - start
    assert layout.key_stop == 8185 and layout.padding_key == 6147
    valid_mask = _native(layout, layout.valid_queries, 8192, 512)
    independent = _oracle_window_documents(positions[:, :8185], start, layout.valid_queries, 512)
    torch.testing.assert_close(valid_mask[:, :8185], independent)
    assert not valid_mask[:, 8185:].any()


def _capture_integer_work(monkeypatch: pytest.MonkeyPatch) -> tuple[list[torch.dtype], list[torch.dtype]]:
    """Observe real value-topk and concatenation dtypes without replacing their math."""
    topk_dtypes: list[torch.dtype] = []
    concat_dtypes: list[torch.dtype] = []
    original_topk = torch.Tensor.topk
    original_cat = torch.cat

    def topk(
        tensor: torch.Tensor, k: int, dim: int = -1, largest: bool = True, sorted: bool = True
    ) -> torch.return_types.topk:
        """Run the original top-k and record integer inputs.

        Args:
            tensor: Values of shape [..., regions] with arbitrary leading dimensions.
            k: Number of retained values.
            dim: Selected tensor axis.
            largest: Select greatest values when true.
            sorted: Sort the returned values when true.

        Returns:
            Original values and indices with the selected axis shortened to k.
        """
        if not tensor.is_floating_point():
            topk_dtypes.append(tensor.dtype)
        return original_topk(tensor, k, dim=dim, largest=largest, sorted=sorted)

    def cat(tensors: Sequence[torch.Tensor], dim: int = 0) -> torch.Tensor:
        """Run the original concatenation and record integer result dtypes.

        Args:
            tensors: Tensors with equal shape except along the concatenated axis.
            dim: Concatenated axis.

        Returns:
            Original concatenation, allocating independent output storage.
        """
        result = original_cat(tensors, dim=dim)
        if all(not tensor.is_floating_point() for tensor in tensors):
            concat_dtypes.append(result.dtype)
        return result

    monkeypatch.setattr(torch.Tensor, "topk", topk)
    monkeypatch.setattr(torch, "cat", cat)
    return topk_dtypes, concat_dtypes


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("offset,length", [(0, 35), (3, 13), (17, 18)])
def test_indexer_temporary_ids_preserve_independent_selection(
    monkeypatch: pytest.MonkeyPatch, tied: bool, offset: int, length: int
) -> None:
    module = _indexer()
    if tied:
        with torch.no_grad():
            module.w_proj.weight.zero_()
    positions = torch.cat([torch.arange(n) for n in (3, 11, 1, 20)])[None]
    padding = torch.zeros_like(positions, dtype=torch.bool)
    hidden = torch.randn(1, 35, 32)
    keys, logits = module.project_keys(hidden, positions)
    local, local_positions = hidden[:, offset : offset + length], positions[:, offset : offset + length]
    meta = region_metadata(positions, padding, query_start=offset, queries=length, region_size=2, topk=2)
    expected = _oracle(module, local, local_positions, keys, logits, positions, padding, offset=offset)
    original_current = meta.current.clone()
    topk_dtypes, concat_dtypes = _capture_integer_work(monkeypatch)
    actual = module(local, local_positions, keys, logits, positions, padding, query_start=offset, metadata=meta)
    torch.testing.assert_close(actual.region_ids, expected, rtol=0, atol=0)
    assert topk_dtypes and set(topk_dtypes) == {torch.int32}
    assert concat_dtypes and set(concat_dtypes) == {torch.int32}
    assert actual.region_ids.dtype == torch.int32 and actual.region_ids.shape[-1] == 3
    assert meta.current.dtype == torch.int64
    torch.testing.assert_close(meta.current, original_current, rtol=0, atol=0)
    assert bool(meta.sparse_queries.any())


@pytest.mark.parametrize("sentinel", [2**31 - 2, 2**31 - 1, 2**31, 2**31 + 1])
def test_indexer_temporary_sentinel_width_boundary(monkeypatch: pytest.MonkeyPatch, sentinel: int) -> None:
    """Exercise real forward's sentinel branch with small referenced region storage.

    Only the unused ID sentinel is inflated; all score tiles still reference the
    original small region tables. This is a temporary-ID component boundary
    test, not evidence for billion-token region_metadata or model execution.
    The wide template on the fallback case preserves the pre-existing caller's
    storage rather than claiming the normal int32 route can hold this sentinel.
    """
    module = _indexer()
    with torch.no_grad():
        module.w_proj.weight.zero_()
    positions = torch.arange(12)[None]
    padding = torch.zeros_like(positions, dtype=torch.bool)
    hidden = torch.randn(1, 12, 32)
    keys, logits = module.project_keys(hidden, positions)
    meta = region_metadata(positions, padding, query_start=0, queries=12, region_size=2, topk=2)
    assert meta.region_sources is not None
    expected = _oracle(module, hidden, positions, keys, logits, positions, padding, offset=0).long()
    expected.masked_fill_(expected == meta.region_count, sentinel)
    dtype = torch.int32 if sentinel <= 2**31 - 1 else torch.int64
    template = meta.dense_region_ids.long()
    template.masked_fill_(template == meta.region_count, sentinel)
    inflated = replace(meta, region_count=sentinel, dense_region_ids=template.to(dtype))
    topk_dtypes, concat_dtypes = _capture_integer_work(monkeypatch)
    actual = module(hidden, positions, keys, logits, positions, padding, metadata=inflated)
    torch.testing.assert_close(actual.region_ids.long(), expected, rtol=0, atol=0)
    assert topk_dtypes and set(topk_dtypes) == {dtype}
    assert concat_dtypes and set(concat_dtypes) == {dtype}
    assert (actual.region_ids[0, 0, 0] == sentinel).sum() == 2
    assert actual.region_ids[0, 0, -1].tolist() == [0, 1, 5]
