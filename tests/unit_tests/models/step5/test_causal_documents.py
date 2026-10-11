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

"""Independent document masks for native unequal-Q/K causal prefixes."""

from dataclasses import replace

import pytest
import torch

from nemo_automodel.components.models.step5.sparse import CausalDocumentLayout, region_metadata


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


def _oracle(positions: torch.Tensor, start: int, queries: int, padding: torch.Tensor | None = None) -> torch.Tensor:
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
    torch.testing.assert_close(_native_mask(layout, queries, positions.shape[1]), _oracle(positions, start, queries))


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
    expected = _oracle(positions, 2, 4)
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
    expected = _oracle(positions, start, queries, padding)
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
