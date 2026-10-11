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

"""Native SWA document/halo layout against independent positional predicates."""

from dataclasses import replace

import pytest
import torch

from nemo_automodel.components.models.step5.backend import Step5BackendConfig
from nemo_automodel.components.models.step5.sparse import CausalDocumentLayout, region_metadata


def _native(layout: CausalDocumentLayout, queries: int, keys: int, window: int) -> torch.Tensor:
    result = torch.zeros(queries, keys, dtype=torch.bool)
    qb, kb = layout.cu_seqlens_q.tolist(), layout.cu_seqlens_k.tolist()
    for q0, q1, k0, k1 in zip(qb, qb[1:], kb, kb[1:]):
        distance = torch.arange(q1 - q0)[:, None] + k1 - k0 - q1 + q0 - torch.arange(k1 - k0)[None]
        result[q0:q1, layout.key_start + k0 : layout.key_start + k1] = (distance >= 0) & (distance < window)
    return result


def _oracle(positions: torch.Tensor, start: int, queries: int, window: int) -> torch.Tensor:
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
        _native(layout, queries, positions.shape[-1], window), _oracle(positions, start, queries, window)
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
    expected = _oracle(positions, 5, 7, 3)
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
    torch.testing.assert_close(native, _oracle(positions, start, queries, 512))
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
    independent = _oracle(positions[:, :8185], start, layout.valid_queries, 512)
    torch.testing.assert_close(valid_mask[:, :8185], independent)
    assert not valid_mask[:, 8185:].any()
