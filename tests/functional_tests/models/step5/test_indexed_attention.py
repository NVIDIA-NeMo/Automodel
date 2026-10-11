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

"""Actual indexed CUDA attention against independent document/route math."""

import pytest
import torch
from torch.nn import functional as F

from nemo_automodel.components.models.step5.indexed_attention import _inverse_edges, selected_attention
from nemo_automodel.components.models.step5.sparse import IndexedRegionLayout, RegionSelection

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _packed_case(
    batch: int, keys: int, query_start: int, padding_tokens: int, *, groups: int = 2, heads_per_group: int = 2
) -> tuple[torch.Tensor, torch.Tensor, RegionSelection, torch.Tensor]:
    """Enumerate physical regions and allowed edges on CPU, without model helpers."""
    positions = torch.zeros(batch, keys, dtype=torch.int64)
    documents = torch.zeros_like(positions)
    regions = torch.zeros_like(positions)
    padding = torch.zeros_like(positions, dtype=torch.bool)
    starts_per_batch, ends_per_batch = [], []
    for b in range(batch):
        # Different boundaries ensure B2 cannot reuse B1's document metadata.
        boundary = min(137 + 74 * b, (keys - padding_tokens) // 2)
        lengths = (boundary, keys - padding_tokens - boundary)
        cursor, region = 0, -1
        starts, ends = [], []
        previous = None
        for document, length in enumerate(lengths, start=1):
            for position in range(length):
                identity = (document, position // 8)
                if identity != previous:
                    if starts:
                        ends.append(cursor)
                    starts.append(cursor)
                    region += 1
                    previous = identity
                positions[b, cursor] = position
                documents[b, cursor] = document
                regions[b, cursor] = region
                cursor += 1
        # Packed padding resets positions but retains the final document ID.
        # Its final physical region may be much longer than eight tokens.
        ends.append(cursor)
        starts.append(cursor)
        ends.append(keys)
        regions[b, cursor:] = region + 1
        documents[b, cursor:] = len(lengths)
        padding[b, cursor:] = True
        starts_per_batch.append(starts)
        ends_per_batch.append(ends)
    region_count = max(len(starts) for starts in starts_per_batch)
    starts = torch.full((batch, region_count), keys, dtype=torch.int32)
    ends = torch.zeros_like(starts)
    for b in range(batch):
        starts[b, : len(starts_per_batch[b])] = torch.tensor(starts_per_batch[b], dtype=torch.int32)
        ends[b, : len(ends_per_batch[b])] = torch.tensor(ends_per_batch[b], dtype=torch.int32)
    queries, slots = keys - query_start, 6
    ids = torch.full((batch, groups, queries, slots), region_count, dtype=torch.int32)
    allowed = torch.zeros(batch, groups, queries, keys, dtype=torch.bool)
    for b in range(batch):
        for group in range(groups):
            for query in range(queries):
                physical = query_start + query
                if query == min(17, queries - 1):
                    continue  # Entire selected row is sentinel: exact zero output/gradient.
                document = int(documents[b, physical])
                first_key = int((documents[b] == document).nonzero()[0])
                first = int(regions[b, first_key])
                current = int(regions[b, physical])
                chosen = sorted((first, max(first, current - 2 * (group + 1)), current, current))
                ids[b, group, query, :4] = torch.tensor(chosen, dtype=torch.int32)
                # Set membership, including duplicate IDs, is defined independently.
                selected = torch.zeros(keys, dtype=torch.bool)
                for region_id in set(chosen):
                    selected |= regions[b] == region_id
                allowed[b, group, query] = (
                    selected & (documents[b] == document) & (positions[b] <= positions[b, physical]) & ~padding[b]
                )
    selection = RegionSelection(
        ids.cuda(),
        regions.cuda(),
        documents[:, query_start:].cuda(),
        documents.cuda(),
        region_count,
        indexed_layout=IndexedRegionLayout(starts.cuda(), ends.cuda()),
    )
    return positions.cuda(), padding.cuda(), selection, allowed.cuda().repeat_interleave(heads_per_group, dim=1)


def _assert_parity(
    batch: int, keys: int, query_start: int, padding_tokens: int, *, groups: int = 2, heads_per_group: int = 2
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the existing 0.015 forward / 0.012 gradient gates to real kernels."""
    torch.manual_seed(6101)
    positions, padding, selection, allowed = _packed_case(
        batch, keys, query_start, padding_tokens, groups=groups, heads_per_group=heads_per_group
    )
    queries, heads, dim = keys - query_start, groups * heads_per_group, 192
    query = torch.randn(batch, queries, heads, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    # Match CP's sequence-major joint K/V storage, without making global copies.
    joint = torch.randn(keys, batch, 2, groups, dim, device="cuda", dtype=torch.bfloat16)
    key, value = joint.transpose(0, 1).unbind(2)
    key, value = key.detach().requires_grad_(), value.detach().requires_grad_()
    assert not key.is_contiguous() and not value.is_contiguous()
    output = selected_attention(
        query,
        key,
        value,
        positions[:, query_start:],
        positions,
        padding,
        selection=selection,
        region_size=8,
    )
    reference = F.scaled_dot_product_attention(
        query.float().transpose(1, 2),
        key.float().transpose(1, 2).repeat_interleave(heads // groups, dim=1),
        value.float().transpose(1, 2).repeat_interleave(heads // groups, dim=1),
        attn_mask=allowed,
    ).transpose(1, 2)
    torch.testing.assert_close(output.float(), reference, atol=0.015, rtol=0.015)
    empty = min(17, queries - 1)
    assert torch.count_nonzero(output[:, empty]) == 0
    upstream = torch.randn_like(output)
    assert torch.count_nonzero(upstream[:, -padding_tokens:]) > 0
    actual = torch.autograd.grad(output, (query, key, value), upstream)
    expected = torch.autograd.grad(reference, (query, key, value), upstream.float())
    for actual_gradient, expected_gradient in zip(actual, expected):
        assert torch.isfinite(actual_gradient).all()
        relative = (actual_gradient.float() - expected_gradient.float()).square().sum().sqrt()
        relative /= expected_gradient.float().square().sum().sqrt()
        assert relative < 0.012, relative.item()
    assert torch.count_nonzero(actual[0][:, empty]) == 0
    for gradient in actual[1:]:
        assert torch.count_nonzero(gradient[:, -padding_tokens:]) == 0
    return actual


@pytest.mark.parametrize("batch,query_start", [(1, 0), (2, 0), (1, 517), (2, 517)])
def test_selected_attention_matches_independent_packed_math(batch: int, query_start: int) -> None:
    """Exercise shifted packed documents, GQA, duplicate/sentinel routes and empty rows."""
    _assert_parity(batch, 1153, query_start, padding_tokens=3)


def test_selected_attention_matches_native_head_groups() -> None:
    """Exercise all 64 query heads and four KV groups at the native 16:1 ratio."""
    _assert_parity(2, 257, query_start=19, padding_tokens=13, groups=4, heads_per_group=16)


@pytest.mark.parametrize("batch", [1, 2])
def test_selected_attention_initializes_long_padding_gradients(monkeypatch: pytest.MonkeyPatch, batch: int) -> None:
    """Poison old empty K/V-gradient allocations beyond the region kernel's eight stores."""
    keys, groups, dim = 257, 2, 192
    original_empty = torch.empty

    def poison_empty(*size: int | tuple[int, ...] | torch.Size, **options: object) -> torch.Tensor:
        tensor = original_empty(*size, **options)
        if tensor.shape == (batch, keys, groups, dim) and tensor.is_floating_point():
            tensor.fill_(torch.nan)
        return tensor

    # The defective empty allocation left the >8-token physical padding tail
    # untouched. Zero initialization is required independently of allocator reuse.
    monkeypatch.setattr(torch, "empty", poison_empty)
    _assert_parity(batch, keys, query_start=19, padding_tokens=113)


def test_inverse_edges_matches_independent_sets() -> None:
    """Check exact CUDA edge sets without assuming atomic insertion order."""
    regions, batch, groups, queries, slots = 7, 2, 2, 5, 6
    ids = torch.full((batch, groups, queries, slots), regions, dtype=torch.int32)
    for b in range(batch):
        for g in range(groups):
            for q in range(queries - 1):
                chosen = sorted([b, b, (q + g + b) % regions, (2 * q + g) % regions])
                ids[b, g, q, :4] = torch.tensor(chosen)
    offsets, edges = _inverse_edges(ids.cuda(), regions)
    offsets, edges = offsets.cpu().tolist(), edges.cpu().tolist()
    assert offsets[0] == 0
    expected_count = 0
    for b in range(batch):
        for g in range(groups):
            for region in range(regions):
                row = (b * groups + g) * regions + region
                expected = [q for q in range(queries) if region in ids[b, g, q].tolist()]
                actual = edges[offsets[row] : offsets[row + 1]]
                assert len(actual) == len(set(actual))
                assert sorted(actual) == expected
                expected_count += len(expected)
    assert offsets[-1] == expected_count
