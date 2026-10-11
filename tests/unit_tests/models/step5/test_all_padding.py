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

"""All-padding CSA document semantics and explicit autograd dependencies."""

import pytest
import torch

from nemo_automodel.components.models.step5.fa4 import _all_padding_attention
from nemo_automodel.components.models.step5.sparse import Step5SparseIndexer, region_metadata
from tests.unit_tests.models.step5.test_model import tiny_config


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
