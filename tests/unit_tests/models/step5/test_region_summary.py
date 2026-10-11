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

"""Lossless compression of only the historical CSA regions used by a CP shard."""

from dataclasses import replace

import pytest
import torch

from nemo_automodel.components.models.step5 import sparse
from nemo_automodel.components.models.step5.kernels import gather_region_payload
from nemo_automodel.components.models.step5.sparse import (
    RegionMetadata,
    Step5SparseIndexer,
    _summarize_region_payload,
    _summary_ranges,
    region_metadata,
)
from tests.unit_tests.models.step5.test_model import tiny_config


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
