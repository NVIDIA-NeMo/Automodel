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

"""Independent CSA selection checks for packed prefixes and sparse histories."""

import pytest
import torch

from nemo_automodel.components.models.step5.sparse import Step5SparseIndexer, region_metadata
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
