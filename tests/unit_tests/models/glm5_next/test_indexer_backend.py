# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration, empty-input and CUDA numerical contracts for the optional cuDNN indexer."""

from dataclasses import asdict

import pytest
import torch
import torch.nn.functional as F

from nemo_automodel.components.models.common.cudnn_sparse_attention import (
    cudnn_indexer_scores,
    is_cudnn_sparse_attention_available,
)
from nemo_automodel.components.models.glm5_next.backend import Glm5NextBackendConfig
from nemo_automodel.components.models.glm5_next.config import Glm5NextTextConfig
from nemo_automodel.components.models.glm5_next.layers import Glm5NextKPoolIndexer, Glm5NextSparseAttention
from tests.unit_tests.models.glm5_next.conftest import tiny_backend, tiny_glm5_next_config


@pytest.mark.parametrize("indexer", ["torch", "cudnn"])
def test_indexer_backend_is_independent_of_attention(indexer: str) -> None:
    backend = Glm5NextBackendConfig(**{**asdict(tiny_backend()), "indexer": indexer})
    module = Glm5NextSparseAttention(tiny_glm5_next_config().text_config, 3, backend)
    assert module.indexer.backend == indexer


def test_shared_backend_preserves_torch_indexer() -> None:
    module = Glm5NextSparseAttention(tiny_glm5_next_config().text_config, 3, tiny_backend())
    assert module.indexer.backend == "torch"


def test_unknown_indexer_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported indexer"):
        Glm5NextBackendConfig(**{**asdict(tiny_backend()), "indexer": "invalid"})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA contract")
@pytest.mark.parametrize(("queries", "pools"), [(0, 3), (3, 0)])
def test_empty_index_scores_need_no_kernel(queries: int, pools: int) -> None:
    q = torch.empty((1, queries, 32, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.empty((pools, 128), device="cuda", dtype=torch.bfloat16)
    w = torch.empty((1, queries, 32), device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(queries, device="cuda")
    scores = cudnn_indexer_scores(q, k, w, positions, 4, 1 / 64)
    assert scores.shape == (1, queries, pools)
    assert scores.dtype == torch.float32
    assert scores.device == q.device


@pytest.mark.skipif(
    not torch.cuda.is_available() or not is_cudnn_sparse_attention_available(),
    reason="requires CUDA and cuDNN DSA",
)
@pytest.mark.parametrize("query_start", [37, 40])
def test_cudnn_indexer_scores_and_selection_match_fp32_oracle(query_start: int) -> None:
    """Nonempty BF16 KPool4 scores, 32 heads of width 128, signed weights and offset queries.

    The query chunk starts at a nonzero document position (inside a pool for 37, on a pool
    boundary for 40), so visibility depends on the causal offset. Each document has more
    visible pools than the 8-pool budget.
    """
    pool, topk_pools, length = 4, 8, 96
    config = Glm5NextTextConfig(
        hidden_size=256,
        q_lora_rank=64,
        index_n_heads=32,
        index_head_dim=128,
        index_kpool=pool,
        index_topk=pool * topk_pools,
        index_kpool_always_select_tail=True,
    )
    torch.manual_seed(1009)
    indexer = Glm5NextKPoolIndexer(config, 3, torch.bfloat16, backend="cudnn").cuda()
    with torch.no_grad():
        for parameter in indexer.parameters():
            parameter.normal_(0.0, 0.08)
        indexer.k_norm.weight.fill_(1.0)
    hidden = torch.randn(1, length, config.hidden_size, device="cuda", dtype=torch.bfloat16)
    query_resid = torch.randn(1, length, config.q_lora_rank, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(query_start, length, device="cuda")
    query_hidden, query_resid_chunk = hidden[:, query_start:], query_resid[:, query_start:]
    pool_keys, pool_indices = indexer.prepare_pools(hidden)

    # Independent FP32 oracle on the same BF16 projections: sum_h w[q, h] * relu(q_h . k_p) * scale.
    queries = indexer.wq_b(query_resid_chunk).view(1, -1, config.index_n_heads, config.index_head_dim)
    weights = indexer.weights_proj(query_hidden)
    assert (weights < 0).any() and (weights > 0).any()
    scale = indexer.softmax_scale * config.index_n_heads**-0.5
    oracle = (
        torch.einsum(
            "qh,qhp->qp", weights[0].float(), F.relu(torch.einsum("qhd,pd->qhp", queries[0].float(), pool_keys.float()))
        )
        * scale
    )
    visible = pool_indices[:, -1][None, :] <= positions[:, None]
    actual = cudnn_indexer_scores(queries, pool_keys, weights, positions, pool, scale)[0]
    assert actual.dtype == torch.float32 and actual.shape == oracle.shape
    assert torch.isneginf(actual[~visible]).all(), "future pools must be unselectable"
    observed, reference = actual[visible].double(), oracle[visible].double()
    scale_rms = reference.square().mean().sqrt()
    assert (observed - reference).abs().max() / scale_rms <= 0.02

    # Selection through the module: compare raw key sets where the oracle's k-th/(k+1)-th margin is separated.
    selected = indexer.select(query_hidden, query_resid_chunk, positions, pool_keys, pool_indices, length)[0]
    assert (selected <= positions[:, None]).all(), "no query may select a future key"
    margin = 0.05 * float(scale_rms)
    checked = 0
    for row, position in enumerate(positions.tolist()):
        scores = oracle[row].masked_fill(~visible[row], float("-inf"))
        assert int(visible[row].sum()) > topk_pools
        ranked = scores.sort(descending=True).values
        if ranked[topk_pools - 1] - ranked[topk_pools] < margin:
            continue
        chosen = scores.topk(topk_pools).indices
        expected_keys = set(pool_indices[chosen].flatten().tolist())
        tail_start = (position + 1) // pool * pool
        expected_keys |= set(range(tail_start, position + 1))
        assert set(selected[row][selected[row] >= 0].tolist()) == expected_keys, position
        checked += 1
    assert checked >= len(positions) // 2, f"only {checked} of {len(positions)} queries had separated scores"
