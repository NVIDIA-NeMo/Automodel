# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration and empty-input contracts for the optional cuDNN indexer."""

from dataclasses import asdict

import pytest
import torch

from nemo_automodel.components.models.common.cudnn_sparse_attention import cudnn_indexer_scores
from nemo_automodel.components.models.glm5_next.backend import Glm5NextBackendConfig
from nemo_automodel.components.models.glm5_next.layers import Glm5NextSparseAttention
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
