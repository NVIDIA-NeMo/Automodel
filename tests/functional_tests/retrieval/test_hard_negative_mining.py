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

"""Real-CUDA functional coverage for hard-negative mining."""

import numpy as np
import pytest
import torch

from nemo_automodel.recipes.retrieval.mine_hard_negatives import MineHardNegativesRecipe

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def test_mining_ranks_negatives_and_filters_false_negatives() -> None:
    """Mine by GPU similarity while preserving duplicate-positive score order."""
    recipe = MineHardNegativesRecipe.__new__(MineHardNegativesRecipe)
    queries = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    documents = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.95, 0.20],
            [0.20, 0.95],
            [-1.0, 0.0],
            [0.0, -1.0],
        ],
        dtype=np.float32,
    )
    positives = [[0, 0], [1]]

    neg_indices, neg_scores, pos_scores = recipe._mine_hard_negatives(
        queries,
        documents,
        positives,
        batch_size=2,
        num_negs=2,
    )

    assert neg_indices == [[2, 3], [3, 2]]
    np.testing.assert_allclose(neg_scores, [[0.95, 0.20], [0.95, 0.20]])
    assert pos_scores[0] == pytest.approx([1.0, 1.0])
    assert pos_scores[1] == pytest.approx([1.0])

    filtered_indices, filtered_scores, _ = recipe._mine_hard_negatives(
        queries,
        documents,
        positives,
        batch_size=1,
        num_negs=2,
        hard_neg_margin=0.1,
        hard_neg_margin_type="abs",
    )

    assert 2 not in filtered_indices[0]
    assert 3 not in filtered_indices[1]
    assert all(score <= 0.9 for row in filtered_scores for score in row)
