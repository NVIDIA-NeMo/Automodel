# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import pytest
import torch

from nemo_automodel.components.datasets.llm.mock_prefix_tree import build_mock_rollout_dataset
from nemo_automodel.components.datasets.llm.prefix_tree import (
    CROSS_ENTROPY_IGNORE_IDX,
    FoldedRollouts,
    fold_shared_prefix_rollouts,
    prefix_tree_collate_fn,
)


def test_fold_basic_layout_and_mask():
    folded = fold_shared_prefix_rollouts([1, 2], [[10, 11], [20, 21, 22]])

    # Flat dedup layout: prompt once, then each completion.
    assert folded.input_ids == [1, 2, 10, 11, 20, 21, 22]
    # Pre-shifted next-token labels: prompt masked, each completion predicts its
    # own next token, and each completion's last token is masked (no successor).
    _ = CROSS_ENTROPY_IGNORE_IDX
    assert folded.labels == [_, _, 11, _, 21, 22, _]
    # Each completion's positions continue from the prompt length (P=2).
    assert folded.position_ids == [0, 1, 2, 3, 2, 3, 4]
    # Tree structure: node 0 is the prompt, nodes 1..N the completions; each path
    # is prompt -> completion. The magi backend turns this into the AttnMaskSpec.
    assert folded.node_lengths == [2, 2, 3]
    assert folded.sample_paths == [[0, 1], [0, 2]]


def test_fold_single_completion():
    folded = fold_shared_prefix_rollouts([5], [[7, 8]])
    assert folded.input_ids == [5, 7, 8]
    # Prompt masked; token 7 predicts 8; last token 8 masked.
    assert folded.labels == [CROSS_ENTROPY_IGNORE_IDX, 8, CROSS_ENTROPY_IGNORE_IDX]
    assert folded.position_ids == [0, 1, 2]
    assert folded.node_lengths == [1, 2]
    assert folded.sample_paths == [[0, 1]]


def test_fold_empty_prompt_collapses_to_varlen_blocks():
    folded = fold_shared_prefix_rollouts([], [[10, 11], [20]])
    assert folded.input_ids == [10, 11, 20]
    # No prompt: each completion still shifts within itself, last token masked.
    assert folded.labels == [11, CROSS_ENTROPY_IGNORE_IDX, CROSS_ENTROPY_IGNORE_IDX]
    # Each completion restarts at position 0.
    assert folded.position_ids == [0, 1, 0]
    # Empty prompt node 0; each completion is its own block (no shared ancestor).
    assert folded.node_lengths == [0, 2, 1]
    assert folded.sample_paths == [[1], [2]]


def test_fold_custom_ignore_idx():
    folded = fold_shared_prefix_rollouts([1], [[10, 11]], ignore_idx=-1)
    # Custom ignore value used for the masked prompt and masked last token.
    assert folded.labels == [-1, 11, -1]


def test_fold_returns_folded_rollouts_instance():
    folded = fold_shared_prefix_rollouts([1], [[2]])
    assert isinstance(folded, FoldedRollouts)


@pytest.mark.parametrize(
    "prompt, completions, match",
    [
        ([1], [], "non-empty"),
        ([1], [[10], []], "every completion must be non-empty"),
    ],
)
def test_fold_validation_errors(prompt, completions, match):
    with pytest.raises(ValueError, match=match):
        fold_shared_prefix_rollouts(prompt, completions)


def test_collate_builds_batched_tensors():
    group = {"prompt_ids": [1, 2], "completions": [[10, 11], [20]]}
    batch = prefix_tree_collate_fn([group])

    assert set(batch) == {"input_ids", "labels", "position_ids", "prefix_tree"}
    for key in ("input_ids", "labels", "position_ids"):
        assert batch[key].shape == (1, 5)
        assert batch[key].dtype == torch.long
    assert torch.equal(batch["input_ids"], torch.tensor([[1, 2, 10, 11, 20]]))
    assert torch.equal(batch["labels"], torch.tensor([[-100, -100, 11, -100, -100]]))
    # Tree structure (node_lengths, sample_paths) for the magi backend to build the spec.
    assert batch["prefix_tree"] == ([2, 2, 1], [[0, 1], [0, 2]])


def test_collate_rejects_multi_group_batch():
    group = {"prompt_ids": [1], "completions": [[2]]}
    with pytest.raises(ValueError, match="local_batch_size=1 only"):
        prefix_tree_collate_fn([group, group])


def test_build_mock_rollout_dataset_shape_and_determinism():
    ds = build_mock_rollout_dataset(num_groups=3, completions_per_group=4, prompt_len=8, completion_len=5)
    assert len(ds) == 3
    group = ds[0]
    assert len(group["prompt_ids"]) == 8
    assert len(group["completions"]) == 4
    assert all(len(c) == 5 for c in group["completions"])
    # Same seed -> identical data; it must be foldable by the collate.
    again = build_mock_rollout_dataset(num_groups=3, completions_per_group=4, prompt_len=8, completion_len=5)
    assert ds == again
    assert prefix_tree_collate_fn([group])["input_ids"].shape == (1, 8 + 4 * 5)
