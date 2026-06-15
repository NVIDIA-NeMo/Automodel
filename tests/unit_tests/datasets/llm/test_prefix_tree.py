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

from nemo_automodel.components.datasets.llm.prefix_tree import (
    CROSS_ENTROPY_IGNORE_IDX,
    FoldedRollouts,
    build_mock_rollout_dataset,
    fold_shared_prefix_rollouts,
    prefix_tree_collate_fn,
)


def test_fold_basic_layout_and_mask():
    folded = fold_shared_prefix_rollouts([1, 2], [[10, 11], [20, 21, 22]])

    # Flat dedup layout: prompt once, then each completion.
    assert folded.input_ids == [1, 2, 10, 11, 20, 21, 22]
    # Prompt masked; completions supervised with their own ids.
    assert folded.labels == [CROSS_ENTROPY_IGNORE_IDX, CROSS_ENTROPY_IGNORE_IDX, 10, 11, 20, 21, 22]
    # Each completion's positions continue from the prompt length (P=2).
    assert folded.position_ids == [0, 1, 2, 3, 2, 3, 4]
    assert folded.node_lengths == [2, 2, 3]
    assert folded.sample_paths == [[0, 1], [0, 2]]

    spec = folded.spec
    assert spec.total_seqlen == 7
    assert spec.q_ranges == [[0, 2], [2, 4], [2, 4], [4, 7], [4, 7]]
    assert spec.k_ranges == [[0, 2], [2, 4], [0, 2], [4, 7], [0, 2]]
    assert spec.mask_types == ["causal", "causal", "full", "causal", "full"]
    # Per-sample flat ranges: prompt node + completion node.
    assert folded.sample_token_ranges == [[[0, 2], [2, 4]], [[0, 2], [4, 7]]]


def test_fold_single_completion():
    folded = fold_shared_prefix_rollouts([5], [[7, 8]])
    assert folded.input_ids == [5, 7, 8]
    assert folded.labels == [CROSS_ENTROPY_IGNORE_IDX, 7, 8]
    assert folded.position_ids == [0, 1, 2]
    assert folded.sample_paths == [[0, 1]]
    assert folded.spec.q_ranges == [[0, 1], [1, 3], [1, 3]]
    assert folded.spec.mask_types == ["causal", "causal", "full"]


def test_fold_empty_prompt_collapses_to_varlen_blocks():
    folded = fold_shared_prefix_rollouts([], [[10, 11], [20]])
    assert folded.input_ids == [10, 11, 20]
    # No prompt -> nothing masked.
    assert folded.labels == [10, 11, 20]
    # Each completion restarts at position 0.
    assert folded.position_ids == [0, 1, 0]
    assert folded.node_lengths == [0, 2, 1]
    assert folded.sample_paths == [[1], [2]]
    # Block-diagonal causal: each completion attends only to itself.
    assert folded.spec.q_ranges == [[0, 2], [2, 3]]
    assert folded.spec.k_ranges == [[0, 2], [2, 3]]
    assert folded.spec.mask_types == ["causal", "causal"]


def test_fold_custom_completion_labels_and_ignore_idx():
    folded = fold_shared_prefix_rollouts(
        [1],
        [[10, 11]],
        completion_labels=[[-1, 11]],
        ignore_idx=-1,
    )
    assert folded.labels == [-1, -1, 11]


def test_fold_returns_folded_rollouts_instance():
    folded = fold_shared_prefix_rollouts([1], [[2]])
    assert isinstance(folded, FoldedRollouts)


@pytest.mark.parametrize(
    "prompt, completions, labels, match",
    [
        ([1], [], None, "non-empty"),
        ([1], [[10], []], None, "every completion must be non-empty"),
        ([1], [[10]], [[1], [2]], "one entry per completion"),
        ([1], [[10, 11]], [[1]], "match its completion length"),
    ],
)
def test_fold_validation_errors(prompt, completions, labels, match):
    with pytest.raises(ValueError, match=match):
        fold_shared_prefix_rollouts(prompt, completions, completion_labels=labels)


def test_collate_builds_batched_tensors():
    group = {"prompt_ids": [1, 2], "completions": [[10, 11], [20]]}
    batch = prefix_tree_collate_fn([group])

    assert set(batch) == {"input_ids", "labels", "position_ids", "magi_attn_spec"}
    for key in ("input_ids", "labels", "position_ids"):
        assert batch[key].shape == (1, 5)
        assert batch[key].dtype == torch.long
    assert torch.equal(batch["input_ids"], torch.tensor([[1, 2, 10, 11, 20]]))
    assert torch.equal(batch["labels"], torch.tensor([[-100, -100, 10, 11, 20]]))
    # The spec the recipe activates matches a direct fold of the same group.
    expected = fold_shared_prefix_rollouts([1, 2], [[10, 11], [20]]).spec
    assert batch["magi_attn_spec"].fingerprint() == expected.fingerprint()


def test_collate_honors_completion_labels():
    group = {"prompt_ids": [1], "completions": [[10, 11]], "completion_labels": [[-100, 11]]}
    batch = prefix_tree_collate_fn([group])
    assert torch.equal(batch["labels"], torch.tensor([[-100, -100, 11]]))


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
