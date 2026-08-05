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

"""Deterministic mock shared-prefix rollout data for prefix-tree smoke runs."""

import random


def build_mock_rollout_dataset(
    *,
    num_groups: int = 16,
    completions_per_group: int = 4,
    prompt_len: int = 32,
    completion_len: int = 16,
    vocab_size: int = 1024,
    seed: int = 0,
) -> list[dict]:
    """Build a deterministic mock shared-prefix rollout dataset for smoke runs.

    Each group is one shared prompt with ``completions_per_group`` completions, in
    the ``{"prompt_ids", "completions"}`` schema consumed by
    ``prefix_tree_collate_fn``. Token ids are random in ``[2, vocab_size)``; this
    is a pipeline smoke, not a quality dataset.

    Args:
        num_groups: number of rollout groups.
        completions_per_group: completions (leaves) sharing each prompt.
        prompt_len: shared prompt length per group.
        completion_len: length of each completion.
        vocab_size: upper bound (exclusive) for random token ids.
        seed: RNG seed for reproducibility.

    Returns:
        A list of ``{"prompt_ids": list[int], "completions": list[list[int]]}``.
    """
    rng = random.Random(seed)

    def _ids(n: int) -> list[int]:
        return [rng.randint(2, vocab_size - 1) for _ in range(n)]

    return [
        {
            "prompt_ids": _ids(prompt_len),
            "completions": [_ids(completion_len) for _ in range(completions_per_group)],
        }
        for _ in range(num_groups)
    ]
