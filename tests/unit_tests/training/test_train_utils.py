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

from nemo_automodel.components.training.utils import count_tail_padding


def test_docstring_example():
    labels = torch.tensor(
        [
            [-100, 1, 1, -100, -100],  # 2 tail -100s
            [-100, -100, 2, 3, 4],  # 0 tail -100s
            [5, 6, -100, -100, -100],  # 3 tail -100s
        ]
    )
    assert count_tail_padding(labels) == 5


@pytest.mark.parametrize(
    "labels, expected",
    [
        # No padding at all
        (torch.tensor([[1, 2, 3], [4, 5, 6]]), 0),
        # Entire sequence is padding
        (torch.full((2, 4), -100), 8),
        # Different ignore label
        (torch.tensor([[9, 0, 0], [0, 0, 0]]), 5),
    ],
)
def test_various_cases(labels, expected):
    """
    Covers:
    1. no ignore_label present
    2. every position is ignore_label
    3. custom ignore_label value (0)
    """
    ignore_label = 0 if (labels == 0).any() else -100
    assert count_tail_padding(labels, ignore_label=ignore_label) == expected


def test_random_shapes():
    """
    Generate random examples and compare with a simple-but-slow reference
    implementation to guard against shape / broadcasting regressions.
    """
    torch.manual_seed(0)
    for _ in range(10):
        batch = torch.randint(
            1,
            8,
            size=(
                torch.randint(1, 5, ()).item(),  # batch size
                torch.randint(1, 10, ()).item(),
            ),
        )  # seq len
        # randomly sprinkle ignore tokens
        mask = torch.rand_like(batch.float()) < 0.3
        batch[mask] = -100

        # brute-force reference
        ref = 0
        for row in batch:
            idx = (row != -100).nonzero(as_tuple=True)[0]
            if len(idx) == 0:
                ref += row.numel()
            else:
                ref += (row[idx[-1] + 1 :] == -100).sum().item()

        assert count_tail_padding(batch) == ref
