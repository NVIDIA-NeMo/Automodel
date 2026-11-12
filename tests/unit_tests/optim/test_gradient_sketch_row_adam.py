# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_automodel.components.optim.gradient_sketch.row_adam import row_adam_direction


def test_running_rms_reference():
    history = [torch.tensor([[1.0, 3.0], [2.0, 4.0]]), torch.tensor([[2.0, 1.0], [8.0, 2.0]])]
    moment = torch.zeros(2, 1)
    for index, sketch in enumerate(history, 1):
        expected_moment = sum(
            (1 - 0.95) * 0.95 ** (index - 1 - j) * g.square().mean(1, keepdim=True)
            for j, g in enumerate(history[:index])
        )
        expected = sketch / ((expected_moment / (1 - 0.95**index)).sqrt() + 1e-8)
        torch.testing.assert_close(row_adam_direction(sketch, moment, step=index, beta2=0.95, epsilon=1e-8), expected)
        torch.testing.assert_close(moment, expected_moment)


def test_inplace_matches_out_of_place_and_reuses_storage():
    sketch = torch.tensor([[1.0, 3.0], [2.0, 4.0], [0.5, -1.0]])
    expected = row_adam_direction(sketch.clone(), torch.zeros(3, 1), step=3, beta2=0.9, epsilon=1e-8)
    moment = torch.zeros(3, 1)
    actual = row_adam_direction(sketch, moment, step=3, beta2=0.9, epsilon=1e-8, inplace=True)
    assert actual is sketch
    torch.testing.assert_close(actual, expected)
    assert torch.all(moment > 0)


def test_rejects_invalid_arguments():
    with pytest.raises(ValueError):
        row_adam_direction(torch.ones(2, 2), torch.zeros(2, 1), step=0, beta2=0.9, epsilon=1e-8)
    with pytest.raises(ValueError):
        row_adam_direction(torch.ones(2, 2), torch.zeros(2, 1), step=1, beta2=0.9, epsilon=float("nan"))
    with pytest.raises(ValueError, match="one second moment"):
        row_adam_direction(torch.ones(2, 2), torch.zeros(2, 2), step=1, beta2=0.9, epsilon=1e-8)
    with pytest.raises(FloatingPointError):
        row_adam_direction(torch.tensor([[float("nan"), 1.0]]), torch.zeros(1, 1), step=1, beta2=0.9, epsilon=1e-8)


def test_nonfinite_sketch_leaves_moment_unchanged():
    moment = torch.full((1, 1), 0.5)
    with pytest.raises(FloatingPointError):
        row_adam_direction(torch.tensor([[float("inf"), 1.0]]), moment, step=1, beta2=0.9, epsilon=1e-8)
    torch.testing.assert_close(moment, torch.full((1, 1), 0.5))
