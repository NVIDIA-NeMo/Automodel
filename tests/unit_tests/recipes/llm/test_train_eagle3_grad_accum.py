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

"""Unit tests for EAGLE-3 recipe gradient-accumulation arithmetic.

The trainer flushes a trailing partial accumulation window at the end of
every epoch so micro-batches in a non-divisible epoch are not silently
discarded. The LR scheduler must therefore be sized against the *ceil*
of ``num_batches / grad_accumulation_steps``, not the floor -- otherwise
``progress`` saturates and the trailing flushes train at
``min_lr_ratio``.
"""

from __future__ import annotations

import pytest

from nemo_automodel.recipes.llm.train_eagle3 import _optim_steps_per_epoch


@pytest.mark.parametrize(
    "num_batches,accum,expected",
    [
        (10, 1, 10),
        (10, 2, 5),
        (10, 3, 4),  # 3 full windows + 1 trailing micro-batch -> 4 steps
        (10, 4, 3),  # 2 full windows + 2 trailing -> 3 steps
        (1, 4, 1),  # entire epoch is one trailing flush
        (4, 4, 1),
        (5, 4, 2),
        (0, 4, 0),  # iterable dataloader / no length
    ],
)
def test_optim_steps_per_epoch_uses_ceil_division(num_batches, accum, expected):
    assert _optim_steps_per_epoch(num_batches, accum) == expected


def test_optim_steps_per_epoch_handles_invalid_inputs():
    assert _optim_steps_per_epoch(0, 1) == 0
    assert _optim_steps_per_epoch(-1, 4) == 0
    assert _optim_steps_per_epoch(10, 0) == 0
    assert _optim_steps_per_epoch(10, -1) == 0
