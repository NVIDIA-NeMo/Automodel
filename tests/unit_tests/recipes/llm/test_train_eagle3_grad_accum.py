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

from nemo_automodel.recipes.llm.train_eagle3 import _optim_steps_per_epoch, _should_sync_grads


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


def _sync(pending, batch_idx, *, accum=4, batches_per_epoch=10, is_ddp=True):
    return _should_sync_grads(
        pending_micro_batches=pending,
        grad_accumulation_steps=accum,
        batch_idx=batch_idx,
        batches_per_epoch=batches_per_epoch,
        is_ddp=is_ddp,
    )


def test_should_sync_always_true_without_ddp():
    for pending in range(4):
        assert _sync(pending, batch_idx=0, is_ddp=False) is True


def test_should_sync_only_on_window_close_under_ddp():
    assert _sync(0, batch_idx=0) is False
    assert _sync(1, batch_idx=1) is False
    assert _sync(2, batch_idx=2) is False
    assert _sync(3, batch_idx=3) is True  # window closer


def test_should_sync_on_epoch_final_batch_even_mid_window():
    # The trailing-flush step consumes the last batch's grads -> must sync.
    assert _sync(0, batch_idx=9, batches_per_epoch=10) is True
    assert _sync(2, batch_idx=9, batches_per_epoch=10) is True


def test_should_sync_every_step_when_length_unknown():
    for pending in range(4):
        assert _sync(pending, batch_idx=pending, batches_per_epoch=None) is True


def test_should_sync_every_step_when_accum_is_one():
    for batch_idx in range(5):
        assert _sync(0, batch_idx=batch_idx, accum=1) is True
