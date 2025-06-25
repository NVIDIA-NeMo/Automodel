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
from nemo_automodel.training.step_scheduler import StepScheduler


class _DummySampler:
    """Mimics DistributedSampler API – only set_epoch is required."""
    def __init__(self):
        """
        ctor
        """
        self.last_epoch = None

    def set_epoch(self, epoch: int):
        """
        set epoch

        Args:
            epoch (int): current epoch
        """
        self.last_epoch = epoch


class _DummyDataLoader:
    """
    A minimal iterable that behaves like a torch DataLoader for testing:
      • __iter__ returns the underlying list in order
      • __len__  returns fixed length
      • sampler  exposes .set_epoch()
    """
    def __init__(self, length: int):
        self._data    = list(range(length))
        self.sampler  = _DummySampler()

    def __iter__(self):
        # Real DataLoader returns a *fresh* iterator each time, replicate that.
        return iter(self._data)

    def __len__(self):
        return len(self._data)


@pytest.fixture
def dataloader():
    """Return a 5-item dummy dataloader shared by most tests."""
    return _DummyDataLoader(length=15)


def test_iteration_respects_max_steps(dataloader):
    """
    The iterator must stop once `max_steps` is reached even if the dataloader
    still has items left.
    """
    sch = StepScheduler(
        grad_acc_steps=1,
        ckpt_every_steps=100,   # large so it never triggers here
        dataloader=dataloader,
        max_steps=4,
    )

    yielded = list(iter(sch))
    n = len(yielded)
    assert n == 4, f"Scheduler should yield exactly `max_steps` batches but got {n}."
    assert sch.step == 4, f"Internal step counter must equal the number of yields but got {sch.step}."


def test_iteration_respects_max_steps_gacc2(dataloader):
    """
    The iterator must stop once `max_steps` is reached even if the dataloader
    still has items left.
    """
    sch = StepScheduler(
        grad_acc_steps=2,
        ckpt_every_steps=100,   # large so it never triggers here
        dataloader=dataloader,
        max_steps=4,
    )

    yielded = list(iter(sch))
    n = len(yielded)
    assert n == 8, f"Scheduler should yield exactly `max_steps` batches but got {n}."
    assert sch.step == 4, f"Internal step counter must equal the number of yields but got {sch.step}."

def test_optimizer_step_frequency(dataloader):
    """
    `is_optim_step` should be True every `grad_acc_steps` calls.
    """
    sch = StepScheduler(
        grad_acc_steps=3,       # every 3 physical steps => 3,6,9 ...
        ckpt_every_steps=99,
        dataloader=dataloader,
        max_steps=4,
    )

    optim_flags = []
    for _ in sch:
        optim_flags.append(sch.is_optim_step)

    assert optim_flags == [
        False, False, True,     # 1 .. 3
        False, False, True,     # 4 .. 6
        False, False, True,     # 7 .. 9
        False, False, True,     # 10 .. 12
    ]


def test_validation_step_logic(dataloader):
    """
    With `val_every_steps=N` and `grad_acc_steps=1`, validation should trigger
    every N *optimizer* steps.
    """
    sch = StepScheduler(
        grad_acc_steps=1,
        ckpt_every_steps=999,
        dataloader=dataloader,
        val_every_steps=2,      # every 2 optimizer steps
        max_steps=6,
    )

    val_flags = []
    for _ in sch:
        val_flags.append(sch.is_val_step)
    assert val_flags == [False, True, False, True, False, True]


def test_checkpoint_logic(dataloader):
    """
    A checkpoint is taken when
      • global step % ckpt_every_steps == 0  (and step != 0) OR
      • the batch is the final batch of an epoch.
    Verify both conditions.
    """
    sch = StepScheduler(
        grad_acc_steps=1,
        ckpt_every_steps=4,
        dataloader=dataloader,   # len == 5
        max_steps=10,            # two full epochs
    )

    ckpt_flags = []
    for _ in sch:
        ckpt_flags.append(sch.is_ckpt_step)

    # Expected pattern for steps 1..10 (see analysis section)
    expected = [False, False, False, True,    # 1-4  (4 hits modulus condition)
                False, False, False, True,    # 5-8  (8 hits modulus)
                False, False]
    assert ckpt_flags == expected


def test_state_dict_roundtrip(dataloader):
    """
    Saving and re-loading `state_dict` should restore both step and epoch.
    """
    # Part-1: advance some steps
    sch1 = StepScheduler(
        grad_acc_steps=1,
        ckpt_every_steps=10,
        dataloader=dataloader,
        start_epoch=3,
    )
    for _ in range(7):
        next(iter(sch1))    # advance 7 steps

    # Part-2: new instance loads saved state
    sch2 = StepScheduler(
        grad_acc_steps=1,
        ckpt_every_steps=10,
        dataloader=dataloader,
    )
    sch2.load_state_dict(sch1.state_dict())

    assert sch2.step  == sch1.step
    assert sch2.epoch == sch1.epoch


def test_set_epoch_calls_sampler(dataloader):
    """
    `set_epoch` should forward the epoch value to the dataloader's sampler.
    """
    sch = StepScheduler(
        grad_acc_steps=1,
        ckpt_every_steps=10,
        dataloader=dataloader,
    )

    sch.set_epoch(42)
    assert dataloader.sampler.last_epoch == 42
