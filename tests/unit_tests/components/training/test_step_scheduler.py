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

from nemo_automodel.components.training.step_scheduler import StepScheduler, _calculate_max_steps


class SizedDataLoader:
    def __init__(self, num_batches: int, global_batch_size: int = 1, local_batch_size: int = 1):
        self.num_batches = num_batches
        # self.global_batch_size = global_batch_size
        # self.local_batch_size = local_batch_size

    def __iter__(self):
        for i in range(self.num_batches):
            # ans = []
            # for j in range(self.global_batch_size):
            yield {"batch": (i, 0)}

    def __len__(self):
        return self.num_batches #* (self.global_batch_size // self.local_batch_size)


class IterableDataLoader:
    def __init__(self, num_batches: int):
        self.num_batches = num_batches

    def __iter__(self):
        for i in range(self.num_batches):
            yield {"batch": i}

    def __len__(self):
        raise NotImplementedError("IterableDataLoader does not support __len__")


def test_iteration_groups_and_epoch_increment_sized():
    # grad_acc_steps = global // (local * dp) = 8 // (2 * 2) = 2
    dataloader = SizedDataLoader(num_batches=5)
    scheduler = StepScheduler(
        global_batch_size=8,
        local_batch_size=2,
        dp_size=2,
        ckpt_every_steps=1000,  # effectively disabled for this test
        dataloader=dataloader,
        num_epochs=1,
        max_steps=1000,
    )

    groups = []
    for group in scheduler:
        groups.append([b for b in group])

    # Expect two full groups of 2 and a final remainder group of 1
    assert [len(g) for g in groups] == [2, 2, 1]
    # One epoch completed and 3 steps performed
    assert scheduler.step == 3
    assert scheduler.epoch == 1
@pytest.mark.parametrize(
    "max_steps, ckpt_every_steps",
    [
        (11, 1),
        (3, 1),
        (3, 2),
        (3, 3),
        (5, 3),
        (6, 2),
        (10, 4),
    ],
)
def test_resume(max_steps, ckpt_every_steps):
    from copy import deepcopy
    max_steps = 10
    dataloader = SizedDataLoader(num_batches=max_steps)
    scheduler = StepScheduler(
        global_batch_size=1,  # grad_acc_steps = 1
        local_batch_size=1,
        dp_size=1,
        ckpt_every_steps=ckpt_every_steps,
        dataloader=dataloader,
        num_epochs=1,
        max_steps=max_steps,
    )

    ref_outputs = []
    ref_state = None
    saved_is_ckpt = None
    start_collecting = False
    for i, _  in enumerate(scheduler):
        if i == 2:
            ref_state = deepcopy(scheduler.state_dict())
            saved_is_ckpt = scheduler.is_ckpt_step
            # Start collecting from the NEXT iteration after snapshot
            start_collecting = True
            continue
        if start_collecting:
            # record exact values; sequence starts at step ref_state['step']
            ref_outputs.append((scheduler.step, scheduler.is_val_step, scheduler.is_ckpt_step))

    del scheduler
    scheduler = StepScheduler(
        global_batch_size=1,  # grad_acc_steps = 1
        local_batch_size=1,
        dp_size=1,
        ckpt_every_steps=ckpt_every_steps,
        dataloader=dataloader,
        num_epochs=1,
        max_steps=max_steps,
    )

    scheduler.load_state_dict(ref_state)
    for j, _  in enumerate(scheduler):
        expected_step, expected_is_val, expected_is_ckpt = ref_outputs.pop(0)
        # Ensure we don't checkpoint immediately after resume if we saved on a checkpoint step
        if j == 0 and saved_is_ckpt and ckpt_every_steps > 1:
            expected_is_ckpt = False
        assert (expected_step, expected_is_val, expected_is_ckpt) == (
            scheduler.step, scheduler.is_val_step, scheduler.is_ckpt_step
        )
        # step at resume should be ref_state['step'] + j
        assert scheduler.step == j + ref_state["step"]
    assert len(ref_outputs) == 0

@pytest.mark.parametrize(
    "max_steps, ckpt_every_steps, global_batch_size, local_batch_size, is_ckpt_step",
    [
        (1, 1, 1, 1, [True]),
        (3, 1, 2, 2, [True, True, True]),
        (3, 2, 2, 1, [False, True, True]),
        (3, 3, 2, 2, [False, False, True]),
        (5, 3, 1, 1, [False, False, True, False, True]),
        (6, 2, 2, 1, [False, True, False, True, False, True]),
        (10, 4, 4, 2, [False, False, False, True, False, False, False, True, False, True]),
    ],
)
def test_is_ckpt_step_parametrized_iterable(max_steps, ckpt_every_steps, global_batch_size, local_batch_size, is_ckpt_step):
    dataloader = SizedDataLoader(
        num_batches=max_steps * (global_batch_size // local_batch_size),
    )
    scheduler = StepScheduler(
        global_batch_size=global_batch_size,
        local_batch_size=local_batch_size,
        dp_size=1,
        ckpt_every_steps=ckpt_every_steps,
        dataloader=dataloader,
        num_epochs=1,
        max_steps=max_steps,
    )

    periodic_ckpt_steps = []
    assert len(is_ckpt_step) == max_steps
    for i, batches in enumerate(scheduler):
        assert len(batches) == global_batch_size // local_batch_size
        # After each yielded group, scheduler.step has been incremented
        # Record steps where the periodic checkpoint condition fires
        assert is_ckpt_step.pop(0) == scheduler.is_ckpt_step, i
        if scheduler.is_ckpt_step:
            periodic_ckpt_steps.append(scheduler.step)
        assert scheduler.step == i
    assert len(is_ckpt_step) == 0

    # Finished should trigger a checkpoint at the end regardless of periodicity
    assert scheduler.step == max_steps
    assert scheduler.is_ckpt_step is True

@pytest.mark.parametrize(
    "expected_last_batch_steps",
    [
        ([3, 7, 9])
    ],
)
def test_is_ckpt_step_triggers_on_last_batch_with_sized_dataloader(expected_last_batch_steps):
    epoch_len = 4  # number of micro-batches per epoch
    dataloader = SizedDataLoader(num_batches=epoch_len)
    scheduler = StepScheduler(
        global_batch_size=1,  # grad_acc_steps = 1 so step aligns with micro-batches
        local_batch_size=1,
        dp_size=1,
        ckpt_every_steps=1000,  # disable periodic checkpointing
        dataloader=dataloader,
        num_epochs=100,  # large to allow multiple epochs until max_steps
        max_steps=10,
    )

    last_batch_trigger_steps = []
    # Iterate over epochs using the provided epochs generator
    for _ in scheduler.epochs:
        for _ in scheduler:
            if scheduler.is_ckpt_step:
                last_batch_trigger_steps.append(scheduler.step)

    # Expect a trigger at the end of each epoch (steps 3, 7 for max_steps=10, epoch_len=4)
    assert last_batch_trigger_steps == expected_last_batch_steps

    # Finished also triggers checkpoint
    assert scheduler.step == 10
    assert scheduler.is_ckpt_step is True
    assert scheduler.state_dict() == {"step": 10, "epoch": 2, "max_steps": 10}

@pytest.mark.parametrize(
    "max_steps, ckpt_every_steps, epoch, num_epochs, global_batch_size, local_batch_size, num_batches, is_ckpt_step",
    [
        (None, 1000, 0, 1, 64, 1, 317 + 1, [False] * 317 + [True]),
        (1000, 1000, 0, 1, 64, 1, 317 + 1, [False] * 317 + [True]),
    ],
)
def test_ckpt_every_steps_larger_than_max_steps(max_steps, ckpt_every_steps, epoch, num_epochs, global_batch_size, local_batch_size, num_batches, is_ckpt_step):
    dataloader = SizedDataLoader(
        num_batches=num_batches * (global_batch_size // local_batch_size),
    )
    scheduler = StepScheduler(
        global_batch_size=global_batch_size,
        local_batch_size=local_batch_size,
        dp_size=1,
        ckpt_every_steps=ckpt_every_steps,
        dataloader=dataloader,
        start_epoch=epoch,
        num_epochs=num_epochs,
        max_steps=max_steps,
    )

    for i, batches in enumerate(scheduler):
        val = is_ckpt_step.pop(0)
        assert val == scheduler.is_ckpt_step, i
        assert val == scheduler.is_last_batch, i
        if max_steps is None:
            assert val == scheduler.is_last_step, i
    assert len(is_ckpt_step) == 0

@pytest.mark.parametrize(
    "num_epochs, epoch_len, expected_max_steps",
    [
        (1, 10, 10),
        (1, None, 9223372036854775807),
        (2, 10, 20),
    ],
)
def test_calculate_max_steps(num_epochs, epoch_len, expected_max_steps):
    assert _calculate_max_steps(num_epochs, epoch_len) == expected_max_steps
