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
import os
import signal as _signal

from nemo_automodel.components.training.step_scheduler import (
    StepScheduler,
    StepSchedulerConfig,
    _calculate_max_steps,
)


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
        return self.num_batches  # * (self.global_batch_size // self.local_batch_size)


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
    for i, _ in enumerate(scheduler):
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
    for j, _ in enumerate(scheduler):
        expected_step, expected_is_val, expected_is_ckpt = ref_outputs.pop(0)
        # Ensure we don't checkpoint immediately after resume if we saved on a checkpoint step
        if j == 0 and saved_is_ckpt and ckpt_every_steps > 1:
            expected_is_ckpt = False
        assert (expected_step, expected_is_val, expected_is_ckpt) == (
            scheduler.step,
            scheduler.is_val_step,
            scheduler.is_ckpt_step,
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
        (
            10,
            4,
            4,
            2,
            [False, False, False, True, False, False, False, True, False, True],
        ),
    ],
)
def test_is_ckpt_step_parametrized_iterable(
    max_steps, ckpt_every_steps, global_batch_size, local_batch_size, is_ckpt_step
):
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
    [([3, 7, 9])],
)
def test_is_ckpt_step_triggers_on_last_batch_with_sized_dataloader(
    expected_last_batch_steps,
):
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
    assert scheduler.state_dict() == {"step": 10, "epoch": 2}


@pytest.mark.parametrize(
    "max_steps, ckpt_every_steps, epoch, num_epochs, global_batch_size, local_batch_size, num_batches, is_ckpt_step",
    [
        (None, 1000, 0, 1, 64, 1, 317 + 1, [False] * 317 + [True]),
        (1000, 1000, 0, 1, 64, 1, 317 + 1, [False] * 317 + [True]),
    ],
)
def test_ckpt_every_steps_larger_than_max_steps(
    max_steps,
    ckpt_every_steps,
    epoch,
    num_epochs,
    global_batch_size,
    local_batch_size,
    num_batches,
    is_ckpt_step,
):
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
        # is_last_step must fire on the final batch regardless of whether the run
        # ends by hitting max_steps or by exhausting epochs (here max_steps=1000
        # is never reached; the single epoch runs out first).
        assert val == scheduler.is_last_step, i
    assert len(is_ckpt_step) == 0


def test_is_last_step_when_epochs_exhausted_before_max_steps():
    """Regression: epochs can run out before max_steps with a small dataset.

    Mirrors the diffusion finetune CI setup that regressed (e.g. hunyuan_t2v_flow):
    max_steps=100 but only num_epochs * epoch_len = 30 * 2 = 60 steps of data
    exist, with periodic (ckpt_every_steps=100) and epoch-boundary
    (save_checkpoint_every_epoch=False) checkpointing both disabled. The final
    step must still be flagged so the final/consolidated checkpoint is written;
    otherwise no checkpoint is ever saved.
    """
    num_epochs = 30
    epoch_len = 2
    max_steps = 100  # never reached: only num_epochs * epoch_len = 60 steps of data

    scheduler = StepScheduler(
        global_batch_size=1,
        local_batch_size=1,
        dp_size=1,
        ckpt_every_steps=max_steps,  # periodic save never fires (60 < 100)
        save_checkpoint_every_epoch=False,
        dataloader=SizedDataLoader(num_batches=epoch_len),
        num_epochs=num_epochs,
        max_steps=max_steps,
    )

    last_step_flags = []
    ckpt_step_flags = []
    for _epoch in scheduler.epochs:
        for _batches in scheduler:
            last_step_flags.append(scheduler.is_last_step)
            ckpt_step_flags.append(scheduler.is_ckpt_step)

    # All 60 steps ran (max_steps was never the limiter).
    assert len(last_step_flags) == num_epochs * epoch_len
    # is_last_step -- and therefore is_ckpt_step -- fire exactly once, on the final step.
    assert sum(last_step_flags) == 1
    assert last_step_flags[-1] is True
    assert sum(ckpt_step_flags) == 1
    assert ckpt_step_flags[-1] is True


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


@pytest.mark.parametrize(
    "dataloader, is_iterable",
    [
        (SizedDataLoader(num_batches=10), False),
        (IterableDataLoader(num_batches=10), True),
    ],
)
def test_ckpt_every_steps_is_none(dataloader, is_iterable):
    scheduler = StepScheduler(
        global_batch_size=1,
        local_batch_size=1,
        dp_size=1,
        ckpt_every_steps=None,
        dataloader=dataloader,
        num_epochs=1,
        max_steps=10,
    )
    if is_iterable:
        assert scheduler.epoch_len is None
        assert scheduler.ckpt_every_steps is 10 // 2
    else:
        assert scheduler.epoch_len is 10
        assert scheduler.ckpt_every_steps is 10


def test_iterable_dataloader():
    dataloader = IterableDataLoader(num_batches=10)
    StepScheduler(
        global_batch_size=1,
        local_batch_size=1,
        dp_size=1,
        ckpt_every_steps=1000,
        dataloader=dataloader,
        num_epochs=1,
        max_steps=10,
    )


def test_is_gc_step_disabled_by_default():
    dataloader = SizedDataLoader(num_batches=5)
    scheduler = StepScheduler(
        global_batch_size=1,
        local_batch_size=1,
        dp_size=1,
        dataloader=dataloader,
        num_epochs=1,
        max_steps=5,
    )
    assert scheduler.is_gc_step is False


def test_is_gc_step_every_n_steps():
    dataloader = SizedDataLoader(num_batches=7)
    scheduler = StepScheduler(
        global_batch_size=1,
        local_batch_size=1,
        dp_size=1,
        dataloader=dataloader,
        num_epochs=1,
        max_steps=7,
        gc_every_steps=3,
    )

    observed = []
    for _ in scheduler:
        observed.append(scheduler.is_gc_step)

    assert observed == [True, False, False, True, False, False, True]


def test_gc_every_steps_must_be_positive():
    dataloader = SizedDataLoader(num_batches=1)
    with pytest.raises(ValueError, match="gc_every_steps must be greater than 0 if not None"):
        StepScheduler(
            global_batch_size=1,
            local_batch_size=1,
            dp_size=1,
            dataloader=dataloader,
            num_epochs=1,
            max_steps=1,
            gc_every_steps=0,
        )


def test_set_epoch():
    dataloader = SizedDataLoader(num_batches=10)
    scheduler = StepScheduler(
        global_batch_size=1,
        local_batch_size=1,
        dp_size=1,
        ckpt_every_steps=1000,
        dataloader=dataloader,
        num_epochs=1,
        max_steps=10,
    )
    scheduler.set_epoch(2)
    assert scheduler.epoch == 2


def test_scheduler_max_steps_priority():
    dataloader = SizedDataLoader(num_batches=20)
    scheduler = StepScheduler(
        global_batch_size=1,
        local_batch_size=1,
        dp_size=1,
        ckpt_every_steps=10,
        dataloader=dataloader,
        max_steps=1000,
    )

    assert scheduler.num_epochs != 10, "Scheduler defaulted to 10 epochs despite max_steps being set!"

    scheduler_default = StepScheduler(
        global_batch_size=1,
        local_batch_size=1,
        dp_size=1,
        ckpt_every_steps=10,
        dataloader=dataloader,
    )

    assert scheduler_default.num_epochs == 10, "Should default to 10 epochs if nothing is provided"


def test_scheduler_num_epochs_derived_from_max_steps():
    dataloader = SizedDataLoader(num_batches=10)
    scheduler = StepScheduler(
        global_batch_size=1,
        local_batch_size=1,
        dp_size=1,
        ckpt_every_steps=10,
        dataloader=dataloader,
        max_steps=15,
    )

    print(scheduler.num_epochs)
    assert scheduler.num_epochs == 2, "Derived number of epochs should be 2"


def test_save_checkpoint_every_epoch_false_disables_epoch_boundary_checkpoints():
    """When save_checkpoint_every_epoch=False, epoch-end checkpoints don't trigger,
    but last-step checkpoints still do."""
    epoch_len = 4
    dataloader = SizedDataLoader(num_batches=epoch_len)
    scheduler = StepScheduler(
        global_batch_size=1,
        local_batch_size=1,
        dp_size=1,
        ckpt_every_steps=1000,  # disable periodic checkpointing
        save_checkpoint_every_epoch=False,
        dataloader=dataloader,
        num_epochs=100,
        max_steps=10,
    )

    ckpt_trigger_steps = []
    for _ in scheduler.epochs:
        for _ in scheduler:
            if scheduler.is_ckpt_step:
                ckpt_trigger_steps.append(scheduler.step)

    # Only the last step should trigger (step 9, is_last_step=True)
    # Epoch boundaries at steps 3 and 7 should NOT trigger
    assert ckpt_trigger_steps == [9]

    assert scheduler.step == 10
    assert scheduler.is_ckpt_step is True


def test_save_checkpoint_every_epoch_false_preserves_periodic_checkpoints():
    """Periodic checkpoints still fire when save_checkpoint_every_epoch=False."""
    epoch_len = 4
    dataloader = SizedDataLoader(num_batches=epoch_len)
    scheduler = StepScheduler(
        global_batch_size=1,
        local_batch_size=1,
        dp_size=1,
        ckpt_every_steps=5,
        save_checkpoint_every_epoch=False,
        dataloader=dataloader,
        num_epochs=100,
        max_steps=10,
    )

    ckpt_trigger_steps = []
    for _ in scheduler.epochs:
        for _ in scheduler:
            if scheduler.is_ckpt_step:
                ckpt_trigger_steps.append(scheduler.step)

    # Periodic at step 4 (step%5==4), last step at 9
    # Epoch boundaries at 3 and 7 should NOT trigger
    assert ckpt_trigger_steps == [4, 9]


# ---------------------------------------------------------------------------
# Preemption checkpointing
# ---------------------------------------------------------------------------


def _make_scheduler(**overrides):
    kwargs = dict(
        global_batch_size=2,
        local_batch_size=1,
        dp_size=2,
        dataloader=SizedDataLoader(num_batches=8),
        ckpt_every_steps=1000,
        num_epochs=1,
    )
    kwargs.update(overrides)
    return StepScheduler(**kwargs)


def test_preemption_disable_installs_no_handler():
    """preemption_signal=None must not touch signal handlers and never report sigterm."""
    original = _signal.getsignal(_signal.SIGTERM)
    scheduler = _make_scheduler(preemption_signal=None)
    assert scheduler.sig_handler is None
    assert _signal.getsignal(_signal.SIGTERM) is original
    assert scheduler.sigterm_received is False
    assert scheduler.is_ckpt_step is False


def test_preemption_invalid_signal_name_raises():
    with pytest.raises(ValueError):
        _make_scheduler(preemption_signal="SIGTYPO")


def test_preemption_signal_triggers_checkpoint_and_stops_iteration():
    """
    Delivering the configured signal mid-training must make is_ckpt_step
    True at the nest step boundary, stop iteration, and leave a resumable state_dict.
    """
    scheduler = _make_scheduler(preemption_signal="SIGUSR1")
    try:
        steps_run = 0
        for _batches in scheduler:
            steps_run += 1
            if steps_run == 2:
                os.kill(os.getpid(), _signal.SIGUSR1)
                # Recipes consult is_ckpt_step once per step; mirrors train loop.
            if scheduler.is_ckpt_step:
                break_step = scheduler.step
                assert scheduler.sigterm_flag is True
                break
        else:
            pytest.fail("is_ckpt_step never become True after signal delivery")
        assert steps_run == 2
        # state_dict stores step + 1 so the resumed run continues after the
        # preemption checkpoint instead of repeating the step.
        assert scheduler.state_dict()["step"] == break_step + 1
    finally:
        scheduler.sig_handler.release()


def test_preemption_iteration_stops_without_explicit_break():
    """__iter__ must terminate on its own once the flag is observed."""
    scheduler = _make_scheduler(preemption_signal="SIGUSR1")
    try:
        seen = []
        for _batches in scheduler:
            seen.append(scheduler.step)
            os.kill(os.getpid(), _signal.SIGUSR1)
            assert scheduler.is_ckpt_step is True
        assert seen == [0]
    finally:
        scheduler.sig_handler.release()


def test_preemption_poll_is_cached_once_per_step():
    """
    sigterm_received must issue at most one cross-rank poll per step, no matter
    how many properties consult it within the step.
    """
    scheduler = _make_scheduler(preemption_signal="SIGUSR1")
    try:
        calls = {"n": 0}
        real = scheduler.sig_handler.signals_received

        def counting():
            calls["n"] += 1
            return real()

        scheduler.sig_handler.signals_received = counting

        # Multiple accesses at the same step -> exactly one poll.
        _ = scheduler.is_val_step
        _ = scheduler.is_ckpt_step
        _ = scheduler.sigterm_received
        assert calls["n"] == 1
    finally:
        scheduler.sig_handler.release()


def test_preemption_flag_is_sticky_and_stops_polling():
    """once set, the flag stays True with no further collective calls."""
    scheduler = _make_scheduler(preemption_signal="SIGUSR1")
    try:
        os.kill(os.getpid(), _signal.SIGUSR1)
        assert scheduler.sigterm_received is True

        calls = {"n": 0}

        def counting():
            calls["n"] += 1
            return [True]

        scheduler.sig_handler.signals_received = counting
        scheduler.step += 1
        assert scheduler.sigterm_received is True
        assert calls["n"] == 0
    finally:
        scheduler.sig_handler.release()


def test_step_scheduler_config_exposes_preemption_signal():
    """The tupes config must plub preemption_signal through build()."""
    cfg = StepSchedulerConfig(
        global_batch_size=2,
        num_epochs=1,
        preemption_signal=None,
    )
    scheduler = cfg.build(SizedDataLoader(num_batches=4), dp_group_size=2, local_batch_size=1)
    assert scheduler.sig_handler is None

    cfg = StepSchedulerConfig(
        global_batch_size=2,
        num_epochs=1,
        preemption_signal="SIGUSR2",
    )
    scheduler = cfg.build(SizedDataLoader(num_batches=4), dp_group_size=2, local_batch_size=1)
    try:
        assert scheduler.sig_handler is not None
        assert scheduler.sig_handler.sigs == [_signal.SIGUSR2]
    finally:
        scheduler.sig_handler.release()
