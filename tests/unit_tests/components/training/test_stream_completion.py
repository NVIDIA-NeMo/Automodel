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

from nemo_automodel.components.training.step_scheduler import StepSchedulerConfig


class _Stream:
    def __init__(self, count: int, offset: int = 0):
        self.values = iter(range(1 + offset, count + 1 + offset))
        self.read = 0

    def __iter__(self):
        return self

    def __next__(self):
        value = next(self.values)
        self.read += 1
        return {"value": value}


def _scheduler(source, *, accumulation: int = 2, max_steps: int = 20, dp_size: int = 1):
    return StepSchedulerConfig(
        global_batch_size=accumulation * dp_size,
        max_steps=max_steps,
        preemption_signal=None,
    ).build(source, dp_size, 1, synchronized_stream=True)


@pytest.mark.parametrize("count", [4, 5])
def test_finite_stream_completes_only_full_steps_and_does_not_restart(count):
    source = _Stream(count)
    scheduler = _scheduler(source)
    observed = []
    inside_states = []
    epochs = []
    for epoch in scheduler.epochs:
        epochs.append(epoch)
        scheduler.set_epoch(epoch)
        for batches in scheduler:
            observed.append([batch["value"] for batch in batches])
            inside_states.append(scheduler.state_dict()["step"])
    assert observed == [[1, 2], [3, 4]]
    assert inside_states == [1, 2]
    assert epochs == [0]
    assert scheduler.stream_exhausted
    assert scheduler.is_last_step
    assert scheduler.state_dict() == {"step": 2, "epoch": 1, "stream_exhausted": True}
    assert list(scheduler) == []
    assert list(scheduler.epochs) == []
    assert source.read == count


def test_step_cap_does_not_read_the_next_optimizer_step():
    source = _Stream(20)
    scheduler = _scheduler(source, max_steps=2)
    assert scheduler.state_dict()["step"] == 0
    assert len(list(scheduler)) == 2
    assert source.read == 4
    assert not scheduler.stream_exhausted
    assert scheduler.state_dict()["step"] == 2


@pytest.mark.parametrize("count", [0, 1])
def test_stream_without_a_full_step_is_an_error(count):
    scheduler = _scheduler(_Stream(count))
    with pytest.raises(RuntimeError, match="before a complete optimizer step"):
        list(scheduler)
    assert scheduler.step == 0


def test_terminal_state_restore_does_not_consume_data():
    scheduler = _scheduler(_Stream(3))
    list(scheduler)
    state = scheduler.state_dict()
    fresh = _Stream(10)
    restored = _scheduler(fresh)
    restored.load_state_dict(state)
    assert list(restored.epochs) == []
    assert list(restored) == []
    assert fresh.read == 0
    assert restored.state_dict() == state


def test_old_checkpoint_restores_as_a_live_stream():
    scheduler = _scheduler(_Stream(2))
    scheduler.load_state_dict({"step": 4, "epoch": 0})
    assert len(list(scheduler)) == 1
    assert scheduler.state_dict()["step"] == 5


def test_config_infers_stream_capability_and_preserves_explicit_override():
    source = _Stream(2)
    source.synchronized_stream = True
    config = StepSchedulerConfig(global_batch_size=1, max_steps=3, preemption_signal=None)
    inferred = config.build(source, 1, 1)
    assert len(list(inferred)) == 2
    assert inferred.stream_exhausted
    assert inferred.state_dict()["stream_exhausted"] is True

    overridden = config.build(source, 1, 1, synchronized_stream=False)
    assert "stream_exhausted" not in overridden.state_dict()
    native = config.build(_Stream(2), 1, 1)
    assert "stream_exhausted" not in native.state_dict()

    # Direct construction retains its pre-existing explicit opt-in contract.
    from nemo_automodel.components.training.step_scheduler import StepScheduler

    direct = StepScheduler(1, 1, 1, source, max_steps=3, preemption_signal=None)
    assert "stream_exhausted" not in direct.state_dict()
