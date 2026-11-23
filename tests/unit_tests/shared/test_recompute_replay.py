# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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


import threading
from contextlib import contextmanager

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from nemo_automodel.shared.recompute_replay import (
    RecomputeReplay,
    RecomputeReplayRecorder,
    in_checkpoint_recompute,
    mark_checkpoint_recompute,
)


def test_recorder_replays_in_order_then_reports_misses() -> None:
    recorder: RecomputeReplayRecorder[str] = RecomputeReplayRecorder()
    recorder.record("first")
    recorder.record("second")
    assert len(recorder) == 2
    assert recorder.records == ("first", "second")
    assert recorder.take() == "first"
    assert recorder.take() == "second"
    assert recorder.take() is None
    assert recorder.replay_misses == 1
    recorder.rewind()
    assert recorder.take() == "first"


def test_scope_binds_and_restores_thread_state() -> None:
    channel: RecomputeReplay[str] = RecomputeReplay("test")
    assert channel.current() is None
    recorder: RecomputeReplayRecorder[str] = RecomputeReplayRecorder()
    with channel.scope(recorder, "record"):
        assert channel.current() == (recorder, "record")
        with channel.scope(recorder, "replay"):
            assert channel.current() == (recorder, "replay")
        assert channel.current() == (recorder, "record")
    assert channel.current() is None
    with channel.scope(None, "record"):
        assert channel.current() is None
    with pytest.raises(ValueError, match="Unsupported test replay mode"):
        with channel.scope(recorder, "rewind"):
            pass


def test_channels_are_independent() -> None:
    first: RecomputeReplay[int] = RecomputeReplay("first")
    second: RecomputeReplay[int] = RecomputeReplay("second")
    recorder: RecomputeReplayRecorder[int] = RecomputeReplayRecorder()
    with first.scope(recorder, "record"):
        assert first.current() == (recorder, "record")
        assert second.current() is None


class _HookedRecorder(RecomputeReplayRecorder[int]):
    """Recorder that logs the channel hooks into the enclosing event list."""

    events: list[str] = []

    def on_record_exit(self) -> None:
        self.events.append(f"record exit {len(self)}")

    def on_replay_enter(self) -> None:
        self.events.append("replay enter")


def test_checkpoint_context_fn_records_then_replays_per_call() -> None:
    channel: RecomputeReplay[int] = RecomputeReplay("test", recorder_factory=_HookedRecorder)
    events: list[str] = []
    _HookedRecorder.events = events

    @contextmanager
    def tracked(name: str):
        events.append(f"enter {name}")
        yield
        events.append(f"exit {name}")

    def context_fn():
        return tracked("forward"), tracked("recompute")

    wrapped = channel.checkpoint_context_fn(context_fn)

    def decide(value: int) -> int:
        binding = channel.current()
        assert binding is not None
        recorder, mode = binding
        if mode == "replay":
            cached = recorder.take()
            if cached is not None:
                return cached
        recorder.record(value)
        return value

    forward_ctx, recompute_ctx = wrapped()
    with forward_ctx:
        assert [decide(1), decide(2)] == [1, 2]
    # The record-exit hook runs after the forward context has exited.
    assert events == ["enter forward", "exit forward", "record exit 2"]
    with recompute_ctx:
        # Replay returns the forward decisions regardless of the recomputed value.
        assert [decide(10), decide(20)] == [1, 2]
        # A third call outruns the log and falls back to the fresh decision.
        assert decide(30) == 30
    # The replay-enter hook runs before the recompute context opens.
    assert events == [
        "enter forward",
        "exit forward",
        "record exit 2",
        "replay enter",
        "enter recompute",
        "exit recompute",
    ]
    assert channel.current() is None

    # A second checkpoint call gets its own recorder.
    forward_ctx, recompute_ctx = wrapped()
    with forward_ctx:
        assert decide(7) == 7
    with recompute_ctx:
        assert decide(0) == 7


def test_checkpoint_context_fn_without_inner_contexts() -> None:
    channel: RecomputeReplay[int] = RecomputeReplay("test")
    forward_ctx, recompute_ctx = channel.checkpoint_context_fn(None)()
    with forward_ctx:
        assert channel.current() is not None and channel.current()[1] == "record"
    with recompute_ctx:
        assert channel.current() is not None and channel.current()[1] == "replay"


def test_mark_checkpoint_recompute_marks_only_the_recompute_context() -> None:
    events: list[str] = []

    @contextmanager
    def tracked(name: str):
        events.append(f"enter {name}")
        yield
        events.append(f"exit {name}")

    wrapped = mark_checkpoint_recompute(lambda: (tracked("forward"), tracked("recompute")))
    forward_ctx, recompute_ctx = wrapped()
    assert not in_checkpoint_recompute()
    with forward_ctx:
        assert not in_checkpoint_recompute()
    with recompute_ctx:
        assert in_checkpoint_recompute()
        # The mark belongs to the thread that runs the recompute.
        seen_in_other_thread: list[bool] = []
        worker = threading.Thread(target=lambda: seen_in_other_thread.append(in_checkpoint_recompute()))
        worker.start()
        worker.join()
        assert seen_in_other_thread == [False]
    assert not in_checkpoint_recompute()
    # The supplied contexts are entered unchanged, the recompute one inside the mark.
    assert events == ["enter forward", "exit forward", "enter recompute", "exit recompute"]


def test_mark_checkpoint_recompute_is_cleared_when_the_recompute_raises() -> None:
    _, recompute_ctx = mark_checkpoint_recompute(None)()
    with pytest.raises(RuntimeError, match="early stop"):
        with recompute_ctx:
            assert in_checkpoint_recompute()
            raise RuntimeError("early stop")
    assert not in_checkpoint_recompute()


def test_mark_checkpoint_recompute_nests() -> None:
    _, outer = mark_checkpoint_recompute(None)()
    _, inner = mark_checkpoint_recompute(None)()
    with outer:
        with inner:
            assert in_checkpoint_recompute()
        assert in_checkpoint_recompute()
    assert not in_checkpoint_recompute()


def test_mark_checkpoint_recompute_follows_a_non_reentrant_checkpoint() -> None:
    weight = torch.nn.Parameter(torch.tensor([2.0]))
    seen: list[bool] = []

    def body(x: torch.Tensor) -> torch.Tensor:
        seen.append(in_checkpoint_recompute())
        return (x * weight).sin()

    x = torch.tensor([1.0], requires_grad=True)
    y = checkpoint(body, x, use_reentrant=False, context_fn=mark_checkpoint_recompute(None))
    assert seen == [False]
    y.backward()
    # The backward replays the body once, inside the mark, and leaves no mark behind.
    assert seen == [False, True]
    assert not in_checkpoint_recompute()
    assert weight.grad is not None

    # An unmarked checkpoint replays outside the mark.
    seen.clear()
    checkpoint(body, x, use_reentrant=False).backward()
    assert seen == [False, False]
