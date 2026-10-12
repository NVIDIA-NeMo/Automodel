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


from contextlib import contextmanager

import pytest

from nemo_automodel.components.distributed.recompute_replay import RecomputeReplay, RecomputeReplayRecorder


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


def test_checkpoint_context_fn_records_then_replays_per_call() -> None:
    channel: RecomputeReplay[int] = RecomputeReplay("test")
    events: list[str] = []
    finalized: list[int] = []

    @contextmanager
    def tracked(name: str):
        events.append(f"enter {name}")
        yield
        events.append(f"exit {name}")

    def context_fn():
        return tracked("forward"), tracked("recompute")

    wrapped = channel.checkpoint_context_fn(context_fn, on_record_exit=lambda rec: finalized.append(len(rec)))

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
    assert finalized == [2]
    with recompute_ctx:
        # Replay returns the forward decisions regardless of the recomputed value.
        assert [decide(10), decide(20)] == [1, 2]
        # A third call outruns the log and falls back to the fresh decision.
        assert decide(30) == 30
    assert events == ["enter forward", "exit forward", "enter recompute", "exit recompute"]
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
