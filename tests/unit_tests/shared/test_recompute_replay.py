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
import torch
from torch.utils.checkpoint import CheckpointPolicy, checkpoint

from nemo_automodel.shared.recompute_replay import (
    RecomputeReplay,
    RecomputeReplayRecorder,
    ReenterableContext,
    replayable_selective_checkpoint_contexts,
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


def test_retained_graph_replays_the_original_record_on_every_backward() -> None:
    channel: RecomputeReplay[float] = RecomputeReplay("test")
    modes: list[str] = []

    def block(x: torch.Tensor) -> torch.Tensor:
        bound = channel.current()
        assert bound is not None
        recorder, mode = bound
        modes.append(mode)
        if mode == "record":
            recorder.record(3.0)
            return x * x * 3.0
        taken = recorder.take()
        assert taken == 3.0
        return x * x * taken

    x = torch.arange(1.0, 4.0, requires_grad=True)
    y = checkpoint(block, x, use_reentrant=False, context_fn=channel.checkpoint_context_fn(None))
    for retain in (True, True, False):
        (grad,) = torch.autograd.grad(y.sum(), x, retain_graph=retain)
        torch.testing.assert_close(grad, 6.0 * x.detach())
    # One recorded forward, then a replay that took the record on each of the three backward passes.
    assert modes == ["record", "replay", "replay", "replay"]


def test_reenterable_context_builds_a_fresh_inner_context_per_entry() -> None:
    entries: list[int] = []

    @contextmanager
    def once():
        entries.append(len(entries))
        yield

    context = ReenterableContext(once)
    for _ in range(3):
        with context:
            pass
    assert entries == [0, 1, 2]


def test_nested_contexts_survive_a_split_input_and_weight_backward() -> None:
    """A split backward (input pass, then weight pass) recomputes the region twice through every context."""
    channel: RecomputeReplay[float] = RecomputeReplay("test")
    recomputes: list[str] = []
    weight = torch.nn.Parameter(torch.tensor([0.5, -1.0, 2.0]))

    def policy(ctx, func, *args, **kwargs):
        return CheckpointPolicy.MUST_SAVE if func == torch.ops.aten.sin.default else CheckpointPolicy.PREFER_RECOMPUTE

    def inner_context_fn():
        forward_context, recompute_context = replayable_selective_checkpoint_contexts(policy)

        @contextmanager
        def recompute():
            recomputes.append("enter")
            with recompute_context:
                yield

        return forward_context, ReenterableContext(recompute)

    def block(x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x) * weight * x

    x = torch.arange(1.0, 4.0, requires_grad=True)
    y = checkpoint(block, x, use_reentrant=False, context_fn=channel.checkpoint_context_fn(inner_context_fn))
    (dx,) = torch.autograd.grad(y.sum(), x, retain_graph=True)
    (dw,) = torch.autograd.grad(y.sum(), weight)

    xd = x.detach()
    torch.testing.assert_close(dx, weight.detach() * (torch.cos(xd) * xd + torch.sin(xd)))
    torch.testing.assert_close(dw, torch.sin(xd) * xd)
    assert recomputes == ["enter", "enter"]
