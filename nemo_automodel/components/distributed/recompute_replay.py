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


"""Record-and-replay of forward-pass decisions across activation-checkpoint recompute.

Activation checkpointing reruns a block during backward. Some of the block's work
is a deterministic decision that the recompute would reproduce bit for bit -- a
frozen indexer's top-k routes, an expert-parallel dispatch layout -- yet costs
real host or communication time to recompute. A :class:`RecomputeReplay` channel
lets the checkpoint forward record such decisions and the recompute take them
back in the same order, falling back to recomputing when the log runs out.

Each checkpointed call gets its own :class:`RecomputeReplayRecorder` from
:meth:`RecomputeReplay.checkpoint_context_fn`, so concurrent microbatches and
repeated calls of one block never share a log. The binding is thread-local so
pipeline-parallel worker threads stay independent.

This is distinct from :mod:`nemo_automodel.components.moe.router_replay`, which
replays MoE top-k across two separate forward passes for RL (R3) through a
process-global, per-gate registry.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import Generic, TypeVar

__all__ = ["RecomputeReplay", "RecomputeReplayRecorder"]

T = TypeVar("T")

CheckpointContextFn = Callable[[], tuple[AbstractContextManager, AbstractContextManager]]


class RecomputeReplayRecorder(Generic[T]):
    """Ordered log of one checkpointed call's forward decisions, consumed on recompute."""

    def __init__(self) -> None:
        self._records: list[T] = []
        self._cursor = 0
        self.replay_misses = 0

    def record(self, entry: T) -> None:
        """Append one forward decision."""
        self._records.append(entry)

    def take(self) -> T | None:
        """Return the next recorded decision, or ``None`` when replay outruns the log."""
        if self._cursor >= len(self._records):
            self.replay_misses += 1
            return None
        entry = self._records[self._cursor]
        self._cursor += 1
        return entry

    def rewind(self) -> None:
        """Restart replay from the first record."""
        self._cursor = 0

    @property
    def records(self) -> tuple[T, ...]:
        """The recorded decisions in forward order."""
        return tuple(self._records)

    def __len__(self) -> int:
        return len(self._records)


class _Binding(threading.local):
    """Thread-local ``(recorder, mode)`` binding; ``__init__`` runs once per thread."""

    def __init__(self) -> None:
        self.recorder: RecomputeReplayRecorder | None = None
        self.mode: str | None = None


class RecomputeReplay(Generic[T]):
    """One named record/replay channel with its own thread-local binding.

    Consumers create a module-level channel and consult :meth:`current` where the
    decision is made: record the fresh decision under ``"record"``, take a stored one
    under ``"replay"``. :meth:`checkpoint_context_fn` binds the channel around a
    ``torch.utils.checkpoint`` context factory.

    Args:
        name: Channel name used in error messages.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._binding = _Binding()

    @contextmanager
    def scope(self, recorder: RecomputeReplayRecorder[T] | None, mode: str) -> Iterator[None]:
        """Bind ``recorder`` in ``mode`` (``"record"`` or ``"replay"``) for the enclosed region.

        Args:
            recorder: Log shared by the forward and recompute of one checkpointed call;
                ``None`` disables replay inside the scope.
            mode: ``"record"`` or ``"replay"``.
        """
        if mode not in ("record", "replay"):
            raise ValueError(f"Unsupported {self.name} replay mode: {mode!r}")
        previous_recorder, previous_mode = self._binding.recorder, self._binding.mode
        self._binding.recorder = recorder
        self._binding.mode = mode if recorder is not None else None
        try:
            yield
        finally:
            self._binding.recorder = previous_recorder
            self._binding.mode = previous_mode

    def current(self) -> tuple[RecomputeReplayRecorder[T], str] | None:
        """Return the active ``(recorder, mode)``, or ``None`` outside a scope."""
        recorder, mode = self._binding.recorder, self._binding.mode
        if recorder is None or mode is None:
            return None
        return recorder, mode

    def checkpoint_context_fn(
        self,
        context_fn: CheckpointContextFn | None,
        *,
        on_record_exit: Callable[[RecomputeReplayRecorder[T]], None] | None = None,
    ) -> CheckpointContextFn:
        """Wrap a checkpoint ``context_fn`` so this channel records in forward and replays in recompute.

        Args:
            context_fn: Existing ``torch.utils.checkpoint`` context factory returning the
                ``(forward, recompute)`` context managers, or ``None`` for no other contexts.
            on_record_exit: Optional hook run on the recorder after the forward context
                exits, for work that must stay out of the checkpointed op trace.

        Returns:
            A context factory that binds a fresh recorder per checkpoint call.
        """

        def wrapped_context_fn() -> tuple[AbstractContextManager, AbstractContextManager]:
            forward_context, recompute_context = (
                context_fn() if context_fn is not None else (nullcontext(), nullcontext())
            )
            recorder: RecomputeReplayRecorder[T] = RecomputeReplayRecorder()

            @contextmanager
            def scoped(mode: str, inner: AbstractContextManager) -> Iterator[None]:
                if mode == "replay":
                    recorder.rewind()
                with self.scope(recorder, mode):
                    with inner:
                        yield
                    if mode == "record" and on_record_exit is not None:
                        on_record_exit(recorder)

            return scoped("record", forward_context), scoped("replay", recompute_context)

        return wrapped_context_fn
