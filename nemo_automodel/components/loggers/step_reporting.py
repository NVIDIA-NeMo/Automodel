# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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


"""CUDA step timing and deferred host metrics for language-model recipes."""

from collections import defaultdict, deque
from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter

import torch
import torch.distributed as dist

from nemo_automodel.components.loggers.metric_logger import MetricsSample

StepMarker = torch.cuda.Event | float

# A poll finishes the oldest records beyond this many, waiting if it must, so a slow
# timing reduction stalls training instead of retaining every later step's snapshot.
MAX_OUTSTANDING_REPORTS = 8


@dataclass
class _MetricReadback:
    keys: list[str]
    shapes: list[torch.Size]
    values: torch.Tensor


@dataclass
class _PendingStep:
    sample: MetricsSample
    spans: dict[str, tuple[StepMarker, StepMarker]]
    ready: torch.cuda.Event | None
    remote_log: bool
    readbacks: list[_MetricReadback]


class DeferredStepReporter:
    """Read completed step metrics and aggregate rank times on CPU."""

    def __init__(self, device: torch.device, *, process_group: dist.ProcessGroup | None = None) -> None:
        """Set the training device and create a CPU process group for the reporting ranks.

        Args:
            device: Training device assigned to the calling rank.
            process_group: Ranks that run the training step together and share its time, such as
                the students of separate-mesh distillation. None means every rank. The reporter
                owns a Gloo group with the same members and destroys it in ``close``.
        """
        self._device = device
        self._group = _new_gloo_group(process_group)
        self._pending: deque[_PendingStep] = deque()
        self._reducing: deque[tuple[_PendingStep, torch.Tensor, dist.Work | None]] = deque()

    def mark(self) -> StepMarker:
        """Record a CUDA event or a CPU timestamp on the training device."""
        if self._device.type != "cuda":
            return perf_counter()
        event = torch.cuda.Event(enable_timing=True)
        event.record(torch.cuda.current_stream(self._device))
        return event

    @torch.no_grad()
    def enqueue(
        self,
        sample: MetricsSample,
        *,
        start: StepMarker,
        end: StepMarker,
        remote_log: bool,
        phases: Mapping[str, tuple[StepMarker, StepMarker]] | None = None,
    ) -> None:
        """Queue a detached metrics snapshot for the step identified by sample.

        Args:
            sample: Step metadata and metrics. CUDA values are scalars or vectors
                such as expert counts. Their storage may be reused after enqueue.
                CUDA results replace the original tensor entries on completion.
            start: Timing event recorded before step work began.
            end: Timing event ordered after the step's compute and communication.
            remote_log: Whether the step is scheduled for remote logging.
            phases: Additional named start and end markers from this step.
        """
        by_dtype: dict[torch.dtype, list[tuple[str, torch.Tensor]]] = defaultdict(list)
        for key, value in sample.metrics.items():
            if not isinstance(value, torch.Tensor):
                continue
            if value.is_cuda:
                by_dtype[value.dtype].append((key, value.detach()))
            else:
                sample.metrics[key] = value.item() if value.ndim == 0 else value.detach().clone()

        readbacks = []
        ready = end if isinstance(end, torch.cuda.Event) else None
        if by_dtype and self._device.type == "cuda":
            for entries in by_dtype.values():
                values = torch.cat([value.reshape(-1) for _, value in entries])
                # PyTorch allocates a pinned CPU destination for this copy.
                host = values.to("cpu", non_blocking=True)
                readbacks.append(
                    _MetricReadback([key for key, _ in entries], [value.shape for _, value in entries], host)
                )
            ready = torch.cuda.current_stream(self._device).record_event()
        self._pending.append(_PendingStep(sample, {"": (start, end), **(phases or {})}, ready, remote_log, readbacks))

    def poll(self, *, wait: bool = False) -> list[tuple[MetricsSample, bool, float]]:
        """Return completed samples in order, with their logging flag and max rank seconds.

        Args:
            wait: Drain pending records, including incomplete reads. Use at
                lifecycle boundaries that already need the exact completed sample.
                Without it, only records beyond ``MAX_OUTSTANDING_REPORTS`` are waited for.

        Returns:
            Completed samples, their remote logging flags, and max rank durations.
        """
        # The oldest records past the cap finish now. Records in _reducing are older than
        # those in _pending, and both deques are in step order.
        must_finish = len(self._reducing) + len(self._pending) - MAX_OUTSTANDING_REPORTS
        completed = []
        while self._pending:
            record = self._pending[0]
            if record.ready is not None and not record.ready.query():
                if not wait and len(self._reducing) >= must_finish:
                    break
                record.ready.synchronize()
            elapsed = torch.tensor(
                [_elapsed_seconds(start, end) for start, end in record.spans.values()],
                dtype=torch.float64,
                device="cpu",
            )
            work = (
                dist.all_reduce(elapsed, op=dist.ReduceOp.MAX, group=self._group, async_op=True)
                if self._group is not None
                else None
            )
            self._reducing.append((self._pending.popleft(), elapsed, work))

        while self._reducing:
            record, elapsed, work = self._reducing[0]
            if work is not None:
                if not wait and len(completed) >= must_finish and not work.is_completed():
                    break
                work.wait()

            for readback in record.readbacks:
                chunks = readback.values.split([shape.numel() for shape in readback.shapes])
                for key, shape, chunk in zip(readback.keys, readback.shapes, chunks, strict=True):
                    record.sample.metrics[key] = chunk.view(shape).clone() if shape else chunk.item()
            step_elapsed, *phases = elapsed.tolist()
            for name, seconds in zip(tuple(record.spans)[1:], phases, strict=True):
                record.sample.metrics[f"{name}-time"] = seconds
            completed.append((record.sample, record.remote_log, step_elapsed))
            self._reducing.popleft()
        return completed

    def close(self) -> None:
        """Release the reporting process group after the caller drains pending samples.

        Samples still pending are dropped. After a rank-local failure the timing reduction
        cannot finish without the other ranks, so close does not wait for it.
        """
        if self._group is not None:
            dist.destroy_process_group(self._group)


def _new_gloo_group(process_group: dist.ProcessGroup | None) -> dist.ProcessGroup | None:
    """Create a Gloo group with the members of ``process_group``, or None for a single reporting rank."""
    if not dist.is_initialized():
        return None
    if process_group is None:
        return dist.new_group(backend="gloo") if dist.get_world_size() > 1 else None
    ranks = dist.get_process_group_ranks(process_group)
    if len(ranks) == 1:
        return None
    # Only the members create the group, so ranks outside it (separate-mesh distillation
    # teachers, for example) never wait in this call.
    return dist.new_group(ranks=ranks, backend="gloo", use_local_synchronization=True)


def _elapsed_seconds(start: StepMarker, end: StepMarker) -> float:
    if isinstance(start, torch.cuda.Event) and isinstance(end, torch.cuda.Event):
        return start.elapsed_time(end) / 1000.0
    if isinstance(start, torch.cuda.Event) or isinstance(end, torch.cuda.Event):
        raise TypeError("A span's start and end markers must both be CUDA events or both be CPU timestamps")
    return end - start
