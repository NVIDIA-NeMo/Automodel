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
from dataclasses import dataclass, field

import torch
import torch.distributed as dist

from nemo_automodel.components.loggers.metric_logger import MetricsSample


@dataclass
class _MetricReadback:
    keys: list[str]
    shapes: list[torch.Size]
    values: torch.Tensor


@dataclass
class _PendingStep:
    sample: MetricsSample
    start: torch.cuda.Event
    end: torch.cuda.Event
    ready: torch.cuda.Event
    remote_log: bool
    readbacks: list[_MetricReadback]
    elapsed: torch.Tensor = field(default_factory=lambda: torch.empty((), dtype=torch.float64))
    timing_work: dist.Work | None = None


class DeferredStepReporter:
    """Read CUDA metrics after completion and aggregate rank times on CPU."""

    def __init__(self, device: torch.device) -> None:
        """Create a reporting stream and a CPU process group in rank order.

        Args:
            device: CUDA device assigned to the calling rank.
        """
        if device.type != "cuda":
            raise ValueError("defer_step_metrics requires a CUDA training device")
        self._device = device
        self._stream = torch.cuda.Stream(device=device)
        self._group = dist.new_group(backend="gloo") if dist.is_initialized() and dist.get_world_size() > 1 else None
        self._pending: deque[_PendingStep] = deque()
        self._reducing: deque[_PendingStep] = deque()
        self._buffers: dict[tuple[torch.dtype, int], list[torch.Tensor]] = defaultdict(list)

    def mark(self) -> torch.cuda.Event:
        """Record a timing marker on the current stream of the training device."""
        event = torch.cuda.Event(enable_timing=True)
        event.record(torch.cuda.current_stream(self._device))
        return event

    @torch.no_grad()
    def enqueue(
        self,
        sample: MetricsSample,
        *,
        start: torch.cuda.Event,
        end: torch.cuda.Event,
        remote_log: bool,
    ) -> None:
        """Queue a detached metrics snapshot for the step identified by sample.

        Args:
            sample: Step metadata and metrics. CUDA values are scalars or vectors
                such as expert counts. Their storage may be reused after enqueue.
                Resolved values replace the original tensor entries on completion.
            start: Timing event recorded before step work began.
            end: Timing event ordered after the step's compute and communication.
            remote_log: Whether the step is scheduled for remote logging.
        """
        by_dtype: dict[torch.dtype, list[tuple[str, torch.Tensor]]] = defaultdict(list)
        for key, value in sample.metrics.items():
            if isinstance(value, torch.Tensor):
                if value.is_cuda:
                    by_dtype[value.dtype].append((key, value.detach()))
                else:
                    sample.metrics[key] = value.detach().clone()

        packed = [(entries, torch.cat([value.reshape(-1) for _, value in entries])) for entries in by_dtype.values()]
        produced = torch.cuda.Event()
        produced.record(torch.cuda.current_stream(self._device))
        ready = torch.cuda.Event()
        readbacks = []
        with torch.cuda.stream(self._stream):
            self._stream.wait_event(produced)
            for entries, values in packed:
                available = self._buffers[(values.dtype, values.numel())]
                host = (
                    available.pop()
                    if available
                    else torch.empty(values.numel(), dtype=values.dtype, device="cpu", pin_memory=True)
                )
                host.copy_(values, non_blocking=True)
                values.record_stream(self._stream)
                readbacks.append(
                    _MetricReadback([key for key, _ in entries], [value.shape for _, value in entries], host)
                )
            ready.record(self._stream)
        self._pending.append(_PendingStep(sample, start, end, ready, remote_log, readbacks))

    def poll(self, *, wait: bool = False) -> list[tuple[MetricsSample, bool, float]]:
        """Return completed samples in order, with their logging flag and max rank seconds.

        Args:
            wait: Drain pending records, including incomplete reads. Use at
                lifecycle boundaries that already need the exact completed sample.

        Returns:
            Completed samples, their remote logging flags, and max rank durations.
        """
        completed = []
        while self._pending:
            record = self._pending[0]
            if not record.ready.query():
                if not wait:
                    break
                record.ready.synchronize()
            record.elapsed.fill_(record.start.elapsed_time(record.end) / 1000.0)
            if self._group is not None:
                record.timing_work = dist.all_reduce(
                    record.elapsed, op=dist.ReduceOp.MAX, group=self._group, async_op=True
                )
            self._reducing.append(self._pending.popleft())

        while self._reducing:
            record = self._reducing[0]
            if record.timing_work is not None:
                if not wait and not record.timing_work.is_completed():
                    break
                record.timing_work.wait()

            for readback in record.readbacks:
                offset = 0
                for key, shape in zip(readback.keys, readback.shapes):
                    value = readback.values.narrow(0, offset, shape.numel()).view(shape)
                    record.sample.metrics[key] = value.item() if not shape else value.clone()
                    offset += shape.numel()
                self._buffers[(readback.values.dtype, readback.values.numel())].append(readback.values)
            for key, value in record.sample.metrics.items():
                if isinstance(value, torch.Tensor) and value.ndim == 0:
                    record.sample.metrics[key] = value.item()
            completed.append((record.sample, record.remote_log, record.elapsed.item()))
            self._reducing.popleft()
        return completed

    def close(self) -> None:
        """Release the reporting process group after the caller drains pending samples."""
        if self._group is not None:
            dist.destroy_process_group(self._group)
