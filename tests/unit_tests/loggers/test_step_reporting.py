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


import time
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import nemo_automodel.components.loggers.step_reporting as step_reporting
from nemo_automodel.components.loggers.metric_logger import MetricsSample
from nemo_automodel.components.loggers.step_reporting import MAX_OUTSTANDING_REPORTS, DeferredStepReporter
from nemo_automodel.components.training.signal_handler import all_gather_item


def _reporting_worker(rank: int, rendezvous: str) -> None:
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", rank=rank, world_size=2, init_method=rendezvous, timeout=timedelta(seconds=30))
    control_group = dist.new_group(backend="gloo")
    reporter = DeferredStepReporter(device)
    samples = []
    boundaries = []
    entries = torch.arange(4, device=device, dtype=torch.int64)
    cpu_step = torch.tensor(7, device="cpu")
    try:
        torch.cuda.set_sync_debug_mode("error")
        assert all_gather_item(rank, torch.int32, group=control_group) == [0, 1]
        for step in (7, 8):
            start = reporter.mark()
            local_loss = (entries + step).float().sum()
            end = reporter.mark()
            boundaries.append((start, end))
            sample = MetricsSample(
                step=step,
                epoch=2,
                metrics={
                    "loss": local_loss,
                    "experts": entries,
                    "expert_total": entries.sum(),
                    "empty": entries[:0],
                    "cpu_step": cpu_step,
                },
            )
            reporter.enqueue(sample, start=start, end=end, remote_log=step == 7, phases={"optimizer": (start, end)})
            # The snapshot owns its values even when the producer reuses storage.
            entries.add_(10)
            cpu_step.add_(1)
            samples.extend(reporter.poll())
        deadline = time.monotonic() + 20
        while len(samples) != 2 and time.monotonic() < deadline:
            samples.extend(reporter.poll())
            time.sleep(0.001)
        assert [sample.step for sample, _, _ in samples] == [7, 8]
        assert [remote for _, remote, _ in samples] == [True, False]
        for i, (sample, _, seconds) in enumerate(samples):
            start, end = boundaries[i]
            local_time = torch.tensor(start.elapsed_time(end) / 1000, dtype=torch.float64)
            dist.all_reduce(local_time, group=control_group, op=dist.ReduceOp.MAX)
            assert seconds == local_time.item()
            assert sample.metrics["optimizer-time"] == seconds
            assert sample.metrics["loss"] == 34 + 44 * i
            assert sample.metrics["expert_total"] == 6 + 40 * i
            assert sample.metrics["cpu_step"] == 7 + i
            torch.testing.assert_close(sample.metrics["empty"], torch.empty(0, dtype=torch.int64))
            torch.testing.assert_close(sample.metrics["experts"], torch.arange(4) + 10 * i)
    finally:
        torch.cuda.set_sync_debug_mode("default")
        reporter.close()
        dist.destroy_process_group(control_group)
        dist.destroy_process_group()


@pytest.mark.skipif(
    torch.cuda.device_count() < 2 or not dist.is_nccl_available(), reason="needs two CUDA GPUs and NCCL"
)
def test_deferred_reporting_with_nccl_training_and_cpu_control(tmp_path):
    rendezvous = (tmp_path / "rdzv").as_uri()
    mp.spawn(_reporting_worker, args=(rendezvous,), nprocs=2)


def _sample(step: int, **metrics: torch.Tensor) -> MetricsSample:
    return MetricsSample(step=step, epoch=0, metrics=dict(metrics))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_poll_returns_a_readback_only_after_its_copy_finishes():
    device = torch.device("cuda", torch.cuda.current_device())
    reporter = DeferredStepReporter(device)
    # Prime the pinned allocator with different values. A fresh 128 MiB host
    # allocation can take longer than the queued GPU work on fast devices.
    start = reporter.mark()
    stale = torch.full((1 << 24,), -1, dtype=torch.int64, device=device)
    reporter.enqueue(_sample(-1, vector=stale), start=start, end=reporter.mark(), remote_log=False)
    reporter.poll(wait=True)
    del stale

    busy = torch.randn(4096, 4096, device=device)
    start = reporter.mark()
    for _ in range(30):
        # Keep the stream busy, so polling starts long before the 128 MB copy below runs.
        busy = torch.tanh(busy @ busy)
    vector = torch.arange(1 << 24, device=device)
    end = reporter.mark()
    reporter.enqueue(_sample(0, vector=vector), start=start, end=end, remote_log=False)

    completed = []
    while not completed:
        completed = reporter.poll()
    reporter.close()

    torch.testing.assert_close(completed[0][0].metrics["vector"], torch.arange(1 << 24))


@pytest.mark.cpu_only
def test_poll_returns_records_in_step_order_with_their_own_phase_times():
    reporter = DeferredStepReporter(torch.device("cpu"))
    # Both records are pending together, and each phase differs from its step span and from the other phase.
    reporter.enqueue(_sample(0), start=0.0, end=2.0, remote_log=False, phases={"optimizer": (0.5, 1.0)})
    reporter.enqueue(_sample(1), start=2.0, end=5.0, remote_log=True, phases={"optimizer": (2.25, 3.0)})

    completed = reporter.poll(wait=True)
    reporter.close()

    assert [(s.step, remote, seconds, s.metrics["optimizer-time"]) for s, remote, seconds in completed] == [
        (0, False, 2.0, 0.5),
        (1, True, 3.0, 0.75),
    ]


@pytest.mark.cpu_only
def test_cpu_tensor_metrics_are_snapshots_of_the_producer():
    reporter = DeferredStepReporter(torch.device("cpu"))
    counts, total = torch.arange(4), torch.tensor(6)
    reporter.enqueue(_sample(0, counts=counts, total=total), start=0.0, end=1.0, remote_log=False)
    counts.add_(10)
    total.add_(10)

    ((sample, _, _),) = reporter.poll()
    reporter.close()

    torch.testing.assert_close(sample.metrics["counts"], torch.arange(4))
    assert sample.metrics["total"] == 6


class _HeldReduction:
    """A timing reduction that finishes only when waited for, like one stalled behind a slow peer."""

    def __init__(self) -> None:
        self.done = False

    def is_completed(self) -> bool:
        return self.done

    def wait(self) -> bool:
        self.done = True
        return True


@pytest.mark.cpu_only
def test_outstanding_reports_stay_bounded_while_timing_reductions_stall(monkeypatch):
    reductions = []

    def held_all_reduce(tensor, *, op, group, async_op):
        reductions.append(_HeldReduction())
        return reductions[-1]

    monkeypatch.setattr(step_reporting, "_new_gloo_group", lambda process_group: "reporting-group")
    monkeypatch.setattr(dist, "all_reduce", held_all_reduce)
    monkeypatch.setattr(dist, "destroy_process_group", lambda group: None)
    reporter = DeferredStepReporter(torch.device("cpu"))

    delivered = []
    for step in range(50):
        reporter.enqueue(_sample(step, value=torch.tensor(step)), start=0.0, end=float(step), remote_log=False)
        delivered += reporter.poll()
        assert sum(not reduction.done for reduction in reductions) <= MAX_OUTSTANDING_REPORTS
    # Below the cap a poll never waits, so each poll past it finishes exactly the oldest record.
    assert len(delivered) == 50 - MAX_OUTSTANDING_REPORTS
    delivered += reporter.poll(wait=True)
    reporter.close()

    assert [(s.step, s.metrics["value"], seconds) for s, _, seconds in delivered] == [
        (step, step, float(step)) for step in range(50)
    ]


def _student_teacher_worker(rank: int, rendezvous: str) -> None:
    dist.init_process_group("gloo", rank=rank, world_size=3, init_method=rendezvous, timeout=timedelta(seconds=30))
    students = dist.new_group(ranks=[0, 1])
    command = torch.zeros(1, dtype=torch.int64)
    try:
        if rank == 2:
            # A separate-mesh teacher builds no reporter and waits for the students' next command.
            dist.recv(command, src=0)
            assert command.item() == 7
            return
        reporter = DeferredStepReporter(torch.device("cpu"), process_group=students)
        try:
            reporter.enqueue(_sample(0), start=0.0, end=1.0 + rank, remote_log=False)
            ((_, _, seconds),) = reporter.poll(wait=True)
        finally:
            reporter.close()
        assert seconds == 2.0  # the slower student's step time
        if rank == 0:
            dist.send(command.fill_(7), dst=2)
    finally:
        dist.destroy_process_group()


@pytest.mark.cpu_only
def test_reporting_group_spans_only_the_training_ranks(tmp_path):
    """Separate-mesh distillation builds reporters on the students only, so the teacher must never be needed."""
    mp.spawn(_student_teacher_worker, args=((tmp_path / "rdzv").as_uri(),), nprocs=3)
