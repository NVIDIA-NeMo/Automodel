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

from nemo_automodel.components.loggers.metric_logger import MetricsSample
from nemo_automodel.components.loss.token_normalization import normalize_token_loss
from nemo_automodel.components.training.signal_handler import all_gather_item
from nemo_automodel.recipes.llm.step_reporting import DeferredStepReporter


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("count", [0, 1_000_003])
def test_device_token_normalization_preserves_loss_and_gradient(dtype, count):
    loss = torch.tensor(17.0, dtype=dtype, requires_grad=True)
    expected = normalize_token_loss(loss, count)
    actual = normalize_token_loss(loss, torch.tensor(count))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    (expected_grad,) = torch.autograd.grad(expected, loss)
    (actual_grad,) = torch.autograd.grad(actual, loss)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


def _reporting_worker(rank: int, rendezvous: str) -> None:
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", rank=rank, world_size=2, init_method=rendezvous, timeout=timedelta(seconds=30))
    control_group = dist.new_group(backend="gloo")
    reporter = DeferredStepReporter(device)
    samples = []
    boundaries = []
    entries = torch.arange(4, device=device, dtype=torch.int64)
    try:
        torch.cuda.set_sync_debug_mode("error")
        assert all_gather_item(rank, torch.int32, group=control_group) == [0, 1]
        for step in (7, 8):
            start = reporter.mark()
            local_loss = (entries + step).float().sum()
            end = reporter.mark()
            boundaries.append((start, end))
            sample = MetricsSample(step=step, epoch=2, metrics={"loss": local_loss, "experts": entries})
            reporter.enqueue(sample, start=start, end=end, remote_log=step == 7)
            # The snapshot owns its values even when the producer reuses storage.
            entries.add_(10)
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
            assert sample.metrics["loss"] == 34 + 44 * i
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
