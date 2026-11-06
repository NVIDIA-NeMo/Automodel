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

"""Real DDP finite-stream exhaustion and update equivalence."""

import json
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

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


def _distributed_worker(rank: int, rendezvous: str, result_dir: str, accumulation: int):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        # The second rank has an entire extra window when rank zero ends.
        source = _Stream(5 if rank == 0 else 8, offset=rank)
        scheduler = _scheduler(source, accumulation=accumulation, dp_size=2)
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)
        model = DistributedDataParallel(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        weights = []
        for _ in scheduler.epochs:
            for batches in scheduler:
                optimizer.zero_grad()
                for batch in batches:
                    inputs = torch.tensor([[float(batch["value"])]])
                    loss = model(inputs).square().mean() / accumulation
                    loss.backward()
                optimizer.step()
                weights.append(model.module.weight.item())
        result = {"weights": weights, "state": scheduler.state_dict(), "read": source.read}

        # A source failure must reach every peer before another model step.
        def failing_source():
            yield {"value": 1}
            yield {"value": 2}
            if rank == 0:
                raise ValueError("unreadable document")
            yield {"value": 3}
            yield {"value": 4}

        failed = _scheduler(failing_source(), dp_size=2)
        try:
            list(failed)
        except RuntimeError as exc:
            result["error"] = str(exc)
            result["cause"] = str(exc.__cause__)
        else:
            raise AssertionError("A source failure did not terminate all ranks")
        Path(result_dir, f"rank-{rank}.json").write_text(json.dumps(result))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("accumulation", [1, 2])
def test_real_two_rank_uneven_exhaustion_matches_full_batch_updates(tmp_path, accumulation):
    rendezvous = (tmp_path / "rendezvous").as_uri()
    mp.spawn(_distributed_worker, args=(rendezvous, str(tmp_path), accumulation), nprocs=2, join=True)
    # Single-process global-batch reference, including both accumulation microbatches.
    reference = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        reference.weight.fill_(1.0)
    optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
    expected = []
    for start in range(0, 5 - accumulation + 1, accumulation):
        values = list(range(start + 1, start + accumulation + 1)) + list(range(start + 2, start + accumulation + 2))
        optimizer.zero_grad()
        reference(torch.tensor(values, dtype=torch.float32).reshape(-1, 1)).square().mean().backward()
        optimizer.step()
        expected.append(reference.weight.item())
    for rank in range(2):
        result = json.loads((tmp_path / f"rank-{rank}.json").read_text())
        assert result["weights"] == pytest.approx(expected, abs=1e-7)
        assert result["state"] == {"step": 5 // accumulation, "epoch": 1, "stream_exhausted": True}
        assert result["read"] == (5 if rank == 0 else 6)
        assert "synchronized training stream failed" in result["error"]
        assert result["cause"] == ("unreadable document" if rank == 0 else "None")
