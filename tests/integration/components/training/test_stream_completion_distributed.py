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

from nemo_automodel.components.training.step_scheduler import StepSchedulerConfig, TokenCounts


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


# Two DP groups represent two TP/PP coordinates. Deliberately different data
# at those coordinates proves their token totals are not multiplied or mixed.
def _fused_statistics_worker(rank: int, rendezvous: str, result_dir: str):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=4, timeout=timedelta(seconds=60))
    try:
        groups = [dist.new_group([0, 2]), dist.new_group([1, 3])]
        slot = rank % 2
        counts_per_batch = rank + 1
        source = _Stream(1 if rank == 3 else 2)
        scheduler = _scheduler(source, accumulation=1, dp_size=2)
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)
        model = DistributedDataParallel(model, process_group=groups[slot])
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        reductions = []
        all_reduce = dist.all_reduce

        def record_reduce(tensor, *args, **kwargs):
            reductions.append((tensor.numel(), kwargs.get("group") is None))
            return all_reduce(tensor, *args, **kwargs)

        dist.all_reduce = record_reduce

        def count_tokens(batches):
            return TokenCounts(counts_per_batch, counts_per_batch + 2)

        observed = []
        for batches, counts in scheduler.iter_steps(count_tokens, statistics_group_index=slot):
            expected_supervised = 4 if slot == 0 else 6
            assert counts == TokenCounts(expected_supervised, expected_supervised + 4)
            inputs = torch.full((counts_per_batch, 1), float(rank + 1))
            optimizer.zero_grad()
            # DDP averages gradients. The fused count must preserve the old DP
            # denominator so this matches a single-process token-weighted update.
            loss = model(inputs).square().sum() * 2 / counts.supervised
            loss.backward()
            optimizer.step()
            observed.append(counts.supervised)
        assert observed == [4 if slot == 0 else 6]
        assert scheduler.state_dict() == {"step": 1, "epoch": 1, "stream_exhausted": True}
        assert len(reductions) == 2  # one successful window, then global exhaustion

        # EOF before the first step is still a coordinated error, including
        # when only a model-parallel peer (rank 3) has no data.
        empty = _scheduler(_Stream(0 if rank == 3 else 1), accumulation=1, dp_size=2)
        try:
            list(empty.iter_steps(count_tokens, statistics_group_index=slot))
        except RuntimeError as exc:
            assert "before a complete optimizer step" in str(exc)
        else:
            raise AssertionError("Initial uneven EOF was not coordinated")

        # A source failure wins over EOF on another rank.
        def failed_source():
            if rank == 2:
                raise ValueError("unreadable document")
            if rank != 3:
                yield {"value": 1}

        failed = _scheduler(failed_source(), accumulation=1, dp_size=2)
        try:
            list(failed.iter_steps(count_tokens, statistics_group_index=slot))
        except RuntimeError as exc:
            assert "synchronized training stream failed" in str(exc)
            assert str(exc.__cause__) == ("unreadable document" if rank == 2 else "None")
        else:
            raise AssertionError("Source error did not terminate all model-parallel peers")

        def invalid_counts(batches):
            if rank == 1:
                raise ValueError("invalid labels")
            return count_tokens(batches)

        failed = _scheduler(_Stream(1), accumulation=1, dp_size=2)
        try:
            list(failed.iter_steps(invalid_counts, statistics_group_index=slot))
        except RuntimeError as exc:
            assert "synchronized training stream failed" in str(exc)
            assert str(exc.__cause__) == ("invalid labels" if rank == 1 else "None")
        else:
            raise AssertionError("Token preparation error did not terminate every rank")
        assert reductions == [(6, True)] * 5
        Path(result_dir, f"fused-rank-{rank}.json").write_text(
            json.dumps({"weight": model.module.weight.item(), "slot": slot, "reductions": len(reductions)})
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.cpu_only
def test_fused_stream_statistics_preserve_dp_updates_and_coordinate_model_parallel_peers(tmp_path):
    rendezvous = (tmp_path / "fused-rendezvous").as_uri()
    mp.spawn(_fused_statistics_worker, args=(rendezvous, str(tmp_path)), nprocs=4, join=True)
    for rank in range(4):
        result = json.loads((tmp_path / f"fused-rank-{rank}.json").read_text())
        peers = (0, 2) if result["slot"] == 0 else (1, 3)
        reference = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            reference.weight.fill_(1.0)
        inputs = torch.tensor([float(r + 1) for r in peers for _ in range(r + 1)]).reshape(-1, 1)
        optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
        reference(inputs).square().mean().backward()
        optimizer.step()
        assert result["weight"] == pytest.approx(reference.weight.item(), abs=1e-7)
