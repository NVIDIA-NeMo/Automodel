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

"""Real Gloo collectives exercise validation with empty and uneven rank shards."""

from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.pipelining import PipelineStage, ScheduleGPipe
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import default_collate

from nemo_automodel.components.datasets.loader import ParallelAwareDataloader
from nemo_automodel.components.training.validation import synchronized_validation_batches
from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction


@pytest.fixture(autouse=True)
def _cpu_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep spawned Gloo workers on CPU, including when pytest runs on GPU hosts."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")


class CollectiveModel(nn.Module):
    """A tiny DDP model whose buffer broadcast requires every rank to forward."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 2, bias=False)
        self.register_buffer("offset", torch.zeros(1))
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor([[0.25], [-0.5]]))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Map ``inputs`` of shape [batch, 1] to logits of shape [batch, 2]."""
        # FSDP/EP require collectives even in eval. This unconditional collective
        # makes mismatched forward counts fail in the CPU test as well.
        token = self.offset.clone()
        dist.all_reduce(token)
        return self.linear(inputs) + token


class ValidationRecipe(TrainFinetuneRecipeForNextTokenPrediction):
    """Exercise the production validation loop with a small collective-bearing model."""

    def _forward_backward_step(
        self,
        idx: int,
        batch: dict[str, torch.Tensor],
        *,
        loss_buffer: list[torch.Tensor],
        num_label_tokens: int | None,
        num_batches: int,
        is_train: bool = True,
    ) -> None:
        logits = self.model_parts[0](batch["input_ids"])
        loss_buffer.append(nn.functional.cross_entropy(logits, batch["labels"].reshape(-1), reduction="sum"))


def _worker(rank: int, rendezvous: str, samples: int, masked: bool) -> None:
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=40))
    try:
        model = DistributedDataParallel(CollectiveModel())
        data = [
            {"input_ids": torch.tensor([float(i + 1)]), "labels": torch.tensor([-100 if masked else i % 2])}
            for i in range(samples)
        ]
        loader = ParallelAwareDataloader(
            data, dp_rank=rank, dp_world_size=2, batch_size=2, collate_fn=default_collate, is_training=False
        )
        recipe = ValidationRecipe.__new__(ValidationRecipe)
        recipe.model_parts = [model]
        recipe.dist_env = SimpleNamespace(device=torch.device("cpu"), is_main=rank == 0)
        recipe.pp_enabled = False
        recipe.optimizer = [SimpleNamespace(param_groups=[{"lr": 0.01}])]
        recipe.step_scheduler = SimpleNamespace(step=0, epoch=0)
        result = recipe._run_validation_epoch(loader)
        assert result.metrics["num_label_tokens"] == (0 if masked else samples)
        if samples and not masked:
            inputs = torch.tensor([[float(i + 1)] for i in range(samples)])
            labels = torch.tensor([i % 2 for i in range(samples)])
            expected = nn.functional.cross_entropy(model.module.linear(inputs), labels).item()
            assert result.metrics["val_loss"] == pytest.approx(expected, rel=1e-6)
        else:
            assert torch.isnan(torch.tensor(result.metrics["val_loss"]))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(("samples", "masked"), [(0, False), (1, False), (3, False), (5, False), (3, True)])
def test_validation_counts_each_example_once(tmp_path: Path, samples: int, masked: bool) -> None:
    mp.spawn(_worker, args=(f"file://{tmp_path / 'rdzv'}", samples, masked), nprocs=2, join=True)


def _error_worker(rank: int, rendezvous: str) -> None:
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=40))
    try:

        def batches() -> Iterator[dict[str, torch.Tensor]]:
            yield {"input_ids": torch.tensor([[1]])}
            if rank == 1:
                raise ValueError("bad input")
            yield {"input_ids": torch.tensor([[1]])}

        with pytest.raises(RuntimeError, match="validation loader failed"):
            list(synchronized_validation_batches(batches()))
    finally:
        dist.destroy_process_group()


def test_loader_errors_fail_on_every_rank(tmp_path: Path) -> None:
    mp.spawn(_error_worker, args=(f"file://{tmp_path / 'rdzv'}",), nprocs=2, join=True)


class PipelineValidationRecipe(ValidationRecipe):
    """Use a real two-stage schedule and DP groups in the production validation loop."""

    def _get_dp_group(self, include_cp: bool = False) -> dist.ProcessGroup:
        return self.device_mesh["dp"].get_group()

    def _forward_backward_step(
        self,
        idx: int,
        batch: dict[str, torch.Tensor],
        *,
        loss_buffer: list[torch.Tensor],
        num_label_tokens: int | None,
        num_batches: int,
        is_train: bool = True,
    ) -> None:
        inputs = (batch["input_ids"],) if self.device_mesh["pp"].get_local_rank() == 0 else ()
        self.schedule.eval(*inputs, target=batch["labels"].reshape(-1), losses=loss_buffer)
        if not loss_buffer:
            loss_buffer.append(torch.tensor(0.0))


def _pipeline_worker(rank: int, rendezvous: str, samples: int) -> None:
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=4, timeout=timedelta(seconds=40))
    try:
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "pp"))
        stage_index = mesh["pp"].get_local_rank()
        module = nn.Linear(1, 1 if stage_index == 0 else 2, bias=False)
        with torch.no_grad():
            module.weight.copy_(torch.tensor([[1.0]]) if stage_index == 0 else torch.tensor([[0.25], [-0.5]]))
        stage = PipelineStage(
            module, stage_index, 2, torch.device("cpu"), input_args=torch.ones(1, 1), group=mesh["pp"].get_group()
        )
        data = [{"input_ids": torch.tensor([float(i + 1)]), "labels": torch.tensor([i % 2])} for i in range(samples)]
        loader = ParallelAwareDataloader(
            data,
            dp_rank=mesh["dp"].get_local_rank(),
            dp_world_size=2,
            batch_size=2,
            collate_fn=default_collate,
            is_training=False,
            pp_enabled=True,
        )
        recipe = PipelineValidationRecipe.__new__(PipelineValidationRecipe)
        recipe.schedule = ScheduleGPipe(stage, n_microbatches=2, loss_fn=nn.CrossEntropyLoss(reduction="sum"))
        recipe.model_parts = [module]
        recipe.device_mesh = mesh
        recipe.dist_env = SimpleNamespace(device=torch.device("cpu"), is_main=rank == 0)
        recipe.pp_enabled = True
        recipe.optimizer = [SimpleNamespace(param_groups=[{"lr": 0.01}])]
        recipe.step_scheduler = SimpleNamespace(step=0, epoch=0)
        result = recipe._run_validation_epoch(loader)
        inputs = torch.tensor([[float(i + 1)] for i in range(samples)])
        labels = torch.tensor([i % 2 for i in range(samples)])
        expected = nn.functional.cross_entropy(inputs @ torch.tensor([[0.25, -0.5]]), labels).item()
        assert result.metrics["num_label_tokens"] == samples
        assert result.metrics["val_loss"] == pytest.approx(expected, rel=1e-6)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("samples", [1, 5])
def test_pipeline_validation_retains_partial_and_empty_shards(tmp_path: Path, samples: int) -> None:
    mp.spawn(_pipeline_worker, args=(f"file://{tmp_path / 'rdzv'}", samples), nprocs=4, join=True)
