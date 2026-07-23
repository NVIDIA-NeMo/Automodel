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

"""Scheduled distributed CPU coverage for Domino evaluation reductions."""

from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nemo_automodel.components.speculative.dflash.core import NoValidAnchorsError
from nemo_automodel.recipes.llm.train_domino import TrainDominoRecipe


def _distributed_domino_eval_worker(rank: int, world_size: int, init_file: str, output_dir: str) -> None:
    """Run uneven Domino validation on one CPU process.

    Rank 0 contributes no valid batch while rank 1 contributes scalar step
    statistics. Both ranks must enter the same Gloo collectives and produce the
    same globally reduced result.
    """
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        batch = {
            "input_ids": torch.zeros(1, 4, dtype=torch.long),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
            "loss_mask": torch.ones(1, 4),
        }
        recipe = TrainDominoRecipe.__new__(TrainDominoRecipe)
        recipe.device = torch.device("cpu")
        recipe.val_dataloader = [batch]
        recipe.trainer_module = SimpleNamespace(eval=lambda: None, train=lambda: None)
        recipe.target_wrapper = SimpleNamespace(generate_batch=lambda **kwargs: SimpleNamespace(**kwargs))
        result = (
            NoValidAnchorsError("rank has no valid anchors")
            if rank == 0
            else SimpleNamespace(
                loss=torch.tensor(2.0),
                loss_weight=torch.tensor(4.0),
                accuracy=torch.tensor(0.8),
                valid_tokens=torch.tensor(5.0),
                correct_tokens=torch.tensor(4.0),
                accept_len_sum=torch.tensor(6.0),
                valid_blocks=torch.tensor(2.0),
                final_loss=torch.tensor(1.5),
                base_loss=torch.tensor(3.0),
                base_correct_tokens=torch.tensor(2.0),
                base_accept_len_sum=torch.tensor(4.0),
            )
        )

        def _run_step(_target_batch: SimpleNamespace) -> SimpleNamespace:
            """Return the per-rank scalar metric tensors.

            Args:
                _target_batch: Namespace containing input tensors of shape
                    [batch, sequence].

            Returns:
                Namespace containing scalar metric tensors.
            """
            if isinstance(result, Exception):
                raise result
            return result

        recipe._run_trainer_step = _run_step
        metrics = recipe._run_eval()
        torch.save(metrics, Path(output_dir) / f"rank_{rank}.pt")
    finally:
        dist.destroy_process_group()


def test_distributed_domino_eval_handles_rank_with_no_valid_batches(tmp_path: Path) -> None:
    init_file = tmp_path / "gloo_init"
    mp.spawn(
        _distributed_domino_eval_worker,
        args=(2, str(init_file), str(tmp_path)),
        nprocs=2,
        join=True,
    )

    expected = {
        "val_loss": 2.0,
        "val_accuracy": pytest.approx(0.8),
        "val_accept_len": 3.0,
        "val_final_loss": 1.5,
        "val_base_loss": 3.0,
        "val_base_accuracy": pytest.approx(0.4),
        "val_base_accept_len": 2.0,
    }
    assert torch.load(tmp_path / "rank_0.pt", weights_only=True) == expected
    assert torch.load(tmp_path / "rank_1.pt", weights_only=True) == expected
