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

import json
import math
import random
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from transformers.modeling_outputs import CausalLMOutput

from nemo_automodel.components.models.gs1.training import GS1DecisionExample, GS1DecisionObjective
from nemo_automodel.recipes.decision.train_gs1 import TrainGS1DecisionRecipe

pytestmark = pytest.mark.cpu_only


@dataclass(frozen=True)
class _Branch:
    suffix: tuple[int, ...]
    token_roles: tuple[str, ...]
    candidates: tuple[str, ...]


class _TinyDecisionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor([0.2, -0.1, 0.05]))
        self.unused = nn.Parameter(torch.tensor(1.0))

    def forward(self, input_ids, attention_mask, branches=None, **kwargs):
        features = (input_ids.float() * attention_mask).sum(dim=1, keepdim=True)
        width = max(len(branch.candidates) for branch in branches)
        return CausalLMOutput(logits=features * self.logit_scale[:width])


class _Checkpointer:
    def __init__(self, output: Path, rank: int) -> None:
        self.config = SimpleNamespace(enabled=True)
        self.output = output
        self.rank = rank
        self.saved = []
        self.lifecycle = SimpleNamespace(reserve=lambda path: None, publish=lambda *args, **kwargs: None)

    def save_model(self, model, path, **kwargs):
        assert not isinstance(model, torch.nn.parallel.DistributedDataParallel)
        self.saved.append(model)
        (self.output / f"unwrapped{self.rank}").touch()

    def async_wait(self):
        pass

    def finalize(self):
        pass


def _examples() -> list[GS1DecisionExample]:
    specs = [(2, (0.8, 0.2)), (4, (0.3, 0.7)), (3, (0.2, 0.5, 0.3)), (5, (0.6, 0.4)), (1, (0.1, 0.2, 0.7))]
    examples = []
    for index, (length, target) in enumerate(specs):
        candidates = tuple(str(i) for i in range(len(target)))
        suffix = tuple(range(10, 10 + length))
        state = tuple(range(1, index + 2))
        examples.append(
            GS1DecisionExample(str(index), "q", (*state, *suffix), _Branch(suffix, (), candidates), len(state), target)
        )
    return examples


def _recipe(output: Path, rank: int, world: int) -> TrainGS1DecisionRecipe:
    torch.manual_seed(7)
    recipe = TrainGS1DecisionRecipe.__new__(TrainGS1DecisionRecipe)
    recipe.model = torch.nn.parallel.DistributedDataParallel(_TinyDecisionModel(), find_unused_parameters=True)
    recipe.examples = _examples()
    recipe.objective = GS1DecisionObjective()
    recipe.pad_token_id = 0
    recipe.device = torch.device("cpu")
    recipe.tokenizer = None
    recipe.seed = 17
    recipe.local_batch_size = 2
    recipe.epochs = 1
    recipe.max_grad_norm = 0.15
    recipe.device_mesh = None
    recipe.optimizers = [torch.optim.SGD(recipe.model.parameters(), lr=0.01)]
    recipe.dist_env = SimpleNamespace(is_main=rank == 0)
    recipe.output_dir = output
    recipe.cfg = SimpleNamespace(get=lambda key, default=None: default)
    recipe.checkpointer = _Checkpointer(output, rank)
    recipe._get_dp_rank = lambda: rank
    recipe._get_dp_group_size = lambda: world
    return recipe


def _worker(rank: int, rendezvous: str, output: str) -> None:
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=45)
    )
    recipe = _recipe(Path(output), rank, 2)
    try:
        recipe.run_train_validation_loop()
        torch.save(recipe.model.module.state_dict(), Path(output) / f"rank{rank}.pt")
    finally:
        dist.destroy_process_group()


def _reference() -> tuple[dict[str, torch.Tensor], list[float]]:
    torch.manual_seed(7)
    model = _TinyDecisionModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    examples = _examples()
    indices = list(range(len(examples)))
    random.Random(17).shuffle(indices)
    indices.extend(indices[index % len(indices)] for index in range(6 - len(indices)))
    ranks = [indices[rank:6:2] for rank in range(2)]
    norms = []
    for start in range(0, len(ranks[0]), 2):
        items = [examples[index] for rank in ranks for index in rank[start : start + 2]]
        sequence_length = max(len(item.input_ids) for item in items)
        input_ids = torch.zeros((len(items), sequence_length), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        option_count = max(len(item.target_probs) for item in items)
        targets = torch.zeros((len(items), option_count))
        option_mask = torch.zeros((len(items), option_count), dtype=torch.bool)
        for row, item in enumerate(items):
            length = len(item.input_ids)
            input_ids[row, -length:] = torch.tensor(item.input_ids)
            attention_mask[row, -length:] = True
            count = len(item.target_probs)
            targets[row, :count] = torch.tensor(item.target_probs)
            option_mask[row, :count] = True
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids, attention_mask, branches=[item.branch for item in items]).logits
        loss, _ = GS1DecisionObjective()(logits, targets, option_mask)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.15)
        optimizer.step()
        norms.append(float(norm))
    return model.state_dict(), norms


def test_two_rank_optimizer_loop_matches_padded_fp32_reference(tmp_path: Path) -> None:
    output = tmp_path / "distributed"
    output.mkdir()
    mp.spawn(_worker, args=(str(tmp_path / "rendezvous"), str(output)), nprocs=2, join=True)
    expected, norms = _reference()
    rank0 = torch.load(output / "rank0.pt", weights_only=True)
    rank1 = torch.load(output / "rank1.pt", weights_only=True)
    for name, value in expected.items():
        torch.testing.assert_close(rank0[name], value, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(rank1[name], value, atol=1e-6, rtol=1e-6)
    assert (output / "unwrapped0").exists() and (output / "unwrapped1").exists()
    actual = [json.loads(line) for line in (output / "training_log.jsonl").read_text().splitlines()]
    assert len(actual) == len(norms) == 2
    for row, norm in zip(actual, norms, strict=True):
        assert math.isclose(row["grad_norm"], norm, rel_tol=1e-5, abs_tol=1e-6)


def test_disabled_checkpoint_is_a_noop(tmp_path: Path) -> None:
    recipe = TrainGS1DecisionRecipe.__new__(TrainGS1DecisionRecipe)
    recipe.checkpointer = SimpleNamespace(
        config=SimpleNamespace(enabled=False),
        lifecycle=SimpleNamespace(reserve=Mock(), publish=Mock()),
        save_model=Mock(),
        async_wait=Mock(),
    )
    recipe.output_dir = tmp_path
    recipe._save_final_checkpoint()
    assert not (tmp_path / "final").exists()
    recipe.checkpointer.lifecycle.reserve.assert_not_called()
    recipe.checkpointer.save_model.assert_not_called()
    recipe.checkpointer.lifecycle.publish.assert_not_called()
