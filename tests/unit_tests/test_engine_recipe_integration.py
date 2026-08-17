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

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy
from nemo_automodel.components.models.nemotron_parse.nemotron_parse_loss import NemotronParseLoss
from nemo_automodel.engine import Engine, collate_prebatched
from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction
from nemo_automodel.recipes.vlm.finetune import FinetuneRecipeForVLM


class _Config(dict):
    def __init__(self):
        super().__init__()
        self.mtp = SimpleNamespace(scaling_factor=None, ignore_index=-100)


class _TinyLM(nn.Module):
    def __init__(self, *, vlm: bool = False):
        super().__init__()
        self.embedding = nn.Embedding(8, 4)
        self.vision = nn.Linear(1, 4) if vlm else None
        self.output = nn.Linear(4, 8)
        self.forward_calls = 0

    def forward(self, input_ids, pixel_values=None):
        self.forward_calls += 1
        hidden = self.embedding(input_ids)
        if pixel_values is not None:
            hidden = hidden + self.vision(pixel_values).unsqueeze(1)
        return SimpleNamespace(logits=self.output(hidden))


class _CountingSGD(torch.optim.SGD):
    def __init__(self, parameters):
        super().__init__(parameters, lr=0.1)
        self.step_calls = 0
        self.zero_calls = 0

    def step(self, closure=None):
        self.step_calls += 1
        return super().step(closure)

    def zero_grad(self, *args, **kwargs):
        self.zero_calls += 1
        return super().zero_grad(*args, **kwargs)


@pytest.mark.parametrize(
    ("recipe_cls", "vlm", "loss_kind"),
    [
        (TrainFinetuneRecipeForNextTokenPrediction, False, "masked"),
        (FinetuneRecipeForVLM, True, "masked"),
        (FinetuneRecipeForVLM, True, "nemotron_parse"),
    ],
)
def test_recipes_run_one_datum_engine_window_then_one_optimizer_step(recipe_cls, vlm, loss_kind):
    model = _TinyLM(vlm=vlm)
    reference = _TinyLM(vlm=vlm)
    reference.load_state_dict(model.state_dict())
    recipe = object.__new__(recipe_cls)
    recipe.cfg = _Config()
    recipe.loss_fn = (
        NemotronParseLoss(class_token_start_idx=100, reduction="sum")
        if loss_kind == "nemotron_parse"
        else MaskedCrossEntropy()
    )
    recipe.model_parts = [model]
    recipe.device_mesh = None
    recipe.moe_mesh = None
    recipe.pp_enabled = False
    recipe.dist_env = SimpleNamespace(device=torch.device("cpu"), world_size=1, is_main=True)
    recipe.distributed_config = SimpleNamespace(defer_fsdp_grad_sync=True)
    recipe.engine = Engine(model, device="cpu", microbatch_size=1, collate_fn=collate_prebatched)
    optimizer = _CountingSGD(model.parameters())
    recipe.optimizer = [optimizer]
    recipe.lr_scheduler = None
    recipe.checkpointer = SimpleNamespace(maybe_wait_for_staging=lambda: None)
    recipe.step_scheduler = SimpleNamespace(step=1, epoch=0, is_remote_logging_step=False)
    recipe.timestamp = time.perf_counter() - 1.0

    reductions = 0

    def local_reduce(value, include_cp=False):
        nonlocal reductions
        reductions += 1
        return value

    recipe._dp_allreduce = local_reduce
    batches = [
        {"input_ids": torch.tensor([[1, 2, 3]]), "labels": torch.tensor([[2, 3, -100]])},
        {"input_ids": torch.tensor([[4, 5]]), "labels": torch.tensor([[5, 6]])},
    ]
    if vlm:
        batches[0]["pixel_values"] = torch.tensor([[0.5]])
        batches[1]["pixel_values"] = torch.tensor([[1.5]])

    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
    denominator = sum((batch["labels"] != -100).sum() for batch in batches)
    reference_loss = (
        sum(
            F.cross_entropy(
                reference(
                    batch["input_ids"],
                    pixel_values=batch.get("pixel_values"),
                ).logits.flatten(0, 1),
                batch["labels"].flatten(),
                ignore_index=-100,
                reduction="sum",
            )
            for batch in batches
        )
        / denominator
    )
    reference_loss.backward()
    reference_optimizer.step()

    metrics = recipe._run_train_optim_step(batches, max_grad_norm=None)

    assert metrics.metrics["loss"] == pytest.approx(reference_loss.item())
    assert model.forward_calls == 2
    assert optimizer.step_calls == 1
    assert optimizer.zero_calls == 1
    assert reductions == 2  # token counters only; Engine already reduced the loss
    for actual, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected)
