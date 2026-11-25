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

"""Actual native recipe: common main-token denominator, unequal and empty rows.

R1/3,K1/2 are emulated recipe scaling tests, not distributed collective tests.
The reference oracle uses a different, explicit row+row objective.
"""

import copy

import pytest
import torch
from torch.nn import functional as F

from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy
from nemo_automodel.components.loss.mtp import calculate_mtp_loss
from nemo_automodel.components.loss.utils import calculate_loss
from nemo_automodel.recipes.llm.train_ft import (
    TrainFinetuneRecipeForNextTokenPrediction,
)
from tests.unit_tests.models.ernie5_5.test_mtp_training_local import batch, model
from tests.unit_tests.recipes.test_train_ft import (
    TestRunTrainOptimStepSetsMoEScale as _OptimizerStepFixture,
)
from tests.unit_tests.recipes.test_train_ft import (
    _make_mtp_forward_backward_recipe,
)

pytestmark = pytest.mark.cpu_only


def _rows(ranks: int, microbatches: int) -> list[list[dict]]:
    result = []
    for rank in range(ranks):
        rows = []
        for index in range(microbatches):
            tokens, labels = (value.clone() for value in batch())
            labels[:, : (rank * 2 + index) % 5] = -100
            if rank == 0 and index == 1:
                labels.fill_(-100)
            rows.append({"input_ids": tokens, "labels": labels})
        result.append(rows)
    return result


@pytest.mark.parametrize("ranks,microbatches", [(1, 1), (1, 2), (3, 1), (3, 2)])
def test_real_native_recipe_matches_common_token_mean_for_main_and_mtp(
    monkeypatch: pytest.MonkeyPatch, ranks: int, microbatches: int
) -> None:
    rows = _rows(ranks, microbatches)
    total_main = sum(int((row["labels"] != -100).sum()) for rank in rows for row in rank)
    reference = model()
    initial = copy.deepcopy(reference.state_dict())
    objective = 0
    for rank_rows in rows:
        for row in rank_rows:
            output = reference(row["input_ids"])
            labels = row["labels"]
            targets = torch.cat((labels[:, 1:], torch.full_like(labels[:, :1], -100)), dim=1)
            main_sum = F.cross_entropy(output.logits.float().flatten(0, 1), labels.flatten(), reduction="sum")
            auxiliary_sum = F.cross_entropy(
                output.mtp_per_depth_logits[0].float().flatten(0, 1),
                targets.flatten(),
                reduction="sum",
            )
            objective = objective + (main_sum + 0.1 * auxiliary_sum) / total_main
    objective.backward()
    actual_grads = []
    logged_losses = []
    for rank_rows in rows:
        candidate = model()
        candidate.load_state_dict(initial)
        recipe = _make_mtp_forward_backward_recipe(candidate, monkeypatch, calculate_mtp_loss=calculate_mtp_loss)
        object.__setattr__(recipe, "loss_fn", MaskedCrossEntropy())
        recipe.cfg.mtp.scaling_factor = None
        object.__setattr__(recipe, "_get_dp_group_size", lambda include_cp=False: ranks)
        monkeypatch.setattr("nemo_automodel.recipes.llm.train_ft.calculate_loss", calculate_loss)
        losses = []
        for index, row in enumerate(rank_rows):
            recipe._forward_backward_step(
                index,
                copy.deepcopy(row),
                loss_buffer=losses,
                num_label_tokens=total_main,
                num_batches=microbatches,
                is_train=True,
            )
        logged_losses.extend(losses)
        actual_grads.append(
            {
                name: None if parameter.grad is None else parameter.grad.clone()
                for name, parameter in candidate.named_parameters()
            }
        )
    # The recipe logs globally normalized contributions before its R multiplier.
    torch.testing.assert_close(sum(logged_losses), objective.detach())
    for name, parameter in reference.named_parameters():
        gradients = [rank[name] for rank in actual_grads]
        if parameter.grad is None:
            assert all(gradient is None for gradient in gradients), name
        else:
            assert all(gradient is not None for gradient in gradients), name
            reduced = sum(gradients) / ranks
            torch.testing.assert_close(reduced, parameter.grad, rtol=1e-4, atol=1e-8, msg=name)


@pytest.mark.parametrize("all_empty", [False, True])
def test_complete_recipe_optimizer_step_counts_unequal_and_empty_rows(
    monkeypatch: pytest.MonkeyPatch, all_empty: bool
) -> None:
    """Exercise the recipe's own token counter, forward, losses and optimizer."""
    rows = _rows(1, 2)[0]
    tokens, labels = (value.clone() for value in batch())
    labels[:, :5] = -100
    rows.append({"input_ids": tokens, "labels": labels})
    if all_empty:
        for row in rows:
            row["labels"].fill_(-100)
    total = sum(int((row["labels"] != -100).sum()) for row in rows)
    counts = [int((row["labels"] != -100).sum()) for row in rows]
    assert counts[1] == 0 and (all_empty or counts[0] != counts[2])
    reference = model()
    candidate = model()
    candidate.load_state_dict(reference.state_dict())
    objective = 0
    for row in rows:
        output = reference(row["input_ids"])
        labels = row["labels"]
        targets = torch.cat((labels[:, 1:], torch.full_like(labels[:, :1], -100)), dim=1)
        main = F.cross_entropy(output.logits.double().flatten(0, 1), labels.flatten(), reduction="sum")
        auxiliary = F.cross_entropy(
            output.mtp_per_depth_logits[0].double().flatten(0, 1),
            targets.flatten(),
            reduction="sum",
        )
        objective = objective + (main + 0.1 * auxiliary) / max(total, 1)
    objective.backward()
    fixture = _OptimizerStepFixture()
    fixture.setup_method()
    recipe = fixture._make_recipe(monkeypatch, pp_enabled=False, dp_group_size=1)
    # The existing forward harness only patches CP=1 plumbing. Restore both
    # production loss functions and bind the real forward on the whole-step recipe.
    forward_recipe = _make_mtp_forward_backward_recipe(candidate, monkeypatch, calculate_mtp_loss=calculate_mtp_loss)
    recipe.distributed_config = forward_recipe.distributed_config
    monkeypatch.setattr("nemo_automodel.recipes.llm.train_ft.calculate_loss", calculate_loss)
    recipe.model_parts = [candidate]
    recipe.loss_fn = MaskedCrossEntropy()
    recipe.cfg.mtp = ConfigNode({"ignore_index": -100, "scaling_factor": None})
    monkeypatch.setattr(
        recipe,
        "_forward_backward_step",
        TrainFinetuneRecipeForNextTokenPrediction._forward_backward_step.__get__(recipe),
    )
    optimizer = torch.optim.SGD([value for value in candidate.parameters() if value.requires_grad], lr=0.01)
    recipe.optimizer = [optimizer]
    initial = {name: parameter.detach().clone() for name, parameter in candidate.named_parameters()}
    captured = {}

    def capture_gradients(_optimizer: torch.optim.Optimizer, _args: tuple, _kwargs: dict) -> None:
        captured.update(
            {
                name: None if parameter.grad is None else parameter.grad.detach().clone()
                for name, parameter in candidate.named_parameters()
            }
        )

    handle = optimizer.register_step_pre_hook(capture_gradients)
    try:
        sample = recipe._run_train_optim_step(copy.deepcopy(rows))
    finally:
        handle.remove()
        fixture.teardown_method()
    assert sample.metrics["num_label_tokens"] == total
    assert sample.metrics["loss"] == pytest.approx(float(objective.detach()), rel=2e-6, abs=1e-8)
    for name, expected in reference.named_parameters():
        actual = captured[name]
        if expected.grad is None:
            assert actual is None, name
            torch.testing.assert_close(candidate.get_parameter(name), initial[name], rtol=0, atol=0, msg=name)
        else:
            assert actual is not None and bool(torch.isfinite(actual).all()), name
            torch.testing.assert_close(actual, expected.grad, rtol=1e-4, atol=1e-8, msg=name)
            torch.testing.assert_close(
                candidate.get_parameter(name),
                initial[name] - 0.01 * expected.grad,
                rtol=1e-6,
                atol=1e-8,
                msg=name,
            )
            if all_empty:
                assert not actual.any(), name
