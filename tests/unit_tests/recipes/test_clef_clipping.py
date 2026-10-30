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

"""Numerical regressions for the actual CLEF loop's joint gradient budget."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.components.datasets.decision.records import question_targets
from nemo_automodel.components.loss.decision import DecisionLossConfig
from nemo_automodel.components.loss.decision_batch import DecisionTargets, QuestionLogits, assemble_decision_batch
from nemo_automodel.components.models.clef.joint_schema_head import (
    ClefHeadConfig,
    EncodedQuestion,
    EncodedRecord,
    LexicalTable,
    clef_option_orders,
)
from nemo_automodel.recipes._typed_config import RecipeConfig
from nemo_automodel.recipes.decision.train_clef import TrainClefDecisionRecipe


class _TinyBackbone(nn.Module):
    def __init__(self, trainable: bool) -> None:
        super().__init__()
        self.embedding = nn.Embedding(16, 8)
        self.projection = nn.Linear(8, 8, bias=False)
        self.requires_grad_(trainable)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        use_cache: bool,
        logits_to_keep: int,
        output_hidden_states: bool,
    ) -> CausalLMOutputWithPast:
        """Return projected text states through the production hidden-state API.

        Args:
            input_ids: Long tensor of shape [batch, sequence].
            attention_mask: Long tensor of shape [batch, sequence].
            use_cache: Whether an autoregressive cache is requested.
            logits_to_keep: Number of requested language-model logits.
            output_hidden_states: Whether final hidden states are requested.

        Returns:
            Model output containing hidden states of shape [batch, sequence, hidden].
        """
        return CausalLMOutputWithPast(hidden_states=(self.projection(self.embedding(input_ids)),))


def _records() -> list:
    items = []
    for index, count in enumerate((1, 3)):
        questions = tuple(EncodedQuestion(f"q{i}", 1, (0, 1), ((1, 2), (2, 3)), ("a", "b")) for i in range(count))
        record = {
            "id": str(index),
            "state": "example",
            "questions": {
                question.question_id: {"type": "choice", "criteria": {"a": "A", "b": "B"}} for question in questions
            },
            "answers": {question.question_id: "a" if index == 0 else "b" for question in questions},
        }
        items.append((record, EncodedRecord((1, 2, 3, 4), questions, str(index))))
    return items


def _recipe(output: Path, *, trainable_backbone: bool) -> TrainClefDecisionRecipe:
    torch.manual_seed(7)
    recipe = TrainClefDecisionRecipe.__new__(TrainClefDecisionRecipe)
    recipe.model = _TinyBackbone(trainable_backbone)
    recipe.head = ClefHeadConfig(hidden_size=8, width=8, routing_layers=1, layers=1, heads=2, feedforward=16).build()
    recipe.lexical = LexicalTable(list(range(16)), recipe.model.embedding.weight.detach().clone())
    recipe.pad_token_id = 0
    recipe.device = torch.device("cpu")
    recipe.device_mesh = None
    recipe.peft_config = None
    recipe.seed = 7
    recipe.tag = "sft"
    recipe.dist_env = SimpleNamespace(is_main=True)
    recipe.output_dir = output
    recipe.loss_config = DecisionLossConfig(label_smoothing=0.0, brier_weight=0.0, ordinal_weight=0.0)
    recipe.cfg = RecipeConfig(
        ConfigNode(
            {
                "lr_scheduler": None,
                "train": {
                    "epochs": 1,
                    "local_batch_size": 1,
                    "global_batch_size": 2,
                    "warmup_ratio": 0.0,
                    "max_grad_norm": 0.1,
                    "val_every_steps": 100,
                },
            }
        )
    )
    recipe.splits = {"train": _records()}
    recipe.optimizer = [torch.optim.SGD(recipe.model.parameters(), lr=0.05)] if trainable_backbone else []
    recipe.head_optimizer = torch.optim.SGD(recipe.head.parameters(), lr=0.05)
    recipe._get_dp_group_size = lambda: 1
    recipe._get_dp_rank = lambda: 0
    recipe._dp_group = lambda: None
    # Persistence is orthogonal to this clipping regression; exercise real stage I/O separately.
    recipe._save_stage = lambda path, **kwargs: None
    return recipe


def _parameters(recipe: TrainClefDecisionRecipe) -> dict[str, nn.Parameter]:
    return {
        f"{part_name}.{name}": parameter
        for part_name, part in (("backbone", recipe.model), ("head", recipe.head))
        for name, parameter in part.named_parameters()
        if parameter.requires_grad
    }


@pytest.mark.parametrize("trainable_backbone", [False, True])
def test_actual_clef_loop_uses_one_combined_gradient_budget(tmp_path, trainable_backbone):
    actual = _recipe(tmp_path, trainable_backbone=trainable_backbone)
    reference = _recipe(tmp_path, trainable_backbone=trainable_backbone)
    items = reference.splits["train"]
    questions = [question_targets(record, clef_option_orders(encoded)) for record, encoded in items]
    targets = DecisionTargets.from_questions(record_ids=[str(record["id"]) for record, _ in items], questions=questions)
    scores = [
        [
            QuestionLogits(str(record["id"]), target.question_id, target.option_ids, row)
            for target, row in zip(record_targets, record_scores, strict=True)
        ]
        for (record, _), record_targets, record_scores in zip(items, questions, reference._forward(items), strict=True)
    ]
    batch = assemble_decision_batch(logits=scores, targets=targets)
    loss, _ = reference.loss_config.build()(batch.logits, batch.targets)
    loss.backward()
    expected_parameters = _parameters(reference)
    expected_norm = nn.utils.clip_grad_norm_(list(expected_parameters.values()), 0.1)
    assert float(expected_norm) > 0.1
    expected_gradients = {name: parameter.grad.detach().clone() for name, parameter in expected_parameters.items()}
    for optimizer in [*reference.optimizer, reference.head_optimizer]:
        optimizer.step()

    applied_gradients = {}

    def capture_before_step(optimizer, args, kwargs):
        for name, parameter in _parameters(actual).items():
            if parameter.grad is not None:
                applied_gradients[name] = parameter.grad.detach().clone()

    handles = [
        optimizer.register_step_pre_hook(capture_before_step)
        for optimizer in [*actual.optimizer, actual.head_optimizer]
    ]
    try:
        actual._train()
    finally:
        for handle in handles:
            handle.remove()

    assert applied_gradients.keys() == expected_gradients.keys()
    applied_norm = torch.linalg.vector_norm(torch.cat([gradient.flatten() for gradient in applied_gradients.values()]))
    assert float(applied_norm) <= 0.100002
    for name, parameter in _parameters(actual).items():
        torch.testing.assert_close(applied_gradients[name], expected_gradients[name], atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(parameter, expected_parameters[name], atol=2e-6, rtol=2e-5)
    training_rows = [json.loads(line) for line in (tmp_path / "training_log.jsonl").read_text().splitlines()]
    reported_norm = next(row["grad_norm"] for row in training_rows if "grad_norm" in row)
    assert reported_norm > 0
    assert reported_norm == pytest.approx(float(expected_norm), rel=2e-5, abs=2e-6)
