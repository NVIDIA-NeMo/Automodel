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

"""CPU/Gloo checks for the actual CLEF optimizer-window normalization path.

These exercise replicated-head DP collectives. FSDP sharded-backbone acceptance
still requires the separate supported GPU test environment.
"""

import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from nemo_automodel.components.checkpoint.config import CheckpointingConfig
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


class TinyFrozenBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 8)
        self.requires_grad_(False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        use_cache: bool,
        logits_to_keep: int,
        output_hidden_states: bool,
    ):
        """Return text states through the public hidden-state contract.

        Args:
            input_ids: Long tensor of shape [batch, sequence].
            attention_mask: Long tensor of shape [batch, sequence].
            use_cache: Whether an autoregressive cache is requested.
            logits_to_keep: Language-model logit slice length.
            output_hidden_states: Whether hidden states are requested.

        Returns:
            Object containing one hidden-state tensor [batch, sequence, hidden].
        """
        return SimpleNamespace(hidden_states=(self.embedding(input_ids),))


def _items(accumulation, empty_rank, all_empty=False):
    items = []
    for index in range(2 * accumulation):
        count = 1 if index % 2 == 0 else 3
        questions = tuple(EncodedQuestion(f"q{i}", 1, (0, 1), ((1, 2), (2, 3)), ("a", "b")) for i in range(count))
        record = {
            "id": str(index),
            "state": "example",
            "questions": {q.question_id: {"type": "choice", "criteria": {"a": "A", "b": "B"}} for q in questions},
            "answers": {}
            if all_empty or (empty_rank and index % 2 == 0)
            else {q.question_id: "a" if index % 2 == 0 else "b" for q in questions},
        }
        items.append((record, EncodedRecord((1, 2, 3, 4), questions, str(index))))
    return items


def _recipe(items, output, rank, world, accumulation, max_norm, epochs=1):
    torch.manual_seed(7)
    recipe = TrainClefDecisionRecipe.__new__(TrainClefDecisionRecipe)
    recipe.model = TinyFrozenBackbone()
    recipe.head = ClefHeadConfig(hidden_size=8, width=8, routing_layers=1, layers=1, heads=2, feedforward=16).build()
    recipe.lexical = LexicalTable(list(range(16)), recipe.model.embedding.weight.detach().clone())
    recipe.pad_token_id = 0
    recipe.device = torch.device("cpu")
    recipe.device_mesh = None
    recipe.peft_config = None
    recipe.seed = 7
    recipe.tag = "sft"
    recipe.dist_env = SimpleNamespace(is_main=rank == 0)
    recipe.output_dir = output
    recipe.loss_config = DecisionLossConfig(label_smoothing=0.0, brier_weight=0.0, ordinal_weight=0.0)
    recipe.cfg = RecipeConfig(
        ConfigNode(
            {
                "train": {
                    "epochs": epochs,
                    "local_batch_size": 1,
                    "global_batch_size": world * accumulation,
                    "max_grad_norm": max_norm,
                    "val_every_steps": 100,
                },
            }
        )
    )
    recipe.checkpointer = CheckpointingConfig(checkpoint_dir=str(output), is_peft=False, save_consolidated=False).build(
        dp_rank=rank, tp_rank=0, pp_rank=0, process_group=dist.group.WORLD if dist.is_initialized() else None
    )
    recipe.splits = {"train": items}
    recipe.optimizer = []
    recipe.head_optimizer = torch.optim.SGD(recipe.head.parameters(), lr=0.05)
    recipe._get_dp_group_size = lambda: world
    recipe._get_dp_rank = lambda: rank
    recipe._dp_group = lambda: dist.group.WORLD
    return recipe


def _worker(rank, rendezvous, output, accumulation, empty_rank, max_norm, all_empty):
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )
    run = None
    try:
        run = _recipe(_items(accumulation, empty_rank, all_empty), Path(output), rank, 2, accumulation, max_norm)
        try:
            run._train()
        except ValueError as error:
            if not all_empty:
                raise
            assert "no supervised labels or explicit rewards on any DP rank" in str(error)
        else:
            assert not all_empty
            if rank == 0:
                torch.save(run.head.state_dict(), Path(output) / "actual.pt")
    finally:
        if run is not None:
            run.checkpointer.close()
        dist.destroy_process_group()


@pytest.mark.parametrize("accumulation,empty_rank,max_norm", [(1, False, 1e6), (2, False, 0.1), (1, True, 1e6)])
def test_replicated_head_dp_step_matches_single_process_question_mean(tmp_path, accumulation, empty_rank, max_norm):
    output = tmp_path / "run"
    output.mkdir()
    mp.spawn(
        _worker,
        args=(str(tmp_path / "rendezvous"), str(output), accumulation, empty_rank, max_norm, False),
        nprocs=2,
        join=True,
    )
    items = _items(accumulation, empty_rank)
    reference = _recipe(items, output, 0, 1, accumulation, max_norm)
    reference.checkpointer.close()
    nested_targets = [question_targets(record, clef_option_orders(encoded)) for record, encoded in items]
    targets = DecisionTargets.from_questions(record_ids=[record["id"] for record, _ in items], questions=nested_targets)
    nested_logits = reference._forward(items)
    identified = [
        [
            QuestionLogits(record["id"], target.question_id, target.option_ids, row)
            for target, row in zip(record_targets, record_logits, strict=True)
        ]
        for (record, _), record_targets, record_logits in zip(items, nested_targets, nested_logits, strict=True)
    ]
    assembled = assemble_decision_batch(logits=identified, targets=targets)
    loss, _ = reference.loss_config.build()(assembled.logits, assembled.targets)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(reference.head.parameters(), max_norm)
    reference.head_optimizer.step()
    actual = torch.load(output / "actual.pt", map_location="cpu", weights_only=True)
    for name, value in reference.head.state_dict().items():
        torch.testing.assert_close(actual[name], value, atol=2e-6, rtol=2e-5)


def test_all_feedback_free_ranks_fail_together_before_backward(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    mp.spawn(_worker, args=(str(tmp_path / "rendezvous"), str(output), 1, True, 1e6, True), nprocs=2, join=True)
    assert not (output / "step_00000001").exists()
    assert not (output / "actual.pt").exists()


def _rl_recipe(output, rank, world):
    items = _items(2, False)
    for record, encoded in items:
        record["rewards"] = {question.question_id: {"a": 0.0, "b": 1.0} for question in encoded.questions}
    recipe = _recipe(items, output, rank, world, 4 // world, 1.0, epochs=2)
    recipe.loss_config = DecisionLossConfig(mode="rl", kl_coef=0.2, anchor_weight=0.0)
    recipe.mode = "train"
    recipe.slice_keys = []
    # Three records force evaluation padding on two ranks. Give step zero perfect
    # validation labels so later updates cannot win and the main loop must restore it.
    validation = _items(2, False)[:3]
    with torch.no_grad():
        initial = recipe._forward(validation)
    for (record, encoded), logits in zip(validation, initial, strict=True):
        record["answers"] = {
            question.question_id: question.option_ids[int(row.argmax())]
            for question, row in zip(encoded.questions, logits, strict=True)
        }
    recipe.splits["validation"] = validation
    train = recipe._train

    def capture_trained_weights():
        train()
        if rank == 0:
            torch.save(recipe.head.state_dict(), output / "trained.pt")

    recipe._train = capture_trained_weights
    return recipe


def _rl_worker(rank, rendezvous, output):
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )
    recipe = None
    try:
        recipe = _rl_recipe(Path(output), rank, 2)
        recipe.run_train_validation_loop()
        summary = recipe.evaluate("validation")
        assert summary["overall"]["records"] == 3
        assert summary["overall"]["accuracy"] == 1.0
        if rank == 0:
            torch.save(recipe.head.state_dict(), Path(output) / "selected.pt")
    finally:
        if recipe is not None:
            recipe.checkpointer.close()
        dist.destroy_process_group()


def test_rl_reference_padded_evaluation_and_best_stage_match_single_process(tmp_path):
    distributed = tmp_path / "distributed"
    distributed.mkdir()
    mp.spawn(_rl_worker, args=(str(tmp_path / "rendezvous"), str(distributed)), nprocs=2, join=True)
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir()
    reference = _rl_recipe(reference_dir, 0, 1)
    try:
        reference.run_train_validation_loop()
        trained = torch.load(distributed / "trained.pt", weights_only=True)
        selected = torch.load(distributed / "selected.pt", weights_only=True)
        expected_trained = torch.load(reference_dir / "trained.pt", weights_only=True)
        assert any(not torch.equal(trained[name], selected[name]) for name in trained)
        for name, parameter in reference.head.state_dict().items():
            torch.testing.assert_close(trained[name], expected_trained[name], atol=3e-6, rtol=3e-5)
            torch.testing.assert_close(selected[name], parameter, atol=3e-6, rtol=3e-5)
        actual_summary = json.loads((distributed / "metrics_validation.json").read_text())
        expected_summary = json.loads((reference_dir / "metrics_validation.json").read_text())
        assert actual_summary == expected_summary
        rows = [json.loads(line) for line in (distributed / "training_log.jsonl").read_text().splitlines()]
        assert max(row.get("kl", 0.0) for row in rows) > 0
    finally:
        reference.checkpointer.close()
