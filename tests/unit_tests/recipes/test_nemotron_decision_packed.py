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

"""Nemotron Decision recipe with packed reads: rows, per-question loss, multi-question evaluation, single-read eval."""

from __future__ import annotations

import dataclasses

import pytest
import torch
from transformers.models.ministral3.configuration_ministral3 import Ministral3Config

from nemo_automodel.components.datasets.decision.records import validate_record
from nemo_automodel.components.loss.decision import DecisionLossConfig
from nemo_automodel.components.loss.decision_batch import QuestionLogits, assemble_decision_batch
from nemo_automodel.components.models.nemotron_decision.model import NemotronForDecision
from nemo_automodel.components.models.nemotron_decision.scorer import gather_head_rows
from nemo_automodel.components.models.nemotron_decision.serving_prompt import DecisionContract, readout_token_ids
from nemo_automodel.recipes.decision.train_nemotron_decision import TrainNemotronDecisionRecipe, _PackedRow
from tests.unit_tests.models.nemotron_decision.char_tokenizer import MASK, STATE_END, char_tokenizer, write_contract

pytestmark = pytest.mark.cpu_only


def _records():
    records = []
    for i in range(4):
        record = {
            "id": f"r{i}",
            "state": {"ticket": f"ticket {i}: refund of ${100 * i}"},
            "questions": {
                "urgent": {"type": "noul", "instructions": "Urgent?"},
                "team": {"type": "choice", "instructions": "Team?", "criteria": {"billing": "Refunds", "tech": "Bugs"}},
                "mood": {"type": "score", "instructions": "Mood?", "criteria": ["calm", "upset", "angry"]},
            },
            "answers": {"urgent": "true" if i % 2 else "false", "team": "billing", "mood": str(i % 3)},
        }
        if i % 2:
            record["soft_labels"] = {"team": {"billing": 0.7, "tech": 0.3}}
        validate_record(record)
        records.append(record)
    return records


@pytest.fixture
def recipe(tmp_path):
    torch.manual_seed(0)
    recipe = TrainNemotronDecisionRecipe.__new__(TrainNemotronDecisionRecipe)
    recipe.tokenizer = char_tokenizer()
    recipe.contract = DecisionContract.from_pretrained(write_contract(tmp_path / "ckpt"))
    recipe.mask_token_id = recipe.tokenizer.convert_tokens_to_ids(MASK)
    recipe.max_tokens = 4096
    recipe.packed_read = True
    recipe.prompt_format = "json-sep"
    recipe.state_end_token = STATE_END
    recipe.state_end_token_id = recipe.tokenizer.convert_tokens_to_ids(STATE_END)
    recipe.loss_config = DecisionLossConfig(
        mode="sft", label_smoothing=0.0, brier_weight=0.0, ordinal_weight=0.0, soft_label_kl_weight=0.0
    )
    recipe.loss_fn = recipe.loss_config.build()
    recipe.seed = 101
    recipe.slice_keys = []
    recipe.device = torch.device("cpu")
    recipe._get_dp_group_size = lambda: 1
    recipe._get_dp_rank = lambda: 0
    config = Ministral3Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=4096,
        mask_token_id=recipe.mask_token_id,
        architectures=["NemotronForDecision"],
        attn_implementation="sdpa",
    )
    recipe.model = NemotronForDecision(config)
    recipe.model.get_output_embeddings().weight.requires_grad_(False)
    tokens = list(readout_token_ids(recipe.tokenizer, recipe.contract))
    rows = gather_head_rows(recipe.model, tokens)
    recipe.head_rows = {token: rows[i] for i, token in enumerate(tokens)}
    return recipe


def test_packed_rows_keep_a_records_supervised_questions_in_one_read(recipe):
    record = {
        "id": "r",
        "state": {"alert": "x"},
        "questions": {
            "gold": {"type": "noul", "instructions": "Escalate?"},
            "soft": {"type": "noul", "instructions": "Page?"},
            "reward": {"type": "noul", "instructions": "Close?"},
            "unsupervised": {"type": "noul", "instructions": "Notify?"},
            "unreadable": {"type": "noul"},
        },
        "answers": {"gold": "true"},
        "soft_labels": {"soft": {"true": 0.9, "false": 0.1}},
        "rewards": {"reward": {"true": 1.0, "false": 0.0}},
    }
    validate_record(record)
    (row,), skipped = recipe._encode_packed_split([record], train=True)
    assert [p.question_id for p in row.prompts] == ["gold", "soft"] and skipped == 1
    assert all(t.option_ids == p.labels for p, t in zip(row.prompts, row.targets, strict=True))
    read = recipe._pack(row.prompts)
    assert len(read.slots) == 2 and read.input_ids[read.state_length - 1] == recipe.state_end_token_id
    (row,), _ = recipe._encode_packed_split([record], train=False)
    assert [p.question_id for p in row.prompts] == ["gold", "reward"]
    # A read longer than data.max_tokens splits into reads of the same state; no question is lost.
    recipe.max_tokens = len(row.prompts[0].input_ids) + 2
    rows, skipped = recipe._encode_packed_split([record], train=False)
    assert [[p.question_id for p in r.prompts] for r in rows] == [["gold"], ["reward"]] and skipped == 1


def _predictions(recipe, rows):
    recipe.decisions = {"test": rows}
    summary, predictions = recipe.evaluate("test")
    return summary, {(p["record_id"], p["question_id"]): p["probs"] for p in predictions}


def test_multi_question_eval_equals_each_question_in_its_own_pack_of_the_same_state(recipe):
    rows, _ = recipe._encode_packed_split(_records(), train=False)
    assert [len(row.prompts) for row in rows] == [3, 3, 3, 3]
    alone = [
        _PackedRow(row.record, (prompt,), (target,))
        for row in rows
        for prompt, target in zip(row.prompts, row.targets, strict=True)
    ]
    summary, together = _predictions(recipe, rows)
    summary_alone, separate = _predictions(recipe, alone)
    assert together.keys() == separate.keys() and len(together) == 12
    for key, probs in together.items():
        torch.testing.assert_close(torch.tensor(probs), torch.tensor(separate[key]), atol=1e-6, rtol=1e-5)
    # Metrics per question and per state: exact_match is the share of states with every question right.
    assert summary["overall"]["n"] == 12 and summary["overall"]["records"] == 4
    assert summary["overall"]["exact_match"] == pytest.approx(summary_alone["overall"]["exact_match"])
    assert set(summary["questions"]) == {"urgent", "team", "mood"}


def test_the_same_weights_also_evaluate_one_question_per_read(recipe):
    _, packed = _predictions(recipe, recipe._encode_packed_split(_records(), train=False)[0])
    recipe.packed_read = False
    decisions, _ = recipe._encode_split(_records(), train=False)
    _, single = _predictions(recipe, decisions)
    assert single.keys() == packed.keys()
    # Single reads let the state attend to the question, so the reads differ; both are valid distributions.
    assert any(not torch.allclose(torch.tensor(single[k]), torch.tensor(packed[k])) for k in packed)
    assert all(sum(p) == pytest.approx(1.0) for p in single.values())


def _question_loss(recipe, row, prompt, logits):
    targets = recipe._row_targets(row.record, [recipe._target(row.record, prompt)])
    scores = [[QuestionLogits(str(row.record["id"]), prompt.question_id, prompt.labels, logits)]]
    return recipe.loss_fn(assemble_decision_batch(logits=scores, targets=targets).logits, targets)[0]


@pytest.mark.parametrize(
    "loss",
    [
        {"soft_label_kl_weight": 0.0},  # soft labels replace the gold target where present
        {"soft_label_kl_weight": 0.5},  # gold cross-entropy plus KL to the soft labels
        {"soft_label_kl_weight": 0.0, "label_smoothing": 0.1, "brier_weight": 1.0, "ordinal_weight": 0.5},
    ],
    ids=["soft", "kl", "smoothed-brier-rps"],
)
def test_packed_row_loss_is_the_mean_over_its_questions_and_a_step_averages_rows(recipe, loss):
    recipe.loss_config = dataclasses.replace(recipe.loss_config, **loss)
    recipe.loss_fn = recipe.loss_config.build()
    rows, _ = recipe._encode_packed_split(_records(), train=True)
    window = [0, 1]
    with torch.no_grad():
        expected = torch.stack(
            [
                torch.stack(
                    [
                        _question_loss(recipe, rows[i], prompt, logits)
                        for prompt, logits in zip(rows[i].prompts, recipe._packed_logits(rows[i]), strict=True)
                    ]
                ).mean()
                for i in window
            ]
        ).mean()
    recipe.model.zero_grad()
    got = recipe._packed_window(rows, window, rank=0, world=1, epoch=0, canonical_fraction=1.0)
    torch.testing.assert_close(got, expected, atol=1e-6, rtol=1e-5)
    grads = [p.grad for p in recipe.model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads) and any(g.abs().sum() > 0 for g in grads)
    assert recipe.model.get_output_embeddings().weight.grad is None


def test_hard_labels_only(recipe):
    records = _records()
    for record in records:
        record.pop("soft_labels", None)
    rows, _ = recipe._encode_packed_split(records, train=True)
    got = recipe._packed_window(rows, [2, 3], rank=0, world=1, epoch=0, canonical_fraction=1.0)
    with torch.no_grad():
        nll = [
            torch.stack(
                [
                    -logits.log_softmax(-1)[target.label]
                    for logits, target in zip(recipe._packed_logits(rows[i]), rows[i].targets, strict=True)
                ]
            ).mean()
            for i in (2, 3)
        ]
    torch.testing.assert_close(got, torch.stack(nll).mean(), atol=1e-6, rtol=1e-5)


def test_option_shuffle_recodes_each_question_and_keeps_the_shared_state(recipe):
    rows, _ = recipe._encode_packed_split(_records(), train=True)
    shuffled = [recipe._train_prompts(rows[0], seed, 0.0) for seed in range(8)]
    assert any(p.labels != q.labels for prompts in shuffled for p, q in zip(prompts, rows[0].prompts, strict=True))
    for prompts in shuffled:
        read = recipe._pack(prompts)
        assert read.state_length == rows[0].prompts[0].state_length
        targets = [recipe._target(rows[0].record, p) for p in prompts]
        assert all(t.option_ids == p.labels for t, p in zip(targets, prompts, strict=True))
    assert recipe._train_prompts(rows[0], 0, 1.0) == rows[0].prompts


def test_reads_default_to_one_question_per_read_with_json_prompts():
    recipe = TrainNemotronDecisionRecipe.__new__(TrainNemotronDecisionRecipe)
    assert (recipe.packed_read, recipe.prompt_format, recipe.read_layout) == (False, "json", None)
