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

"""CPU unit tests for decision-model post-training utilities."""

import json
import math
from pathlib import Path

import pytest
import torch

from nemo_automodel.components.datasets.decision import records as schema
from nemo_automodel.components.eval.decision import metrics
from nemo_automodel.components.loss import decision as losses
from nemo_automodel.components.loss import decision_batch as batching
from nemo_automodel.components.models.clef import joint_schema_head as clef
from nemo_automodel.components.models.clm import heads as clm
from nemo_automodel.components.models.clm import prompting as clm_prompting

QUESTIONS = {
    "verdict": {
        "type": "choice",
        "instructions": "What is the verdict?",
        "criteria": {"malicious": "Real attack.", "benign_authorized": "Authorized.", "false_positive": "Misfire."},
    },
    "severity": {"type": "score", "instructions": "How severe?", "criteria": ["Info", "Low", "Medium", "High"]},
    "needs_human": {"type": "noul", "instructions": "Does a human need to review this?"},
}


def make_record(rid="r1", verdict="malicious", severity="2", needs_human="true"):
    return {
        "id": rid,
        "state": {"alert": {"title": "PsExec to 3 hosts", "events": ["a", "b"]}, "asset": {"criticality": "high"}},
        "questions": QUESTIONS,
        "answers": {"verdict": verdict, "severity": severity, "needs_human": needs_human},
        "rewards": {
            "verdict": {"malicious": 1.0, "benign_authorized": -1.0, "false_positive": -1.0},
            "severity": {"0": 0.25, "1": 0.5, "2": 1.0, "3": 0.75},
            "needs_human": {"true": 1.0, "false": -0.5},
        },
        "meta": {"evidence": "clear"},
    }


class CharTokenizer:
    """Deterministic offline tokenizer: one id per character (enough for span bookkeeping)."""

    pad_token_id = 0

    class _Out:
        def __init__(self, ids):
            self.input_ids = ids

    def __call__(self, text, add_special_tokens=False):
        return self._Out([1 + (ord(c) % 250) for c in text])


# ---------------------------------------------------------------------------------------------------------------------
def test_validate_and_targets_follow_model_order():
    record = make_record()
    schema.validate_record(record)
    order = {"verdict": ["false_positive", "malicious", "benign_authorized"], "needs_human": ["false", "true"]}
    targets = schema.question_targets(record, order)
    assert [t.question_id for t in targets] == ["verdict", "needs_human"]
    assert targets[0].label == 1 and targets[0].rewards == (-1.0, 1.0, -1.0)
    assert targets[1].label == 1 and targets[1].rewards == (-0.5, 1.0)


def test_validate_rejects_unknown_answer():
    record = make_record(verdict="ransomware")
    with pytest.raises(ValueError):
        schema.validate_record(record)


@pytest.mark.parametrize(
    "rewards",
    [
        {},
        {"malicious": 1.0},
        {"malicious": float("nan"), "benign_authorized": 0.0, "false_positive": 0.0},
        {"malicious": "bad", "benign_authorized": 0.0, "false_positive": 0.0},
    ],
)
def test_validate_rejects_malformed_reward_tables_before_training(rewards: dict[str, object]) -> None:
    record = make_record()
    record["rewards"]["verdict"] = rewards
    with pytest.raises(ValueError, match="record 'r1', question 'verdict': rewards"):
        schema.validate_record(record)


def test_validate_rejects_rewards_for_an_unknown_question() -> None:
    record = make_record()
    record["rewards"]["unknown"] = {"option": 1.0}
    with pytest.raises(ValueError, match="rewards for unknown question"):
        schema.validate_record(record)


def test_sft_loss_prefers_gold_and_is_finite():
    record = make_record()
    order = {q: schema.canonical_option_ids(QUESTIONS[q]) for q in QUESTIONS}
    t = batching.DecisionTargets.from_questions(
        record_ids=[record["id"]], questions=[schema.question_targets(record, order)]
    )
    logits = torch.zeros(t.option_mask.shape, requires_grad=True)
    loss, m = losses.DecisionLossConfig(mode="sft").build()(logits, t)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    # gradient descent raises the gold logit
    for i, label in enumerate(t.labels.tolist()):
        assert logits.grad[i, label] < 0
    # padded options get no gradient
    assert (logits.grad[~t.option_mask] == 0).all()


def test_expected_reward_gradient_is_exact_policy_gradient():
    t = batching.DecisionTargets(
        labels=torch.tensor([0]),
        option_mask=torch.tensor([[True, True, True]]),
        question_types=torch.tensor([1]),
        rewards=torch.tensor([[1.0, -1.0, 0.5]]),
        reward_mask=torch.tensor([True]),
    )
    logits = torch.tensor([[0.3, -0.2, 0.1]], requires_grad=True)
    cfg = losses.DecisionLossConfig(mode="rl", rl_objective="expected_reward", kl_coef=0.0, anchor_weight=0.0)
    loss, m = cfg.build()(logits, t)
    loss.backward()
    p = torch.softmax(logits.detach(), -1)[0]
    r = t.rewards[0]
    expected_grad = -(p * (r - (p * r).sum()))
    assert torch.allclose(logits.grad[0], expected_grad, atol=1e-6)
    assert math.isclose(m["expected_reward"].item(), (p * r).sum().item(), rel_tol=1e-5)


def test_grpo_matches_expected_reward_in_expectation():
    t = batching.DecisionTargets(
        labels=torch.tensor([0]),
        option_mask=torch.tensor([[True, True, True, False]]),
        question_types=torch.tensor([1]),
        rewards=torch.tensor([[1.0, -1.0, 0.5, 0.0]]),
        reward_mask=torch.tensor([True]),
    )
    logits = torch.tensor([[0.3, -0.2, 0.1, 0.0]], requires_grad=True)
    cfg = losses.DecisionLossConfig(
        mode="rl", rl_objective="grpo", num_samples=20000, normalize_advantages=False, kl_coef=0.0, anchor_weight=0.0
    )
    loss, _ = cfg.build()(logits, t, generator=torch.Generator().manual_seed(0))
    loss.backward()
    p = torch.softmax(logits.detach()[0, :3], -1)
    r = t.rewards[0, :3]
    exact = -(p * (r - (p * r).sum()))
    assert torch.allclose(logits.grad[0, :3], exact, atol=2e-2)
    assert logits.grad[0, 3] == 0


def test_kl_to_identical_reference_is_zero():
    t = batching.DecisionTargets.from_questions(
        record_ids=["r1"],
        questions=[
            schema.question_targets(make_record(), {q: schema.canonical_option_ids(QUESTIONS[q]) for q in QUESTIONS})
        ],
    )
    logits = torch.randn(t.option_mask.shape)
    ref = losses.masked_log_softmax(logits, t.option_mask)
    t.ref_logprobs = ref
    _, m = losses.DecisionLossConfig(mode="rl").build()(logits, t)
    assert abs(m["kl"].item()) < 1e-6


def test_metrics_summary():
    ev = metrics.DecisionEvaluator(slice_keys=["evidence"])
    ev.add("r1", "verdict", ["a", "b"], [0.9, 0.1], 0, [1.0, -1.0], {"evidence": "clear"})
    ev.add("r1", "severity", ["0", "1"], [0.4, 0.6], 0, [1.0, 0.5], {"evidence": "clear"})
    ev.add("r2", "verdict", ["a", "b"], [0.2, 0.8], 1, [-1.0, 1.0], {"evidence": "subtle"})
    s = ev.summary()
    assert s["questions"]["verdict"]["accuracy"] == 1.0
    assert s["overall"]["exact_match"] == 0.5
    assert math.isclose(s["overall"]["reward"], (1.0 + 0.5 + 1.0) / 3)


# ---------------------------------------------------------------------------------------------------------------------
def _clm_reference() -> dict:
    path = Path(__file__).resolve().parents[1] / "models/clm/fixtures/reference.json"
    return json.loads(path.read_text())


def test_clm_rendering_matches_pinned_reference() -> None:
    fixture = _clm_reference()
    assert fixture["provenance"]["revision"] == "d5f9ef0fd9bde185df0ceaad4f4ecc6cfe8c34f6"
    assert clm_prompting.clm_pairs(fixture["record"]) == {k: tuple(v) for k, v in fixture["rendered_pairs"].items()}


def test_clm_heads_roundtrip_and_logits(tmp_path):
    cfg = {"width": 64, "depth": 3, "projection_dim": 16, "layernorm": True, "residual": True, "hidden_size": 32}
    heads = clm.CLMHeadConfig(
        hidden_size=cfg["hidden_size"],
        width=cfg["width"],
        depth=cfg["depth"],
        projection_dim=cfg["projection_dim"],
        activation="gelu",
        layer_norm=cfg["layernorm"],
        residual=cfg["residual"],
    ).build()
    path = tmp_path / "heads.pt"
    clm.save_clm_heads(heads, path)
    loaded = clm.load_clm_heads(path)
    states, options = torch.randn(2, 32), torch.randn(5, 32)
    a = heads.question_logits(states, options, [3, 2])
    b = loaded.question_logits(states, options, [3, 2])
    assert [x.shape[0] for x in a] == [3, 2]
    for x, y in zip(a, b):
        assert torch.allclose(x, y, atol=1e-6)
    assert all(x.abs().max() <= heads.scale + 1e-4 for x in a)


def test_clm_heads_match_pinned_reference_forward_and_backward() -> None:
    fixture = _clm_reference()
    cfg = fixture["config"]
    heads = clm.CLMHeadConfig(
        hidden_size=cfg["hidden_size"],
        width=cfg["width"],
        depth=cfg["depth"],
        projection_dim=cfg["projection_dim"],
        activation=cfg["activation"],
        layer_norm=cfg["layernorm"],
        residual=cfg["residual"],
    ).build(initial_temperature=math.exp(-fixture["logit_scale"]))
    for name in ("state_head", "action_head"):
        getattr(heads, name).load_state_dict({key: torch.tensor(value) for key, value in fixture[name].items()})
    states = torch.tensor(fixture["state_embeddings"], requires_grad=True)
    actions = torch.tensor(fixture["action_embeddings"], requires_grad=True)
    torch.testing.assert_close(
        heads.project_states(states), torch.tensor(fixture["state_projections"]), atol=1e-6, rtol=1e-5
    )
    torch.testing.assert_close(
        heads.project_actions(actions), torch.tensor(fixture["action_projections"]), atol=1e-6, rtol=1e-5
    )
    logits = heads(states, actions)
    torch.testing.assert_close(logits, torch.tensor(fixture["logits"]), atol=2e-6, rtol=1e-5)
    (logits * torch.tensor(fixture["upstream_gradient"])).sum().backward()
    torch.testing.assert_close(states.grad, torch.tensor(fixture["state_gradients"]), atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(actions.grad, torch.tensor(fixture["action_gradients"]), atol=2e-6, rtol=1e-5)


def test_embedding_table_and_clm_batch():
    records = [make_record("a"), make_record("b", verdict="false_positive")]
    texts = clm.collect_clm_texts(records)
    table = clm.EmbeddingTable(texts, torch.randn(len(texts), 32))
    states, options, counts, orders = clm.clm_batch(records, table)
    assert states.shape == (6, 32) and options.shape[0] == sum(counts) == 2 * (3 + 4 + 2)
    assert orders[0]["needs_human"] == ["false", "true"]


# ---------------------------------------------------------------------------------------------------------------------
def tiny_head():
    torch.manual_seed(0)
    return clef.ClefHeadConfig(hidden_size=32, width=16, routing_layers=1, layers=1, heads=2, feedforward=32).build()


def test_clef_encode_head_and_targets():
    tok = CharTokenizer()
    records = [make_record("a"), make_record("b", verdict="benign_authorized", severity="0", needs_human="false")]
    encoded = [clef.encode_record(tok, r, max_length=4096) for r in records]
    batch = clef.collate_records(encoded, tok.pad_token_id, torch.device("cpu"))
    head = tiny_head()
    vocab = torch.randn(256, 32)
    hidden = torch.randn(batch["input_ids"].shape + (32,))
    out = head(hidden, batch["input_ids"], batch["attention_mask"], batch["records"], vocab)
    assert [len(x) for x in out] == [3, 3]
    order = clef.clef_option_orders(encoded[0])
    assert order["verdict"] == ("benign_authorized", "false_positive", "malicious")  # Clef sorts choice keys
    assert order["needs_human"] == ("true", "false")
    targets = [schema.question_targets(r, clef.clef_option_orders(e)) for r, e in zip(records, encoded)]
    t = batching.DecisionTargets.from_questions(record_ids=[r["id"] for r in records], questions=targets)
    identified = [
        [
            batching.QuestionLogits(record["id"], target.question_id, target.option_ids, row)
            for target, row in zip(record_targets, record_logits, strict=True)
        ]
        for record, record_targets, record_logits in zip(records, targets, out, strict=True)
    ]
    batch = batching.assemble_decision_batch(logits=identified, targets=t)
    loss, _ = losses.DecisionLossConfig(mode="sft").build()(batch.logits, batch.targets)
    loss.backward()
    assert head.prior_logit_scale.grad is not None


def test_clef_lexical_table_matches_dense_lookup():
    tok = CharTokenizer()
    encoded = [clef.encode_record(tok, make_record())]
    batch = clef.collate_records(encoded, tok.pad_token_id, torch.device("cpu"))
    head = tiny_head().eval()
    vocab = torch.randn(256, 32)
    ids = clef.option_token_ids(encoded)
    table = clef.LexicalTable(ids, vocab[ids])
    hidden = torch.randn(batch["input_ids"].shape + (32,))
    with torch.no_grad():
        a = head(hidden, batch["input_ids"], batch["attention_mask"], batch["records"], vocab)
        b = head(hidden, batch["input_ids"], batch["attention_mask"], batch["records"], table)
    for x, y in zip(a[0], b[0]):
        assert torch.allclose(x, y)


def test_clef_head_save_load_roundtrip(tmp_path):
    head = tiny_head()
    clef.save_joint_head(head, tmp_path, dtype=torch.float32)
    loaded = clef.load_joint_head(tmp_path)
    for (k, v), (k2, v2) in zip(head.state_dict().items(), loaded.state_dict().items()):
        assert k == k2 and torch.equal(v, v2)


@pytest.mark.parametrize("temperature", [0.5, 2.0])
def test_grpo_temperature_scores_the_sampling_policy(temperature):
    targets = batching.DecisionTargets(
        labels=torch.tensor([0]),
        option_mask=torch.tensor([[True, True, True]]),
        question_types=torch.tensor([1]),
        rewards=torch.tensor([[1.0, -1.0, 0.5]]),
        reward_mask=torch.tensor([True]),
    )
    logits = torch.tensor([[0.3, -0.2, 0.1]], requires_grad=True)
    cfg = losses.DecisionLossConfig(
        mode="rl",
        rl_objective="grpo",
        num_samples=20000,
        sample_temperature=temperature,
        normalize_advantages=False,
        kl_coef=0.0,
        anchor_weight=0.0,
    )
    loss, _ = cfg.build()(logits, targets, generator=torch.Generator().manual_seed(7))
    loss.backward()
    probabilities = torch.softmax(logits.detach()[0] / temperature, -1)
    rewards = targets.rewards[0]
    expected = -(probabilities * (rewards - (probabilities * rewards).sum())) / temperature
    torch.testing.assert_close(logits.grad[0], expected, atol=2e-2, rtol=0)


@pytest.mark.parametrize(
    "settings",
    [
        {"label_smoothing": -0.1},
        {"label_smoothing": 1.1},
        {"rl_objective": "grpo", "num_samples": 1},
        {"sample_temperature": 0.0},
    ],
)
def test_invalid_objective_settings_are_rejected(settings):
    with pytest.raises(ValueError):
        losses.DecisionLossConfig(**settings).build()


def test_rl_without_labels_or_rewards_does_not_invent_option_zero():
    logits = torch.zeros(1, 2, requires_grad=True)
    targets = batching.DecisionTargets(
        labels=torch.tensor([-100]), option_mask=torch.ones(1, 2, dtype=torch.bool), question_types=torch.tensor([1])
    )
    loss, metrics = losses.DecisionLossConfig(mode="rl").build()(logits, targets)
    loss.backward()
    torch.testing.assert_close(loss, torch.tensor(0.0), rtol=0, atol=0)
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits), rtol=0, atol=0)
    assert metrics["normalization_weight"] == 0


def test_rl_explicit_reward_learns_without_gold_label():
    logits = torch.zeros(1, 2, requires_grad=True)
    targets = batching.DecisionTargets(
        labels=torch.tensor([-100]),
        option_mask=torch.ones(1, 2, dtype=torch.bool),
        question_types=torch.tensor([1]),
        rewards=torch.tensor([[-1.0, 1.0]]),
        reward_mask=torch.tensor([True]),
    )
    loss, _ = losses.DecisionLossConfig(mode="rl", kl_coef=0.0).build()(logits, targets)
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.tensor([[0.5, -0.5]]), rtol=0, atol=0)


def test_rl_mixed_feedback_batch_matches_feedback_only_loss_and_gradient():
    cfg = losses.DecisionLossConfig(mode="rl", kl_coef=0.0, anchor_weight=0.2)
    mixed_logits = torch.tensor([[0.9, -0.4], [0.2, 0.7]], requires_grad=True)
    single_logits = mixed_logits.detach()[1:].clone().requires_grad_()
    mixed_targets = batching.DecisionTargets(
        labels=torch.tensor([-100, 1]),
        option_mask=torch.ones(2, 2, dtype=torch.bool),
        question_types=torch.tensor([1, 1]),
        weights=torch.tensor([17.0, 3.0]),
    )
    single_targets = batching.DecisionTargets(
        labels=torch.tensor([1]),
        option_mask=torch.ones(1, 2, dtype=torch.bool),
        question_types=torch.tensor([1]),
        weights=torch.tensor([3.0]),
    )
    mixed_loss, _ = cfg.build()(mixed_logits, mixed_targets)
    single_loss, _ = cfg.build()(single_logits, single_targets)
    mixed_loss.backward()
    single_loss.backward()
    torch.testing.assert_close(mixed_loss, single_loss, rtol=0, atol=0)
    torch.testing.assert_close(mixed_logits.grad[0], torch.zeros(2), rtol=0, atol=0)
    torch.testing.assert_close(mixed_logits.grad[1:], single_logits.grad, rtol=0, atol=0)


@pytest.mark.parametrize("mask", [torch.tensor([[False, False]]), torch.tensor([[1, 1]]), torch.tensor([[True]])])
def test_invalid_option_masks_fail_before_softmax(mask):
    logits = torch.zeros(1, 2, requires_grad=True)
    targets = batching.DecisionTargets(labels=torch.tensor([0]), option_mask=mask, question_types=torch.tensor([1]))
    with pytest.raises(ValueError, match="option"):
        losses.DecisionLossConfig().build()(logits, targets)
    assert logits.grad is None


@pytest.mark.parametrize(
    "kind,criteria",
    [("choice", ["A", "B"]), ("choice", {}), ("score", {"0": "Low", "1": "High"}), ("score", ["Only"])],
)
def test_malformed_criteria_fail_before_model_specific_encoding(kind, criteria):
    record = {"id": "invalid", "state": "text", "questions": {"pick": {"type": kind, "criteria": criteria}}}
    with pytest.raises(ValueError, match="criteria"):
        schema.validate_record(record)


@pytest.mark.parametrize(
    "overrides",
    [
        {"probabilities": []},
        {"probabilities": [1.0]},
        {"probabilities": [float("nan"), 0.5]},
        {"probabilities": [-0.1, 1.1]},
        {"probabilities": [0.2, 0.3]},
        {"label": 2},
        {"rewards": [1.0]},
        {"rewards": [1.0, float("inf")]},
    ],
)
def test_metrics_reject_misaligned_or_invalid_decisions(overrides):
    evaluator = metrics.DecisionEvaluator()
    inputs = dict(record_id="r", question_id="q", option_ids=["a", "b"], probabilities=[0.8, 0.2], label=0)
    inputs.update(overrides)
    with pytest.raises(ValueError):
        evaluator.add(**inputs)
    assert evaluator.predictions() == []


def test_metrics_reject_truncated_batch():
    with pytest.raises(ValueError):
        metrics.DecisionEvaluator().add_batch([{"id": "r"}], [], [])
    target = schema.QuestionTarget("q", 0, ("a", "b"), 0, None)
    with pytest.raises(ValueError):
        metrics.DecisionEvaluator().add_batch([{"id": "r"}], [[target]], [[]])
