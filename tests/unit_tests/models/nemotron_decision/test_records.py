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

"""One-decision conversion to SystemOne records, splitting, and per-question encoding with aligned targets."""

from __future__ import annotations

import pytest

from nemo_automodel.components.datasets.decision.records import question_targets, validate_record
from nemo_automodel.components.models.nemotron_decision.records import (
    convert_simple_line,
    encode_question,
    in_heldout,
)
from nemo_automodel.components.models.nemotron_decision.serving_prompt import DecisionContract
from tests.unit_tests.models.nemotron_decision.char_tokenizer import MASK, char_tokenizer, write_contract

LINE = {
    "id": "t-1",
    "state": {"ticket": "refund"},
    "question": "Which queue?",
    "options": {"billing": "Refunds", "technical": "Bugs", "sales": "Contracts"},
    "label": "billing",
    "soft_labels": {"billing": 3, "sales": 1},
    "group": "customer-1",
}


def test_simple_line_converts_to_a_valid_single_question_record():
    record = convert_simple_line(LINE)
    validate_record(record)
    assert record["questions"]["decision"]["criteria"] == LINE["options"]
    assert record["answers"] == {"decision": "billing"} and record["meta"] == {"group": "customer-1"}
    assert record["soft_labels"] == {"decision": {"billing": 0.75, "technical": 0.0, "sales": 0.25}}


@pytest.mark.parametrize(
    "change,error",
    [
        ({"label": "refunds"}, "not one of"),
        ({"options": {"billing": "x"}}, "at least 2"),
        ({"soft_labels": {"unknown": 1.0}}, "soft_labels"),
        ({"soft_labels": {"billing": -1.0, "sales": 2.0}}, "soft_labels"),
        ({"type": "ranking"}, "unknown type"),
        ({"type": "noul", "label": "maybe", "options": None}, "true/false"),
    ],
)
def test_simple_line_conversion_rejects_inconsistent_lines(change, error):
    with pytest.raises(ValueError, match=error):
        convert_simple_line({**LINE, **change})


def test_noul_and_score_conversion():
    noul = convert_simple_line(
        {"state": "s", "question": "q", "type": "noul", "label": "Yes", "soft_labels": {"no": 1, "yes": 3}}
    )
    validate_record(noul)
    assert noul["answers"]["decision"] == "true"
    assert noul["soft_labels"]["decision"] == {"false": 0.25, "true": 0.75}
    score = convert_simple_line(
        {"state": "s", "question": "q", "type": "score", "options": ["low", "high"], "label": 1}
    )
    validate_record(score)
    assert score["answers"]["decision"] == "1" and score["questions"]["decision"]["criteria"] == ["low", "high"]


def test_heldout_split_is_deterministic_and_keeps_groups_together():
    records = [convert_simple_line({**LINE, "id": f"t{i}", "group": f"g{i // 3}"}) for i in range(300)]
    first = [in_heldout(r, 0.2, "0") for r in records]
    assert first == [in_heldout(r, 0.2, "0") for r in records]
    by_group = {}
    for record, held in zip(records, first, strict=True):
        by_group.setdefault(record["meta"]["group"], set()).add(held)
    assert all(len(v) == 1 for v in by_group.values()) and 0 < sum(first) < len(records)


def test_targets_follow_the_shuffled_prompt_order(tmp_path):
    tok = char_tokenizer()
    contract = DecisionContract.from_pretrained(write_contract(tmp_path / "ckpt"))
    mask = tok.convert_tokens_to_ids(MASK)
    record = convert_simple_line(LINE)
    for seed in (None, 1, 2, 3, 4, 5):
        prompt = encode_question(
            tok, contract, record, "decision", mask_token_id=mask, max_tokens=4096, shuffle_seed=seed
        )
        (target,) = question_targets(record, {prompt.question_id: prompt.labels})
        assert target.option_ids == prompt.labels and prompt.labels[target.label] == "billing"
        expected = record["soft_labels"]["decision"]
        assert dict(zip(prompt.labels, target.soft_labels, strict=True)) == pytest.approx(expected)


def test_every_question_of_a_multi_question_record_is_encoded_on_its_own(tmp_path):
    tok = char_tokenizer()
    contract = DecisionContract.from_pretrained(write_contract(tmp_path / "ckpt"))
    mask = tok.convert_tokens_to_ids(MASK)
    record = {
        "id": "r1",
        "state": {"alert": "x"},
        "questions": {"a": {"type": "noul", "instructions": "?"}, "b": {"type": "noul", "instructions": "??"}},
        "answers": {"a": "true"},
    }
    validate_record(record)
    prompts = {
        qid: encode_question(tok, contract, record, qid, mask_token_id=mask, max_tokens=4096)
        for qid in record["questions"]
    }
    assert prompts["a"].input_ids != prompts["b"].input_ids
    targets = question_targets(record, {qid: p.labels for qid, p in prompts.items()})
    assert [t.label for t in targets] == [prompts["a"].labels.index("true"), -100]


def test_questions_this_model_cannot_read_are_not_encoded(tmp_path):
    tok = char_tokenizer()
    contract = DecisionContract.from_pretrained(write_contract(tmp_path / "ckpt"))
    mask = tok.convert_tokens_to_ids(MASK)
    record = convert_simple_line(LINE)
    image = {**record, "state": {"image": "data:..."}}
    images = {**record, "images": ["data:..."]}
    no_instructions = {
        **record,
        "questions": {"decision": {k: v for k, v in record["questions"]["decision"].items() if k != "instructions"}},
    }
    for bad in (image, images, no_instructions):
        assert encode_question(tok, contract, bad, "decision", mask_token_id=mask, max_tokens=4096) is None
    length = len(encode_question(tok, contract, record, "decision", mask_token_id=mask, max_tokens=4096).input_ids)
    assert encode_question(tok, contract, record, "decision", mask_token_id=mask, max_tokens=length + 1) is not None
    assert (
        encode_question(tok, contract, record, "decision", mask_token_id=mask, max_tokens=length) is None
    )  # the mask slot must fit too


def test_merge_by_state_keeps_a_states_questions_together_and_the_splits_unchanged():
    from nemo_automodel.components.models.nemotron_decision.records import merge_by_state

    lines = []
    for i in range(60):
        state = {"ticket": f"ticket {i // 3}"}
        lines.append({**LINE, "id": f"t{i}", "state": state, "group": f"g{i // 6}"})
    lines.append({**LINE, "id": "other-key-order", "state": {"b": 1, "a": 2}})
    lines.append({**LINE, "id": "same-values", "state": {"a": 2, "b": 1}})  # key order differs: not merged
    lines.append(
        {
            "id": "n1",
            "state": {"ticket": "ticket 0"},
            "question": "Urgent?",
            "type": "noul",
            "label": "yes",
            "group": "g0",
        }
    )
    records = [convert_simple_line(line) for line in lines]
    merged = merge_by_state(records)
    for record in merged:
        validate_record(record)
    first = merged[0]
    assert first["id"] == "t0" and list(first["questions"]) == ["t0", "t1", "t2", "n1"]
    assert first["answers"] == {"t0": "billing", "t1": "billing", "t2": "billing", "n1": "true"}
    assert set(first["soft_labels"]) == {"t0", "t1", "t2"} and first["meta"] == {"group": "g0"}
    assert len(merged) == 20 + 2 and sum(len(r["questions"]) for r in merged) == len(records)
    # Every question lands in the split its line would.
    by_line = {r["id"]: in_heldout(r, 0.3, "0") for r in records}
    for record in merged:
        assert {by_line[qid] for qid in record["questions"]} == {in_heldout(record, 0.3, "0")}
    with pytest.raises(ValueError, match="single-question"):
        merge_by_state(merged[:1] + [merged[0]])
    with pytest.raises(ValueError, match="duplicate"):
        merge_by_state([records[0], records[0]])


def test_json_sep_questions_of_a_record_share_the_state_prefix(tmp_path):
    from nemo_automodel.components.models.nemotron_decision.packed import shared_state_length
    from tests.unit_tests.models.nemotron_decision.char_tokenizer import STATE_END

    tok = char_tokenizer()
    contract = DecisionContract.from_pretrained(write_contract(tmp_path / "ckpt"))
    mask = tok.convert_tokens_to_ids(MASK)
    record = {
        "id": "r1",
        "state": {"alert": "x"},
        "questions": {
            "a": {"type": "noul", "instructions": "?"},
            "b": {"type": "score", "instructions": "How bad?", "criteria": ["low", "high"]},
        },
    }
    prompts = [
        encode_question(tok, contract, record, qid, mask_token_id=mask, max_tokens=4096, prompt_format="json-sep")
        for qid in record["questions"]
    ]
    state_length = shared_state_length([p.input_ids for p in prompts], tok.convert_tokens_to_ids(STATE_END))
    assert all(p.state_length == state_length for p in prompts)
    plain = encode_question(tok, contract, record, "a", mask_token_id=mask, max_tokens=4096)
    assert len(prompts[0].input_ids) == len(plain.input_ids) + 1  # json-sep adds exactly the state-end token
