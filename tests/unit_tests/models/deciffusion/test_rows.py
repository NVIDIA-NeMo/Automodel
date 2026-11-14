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

"""Training-row conversion, splitting and encoding (labels, soft targets, rejection reasons)."""

from __future__ import annotations

import pytest

from nemo_automodel.components.models.deciffusion.rows import (
    convert_simple_row,
    encode_row,
    in_heldout,
    rows_from_systemone,
)
from tests.unit_tests.models.deciffusion.char_tokenizer import CharTokenizer

LINE = {
    "id": "t-1",
    "state": {"ticket": "refund"},
    "question": "Which queue?",
    "options": {"billing": "Refunds", "technical": "Bugs", "sales": "Contracts"},
    "label": "billing",
    "soft_labels": {"billing": 3, "sales": 1},
    "group": "customer-1",
}


def test_simple_row_conversion_normalizes_soft_labels_over_all_options():
    row = convert_simple_row(LINE)
    assert row["input"]["questions"]["decision"]["criteria"] == LINE["options"]
    assert row["reference"] == {"target": "billing"} and row["group"] == "customer-1"
    assert row["soft_labels"] == {"billing": 0.75, "technical": 0.0, "sales": 0.25}


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
def test_simple_row_conversion_rejects_inconsistent_lines(change, error):
    with pytest.raises(ValueError, match=error):
        convert_simple_row({**LINE, **change})


def test_noul_and_score_conversion():
    noul = convert_simple_row(
        {"state": "s", "question": "q", "type": "noul", "label": "Yes", "soft_labels": {"no": 1, "yes": 3}}
    )
    assert noul["reference"]["target"] == "true" and noul["soft_labels"] == {"false": 0.25, "true": 0.75}
    score = convert_simple_row({"state": "s", "question": "q", "type": "score", "options": ["low", "high"], "label": 1})
    assert score["reference"]["target"] == "1" and score["input"]["questions"]["decision"]["criteria"] == [
        "low",
        "high",
    ]


def test_systemone_records_flatten_to_one_row_per_answered_question_in_one_group():
    record = {
        "id": "r1",
        "state": {"alert": "x"},
        "questions": {"a": {"type": "noul", "instructions": "?"}, "b": {"type": "noul", "instructions": "??"}},
        "answers": {"a": "True"},
    }
    rows = rows_from_systemone(record)
    assert [r["id"] for r in rows] == ["r1/a"] and rows[0]["reference"]["target"] == "true" and rows[0]["group"] == "r1"


def test_heldout_split_is_deterministic_and_keeps_groups_together():
    rows = [convert_simple_row({**LINE, "id": f"t{i}", "group": f"g{i // 3}"}) for i in range(300)]
    first = [in_heldout(r, 0.2, "0") for r in rows]
    assert first == [in_heldout(r, 0.2, "0") for r in rows]
    by_group = {}
    for row, held in zip(rows, first, strict=True):
        by_group.setdefault(row["group"], set()).add(held)
    assert all(len(v) == 1 for v in by_group.values()) and 0 < sum(first) < len(rows)


def test_encoded_gold_and_soft_targets_follow_the_shuffled_prompt_order():
    tok = CharTokenizer()
    row = convert_simple_row(LINE)
    for seed in (None, 1, 2, 3, 4, 5):
        example = encode_row(tok, row, max_tokens=4096, shuffle_seed=seed)
        labels = example.prompt.labels
        assert labels[example.gold] == "billing"
        assert dict(zip(labels, example.soft, strict=True)) == pytest.approx(row["soft_labels"])


def test_soft_gold_mask_drops_soft_labels_that_disagree_with_gold():
    tok = CharTokenizer()
    row = convert_simple_row({**LINE, "soft_labels": {"sales": 0.9, "billing": 0.1}})
    assert encode_row(tok, row, max_tokens=4096).soft is not None
    assert encode_row(tok, row, max_tokens=4096, soft_gold_mask=True).soft is None


def test_rows_that_are_not_single_text_decisions_are_not_encoded():
    tok = CharTokenizer()
    row = convert_simple_row(LINE)
    two = {
        **row,
        "input": {
            **row["input"],
            "questions": {"a": row["input"]["questions"]["decision"]} | {"b": row["input"]["questions"]["decision"]},
        },
    }
    image = {**row, "input": {**row["input"], "state": {"image": "data:..."}}}
    unlabeled = {k: v for k, v in row.items() if k != "reference"}
    wrong_gold = {**row, "reference": {"target": "refunds"}}
    for bad in (two, image, unlabeled, wrong_gold):
        assert encode_row(tok, bad, max_tokens=4096) is None
    length = len(encode_row(tok, row, max_tokens=4096).prompt.input_ids)
    assert encode_row(tok, row, max_tokens=length + 1) is not None
    assert encode_row(tok, row, max_tokens=length) is None  # the mask slot must fit too
