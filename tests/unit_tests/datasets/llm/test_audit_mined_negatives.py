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

import json
import sys

from examples.retrieval.data_utils.audit_mined_negatives import audit_training_data
from examples.retrieval.data_utils.audit_mined_negatives import main as audit_main


def test_audit_mined_negatives_reports_and_cleans_invalid_rows():
    training_data = {
        "corpus": {"path": "/corpus"},
        "data": [
            {
                "question_id": "q0_0",
                "original_question_id": "q0",
                "question": "Which document is positive?",
                "corpus_id": "demo",
                "pos_doc": [{"id": 1, "score": 0.8}],
                "neg_doc": [
                    {"id": "1", "score": 0.9},
                    {"id": "2", "score": 0.4},
                    {"id": "2", "score": 0.3},
                    {"id": "3", "score": float("-inf")},
                    {"id": "4"},
                ],
            }
        ],
    }

    summary, cleaned, findings = audit_training_data(training_data, drop_invalid_negatives=True)

    assert summary == {
        "records": 1,
        "negatives": 5,
        "rows_with_findings": 1,
        "missing_positive_score": 0,
        "non_finite_positive_score": 0,
        "negative_is_known_positive": 1,
        "duplicate_negative": 1,
        "missing_negative_score": 1,
        "non_finite_negative_score": 1,
        "rows_with_too_few_negatives": 0,
        "dropped_negatives": 4,
        "total_findings": 4,
    }
    assert cleaned["corpus"] == training_data["corpus"]
    assert cleaned["data"] == [
        {
            "question_id": "q0_0",
            "original_question_id": "q0",
            "question": "Which document is positive?",
            "corpus_id": "demo",
            "pos_doc": [{"id": 1, "score": 0.8}],
            "neg_doc": [{"id": "2", "score": 0.4}],
        }
    ]
    assert findings[0]["original_question_id"] == "q0"


def test_audit_mined_negatives_reports_positive_score_findings():
    training_data = {
        "data": [
            {
                "question_id": "q0",
                "pos_doc": [{"id": "1"}, {"id": "2", "score": float("nan")}],
                "neg_doc": [{"id": "3", "score": 0.1}],
            }
        ]
    }

    summary, _, findings = audit_training_data(training_data)

    assert summary["missing_positive_score"] == 1
    assert summary["non_finite_positive_score"] == 1
    assert summary["total_findings"] == 2
    assert findings[0]["positive_id"] == "1"
    assert findings[1]["positive_id"] == "2"


def test_audit_mined_negatives_preserves_records_without_findings():
    training_data = {
        "corpus": {"path": "/corpus"},
        "data": [
            {
                "question_id": "q0",
                "question": "Which document is positive?",
                "corpus_id": "demo",
                "pos_doc": [{"id": "1", "score": 0.8}],
                "neg_doc": [{"id": "2", "score": 0.2}],
            }
        ],
    }

    summary, cleaned, findings = audit_training_data(training_data, drop_invalid_negatives=True)

    assert summary["total_findings"] == 0
    assert cleaned == training_data
    assert findings == []


def test_audit_cli_writes_cleaned_output_and_exits_zero(tmp_path, monkeypatch, capsys):
    input_file = tmp_path / "mined.json"
    output_file = tmp_path / "cleaned.json"
    input_file.write_text(
        json.dumps(
            {
                "corpus": {"path": "/corpus"},
                "data": [
                    {
                        "question_id": "q0",
                        "question": "Which document is positive?",
                        "corpus_id": "demo",
                        "pos_doc": [{"id": "1", "score": 0.8}],
                        "neg_doc": [{"id": "1", "score": 1.0}, {"id": "2"}],
                    }
                ],
            }
        )
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_mined_negatives.py",
            str(input_file),
            "--drop-invalid-negatives",
            "--output",
            str(output_file),
        ],
    )

    assert audit_main() == 0
    report = json.loads(capsys.readouterr().out)
    cleaned = json.loads(output_file.read_text())

    assert report["summary"]["total_findings"] == 2
    assert report["remaining_summary"]["total_findings"] == 0
    assert cleaned["data"][0]["neg_doc"] == []


def test_audit_cli_exits_nonzero_when_findings_remain(tmp_path, monkeypatch):
    input_file = tmp_path / "mined.json"
    input_file.write_text(
        json.dumps(
            {
                "data": [
                    {
                        "question_id": "q0",
                        "pos_doc": [{"id": "1", "score": 0.8}],
                        "neg_doc": [{"id": "1", "score": 1.0}],
                    }
                ]
            }
        )
    )
    monkeypatch.setattr(sys, "argv", ["audit_mined_negatives.py", str(input_file)])

    assert audit_main() == 1


def test_audit_cli_allow_findings_exits_zero(tmp_path, monkeypatch):
    input_file = tmp_path / "mined.json"
    input_file.write_text(
        json.dumps(
            {
                "data": [
                    {
                        "question_id": "q0",
                        "pos_doc": [{"id": "1", "score": 0.8}],
                        "neg_doc": [{"id": "1", "score": 1.0}],
                    }
                ]
            }
        )
    )
    monkeypatch.setattr(sys, "argv", ["audit_mined_negatives.py", str(input_file), "--allow-findings"])

    assert audit_main() == 0


def test_audit_cli_min_negatives_catches_rows_cleaned_to_empty(tmp_path, monkeypatch, capsys):
    input_file = tmp_path / "mined.json"
    output_file = tmp_path / "cleaned.json"
    input_file.write_text(
        json.dumps(
            {
                "data": [
                    {
                        "question_id": "q0",
                        "pos_doc": [{"id": "1", "score": 0.8}],
                        "neg_doc": [{"id": "1", "score": 1.0}],
                    }
                ]
            }
        )
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_mined_negatives.py",
            str(input_file),
            "--drop-invalid-negatives",
            "--output",
            str(output_file),
            "--min-negatives",
            "1",
        ],
    )

    assert audit_main() == 1
    report = json.loads(capsys.readouterr().out)

    assert report["remaining_summary"]["rows_with_too_few_negatives"] == 1
    assert report["remaining_findings"][0]["issue"] == "too_few_negatives"
