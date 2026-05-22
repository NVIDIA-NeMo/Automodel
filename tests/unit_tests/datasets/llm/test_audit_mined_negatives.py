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

from examples.retrieval.data_utils.audit_mined_negatives import audit_training_data


def test_audit_mined_negatives_reports_and_cleans_invalid_rows():
    training_data = {
        "corpus": {"path": "/corpus"},
        "data": [
            {
                "question_id": "q0_0",
                "original_question_id": "q0",
                "question": "Which document is positive?",
                "corpus_id": "demo",
                "pos_doc": [{"id": 1}],
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
        "negative_is_known_positive": 1,
        "duplicate_negative": 1,
        "missing_negative_score": 1,
        "non_finite_negative_score": 1,
        "dropped_negatives": 3,
        "total_findings": 4,
    }
    assert cleaned["corpus"] == training_data["corpus"]
    assert cleaned["data"] == [
        {
            "question_id": "q0_0",
            "original_question_id": "q0",
            "question": "Which document is positive?",
            "corpus_id": "demo",
            "pos_doc": [{"id": 1}],
            "neg_doc": [{"id": "2", "score": 0.4}, {"id": "4"}],
        }
    ]
    assert findings[0]["original_question_id"] == "q0"


def test_audit_mined_negatives_preserves_records_without_findings():
    training_data = {
        "corpus": {"path": "/corpus"},
        "data": [
            {
                "question_id": "q0",
                "question": "Which document is positive?",
                "corpus_id": "demo",
                "pos_doc": [{"id": "1"}],
                "neg_doc": [{"id": "2", "score": 0.2}],
            }
        ],
    }

    summary, cleaned, findings = audit_training_data(training_data, drop_invalid_negatives=True)

    assert summary["total_findings"] == 0
    assert cleaned == training_data
    assert findings == []
