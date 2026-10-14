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

import pytest

from examples.retrieval.data_utils.unroll_pos_docs import unroll_training_data


def test_unroll_preserves_sibling_positives_and_filters_stringified_negatives():
    records = [
        {
            "question_id": "q0",
            "question": "Which documents are positive?",
            "corpus_id": "demo",
            "pos_doc": [{"id": 1}, {"id": "2"}],
            "neg_doc": [{"id": "1"}, {"id": 2}, {"id": "3"}],
        }
    ]

    unrolled = unroll_training_data(records)

    assert unrolled == [
        {
            "question_id": "q0_0",
            "original_question_id": "q0",
            "question": "Which documents are positive?",
            "corpus_id": "demo",
            "pos_doc": [{"id": 1}, {"id": "2"}],
            "neg_doc": [{"id": "3"}],
        },
        {
            "question_id": "q0_1",
            "original_question_id": "q0",
            "question": "Which documents are positive?",
            "corpus_id": "demo",
            "pos_doc": [{"id": "2"}, {"id": 1}],
            "neg_doc": [{"id": "3"}],
        },
    ]


def test_unroll_keeps_single_positive_question_id_and_filters_negatives():
    records = [
        {
            "question_id": "q0",
            "question": "Which document is positive?",
            "corpus_id": "demo",
            "pos_doc": [{"id": 1}],
            "neg_doc": [{"id": "1"}, {"id": "2"}],
        }
    ]

    assert unroll_training_data(records) == [
        {
            "question_id": "q0",
            "question": "Which document is positive?",
            "corpus_id": "demo",
            "pos_doc": [{"id": 1}],
            "neg_doc": [{"id": "2"}],
        }
    ]


def test_unroll_raises_for_missing_document_id():
    records = [
        {
            "question_id": "q0",
            "question": "Which document is positive?",
            "corpus_id": "demo",
            "pos_doc": [{"id": "1"}],
            "neg_doc": [{}],
        }
    ]

    with pytest.raises(ValueError, match="missing an id"):
        unroll_training_data(records)
