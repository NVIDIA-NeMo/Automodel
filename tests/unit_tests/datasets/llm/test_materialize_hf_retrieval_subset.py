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

from examples.retrieval.data_utils import materialize_hf_retrieval_subset


class _FakeCorpusInfo:
    metadata = {"corpus_id": "demo", "query_instruction": "query:"}
    corpus_id = "demo"

    def get_all_ids(self):
        return ["d0"]

    def get_document_by_id(self, doc_id):
        return {"text": f"document {doc_id}"}


def test_materialize_hf_retrieval_subset_writes_local_corpus_json(tmp_path, monkeypatch):
    data_list = [
        {
            "question_id": "q0",
            "original_question_id": "q0",
            "question": "What is the document?",
            "corpus_id": "demo",
            "pos_doc": [{"id": "d0"}],
            "neg_doc": [{"id": "d1"}],
        }
    ]
    monkeypatch.setattr(
        materialize_hf_retrieval_subset,
        "_load_hf_subset",
        lambda repo_id, subset: (data_list, _FakeCorpusInfo()),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "materialize_hf_retrieval_subset.py",
            "nvidia/embed-nemotron-dataset-v1",
            "FEVER",
            str(tmp_path),
        ],
    )

    assert materialize_hf_retrieval_subset.main() == 0

    train_json = json.loads((tmp_path / "train.json").read_text())
    metadata = json.loads((tmp_path / "FEVER_corpus" / "merlin_metadata.json").read_text())

    assert train_json["corpus"] == [{"path": "./FEVER_corpus"}]
    assert train_json["data"] == data_list
    assert metadata["class"] == "TextQADataset"
    assert metadata["corpus_id"] == "demo"
    assert (tmp_path / "FEVER_corpus" / "train.parquet").exists()
