# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import os
from pathlib import Path
import json
import pytest

from nemo_automodel.components.datasets.llm import retrieval_dataset_inline as rdi

def load_jsonl_one_line(path):
    ans = []
    with open(path, "r") as f:
        for line in f:
            ans.append(json.loads(line))
            break # only load the first line
    return ans

def _embedding_testdata_training_file() -> Path:
    return Path(os.environ["TEST_DATA_DIR"]) / "embedding_testdata" / "training.jsonl"


def test_retrieval_dataset_inline_embedding_testdata_smoke():
    data_file = _embedding_testdata_training_file()
    if not data_file.exists():
        pytest.skip(f"Missing embedding test data file: {data_file}")

    ds = rdi.make_retrieval_dataset(
        data_dir_list=str(data_file),
        data_type="train",
        train_n_passages=2,  # 1 positive + 1 negative
        do_shuffle=False,
        max_train_samples=1,
    )

    assert len(ds) >= 1

    ex = ds[0]
    assert isinstance(ex.get("question"), str) and ex["question"]
    assert isinstance(ex.get("doc_text"), list) and len(ex["doc_text"]) == 2
    assert isinstance(ex["doc_text"][0], str)
    assert isinstance(ex.get("doc_image"), list) and len(ex["doc_image"]) == 2
    assert isinstance(ex.get("query_instruction"), str)
    assert isinstance(ex.get("passage_instruction"), str)

    payload = load_jsonl_one_line(data_file)
    assert ex['doc_text'][0] == payload[0]['pos_doc']
    assert ex['doc_text'][1] == payload[0]['neg_doc'][0]
    assert ex['question'] == payload[0]['query']

