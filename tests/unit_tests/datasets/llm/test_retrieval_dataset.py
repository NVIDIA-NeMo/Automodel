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

import json
from typing import Any, Dict

import pytest
from datasets import Dataset

import nemo_automodel.components.datasets.llm.retrieval_dataset as rd


class DummyImage:
    def __init__(self):
        self.convert_called_with = None

    def convert(self, mode: str):
        self.convert_called_with = mode
        return self


class DummyCorpus(rd.AbstractDataset):
    def __init__(self, id_to_doc: Dict[str, Dict[str, Any]], query_instruction: str = "", passage_instruction: str = ""):
        self._id_to_doc = id_to_doc
        self._query_instruction = query_instruction
        self._passage_instruction = passage_instruction

    @property
    def query_instruction(self):
        return self._query_instruction

    @property
    def passage_instruction(self):
        return self._passage_instruction

    def get_document_by_id(self, id):
        return self._id_to_doc[str(id)]

    def get_all_ids(self):
        return sorted(list(self._id_to_doc.keys()))


def _mock_hf_load_dataset_returning(train_examples):
    # Returns a function suitable for monkeypatching rd.load_dataset
    def _loader(path):
        return {"train": Dataset.from_list(train_examples)}

    return _loader


def test_load_corpus_metadata_and_load_corpus_success(tmp_path, monkeypatch):
    corpus_dir = tmp_path / "corpusA"
    corpus_dir.mkdir()
    (corpus_dir / "merlin_metadata.json").write_text(json.dumps({"class": "TextQADataset", "corpus_id": "corpusA"}))

    # Provide minimal HF dataset for TextQADataset
    monkeypatch.setattr(
        rd,
        "load_dataset",
        _mock_hf_load_dataset_returning(
            [
                {"id": "1", "text": "Doc 1"},
                {"id": "2", "text": "Doc 2"},
            ]
        ),
    )

    corpus_id, corpus = rd.load_corpus(str(corpus_dir))
    assert corpus_id == "corpusA"
    assert isinstance(corpus, rd.TextQADataset)
    doc = corpus.get_document_by_id("1")
    assert doc["text"] == "Doc 1"


def test_add_corpus_duplicate_rules(tmp_path, monkeypatch):
    path1 = tmp_path / "corpus"
    path2 = tmp_path / "corpus2"
    path1.mkdir()
    path2.mkdir()

    meta = {"class": "TextQADataset", "corpus_id": "same_id"}
    (path1 / "merlin_metadata.json").write_text(json.dumps(meta))
    (path2 / "merlin_metadata.json").write_text(json.dumps(meta))

    monkeypatch.setattr(
        rd,
        "load_dataset",
        _mock_hf_load_dataset_returning([{"id": "a", "text": "A"}]),
    )

    corpus_dict = {}
    # First add is fine
    rd.add_corpus({"path": str(path1)}, corpus_dict)
    assert "same_id" in corpus_dict

    # Adding same corpus id with same path is a no-op (no error)
    rd.add_corpus({"path": str(path1)}, corpus_dict)
    assert corpus_dict["same_id"].path == str(path1)

    # Adding same id but different path must raise
    with pytest.raises(ValueError):
        rd.add_corpus({"path": str(path2)}, corpus_dict)


def test_load_datasets_normalizes_and_errors(tmp_path, monkeypatch):
    corpus_dir = tmp_path / "corpusA"
    corpus_dir.mkdir()
    (corpus_dir / "merlin_metadata.json").write_text(json.dumps({"class": "TextQADataset", "corpus_id": "corpusA"}))

    # TextQADataset source
    monkeypatch.setattr(
        rd,
        "load_dataset",
        _mock_hf_load_dataset_returning([{"id": "p1", "text": "pos1"}, {"id": "n1", "text": "neg1"}, {"id": "n2", "text": "neg2"}]),
    )

    data_ok = {
        "corpus": [{"path": str(corpus_dir)}],
        "data": [
            {
                "question_id": "q1",
                "question": "What?",
                "corpus_id": "corpusA",
                "pos_doc": [{"id": "p1"}],
                "neg_doc": [{"id": "n1"}, "n2"],
            }
        ],
    }
    f_ok = tmp_path / "train.json"
    f_ok.write_text(json.dumps(data_ok))

    dataset, corpus_dict = rd.load_datasets(str(f_ok))
    assert len(dataset) == 1
    row = dataset[0]
    assert row["question_id"] == "q1"
    assert row["pos_doc"][0]["id"] == "p1"
    assert row["neg_doc"][0]["id"] == "n1" and row["neg_doc"][1]["id"] == "n2"
    assert "corpusA" in corpus_dict

    # Missing required field should raise
    bad = {
        "corpus": [{"path": str(corpus_dir)}],
        "data": [
            {
                # "question_id" missing
                "question": "What?",
                "corpus_id": "corpusA",
                "pos_doc": [{"id": "p1"}],
                "neg_doc": [{"id": "n1"}],
            }
        ],
    }
    f_bad = tmp_path / "bad.json"
    f_bad.write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        rd.load_datasets(str(f_bad))


def test_transform_func_single_batched():
    corpus_dict = {
        "corpusA": DummyCorpus(
            {
                "p": {"text": "pos", "image": "", "nr_ocr": ""},
                "n1": {"text": "neg1", "image": "", "nr_ocr": ""},
                "n2": {"text": "neg2", "image": "", "nr_ocr": ""},
            }
        )
    }
    # Batched path
    examples_batched = {
        "question": ["Q"],
        "corpus_id": ["corpusA"],
        "pos_doc": [[{"id": "p"}]],
        "neg_doc": [[{"id": "n1"}, {"id": "n2"}]],
    }
    out = rd._transform_func(examples_batched, num_neg_docs=2, corpus_dict=corpus_dict)
    assert out["question"] == ["Q"]
    assert out["doc_text"][0] == ["pos", "neg1", "neg2"]
    assert len(out["doc_image"][0]) == 3

    # Single (non-batched) path
    examples_single = {
        "question": "Q",
        "corpus_id": "corpusA",
        "pos_doc": [{"id": "p"}],
        "neg_doc": [{"id": "n1"}, {"id": "n2"}],
    }
    out_single = rd._transform_func(examples_single, num_neg_docs=1, corpus_dict=corpus_dict)
    assert out_single["question"] == "Q"
    assert out_single["doc_text"] == ["pos", "neg1"]


def test_transform_func_image_conversion():
    img = DummyImage()
    corpus_dict = {
        "c": DummyCorpus({"p": {"text": "t", "image": img, "nr_ocr": ""}}),
    }
    examples = {"question": ["Q"], "corpus_id": ["c"], "pos_doc": [[{"id": "p"}]], "neg_doc": [[{"id": "p"}]]}
    out = rd._transform_func(examples, num_neg_docs=1, corpus_dict=corpus_dict)
    # conversion called
    assert isinstance(out["doc_image"][0][0], DummyImage)
    assert img.convert_called_with == "RGB"
    # text is preserved (trim logic without leading spaces result)
    assert out["doc_text"][0][0] == "t"


def _make_train_file(tmp_path, corpus_dir, data_len=1, corpus_id="corpusA"):
    data = []
    for i in range(data_len):
        data.append(
            {
                "question_id": f"q{i}",
                "question": f"Q{i}",
                "corpus_id": corpus_id,
                "pos_doc": [{"id": "p"}],
                "neg_doc": [{"id": "n1"}, {"id": "n2"}, {"id": "n3"}],
            }
        )
    d = {"corpus": [{"path": str(corpus_dir)}], "data": data}
    f = tmp_path / "train_data.json"
    f.write_text(json.dumps(d))
    return f


def test_make_retrieval_dataset_train_and_eval(tmp_path, monkeypatch):
    corpus_dir = tmp_path / "corpusA"
    corpus_dir.mkdir()
    (corpus_dir / "merlin_metadata.json").write_text(json.dumps({"class": "TextQADataset", "corpus_id": "corpusA"}))

    # HF data backing the corpus ids used in file
    monkeypatch.setattr(
        rd,
        "load_dataset",
        _mock_hf_load_dataset_returning(
            [{"id": "p", "text": "P"}, {"id": "n1", "text": "N1"}, {"id": "n2", "text": "N2"}, {"id": "n3", "text": "N3"}]
        ),
    )

    train_file = _make_train_file(tmp_path, corpus_dir, data_len=2)

    # Train mode: set_transform uses train_n_passages - 1 negatives
    ds_train = rd.make_retrieval_dataset(
        data_dir_list=str(train_file), data_type="train", train_n_passages=3, max_train_samples=1
    )
    assert len(ds_train) == 1
    ex = ds_train[0]
    assert len(ex["doc_text"]) == 3  # 1 pos + 2 neg

    # Eval mode
    ds_eval = rd.make_retrieval_dataset(data_dir_list=str(train_file), data_type="eval", eval_negative_size=2)
    ex_e = ds_eval[0]
    assert len(ex_e["doc_text"]) == 3


def test_abstract_dataset_methods_cover_pass():
     # Directly call abstract methods as unbound functions to execute 'pass' lines
     assert rd.AbstractDataset.get_document_by_id(None, None) is None
     assert rd.AbstractDataset.get_all_ids(None) is None


def test_textqa_get_all_ids(tmp_path, monkeypatch):
     corpus_dir = tmp_path / "corpusB"
     corpus_dir.mkdir()
     (corpus_dir / "merlin_metadata.json").write_text(json.dumps({"class": "TextQADataset", "corpus_id": "B"}))
     monkeypatch.setattr(
         rd,
         "load_dataset",
         _mock_hf_load_dataset_returning(
             [
                 {"id": "2", "text": "t2"},
                 {"id": "1", "text": "t1"},
             ]
         ),
     )
     _, corpus = rd.load_corpus(str(corpus_dir))
     assert corpus.get_all_ids() == ["1", "2"]


def test_load_corpus_metadata_missing_file(tmp_path):
     empty_dir = tmp_path / "empty_corpus"
     empty_dir.mkdir()
     with pytest.raises(ValueError) as e:
         rd.load_corpus_metadata(str(empty_dir))
     assert "merlin_metadata.json" in str(e.value)


def test_load_corpus_invalid_class():
     with pytest.raises(ValueError) as e:
         rd.load_corpus("/unused", metadata={"class": "UnknownDataset", "corpus_id": "x"})
     assert "DatasetClass is not implemented" in str(e.value)


def test_add_corpus_requires_dict(tmp_path):
     with pytest.raises(ValueError):
         rd.add_corpus({"path": str(tmp_path)}, None)


def test_load_datasets_type_coercion_and_concatenate_false(tmp_path, monkeypatch):
     corpus_dir = tmp_path / "corpusC"
     corpus_dir.mkdir()
     (corpus_dir / "merlin_metadata.json").write_text(json.dumps({"class": "TextQADataset", "corpus_id": "C"}))
     monkeypatch.setattr(
         rd,
         "load_dataset",
         _mock_hf_load_dataset_returning(
             [
                 {"id": "101", "text": "p"},
                 {"id": "202", "text": "n202"},
                 {"id": "x", "text": "nx"},
             ]
         ),
     )
     data = {
         "corpus": [{"path": str(corpus_dir)}],
         "data": [
             {
                 "question_id": "q",
                 "question": "Q",
                 "corpus_id": "C",
                 "pos_doc": [101],  # int -> coerced to "101" via lines 140-141
                 "neg_doc": [202, "x"],  # 202 -> "202" via lines 149-150; "x" unchanged
             }
         ],
     }
     f = tmp_path / "data.json"
     f.write_text(json.dumps(data))
     datasets_list, corpus_dict = rd.load_datasets(str(f), concatenate=False)
     assert isinstance(datasets_list, list) and len(datasets_list) == 1
     row = datasets_list[0][0]
     assert row["pos_doc"][0]["id"] == "101"
     assert [d["id"] for d in row["neg_doc"]] == ["202", "x"]
     assert "C" in corpus_dict


def test_transform_func_positive_else_and_text_empty_branch():
     # Covers line 198 (positives not list) and 228 (text empty and no image)
     corpus = DummyCorpus({"p": {"text": "", "image": "", "nr_ocr": ""}, "n": {"text": "n", "image": "", "nr_ocr": ""}})
     corpus_dict = {"c": corpus}
     # Non-batched example with pos_doc as dict (not list)
     examples_single = {"question": "Q", "corpus_id": "c", "pos_doc": {"id": "p"}, "neg_doc": [{"id": "n"}]}
     out = rd._transform_func(examples_single, num_neg_docs=1, corpus_dict=corpus_dict)
     # Positive text becomes "" (line 228), negative is "n"
     assert out["doc_text"] == ["", "n"]


def test_make_retrieval_dataset_shuffle_branch(tmp_path, monkeypatch):
    corpus_dir = tmp_path / "corpusD"
    corpus_dir.mkdir()
    (corpus_dir / "merlin_metadata.json").write_text(json.dumps({"class": "TextQADataset", "corpus_id": "D"}))
    monkeypatch.setattr(
        rd,
        "load_dataset",
        _mock_hf_load_dataset_returning(
            [{"id": "p", "text": "P"}, {"id": "n1", "text": "N1"}, {"id": "n2", "text": "N2"}]
        ),
    )
    train_file = _make_train_file(tmp_path, corpus_dir, data_len=3, corpus_id="D")
    ds = rd.make_retrieval_dataset(
        data_dir_list=str(train_file),
        data_type="train",
        train_n_passages=2,
        do_shuffle=True,
        max_train_samples=2,
    )
    ex0 = ds[0]
    assert len(ex0["doc_text"]) == 2


def test_make_retrieval_dataset_invalid_type(tmp_path, monkeypatch):
    corpus_dir = tmp_path / "corpusE"
    corpus_dir.mkdir()
    (corpus_dir / "merlin_metadata.json").write_text(json.dumps({"class": "TextQADataset", "corpus_id": "E"}))
    monkeypatch.setattr(
        rd,
        "load_dataset",
        _mock_hf_load_dataset_returning([{"id": "p", "text": "P"}, {"id": "n", "text": "N"}]),
    )
    train_file = _make_train_file(tmp_path, corpus_dir, data_len=1)
    with pytest.raises(ValueError):
        rd.make_retrieval_dataset(str(train_file), data_type="invalid")


def test_transform_func_epoch_cycling():
    """Test epoch-based cycling through multiple positive documents."""
    corpus_dict = {
        "corpusA": DummyCorpus(
            {
                "p1": {"text": "pos1", "image": "", "nr_ocr": ""},
                "p2": {"text": "pos2", "image": "", "nr_ocr": ""},
                "p3": {"text": "pos3", "image": "", "nr_ocr": ""},
                "n1": {"text": "neg1", "image": "", "nr_ocr": ""},
            }
        )
    }

    # Example with multiple positive docs
    examples = {
        "question": ["Q"],
        "corpus_id": ["corpusA"],
        "pos_doc": [[{"id": "p1"}, {"id": "p2"}, {"id": "p3"}]],
        "neg_doc": [[{"id": "n1"}]],
    }

    # Epoch 0: Should select first positive (p1)
    examples["epoch"] = 0
    out_0 = rd._transform_func(examples, num_neg_docs=1, corpus_dict=corpus_dict)
    assert out_0["doc_text"][0][0] == "pos1"

    # Epoch 1: Should select second positive (p2)
    examples["epoch"] = 1
    out_1 = rd._transform_func(examples, num_neg_docs=1, corpus_dict=corpus_dict)
    assert out_1["doc_text"][0][0] == "pos2"

    # Epoch 2: Should select third positive (p3)
    examples["epoch"] = 2
    out_2 = rd._transform_func(examples, num_neg_docs=1, corpus_dict=corpus_dict)
    assert out_2["doc_text"][0][0] == "pos3"

    # Epoch 3: Should cycle back to first positive (p1)
    examples["epoch"] = 3
    out_3 = rd._transform_func(examples, num_neg_docs=1, corpus_dict=corpus_dict)
    assert out_3["doc_text"][0][0] == "pos1"


def test_retrieval_transform_set_epoch():
    """Test the RetrievalTransform stateful class and set_epoch method."""
    corpus_dict = {
        "corpusA": DummyCorpus(
            {
                "p1": {"text": "pos1", "image": "", "nr_ocr": ""},
                "p2": {"text": "pos2", "image": "", "nr_ocr": ""},
                "n1": {"text": "neg1", "image": "", "nr_ocr": ""},
            }
        )
    }

    # Create a dataset with multiple positive docs
    dataset = Dataset.from_list([
        {
            "question_id": "q1",
            "question": "Q",
            "corpus_id": "corpusA",
            "pos_doc": [{"id": "p1"}, {"id": "p2"}],
            "neg_doc": [{"id": "n1"}]
        }
    ])

    # Use the RetrievalTransform class
    transform = rd.RetrievalTransform(num_neg_docs=1, corpus_dict=corpus_dict)
    dataset.set_transform(transform)
    dataset.set_epoch = transform.set_epoch

    # Initial transform (epoch 0 default)
    item_0 = dataset[0]
    assert item_0["doc_text"][0] == "pos1"

    # Update to epoch 1 using the exposed set_epoch method
    dataset.set_epoch(1)
    item_1 = dataset[0]
    assert item_1["doc_text"][0] == "pos2"

    # Cycle back to epoch 0
    dataset.set_epoch(0)
    item_back = dataset[0]
    assert item_back["doc_text"][0] == "pos1"


def test_use_dataset_instruction_from_metadata(tmp_path, monkeypatch):
    """Test that use_dataset_instruction correctly loads and applies instructions from metadata."""
    corpus_dir = tmp_path / "squad_corpus"
    corpus_dir.mkdir()

    # Create metadata with query and passage instructions as in merlin_metadata.json
    metadata = {
        "corpus_id": "squad",
        "class": "TextQADataset",
        "query_instruction": "Instruct: Given a question, retrieve Wikipedia passages that answer the question\nQuery:",
        "passage_instruction": "",
        "task_type": "Retrieval",
    }
    (corpus_dir / "merlin_metadata.json").write_text(json.dumps(metadata))

    # Mock HF dataset
    monkeypatch.setattr(
        rd,
        "load_dataset",
        _mock_hf_load_dataset_returning(
            [
                {"id": "doc1", "text": "Paris is the capital of France"},
                {"id": "doc2", "text": "London is the capital of England"},
            ]
        ),
    )

    # Use add_corpus to properly create CorpusInfo object
    corpus_dict = {}
    rd.add_corpus({"path": str(corpus_dir)}, corpus_dict)

    # Verify metadata properties are accessible through CorpusInfo
    assert "squad" in corpus_dict
    corpus_info = corpus_dict["squad"]
    assert corpus_info.corpus_id == "squad"
    assert corpus_info.query_instruction == metadata["query_instruction"]
    assert corpus_info.passage_instruction == metadata["passage_instruction"]
    assert corpus_info.task_type == metadata["task_type"]


def test_transform_func_with_use_dataset_instruction():
    """Test that _transform_func includes query and passage instructions when use_dataset_instruction=True."""

    query_instruction = "Instruct: Given a question, retrieve Wikipedia passages that answer the question\nQuery:"
    passage_instruction = ""

    corpus_dict = {
        "squad": DummyCorpus(
            {
                "p1": {"text": "positive doc", "image": "", "nr_ocr": ""},
                "n1": {"text": "negative doc", "image": "", "nr_ocr": ""},
            },
            query_instruction=query_instruction,
            passage_instruction=passage_instruction,
        )
    }

    # Test with use_dataset_instruction=True
    examples_with_instruction = {
        "question": ["What is the capital?"],
        "corpus_id": ["squad"],
        "pos_doc": [[{"id": "p1"}]],
        "neg_doc": [[{"id": "n1"}]],
    }

    out_with_instruction = rd._transform_func(
        examples_with_instruction,
        num_neg_docs=1,
        corpus_dict=corpus_dict,
        use_dataset_instruction=True,
    )

    # Verify that query_instruction and passage_instruction fields are populated
    assert "query_instruction" in out_with_instruction
    assert "passage_instruction" in out_with_instruction
    assert out_with_instruction["query_instruction"][0] == query_instruction
    assert out_with_instruction["passage_instruction"][0] == passage_instruction

    # Test with use_dataset_instruction=False
    out_without_instruction = rd._transform_func(
        examples_with_instruction,
        num_neg_docs=1,
        corpus_dict=corpus_dict,
        use_dataset_instruction=False,
    )

    # Verify that instruction fields are empty strings when disabled
    assert out_without_instruction["query_instruction"][0] == ""
    assert out_without_instruction["passage_instruction"][0] == ""

    # Both should have same question and doc_text content
    assert out_with_instruction["question"] == out_without_instruction["question"]
    assert out_with_instruction["doc_text"] == out_without_instruction["doc_text"]
