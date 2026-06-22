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

import importlib.machinery
import importlib.util
import json
import os
import sys
import types

import pytest
from torch.utils.data import IterableDataset

if "torchvision" not in sys.modules and importlib.util.find_spec("torchvision") is None:
    torchvision = types.ModuleType("torchvision")
    transforms = types.ModuleType("torchvision.transforms")
    functional = types.ModuleType("torchvision.transforms.functional")
    torchvision.__spec__ = importlib.machinery.ModuleSpec("torchvision", loader=None)
    transforms.__spec__ = importlib.machinery.ModuleSpec("torchvision.transforms", loader=None)
    functional.__spec__ = importlib.machinery.ModuleSpec("torchvision.transforms.functional", loader=None)

    class _InterpolationMode:
        BICUBIC = "bicubic"

    functional.InterpolationMode = _InterpolationMode
    transforms.functional = functional
    torchvision.transforms = transforms
    sys.modules["torchvision"] = torchvision
    sys.modules["torchvision.transforms"] = transforms
    sys.modules["torchvision.transforms.functional"] = functional

import nemo_automodel.components.datasets.llm.retrieval_dataset_resolved as rdr
from nemo_automodel.components.datasets.llm import retrieval_dataset_normalized as nd
from tools.retrieval import prepare_normalized_vl_retrieval_data as prep_norm
from tools.retrieval import prepare_resolved_vl_retrieval_data as prep
from tools.retrieval import warm_retrieval_hf_cache as warm


class _DummyImage:
    size = (2, 2)


class _DummyRetrievalDataset:
    def __init__(self):
        self.examples = [
            {"question": "Q0", "doc_text": ["P0", "N0"], "doc_image": [_DummyImage(), ""]},
            {"question": "Q1", "doc_text": ["P1", "N1"], "doc_image": ["", _DummyImage()]},
        ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class _FakeCorpusInfo:
    def __init__(self, docs):
        self.metadata = {"corpus_id": "test", "query_instruction": "query:", "passage_instruction": "passage:"}
        self.docs = docs

    def get_document_by_id(self, doc_id):
        return self.docs[doc_id]


def _write_warm_cache_config(path, dataset_target):
    path.write_text(
        "\n".join(
            [
                "dataloader:",
                "  dataset:",
                f"    _target_: {dataset_target}",
                "    data_dir_list:",
                "      - /tmp/train.json",
                "    model_type: bi_encoder",
                "    data_type: train",
                "    n_passages: 2",
                "    seed: 123",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _patch_resolved_source(monkeypatch, image_mod, *, num_examples=1, n_passages=2, question_prefix=""):
    monkeypatch.setattr(
        prep, "load_datasets", lambda data_dir_list, concatenate, seed: ([{"idx": i} for i in range(num_examples)], {})
    )

    def _transform_func(item, num_neg_docs, corpus_dict, use_dataset_instruction):
        idx = item["idx"]
        doc_count = n_passages
        return {
            "question": f"{question_prefix}Q{idx}",
            "doc_text": [f"D{idx}-{doc_idx}" for doc_idx in range(doc_count)],
            "doc_image": [image_mod.new("RGB", (2, 2), color="blue")] + [""] * (doc_count - 1),
            "doc_id": [f"d{idx}-{doc_idx}" for doc_idx in range(doc_count)],
            "query_instruction": "",
            "passage_instruction": "",
        }

    monkeypatch.setattr(prep, "_transform_func", _transform_func)


def _patch_normalized_source(monkeypatch, image_mod, *, fail_on=None):
    image = image_mod.new("RGB", (2, 2), color="blue")

    def fake_load_datasets(data_dir_list, concatenate, seed):
        entry = data_dir_list[0]
        source_path = entry["path"] if isinstance(entry, dict) else entry
        if fail_on is not None and fail_on in str(source_path):
            raise RuntimeError(f"failed source: {source_path}")
        source_name = "a" if "source_a" in str(source_path) else "b"
        return (
            [
                {
                    "question_id": f"{source_name}-q0",
                    "question": f"{source_name} Q0",
                    "corpus_id": "test",
                    "pos_doc": [{"id": f"{source_name}-p0"}],
                    "neg_doc": [{"id": f"{source_name}-n0"}],
                },
                {
                    "question_id": f"{source_name}-q1",
                    "question": f"{source_name} Q1",
                    "corpus_id": "test",
                    "pos_doc": [{"id": f"{source_name}-p1"}],
                    "neg_doc": [{"id": f"{source_name}-n1"}],
                },
            ],
            {
                "test": _FakeCorpusInfo(
                    {
                        f"{source_name}-p0": {"text": f"{source_name} positive 0", "image": image},
                        f"{source_name}-n0": {"text": f"{source_name} negative 0", "image": ""},
                        f"{source_name}-p1": {"text": f"{source_name} positive 1", "image": ""},
                        f"{source_name}-n1": {"text": f"{source_name} negative 1", "image": ""},
                    }
                )
            },
        )

    monkeypatch.setattr(prep_norm, "load_datasets", fake_load_datasets)


def test_warm_retrieval_hf_cache_builds_original_dataset_from_config(tmp_path):
    config_path = tmp_path / "config.yaml"
    _write_warm_cache_config(config_path, "nemo_automodel.components.datasets.llm.make_retrieval_dataset")
    cache_dir = tmp_path / "hf_cache"
    report_path = tmp_path / "report.json"
    captured_kwargs = {}
    old_env = {name: os.environ.get(name) for name in warm._configure_hf_cache(None)}

    def dataset_factory(
        data_dir_list,
        model_type="bi_encoder",
        data_type="train",
        n_passages=5,
        seed=42,
        do_shuffle=False,
        max_train_samples=None,
    ):
        captured_kwargs.update(
            {
                "data_dir_list": data_dir_list,
                "model_type": model_type,
                "data_type": data_type,
                "n_passages": n_passages,
                "seed": seed,
                "do_shuffle": do_shuffle,
                "max_train_samples": max_train_samples,
            }
        )
        return _DummyRetrievalDataset()

    try:
        report = warm.warm_retrieval_hf_cache(
            str(config_path),
            cache_dir=str(cache_dir),
            touch_samples=2,
            max_train_samples=1,
            report_path=str(report_path),
            dataset_factory=dataset_factory,
        )
    finally:
        for name, value in old_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    assert captured_kwargs["data_dir_list"] == ["/tmp/train.json"]
    assert captured_kwargs["n_passages"] == 2
    assert captured_kwargs["seed"] == 123
    assert captured_kwargs["max_train_samples"] == 1
    assert report["dataset_length"] == 2
    assert report["touched_documents"] == 4
    assert report["touched_images"] == 2
    assert json.loads(report_path.read_text(encoding="utf-8"))["touch_samples"] == 2


def test_warm_retrieval_hf_cache_rejects_resolved_dataset_config(tmp_path):
    config_path = tmp_path / "config.yaml"
    _write_warm_cache_config(config_path, "nemo_automodel.components.datasets.llm.make_resolved_retrieval_dataset")

    with pytest.raises(ValueError, match="only supports the original retrieval dataset"):
        warm.warm_retrieval_hf_cache(
            str(config_path),
            dataset_factory=lambda data_dir_list: _DummyRetrievalDataset(),
        )


def test_prepare_resolved_vl_retrieval_data_writes_arrow_packed_images(tmp_path, monkeypatch):
    pytest.importorskip("datasets")
    pytest.importorskip("datasets.arrow_writer")
    image_mod = pytest.importorskip("PIL.Image")
    _patch_resolved_source(monkeypatch, image_mod)

    output_dir = tmp_path / "resolved"
    metadata = prep.resolve_dataset(
        data_dir_list=["train.json"],
        output_dir=output_dir,
        n_passages=2,
        samples_per_shard=10,
        seed=42,
        max_samples=None,
        use_dataset_instruction=False,
        jpeg_quality=90,
    )

    assert metadata["format"] == "nemo_automodel_resolved_vl_retrieval_arrow"
    assert metadata["image_storage"] == "arrow"
    assert metadata["num_records"] == 1
    assert metadata["shards"] == ["shard-00000.arrow"]

    dataset = rdr.make_resolved_retrieval_dataset(data_dir_list=str(output_dir), n_passages=2)
    assert not isinstance(dataset, IterableDataset)
    assert len(dataset) == 1
    example = dataset[0]
    assert example["question"] == "Q0"
    assert example["doc_text"] == ["D0-0", "D0-1"]
    assert example["doc_id"] == ["d0-0", "d0-1"]
    assert example["doc_image"][0].mode == "RGB"
    assert example["doc_image"][1] == ""


def test_resolved_retrieval_arrow_dataset_is_map_style_not_rank_sharded(tmp_path, monkeypatch):
    pytest.importorskip("datasets")
    pytest.importorskip("datasets.arrow_writer")
    image_mod = pytest.importorskip("PIL.Image")
    _patch_resolved_source(monkeypatch, image_mod, num_examples=4, n_passages=1)

    output_dir = tmp_path / "resolved"
    prep.resolve_dataset(
        data_dir_list=["train.json"],
        output_dir=output_dir,
        n_passages=1,
        samples_per_shard=2,
        seed=42,
        max_samples=None,
        use_dataset_instruction=False,
        jpeg_quality=90,
    )

    dataset = rdr.make_resolved_retrieval_dataset(data_dir_list=str(output_dir), n_passages=1, decode_images=False)

    assert not isinstance(dataset, IterableDataset)
    assert len(dataset) == 4
    assert [dataset[idx]["question"] for idx in range(len(dataset))] == ["Q0", "Q1", "Q2", "Q3"]
    assert [dataset[idx]["doc_image"] for idx in range(len(dataset))] == [["packed:0"]] * 4


def test_resolved_retrieval_arrow_dataset_rejects_wrong_passage_count(tmp_path, monkeypatch):
    pytest.importorskip("datasets")
    pytest.importorskip("datasets.arrow_writer")
    image_mod = pytest.importorskip("PIL.Image")
    _patch_resolved_source(monkeypatch, image_mod, n_passages=1)

    output_dir = tmp_path / "resolved"
    prep.resolve_dataset(
        data_dir_list=["train.json"],
        output_dir=output_dir,
        n_passages=1,
        samples_per_shard=10,
        seed=42,
        max_samples=None,
        use_dataset_instruction=False,
        jpeg_quality=90,
    )
    dataset = rdr.make_resolved_retrieval_dataset(str(output_dir), n_passages=2, decode_images=False)

    with pytest.raises(ValueError, match="expected 2"):
        dataset[0]


def test_prepare_normalized_vl_retrieval_data_writes_portable_arrow_bundle(tmp_path, monkeypatch):
    pytest.importorskip("datasets")
    pytest.importorskip("datasets.arrow_writer")
    image_mod = pytest.importorskip("PIL.Image")

    image = image_mod.new("RGB", (2, 2), color="blue")
    monkeypatch.setattr(
        prep_norm,
        "load_datasets",
        lambda data_dir_list, concatenate, seed: (
            [
                {
                    "question_id": "q0",
                    "question": "Q",
                    "corpus_id": "test",
                    "pos_doc": [{"id": "p"}],
                    "neg_doc": [{"id": "n"}],
                },
                {
                    "question_id": "q1",
                    "question": "Q again",
                    "corpus_id": "test",
                    "pos_doc": [{"id": "p"}],
                    "neg_doc": [{"id": "n2"}],
                },
            ],
            {
                "test": _FakeCorpusInfo(
                    {
                        "p": {"text": "positive", "image": image},
                        "n": {"text": "negative", "image": ""},
                        "n2": {"text": "negative two", "image": ""},
                    }
                )
            },
        ),
    )

    output_dir = tmp_path / "normalized"
    metadata = prep_norm.prepare_normalized_dataset(
        data_dir_list=["train.json"],
        output_dir=output_dir,
        samples_per_shard=10,
        docs_per_shard=10,
        seed=42,
        max_samples=None,
        jpeg_quality=90,
    )

    assert metadata["format"] == "nemo_automodel_normalized_vl_retrieval_arrow"
    assert metadata["version"] == 3
    assert metadata["num_records"] == 2
    assert len(metadata["sources"]) == 1
    assert metadata["sources"][0]["source_index"] == 0
    assert metadata["sources"][0]["source_name"] == "train"
    assert metadata["sources"][0]["source_entry"] == "train.json"
    assert metadata["sources"][0]["path"] == "sources/source-00000"
    assert metadata["sources"][0]["num_records"] == 2
    assert metadata["sources"][0]["source_key"]
    assert (output_dir / "metadata.json").is_file()
    source_dir = output_dir / "sources" / "source-00000"
    source_metadata = json.loads((source_dir / "metadata.json").read_text(encoding="utf-8"))
    assert source_metadata["train_shards"] == ["train/train-00000.arrow"]
    assert source_metadata["corpora"][0]["num_docs"] == 3

    dataset = nd.make_normalized_retrieval_dataset(
        data_dir_list=str(output_dir),
        n_passages=2,
        use_dataset_instruction=True,
    )

    assert len(dataset) == 2
    example = dataset[0]
    assert example["question"] == "Q"
    assert example["doc_text"] == ["positive", "negative"]
    assert example["doc_id"] == ["p", "n"]
    assert example["query_instruction"] == "query:"
    assert example["passage_instruction"] == "passage:"
    assert example["doc_image"][0].mode == "RGB"
    assert example["doc_image"][1] == ""

    for shard in (source_dir / "corpus" / "test").glob("*.arrow"):
        shard.unlink()
    resumed_metadata = prep_norm.prepare_normalized_dataset(
        data_dir_list=["train.json"],
        output_dir=output_dir,
        samples_per_shard=10,
        docs_per_shard=10,
        seed=42,
        max_samples=None,
        jpeg_quality=90,
        resume=True,
    )

    assert resumed_metadata["num_records"] == 2
    resumed_source_metadata = json.loads((source_dir / "metadata.json").read_text(encoding="utf-8"))
    assert resumed_source_metadata["corpora"][0]["num_docs"] == 3


def test_prepare_normalized_vl_retrieval_data_preserves_source_bundles(tmp_path, monkeypatch):
    pytest.importorskip("datasets")
    pytest.importorskip("datasets.arrow_writer")
    image_mod = pytest.importorskip("PIL.Image")
    _patch_normalized_source(monkeypatch, image_mod)

    output_dir = tmp_path / "normalized_multi"
    metadata = prep_norm.prepare_normalized_dataset(
        data_dir_list=[{"path": "source_a.json", "num_samples": None}, {"path": "source_b.json", "num_samples": None}],
        output_dir=output_dir,
        samples_per_shard=10,
        docs_per_shard=10,
        seed=42,
        max_samples=None,
        jpeg_quality=90,
    )

    assert metadata["version"] == 3
    assert metadata["num_records"] == 4
    assert [source["path"] for source in metadata["sources"]] == ["sources/source-00000", "sources/source-00001"]

    dataset = nd.make_normalized_retrieval_dataset(data_dir_list=str(output_dir), n_passages=2)
    assert len(dataset) == 4
    assert sorted(dataset[idx]["question"] for idx in range(len(dataset))) == ["a Q0", "a Q1", "b Q0", "b Q1"]

    sampled_dataset = nd.make_normalized_retrieval_dataset(
        data_dir_list=[
            {"path": str(output_dir / "sources" / "source-00000"), "num_samples": 1},
            {"path": str(output_dir / "sources" / "source-00001"), "num_samples": None},
        ],
        n_passages=2,
        seed=42,
    )
    questions = [sampled_dataset[idx]["question"] for idx in range(len(sampled_dataset))]
    assert len(questions) == 3
    assert sum(question.startswith("a ") for question in questions) == 1
    assert sum(question.startswith("b ") for question in questions) == 2


def test_prepare_normalized_vl_retrieval_data_appends_sources_safely(tmp_path, monkeypatch):
    pytest.importorskip("datasets")
    pytest.importorskip("datasets.arrow_writer")
    image_mod = pytest.importorskip("PIL.Image")
    _patch_normalized_source(monkeypatch, image_mod)

    output_dir = tmp_path / "normalized_append"
    prep_norm.prepare_normalized_dataset(
        data_dir_list=[{"path": "source_a.json", "num_samples": None}],
        output_dir=output_dir,
        samples_per_shard=10,
        docs_per_shard=10,
        seed=42,
        max_samples=None,
        jpeg_quality=90,
    )

    appended_metadata = prep_norm.prepare_normalized_dataset(
        data_dir_list=[{"path": "source_b.json", "num_samples": None}],
        output_dir=output_dir,
        samples_per_shard=10,
        docs_per_shard=10,
        seed=42,
        max_samples=None,
        jpeg_quality=90,
        append=True,
    )

    assert appended_metadata["num_records"] == 4
    assert [source["path"] for source in appended_metadata["sources"]] == [
        "sources/source-00000",
        "sources/source-00001",
    ]
    assert (output_dir / "sources" / "source-00000").is_dir()
    assert (output_dir / "sources" / "source-00001").is_dir()

    duplicate_metadata = prep_norm.prepare_normalized_dataset(
        data_dir_list=[{"path": "source_b.json", "num_samples": None}],
        output_dir=output_dir,
        samples_per_shard=10,
        docs_per_shard=10,
        seed=42,
        max_samples=None,
        jpeg_quality=90,
        append=True,
    )

    assert duplicate_metadata["num_records"] == 4
    assert len(duplicate_metadata["sources"]) == 2

    _patch_normalized_source(monkeypatch, image_mod, fail_on="source_c")
    before_failure_metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    with pytest.raises(RuntimeError, match="failed source"):
        prep_norm.prepare_normalized_dataset(
            data_dir_list=[{"path": "source_c.json", "num_samples": None}],
            output_dir=output_dir,
            samples_per_shard=10,
            docs_per_shard=10,
            seed=42,
            max_samples=None,
            jpeg_quality=90,
            append=True,
        )

    after_failure_metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert after_failure_metadata == before_failure_metadata
    assert not (output_dir / "sources" / "source-00002").exists()
    assert not list((output_dir / "sources").glob(".source-00002.tmp-*"))


def test_prepare_resolved_vl_retrieval_data_parallel_build_shard(tmp_path, monkeypatch):
    pytest.importorskip("datasets")
    pytest.importorskip("datasets.arrow_writer")
    image_mod = pytest.importorskip("PIL.Image")
    _patch_resolved_source(monkeypatch, image_mod, num_examples=5, n_passages=2)

    output_dir = tmp_path / "resolved"
    metadata = prep.resolve_dataset(
        data_dir_list=["train.json"],
        output_dir=output_dir,
        n_passages=2,
        samples_per_shard=1,
        seed=42,
        max_samples=5,
        use_dataset_instruction=False,
        jpeg_quality=90,
        num_build_shards=2,
        build_shard_index=1,
    )

    assert metadata["num_records"] == 2
    assert metadata["num_build_shards"] == 2
    assert metadata["build_shard_index"] == 1
    assert metadata["shards"] == ["shard-00001-00000.arrow", "shard-00001-00001.arrow"]
    assert (output_dir / "metadata-build-shard-00001.json").is_file()

    dataset = rdr.make_resolved_retrieval_dataset(data_dir_list=str(output_dir), n_passages=2, decode_images=False)
    assert [dataset[idx]["question"] for idx in range(len(dataset))] == ["Q1", "Q3"]
    assert dataset[0]["doc_image"] == ["packed:0", ""]


def test_resolved_retrieval_arrow_dataset_accepts_source_list_num_samples(tmp_path, monkeypatch):
    pytest.importorskip("datasets")
    pytest.importorskip("datasets.arrow_writer")
    image_mod = pytest.importorskip("PIL.Image")

    output_a = tmp_path / "resolved_a"
    _patch_resolved_source(monkeypatch, image_mod, num_examples=2, n_passages=1, question_prefix="A")
    prep.resolve_dataset(
        data_dir_list=["source_a.json"],
        output_dir=output_a,
        n_passages=1,
        samples_per_shard=10,
        seed=42,
        max_samples=None,
        use_dataset_instruction=False,
        jpeg_quality=90,
    )

    output_b = tmp_path / "resolved_b"
    _patch_resolved_source(monkeypatch, image_mod, num_examples=2, n_passages=1, question_prefix="B")
    prep.resolve_dataset(
        data_dir_list=["source_b.json"],
        output_dir=output_b,
        n_passages=1,
        samples_per_shard=10,
        seed=42,
        max_samples=None,
        use_dataset_instruction=False,
        jpeg_quality=90,
    )

    dataset = rdr.make_resolved_retrieval_dataset(
        data_dir_list=[
            {"path": str(output_a), "num_samples": 1},
            {"path": str(output_b), "num_samples": None},
        ],
        n_passages=1,
        decode_images=False,
    )

    assert len(dataset) == 3
    assert [dataset[idx]["question"] for idx in range(len(dataset))] == ["AQ0", "BQ0", "BQ1"]
