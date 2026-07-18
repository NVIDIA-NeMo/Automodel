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
import subprocess
import sys
import types
from pathlib import Path

import pytest

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

from nemo_automodel.components.datasets.llm import retrieval_dataset_normalized as nd
from tools.retrieval import prepare_normalized_vl_retrieval_data as prep_norm
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


class _FingerprintDataset(list):
    def __init__(self, rows, *, fingerprint=None, fail_after=None):
        super().__init__(rows)
        self._fingerprint = fingerprint or json.dumps(rows, sort_keys=True)
        self.fail_after = fail_after

    def __iter__(self):
        for idx in range(len(self)):
            if self.fail_after is not None and idx == self.fail_after:
                raise RuntimeError("interrupted source iteration")
            yield self[idx]


class _FakeCorpusInfo:
    def __init__(self, docs, *, corpus_id="test", fingerprint=None):
        self.metadata = {
            "corpus_id": corpus_id,
            "query_instruction": "query:",
            "passage_instruction": "passage:",
        }
        self.docs = docs
        corpus_fingerprint = fingerprint or f"{corpus_id}:{','.join(sorted(docs))}"
        self.corpus = types.SimpleNamespace(data=types.SimpleNamespace(_fingerprint=corpus_fingerprint))

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


def _write_current_retrieval_config(path):
    path.write_text(
        "\n".join(
            [
                "dataset:",
                "  _target_: nemo_automodel.components.datasets.llm.retrieval_dataset.RetrievalDatasetConfig",
                "  data_dir_list:",
                "    - /tmp/current-train.json",
                "  model_type: bi_encoder",
                "  data_type: train",
                "  n_passages: 3",
                "  seed: 321",
                "dataloader:",
                "  batch_size: 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _patch_normalized_source(monkeypatch, image_mod, *, fail_on=None):
    image = image_mod.new("RGB", (2, 2), color="blue")

    def fake_load_datasets(data_dir_list, concatenate, seed):
        entry = data_dir_list[0]
        source_path = entry["path"] if isinstance(entry, dict) else entry
        if fail_on is not None and fail_on in str(source_path):
            raise RuntimeError(f"failed source: {source_path}")
        source_name = "a" if "source_a" in str(source_path) else "b"
        return (
            _FingerprintDataset(
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
                ]
            ),
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


def test_retrieval_tools_read_current_top_level_dataset_config(tmp_path):
    config_path = tmp_path / "config.yaml"
    _write_current_retrieval_config(config_path)

    assert prep_norm._load_dataset_args_from_config(str(config_path)) == (["/tmp/current-train.json"], 3)
    dataset_cfg = warm._extract_dataset_config(config_path)
    assert dataset_cfg["data_dir_list"] == ["/tmp/current-train.json"]
    assert dataset_cfg["n_passages"] == 3


def test_normalized_corpus_merge_deduplicates_equal_docs_and_rejects_conflicts():
    datasets = pytest.importorskip("datasets")

    def corpus(rows, path):
        dataset = datasets.Dataset.from_dict(
            {
                "id": [row[0] for row in rows],
                "text": [row[1] for row in rows],
                "image_jpeg": [b""] * len(rows),
                "nr_ocr": [""] * len(rows),
                "complex_ocr": [""] * len(rows),
            }
        )
        return nd.CorpusInfo(
            {"corpus_id": "shared"},
            nd.NormalizedArrowCorpusDataset(dataset, path=path),
        )

    first = corpus([("shared-doc", "same"), ("first-only", "first")], "first")
    second = corpus([("shared-doc", "same"), ("second-only", "second")], "second")
    merged = nd._merge_corpus_dicts([{"shared": first}, {"shared": second}])
    assert merged["shared"].corpus.get_all_ids() == ["first-only", "second-only", "shared-doc"]

    conflicting = corpus([("shared-doc", "different")], "conflicting")
    with pytest.raises(ValueError, match="Conflicting document payloads.*shared-doc"):
        nd._merge_corpus_dicts([{"shared": first}, {"shared": conflicting}])


def test_normalized_resume_rejects_equal_count_with_wrong_document_ids(tmp_path, monkeypatch):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "corpus-00000.arrow").touch()

    class FakeShard:
        def __len__(self):
            return 2

        def __getitem__(self, key):
            assert key == "id"
            return ["expected-a", "stale-c"]

    fake_datasets = types.SimpleNamespace(
        Dataset=types.SimpleNamespace(from_file=lambda path: FakeShard()),
    )
    monkeypatch.setattr(prep_norm, "safe_import", lambda name: (True, fake_datasets))

    metadata = prep_norm._existing_corpus_metadata_if_complete(
        "test",
        types.SimpleNamespace(metadata={"corpus_id": "test"}),
        corpus_dir,
        {"expected-a", "expected-b"},
    )
    assert metadata is None


def test_normalized_corpus_paths_are_contained_and_collision_safe(tmp_path):
    pytest.importorskip("datasets")
    pytest.importorskip("datasets.arrow_writer")

    output_dir = tmp_path / "normalized"
    (output_dir / "corpus").mkdir(parents=True)
    sentinel = output_dir / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    corpus_ids = [".", "..", "a/b", "a__b"]
    refs_by_corpus = {corpus_id: {f"{corpus_id}-doc"} for corpus_id in corpus_ids}
    corpus_dict = {
        corpus_id: _FakeCorpusInfo(
            {f"{corpus_id}-doc": {"text": corpus_id, "image": ""}},
            corpus_id=corpus_id,
        )
        for corpus_id in corpus_ids
    }

    metadata = prep_norm._write_corpus_shards(
        corpus_dict,
        refs_by_corpus,
        output_dir,
        docs_per_shard=10,
        jpeg_quality=90,
    )
    corpus_dirs = {Path(corpus["shards"][0]).parts[1] for corpus in metadata}
    assert len(corpus_dirs) == len(corpus_ids)
    assert prep_norm._safe_corpus_dir_name("a/b") != prep_norm._safe_corpus_dir_name("a__b")
    assert all(name not in {".", ".."} for name in corpus_dirs)

    for corpus in metadata:
        for shard in corpus["shards"]:
            (output_dir / shard).unlink()
    prep_norm._write_corpus_shards(
        corpus_dict,
        refs_by_corpus,
        output_dir,
        docs_per_shard=10,
        jpeg_quality=90,
        resume=True,
    )

    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert (output_dir / "corpus").is_dir()


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


def test_warm_retrieval_hf_cache_rejects_normalized_dataset_config(tmp_path):
    config_path = tmp_path / "config.yaml"
    _write_warm_cache_config(
        config_path,
        "nemo_automodel.components.datasets.llm.retrieval_dataset_normalized.NormalizedRetrievalDatasetConfig",
    )

    with pytest.raises(ValueError, match="only supports the original retrieval dataset"):
        warm.warm_retrieval_hf_cache(
            str(config_path),
            dataset_factory=lambda data_dir_list: _DummyRetrievalDataset(),
        )


def test_prepare_normalized_vl_retrieval_data_writes_portable_arrow_bundle(tmp_path, monkeypatch):
    pytest.importorskip("datasets")
    pytest.importorskip("datasets.arrow_writer")
    image_mod = pytest.importorskip("PIL.Image")

    image = image_mod.new("RGB", (2, 2), color="blue")
    monkeypatch.setattr(
        prep_norm,
        "load_datasets",
        lambda data_dir_list, concatenate, seed: (
            _FingerprintDataset(
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
                ]
            ),
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


def test_normalized_resume_rewrites_partial_train_shards(tmp_path, monkeypatch):
    pytest.importorskip("datasets")
    pytest.importorskip("datasets.arrow_writer")

    rows = [
        {
            "question_id": "q0",
            "question": "Q0",
            "corpus_id": "test",
            "pos_doc": [{"id": "p0"}],
            "neg_doc": [{"id": "n0"}],
        },
        {
            "question_id": "q1",
            "question": "Q1",
            "corpus_id": "test",
            "pos_doc": [{"id": "p1"}],
            "neg_doc": [{"id": "n1"}],
        },
    ]
    docs = {
        "p0": {"text": "positive 0", "image": ""},
        "n0": {"text": "negative 0", "image": ""},
        "p1": {"text": "positive 1", "image": ""},
        "n1": {"text": "negative 1", "image": ""},
    }
    load_count = 0

    def load_source(data_dir_list, concatenate, seed):
        nonlocal load_count
        load_count += 1
        fail_after = 1 if load_count == 1 else None
        return (
            _FingerprintDataset(rows, fingerprint="query-v1", fail_after=fail_after),
            {"test": _FakeCorpusInfo(docs, fingerprint="corpus-v1")},
        )

    monkeypatch.setattr(prep_norm, "load_datasets", load_source)
    output_dir = tmp_path / "normalized_partial"
    prep_kwargs = {
        "data_dir_list": ["source.json"],
        "output_dir": output_dir,
        "samples_per_shard": 10,
        "docs_per_shard": 10,
        "seed": 42,
        "max_samples": None,
        "jpeg_quality": 90,
    }

    with pytest.raises(RuntimeError, match="interrupted source iteration"):
        prep_norm.prepare_normalized_dataset(**prep_kwargs)

    train_dir = output_dir / "sources" / "source-00000" / "train"
    assert list(train_dir.glob("train-*.arrow"))
    assert not (train_dir / prep_norm._TRAIN_COMPLETION_FILENAME).exists()

    metadata = prep_norm.prepare_normalized_dataset(**prep_kwargs, resume=True)
    assert metadata["num_records"] == 2
    assert (train_dir / prep_norm._TRAIN_COMPLETION_FILENAME).is_file()
    dataset = nd.make_normalized_retrieval_dataset(data_dir_list=str(output_dir), n_passages=2)
    assert len(dataset) == 2

    completion_path = train_dir / prep_norm._TRAIN_COMPLETION_FILENAME
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["num_records"] = 1
    completion_path.write_text(json.dumps(completion), encoding="utf-8")
    rewritten_metadata = prep_norm.prepare_normalized_dataset(**prep_kwargs, resume=True)
    assert rewritten_metadata["num_records"] == 2
    rewritten_completion = json.loads(completion_path.read_text(encoding="utf-8"))
    assert rewritten_completion["num_records"] == 2


@pytest.mark.parametrize("changed_part", ["query", "corpus"])
def test_normalized_resume_rejects_changed_source_contents(tmp_path, monkeypatch, changed_part):
    pytest.importorskip("datasets")
    pytest.importorskip("datasets.arrow_writer")

    revisions = {"query": "v1", "corpus": "v1"}
    rows = [
        {
            "question_id": "q0",
            "question": "Q0",
            "corpus_id": "test",
            "pos_doc": [{"id": "p0"}],
            "neg_doc": [{"id": "n0"}],
        }
    ]
    docs = {
        "p0": {"text": "positive", "image": ""},
        "n0": {"text": "negative", "image": ""},
    }

    def load_source(data_dir_list, concatenate, seed):
        return (
            _FingerprintDataset(rows, fingerprint=f"query-{revisions['query']}"),
            {"test": _FakeCorpusInfo(docs, fingerprint=f"corpus-{revisions['corpus']}")},
        )

    monkeypatch.setattr(prep_norm, "load_datasets", load_source)
    output_dir = tmp_path / f"normalized_changed_{changed_part}"
    prep_kwargs = {
        "data_dir_list": ["source.json"],
        "output_dir": output_dir,
        "samples_per_shard": 10,
        "docs_per_shard": 10,
        "seed": 42,
        "max_samples": None,
        "jpeg_quality": 90,
    }
    prep_norm.prepare_normalized_dataset(**prep_kwargs)

    revisions[changed_part] = "v2"
    with pytest.raises(ValueError, match="source or prep options changed"):
        prep_norm.prepare_normalized_dataset(**prep_kwargs, resume=True)


def test_normalized_resume_rejects_unverifiable_source_contents(tmp_path, monkeypatch):
    pytest.importorskip("datasets")
    pytest.importorskip("datasets.arrow_writer")

    rows = [
        {
            "question_id": "q0",
            "question": "Q0",
            "corpus_id": "test",
            "pos_doc": [{"id": "p0"}],
            "neg_doc": [{"id": "n0"}],
        }
    ]
    docs = {
        "p0": {"text": "positive", "image": ""},
        "n0": {"text": "negative", "image": ""},
    }
    monkeypatch.setattr(
        prep_norm,
        "load_datasets",
        lambda data_dir_list, concatenate, seed: (rows, {"test": _FakeCorpusInfo(docs)}),
    )
    output_dir = tmp_path / "normalized_unverifiable"
    prep_kwargs = {
        "data_dir_list": ["source.json"],
        "output_dir": output_dir,
        "samples_per_shard": 10,
        "docs_per_shard": 10,
        "seed": 42,
        "max_samples": None,
        "jpeg_quality": 90,
    }
    prep_norm.prepare_normalized_dataset(**prep_kwargs)

    with pytest.raises(ValueError, match="does not provide verifiable content fingerprints"):
        prep_norm.prepare_normalized_dataset(**prep_kwargs, resume=True)


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


def test_prepare_normalized_vl_retrieval_data_finalizes_independent_sources(tmp_path, monkeypatch):
    pytest.importorskip("datasets")
    pytest.importorskip("datasets.arrow_writer")
    image_mod = pytest.importorskip("PIL.Image")
    _patch_normalized_source(monkeypatch, image_mod)

    output_dir = tmp_path / "normalized_array"
    data_dir_list = [{"path": "source_a.json", "num_samples": None}, {"path": "source_b.json", "num_samples": None}]
    source_1 = prep_norm._prepare_normalized_source(
        data_dir_list=data_dir_list,
        output_dir=output_dir,
        source_index=1,
        samples_per_shard=10,
        docs_per_shard=10,
        seed=42,
        max_samples=None,
        jpeg_quality=90,
    )

    assert source_1["path"] == "sources/source-00001"
    assert not (output_dir / "metadata.json").exists()
    with pytest.raises(FileNotFoundError, match="source-00000"):
        prep_norm._finalize_normalized_sources(
            data_dir_list,
            output_dir,
            samples_per_shard=10,
            docs_per_shard=10,
        )

    source_0 = prep_norm._prepare_normalized_source(
        data_dir_list=data_dir_list,
        output_dir=output_dir,
        source_index=0,
        samples_per_shard=10,
        docs_per_shard=10,
        seed=42,
        max_samples=None,
        jpeg_quality=90,
    )
    metadata = prep_norm._finalize_normalized_sources(
        data_dir_list,
        output_dir,
        samples_per_shard=10,
        docs_per_shard=10,
    )

    assert [source["path"] for source in metadata["sources"]] == [source_0["path"], source_1["path"]]
    assert metadata["num_records"] == 4
    dataset = nd.make_normalized_retrieval_dataset(data_dir_list=str(output_dir), n_passages=2)
    assert len(dataset) == 4


def test_normalized_array_launcher_generates_source_and_finalizer_jobs(tmp_path):
    repo_dir = Path(__file__).resolve().parents[4]
    launcher = repo_dir / "tools/retrieval/submit_prepare_normalized_vl_retrieval_data_cpu_array.sh"
    config_path = tmp_path / "config.yaml"
    config_path.write_text("dataset:\n  data_dir_list:\n    - source_a.json\n    - source_b.json\n", encoding="utf-8")

    capture_dir = tmp_path / "captured_sbatch"
    capture_dir.mkdir()
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_sbatch = fake_bin / "sbatch"
    fake_sbatch.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

capture_dir = Path(os.environ["SBATCH_CAPTURE_DIR"])
counter_path = capture_dir / "counter"
index = int(counter_path.read_text(encoding="utf-8")) if counter_path.exists() else 0
(capture_dir / f"args-{index}.json").write_text(json.dumps(sys.argv[1:]), encoding="utf-8")
(capture_dir / f"script-{index}.sh").write_text(sys.stdin.read(), encoding="utf-8")
counter_path.write_text(str(index + 1), encoding="utf-8")
print(1000 + index)
""",
        encoding="utf-8",
    )
    fake_sbatch.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "SBATCH_CAPTURE_DIR": str(capture_dir),
            "REPO_DIR": str(repo_dir),
            "CONFIG": str(config_path),
            "OUT_DIR": str(tmp_path / "normalized"),
            "CACHE_DIR": str(tmp_path / "cache"),
            "HF_CACHE": str(tmp_path / "cache/hf"),
            "TRITON_CACHE": str(tmp_path / "cache/triton"),
            "NUM_SOURCES": "2",
            "ARRAY_PARALLELISM": "1",
            "RUN_NAME": "normalized-array-test",
        }
    )
    result = subprocess.run(
        ["bash", str(launcher)],
        cwd=repo_dir,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    source_args = json.loads((capture_dir / "args-0.json").read_text(encoding="utf-8"))
    source_script = (capture_dir / "script-0.sh").read_text(encoding="utf-8")
    finalizer_args = json.loads((capture_dir / "args-1.json").read_text(encoding="utf-8"))
    finalizer_script = (capture_dir / "script-1.sh").read_text(encoding="utf-8")
    assert "--array=0-1%1" in source_args
    assert '--source-index "${SLURM_ARRAY_TASK_ID}"' in source_script
    assert '--container-image="nvcr.io#nvidia/nemo-automodel:26.06.00"' in source_script
    assert "--dependency=afterok:1000" in finalizer_args
    assert "--finalize-sources" in finalizer_script
    assert "Submitted normalized source array job: 1000" in result.stdout
    assert "Submitted normalized finalizer job: 1001" in result.stdout


def test_prepare_normalized_vl_retrieval_data_rejects_resume_with_changed_inputs(tmp_path, monkeypatch):
    pytest.importorskip("datasets")
    pytest.importorskip("datasets.arrow_writer")
    image_mod = pytest.importorskip("PIL.Image")
    _patch_normalized_source(monkeypatch, image_mod)

    output_dir = tmp_path / "normalized_resume"
    prep_norm.prepare_normalized_dataset(
        data_dir_list=[{"path": "source_a.json", "num_samples": None}],
        output_dir=output_dir,
        samples_per_shard=10,
        docs_per_shard=10,
        seed=42,
        max_samples=None,
        jpeg_quality=90,
    )

    with pytest.raises(ValueError, match="source or prep options changed"):
        prep_norm.prepare_normalized_dataset(
            data_dir_list=[{"path": "source_b.json", "num_samples": None}],
            output_dir=output_dir,
            samples_per_shard=10,
            docs_per_shard=10,
            seed=42,
            max_samples=None,
            jpeg_quality=90,
            resume=True,
        )


def test_prepare_normalized_vl_retrieval_data_rejects_resume_with_append(tmp_path):
    with pytest.raises(ValueError, match="mutually exclusive"):
        prep_norm.prepare_normalized_dataset(
            data_dir_list=["source.json"],
            output_dir=tmp_path / "normalized",
            samples_per_shard=10,
            docs_per_shard=10,
            seed=42,
            max_samples=None,
            jpeg_quality=90,
            resume=True,
            append=True,
        )


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
