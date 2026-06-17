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
from tools.retrieval import prepare_resolved_vl_retrieval_data as prep
from tools.retrieval import warm_retrieval_hf_cache as warm


def _write_jsonl(path, records):
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


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


def test_resolved_retrieval_dataset_opens_relative_images(tmp_path):
    image_mod = pytest.importorskip("PIL.Image")

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image_path = image_dir / "doc.jpg"
    image_mod.new("RGB", (2, 2), color="red").save(image_path)

    data_path = tmp_path / "shard-00000.jsonl"
    _write_jsonl(
        data_path,
        [
            {
                "question": "Q",
                "doc_text": ["positive", "negative"],
                "doc_image": ["images/doc.jpg", ""],
                "doc_id": ["p", "n"],
            }
        ],
    )

    dataset = rdr.make_resolved_retrieval_dataset(data_dir_list=str(data_path), n_passages=2)

    assert len(dataset) == 1
    example = next(iter(dataset))
    assert example["question"] == "Q"
    assert example["doc_text"] == ["positive", "negative"]
    assert example["doc_id"] == ["p", "n"]
    assert example["doc_image"][0].mode == "RGB"
    assert example["doc_image"][1] == ""


def test_resolved_retrieval_dataset_rank_sharding(tmp_path, monkeypatch):
    data_path = tmp_path / "shard-00000.jsonl"
    _write_jsonl(
        data_path,
        [{"question": f"Q{i}", "doc_text": [f"P{i}"], "doc_image": [""]} for i in range(5)],
    )
    monkeypatch.setattr(rdr, "_get_dist_info", lambda: (1, 2))

    dataset = rdr.make_resolved_retrieval_dataset(str(data_path), n_passages=1, decode_images=False)

    assert len(dataset) == 2
    assert [example["question"] for example in dataset] == ["Q1", "Q3"]


def test_resolved_retrieval_dataset_num_samples_caps_iteration(tmp_path):
    data_path = tmp_path / "shard-00000.jsonl"
    _write_jsonl(
        data_path,
        [{"question": f"Q{i}", "doc_text": [f"P{i}"], "doc_image": [""]} for i in range(5)],
    )

    dataset = rdr.make_resolved_retrieval_dataset(
        str(data_path),
        n_passages=1,
        decode_images=False,
        num_samples=3,
        repeat=2,
    )

    assert len(dataset) == 6
    assert [example["question"] for example in dataset] == ["Q0", "Q1", "Q2", "Q0", "Q1", "Q2"]


def test_resolved_retrieval_dataset_rejects_wrong_passage_count(tmp_path):
    data_path = tmp_path / "shard-00000.jsonl"
    _write_jsonl(data_path, [{"question": "Q", "doc_text": ["P"], "doc_image": [""]}])

    dataset = rdr.make_resolved_retrieval_dataset(str(data_path), n_passages=2, decode_images=False)

    with pytest.raises(ValueError, match="expected 2"):
        next(iter(dataset))


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


def test_prepare_resolved_vl_retrieval_data_writes_jsonl_and_images(tmp_path, monkeypatch):
    image_mod = pytest.importorskip("PIL.Image")

    monkeypatch.setattr(prep, "load_datasets", lambda data_dir_list, concatenate, seed: ([{"raw": "row"}], {}))
    monkeypatch.setattr(
        prep,
        "_transform_func",
        lambda item, num_neg_docs, corpus_dict, use_dataset_instruction: {
            "question": "Q",
            "doc_text": ["P", "N"],
            "doc_image": [image_mod.new("RGB", (2, 2), color="blue"), ""],
            "doc_id": ["p", "n"],
            "query_instruction": "",
            "passage_instruction": "",
        },
    )

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

    assert metadata["num_records"] == 1
    assert (output_dir / "metadata.json").is_file()
    assert (output_dir / "images" / "00000000_00.jpg").is_file()
    row = json.loads((output_dir / "shard-00000.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["question"] == "Q"
    assert row["doc_image"] == ["images/00000000_00.jpg", ""]


def test_prepare_resolved_vl_retrieval_data_writes_sqlite_packed_images(tmp_path, monkeypatch):
    image_mod = pytest.importorskip("PIL.Image")

    monkeypatch.setattr(prep, "load_datasets", lambda data_dir_list, concatenate, seed: ([{"raw": "row"}], {}))
    monkeypatch.setattr(
        prep,
        "_transform_func",
        lambda item, num_neg_docs, corpus_dict, use_dataset_instruction: {
            "question": "Q",
            "doc_text": ["P", "N"],
            "doc_image": [image_mod.new("RGB", (2, 2), color="blue"), ""],
            "doc_id": ["p", "n"],
            "query_instruction": "",
            "passage_instruction": "",
        },
    )

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
        image_storage="sqlite",
    )

    assert metadata["format"] == "nemo_automodel_resolved_vl_retrieval_sqlite"
    assert metadata["image_storage"] == "sqlite"
    assert metadata["num_records"] == 1
    assert metadata["shards"] == ["shard-00000.sqlite"]
    assert not (output_dir / "images").exists()

    dataset = rdr.make_resolved_retrieval_dataset(data_dir_list=str(output_dir), n_passages=2)
    example = next(iter(dataset))
    assert example["question"] == "Q"
    assert example["doc_text"] == ["P", "N"]
    assert example["doc_id"] == ["p", "n"]
    assert example["doc_image"][0].mode == "RGB"
    assert example["doc_image"][1] == ""


def test_prepare_resolved_vl_retrieval_data_writes_parquet_packed_images(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    pytest.importorskip("pyarrow.parquet")
    image_mod = pytest.importorskip("PIL.Image")

    monkeypatch.setattr(prep, "load_datasets", lambda data_dir_list, concatenate, seed: ([{"raw": "row"}], {}))
    monkeypatch.setattr(
        prep,
        "_transform_func",
        lambda item, num_neg_docs, corpus_dict, use_dataset_instruction: {
            "question": "Q",
            "doc_text": ["P", "N"],
            "doc_image": [image_mod.new("RGB", (2, 2), color="blue"), ""],
            "doc_id": ["p", "n"],
            "query_instruction": "",
            "passage_instruction": "",
        },
    )

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
        image_storage="parquet",
        parquet_row_group_size=1,
    )

    assert metadata["format"] == "nemo_automodel_resolved_vl_retrieval_parquet"
    assert metadata["image_storage"] == "parquet"
    assert metadata["num_records"] == 1
    assert metadata["parquet_row_group_size"] == 1
    assert metadata["shards"] == ["shard-00000.parquet"]
    assert not (output_dir / "images").exists()

    dataset = rdr.make_resolved_retrieval_dataset(data_dir_list=str(output_dir), n_passages=2)
    example = next(iter(dataset))
    assert example["question"] == "Q"
    assert example["doc_text"] == ["P", "N"]
    assert example["doc_id"] == ["p", "n"]
    assert example["doc_image"][0].mode == "RGB"
    assert example["doc_image"][1] == ""


def test_resolved_retrieval_parquet_dataset_rank_sharding(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    pytest.importorskip("pyarrow.parquet")
    image_mod = pytest.importorskip("PIL.Image")

    monkeypatch.setattr(
        prep, "load_datasets", lambda data_dir_list, concatenate, seed: ([{"idx": i} for i in range(4)], {})
    )

    def _transform_func(item, num_neg_docs, corpus_dict, use_dataset_instruction):
        idx = item["idx"]
        return {
            "question": f"Q{idx}",
            "doc_text": [f"P{idx}"],
            "doc_image": [image_mod.new("RGB", (2, 2), color="blue")],
            "query_instruction": "",
            "passage_instruction": "",
        }

    monkeypatch.setattr(prep, "_transform_func", _transform_func)
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
        image_storage="parquet",
        parquet_row_group_size=1,
    )
    monkeypatch.setattr(rdr, "_get_dist_info", lambda: (1, 2))

    dataset = rdr.make_resolved_retrieval_dataset(data_dir_list=str(output_dir), n_passages=1, decode_images=False)

    assert [example["question"] for example in dataset] == ["Q1", "Q3"]
    assert [example["doc_image"] for example in dataset] == [["packed:0"], ["packed:0"]]


def test_prepare_resolved_vl_retrieval_data_parallel_build_shard(tmp_path, monkeypatch):
    image_mod = pytest.importorskip("PIL.Image")

    monkeypatch.setattr(
        prep, "load_datasets", lambda data_dir_list, concatenate, seed: ([{"idx": i} for i in range(5)], {})
    )

    def _transform_func(item, num_neg_docs, corpus_dict, use_dataset_instruction):
        idx = item["idx"]
        return {
            "question": f"Q{idx}",
            "doc_text": [f"P{idx}", f"N{idx}"],
            "doc_image": [image_mod.new("RGB", (2, 2), color="blue"), ""],
            "doc_id": [f"p{idx}", f"n{idx}"],
            "query_instruction": "",
            "passage_instruction": "",
        }

    monkeypatch.setattr(prep, "_transform_func", _transform_func)

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
    assert metadata["shards"] == ["shard-00001-00000.jsonl", "shard-00001-00001.jsonl"]
    assert (output_dir / "metadata-build-shard-00001.json").is_file()
    assert (output_dir / "images" / "shard-00001" / "00000001_00.jpg").is_file()

    rows = [
        json.loads((output_dir / shard_path).read_text(encoding="utf-8").strip()) for shard_path in metadata["shards"]
    ]
    assert [row["question"] for row in rows] == ["Q1", "Q3"]
    assert rows[0]["doc_image"] == ["images/shard-00001/00000001_00.jpg", ""]
