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
import json
import sys
import types

import pytest

if "torchvision" not in sys.modules:
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
from examples.retrieval.bi_encoder import prepare_resolved_vl_retrieval_data as prep


def _write_jsonl(path, records):
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


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
