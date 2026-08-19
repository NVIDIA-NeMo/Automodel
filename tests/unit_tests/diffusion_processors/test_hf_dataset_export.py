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
"""Tests for Hugging Face diffusion dataset materialization."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from tools.diffusion.data import hf_dataset_export


class FakeDataset:
    """Small dataset-like object for materialization tests."""

    def __init__(self, rows, *, config_name=None):
        self.rows = rows
        self.column_names = list(rows[0]) if rows else []
        self.features = {}
        self.info = SimpleNamespace(config_name=config_name)

    def __iter__(self):
        return iter(self.rows)


def test_materialize_hf_image_dataset_writes_jsonl_captions(tmp_path, monkeypatch):
    rows = [
        {"image": Image.new("RGB", (8, 8), color="red"), "text": "red square"},
        {"image": Image.new("RGB", (8, 8), color="blue"), "text": "blue square"},
    ]
    monkeypatch.setattr(hf_dataset_export, "_load_hf_dataset", lambda *args, **kwargs: FakeDataset(rows))

    export = hf_dataset_export.materialize_hf_dataset(
        "org/images",
        tmp_path,
        media_type="image",
        caption_field="internvl",
        max_items=1,
    )

    assert export.total_items == 1
    assert export.media_column == "image"
    assert export.caption_column == "text"
    assert (tmp_path / "hf_sample_00000000.png").exists()

    caption_lines = (tmp_path / "hf_internvl.json").read_text(encoding="utf-8").splitlines()
    assert len(caption_lines) == 1
    assert json.loads(caption_lines[0]) == {
        "file_name": "hf_sample_00000000.png",
        "internvl": "red square",
    }


def test_materialize_hf_video_dataset_writes_sidecar_captions(tmp_path, monkeypatch):
    rows = [{"video": {"bytes": b"fake-video", "path": "source.mov"}, "caption": "a short clip"}]
    monkeypatch.setattr(hf_dataset_export, "_load_hf_dataset", lambda *args, **kwargs: FakeDataset(rows))

    export = hf_dataset_export.materialize_hf_dataset(
        "org/videos",
        tmp_path,
        media_type="video",
        caption_field="caption",
    )

    assert export.total_items == 1
    assert export.media_column == "video"
    assert export.caption_column == "caption"
    assert (tmp_path / "hf_sample_00000000.mov").read_bytes() == b"fake-video"
    assert json.loads((tmp_path / "hf_sample_00000000.json").read_text(encoding="utf-8")) == {"caption": "a short clip"}


def test_materialize_hf_video_dataset_accepts_url_mapping(tmp_path, monkeypatch):
    source_path = tmp_path / "source.mov"
    source_path.write_bytes(b"fake-video")
    rows = [{"video": {"url": str(source_path)}, "caption": "a local clip"}]
    monkeypatch.setattr(hf_dataset_export, "_load_hf_dataset", lambda *args, **kwargs: FakeDataset(rows))

    export_dir = tmp_path / "export"
    export = hf_dataset_export.materialize_hf_dataset("org/videos", export_dir, media_type="video")

    assert export.total_items == 1
    assert (export_dir / "hf_sample_00000000.mov").read_bytes() == b"fake-video"


def test_materialize_hf_dataset_rejects_existing_export(tmp_path, monkeypatch):
    rows = [{"image": Image.new("RGB", (8, 8), color="red"), "text": "red square"}]
    monkeypatch.setattr(hf_dataset_export, "_load_hf_dataset", lambda *args, **kwargs: FakeDataset(rows))

    hf_dataset_export.materialize_hf_dataset("org/images", tmp_path, media_type="image")
    media_path = tmp_path / "hf_sample_00000000.png"
    caption_path = tmp_path / "hf_internvl.json"
    original_media = media_path.read_bytes()
    original_captions = caption_path.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="HF materialization directory is not empty; choose a new directory"):
        hf_dataset_export.materialize_hf_dataset("org/images", tmp_path, media_type="image")

    assert media_path.read_bytes() == original_media
    assert caption_path.read_text(encoding="utf-8") == original_captions


def test_materialize_hf_dataset_errors_when_media_column_cannot_be_inferred(tmp_path, monkeypatch):
    monkeypatch.setattr(
        hf_dataset_export,
        "_load_hf_dataset",
        lambda *args, **kwargs: FakeDataset([{"text": "caption only"}]),
    )

    with pytest.raises(ValueError, match="Pass --dataset_media_column"):
        hf_dataset_export.materialize_hf_dataset("org/images", tmp_path, media_type="image")


def test_materialize_hf_image_edit_dataset_writes_ordered_manifest_and_deduplicates_media(tmp_path, monkeypatch):
    rows = [
        {
            "img_id": "42",
            "turn_index": 2,
            "source_img": Image.new("RGB", (8, 6), color="red"),
            "mask_img": Image.new("L", (8, 6), color="white"),
            "instruction": "make it blue",
            "target_img": Image.new("RGB", (8, 6), color="blue"),
        }
    ]
    load_calls = []

    def fake_load_hf_dataset(*args, **kwargs):
        load_calls.append((args, kwargs))
        return FakeDataset(rows, config_name="magicbrush")

    monkeypatch.setattr(hf_dataset_export, "_load_hf_dataset", fake_load_hf_dataset)
    mappings = [
        hf_dataset_export.HFDatasetMediaMapping("target", "target_img"),
        hf_dataset_export.HFDatasetMediaMapping("context", "source_img"),
        hf_dataset_export.HFDatasetMediaMapping("condition", "source_img"),
    ]

    export = hf_dataset_export.materialize_hf_dataset(
        "org/image-edits",
        tmp_path,
        media_type="image-edit",
        split="dev",
        config_name="magicbrush",
        media_mappings=mappings,
        caption_column="instruction",
    )

    assert export.total_items == 1
    assert export.media_column == "target_img"
    assert export.media_mappings == tuple(mappings)
    assert export.manifest_file == tmp_path / "hf_image_edit_manifest.jsonl"
    assert export.dataset_config_name == "magicbrush"
    assert len(list((tmp_path / "media").glob("*"))) == 2
    assert load_calls[0][1]["config_name"] == "magicbrush"

    manifest_row = json.loads(export.manifest_file.read_text(encoding="utf-8"))
    assert manifest_row["id"] == "dev:00000000"
    assert manifest_row["prompt"] == "make it blue"
    assert [entry["role"] for entry in manifest_row["media"]] == ["target", "context", "condition"]
    assert manifest_row["media"][1]["file_name"] == manifest_row["media"][2]["file_name"]
    assert all(not Path(entry["file_name"]).is_absolute() for entry in manifest_row["media"])
    assert manifest_row["metadata"]["dataset_name"] == "org/image-edits"
    assert manifest_row["metadata"]["dataset_config_name"] == "magicbrush"
    assert manifest_row["metadata"]["dataset_split"] == "dev"
    assert manifest_row["metadata"]["row_index"] == 0
    assert manifest_row["metadata"]["row"]["img_id"] == "42"
    assert manifest_row["metadata"]["row"]["turn_index"] == 2
    mask_metadata = manifest_row["metadata"]["row"]["mask_img"]
    assert mask_metadata["media_type"] == "image"
    assert not Path(mask_metadata["file_name"]).is_absolute()
    assert (tmp_path / mask_metadata["file_name"]).exists()


@pytest.mark.parametrize(
    ("mappings", "match"),
    [
        ([hf_dataset_export.HFDatasetMediaMapping("context", "source")], "exactly one target"),
        ([hf_dataset_export.HFDatasetMediaMapping("target", "target")], "at least one context"),
        (
            [
                hf_dataset_export.HFDatasetMediaMapping("target", "target"),
                hf_dataset_export.HFDatasetMediaMapping("reference", "source"),
            ],
            "Unsupported image-edit media role",
        ),
    ],
)
def test_materialize_hf_image_edit_dataset_validates_roles(tmp_path, mappings, match):
    with pytest.raises(ValueError, match=match):
        hf_dataset_export.materialize_hf_dataset(
            "org/image-edits",
            tmp_path,
            media_type="image-edit",
            media_mappings=mappings,
        )


def test_materialize_hf_dataset_rejects_legacy_column_with_media_mappings(tmp_path):
    with pytest.raises(ValueError, match="--dataset_media_column cannot be combined with --dataset_media_mapping"):
        hf_dataset_export.materialize_hf_dataset(
            "org/image-edits",
            tmp_path,
            media_type="image-edit",
            media_column="target_img",
            media_mappings=[
                hf_dataset_export.HFDatasetMediaMapping("target", "target_img"),
                hf_dataset_export.HFDatasetMediaMapping("context", "source_img"),
            ],
        )


def test_materialize_hf_image_edit_dataset_validates_mapping_columns(tmp_path, monkeypatch):
    rows = [{"target_img": Image.new("RGB", (8, 8)), "instruction": "edit it"}]
    monkeypatch.setattr(hf_dataset_export, "_load_hf_dataset", lambda *args, **kwargs: FakeDataset(rows))

    with pytest.raises(ValueError, match="source_img"):
        hf_dataset_export.materialize_hf_dataset(
            "org/image-edits",
            tmp_path,
            media_type="image-edit",
            media_mappings=[
                hf_dataset_export.HFDatasetMediaMapping("target", "target_img"),
                hf_dataset_export.HFDatasetMediaMapping("context", "source_img"),
            ],
            caption_column="instruction",
        )


def test_materialize_hf_dataset_forwards_load_kwargs(tmp_path, monkeypatch):
    calls = []

    def fake_load_dataset(dataset_name, **kwargs):
        calls.append((dataset_name, kwargs))
        return FakeDataset([{"image": Image.new("RGB", (8, 8)), "text": "caption"}])

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

    export = hf_dataset_export.materialize_hf_dataset(
        "org/image-edits",
        tmp_path,
        media_type="image",
        split="dev",
        config_name="default",
        streaming=True,
        trust_remote_code=False,
    )

    assert export.total_items == 1
    assert calls == [
        (
            "org/image-edits",
            {
                "split": "dev",
                "streaming": True,
                "name": "default",
                "trust_remote_code": False,
            },
        )
    ]
