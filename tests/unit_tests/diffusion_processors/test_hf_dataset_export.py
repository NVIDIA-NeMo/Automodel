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

import pytest
from PIL import Image

from tools.diffusion.data import hf_dataset_export


class FakeDataset:
    """Small dataset-like object for materialization tests."""

    def __init__(self, rows):
        self.rows = rows
        self.column_names = list(rows[0]) if rows else []
        self.features = {}

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
