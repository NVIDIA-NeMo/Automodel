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
"""Tests for the generic image-edit preprocessing CLI."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from tools.diffusion.data.hf_dataset_export import HFDatasetExport, HFDatasetMediaMapping

if importlib.util.find_spec("cv2") is None:
    sys.modules["cv2"] = ModuleType("cv2")

from tools.diffusion import preprocessing_multiprocess


def test_image_edit_cli_materializes_and_invokes_configured_encoder(tmp_path, monkeypatch):
    calls = {}
    source_manifest = tmp_path / "materialized" / "hf_image_edit_manifest.jsonl"

    def fake_materialize(dataset_name, output_dir, **kwargs):
        calls["materialize"] = (dataset_name, Path(output_dir), kwargs)
        return HFDatasetExport(
            media_dir=Path(output_dir),
            total_items=1,
            media_column="target_img",
            caption_column="instruction",
            media_mappings=tuple(kwargs["media_mappings"]),
            manifest_file=source_manifest,
            dataset_revision="a" * 40,
            dataset_config_name="magicbrush",
        )

    class FakeEncoder:
        def __init__(self, *, model_name, revision, max_sequence_length):
            calls["constructor"] = (model_name, revision, max_sequence_length)

        def encode_manifest(
            self,
            *,
            manifest_path,
            output_dir,
            max_pixels,
            resolution_preset,
            num_gpus,
            verify,
        ):
            calls["encode"] = {
                "manifest_path": manifest_path,
                "output_dir": output_dir,
                "max_pixels": max_pixels,
                "resolution_preset": resolution_preset,
                "num_gpus": num_gpus,
                "verify": verify,
            }
            return output_dir / "metadata.json"

    monkeypatch.setattr(preprocessing_multiprocess, "materialize_hf_dataset", fake_materialize)
    monkeypatch.setattr(preprocessing_multiprocess, "_resolve_processor_target", lambda target: FakeEncoder)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "preprocessing_multiprocess.py",
            "image-edit",
            "--dataset_name",
            "org/image-edits",
            "--dataset_revision",
            "pinned-revision",
            "--dataset_split",
            "dev",
            "--dataset_streaming",
            "--dataset_media_mapping",
            "target=target_img",
            "--dataset_media_mapping",
            "context=source_img",
            "--dataset_media_mapping",
            "condition=source_img",
            "--dataset_caption_column",
            "instruction",
            "--processor_target",
            "package.module.Encoder",
            "--model_name",
            "org/model",
            "--model_revision",
            "model-branch",
            "--max_sequence_length",
            "384",
            "--max_items",
            "64",
            "--resolution_preset",
            "1024p",
            "--num_gpus",
            "8",
            "--verify",
            "--output_dir",
            str(tmp_path / "cache"),
        ],
    )
    preprocessing_multiprocess.main()

    dataset_name, export_dir, materialize_kwargs = calls["materialize"]
    assert dataset_name == "org/image-edits"
    assert export_dir == tmp_path / "cache" / "_hf_dataset" / "image_edit"
    assert materialize_kwargs["media_type"] == "image-edit"
    assert materialize_kwargs["revision"] == "pinned-revision"
    assert materialize_kwargs["split"] == "dev"
    assert materialize_kwargs["streaming"] is True
    assert materialize_kwargs["max_items"] == 64
    assert materialize_kwargs["media_mappings"] == [
        HFDatasetMediaMapping("target", "target_img"),
        HFDatasetMediaMapping("context", "source_img"),
        HFDatasetMediaMapping("condition", "source_img"),
    ]
    assert calls["constructor"] == ("org/model", "model-branch", 384)
    assert calls["encode"] == {
        "manifest_path": source_manifest,
        "output_dir": tmp_path / "cache",
        "max_pixels": 1024 * 1024,
        "resolution_preset": "1024p",
        "num_gpus": 8,
        "verify": True,
    }


def test_image_edit_preprocessing_requires_encoder_contract(tmp_path, monkeypatch):
    export = HFDatasetExport(
        media_dir=tmp_path,
        total_items=1,
        media_column="target_img",
        caption_column="instruction",
        manifest_file=tmp_path / "manifest.jsonl",
    )

    class MissingEncodeManifest:
        def __init__(self, *, model_name, revision, max_sequence_length):
            pass

    monkeypatch.setattr(preprocessing_multiprocess, "materialize_hf_dataset", lambda *args, **kwargs: export)
    monkeypatch.setattr(
        preprocessing_multiprocess,
        "_resolve_processor_target",
        lambda target: MissingEncodeManifest,
    )

    with pytest.raises(TypeError, match=r"must implement encode_manifest\(\.\.\.\)"):
        preprocessing_multiprocess._preprocess_image_edit_dataset(
            dataset_name="org/image-edits",
            dataset_revision="pinned-revision",
            dataset_split="dev",
            dataset_config_name=None,
            dataset_media_mappings=[
                HFDatasetMediaMapping("target", "target_img"),
                HFDatasetMediaMapping("context", "source_img"),
            ],
            dataset_caption_column="instruction",
            dataset_dir=None,
            dataset_streaming=True,
            dataset_trust_remote_code=None,
            output_dir=str(tmp_path / "cache"),
            processor_target="package.module.Encoder",
            model_name=None,
            model_revision=None,
            max_sequence_length=512,
            max_items=64,
            max_pixels=1024 * 1024,
            resolution_preset="1024p",
            num_gpus=8,
            verify=False,
        )
