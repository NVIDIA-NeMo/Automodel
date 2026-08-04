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

"""Tests for ``try_load_blend_from_json`` accepting the Megatron-LM flat-list form.

The same ``MegatronPretraining.paths`` docstring already documents flat-zipped
``["w1", "p1", "w2", "p2", ...]`` as a valid *inline* list form. This test
suite covers the symmetric JSON-file path so users emitting blends in the
canonical Megatron-LM convention (and from ecosystem tooling such as
Megatron-Bridge) do not need to wrap their config in a dict-of-splits just to
satisfy the parser.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nemo_automodel.components.datasets.llm.megatron_dataset import try_load_blend_from_json


def _write_json(tmp_path: Path, name: str, payload) -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(payload))
    return p


class TestFlatListBlendJson:
    def test_flat_list_returned_as_is(self, tmp_path):
        blend = ["0.3", "/data/dataset_1", "0.7", "/data/dataset_2"]
        p = _write_json(tmp_path, "blend.json", blend)
        assert try_load_blend_from_json(p) == blend

    def test_flat_list_of_prefixes_only(self, tmp_path):
        """Flat list of prefixes without weights is also accepted."""
        blend = ["/data/dataset_1", "/data/dataset_2"]
        p = _write_json(tmp_path, "blend.json", blend)
        assert try_load_blend_from_json(p) == blend

    def test_flat_list_with_s3_prefixes(self, tmp_path):
        """Megatron-LM blends typically interleave weights and (s3) prefixes."""
        blend = [
            "0.5",
            "s3://bucket/shard1",
            "0.5",
            "s3://bucket/shard2",
        ]
        p = _write_json(tmp_path, "blend.json", blend)
        assert try_load_blend_from_json(p) == blend


class TestDictOfSplitsStillWorks:
    """Regression: pre-existing dict-of-splits form must remain unchanged."""

    def test_dict_passes_through_with_normalization(self, tmp_path):
        blend = {
            "train": ["30", "/data/ds1", "70", "/data/ds2"],
            "valid": ["/data/val_ds"],
        }
        p = _write_json(tmp_path, "blend.json", blend)
        out = try_load_blend_from_json(p)
        assert out["train"] == blend["train"]
        # 'valid' alias normalized to 'validation'
        assert out["validation"] == blend["valid"]
        assert "valid" not in out


class TestInvalidJson:
    def test_scalar_root_rejected(self, tmp_path):
        p = _write_json(tmp_path, "blend.json", 42)
        with pytest.raises(ValueError, match="must contain a list or dictionary"):
            try_load_blend_from_json(p)

    def test_non_json_extension_returns_none(self, tmp_path):
        p = tmp_path / "blend.yaml"
        p.write_text("ignored")
        assert try_load_blend_from_json(p) is None

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            try_load_blend_from_json(tmp_path / "nope.json")

    def test_malformed_json_raises(self, tmp_path):
        p = tmp_path / "blend.json"
        p.write_text("{ not valid json ")
        with pytest.raises(ValueError, match="Invalid JSON"):
            try_load_blend_from_json(p)
