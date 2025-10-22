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

import pytest

from nemo_automodel.components.datasets.llm.megatron_dataset import try_load_blend_from_json


def test_try_load_blend_from_json_success(tmp_path):
    """Test loading a valid JSON blend file with split normalization."""
    blend_config = {
        "train": ["30", "path/to/dataset1", "70", "path/to/dataset2"],
        "valid": ["path/to/val_dataset"],
        "test": ["path/to/test_dataset"],
    }

    json_file = tmp_path / "blend.json"
    with open(json_file, "w") as f:
        json.dump(blend_config, f)

    result = try_load_blend_from_json(json_file)

    assert result is not None
    assert "train" in result
    assert "validation" in result  # 'valid' normalized to 'validation'
    assert "test" in result
    assert result["validation"] == ["path/to/val_dataset"]


def test_try_load_blend_from_json_non_json_returns_none(tmp_path):
    """Test that non-JSON files return None."""
    text_file = tmp_path / "not_json.txt"
    text_file.write_text("some text")

    assert try_load_blend_from_json(text_file) is None


def test_try_load_blend_from_json_file_not_found(tmp_path):
    """Test that FileNotFoundError is raised for non-existent JSON files."""
    with pytest.raises(FileNotFoundError, match="Blend JSON file not found"):
        try_load_blend_from_json(tmp_path / "nonexistent.json")


def test_try_load_blend_from_json_invalid_json(tmp_path):
    """Test that ValueError is raised for malformed JSON."""
    json_file = tmp_path / "invalid.json"
    json_file.write_text("{'invalid': json}")

    with pytest.raises(ValueError, match="Invalid JSON in blend file"):
        try_load_blend_from_json(json_file)


def test_try_load_blend_from_json_wrong_type(tmp_path):
    """Test that ValueError is raised when JSON is not a dictionary."""
    json_file = tmp_path / "list.json"
    with open(json_file, "w") as f:
        json.dump(["not", "a", "dict"], f)

    with pytest.raises(ValueError, match="Blend JSON file must contain a dictionary"):
        try_load_blend_from_json(json_file)




