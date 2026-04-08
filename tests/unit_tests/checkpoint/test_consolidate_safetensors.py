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
import os
from unittest.mock import MagicMock

import pytest

from nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors import (
    _write_sub_tensor_to_file_optimized,
)
from nemo_automodel.components.checkpoint._backports.hf_storage import (
    _DIFFUSERS_INDEX_FN,
    _HuggingFaceStorageWriter,
    _maybe_rename_index_for_diffusers,
)


@pytest.mark.run_only_on("CPU")
def test_write_scalar_tensor(tmp_path):
    """Ensure that a 0-dim (scalar) tensor shard is written to the output file.

    Regression test for a bug where `_write_sub_tensor_to_file_optimized` used to
    early-return on ``tensor_shape == []`` and therefore omitted scalar payloads,
    which produced corrupt `.safetensors` files (incomplete metadata).
    """

    # Prepare an empty temporary output file
    output_file = tmp_path / "scalar_tensor.bin"
    output_file.write_bytes(b"")  # create the file

    # Fake scalar tensor payload (2-byte BF16 value)
    sub_tensor_bytes = b"\x34\x12"
    element_size = len(sub_tensor_bytes)  # 2 bytes for BF16

    # Prepare destination buffer for a scalar tensor (element_size bytes)
    full_tensor_mv = memoryview(bytearray(element_size))

    # Call the routine under test: scalar has empty shapes and offsets
    _write_sub_tensor_to_file_optimized(
        full_tensor_mv,
        sub_tensor_bytes,
        element_size,
        tensor_shape=[],  # scalar
        sub_tensor_offsets=[],
        sub_tensor_shape=[],
    )

    # Emulate file write as done by the caller in production code
    output_file.write_bytes(full_tensor_mv.tobytes())

    # The file must now contain exactly the scalar payload
    written = output_file.read_bytes()
    assert written == sub_tensor_bytes
    assert os.path.getsize(output_file) == element_size


# =============================================================================
# Tests for _maybe_rename_index_for_diffusers
# =============================================================================


class TestMaybeRenameIndexForDiffusers:
    """Tests for the shared rename helper."""

    def test_renames_when_index_exists(self, tmp_path):
        index_file = tmp_path / "model.safetensors.index.json"
        index_file.write_text('{"metadata": {}}')

        _maybe_rename_index_for_diffusers(str(tmp_path))

        assert not index_file.exists()
        assert (tmp_path / _DIFFUSERS_INDEX_FN).exists()
        assert json.loads((tmp_path / _DIFFUSERS_INDEX_FN).read_text()) == {"metadata": {}}

    def test_noop_when_index_missing(self, tmp_path):
        """No error when the source index file does not exist."""
        _maybe_rename_index_for_diffusers(str(tmp_path))

        assert not (tmp_path / _DIFFUSERS_INDEX_FN).exists()

    def test_preserves_other_files(self, tmp_path):
        """Other files in the directory are untouched."""
        index_file = tmp_path / "model.safetensors.index.json"
        index_file.write_text("{}")
        other_file = tmp_path / "model-00001-of-00001.safetensors"
        other_file.write_bytes(b"\x00")

        _maybe_rename_index_for_diffusers(str(tmp_path))

        assert other_file.exists()
        assert other_file.read_bytes() == b"\x00"


# =============================================================================
# Tests for _HuggingFaceStorageWriter.finish — single-rank consolidation path
# =============================================================================


class TestStorageWriterFinishDiffusersCompatible:
    """Tests that finish() renames the index when diffusers_compatible=True."""

    @pytest.fixture()
    def _mock_consolidate(self, monkeypatch):
        """Patch consolidate_safetensors_files so finish() doesn't need real shards."""
        self._consolidate_calls = []

        def _fake(*, input_dir, output_dir, **kwargs):
            self._consolidate_calls.append(output_dir)
            index = os.path.join(output_dir, "model.safetensors.index.json")
            with open(index, "w") as f:
                json.dump({"weight_map": {}}, f)

        monkeypatch.setattr(
            "nemo_automodel.components.checkpoint._backports.hf_storage.consolidate_safetensors_files",
            _fake,
        )

    @pytest.mark.usefixtures("_mock_consolidate")
    def test_finish_renames_index_when_diffusers_compatible(self, tmp_path):
        consolidated_dir = tmp_path / "consolidated"
        consolidated_dir.mkdir()

        writer = _HuggingFaceStorageWriter(
            path=str(tmp_path / "shards"),
            save_sharded=True,
            consolidated_output_path=str(consolidated_dir),
            diffusers_compatible=True,
        )

        writer.finish(metadata=MagicMock(), results=[[]])

        assert not (consolidated_dir / "model.safetensors.index.json").exists()
        assert (consolidated_dir / _DIFFUSERS_INDEX_FN).exists()

    @pytest.mark.usefixtures("_mock_consolidate")
    def test_finish_preserves_index_name_when_not_diffusers_compatible(self, tmp_path):
        consolidated_dir = tmp_path / "consolidated"
        consolidated_dir.mkdir()

        writer = _HuggingFaceStorageWriter(
            path=str(tmp_path / "shards"),
            save_sharded=True,
            consolidated_output_path=str(consolidated_dir),
            diffusers_compatible=False,
        )

        writer.finish(metadata=MagicMock(), results=[[]])

        assert (consolidated_dir / "model.safetensors.index.json").exists()
        assert not (consolidated_dir / _DIFFUSERS_INDEX_FN).exists()

    def test_finish_early_return_when_no_consolidated_path(self):
        """finish() returns early when no consolidated_output_path is set."""
        writer = _HuggingFaceStorageWriter(
            path="/fake/path",
            save_sharded=True,
            consolidated_output_path=None,
            diffusers_compatible=True,
        )

        # Should not raise — returns before attempting consolidation or rename
        writer.finish(metadata=MagicMock(), results=[[]])
