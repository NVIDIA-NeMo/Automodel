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
import logging
import os
from unittest.mock import MagicMock

import pytest
import torch
from safetensors.torch import load_file, save_file

from nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors import (
    _write_sub_tensor_to_file_optimized,
    consolidate_safetensors_files,
    resolve_dtype_cast,
)
from nemo_automodel.components.checkpoint._backports.hf_storage import (
    _DIFFUSERS_INDEX_FN,
    _HuggingFaceStorageWriter,
    _maybe_rename_index_for_diffusers,
)
from nemo_automodel.components.checkpoint._backports.hf_utils import CUSTOM_METADATA_KEY


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


@pytest.mark.run_only_on("CPU")
def test_consolidate_casts_float_tensors_only_when_cast_dtype_is_set(tmp_path, caplog):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    tensors = {
        "float_weight": torch.arange(4, dtype=torch.float32).reshape(2, 2),
        "int_weight": torch.arange(4, dtype=torch.int64),
    }
    dcp_metadata = {name: {"saved_offsets": [0 for _ in tensor.shape]} for name, tensor in tensors.items()}
    save_file(
        tensors,
        input_dir / "model-00001-of-00001.safetensors",
        metadata={CUSTOM_METADATA_KEY: json.dumps(dcp_metadata)},
    )
    caplog.set_level(logging.INFO)

    consolidate_safetensors_files(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        fqn_to_index_mapping={"float_weight": 1, "int_weight": 1},
        cast_dtype=torch.bfloat16,
    )

    output_tensors = load_file(output_dir / "model-00001-of-00001.safetensors")
    assert output_tensors["float_weight"].dtype is torch.bfloat16
    assert output_tensors["int_weight"].dtype is torch.int64
    torch.testing.assert_close(output_tensors["float_weight"], tensors["float_weight"].to(torch.bfloat16))
    torch.testing.assert_close(output_tensors["int_weight"], tensors["int_weight"])

    with open(output_dir / "model.safetensors.index.json", "r") as f:
        index = json.load(f)
    expected_total_size = tensors["float_weight"].numel() * 2 + tensors["int_weight"].numel() * 8
    assert index["metadata"]["total_size"] == expected_total_size
    assert "Requested cast dtype torch.bfloat16 for consolidation." in caplog.text
    assert "tensors already in this dtype, FP8 tensors, and non-floating tensors are unchanged." in caplog.text


@pytest.mark.run_only_on("CPU")
def test_consolidate_cast_dtype_does_not_cast_fp8_tensors(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    tensors = {
        "fp8_weight": torch.arange(4, dtype=torch.float32).to(torch.float8_e4m3fn),
        "fp32_weight": torch.arange(4, dtype=torch.float32),
    }
    dcp_metadata = {name: {"saved_offsets": [0 for _ in tensor.shape]} for name, tensor in tensors.items()}
    save_file(
        tensors,
        input_dir / "model-00001-of-00001.safetensors",
        metadata={CUSTOM_METADATA_KEY: json.dumps(dcp_metadata)},
    )

    consolidate_safetensors_files(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        fqn_to_index_mapping={"fp8_weight": 1, "fp32_weight": 1},
        cast_dtype=torch.bfloat16,
    )

    output_tensors = load_file(output_dir / "model-00001-of-00001.safetensors")
    assert output_tensors["fp8_weight"].dtype is torch.float8_e4m3fn
    assert output_tensors["fp32_weight"].dtype is torch.bfloat16
    torch.testing.assert_close(output_tensors["fp8_weight"].to(torch.float32), tensors["fp8_weight"].to(torch.float32))
    torch.testing.assert_close(output_tensors["fp32_weight"], tensors["fp32_weight"].to(torch.bfloat16))


@pytest.mark.run_only_on("CPU")
def test_consolidate_preserves_original_hf_float_dtypes_when_available(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    tensors = {
        "bf16_weight": torch.arange(4, dtype=torch.float32).reshape(2, 2),
        "fp32_weight": torch.arange(4, dtype=torch.float32),
        "missing_original_dtype": torch.arange(2, dtype=torch.float32),
    }
    dcp_metadata = {name: {"saved_offsets": [0 for _ in tensor.shape]} for name, tensor in tensors.items()}
    save_file(
        tensors,
        input_dir / "model-00001-of-00001.safetensors",
        metadata={CUSTOM_METADATA_KEY: json.dumps(dcp_metadata)},
    )

    consolidate_safetensors_files(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        fqn_to_index_mapping={name: 1 for name in tensors},
        fqn_to_dtype_mapping={"bf16_weight": "BF16", "fp32_weight": "F32"},
    )

    output_tensors = load_file(output_dir / "model-00001-of-00001.safetensors")
    assert output_tensors["bf16_weight"].dtype is torch.bfloat16
    assert output_tensors["fp32_weight"].dtype is torch.float32
    assert output_tensors["missing_original_dtype"].dtype is torch.float32
    torch.testing.assert_close(output_tensors["bf16_weight"], tensors["bf16_weight"].to(torch.bfloat16))
    torch.testing.assert_close(output_tensors["fp32_weight"], tensors["fp32_weight"])
    torch.testing.assert_close(output_tensors["missing_original_dtype"], tensors["missing_original_dtype"])

    with open(output_dir / "model.safetensors.index.json", "r") as f:
        index = json.load(f)
    expected_total_size = (
        tensors["bf16_weight"].numel() * 2
        + tensors["fp32_weight"].numel() * 4
        + tensors["missing_original_dtype"].numel() * 4
    )
    assert index["metadata"]["total_size"] == expected_total_size


@pytest.mark.run_only_on("CPU")
def test_consolidate_keeps_saved_dtype_when_original_hf_dtype_metadata_is_missing(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    tensors = {
        "fp32_weight": torch.arange(4, dtype=torch.float32),
        "bf16_weight": torch.arange(4, dtype=torch.bfloat16),
    }
    dcp_metadata = {name: {"saved_offsets": [0 for _ in tensor.shape]} for name, tensor in tensors.items()}
    save_file(
        tensors,
        input_dir / "model-00001-of-00001.safetensors",
        metadata={CUSTOM_METADATA_KEY: json.dumps(dcp_metadata)},
    )

    consolidate_safetensors_files(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        fqn_to_index_mapping={name: 1 for name in tensors},
    )

    output_tensors = load_file(output_dir / "model-00001-of-00001.safetensors")
    assert output_tensors["fp32_weight"].dtype is torch.float32
    assert output_tensors["bf16_weight"].dtype is torch.bfloat16
    torch.testing.assert_close(output_tensors["fp32_weight"], tensors["fp32_weight"])
    torch.testing.assert_close(output_tensors["bf16_weight"], tensors["bf16_weight"])


@pytest.mark.run_only_on("CPU")
def test_consolidate_keeps_float_when_original_dtype_is_quantized(tmp_path, caplog):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    tensors = {"weight": torch.arange(4, dtype=torch.float32)}
    dcp_metadata = {"weight": {"saved_offsets": [0]}}
    save_file(
        tensors,
        input_dir / "model-00001-of-00001.safetensors",
        metadata={CUSTOM_METADATA_KEY: json.dumps(dcp_metadata)},
    )
    caplog.set_level(logging.WARNING)

    consolidate_safetensors_files(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        fqn_to_index_mapping={"weight": 1},
        fqn_to_dtype_mapping={"weight": "F8_E4M3"},
    )

    output_tensors = load_file(output_dir / "model-00001-of-00001.safetensors")
    assert output_tensors["weight"].dtype is torch.float32
    torch.testing.assert_close(output_tensors["weight"], tensors["weight"])
    assert "Original checkpoint tensor(s) were quantized or packed" in caplog.text
    assert "weight: original F8_E4M3, saved F32" in caplog.text


def test_resolve_dtype_cast_accepts_aliases_and_none():
    assert resolve_dtype_cast(None) is None
    assert resolve_dtype_cast("") is None
    assert resolve_dtype_cast("none") is None
    assert resolve_dtype_cast("bf16") is torch.bfloat16
    assert resolve_dtype_cast("torch.float16") is torch.float16


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
