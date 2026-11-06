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

"""Base-model loads pick the upstream safetensors reader when it can decode every dtype of the checkpoint."""

import json
import struct

import pytest
import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import save_file

from nemo_automodel.components.checkpoint._backports.hf_storage import _HuggingFaceStorageReader
from nemo_automodel.components.checkpoint.checkpointing import (
    Checkpointer,
    CheckpointingConfig,
    _safetensors_decodes_checkpoint_dtypes,
)

torch.distributed.checkpoint.hf_storage = pytest.importorskip("torch.distributed.checkpoint.hf_storage")
from torch.distributed.checkpoint.hf_storage import HuggingFaceStorageReader  # noqa: E402


def _write_e8m0_checkpoint(path) -> None:
    """A minimal safetensors file whose only tensor uses a dtype the safetensors package does not decode."""
    header = {"scale": {"dtype": "F8_E8M0", "shape": [4], "data_offsets": [0, 4]}}
    payload = json.dumps(header).encode()
    with open(path, "wb") as stream:
        stream.write(struct.pack("<Q", len(payload)))
        stream.write(payload)
        stream.write(bytes([0x7F, 0x7F, 0x7F, 0x7F]))


def test_decodable_dtypes_select_the_upstream_reader(tmp_path):
    save_file(
        {
            "w": torch.zeros(4, 4, dtype=torch.bfloat16),
            "codes": torch.zeros(4, 2, dtype=torch.uint8),
            "scales": torch.zeros(4, 1, dtype=torch.float8_e4m3fn),
            "g": torch.tensor(1.0),
        },
        str(tmp_path / "model.safetensors"),
    )
    assert _safetensors_decodes_checkpoint_dtypes(str(tmp_path))
    checkpointer = Checkpointer(
        CheckpointingConfig(
            enabled=True,
            checkpoint_dir=str(tmp_path / "out"),
            model_save_format="safetensors",
            model_cache_dir=str(tmp_path),
            model_repo_id=str(tmp_path),
            save_consolidated=False,
        ),
        dp_rank=0,
        tp_rank=0,
        pp_rank=0,
        moe_mesh=None,
    )
    reader = checkpointer._get_storage_reader(str(tmp_path), key_mapping=None, is_init_step=True, is_safetensors=True)
    assert isinstance(reader, HuggingFaceStorageReader) and not isinstance(reader, _HuggingFaceStorageReader)
    # A key mapping still needs the backport (the upstream reader cannot rename keys).
    mapped = checkpointer._get_storage_reader(
        str(tmp_path), key_mapping={"^w$": "weight"}, is_init_step=True, is_safetensors=True
    )
    assert isinstance(mapped, _HuggingFaceStorageReader)


def test_undecodable_dtypes_keep_the_backport_reader(tmp_path):
    _write_e8m0_checkpoint(tmp_path / "model.safetensors")
    assert not _safetensors_decodes_checkpoint_dtypes(str(tmp_path))
    checkpointer = Checkpointer(
        CheckpointingConfig(
            enabled=True,
            checkpoint_dir=str(tmp_path / "out"),
            model_save_format="safetensors",
            model_cache_dir=str(tmp_path),
            model_repo_id=str(tmp_path),
            save_consolidated=False,
        ),
        dp_rank=0,
        tp_rank=0,
        pp_rank=0,
        moe_mesh=None,
    )
    reader = checkpointer._get_storage_reader(str(tmp_path), key_mapping=None, is_init_step=True, is_safetensors=True)
    assert isinstance(reader, _HuggingFaceStorageReader)


def test_non_local_or_empty_paths_keep_the_backport_reader(tmp_path):
    assert not _safetensors_decodes_checkpoint_dtypes("hf://org/repo")
    assert not _safetensors_decodes_checkpoint_dtypes(str(tmp_path))  # no safetensors files
    (tmp_path / "model.safetensors").write_bytes(b"\x10\x00\x00\x00\x00\x00\x00\x00not json")
    assert not _safetensors_decodes_checkpoint_dtypes(str(tmp_path))  # unreadable header


def _checkpointer(tmp_path, checkpoint_dir) -> Checkpointer:
    return Checkpointer(
        CheckpointingConfig(
            enabled=True,
            checkpoint_dir=str(tmp_path / "out"),
            model_save_format="safetensors",
            model_cache_dir=str(tmp_path),
            model_repo_id=str(checkpoint_dir),
            save_consolidated=False,
        ),
        dp_rank=0,
        tp_rank=0,
        pp_rank=0,
        moe_mesh=None,
    )


def test_upstream_reader_parses_the_headers_once_per_checkpoint(tmp_path, monkeypatch):
    """A part-by-part base load issues one dcp.load per part with one reader: the headers are parsed once."""
    first, second = tmp_path / "first", tmp_path / "second"
    for directory in (first, second):
        directory.mkdir()
        save_file({"a": torch.ones(2, 2)}, str(directory / "model-00001-of-00002.safetensors"))
        save_file({"b": torch.full((3,), 2.0)}, str(directory / "model-00002-of-00002.safetensors"))
    parses = []
    original = HuggingFaceStorageReader.read_metadata

    def counting_read_metadata(self):
        parses.append(str(self.path))
        return original(self)

    monkeypatch.setattr(HuggingFaceStorageReader, "read_metadata", counting_read_metadata)
    reader = _checkpointer(tmp_path, first)._get_storage_reader(
        str(first), key_mapping=None, is_init_step=True, is_safetensors=True
    )
    assert isinstance(reader, HuggingFaceStorageReader)

    part_a = {"a": torch.zeros(2, 2)}
    dcp.load(part_a, checkpoint_id=str(first), storage_reader=reader)
    part_b = {"b": torch.zeros(3)}
    dcp.load(part_b, checkpoint_id=str(first), storage_reader=reader)

    assert torch.equal(part_a["a"], torch.ones(2, 2)) and torch.equal(part_b["b"], torch.full((3,), 2.0))
    assert parses == [str(first)]
    # Moving the reader to another checkpoint drops the cache.
    dcp.load({"a": torch.zeros(2, 2)}, checkpoint_id=str(second), storage_reader=reader)
    assert parses == [str(first), str(second)]
