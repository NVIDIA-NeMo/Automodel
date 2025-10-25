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
import sys
from pathlib import Path

# Ensure tools directory is on sys.path so we can import the module directly
TOOLS_DIR = Path(__file__).resolve().parents[3] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import pytest


def test_copy_metadata_files_moves_and_cleans(tmp_path):
    from offline_hf_consolidation import copy_metadata_files

    meta_dir = tmp_path / ".hf_metadata"
    out_dir = tmp_path / "out"
    meta_dir.mkdir()
    out_dir.mkdir()

    # Files: mapping (should be skipped) and another metadata file (should be moved)
    (meta_dir / "fqn_to_file_index_mapping.json").write_text(json.dumps({"layer.weight": 0}))
    (meta_dir / "config.json").write_text("{}")

    copy_metadata_files(str(meta_dir), str(out_dir))

    # Input metadata directory should be removed
    assert not meta_dir.exists()
    # Non-mapping files moved
    assert (out_dir / "config.json").exists()
    # Mapping file should not be copied into the output directory here
    assert not (out_dir / "fqn_to_file_index_mapping.json").exists()


def test_main_happy_path_calls_consolidate_and_copies(tmp_path, monkeypatch):
    import offline_hf_consolidation as script

    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    meta_dir = in_dir / ".hf_metadata"
    meta_dir.mkdir(parents=True)
    out_dir.mkdir(parents=True)

    mapping = {"layer.weight": 0}
    (meta_dir / "fqn_to_file_index_mapping.json").write_text(json.dumps(mapping))
    (meta_dir / "config.json").write_text("{}")

    # Make input dir itself exist (the script expects it)
    in_dir.mkdir(exist_ok=True)

    # Patch out distributed and consolidation side effects
    monkeypatch.setattr(script, "initialize_distributed", lambda backend: None)
    monkeypatch.setattr(script, "get_world_size_safe", lambda: 1)
    monkeypatch.setattr(script, "get_rank_safe", lambda: 0)

    captured = {}

    def fake_consolidate(input_dir, output_dir, fqn_to_index_mapping, num_threads):
        captured["args"] = (input_dir, output_dir, fqn_to_index_mapping, num_threads)

    monkeypatch.setattr(script, "consolidate_safetensors_files_on_every_rank", fake_consolidate)

    # Provide CLI args; force backend to gloo to avoid depending on CUDA availability
    argv = [
        "prog",
        "--model-name",
        "meta-llama/Llama-3.2-1B",
        "--input-dir",
        str(in_dir),
        "--output-dir",
        str(out_dir),
        "--num-threads",
        "2",
        "--backend",
        "gloo",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    script.main()

    # Consolidation called with expected arguments
    assert "args" in captured
    called_in_dir, called_out_dir, called_mapping, called_threads = captured["args"]
    assert Path(called_in_dir) == in_dir
    assert Path(called_out_dir) == out_dir
    assert called_mapping == mapping
    assert called_threads == 2

    # Metadata copied (except mapping) and source metadata dir removed
    assert (out_dir / "config.json").exists()
    assert not (out_dir / "fqn_to_file_index_mapping.json").exists()
    assert not meta_dir.exists()


def test_main_raises_if_missing_metadata(tmp_path, monkeypatch):
    import offline_hf_consolidation as script

    in_dir = tmp_path / "in_no_meta"
    out_dir = tmp_path / "out_no_meta"
    in_dir.mkdir()

    monkeypatch.setattr(script, "initialize_distributed", lambda backend: None)
    monkeypatch.setattr(script, "get_world_size_safe", lambda: 1)
    monkeypatch.setattr(script, "get_rank_safe", lambda: 0)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--model-name",
            "meta-llama/Llama-3.2-1B",
            "--input-dir",
            str(in_dir),
            "--output-dir",
            str(out_dir),
            "--backend",
            "gloo",
        ],
    )

    with pytest.raises(FileNotFoundError):
        script.main()


