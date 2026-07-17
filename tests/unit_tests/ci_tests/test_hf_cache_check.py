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

"""Tests for ``tests/ci_tests/utils/hf_cache_check.py``."""

import json
import sys
from pathlib import Path

import pytest
import yaml

UTILS_DIR = Path(__file__).resolve().parents[3] / "tests" / "ci_tests" / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

import hf_cache_check  # noqa: E402

LLM_RECIPE = {
    "model": {
        "_target_": "nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained",
        "pretrained_model_name_or_path": "meta-llama/Llama-3.2-1B",
    },
    "dataset": {"path_or_dataset": "rowan/hellaswag"},
    "ci": {
        "checkpoint_robustness": {
            "model": {
                "pretrained_model_name_or_path": "meta-llama/Llama-3.2-3B-Instruct",
            },
            "tokenizer_name": "meta-llama/Llama-3.2-3B-Instruct",
        },
    },
}

VLM_RECIPE = {
    "model": {
        "_target_": "nemo_automodel.NeMoAutoModelForImageTextToText.from_pretrained",
        "pretrained_model_name_or_path": "Qwen/Qwen3-VL-4B-Thinking",
    },
    "processor": {
        "_target_": "transformers.AutoProcessor.from_pretrained",
        "pretrained_model_name_or_path": "Qwen/Qwen3-VL-4B-Thinking",
    },
    "dataset": {"path_or_dataset": "mmoukouba/MedPix-VQA"},
}

LOCAL_PATH_RECIPE = {
    "model": {"pretrained_model_name_or_path": "/opt/checkpoints/local"},
    "processor": {"pretrained_model_name_or_path": "./relative/path"},
    "ci": {"checkpoint_robustness": {"tokenizer_name": "meta-llama/Llama-3.2-1B"}},
}


def _write_recipe(tmp_path: Path, name: str, recipe: dict) -> Path:
    path = tmp_path / name
    # sort_keys=False preserves insertion order so the walker's output is stable
    # (real recipes are hand-authored top-down; sorted dumps would obscure that).
    path.write_text(yaml.safe_dump(recipe, sort_keys=False), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Recipe walker
# ---------------------------------------------------------------------------


def test_extract_repo_ids_llm_recipe(tmp_path):
    path = _write_recipe(tmp_path, "llm.yaml", LLM_RECIPE)
    assert hf_cache_check._extract_repo_ids(path) == [
        "meta-llama/Llama-3.2-1B",
        "meta-llama/Llama-3.2-3B-Instruct",
    ]


def test_extract_repo_ids_vlm_recipe_dedupes(tmp_path):
    # model + processor reference the same repo — expect one entry.
    path = _write_recipe(tmp_path, "vlm.yaml", VLM_RECIPE)
    assert hf_cache_check._extract_repo_ids(path) == ["Qwen/Qwen3-VL-4B-Thinking"]


def test_extract_repo_ids_skips_local_paths(tmp_path):
    # Absolute and relative filesystem paths are not HF repo IDs; the tokenizer
    # string is the only genuine repo id and should survive.
    path = _write_recipe(tmp_path, "local.yaml", LOCAL_PATH_RECIPE)
    assert hf_cache_check._extract_repo_ids(path) == ["meta-llama/Llama-3.2-1B"]


@pytest.mark.parametrize(
    "value,expected",
    [
        ("meta-llama/Llama-3.2-1B", True),
        ("Qwen/Qwen3-VL-4B-Thinking", True),
        ("/opt/checkpoints/local", False),
        ("./relative/path", False),
        ("no-slash", False),
        ("too/many/slashes", False),
        ("", False),
        ("has space/name", False),
    ],
)
def test_is_hf_repo_id(value, expected):
    assert hf_cache_check._is_hf_repo_id(value) is expected


# ---------------------------------------------------------------------------
# Cache inspection
# ---------------------------------------------------------------------------


def _seed_cache(hf_home: Path, repo_id: str, with_file: bool = True) -> None:
    """Create a ``$HF_HOME/hub/models--<org>--<name>/snapshots/<rev>/`` layout."""
    folder = "models--" + repo_id.replace("/", "--")
    rev_dir = hf_home / "hub" / folder / "snapshots" / "abc123"
    rev_dir.mkdir(parents=True, exist_ok=True)
    if with_file:
        (rev_dir / "config.json").write_text("{}", encoding="utf-8")


def test_check_cache_partitions_hits_and_misses(tmp_path):
    hf_home = tmp_path / "hf_home"
    _seed_cache(hf_home, "meta-llama/Llama-3.2-1B")
    # 3B-Instruct is deliberately left un-seeded so it registers as missing.

    recipe = _write_recipe(tmp_path, "llm.yaml", LLM_RECIPE)
    result = hf_cache_check.check_cache(recipe, hf_home)

    assert result["cached"] == ["meta-llama/Llama-3.2-1B"]
    assert result["missing"] == ["meta-llama/Llama-3.2-3B-Instruct"]
    assert result["hf_home"] == str(hf_home)


def test_check_cache_empty_snapshot_dir_counts_as_missing(tmp_path):
    hf_home = tmp_path / "hf_home"
    _seed_cache(hf_home, "meta-llama/Llama-3.2-1B", with_file=False)
    _seed_cache(hf_home, "meta-llama/Llama-3.2-3B-Instruct", with_file=False)

    recipe = _write_recipe(tmp_path, "llm.yaml", LLM_RECIPE)
    result = hf_cache_check.check_cache(recipe, hf_home)

    assert result["cached"] == []
    assert set(result["missing"]) == {
        "meta-llama/Llama-3.2-1B",
        "meta-llama/Llama-3.2-3B-Instruct",
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_main_writes_json_output(tmp_path, monkeypatch, capsys):
    hf_home = tmp_path / "hf_home"
    _seed_cache(hf_home, "Qwen/Qwen3-VL-4B-Thinking")
    monkeypatch.setenv("HF_HOME", str(hf_home))

    recipe = _write_recipe(tmp_path, "vlm.yaml", VLM_RECIPE)
    output = tmp_path / "out.json"

    monkeypatch.setattr(sys, "argv", ["hf_cache_check.py", "--config", str(recipe), "--output", str(output)])
    assert hf_cache_check.main() == 0

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["cached"] == ["Qwen/Qwen3-VL-4B-Thinking"]
    assert payload["missing"] == []


def test_main_without_hf_home_marks_everything_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_HOME", raising=False)

    recipe = _write_recipe(tmp_path, "llm.yaml", LLM_RECIPE)
    output = tmp_path / "out.json"

    monkeypatch.setattr(sys, "argv", ["hf_cache_check.py", "--config", str(recipe), "--output", str(output)])
    assert hf_cache_check.main() == 0

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["cached"] == []
    assert set(payload["missing"]) == {
        "meta-llama/Llama-3.2-1B",
        "meta-llama/Llama-3.2-3B-Instruct",
    }
    assert payload["hf_home"] == ""
