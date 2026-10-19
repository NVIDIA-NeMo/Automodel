# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

import os
import re
import subprocess
from pathlib import Path

from packaging.requirements import Requirement
from transformers import AutoModel
from transformers.models.gemma4.configuration_gemma4 import Gemma4VisionConfig

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # ty: ignore[unresolved-import]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SETUP_SCRIPT = _REPO_ROOT / "examples/convergence/tulu3/eval/setup_lm_eval.sh"
_PYPROJECT = _REPO_ROOT / "pyproject.toml"


def test_eval_transformers_pin_matches_training_environment() -> None:
    """The isolated vLLM environment must understand configs exported during training."""
    dependencies = tomllib.loads(_PYPROJECT.read_text())["project"]["dependencies"]
    project_requirement = next(
        Requirement(value) for value in dependencies if Requirement(value).name == "transformers"
    )
    setup_versions = re.findall(r'"transformers==([^" ]+)"', _SETUP_SCRIPT.read_text())

    assert setup_versions == [str(project_requirement.specifier).removeprefix("==")]


def test_setup_refreshes_existing_eval_environment(tmp_path: Path, monkeypatch) -> None:
    """A reused virtual environment must still receive the current dependency pins."""
    install_dir = tmp_path / "lm-evaluation-harness"
    (install_dir / ".git").mkdir(parents=True)
    activate = install_dir / ".venv/bin/activate"
    activate.parent.mkdir(parents=True)
    activate.touch()

    calls = tmp_path / "uv-calls"
    uv = tmp_path / "uv"
    uv.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$UV_CALLS"\n')
    uv.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    monkeypatch.setenv("UV_CALLS", str(calls))

    subprocess.run(["bash", str(_SETUP_SCRIPT), str(install_dir)], check=True)

    assert "pip install -e .[vllm] vllm==0.28.0 transformers==5.18.0 nvidia-cutlass-dsl>=4.6.0" in calls.read_text()


def test_eval_transformers_supports_gemma4_axial_vision_rope() -> None:
    """Construct the Gemma4 vision path that vLLM initializes for text evaluation."""
    config = Gemma4VisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        position_embedding_size=16,
        rope_parameters={"rope_type": "axial", "rope_theta": 100.0},
    )

    model = AutoModel.from_config(config)

    assert model.config.rope_parameters["rope_type"] == "axial"
