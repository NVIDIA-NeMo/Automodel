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
"""Guard tests for examples/vlm_finetune/qwen3_omni_asr/train.sh.

These tests exercise only the launcher's input-validation block by setting
PY=/bin/true (so the eventual ``exec`` never actually starts training) and by
pointing WENETSPEECH_WU_PATH at a literal stub string.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
TRAIN_SH = REPO_ROOT / "examples" / "vlm_finetune" / "qwen3_omni_asr" / "train.sh"


@pytest.mark.skipif(not TRAIN_SH.exists(), reason="train.sh not present")
@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not on PATH")
def test_train_sh_rejects_nproc_per_node_other_than_8():
    """The single-node 8-GPU contract must reject any NPROC_PER_NODE != 8."""
    env = {
        **os.environ,
        "NPROC_PER_NODE": "4",
        "WENETSPEECH_WU_PATH": "stub",
        "PY": "/bin/true",
    }
    result = subprocess.run(
        ["bash", str(TRAIN_SH)],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode != 0, (
        f"train.sh accepted NPROC_PER_NODE=4 (stdout={result.stdout!r}, stderr={result.stderr!r})"
    )
    assert "NPROC_PER_NODE must be exactly 8" in (result.stderr + result.stdout)


@pytest.mark.skipif(not TRAIN_SH.exists(), reason="train.sh not present")
@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not on PATH")
@pytest.mark.parametrize("bad_value", ["1", "2", "7", "16", "abc"])
def test_train_sh_rejects_various_bad_nproc_values(bad_value):
    """Any concrete non-8 NPROC_PER_NODE must be rejected.

    Note: an empty string is treated as "use default 8" by bash's ``${VAR:-8}``,
    so it is intentionally NOT included here.
    """
    env = {
        **os.environ,
        "NPROC_PER_NODE": bad_value,
        "WENETSPEECH_WU_PATH": "stub",
        "PY": "/bin/true",
    }
    result = subprocess.run(
        ["bash", str(TRAIN_SH)],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode != 0


@pytest.mark.skipif(not TRAIN_SH.exists(), reason="train.sh not present")
@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not on PATH")
def test_train_sh_rejects_missing_wenetspeech_path():
    """WENETSPEECH_WU_PATH is mandatory; missing must fail-fast with a clear message."""
    env = {k: v for k, v in os.environ.items() if k != "WENETSPEECH_WU_PATH"}
    env["NPROC_PER_NODE"] = "8"
    env["PY"] = "/bin/true"
    result = subprocess.run(
        ["bash", str(TRAIN_SH)],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode != 0
    assert "WENETSPEECH_WU_PATH" in (result.stderr + result.stdout)


@pytest.mark.skipif(not TRAIN_SH.exists(), reason="train.sh not present")
@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not on PATH")
def test_train_sh_passes_guards_with_nproc_8():
    """Setting NPROC_PER_NODE=8 + valid WENETSPEECH_WU_PATH + PY=/bin/true must clear the guards.

    /bin/true ignores all arguments and exits 0, so the script returns 0 if and
    only if the validation block passes and the final ``exec`` lands.
    """
    env = {
        **os.environ,
        "NPROC_PER_NODE": "8",
        "WENETSPEECH_WU_PATH": "stub",
        "PY": "/bin/true",
    }
    result = subprocess.run(
        ["bash", str(TRAIN_SH)],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, (
        f"train.sh failed past guards: stdout={result.stdout!r} stderr={result.stderr!r}"
    )


@pytest.mark.skipif(not TRAIN_SH.exists(), reason="train.sh not present")
def test_train_sh_is_executable():
    """train.sh must be marked executable for the launcher convention."""
    assert os.access(TRAIN_SH, os.X_OK), f"train.sh is not executable: {TRAIN_SH}"


@pytest.mark.skipif(not TRAIN_SH.exists(), reason="train.sh not present")
def test_train_sh_invokes_nemo_automodel_cli_app():
    """The launcher must dispatch to nemo_automodel.cli.app via torch.distributed.run."""
    text = TRAIN_SH.read_text()
    assert "torch.distributed.run" in text
    assert "nemo_automodel.cli.app" in text
