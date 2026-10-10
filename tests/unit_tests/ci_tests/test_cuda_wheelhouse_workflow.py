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

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

_MODE_SCRIPT = Path(".github/scripts/select-uv-install-modes.sh").resolve()
_VERIFY_SCRIPT = Path(".github/scripts/verify-cuda-wheelhouse-provenance.sh").resolve()


def _git(directory: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(directory), *args], text=True).strip()


@pytest.mark.parametrize(
    ("event", "path", "force", "expect_source"),
    [
        ("push", "README.md", "false", False),
        ("push", "pyproject.toml", "false", True),
        ("push", "uv.lock", "false", True),
        ("push", ".github/scripts/build-cuda-wheelhouse.sh", "false", True),
        ("push", ".github/workflows/install-test.yml", "false", True),
        ("schedule", "README.md", "false", True),
        ("workflow_dispatch", "README.md", "true", True),
    ],
)
def test_source_coverage_selection(tmp_path: Path, event: str, path: str, force: str, expect_source: bool) -> None:
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.name", "Test")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "-c", "commit.gpgsign=false", "commit", "--allow-empty", "-qm", "base")
    _git(tmp_path, "update-ref", "refs/remotes/origin/main", "HEAD")
    target = tmp_path / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("changed\n")
    _git(tmp_path, "add", path)
    _git(tmp_path, "-c", "commit.gpgsign=false", "commit", "-qm", "change")
    output = tmp_path / "output"
    subprocess.run(
        ["bash", str(_MODE_SCRIPT)],
        cwd=tmp_path,
        env={
            **os.environ,
            "GITHUB_EVENT_NAME": event,
            "GITHUB_REF_NAME": "pull-request/1",
            "FORCE_SOURCE_BUILD": force,
            "GITHUB_OUTPUT": str(output),
        },
        check=True,
    )
    modes = json.loads(output.read_text().split("=", 1)[1])
    assert modes == (["wheelhouse", "source"] if expect_source else ["wheelhouse"])


@pytest.mark.parametrize("cache_hit", [True, False])
@pytest.mark.parametrize("valid_signature", [True, False])
def test_provenance_gate_requires_correct_identity_and_success(
    tmp_path: Path, cache_hit: bool, valid_signature: bool
) -> None:
    # Capture the verifier policy and simulate its exit status; real signature
    # verification remains the responsibility of `gh attestation verify`.
    gh = tmp_path / "gh"
    gh.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$ARGS_FILE"\nexit "$VERIFY_STATUS"\n')
    gh.chmod(0o755)
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"wheels":{}}\n')
    args_file = tmp_path / "arguments"
    output = tmp_path / "output"
    result = subprocess.run(
        ["bash", str(_VERIFY_SCRIPT)],
        env={
            **os.environ,
            "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
            "ARGS_FILE": str(args_file),
            "VERIFY_STATUS": "0" if valid_signature else "1",
            "GITHUB_OUTPUT": str(output),
            "GITHUB_REPOSITORY": "NVIDIA-NeMo/Automodel",
            "GITHUB_REF": "refs/heads/pull-request/1",
            "GITHUB_SHA": "a" * 40,
            "WHEELHOUSE_DIR": str(tmp_path),
            "WHEELHOUSE_CACHE_HIT": str(cache_hit).lower(),
        },
        check=False,
    )
    args = args_file.read_text().splitlines()
    assert args[args.index("--repo") + 1] == "NVIDIA-NeMo/Automodel"
    assert args[args.index("--signer-workflow") + 1] == "NVIDIA-NeMo/Automodel/.github/workflows/install-test.yml"
    assert args[args.index("--source-ref") + 1] == ("refs/heads/main" if cache_hit else "refs/heads/pull-request/1")
    if not cache_hit:
        assert args[args.index("--source-digest") + 1] == "a" * 40
        assert args[args.index("--signer-digest") + 1] == "a" * 40
    if valid_signature:
        assert result.returncode == 0
        assert output.read_text() == f"manifest_sha256={hashlib.sha256(manifest.read_bytes()).hexdigest()}\n"
    else:
        assert result.returncode != 0
        assert not output.exists()
