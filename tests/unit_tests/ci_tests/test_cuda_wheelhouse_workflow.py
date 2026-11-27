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
import sys
from pathlib import Path

import pytest
import yaml

_MODE_SCRIPT = Path(".github/scripts/select-uv-install-modes.sh").resolve()
_VERIFY_SCRIPT = Path(".github/scripts/verify-cuda-wheelhouse-provenance.sh").resolve()


@pytest.mark.parametrize("mode", ["source", "wheelhouse"])
def test_uv_install_keeps_wheel_only_wrapper_and_builds_native_extensions(tmp_path: Path, mode: str) -> None:
    workflow = yaml.safe_load(Path(".github/workflows/install-test.yml").read_text())
    step = next(step for step in workflow["jobs"]["ngc-cuda-test-uv"]["steps"] if step["name"] == "Install project")
    script = step["run"]
    script = script[script.index('requirements="$(') : script.index("\nuv pip install --no-deps -e .")]
    script = script.replace("${{ matrix.python-version }}", "3.12")
    requirements = [
        "causal-conv1d==1.6.0",
        "flash-attn==2.8.3",
        "mamba-ssm==2.3.0",
        "transformer-engine[pytorch]==2.19.0",
        "transformer-engine-torch==2.19.0",
    ]
    python = tmp_path / "python"
    python.write_text(f"#!{sys.executable}\nprint({chr(10).join(requirements)!r})\n")
    python.chmod(0o755)
    calls = tmp_path / "calls.jsonl"
    uv = tmp_path / "uv"
    uv.write_text(
        f"#!{sys.executable}\nimport json, os, sys\n"
        "with open(os.environ['UV_CALLS'], 'a') as output:\n"
        "    output.write(json.dumps({'args': sys.argv[1:], 'mamba': os.getenv('MAMBA_FORCE_BUILD'), "
        "'causal': os.getenv('CAUSAL_CONV1D_FORCE_BUILD')}) + '\\n')\n"
    )
    uv.chmod(0o755)
    subprocess.run(
        ["bash", "-eu", "-o", "pipefail", "-c", script],
        env={
            **os.environ,
            "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
            "INSTALL_MODE": mode,
            "TORCH_CU_INDEX": "https://download.pytorch.org/whl/cu130",
            "UV_CALLS": str(calls),
        },
        check=True,
    )
    invocations = [json.loads(line) for line in calls.read_text().splitlines()]
    sync = invocations[-1]
    assert sync["args"][:8] == [
        "sync",
        "--link-mode",
        "copy",
        "--locked",
        "--all-groups",
        "--no-group",
        "magi",
        "--extra",
    ]
    if mode == "source":
        assert len(invocations) == 1
        assert sync["args"][8:] == [
            "all",
            "--no-cache",
            "--no-binary-package",
            "causal-conv1d",
            "--no-binary-package",
            "flash-attn",
            "--no-binary-package",
            "mamba-ssm",
            "--no-binary-package",
            "transformer-engine-torch",
        ]
        assert sync["mamba"] == sync["causal"] == "TRUE"
    else:
        assert len(invocations) == 2
        assert invocations[0]["args"] == [
            "pip",
            "install",
            "--no-config",
            "--python",
            "./venv/bin/python",
            "--no-deps",
            "--no-index",
            "--only-binary",
            ":all:",
            "--find-links",
            "/tmp/cuda-wheelhouse",
            *requirements,
        ]
        assert sync["args"][8:] == ["all"]


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
            "WHEELHOUSE_REUSED": str(cache_hit).lower(),
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


@pytest.mark.parametrize(
    ("branch", "target", "release", "writer"),
    [
        ("main", "", "", "true"),
        ("r0.7.0", "", "r0.7.0", "true"),
        ("r0.10.0", "", "r0.10.0", "true"),
        ("pull-request/1", "r0.7.0", "r0.7.0", "false"),
        ("pull-request/1", "main", "", "false"),
        ("pull-request/1", "feature/base", "", "false"),
        ("pull-request/1", "r0.7.0-rc1", "", "false"),
        ("r0.7.0-rc1", "", "", "false"),
        ("r0.7.1", "", "", "false"),
        ("deploy-release/r0.7.0", "", "", "false"),
    ],
)
def test_cache_scope_limits_writers_and_uses_actual_pr_target(
    tmp_path: Path, branch: str, target: str, release: str, writer: str
) -> None:
    output = tmp_path / "output"
    subprocess.run(
        ["bash", ".github/scripts/select-ci-cache-scope.sh"],
        env={
            **os.environ,
            "GITHUB_REF_NAME": branch,
            "GITHUB_REF": f"refs/heads/{branch}",
            "PR_BASE_REF": target,
            "GITHUB_OUTPUT": str(output),
        },
        check=True,
    )
    values = dict(line.split("=", 1) for line in output.read_text().splitlines())
    assert values == {
        "base_branch": target or release or "main",
        "release_branch": release,
        "cache_writer": writer,
    }


@pytest.mark.parametrize("target", ["", "../main", "main\ncache_writer=true"])
def test_copied_pr_requires_valid_target_metadata(tmp_path: Path, target: str) -> None:
    output = tmp_path / "output"
    result = subprocess.run(
        ["bash", ".github/scripts/select-ci-cache-scope.sh"],
        env={
            **os.environ,
            "GITHUB_REF_NAME": "pull-request/1",
            "GITHUB_REF": "refs/heads/pull-request/1",
            "PR_BASE_REF": target,
            "GITHUB_OUTPUT": str(output),
        },
        check=False,
    )
    assert result.returncode != 0
    assert not output.exists()


@pytest.mark.parametrize(
    ("branch", "base_branch", "path", "expected"),
    [
        ("pull-request/1", "r0.7.0", "README.md", ["wheelhouse"]),
        ("pull-request/1", "r0.7.0", "pyproject.toml", ["wheelhouse", "source"]),
        ("pull-request/1", "main", "README.md", ["wheelhouse", "source"]),
        ("pull-request/1", "r0.6.0", "README.md", ["wheelhouse", "source"]),
        ("r0.7.0", "r0.7.0", "README.md", ["wheelhouse"]),
        ("r0.7.0", "r0.7.0", "pyproject.toml", ["wheelhouse", "source"]),
    ],
)
@pytest.mark.runtime_budget(15, reason="Constructs release Git history and runs the real Bash source-mode selector")
def test_cherry_pick_source_selection_ignores_historical_release_divergence(
    tmp_path: Path, branch: str, base_branch: str, path: str, expected: list[str]
) -> None:
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.name", "Test")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "-c", "commit.gpgsign=false", "commit", "--allow-empty", "-qm", "base")
    _git(tmp_path, "update-ref", "refs/remotes/origin/main", "HEAD")
    (tmp_path / "pyproject.toml").write_text("historical release configuration\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "-c", "commit.gpgsign=false", "commit", "-qm", "release diverges")
    before = _git(tmp_path, "rev-parse", "HEAD")
    _git(tmp_path, "update-ref", "refs/remotes/origin/r0.7.0", before)
    (tmp_path / path).write_text("cherry-picked change\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "-c", "commit.gpgsign=false", "commit", "-qm", "cherry-pick")
    output = tmp_path / "output"
    subprocess.run(
        ["bash", str(_MODE_SCRIPT)],
        cwd=tmp_path,
        env={
            **os.environ,
            "GITHUB_EVENT_NAME": "push",
            "GITHUB_REF_NAME": branch,
            "BASE_BRANCH": base_branch,
            "BEFORE_SHA": before,
            "GITHUB_OUTPUT": str(output),
        },
        check=True,
    )
    assert json.loads(output.read_text().split("=", 1)[1]) == expected


@pytest.mark.parametrize(
    ("signer", "release", "reused", "accepted"),
    [
        ("main", "r0.7.0", True, True),
        ("r0.7.0", "r0.7.0", True, True),
        ("r0.6.0", "r0.7.0", True, False),
        ("pull-request/2", "r0.7.0", True, False),
        ("r0.7.0", "", True, False),
        ("r0.7.0", "r0.7.0", False, False),
        ("pull-request/1", "r0.7.0", False, True),
    ],
)
def test_release_provenance_accepts_only_main_or_target_release(
    tmp_path: Path, signer: str, release: str, reused: bool, accepted: bool
) -> None:
    gh = tmp_path / "gh"
    gh.write_text(
        f"#!{sys.executable}\nimport os, sys\n"
        "ref = sys.argv[sys.argv.index('--source-ref') + 1]\n"
        "sys.exit(0 if ref == 'refs/heads/' + os.environ['SIGNER'] else 1)\n"
    )
    gh.chmod(0o755)
    (tmp_path / "manifest.json").write_text('{"wheels":{}}\n')
    output = tmp_path / "output"
    result = subprocess.run(
        ["bash", str(_VERIFY_SCRIPT)],
        env={
            **os.environ,
            "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
            "SIGNER": signer,
            "RELEASE_BRANCH": release,
            "WHEELHOUSE_REUSED": str(reused).lower(),
            "GITHUB_REPOSITORY": "NVIDIA-NeMo/Automodel",
            "GITHUB_REF": "refs/heads/pull-request/1",
            "GITHUB_SHA": "a" * 40,
            "WHEELHOUSE_DIR": str(tmp_path),
            "GITHUB_OUTPUT": str(output),
        },
        check=False,
    )
    assert (result.returncode == 0) == accepted
    assert output.exists() == accepted


@pytest.mark.parametrize("tag", ["main", "r0.7.0"])
def test_tags_do_not_become_canonical_cache_writers(tmp_path: Path, tag: str) -> None:
    output = tmp_path / "output"
    subprocess.run(
        ["bash", ".github/scripts/select-ci-cache-scope.sh"],
        env={
            **os.environ,
            "GITHUB_REF": f"refs/tags/{tag}",
            "GITHUB_REF_NAME": tag,
            "GITHUB_OUTPUT": str(output),
        },
        check=True,
    )
    values = dict(line.split("=", 1) for line in output.read_text().splitlines())
    assert values == {"base_branch": "main", "release_branch": "", "cache_writer": "false"}
