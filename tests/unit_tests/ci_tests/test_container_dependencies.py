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
import shlex
import subprocess
from pathlib import Path

import pytest

_SCRIPT = Path(".github/scripts/prepare-container-dependencies.sh").resolve()
_INPUTS = (
    "docker/Dockerfile",
    "docker/common/deepep.patch",
    "docker/common/uv-pytorch.toml",
    "docker/common/uv-pytorch.lock",
    "docker/common/update_pyproject_pytorch.sh",
    "scripts/setup_uccl_ep.sh",
    "pyproject.toml",
    "uv.lock",
    ".dockerignore",
    "nemo_automodel/__init__.py",
    "nemo_automodel/package_info.py",
    ".github/container-build-args.env",
    ".github/scripts/prepare-container-dependencies.sh",
)
_BASE_DIGEST = "sha256:" + "a" * 64
_DEPS_DIGEST = "sha256:" + "b" * 64


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    for name in _INPUTS:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("dependency input\n")
    (tmp_path / "docker/Dockerfile").write_text("ARG PYTORCH_IMAGE=example/pytorch:base\n")
    (tmp_path / ".github/container-build-args.env").write_text("""BASE_IMAGE=pytorch
UV_SYNC_ARGS=--locked
""")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    docker = tmp_path / "bin/docker"
    docker.parent.mkdir()
    docker.write_text("""#!/bin/bash
set -eu
echo "$4" >> "$LOOKUPS"
if [[ $4 == example/pytorch:base ]]; then
  echo "$BASE_DIGEST"
elif [[ $REGISTRY_ERROR != none ]]; then
  echo "$REGISTRY_ERROR" >&2
  exit 1
elif [[ $AVAILABLE == own && $4 == *:deps-branch-* ]] || [[ $AVAILABLE == main && $4 == *:deps-main-* ]]; then
  echo "$DEPS_DIGEST"
else
  echo 'manifest unknown' >&2
  exit 1
fi
""")
    docker.chmod(0o755)
    return tmp_path


def _prepare(checkout: Path, **overrides: str) -> tuple[subprocess.CompletedProcess[str], dict[str, str]]:
    output = checkout / "output"
    output.unlink(missing_ok=True)
    result = subprocess.run(
        ["bash", str(_SCRIPT)],
        cwd=checkout,
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "PATH": f"{checkout / 'bin'}:{os.environ['PATH']}",
            "PLATFORM": "linux/arm64",
            "REGISTRY": "registry.example/ci",
            "CACHE_KEY": "4269",
            "GITHUB_OUTPUT": str(output),
            "BASE_DIGEST": _BASE_DIGEST,
            "DEPS_DIGEST": _DEPS_DIGEST,
            "LOOKUPS": str(checkout / "lookups"),
            "AVAILABLE": "none",
            "REGISTRY_ERROR": "none",
            **overrides,
        },
        check=False,
    )
    values = {}
    if output.exists():
        scalars, build_args = output.read_text().split("build-args<<AUTOMODEL_BUILD_ARGS\n")
        values = dict(line.split("=", 1) for line in scalars.splitlines())
        values["build-args"] = build_args.removesuffix("AUTOMODEL_BUILD_ARGS\n")
    return result, values


@pytest.mark.parametrize("path", _INPUTS)
@pytest.mark.runtime_budget(
    15,
    reason="Runs the real Git/Bash dependency lookup twice; GPU CI measured 5.71s under suite load",
)
def test_dependency_input_changes_invalidate_image(checkout: Path, path: str) -> None:
    result, before = _prepare(checkout)
    assert result.returncode == 0, result.stderr
    with (checkout / path).open("a") as stream:
        stream.write("# dependency changed\n")
    result, after = _prepare(checkout)
    assert result.returncode == 0, result.stderr
    assert after["image"] != before["image"]


@pytest.mark.runtime_budget(
    15,
    reason="Runs two real Git/Bash dependency lookups and stages a source change; comparable CI cases exceed 5s",
)
def test_source_only_changes_reuse_key(checkout: Path) -> None:
    result, before = _prepare(checkout)
    assert result.returncode == 0, result.stderr
    path = checkout / "nemo_automodel/model.py"
    path.write_text("changed application code\n")
    subprocess.run(["git", "-C", str(checkout), "add", "."], check=True)
    result, after = _prepare(checkout)
    assert result.returncode == 0, result.stderr
    assert after["image"] == before["image"]


@pytest.mark.parametrize("override", [{"PLATFORM": "linux/amd64"}, {"BASE_DIGEST": "sha256:" + "c" * 64}])
@pytest.mark.runtime_budget(
    15,
    reason="Runs the real Git/Bash dependency lookup twice; CPU CI measured 5.03s under suite load",
)
def test_abi_changes_invalidate_image(checkout: Path, override: dict[str, str]) -> None:
    _, before = _prepare(checkout)
    result, after = _prepare(checkout, **override)
    assert result.returncode == 0, result.stderr
    assert after["image"] != before["image"]
    assert f"PYTORCH_IMAGE=example/pytorch:base@{override.get('BASE_DIGEST', _BASE_DIGEST)}" in after["build-args"]


@pytest.mark.parametrize("available", ["none", "own", "main"])
def test_lookup_reuses_own_or_main_and_pins_digest(checkout: Path, available: str) -> None:
    result, output = _prepare(checkout, AVAILABLE=available)
    assert result.returncode == 0, result.stderr
    assert output["cache-hit"] == str(available != "none").lower()
    assert output["digest"] == (_DEPS_DIGEST if available != "none" else "")
    assert (":deps-main-" in output["image"]) == (available == "main")
    assert all(
        "deps-main-" in line or "deps-branch-" in line for line in (checkout / "lookups").read_text().splitlines()[1:]
    )


def test_main_never_consumes_branch_images(checkout: Path) -> None:
    result, output = _prepare(checkout, CACHE_KEY="main", AVAILABLE="own")
    assert result.returncode == 0, result.stderr
    assert output["cache-hit"] == "false"
    assert ":deps-main-" in output["image"]
    assert "deps-branch-" not in (checkout / "lookups").read_text()


@pytest.mark.runtime_budget(
    15,
    reason="Runs the real Git/Bash dependency lookup three times; CI measured up to 8.43s under suite load",
)
def test_release_builds_keep_a_stable_separate_image_scope(checkout: Path) -> None:
    _, pr = _prepare(checkout)
    result, release = _prepare(checkout, CACHE_KEY="deploy-release-r0.7.0")
    assert result.returncode == 0, result.stderr
    _, repeated = _prepare(checkout, CACHE_KEY="deploy-release-r0.7.0")
    assert release == repeated
    assert release["image"] != pr["image"]
    assert release["image"].rsplit("-", 1)[1] == pr["image"].rsplit("-", 1)[1]


@pytest.mark.parametrize("error", ["unauthorized", "connection refused", "TLS handshake timeout"])
def test_registry_errors_do_not_trigger_recompilation(checkout: Path, error: str) -> None:
    result, output = _prepare(checkout, REGISTRY_ERROR=error)
    assert result.returncode != 0
    assert error in result.stderr
    assert not output


@pytest.mark.parametrize("variable", ["BASE_DIGEST", "DEPS_DIGEST"])
def test_invalid_digest_cannot_be_used(checkout: Path, variable: str) -> None:
    result, output = _prepare(checkout, AVAILABLE="own", **{variable: "latest"})
    assert result.returncode != 0
    assert not output


def test_every_dependency_copy_input_is_keyed() -> None:
    # Guard the maintenance boundary: a newly added COPY must not silently reuse
    # an image built from older inputs. COPY --from reads already-keyed stages.
    dependencies = Path("docker/Dockerfile").read_text().split("FROM automodel_dependencies AS automodel_final")[0]
    for line in dependencies.splitlines():
        if line.startswith("COPY ") and "--from=" not in line:
            sources = [token for token in shlex.split(line)[1:-1] if not token.startswith("--")]
            assert set(sources) <= set(_INPUTS)
