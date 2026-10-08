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

"""CPU regression tests for the container's fresh-process Ray HTTP guard."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GUARD = REPO_ROOT / "docker/common/verify_ray_aiohttp.py"


def _run_guard(tmp_path: Path, *, installed: str = "3.14.3", bundled: str | None = None) -> subprocess.CompletedProcess:
    # Model Ray's real agent import order without starting a node in fast unit tests.
    ray = tmp_path / "ray"
    ray.mkdir()
    (ray / "__init__.py").write_text("")
    agent = ray / "_private/runtime_env/agent"
    agent.mkdir(parents=True)
    (agent / "main.py").write_text(
        "import sys\nfrom pathlib import Path\n"
        "sys.path.insert(0, str(Path(__file__).parent / 'thirdparty_files'))\n"
        "import aiohttp\nfrom aiohttp import web\n"
    )
    metadata = tmp_path / f"aiohttp-{installed}.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(f"Metadata-Version: 2.1\nName: aiohttp\nVersion: {installed}\n")
    locations = [(tmp_path, installed)]
    if bundled is not None:
        locations.append((agent / "thirdparty_files", bundled))
    for root, version in locations:
        package = root / "aiohttp"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text(f"__version__ = {version!r}\n")
        (package / "web.py").write_text("class Application: pass\n")
    return subprocess.run(
        [sys.executable, str(GUARD)],
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=10,
    )


def test_agent_uses_patched_project_aiohttp(tmp_path: Path) -> None:
    result = _run_guard(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "Ray uses aiohttp 3.14.3" in result.stderr


def test_rejects_unpatched_project_aiohttp(tmp_path: Path) -> None:
    result = _run_guard(tmp_path, installed="3.14.1")
    assert result.returncode != 0
    assert "aiohttp>=3.14.3 is required" in result.stderr


@pytest.mark.parametrize("bundled", ["3.14.1", "3.14.3"])
def test_rejects_shadowed_aiohttp_even_with_matching_version(tmp_path: Path, bundled: str) -> None:
    result = _run_guard(tmp_path, bundled=bundled)
    assert result.returncode != 0
    assert "Ray imports a shadowed aiohttp" in result.stderr


def test_rejects_preimported_aiohttp() -> None:
    result = subprocess.run(
        [sys.executable, "-c", f"import aiohttp, runpy; runpy.run_path({str(GUARD)!r}, run_name='__main__')"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0
    assert "fresh process" in result.stderr


def test_docker_removes_bundle_after_sync_and_verifies_before_native_checks() -> None:
    dockerfile = (REPO_ROOT / "docker/Dockerfile").read_text()
    sync = dockerfile.index("uv sync --extra")
    removal = dockerfile.index("/site-packages/ray/_private/runtime_env/agent/thirdparty_files")
    guard = dockerfile.index("RUN python docker/common/verify_ray_aiohttp.py")
    native = dockerfile.index("MAX_JOBS=2 python docker/common/verify_native_runtime.py")
    assert sync < removal < guard < native
