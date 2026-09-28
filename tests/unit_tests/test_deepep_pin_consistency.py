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

"""DeepEP is pinned twice: once for the container build and once for uv/pip installs.

`docker/Dockerfile` builds DeepEP from source at `ARG DEEPEP_COMMIT`, which is what every
CI job and every published image actually runs. `pyproject.toml` pins the same dependency
under `[tool.uv.sources]` for anyone installing `nemo_automodel[moe]` outside the container.
Nothing links the two, so bumping one and forgetting the other silently ships a different
DeepEP to users than the one the tests validated. These tests fail on that drift.
"""

import re
from pathlib import Path

try:
    import tomllib  # ty: ignore[unresolved-import]
except ModuleNotFoundError:
    import tomli as tomllib  # ty: ignore[unresolved-import]

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"
DOCKERFILE_PATH = REPO_ROOT / "docker" / "Dockerfile"

_FULL_SHA = r"[0-9a-f]{40}"
_DOCKERFILE_ARG = re.compile(rf"^ARG\s+DEEPEP_COMMIT=({_FULL_SHA})\s*$", re.MULTILINE)
# `git rev-parse --short` never abbreviates below 7 characters.
_MIN_ABBREV_LEN = 7


def _load_pyproject() -> dict:
    with PYPROJECT_PATH.open("rb") as handle:
        return tomllib.load(handle)


def _pyproject_deepep_rev() -> str:
    """The commit `uv`/`pip` resolve `deep_ep` from, per `[tool.uv.sources]`."""
    sources = _load_pyproject()["tool"]["uv"]["sources"]
    assert "deep_ep" in sources, f"[tool.uv.sources] in {PYPROJECT_PATH.name} no longer declares `deep_ep`"

    rev = sources["deep_ep"]["rev"]
    assert re.fullmatch(_FULL_SHA, rev), (
        f"[tool.uv.sources] deep_ep.rev must be a full 40-character commit SHA so it can be compared "
        f"against DEEPEP_COMMIT in docker/Dockerfile, got {rev!r}"
    )
    return rev


def _pyproject_deepep_declared_version() -> str:
    """The hand-written `[[tool.uv.dependency-metadata]]` version used to skip a build during resolution."""
    entries = _load_pyproject()["tool"]["uv"]["dependency-metadata"]
    matches = [entry for entry in entries if entry["name"] == "deep_ep"]
    assert len(matches) == 1, (
        f"expected exactly one [[tool.uv.dependency-metadata]] entry named `deep_ep` in "
        f"{PYPROJECT_PATH.name}, found {len(matches)}"
    )
    return matches[0]["version"]


def _dockerfile_deepep_commit() -> str:
    """The commit the container builds DeepEP from, per `ARG DEEPEP_COMMIT`."""
    matches = _DOCKERFILE_ARG.findall(DOCKERFILE_PATH.read_text())
    assert len(matches) == 1, (
        f"expected exactly one `ARG DEEPEP_COMMIT=<40-char sha>` line in docker/Dockerfile, "
        f"found {len(matches)}; this test cannot tell which pin is authoritative otherwise"
    )
    return matches[0]


def test_pyproject_deepep_rev_matches_dockerfile_commit():
    """The uv/pip pin and the container build must resolve to the same DeepEP commit."""
    pyproject_rev = _pyproject_deepep_rev()
    dockerfile_commit = _dockerfile_deepep_commit()

    assert pyproject_rev == dockerfile_commit, (
        "DeepEP pin drift: the container and the uv/pip install path would ship different DeepEP builds.\n"
        f"  pyproject.toml  [tool.uv.sources] deep_ep.rev = {pyproject_rev}\n"
        f"  docker/Dockerfile           ARG DEEPEP_COMMIT = {dockerfile_commit}\n"
        "Bump whichever is stale, then refresh both lock files (see CONTRIBUTING.md) so "
        "`uv.lock` and `docker/common/uv-pytorch.lock` agree."
    )


def test_deepep_dependency_metadata_version_matches_pinned_rev():
    """The declared `1.2.1+<short-sha>` local version must name the commit that is actually pinned.

    `uv` trusts this string instead of building DeepEP during resolution; if it names a different
    commit than `deep_ep.rev`, the lock file records a version that the built wheel never produces.
    """
    rev = _pyproject_deepep_rev()
    declared = _pyproject_deepep_declared_version()

    assert "+" in declared, (
        f"[[tool.uv.dependency-metadata]] deep_ep.version must carry a `+<short-sha>` local segment "
        f"identifying the pinned commit, got {declared!r}"
    )
    short_sha = declared.rsplit("+", 1)[1]

    assert len(short_sha) >= _MIN_ABBREV_LEN and rev.startswith(short_sha), (
        "DeepEP declared version does not match the pinned commit.\n"
        f"  [tool.uv.sources]            deep_ep.rev = {rev}\n"
        f"  [[tool.uv.dependency-metadata]]  version = {declared}\n"
        f"`deep_ep` builds as `1.2.1+$(git rev-parse --short HEAD)`, so the local segment must be "
        f"an abbreviation of the pinned rev (expected `+{rev[:_MIN_ABBREV_LEN]}`, got `+{short_sha}`)."
    )
