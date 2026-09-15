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
for anyone installing `nemo_automodel[moe]` outside the container -- either as a
`[tool.uv.sources]` entry carrying a commit SHA, or as a `git+...@<branch>` requirement in
the `moe-hybridep` dependency group, in which case the lock files hold the resolved SHA.
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
LOCK_PATHS = (REPO_ROOT / "uv.lock", REPO_ROOT / "docker" / "common" / "uv-pytorch.lock")

_FULL_SHA = r"[0-9a-f]{40}"
_DOCKERFILE_ARG = re.compile(rf"^ARG\s+DEEPEP_COMMIT=({_FULL_SHA})\s*$", re.MULTILINE)
# `deep_ep @ git+<url>@<ref>`, the PEP 508 direct reference used by the dependency group.
_DIRECT_REFERENCE = re.compile(r"^deep[-_]ep\s*@\s*git\+(?P<url>[^@\s]+)@(?P<ref>\S+)$")
# uv records a locked git dependency as `<url>?rev=<ref>#<sha>`.
_LOCKED_GIT_SOURCE = re.compile(rf"^(?P<url>[^?]+)\?rev=(?P<ref>[^#]+)#(?P<sha>{_FULL_SHA})$")
# `git rev-parse --short` never abbreviates below 7 characters.
_MIN_ABBREV_LEN = 7


def _load_toml(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _pyproject_deepep_requirement() -> tuple[str, str]:
    """The `(repository url, git ref)` `uv`/`pip` resolve `deep_ep` from.

    The ref is a commit SHA when declared under `[tool.uv.sources]`, and a branch name when
    declared as a direct reference in the `moe-hybridep` dependency group.
    """
    pyproject = _load_toml(PYPROJECT_PATH)

    source = pyproject["tool"]["uv"]["sources"].get("deep_ep")
    if source is not None:
        return source["git"], source["rev"]

    group = pyproject["dependency-groups"].get("moe-hybridep", [])
    matches = [m for m in (_DIRECT_REFERENCE.match(entry.strip()) for entry in group) if m]
    assert len(matches) == 1, (
        f"{PYPROJECT_PATH.name} must pin `deep_ep` exactly once, either under [tool.uv.sources] or as a "
        f"`deep_ep @ git+<url>@<ref>` entry in the `moe-hybridep` dependency group; found {len(matches)} "
        "of the latter and none of the former"
    )
    return matches[0]["url"], matches[0]["ref"]


def _lockfile_deepep_source(lock_path: Path) -> tuple[str, str, str]:
    """The `(repository url, git ref, commit sha)` a lock file resolved `deep_ep` to."""
    packages = [pkg for pkg in _load_toml(lock_path)["package"] if pkg["name"] == "deep-ep"]
    assert len(packages) == 1, f"expected exactly one `deep-ep` package in {lock_path.name}, found {len(packages)}"

    git_source = packages[0]["source"].get("git")
    assert git_source is not None, f"`deep-ep` in {lock_path.name} is not resolved from git: {packages[0]['source']}"

    match = _LOCKED_GIT_SOURCE.match(git_source)
    assert match, f"could not parse `deep-ep` git source in {lock_path.name}: {git_source!r}"
    return match["url"], match["ref"], match["sha"]


def _pinned_deepep_rev() -> str:
    """The DeepEP commit the uv/pip install path resolves to, cross-checked across both lock files."""
    declared_url, declared_ref = _pyproject_deepep_requirement()
    if re.fullmatch(_FULL_SHA, declared_ref):
        expected = declared_ref
    else:
        expected = None

    revs = {}
    for lock_path in LOCK_PATHS:
        url, ref, sha = _lockfile_deepep_source(lock_path)
        assert (url, ref) == (declared_url, declared_ref), (
            f"{lock_path.name} resolves `deep_ep` from a different source than {PYPROJECT_PATH.name}:\n"
            f"  {PYPROJECT_PATH.name}: {declared_url}@{declared_ref}\n"
            f"  {lock_path.name}: {url}@{ref}\n"
            "Refresh both lock files (see CONTRIBUTING.md) after changing the pin."
        )
        revs[lock_path.name] = sha

    assert len(set(revs.values())) == 1, f"lock files disagree on the pinned DeepEP commit: {revs}"
    resolved = next(iter(revs.values()))

    assert expected is None or expected == resolved, (
        f"[tool.uv.sources] deep_ep.rev is {expected} but the lock files resolved {resolved}"
    )
    return resolved


def _pyproject_deepep_declared_version() -> str:
    """The hand-written `[[tool.uv.dependency-metadata]]` version used to skip a build during resolution."""
    entries = _load_toml(PYPROJECT_PATH)["tool"]["uv"]["dependency-metadata"]
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
    pinned_rev = _pinned_deepep_rev()
    dockerfile_commit = _dockerfile_deepep_commit()

    assert pinned_rev == dockerfile_commit, (
        "DeepEP pin drift: the container and the uv/pip install path would ship different DeepEP builds.\n"
        f"  pyproject.toml + lock files      deep_ep = {pinned_rev}\n"
        f"  docker/Dockerfile        ARG DEEPEP_COMMIT = {dockerfile_commit}\n"
        "Bump whichever is stale, then refresh both lock files (see CONTRIBUTING.md) so "
        "`uv.lock` and `docker/common/uv-pytorch.lock` agree."
    )


def test_deepep_dependency_metadata_version_matches_pinned_rev():
    """The declared `1.2.1+<short-sha>` local version must name the commit that is actually pinned.

    `uv` trusts this string instead of building DeepEP during resolution; if it names a different
    commit than the pinned rev, the lock file records a version that the built wheel never produces.
    """
    rev = _pinned_deepep_rev()
    declared = _pyproject_deepep_declared_version()

    assert "+" in declared, (
        f"[[tool.uv.dependency-metadata]] deep_ep.version must carry a `+<short-sha>` local segment "
        f"identifying the pinned commit, got {declared!r}"
    )
    short_sha = declared.rsplit("+", 1)[1]

    assert len(short_sha) >= _MIN_ABBREV_LEN and rev.startswith(short_sha), (
        "DeepEP declared version does not match the pinned commit.\n"
        f"  pyproject.toml + lock files      deep_ep = {rev}\n"
        f"  [[tool.uv.dependency-metadata]]  version = {declared}\n"
        f"`deep_ep` builds as `1.2.1+$(git rev-parse --short HEAD)`, so the local segment must be "
        f"an abbreviation of the pinned rev (expected `+{rev[:_MIN_ABBREV_LEN]}`, got `+{short_sha}`)."
    )
