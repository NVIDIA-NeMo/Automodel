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

"""Structural checks for the hierarchical ``AGENTS.md`` instruction files.

Agent instructions are only useful while they are true. These tests keep the
hierarchy self-consistent and keep every path an instruction file names
resolvable, so a rename that invalidates guidance fails a PR instead of quietly
misleading the next agent that reads it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT_AGENTS = REPO_ROOT / "AGENTS.md"

IMPORT_LINE = "@AGENTS.md"

ROOT_MAX_LINES = 200
NESTED_MAX_LINES = 80

# Directories that never hold source we document, so we do not walk into them.
SKIP_DIRS = {".git", ".venv", "node_modules", "__pycache__", ".pytest_cache", "build", "dist"}

# Backtick spans that name a path-like token. Requires a separator or a known
# suffix so prose such as `build(...)` or `MODEL_ARCH_MAPPING` is not treated as
# a filesystem path.
PATH_SPAN = re.compile(r"`([A-Za-z0-9_.][A-Za-z0-9_./-]*)`")
PATH_SUFFIXES = (".py", ".md", ".mdx", ".yml", ".yaml", ".toml", ".json", ".sh", ".mjs", ".lock")

# Tokens that look path-like but are commands, packages, or prose.
PATH_EXEMPT = {
    "AGENTS.md",
    "CLAUDE.md",
    "SKILL.md",
    "docs.yml",
    "fern.config.json",
    "config.json",
    "__init__.py",
    "model.py",
    "state_dict_adapter.py",
    "config.py",
    "layers.py",
    "rope_utils.py",
}


def _agents_files() -> list[Path]:
    found = []
    for path in REPO_ROOT.rglob("AGENTS.md"):
        if SKIP_DIRS & set(path.relative_to(REPO_ROOT).parts):
            continue
        found.append(path)
    return sorted(found)


def _rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


AGENTS_FILES = _agents_files()
NESTED_FILES = [p for p in AGENTS_FILES if p != ROOT_AGENTS]


def test_repo_has_a_root_and_nested_agents_files():
    assert ROOT_AGENTS.is_file()
    assert NESTED_FILES, "hierarchy collapsed to a single root AGENTS.md"


@pytest.mark.parametrize("agents_path", AGENTS_FILES, ids=_rel)
def test_every_agents_file_has_a_claude_import(agents_path: Path):
    """Claude Code reads ``CLAUDE.md`` and not ``AGENTS.md``.

    Without the sibling file the directory's rules are invisible to Claude Code
    while remaining visible to Codex, which is worse than having no file at all:
    the two agents would work from different rules.

    The sibling is a regular file containing the single import ``@AGENTS.md``,
    not a symlink. A symlink needs Administrator or Developer Mode to create on
    Windows and is materialised as a text file by checkouts and archive
    exporters that do not preserve symlinks, which silently turns the rules into
    the literal string "AGENTS.md". The import is a plain file everywhere and
    resolves relative to the file that contains it, so it always picks up the
    sibling rather than the repository root.
    """
    claude_path = agents_path.parent / "CLAUDE.md"

    assert not claude_path.is_symlink(), (
        f"{_rel(claude_path)} is a symlink; use a regular file containing `{IMPORT_LINE}` instead"
    )
    assert claude_path.is_file(), f"{_rel(claude_path)} is missing"
    assert claude_path.read_text().strip() == IMPORT_LINE, f"{_rel(claude_path)} must contain exactly `{IMPORT_LINE}`"


@pytest.mark.parametrize("agents_path", AGENTS_FILES, ids=_rel)
def test_agents_files_stay_within_line_budget(agents_path: Path):
    """Long instruction files measurably reduce adherence."""
    limit = ROOT_MAX_LINES if agents_path == ROOT_AGENTS else NESTED_MAX_LINES
    line_count = len(agents_path.read_text().splitlines())

    assert line_count <= limit, f"{_rel(agents_path)} has {line_count} lines (limit {limit})"


def test_root_routing_table_matches_the_files_on_disk():
    """The root file is the index; a stale index sends agents to nothing."""
    listed = {
        match
        for match in PATH_SPAN.findall(ROOT_AGENTS.read_text())
        if match.endswith("AGENTS.md") and match != "AGENTS.md"
    }
    on_disk = {_rel(p) for p in NESTED_FILES}

    assert listed == on_disk, (
        f"routing table out of sync; listed-but-missing={sorted(listed - on_disk)}, "
        f"on-disk-but-unlisted={sorted(on_disk - listed)}"
    )


@pytest.mark.parametrize("agents_path", AGENTS_FILES, ids=_rel)
def test_paths_named_in_instructions_exist(agents_path: Path):
    """Every backticked path must resolve, relative to the file or the repo root.

    This is the check that catches guidance drift: a directory rename that
    leaves an instruction file pointing at a path that no longer exists.
    """
    missing = []
    for token in PATH_SPAN.findall(agents_path.read_text()):
        if token in PATH_EXEMPT:
            continue
        # A token counts as a path only when it ends in "/" (an explicit
        # directory) or its last segment carries a known file suffix. That
        # keeps repo slugs ("NVIDIA-NeMo/FW-CI-templates") and branch-name
        # examples ("jdoe/fix/rope-scaling") out of the check. Write
        # directories with a trailing slash so they are covered.
        if not (token.endswith("/") or token.rsplit("/", 1)[-1].endswith(PATH_SUFFIXES)):
            continue
        if (agents_path.parent / token).exists() or (REPO_ROOT / token).exists():
            continue
        missing.append(token)

    assert not missing, f"{_rel(agents_path)} names paths that do not exist: {sorted(set(missing))}"


@pytest.mark.parametrize("agents_path", NESTED_FILES, ids=_rel)
def test_nested_files_declare_that_they_are_additive(agents_path: Path):
    """Claude Code and Codex both concatenate root-to-leaf; the agents.md spec
    says the nearest file wins. Nested files must read correctly under either
    interpretation, so each one states that it extends the root rather than
    replacing it.
    """
    head = "\n".join(agents_path.read_text().splitlines()[:6]).casefold()

    assert "adds to the repository root" in head, (
        f"{_rel(agents_path)} must open by stating that it adds to the root AGENTS.md"
    )
