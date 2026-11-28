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

"""Lint open pull requests as they would land on main.

CI lints a pull request against main as it was when that CI run started. A pull
request that stays open while main changes can therefore merge code that no
longer lints, such as a new call to a helper that main has since renamed. This
tool merges each pull request into a main commit in a scratch worktree, runs ruff
on the result, and reports the ruff findings and unformatted files that the main
commit itself does not have, so a broken main does not fail every pull request.

Pull request content is untrusted data. The tool never imports or executes it,
checks symbolic links out as plain files, skips Git LFS downloads, and disables
Git hooks. Run it without credentials in its environment.

Run it from the root of a clone whose ``origin`` is the GitHub repository.
``--prs`` names a JSON list of ``{"number", "head_sha"}`` objects, each optionally
carrying the ``state`` and ``description`` of the status already posted for that
head. The tool writes ``results.json``, the statuses whose state or description
changes, and ``summary.md``, which lists every pull request that does not lint
cleanly, to ``--output-dir``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from html import escape
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

_GIT_OPTIONS = (
    "-c",
    "core.symlinks=false",
    "-c",
    "core.hooksPath=/dev/null",
    "-c",
    "user.name=lint-merged-with-main",
    "-c",
    "user.email=lint-merged-with-main@invalid",
)
_FETCH_TIMEOUT_S = 600
_GIT_TIMEOUT_S = 120
_RUFF_TIMEOUT_S = 300
_REFORMAT_PREFIX = "Would reformat: "
# The code ruff 0.12 gives a file that does not parse. A finding without a code gets it too.
_INVALID_SYNTAX = "invalid-syntax"

State = Literal["success", "failure", "error"]


@dataclass(frozen=True)
class PullRequest:
    """An open pull request and the status already posted for its head."""

    number: int
    head_sha: str
    state: str | None = None
    description: str | None = None


@dataclass(frozen=True)
class Status:
    """A commit status to post on a pull request head."""

    number: int
    head_sha: str
    state: State
    description: str


@dataclass(frozen=True)
class _Finding:
    path: str
    row: int
    column: int
    code: str
    message: str

    @property
    def key(self) -> tuple[str, str, str]:
        """Identify the finding without its position, which merges shift."""
        return self.path, self.code, self.message

    def __str__(self) -> str:
        return f"{self.path}:{self.row}:{self.column}: {self.code} {self.message}"


@dataclass(frozen=True)
class _Lint:
    findings: list[_Finding]
    unformatted: frozenset[str]


class _LintError(Exception):
    """Ruff could not lint a tree."""


def _git(
    cwd: Path, *args: str, check: bool = True, timeout: int = _GIT_TIMEOUT_S, stdin: str | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *_GIT_OPTIONS, *args],
        cwd=cwd,
        env={**os.environ, "GIT_LFS_SKIP_SMUDGE": "1", "GIT_TERMINAL_PROMPT": "0"},
        check=check,
        capture_output=True,
        text=True,
        timeout=timeout,
        input=stdin,
    )


def _one_line(text: str) -> str:
    """Escape line breaks so that a file name cannot start a workflow command line."""
    return text.replace("\r", "\\r").replace("\n", "\\n")


def _ruff(ruff: Path, tree: Path, cache: Path) -> _Lint:
    check = subprocess.run(
        [str(ruff), "check", "--output-format", "json", "--exit-zero", "--cache-dir", str(cache), "."],
        cwd=tree,
        capture_output=True,
        text=True,
        timeout=_RUFF_TIMEOUT_S,
    )
    if check.returncode != 0:
        raise _LintError(f"ruff check exited with {check.returncode}")
    try:
        findings = [
            _Finding(
                path=Path(item["filename"]).relative_to(tree).as_posix(),
                row=item["location"]["row"],
                column=item["location"]["column"],
                code=item["code"] or _INVALID_SYNTAX,
                message=item["message"],
            )
            for item in json.loads(check.stdout)
        ]
    except (KeyError, TypeError, ValueError) as exc:
        raise _LintError("ruff check printed unexpected output") from exc
    # Exit code 2 also covers files that do not parse, which `ruff check` already reports.
    formatted = subprocess.run(
        [str(ruff), "format", "--check", "--cache-dir", str(cache), "."],
        cwd=tree,
        capture_output=True,
        text=True,
        timeout=_RUFF_TIMEOUT_S,
    )
    unformatted = frozenset(
        line.removeprefix(_REFORMAT_PREFIX)
        for line in formatted.stdout.splitlines()
        if line.startswith(_REFORMAT_PREFIX)
    )
    if formatted.returncode == 2 and not any(finding.code == _INVALID_SYNTAX for finding in findings):
        raise _LintError("ruff format exited with 2")
    return _Lint(findings, unformatted)


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _lint_merge(
    tree: Path, base: str, pull_request: PullRequest, ruff: Path, cache: Path, baseline: _Lint
) -> tuple[Status, list[str]]:
    """Return the status of one pull request merged into base, and the lines that explain it."""
    _git(tree, "reset", "--quiet", "--hard", base)
    merge = _git(tree, "merge", "--quiet", "--no-ff", "--no-commit", pull_request.head_sha, check=False)
    if merge.returncode != 0:
        conflicts = _git(tree, "diff", "--name-only", "--diff-filter=U").stdout.splitlines()
        if not conflicts:
            raise _LintError(f"git merge exited with {merge.returncode}")
        return (
            Status(
                pull_request.number,
                pull_request.head_sha,
                "failure",
                "Conflicts with main, so the merge was not linted",
            ),
            [f"conflict: {path}" for path in conflicts],
        )
    merged = _ruff(ruff, tree, cache)
    new_counts = Counter(finding.key for finding in merged.findings) - Counter(
        finding.key for finding in baseline.findings
    )
    new_findings = [finding for finding in merged.findings if finding.key in new_counts]
    new_unformatted = sorted(merged.unformatted - baseline.unformatted)
    explanation = [str(finding) for finding in new_findings] + [
        f"{path}: needs ruff format" for path in new_unformatted
    ]
    if not explanation:
        return Status(
            pull_request.number, pull_request.head_sha, "success", "Merged into main: no new ruff findings"
        ), []
    problems = []
    if new_counts:
        problems.append(_plural(new_counts.total(), "new ruff finding"))
    if new_unformatted:
        problems.append(_plural(len(new_unformatted), "file") + " to reformat")
    description = "Merged into main: " + ", ".join(problems)
    return Status(pull_request.number, pull_request.head_sha, "failure", description), explanation


def lint_pull_requests(base: str, pull_requests: list[PullRequest], ruff: Path) -> tuple[list[Status], str]:
    """Merge each pull request into base, lint the merge, and describe the outcome.

    Args:
        base: Commit to merge each pull request into.
        pull_requests: Pull requests to lint. A pull request whose head moved after it
            was listed is skipped, since a newer run lints its new head.
        ruff: Ruff executable.

    Returns:
        The statuses whose state or description differs from the status already
        posted, and a Markdown summary of the pull requests that do not lint cleanly.
    """
    repo = Path.cwd()
    base_sha = _git(repo, "rev-parse", "--verify", f"{base}^{{commit}}").stdout.strip()
    if not pull_requests:
        return [], "## Lint merged with main\n\nNo open pull requests to lint.\n"
    refs = {pull_request.number: f"refs/lint-merged-with-main/{pull_request.number}" for pull_request in pull_requests}
    statuses: list[Status] = []
    rows: list[str] = []
    details: list[str] = []
    linted = 0
    try:
        _git(
            repo,
            "fetch",
            "--quiet",
            "--no-tags",
            "--force",
            "origin",
            *(f"refs/pull/{number}/head:{ref}" for number, ref in refs.items()),
            timeout=_FETCH_TIMEOUT_S,
        )
        with tempfile.TemporaryDirectory() as scratch:
            tree = Path(scratch, "tree").resolve()
            cache = Path(scratch, "ruff-cache")
            _git(repo, "worktree", "add", "--quiet", "--detach", str(tree), base_sha)
            try:
                baseline = _ruff(ruff, tree, cache)
                for pull_request in pull_requests:
                    head = _git(repo, "rev-parse", refs[pull_request.number]).stdout.strip()
                    if head != pull_request.head_sha:
                        logger.info("#%d: head moved to %s; skipped", pull_request.number, head)
                        continue
                    linted += 1
                    try:
                        status, explanation = _lint_merge(tree, base_sha, pull_request, ruff, cache, baseline)
                    except (_LintError, subprocess.SubprocessError) as exc:
                        reason = str(exc) if isinstance(exc, _LintError) else type(exc).__name__
                        status = Status(
                            pull_request.number, pull_request.head_sha, "error", f"Could not lint the merge: {reason}"
                        )
                        explanation = []
                    logger.info("#%d: %s", pull_request.number, status.description)
                    for line in explanation:
                        logger.info("#%d:   %s", pull_request.number, _one_line(line))
                    if (status.state, status.description) != (pull_request.state, pull_request.description):
                        statuses.append(status)
                    if status.state != "success":
                        rows.append(f"| #{pull_request.number} | {status.state} | {escape(status.description)} |")
                    if explanation:
                        lines = escape("\n".join(explanation))
                        details.append(
                            f"<details><summary>#{pull_request.number}</summary>\n\n<pre>{lines}</pre>\n</details>"
                        )
            finally:
                _git(repo, "worktree", "remove", "--force", str(tree), check=False)
    finally:
        _git(repo, "update-ref", "--stdin", check=False, stdin="".join(f"delete {ref}\n" for ref in refs.values()))
    summary = [
        "## Lint merged with main",
        "",
        f"Merged {_plural(linted, 'pull request')} into main at `{base_sha[:12]}` and ran ruff.",
        "",
    ]
    if rows:
        summary += ["| Pull request | State | Description |", "| --- | --- | --- |", *rows, "", *details]
    else:
        summary.append("Every merge lints cleanly.")
    return statuses, "\n".join(summary) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Lint pull requests merged into main and write the statuses to post."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True, help="Main commit to merge each pull request into.")
    parser.add_argument("--prs", type=Path, required=True, help="JSON list of the pull requests to lint.")
    parser.add_argument("--ruff", type=Path, required=True, help="Ruff executable.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for results.json and summary.md.")
    args = parser.parse_args(argv)
    pull_requests = [PullRequest(**item) for item in json.loads(args.prs.read_text())]
    try:
        statuses, summary = lint_pull_requests(args.base, pull_requests, args.ruff)
    except (OSError, ValueError, _LintError, subprocess.SubprocessError) as exc:
        logger.error("Cannot lint pull requests merged into main: %s", exc)
        return 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(json.dumps([asdict(status) for status in statuses], indent=2) + "\n")
    (args.output_dir / "summary.md").write_text(summary)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
