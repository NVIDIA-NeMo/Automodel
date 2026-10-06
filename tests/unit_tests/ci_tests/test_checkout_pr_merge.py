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

"""Tests for .github/scripts/checkout-pr-merge.sh.

Each test builds a real origin repository shaped like the 2026-10-04 incident:
a PR branched before ``main`` added a rule (``contract.txt``), so the PR head
alone cannot see the rule but GitHub's test merge commit can. ``gh`` is replaced
by a stub that replays canned ``pulls/<N>`` lookups.
"""

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / ".github/scripts/checkout-pr-merge.sh"
PR = 7

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="requires git")


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=ci", "-c", "user.email=ci@example.com", "-c", "init.defaultBranch=main", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _commit(work: Path, path: str, text: str) -> str:
    (work / path).parent.mkdir(parents=True, exist_ok=True)
    (work / path).write_text(text)
    _git(work, "add", path)
    _git(work, "commit", "-q", "-m", f"add {path}")
    return _git(work, "rev-parse", "HEAD")


def _merge(work: Path, base: str, head: str) -> str:
    """Build a test merge commit the way GitHub does: parents are [base, head]."""
    _git(work, "checkout", "-q", "--detach", base)
    _git(work, "merge", "-q", "--no-ff", "--no-edit", head)
    return _git(work, "rev-parse", "HEAD")


@dataclass
class Repo:
    origin: Path
    work: Path
    base: str  # main after the rule landed
    head: str  # vetted PR head, branched before the rule
    merge: str  # test merge of head into base


@pytest.fixture
def repo(tmp_path: Path) -> Repo:
    origin = tmp_path / "origin.git"
    work = tmp_path / "work"
    _git(tmp_path, "init", "-q", "--bare", str(origin))
    _git(tmp_path, "init", "-q", str(work))
    _git(work, "remote", "add", "origin", origin.as_uri())

    fork_point = _commit(work, "models/base.py", "x = 1\n")
    head = _commit(work, "models/new_model.py", "import datasets\n")
    _git(work, "push", "-q", "origin", f"{head}:refs/heads/pull-request/{PR}")

    _git(work, "checkout", "-q", "--detach", fork_point)
    base = _commit(work, "contract.txt", "models must not import datasets\n")
    _git(work, "push", "-q", "origin", f"{base}:refs/heads/main")

    merge = _merge(work, base, head)
    _git(work, "push", "-q", "origin", f"{merge}:refs/pull/{PR}/merge")
    return Repo(origin=origin, work=work, base=base, head=head, merge=merge)


@dataclass
class Result:
    returncode: int
    output: str
    checked_out: str
    runner: Path
    gh_calls: list[str]


def _run(tmp_path: Path, repo: Repo, responses: list[str], *, ref: str = f"refs/heads/pull-request/{PR}") -> Result:
    """Clone the vetted head like actions/checkout, then run the script against it."""
    runner = tmp_path / "runner"
    _git(tmp_path, "clone", "-q", "--depth=1", "--branch", f"pull-request/{PR}", repo.origin.as_uri(), str(runner))

    fake = tmp_path / "fake_gh"
    fake.mkdir()
    (fake / "responses").write_text("".join(f"{line}\n" for line in responses))
    gh = fake / "gh"
    gh.write_text(
        "#!/bin/sh\n"
        'printf "%s\\n" "$*" >> "$FAKE_GH_DIR/calls"\n'
        'n=$(wc -l < "$FAKE_GH_DIR/calls")\n'
        'line=$(sed -n "${n}p" "$FAKE_GH_DIR/responses")\n'
        '[ -n "$line" ] || line=$(tail -n 1 "$FAKE_GH_DIR/responses")\n'
        'printf "%s\\n" "$line"\n'
    )
    gh.chmod(0o755)

    env = {
        **os.environ,
        "PATH": f"{fake}{os.pathsep}{os.environ['PATH']}",
        "FAKE_GH_DIR": str(fake),
        "GITHUB_REF": ref,
        "GITHUB_SHA": repo.head,
        "GITHUB_REPOSITORY": "NVIDIA-NeMo/Automodel",
        "MERGE_LOOKUP_ATTEMPTS": "3",
        "MERGE_LOOKUP_DELAY": "0",
    }
    proc = subprocess.run(["bash", str(SCRIPT)], cwd=runner, env=env, capture_output=True, text=True)
    calls_file = fake / "calls"
    return Result(
        returncode=proc.returncode,
        output=proc.stdout + proc.stderr,
        checked_out=_git(runner, "rev-parse", "HEAD"),
        runner=runner,
        gh_calls=calls_file.read_text().splitlines() if calls_file.exists() else [],
    )


def test_checks_out_the_merge_so_rules_added_to_main_apply(tmp_path, repo):
    result = _run(tmp_path, repo, [f"true|{repo.merge}|main"])

    assert result.returncode == 0, result.output
    assert result.checked_out == repo.merge
    assert (result.runner / "contract.txt").exists(), "the rule that landed on main must be visible"
    assert (result.runner / "models/new_model.py").exists(), "the PR's own change must still be there"
    assert f"merged into main at {repo.base}" in result.output
    assert result.gh_calls == [
        f"api repos/NVIDIA-NeMo/Automodel/pulls/{PR} --jq [.mergeable, .merge_commit_sha, .base.ref] "
        '| map(if . == null then "" else tostring end) | join("|")'
    ]


def test_waits_while_github_is_still_computing_mergeability(tmp_path, repo):
    result = _run(tmp_path, repo, ["||main", f"true|{repo.merge}|main"])

    assert result.returncode == 0, result.output
    assert result.checked_out == repo.merge
    assert len(result.gh_calls) == 2


def test_keeps_the_head_when_the_merge_was_built_from_an_unvetted_head(tmp_path, repo):
    """A push after copy-pr-bot vetted the head moves the merge ref; never check that code out."""
    _git(repo.work, "checkout", "-q", "--detach", repo.head)
    unvetted = _commit(repo.work, "models/unvetted.py", "y = 2\n")
    unvetted_merge = _merge(repo.work, repo.base, unvetted)
    _git(repo.work, "push", "-q", "-f", "origin", f"{unvetted_merge}:refs/pull/{PR}/merge")

    result = _run(tmp_path, repo, [f"true|{unvetted_merge}|main"])

    assert result.returncode == 0, result.output
    assert result.checked_out == repo.head
    assert f"built from {unvetted}, not the vetted head {repo.head}" in result.output
    assert not (result.runner / "models/unvetted.py").exists()


def test_keeps_the_head_when_the_merge_ref_moved_after_the_lookup(tmp_path, repo):
    stale = "0" * 40
    result = _run(tmp_path, repo, [f"true|{stale}|main"])

    assert result.returncode == 0, result.output
    assert result.checked_out == repo.head
    assert "moved during lookup" in result.output


@pytest.mark.parametrize(
    ("responses", "message"),
    [
        (["false||main"], "has no clean merge with main (mergeable=false)"),
        (["||main"], "has no clean merge with main (mergeable=pending)"),
        (["true||main"], "has no test merge commit"),
    ],
    ids=["conflicting", "never-resolves", "no-merge-commit"],
)
def test_keeps_the_head_when_github_has_no_usable_merge(tmp_path, repo, responses, message):
    result = _run(tmp_path, repo, responses)

    assert result.returncode == 0, result.output
    assert result.checked_out == repo.head
    assert "::warning title=Checking the PR head only::" in result.output
    assert message in result.output


def test_ignores_refs_that_are_not_pull_request_mirrors(tmp_path, repo):
    result = _run(tmp_path, repo, [f"true|{repo.merge}|main"], ref="refs/heads/main")

    assert result.returncode == 0, result.output
    assert result.checked_out == repo.head
    assert result.gh_calls == []


@pytest.mark.parametrize("job_name", ["linting", "import_linting", "type_checking"])
def test_static_check_jobs_check_out_the_merge_before_running(job_name):
    workflow = yaml.safe_load((ROOT / ".github/workflows/cicd-main.yml").read_text())
    steps = workflow["jobs"][job_name]["steps"]

    assert steps[0]["uses"].startswith("actions/checkout@")
    merge_step = steps[1]
    assert merge_step["run"] == "bash .github/scripts/checkout-pr-merge.sh"
    assert merge_step["if"] == "startsWith(github.ref, 'refs/heads/pull-request/')"
    assert merge_step["env"] == {"GH_TOKEN": "${{ github.token }}"}
    assert workflow["permissions"]["pull-requests"] == "read"
