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

import json
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


@pytest.mark.parametrize(
    "case",
    [
        "valid",
        "stale-head",
        "stale-base",
        "wrong-pr-parent",
        "single-parent",
        "extra-parent",
        "closed",
        "pending",
        "main",
        "queue",
        "deployment",
    ],
)
@pytest.mark.parametrize("builder", ["workflow", "composite"])
def test_container_build_resolves_only_the_vetted_revision(tmp_path: Path, case: str, builder: str) -> None:
    """Pin PR images to the exact GitHub test merge and preserve the event SHA for other revisions."""
    if builder == "workflow":
        workflow = yaml.safe_load((ROOT / ".github/workflows/cicd-main.yml").read_text())
        jobs = workflow["jobs"]
        resolver = jobs["cicd-compute-build-matrix"]["steps"][0]
        assert jobs["cicd-compute-build-matrix"]["outputs"]["sha"] == "${{ steps.revision.outputs.sha }}"
        assert "needs.cicd-compute-build-matrix.result == 'success'" in jobs["cicd-container-build"]["if"]
        for name in ("cicd-container-build", "cicd-unit-tests", "cicd-e2e-tests", "cicd-e2e-tests-gb200", "Coverage"):
            checkout = jobs[name]["steps"][0]["with"]
            assert checkout["ref"] == "${{ needs.cicd-compute-build-matrix.outputs.sha }}"
            assert checkout["persist-credentials"] is False
            if name != "cicd-container-build":
                assert "needs.cicd-container-build.result == 'success'" in jobs[name]["if"]
    else:
        action = yaml.safe_load((ROOT / ".github/actions/build-container/action.yml").read_text())
        install_gh, resolver, checkout, *_ = action["runs"]["steps"]
        assert install_gh["name"] == "Install GH CLI"
        assert checkout["with"]["ref"] == "${{ steps.revision.outputs.sha }}"
        assert checkout["with"]["persist-credentials"] is False
    assert resolver["id"] == "revision"

    head, base, merge, other = (c * 40 for c in "abcd")
    pr = {
        "state": "closed" if case == "closed" else "open",
        "head": {"sha": other if case == "stale-head" else head},
        "base": {"sha": base},
        "merge_commit_sha": None if case == "pending" else merge,
    }
    parents = {
        "stale-base": [other, head],
        "wrong-pr-parent": [base, other],
        "single-parent": [head],
        "extra-parent": [base, head, other],
    }.get(case, [base, head])
    ref = {"main": "main", "queue": "gh-readonly-queue/main/pr-7-test", "deployment": "deploy-release/test"}.get(
        case, "pull-request/7"
    )
    output = tmp_path / "outputs"
    fixture = """
gh() {
  [[ "$1" == api && "$3" == --jq && $# == 4 ]] || return 1
  case "$2" in
    repos/example/repo/pulls/7) jq -r "$4" <<< "$PR_JSON" ;;
    "repos/example/repo/git/commits/$MERGE_SHA") jq -r "$4" <<< "$COMMIT_JSON" ;;
    *) return 1 ;;
  esac
}
"""
    env = {
        "PATH": os.environ["PATH"],
        "GITHUB_REF": f"refs/heads/{ref}",
        "GITHUB_SHA": head,
        "GITHUB_REPOSITORY": "example/repo",
        "GITHUB_OUTPUT": str(output),
        "MERGE_SHA": merge,
        "PR_JSON": json.dumps(pr),
        "COMMIT_JSON": json.dumps({"parents": [{"sha": parent} for parent in parents]}),
    }
    result = subprocess.run(["bash"], input=fixture + resolver["run"], text=True, env=env, capture_output=True)
    if case in {"valid", "main", "queue", "deployment"}:
        assert result.returncode == 0, result.stderr
        assert output.read_text().splitlines() == (
            [f"sha={merge}", "pr_number=7", f"base={base}"]
            if case == "valid"
            else [f"sha={head}", "pr_number=", "base="]
        )
    else:
        assert result.returncode != 0
        assert not output.exists()


def test_container_action_builds_without_caller_credentials() -> None:
    """Keep the deprecated input compatible and omit credentials from the caller's Dockerfile."""
    action = yaml.safe_load((ROOT / ".github/actions/build-container/action.yml").read_text())
    assert action["inputs"]["PAT"]["required"] is False
    assert "deprecationMessage" in action["inputs"]["PAT"]
    steps = action["runs"]["steps"]
    assert "inputs.PAT" not in json.dumps(steps)
    build = next(step["with"] for step in steps if step["name"] == "Build and push")
    assert build["context"] == "."
    assert not {"secrets", "secret-envs", "secret-files", "ssh"} & build.keys()
