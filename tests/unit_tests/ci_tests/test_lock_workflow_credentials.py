# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check credential boundaries for the dependency and PR lockfile workflows."""

import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

WORKFLOWS = Path(__file__).resolve().parents[3] / ".github/workflows"
LOCKS = ("uv.lock", "docker/common/uv-pytorch.lock")


def test_dependency_update_uses_separate_git_and_pr_credentials() -> None:
    """The scheduled resolver has read access; the signed PR uses separate native and App credentials."""
    caller = yaml.safe_load((WORKFLOWS / "dependabot.yml").read_text())["jobs"]
    assert caller["get-release-branch-names"]["permissions"] == {"contents": "read"}
    assert caller["get-release-branch-names"]["if"] == "github.ref == 'refs/heads/main'"
    assert set(caller["bump-tags"]["secrets"]) == {"PR_APP_PRIVATE_KEY", "SSH_KEY", "SSH_PWD"}
    jobs = yaml.safe_load((WORKFLOWS / "_update_dependencies.yml").read_text())["jobs"]
    assert jobs["update-lockfile"]["permissions"] == {"contents": "read"}
    create = jobs["create-pr"]
    assert create["environment"] == "main"
    assert create["permissions"] == {"contents": "write"}
    steps = {step["name"]: step for step in create["steps"]}
    assert steps["Import GPG key"]["with"]["git_commit_gpgsign"] is True
    app = steps["Create token to manage the bump pull request"]["with"]
    assert app == {
        "app-id": "${{ vars.PR_APP_ID }}",
        "private-key": "${{ secrets.PR_APP_PRIVATE_KEY }}",
        "permission-pull-requests": "write",
    }
    pr = steps["Create Bump PR"]["with"]
    assert pr["token"] == "${{ steps.pr-token.outputs.token }}"
    assert pr["git-token"] == "${{ github.token }}"
    assert pr["signoff"] is True
    assert tuple(pr["add-paths"].splitlines()) == LOCKS


@pytest.mark.parametrize(
    ("protected", "stale"),
    [(False, False), (True, False), (False, True)],
)
def test_uv_writer_updates_only_expected_head_and_paths(tmp_path: Path, protected: bool, stale: bool) -> None:
    """Run the writer against a local remote and enforce same-head, unprotected-branch updates."""
    jobs = yaml.safe_load((WORKFLOWS / "uv-lock-generation.yml").read_text())["jobs"]
    read_job = jobs["check_uv_lock_and_update"]
    assert read_job["permissions"] == {"contents": "read"}
    assert (
        next(step for step in read_job["steps"] if step["name"] == "Checkout")["with"]["persist-credentials"] is False
    )
    write_job = jobs["push-locks"]
    assert "github.event.pull_request.head.repo.full_name == github.repository" in write_job["if"]
    steps = {step["name"]: step for step in write_job["steps"]}
    assert steps["Checkout the PR commit"]["with"]["ref"] == "${{ github.event.pull_request.head.sha }}"
    script = steps["Commit and push uv lock updates"]["run"]
    remote, checkout = tmp_path / "remote", tmp_path / "work"
    checkout.mkdir()
    branch = "topic/uv-locks"
    local_env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path), "GIT_CONFIG_NOSYSTEM": "1"}

    def git(*args: str) -> str:
        """Run Git with isolated configuration in the local checkout."""
        return subprocess.check_output(["git", *args], cwd=checkout, env=local_env, text=True).strip()

    git("init", "--bare", str(remote))
    git("init", "-b", branch)
    git("config", "user.name", "Fixture")
    git("config", "user.email", "fixture@example.org")
    for lock in LOCKS:
        (checkout / lock).parent.mkdir(parents=True, exist_ok=True)
        (checkout / lock).write_text("original")
        artifact = tmp_path / "uv-locks" / lock
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text("updated")
    (tmp_path / "uv-locks/extra-file").write_text("unexpected")
    git("add", "--", *LOCKS)
    git("commit", "-m", "original")
    head = git("rev-parse", "HEAD")
    git("remote", "add", "origin", str(remote))
    git("push", "origin", branch)
    git("checkout", "--detach", head)
    fixture = 'gh() { [[ "$1" == api && "$2" == repos/example/repo/branches/topic%2Fuv-locks ]] && echo "$REPLY"; }\n'
    result = subprocess.run(
        ["bash"],
        input=fixture + script,
        cwd=checkout,
        text=True,
        capture_output=True,
        env={
            **local_env,
            "RUNNER_TEMP": str(tmp_path),
            "GITHUB_REPOSITORY": "example/repo",
            "PR_BRANCH": branch,
            "PR_HEAD": head,
            "REPLY": json.dumps({"protected": protected, "commit": {"sha": "f" * 40 if stale else head}}),
        },
    )
    tip = git("--git-dir", str(remote), "rev-parse", f"refs/heads/{branch}")
    if protected or stale:
        assert result.returncode != 0
        assert tip == head
    else:
        assert result.returncode == 0, result.stderr
        assert git("rev-parse", f"{tip}^") == head
        assert set(git("diff", "--name-only", head, tip).splitlines()) == set(LOCKS)
        assert "Signed-off-by:" in git("log", "-1", "--format=%B", tip)
