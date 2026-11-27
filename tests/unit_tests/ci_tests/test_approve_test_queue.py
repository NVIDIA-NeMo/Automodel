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

import ast
import json
import os
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import requests
import yaml

_WORKFLOW = Path(".github/workflows/cicd-approve-test-queue.yml")


def _approval_script() -> ast.Module:
    workflow = yaml.safe_load(_WORKFLOW.read_text())
    steps = workflow["jobs"]["approve-queue"]["steps"]
    approval_step = next(step for step in steps if step["name"] == "Approve waiting deployments")
    return ast.parse(approval_step["run"])


def _internal_contributor_scope() -> tuple[set[str], ast.FunctionDef]:
    script = _approval_script()
    service_accounts_assignment = next(
        node
        for node in script.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "INTERNAL_SERVICE_ACCOUNTS" for target in node.targets)
    )
    service_accounts = ast.literal_eval(service_accounts_assignment.value)
    classifier = next(
        node for node in script.body if isinstance(node, ast.FunctionDef) and node.name == "is_internal_contributor"
    )
    return service_accounts, classifier


def _is_internal_contributor(pr_info: dict, sso_users: dict) -> bool:
    service_accounts, classifier = _internal_contributor_scope()
    module = ast.Module(body=[classifier], type_ignores=[])
    namespace = {"INTERNAL_SERVICE_ACCOUNTS": service_accounts, "sso_users": sso_users}
    exec(compile(module, str(_WORKFLOW), "exec"), namespace)
    return namespace["is_internal_contributor"](pr_info)


def test_svcnemo_autobot_uses_internal_queue_without_sso_membership():
    assert _is_internal_contributor({"user": {"login": "svcnemo-autobot"}}, {})


def test_external_contributor_stays_external_without_sso_membership():
    assert not _is_internal_contributor({"user": {"login": "external-contributor"}}, {})


def test_nvidia_member_stays_internal():
    assert _is_internal_contributor(
        {"user": {"login": "nvidia-contributor"}},
        {"nvidia-contributor": {"org_roles": ["NVIDIA:Member"]}},
    )


def test_unavailable_roster_download_stops_queue(tmp_path: Path) -> None:
    """A failed asset request must stop the job before any classification."""
    steps = yaml.safe_load(_WORKFLOW.read_text())["jobs"]["approve-queue"]["steps"]
    download = next(step for step in steps if step["name"] == "Download SSO users list")
    gh = tmp_path / "gh"
    gh.write_text("#!/bin/sh\nexit 1\n")
    gh.chmod(0o755)
    result = subprocess.run(
        ["bash", "-e", "-c", download["run"]],
        cwd=tmp_path,
        env={"PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}"},
        check=False,
    )
    assert result.returncode != 0
    assert not (tmp_path / "users_sso.json").exists()


def _response(data: dict | list, status: int = 200) -> requests.Response:
    """Build a GitHub response without contacting a service."""
    response = requests.Response()
    response.status_code = status
    response._content = json.dumps(data).encode()
    return response


def _run_approval(tmp_path: Path, roster: str, get: Mock) -> None:
    """Run the queue with fake reads and reject any unexpected approval."""
    path = tmp_path / "users_sso.json"
    path.write_text(roster)
    env = {
        "SSO_USERS_FILE": str(path),
        "APPROVAL_TOKEN": "unused",
        "READ_TOKEN": "unused",
        "GITHUB_REPOSITORY": "example/model",
        "MATRIX_BRANCH": "main",
        "CONTRIBUTOR_TYPE": "internal",
        "MAX_CONCURRENCY": "3",
    }
    steps = yaml.safe_load(_WORKFLOW.read_text())["jobs"]["approve-queue"]["steps"]
    approve = next(step for step in steps if step["name"] == "Approve waiting deployments")
    code = approve["run"].replace("${{ matrix.branch }}", "main")
    with (
        patch.dict(os.environ, env, clear=True),
        patch("requests.get", get),
        patch("requests.post", side_effect=AssertionError("Unexpected approval")),
        patch("time.sleep"),
    ):
        exec(compile(code, str(_WORKFLOW), "exec"), {})


@pytest.mark.parametrize(
    "roster",
    [
        "",
        "not JSON",
        "[]",
        "{}",
        '{"private":1}',
        '{"private":{"org_roles":null}}',
        '{"private":{"org_roles":null,"nvidia_email":"user@nvidia.com.example"}}',
        '{"private":{"org_roles":[3]}}',
        '{"private":{"org_roles":"bad"}}',
    ],
)
def test_invalid_roster_fails_before_api_reads(tmp_path: Path, roster: str) -> None:
    """Unusable identity data must stop processing even when the queue is empty."""
    get = Mock(side_effect=AssertionError("Unexpected GitHub read"))
    with pytest.raises(ValueError):
        _run_approval(tmp_path, roster, get)
    get.assert_not_called()


@pytest.mark.parametrize("status", [401, 403, 404, 429, 503])
def test_unavailable_pr_metadata_fails_queue(tmp_path: Path, status: int) -> None:
    """A denied or exhausted PR lookup must not turn a matching branch into a nonmatch."""
    run = {"id": 10, "name": "CICD NeMo", "head_branch": "pull-request/11"}
    attempts = 5 if status in (429, 503) else 1
    responses = [_response({"workflow_runs": [run]}), _response({"workflow_runs": []})]
    get = Mock(side_effect=responses + [_response({"message": "private body"}, status)] * attempts)
    with pytest.raises(RuntimeError, match="GET pulls/11") as error:
        _run_approval(tmp_path, '{"member":{"org_roles":[]}}', get)
    assert "private body" not in str(error.value)
    expected = f"HTTP {status}" if attempts == 1 else "exhausted 5 attempts"
    assert expected in str(error.value)
    assert get.call_count == len(responses) + attempts


@pytest.mark.parametrize("metadata", [{}, {"base": {"ref": "main"}}, {"user": {"login": "member"}}])
def test_incomplete_pr_metadata_fails_queue(tmp_path: Path, metadata: dict) -> None:
    """Classifying a PR requires a known author and base branch."""
    run = {"id": 10, "name": "CICD NeMo", "head_branch": "pull-request/11"}
    get = Mock(side_effect=[_response({"workflow_runs": [run]}), _response({"workflow_runs": []}), _response(metadata)])
    with pytest.raises(ValueError, match="PR #11"):
        _run_approval(tmp_path, '{"member":{}}', get)


@pytest.mark.parametrize("guest", [{"extra": "metadata"}, {"org_roles": None, "nvidia_email": "guest@NVIDIA.com"}])
def test_queue_filters_supported_runs_with_one_pr_read_each(
    tmp_path: Path, capsys: pytest.CaptureFixture, guest: dict
) -> None:
    """Ordinary branches, other PR targets and external authors stay outside the internal main queue."""
    branches = [
        None,
        "main",
        "mcore-testing-a",
        "pull-request/11-extra",
        "pull-request/11",
        "pull-request/12",
        "pull-request/13",
    ]
    runs = [{"id": index, "name": "CICD NeMo", "head_branch": branch} for index, branch in enumerate(branches)]
    responses = [_response({"workflow_runs": runs}), _response({"workflow_runs": []})]
    responses.extend(
        _response({"base": {"ref": base}, "user": {"login": user}})
        for base, user in [("main", "member"), ("dev", "member"), ("main", "guest")]
    )
    get = Mock(side_effect=responses + [_response({"workflow_runs": []})])
    roster = {"member": {"org_roles": ["NVIDIA:Member", "Other:Role"]}, "guest": guest}
    _run_approval(tmp_path, json.dumps(roster), get)
    assert "Current queued workflows (PRs targeting main, internal): 1" in capsys.readouterr().out
    assert [call.args[0].split("/")[-1] for call in get.call_args_list if "/pulls/" in call.args[0]] == [
        "11",
        "12",
        "13",
    ]


@pytest.mark.parametrize("deployments", [[], [{"environment": {"name": "other", "id": 12}}]])
def test_no_pending_test_deployment_skips_approval(tmp_path: Path, deployments: list) -> None:
    """An empty or unrelated environment list can follow a successful waiting-run lookup."""
    waiting = {"id": 10, "name": "CICD NeMo", "head_branch": "pull-request/11", "created_at": "", "display_title": ""}
    empty = _response({"workflow_runs": []})
    get = Mock(
        side_effect=[
            empty,
            empty,
            _response({"workflow_runs": [waiting]}),
            _response({"base": {"ref": "main"}, "user": {"login": "member"}}),
            _response(deployments),
        ]
    )
    _run_approval(tmp_path, '{"member":{"org_roles":["NVIDIA:Member"]}}', get)
    assert get.call_args.args[0].endswith("/actions/runs/10/pending_deployments")
