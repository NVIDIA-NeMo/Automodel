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

import json
import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml

WORKFLOWS = Path(__file__).resolve().parents[3] / ".github/workflows"


def test_limit_legacy_secrets_to_existing_integrations() -> None:
    """Keep the human queue reviewer and external integrations out of build and test jobs."""
    permitted = {
        "PAT": {("cicd-approve-test-queue.yml", "approve-queue")},
        "NVIDIA_MANAGEMENT_ORG_PAT": {
            ("cicd-main.yml", "pre-flight"),
            ("cicd-approve-test-queue.yml", "approve-queue"),
            ("community-bot.yml", "community-bot"),
        },
        "NVSKILLS_CI_DISPATCH_TOKEN": {("request-nvskills-ci.yml", "request")},
    }
    for file in WORKFLOWS.glob("*.y*ml"):
        workflow = yaml.safe_load(file.read_text())
        assert not re.search(r"\b(?:BOT_KEY|BOT_ID)\b", str(workflow), re.IGNORECASE), file.name
        for name, job in workflow["jobs"].items():
            assert job.get("secrets") != "inherit", (file.name, name)
            for secret in re.findall(r"\bsecrets\.(\w+)", str(job)):
                if secret in permitted:
                    assert (file.name, name) in permitted[secret], (file.name, name, secret)


def test_changelog_requires_successful_compare_of_requested_refs() -> None:
    """Prevent a failed comparison from opening a changelog pull request."""
    workflow = yaml.safe_load((WORKFLOWS / "changelog-build.yml").read_text())
    step = next(step for step in workflow["jobs"]["changelog"]["steps"] if step["name"] == "Build Changelog")
    assert step["with"]["failOnError"] == "true"
    assert step["with"]["fromTag"] == "${{ inputs.last-release-tag }}"
    assert step["with"]["toTag"] == "${{ inputs.release-branch }}"


@pytest.mark.parametrize("current_head", [True, False])
def test_guardwords_uses_only_the_current_mirror(tmp_path: Path, current_head: bool) -> None:
    """Only submit the reviewed PR's public metadata to the catalog-scanning job."""
    workflow = yaml.safe_load((WORKFLOWS / "guardwords-check.yml").read_text())
    step = workflow["jobs"]["pr-info"]["steps"][0]
    base, pushed, other = "a" * 40, "b" * 40, "c" * 40
    gh = tmp_path / "gh"
    gh.write_text(f"#!/bin/sh\nprintf '%s\\t%s\\n' {base} {pushed if current_head else other}\n")
    gh.chmod(0o755)
    output = tmp_path / "output"
    env = {
        **os.environ,
        "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
        "GITHUB_OUTPUT": str(output),
        "GITHUB_REF_NAME": "pull-request/17",
        "GITHUB_REPOSITORY": "example/model",
        "GITHUB_SHA": pushed,
    }
    result = subprocess.run(["bash", "-eu", "-o", "pipefail", "-c", step["run"]], env=env, check=False)
    if current_head:
        assert result.returncode == 0
        assert output.read_text().splitlines() == ["number=17", f"base={base}"]
    else:
        assert result.returncode != 0
        assert not output.exists()
    assert workflow["jobs"]["guardwords-check"]["secrets"] == {
        "GUARDWORDS_DEPLOY_KEY": "${{ secrets.GUARDWORDS_DEPLOY_KEY }}"
    }


@pytest.mark.parametrize("file", ["cicd-approve-test-queue.yml", "dependabot.yml"])
@pytest.mark.parametrize("team", ["", "S123"])
def test_failure_alert_routes_to_ci_channel(file: str, team: str, tmp_path: Path) -> None:
    """Report the failing workflow and run, mentioning the team only when configured."""
    caller = yaml.safe_load((WORKFLOWS / file).read_text())
    notify = caller["jobs"]["notify"]
    assert notify["if"] == "failure()"
    assert notify["permissions"] == {}
    assert notify["secrets"] == {
        "SLACK_CI_CHANNEL_WEBHOOK": "${{ secrets.SLACK_CI_CHANNEL_WEBHOOK }}",
        "SLACK_TEAM_GROUP_ID": "${{ secrets.SLACK_TEAM_GROUP_ID }}",
    }
    called = yaml.safe_load((WORKFLOWS / Path(notify["uses"]).name).read_text())
    compose, send = called["jobs"]["notify"]["steps"]
    assert called["permissions"] == {}
    assert compose["env"]["SLACK_TEAM_GROUP_ID"] == "${{ secrets.SLACK_TEAM_GROUP_ID }}"
    assert re.fullmatch(r"NVIDIA-NeMo/FW-CI-templates/\.github/actions/send-slack-alert@[0-9a-f]{40}", send["uses"])
    assert send["with"] == {
        "message": "${{ steps.alert.outputs.message }}",
        "webhook": "${{ secrets.SLACK_CI_CHANNEL_WEBHOOK }}",
    }
    output = tmp_path / "output"
    env = {
        "PATH": os.environ["PATH"],
        "GITHUB_OUTPUT": str(output),
        "GITHUB_SERVER_URL": "https://github.com",
        "GITHUB_REPOSITORY": "example/model",
        "GITHUB_RUN_ID": "17",
        "GITHUB_WORKFLOW": caller["name"],
        "SLACK_TEAM_GROUP_ID": team,
    }
    subprocess.run(["bash", "-eu", "-c", compose["run"]], check=True, env=env)
    mention = " cc <!subteam^S123>" if team else ""
    link = f"<https://github.com/example/model/actions/runs/17|{caller['name']}>"
    assert output.read_text() == f"message={link} failed.{mention}\n"


@pytest.mark.parametrize(
    ("file", "name", "target"),
    [
        ("cicd-main.yml", "Nemo_CICD_Test", "cicd-unit-tests"),
        ("copyright-check.yml", "copyright-check-summary", "copyright-check"),
        ("install-test.yml", "install-test-summary", "pip-test"),
    ],
)
def test_required_summaries_use_completed_dependencies(file: str, name: str, target: str) -> None:
    """Require evidence of success, with explicit exceptions for optional or skipped jobs."""
    job = yaml.safe_load((WORKFLOWS / file).read_text())["jobs"][name]
    assert job["permissions"] == {}
    (step,) = job["steps"]
    defaults = {key: {"result": "success"} for key in job["needs"]}

    def accepts(overrides: dict[str, str], *, present: bool = True, **settings: str) -> bool:
        results = defaults | {key: {"result": status} for key, status in overrides.items()} if present else {}
        env = {
            **os.environ,
            "RESULTS": json.dumps(results),
            "PREFLIGHT": results.get("pre-flight", {}).get("result", ""),
            "COPYRIGHT": results.get("copyright-check", {}).get("result", ""),
            "ALLOW_SKIPS": "false",
            "CHECK_GB200": "true",
            "CI_WORKLOAD": "false",
            **settings,
        }
        return subprocess.run(["bash", "-eu", "-c", step["run"]], env=env, capture_output=True).returncode == 0

    assert accepts({})
    assert not accepts({}, present=False)
    assert not accepts({target: "failure"})
    assert not accepts({target: "cancelled"})
    assert not accepts({target: "skipped"})
    assert accepts({target: "skipped"}, ALLOW_SKIPS="true")
    assert not accepts({target: "failure"}, ALLOW_SKIPS="true")
    assert not accepts({"pre-flight": "failure"}, ALLOW_SKIPS="true", CI_WORKLOAD="true")
    if name == "Nemo_CICD_Test":
        assert accepts({"cicd-e2e-tests-gb200": "skipped"}, CHECK_GB200="false")
        assert not accepts({"cicd-e2e-tests-gb200": "skipped"})
        assert accepts({target: "failure"}, CI_WORKLOAD="true")
