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
from pathlib import Path

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
