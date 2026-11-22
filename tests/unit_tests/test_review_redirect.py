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


from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_review_redirect_is_least_privilege_and_does_not_execute_review():
    workflow = yaml.safe_load((ROOT / ".github/workflows/claude-review.yml").read_text())
    job = workflow["jobs"]["redirect-to-review"]
    condition = job["if"]
    assert "github.event_name == 'issue_comment'" in condition
    assert "github.event.issue.pull_request" in condition
    assert "github.event.comment.user.type != 'Bot'" in condition
    for command in ("/claude review", "/claude strict-review"):
        assert f"contains(github.event.comment.body, '{command}')" in condition
    assert job["permissions"] == {"pull-requests": "write"}
    assert "uses" not in job
    assert "secrets" not in job
    assert len(job["steps"]) == 1
    step = job["steps"][0]
    assert "uses" not in step
    assert step["run"] == 'gh pr comment "$PR_NUMBER" --repo "$REPO" --body "$NOTICE"'
    assert "${{" not in step["run"]
    assert "model=claude" in step["env"]["NOTICE"]
    assert "model=codex" in step["env"]["NOTICE"]
    assert "/review help" in step["env"]["NOTICE"]
    assert job["env"]["REVIEW_COMMAND"] == (
        "${{ contains(github.event.comment.body, '/claude strict-review') && '/review mode=strict' || '/review' }}"
    )


def test_formal_review_rubric_is_inert_and_uses_formal_submission():
    rubric = (ROOT / "skills/pr-review/SKILL.md").read_text()
    metadata = yaml.safe_load(rubric.split("---", 2)[1])
    assert metadata["name"] == "pr-review"
    assert metadata["disable-model-invocation"] is True
    assert metadata["user-invocable"] is False
    assert "mode=light" in rubric
    assert "mode=strict" in rubric
    assert "Do not run GitHub commands or" in rubric
    assert "Never approve an incomplete" in rubric


def test_legacy_review_execution_is_removed():
    workflow = yaml.safe_load((ROOT / ".github/workflows/claude-review.yml").read_text())
    assert set(workflow["jobs"]) == {"redirect-to-review"}
    assert workflow["permissions"] == {}
