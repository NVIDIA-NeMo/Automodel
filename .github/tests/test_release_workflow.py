# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from NVIDIA-NeMo/Gym under Apache-2.0.
# https://github.com/NVIDIA-NeMo/Gym/blob/e58425147cf1729131f8cf3f16dad5fe266944dd/.github/tests/test_release_workflow.py
# SPDX-License-Identifier: Apache-2.0
# /// script
# dependencies = ["pyyaml>=6.0,<7"]
# ///
"""Exercise release guards without credentials, uploads, or package builds."""

import json
import os
import re
import subprocess
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = yaml.safe_load(next((ROOT / ".github/workflows").glob("release.y*ml")).read_text())
JOBS = WORKFLOW["jobs"]


def evaluate(expression: str, context: dict[str, object]) -> bool:
    """Evaluate the boolean and equality subset used by release job guards."""
    source = expression.strip().removeprefix("${{").removesuffix("}}")
    source = source.replace("!cancelled()", "True").replace("!failure()", "True")
    source = re.sub(r"\b(?:github|inputs|needs)\.[\w.-]+", lambda match: repr(context[match[0]]), source)
    source = source.replace("&&", " and ").replace("||", " or ")
    source = re.sub(r"!(?!=)", "not ", source)
    return bool(eval(" ".join(source.split()), {"__builtins__": {}}, {}))


class ReleaseGuards(unittest.TestCase):
    """Cover publication eligibility and unsuccessful or skipped dependencies."""

    def setUp(self) -> None:
        """Create a successful manual release context for the canonical repo."""
        self.context = {
            "github.repository": "NVIDIA-NeMo/Automodel",
            "github.event_name": "workflow_dispatch",
            "inputs.dry-run": False,
            "needs.validate-inputs.result": "success",
            "needs.release-workflow-tests.result": "success",
            "needs.bump.result": "success",
            "needs.build-wheel.result": "success",
            "needs.publish-wheel.result": "success",
        }

    def test_publish_requires_manual_non_dry_run(self) -> None:
        """Pushes and dry runs cannot request the production upload job."""
        for event in ("push", "workflow_dispatch"):
            for dry_run in (False, True):
                with self.subTest(event=event, dry_run=dry_run):
                    context = dict(self.context, **{"github.event_name": event, "inputs.dry-run": dry_run})
                    self.assertEqual(
                        evaluate(JOBS["publish-wheel"]["if"], context), event == "workflow_dispatch" and not dry_run
                    )
        self.context["github.repository"] = "example/fork"
        self.assertFalse(evaluate(JOBS["publish-wheel"]["if"], self.context))

    def test_missing_build_blocks_upload(self) -> None:
        """Failed, cancelled and skipped builds cannot reach publication."""
        for state in ("failure", "cancelled", "skipped"):
            with self.subTest(state=state):
                self.context["needs.build-wheel.result"] = state
                self.assertFalse(evaluate(JOBS["publish-wheel"]["if"], self.context))

    def test_regression_failures_block_version_preparation(self) -> None:
        """Require the regression job to succeed before version preparation."""
        self.assertIn("release-workflow-tests", JOBS["bump"]["needs"])
        self.assertTrue(evaluate(JOBS["bump"]["if"], self.context))
        for state in ("failure", "cancelled", "skipped"):
            with self.subTest(state=state):
                self.context["needs.release-workflow-tests.result"] = state
                self.assertFalse(evaluate(JOBS["bump"]["if"], self.context))

    def test_real_finalize_requires_upload(self) -> None:
        """A missing upload cannot finalize a real release; dry runs may continue."""
        for state in ("failure", "cancelled", "skipped"):
            with self.subTest(state=state):
                self.context["needs.publish-wheel.result"] = state
                self.context["inputs.dry-run"] = False
                self.assertFalse(evaluate(JOBS["finalize"]["if"], self.context))
                self.context["inputs.dry-run"] = True
                self.assertTrue(evaluate(JOBS["finalize"]["if"], self.context))

    def test_summary_rejects_missing_required_jobs(self) -> None:
        """Run the actual shell summary against failures and permitted skips."""
        script = JOBS["release-summary"]["steps"][0]["run"]
        for publish in (False, True):
            baseline = {name: {"result": "success"} for name in JOBS["release-summary"]["needs"]}
            baseline["publish-wheel"]["result"] = "success" if publish else "skipped"
            for job in (
                "release-workflow-tests",
                "validate-inputs",
                "bump",
                "build-wheel",
                "publish-wheel",
                "finalize",
            ):
                for state in ("success", "failure", "cancelled", "skipped"):
                    with self.subTest(publish=publish, job=job, state=state):
                        needs = {name: dict(value) for name, value in baseline.items()}
                        needs[job]["result"] = state
                        env = dict(
                            os.environ,
                            NEEDS=json.dumps(needs),
                            PUBLISH_EXPECTED=str(publish).lower(),
                            RELEASE_SKIPPED="false",
                        )
                        result = subprocess.run(
                            ["bash", "-euo", "pipefail", "-c", script], env=env, capture_output=True
                        )
                        expected = state == ("skipped" if job == "publish-wheel" and not publish else "success")
                        self.assertEqual(result.returncode == 0, expected, result.stderr.decode())

    def test_validate_ref_treats_input_as_data(self) -> None:
        """Only full immutable SHAs pass; branch names and shell payloads fail."""
        script = JOBS["validate-inputs"]["steps"][0]["run"]
        for ref in ("a" * 40, "main", "abc1234", "$(exit 0)", "a" * 40 + "\n"):
            with self.subTest(ref=ref):
                result = subprocess.run(
                    ["bash", "-euo", "pipefail", "-c", script],
                    env=dict(os.environ, RELEASE_REF=ref),
                    capture_output=True,
                )
                self.assertEqual(result.returncode == 0, ref == "a" * 40)

    def test_upload_has_no_checkout_or_password(self) -> None:
        """Keep source execution and build credentials outside the OIDC job."""
        self.assertEqual(JOBS["build-wheel"]["permissions"], {"contents": "read"})
        self.assertNotIn("secrets", JOBS["build-wheel"])
        upload = JOBS["publish-wheel"]
        self.assertEqual(upload["permissions"], {"id-token": "write"})
        self.assertEqual(upload["environment"]["name"], "pypi")
        self.assertEqual(len(upload["steps"]), 2)
        for step in upload["steps"]:
            self.assertNotIn("run", step)
            self.assertNotIn("password", step.get("with", {}))
            self.assertNotIn("checkout", step["uses"])

    def test_dispatch_guard_preserves_push_rehearsals(self) -> None:
        """Only manual dispatches from refs other than main fail this step."""
        step = next(step for step in JOBS["validate-inputs"]["steps"] if step["name"] == "Require dispatch from main")
        for event in ("workflow_dispatch", "push"):
            for ref in ("refs/heads/main", "refs/tags/main", "refs/heads/r0.7.0", "refs/heads/pull-request/123"):
                with self.subTest(event=event, ref=ref):
                    rejected = evaluate(step["if"], {"github.event_name": event, "github.ref": ref})
                    self.assertEqual(rejected, event == "workflow_dispatch" and ref != "refs/heads/main")
        self.assertNotEqual(subprocess.run(["bash"], input=step["run"], text=True, capture_output=True).returncode, 0)

    def test_release_commit_belongs_to_protected_target(self) -> None:
        """Only a matching protected branch containing the requested SHA can be released."""
        step = next(step for step in JOBS["validate-inputs"]["steps"] if step["name"].startswith("Verify release"))
        self.assertEqual(step["if"], "github.event_name == 'workflow_dispatch'")
        candidate_sha = "a" * 40
        target_sha = "b" * 40
        fixture = """
gh() {
    [[ $1 == api && $3 == --jq && $# == 4 ]] || return 1
    case "$2" in
      "repos/example/repo/branches/$TARGET_BRANCH") reply=$BRANCH_REPLY ;;
      "repos/example/repo/compare/$RELEASE_REF...$TIP") reply=$COMPARE_REPLY ;;
      *) return 1 ;;
    esac
    jq -r "$4" <<< "$reply"
}
"""
        for branch, protected, comparison, accepts in (
            ("main", True, "identical", True),
            ("r0.7.0", True, "ahead", True),
            ("v0.2.0", True, "identical", True),
            ("r0.7.0", False, "identical", False),
            ("r0.7.0", True, "behind", False),
            ("main", True, "diverged", False),
            ("r0.7.0-preview", True, "ahead", False),
            ("pull-request/123", True, "ahead", False),
        ):
            with self.subTest(branch=branch, protected=protected, comparison=comparison):
                env = dict(
                    os.environ,
                    GITHUB_REPOSITORY="example/repo",
                    RELEASE_REF=candidate_sha,
                    TIP=target_sha,
                    TARGET_BRANCH=branch,
                    BRANCH_REPLY=json.dumps({"protected": protected, "commit": {"sha": target_sha}}),
                    COMPARE_REPLY=json.dumps({"status": comparison}),
                )
                result = subprocess.run(
                    ["bash", "-euo", "pipefail"], input=fixture + step["run"], text=True, env=env, capture_output=True
                )
                self.assertEqual(result.returncode == 0, accepts, result.stderr)

    def test_release_jobs_use_only_the_environment_app_for_github_writes(self) -> None:
        """Release and freeze use the protected App while PR rehearsals retain read access."""
        freeze = yaml.safe_load((ROOT / ".github/workflows/release-freeze.yml").read_text())["jobs"]["code-freeze"]
        self.assertEqual(freeze["with"]["freeze-source-branch"], "main")
        self.assertEqual(freeze["with"]["freeze-commit"], "${{ inputs.freeze-commit }}")
        for job in (freeze, JOBS["bump"], JOBS["finalize"]):
            self.assertEqual(job["with"]["release-environment"], "automodel-release")
            self.assertNotIn("app-id", job["with"])
            self.assertEqual(job["secrets"]["RELEASE_APP_PRIVATE_KEY"], "${{ secrets.RELEASE_APP_PRIVATE_KEY }}")
            self.assertTrue(set(job["secrets"]) <= {"RELEASE_APP_PRIVATE_KEY", "SLACK_WEBHOOK", "SLACK_WEBHOOK_ADMIN"})
        self.assertEqual(JOBS["bump"]["permissions"], {"contents": "read"})
        self.assertEqual(JOBS["finalize"]["permissions"], {"contents": "read", "pull-requests": "read"})

    def test_fern_is_the_documentation_publisher(self) -> None:
        """Reject retired Sphinx dispatches before version and package publication."""
        docs = next(
            step for step in JOBS["validate-inputs"]["steps"] if step["name"] == "Check documentation publisher"
        )
        self.assertEqual(docs["if"], "inputs.publish-docs")
        result = subprocess.run(["bash"], input=docs["run"], text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Publish Fern Docs", result.stdout)
        self.assertIs(JOBS["finalize"]["with"]["publish-docs"], False)


if __name__ == "__main__":
    unittest.main()
