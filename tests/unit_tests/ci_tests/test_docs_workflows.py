# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

_WORKFLOWS = Path(__file__).resolve().parents[3] / ".github/workflows"


def _run_pr_navigation(workspace: Path) -> subprocess.CompletedProcess[str]:
    workflow = yaml.safe_load((_WORKFLOWS / "fern-docs-preview.yml").read_text())
    step = next(
        step for step in workflow["jobs"]["preview"]["steps"] if step["name"] == "Use validated PR nightly navigation"
    )
    script = step["run"].split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    return subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=workspace,
        env={"RUNNER_TEMP": str(workspace / "runner")},
        capture_output=True,
        text=True,
    )


def test_preview_preserves_trusted_navigation_targets(tmp_path: Path) -> None:
    """PR navigation is used while trusted config and fallback content are retained."""
    trusted = {
        "fern/versions/nightly.yml": (
            "navigation:\n  - page: Old\n    path: ../../old/page.mdx\n  - folder: ../product-docs/reference\n"
        ),
        "fern/versions/v0.5.yml": "trusted archived navigation",
        "fern/docs.yml": "trusted site configuration",
        "fern/components/Badge.tsx": "trusted component",
        "old/page.mdx": "![diagram](./diagram.png)",
        "old/diagram.png": "trusted image",
        "deleted.mdx": "deleted in PR",
        "unchanged.mdx": "original text",
    }
    proposed = {
        "fern/versions/nightly.yml": (
            "navigation:\n  - section: Tutorials\n    skip-slug: true\n    contents:\n"
            "      - page: New\n        path: ../../new/page.mdx\n        slug: recipes-e2e-examples/new\n"
            "  - folder: ../product-docs/reference\n"
        ),
        "fern/versions/v0.5.yml": "PR archived navigation must not be used",
        "fern/components/Badge.tsx": "PR component must not be used",
        "fern/docs.yml": "PR configuration must not be used",
        "new/page.mdx": "moved page",
        "unchanged.mdx": "updated text",
        "injected.py": "PR code must not be copied",
        ".hidden.mdx": "hidden content must not be copied",
    }
    for source, files in (("trusted-source", trusted), ("pr-source", proposed)):
        for name, content in files.items():
            path = tmp_path / source / "docs" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
    # Record a base and a PR commit so the historical deletion logic can be
    # exercised against this fixture as well as the fixed workflow.
    pr = tmp_path / "pr-source"
    shutil.rmtree(pr / "docs")
    shutil.copytree(tmp_path / "trusted-source/docs", pr / "docs")
    git = ["git", "-c", "core.hooksPath=/dev/null", "-c", "user.name=Test", "-c", "user.email=test@example.com"]
    for args in (
        ["init", "-q"],
        ["add", "."],
        ["commit", "-qm", "base"],
        ["update-ref", "refs/remotes/origin/main", "HEAD"],
    ):
        subprocess.run([*git, *args], cwd=pr, check=True, capture_output=True)
    shutil.rmtree(pr / "docs")
    for name, content in proposed.items():
        path = pr / "docs" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    for args in (["add", "-A"], ["commit", "-qm", "move docs"]):
        subprocess.run([*git, *args], cwd=pr, check=True, capture_output=True)
    workflow = yaml.safe_load((_WORKFLOWS / "fern-docs-preview.yml").read_text())
    step = next(step for step in workflow["jobs"]["preview"]["steps"] if step["name"] == "Prepare preview docs")
    subprocess.run(
        ["bash", "-e", "-c", step["run"]],
        cwd=tmp_path,
        env={**os.environ, "RUNNER_TEMP": str(tmp_path / "runner")},
        check=True,
        capture_output=True,
        text=True,
    )
    preview = tmp_path / "runner/fern-preview/docs"
    for name, content in trusted.items():
        assert (preview / name).read_text() == ("updated text" if name == "unchanged.mdx" else content)
    assert (preview / "new/page.mdx").read_text() == "moved page"
    for excluded in ("injected.py", ".hidden.mdx"):
        assert not (preview / excluded).exists()

    result = _run_pr_navigation(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert yaml.safe_load((preview / "fern/versions/nightly.yml").read_text()) == yaml.safe_load(
        proposed["fern/versions/nightly.yml"]
    )
    for name in ("fern/docs.yml", "fern/versions/v0.5.yml", "fern/components/Badge.tsx"):
        assert (preview / name).read_text() == trusted[name]


@pytest.mark.parametrize(
    "proposed",
    [
        "navigation:\n  - page: Private\n    path: ../../../../private.mdx\n",
        "navigation:\n  - page: Private\n    path: /proc/self/environ\n",
        "navigation:\n  - page: Remote\n    path: https://example.com/page.mdx\n",
        "navigation:\n  - page: Config\n    path: ../docs.yml\n",
        "navigation:\n  - page: Symlink\n    path: ../../link.mdx\n",
        "navigation:\n  - folder: /proc/self\n",
        "navigation:\n  - section: Unsafe\n    icon: ../../../../private.svg\n",
        "navigation: []\njs: ../../../../private.js\n",
        "navigation:\n  - page: Safe\n    path: ../../safe.mdx\n    api: malicious\n",
        "navigation: &cycle\n  - section: Recursive\n    contents: *cycle\n",
        "!!python/object/apply:os.system ['touch injected']",
    ],
)
def test_preview_rejects_unsafe_navigation(tmp_path: Path, proposed: str) -> None:
    preview = tmp_path / "runner/fern-preview/docs"
    destination = preview / "fern/versions/nightly.yml"
    destination.parent.mkdir(parents=True)
    original = "navigation:\n  - page: Safe\n    path: ../../safe.mdx\n"
    destination.write_text(original)
    (preview / "safe.mdx").write_text("safe page")
    (preview / "fern/docs.yml").write_text("trusted config")
    private = tmp_path / "runner/private.mdx"
    private.write_text("SENTINEL_SECRET")
    (preview / "link.mdx").symlink_to(private)
    source = tmp_path / "pr-source/docs/fern/versions/nightly.yml"
    source.parent.mkdir(parents=True)
    source.write_text(proposed)

    result = _run_pr_navigation(tmp_path)

    assert result.returncode != 0
    assert destination.read_text() == original
    assert "SENTINEL_SECRET" not in result.stdout + result.stderr
    assert not (tmp_path / "injected").exists()


@pytest.mark.parametrize("job", ["linting", "import_linting", "type_checking"])
def test_code_lint_jobs_use_docs_only_preflight(job: str) -> None:
    """All code lint jobs honor the same docs-only decision as training CI."""
    workflow = yaml.safe_load((_WORKFLOWS / "cicd-main.yml").read_text())
    assert "pre-flight" in workflow["jobs"][job]["needs"]
    assert workflow["jobs"][job]["if"] == "needs.pre-flight.outputs.docs_only != 'true'"


@pytest.mark.parametrize(
    ("docs_only", "preflight", "results", "expected"),
    [
        ("true", "success", ("skipped", "skipped", "skipped"), 0),
        ("false", "success", ("success", "success", "success"), 0),
        ("false", "success", ("skipped", "skipped", "skipped"), 1),
        ("", "failure", ("skipped", "skipped", "skipped"), 1),
        ("true", "cancelled", ("skipped", "skipped", "skipped"), 1),
        ("true", "success", ("failure", "skipped", "skipped"), 1),
        ("true", "success", ("skipped", "cancelled", "skipped"), 1),
        ("false", "success", ("success", "success", "failure"), 1),
    ],
)
def test_lint_gate_accepts_only_success_or_intentional_docs_skip(
    docs_only: str, preflight: str, results: tuple[str, str, str], expected: int
) -> None:
    """Run the actual aggregate gate with passing, skipped, and failed jobs."""
    workflow = yaml.safe_load((_WORKFLOWS / "cicd-main.yml").read_text())
    job = workflow["jobs"]["Nemo_Linting_Test"]
    assert "pre-flight" in job["needs"]
    assert job["if"] == "always()"
    result = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", job["steps"][0]["run"]],
        env={
            **os.environ,
            "PRE_FLIGHT_RESULT": preflight,
            "DOCS_ONLY": docs_only,
            "LINTING_RESULT": json.dumps({"result": results[0]}),
            "IMPORT_LINTING_RESULT": json.dumps({"result": results[1]}),
            "TYPE_CHECKING_RESULT": json.dumps({"result": results[2]}),
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == expected, result.stdout + result.stderr
