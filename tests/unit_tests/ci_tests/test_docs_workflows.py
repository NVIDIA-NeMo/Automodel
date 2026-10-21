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


def test_preview_generates_tables_without_publishing_credentials() -> None:
    workflow = yaml.safe_load((_WORKFLOWS / "fern-docs-preview.yml").read_text())
    generation = workflow["jobs"]["generate-tables"]
    publishing = workflow["jobs"]["preview"]
    assert generation["permissions"] == {"contents": "read"}
    assert "secrets." not in yaml.safe_dump(generation)
    checkout = next(step for step in generation["steps"] if step.get("uses", "").startswith("actions/checkout@"))
    assert checkout["with"]["persist-credentials"] is False
    assert checkout["with"]["ref"] == "${{ github.sha }}"
    assert "generate-tables" in publishing["needs"]
    assert "sync_model_coverage_tables.py" not in yaml.safe_dump(publishing)
    upload = next(step for step in generation["steps"] if step["name"] == "Upload generated Markdown tables")
    download = next(step for step in publishing["steps"] if step["name"] == "Download generated Markdown tables")
    assert upload["with"]["name"] == download["with"]["name"] == "model-coverage-${{ github.sha }}"


@pytest.mark.parametrize("invalid", [None, "missing", "artifact-symlink", "docs-symlink"])
def test_preview_restores_only_generated_markdown(tmp_path: Path, invalid: str | None) -> None:
    workflow = yaml.safe_load((_WORKFLOWS / "fern-docs-preview.yml").read_text())
    upload = next(
        step
        for step in workflow["jobs"]["generate-tables"]["steps"]
        if step["name"] == "Upload generated Markdown tables"
    )
    restore = next(
        step for step in workflow["jobs"]["preview"]["steps"] if step["name"] == "Restore generated Markdown tables"
    )
    artifact = tmp_path / "runner/model-coverage"
    docs = tmp_path / "pr-source/docs"
    pages = [Path(path).relative_to("docs") for path in upload["with"]["path"].splitlines()]
    for page in pages:
        assert page.suffix == ".mdx" and ".." not in page.parts
        for root, content in ((artifact, "generated table"), (docs, "original table")):
            destination = root / page
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(content)
    (artifact / "injected.py").write_text("raise RuntimeError('must not execute')")
    if invalid == "missing":
        (artifact / pages[0]).unlink()
    elif invalid in ("artifact-symlink", "docs-symlink"):
        root = artifact if invalid == "artifact-symlink" else docs
        (root / "symlink.mdx").symlink_to(artifact / pages[0])

    result = subprocess.run(
        ["bash", "-e", "-c", restore["run"]],
        cwd=tmp_path,
        env={**os.environ, "RUNNER_TEMP": str(tmp_path / "runner")},
        capture_output=True,
        text=True,
    )

    if invalid:
        assert result.returncode != 0
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        assert all((docs / page).read_text() == "generated table" for page in pages)
    assert not (docs / "injected.py").exists()


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


def test_preview_drops_legacy_redirects_that_shadow_new_pages(tmp_path: Path) -> None:
    preview = tmp_path / "runner/fern-preview/docs"
    destination = preview / "fern/versions/nightly.yml"
    destination.parent.mkdir(parents=True)
    navigation = "navigation:\n  - page: BF16\n    path: ../../model.mdx\n"
    destination.write_text(navigation)
    (preview / "model.mdx").write_text("---\nslug: model-coverage/org/Model-BF16\n---\nBF16 page\n")
    stale = {"source": "/nemo/automodel/nightly/model-coverage/org/Model-BF16", "destination": "/old"}
    retained = {"source": "/nemo/automodel/nightly/old-guide", "destination": "/guide"}
    trusted = {"title": "Trusted", "js": ["./trusted.js"], "redirects": [stale, retained]}
    (preview / "fern/docs.yml").write_text(yaml.safe_dump(trusted))
    source = tmp_path / "pr-source/docs/fern/versions/nightly.yml"
    source.parent.mkdir(parents=True)
    source.write_text(navigation)
    (source.parent.parent / "docs.yml").write_text("js: ['https://untrusted.example/code.js']\n")

    result = _run_pr_navigation(tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    assert yaml.safe_load((preview / "fern/docs.yml").read_text()) == {**trusted, "redirects": [retained]}


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


@pytest.mark.parametrize("invalid_slug", [False, True])
def test_preview_navigation_uses_pr_cards_instead_of_retained_fallbacks(tmp_path: Path, invalid_slug: bool) -> None:
    """Renames and deletions affect navigation even while old pages remain staged."""
    categories = {
        "llm": "large-language-models",
        "vlm": "vision-language-models",
        "multimodal": "multimodal",
        "omni": "omni",
        "dllm": "dllm",
        "diffusion": "diffusion",
        "embedding": "embedding-models",
        "reranker": "reranking-models",
    }
    provider = {
        "section": "Trusted Provider",
        "slug": "provider",
        "icon": "trusted-icon",
        "path": "../../model-coverage/llm/provider/index.mdx",
        "contents": [
            {"page": "Model-V1", "path": "../../model-coverage/llm/provider/old-name.mdx"},
            {"page": "Deleted", "path": "../../model-coverage/llm/provider/deleted.mdx"},
        ],
    }
    navigation = [
        {"page": "Guide", "path": "../../guides/guide.mdx"},
        {
            "section": "Model Coverage",
            "contents": [
                {"section": directory, "slug": slug, "contents": [provider] if directory == "llm" else []}
                for directory, slug in categories.items()
            ],
        },
    ]
    route = "/nemo/automodel/nightly/model-coverage/large-language-models/provider/Model-V1"
    unrelated_redirect = {"source": "/old-guide", "destination": "/guide"}
    config = {"title": "Trusted Site", "redirects": [{"source": route, "destination": "/retired"}, unrelated_redirect]}
    trusted = {
        "fern/versions/nightly.yml": yaml.safe_dump({"navigation": navigation}),
        "fern/docs.yml": yaml.safe_dump(config),
        "guides/guide.mdx": "Trusted guide",
        "model-coverage/llm/provider/index.mdx": "Trusted provider index",
        # The old filename intentionally disagrees with its canonical slug.
        "model-coverage/llm/provider/old-name.mdx": "---\nslug: model-coverage/large-language-models/provider/Model-V1\n---\nOld card",
        "model-coverage/llm/provider/deleted.mdx": "---\nslug: model-coverage/large-language-models/provider/deleted\n---\nDeleted card",
    }
    slug = "wrong-route" if invalid_slug else route.removeprefix("/nemo/automodel/nightly/")
    proposed = {
        "fern/versions/nightly.yml": "PR navigation must not be used",
        "fern/docs.yml": "PR configuration must not be used",
        "model-coverage/llm/provider/index.mdx": "Updated provider index",
        "model-coverage/llm/provider/Model-V1.mdx": f"---\nslug: {slug}\n---\nRenamed card",
    }
    for source, files in (("trusted-source", trusted), ("pr-source", proposed)):
        for name, content in files.items():
            path = tmp_path / source / "docs" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)

    workflow = yaml.safe_load((_WORKFLOWS / "fern-docs-preview.yml").read_text())
    steps = {step["name"]: step for step in workflow["jobs"]["preview"]["steps"]}
    env = {**os.environ, "RUNNER_TEMP": str(tmp_path / "runner")}
    subprocess.run(
        ["bash", "-e", "-c", steps["Prepare preview docs"]["run"]],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    result = subprocess.run(
        ["bash", "-e", "-c", steps["Resolve model card paths in trusted navigation"]["run"]],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    if invalid_slug:
        assert result.returncode != 0
        assert "Model card filename must match its public checkpoint URL" in result.stderr
        assert "Model-V1.mdx" in result.stderr
        return
    assert result.returncode == 0, result.stderr

    preview = tmp_path / "runner/fern-preview/docs"
    assert (preview / "model-coverage/llm/provider/old-name.mdx").is_file()
    assert (preview / "model-coverage/llm/provider/deleted.mdx").is_file()
    actual = yaml.safe_load((preview / "fern/versions/nightly.yml").read_text())["navigation"]
    assert actual[0] == navigation[0]
    actual_provider = actual[1]["contents"][0]["contents"][0]
    assert actual_provider == {
        **provider,
        "contents": [{"page": "Model-V1", "path": "../../model-coverage/llm/provider/Model-V1.mdx"}],
    }
    assert yaml.safe_load((preview / "fern/docs.yml").read_text()) == {
        **config,
        "redirects": [unrelated_redirect],
    }


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
