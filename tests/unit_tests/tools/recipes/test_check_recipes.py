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

"""Exercise the shared CLI with real validators and disposable Git history."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

SOURCE_ROOT = Path(__file__).resolve().parents[4]
VALID_RECIPE = "ci:\n  cluster_tag: h100\n  recipe_owner: test\n  time: '00:10:00'\n  nodes: 1\n"


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        [
            "git",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "user.name=Recipe Test",
            "-c",
            "user.email=recipe-test@example.com",
            *args,
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    root = tmp_path / "checkout"
    for name in (
        "tools/recipes/check_recipes.py",
        "tools/lint_example_yamls.py",
        "tests/ci_tests/utils/validate_nightly_recipes.py",
        "tests/ci_tests/utils/validate_generate_ci.py",
        "tests/ci_tests/utils/validate_new_recipe_ci.py",
        "tests/ci_tests/utils/generate_ci_tests.py",
    ):
        dest = root / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE_ROOT / name, dest)
    for folder in ("llm_finetune", "vlm_finetune", "diffusion_finetune", "llm_benchmark", "vlm_benchmark"):
        path = root / "tests/ci_tests/configs" / folder / "override_recipes.yml"
        path.parent.mkdir(parents=True)
        path.write_text("{}\n")
    recipes = root / "examples/diffusion/finetune/model"
    recipes.mkdir(parents=True)
    (recipes / "existing.yaml").write_text("ci:\n  cluster_tag: h100\n")
    _git(root, "init", "-q")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "baseline")
    _git(root, "update-ref", "refs/remotes/origin/main", "HEAD")
    return root


def _check(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(root / "tools/recipes/check_recipes.py"), *args],
        # The entry point must work even when invoked from outside its checkout.
        cwd=root.parent,
        capture_output=True,
        text=True,
    )


def test_existing_recipe_is_grandfathered(checkout: Path) -> None:
    result = _check(checkout)
    assert result.returncode == 0, result.stderr
    assert "All recipe checks passed" in result.stderr
    assert "0 newly added CI recipe(s)" in result.stderr


@pytest.mark.parametrize("state", ["untracked", "staged", "committed"])
def test_added_recipe_metadata_and_paths_with_spaces(checkout: Path, state: str) -> None:
    recipe = checkout / "examples/diffusion/finetune/model/new 'recipe.yaml"
    recipe.write_text("ci:\n  cluster_tag: h100\n")
    if state != "untracked":
        _git(checkout, "add", ".")
    if state == "committed":
        _git(checkout, "commit", "-qm", "new recipe")
    result = _check(checkout)
    assert result.returncode == 1
    assert "missing required `ci.recipe_owner`" in result.stderr
    assert "new 'recipe.yaml" in result.stderr
    recipe.write_text(VALID_RECIPE)
    result = _check(checkout)
    assert result.returncode == 0, result.stderr
    assert "1 newly added CI recipe(s)" in result.stderr


def test_explicit_pr_range_excludes_untracked_metadata(checkout: Path) -> None:
    recipe = checkout / "examples/diffusion/finetune/model/new.yaml"
    recipe.write_text(VALID_RECIPE)
    _git(checkout, "add", ".")
    _git(checkout, "commit", "-qm", "valid addition")
    (recipe.parent / "untracked.yaml").write_text("ci:\n  cluster_tag: h100\n")
    result = _check(checkout, "--base-ref", "origin/main", "--head-ref", "HEAD")
    assert result.returncode == 0, result.stderr
    assert "1 newly added CI recipe(s)" in result.stderr
    assert _check(checkout).returncode == 1


def test_invalid_base_fails_instead_of_skipping_metadata(checkout: Path) -> None:
    result = _check(checkout, "--base-ref", "missing-ref")
    assert result.returncode != 0
    assert "Cannot determine added recipes" in result.stderr
    assert "All recipe checks passed" not in result.stderr


@pytest.mark.parametrize(
    ("failure", "diagnostic"),
    [
        ("yaml", "YAML syntax error"),
        ("nightly", "Missing nightly recipe references"),
        ("generator", "combos failed"),
    ],
)
def test_validator_failures_reach_cli(checkout: Path, failure: str, diagnostic: str) -> None:
    config = checkout / "tests/ci_tests/configs/diffusion_finetune"
    if failure == "yaml":
        (checkout / "examples/diffusion/finetune/model/existing.yaml").write_text("ci: [\n")
    elif failure == "nightly":
        (config / "nightly_recipes.yml").write_text("examples_dir: diffusion/finetune\nconfigs: [model/missing.yaml]\n")
    else:
        (config / "override_recipes.yml").write_text("[not-a-mapping]\n")
    result = _check(checkout)
    assert result.returncode == 1
    assert diagnostic in result.stderr
    assert "All recipe checks passed" not in result.stderr


def test_merge_base_does_not_treat_diverged_main_recipe_as_added(checkout: Path) -> None:
    base = _git(checkout, "rev-parse", "HEAD")
    _git(checkout, "checkout", "-qb", "main-advance")
    existing = checkout / "examples/diffusion/finetune/model/existing.yaml"
    existing.unlink()
    _git(checkout, "add", ".")
    _git(checkout, "commit", "-qm", "remove old recipe upstream")
    _git(checkout, "update-ref", "refs/remotes/origin/main", "HEAD")
    _git(checkout, "checkout", "--detach", base)
    result = _check(checkout)
    assert result.returncode == 0, result.stderr
    assert "0 newly added CI recipe(s)" in result.stderr


def test_ci_calls_shared_checker_with_pr_refs() -> None:
    workflow = yaml.safe_load((SOURCE_ROOT / ".github/workflows/validate-recipe-configs.yml").read_text())
    paths = workflow[True]["pull_request"]["paths"]
    assert "tools/recipes/**" in paths
    assert "tests/unit_tests/tools/recipes/**" in paths
    steps = workflow["jobs"]["validate-nightly-recipes"]["steps"]
    check = next(step for step in steps if step.get("name") == "Check recipes")
    assert check["run"] == (
        'uv run --script tools/recipes/check_recipes.py --base-ref "$BASE_SHA" --head-ref "$HEAD_SHA"'
    )
    assert check["env"] == {
        "BASE_SHA": "${{ github.event.pull_request.base.sha }}",
        "HEAD_SHA": "${{ github.event.pull_request.head.sha }}",
    }
    assert any("tests/unit_tests/tools/recipes" in step.get("run", "") for step in steps)
