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

#!/usr/bin/env python3
"""Validate that a newly added CI recipe declares a complete ``ci:`` section.

A recipe YAML placed under a CI-enabled ``examples/`` folder is run by the
nightly/release/performance pipelines. Auto-discovery scopes (see
``AUTO_DISCOVER_SCOPES`` in ``generate_ci_tests.py``) enroll *every* YAML under
their folder simply by its presence on disk, so a new file silently becomes a
CI test. This check enforces, on newly added recipes only, that the recipe
declares who owns it, how long it should run, and how many nodes it needs -- so
the generated job is accountable and correctly sized rather than falling back to
opaque defaults.

The set of CI-enabled recipes is derived entirely from configuration already in
this repo (no nemo-ci dependency): the auto-discover subpaths plus every recipe
referenced by a ``tests/ci_tests/configs/<folder>/*_recipes.yml`` list.

Existing recipes are grandfathered: the caller passes only the files added in
the change (``git diff --diff-filter=A``), so this never retroactively fails
recipes that predate the check.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml
from ruamel.yaml import YAML

REPO_ROOT = Path(__file__).resolve().parents[3]
UTILS_DIR = Path(__file__).resolve().parent
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from generate_ci_tests import AUTO_DISCOVER_SCOPES, resolve_recipe_path  # noqa: E402

# ci: keys every newly added CI recipe must declare.
REQUIRED_CI_FIELDS = ("recipe_owner", "time", "nodes")

_ruamel = YAML()


@dataclass(frozen=True)
class CiError:
    """A missing/invalid ``ci:`` declaration with source location."""

    path: Path
    line: int
    message: str


def auto_discover_subpaths() -> set[str]:
    """Return the examples/ subpaths auto-enrolled in CI (e.g. ``diffusion/finetune``)."""
    return {subpath for scope in AUTO_DISCOVER_SCOPES.values() for subpath in scope.values()}


def listed_recipe_paths(automodel_dir: Path) -> set[Path]:
    """Return every ``examples/...`` recipe referenced by a ``*_recipes.yml`` list.

    Override files (``override_recipes.yml``) carry no ``configs:`` list and are
    skipped implicitly.
    """
    configs_root = automodel_dir / "tests" / "ci_tests" / "configs"
    paths: set[Path] = set()
    for recipe_list in sorted(configs_root.glob("*/*_recipes.yml")):
        with recipe_list.open("r", encoding="utf-8") as f:
            data = _ruamel.load(f) or {}
        configs = data.get("configs")
        if not configs:
            continue
        examples_dir = data.get("examples_dir", recipe_list.parent.name)
        for config in configs:
            paths.add(resolve_recipe_path(config, examples_dir))
    return paths


def is_ci_enabled(rel_path: Path, subpaths: set[str], listed: set[Path]) -> bool:
    """Whether ``rel_path`` (relative to the repo root) is a recipe CI will run."""
    if rel_path in listed:
        return True
    return any(_is_relative_to(rel_path, Path("examples") / subpath) for subpath in subpaths)


def validate_recipe(path: Path, automodel_dir: Path) -> list[CiError]:
    """Return CI-section errors for one added recipe YAML."""
    rel_path = _relative_path(path, automodel_dir)
    text = path.read_text(encoding="utf-8")
    try:
        document = yaml.compose(text)
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        return [CiError(path, _yaml_error_line(exc), f"YAML syntax error: {exc}")]

    if not isinstance(data, dict) or "ci" not in data:
        return [
            CiError(
                path,
                1,
                f"Recipe `{rel_path}` runs in CI but has no top-level `ci:` section. "
                f"Add a `ci:` block declaring: {', '.join(REQUIRED_CI_FIELDS)}.",
            )
        ]

    ci_block = data.get("ci")
    ci_line = _ci_key_line(document)
    if not isinstance(ci_block, dict):
        return [CiError(path, ci_line, "Top-level `ci:` section must be a mapping.")]

    errors: list[CiError] = []
    for field in REQUIRED_CI_FIELDS:
        value = ci_block.get(field)
        if value is None or (isinstance(value, str) and not value.strip()):
            errors.append(
                CiError(
                    path,
                    ci_line,
                    f"Recipe `{rel_path}` is missing required `ci.{field}`; "
                    f"a CI recipe must declare its owner, run time, and node count.",
                )
            )
    return errors


def format_errors(errors: list[CiError], automodel_dir: Path) -> str:
    """Format CI-section errors for CLI output."""
    lines = ["New recipe `ci:` section validation failed:"]
    for error in errors:
        path = _relative_path(error.path, automodel_dir)
        lines.append(f"  {path}:{error.line}: {error.message}")
    lines.append(
        "\nEvery recipe added under a CI-enabled examples/ folder must declare a `ci:` "
        "section with `recipe_owner`, `time`, and `nodes`. See tests/ci_tests/README.md."
    )
    return "\n".join(lines)


def _ci_key_line(document) -> int:
    if isinstance(document, yaml.MappingNode):
        for key_node, _value_node in document.value:
            if getattr(key_node, "value", None) == "ci":
                return key_node.start_mark.line + 1
    return 1


def _is_relative_to(path: Path, prefix: Path) -> bool:
    try:
        path.relative_to(prefix)
    except ValueError:
        return False
    return True


def _relative_path(path: Path, automodel_dir: Path) -> Path:
    try:
        return path.resolve().relative_to(automodel_dir.resolve())
    except ValueError:
        return path


def _yaml_error_line(exc: yaml.YAMLError) -> int:
    mark = getattr(exc, "problem_mark", None) or getattr(exc, "context_mark", None)
    return mark.line + 1 if mark is not None else 1


def main(argv: list[str] | None = None) -> int:
    """Validate the ``ci:`` section on newly added recipe YAMLs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="Added YAML files to validate (typically from git diff).")
    parser.add_argument("--automodel-dir", type=Path, default=Path.cwd(), help="Path to the AutoModel repository root.")
    args = parser.parse_args(argv)

    automodel_dir = args.automodel_dir.resolve()
    subpaths = auto_discover_subpaths()
    listed = listed_recipe_paths(automodel_dir)

    errors: list[CiError] = []
    checked = 0
    for path in args.paths:
        if path.suffix not in {".yaml", ".yml"}:
            continue
        abs_path = path if path.is_absolute() else (automodel_dir / path)
        rel_path = _relative_path(abs_path, automodel_dir)
        if not is_ci_enabled(rel_path, subpaths, listed):
            continue
        if not abs_path.is_file():
            continue
        checked += 1
        errors.extend(validate_recipe(abs_path, automodel_dir))

    if errors:
        print(format_errors(errors, automodel_dir), file=sys.stderr)
        return 1

    print(f"Validated `ci:` section on {checked} newly added CI recipe(s).", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
