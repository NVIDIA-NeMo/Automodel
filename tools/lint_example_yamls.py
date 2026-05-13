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

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True)
class LintError:
    """A YAML lint failure with source location."""

    path: Path
    line: int
    message: str


EXCLUDED_RECIPE_PREFIXES = (
    Path("examples/diffusion"),
    Path("examples/convergence"),
)

EXCLUDED_RECIPE_FILES = {
    Path("examples/retrieval/data_utils/mining_config.yaml"),
}

RECIPE_TARGET_HELP = (
    "Add one of: `recipe: TrainFinetuneRecipeForNextTokenPrediction`, "
    "`recipe: nemo_automodel.recipes.llm.train_ft.TrainFinetuneRecipeForNextTokenPrediction`, "
    "or `recipe: {_target_: nemo_automodel.recipes.llm.train_ft.TrainFinetuneRecipeForNextTokenPrediction}`."
)


def collect_example_yamls(automodel_dir: Path) -> list[Path]:
    """Return example YAML files linted by default."""
    examples_dir = automodel_dir / "examples"
    return sorted([*examples_dir.rglob("*.yaml"), *examples_dir.rglob("*.yml")])


def lint_yaml_file(path: Path, automodel_dir: Path) -> list[LintError]:
    """Lint one YAML file from ``examples/``."""
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        return [LintError(path, 1, f"YAML file must be UTF-8 encoded: {exc}")]

    return lint_yaml_text(text, path, automodel_dir)


def lint_yaml_text(text: str, path: Path, automodel_dir: Path) -> list[LintError]:
    """Lint YAML text as if it came from ``path``."""
    try:
        document = yaml.compose(text)
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        line = _yaml_error_line(exc)
        return [LintError(path, line, f"YAML syntax error: {exc}")]

    if document is None:
        return []
    if not isinstance(document, yaml.MappingNode):
        return [LintError(path, document.start_mark.line + 1, "Top-level YAML document must be a mapping.")]
    if not isinstance(data, dict):
        return [LintError(path, document.start_mark.line + 1, "Top-level YAML document must parse to a mapping.")]

    key_lines = _top_level_key_lines(document)
    errors = _lint_top_level_keys(path, key_lines)
    if _should_require_recipe(path, automodel_dir):
        errors.extend(_lint_recipe_target(path, data, key_lines, automodel_dir))
    return errors


def format_errors(errors: list[LintError], automodel_dir: Path) -> str:
    """Format lint errors for CLI output."""
    lines = ["YAML lint failures:"]
    for error in errors:
        path = _relative_path(error.path, automodel_dir)
        lines.append(f"  {path}:{error.line}: {error.message}")
    return "\n".join(lines)


def _lint_top_level_keys(path: Path, key_lines: list[tuple[str, int]]) -> list[LintError]:
    errors: list[LintError] = []
    keys = [key for key, _line in key_lines]
    seen: dict[str, int] = {}
    for key, line in key_lines:
        first_line = seen.get(key)
        if first_line is not None:
            errors.append(
                LintError(path, line, f"Duplicate top-level key `{key}`; first defined on line {first_line}.")
            )
        seen[key] = line

    if "recipe" in keys and keys[0] != "recipe":
        recipe_line = dict(key_lines)["recipe"]
        errors.append(LintError(path, recipe_line, "Top-level `recipe` section must be the first section."))

    if "ci" in keys and keys[-1] != "ci":
        ci_line = dict(key_lines)["ci"]
        errors.append(LintError(path, ci_line, "Top-level `ci` section must be the last section."))

    return errors


def _lint_recipe_target(
    path: Path, data: dict, key_lines: list[tuple[str, int]], automodel_dir: Path
) -> list[LintError]:
    from nemo_automodel.cli.app import resolve_recipe_name

    recipe_line = dict(key_lines).get("recipe", 1)
    raw = _get_recipe_target(data)
    if raw is None:
        return [LintError(path, recipe_line, f"Missing recipe target. {RECIPE_TARGET_HELP}")]

    try:
        fqn = resolve_recipe_name(raw)
    except ValueError as exc:
        return [LintError(path, recipe_line, str(exc))]

    try:
        source_file, class_name = _fqn_to_source_file(fqn, automodel_dir)
    except ValueError as exc:
        return [LintError(path, recipe_line, str(exc))]

    if not source_file.is_file():
        rel_source = _relative_path(source_file, automodel_dir)
        return [LintError(path, recipe_line, f"Recipe module not found: {rel_source} (from target `{fqn}`).")]

    source = source_file.read_text(encoding="utf-8")
    if not re.search(rf"^class\s+{re.escape(class_name)}\b", source, re.MULTILINE):
        rel_source = _relative_path(source_file, automodel_dir)
        return [LintError(path, recipe_line, f"Recipe class `{class_name}` not found in {rel_source}.")]

    return []


def _get_recipe_target(data: dict) -> str | None:
    recipe = data.get("recipe")
    if isinstance(recipe, str) and recipe.strip():
        return recipe.strip()
    if isinstance(recipe, dict):
        target = recipe.get("_target_")
        if isinstance(target, str) and target.strip():
            return target.strip()
    return None


def _fqn_to_source_file(fqn: str, automodel_dir: Path) -> tuple[Path, str]:
    parts = fqn.rsplit(".", 1)
    if len(parts) != 2:
        raise ValueError(f"Expected recipe target in `module.ClassName` form, got: `{fqn}`.")
    module_dotted, class_name = parts
    module_rel = Path(*module_dotted.split(".")).with_suffix(".py")
    return automodel_dir / module_rel, class_name


def _top_level_key_lines(document: yaml.MappingNode) -> list[tuple[str, int]]:
    key_lines: list[tuple[str, int]] = []
    for key_node, _value_node in document.value:
        if isinstance(key_node.value, str):
            key_lines.append((key_node.value, key_node.start_mark.line + 1))
    return key_lines


def _should_require_recipe(path: Path, automodel_dir: Path) -> bool:
    rel_path = _relative_path(path, automodel_dir)
    if rel_path in EXCLUDED_RECIPE_FILES:
        return False
    return not any(_is_relative_to(rel_path, prefix) for prefix in EXCLUDED_RECIPE_PREFIXES)


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
    if mark is None:
        return 1
    return mark.line + 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Lint YAML files under examples/.")
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Specific YAML files to lint. Defaults to every YAML file under examples/.",
    )
    parser.add_argument("--automodel-dir", type=Path, default=Path.cwd(), help="Path to the AutoModel repository root.")
    args = parser.parse_args(argv)

    automodel_dir = args.automodel_dir.resolve()
    paths = [path.resolve() for path in args.paths] if args.paths else collect_example_yamls(automodel_dir)
    errors: list[LintError] = []
    for path in paths:
        if path.suffix not in {".yaml", ".yml"}:
            continue
        errors.extend(lint_yaml_file(path, automodel_dir))

    if errors:
        print(format_errors(errors, automodel_dir), file=sys.stderr)
        return 1

    print(f"Linted {len(paths)} YAML file(s).", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
