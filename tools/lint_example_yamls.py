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
from datetime import datetime
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The CI pipeline generator owns the `ci.nemo_ci_schedule` values it reads.
from tests.ci_tests.utils.generate_ci_tests import (  # noqa: E402
    NEMO_CI_SCHEDULE_FOLDERS,
    nemo_ci_schedule_folder,
    nemo_ci_schedule_scopes,
)


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
    Path("examples/retrieval/data_utils/mining_multimodal_config.yaml"),
    # Command-only configs: launched via
    # `python -m nemo_automodel.recipes.speculative_decoding.precompute_dspark_dist -c <config>`
    # (torchrun for multi-node), so they have no recipe-class entry point by design.
    Path("examples/speculative/dspark/deepseek_v4_flash_dspark_precompute.yaml"),
    Path("examples/speculative/dspark/glm_5.2_dspark_precompute.yaml"),
    # A plain YAML list of bench_sweep.py dataset entries, not a recipe config
    # (consumed via --datasets-config, launched with python -m ... bench_sweep).
    Path("examples/speculative/bench_sweep/spec_bench_datasets.yaml"),
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
        if _relative_path(path, automodel_dir) in EXCLUDED_RECIPE_FILES:
            return []
        return [*_lint_ci_cluster_tag(path, {}, []), *_lint_ci_nemo_ci_schedule(path, {}, None, [], automodel_dir)]
    if not isinstance(document, yaml.MappingNode):
        return [LintError(path, document.start_mark.line + 1, "Top-level YAML document must be a mapping.")]
    if not isinstance(data, dict):
        return [LintError(path, document.start_mark.line + 1, "Top-level YAML document must parse to a mapping.")]

    key_lines = _top_level_key_lines(document)
    errors = _lint_top_level_keys(path, key_lines)
    errors.extend(_lint_wandb_disabled(path, data, key_lines))
    if _relative_path(path, automodel_dir) not in EXCLUDED_RECIPE_FILES:
        errors.extend(_lint_ci_cluster_tag(path, data, key_lines))
        errors.extend(_lint_ci_perf_testing(path, data, document, key_lines))
        errors.extend(_lint_ci_nemo_ci_schedule(path, data, document, key_lines, automodel_dir))
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


def _lint_ci_cluster_tag(path: Path, data: dict, key_lines: list[tuple[str, int]]) -> list[LintError]:
    """Require an explicit GPU tag unless the recipe is marked deprecated."""
    ci = data.get("ci")
    if isinstance(ci, dict):
        cluster_tag = ci.get("cluster_tag")
        if ci.get("deprecated") is True or (isinstance(cluster_tag, str) and cluster_tag.strip()):
            return []
    return [
        LintError(
            path,
            dict(key_lines).get("ci", 1),
            "Recipe must declare a nonempty `ci.cluster_tag` (e.g. h100, gb200, gb10) or `ci.deprecated: true`.",
        )
    ]


def _lint_ci_nemo_ci_schedule(
    path: Path,
    data: dict,
    document: yaml.MappingNode | None,
    key_lines: list[tuple[str, int]],
    automodel_dir: Path,
) -> list[LintError]:
    """Require ``ci.nemo_ci_schedule``, the NeMo-CI scopes that run the recipe.

    The value is ``nightly``, ``release``, ``nightly,release``, or ``skipped``. A deprecated
    recipe, or one outside ``NEMO_CI_SCHEDULE_FOLDERS``, runs in neither scope, so it
    must declare ``skipped``.

    Args:
        path: Path of the YAML file being linted.
        data: The parsed YAML mapping.
        document: The composed top-level YAML mapping node, used for line numbers.
        key_lines: ``(key, line)`` pairs for the document's top-level keys.
        automodel_dir: Repository root, used to locate the recipe's test folder.

    Returns:
        A lint error for a missing, invalid, or inapplicable value; empty if the value is valid.
    """
    ci = data.get("ci")
    ci = ci if isinstance(ci, dict) else {}
    line = _mapping_key_lines(_child_node(document, "ci")).get("nemo_ci_schedule", dict(key_lines).get("ci", 1))
    try:
        scopes = nemo_ci_schedule_scopes(ci)
    except ValueError as exc:
        return [LintError(path, line, f"{exc}.")]
    if not scopes:
        return []
    if ci.get("deprecated") is True:
        return [LintError(path, line, "A deprecated recipe runs no tests; set `ci.nemo_ci_schedule: skipped`.")]
    if nemo_ci_schedule_folder(_relative_path(path, automodel_dir)) is None:
        folders = ", ".join(f"examples/{subpath}" for subpath in NEMO_CI_SCHEDULE_FOLDERS.values())
        return [
            LintError(
                path,
                line,
                f"NeMo-CI schedules run only recipes under {folders}; set `ci.nemo_ci_schedule: skipped`.",
            )
        ]
    return []


PERF_TESTING_KEYS = ("schedule", "time", "fake_balanced_gate", "validation")
PERF_TESTING_SCHEDULES = ("nightly", "weekly", "disabled")
CI_TIME_FORMAT = "%H:%M:%S"  # Same format that tests/ci_tests/utils/generate_ci_tests.py parses for `ci.time`.


def _lint_ci_perf_testing(
    path: Path, data: dict, document: yaml.MappingNode, key_lines: list[tuple[str, int]]
) -> list[LintError]:
    """Validate the optional ``ci.perf_testing`` section.

    The section configures automatic performance measurements of the recipe:

    - ``schedule``: required, one of ``PERF_TESTING_SCHEDULES``. ``disabled`` keeps the settings but pauses the runs.
    - ``time``: ``HH:MM:SS`` wall time of the run, parsed like ``ci.time``.
    - ``fake_balanced_gate``: optional boolean; the only performance-specific model override.
    - ``validation``: optional boolean, default false; true preserves the recipe's validation cadence.

    ``ci.deprecated: true`` applies to the section as well: a deprecated recipe runs no tests.

    Args:
        path: Path of the YAML file being linted.
        data: The parsed YAML mapping.
        document: The composed top-level YAML mapping node, used for line numbers and duplicate keys.
        key_lines: ``(key, line)`` pairs for the document's top-level keys.

    Returns:
        Lint errors for every invalid field. Empty if the section is absent or valid.
    """
    ci = data.get("ci")
    if not isinstance(ci, dict) or "perf_testing" not in ci:
        return []
    ci_node = _child_node(document, "ci")
    section_node = _child_node(ci_node, "perf_testing")
    section_line = section_node.start_mark.line + 1 if section_node is not None else dict(key_lines).get("ci", 1)
    # Only `perf_testing` is checked here; duplicates of other `ci` keys are out of scope for this section.
    perf_lines = [key_node.start_mark.line + 1 for key_node, _ in ci_node.value if key_node.value == "perf_testing"]
    errors = [
        LintError(path, line, f"Duplicate key `perf_testing` in `ci`; first defined on line {perf_lines[0]}.")
        for line in perf_lines[1:]
    ]
    section = ci["perf_testing"]
    if not isinstance(section, dict):
        return [*errors, LintError(path, section_line, "`ci.perf_testing` must be a mapping.")]

    section_lines = _mapping_key_lines(section_node)

    def error(message: str, line: int | None = None) -> None:
        errors.append(LintError(path, line or section_line, f"`ci.perf_testing`: {message}"))

    errors.extend(_lint_duplicate_keys(path, section_node, "ci.perf_testing"))
    for key in section:
        if not isinstance(key, str):
            error(f"key `{key}` must be a string.", section_lines.get(str(key)))
        elif key not in PERF_TESTING_KEYS:
            error(f"unknown key `{key}`; allowed keys are {', '.join(PERF_TESTING_KEYS)}.", section_lines.get(key))

    errors.extend(_lint_perf_testing_values(path, section, section_lines, default_line=section_line))
    return errors


def _lint_perf_testing_values(
    path: Path, section: dict[str, object], section_lines: dict[str, int], *, default_line: int
) -> list[LintError]:
    """Validate performance field values after the container and key checks."""
    errors: list[LintError] = []

    def error(message: str, line: int | None = None) -> None:
        errors.append(LintError(path, line or default_line, f"`ci.perf_testing`: {message}"))

    if "schedule" not in section:
        error("`schedule` is required.")
    elif section["schedule"] not in PERF_TESTING_SCHEDULES:
        error(f"`schedule` must be one of {', '.join(PERF_TESTING_SCHEDULES)}.", section_lines.get("schedule"))
    if "time" in section and not _is_ci_time(section["time"]):
        error('`time` must be a "HH:MM:SS" string, the same format as `ci.time`.', section_lines.get("time"))
    for key in ("fake_balanced_gate", "validation"):
        if key in section and not isinstance(section[key], bool):
            error(f"`{key}` must be a boolean.", section_lines.get(key))
    return errors


def _is_ci_time(value: object) -> bool:
    """Return whether ``value`` parses like ``ci.time`` (a ``%H:%M:%S`` string)."""
    if not isinstance(value, str):
        return False
    try:
        datetime.strptime(value, CI_TIME_FORMAT)
    except ValueError:
        return False
    return True


def _child_node(node: yaml.Node | None, key: str) -> yaml.Node | None:
    """Return the value node of the last ``key`` in a mapping node, or None.

    The last occurrence matches the value that ``yaml.safe_load`` keeps for a duplicate key.
    """
    if not isinstance(node, yaml.MappingNode):
        return None
    value = None
    for key_node, value_node in node.value:
        if key_node.value == key:
            value = value_node
    return value


def _mapping_key_lines(node: yaml.Node | None) -> dict[str, int]:
    """Map each key text of a mapping node to the line of its last occurrence, the one ``yaml.safe_load`` keeps."""
    lines: dict[str, int] = {}
    if isinstance(node, yaml.MappingNode):
        for key_node, _value_node in node.value:
            lines[key_node.value] = key_node.start_mark.line + 1
    return lines


def _lint_duplicate_keys(path: Path, node: yaml.Node | None, name: str) -> list[LintError]:
    """Report keys that appear more than once in a mapping node; YAML silently keeps the last one."""
    errors: list[LintError] = []
    seen: dict[str, int] = {}
    if isinstance(node, yaml.MappingNode):
        for key_node, _value_node in node.value:
            line = key_node.start_mark.line + 1
            if key_node.value in seen:
                errors.append(
                    LintError(
                        path,
                        line,
                        f"Duplicate key `{key_node.value}` in `{name}`; first defined on line {seen[key_node.value]}.",
                    )
                )
            seen.setdefault(key_node.value, line)
    return errors


def _lint_wandb_disabled(path: Path, data: dict, key_lines: list[tuple[str, int]]) -> list[LintError]:
    """Require an example ``wandb:`` block to set ``enable: false``.

    Weights & Biases logging is opt-in for examples: a present ``wandb:`` block
    logs by default, which fails or noisily logs for anyone who runs the example
    without W&B credentials. Example configs ship the block with ``enable: false``
    so users opt in by flipping it to ``true`` (no need to comment/uncomment the
    block). A config with no ``wandb:`` block is fine.

    Args:
        path: Path of the YAML file being linted.
        data: The parsed YAML mapping.
        key_lines: ``(key, line)`` pairs for the document's top-level keys.

    Returns:
        A lint error if a top-level ``wandb`` block is present without
        ``enable: false``.
    """
    if "wandb" not in data:
        return []
    wandb_line = dict(key_lines).get("wandb", 1)
    block = data.get("wandb")
    enable = block.get("enable") if isinstance(block, dict) else None
    if enable is False:
        return []
    return [
        LintError(
            path,
            wandb_line,
            "Example `wandb:` block must set `enable: false` (W&B logging is opt-in); "
            "add `enable: false` so users opt in by flipping it to `true`.",
        )
    ]


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
    """Run the example YAML linter."""
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
