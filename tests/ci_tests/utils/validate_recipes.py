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
"""Validate that every recipe path in tests/ci_tests/configs/<folder>/nightly_recipes.yml exists under examples/."""

import argparse
import os
import sys
from pathlib import Path

from ruamel.yaml import YAML

yaml = YAML()


SUMMARY_HEADER = "## Nightly recipe validation failed"

SUMMARY_INTRO = (
    "One or more recipes listed in `nightly_recipes.yml` do not exist on disk.\n"
    "AutoModel's nightly CI generator skips missing recipes at runtime, but a "
    "broken reference still means the listed test never runs. Fix it before merge."
)

SUMMARY_HOW_TO_FIX = """### How to fix

For each missing entry above, choose one:

1. **Add the recipe** — create the YAML at the path shown. Typical when a recipe was renamed or moved without updating `nightly_recipes.yml`.
2. **Remove from nightly** — delete the offending line from `tests/ci_tests/configs/<folder>/nightly_recipes.yml` if the recipe is genuinely gone.
3. **Mark as known issue** — add `ci.known_issue_id: AM-xxx` (or `ci.allow_failure: true`) to the recipe YAML. See `tests/ci_tests/README.md`.

Inline annotations are also attached to the offending lines in the **Files Changed** tab."""

STDERR_HEADER = "Missing nightly recipe references:"

STDERR_HOW_TO_FIX = """
------------------------------------------------------------
How to fix - for each missing entry above, choose one:

  1. Add the recipe at the path shown (typical when a recipe was
     renamed or moved without updating nightly_recipes.yml).

  2. Remove the line from
     tests/ci_tests/configs/<folder>/nightly_recipes.yml
     if the recipe is genuinely gone.

  3. Mark as known issue: add ci.known_issue_id: AM-xxx
     (or ci.allow_failure: true) to the recipe YAML.
     See tests/ci_tests/README.md.
------------------------------------------------------------"""

ANNOTATION_MSG_TEMPLATE = (
    "Recipe '{config}' not found at {rel_path}. "
    "Add the recipe, remove this line, or set ci.known_issue_id on the recipe."
)


def collect_nightly_lists(automodel_dir: Path):
    """Yield (recipe_list_path, examples_dir, [(config, line_number)]) for each nightly_recipes.yml."""
    configs_root = automodel_dir / "tests" / "ci_tests" / "configs"
    for recipe_list in sorted(configs_root.glob("*/nightly_recipes.yml")):
        with recipe_list.open("r", encoding="utf-8") as f:
            data = yaml.load(f) or {}
        configs = data.get("configs") or []
        examples_dir = data.get("examples_dir", recipe_list.parent.name)
        # ruamel.yaml round-trip mode tracks line/col per sequence entry in .lc.data.
        lc = getattr(configs, "lc", None)
        entries = []
        for i, config in enumerate(configs):
            line = None
            if lc is not None and lc.data and i in lc.data:
                line = lc.data[i][0] + 1
            entries.append((config, line))
        yield recipe_list, examples_dir, entries


def _gh_escape(value: str) -> str:
    return value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def _write_stderr_report(missing_by_list, automodel_dir):
    print(STDERR_HEADER, file=sys.stderr)
    for recipe_list, items in missing_by_list.items():
        rel_list = recipe_list.relative_to(automodel_dir)
        print(f"\n  {rel_list}:", file=sys.stderr)
        for config, rel_path, line in items:
            loc = f"line {line}" if line else "line ?"
            print(f"    - [{loc}] {config} -> {rel_path} (not found)", file=sys.stderr)
    print(STDERR_HOW_TO_FIX, file=sys.stderr)


def _emit_gh_annotations(missing_by_list, automodel_dir):
    for recipe_list, items in missing_by_list.items():
        rel_list = recipe_list.relative_to(automodel_dir)
        for config, rel_path, line in items:
            msg = _gh_escape(ANNOTATION_MSG_TEMPLATE.format(config=config, rel_path=rel_path))
            line_part = f",line={line}" if line else ""
            print(f"::error file={rel_list}{line_part}::{msg}")


def _write_step_summary(summary_path, missing_by_list, automodel_dir):
    sections = [SUMMARY_HEADER, "", SUMMARY_INTRO, "", "### Missing entries", ""]
    for recipe_list, items in missing_by_list.items():
        rel_list = recipe_list.relative_to(automodel_dir)
        sections.append(f"**`{rel_list}`**")
        sections.append("")
        for config, rel_path, line in items:
            loc = f"L{line}" if line else "L?"
            sections.append(f"- {loc}: `{config}` → `{rel_path}` (not found)")
        sections.append("")
    sections.append(SUMMARY_HOW_TO_FIX)
    sections.append("")
    with open(summary_path, "a", encoding="utf-8") as f:
        f.write("\n".join(sections))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--automodel-dir", type=str, required=True, help="Path to Automodel directory")
    args = parser.parse_args()

    automodel_dir = Path(args.automodel_dir).resolve()
    missing_by_list: dict[Path, list[tuple[str, Path, int | None]]] = {}

    for recipe_list, examples_dir, entries in collect_nightly_lists(automodel_dir):
        for config, line in entries:
            rel_path = Path("examples") / examples_dir / config
            if not (automodel_dir / rel_path).is_file():
                missing_by_list.setdefault(recipe_list, []).append((config, rel_path, line))

    if not missing_by_list:
        print("All nightly recipe references valid.")
        return 0

    _write_stderr_report(missing_by_list, automodel_dir)

    if os.environ.get("GITHUB_ACTIONS") == "true":
        _emit_gh_annotations(missing_by_list, automodel_dir)

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        _write_step_summary(summary_path, missing_by_list, automodel_dir)

    return 1


if __name__ == "__main__":
    sys.exit(main())
