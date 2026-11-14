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
"""Validate the recipe lists in tests/ci_tests/configs/<folder>/<scope>_recipes.yml.

Only RECIPE_LIST_SCOPES in generate_ci_tests.py read a recipe list. The nightly and
release scopes select recipes by their ``ci.nemo_ci_schedule`` value, so a list for any
other scope would be silently ignored. Every entry in a list that is read must
exist under examples/.
"""

import argparse
import sys
from pathlib import Path

from ruamel.yaml import YAML

# Allow `from generate_ci_tests import ...` when this script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_ci_tests import RECIPE_LIST_SCOPES  # noqa: E402

yaml = YAML()


MISSING_HEADER = "Missing recipe list references:"

MISSING_HOW_TO_FIX = """
------------------------------------------------------------
How to fix - for each missing entry above, choose one:

  1. Add the recipe at the path shown (typical when a recipe was
     renamed or moved without updating the recipe list).

  2. Remove the line from the recipe list if the recipe is
     genuinely gone.

See tests/ci_tests/README.md for details.
------------------------------------------------------------"""

UNREAD_HEADER = "Recipe lists that no CI scope reads:"

UNREAD_HOW_TO_FIX = f"""
------------------------------------------------------------
Only these scopes read a recipe list: {", ".join(RECIPE_LIST_SCOPES)}.
The nightly and release scopes run the recipes whose `ci.nemo_ci_schedule`
selects them. Delete each list above and set `ci.nemo_ci_schedule` in the
recipes instead. See tests/ci_tests/README.md for details.
------------------------------------------------------------"""


def collect_recipe_lists(automodel_dir: Path):
    """Yield (recipe_list_path, scope, examples_dir, [(config, line_number)]) for each recipe list."""
    configs_root = automodel_dir / "tests" / "ci_tests" / "configs"
    for recipe_list in sorted(configs_root.glob("*/*_recipes.yml")):
        # override_recipes.yml holds exemptions, not a scope's recipes.
        if recipe_list.name == "override_recipes.yml":
            continue
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
        yield recipe_list, recipe_list.stem.removesuffix("_recipes"), examples_dir, entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--automodel-dir", type=str, required=True, help="Path to Automodel directory")
    args = parser.parse_args()

    automodel_dir = Path(args.automodel_dir).resolve()
    unread_lists: list[Path] = []
    missing_by_list: dict[Path, list[tuple[str, Path, int | None]]] = {}

    for recipe_list, scope, examples_dir, entries in collect_recipe_lists(automodel_dir):
        if scope not in RECIPE_LIST_SCOPES:
            unread_lists.append(recipe_list)
            continue
        for config, line in entries:
            rel_path = Path("examples") / examples_dir / config
            if not (automodel_dir / rel_path).is_file():
                missing_by_list.setdefault(recipe_list, []).append((config, rel_path, line))

    if unread_lists:
        print(UNREAD_HEADER, file=sys.stderr)
        for recipe_list in unread_lists:
            print(f"  {recipe_list.relative_to(automodel_dir)}", file=sys.stderr)
        print(UNREAD_HOW_TO_FIX, file=sys.stderr)

    if missing_by_list:
        print(MISSING_HEADER, file=sys.stderr)
        for recipe_list, items in missing_by_list.items():
            rel_list = recipe_list.relative_to(automodel_dir)
            print(f"\n  {rel_list}:", file=sys.stderr)
            for config, rel_path, line in items:
                loc = f"line {line}" if line else "line ?"
                print(f"    - [{loc}] {config} -> {rel_path} (not found)", file=sys.stderr)
        print(MISSING_HOW_TO_FIX, file=sys.stderr)

    if unread_lists or missing_by_list:
        return 1
    print("All recipe list references valid.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
