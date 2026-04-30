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
import sys
from pathlib import Path

from ruamel.yaml import YAML

yaml = YAML()


REPORT_HEADER = "Missing nightly recipe references:"

REPORT_HOW_TO_FIX = """
------------------------------------------------------------
How to fix - for each missing entry above, choose one:

  1. Add the recipe at the path shown (typical when a recipe was
     renamed or moved without updating nightly_recipes.yml).

  2. Remove the line from
     tests/ci_tests/configs/<folder>/nightly_recipes.yml
     if the recipe is genuinely gone.

See tests/ci_tests/README.md for details.
------------------------------------------------------------"""


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
        print("All nightly recipe references valid.", file=sys.stderr)
        return 0

    print(REPORT_HEADER, file=sys.stderr)
    for recipe_list, items in missing_by_list.items():
        rel_list = recipe_list.relative_to(automodel_dir)
        print(f"\n  {rel_list}:", file=sys.stderr)
        for config, rel_path, line in items:
            loc = f"line {line}" if line else "line ?"
            print(f"    - [{loc}] {config} -> {rel_path} (not found)", file=sys.stderr)
    print(REPORT_HOW_TO_FIX, file=sys.stderr)

    return 1


if __name__ == "__main__":
    sys.exit(main())
