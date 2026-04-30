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


def collect_nightly_lists(automodel_dir: Path):
    """Yield (recipe_list_path, examples_dir, configs) for each nightly_recipes.yml."""
    configs_root = automodel_dir / "tests" / "ci_tests" / "configs"
    for recipe_list in sorted(configs_root.glob("*/nightly_recipes.yml")):
        with recipe_list.open("r", encoding="utf-8") as f:
            data = yaml.load(f) or {}
        configs = data.get("configs") or []
        examples_dir = data.get("examples_dir", recipe_list.parent.name)
        yield recipe_list, examples_dir, configs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--automodel-dir", type=str, required=True, help="Path to Automodel directory")
    args = parser.parse_args()

    automodel_dir = Path(args.automodel_dir).resolve()
    missing_by_list: dict[Path, list[tuple[str, Path]]] = {}

    for recipe_list, examples_dir, configs in collect_nightly_lists(automodel_dir):
        for config in configs:
            rel_path = Path("examples") / examples_dir / config
            if not (automodel_dir / rel_path).is_file():
                missing_by_list.setdefault(recipe_list, []).append((config, rel_path))

    if not missing_by_list:
        print("All nightly recipe references valid.")
        return 0

    print("Missing nightly recipe references:", file=sys.stderr)
    for recipe_list, entries in missing_by_list.items():
        rel_list = recipe_list.relative_to(automodel_dir)
        print(f"\n  {rel_list}:", file=sys.stderr)
        for config, rel_path in entries:
            print(f"    - {config} -> {rel_path} (not found)", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
