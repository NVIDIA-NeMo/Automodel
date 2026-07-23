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
"""Validate that generate_ci_tests.generate_pipeline runs cleanly for every
(test_folder, scope) combination defined in this repo.

The matrix is auto-discovered from two sources of truth so the check stays current
without manual maintenance:
  * AUTO_DISCOVER_SCOPES in generate_ci_tests.py (e.g. release, performance)
  * tests/ci_tests/configs/<folder>/<scope>_recipes.yml files

Each combination is invoked; failures are collected and reported together so a
single broken combo does not hide the rest.
"""

import argparse
import sys
import traceback
from pathlib import Path

# Allow `from generate_ci_tests import ...` when this script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_ci_tests import AUTO_DISCOVER_SCOPES, generate_pipeline  # noqa: E402


def collect_matrix(automodel_dir: Path) -> list[tuple[str, str]]:
    """Discover every (test_folder, scope) pair worth validating."""
    matrix: set[tuple[str, str]] = set()

    for scope, folders in AUTO_DISCOVER_SCOPES.items():
        for folder in folders:
            matrix.add((folder, scope))

    configs_root = automodel_dir / "tests" / "ci_tests" / "configs"
    for recipe_list in configs_root.glob("*/*_recipes.yml"):
        # override_recipes.yml is not a scope; skip it.
        if recipe_list.name == "override_recipes.yml":
            continue
        folder = recipe_list.parent.name
        scope = recipe_list.stem[: -len("_recipes")]
        matrix.add((folder, scope))

    # Also exercise every config folder with scope=nightly so folders without a
    # nightly_recipes.yml hit the empty-pipeline path.
    for config_dir in configs_root.iterdir():
        if config_dir.is_dir():
            matrix.add((config_dir.name, "nightly"))

    return sorted(matrix)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--automodel-dir", type=str, required=True, help="Path to Automodel directory")
    args = parser.parse_args()

    automodel_dir = Path(args.automodel_dir).resolve()
    matrix = collect_matrix(automodel_dir)

    successes: list[tuple[str, str, int]] = []
    failures: list[tuple[str, str, str]] = []

    for folder, scope in matrix:
        try:
            pipeline = generate_pipeline(str(automodel_dir), scope, folder)
        except Exception:
            failures.append((folder, scope, traceback.format_exc()))
            continue
        job_count = sum(1 for k in pipeline if k not in ("include", "stages"))
        successes.append((folder, scope, job_count))

    for folder, scope, job_count in successes:
        print(f"OK   {folder}/{scope}: {job_count} jobs", file=sys.stderr)
    for folder, scope, tb in failures:
        print(f"\nFAIL {folder}/{scope}:\n{tb}", file=sys.stderr)

    if failures:
        print(
            f"\n{len(failures)} of {len(matrix)} (test_folder, scope) combos failed.",
            file=sys.stderr,
        )
        return 1
    print(
        f"\nAll {len(matrix)} (test_folder, scope) combos generated successfully.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
