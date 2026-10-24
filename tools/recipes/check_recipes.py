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

# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["pyyaml", "ruamel.yaml"]
# ///

"""Run the recipe checks used by contributors, agents, and CI.

Run ``uv run --script tools/recipes/check_recipes.py`` from the repository root.
Only the new-recipe metadata check is limited to added files; all other checks
validate the current checkout. No training dependencies or GPUs are required.
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _added_recipes(repo_root: Path, base_ref: str, head_ref: str | None) -> list[str]:
    merge_base = subprocess.run(
        ["git", "merge-base", "--", base_ref, head_ref or "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    diff = subprocess.run(
        [
            "git",
            "diff",
            "--name-only",
            "--diff-filter=A",
            "-z",
            merge_base,
            *([head_ref] if head_ref else []),
            "--",
            "examples/",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    paths = set(diff.split("\0"))
    if head_ref is None:
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "-z", "--", "examples/"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        paths.update(untracked.split("\0"))
    return sorted(path for path in paths if Path(path).suffix in {".yaml", ".yml"})


def main() -> int:
    """Run recipe validators and return the first failing command's status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-ref",
        default="origin/main",
        help="Base Git ref for detecting added recipes (default: origin/main; fetch it first).",
    )
    parser.add_argument(
        "--head-ref",
        help="Compare committed additions at this ref; by default include local changes and untracked recipes.",
    )
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    try:
        added = _added_recipes(repo_root, args.base_ref, args.head_ref)
        for script, paths in (
            ("tools/lint_example_yamls.py", []),
            ("tests/ci_tests/utils/validate_nightly_recipes.py", []),
            ("tests/ci_tests/utils/validate_generate_ci.py", []),
            ("tests/ci_tests/utils/validate_new_recipe_ci.py", added),
        ):
            logger.info("Running %s", script)
            status = subprocess.run(
                [sys.executable, script, "--automodel-dir", str(repo_root), *paths],
                cwd=repo_root,
                check=False,
            ).returncode
            if status:
                return status
    except subprocess.CalledProcessError as error:
        logger.error(
            "Cannot determine added recipes; fetch the base/head refs and their history: %s", error.stderr.strip()
        )
        return error.returncode
    except OSError as error:
        logger.error("Recipe check could not run: %s", error)
        return 1
    logger.info("All recipe checks passed.")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
