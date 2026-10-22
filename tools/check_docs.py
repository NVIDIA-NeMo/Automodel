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
# dependencies = ["pytest", "pytest-timeout", "pyyaml", "pillow"]
# ///

"""The single docs-check entry point for contributors, agents, and CI.

Run ``uv run --script tools/check_docs.py`` from the repository root. Keep docs
validation orchestration here so local checks and CI enforce the same rules.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def main() -> int:
    """Check source contracts and rendered docs, returning the failing command's status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keep-generated",
        action="store_true",
        help="Keep generated model-coverage tables after a successful check for publication.",
    )
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    try:
        # Source contracts must run before generation changes the checked files.
        tests = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "--noconftest",
                "tests/unit_tests/ci_tests/test_docs_workflows.py",
                "tests/unit_tests/tools/test_sync_model_coverage_tables.py",
                "tests/unit_tests/tools/test_check_docs.py",
                "tests/unit_tests/_transformers/test_doc_coverage.py::"
                "test_every_model_card_has_architecture_reference_table",
            ],
            cwd=repo_root,
            check=False,
        )
        if tests.returncode:
            return tests.returncode

        version = json.loads((repo_root / "docs/fern/fern.config.json").read_text(encoding="utf-8"))["version"]
        generator = "tests/ci_tests/utils/sync_model_coverage_tables.py"
        status = 1
        keep_tables = False
        try:
            status = subprocess.run([sys.executable, generator], cwd=repo_root, check=False).returncode
            if status == 0:
                status = subprocess.run(
                    ["npx", "--yes", f"--package=fern-api@{version}", "--", "make", "-C", "docs/fern", "_docs-check"],
                    cwd=repo_root,
                    env={**os.environ, "FERN_NO_VERSION_REDIRECTION": "true"},
                    check=False,
                ).returncode
            if status == 0:
                status = subprocess.run(["git", "diff", "--check"], cwd=repo_root, check=False).returncode
            keep_tables = args.keep_generated and status == 0
        finally:
            # The Makefile removes generated styles. Retain tables only when
            # publishing requested them and every check completed successfully.
            if not keep_tables:
                cleanup = subprocess.run([sys.executable, generator, "--clean"], cwd=repo_root, check=False)
                status = status or cleanup.returncode
        return status
    except OSError as error:
        logger.error("Docs check could not run: %s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
