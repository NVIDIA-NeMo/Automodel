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

"""Run annotation policy, signature completeness, and opt-in ty checks together.

Usage: uv run --group linting python tools/check_types.py
Run from the repository root. No training modules are imported by this runner.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import tomllib
from pathlib import Path


def main() -> int:
    """Require complete signatures and consistent types in ty's opt-in scope."""
    config = tomllib.loads(Path("ty.toml").read_text())
    targets = sorted({pattern.removesuffix("/**") for entry in config["overrides"] for pattern in entry["include"]})
    if not targets or any(not Path(target).exists() or any(c in target for c in "*?[") for target in targets):
        raise ValueError("ty overrides must name existing files or directories (optionally ending in /**)")
    policy = subprocess.run([sys.executable, "tools/check_annotation_policy.py"], check=False)
    logging.getLogger(__name__).info("Annotation and ty check targets: %s", " ".join(targets))
    excludes = config.get("src", {}).get("exclude", [])
    annotations = subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--isolated",
            "--ignore-noqa",
            "--select",
            "ANN001,ANN002,ANN003,ANN201,ANN202,ANN204,ANN205,ANN206",
            "--force-exclude",
            *(["--exclude", ",".join(excludes)] if excludes else []),
            *targets,
        ],
        check=False,
    )
    checker = subprocess.run([sys.executable, "-m", "ty", "check", "--verbose", *targets], check=False)
    return int(policy.returncode != 0 or annotations.returncode != 0 or checker.returncode != 0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
