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

"""Fail a CI training job when its logged loss or gradient norm is non-finite."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

_STEP_METRICS = re.compile(
    r"\bstep\s+(?P<step>\d+)\s*\|.*?\bloss\s+(?P<loss>\S+)\s*\|\s*grad_norm\s+(?P<grad_norm>\S+)"
)


def assert_finite_train_metrics(log_path: Path) -> int:
    """Validate every training-step loss and gradient norm in a log.

    Args:
        log_path: Training log written by the NeMo CI Slurm job.

    Returns:
        Zero when at least one step was found and all metrics are finite;
        otherwise one.
    """
    metrics_found = 0
    failures: list[str] = []
    with log_path.open(encoding="utf-8", errors="replace") as log:
        for line in log:
            match = _STEP_METRICS.search(line)
            if match is None:
                continue
            metrics_found += 1
            for metric_name in ("loss", "grad_norm"):
                raw_value = match.group(metric_name).rstrip(",")
                try:
                    value = float(raw_value)
                except ValueError:
                    failures.append(f"step {match.group('step')} {metric_name}={raw_value!r}")
                    continue
                if not math.isfinite(value):
                    failures.append(f"step {match.group('step')} {metric_name}={raw_value}")

    if metrics_found == 0:
        print(f"ERROR: no training-step loss/grad_norm metrics found in {log_path}")
        return 1
    if failures:
        print("ERROR: non-finite training metrics: " + ", ".join(failures))
        return 1
    print(f"Validated finite loss and grad_norm for {metrics_found} training steps.")
    return 0


def main() -> int:
    """Parse CLI arguments and validate the requested training log."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, type=Path, help="NeMo CI Slurm training log")
    args = parser.parse_args()
    return assert_finite_train_metrics(args.log)


if __name__ == "__main__":
    raise SystemExit(main())
