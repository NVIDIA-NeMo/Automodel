#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Per-file coverage floor check.

Compares per-file coverage between a base (main) and head (PR) coverage report,
and fails if any modified file that was above the floor drops below it.

Optionally, ``--prevent-regression`` also fails if *any* modified file loses
coverage, regardless of where it stands relative to the floor.

Usage::

    python tools/check_file_coverage.py \
        --base-json baseline-coverage.json \
        --head-json head-coverage.json \
        --floor 80 \
        --prevent-regression
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FileResult:
    path: str
    base_pct: float | None
    head_pct: float | None
    violated_floor: bool = False
    regressed: bool = False


@dataclass
class CheckResult:
    passed: bool
    violations: list[FileResult] = field(default_factory=list)
    all_results: list[FileResult] = field(default_factory=list)


def load_coverage_json(path: Path) -> dict[str, float]:
    """Return ``{filepath: percent_covered}`` from a ``coverage json`` output file."""
    data = json.loads(path.read_text())
    files = data.get("files", {})
    result: dict[str, float] = {}
    for fpath, fdata in files.items():
        summary = fdata.get("summary", {})
        pct = summary.get("percent_covered", 0.0)
        result[fpath] = round(pct, 2)
    return result


def get_changed_files(base_ref: str = "origin/main") -> list[str]:
    """Return list of files changed relative to *base_ref* using git."""
    result = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=ACMR", f"{base_ref}...HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        result = subprocess.run(
            ["git", "diff", "--name-only", "--diff-filter=ACMR", base_ref, "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
    if result.returncode != 0:
        print(f"WARNING: git diff failed (rc={result.returncode}): {result.stderr.strip()}", file=sys.stderr)
        return []
    return [f for f in result.stdout.strip().splitlines() if f.endswith(".py")]


def check_coverage(
    base_coverage: dict[str, float],
    head_coverage: dict[str, float],
    changed_files: list[str],
    floor: float,
    prevent_regression: bool,
) -> CheckResult:
    violations: list[FileResult] = []
    all_results: list[FileResult] = []

    for fpath in sorted(changed_files):
        base_pct = base_coverage.get(fpath)
        head_pct = head_coverage.get(fpath)

        if head_pct is None:
            continue

        r = FileResult(path=fpath, base_pct=base_pct, head_pct=head_pct)

        if base_pct is not None and base_pct >= floor and head_pct < floor:
            r.violated_floor = True

        if prevent_regression and base_pct is not None and head_pct < base_pct:
            r.regressed = True

        all_results.append(r)
        if r.violated_floor or r.regressed:
            violations.append(r)

    return CheckResult(passed=len(violations) == 0, violations=violations, all_results=all_results)


def _fmt_pct(pct: float | None) -> str:
    return f"{pct:6.2f}%" if pct is not None else "   N/A "


def print_report(result: CheckResult, floor: float, prevent_regression: bool) -> None:
    if not result.all_results:
        print("No changed Python files with coverage data found. Nothing to check.")
        return

    col_file = max(len(r.path) for r in result.all_results)
    col_file = max(col_file, 4)
    header = f"{'File':<{col_file}}  {'Base':>8}  {'Head':>8}  {'Status'}"
    sep = "-" * len(header)

    print(f"\n{'Per-File Coverage Check':^{len(header)}}")
    print(f"{'Floor: ' + str(floor) + '%':^{len(header)}}")
    if prevent_regression:
        print(f"{'(regression prevention ON)':^{len(header)}}")
    print(sep)
    print(header)
    print(sep)

    for r in result.all_results:
        status = "OK"
        if r.violated_floor:
            status = f"FAIL (dropped below {floor}%)"
        elif r.regressed:
            status = f"FAIL (regressed: {r.base_pct:.2f}% -> {r.head_pct:.2f}%)"
        print(f"{r.path:<{col_file}}  {_fmt_pct(r.base_pct)}  {_fmt_pct(r.head_pct)}  {status}")

    print(sep)

    if result.passed:
        print(f"\nPASSED: All {len(result.all_results)} checked file(s) meet coverage requirements.\n")
    else:
        print(f"\nFAILED: {len(result.violations)} file(s) violated coverage requirements.\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check per-file coverage floor on changed files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--base-json", type=Path, required=True, help="coverage json from the base branch (main)")
    parser.add_argument("--head-json", type=Path, required=True, help="coverage json from the PR head")
    parser.add_argument("--floor", type=float, default=80.0, help="coverage floor percentage (default: 80)")
    parser.add_argument(
        "--prevent-regression",
        action="store_true",
        default=False,
        help="also fail if any changed file's coverage decreased",
    )
    parser.add_argument(
        "--changed-files",
        nargs="*",
        default=None,
        help="explicit list of changed files (auto-detected from git if omitted)",
    )
    parser.add_argument("--base-ref", default="origin/main", help="git ref for the base branch (default: origin/main)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    base_coverage = load_coverage_json(args.base_json)
    head_coverage = load_coverage_json(args.head_json)

    if args.changed_files is not None:
        changed_files = args.changed_files
    else:
        changed_files = get_changed_files(args.base_ref)

    if not changed_files:
        print("No changed Python files detected. Nothing to check.")
        return 0

    result = check_coverage(
        base_coverage=base_coverage,
        head_coverage=head_coverage,
        changed_files=changed_files,
        floor=args.floor,
        prevent_regression=args.prevent_regression,
    )

    print_report(result, floor=args.floor, prevent_regression=args.prevent_regression)
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
