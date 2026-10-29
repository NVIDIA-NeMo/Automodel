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

"""Rebuild ``tests/unit_tests/gpu_shard_durations.json`` from GPU unit-test shard logs.

Save the log of every ``L0_Unit_Tests_GPU_*`` job from one green run, for example with
``gh run view <run-id> --repo NVIDIA-NeMo/Automodel --log --job <job-id> > shard1.log``,
then run ``python tools/update_gpu_shard_durations.py shard1.log shard2.log --source "run <run-id>"``.

pytest's default progress output prints one line per test file and ends that line when the
next file starts. The time between consecutive file lines is therefore that file's wall
time, including its fixtures and teardown.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "tests" / "unit_tests" / "gpu_shard_durations.json"
_UNIT_TESTS_PREFIX = "tests/unit_tests/"
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
# Both the job-log API and `gh run view --log` (tab-separated job and step prefix) put a timestamp first.
_TIMESTAMPED_RE = re.compile(r"(?:^|\t)(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?)\d*Z (.*)$")
_FILE_LINE_RE = re.compile(r"^(tests/unit_tests/\S+\.py)(?: |$)")
_WRAPPED_LINE_RE = re.compile(r"^[.sxXFE]+\s*\[\s*\d+%\]$")


def parse_file_durations(log_text: str) -> dict[str, float]:
    """Return seconds per unit-test file from one pytest run in a GitHub Actions log.

    Args:
        log_text: Job log whose lines start with GitHub Actions timestamps.

    Returns:
        Seconds per test file, keyed by its path relative to ``tests/unit_tests``.

    Raises:
        ValueError: If the log has no pytest progress lines.
    """
    durations: dict[str, float] = {}
    previous_time: datetime | None = None
    current_file: str | None = None
    for raw_line in log_text.splitlines():
        match = _TIMESTAMPED_RE.search(_ANSI_RE.sub("", raw_line))
        if match is None:
            continue
        time = datetime.fromisoformat(match.group(1))
        body = match.group(2).rstrip()
        file_match = _FILE_LINE_RE.match(body)
        if file_match is not None:
            current_file = file_match.group(1).removeprefix(_UNIT_TESTS_PREFIX)
        elif current_file is None:
            previous_time = time  # collection output; the first file starts after it
            continue
        elif body.startswith(("=", "##[")):
            break  # session summary, or the runner stopped the attempt
        elif not _WRAPPED_LINE_RE.match(body):
            continue  # other output between progress lines
        if previous_time is None:
            previous_time = time
        durations[current_file] = durations.get(current_file, 0.0) + (time - previous_time).total_seconds()
        previous_time = time
    if not durations:
        raise ValueError("no pytest progress lines found")
    return durations


def main(argv: list[str] | None = None) -> int:
    """Merge per-file durations from shard logs and write the JSON used for GPU sharding."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("logs", nargs="+", type=Path, help="Job log of each GPU unit-test shard from one run.")
    parser.add_argument("--source", required=True, help="Where the logs came from, recorded in the JSON.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="JSON file to write.")
    args = parser.parse_args(argv)

    seconds: dict[str, float] = {}
    for log in args.logs:
        try:
            durations = parse_file_durations(log.read_text(encoding="utf-8", errors="replace"))
        except ValueError as exc:
            print(f"error: {log}: {exc}", file=sys.stderr)
            return 1
        for path, duration in durations.items():
            seconds[path] = seconds.get(path, 0.0) + duration

    document = {"source": args.source, "seconds": {path: round(seconds[path], 1) for path in sorted(seconds)}}
    args.output.write_text(json.dumps(document, indent=2) + "\n")
    print(f"Wrote {len(seconds)} test files ({sum(seconds.values()):.0f} s) to {args.output}.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
