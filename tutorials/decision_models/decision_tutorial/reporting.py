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

"""Inspect locally generated framework reports without fixed performance claims."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TypedDict


class StageResult(TypedDict):
    """Held-out measurements for one stage of a local experiment."""

    stage: str
    records: int
    decisions: int
    accuracy: float
    exact_match: float
    nll: float
    brier: float
    reward: float


def summarize_run(run_dir: Path, *, stages: tuple[str, ...] = ("base", "sft")) -> list[StageResult]:
    """Read native test reports and require equal record and decision counts.

    Equal counts do not establish record identity. Retain the same split manifest
    and input files across stages; verify their hashes separately."""
    rows: list[StageResult] = []
    for stage in stages:
        path = run_dir / stage / "metrics_test.json"
        overall = json.loads(path.read_text(encoding="utf-8"))["overall"]
        row: StageResult = {
            "stage": stage,
            "records": int(overall["records"]),
            "decisions": int(overall["n"]),
            **{key: float(overall[key]) for key in ("accuracy", "exact_match", "nll", "brier", "reward")},
        }
        if rows and (row["records"], row["decisions"]) != (rows[0]["records"], rows[0]["decisions"]):
            raise ValueError("Stage reports have different record or decision counts")
        rows.append(row)
    return rows


def write_summary(rows: list[StageResult], path: Path) -> None:
    """Write a fresh CSV summary; accuracy and utility may move differently."""
    if not rows:
        raise ValueError("At least one stage is required")
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(StageResult.__annotations__))
        writer.writeheader()
        writer.writerows(rows)
