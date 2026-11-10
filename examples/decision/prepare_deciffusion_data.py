# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Prepare one-question training rows for ``train_deciffusion`` from public fictional data.

  # simple one-decision-per-line JSONL (e.g. the synthetic support-routing example)
  uv run python examples/decision/prepare_deciffusion_data.py simple --input support_routing.jsonl \
      --output data/support_routing --split three-way
  # SystemOne records (e.g. the fictional SOC-triage tutorial data with train/validation/test files)
  uv run python examples/decision/prepare_deciffusion_data.py systemone --input-dir data/soc_triage \
      --output data/soc_triage_rows

``--split reference`` reproduces the reference two-way split (``train``/``heldout``, held-out fraction 0.2, seed
``0``); its held-out file selects checkpoints, so it is validation, not an untouched test. ``--split three-way`` writes
group-disjoint ``train``/``validation``/``test`` files for a separate final test.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from nemo_automodel.components.models.deciffusion.rows import (
    convert_simple_row,
    in_heldout,
    read_jsonl,
    rows_from_systemone,
    write_jsonl,
)


def _summary(splits: dict[str, list[dict]]) -> dict:
    groups = {name: {r.get("group") or r["id"] for r in rows} for name, rows in splits.items()}
    names = list(groups)
    overlap = sum(len(groups[a] & groups[b]) for i, a in enumerate(names) for b in names[i + 1 :])
    return {
        "rows": {name: len(rows) for name, rows in splits.items()},
        "groups": {name: len(g) for name, g in groups.items()},
        "group_overlap": overlap,
        "types": dict(
            Counter(next(iter(r["input"]["questions"].values()))["type"] for rows in splits.values() for r in rows)
        ),
        "soft_label_rows": sum("soft_labels" in r for rows in splits.values() for r in rows),
    }


def main() -> None:
    """Convert and split rows; print a JSON summary."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="source", required=True)
    simple = sub.add_parser("simple")
    simple.add_argument("--input", required=True)
    simple.add_argument("--split", choices=["reference", "three-way"], default="three-way")
    simple.add_argument("--seed", default="0")
    simple.add_argument("--heldout-fraction", type=float, default=0.2)
    simple.add_argument("--test-fraction", type=float, default=0.2, help="three-way: share of groups for test")
    systemone = sub.add_parser("systemone")
    systemone.add_argument("--input-dir", required=True, help="directory with train/validation/test .jsonl")
    for p in (simple, systemone):
        p.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.source == "systemone":
        splits = {}
        for name in ("train", "validation", "test"):
            path = Path(args.input_dir) / f"{name}.jsonl"
            if path.exists():
                splits[name] = [row for record in read_jsonl(path) for row in rows_from_systemone(record)]
    else:
        rows, errors = [], 0
        for number, line in enumerate(read_jsonl(args.input), 1):
            try:
                rows.append(convert_simple_row(line, number))
            except (ValueError, KeyError, TypeError) as error:
                errors += 1
                print(f"{args.input}:{number}: skipped: {error}", file=sys.stderr)
        if args.split == "reference":
            splits = {
                "train": [r for r in rows if not in_heldout(r, args.heldout_fraction, args.seed)],
                "heldout": [r for r in rows if in_heldout(r, args.heldout_fraction, args.seed)],
            }
        else:
            test = [r for r in rows if in_heldout(r, args.test_fraction, f"{args.seed}|test")]
            rest = [r for r in rows if not in_heldout(r, args.test_fraction, f"{args.seed}|test")]
            fraction = args.heldout_fraction / (1 - args.test_fraction)
            splits = {
                "train": [r for r in rest if not in_heldout(r, fraction, f"{args.seed}|validation")],
                "validation": [r for r in rest if in_heldout(r, fraction, f"{args.seed}|validation")],
                "test": test,
            }
        if errors:
            print(f"skipped {errors} malformed rows", file=sys.stderr)
    for name, rows in splits.items():
        write_jsonl(Path(args.output) / f"{name}.jsonl", rows)
    print(json.dumps(_summary(splits), indent=2))


if __name__ == "__main__":
    main()
