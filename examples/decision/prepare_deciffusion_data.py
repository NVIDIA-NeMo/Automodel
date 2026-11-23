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

"""Convert one-decision JSONL into single-question SystemOne records with group-disjoint splits.

  # simple one-decision-per-line JSONL (e.g. a synthetic support-routing dataset)
  uv run python examples/decision/prepare_deciffusion_data.py --input support_routing.jsonl \
      --output data/support_routing --split three-way

SystemOne records, such as the fictional SOC-triage tutorial data, need no conversion: point ``data.*`` at them.
``--split two-way`` writes a ``train``/``heldout`` split (held-out fraction 0.2, seed ``0``); its held-out file
selects checkpoints, so it is validation, not an untouched test. ``--split three-way`` writes group-disjoint
``train``/``validation``/``test`` files for a separate final test.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from nemo_automodel.components.datasets.decision.records import validate_record
from nemo_automodel.components.models.deciffusion.records import (
    convert_simple_line,
    in_heldout,
    read_jsonl,
    write_jsonl,
)


def _summary(splits: dict[str, list[dict]]) -> dict:
    groups = {name: {(r.get("meta") or {}).get("group") or r["id"] for r in rows} for name, rows in splits.items()}
    names = list(groups)
    overlap = sum(len(groups[a] & groups[b]) for i, a in enumerate(names) for b in names[i + 1 :])
    return {
        "records": {name: len(rows) for name, rows in splits.items()},
        "groups": {name: len(g) for name, g in groups.items()},
        "group_overlap": overlap,
        "types": dict(Counter(q["type"] for rows in splits.values() for r in rows for q in r["questions"].values())),
        "soft_label_records": sum("soft_labels" in r for rows in splits.values() for r in rows),
    }


def main() -> None:
    """Convert, validate and split one-decision lines; print a JSON summary."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=["two-way", "three-way"], default="three-way")
    parser.add_argument("--seed", default="0")
    parser.add_argument("--heldout-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2, help="three-way: share of groups for test")
    args = parser.parse_args()

    records, errors = [], 0
    for number, line in enumerate(read_jsonl(args.input), 1):
        try:
            record = convert_simple_line(line, number)
            validate_record(record)
        except (ValueError, KeyError, TypeError) as error:
            errors += 1
            print(f"{args.input}:{number}: skipped: {error}", file=sys.stderr)
            continue
        records.append(record)
    if len({r["id"] for r in records}) != len(records):
        raise SystemExit(f"{args.input}: record IDs must be unique")
    if args.split == "two-way":
        splits = {
            "train": [r for r in records if not in_heldout(r, args.heldout_fraction, args.seed)],
            "heldout": [r for r in records if in_heldout(r, args.heldout_fraction, args.seed)],
        }
    else:
        test = [r for r in records if in_heldout(r, args.test_fraction, f"{args.seed}|test")]
        rest = [r for r in records if not in_heldout(r, args.test_fraction, f"{args.seed}|test")]
        fraction = args.heldout_fraction / (1 - args.test_fraction)
        splits = {
            "train": [r for r in rest if not in_heldout(r, fraction, f"{args.seed}|validation")],
            "validation": [r for r in rest if in_heldout(r, fraction, f"{args.seed}|validation")],
            "test": test,
        }
    if errors:
        print(f"skipped {errors} malformed lines", file=sys.stderr)
    for name, rows in splits.items():
        write_jsonl(Path(args.output) / f"{name}.jsonl", rows)
    print(json.dumps(_summary(splits), indent=2))


if __name__ == "__main__":
    main()
