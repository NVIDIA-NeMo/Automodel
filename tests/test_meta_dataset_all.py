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

"""Smoke-test every dataset in a meta JSON file.

For each dataset, load the first 100 lines of its JSONL, convert to
Automodel conversation format, and run basic sanity checks.

Usage:
    python tests/test_meta_dataset_all.py /path/to/sft_v15.json
    python tests/test_meta_dataset_all.py /path/to/sft_v15.json --samples 50
    python tests/test_meta_dataset_all.py /path/to/sft_v15.json --dataset ViCA-322K
"""

import argparse
import json
import math
import os
import sys
import traceback

# Make sure the project root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nemo_automodel.components.datasets.vlm.datasets import (
    _convert_sharegpt_to_conversation,
    _load_json_or_jsonl,
    _resolve_lmdb_image,
)


def load_first_n_lines(file_path, n):
    """Load the first *n* records from a JSON/JSONL file."""
    if file_path.endswith(".jsonl"):
        rows = []
        with open(file_path) as f:
            for i, line in enumerate(f):
                if i >= n:
                    break
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    else:
        data = _load_json_or_jsonl(file_path)
        return data[:n]


def check_conversation(conv, ds_name, idx):
    """Run sanity checks on a single converted example."""
    errors = []
    if "conversation" not in conv:
        errors.append("missing 'conversation' key")
        return errors

    turns = conv["conversation"]
    if len(turns) == 0:
        errors.append("empty conversation")
        return errors

    for t, turn in enumerate(turns):
        if "role" not in turn:
            errors.append(f"turn {t}: missing 'role'")
            continue
        if turn["role"] not in ("user", "assistant"):
            errors.append(f"turn {t}: unexpected role '{turn['role']}'")
        if "content" not in turn:
            errors.append(f"turn {t}: missing 'content'")
            continue
        if not isinstance(turn["content"], list):
            errors.append(f"turn {t}: content is {type(turn['content']).__name__}, expected list")
            continue
        for p, part in enumerate(turn["content"]):
            if "type" not in part:
                errors.append(f"turn {t} part {p}: missing 'type'")
                continue
            if part["type"] == "image":
                img = part.get("image")
                if img is None:
                    errors.append(f"turn {t} part {p}: image is None")
                elif isinstance(img, str):
                    if "::" in img:
                        errors.append(f"turn {t} part {p}: unresolved LMDB path '{img}'")
                    elif not os.path.exists(img):
                        errors.append(f"turn {t} part {p}: image not found: {img}")
                # PIL Image is OK
            elif part["type"] == "video":
                vid = part.get("video")
                if vid is None:
                    errors.append(f"turn {t} part {p}: video is None")
                elif isinstance(vid, str) and not os.path.exists(vid):
                    errors.append(f"turn {t} part {p}: video not found: {vid}")
            elif part["type"] == "text":
                pass  # text is always OK
            else:
                errors.append(f"turn {t} part {p}: unknown type '{part['type']}'")
    return errors


def test_dataset(ds_name, ds_config, n_samples, meta_dir):
    """Test a single dataset entry from the meta file."""
    file_name = ds_config.get("file_name")
    if not file_name:
        return False, "missing 'file_name'", 0

    if not os.path.isabs(file_name):
        file_name = os.path.join(meta_dir, file_name)

    if not os.path.exists(file_name):
        return False, f"file not found: {file_name}", 0

    columns = ds_config.get("columns", {})
    tags = ds_config.get("tags", {})
    media_dir = ds_config.get("media_dir")

    try:
        raw_data = load_first_n_lines(file_name, n_samples)
    except Exception as e:
        return False, f"failed to load data: {e}", 0

    n_loaded = len(raw_data)
    all_errors = []

    for i, example in enumerate(raw_data):
        try:
            converted = _convert_sharegpt_to_conversation(
                example,
                columns=columns,
                tags=tags,
                media_dir=media_dir,
            )
            errs = check_conversation(converted, ds_name, i)
            for e in errs:
                all_errors.append(f"  sample {i}: {e}")
        except Exception as e:
            all_errors.append(f"  sample {i}: EXCEPTION: {e}")

    if all_errors:
        return False, "\n".join(all_errors), n_loaded
    return True, "OK", n_loaded


def main():
    parser = argparse.ArgumentParser(description="Test all datasets in a meta JSON file")
    parser.add_argument("meta_file", help="Path to the meta JSON file (e.g. sft_v15.json)")
    parser.add_argument("--samples", type=int, default=100, help="Samples per dataset (default: 100)")
    parser.add_argument("--dataset", type=str, default=None, help="Test only this dataset name")
    args = parser.parse_args()

    with open(args.meta_file) as f:
        meta = json.load(f)
    meta_dir = os.path.dirname(os.path.abspath(args.meta_file))

    if args.dataset:
        if args.dataset not in meta:
            print(f"ERROR: '{args.dataset}' not found in meta file")
            sys.exit(1)
        selected = {args.dataset: meta[args.dataset]}
    else:
        selected = meta

    total = len(selected)
    passed = 0
    failed = 0
    failed_names = []

    print(f"Testing {total} dataset(s), {args.samples} samples each\n")
    print(f"{'Dataset':<50s} {'Loaded':>7s}  {'Status'}")
    print("-" * 80)

    for ds_name, ds_config in selected.items():
        ok, msg, n_loaded = test_dataset(ds_name, ds_config, args.samples, meta_dir)
        status_str = f"  {n_loaded:>5d}" if n_loaded else "      ?"
        if ok:
            passed += 1
            print(f"{ds_name:<50s} {status_str}  PASS")
        else:
            failed += 1
            failed_names.append(ds_name)
            # Print first line of error inline, rest below
            first_line = msg.split("\n")[0]
            print(f"{ds_name:<50s} {status_str}  FAIL  {first_line}")
            if "\n" in msg:
                remaining = msg.split("\n")[1:]
                for line in remaining[:5]:  # Cap verbose output
                    print(f"{'':>60s}{line}")
                if len(remaining) > 5:
                    print(f"{'':>60s}... and {len(remaining) - 5} more errors")

    print("-" * 80)
    print(f"\nTotal: {total}  Passed: {passed}  Failed: {failed}")
    if failed_names:
        print(f"\nFailed datasets:")
        for name in failed_names:
            print(f"  - {name}")
        sys.exit(1)


if __name__ == "__main__":
    main()
