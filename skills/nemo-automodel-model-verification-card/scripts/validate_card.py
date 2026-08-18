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

#!/usr/bin/env python3

"""Validate a NeMo AutoModel model verification card."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import yaml

ALLOWED_STATUSES = {"verified", "not_verified"}
TRAINING_METRICS = (
    "initial_loss",
    "final_loss",
    "last_10_steps_step_time_ms_avg",
    "last_10_steps_model_tflops_per_gpu_avg",
    "last_10_steps_tokens_per_second_per_gpu_avg",
)


def _mapping(value: object, location: str, errors: list[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        errors.append(f"{location} must be a mapping")
        return {}
    return value


def _string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_card(card: object) -> list[str]:
    """Return validation errors for a parsed verification card.

    Args:
        card: Parsed YAML document.

    Returns:
        Human-readable validation errors. An empty list means the card is valid.
    """
    errors: list[str] = []
    root = _mapping(card, "card", errors)
    for field in ("title", "summary", "model", "verification_environment", "verification_index", "items"):
        if field not in root:
            errors.append(f"card is missing `{field}`")
    for field in ("title", "summary"):
        if not _string(root.get(field)):
            errors.append(f"card.{field} must be a non-empty string")

    model = _mapping(root.get("model"), "model", errors)
    for field in ("hf_id", "hf_revision", "architecture"):
        if not _string(model.get(field)):
            errors.append(f"model.{field} must be a non-empty string")

    environment = _mapping(root.get("verification_environment"), "verification_environment", errors)
    if not _string(environment.get("automodel_commit")):
        errors.append("verification_environment.automodel_commit must be a non-empty string")

    index = _mapping(root.get("verification_index"), "verification_index", errors)
    items = _mapping(root.get("items"), "items", errors)
    indexed: dict[tuple[str, str], tuple[str, str]] = {}

    for category, hardware_map_value in index.items():
        hardware_map = _mapping(hardware_map_value, f"verification_index.{category}", errors)
        for hardware, buckets_value in hardware_map.items():
            location = f"verification_index.{category}.{hardware}"
            buckets = _mapping(buckets_value, location, errors)
            if set(buckets) != ALLOWED_STATUSES:
                errors.append(f"{location} must contain exactly `verified` and `not_verified` lists")
            for status in ALLOWED_STATUSES:
                names = buckets.get(status)
                if not isinstance(names, list) or any(not _string(name) for name in names):
                    errors.append(f"{location}.{status} must be a list of non-empty item names")
                    continue
                for name in names:
                    key = (name, str(hardware))
                    previous = indexed.get(key)
                    if previous is not None:
                        errors.append(
                            f"items.{name}.{hardware} is indexed more than once: "
                            f"{previous[0]}/{previous[1]} and {category}/{status}"
                        )
                    indexed[key] = (str(category), status)

    actual: set[tuple[str, str]] = set()
    for name, hardware_map_value in items.items():
        hardware_map = _mapping(hardware_map_value, f"items.{name}", errors)
        for hardware, leaf_value in hardware_map.items():
            location = f"items.{name}.{hardware}"
            leaf = _mapping(leaf_value, location, errors)
            key = (str(name), str(hardware))
            actual.add(key)
            status = leaf.get("status")
            if status not in ALLOWED_STATUSES:
                errors.append(f"{location}.status must be `verified` or `not_verified`")
                continue
            expected = indexed.get(key)
            if expected is None:
                errors.append(f"{location} is missing from verification_index")
                continue
            category, indexed_status = expected
            if status != indexed_status:
                errors.append(f"{location}.status is `{status}` but the index records `{indexed_status}`")

            if status == "verified":
                for field in ("precision", "automodel_commit", "last_verified", "expected_result"):
                    if not _string(leaf.get(field)):
                        errors.append(f"{location}.{field} must be a non-empty string for verified items")
                if not (_string(leaf.get("recipe")) or _string(leaf.get("command"))):
                    errors.append(f"{location} must include a recipe or command for verified items")
                if category in {"training", "performance"}:
                    metrics = _mapping(leaf.get("metrics"), f"{location}.metrics", errors)
                    for metric in TRAINING_METRICS:
                        value = metrics.get(metric)
                        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                            errors.append(f"{location}.metrics.{metric} must be a finite number")

            if name == "checkpoint_resume" and status == "verified":
                comparison = _mapping(leaf.get("resume_comparison"), f"{location}.resume_comparison", errors)
                shared_steps = comparison.get("shared_steps")
                passed_steps = comparison.get("passed_steps")
                if not isinstance(shared_steps, int) or shared_steps <= 0:
                    errors.append(f"{location}.resume_comparison.shared_steps must be a positive integer")
                if passed_steps != shared_steps:
                    errors.append(f"{location}.resume_comparison.passed_steps must equal shared_steps")
                for field in ("learning_rate_exact", "consumed_tokens_exact"):
                    if comparison.get(field) is not True:
                        errors.append(f"{location}.resume_comparison.{field} must be true")
                if not _string(comparison.get("loss_tolerance")):
                    errors.append(f"{location}.resume_comparison.loss_tolerance must be a non-empty string")

            if name == "sft_long_context":
                contract = _mapping(leaf.get("verification_contract"), f"{location}.verification_contract", errors)
                dataset = str(contract.get("dataset", "")).casefold().replace("_", "").replace("-", "")
                if "coderforge" not in dataset:
                    errors.append(f"{location}.verification_contract.dataset must identify CoderForge")
                if contract.get("sequence_length") != 131072:
                    errors.append(f"{location}.verification_contract.sequence_length must be 131072")

    for name, hardware in sorted(indexed.keys() - actual):
        errors.append(f"verification_index references missing items.{name}.{hardware}")
    return errors


def main(argv: list[str] | None = None) -> int:
    """Validate a card from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("card", type=Path, help="Path to a *_verification_card.yaml file")
    args = parser.parse_args(argv)

    if not args.card.name.endswith("_verification_card.yaml"):
        print("error: card filename must end in _verification_card.yaml", file=sys.stderr)
        return 1
    try:
        card = yaml.safe_load(args.card.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        print(f"error: could not load {args.card}: {error}", file=sys.stderr)
        return 1

    errors = validate_card(card)
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"Validated {args.card}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
