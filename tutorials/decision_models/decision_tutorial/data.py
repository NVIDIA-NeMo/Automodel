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

"""Deterministic, fictional typed-decision data for tutorial smoke tests."""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import logging
import random
import re
from pathlib import Path
from typing import Literal, TypedDict

LOGGER = logging.getLogger(__name__)


class Question(TypedDict):
    """A question in the public typed-decision request format."""

    type: Literal["choice", "score", "noul"]
    instructions: str
    criteria: dict[str, str] | list[str]


class Record(TypedDict):
    """One fictional observation, gold decisions and complete option utilities."""

    id: str
    state: dict[str, object]
    questions: dict[str, Question]
    answers: dict[str, str]
    rewards: dict[str, dict[str, float]]
    meta: dict[str, str]


QUESTIONS: dict[str, Question] = {
    "verdict": {
        "type": "choice",
        "instructions": "What explains the observed security alert?",
        "criteria": {
            "malicious": "The evidence supports an unauthorized attack.",
            "benign_authorized": "The activity is authorized and expected.",
            "false_positive": "A sensor artifact explains the alert.",
        },
    },
    "action": {
        "type": "choice",
        "instructions": "Which response follows the tutorial policy?",
        "criteria": {
            "close": "Close the alert.",
            "monitor": "Monitor the asset without containment.",
            "escalate": "Ask an analyst to investigate.",
            "contain": "Contain the affected asset.",
        },
    },
    "severity": {
        "type": "score",
        "instructions": "How severe is the event under the tutorial policy?",
        "criteria": ["Informational", "Low", "Medium", "High", "Critical"],
    },
    "attack_tactic": {
        "type": "choice",
        "instructions": "Which behavior generated this alert?",
        "criteria": {
            "credential_access": "An attempt to obtain or misuse credentials.",
            "execution": "A process or script execution event.",
        },
    },
    "needs_human": {
        "type": "noul",
        "instructions": "Does the tutorial policy require analyst review?",
        "criteria": {"false": "No analyst review is required.", "true": "An analyst must review the alert."},
    },
}


def option_ids(question: Question) -> list[str]:
    """Return semantic option IDs without imposing a model's native order."""
    criteria = question["criteria"]
    return [str(index) for index in range(len(criteria))] if isinstance(criteria, list) else list(criteria)


def make_records(*, count: int, seed: int) -> list[Record]:
    """Create new fictional records; identifiers use reserved example ranges."""
    if count < 1:
        raise ValueError("count must be positive")
    rng = random.Random(seed)
    records: list[Record] = []
    for index in range(count):
        verdict = rng.choice(["malicious", "benign_authorized", "false_positive"])
        ambiguous = rng.choice([False, True])
        critical = rng.choice([False, True])
        tactic = rng.choice(["credential_access", "execution"])
        explanations = {
            "malicious": "A controlled test artifact matches the detection and no authorized change exists.",
            "benign_authorized": "A signed maintenance ticket and the approved schedule explain the activity.",
            "false_positive": "A replayed sensor event does not correspond to activity on this asset.",
        }
        action = "escalate" if ambiguous else ("contain" if verdict == "malicious" else "close")
        severity = 3 + int(critical) if verdict == "malicious" else int(verdict == "benign_authorized")
        answers = {
            "verdict": verdict,
            "action": action,
            "severity": str(severity),
            "attack_tactic": tactic,
            "needs_human": "true" if ambiguous or (critical and verdict == "malicious") else "false",
        }
        rewards = {
            name: {option: float(option == answers[name]) for option in option_ids(question)}
            for name, question in QUESTIONS.items()
        }
        rewards["severity"] = {str(level): 1.0 - abs(level - severity) / 4 for level in range(5)}
        if verdict == "malicious":
            rewards["action"]["close"] = -1.0
            rewards["verdict"]["benign_authorized"] = -1.0
        address = f"192.0.2.{1 + index % 254}"
        records.append(
            {
                "id": f"fictional-{seed}-{index:05d}",
                "state": {
                    "event_id": f"event-{seed}-{index:05d}",
                    "asset": f"host-{index:05d}.example",
                    "source_ip": address,
                    "contact": f"analyst-{index:05d}@example.com",
                    "behavior": "credential read attempt" if tactic == "credential_access" else "script execution",
                    "evidence": explanations[verdict],
                    "evidence_quality": "ambiguous" if ambiguous else "clear",
                    "asset_impact": "critical" if critical else "ordinary",
                },
                "questions": json.loads(json.dumps(QUESTIONS)),
                "answers": answers,
                "rewards": rewards,
                "meta": {"family": tactic, "verdict": verdict, "evidence": "ambiguous" if ambiguous else "clear"},
            }
        )
    return records


def validate_summary(summary: str) -> None:
    """Reject obvious non-example identifiers in optional generated prose."""
    for email in re.findall(r"[A-Za-z0-9_.+%-]+@([A-Za-z0-9.-]+\.[A-Za-z]{2,})", summary):
        if email != "example.com" and not email.endswith(".example"):
            raise ValueError("Generated prose contains a non-example email domain")
    ranges = tuple(ipaddress.ip_network(net) for net in ("192.0.2.0/24", "198.51.100.0/24", "203.0.113.0/24"))
    for value in re.findall(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])", summary):
        address = ipaddress.ip_address(value)
        if not any(address in network for network in ranges):
            raise ValueError("Generated prose contains a non-documentation IP")
    if re.search(r"-----BEGIN .*PRIVATE KEY-----|\b(?:nvapi|sk)-[A-Za-z0-9_-]{16,}", summary):
        raise ValueError("Generated prose contains a credential-like string")


def validate_splits(splits: dict[str, list[Record]]) -> None:
    """Reject overlapping records, invalid targets and non-example identifiers."""
    seen_ids: set[str] = set()
    seen_states: set[str] = set()
    documentation_range = ipaddress.ip_network("192.0.2.0/24")
    for split, records in splits.items():
        if not records:
            raise ValueError(f"Empty split: {split}")
        for record in records:
            state = json.dumps(record["state"], sort_keys=True)
            if record["id"] in seen_ids or state in seen_states:
                raise ValueError(f"Repeated record or state: {record['id']}")
            seen_ids.add(record["id"])
            seen_states.add(state)
            if record["questions"] != QUESTIONS:
                raise ValueError("Question schema differs from the tutorial schema")
            if ipaddress.ip_address(str(record["state"]["source_ip"])) not in documentation_range:
                raise ValueError("source_ip must use the documentation range")
            if not str(record["state"]["contact"]).endswith("@example.com"):
                raise ValueError("contact must use example.com")
            if not str(record["state"]["asset"]).endswith(".example"):
                raise ValueError("asset must use the reserved .example suffix")
            for name, question in QUESTIONS.items():
                options = option_ids(question)
                if record["answers"][name] not in options:
                    raise ValueError(f"Unknown gold option for {name}")
                if set(record["rewards"][name]) != set(options):
                    raise ValueError(f"Utility options differ for {name}")


def write_dataset(
    output_dir: Path, *, train_records: int = 128, validation_records: int = 32, test_records: int = 32, seed: int = 7
) -> dict[str, list[Record]]:
    """Write a fresh dataset directory and a content-hash manifest.

    Existing output directories are rejected so an experiment's splits cannot
    be silently replaced. This fixture tests mechanics, not security efficacy.
    """
    if output_dir.exists():
        raise FileExistsError(f"Choose a fresh dataset directory: {output_dir}")
    counts = {"train": train_records, "validation": validation_records, "test": test_records}
    if any(count < 1 for count in counts.values()):
        raise ValueError("Every split must contain at least one record")
    records = make_records(count=sum(counts.values()), seed=seed)
    splits: dict[str, list[Record]] = {}
    start = 0
    for name, count in counts.items():
        splits[name] = records[start : start + count]
        start += count
    validate_splits(splits)
    output_dir.mkdir(parents=True)
    hashes = {}
    for name, part in splits.items():
        path = output_dir / f"{name}.jsonl"
        path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in part), encoding="utf-8")
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output_dir / "questions.json").write_text(json.dumps(QUESTIONS, indent=2) + "\n", encoding="utf-8")
    manifest = {"kind": "deterministic-fictional-smoke-data", "seed": seed, "counts": counts, "sha256": hashes}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return splits


def main() -> None:
    """Generate fresh local smoke data without a network or model download."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-records", type=int, default=128)
    parser.add_argument("--validation-records", type=int, default=32)
    parser.add_argument("--test-records", type=int, default=32)
    args = parser.parse_args()
    splits = write_dataset(
        args.output_dir,
        train_records=args.train_records,
        validation_records=args.validation_records,
        test_records=args.test_records,
        seed=args.seed,
    )
    LOGGER.info("Wrote %s", {name: len(records) for name, records in splits.items()})


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
