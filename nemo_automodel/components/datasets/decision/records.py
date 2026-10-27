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

"""SystemOne records: typed questions, option orders, and labelled decision examples.

A *record* is the Jev / SystemOne ``POST /v1/systemone`` request body plus supervision::

    {
        "id": "soc-1a2b",
        "state": {...},                                   # any JSON value
        "questions": {"verdict": {"type": "choice", "instructions": "...", "criteria": {...}}, ...},
        "answers": {"verdict": "malicious", ...},         # gold option id per question (SFT)
        "rewards": {"verdict": {"malicious": 1.0, ...}},  # optional per-option rewards (RL)
        "meta": {...},                                     # optional slicing metadata
    }

Question types follow the SystemOne API: ``noul`` (true/false), ``choice`` (named options) and ``score``
(ordered levels ``"0".."K-1"``).  Each model family enumerates options in its own order, so every tensor
in this package is built from an explicit list of option ids rather than from dictionary order.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, TypeVar

QUESTION_TYPES = ("noul", "choice", "score")
QUESTION_TYPE_IDS = {"noul": 0, "choice": 1, "score": 2}


_Record = TypeVar("_Record")


def pad_and_shard_records(
    records: Sequence[_Record], *, rank: int, world_size: int, shuffle_seed: int | None = None
) -> list[_Record]:
    """Pad cyclically and return equal-length data-parallel record shards.

    Equal shard lengths keep collective-bearing forwards aligned across ranks.
    Padding duplicates existing records; callers that aggregate evaluation
    predictions must deduplicate by record ID. An empty input yields empty shards
    on every rank.

    Args:
        records: Ordered input records.
        rank: Data-parallel rank in [0, world_size).
        world_size: Positive data-parallel group size.
        shuffle_seed: Deterministic shuffle seed, or None to preserve record order.

    Returns:
        This rank's ceil(len(records) / world_size) records.
    """
    if world_size <= 0 or rank < 0 or rank >= world_size:
        raise ValueError("world_size must be positive and rank must be in [0, world_size)")
    items = list(records)
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(items)
    if not items:
        return []
    total = math.ceil(len(items) / world_size) * world_size
    padded = (items * math.ceil(total / len(items)))[:total]
    return padded[rank::world_size]


def load_records(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    """Read a JSONL split, requiring unique nonempty stringified record IDs.

    Validate before distributed padding, which may intentionally repeat records.
    Numeric IDs are preserved but share an identity with their string equivalent.

    Args:
        path: JSONL file containing one unpadded input split.
        limit: Optional maximum number of records to read and validate.

    Returns:
        Loaded records in file order, preserving their original ID values.

    Raises:
        ValueError: If a loaded record has a missing, empty, or duplicate ID.
    """
    records = []
    seen_ids: set[str] = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record_id = record.get("id")
            if record_id is None or not str(record_id).strip():
                raise ValueError(f"{path}: each record must have a nonempty record ID")
            canonical_id = str(record_id)
            if canonical_id in seen_ids:
                raise ValueError(f"{path}: duplicate record ID {canonical_id!r}")
            seen_ids.add(canonical_id)
            records.append(record)
            if limit is not None and len(records) >= limit:
                break
    return records


def validate_record(record: dict[str, Any]) -> None:
    """Raise ``ValueError`` if a record is not a well-formed SystemOne example."""
    if "state" not in record:
        raise ValueError(f"record {record.get('id')!r} has no state")
    questions = record.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise ValueError(f"record {record.get('id')!r} has no questions")
    for qid, question in questions.items():
        qtype = question.get("type")
        if qtype not in QUESTION_TYPES:
            raise ValueError(f"{qid}: type must be one of {QUESTION_TYPES}, got {qtype!r}")
        criteria = question.get("criteria")
        if qtype == "choice" and (not isinstance(criteria, dict) or not criteria):
            raise ValueError(f"{qid}: choice criteria must be a nonempty option mapping")
        if qtype == "score" and (not isinstance(criteria, list) or len(criteria) < 2):
            raise ValueError(f"{qid}: score criteria must be a list with at least two levels")
    for qid, answer in (record.get("answers") or {}).items():
        if qid not in questions:
            raise ValueError(f"answer for unknown question {qid!r}")
        if str(answer) not in canonical_option_ids(questions[qid]):
            raise ValueError(f"{qid}: answer {answer!r} is not an allowed option")
    for qid, rewards in (record.get("rewards") or {}).items():
        if qid not in questions:
            raise ValueError(f"record {record.get('id')!r}: rewards for unknown question {qid!r}")
        options = set(canonical_option_ids(questions[qid]))
        if not isinstance(rewards, dict) or set(rewards) != options:
            raise ValueError(
                f"record {record.get('id')!r}, question {qid!r}: rewards must cover exactly {sorted(options)}"
            )
        try:
            finite = all(math.isfinite(float(value)) for value in rewards.values())
        except (TypeError, ValueError) as exc:
            raise ValueError(f"record {record.get('id')!r}, question {qid!r}: rewards must be finite numbers") from exc
        if not finite:
            raise ValueError(f"record {record.get('id')!r}, question {qid!r}: rewards must be finite numbers")


def canonical_option_ids(question: dict[str, Any]) -> list[str]:
    """All option ids of a question (order-free set semantics; used for validation)."""
    qtype = question["type"]
    if qtype == "noul":
        return ["true", "false"]
    if qtype == "choice":
        return [str(k) for k in question["criteria"]]
    return [str(i) for i in range(len(question["criteria"]))]


@dataclass(frozen=True)
class QuestionTarget:
    """Supervision for one question, aligned with a model-specific option order."""

    question_id: str
    question_type: int
    option_ids: tuple[str, ...]
    label: int  # index into option_ids, or -100 when the record has no gold answer
    rewards: tuple[float, ...] | None  # aligned with option_ids


def question_targets(record: dict[str, Any], option_orders: dict[str, Iterable[str]]) -> list[QuestionTarget]:
    """Align a record's answers and rewards with the option order a model uses.

    Args:
        record: SystemOne record.
        option_orders: ``{question_id: option ids in model order}``, in the order the model returns questions.
    """
    answers = record.get("answers") or {}
    rewards = record.get("rewards") or {}
    targets = []
    for qid, option_ids in option_orders.items():
        option_ids = tuple(str(o) for o in option_ids)
        question = record["questions"][qid]
        answer = answers.get(qid)
        label = option_ids.index(str(answer)) if answer is not None else -100
        reward_table = rewards.get(qid)
        aligned = tuple(float(reward_table[o]) for o in option_ids) if reward_table is not None else None
        targets.append(
            QuestionTarget(
                question_id=qid,
                question_type=QUESTION_TYPE_IDS[question["type"]],
                option_ids=option_ids,
                label=label,
                rewards=aligned,
            )
        )
    return targets
