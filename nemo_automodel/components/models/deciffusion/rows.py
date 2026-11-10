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

"""Training rows for Deciffusion decision post-training: one text question, one label, optional soft labels.

A row is ``{"id", "input": {"state", "questions": {key: question}}, "reference": {"target"}, "soft_labels"?, "group"?}``.
Rows come from two public sources: SystemOne records (several questions per state, flattened here to one row per
question) and simple one-decision-per-line JSONL (``state``, ``question``, ``type``, ``options``, ``label``).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from nemo_automodel.components.models.deciffusion.serving_prompt import (
    PreparedQuestion,
    prepare_question,
    question_options,
)

NOUL_ALIASES = {"true": "true", "yes": "true", "1": "true", "false": "false", "no": "false", "0": "false"}


@dataclass(frozen=True)
class DecisionExample:
    """A tokenized training/evaluation example.

    Attributes:
        row_id: Identifier of the source row.
        prompt: The serving prompt (labels in prompt order, option codes, token ids without the mask).
        gold: Index of the gold label within ``prompt.labels``.
        soft: Normalized soft-label distribution aligned with ``prompt.labels`` or ``None``.
    """

    row_id: str
    prompt: PreparedQuestion
    gold: int
    soft: tuple[float, ...] | None


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read one JSON value per ``\\n``-terminated line (not ``str.splitlines``: U+2028 is legal inside JSON strings)."""
    rows = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").split("\n"), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{number}: invalid JSON row: {error}") from None
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write rows as UTF-8 JSONL, creating parent directories."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_label(kind: str, label: Any) -> str:
    """Normalize a gold/soft label key (noul labels are lower-case ``true``/``false``)."""
    if kind == "noul":
        normalized = NOUL_ALIASES.get(str(label).strip().lower())
        if normalized is None:
            raise ValueError(f"noul label must be true/false, got {label!r}")
        return normalized
    return str(label)


def convert_simple_row(line: dict[str, Any], number: int = 0) -> dict[str, Any]:
    """Convert one simple decision line into a training row.

    Raises:
        ValueError: If a field is missing or the label/options are inconsistent.
    """
    for name in ("state", "question", "label"):
        if name not in line:
            raise ValueError(f'missing "{name}"')
    kind = line.get("type", "choice")
    options = line.get("options")
    question: dict[str, Any] = {"type": kind, "instructions": str(line["question"])}
    if kind == "choice":
        if isinstance(options, list):
            options = {str(o): str(o) for o in options}
        if not isinstance(options, dict) or len(options) < 2:
            raise ValueError('choice needs "options": {key: description} with at least 2 entries')
        question["criteria"] = {str(k): str(v) for k, v in options.items()}
    elif kind == "noul":
        if options:
            question["criteria"] = {"true": str(options["true"]), "false": str(options["false"])}
    elif kind == "score":
        if not isinstance(options, list) or len(options) < 2:
            raise ValueError('score needs "options": [level 0 description, level 1, ...]')
        question["criteria"] = [str(o) for o in options]
    else:
        raise ValueError(f"unknown type {kind!r}")
    labels, _ = question_options(question)
    label = normalize_label(kind, line["label"])
    if label not in labels:
        raise ValueError(f"label {label!r} is not one of {labels}")
    row: dict[str, Any] = {
        "id": str(line.get("id", f"row-{number}")),
        "input": {"state": line["state"], "questions": {"decision": question}},
        "reference": {"target": label},
    }
    soft = line.get("soft_labels")
    if soft:
        soft = {normalize_label(kind, k): float(v) for k, v in soft.items()}
        if set(soft) - set(labels) or any(v < 0 for v in soft.values()) or sum(soft.values()) <= 0:
            raise ValueError(f"soft_labels must be non-negative over the labels {labels}")
        total = sum(soft.values())
        row["soft_labels"] = {name: soft.get(name, 0.0) / total for name in labels}
    if "group" in line:
        row["group"] = str(line["group"])
    return row


def rows_from_systemone(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten a SystemOne record (``state``, ``questions``, ``answers``) to one training row per answered question.

    Questions without an answer are skipped. ``group`` is the record id, so all questions of a state stay together.
    """
    rows = []
    for key, question in record["questions"].items():
        if key not in record.get("answers", {}):
            continue
        rows.append(
            {
                "id": f"{record['id']}/{key}",
                "input": {"state": record["state"], "questions": {key: question}},
                "reference": {"target": normalize_label(question["type"], record["answers"][key])},
                "group": str(record["id"]),
            }
        )
    return rows


def in_heldout(row: dict[str, Any], fraction: float, seed: str | int) -> bool:
    """Deterministic group-disjoint split: hash of ``seed|group`` (or the state) below ``fraction``."""
    key = row.get("group") or json.dumps(row["input"]["state"], sort_keys=True, ensure_ascii=False)
    digest = int(hashlib.sha256(f"{seed}|{key}".encode()).hexdigest()[:8], 16) / 16**8
    return digest < fraction


def has_image(row: dict[str, Any]) -> bool:
    """True for image rows (top-level image(s) or ``state.image``): never trained as text decisions."""
    state = (row.get("input") or {}).get("state")
    return bool(row.get("image") or row.get("images") or (isinstance(state, dict) and state.get("image")))


def encode_row(
    tokenizer: Any,
    row: dict[str, Any],
    *,
    max_tokens: int,
    shuffle_seed: int | None = None,
    soft_gold_mask: bool = False,
) -> DecisionExample | None:
    """Encode a row as a serving prompt, or ``None`` if it is not a trainable text decision.

    Rows with several questions, images, no label, a gold label outside the options, a malformed question or a prompt
    (plus its mask) longer than ``max_tokens`` are not encodable; the caller counts and reports them.

    Args:
        tokenizer: Tokenizer of the checkpoint.
        row: Training row.
        max_tokens: Longest allowed prompt including the mask slot.
        shuffle_seed: Option-shuffle seed (``None`` keeps the serving order).
        soft_gold_mask: Drop soft labels whose mode differs from the gold label (the row then trains on gold alone).
    """
    questions = (row.get("input") or {}).get("questions")
    if not isinstance(questions, dict) or len(questions) != 1 or has_image(row):
        return None
    if "target" not in (row.get("reference") or {}):
        return None
    key, question = next(iter(questions.items()))
    try:
        prompt = prepare_question(tokenizer, row["input"]["state"], key, question, shuffle_seed=shuffle_seed)
        gold = normalize_label(prompt.kind, row["reference"]["target"])
    except ValueError:
        return None
    if gold not in prompt.labels or len(prompt.input_ids) + 1 > max_tokens:
        return None
    soft: tuple[float, ...] | None = None
    given = row.get("soft_labels")
    if given:
        values = [float(given.get(label, 0.0)) for label in prompt.labels]
        total = sum(values)
        if total > 0:
            soft = tuple(v / total for v in values)
        if (
            soft is not None
            and soft_gold_mask
            and max(range(len(soft)), key=soft.__getitem__) != prompt.labels.index(gold)
        ):
            soft = None
    return DecisionExample(str(row.get("id")), prompt, prompt.labels.index(gold), soft)
