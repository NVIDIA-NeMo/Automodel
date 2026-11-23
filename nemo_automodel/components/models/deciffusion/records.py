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

"""SystemOne records for Deciffusion decision post-training: one-question encoding and one-decision JSONL conversion.

The recipe reads SystemOne records (``{"id", "state", "questions", "answers"}`` plus optional ``rewards``,
``soft_labels`` and ``meta``) and reads every question of a record through its own serving prompt. Simple
one-decision-per-line JSONL (``state``, ``question``, ``type``, ``options``, ``label``, optional ``soft_labels`` and
``group``) converts to single-question records, with ``group`` kept in ``meta`` for group-disjoint splits.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from nemo_automodel.components.models.deciffusion.serving_prompt import (
    DecisionContract,
    PreparedQuestion,
    prepare_question,
    question_options,
)

NOUL_ALIASES = {"true": "true", "yes": "true", "1": "true", "false": "false", "no": "false", "0": "false"}
SIMPLE_QUESTION_ID = "decision"


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


def convert_simple_line(line: dict[str, Any], number: int = 0) -> dict[str, Any]:
    """Convert one simple decision line into a single-question SystemOne record.

    Args:
        line: ``{"state", "question", "label"}`` plus optional ``id``, ``type`` (default ``choice``), ``options``,
            ``soft_labels`` and ``group``.
        number: Line number, used for the record ID when the line has none.

    Returns:
        A record whose only question is ``"decision"``. Soft labels are normalized over all options, and ``group``
        is stored as ``meta["group"]``.

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
    record: dict[str, Any] = {
        "id": str(line.get("id", f"row-{number}")),
        "state": line["state"],
        "questions": {SIMPLE_QUESTION_ID: question},
        "answers": {SIMPLE_QUESTION_ID: label},
    }
    soft = line.get("soft_labels")
    if soft:
        soft = {normalize_label(kind, k): float(v) for k, v in soft.items()}
        if set(soft) - set(labels) or any(v < 0 for v in soft.values()) or sum(soft.values()) <= 0:
            raise ValueError(f"soft_labels must be non-negative over the labels {labels}")
        total = sum(soft.values())
        record["soft_labels"] = {SIMPLE_QUESTION_ID: {name: soft.get(name, 0.0) / total for name in labels}}
    if "group" in line:
        record["meta"] = {"group": str(line["group"])}
    return record


def in_heldout(record: dict[str, Any], fraction: float, seed: str | int) -> bool:
    """Deterministic group-disjoint split: hash of ``seed|meta.group`` (or the state) below ``fraction``."""
    key = (record.get("meta") or {}).get("group") or json.dumps(record["state"], sort_keys=True, ensure_ascii=False)
    digest = int(hashlib.sha256(f"{seed}|{key}".encode()).hexdigest()[:8], 16) / 16**8
    return digest < fraction


def has_media(record: dict[str, Any]) -> bool:
    """True for records with images or videos (top-level, or ``state.image``): never read as text decisions."""
    state = record.get("state")
    return bool(
        record.get("image")
        or record.get("images")
        or record.get("videos")
        or (isinstance(state, dict) and state.get("image"))
    )


def encode_question(
    tokenizer: Any,
    contract: DecisionContract,
    record: dict[str, Any],
    question_id: str,
    *,
    mask_token_id: int,
    max_tokens: int,
    shuffle_seed: int | None = None,
) -> PreparedQuestion | None:
    """Encode one question of a record as its decision prompt, or ``None`` if this model cannot read it.

    A question is not readable when the record carries media, the question does not fit the checkpoint's contract (no
    instructions, unsupported fields, or more options than codes), or the prompt plus its mask is longer than
    ``max_tokens``. The caller counts these questions.

    Args:
        tokenizer: Tokenizer of the checkpoint.
        contract: The checkpoint's decision contract.
        record: Validated SystemOne record that holds the question.
        question_id: Key of the question in ``record["questions"]``.
        mask_token_id: The checkpoint's mask token.
        max_tokens: Longest allowed prompt including the mask slot.
        shuffle_seed: Option-shuffle seed (``None`` keeps the checkpoint's option order).
    """
    if has_media(record):
        return None
    try:
        prompt = prepare_question(
            tokenizer,
            contract,
            record["state"],
            question_id,
            record["questions"][question_id],
            mask_token_id=mask_token_id,
            shuffle_seed=shuffle_seed,
        )
    except ValueError:
        return None
    if len(prompt.input_ids) + 1 > max_tokens:
        return None
    return prompt
