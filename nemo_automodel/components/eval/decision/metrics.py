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

"""Evaluation metrics for decision models: accuracy, joint exact match, decision reward, and calibration."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Protocol, Sequence


class EvaluationTargetProtocol(Protocol):
    """Read-only supervision required by this component."""

    @property
    def question_id(self) -> str:
        """Question identity within its record."""
        ...

    @property
    def option_ids(self) -> tuple[str, ...]:
        """Option identities in model-output order."""
        ...

    @property
    def label(self) -> int:
        """Gold option index, or -100 when unlabelled."""
        ...

    @property
    def rewards(self) -> tuple[float, ...] | None:
        """Optional utilities aligned with the option identities."""
        ...


class DecisionEvaluator:
    """Accumulates per-question predictions and summarises them.

    Call :meth:`add` once per (record, question) with the model's probabilities in the model's option order.
    """

    def __init__(self, slice_keys: Sequence[str] = (), num_bins: int = 10) -> None:
        """Set metadata slices and the number of confidence calibration bins."""
        if num_bins < 1:
            raise ValueError("num_bins must be positive")
        self.slice_keys = tuple(slice_keys)
        self.num_bins = num_bins
        self.rows: list[dict[str, Any]] = []

    def add(
        self,
        record_id: str,
        question_id: str,
        option_ids: Sequence[str],
        probabilities: Sequence[float],
        label: int,
        rewards: Sequence[float] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        """Add one identified decision with probabilities in option_ids order.

        Args:
            record_id: Input record identity, shared by its questions.
            question_id: Question identity within that record.
            option_ids: Allowed options in model-output order.
            probabilities: Probability for each allowed option, in the same order.
            label: Correct option index, or a negative value for unlabelled data.
            rewards: Optional utility for each option in the same order.
            meta: Optional record metadata used to group evaluation slices.
        """
        probs = [float(p) for p in probabilities]
        if not probs or len(probs) != len(option_ids):
            raise ValueError("probabilities must have one value per nonempty option_ids")
        if label >= len(probs):
            raise ValueError("label must be a valid option index or negative for unlabelled data")
        if any(not math.isfinite(p) or p < 0 for p in probs) or not math.isclose(sum(probs), 1.0, abs_tol=1e-5):
            raise ValueError("probabilities must be finite, nonnegative and sum to one")
        if rewards is not None and (len(rewards) != len(probs) or any(not math.isfinite(r) for r in rewards)):
            raise ValueError("rewards must have one finite value per option")
        pred = max(range(len(probs)), key=probs.__getitem__)
        self.rows.append(
            {
                "record_id": record_id,
                "question_id": question_id,
                "option_ids": list(option_ids),
                "probs": probs,
                "pred": pred,
                "label": int(label),
                "rewards": list(rewards) if rewards is not None else None,
                "meta": meta or {},
            }
        )

    def add_batch(
        self,
        records: Sequence[dict[str, Any]],
        targets_per_record: Sequence[Sequence[EvaluationTargetProtocol]],
        probs_per_record: Sequence[Sequence[Sequence[float]]],
    ) -> None:
        """Accumulate records with structurally typed targets and aligned probabilities.

        Args:
            records: Input records with IDs and optional slicing metadata.
            targets_per_record: Targets grouped by record, in model-output order.
            probs_per_record: Option probabilities grouped by record and question,
                in each target's option order.
        """
        for record, targets, probs in zip(records, targets_per_record, probs_per_record, strict=True):
            for target, p in zip(targets, probs, strict=True):
                self.add(
                    str(record.get("id")),
                    target.question_id,
                    target.option_ids,
                    p,
                    target.label,
                    target.rewards,
                    record.get("meta"),
                )

    # ------------------------------------------------------------------------------------------------------------------
    def _question_summary(self, rows: list[dict]) -> dict[str, float]:
        labelled = [r for r in rows if r["label"] >= 0]
        out: dict[str, float] = {"n": float(len(rows))}
        if labelled:
            out["accuracy"] = sum(r["pred"] == r["label"] for r in labelled) / len(labelled)
            out["nll"] = sum(-math.log(max(r["probs"][r["label"]], 1e-12)) for r in labelled) / len(labelled)
            out["brier"] = sum(
                sum((p - (i == r["label"])) ** 2 for i, p in enumerate(r["probs"])) for r in labelled
            ) / len(labelled)
            out["ece"] = self._ece(labelled)
        rewarded = [r for r in rows if r["rewards"] is not None]
        if rewarded:
            out["reward"] = sum(r["rewards"][r["pred"]] for r in rewarded) / len(rewarded)
            out["expected_reward"] = sum(sum(p * x for p, x in zip(r["probs"], r["rewards"])) for r in rewarded) / len(
                rewarded
            )
        return out

    def _ece(self, rows: list[dict]) -> float:
        bins: list[list[tuple[float, bool]]] = [[] for _ in range(self.num_bins)]
        for r in rows:
            conf = r["probs"][r["pred"]]
            bins[min(int(conf * self.num_bins), self.num_bins - 1)].append((conf, r["pred"] == r["label"]))
        total = len(rows)
        return sum(
            len(b) / total * abs(sum(c for c, _ in b) / len(b) - sum(ok for _, ok in b) / len(b)) for b in bins if b
        )

    def _exact_match(self, rows: list[dict]) -> float:
        by_record: dict[str, bool] = {}
        for r in rows:
            if r["label"] < 0:
                continue
            by_record[r["record_id"]] = by_record.get(r["record_id"], True) and r["pred"] == r["label"]
        return sum(by_record.values()) / max(len(by_record), 1)

    def summary(self) -> dict[str, Any]:
        """Return ``{"overall": {...}, "questions": {qid: {...}}, "slices": {key: {value: {...}}}}``."""
        by_question: dict[str, list[dict]] = defaultdict(list)
        for r in self.rows:
            by_question[r["question_id"]].append(r)
        questions = {qid: self._question_summary(rows) for qid, rows in by_question.items()}
        overall = self._question_summary(self.rows)
        overall["exact_match"] = self._exact_match(self.rows)
        overall["records"] = float(len({r["record_id"] for r in self.rows}))
        slices: dict[str, dict[str, dict[str, float]]] = {}
        for key in self.slice_keys:
            groups: dict[str, list[dict]] = defaultdict(list)
            for r in self.rows:
                groups[str(r["meta"].get(key))].append(r)
            slices[key] = {
                value: {**self._question_summary(rows), "exact_match": self._exact_match(rows)}
                for value, rows in sorted(groups.items())
            }
        return {"overall": overall, "questions": questions, "slices": slices}

    def predictions(self) -> list[dict[str, Any]]:
        """Return the accumulated identified decisions and their aligned probabilities."""
        return list(self.rows)


def save_summary(summary: dict[str, Any], path: str | Path) -> None:
    """Write a :meth:`DecisionEvaluator.summary` dictionary as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2))
