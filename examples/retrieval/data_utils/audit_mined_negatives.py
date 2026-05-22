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

"""Audit mined retrieval negatives for common false-negative and score issues."""

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _doc_id(doc: Any) -> str:
    """Return a document ID from either a raw ID or a {"id": ...} mapping."""
    raw_id = doc.get("id") if isinstance(doc, dict) else doc
    if raw_id is None:
        raise ValueError(f"Document entry is missing an id: {doc!r}")
    return str(raw_id)


def _score_state(doc: Any) -> str:
    """Classify a mined negative score as finite, missing, or non-finite."""
    if not isinstance(doc, dict) or "score" not in doc:
        return "missing"
    try:
        score = float(doc["score"])
    except (TypeError, ValueError):
        return "non_finite"
    return "finite" if math.isfinite(score) else "non_finite"


def audit_records(
    records: list[dict[str, Any]],
    *,
    drop_invalid_negatives: bool = False,
    max_findings: int = 20,
) -> tuple[dict[str, int], list[dict[str, Any]], list[dict[str, Any]]]:
    """Audit retrieval records and optionally drop invalid mined negatives.

    Args:
        records: Retrieval training records from the top-level ``data`` field.
        drop_invalid_negatives: Drop negatives that duplicate positives, duplicate
            another negative in the same row, or have a non-finite score.
        max_findings: Maximum example findings to return.

    Returns:
        A tuple of ``(summary, cleaned_records, finding_examples)``.
    """
    summary = {
        "records": len(records),
        "negatives": 0,
        "rows_with_findings": 0,
        "negative_is_known_positive": 0,
        "duplicate_negative": 0,
        "missing_negative_score": 0,
        "non_finite_negative_score": 0,
        "dropped_negatives": 0,
        "total_findings": 0,
    }
    cleaned_records = []
    findings = []

    for row_idx, record in enumerate(records):
        pos_ids = {_doc_id(doc) for doc in record.get("pos_doc", [])}
        seen_neg_ids = set()
        cleaned_negatives = []
        row_has_findings = False

        for neg_doc in record.get("neg_doc", []):
            summary["negatives"] += 1
            neg_id = _doc_id(neg_doc)
            should_drop = False

            if neg_id in pos_ids:
                summary["negative_is_known_positive"] += 1
                should_drop = True
                row_has_findings = True
                _append_finding(findings, max_findings, row_idx, record, neg_id, "negative_is_known_positive")

            if neg_id in seen_neg_ids:
                summary["duplicate_negative"] += 1
                should_drop = True
                row_has_findings = True
                _append_finding(findings, max_findings, row_idx, record, neg_id, "duplicate_negative")
            seen_neg_ids.add(neg_id)

            score_state = _score_state(neg_doc)
            if score_state == "missing":
                summary["missing_negative_score"] += 1
                row_has_findings = True
                _append_finding(findings, max_findings, row_idx, record, neg_id, "missing_negative_score")
            elif score_state == "non_finite":
                summary["non_finite_negative_score"] += 1
                should_drop = True
                row_has_findings = True
                _append_finding(findings, max_findings, row_idx, record, neg_id, "non_finite_negative_score")

            if drop_invalid_negatives and should_drop:
                summary["dropped_negatives"] += 1
                continue
            cleaned_negatives.append(neg_doc)

        if row_has_findings:
            summary["rows_with_findings"] += 1
        cleaned_record = dict(record)
        cleaned_record["neg_doc"] = cleaned_negatives
        cleaned_records.append(cleaned_record)

    summary["total_findings"] = (
        summary["negative_is_known_positive"]
        + summary["duplicate_negative"]
        + summary["missing_negative_score"]
        + summary["non_finite_negative_score"]
    )
    return summary, cleaned_records, findings


def _append_finding(
    findings: list[dict[str, Any]],
    max_findings: int,
    row_idx: int,
    record: dict[str, Any],
    neg_id: str,
    issue: str,
) -> None:
    """Append a compact finding example up to the configured limit."""
    if len(findings) >= max_findings:
        return
    finding = {
        "row": row_idx,
        "question_id": record.get("question_id"),
        "negative_id": neg_id,
        "issue": issue,
    }
    if "original_question_id" in record:
        finding["original_question_id"] = record["original_question_id"]
    findings.append(finding)


def audit_training_data(
    training_data: dict[str, Any],
    *,
    drop_invalid_negatives: bool = False,
    max_findings: int = 20,
) -> tuple[dict[str, int], dict[str, Any], list[dict[str, Any]]]:
    """Audit a top-level retrieval JSON object."""
    records = training_data.get("data", [])
    summary, cleaned_records, findings = audit_records(
        records,
        drop_invalid_negatives=drop_invalid_negatives,
        max_findings=max_findings,
    )
    cleaned_training_data = dict(training_data)
    cleaned_training_data["data"] = cleaned_records
    return summary, cleaned_training_data, findings


def main() -> int:
    """Run the mined-negative audit CLI."""
    parser = argparse.ArgumentParser(description="Audit mined retrieval negatives before reusing them for training")
    parser.add_argument("input_file", type=str, help="Path to mined retrieval JSON")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Optional path to write a cleaned retrieval JSON",
    )
    parser.add_argument(
        "--drop-invalid-negatives",
        action="store_true",
        help="Drop negatives that are known positives, duplicate row negatives, or have non-finite scores",
    )
    parser.add_argument("--max-findings", type=int, default=20, help="Maximum finding examples to print")
    parser.add_argument(
        "--allow-findings",
        action="store_true",
        help="Exit with status 0 even when the audit reports findings",
    )
    args = parser.parse_args()

    input_path = Path(args.input_file)
    with open(input_path, "r") as f:
        training_data = json.load(f)

    summary, cleaned_training_data, findings = audit_training_data(
        training_data,
        drop_invalid_negatives=args.drop_invalid_negatives,
        max_findings=args.max_findings,
    )

    print(json.dumps({"summary": summary, "findings": findings}, indent=2))

    if args.output is not None:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(cleaned_training_data, f, indent=2)

    return 1 if summary["total_findings"] and not args.allow_findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
