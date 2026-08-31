# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compare HY4 sampled logits from two distributed-training topologies."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

import torch
import torch.nn.functional as F


def _input_hashes(artifact_dir: Path) -> dict[tuple[int, int], str]:
    """Map ``(dp_rank, forward_call)`` to exact packed-input hashes."""
    result: dict[tuple[int, int], str] = {}
    for path in sorted(artifact_dir.glob("input-rank*-call*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        key = (int(record["dp_rank"]), int(record["call"]))
        digest = str(record["input_sha256"])
        previous = result.setdefault(key, digest)
        if previous != digest:
            raise ValueError(f"Conflicting packed-input hashes for {key} in {artifact_dir}.")
    return result


def _logit_records(artifact_dir: Path, calls_per_step: int) -> dict[tuple[int, str], dict[str, Any]]:
    """Load sampled logits shaped ``[positions, vocab]`` keyed by step and input hash."""
    input_hashes = _input_hashes(artifact_dir)
    records: dict[tuple[int, str], dict[str, Any]] = {}
    for path in sorted(artifact_dir.glob("logits-rank*-call*.pt")):
        record = torch.load(path, map_location="cpu", weights_only=True)
        call = int(record["call"])
        digest = record.get("input_sha256") or input_hashes.get((int(record["dp_rank"]), call))
        if digest is None:
            raise ValueError(f"No packed-input hash found for {path}.")
        key = (call // calls_per_step, str(digest))
        if key in records:
            raise ValueError(f"Duplicate logits for step/input key {key} in {artifact_dir}.")
        record["path"] = str(path)
        records[key] = record
    if not records:
        raise ValueError(f"No logits artifacts found in {artifact_dir}.")
    return records


def _percentile(values: list[float], quantile: float) -> float:
    """Return a nearest-rank percentile from a non-empty scalar list."""
    if not values:
        raise ValueError("Cannot compute a percentile from an empty list.")
    ordered = sorted(values)
    index = min(math.ceil(quantile * len(ordered)) - 1, len(ordered) - 1)
    return ordered[max(index, 0)]


def compare(
    reference_dir: Path,
    candidate_dir: Path,
    *,
    reference_calls_per_step: int,
    candidate_calls_per_step: int,
) -> dict[str, Any]:
    """Compare paired ``[positions, vocab]`` logits and return aggregate metrics."""
    reference = _logit_records(reference_dir, reference_calls_per_step)
    candidate = _logit_records(candidate_dir, candidate_calls_per_step)
    if reference.keys() != candidate.keys():
        missing = sorted(reference.keys() - candidate.keys())
        extra = sorted(candidate.keys() - reference.keys())
        raise ValueError(f"Topology runs used different packed samples: missing={missing[:8]}, extra={extra[:8]}.")

    token_kls: list[float] = []
    sample_cosines: list[float] = []
    max_abs = 0.0
    top1_equal = 0
    top1_total = 0
    per_step: dict[int, dict[str, list[float] | int]] = {}
    for key in sorted(reference):
        ref = reference[key]
        cand = candidate[key]
        if ref["full_logits_shape"] != cand["full_logits_shape"]:
            raise ValueError(
                f"Full logits shape mismatch for {key}: {ref['full_logits_shape']} vs {cand['full_logits_shape']}."
            )
        if not torch.equal(ref["positions"], cand["positions"]):
            raise ValueError(f"Sampled token positions differ for {key}.")

        # The flattened cosine spans millions of vocabulary logits, while the
        # KL values are small differences between nearly identical
        # distributions.  Accumulate both in FP64 so floating-point reduction
        # error cannot produce an invalid cosine greater than one or dominate
        # the measured topology difference.
        ref_logits = ref["logits"].double()
        cand_logits = cand["logits"].double()
        if ref_logits.shape != cand_logits.shape:
            raise ValueError(f"Sampled logits shape mismatch for {key}: {ref_logits.shape} vs {cand_logits.shape}.")
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
        cand_log_probs = F.log_softmax(cand_logits, dim=-1)
        kls = torch.sum(ref_log_probs.exp() * (ref_log_probs - cand_log_probs), dim=-1)
        kls_list = [float(value) for value in kls]
        token_kls.extend(kls_list)
        cosine = float(F.cosine_similarity(ref_logits.flatten(), cand_logits.flatten(), dim=0))
        sample_cosines.append(cosine)
        max_abs = max(max_abs, float((ref_logits - cand_logits).abs().max()))
        top1_equal += int((ref_logits.argmax(dim=-1) == cand_logits.argmax(dim=-1)).sum())
        top1_total += ref_logits.shape[0]

        step = key[0]
        step_metrics = per_step.setdefault(step, {"kls": [], "cosines": [], "samples": 0})
        step_metrics["kls"].extend(kls_list)
        step_metrics["cosines"].append(cosine)
        step_metrics["samples"] += 1

    per_step_report = {
        str(step): {
            "samples": values["samples"],
            "mean_kl": mean(values["kls"]),
            "max_kl": max(values["kls"]),
            "mean_cosine": mean(values["cosines"]),
        }
        for step, values in sorted(per_step.items())
    }
    return {
        "reference_dir": str(reference_dir),
        "candidate_dir": str(candidate_dir),
        "sample_count": len(reference),
        "token_position_count": len(token_kls),
        "mean_kl": mean(token_kls),
        "p95_kl": _percentile(token_kls, 0.95),
        "max_kl": max(token_kls),
        "mean_cosine": mean(sample_cosines),
        "min_cosine": min(sample_cosines),
        "max_abs_logit_diff": max_abs,
        "top1_agreement": top1_equal / top1_total,
        "per_step": per_step_report,
    }


def main() -> None:
    """Parse artifact paths, enforce parity thresholds, and emit a JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_dir", type=Path, help="EP8/PP1 reference artifact directory")
    parser.add_argument("candidate_dir", type=Path, help="EP4/PP2 candidate artifact directory")
    parser.add_argument("--reference-calls-per-step", type=int, default=1)
    parser.add_argument("--candidate-calls-per-step", type=int, default=2)
    parser.add_argument("--mean-kl-tol", type=float, default=1.0e-3)
    parser.add_argument("--max-kl-tol", type=float, default=0.1)
    parser.add_argument("--cosine-min", type=float, default=0.999)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = compare(
        args.reference_dir,
        args.candidate_dir,
        reference_calls_per_step=args.reference_calls_per_step,
        candidate_calls_per_step=args.candidate_calls_per_step,
    )
    serialized = json.dumps(report, indent=2, sort_keys=True)
    print(serialized)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")

    failures = []
    if report["mean_kl"] > args.mean_kl_tol:
        failures.append(f"mean KL {report['mean_kl']:.6g} > {args.mean_kl_tol:.6g}")
    if report["max_kl"] > args.max_kl_tol:
        failures.append(f"max KL {report['max_kl']:.6g} > {args.max_kl_tol:.6g}")
    if report["min_cosine"] < args.cosine_min:
        failures.append(f"min cosine {report['min_cosine']:.6g} < {args.cosine_min:.6g}")
    if failures:
        raise SystemExit("HY4 topology logits parity failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
