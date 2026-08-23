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

"""Opt-in reference shape-sensitivity diagnostics for checkpoint parity."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch

from tests.functional_tests.checkpoint_robustness.parity_metrics import _compute_parity_metrics


@dataclass(frozen=True)
class _ShapeDiagnosticConfig:
    """Validated configuration for non-gating vanilla-HF shape probes."""

    sweep_lengths: tuple[int, ...] = ()

    def lengths(self, *, parity_sequence_length: int, gate_sequence_length: int | None) -> tuple[int, ...]:
        """Return unique standalone-forward lengths, including the standing gate probe."""
        lengths = set(self.sweep_lengths)
        if gate_sequence_length is not None and gate_sequence_length < parity_sequence_length:
            lengths.add(gate_sequence_length)
        return tuple(sorted(lengths))


def _normalize_shape_diagnostic_config(
    raw_config: object,
    *,
    parity_sequence_length: int,
) -> _ShapeDiagnosticConfig:
    """Validate a ``ci.checkpoint_robustness.shape_diagnostic`` mapping."""
    if raw_config is None:
        return _ShapeDiagnosticConfig()
    if not isinstance(raw_config, Mapping):
        raise ValueError("shape_diagnostic must be a mapping with an optional sweep_lengths field")
    unknown_fields = set(raw_config) - {"sweep_lengths"}
    if unknown_fields:
        raise ValueError(f"Unknown shape_diagnostic fields: {sorted(unknown_fields)}")

    raw_lengths = raw_config.get("sweep_lengths", ())
    if not isinstance(raw_lengths, (list, tuple)):
        raise ValueError("shape_diagnostic.sweep_lengths must be a list of positive integers")

    lengths: list[int] = []
    for raw_length in raw_lengths:
        if isinstance(raw_length, bool) or not isinstance(raw_length, int):
            raise ValueError("shape_diagnostic.sweep_lengths must contain only positive integers")
        if raw_length <= 0 or raw_length >= parity_sequence_length:
            raise ValueError(
                "shape_diagnostic.sweep_lengths entries must be shorter than parity_sequence_length "
                f"({raw_length} is invalid for {parity_sequence_length})"
            )
        lengths.append(raw_length)
    return _ShapeDiagnosticConfig(sweep_lengths=tuple(sorted(set(lengths))))


def _build_shape_diagnostic_report(
    base_logits: torch.Tensor,
    standalone_logits: Mapping[int, torch.Tensor],
    *,
    parity_document_sha256: str,
    phase: str,
    gate_sequence_length: int | None,
    sweep_lengths: tuple[int, ...] = (),
    router_diagnostics: Mapping[int, Mapping[str, object]] | None = None,
) -> dict[str, object]:
    """Build HF-full-prefix versus HF-standalone shape-sensitivity evidence."""
    base_sequence_length = base_logits.shape[-2]
    points: dict[int, dict[str, object]] = {}
    for sequence_length, candidate_logits in sorted(standalone_logits.items()):
        if sequence_length <= 0 or sequence_length >= base_sequence_length:
            raise ValueError(f"Shape diagnostic length must be in [1, {base_sequence_length}), got {sequence_length}")
        reference_prefix = base_logits[..., :sequence_length, :]
        if reference_prefix.shape != candidate_logits.shape:
            raise ValueError(
                f"Shape diagnostic length {sequence_length} has mismatched logits: "
                f"{tuple(reference_prefix.shape)} != {tuple(candidate_logits.shape)}"
            )
        bitwise_equal = torch.equal(reference_prefix, candidate_logits)
        purposes = []
        if sequence_length in sweep_lengths:
            purposes.append("calibration_sweep")
        if gate_sequence_length == sequence_length:
            purposes.append("standing_gate_context")
        point: dict[str, object] = {
            "sequence_length": sequence_length,
            "purposes": purposes,
            "comparison": "hf_full_prefix_vs_hf_standalone",
            "bitwise_equal": bitwise_equal,
            "interpretation": (
                "same_kernel_regime_not_probing" if bitwise_equal else "shape_sensitive_numeric_path_observed"
            ),
            "metrics": _compute_parity_metrics(reference_prefix, candidate_logits).to_dict(),
        }
        if router_diagnostics is not None and sequence_length in router_diagnostics:
            point["router_diagnostics"] = dict(router_diagnostics[sequence_length])
        points[sequence_length] = point

    return {
        "schema_version": 1,
        "diagnostic": "vanilla_hf_shape_sensitivity",
        "phase": phase,
        "enforced": False,
        "parity_document_sha256": parity_document_sha256,
        "base_sequence_length": base_sequence_length,
        "gate_sequence_length": gate_sequence_length,
        "points": points,
    }


def _persist_shape_diagnostic_report(report: Mapping[str, object], report_path: Path) -> None:
    """Persist full shape evidence and print one concise CI summary line."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_report_path = report_path.with_suffix(".tmp")
    temporary_report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary_report_path.replace(report_path)

    points = report["points"]
    assert isinstance(points, Mapping)
    summary = {
        "phase": report["phase"],
        "base_sequence_length": report["base_sequence_length"],
        "enforced": False,
        "points": {
            str(length): {
                "bitwise_equal": point["bitwise_equal"],
                "mean_kl": point["metrics"]["mean_kl"],
                "p95_kl": point["metrics"]["p95_kl"],
                "self_flip_token_count": (point.get("router_diagnostics") or {}).get("tokens_with_any_flip_count"),
                "sustained_flip_mean_kl": (
                    (point.get("router_diagnostics") or {}).get("sustained_flip_final_token_kl") or {}
                ).get("mean_kl"),
            }
            for length, point in points.items()
        },
        "report_path": str(report_path),
    }
    print(f"CHECKPOINT_SHAPE_DIAGNOSTICS {json.dumps(summary, sort_keys=True)}")
