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

"""Shared full-logit metrics and numerical profiles for checkpoint parity."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal

import torch
import torch.nn.functional as F

_ComparisonKind = Literal["same_implementation", "cross_framework", "cross_topology"]
_PARITY_COMPARISONS = {"source_load", "automodel_reload", "hf_reload", "cross_tp"}
_PARITY_OVERRIDE_METRICS = {"mean_kl", "p95_kl", "cosine_similarity"}


@dataclass(frozen=True)
class _ParityMetrics:
    """Summary statistics for one full-logit comparison."""

    token_count: int
    vocab_size: int
    mean_kl: float
    p95_kl: float
    max_kl: float
    cosine_similarity: float
    mean_absolute_logit_difference: float
    max_absolute_logit_difference: float

    def to_dict(self) -> dict[str, int | float]:
        """Return a JSON-serializable metric mapping."""
        return asdict(self)


@dataclass(frozen=True)
class _ParityThresholds:
    """Mean KL, p95 KL, and cosine gates selected by one numerical profile."""

    mean_kl: float
    p95_kl: float
    cosine_similarity: float

    def to_dict(self) -> dict[str, float]:
        """Return a JSON-serializable threshold mapping."""
        return asdict(self)


_PARITY_PROFILES: dict[str, dict[_ComparisonKind, _ParityThresholds]] = {
    "strict": {
        "same_implementation": _ParityThresholds(mean_kl=1e-7, p95_kl=1e-6, cosine_similarity=0.999999),
        "cross_framework": _ParityThresholds(mean_kl=1e-4, p95_kl=1e-3, cosine_similarity=0.9999),
        "cross_topology": _ParityThresholds(mean_kl=1e-6, p95_kl=1e-5, cosine_similarity=0.99999),
    },
    "standard": {
        "same_implementation": _ParityThresholds(mean_kl=3e-3, p95_kl=1.2e-2, cosine_similarity=0.999),
        "cross_framework": _ParityThresholds(mean_kl=6e-3, p95_kl=3e-2, cosine_similarity=0.998),
        "cross_topology": _ParityThresholds(mean_kl=6e-3, p95_kl=3e-2, cosine_similarity=0.998),
    },
    "relaxed": {
        "same_implementation": _ParityThresholds(mean_kl=2e-2, p95_kl=5e-2, cosine_similarity=0.995),
        "cross_framework": _ParityThresholds(mean_kl=2.5e-2, p95_kl=1e-1, cosine_similarity=0.99),
        "cross_topology": _ParityThresholds(mean_kl=2e-2, p95_kl=5e-2, cosine_similarity=0.995),
    },
}


def _validate_logits(logits: torch.Tensor, *, chunk_tokens: int = 16) -> tuple[int, int]:
    """Validate a complete logit tensor without allocating another full-size tensor.

    Args:
        logits: Tensor of shape [..., vocab], with arbitrary leading token dimensions.
        chunk_tokens: Number of flattened tokens checked together.

    Returns:
        The flattened token count and vocabulary size.
    """
    if logits.ndim < 2 or logits.shape[-1] <= 0:
        raise ValueError(f"Expected logits of shape [..., vocab], got {tuple(logits.shape)}")
    if chunk_tokens <= 0:
        raise ValueError(f"chunk_tokens must be positive, got {chunk_tokens}")

    vocab_size = logits.shape[-1]
    flattened_logits = logits.detach().reshape(-1, vocab_size)
    token_count = flattened_logits.shape[0]
    if token_count == 0:
        raise ValueError("Logit tensor must contain at least one token")
    for start in range(0, token_count, chunk_tokens):
        end = min(start + chunk_tokens, token_count)
        if not bool(torch.isfinite(flattened_logits[start:end]).all()):
            raise ValueError(f"Logits contain non-finite values in flattened token range [{start}, {end})")
    return token_count, vocab_size


def _compute_parity_metrics(
    reference_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
    *,
    chunk_tokens: int = 16,
) -> _ParityMetrics:
    """Compute bounded-memory statistics over two complete logit tensors.

    Args:
        reference_logits: Reference tensor of shape [..., vocab], with arbitrary leading token dimensions.
        candidate_logits: Candidate tensor of shape [..., vocab], matching ``reference_logits`` exactly.
        chunk_tokens: Number of flattened tokens processed together. This bounds temporary softmax memory.

    Returns:
        Scalar statistics over every token and vocabulary element. Returned values do not alias the inputs.
    """
    if reference_logits.shape != candidate_logits.shape:
        raise ValueError(
            f"Logit shape mismatch: reference={tuple(reference_logits.shape)}, "
            f"candidate={tuple(candidate_logits.shape)}"
        )
    if reference_logits.ndim < 2 or reference_logits.shape[-1] <= 0:
        raise ValueError(f"Expected logits of shape [..., vocab], got {tuple(reference_logits.shape)}")
    if chunk_tokens <= 0:
        raise ValueError(f"chunk_tokens must be positive, got {chunk_tokens}")

    vocab_size = reference_logits.shape[-1]
    reference_tokens = reference_logits.detach().reshape(-1, vocab_size)
    candidate_tokens = candidate_logits.detach().reshape(-1, vocab_size)
    token_count = reference_tokens.shape[0]
    if token_count == 0:
        raise ValueError("Cannot compare empty logit tensors")

    kl_chunks: list[torch.Tensor] = []
    absolute_difference_sum = 0.0
    max_absolute_difference = 0.0
    dot_product = 0.0
    reference_squared_norm = 0.0
    candidate_squared_norm = 0.0

    for start in range(0, token_count, chunk_tokens):
        end = min(start + chunk_tokens, token_count)
        reference_chunk = reference_tokens[start:end].float()
        candidate_chunk = candidate_tokens[start:end].float()
        if not bool(torch.isfinite(reference_chunk).all()):
            raise ValueError(f"Reference logits contain non-finite values in flattened token range [{start}, {end})")
        if not bool(torch.isfinite(candidate_chunk).all()):
            raise ValueError(f"Candidate logits contain non-finite values in flattened token range [{start}, {end})")

        reference_log_probs = F.log_softmax(reference_chunk, dim=-1)
        candidate_log_probs = F.log_softmax(candidate_chunk, dim=-1)
        reference_probs = reference_log_probs.exp()
        token_kl = (reference_probs * (reference_log_probs - candidate_log_probs)).sum(dim=-1)
        kl_chunks.append(token_kl.cpu())

        absolute_difference = (reference_chunk - candidate_chunk).abs()
        absolute_difference_sum += absolute_difference.sum(dtype=torch.float64).item()
        max_absolute_difference = max(max_absolute_difference, absolute_difference.max().item())
        dot_product += (reference_chunk * candidate_chunk).sum(dtype=torch.float64).item()
        reference_squared_norm += reference_chunk.square().sum(dtype=torch.float64).item()
        candidate_squared_norm += candidate_chunk.square().sum(dtype=torch.float64).item()

    per_token_kl = torch.cat(kl_chunks)
    if not bool(torch.isfinite(per_token_kl).all()):
        raise ValueError("KL divergence contains non-finite values")

    norm_product = math.sqrt(reference_squared_norm * candidate_squared_norm)
    if norm_product == 0.0:
        cosine_similarity = 1.0 if max_absolute_difference == 0.0 else 0.0
    else:
        cosine_similarity = dot_product / norm_product

    return _ParityMetrics(
        token_count=token_count,
        vocab_size=vocab_size,
        mean_kl=per_token_kl.mean().item(),
        p95_kl=torch.quantile(per_token_kl, 0.95).item(),
        max_kl=per_token_kl.max().item(),
        cosine_similarity=cosine_similarity,
        mean_absolute_logit_difference=absolute_difference_sum / reference_logits.numel(),
        max_absolute_logit_difference=max_absolute_difference,
    )


def _resolve_parity_thresholds(profile: str, comparison_kind: _ComparisonKind) -> _ParityThresholds:
    """Resolve one named profile for the requested comparison kind."""
    if profile not in _PARITY_PROFILES:
        raise ValueError(f"Unknown parity tolerance profile {profile!r}; expected one of {sorted(_PARITY_PROFILES)}")
    return _PARITY_PROFILES[profile][comparison_kind]


def _normalize_parity_profile_overrides(raw_overrides: object) -> dict[str, str]:
    """Validate and normalize optional per-comparison profile overrides."""
    if raw_overrides is None:
        return {}
    if not isinstance(raw_overrides, dict):
        raise ValueError("parity_tolerance_profile_overrides must be a mapping")

    non_string_comparisons = [repr(comparison) for comparison in raw_overrides if not isinstance(comparison, str)]
    if non_string_comparisons:
        raise ValueError(
            "parity_tolerance_profile_overrides comparison names must be strings, got "
            + ", ".join(non_string_comparisons)
        )
    unknown_comparisons = sorted(set(raw_overrides) - _PARITY_COMPARISONS)
    if unknown_comparisons:
        raise ValueError(
            "Unknown parity_tolerance_profile_overrides comparisons: "
            f"{', '.join(unknown_comparisons)}; expected one of {sorted(_PARITY_COMPARISONS)}"
        )

    normalized: dict[str, str] = {}
    for comparison, profile in raw_overrides.items():
        if not isinstance(profile, str):
            raise ValueError(f"parity_tolerance_profile_overrides.{comparison} must be a profile name")
        _resolve_parity_thresholds(profile, "same_implementation")
        normalized[comparison] = profile
    return normalized


def _select_parity_profile(default_profile: str, raw_overrides: object, comparison: str) -> str:
    """Select one comparison profile, falling back to the global profile."""
    _resolve_parity_thresholds(default_profile, "same_implementation")
    if comparison not in _PARITY_COMPARISONS:
        raise ValueError(f"Unknown parity comparison {comparison!r}; expected one of {sorted(_PARITY_COMPARISONS)}")
    return _normalize_parity_profile_overrides(raw_overrides).get(comparison, default_profile)


def _apply_parity_threshold_overrides(
    thresholds: _ParityThresholds,
    *,
    mean_kl: float | None = None,
    p95_kl: float | None = None,
    cosine_similarity: float | None = None,
) -> _ParityThresholds:
    """Replace selected profile gates with explicit model-specific values."""
    for name, value in (("mean_kl", mean_kl), ("p95_kl", p95_kl)):
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError(f"{name} threshold override must be finite and non-negative, got {value}")
    if cosine_similarity is not None and (not math.isfinite(cosine_similarity) or not -1.0 <= cosine_similarity <= 1.0):
        raise ValueError(
            f"cosine_similarity threshold override must be finite and between -1 and 1, got {cosine_similarity}"
        )
    return _ParityThresholds(
        mean_kl=thresholds.mean_kl if mean_kl is None else mean_kl,
        p95_kl=thresholds.p95_kl if p95_kl is None else p95_kl,
        cosine_similarity=thresholds.cosine_similarity if cosine_similarity is None else cosine_similarity,
    )


def _normalize_parity_threshold_overrides(raw_overrides: object) -> dict[str, dict[str, float]]:
    """Validate and normalize optional per-comparison profile threshold overrides."""
    if raw_overrides is None:
        return {}
    if not isinstance(raw_overrides, dict):
        raise ValueError("parity_threshold_overrides must be a mapping")

    non_string_comparisons = [repr(comparison) for comparison in raw_overrides if not isinstance(comparison, str)]
    if non_string_comparisons:
        raise ValueError(
            "parity_threshold_overrides comparison names must be strings, got " + ", ".join(non_string_comparisons)
        )
    unknown_comparisons = sorted(set(raw_overrides) - _PARITY_COMPARISONS)
    if unknown_comparisons:
        raise ValueError(
            "Unknown parity_threshold_overrides comparisons: "
            f"{', '.join(unknown_comparisons)}; expected one of {sorted(_PARITY_COMPARISONS)}"
        )

    normalized: dict[str, dict[str, float]] = {}
    for comparison, raw_metrics in raw_overrides.items():
        if not isinstance(raw_metrics, dict):
            raise ValueError(f"parity_threshold_overrides.{comparison} must be a mapping")
        non_string_metrics = [repr(metric) for metric in raw_metrics if not isinstance(metric, str)]
        if non_string_metrics:
            raise ValueError(
                f"parity_threshold_overrides.{comparison} metric names must be strings, got "
                + ", ".join(non_string_metrics)
            )
        unknown_metrics = sorted(set(raw_metrics) - _PARITY_OVERRIDE_METRICS)
        if unknown_metrics:
            raise ValueError(
                f"Unknown parity_threshold_overrides.{comparison} metrics: {', '.join(unknown_metrics)}; "
                f"expected one of {sorted(_PARITY_OVERRIDE_METRICS)}"
            )

        metrics: dict[str, float] = {}
        for metric, raw_value in raw_metrics.items():
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                raise ValueError(f"parity_threshold_overrides.{comparison}.{metric} must be numeric")
            metrics[metric] = float(raw_value)
        _apply_parity_threshold_overrides(
            _ParityThresholds(mean_kl=0.0, p95_kl=0.0, cosine_similarity=0.0),
            mean_kl=metrics.get("mean_kl"),
            p95_kl=metrics.get("p95_kl"),
            cosine_similarity=metrics.get("cosine_similarity"),
        )
        normalized[comparison] = metrics
    return normalized


def _parity_failures(
    metrics: _ParityMetrics,
    thresholds: _ParityThresholds,
) -> tuple[str, ...]:
    """Return failed mean-KL, p95-KL, and cosine gates for a named profile."""
    failures: list[str] = []
    if metrics.mean_kl > thresholds.mean_kl:
        failures.append(f"mean KL {metrics.mean_kl:.6e} > profile threshold {thresholds.mean_kl:.6e}")
    if metrics.p95_kl > thresholds.p95_kl:
        failures.append(f"p95 KL {metrics.p95_kl:.6e} > profile threshold {thresholds.p95_kl:.6e}")
    if metrics.cosine_similarity < thresholds.cosine_similarity:
        failures.append(
            f"cosine similarity {metrics.cosine_similarity:.8f} < profile threshold {thresholds.cosine_similarity:.8f}"
        )
    return tuple(failures)
