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

"""Distributed-correct metric recorder for the DSpark objective.

Ratio metrics (accept-rate, the loss terms, supervision density) are recorded as
separate numerator/denominator sums and only divided after a cross-rank sum, so a
data-parallel run reports the global ``sum(num) / sum(den)`` rather than each
rank's local ratio. Scalar metrics accumulate ``(sum, count)`` and reduce to a
global mean. ``add_metric`` matches the call sites in the objective; the recipe
calls :func:`reduce_metrics` (a collective) on every rank, then logs on rank 0.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

_ratio: dict[str, list[float]] = {}  # key -> [num_sum, den_sum]
_scalar: dict[str, list[float]] = {}  # key -> [value_sum, count]


def _to_float(value) -> float:
    if torch.is_tensor(value):
        value = value.detach()
        return float(value.reshape(()) if value.numel() == 1 else value.float().mean())
    return float(value)


def add_metric(name, value, *, den=None, reduction: str = "dp_sum", tag: str = "train") -> None:
    """Record one metric for the current logging window.

    With ``den`` the metric is a ratio accumulated as ``(num, den)`` sums; without
    it the metric accumulates ``(value, count)`` for a mean. Reduction across ranks
    happens in :func:`reduce_metrics`, not here.
    """
    del reduction  # accepted for call-site compatibility; reduction is dp_sum
    key = f"{tag}/{name}"
    if den is not None:
        entry = _ratio.setdefault(key, [0.0, 0.0])
        entry[0] += _to_float(value)
        entry[1] += _to_float(den)
    else:
        entry = _scalar.setdefault(key, [0.0, 0.0])
        entry[0] += _to_float(value)
        entry[1] += 1.0


def _reduce_pairs(pairs: list[list[float]]) -> list[list[float]]:
    """All-reduce (SUM) a list of ``[a, b]`` pairs across ranks when distributed."""
    if not pairs or not (dist.is_available() and dist.is_initialized()):
        return pairs
    flat = torch.tensor(pairs, dtype=torch.float64)
    if torch.cuda.is_available():
        flat = flat.cuda()
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    return flat.cpu().tolist()


def reduce_metrics() -> dict[str, float]:
    """Collective: all-reduce numerators/denominators across ranks, return globals.

    Must be called on every rank (it issues collectives). Keys are sorted so the
    reduction order matches across ranks; the recorded key set is deterministic.
    """
    out: dict[str, float] = {}
    ratio_keys = sorted(_ratio)
    reduced = _reduce_pairs([_ratio[k] for k in ratio_keys])
    for key, (num, den) in zip(ratio_keys, reduced):
        out[key] = num / den if den > 0 else 0.0
    scalar_keys = sorted(_scalar)
    reduced = _reduce_pairs([_scalar[k] for k in scalar_keys])
    for key, (total, count) in zip(scalar_keys, reduced):
        out[key] = total / count if count > 0 else 0.0
    return out


def get_metrics() -> dict[str, float]:
    """Return locally-reduced metrics (no cross-rank collective; for tests / single process)."""
    out = {k: (n / d if d > 0 else 0.0) for k, (n, d) in _ratio.items()}
    out.update({k: (s / c if c > 0 else 0.0) for k, (s, c) in _scalar.items()})
    return out


def reset_metrics() -> None:
    """Clear the current logging window."""
    _ratio.clear()
    _scalar.clear()


__all__ = ["add_metric", "reduce_metrics", "get_metrics", "reset_metrics"]
