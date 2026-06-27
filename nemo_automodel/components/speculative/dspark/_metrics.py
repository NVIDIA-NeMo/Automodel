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

"""Dependency-free metric recorder for the DSpark training code.

Records training-step scalars without requiring an initialized process group,
so the model and loss run under single-process / CPU tests. Records the most
recent scalar per ``tag/name``; recipe-level logging (W&B / TensorBoard) is
wired separately in the trainer, not here.
"""

from __future__ import annotations

import torch

_metrics: dict[str, float] = {}


def _scalar(value) -> torch.Tensor:
    """Detach and reduce ``value`` to a 0-dim float tensor (no autograd graph)."""
    if torch.is_tensor(value):
        value = value.detach()
        return value.reshape(()).float() if value.numel() == 1 else value.float().mean()
    return torch.tensor(float(value), dtype=torch.float32)


def add_metric(name, value, *, den=None, reduction: str = "dp_sum", tag: str = "train") -> float:
    """Record one scalar (or ``num/den`` ratio) metric for later inspection.

    Does no cross-rank reduction -- the DSpark loss already all-reduces its own
    denominators where it matters for the backward value.
    """
    del reduction  # accepted for signature compatibility; not used here
    key = f"{tag}/{name}"
    if den is not None:
        denom = _scalar(den)
        result = (_scalar(value) / denom).item() if denom.item() != 0 else 0.0
    else:
        result = _scalar(value).item()
    _metrics[key] = result
    return result


def get_metrics() -> dict[str, float]:
    """Return a snapshot of the most recently recorded metrics."""
    return dict(_metrics)


def reset_metrics() -> None:
    """Clear all recorded metrics."""
    _metrics.clear()


__all__ = ["add_metric", "get_metrics", "reset_metrics"]
