# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Typed output contract for training: :class:`ModelOutput` and per-token helpers.

This is the output half of the training data contract; the input half
(``Datum`` / ``collate_datums``) lives in ``components.datasets``. These helpers
operate on model logits — a forward concern — so they belong here rather than
in the datasets layer. The module imports only ``torch`` (no other
``components`` submodules), keeping the "components must not import each other"
independence contract intact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

__all__ = [
    "ModelOutput",
    "selected_token_logprobs",
    "compute_entropy",
    "split_per_datum",
]


@dataclass
class ModelOutput:
    """Per-datum model outputs returned by a forward / forward-backward pass.

    Each list field is aligned to the order of the input ``Datum`` list for the
    local data-parallel rank. A field is ``None`` when the pass did not produce
    it (e.g. ``values`` without a value head, or any field on a non-output
    pipeline stage).

    Args:
        loss: scalar training loss (``None`` for forward-only passes).
        logprobs: per-datum selected-token logprobs, each shape ``[T_i]``.
        entropy: per-datum per-token entropy, each shape ``[T_i]``.
        values: per-datum value-head outputs, each shape ``[T_i]``.
        metrics: scalar metrics (loss components, grad-norm, etc.).
    """

    loss: torch.Tensor | None = None
    logprobs: list[torch.Tensor] | None = None
    entropy: list[torch.Tensor] | None = None
    values: list[torch.Tensor] | None = None
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "loss": self.loss,
            "logprobs": self.logprobs,
            "entropy": self.entropy,
            "values": self.values,
            "metrics": dict(self.metrics),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelOutput":
        return cls(
            loss=data.get("loss"),
            logprobs=data.get("logprobs"),
            entropy=data.get("entropy"),
            values=data.get("values"),
            metrics=dict(data.get("metrics", {})),
        )


def _materialize(logits: torch.Tensor) -> torch.Tensor:
    """Materialize a (possibly vocab-parallel) DTensor to a local full tensor."""
    return logits.full_tensor() if hasattr(logits, "full_tensor") else logits


def selected_token_logprobs(logits: torch.Tensor, target_tokens: torch.Tensor) -> torch.Tensor:
    """Log-probability the model assigns to each target token.

    Args:
        logits: ``[..., T, V]`` raw logits. Vocab-parallel DTensors are
            materialized to full tensors first.
        target_tokens: ``[..., T]`` token ids, broadcast-compatible with
            ``logits`` minus its vocab dim.

    Returns:
        ``[..., T]`` per-position logprob of the corresponding target token.
    """
    logits = _materialize(logits).float()
    logp = torch.log_softmax(logits, dim=-1)
    return logp.gather(-1, target_tokens.unsqueeze(-1).long()).squeeze(-1)


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Per-position entropy of the model's next-token distribution.

    Args:
        logits: ``[..., T, V]`` raw logits (DTensors are materialized first).

    Returns:
        ``[..., T]`` entropy in nats.
    """
    logits = _materialize(logits).float()
    logp = torch.log_softmax(logits, dim=-1)
    return -(logp.exp() * logp).sum(dim=-1)


def split_per_datum(flat: torch.Tensor, seq_lens: list[int] | torch.Tensor) -> list[torch.Tensor]:
    """Split a packed/flat per-token tensor back into per-datum tensors.

    Inverse of THD concatenation. Any trailing padding beyond ``sum(seq_lens)``
    is dropped.

    Args:
        flat: ``[N]`` or ``[1, N]`` per-token values.
        seq_lens: original per-sequence lengths.

    Returns:
        One tensor per datum, in input order.
    """
    if flat.dim() == 2 and flat.shape[0] == 1:
        flat = flat.squeeze(0)
    if isinstance(seq_lens, torch.Tensor):
        seq_lens = seq_lens.tolist()
    out: list[torch.Tensor] = []
    offset = 0
    for length in seq_lens:
        out.append(flat[offset : offset + length])
        offset += length
    return out
