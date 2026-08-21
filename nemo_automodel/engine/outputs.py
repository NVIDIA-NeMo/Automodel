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

"""Structured output records returned by Engine loss callbacks.

The types in this module describe output layout only. The Engine owns
context-parallel restoration and per-Datum routing; constructing these records
does not detach, clone, pad, gather, or otherwise transform tensors.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import torch

__all__ = ["LossFnOutputBatch", "PerTokenOutput"]


def _validate_field_name(name: Any, *, container: str) -> None:
    if not isinstance(name, str) or not name:
        raise TypeError(f"{container} field names must be non-empty strings, got {name!r}")


@dataclass(frozen=True, eq=False)
class PerTokenOutput:
    """One callback tensor aligned with the current local token stream.

    The tensor's leading dimensions must exactly match the current CP-local
    loss ``weights`` shape (``[B, S]`` or ``[T]``); only trailing feature
    dimensions may be added. The tensor must also be on the same device as the
    weights. The Engine therefore knows the token axis without guessing from
    arbitrary shapes.
    ``fill_value`` is used only when restoring a caller position for which a
    backend did not retain a token (for example, a dropped slot in a repacked
    sequence).

    The tensor is deliberately retained by reference rather than cloned. This
    preserves its device, autograd relationship, and avoids copying potentially
    large token outputs; the Engine detaches returned records at its API
    boundary.

    Args:
        tensor: Tensor covering the complete CP-local token stream for the
            current loss-callback invocation.
        fill_value: Scalar value used for positions introduced by layout
            restoration. It must be representable by ``tensor.dtype`` when the
            Engine performs that restoration.
    """

    tensor: torch.Tensor
    fill_value: float | int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.tensor, torch.Tensor):
            raise TypeError(f"PerTokenOutput.tensor must be a torch.Tensor, got {type(self.tensor).__name__}")
        if self.tensor.ndim == 0:
            raise ValueError("PerTokenOutput.tensor must have at least one dimension")
        if not isinstance(self.fill_value, (int, float)):
            raise TypeError(f"PerTokenOutput.fill_value must be an int or float, got {type(self.fill_value).__name__}")


@dataclass(frozen=True, eq=False)
class LossFnOutputBatch:
    """Layout-aware outputs from one loss-callback invocation.

    ``per_token`` holds batch-level local token streams. Callbacks should not
    split those streams into per-Datum tensors themselves: under packed context
    parallelism, different ranks can own different numbers of tokens from one
    Datum. The Engine can restore each complete stream first and then route it
    to the final per-Datum records.

    Restoration requires the active context-parallel backend to report a
    reversible token layout. Backends that do not report one fail explicitly
    when this typed output is requested; legacy output records remain usable.

    ``per_datum`` optionally supplies the ordinary output records that the
    restored token fields will augment. It follows the existing callback
    convention of one mapping per logical Datum in the current microbatch.
    These records are opaque and must already be identical on CP ranks (for
    example, sample IDs copied from a PER_DATUM loss input). A token-derived
    CP-local value belongs in ``per_token`` so the Engine can restore it before
    producing records.

    Input mappings, the record sequence, and each record mapping are copied and
    exposed as read-only views. Tensor and other nested values are intentionally
    retained by reference so the envelope is cheap and preserves autograd until
    the Engine consumes it.

    Args:
        per_token: Non-empty mapping from output field name to its local token
            tensor and restoration fill value. Tensor leading dimensions and
            device must match the callback's loss weights.
        per_datum: Optional sequence of existing per-Datum output mappings.
            These mappings may not already contain a field named by
            ``per_token`` because the Engine will add those fields after token
            restoration.
    """

    per_token: Mapping[str, PerTokenOutput]
    per_datum: Sequence[Mapping[str, Any]] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.per_token, Mapping):
            raise TypeError(f"LossFnOutputBatch.per_token must be a mapping, got {type(self.per_token).__name__}")
        copied_per_token = dict(self.per_token)
        if not copied_per_token:
            raise ValueError("LossFnOutputBatch.per_token cannot be empty")
        for name, output in copied_per_token.items():
            _validate_field_name(name, container="per_token")
            if not isinstance(output, PerTokenOutput):
                raise TypeError(f"per_token field {name!r} must be a PerTokenOutput, got {type(output).__name__}")

        object.__setattr__(self, "per_token", MappingProxyType(copied_per_token))
        if self.per_datum is None:
            return
        if not isinstance(self.per_datum, Sequence) or isinstance(self.per_datum, (str, bytes)):
            raise TypeError(
                "LossFnOutputBatch.per_datum must be a sequence of mappings or None, "
                f"got {type(self.per_datum).__name__}"
            )

        copied_records: list[Mapping[str, Any]] = []
        token_names = set(copied_per_token)
        for index, record in enumerate(self.per_datum):
            if not isinstance(record, Mapping):
                raise TypeError(f"per_datum record {index} must be a mapping, got {type(record).__name__}")
            copied_record = dict(record)
            for name in copied_record:
                _validate_field_name(name, container=f"per_datum record {index}")
            conflicts = token_names.intersection(copied_record)
            if conflicts:
                raise ValueError(f"per_datum record {index} conflicts with per_token fields {sorted(conflicts)}")
            copied_records.append(MappingProxyType(copied_record))
        object.__setattr__(self, "per_datum", tuple(copied_records))
