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

"""Split a batch dict into microbatches.

Used by the Engine to drive gradient accumulation. Slices any tensor-valued
keys along dim 0; passes through opaque (non-tensor, non-list) values.
"""

from __future__ import annotations

from typing import Any

import torch


def split_into_microbatches(batch: dict[str, Any], num_microbatches: int) -> list[dict[str, Any]]:
    """Split ``batch`` into ``num_microbatches`` dict-shaped slices along dim 0.

    Rules:
      - ``torch.Tensor``: sliced along dim 0 into roughly-equal chunks (last chunk
        may be smaller).
      - ``list`` / ``tuple``: split into ``num_microbatches`` sub-lists of roughly
        equal length.
      - Other values (None, ints, strings, dicts of tensors for multimodal
        inputs): broadcast as-is to every microbatch.

    A microbatch dict has the same keys as ``batch``; only the values change.
    """
    if num_microbatches < 1:
        raise ValueError(f"num_microbatches must be >= 1, got {num_microbatches}")
    if num_microbatches == 1:
        return [batch]

    # Discover the batch dimension from any tensor value.
    batch_dim_size: int | None = None
    for value in batch.values():
        if isinstance(value, torch.Tensor) and value.dim() >= 1:
            batch_dim_size = value.shape[0]
            break

    if batch_dim_size is None:
        # No tensors with a batch dim — caller asked for >1 microbatches but the
        # batch isn't splittable. Best effort: replicate the dict.
        return [batch] * num_microbatches

    if batch_dim_size < num_microbatches:
        raise ValueError(
            f"Cannot split a batch of size {batch_dim_size} into "
            f"{num_microbatches} microbatches."
        )

    # Compute chunk sizes (roughly equal; remainder pushed to the front).
    base, rem = divmod(batch_dim_size, num_microbatches)
    sizes = [base + (1 if i < rem else 0) for i in range(num_microbatches)]
    starts = [sum(sizes[:i]) for i in range(num_microbatches)]

    microbatches: list[dict[str, Any]] = []
    for i in range(num_microbatches):
        start, size = starts[i], sizes[i]
        end = start + size
        mb: dict[str, Any] = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor) and value.dim() >= 1 and value.shape[0] == batch_dim_size:
                mb[key] = value[start:end]
            elif isinstance(value, (list, tuple)) and len(value) == batch_dim_size:
                mb[key] = type(value)(value[start:end])
            else:
                # Broadcast scalars / opaque values / batch-irrelevant dicts.
                mb[key] = value
        microbatches.append(mb)
    return microbatches


__all__ = ["split_into_microbatches"]
