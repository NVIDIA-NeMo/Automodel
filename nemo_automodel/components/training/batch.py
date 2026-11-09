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

"""Move model input tensors while retaining host-side packing metadata."""

from collections.abc import Mapping
from typing import Any

import torch


def move_batch_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    """Transfer training/validation inputs without moving deferred attention IDs.

    Args:
        batch: Model inputs containing tensors of arbitrary shape, including
            nested mappings/lists/tuples. ``_packed_seq_ids`` is an integer
            tensor [batch, sequence] retained on CPU only when masks/Flash metadata still need preparation.
            THD and prepared-mask inputs transfer normally.
        device: Destination for all other tensor inputs.

    Returns:
        New containers preserving tensor shapes and dtypes. Deferred attention
        IDs alias the original CPU tensor; no input tensor is mutated. Other
        tensors may alias the inputs when they already occupy the destination.
    """

    def move(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.to(device, non_blocking=True)
        if isinstance(value, Mapping):
            return {key: move(item) for key, item in value.items()}
        if isinstance(value, list):
            return [move(item) for item in value]
        if isinstance(value, tuple):
            return tuple(move(item) for item in value)
        return value

    deferred = (
        batch.get("attention_mask") is None
        and batch.get("qkv_format") != "thd"
        and not any(batch.get(key) is not None for key in ("cu_seq_lens_q", "cu_seq_lens_k"))
    )
    return {key: value if deferred and key == "_packed_seq_ids" else move(value) for key, value in batch.items()}
