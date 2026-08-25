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

"""Contract for DTensor parameters whose local shard is consumed by model-owned collectives."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ModelOwnedDTensorSpec:
    """Describe a logical DTensor whose local shard is consumed by model code.

    The parameter is a real, globally shaped DTensor, so model and optimizer
    checkpoints use PyTorch's native DTensor resharding.  The model routes
    activations directly to the rank that owns a row instead of letting FSDP
    materialize the full parameter, so the parameter must be excluded from
    FSDP ownership and needs an explicit gradient divisor because its
    model-owned backward collective has different averaging semantics from
    FSDP.

    Args:
        process_group: Group whose rank order matches the DTensor shard mesh.
        gradient_divisor: Explicit factor applied to the DTensor gradient
            before clipping.
    """

    process_group: torch.distributed.ProcessGroup
    gradient_divisor: float


def get_model_owned_dtensor_spec(parameter: torch.Tensor) -> ModelOwnedDTensorSpec | None:
    """Return the model-owned contract attached to ``parameter``, if any."""
    return getattr(parameter, "_nemo_model_owned_dtensor_spec", None)
