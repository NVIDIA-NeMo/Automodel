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

"""Marker for parameters whose gradient the model averages over its own owner group."""

import torch

_MODEL_OWNED_GRAD_DIVISOR_ATTR = "_nemo_model_owned_grad_divisor"


def model_owned_grad_divisor(parameter: torch.Tensor) -> float | None:
    """Return the owner-sharding gradient divisor stamped on a parameter.

    Model-owned sharded parameters (for example an Engram table whose rows are
    spread over an owner group by the model's own All-to-All) carry the divisor
    that averages their accumulated gradient over the owner group. The marker
    lives in the parameter's ``__dict__``.

    Args:
        parameter: Parameter or DTensor parameter of arbitrary shape.

    Returns:
        The divisor, or ``None`` for a parameter without an owner-sharding contract.
    """
    return getattr(parameter, _MODEL_OWNED_GRAD_DIVISOR_ATTR, None)


def stamp_model_owned_grad_divisor(parameter: torch.Tensor, divisor: float) -> None:
    """Stamp the owner-sharding gradient divisor on a parameter.

    Args:
        parameter: Parameter or DTensor parameter of arbitrary global and
            per-rank local shape whose gradient the model accumulates over its
            owner group. No axis, dtype, device, or placement restriction
            applies. Only the parameter's Python metadata is mutated; tensor
            values, storage, and DTensor placements are unchanged.
        divisor: Owner-group size that averages the accumulated gradient.
    """
    setattr(parameter, _MODEL_OWNED_GRAD_DIVISOR_ATTR, float(divisor))
