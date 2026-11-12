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


"""Adam-style second moments shared across each sketch row; no first moment."""

import torch
from torch import Tensor


@torch.no_grad()
def row_adam_direction(
    sketch: Tensor,
    second_moment: Tensor,
    *,
    step: int,
    beta2: float,
    epsilon: float,
    inplace: bool = False,
) -> Tensor:
    """Normalize each sketch row by its bias-corrected running RMS.

    The state is indexed by the original weight's output rows, not by the changing
    projection coordinates. With ``inplace`` the sketch is overwritten with the
    direction, avoiding a full-size temporary when the sketch is no longer needed.

    Args:
        sketch: ``[rows, rank]`` accumulated sketch.
        second_moment: ``[rows, 1]`` running second moment, updated in place.
        step: 1-based update count used for bias correction.
        beta2: Second-moment decay in ``[0, 1)``.
        epsilon: Denominator floor.
        inplace: Overwrite ``sketch`` with the direction.

    Returns:
        The normalized direction, same shape as ``sketch``.
    """
    if step < 1 or not 0 <= beta2 < 1 or epsilon <= 0:
        raise ValueError("Invalid RowAdam step, decay, or epsilon")
    if second_moment.shape != (sketch.shape[0], 1):
        raise ValueError("RowAdam requires one second moment per output row")
    # Row mean of squares via a [rows, 1] norm: no full-size temporary, even when inplace.
    row_mean_sq = torch.linalg.vector_norm(sketch, dim=1, keepdim=True, dtype=torch.float32).square_()
    row_mean_sq.div_(sketch.shape[1])
    # A NaN or inf anywhere in a row propagates into its norm, so this check is equivalent.
    if not torch.isfinite(row_mean_sq).all():
        raise FloatingPointError("Nonfinite gradient sketch")
    second_moment.mul_(beta2).add_(row_mean_sq, alpha=1 - beta2)
    denominator = (second_moment / (1 - beta2**step)).sqrt_().add_(epsilon)
    if inplace:
        return sketch.div_(denominator)
    return sketch / denominator
