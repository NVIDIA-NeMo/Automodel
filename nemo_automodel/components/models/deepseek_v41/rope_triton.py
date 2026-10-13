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

"""DeepSeek V4.1 rotary embedding using the shared interleaved CUDA kernel."""

from __future__ import annotations

import torch

from nemo_automodel.components.models.common.rope_triton import HAVE_TRITON as HAVE_TRITON
from nemo_automodel.components.models.common.rope_triton import apply_complex_rope


def apply_rope_triton(values: torch.Tensor, angles: torch.Tensor, *, inverse: bool = False) -> torch.Tensor:
    """Rotate the final channels of ``values`` with the fused Triton kernel.

    Drop-in for ``attention._apply_rope``: the last ``2 * angles.shape[-1]`` channels are
    rotated as adjacent pairs by ``exp(i * angles)`` (its conjugate when ``inverse``) and the
    remaining channels are copied, all in one pass. The backward rotates the gradient by the
    conjugate in one pass; ``angles`` receives no gradient. Inputs that are not on CUDA, whose
    angles require a gradient, or whose dtype is outside BF16/FP16/FP32 use the eager path.

    Args:
        values: Tensor of shape [batch, sequence, channels] or
            [batch, sequence, heads, channels]; non-contiguous inputs are accepted.
        angles: FP32 tensor of shape [batch, sequence, rotary_pairs].
        inverse: Conjugate the rotation, as for the attention output.

    Returns:
        New contiguous tensor with the shape and dtype of ``values``.

    Raises:
        ValueError: ``angles`` does not describe an even rotary width within ``values``.
    """
    rotary_dim = 2 * angles.shape[-1]
    if angles.ndim != 3 or angles.shape[:2] != values.shape[:2] or not 0 < rotary_dim <= values.shape[-1]:
        raise ValueError(
            f"angles of shape {tuple(angles.shape)} do not fit values of shape {tuple(values.shape)}: expected "
            "[batch, sequence, rotary_pairs] with 0 < 2 * rotary_pairs <= channels"
        )
    frequencies = torch.polar(torch.ones_like(angles), angles)
    return apply_complex_rope(values, frequencies, inverse=inverse)
