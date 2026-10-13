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

from nemo_automodel.components.models.common.rope_triton import HAVE_TRITON, apply_complex_rope

_FUSED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def _apply_rope_torch(values: torch.Tensor, angles: torch.Tensor, *, inverse: bool = False) -> torch.Tensor:
    """Eager rotation, mirroring ``attention._apply_rope`` so the fallback needs no circular import.

    Args:
        values: Tensor of shape [batch, sequence, channels] or [batch, sequence, heads, channels].
        angles: FP32 tensor of shape [batch, sequence, rotary_pairs].
        inverse: Conjugate the rotation.

    Returns:
        Tensor with the shape and dtype of ``values``, in independent storage.
    """
    rotary_dim = angles.shape[-1] * 2
    pairs = torch.view_as_complex(values[..., -rotary_dim:].float().unflatten(-1, (-1, 2)).contiguous())
    frequencies = torch.polar(torch.ones_like(angles), angles)
    if values.ndim == 4:
        frequencies = frequencies.unsqueeze(2)
    if inverse:
        frequencies = frequencies.conj()
    rotated = torch.view_as_real(pairs * frequencies).flatten(-2).to(values.dtype)
    return torch.cat((values[..., :-rotary_dim], rotated), dim=-1)


def _uses_fused_kernel(values: torch.Tensor, angles: torch.Tensor) -> bool:
    return (
        values.is_cuda
        and angles.is_cuda
        and not angles.requires_grad
        and angles.dtype == torch.float32
        and values.dtype in _FUSED_DTYPES
        and values.ndim in (3, 4)
    )


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
        RuntimeError: ``values`` is on CUDA but Triton is not importable.
        ValueError: ``angles`` does not describe an even rotary width within ``values``.
    """
    if not _uses_fused_kernel(values, angles):
        return _apply_rope_torch(values, angles, inverse=inverse)
    if not HAVE_TRITON:
        raise RuntimeError(
            "apply_rope_triton needs Triton for CUDA tensors; install triton or disable the Triton RoPE path"
        )
    rotary_dim = 2 * angles.shape[-1]
    if angles.ndim != 3 or angles.shape[:2] != values.shape[:2] or not 0 < rotary_dim <= values.shape[-1]:
        raise ValueError(
            f"angles of shape {tuple(angles.shape)} do not fit values of shape {tuple(values.shape)}: expected "
            "[batch, sequence, rotary_pairs] with 0 < 2 * rotary_pairs <= channels"
        )
    frequencies = torch.polar(torch.ones_like(angles), angles)
    return apply_complex_rope(values, frequencies, inverse=inverse)
