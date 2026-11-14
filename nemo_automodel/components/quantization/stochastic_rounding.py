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

"""Stochastic rounding to the precision of an FP8 E4M3 value.

An E4M3 value has four significant bits. MXFP8 stores E4M3 values with one power-of-two scale per block of 32, so
a number with four significant bits is stored exactly unless it is more than ``2**14`` times smaller than the largest
value of its block. Rounding a tensor to four significant bits at random, up with a probability equal to the
fraction of the rounding step that lies below the value, therefore turns the round-to-nearest MXFP8 cast that
follows into stochastic rounding: the stored value equals the input in expectation.
"""

from __future__ import annotations

import torch

from nemo_automodel.shared.import_utils import safe_import

_HAVE_TRITON, triton = safe_import("triton")
_HAVE_TRITON_LANGUAGE, tl = safe_import("triton.language")
HAVE_TRITON: bool = bool(_HAVE_TRITON and _HAVE_TRITON_LANGUAGE)

E4M3_SIGNIFICANT_BITS = 4
_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
_BLOCK = 1024
# One launch covers at most this many values, so the int32 element offsets that key the random numbers stay in range.
_MAX_ELEMENTS_PER_LAUNCH = 2**30
_FLOAT32_MANTISSA_BITS = 23
# Smallest magnitude that is rounded: below it the rounding step is not a normal FP32 number.
_MIN_ROUNDED_MAGNITUDE = 2.0**-123

if HAVE_TRITON:

    @triton.jit
    def _stochastic_round_kernel(x_ptr, y_ptr, count, seed, STEP_SHIFT: tl.constexpr, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < count
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        exponent = x.to(tl.int32, bitcast=True) & 0x7F800000
        # The rounding step is the value's power of two divided by 2**(significant bits - 1). Values below
        # _MIN_ROUNDED_MAGNITUDE (zero included) pass through unchanged.
        has_step = exponent > STEP_SHIFT
        step = tl.where(has_step, exponent - STEP_SHIFT, 0x3F800000).to(tl.float32, bitcast=True)
        scaled = x / step
        low = tl.floor(scaled)
        rounded = (low + (tl.rand(seed, offsets) < scaled - low).to(tl.float32)) * step
        tl.store(y_ptr + offsets, tl.where(has_step, rounded, x).to(y_ptr.dtype.element_ty), mask=mask)


def _stochastic_round_torch(tensor: torch.Tensor) -> torch.Tensor:
    """Reference implementation with PyTorch operations; see :func:`stochastic_round_to_e4m3`."""
    mantissa, exponent = torch.frexp(tensor.float())
    # |mantissa| is in [0.5, 1), so the scaled value has its leading bit at 2**(significant bits - 1).
    scaled = mantissa * 2**E4M3_SIGNIFICANT_BITS
    low = torch.floor(scaled)
    rounded = torch.ldexp((low + (torch.rand_like(scaled) < scaled - low)) / 2**E4M3_SIGNIFICANT_BITS, exponent)
    return torch.where(tensor.abs() >= _MIN_ROUNDED_MAGNITUDE, rounded.to(tensor.dtype), tensor)


def stochastic_round_to_e4m3(tensor: torch.Tensor) -> torch.Tensor:
    """Round every value to four significant bits, up or down at random, without bias.

    A value between two neighbours with four significant bits moves to the upper one with a probability equal to
    its distance from the lower one divided by their spacing, so the result equals ``tensor`` in expectation.
    Values that already have four significant bits, infinities, NaNs and magnitudes below ``2**-123`` (zero
    included) are returned unchanged. The random numbers come from PyTorch's global generator.

    Args:
        tensor: Floating-point tensor of any shape, in BF16, FP16 or FP32.

    Returns:
        New tensor with the shape, dtype and device of ``tensor``.
    """
    if tensor.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(f"stochastic_round_to_e4m3 supports BF16, FP16 and FP32 tensors, got {tensor.dtype}")
    if not (HAVE_TRITON and tensor.is_cuda and tensor.is_contiguous()):
        return _stochastic_round_torch(tensor)
    output = torch.empty_like(tensor)
    source, destination = tensor.view(-1), output.view(-1)
    step_shift = (E4M3_SIGNIFICANT_BITS - 1) << _FLOAT32_MANTISSA_BITS
    for start in range(0, source.numel(), _MAX_ELEMENTS_PER_LAUNCH):
        chunk = source[start : start + _MAX_ELEMENTS_PER_LAUNCH]
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
        _stochastic_round_kernel[(triton.cdiv(chunk.numel(), _BLOCK),)](
            chunk,
            destination[start : start + _MAX_ELEMENTS_PER_LAUNCH],
            chunk.numel(),
            seed,
            STEP_SHIFT=step_shift,
            BLOCK=_BLOCK,
        )
    return output
