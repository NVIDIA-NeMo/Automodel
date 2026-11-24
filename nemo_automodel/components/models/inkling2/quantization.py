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

"""Nearest NVFP4 and MXFP8 encoding of grouped expert weights in the released Inkling2 layout.

``Inkling2StateDictAdapter`` decodes a released NVFP4 expert value as
``(e2m1(code) * e4m3(block_scale)) * (amax / denominator)`` with one block scale per 16 input columns, one
``amax`` per expert and one ``denominator`` per tensor, and a released MXFP8 value as
``e4m3(value) * 2**(scale - 127)`` with one ``ue8m0`` scale byte per row and 32 input columns. The encoders
below are the nearest inverses that ``hf_quant_config.json`` names (``nvfp4_nearest`` with
``amax_denominator`` scales, ``mxfp8_nearest`` with ``ue8m0`` scales):

* NVFP4: ``amax`` is the expert's largest magnitude, the block scale is ``e4m3(bmax / 6 * (denominator /
  amax))`` saturated at 448, and each code is the nearest E2M1 value of ``w / (block_scale * amax /
  denominator)``, ties to even, saturated at 6. A negative value that rounds to zero keeps its sign
  (code 8), as in the release.
* MXFP8: the scale is the smallest power of two ``2**s`` for which ``bmax / 2**s`` still rounds to a finite
  E4M3 value, that is ``bmax / 2**s <= 464``, and each value is the nearest E4M3 value of ``w / 2**s``, ties
  to even, keeping the sign of zero.

Both are fixed points on released bytes: decoding a released expert and encoding it again with its released
``amax`` reproduces its codes and scales. Arithmetic is FP32; inputs of any floating dtype are widened exactly.
"""

from typing import NamedTuple

import torch

# Experts are encoded about this many input elements at a time, bounding the FP32 temporaries.
_CHUNK_ELEMENTS = 1 << 26
_NVFP4_BLOCK = 16
_MXFP8_BLOCK = 32
_E2M1_MAX = 6.0
_E4M3_MAX = 448.0
# Up to 464, the midpoint between 448 and the next grid point 480 (a NaN encoding), values round to 448.
_E4M3_ROUNDS_FINITE = 464.0
_UE8M0_BIAS = 127
# Midpoints between consecutive E2M1 magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6.
_E2M1_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


class Nvfp4Experts(NamedTuple):
    """Released NVFP4 storage of one grouped expert projection.

    Attributes:
        codes: uint8 tensor of shape [experts, output, input / 2]. Byte ``k`` holds the E2M1 codes of input
            columns ``2k`` (low nibble) and ``2k + 1`` (high nibble); bit 3 of a code is its sign.
        scales: float8_e4m3fn tensor of shape [experts, output, input / 16], one block scale per 16 input columns.
        amax: float32 tensor of shape [experts], the largest magnitude of each expert.
    """

    codes: torch.Tensor
    scales: torch.Tensor
    amax: torch.Tensor


class Mxfp8Experts(NamedTuple):
    """Released MXFP8 storage of one grouped expert projection.

    Attributes:
        values: float8_e4m3fn tensor of shape [experts, output, input].
        scales: uint8 tensor of shape [experts, output, input / 32]; byte ``s`` scales its 32 input columns by
            ``2**(s - 127)``.
    """

    values: torch.Tensor
    scales: torch.Tensor


def _checked_experts(weight: torch.Tensor, block: int) -> None:
    """Reject inputs the released layout cannot hold.

    Args:
        weight: Tensor of shape [experts, output, input], any floating dtype, on any device.
        block: Input columns per scale; ``input`` must be a multiple of it.
    """
    if weight.ndim != 3 or weight.shape[-1] % block:
        raise ValueError(f"Expert weights must be [experts, output, input] with input divisible by {block}")
    if not weight.is_floating_point():
        raise ValueError(f"Expert weights must be floating point, got {weight.dtype}")
    if not torch.isfinite(weight).all():
        raise ValueError("Expert weights must be finite")


def _experts_per_chunk(weight: torch.Tensor) -> int:
    """Experts encoded together so that a chunk holds about ``_CHUNK_ELEMENTS`` elements.

    Args:
        weight: Tensor of shape [experts, output, input].

    Returns:
        A positive number of experts.
    """
    return max(1, _CHUNK_ELEMENTS // max(1, weight.shape[1] * weight.shape[2]))


def _nearest_e2m1(scaled: torch.Tensor) -> torch.Tensor:
    """Round to the nearest E2M1 code, ties to even, saturating at 6.

    Args:
        scaled: FP32 tensor of shape [...], already divided by its decode scale.

    Returns:
        uint8 codes of shape [...]: bits 0-2 index the magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6 and bit 3 is the
        sign bit of ``scaled``, so negative values and -0.0 that round to zero give code 8.
    """
    midpoints = torch.tensor(_E2M1_MIDPOINTS, dtype=torch.float32, device=scaled.device)
    magnitude = scaled.abs()
    # right=True sends a value on a midpoint to the upper code; ties belong to the even code, so step back
    # when the upper code is odd.
    upper = torch.bucketize(magnitude, midpoints, right=True)
    tie = (upper > 0) & (magnitude == midpoints[(upper - 1).clamp(min=0)])
    index = torch.where(tie & (upper % 2 == 1), upper - 1, upper).to(torch.uint8)
    return torch.where(torch.signbit(scaled), index | 8, index)


@torch.no_grad()
def quantize_nvfp4_experts(weight: torch.Tensor, denominator: torch.Tensor) -> Nvfp4Experts:
    """Encode grouped expert weights as released NVFP4 storage.

    Args:
        weight: Tensor of shape [experts, output, input] in any floating dtype, finite, with ``input`` a
            multiple of 16. For gate/up projections the output rows keep the released interleaving; this
            function does not reorder rows.
        denominator: FP32 tensor of shape [1], the tensor's released ``outer_scale_denominator``; finite
            and positive.

    Returns:
        Newly allocated codes, block scales and amax on ``weight``'s device; nothing aliases the inputs.
        An all-zero expert gets amax 0, zero block scales and zero codes.
    """
    _checked_experts(weight, _NVFP4_BLOCK)
    if denominator.shape != (1,) or denominator.dtype != torch.float32:
        raise ValueError("The NVFP4 outer-scale denominator must be an FP32 tensor of shape [1]")
    if not torch.isfinite(denominator).all() or not (denominator > 0).all():
        raise ValueError("The NVFP4 outer-scale denominator must be finite and positive")
    experts, rows, width = weight.shape
    codes = torch.empty((experts, rows, width // 2), dtype=torch.uint8, device=weight.device)
    scales = torch.empty((experts, rows, width // _NVFP4_BLOCK), dtype=torch.float8_e4m3fn, device=weight.device)
    amax = torch.empty(experts, dtype=torch.float32, device=weight.device)
    denominator = denominator.to(weight.device)
    step = _experts_per_chunk(weight)
    for start in range(0, experts, step):
        stop = min(experts, start + step)
        blocks = weight[start:stop].float().reshape(stop - start, rows, width // _NVFP4_BLOCK, _NVFP4_BLOCK)
        expert_amax = blocks.abs().amax(dim=(1, 2, 3))
        nonzero = expert_amax > 0
        encode = torch.where(nonzero, denominator / expert_amax, torch.zeros_like(expert_amax))
        block_amax = blocks.abs().amax(dim=-1)
        block_scale = (block_amax / _E2M1_MAX * encode[:, None, None]).clamp(max=_E4M3_MAX).to(torch.float8_e4m3fn)
        decode = block_scale.float() * (expert_amax / denominator)[:, None, None]
        # A block whose scale rounds to zero decodes to zero whatever its codes: encode signed zeros.
        signed_zeros = torch.copysign(torch.zeros_like(blocks), blocks)
        scaled = torch.where(decode.unsqueeze(-1) > 0, blocks / decode.unsqueeze(-1), signed_zeros)
        block_codes = _nearest_e2m1(scaled).reshape(stop - start, rows, width)
        codes[start:stop] = block_codes[..., 0::2] | (block_codes[..., 1::2] << 4)
        scales[start:stop] = block_scale
        amax[start:stop] = expert_amax
    return Nvfp4Experts(codes, scales, amax)


@torch.no_grad()
def quantize_mxfp8_experts(weight: torch.Tensor) -> Mxfp8Experts:
    """Encode grouped expert weights as released MXFP8 storage.

    Args:
        weight: Tensor of shape [experts, output, input] in any floating dtype, finite, with ``input`` a
            multiple of 32. Rows are encoded independently, one scale per row and 32 input columns.

    Returns:
        Newly allocated E4M3 values and ue8m0 scale bytes on ``weight``'s device; nothing aliases the input.
        Scales are clamped to the representable bytes 0-254 (``2**-127`` to ``2**127``).
    """
    _checked_experts(weight, _MXFP8_BLOCK)
    experts, rows, width = weight.shape
    values = torch.empty((experts, rows, width), dtype=torch.float8_e4m3fn, device=weight.device)
    scales = torch.empty((experts, rows, width // _MXFP8_BLOCK), dtype=torch.uint8, device=weight.device)
    step = _experts_per_chunk(weight)
    for start in range(0, experts, step):
        stop = min(experts, start + step)
        blocks = weight[start:stop].float().reshape(stop - start, rows, width // _MXFP8_BLOCK, _MXFP8_BLOCK)
        block_amax = blocks.abs().amax(dim=-1)
        # block_amax = mantissa * 2**exponent with mantissa in [0.5, 1), so block_amax / 2**(exponent - 9)
        # lies in [256, 512): the smallest admissible shared exponent is exponent - 9 or one above it.
        _, exponent = torch.frexp(block_amax)
        shared = (exponent.to(torch.int32) - 9).clamp(-_UE8M0_BIAS, _UE8M0_BIAS)
        overflow = torch.ldexp(block_amax, -shared) > _E4M3_ROUNDS_FINITE
        shared = (shared + overflow.to(torch.int32)).clamp(max=_UE8M0_BIAS)
        # Any scale holds an all-zero block; the smallest one, byte 0, is the smallest admissible power of two.
        shared = torch.where(block_amax > 0, shared, -_UE8M0_BIAS)
        scaled = torch.ldexp(blocks, -shared.unsqueeze(-1)).clamp(-_E4M3_MAX, _E4M3_MAX)
        values[start:stop] = scaled.to(torch.float8_e4m3fn).reshape(stop - start, rows, width)
        scales[start:stop] = (shared + _UE8M0_BIAS).to(torch.uint8)
    return Mxfp8Experts(values, scales)
