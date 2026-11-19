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

"""Reference cache QAT formats with straight-through activation gradients."""

from __future__ import annotations

from typing import Literal

import torch
from torch.nn import functional as F


def _power_of_two_ceiling(value: torch.Tensor) -> torch.Tensor:
    """Round positive normal FP32 values up to an exactly represented power of two.

    Args:
        value: Positive FP32 tensor of arbitrary shape.

    Returns:
        FP32 tensor of the same shape with integer powers of two, matching the
        official IEEE754 exponent/mantissa operation without logarithm rounding.
    """
    bits = value.contiguous().view(torch.int32)
    exponent = (bits >> 23) & 255
    increment = (bits & ((1 << 23) - 1)) != 0
    return ((exponent + increment.to(torch.int32)) << 23).view(torch.float32)


class _CacheQuantization(torch.autograd.Function):
    """Quantize cache values in forward and pass the activation gradient unchanged."""

    @staticmethod
    def forward(ctx: object, values: torch.Tensor, block_size: int, format: str) -> torch.Tensor:
        """Round independent channel groups without modifying the input.

        Args:
            ctx: Autograd context; no tensors need saving for the straight-through gradient.
            values: Tensor of shape [..., channels], with arbitrary leading dimensions.
            block_size: Consecutive channels sharing a quantization scale.
            format: fp8, mxfp4 (E8M0 scale), or nvfp4 (E4M3 scale).

        Returns:
            Independently stored tensor of shape [..., channels] with input dtype.
        """
        channels = values.shape[-1]
        grouped = F.pad(values.float(), (0, -channels % block_size)).unflatten(-1, (-1, block_size))
        maximum = grouped.abs().amax(-1, keepdim=True)
        if format == "fp8":
            scale = _power_of_two_ceiling(maximum.clamp_min(1e-4) * (1 / 448))
            rounded = (grouped / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float()
        else:
            if format == "nvfp4":
                scale = (maximum.clamp_min(6 * 2**-9) / 6).to(torch.float8_e4m3fn).float()
            else:
                scale = _power_of_two_ceiling(maximum.clamp_min(6 * 2**-126) * (1 / 6))
            normalized = (grouped / scale).clamp(-6, 6)
            magnitude = normalized.abs().contiguous()
            midpoints = magnitude.new_tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
            levels = magnitude.new_tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
            lower = torch.bucketize(magnitude, midpoints)
            tie = magnitude == midpoints[lower.clamp_max(6)]
            selected = lower + (tie & (lower % 2 == 1)).to(lower.dtype)
            rounded = torch.copysign(levels[selected], normalized)
        return (rounded * scale).flatten(-2)[..., :channels].to(values.dtype)

    @staticmethod
    def backward(ctx: object, gradient: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        """Pass the gradient for [..., channels] activations through quantization.

        Args:
            ctx: Unused forward context.
            gradient: Tensor of shape [..., channels] from the cache consumer.

        Returns:
            Gradient of shape [..., channels], followed by no gradients for the
            integer group size and format name. The gradient storage is reused.
        """
        return gradient, None, None


def quantize_cache(values: torch.Tensor, *, format: Literal["fp8", "mxfp4", "nvfp4"], block_size: int) -> torch.Tensor:
    """Apply the released model's quantize/dequantize cache representation.

    Args:
        values: Tensor of shape [..., channels], with arbitrary leading dimensions.
            Complete groups are required by the released kernels; a final partial
            group is zero-padded internally for scaled unit-test configurations.
        format: fp8 for SWA KV, mxfp4 for index Q/K, nvfp4 for compressed KV.
        block_size: Channels per scale: 32 for SWA/indexer, 16 for compressed KV.

    Returns:
        Tensor of shape [..., channels], in the original dtype, with a
        straight-through gradient. Neither input storage nor its aliases are mutated.
    """
    if format not in ("fp8", "mxfp4", "nvfp4") or block_size <= 0:
        raise ValueError("Cache quantization requires fp8/mxfp4/nvfp4 and a positive block_size")
    return _CacheQuantization.apply(values, block_size, format)


# Elements quantized per chunk by native QAT (about 0.8 GiB of FP32 temporaries).
_NATIVE_CHUNK_ELEMENTS = 1 << 25


def _round_native(normalized: torch.Tensor, fp4: bool) -> torch.Tensor:
    """Round scale-normalized values to the nearest native code, ties to even.

    Args:
        normalized: FP32 tensor of arbitrary shape holding values divided by
            their power-of-two block scale. E4M3 input is clamped in place.
        fp4: Round to E2M1 magnitudes (at most 6) instead of E4M3 (at most 448).

    Returns:
        New FP32 tensor of the same shape holding code values in scale units.
    """
    if not fp4:
        return normalized.clamp_(-448, 448).to(torch.float8_e4m3fn).float()
    # E2M1 magnitudes are spaced 0.5 up to 2, 1 up to 4, then 4 and 6.
    # round() breaks ties to even, which is the even mantissa in each range.
    magnitude = normalized.abs()
    spacing = torch.where(magnitude > 4, 2.0, torch.where(magnitude > 2, 1.0, 0.5))
    return torch.copysign(magnitude.div_(spacing).round_().mul_(spacing), normalized)


def spread_native_masters(
    codes: torch.Tensor, scale: torch.Tensor, positions: torch.Tensor, *, fp4: bool
) -> torch.Tensor:
    """Place master weights inside the rounding cells of released native codes.

    A master rebuilt as ``codes * scale`` sits at the centre of its cell, so an
    update smaller than half a cell never changes the quantized weight. Masters
    spread over the whole cell quantize to the same released values, and small
    updates then move a proportional share of them across a cell boundary.

    Args:
        codes: FP32 tensor of shape [rows, columns] with released code values in
            scale units (E2M1 or E4M3 magnitudes with sign). FP4 needs complete
            32-column blocks, one scale per row and block.
        scale: FP32 tensor of power-of-two block scales, broadcastable to
            [rows, columns].
        positions: FP32 tensor of shape [rows, columns] with values in [0, 1),
            the relative position of each master between its cell bounds.
        fp4: Codes are E2M1 routed-expert values instead of E4M3.

    Returns:
        New FP32 tensor of shape [rows, columns]. Rounding ``bf16(master) / scale``
        to the native grid gives ``codes`` for every element, and block maxima
        stay inside the range that keeps the minimal block scale. Elements
        without a usable cell remain at ``codes * scale``: zero and subnormal
        E4M3 codes, all-zero FP4 blocks, and the few candidates that BF16
        rounding carries over a cell boundary. No input is modified.
    """
    if fp4:
        if codes.shape[-1] % 32:
            raise ValueError("FP4 master spreading requires complete 32-column blocks")
        # QAT derives the scale from the block maximum. A released block whose
        # largest code is below 4 is requantized there with a smaller scale and
        # proportionally larger codes: the same values, but narrower cells,
        # because E2M1 spacing is not the same at every magnitude.
        largest = codes.abs().unflatten(-1, (-1, 32)).amax(-1, keepdim=True).expand(-1, -1, 32).flatten(-2)
        movable = largest > 0
        factor = torch.where(movable, _power_of_two_ceiling((largest / 6).clamp_min(2.0**-9)), 1.0)
        codes, scale = codes / factor, scale * factor
        magnitude = codes.abs()
        index = torch.bucketize(magnitude, magnitude.new_tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]))
        upper = magnitude.new_tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0])[index]
        lower = magnitude.new_tensor([0.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])[index]
        top = index == 7
    else:
        magnitude = codes.abs()
        mantissa, exponent = torch.frexp(magnitude)
        step = torch.ldexp(torch.ones_like(magnitude), exponent - 4)
        # The spacing halves below a power of two.
        lower = magnitude - torch.where(mantissa == 0.5, step / 4, step / 2)
        top = magnitude >= 448
        upper = torch.where(top, magnitude, magnitude + step / 2)
        movable = magnitude >= 2**-6
    # BF16 rounding shifts each bound of the composed BF16-then-native cell by
    # half a BF16 step, outward or inward depending on which neighbour a tie
    # goes to. Draw up to the outward bounds so that masters reach the true
    # edge; the check sends the candidates that land outside back to the centre.
    # The top code never moves outward: that would raise the block maximum and
    # with it the block scale.
    upper = torch.where(top, upper, upper + torch.ldexp(torch.ones_like(upper), torch.frexp(upper)[1] - 9))
    lower = lower - torch.ldexp(torch.ones_like(lower), torch.frexp(lower)[1] - 9)
    if fp4:
        lower = torch.where(index == 0, -upper, lower)
    grid = codes * scale
    candidate = torch.where(torch.signbit(codes), -scale, scale) * (lower + positions * (upper - lower))
    kept = movable & (_round_native(candidate.to(torch.bfloat16).float() / scale, fp4) == codes)
    return torch.where(kept, candidate, grid)


class _NativeQuantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx: object, values: torch.Tensor, fp4: bool, row_block: int) -> torch.Tensor:
        """Q/DQ values with native E8M0 scales.

        Args:
            ctx: Unused STE context.
            values: Tensor [..., rows, channels], channels divisible by 32 and
                rows divisible by row_block. Arbitrary leading axes are independent.
            fp4: Use E2M1 instead of E4M3 values.
            row_block: Number of adjacent rows sharing a scale (1 or 32).

        Returns:
            Independent tensor with input shape/dtype. Input storage is unchanged.
        """
        if values.ndim < 2 or values.shape[-1] % 32 or values.shape[-2] % row_block:
            raise ValueError("Native QAT requires complete quantization blocks")
        rows, cols = values.shape[-2:]
        output = torch.empty(values.shape, dtype=values.dtype, device=values.device)
        matrices = values.reshape(-1, rows, cols)
        destinations = output.reshape(-1, rows, cols)
        # Chunks bound the FP32 temporaries (about six live chunk-sized tensors)
        # but must stay large: every chunk costs a fixed number of kernel
        # launches, and with small chunks those launches, not the arithmetic,
        # set the time. Each chunk starts on a scale-row boundary.
        row_step = max(row_block, (_NATIVE_CHUNK_ELEMENTS // cols // row_block) * row_block)
        limit = 6.0 if fp4 else 448.0
        for matrix, destination in zip(matrices, destinations, strict=True):
            for start in range(0, rows, row_step):
                end = min(rows, start + row_step)
                blocks = (
                    matrix[start:end]
                    .to(torch.bfloat16)
                    .float()
                    .reshape((end - start) // row_block, row_block, cols // 32, 32)
                )
                maximum = blocks.abs().amax(dim=(1, 3), keepdim=True)
                scale = _power_of_two_ceiling((maximum / limit).clamp_min(torch.finfo(torch.float32).tiny))
                rounded = _round_native(blocks / scale, fp4)
                destination[start:end].copy_(rounded.mul_(scale).reshape(end - start, cols))
        return output

    @staticmethod
    def backward(ctx: object, gradient: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        """Pass the STE gradient through.

        Args:
            ctx: Unused context.
            gradient: Tensor [..., rows, channels] matching the forward input.

        Returns:
            The same gradient storage and no gradients for format/block metadata.
        """
        return gradient, None, None


def quantize_native(values: torch.Tensor, *, fp4: bool = False, weight: bool = False) -> torch.Tensor:
    """Apply native weight or activation QAT with an identity backward.

    Args:
        values: Tensor [..., rows, channels], with complete 32-channel blocks.
            Dense weights also require complete 32-row blocks. Expert weights
            must be passed in [..., out_features, in_features] checkpoint order.
        fp4: Use MXFP4 routed-expert weights; otherwise E4M3 values.
        weight: Dense weights share 32x32 scales; activations and FP4 weights
            share one scale per row and 32 channels.

    Returns:
        Independent Q/DQ tensor in input shape/dtype, with a straight-through
        gradient. This simulates operands, not low-precision GEMM accumulation.
    """
    return _NativeQuantization.apply(values, fp4, 32 if weight and not fp4 else 1)
