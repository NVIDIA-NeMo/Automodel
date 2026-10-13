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

"""Shared interleaved rotary kernel with PyTorch complex-multiply precision.

The kernel rotates adjacent channel pairs in FP32, rounds once to the input
value dtype, and copies any leading unrotated channels. Frequencies are already
computed by the owning model, so it preserves their exact cosine/sine values.
The explicit FMA order matches CUDA ``c10::complex<float>`` multiplication in
both forward and the conjugate rotation used for the input gradient.
"""

from __future__ import annotations

import torch

from nemo_automodel.shared.import_utils import safe_import

_HAVE_TRITON, triton = safe_import("triton")
_HAVE_TRITON_LANGUAGE, tl = safe_import("triton.language")
HAVE_TRITON: bool = bool(_HAVE_TRITON and _HAVE_TRITON_LANGUAGE)

# Elements of ``values`` handled by one program: 32 per thread at four warps.
_ELEMENTS_PER_PROGRAM = 4096


if HAVE_TRITON:

    @triton.jit
    def _rope_kernel(
        values_ptr,
        frequencies_ptr,
        output_ptr,
        rows,
        sequence,
        heads,
        channels,
        pairs,
        pair_offset,
        stride_batch,
        stride_sequence,
        stride_head,
        NEGATE_SIN: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):  # pragma: no cover - Triton JIT executes on the GPU, outside Python tracing.
        """Copy the leading channels and rotate the trailing pairs of ``BLOCK_R`` rows.

        Every row is read once as ``BLOCK_K`` adjacent channel pairs and written once. Pairs
        before the rotary block are passed through ``tl.where`` rather than multiplied by a
        unit frequency, so non-finite prefix values are copied bit for bit.

        Args:
            values_ptr: Input viewed as [batch, sequence, heads, channels] with a unit channel
                stride; ``heads`` is 1 for three-dimensional inputs.
            frequencies_ptr: Contiguous FP32 tensor of shape [batch * sequence, pairs, 2]
                holding (cos, sin) per rotary pair.
            output_ptr: Contiguous output of shape [rows, channels].
            rows: ``batch * sequence * heads``.
            sequence: Sequence length.
            heads: Head count, 1 for three-dimensional inputs.
            channels: Channel count.
            pairs: Rotary pair count; the last ``2 * pairs`` channels rotate.
            pair_offset: Channel pairs preceding the rotary block, rounded up so the pair grid
                covers channel 0 (it then starts at channel -1 for an odd prefix, masked off).
            stride_batch: Input batch stride in elements.
            stride_sequence: Input sequence stride in elements.
            stride_head: Input head stride in elements; ignored when ``heads`` is 1.
            NEGATE_SIN: Rotate by the conjugate frequency.
            BLOCK_R: Rows handled by one program.
            BLOCK_K: Power-of-two pair extent per row, at least ``pair_offset + pairs``.
        """
        row = tl.program_id(0).to(tl.int64) * BLOCK_R + tl.arange(0, BLOCK_R)
        row_mask = row < rows
        head = row % heads
        token = row // heads
        batch = token // sequence
        position = token % sequence

        pair = tl.arange(0, BLOCK_K)
        component = tl.arange(0, 2)
        channel = (channels - 2 * pairs - 2 * pair_offset) + 2 * pair[:, None] + component[None, :]
        mask = row_mask[:, None, None] & ((channel >= 0) & (channel < channels))[None, :, :]
        input_offset = batch * stride_batch + position * stride_sequence + head * stride_head
        x = tl.load(values_ptr + input_offset[:, None, None] + channel[None, :, :], mask=mask, other=0)
        x0, x1 = tl.split(x)

        rotary = pair - pair_offset
        rotary_mask = (rotary >= 0) & (rotary < pairs)
        frequency_offset = (token[:, None, None] * pairs + rotary[None, :, None]) * 2 + component[None, None, :]
        frequency_mask = row_mask[:, None, None] & rotary_mask[None, :, None] & (component < 2)[None, None, :]
        frequency = tl.load(frequencies_ptr + frequency_offset, mask=frequency_mask, other=0.0)
        cos, sin = tl.split(frequency)
        if NEGATE_SIN:
            sin = -sin

        x0_fp32 = x0.to(tl.float32)
        x1_fp32 = x1.to(tl.float32)
        # c10::complex<float> multiplies as (a*c - b*d, a*d + b*c); nvcc fuses the first product.
        rotated0 = tl.fma(x0_fp32, cos, -(x1_fp32 * sin))
        rotated1 = tl.fma(x0_fp32, sin, x1_fp32 * cos)
        y0 = tl.where(rotary_mask[None, :], rotated0.to(x0.dtype), x0)
        y1 = tl.where(rotary_mask[None, :], rotated1.to(x1.dtype), x1)
        output_offset = row * channels
        tl.store(output_ptr + output_offset[:, None, None] + channel[None, :, :], tl.join(y0, y1), mask=mask)


def _launch(values: torch.Tensor, frequencies: torch.Tensor, *, negate_sin: bool) -> torch.Tensor:
    """Run the fused kernel over ``values`` and return a new contiguous tensor.

    Args:
        values: CUDA tensor of shape [batch, sequence, channels] or
            [batch, sequence, heads, channels]. Any leading strides are read in place; a
            strided channel dimension is made contiguous first.
        frequencies: Contiguous FP32 tensor of shape [batch, sequence, pairs, 2].
        negate_sin: Rotate by the conjugate frequency.

    Returns:
        Contiguous tensor with the shape and dtype of ``values``.
    """
    if values.stride(-1) != 1:
        values = values.contiguous()
    if values.ndim == 3:
        batch_size, sequence, channels = values.shape
        heads, stride_head = 1, 0
    else:
        batch_size, sequence, heads, channels = values.shape
        stride_head = values.stride(2)
    pairs = frequencies.shape[-2]
    rows = batch_size * sequence * heads
    output = torch.empty(values.shape, dtype=values.dtype, device=values.device)
    if rows == 0 or channels == 0:
        return output
    pair_offset = (channels - 2 * pairs + 1) // 2
    block_k = triton.next_power_of_2(pair_offset + pairs)
    block_r = max(1, _ELEMENTS_PER_PROGRAM // (2 * block_k))
    num_warps = min(16, max(4, (2 * block_k * block_r) // 1024))
    _rope_kernel[(triton.cdiv(rows, block_r),)](
        values,
        frequencies,
        output,
        rows,
        sequence,
        heads,
        channels,
        pairs,
        pair_offset,
        values.stride(0),
        values.stride(1),
        stride_head,
        NEGATE_SIN=negate_sin,
        BLOCK_R=block_r,
        BLOCK_K=block_k,
        num_warps=num_warps,
    )
    return output


class _FusedRope(torch.autograd.Function):
    """Rotate with the fused kernel and rotate the gradient by the conjugate frequency."""

    @staticmethod
    def forward(ctx: object, values: torch.Tensor, frequencies: torch.Tensor, inverse: bool) -> torch.Tensor:
        ctx.save_for_backward(frequencies)
        ctx.inverse = inverse
        return _launch(values, frequencies, negate_sin=inverse)

    @staticmethod
    def backward(ctx: object, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, None, None]:
        if not ctx.needs_input_grad[0]:
            return None, None, None
        (frequencies,) = ctx.saved_tensors
        return _launch(grad_output, frequencies, negate_sin=not ctx.inverse), None, None


def apply_complex_rope(values: torch.Tensor, frequencies: torch.Tensor, *, inverse: bool = False) -> torch.Tensor:
    """Rotate trailing channel pairs using a model's cached complex frequencies.

    Args:
        values: CUDA BF16, FP16, or FP32 tensor of shape [batch, sequence, channels]
            or [batch, sequence, heads, channels]. Leading dimensions may be strided.
        frequencies: CUDA complex64 tensor of shape [batch, sequence, rotary_pairs].
            Must not require gradients. The rotary width must fit within channels.
        inverse: Whether to conjugate the rotation.

    Returns:
        New contiguous tensor with the shape and dtype of ``values``. Its input
        gradient uses the conjugate rotation with the same FP32 rounding order.
    """
    return _FusedRope.apply(values, torch.view_as_real(frequencies.resolve_conj()).contiguous(), inverse)
