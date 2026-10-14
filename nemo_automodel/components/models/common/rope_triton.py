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

"""Fused Triton rotary embedding for DeepSeek V4.1, forward and backward in one kernel each.

``_apply_rope`` in :mod:`nemo_automodel.components.models.deepseek_v41.attention` rotates the
last ``2 * rotary_pairs`` channels of a [batch, sequence, channels] or
[batch, sequence, heads, channels] tensor as adjacent complex pairs and copies the remaining
channels. Eagerly that is a complex multiply plus ``torch.cat`` in the forward and several
zero-filled slice gradients in the backward, so every call streams the full tensor through
memory more than once. This module writes the copied and the rotated channels in a single
pass, and its backward rotates the incoming gradient by the conjugate in one pass as well.
``angles`` never receives a gradient here; inputs whose angles require one, inputs that are
not on CUDA, and dtypes outside BF16/FP16/FP32 take the eager path.

Numerics follow the eager path by construction wherever PyTorch allows it:

* The frequencies come from the same ``torch.polar(torch.ones_like(angles), angles)`` call the
  eager path runs, so the ``cos``/``sin`` bits are identical rather than relying on libdevice
  agreeing with the CUDA math library PyTorch was built with. They are computed once per
  forward call and saved for the backward, which is the same footprint as the eager path's
  saved ``frequencies``.
* The complex product is evaluated in FP32 with explicit FMAs. Eager CUDA complex
  multiplication contracts the imaginary component differently across PyTorch/CUDA builds
  and devices. A small eager multiplication selects the matching order once per device;
  only that choice is cached for the process lifetime. The same order is used for the
  conjugate in the backward. Unrecognized arithmetic defaults to the first-product order.
* The FP32 result is rounded to ``values.dtype`` with round-to-nearest-even, exactly as
  ``Tensor.to`` rounds.
"""

from __future__ import annotations

from functools import lru_cache

import torch

from nemo_automodel.shared.import_utils import safe_import

_HAVE_TRITON, triton = safe_import("triton")
_HAVE_TRITON_LANGUAGE, tl = safe_import("triton.language")
HAVE_TRITON: bool = bool(_HAVE_TRITON and _HAVE_TRITON_LANGUAGE)

# Elements of ``values`` handled by one program: 32 per thread at four warps.
_ELEMENTS_PER_PROGRAM = 4096
_FUSED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


@lru_cache(maxsize=16)
@torch.compiler.assume_constant_result
def _imaginary_fma_uses_second_product(device: torch.device) -> bool:
    """Identify eager complex multiplication's contraction for this CUDA device.

    The bounded, process-local cache retains only the arithmetic choice, not tensors.
    Calibration synchronizes on first use, before the first Triton kernel compilation.
    FP64 CPU products distinguish the two FP32 FMA orders without using a CUDA FMA
    implementation as the reference or changing the random-number generator state.

    Args:
        device: CUDA device on which the eager reference will execute.

    Returns:
        Whether the eager reference matches the second-product contraction exactly.
        Otherwise, use the first-product contraction.
    """
    values = torch.tensor(
        [[1.2345, -0.9876], [-1.392927, 0.757227], [0.812345, 1.987654], [-0.765432, -1.234567]],
        dtype=torch.float32,
        device="cpu",
    )
    frequencies = torch.tensor([[0.8765, 0.4814]], dtype=torch.float32, device="cpu")
    x0, x1 = values.double().unbind(-1)
    cos, sin = frequencies.double().unbind(-1)
    real = (x0 * cos - (x1 * sin).float().double()).float()
    second = torch.stack((real, (x1 * cos + (x0 * sin).float().double()).float()), dim=-1)
    actual = torch.view_as_real(
        torch.view_as_complex(values).to(device) * torch.view_as_complex(frequencies).to(device)
    ).cpu()
    return torch.equal(actual, second)


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
        IMAGINARY_SECOND_PRODUCT: tl.constexpr,
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
            IMAGINARY_SECOND_PRODUCT: Fuse ``x1 * cos`` in the imaginary component
                to match the installed eager CUDA complex multiplication.
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
        rotated0 = tl.fma(x0_fp32, cos, -(x1_fp32 * sin))
        # Both branches compute x0 * sin + x1 * cos. FMA rounds its product and
        # sum together; the other product rounds separately, so swapping the
        # fused product can change the last bits. Match the contraction order
        # observed in PyTorch's complex multiply on this device for exact parity.
        if IMAGINARY_SECOND_PRODUCT:
            rotated1 = tl.fma(x1_fp32, cos, x0_fp32 * sin)
        else:
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
        IMAGINARY_SECOND_PRODUCT=_imaginary_fma_uses_second_product(values.device),
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


def _apply_rope_torch(values: torch.Tensor, frequencies: torch.Tensor, *, inverse: bool = False) -> torch.Tensor:
    """Eager rotation, mirroring ``attention._apply_rope`` so the fallback needs no circular import.

    Args:
        values: Tensor of shape [batch, sequence, channels] or [batch, sequence, heads, channels].
        frequencies: Complex tensor of shape [batch, sequence, rotary_pairs].
        inverse: Conjugate the rotation.

    Returns:
        Tensor with the shape and dtype of ``values``, in independent storage.
    """
    rotary_dim = frequencies.shape[-1] * 2
    pairs = torch.view_as_complex(values[..., -rotary_dim:].float().unflatten(-1, (-1, 2)).contiguous())
    if values.ndim == 4:
        frequencies = frequencies.unsqueeze(2)
    if inverse:
        frequencies = frequencies.conj()
    rotated = torch.view_as_real(pairs * frequencies).flatten(-2).to(values.dtype)
    return torch.cat((values[..., :-rotary_dim], rotated), dim=-1)


def _uses_fused_kernel(values: torch.Tensor, frequencies: torch.Tensor) -> bool:
    """Check the shared kernel's device, layout, dtype, and autograd contract.

    Args:
        values: Tensor of shape [batch, sequence, channels] or [batch, sequence, heads, channels].
        frequencies: Complex tensor of shape [batch, sequence, rotary_pairs].

    Returns:
        Whether the inputs can use the Triton implementation.
    """
    return (
        HAVE_TRITON
        and values.is_cuda
        and frequencies.is_cuda
        and not frequencies.requires_grad
        and frequencies.dtype == torch.complex64
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


def apply_complex_rope(values: torch.Tensor, frequencies: torch.Tensor, *, inverse: bool = False) -> torch.Tensor:
    """Rotate trailing channel pairs using a model's cached complex frequencies.

    Args:
        values: Tensor of shape [batch, sequence, channels]
            or [batch, sequence, heads, channels]. Leading dimensions may be strided.
        frequencies: Complex tensor of shape [batch, sequence, rotary_pairs].
            The rotary width must fit within channels. CUDA BF16/FP16/FP32 values
            and cached complex64 frequencies use Triton; other inputs use eager
            operations, preserving frequency gradients when required.
        inverse: Whether to conjugate the rotation.

    Returns:
        New contiguous tensor with the shape and dtype of ``values``. Its input
        gradient uses the conjugate rotation with the same FP32 rounding order.
    """
    if _uses_fused_kernel(values, frequencies):
        return _FusedRope.apply(values, torch.view_as_real(frequencies.resolve_conj()).contiguous(), inverse)
    return _apply_rope_torch(values, frequencies, inverse=inverse)
