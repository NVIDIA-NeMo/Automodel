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

"""One-pass DSV4 partial interleaved RoPE with a conjugate backward."""

from __future__ import annotations

import torch

from nemo_automodel.shared.import_utils import safe_import

_HAVE_TRITON, triton = safe_import("triton")
_HAVE_LANGUAGE, tl = safe_import("triton.language")

if _HAVE_TRITON and _HAVE_LANGUAGE:

    @triton.jit
    def _rotate(
        x,
        cos,
        sin,
        out,
        count,
        SEQUENCE,
        HEADS: tl.constexpr,
        DIM: tl.constexpr,
        ROTARY: tl.constexpr,
        XB,
        XH,
        XS,
        XD,
        CB,
        CS,
        CD,
        SB,
        SS,
        SD,
        INVERSE: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Rotate [batch, heads, sequence, channels] into token-major storage.

        Args:
            x: Input tensor with logical shape [batch, heads, sequence, channels].
            cos: Cosines of shape [batch, sequence, rotary_channels], first half unique.
            sin: Sines with the same logical layout as cos.
            out: Output storage of shape [batch, sequence, heads, channels].
            count: Number of channel pairs in x.
        """
        # Widen before any index arithmetic: H64/D512 activations exceed
        # 2**31 elements at long sequence lengths.
        pair = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        valid = pair < count
        channel = (pair % (DIM // 2)) * 2
        row = pair // (DIM // 2)
        head = row % HEADS
        token = row // HEADS
        position = token % SEQUENCE
        batch = token // SEQUENCE
        offset = batch * XB + head * XH + position * XS + channel * XD
        a = tl.load(x + offset, valid, other=0)
        b = tl.load(x + offset + XD, valid, other=0)
        frequency = (channel - (DIM - ROTARY)) // 2
        rotate = channel >= DIM - ROTARY
        c = tl.load(cos + batch * CB + position * CS + frequency * CD, valid & rotate, other=0).to(tl.float32)
        s = tl.load(sin + batch * SB + position * SS + frequency * SD, valid & rotate, other=0).to(tl.float32)
        if INVERSE:
            s = -s
        af, bf = a.to(tl.float32), b.to(tl.float32)
        real = af * c - bf * s
        imaginary = af * s + bf * c
        tl.store(out + pair * 2, tl.where(rotate, real.to(a.dtype), a), valid)
        tl.store(out + pair * 2 + 1, tl.where(rotate, imaginary.to(b.dtype), b), valid)


def _launch(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rotary: int, inverse: bool) -> torch.Tensor:
    """Run a rotation without materializing a heads-to-tokens transpose.

    Args:
        x: CUDA tensor of shape [batch, heads, sequence, channels].
        cos: CUDA FP32 cosines of shape [batch, sequence, rotary_channels].
        sin: CUDA FP32 sines of the same shape as cos.
        rotary: Even number of trailing channels to rotate.
        inverse: Whether to negate sine for the backward rotation.

    Returns:
        Tensor of shape [batch, heads, sequence, channels] with token-major
        storage; its transpose(1, 2) is contiguous. No input is mutated.
    """
    batch, heads, sequence, dim = x.shape
    output = x.new_empty((batch, sequence, heads, dim))
    pairs = x.numel() // 2
    _rotate[(triton.cdiv(pairs, 1024),)](
        x,
        cos,
        sin,
        output,
        pairs,
        sequence,
        heads,
        dim,
        rotary,
        *x.stride(),
        *cos.stride(),
        *sin.stride(),
        inverse,
        1024,
        enable_fp_fusion=False,
    )
    return output.transpose(1, 2)


class _PartialRoPE(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        rotary: int,
    ) -> torch.Tensor:
        """Rotate trailing channels and retain only the frequency tables.

        Args:
            ctx: Autograd context.
            x: Tensor of shape [batch, heads, sequence, channels].
            cos: FP32 tensor of shape [batch, sequence, rotary_channels].
            sin: FP32 tensor with the same shape as cos.
            rotary: Number of trailing channels to rotate.

        Returns:
            Independent tensor with x's logical shape and dtype.
        """
        ctx.save_for_backward(cos, sin)
        ctx.rotary = rotary
        return _launch(x, cos, sin, rotary, False)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None]:
        """Apply the conjugate rotation to the upstream gradient.

        Args:
            ctx: Saved frequency tables and rotary dimension.
            grad_output: Tensor of shape [batch, heads, sequence, channels].

        Returns:
            Input gradient in the same logical layout, followed by None for
            the fixed cosine, sine and dimension inputs.
        """
        cos, sin = ctx.saved_tensors
        return _launch(grad_output, cos, sin, ctx.rotary, True), None, None, None


def apply_partial_rope_triton(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary: int,
) -> torch.Tensor:
    """Fuse partial interleaved RoPE and its conjugate backward.

    Args:
        x: CUDA FP32/FP16/BF16 tensor of shape [batch, heads, sequence, channels].
            The channel count must be even.
        cos: CUDA tensor of shape [batch, sequence, rotary_channels]; the first
            half contains the unique frequencies. Must not require gradients.
        sin: CUDA tensor with cos's shape and the same fixed-frequency contract.
        rotary: Positive even number of trailing channels, no greater than channels.

    Returns:
        Tensor of shape [batch, heads, sequence, channels] in x's dtype,
        stored in token-major order without aliasing x.
    """
    if not (_HAVE_TRITON and _HAVE_LANGUAGE):
        raise RuntimeError("DeepSeek V4 rope='triton' requires Triton")
    if x.ndim != 4 or x.shape[-1] % 2 or rotary <= 0 or rotary % 2 or rotary > x.shape[-1]:
        raise ValueError("DeepSeek V4 Triton RoPE requires rank-4 input and valid even channel dimensions")
    expected = (x.shape[0], x.shape[2], rotary)
    if cos.shape != expected or sin.shape != expected:
        raise ValueError(f"DeepSeek V4 Triton RoPE expects cosine and sine shape {expected}")
    if not x.is_cuda or cos.device != x.device or sin.device != x.device:
        raise ValueError("DeepSeek V4 Triton RoPE inputs must share one CUDA device")
    if cos.requires_grad or sin.requires_grad:
        raise ValueError("DeepSeek V4 Triton RoPE requires fixed frequency tensors")
    return _PartialRoPE.apply(x, cos, sin, rotary)
