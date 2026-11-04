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

"""Training Q RMSNorm plus partial interleaved RoPE in token-major storage.

The two BF16 rounding boundaries are intentional: normalize before rotating,
and rotate the upstream gradient before applying RMSNorm backward.
"""

from __future__ import annotations

import torch

from nemo_automodel.shared.import_utils import safe_import

_HAVE_TRITON, triton = safe_import("triton")
_HAVE_LANGUAGE, tl = safe_import("triton.language")

if _HAVE_TRITON and _HAVE_LANGUAGE:

    @triton.jit
    def _qnorm_rope(
        x,
        cos,
        sin,
        out,
        rstd,
        rows,
        S,
        H: tl.constexpr,
        D: tl.constexpr,
        R: tl.constexpr,
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
        EPS: tl.constexpr,
        M: tl.constexpr,
        K: tl.constexpr,
    ):
        """Normalize and rotate M query rows per program into token-major storage.

        Args:
            x: Input with logical shape [batch, heads, sequence, channels] and
                strides (XB, XH, XS, XD); D is channels and R rotary_channels.
            cos: Cosines of shape [batch, sequence, rotary_channels] with strides
                (CB, CS, CD); the first half holds the unique pair frequencies.
            sin: Sines with cos's layout and strides (SB, SS, SD).
            out: Contiguous output storage of shape [batch, sequence, heads, channels].
            rstd: FP32 inverse RMS of shape [batch * sequence * heads], one per
                row in the output's token-major order.
            rows: batch * sequence * heads.
        """
        # Widen before any index arithmetic: H64/D512 activations exceed
        # 2**31 elements at long sequence lengths.
        row = tl.program_id(0).to(tl.int64) * M + tl.arange(0, M)
        channel = tl.arange(0, K).to(tl.int64) * 2
        token = row // H
        batch, position, head = token // S, token % S, row % H
        offset = (batch * XB + head * XH + position * XS)[:, None] + channel[None, :] * XD
        mask = (row[:, None] < rows) & (channel[None, :] < D)
        a = tl.load(x + offset, mask, other=0).to(tl.float32)
        b = tl.load(x + offset + XD, mask, other=0).to(tl.float32)
        inv = tl.rsqrt(tl.sum(a * a + b * b, 1) / D + EPS)
        tl.store(rstd + row, inv, row < rows)
        # Preserve the normalization's output cast before the FP32 rotation.
        a = (a * inv[:, None]).to(x.dtype.element_ty).to(tl.float32)
        b = (b * inv[:, None]).to(x.dtype.element_ty).to(tl.float32)
        frequency = (channel - (D - R)) // 2
        rotate = channel >= D - R
        cmask = mask & rotate[None, :]
        c = tl.load(cos + (batch * CB + position * CS)[:, None] + frequency[None, :] * CD, cmask, 0)
        s = tl.load(sin + (batch * SB + position * SS)[:, None] + frequency[None, :] * SD, cmask, 0)
        y0 = tl.where(rotate[None, :], a * c - b * s, a)
        y1 = tl.where(rotate[None, :], a * s + b * c, b)
        output = row[:, None] * D + channel[None, :]
        tl.store(out + output, y0, mask)
        tl.store(out + output + 1, y1, mask)

    @triton.jit
    def _qnorm_rope_backward(
        x,
        grad,
        cos,
        sin,
        rstd,
        out,
        rows,
        S,
        H: tl.constexpr,
        D: tl.constexpr,
        R: tl.constexpr,
        XB,
        XH,
        XS,
        XD,
        GB,
        GH,
        GS,
        GD,
        CB,
        CS,
        CD,
        SB,
        SS,
        SD,
        M: tl.constexpr,
        K: tl.constexpr,
    ):
        """Rotate the upstream gradient back, then apply the RMSNorm input gradient.

        Args:
            x: Saved forward input with logical shape [batch, heads, sequence,
                channels] and strides (XB, XH, XS, XD).
            grad: Upstream gradient with x's logical shape and strides (GB, GH, GS, GD).
            cos: Cosines of shape [batch, sequence, rotary_channels] with strides (CB, CS, CD).
            sin: Sines with cos's layout and strides (SB, SS, SD).
            rstd: FP32 inverse RMS of shape [batch * sequence * heads] from the forward.
            out: Contiguous input-gradient storage of shape [batch, sequence, heads, channels].
            rows: batch * sequence * heads.
        """
        row = tl.program_id(0).to(tl.int64) * M + tl.arange(0, M)
        channel = tl.arange(0, K).to(tl.int64) * 2
        token = row // H
        batch, position, head = token // S, token % S, row % H
        mask = (row[:, None] < rows) & (channel[None, :] < D)
        xo = (batch * XB + head * XH + position * XS)[:, None] + channel[None, :] * XD
        go = (batch * GB + head * GH + position * GS)[:, None] + channel[None, :] * GD
        a = tl.load(x + xo, mask, 0).to(tl.float32)
        b = tl.load(x + xo + XD, mask, 0).to(tl.float32)
        ga = tl.load(grad + go, mask, 0).to(tl.float32)
        gb = tl.load(grad + go + GD, mask, 0).to(tl.float32)
        frequency = (channel - (D - R)) // 2
        rotate = channel >= D - R
        cmask = mask & rotate[None, :]
        c = tl.load(cos + (batch * CB + position * CS)[:, None] + frequency[None, :] * CD, cmask, 0)
        s = tl.load(sin + (batch * SB + position * SS)[:, None] + frequency[None, :] * SD, cmask, 0)
        da = tl.where(rotate[None, :], ga * c + gb * s, ga).to(x.dtype.element_ty).to(tl.float32)
        db = tl.where(rotate[None, :], gb * c - ga * s, gb).to(x.dtype.element_ty).to(tl.float32)
        inv = tl.load(rstd + row, row < rows, 0)
        dot = tl.sum(da * a + db * b, 1) / D
        correction = dot * inv * inv
        dx0 = (da - a * correction[:, None]) * inv[:, None]
        dx1 = (db - b * correction[:, None]) * inv[:, None]
        output = row[:, None] * D + channel[None, :]
        tl.store(out + output, dx0, mask)
        tl.store(out + output + 1, dx1, mask)


class _QNormRoPE(torch.autograd.Function):
    """Autograd implementation with FP32 saved inverse RMS."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        rotary: int,
        eps: float,
    ) -> torch.Tensor:
        """Normalize and rotate Q while saving its FP32 inverse RMS.

        Args:
            ctx: Autograd context.
            x: Query tensor of shape [batch, heads, sequence, channels], any strides.
            cos: Fixed FP32 cosines of shape [batch, sequence, rotary_channels];
                the first half holds the unique pair frequencies.
            sin: Fixed FP32 sines with cos's shape and frequency order.
            rotary: Number of trailing channels to rotate.
            eps: Normalization epsilon.

        Returns:
            Tensor of shape [batch, heads, sequence, channels] in x's dtype, backed
            by [batch, sequence, heads, channels] storage so transpose(1, 2) is
            contiguous. Saves x, the tables, and FP32 inverse RMS of shape
            [batch * sequence * heads] in token-major row order.
        """
        batch, heads, sequence, dim = x.shape
        output = x.new_empty((batch, sequence, heads, dim))
        inv = torch.empty(batch * sequence * heads, device=x.device, dtype=torch.float32)
        rows = inv.numel()
        _qnorm_rope[(triton.cdiv(rows, 4),)](
            x,
            cos,
            sin,
            output,
            inv,
            rows,
            sequence,
            heads,
            dim,
            rotary,
            *x.stride(),
            *cos.stride(),
            *sin.stride(),
            eps,
            4,
            triton.next_power_of_2(dim // 2),
            num_warps=4,
            enable_fp_fusion=False,
        )
        ctx.save_for_backward(x, cos, sin, inv)
        ctx.rotary = rotary
        return output.transpose(1, 2)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None, None]:
        """Apply conjugate rotation then the RMSNorm input gradient.

        Args:
            ctx: Saved input, frequencies, and inverse RMS.
            grad_output: Upstream gradient of shape [batch, heads, sequence,
                channels], any strides.

        Returns:
            Input gradient of shape [batch, heads, sequence, channels] in x's dtype
            with [batch, sequence, heads, channels] storage, then None for the
            fixed cosine, sine, rotary and epsilon arguments.
        """
        x, cos, sin, inv = ctx.saved_tensors
        batch, heads, sequence, dim = x.shape
        output = x.new_empty((batch, sequence, heads, dim))
        _qnorm_rope_backward[(triton.cdiv(inv.numel(), 4),)](
            x,
            grad_output,
            cos,
            sin,
            inv,
            output,
            inv.numel(),
            sequence,
            heads,
            dim,
            ctx.rotary,
            *x.stride(),
            *grad_output.stride(),
            *cos.stride(),
            *sin.stride(),
            4,
            triton.next_power_of_2(dim // 2),
            num_warps=4,
            enable_fp_fusion=False,
        )
        return output.transpose(1, 2), None, None, None, None


def qnorm_rope_triton(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary: int,
    eps: float,
) -> torch.Tensor:
    """Normalize Q and rotate its trailing interleaved pairs without a transpose copy.

    Args:
        x: CUDA tensor with logical shape [batch, heads, sequence, channels].
        cos: Fixed cosine table of shape [batch, sequence, rotary_channels].
        sin: Fixed sine table with cos's shape.
        rotary: Positive even number of trailing channels to rotate.
        eps: Positive RMSNorm epsilon.

    Returns:
        Normalized and rotated tensor with x's logical shape and token-major storage.
    """
    if not (_HAVE_TRITON and _HAVE_LANGUAGE):
        raise RuntimeError("Fused DeepSeek V4 Q normalization and RoPE require Triton")
    if x.ndim != 4 or x.shape[-1] % 2 or rotary <= 0 or rotary % 2 or rotary > x.shape[-1] or eps <= 0:
        raise ValueError("Expected rank-four input, valid even rotary dimensions, and positive epsilon")
    expected = (x.shape[0], x.shape[2], rotary)
    if cos.shape != expected or sin.shape != expected:
        raise ValueError(f"Expected fixed frequency tables of shape {expected}")
    if not x.is_cuda or cos.device != x.device or sin.device != x.device:
        raise ValueError("Q normalization and RoPE inputs must share a CUDA device")
    if cos.requires_grad or sin.requires_grad:
        raise ValueError("Q normalization and RoPE require fixed frequency tables")
    # The model stores rotary outputs in its activation dtype; promote those
    # already-rounded values exactly as the existing partial-RoPE path does.
    return _QNormRoPE.apply(x, cos.float(), sin.float(), rotary, eps)
