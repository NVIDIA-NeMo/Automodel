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

"""MXFP4 (fp4 e2m1 + e8m0 block scales) pack/unpack utilities for MoE expert weights.

The packed layout matches the DeepSeek V4 Flash routed-expert checkpoint format:
two e2m1 values per int8 byte (low nibble at even column index, high nibble at the
following odd column) with one ``float8_e8m0fnu`` scale per 32 contiguous columns.
``MXFP4GroupedMM`` provides a grouped GEMM over packed weights that re-dequantizes
in backward instead of saving the dequantized tensor, so frozen expert weights stay
packed at steady state during LoRA training.
"""

import torch

MXFP4_BLOCK_SIZE = 32

# FP4 e2m1 value table: low 3 bits -> magnitude, MSB -> sign.
# Layout: [positive values for codes 0-7, negative values for codes 8-15].
_FP4_E2M1_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)

# Midpoints between consecutive positive e2m1 magnitudes, used to round to nearest.
_FP4_E2M1_MIDPOINTS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32)


def dequantize_mxfp4(packed: torch.Tensor, scales: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Unpack fp4 e2m1 packed-int8 values and apply the per-32-column e8m0 scale.

    Args:
        packed: int8 tensor of shape ``[..., K // 2]`` holding two e2m1 values per byte.
        scales: ``float8_e8m0fnu`` tensor of shape ``[..., K // 32]``.
        dtype: Output dtype.

    Returns:
        Dequantized tensor of shape ``[..., K]`` in ``dtype``.
    """
    packed_u8 = packed.contiguous().view(torch.uint8)
    low = (packed_u8 & 0x0F).long()
    high = ((packed_u8 >> 4) & 0x0F).long()
    table = _FP4_E2M1_TABLE.to(packed_u8.device)
    # Interleave (low, high) per byte so column indices match the original layout.
    fp4_vals = torch.stack([table[low], table[high]], dim=-1).flatten(-2)

    # Decode e8m0 to fp32: 2^(e - 127), with byte 0 mapping to 0 (all-zero block).
    scale_u8 = scales.contiguous().view(torch.uint8).int()
    scale_f32 = torch.where(
        scale_u8 == 0,
        torch.zeros_like(scale_u8, dtype=torch.float32),
        torch.pow(2.0, (scale_u8 - 127).float()),
    )

    scale_expanded = scale_f32.repeat_interleave(MXFP4_BLOCK_SIZE, dim=-1)
    scale_expanded = scale_expanded[..., : fp4_vals.shape[-1]]
    return (fp4_vals * scale_expanded).to(dtype)


def quantize_mxfp4(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize along the last dim to the packed mxfp4 layout used by ``dequantize_mxfp4``.

    Block scales are computed as ``2^(floor(log2(amax)) - 2)`` so that values that are
    already exactly representable (e.g. a dequantized fp4 checkpoint) round-trip
    value-exactly.

    Args:
        weight: Floating-point tensor of shape ``[..., K]`` with ``K`` divisible by 32.

    Returns:
        Tuple of (int8 packed tensor ``[..., K // 2]``, ``float8_e8m0fnu`` scales ``[..., K // 32]``).
    """
    k = weight.shape[-1]
    assert k % MXFP4_BLOCK_SIZE == 0, f"last dim {k} must be divisible by {MXFP4_BLOCK_SIZE}"

    w = weight.float()
    blocks = w.view(*w.shape[:-1], k // MXFP4_BLOCK_SIZE, MXFP4_BLOCK_SIZE)
    amax = blocks.abs().amax(dim=-1)

    # e2m1 max magnitude is 6 = 1.5 * 2^2, so the shared block exponent is
    # floor(log2(amax)) - 2. amax == 0 maps to scale byte 0 (decoded as 0).
    nonzero = amax > 0
    exp = torch.zeros_like(amax)
    exp[nonzero] = torch.floor(torch.log2(amax[nonzero])) - 2.0
    scale_bytes = torch.where(
        nonzero,
        (exp + 127.0).clamp(1.0, 254.0),
        torch.zeros_like(exp),
    ).to(torch.uint8)
    # Zero-amax blocks divide by 1 instead of 0; their codes are all zero anyway.
    scale = torch.where(nonzero, torch.pow(2.0, (scale_bytes.int() - 127).float()), torch.ones_like(exp))

    # Round each scaled magnitude to the nearest e2m1 magnitude code.
    scaled = blocks.abs() / scale.unsqueeze(-1)
    midpoints = _FP4_E2M1_MIDPOINTS.to(w.device)
    codes = torch.bucketize(scaled, midpoints).to(torch.uint8)
    codes = codes | torch.where(blocks < 0, torch.full_like(codes, 0x08), torch.zeros_like(codes))
    codes = codes.reshape(*w.shape[:-1], k)

    packed = (codes[..., 0::2] | (codes[..., 1::2] << 4)).view(torch.int8)
    return packed.contiguous(), scale_bytes.view(torch.float8_e8m0fnu).contiguous()


class MXFP4GroupedMM(torch.autograd.Function):
    """Grouped GEMM over mxfp4-packed frozen weights with dequantization on the fly.

    Saves only the packed weights for backward and re-dequantizes there, so the
    bf16 weight tensor is a transient in both passes instead of being kept alive
    by autograd. Weights are stored as ``[E, N, K]`` packed along ``K``, which is
    the natural dequantization output and the operand the backward GEMM needs
    directly (``grad_x = grad_out @ W``). The forward needs ``[E, K, N]``, which
    ``torch._grouped_mm`` consumes as a transposed view (cuBLAS transB) — no
    contiguous copy required.

    No weight gradient is produced — the base weights are frozen under LoRA.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, packed: torch.Tensor, scales: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
        w_t = dequantize_mxfp4(packed, scales, x.dtype)  # [E, N, K]
        # Pass the transposed view directly; torch._grouped_mm handles the
        # strided mat2 (transB), avoiding a full bf16 weight copy per forward.
        out = torch._grouped_mm(x, w_t.transpose(-2, -1), offs=offs)
        ctx.save_for_backward(packed, scales, offs)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        packed, scales, offs = ctx.saved_tensors
        w_t = dequantize_mxfp4(packed, scales, grad_out.dtype)  # [E, N, K] == W^T
        grad_x = torch._grouped_mm(grad_out.contiguous(), w_t, offs=offs)
        return grad_x, None, None, None
