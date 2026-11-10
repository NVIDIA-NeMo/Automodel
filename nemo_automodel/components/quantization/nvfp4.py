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

"""NVFP4 (fp4 e2m1 + fp8-e4m3 block scales + fp32 tensor scale) pack/unpack utilities for MoE expert weights.

The packed layout matches the ModelOpt NVFP4 checkpoint format (``hf_quant_config.json``:
``quant_algo: NVFP4, group_size: 16``): two e2m1 values per ``uint8`` byte (the even column in
the low nibble, the following odd column in the high nibble), one ``float8_e4m3fn`` scale per
16 contiguous columns (``weight_scale``) and one ``float32`` scale per tensor
(``weight_scale_2``), so a value decodes as ``e2m1 * e4m3 * fp32``. In-model storage keeps the
per-tensor scale per output row, which lets fused gate/up projections carry the two tensor scales
of their halves exactly: 0.5625 bytes per parameter plus 4 bytes per row.

``NVFP4GroupedMM`` provides a grouped GEMM over packed weights that re-dequantizes in backward
instead of saving the dequantized tensor, so frozen expert weights stay packed at steady state
during LoRA training. No FP4 tensor cores are needed: the GEMM runs on the dequantized bf16
operand, on Hopper or Blackwell alike.
"""

from typing import Protocol

import torch
from torch.autograd.function import FunctionCtx

from nemo_automodel.components.quantization.mxfp4 import (
    _FP4_BYTE_TABLE,
    _FP4_E2M1_MIDPOINTS,
)

NVFP4_BLOCK_SIZE = 16
E2M1_MAX = 6.0
E4M3_MAX = 448.0
# Smallest positive e4m3 value (a subnormal, 2^-9): the smallest block scale the format can store. Zero blocks
# get this scale and all-zero codes, which decodes to exactly zero.
E4M3_MIN_SUBNORMAL = 2.0**-9


# Elements per dequantization chunk: bounds the fp32 transients (index, values, scaled product) to ~1.5 GiB
# regardless of the local expert count, so a per-layer EP shard (hundreds of millions of weights) never
# materializes a full-size fp32 copy.
_DEQUANT_CHUNK_ELEMS = 2**27


def _check_last_dim(k: int) -> None:
    """Reject contraction dims that are not a whole number of 16-column blocks."""
    if k % NVFP4_BLOCK_SIZE != 0:
        raise ValueError(f"last dim {k} must be divisible by {NVFP4_BLOCK_SIZE} for NVFP4 block quantization")


def dequantize_nvfp4(
    packed: torch.Tensor, scales: torch.Tensor, global_scale: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    """Unpack fp4 e2m1 packed-uint8 values and apply the per-16-column e4m3 scale and the row scale.

    Each value decodes as ``e2m1 * float(e4m3 block scale) * row scale`` with fp32 intermediates and a single
    final rounding to ``dtype``; the work runs in row chunks of ``_DEQUANT_CHUNK_ELEMS`` elements to bound the
    transients.

    Args:
        packed: ``uint8`` (or ``int8``) tensor of shape ``[..., K // 2]`` holding two e2m1 values per byte
            (even column in the low nibble).
        scales: ``float8_e4m3fn`` tensor of shape ``[..., K // 16]``.
        global_scale: ``float32`` tensor of shape ``[...]`` (one value per row of ``packed``), broadcast
            over the row's blocks.
        dtype: Output dtype.

    Returns:
        Dequantized tensor of shape ``[..., K]`` in ``dtype``.
    """
    packed_u8 = packed.contiguous().view(torch.uint8)
    lead_shape = packed_u8.shape[:-1]
    k = packed_u8.shape[-1] * 2
    n_blocks = k // NVFP4_BLOCK_SIZE
    if scales.shape != (*lead_shape, n_blocks):
        raise ValueError(f"scales shape {tuple(scales.shape)} does not match packed {tuple(packed_u8.shape)}")
    if global_scale.shape != tuple(lead_shape):
        raise ValueError(f"global_scale shape {tuple(global_scale.shape)} must equal the row shape {tuple(lead_shape)}")
    rows = packed_u8.reshape(-1, k // 2)
    scale_rows = scales.reshape(-1, n_blocks)
    global_rows = global_scale.reshape(-1)
    out = torch.empty((rows.shape[0], k), dtype=dtype, device=packed_u8.device)
    # One gather over the 256-row byte table yields both e2m1 values per byte, interleaved (low, high) so
    # column indices match the original layout. An int32 index halves the index memory of ``.long()``.
    table = _FP4_BYTE_TABLE.to(device=packed_u8.device, dtype=torch.bfloat16)
    step = max(1, _DEQUANT_CHUNK_ELEMS // k)
    for r0 in range(0, rows.shape[0], step):
        chunk = rows[r0 : r0 + step]
        pairs = torch.index_select(table, 0, chunk.reshape(-1).to(torch.int32))  # [n * K // 2, 2]
        fp4_vals = pairs.view(chunk.shape[0], n_blocks, NVFP4_BLOCK_SIZE).to(torch.float32)  # exact: |v| <= 6
        # Block scale times row scale, an fp32 intermediate (the e2m1 values are exact in fp32).
        scale = scale_rows[r0 : r0 + step].to(torch.float32) * global_rows[r0 : r0 + step].unsqueeze(-1)
        out[r0 : r0 + step] = (fp4_vals * scale.unsqueeze(-1)).view(chunk.shape[0], k).to(dtype)
    return out.view(*lead_shape, k)


def quantize_nvfp4(
    weight: torch.Tensor, *, global_scale: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize along the last dim to the packed NVFP4 layout used by ``dequantize_nvfp4``.

    Follows the ModelOpt convention ``global = amax / (6 * 448)``, ``block_scale = e4m3(amax(block) / 6 / global)``
    and ``code = round_to_e2m1(w / (block_scale * global))``, with one global scale per row unless
    ``global_scale`` is given (see ``quantize_nvfp4_modelopt`` for the per-tensor checkpoint convention).

    Args:
        weight: Floating-point tensor of shape ``[..., K]`` with ``K`` divisible by 16.
        global_scale: Optional ``float32`` tensor broadcastable to ``weight.shape[:-1]`` with the row scales
            to use instead of the per-row amax rule.

    Returns:
        Tuple of (``uint8`` packed codes ``[..., K // 2]``, ``float8_e4m3fn`` block scales ``[..., K // 16]``,
        ``float32`` row scales ``[...]``).
    """
    k = weight.shape[-1]
    _check_last_dim(k)
    w = weight.to(torch.float32)
    row_shape = w.shape[:-1]
    blocks = w.reshape(*row_shape, k // NVFP4_BLOCK_SIZE, NVFP4_BLOCK_SIZE)
    if global_scale is None:
        row_amax = w.abs().amax(dim=-1)  # [...]
        global_scale = torch.where(row_amax > 0, row_amax / (E2M1_MAX * E4M3_MAX), torch.ones_like(row_amax))
    else:
        global_scale = (
            torch.as_tensor(global_scale, dtype=torch.float32, device=w.device).expand(row_shape).contiguous()
        )
    block_amax = blocks.abs().amax(dim=-1)  # [..., K // 16]
    block_scale = block_amax / E2M1_MAX / global_scale.unsqueeze(-1)
    block_scale = block_scale.clamp(min=E4M3_MIN_SUBNORMAL, max=E4M3_MAX).to(torch.float8_e4m3fn)
    scale = block_scale.to(torch.float32) * global_scale.unsqueeze(-1)  # exact decode scale, [..., K // 16]
    scaled = blocks / scale.unsqueeze(-1)  # nominally in [-6, 6]; block-scale rounding can push |x| slightly past 6
    magnitude = torch.bucketize(scaled.abs().clamp(max=E2M1_MAX), _FP4_E2M1_MIDPOINTS.to(w.device)).to(torch.uint8)
    codes = torch.where(scaled < 0, magnitude | 8, magnitude).reshape(*row_shape, k)
    packed = codes[..., 0::2] | (codes[..., 1::2] << 4)  # low nibble = even column
    return packed.contiguous(), block_scale.contiguous(), global_scale.contiguous()


def quantize_nvfp4_modelopt(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a 2-D ``[out, in]`` matrix with one tensor scale, as a ModelOpt NVFP4 checkpoint stores it.

    Returns:
        Tuple of (``weight``: ``uint8`` ``[out, in // 2]``, ``weight_scale``: ``float8_e4m3fn`` ``[out, in // 16]``,
        ``weight_scale_2``: 0-dim ``float32``). ``dequantize_nvfp4`` takes ``weight_scale_2`` expanded to one
        value per row.
    """
    amax = weight.to(torch.float32).abs().amax()
    tensor_scale = torch.where(amax > 0, amax / (E2M1_MAX * E4M3_MAX), torch.ones_like(amax))
    packed, block_scale, _ = quantize_nvfp4(weight, global_scale=tensor_scale)
    return packed, block_scale, tensor_scale.reshape(())


class _NVFP4Context(Protocol):
    """Autograd context retaining the 4 packed-weight tensors for backward."""

    @property
    def saved_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...


class NVFP4GroupedMM(torch.autograd.Function):
    """Grouped GEMM over NVFP4-packed frozen weights with dequantization on the fly.

    Saves only the packed codes and scales for backward and re-dequantizes there, so the
    floating-point weight tensor is a transient in both passes instead of being kept alive
    by autograd. Weights are stored as ``[E, N, K]`` packed along ``K``, the natural
    dequantization output and the operand the backward GEMM needs directly
    (``grad_x = grad_out @ W``). The forward needs ``[E, K, N]``, which ``torch._grouped_mm``
    consumes as a transposed view (cuBLAS transB) without a contiguous copy.

    No weight gradient is produced: the base weights are frozen under LoRA.
    """

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        x: torch.Tensor,
        packed: torch.Tensor,
        scales: torch.Tensor,
        global_scale: torch.Tensor,
        offs: torch.Tensor,
    ) -> torch.Tensor:
        """Multiply activations grouped by expert by packed frozen weights.

        Args:
            x: Tensor of shape [tokens, in_dim], grouped contiguously by local expert.
            packed: ``uint8`` tensor of shape [local_experts, out_dim, in_dim // 2].
            scales: ``float8_e4m3fn`` tensor of shape [local_experts, out_dim, in_dim // 16].
            global_scale: ``float32`` tensor of shape [local_experts, out_dim].
            offs: Int32 tensor of shape [local_experts], holding cumulative token counts.
                All tensors must be on the same device.

        Returns:
            Tensor of shape [tokens, out_dim] with the activation dtype.
        """
        w_t = dequantize_nvfp4(packed, scales, global_scale, x.dtype)  # [E, N, K]
        out = torch._grouped_mm(x, w_t.transpose(-2, -1), offs=offs)
        ctx.save_for_backward(packed, scales, global_scale, offs)
        return out

    @staticmethod
    def backward(ctx: _NVFP4Context, *grad_outputs: torch.Tensor) -> tuple[torch.Tensor, None, None, None, None]:
        """Backpropagate through the activations while keeping the packed base frozen.

        Args:
            grad_outputs: Exactly one gradient tensor of shape [tokens, out_dim] with the activation dtype,
                grouped by local expert like the forward input.

        Returns:
            Tuple of (grad_x of shape [tokens, in_dim] in the activation dtype, None for packed, scales,
            global_scale and offs, which receive no gradient).
        """
        (grad_out,) = grad_outputs
        packed, scales, global_scale, offs = ctx.saved_tensors
        w_t = dequantize_nvfp4(packed, scales, global_scale, grad_out.dtype)  # [E, N, K] == W^T
        grad_x = torch._grouped_mm(grad_out.contiguous(), w_t, offs=offs)
        return grad_x, None, None, None, None
