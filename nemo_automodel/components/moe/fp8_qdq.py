# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Block E4M3 fake quantization for BF16 FFN training with identity STE.

This simulates quantization, not FP8 GEMM accumulation. Parameters stay in their
original storage and are quantized afresh on every forward. No Hadamard rotation
or power-of-two scales are used. Weight blocks are 128x128; activation groups
are 128 channels per token. Block-aligned dimensions are required so local TP
shards cannot silently introduce partial-block scales.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd.function import FunctionCtx


@dataclass(frozen=True, kw_only=True)
class FP8QDQConfig:
    """Independent weight and activation fake-quantization switches."""

    weights: bool = False
    activations: bool = False

    @property
    def enabled(self) -> bool:
        """Whether either operand requires fake quantization."""
        return self.weights or self.activations

    def build(self, *, in_features: int, out_features: int, bias: bool, dtype: torch.dtype) -> nn.Linear:
        """Build a BF16-compatible linear with unchanged weight/bias state keys."""
        if in_features % 128 or out_features % 128:
            raise ValueError("FP8 FFN QDQ requires projection dimensions divisible by 128")
        return _QDQLinear(in_features, out_features, bias=bias, dtype=dtype, qdq=self)

    def activation(self, x: torch.Tensor) -> torch.Tensor:
        """Fake-quantize independent channel groups, preserving identity gradients.

        Args:
            x: Local tensor of shape [..., hidden], with arbitrary leading
                dimensions and hidden divisible by 128.
        Returns:
            Tensor with the same shape/dtype; does not mutate x.
        """
        if not self.activations:
            return x
        if x.shape[-1] % 128:
            raise ValueError("FP8 activation QDQ requires hidden divisible by 128")
        with torch.no_grad():
            groups = x.float().reshape(*x.shape[:-1], x.shape[-1] // 128, 128)
            scale = groups.abs().amax(-1, keepdim=True).clamp_min(1e-10) / 448.0
            value = (groups / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scale
            value = value.reshape(x.shape).to(x.dtype)
        return _IdentitySTE.apply(x, value)

    def weight(self, weight: torch.Tensor) -> torch.Tensor:
        """Fake-quantize independent 128x128 blocks in projection layout.

        Args:
            weight: Local tensor of shape [..., out_features, in_features].
                Leading expert axes are independent. Both trailing dimensions
                must be divisible by 128; FSDP weights must first be unsharded.
        Returns:
            Tensor of the same shape/dtype with identity gradients, no mutation.
        """
        if not self.weights:
            return weight
        rows, cols = weight.shape[-2:]
        if rows % 128 or cols % 128:
            raise ValueError("FP8 weight QDQ requires projection dimensions divisible by 128")
        with torch.no_grad():
            blocks = weight.float().reshape(*weight.shape[:-2], rows // 128, 128, cols // 128, 128)
            amax = blocks.abs().amax(dim=(-3, -1), keepdim=True)
            # Match NeMo-RL refit: multiply by 448/amax, dequantize with its
            # reciprocal (rather than independently rounding amax/448).
            multiplier = torch.where(amax == 0, 1.0, 448.0 / amax)
            value = (blocks * multiplier).clamp(-448, 448).to(torch.float8_e4m3fn).float()
            value = (value * multiplier.reciprocal()).reshape(weight.shape).to(weight.dtype)
        return _IdentitySTE.apply(weight, value)


class _IdentitySTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx: FunctionCtx, original: torch.Tensor, quantized: torch.Tensor) -> torch.Tensor:
        """Use quantized values without cancellation in the STE expression.

        Args:
            ctx: Autograd context; no tensors need to be saved.
            original: Tensor of arbitrary shape [...], the gradient recipient.
            quantized: Detached tensor of the same shape/dtype as original.
        Returns:
            Tensor [...] aliasing quantized, with identity gradient to original.
        """
        return quantized

    @staticmethod
    def backward(ctx: FunctionCtx, grad: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Pass the gradient to the original operand only.

        Args:
            ctx: Autograd context.
            grad: Output gradient of shape [...], matching the forward operands.
        Returns:
            Original-operand gradient [...] and no quantized-operand gradient.
        """
        return grad, None


class _QDQLinear(nn.Linear):
    def __init__(
        self, in_features: int, out_features: int, *, bias: bool, dtype: torch.dtype, qdq: FP8QDQConfig
    ) -> None:
        super().__init__(in_features, out_features, bias=bias, dtype=dtype)
        self.qdq = qdq

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply fake quantization at a linear's operand boundaries.

        Args:
            input: Tensor [..., in_features], with arbitrary leading dimensions.
        Returns:
            Tensor [..., out_features] using the ordinary linear compute dtype.
        """
        return F.linear(self.qdq.activation(input), self.qdq.weight(self.weight), self.bias)
