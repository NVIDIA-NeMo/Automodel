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

"""Transformer Engine quantizers for an NVFP4 forward pass with an MXFP8 backward pass.

The forward GEMM of a Transformer Engine ``Linear`` or ``GroupedLinear`` multiplies NVFP4 operands. The two
backward GEMMs multiply MXFP8 operands: the gradient, and the forward operand as the forward pass used it,
dequantized and quantized again along the other axis. Importing this module needs Transformer Engine 2.19 or newer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from transformer_engine.common.recipe import CustomRecipe
from transformer_engine.pytorch.constants import DType
from transformer_engine.pytorch.quantization import QuantizerRole
from transformer_engine.pytorch.quantized_tensor import QuantizedTensor, QuantizedTensorStorage, Quantizer
from transformer_engine.pytorch.tensor.hybrid_tensor import HybridQuantizer
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer

from nemo_automodel.components.quantization.stochastic_rounding import stochastic_round_to_e4m3

# Block-scale bound of 4over6 tensors, the default of Transformer Engine's NVFP4 recipe.
_FOUR_OVER_SIX_E4M3_MAX = 256


class StochasticMXFP8Quantizer(Quantizer):
    """MXFP8 quantizer whose result equals its input in expectation.

    The input is rounded stochastically to the precision of an E4M3 value and then cast by Transformer Engine's
    MXFP8 quantizer, which stores such values exactly (see
    :mod:`nemo_automodel.components.quantization.stochastic_rounding`).
    """

    def __init__(self, *, rowwise: bool = True, columnwise: bool = True) -> None:
        super().__init__(rowwise=rowwise, columnwise=columnwise)
        self._mxfp8 = MXFP8Quantizer(fp8_dtype=DType.kFloat8E4M3, rowwise=rowwise, columnwise=columnwise)

    def _cast(self) -> MXFP8Quantizer:
        """Return the MXFP8 quantizer with this quantizer's current usage and output flags."""
        self._mxfp8.set_usage(rowwise=self.rowwise_usage, columnwise=self.columnwise_usage)
        self._mxfp8.internal = self.internal
        self._mxfp8.optimize_for_gemm = self.optimize_for_gemm
        return self._mxfp8

    def quantize_impl(self, tensor: torch.Tensor) -> QuantizedTensor:
        """Quantize a stochastically rounded copy of ``tensor``.

        Args:
            tensor: Tensor of shape [..., columns]; the leading dimensions are flattened into rows. Rows and
                columns are multiples of 32.

        Returns:
            MXFP8 tensor with the shape of ``tensor``.
        """
        return self._cast().quantize_impl(stochastic_round_to_e4m3(tensor))

    def update_quantized(
        self,
        src: torch.Tensor,
        dst: QuantizedTensor,
        *,
        noop_flag: torch.Tensor | None = None,
    ) -> QuantizedTensor:
        """Quantize a stochastically rounded copy of ``src`` into ``dst``.

        Args:
            src: Tensor of shape [..., columns], with the shape ``dst`` was created for.
            dst: MXFP8 tensor that receives the result in place.
            noop_flag: Optional one-element tensor; a nonzero value skips the cast.

        Returns:
            ``dst``.
        """
        return self._cast().update_quantized(stochastic_round_to_e4m3(src), dst, noop_flag=noop_flag)

    def make_empty(
        self,
        shape: tuple[int, ...],
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
        requires_grad: bool = False,
        pin_memory: bool = False,
    ) -> QuantizedTensor | QuantizedTensorStorage:
        """Allocate an MXFP8 tensor for values of the given shape."""
        return self._cast().make_empty(
            shape, dtype=dtype, device=device, requires_grad=requires_grad, pin_memory=pin_memory
        )

    def copy(self) -> StochasticMXFP8Quantizer:
        """Create a quantizer with the same usage and output flags."""
        quantizer = StochasticMXFP8Quantizer(rowwise=self.rowwise_usage, columnwise=self.columnwise_usage)
        quantizer.internal = self.internal
        quantizer.optimize_for_gemm = self.optimize_for_gemm
        return quantizer

    def is_quantizable(self, inp: torch.Tensor) -> bool:
        """Whether ``inp`` has rows and columns in multiples of the MXFP8 block size."""
        return self._mxfp8.is_quantizable(inp)

    def is_requantization_safe(self) -> bool:
        """Quantizing the same tensor twice gives different results."""
        return False

    def _get_compatible_recipe(self) -> type[CustomRecipe]:
        return CustomRecipe


@dataclass(frozen=True)
class NVFP4ForwardMXFP8BackwardFactory:
    """Quantizer factory of a ``CustomRecipe``: NVFP4 forward GEMMs and MXFP8 backward GEMMs.

    For the input and the weight of a ``Linear`` or ``GroupedLinear`` the forward GEMM reads an NVFP4 operand,
    and the backward GEMM reads that operand dequantized and quantized to MXFP8 along the other axis. Every other
    tensor, the gradient included, is quantized to MXFP8. Instances with equal fields compare equal, so
    Transformer Engine modules keep their quantizers when the recipe is rebuilt.
    """

    weight_blocks_2d: bool
    """Share one block scale per 16x16 weight block instead of per 16 values of a row."""
    row_scaled_activation: bool
    """Scale forward activations per row (one FP32 scale per token) instead of per tensor."""
    four_over_six: Literal["none", "weights", "activations", "all"]
    """Forward operands whose blocks map their maximum to 4 or to 6, whichever gives the smaller error."""
    stochastic_rounding: bool
    """Round the backward copies of the forward operands stochastically instead of to nearest."""

    def __call__(self, role: QuantizerRole | None) -> Quantizer:
        """Build the quantizer for one tensor of a Transformer Engine module; ``None`` is a tensor without a role."""
        if role is not None and role.module_type in ("linear", "grouped_linear"):
            if role.tensor_type == "input":
                return self._forward_operand(
                    row_scaled=self.row_scaled_activation,
                    blocks_2d=False,
                    four_over_six=self.four_over_six in ("activations", "all"),
                )
            if role.tensor_type == "weight":
                return self._forward_operand(
                    row_scaled=False,
                    blocks_2d=self.weight_blocks_2d,
                    four_over_six=self.four_over_six in ("weights", "all"),
                )
        return MXFP8Quantizer(fp8_dtype=DType.kFloat8E4M3)

    def _forward_operand(self, *, row_scaled: bool, blocks_2d: bool, four_over_six: bool) -> HybridQuantizer:
        nvfp4 = NVFP4Quantizer(
            fp4_dtype=DType.kFloat4E2M1,
            with_rht=False,
            with_post_rht_amax=False,
            with_2d_quantization=blocks_2d,
            stochastic_rounding=False,
            row_scaled_nvfp4=row_scaled,
            nvfp4_use_4over6=four_over_six,
            nvfp4_e4m3_max=_FOUR_OVER_SIX_E4M3_MAX if four_over_six else 448,
        )
        mxfp8 = StochasticMXFP8Quantizer() if self.stochastic_rounding else MXFP8Quantizer(fp8_dtype=DType.kFloat8E4M3)
        return HybridQuantizer(
            rowwise_quantizer=nvfp4,
            columnwise_quantizer=mxfp8,
            columnwise_source="rowwise_dequantized",
        )
