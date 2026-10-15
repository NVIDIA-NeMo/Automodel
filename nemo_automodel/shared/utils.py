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
from typing import Any

import torch


def dtype_from_str(val, default=torch.bfloat16):
    """
    Translate a str val of a dtype into the corresponding torch.dtype
    Args:
        val (str): the dotted path of the dtype (e.g., "torch.bfloat16").

    Returns:
        torch.dtype: the actual dtype (e.g., torch.bfloat16)
    """
    if val is None:
        assert isinstance(default, torch.dtype), default
        return default

    if isinstance(val, torch.dtype):
        return val

    # val must be a recognized dtype string below; anything else (e.g. a
    # Mock attribute auto-created in tests that build configs with
    # unittest.mock.Mock) falls back to the default so a call site like
    # get_dtype(getattr(config, "torch_dtype", None), torch.bfloat16) stays
    # safe instead of crashing deep inside str concatenation.
    if not isinstance(val, str):
        assert isinstance(default, torch.dtype), default
        return default

    lut = {
        "torch.float": torch.float,
        "torch.float32": torch.float,
        "torch.float64": torch.float64,
        "torch.double": torch.float64,
        "torch.complex64": torch.complex,
        "torch.cfloat": torch.complex,
        "torch.float16": torch.float16,
        "torch.half": torch.float16,
        "torch.bfloat16": torch.bfloat16,
        "torch.uint8": torch.uint8,
        "torch.int8": torch.int8,
        "torch.int16": torch.int16,
        "torch.short": torch.short,
        "torch.int32": torch.int32,
        "torch.int": torch.int,
        "torch.int64": torch.int64,
        "torch.long": torch.long,
        "torch.bool": torch.bool,
        "bf16": torch.bfloat16,
    }

    val_lower = val.lower()
    if val_lower in lut:
        return lut[val_lower]
    torch_val = "torch." + val_lower
    if torch_val in lut:
        return lut[torch_val]
    raise KeyError(f"Unknown dtype string: {val}")


class _RowSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, values: torch.Tensor) -> torch.Tensor:
        """Reduce groups of rows, then their partial sums.

        Args:
            ctx: Autograd context; saves only the input shape.
            values: Tensor of shape [..., rows, columns], with arbitrary leading dimensions and strides.

        Returns:
            Tensor of shape [..., columns], with the dtype of ``values.sum(-2)``.
        """
        ctx.input_shape = values.shape
        low_precision = values.dtype in (torch.float16, torch.bfloat16)
        dtype = torch.float32 if low_precision else None
        rows = values.shape[-2]
        full_rows = rows // 256 * 256
        # Splitting one dimension with unflatten is a view, including for strided inputs.
        full = values if full_rows == rows else values[..., :full_rows, :]
        partials = full.unflatten(-2, (-1, 256)).sum(dim=-2, dtype=dtype)
        result = partials.sum(dim=-2)
        if full_rows != rows:
            result = result + values[..., full_rows:, :].sum(dim=-2, dtype=dtype)
        return result.to(values.dtype) if low_precision else result

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor]:
        """Broadcast the upstream gradient, as for a single sum.

        Args:
            ctx: Context containing the original [..., rows, columns] shape.
            grad_output: Tensor of shape [..., columns].

        Returns:
            One gradient of shape [..., rows, columns], a broadcast view of ``grad_output``.
        """
        # Autograd through the sliced tail would otherwise allocate full-size slice gradients.
        return (grad_output.unsqueeze(-2).expand(ctx.input_shape),)


def sum_rows(values: torch.Tensor) -> torch.Tensor:
    """Sum the penultimate axis with small CUDA reduction intermediates.

    Short row groups keep CUDA from allocating a large cross-block staging buffer
    for a strided column reduction. FP16/BF16 partials accumulate in FP32 and are
    cast only after the final sum. Floating-point accumulation order can differ
    from ``sum(-2)``. CPU, short reductions and contiguous row axes use native sum.

    Args:
        values: Tensor of shape [..., rows, columns], with arbitrary leading dimensions and strides.

    Returns:
        Tensor of shape [..., columns], with the dtype of ``values.sum(-2)``.
        The input is not mutated. Backward broadcasts the upstream gradient.
    """
    if not values.is_cuda or values.shape[-2] <= 512 or values.stride(-2) == 1:
        return values.sum(dim=-2)
    return _RowSum.apply(values)
