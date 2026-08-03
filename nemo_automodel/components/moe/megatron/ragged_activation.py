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

import torch

from nemo_automodel.shared.import_utils import safe_import

_HAS_TRITON, triton = safe_import("triton")
_HAS_TRITON_LANGUAGE, tl = safe_import("triton.language")
_HAS_TRITON = _HAS_TRITON and _HAS_TRITON_LANGUAGE

_BLOCK_SIZE = 256
_NUM_PROGRAMS = 256
_MAX_INTERMEDIATE_SIZE = 16384


if _HAS_TRITON:

    @triton.jit
    def _ragged_weighted_swiglu_forward_kernel(
        input_ptr,
        weight_ptr,
        expert_offsets_ptr,
        output_ptr,
        intermediate_size: tl.constexpr,
        num_experts: tl.constexpr,
        num_programs: tl.constexpr,
        block_size: tl.constexpr,
    ):
        """Apply weighted SwiGLU only to rows selected by device expert offsets."""
        program = tl.program_id(0)
        valid_rows = tl.load(expert_offsets_ptr + num_experts - 1)
        columns = tl.arange(0, block_size)
        column_mask = columns < intermediate_size
        row = program
        while row < valid_rows:
            gate = tl.load(
                input_ptr + row * intermediate_size * 2 + columns,
                mask=column_mask,
                other=0.0,
            ).to(tl.float32)
            up = tl.load(
                input_ptr + row * intermediate_size * 2 + intermediate_size + columns,
                mask=column_mask,
                other=0.0,
            ).to(tl.float32)
            weight = tl.load(weight_ptr + row).to(tl.float32)
            output = gate * tl.sigmoid(gate) * up * weight
            tl.store(output_ptr + row * intermediate_size + columns, output, mask=column_mask)
            row += num_programs

    @triton.jit
    def _ragged_weighted_swiglu_backward_kernel(
        grad_output_ptr,
        input_ptr,
        weight_ptr,
        expert_offsets_ptr,
        grad_input_ptr,
        grad_weight_ptr,
        intermediate_size: tl.constexpr,
        num_experts: tl.constexpr,
        num_programs: tl.constexpr,
        block_size: tl.constexpr,
    ):
        """Differentiate weighted SwiGLU only over device-selected logical rows."""
        program = tl.program_id(0)
        valid_rows = tl.load(expert_offsets_ptr + num_experts - 1)
        columns = tl.arange(0, block_size)
        column_mask = columns < intermediate_size
        row = program
        while row < valid_rows:
            gate = tl.load(
                input_ptr + row * intermediate_size * 2 + columns,
                mask=column_mask,
                other=0.0,
            ).to(tl.float32)
            up = tl.load(
                input_ptr + row * intermediate_size * 2 + intermediate_size + columns,
                mask=column_mask,
                other=0.0,
            ).to(tl.float32)
            grad_output = tl.load(
                grad_output_ptr + row * intermediate_size + columns,
                mask=column_mask,
                other=0.0,
            ).to(tl.float32)
            weight = tl.load(weight_ptr + row).to(tl.float32)

            sigmoid = tl.sigmoid(gate)
            silu = gate * sigmoid
            scaled_grad = grad_output * weight
            grad_gate = scaled_grad * up * sigmoid * (1.0 + gate * (1.0 - sigmoid))
            grad_up = scaled_grad * silu
            tl.store(
                grad_input_ptr + row * intermediate_size * 2 + columns,
                grad_gate,
                mask=column_mask,
            )
            tl.store(
                grad_input_ptr + row * intermediate_size * 2 + intermediate_size + columns,
                grad_up,
                mask=column_mask,
            )

            activation = (silu * up).to(input_ptr.dtype.element_ty).to(tl.float32)
            grad_weight = tl.sum(tl.where(column_mask, activation * grad_output, 0.0))
            tl.store(grad_weight_ptr + row, grad_weight)
            row += num_programs


class _RaggedWeightedSwiGLU(torch.autograd.Function):
    """Weighted SwiGLU whose logical row count remains GPU-resident."""

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weights: torch.Tensor,
        expert_offsets: torch.Tensor,
    ) -> torch.Tensor:
        """Apply weighted SwiGLU to the logical prefix of max-sized storage.

        Args:
            input: Contiguous CUDA tensor of shape [capacity, 2 * intermediate].
            weights: Contiguous CUDA tensor of shape [capacity, 1].
            expert_offsets: Contiguous CUDA tensor of shape [experts] whose final
                element is the device-resident logical row count.

        Returns:
            Tensor of shape [capacity, intermediate]. Rows at or after the logical
            row count are unspecified and must not be consumed.
        """
        intermediate_size = input.size(1) // 2
        output = torch.empty(
            (input.size(0), intermediate_size),
            dtype=input.dtype,
            device=input.device,
        )
        num_programs = min(_NUM_PROGRAMS, max(1, input.size(0)))
        block_size = triton.next_power_of_2(intermediate_size)
        _ragged_weighted_swiglu_forward_kernel[(num_programs,)](
            input,
            weights,
            expert_offsets,
            output,
            intermediate_size=intermediate_size,
            num_experts=expert_offsets.numel(),
            num_programs=num_programs,
            block_size=block_size,
            num_warps=8,
        )
        ctx.save_for_backward(input, weights, expert_offsets)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        """Differentiate the logical rows of a ragged weighted SwiGLU result.

        Args:
            grad_output: Contiguous CUDA tensor of shape [capacity, intermediate].

        Returns:
            Tuple containing input gradients of shape [capacity, 2 * intermediate],
            weight gradients of shape [capacity, 1], and no offset gradient. Rows
            at or after the logical row count are zero or unspecified as documented
            by the downstream grouped-GEMM offset contract.
        """
        input, weights, expert_offsets = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        intermediate_size = grad_output.size(1)
        grad_input = torch.empty_like(input)
        grad_weights = torch.zeros_like(weights)
        num_programs = min(_NUM_PROGRAMS, max(1, input.size(0)))
        block_size = triton.next_power_of_2(intermediate_size)
        _ragged_weighted_swiglu_backward_kernel[(num_programs,)](
            grad_output,
            input,
            weights,
            expert_offsets,
            grad_input,
            grad_weights,
            intermediate_size=intermediate_size,
            num_experts=expert_offsets.numel(),
            num_programs=num_programs,
            block_size=block_size,
            num_warps=8,
        )
        return grad_input, grad_weights, None


def _can_use_ragged_weighted_swiglu(
    input: torch.Tensor,
    weights: torch.Tensor,
    expert_offsets: torch.Tensor,
) -> bool:
    """Return whether tensors satisfy the ragged SwiGLU kernel contract.

    Args:
        input: Tensor of shape [capacity, 2 * intermediate].
        weights: Tensor of shape [capacity, 1].
        expert_offsets: Tensor of shape [experts].

    Returns:
        True when Triton is available and all tensors meet the CUDA, layout, and
        supported-intermediate-size requirements.
    """
    if input.ndim != 2:
        return False
    intermediate_size = input.size(1) // 2
    return (
        _HAS_TRITON
        and input.is_cuda
        and weights.is_cuda
        and expert_offsets.is_cuda
        and input.is_contiguous()
        and weights.is_contiguous()
        and expert_offsets.is_contiguous()
        and input.size(1) % 2 == 0
        and weights.shape == (input.size(0), 1)
        and expert_offsets.ndim == 1
        and expert_offsets.numel() > 0
        and intermediate_size <= _MAX_INTERMEDIATE_SIZE
    )


def _ragged_weighted_swiglu(
    input: torch.Tensor,
    weights: torch.Tensor,
    expert_offsets: torch.Tensor,
) -> torch.Tensor:
    """Apply weighted SwiGLU using a GPU-resident logical row count.

    Args:
        input: Contiguous CUDA tensor of shape [capacity, 2 * intermediate].
        weights: Contiguous CUDA tensor of shape [capacity, 1].
        expert_offsets: Contiguous CUDA tensor of shape [experts].

    Returns:
        Tensor of shape [capacity, intermediate]. Rows at or after the final
        device offset are unspecified and ignored by grouped GEMM.
    """
    if not _can_use_ragged_weighted_swiglu(input, weights, expert_offsets):
        raise ValueError("ragged weighted SwiGLU requires supported contiguous CUDA tensors")
    return _RaggedWeightedSwiGLU.apply(input, weights, expert_offsets)
