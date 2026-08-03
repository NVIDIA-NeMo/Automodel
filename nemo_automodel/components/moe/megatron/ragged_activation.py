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

RAGGED_SWIGLU = 0
RAGGED_GEGLU = 1
RAGGED_QUICK_GEGLU = 2
RAGGED_SWIGLU_OAI = 3
RAGGED_SWIGLU_CLAMPED = 4
RAGGED_RELU2 = 5

_GATED_ACTIVATIONS = (
    RAGGED_SWIGLU,
    RAGGED_GEGLU,
    RAGGED_QUICK_GEGLU,
    RAGGED_SWIGLU_OAI,
    RAGGED_SWIGLU_CLAMPED,
)


if _HAS_TRITON:
    _TRITON_RAGGED_GEGLU = tl.constexpr(RAGGED_GEGLU)
    _TRITON_RAGGED_QUICK_GEGLU = tl.constexpr(RAGGED_QUICK_GEGLU)
    _TRITON_RAGGED_RELU2 = tl.constexpr(RAGGED_RELU2)

    @triton.jit
    def _ragged_weighted_activation_forward_kernel(
        input_ptr,
        weight_ptr,
        expert_offsets_ptr,
        output_ptr,
        intermediate_size: tl.constexpr,
        num_experts: tl.constexpr,
        num_programs: tl.constexpr,
        block_size: tl.constexpr,
        activation_kind: tl.constexpr,
        alpha: tl.constexpr,
        limit: tl.constexpr,
        linear_offset: tl.constexpr,
        clamp_enabled: tl.constexpr,
    ):
        """Apply a weighted activation only to rows selected by device expert offsets."""
        program = tl.program_id(0)
        valid_rows = tl.load(expert_offsets_ptr + num_experts - 1)
        columns = tl.arange(0, block_size)
        column_mask = columns < intermediate_size
        row = program
        while row < valid_rows:
            if activation_kind == _TRITON_RAGGED_RELU2:
                value = tl.load(
                    input_ptr + row * intermediate_size + columns,
                    mask=column_mask,
                    other=0.0,
                ).to(tl.float32)
                positive = tl.maximum(value, 0.0)
                activation = positive * positive
            else:
                if activation_kind == _TRITON_RAGGED_QUICK_GEGLU:
                    gate_offsets = row * intermediate_size * 2 + columns * 2
                    up_offsets = gate_offsets + 1
                else:
                    gate_offsets = row * intermediate_size * 2 + columns
                    up_offsets = gate_offsets + intermediate_size

                gate = tl.load(input_ptr + gate_offsets, mask=column_mask, other=0.0).to(tl.float32)
                up = tl.load(input_ptr + up_offsets, mask=column_mask, other=0.0).to(tl.float32)
                if clamp_enabled:
                    gate = tl.minimum(gate, limit)
                    up = tl.maximum(tl.minimum(up, limit), -limit)

                if activation_kind == _TRITON_RAGGED_GEGLU:
                    inner = 0.7978845608028654 * (gate + 0.044715 * gate * gate * gate)
                    gated = 0.5 * gate * (1.0 + tl.extra.libdevice.tanh(inner))
                else:
                    gated = gate * tl.sigmoid(alpha * gate)
                activation = gated * (up + linear_offset)

            weight = tl.load(weight_ptr + row).to(tl.float32)
            output = activation * weight
            tl.store(output_ptr + row * intermediate_size + columns, output, mask=column_mask)
            row += num_programs

    @triton.jit
    def _ragged_weighted_activation_backward_kernel(
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
        activation_kind: tl.constexpr,
        alpha: tl.constexpr,
        limit: tl.constexpr,
        linear_offset: tl.constexpr,
        clamp_enabled: tl.constexpr,
    ):
        """Differentiate a weighted activation over device-selected logical rows."""
        program = tl.program_id(0)
        valid_rows = tl.load(expert_offsets_ptr + num_experts - 1)
        columns = tl.arange(0, block_size)
        column_mask = columns < intermediate_size
        row = program
        while row < valid_rows:
            grad_output = tl.load(
                grad_output_ptr + row * intermediate_size + columns,
                mask=column_mask,
                other=0.0,
            ).to(tl.float32)
            weight = tl.load(weight_ptr + row).to(tl.float32)

            if activation_kind == _TRITON_RAGGED_RELU2:
                value = tl.load(
                    input_ptr + row * intermediate_size + columns,
                    mask=column_mask,
                    other=0.0,
                ).to(tl.float32)
                positive = tl.maximum(value, 0.0)
                activation = positive * positive
                grad_value = grad_output * weight * 2.0 * positive * (value > 0.0)
                tl.store(
                    grad_input_ptr + row * intermediate_size + columns,
                    grad_value,
                    mask=column_mask,
                )
            else:
                if activation_kind == _TRITON_RAGGED_QUICK_GEGLU:
                    gate_offsets = row * intermediate_size * 2 + columns * 2
                    up_offsets = gate_offsets + 1
                else:
                    gate_offsets = row * intermediate_size * 2 + columns
                    up_offsets = gate_offsets + intermediate_size

                original_gate = tl.load(input_ptr + gate_offsets, mask=column_mask, other=0.0).to(tl.float32)
                original_up = tl.load(input_ptr + up_offsets, mask=column_mask, other=0.0).to(tl.float32)
                gate = original_gate
                up = original_up
                if clamp_enabled:
                    gate = tl.minimum(gate, limit)
                    up = tl.maximum(tl.minimum(up, limit), -limit)

                if activation_kind == _TRITON_RAGGED_GEGLU:
                    inner = 0.7978845608028654 * (gate + 0.044715 * gate * gate * gate)
                    tanh_inner = tl.extra.libdevice.tanh(inner)
                    gated = 0.5 * gate * (1.0 + tanh_inner)
                    gated_grad = 0.5 * (1.0 + tanh_inner) + 0.5 * gate * (1.0 - tanh_inner * tanh_inner) * (
                        0.7978845608028654 * (1.0 + 3.0 * 0.044715 * gate * gate)
                    )
                else:
                    sigmoid = tl.sigmoid(alpha * gate)
                    gated = gate * sigmoid
                    gated_grad = sigmoid * (1.0 + alpha * gate * (1.0 - sigmoid))

                scaled_grad = grad_output * weight
                grad_gate = scaled_grad * (up + linear_offset) * gated_grad
                grad_up = scaled_grad * gated
                if clamp_enabled:
                    grad_gate = tl.where(original_gate <= limit, grad_gate, 0.0)
                    grad_up = tl.where((original_up >= -limit) & (original_up <= limit), grad_up, 0.0)
                tl.store(grad_input_ptr + gate_offsets, grad_gate, mask=column_mask)
                tl.store(grad_input_ptr + up_offsets, grad_up, mask=column_mask)
                activation = gated * (up + linear_offset)

            activation = activation.to(input_ptr.dtype.element_ty).to(tl.float32)
            grad_weight = tl.sum(tl.where(column_mask, activation * grad_output, 0.0))
            tl.store(grad_weight_ptr + row, grad_weight)
            row += num_programs


class _RaggedWeightedActivation(torch.autograd.Function):
    """Weighted expert activation whose logical row count remains GPU-resident."""

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weights: torch.Tensor,
        expert_offsets: torch.Tensor,
        activation_kind: int,
        alpha: float,
        limit: float,
        linear_offset: float,
    ) -> torch.Tensor:
        """Apply an expert activation to the logical prefix of max-sized storage.

        Args:
            input: Contiguous CUDA tensor of shape [capacity, projection].
            weights: Contiguous CUDA tensor of shape [capacity, 1].
            expert_offsets: Contiguous CUDA tensor of shape [experts] whose final
                element is the device-resident logical row count.
            activation_kind: Integer identifying the activation formula and layout.
            alpha: Sigmoid scale for Quick-GEGLU and SwiGLU-OAI.
            limit: Clamp magnitude for activation variants that require it.
            linear_offset: Offset added to the up projection before multiplication.

        Returns:
            Tensor of shape [capacity, intermediate]. Rows at or after the logical
            row count are unspecified and must not be consumed.
        """
        gated = activation_kind in _GATED_ACTIVATIONS
        intermediate_size = input.size(1) // 2 if gated else input.size(1)
        output = torch.empty(
            (input.size(0), intermediate_size),
            dtype=input.dtype,
            device=input.device,
        )
        num_programs = min(_NUM_PROGRAMS, max(1, input.size(0)))
        block_size = triton.next_power_of_2(intermediate_size)
        clamp_enabled = activation_kind in (RAGGED_QUICK_GEGLU, RAGGED_SWIGLU_CLAMPED) or (
            activation_kind == RAGGED_SWIGLU_OAI and limit > 0.0
        )
        _ragged_weighted_activation_forward_kernel[(num_programs,)](
            input,
            weights,
            expert_offsets,
            output,
            intermediate_size=intermediate_size,
            num_experts=expert_offsets.numel(),
            num_programs=num_programs,
            block_size=block_size,
            activation_kind=activation_kind,
            alpha=alpha,
            limit=limit,
            linear_offset=linear_offset,
            clamp_enabled=clamp_enabled,
            num_warps=8,
        )
        ctx.save_for_backward(input, weights, expert_offsets)
        ctx.activation_kind = activation_kind
        ctx.alpha = alpha
        ctx.limit = limit
        ctx.linear_offset = linear_offset
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None, None, None, None, None]:
        """Differentiate the logical rows of a ragged weighted activation.

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
        clamp_enabled = ctx.activation_kind in (RAGGED_QUICK_GEGLU, RAGGED_SWIGLU_CLAMPED) or (
            ctx.activation_kind == RAGGED_SWIGLU_OAI and ctx.limit > 0.0
        )
        _ragged_weighted_activation_backward_kernel[(num_programs,)](
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
            activation_kind=ctx.activation_kind,
            alpha=ctx.alpha,
            limit=ctx.limit,
            linear_offset=ctx.linear_offset,
            clamp_enabled=clamp_enabled,
            num_warps=8,
        )
        return grad_input, grad_weights, None, None, None, None, None


def _can_use_ragged_weighted_activation(
    input: torch.Tensor,
    weights: torch.Tensor,
    expert_offsets: torch.Tensor,
    activation_kind: int,
) -> bool:
    """Return whether tensors satisfy the ragged activation kernel contract.

    Args:
        input: Tensor of shape [capacity, 2 * intermediate].
        weights: Tensor of shape [capacity, 1].
        expert_offsets: Tensor of shape [experts].
        activation_kind: Integer identifying the activation formula and layout.

    Returns:
        True when Triton is available and all tensors meet the CUDA, layout, and
        supported-intermediate-size requirements.
    """
    if input.ndim != 2:
        return False
    if activation_kind not in (*_GATED_ACTIVATIONS, RAGGED_RELU2):
        return False
    gated = activation_kind in _GATED_ACTIVATIONS
    intermediate_size = input.size(1) // 2 if gated else input.size(1)
    return (
        _HAS_TRITON
        and input.is_cuda
        and weights.is_cuda
        and expert_offsets.is_cuda
        and input.is_contiguous()
        and weights.is_contiguous()
        and expert_offsets.is_contiguous()
        and (not gated or input.size(1) % 2 == 0)
        and weights.shape == (input.size(0), 1)
        and expert_offsets.ndim == 1
        and expert_offsets.numel() > 0
        and intermediate_size <= _MAX_INTERMEDIATE_SIZE
    )


def _ragged_weighted_activation(
    input: torch.Tensor,
    weights: torch.Tensor,
    expert_offsets: torch.Tensor,
    activation_kind: int,
    *,
    alpha: float = 1.0,
    limit: float = 0.0,
    linear_offset: float = 0.0,
) -> torch.Tensor:
    """Apply a weighted expert activation using a GPU-resident logical row count.

    Args:
        input: Contiguous CUDA tensor of shape [capacity, 2 * intermediate].
        weights: Contiguous CUDA tensor of shape [capacity, 1].
        expert_offsets: Contiguous CUDA tensor of shape [experts].
        activation_kind: Integer identifying the activation formula and layout.
        alpha: Sigmoid scale for Quick-GEGLU and SwiGLU-OAI.
        limit: Clamp magnitude for activation variants that require it.
        linear_offset: Offset added to the up projection before multiplication.

    Returns:
        Tensor of shape [capacity, intermediate]. Rows at or after the final
        device offset are unspecified and ignored by grouped GEMM.
    """
    if not _can_use_ragged_weighted_activation(input, weights, expert_offsets, activation_kind):
        raise ValueError("ragged weighted activation requires supported contiguous CUDA tensors")
    return _RaggedWeightedActivation.apply(
        input,
        weights,
        expert_offsets,
        activation_kind,
        alpha,
        limit,
        linear_offset,
    )
