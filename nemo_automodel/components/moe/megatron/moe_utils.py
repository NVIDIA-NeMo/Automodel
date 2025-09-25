# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import torch
import torch.nn.functional as F

try:
    from transformer_engine.pytorch import (
        moe_permute,
        moe_permute_with_probs,
        moe_unpermute,
    )

    HAVE_TE = True
except ImportError:
    HAVE_TE = False


def permute(
    tokens,
    routing_map,
    probs: Optional[torch.Tensor] = None,
    num_out_tokens: Optional[int] = None,
    fused: bool = False,
    drop_and_pad: bool = False,
):
    """Permute the tokens and probs based on the mask.
    Tokens with the same designated expert will be grouped together.
    The shape of mask is [tokens, num_experts], it indicates which experts were selected
    by each token.

    When drop_and_pad=True, in routing_map, the number of non-zeros in each column equals to
    expert capacity. This function exploits this feature to use ops that support cuda graph.

    Args:
        tokens (torch.Tensor): The input token tensor, [num_tokens, hidden].
        routing_map (torch.Tensor): The sparse token to expert mapping, [num_tokens, num_experts].
        probs (torch.Tensor, optional): The probs tensor, [num_tokens, num_experts].
        num_out_tokens (int, optional): The number of output tokens. If None, it's set to
                                        the number of input tokens.
        fused (bool, optional): Whether use the fused permute function.
        drop_and_pad (bool, optional): Whether or not the token dispatcher uses token-drop
                                       and pads the number of tokens to the expert capacity.
                                       If set to true, routing_map has a fixed number of non-zeros
                                       in each column.

    Returns:
        permuted_input (torch.Tensor): The permuted token tensor.
        permuted_probs (torch.Tensor, optional): The permuted probs tensor.
        sorted_indices (torch.Tensor): The tensor of a mapping table for sorted indices used to unpermute the tokens.
    """
    if fused and probs is None:
        if not HAVE_TE or moe_permute is None:
            raise ValueError("moe_permute is not available. Please install TE >= 2.1.0.")
        permuted_input, sorted_indices = moe_permute(tokens, routing_map, num_out_tokens=num_out_tokens)
        return permuted_input, None, sorted_indices

    if fused and probs is not None:
        if not HAVE_TE or moe_permute_with_probs is None:
            raise ValueError("moe_permute_with_probs is not available. Please install TE >= 2.1.0.")
        return moe_permute_with_probs(tokens, probs, routing_map, num_out_tokens=num_out_tokens)

    num_tokens, hidden = tokens.shape
    num_experts = routing_map.shape[1]
    permuted_probs = None
    if drop_and_pad and num_out_tokens is not None:
        capacity = num_out_tokens // num_experts
        assert not routing_map.requires_grad
        # mask [num_tokens, num_experts] -> [num_experts, num_tokens]
        routing_map = routing_map.to(dtype=torch.int8).T.contiguous()
        # use argsort to put indices of all non-zeros in the beginning of list
        # and keep the first `capacity` number of indices
        sorted_indices = routing_map.argsort(dim=-1, descending=True, stable=True)[:, :capacity].contiguous()
        # flatten from [num_experts, capacity] to 1D
        sorted_indices = sorted_indices.view(-1)

        if probs is not None:
            # [num_tokens, num_experts] -> num_experts * num_tokens
            probs_T_1D = probs.T.contiguous().view(-1)
            # get 1D indices of the probs selected by routing_map
            indices_dim0 = torch.arange(num_experts, device=routing_map.device).unsqueeze(-1)
            indices_dim1 = sorted_indices.view(num_experts, capacity)
            indices_1D = (indices_dim0 * num_tokens + indices_dim1).view(-1)
            # get probs from indices
            permuted_probs = probs_T_1D.index_select(0, indices_1D)
    else:
        # mask [num_tokens, num_experts] -> [num_experts, num_tokens]
        routing_map = routing_map.bool().T.contiguous()

        # Create a dense expert-to-token mapping from the sparse token-to-expert mapping
        token_indices = torch.arange(num_tokens, device=routing_map.device).unsqueeze(0).expand(num_experts, -1)
        sorted_indices = token_indices.masked_select(routing_map)

        if probs is not None:
            permuted_probs = probs.T.contiguous().masked_select(routing_map)

    # use the mapping to permute the tokens
    permuted_input = tokens.index_select(0, sorted_indices)

    return permuted_input, permuted_probs, sorted_indices


def unpermute(
    permuted_tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    restore_shape: torch.Size,
    probs: torch.Tensor = None,
    routing_map: torch.Tensor = None,
    fused: bool = False,
    drop_and_pad: bool = False,
):
    """
    Restore the original order of tokens after permutation. If probs are provided, it
    will also apply them to the tokens before restoring the order.

    When drop_and_pad=True, the tensors will have the following properties:
      - In routing_map, the number of non-zeros in each column equals to expert capacity
      - The size of sorted_indices equals to num_experts * capacity, each split of `capacity`
        contains the indices of tokens routed to an expert.
    This function exploits these features to use ops that support cuda graph.

    Args:
        permuted_tokens (torch.Tensor): The permuted token tensor.
        sorted_indices (torch.Tensor): The indices used to sort the tokens.
        restore_shape (torch.Size): The shape of the unpermuted tensor.
        probs (torch.Tensor, optional): The unpermuted probs tensor,
        routing_map (torch.Tensor, optional): Token to expert mapping, shape
            [num_tokens, num_experts].
        fused (bool, optional): Whether use the fused unpermute function.
        drop_and_pad (bool, optional): Whether or not the token dispatcher uses token-drop
                                       and pads the number of tokens to the expert capacity.

    Returns:
        torch.Tensor: The tokens restored to their original order.
    """
    if fused:
        if not HAVE_TE or moe_unpermute is None:
            raise ValueError("moe_unpermute is not available. Please install TE >= 2.1.0.")
        return moe_unpermute(
            permuted_tokens,
            sorted_indices,
            merging_probs=probs,
            restore_shape=restore_shape,
        )

    _, hidden = restore_shape
    input_dtype = permuted_tokens.dtype

    if probs is not None:
        assert routing_map is not None, "Mask must be provided to permute the probs."
        if drop_and_pad:
            num_experts = routing_map.size(1)
            num_permuted_tokens = sorted_indices.size(0)
            capacity = num_permuted_tokens // num_experts
            num_unpermuted_tokens = probs.size(0)

            # [num_unpermuted_tokens, num_experts] -> num_experts * num_unpermuted_tokens
            probs_T_1D = probs.T.contiguous().view(-1)

            # get 1D indices of the probs selected by routing_map
            indices_dim0 = torch.arange(num_experts, device=routing_map.device).unsqueeze(-1)
            indices_dim1 = sorted_indices.view(num_experts, capacity)
            indices_1D = (indices_dim0 * num_unpermuted_tokens + indices_dim1).view(-1)

            # get probs from indices
            permuted_probs = probs_T_1D.index_select(0, indices_1D)
        else:
            permuted_probs = probs.T.contiguous().masked_select(routing_map.T.contiguous())
        # Here may promote permuted_tokens to higher precision (fp32/fp64) if probs is in
        # higher precision due to moe_router_dtype being enabled. This can lead to
        # additional GPU memory usage. Use --moe-permute-fusion flag to avoid this extra memory
        # allocation.
        permuted_tokens = permuted_tokens * permuted_probs.unsqueeze(-1)

    # Create an output tensor filled with zeros
    output_tokens = torch.zeros(restore_shape, dtype=permuted_tokens.dtype, device=permuted_tokens.device)
    # Scatter add the permuted_input back to the original positions
    output_tokens.scatter_add_(0, sorted_indices.unsqueeze(1).expand(-1, hidden), permuted_tokens)
    return output_tokens.to(dtype=input_dtype)


@torch.compile
def swiglu(y):
    y_1, y_2 = torch.chunk(y, 2, -1)
    return F.silu(y_1) * y_2


@torch.compile
def weighted_swiglu(y, weights):
    dtype = y.dtype
    res = swiglu(y) * weights
    return res.to(dtype)


# gradient of tanh approximation of gelu
# gradient of actual gelu is:
# 0.5 * (1. + torch.erf(x * 0.70710678)) + 0.3989423 * x * torch.exp(-0.5 * x * x)
@torch.compile
def swiglu_back(g, y):
    y_1, y_2 = torch.chunk(y, 2, -1)
    return torch.cat(
        (
            g * torch.sigmoid(y_1) * (1 + y_1 * (1 - torch.sigmoid(y_1))) * y_2,
            g * F.silu(y_1),
        ),
        -1,
    )


@torch.compile
def weighted_swiglu_back(g, y, weights):
    input_dtype = y.dtype
    w_dtype = weights.dtype
    input_grad = swiglu_back(g * weights, y)
    # precison of w may be higher than y and g, so we need to cast g to w_dtype
    weights_grad = swiglu(y) * g.to(w_dtype)
    weights_grad = torch.sum(weights_grad, dim=-1, keepdim=True)
    return input_grad.to(input_dtype), weights_grad.to(w_dtype)


class WeightedSwiGLUFunction(torch.autograd.Function):
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weights, fp8_input_store):
        input_for_backward = input.to(torch.float8_e4m3fn) if fp8_input_store else input
        ctx.save_for_backward(input_for_backward, weights)
        ctx.ori_input_dtype = input.dtype
        ctx.fp8_input_store = fp8_input_store
        return weighted_swiglu(input, weights)

    @staticmethod
    def backward(ctx, grad_output):
        input, weights = ctx.saved_tensors
        input = input.to(ctx.ori_input_dtype) if ctx.fp8_input_store else input
        tmp, wgrad = weighted_swiglu_back(grad_output, input, weights)
        return tmp, wgrad, None


def weighted_bias_swiglu_impl(input, weights, fp8_input_store=False):
    """
    Token-wise-weighted bias swiglu fusion.
    """
    ori_shape = input.shape
    assert len(ori_shape) in [2, 3]
    input = input.view(-1, ori_shape[-1])
    output = WeightedSwiGLUFunction.apply(input, weights, fp8_input_store)

    return output if len(ori_shape) == 2 else output.view(ori_shape[0], ori_shape[1], -1)


@torch.compile
def quick_gelu(y: torch.Tensor, alpha: float = 1.702) -> torch.Tensor:
    """Sigmoid approximation of gelu"""
    return y * torch.sigmoid(alpha * y)


@torch.compile
def quick_geglu(y: torch.Tensor, linear_offset: float = 0.0) -> torch.Tensor:
    """Performs Quick-GELU-based GEGLU activation : quick_gelu(y1) * (y2 + offset).

    Args:
        y: Input tensor split into two halves on the last dimension.
        linear_offset: Optional linear offset added to the second half before gating.

    Returns:
        Tensor after applying the GEGLU activation.
    """
    y_1, y_2 = torch.chunk(y, 2, dim=-1)
    return quick_gelu(y_1) * (y_2 + linear_offset)


@torch.compile
def weighted_quick_geglu(y: torch.Tensor, weights: torch.Tensor, linear_offset: float = 0.0) -> torch.Tensor:
    """Token-wise-weighted Quick-GEGLU activation.

    The weights tensor is expected to have the same first-dimension length as ``y`` and a trailing
    singleton dimension so that it broadcasts over the feature dimension.
    """
    dtype = y.dtype
    res = quick_geglu(y, linear_offset) * weights
    return res.to(dtype)


# gradient of sigmoid approximation of gelu
@torch.compile
def quick_geglu_back(g, y, linear_offset: float = 0.0) -> torch.Tensor:
    y_1, y_2 = torch.chunk(y, 2, -1)
    sigmoid_out = torch.sigmoid(1.702 * y_1)
    dy_1 = g * sigmoid_out * (1 + 1.702 * y_1 * (1 - sigmoid_out)) * (y_2 + linear_offset)
    dy_2 = g * y_1 * sigmoid_out
    return torch.cat((dy_1, dy_2), -1)


@torch.compile
def weighted_quick_geglu_back(g, y, weights, linear_offset: float = 0.0):
    """Backward helper for weighted Quick-GEGLU.
    Returns gradient w.r.t input `y` and `weights`.
    """
    input_dtype = y.dtype
    w_dtype = weights.dtype
    # Gradient w.r.t input uses the chain rule with weighting.
    input_grad = quick_geglu_back(g * weights, y, linear_offset)
    # Gradient w.r.t weights is the activation times upstream grad (cast to weight dtype).
    weights_grad = quick_geglu(y, linear_offset) * g.to(w_dtype)
    # Sum across the feature dimension to keep weights shape `[tokens, 1]`.
    weights_grad = torch.sum(weights_grad, dim=-1, keepdim=True)
    return input_grad.to(input_dtype), weights_grad.to(w_dtype)


# ---------------- Weighted Bias Quick-GEGLU helpers -----------------


@torch.compile
def weighted_bias_quick_geglu(
    y: torch.Tensor, bias: torch.Tensor, weights: torch.Tensor, linear_offset: float = 0.0
) -> torch.Tensor:
    """Token-wise weighted Quick-GEGLU activation with bias.

    Args:
        y: Input tensor before bias addition.
        bias: Bias tensor broadcastable to `y`.
        weights: Weight tensor with shape `[tokens, 1]` broadcasting over feature dim.
        linear_offset: Optional linear offset for the second half before gating.

    Returns:
        Activated tensor with same dtype as `y`.
    """
    dtype = y.dtype
    res = quick_geglu(y + bias, linear_offset) * weights
    return res.to(dtype)


@torch.compile
def weighted_bias_quick_geglu_back(g, y, bias, weights, linear_offset: float = 0.0):
    """Backward helper for weighted Quick-GEGLU with bias.

    Returns gradients w.r.t input `y`, `bias`, and `weights`.
    """
    input_dtype = y.dtype
    w_dtype = weights.dtype

    # Forward input with bias
    x = y + bias

    # Gradient w.r.t input (and thus bias) via chain rule
    input_grad = quick_geglu_back(g * weights, x, linear_offset)

    # Gradient w.r.t weights
    weights_grad = quick_geglu(x, linear_offset) * g.to(w_dtype)
    weights_grad = torch.sum(weights_grad, dim=-1, keepdim=True)

    # bias gradient identical to input gradient
    bias_grad = input_grad

    return input_grad.to(input_dtype), bias_grad.to(input_dtype), weights_grad.to(w_dtype)


class WeightedQuickGeGLUFunction(torch.autograd.Function):
    """Autograd function for token-wise weighted Quick-GEGLU (no bias)."""

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weights: torch.Tensor,
        fp8_input_store: bool,
        linear_offset: torch.Tensor,
    ):
        input_for_backward = input.to(torch.float8_e4m3fn) if fp8_input_store else input
        ctx.save_for_backward(input_for_backward, weights, linear_offset)
        ctx.ori_input_dtype = input.dtype
        ctx.fp8_input_store = fp8_input_store
        return weighted_quick_geglu(input, weights, linear_offset)

    @staticmethod
    def backward(ctx, grad_output):
        input, weights, linear_offset = ctx.saved_tensors
        input = input.to(ctx.ori_input_dtype) if ctx.fp8_input_store else input
        input_grad, wgrad = weighted_quick_geglu_back(grad_output, input, weights, linear_offset)
        return input_grad, wgrad, None, None


class WeightedBiasQuickGeGLUFunction(torch.autograd.Function):
    """Autograd function for token-wise weighted Quick-GEGLU with bias support."""

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        bias: torch.Tensor,
        weights: torch.Tensor,
        fp8_input_store: bool,
        linear_offset: torch.Tensor,
    ):
        # Optionally store the input in FP8 for memory savings.
        input_for_backward = input.to(torch.float8_e4m3fn) if fp8_input_store else input

        # Save tensors for backward.
        ctx.save_for_backward(input_for_backward, bias, weights, linear_offset)
        ctx.ori_input_dtype = input.dtype
        ctx.fp8_input_store = fp8_input_store

        # Compute activation using fused helper that includes bias and weighting.
        return weighted_bias_quick_geglu(input, bias, weights, linear_offset)

    @staticmethod
    def backward(ctx, grad_output):
        input, bias, weights, linear_offset = ctx.saved_tensors

        # Restore original input dtype if it was stored in FP8.
        input = input.to(ctx.ori_input_dtype) if ctx.fp8_input_store else input

        input_grad, bias_grad, weights_grad = weighted_bias_quick_geglu_back(
            grad_output, input, bias, weights, linear_offset
        )

        return input_grad, bias_grad, weights_grad, None, None


def weighted_bias_quick_geglu_impl(
    input, bias, weights, fp8_input_store=False, linear_offset=0.0, clamp_value=None, alpha=1.702
):
    """
    Token-wise-weighted bias quick_geglu fusion.
        input: [num_selected_experts * seq_len, hidden_size * 2]
        bias: None
        weights: [num_selected_experts * seq_len, 1]
        fp8_input_store: bool
        linear_offset: float
        output: [num_selected_experts * seq_len, hidden_size]
    """
    ori_shape = input.shape
    assert len(ori_shape) in [2, 3], f"Input shape must be of length 2 or 3, but got {ori_shape=}"
    if clamp_value is not None:
        x_glu, x_linear = input.chunk(2, -1)
        input = torch.cat(
            (
                x_glu.clamp(min=None, max=clamp_value),
                x_linear.clamp(min=-clamp_value, max=clamp_value),
            ),
            -1,
        )
    input = input.view(-1, ori_shape[-1])
    linear_offset = torch.tensor(linear_offset, dtype=input.dtype, device=input.device)
    if bias is not None:
        output = WeightedBiasQuickGeGLUFunction.apply(input, bias, weights, fp8_input_store, linear_offset)
    else:
        output = WeightedQuickGeGLUFunction.apply(input, weights, fp8_input_store, linear_offset)

    return output if len(ori_shape) == 2 else output.view(ori_shape[0], ori_shape[1], -1)


class MoEAuxLossAutoScaler(torch.autograd.Function):
    """An AutoScaler that triggers the backward pass and scales the grad for auxiliary loss."""

    main_loss_backward_scale: torch.Tensor = None

    @staticmethod
    def forward(ctx, output: torch.Tensor, aux_loss: torch.Tensor):
        """Preserve the aux_loss by storing it in the context to avoid garbage collection.

        Args:
            output (torch.Tensor): The output tensor.
            aux_loss (torch.Tensor): The auxiliary loss tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        ctx.save_for_backward(aux_loss)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Compute and scale the gradient for auxiliary loss..

        Args:
            grad_output (torch.Tensor): The gradient of the output.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The gradient of the output, scaled auxiliary loss
                                               gradient.
        """
        (aux_loss,) = ctx.saved_tensors
        if MoEAuxLossAutoScaler.main_loss_backward_scale is None:
            MoEAuxLossAutoScaler.main_loss_backward_scale = torch.tensor(1.0, device=aux_loss.device)
        aux_loss_backward_scale = MoEAuxLossAutoScaler.main_loss_backward_scale
        scaled_aux_loss_grad = torch.ones_like(aux_loss) * aux_loss_backward_scale
        return grad_output, scaled_aux_loss_grad
