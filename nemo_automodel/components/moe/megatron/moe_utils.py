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
