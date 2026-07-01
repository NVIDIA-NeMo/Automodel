# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

import logging
import math
import os
from functools import cache, partial
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn_f
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

from nemo_automodel.components.moe.state_dict_utils import create_dtensor_from_local

try:
    from grouped_gemm import ops
except ImportError:
    print("grouped_gemm is not available. Please run:pip install git+https://github.com/fanshiqing/grouped_gemm@v1.1.4")

from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.megatron.moe_utils import (
    weighted_bias_geglu_impl,
    weighted_bias_swiglu_impl,
)
from nemo_automodel.components.moe.megatron.token_dispatcher import MoEFlexTokenDispatcher, TokenDispatcherConfig
from nemo_automodel.components.moe.mxfp8 import select_grouped_mm

logger = logging.getLogger(__name__)

# ── EP variable-length collective helpers ──


class _AllGatherConcatVarlenFn(Function):
    """All-gather with variable local lengths and autograd-safe backward.

    Backward uses all-reduce + local narrow instead of reduce-scatter to avoid
    monitoredBarrier deadlocks observed with mixed FSDP/EP backward collective ordering.
    """

    @staticmethod
    def forward(ctx, local_tensor: torch.Tensor, group: dist.ProcessGroup, gathered_lens: list[int], max_len: int):
        local_len = local_tensor.size(0)
        if local_len < max_len:
            pad_shape = (max_len - local_len,) + tuple(local_tensor.shape[1:])
            pad = torch.zeros(pad_shape, dtype=local_tensor.dtype, device=local_tensor.device)
            local_padded = torch.cat([local_tensor, pad], dim=0)
        else:
            local_padded = local_tensor

        world_size = len(gathered_lens)
        gathered = [torch.empty_like(local_padded) for _ in range(world_size)]
        dist.all_gather(gathered, local_padded, group=group)
        gathered = [g[:n] for g, n in zip(gathered, gathered_lens)]

        ctx.group = group
        ctx.gathered_lens = gathered_lens
        ctx.rank = dist.get_rank(group)
        return torch.cat(gathered, dim=0)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_full = grad_output.contiguous()
        start = sum(ctx.gathered_lens[: ctx.rank])
        local_len = ctx.gathered_lens[ctx.rank]
        dist.all_reduce(grad_full, op=dist.ReduceOp.SUM, group=ctx.group)
        grad_local = grad_full.narrow(0, start, local_len).contiguous()
        return grad_local, None, None, None


if TYPE_CHECKING:
    from transformer_engine.pytorch import GroupedLinear

    from nemo_automodel.components.models.common.utils import BackendConfig


def is_gated_activation(activation: str) -> bool:
    """Check if activation requires gating (gate_proj + up_proj).

    Gated activations (SwiGLU, Quick-GEGLU) use both gate_proj and up_proj,
    requiring gate_and_up_projs tensor with shape [n_experts, dim, 2*inter_dim].

    Non-gated activations (ReLU²) only use up_proj, requiring up_projs tensor
    with shape [n_experts, dim, inter_dim] - 50% memory savings.
    """
    return activation in ("swiglu", "swiglu_step", "swigluoai", "quick_geglu", "geglu")


def _permute_tokens_for_grouped_mm(
    indices: torch.Tensor,
    weights: torch.Tensor,
    token_mask: torch.Tensor,
    n_local_experts: int,
    experts_start_idx: int,
):
    """Permute tokens by expert assignment and compute offs for torch._grouped_mm.

    Takes the raw router outputs and produces sorted token IDs, routing weights,
    tokens_per_expert counts, and cumulative offsets ready for grouped GEMM.

    Returns:
        sorted_token_ids: Token indices sorted by expert assignment.
        sorted_weights: Routing weights in the same sorted order.
        tokens_per_expert: Count of tokens per local expert.
        offs: Cumulative token counts (int32) for torch._grouped_mm.
    """
    num_tokens, topk = indices.shape
    experts_end_idx = experts_start_idx + n_local_experts

    # Mask invalid tokens
    indices = indices.masked_fill(~token_mask.unsqueeze(-1), -1)

    # Flatten [num_tokens, topk] -> [num_tokens * topk]
    flat_indices = indices.view(-1)
    flat_weights = weights.float().view(-1)
    token_ids = torch.arange(num_tokens, device=indices.device).unsqueeze(1).expand(-1, topk).reshape(-1)

    # Filter to local experts
    local_mask = (flat_indices >= experts_start_idx) & (flat_indices < experts_end_idx)
    local_expert_ids = flat_indices[local_mask] - experts_start_idx
    local_token_ids = token_ids[local_mask]
    local_weights = flat_weights[local_mask]

    # Sort by expert to group tokens contiguously
    sort_order = local_expert_ids.argsort(stable=True)
    sorted_expert_ids = local_expert_ids[sort_order]
    sorted_token_ids = local_token_ids[sort_order]
    sorted_weights = local_weights[sort_order]

    # Compute tokens_per_expert and offs
    tokens_per_expert = torch.bincount(sorted_expert_ids, minlength=n_local_experts)
    offs = tokens_per_expert.cumsum(dim=0).to(torch.int32)

    return sorted_token_ids, sorted_weights, tokens_per_expert, offs


def _mask_routing_metadata(
    weights: torch.Tensor,
    indices: torch.Tensor,
    token_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask compact top-k or TE's dense HybridEP routing metadata."""
    if indices.dtype == torch.bool:
        valid_tokens = token_mask.unsqueeze(-1)
        return weights * valid_tokens.to(weights.dtype), indices & valid_tokens
    return weights, indices.masked_fill(~token_mask.unsqueeze(-1), -1)


def _apply_bias(value, bias, tokens_per_expert, permuted_probs=None):
    """Apply per-expert bias to grouped GEMM output.

    NOTE: torch._grouped_mm accepts a `bias` kwarg in its schema but raises
    "RuntimeError: Bias not supported yet" as of PyTorch 2.9.0.
    Additionally, down projection bias needs weighting by routing probs
    (bias * permuted_probs) which native bias support wouldn't handle.

    Args:
        value: Output from grouped GEMM, shape [total_tokens, features].
        bias: Per-expert bias, shape [num_experts, features].
        tokens_per_expert: Token counts per expert.
        permuted_probs: If provided, bias is weighted by routing probs (for down projection).
    """
    if bias is None:
        return value
    shape = value.shape
    if permuted_probs is not None:
        output = (
            torch.cat(
                [
                    t + b * p
                    for t, b, p in zip(
                        torch.split(value.view(-1, shape[-1]), tokens_per_expert.tolist()),
                        bias,
                        torch.split(permuted_probs, tokens_per_expert.tolist()),
                    )
                ]
            )
            .view(shape)
            .to(value.dtype)
        )
    else:
        output = (
            torch.cat(
                [
                    t + b
                    for t, b in zip(
                        torch.split(
                            value.view(-1, shape[-1]),
                            tokens_per_expert.tolist()
                            if isinstance(tokens_per_expert, torch.Tensor)
                            else tokens_per_expert,
                        ),
                        bias,
                    )
                ]
            )
            .view(shape)
            .to(value.dtype)
        )
    return output


class GroupedExperts(nn.Module):
    """
    Sparse MoE implementation using all-gather/reduce-scatter primitives.

    Supports two compute backends:
    - Per-expert loop with gather/scatter (default)
    - torch._grouped_mm with argsort-based permutation (backend.experts="torch_mm")

    Attributes:
        n_routed_experts (int): Total number of experts in the model.
        gate_and_up_projs (nn.Parameter): Linear layer for gate+up (gated) or just up (non-gated).
        down_projs (nn.Parameter): Linear layer for hidden-to-output transformation.
    """

    def __init__(self, config: MoEConfig, backend: Optional["BackendConfig"] = None):
        """
        Initializes the GroupedExperts module.

        Args:
            config: MoE configuration containing expert parameters.
            backend: Backend configuration. When backend.experts == "torch_mm",
                uses torch._grouped_mm instead of per-expert loop.
        """
        super().__init__()
        self.config = config
        self.n_routed_experts = config.n_routed_experts
        self.expert_bias = config.expert_bias
        self.is_gated = is_gated_activation(config.expert_activation)
        # "torch_mm_mxfp8" dispatches identically to "torch_mm" but routes the grouped
        # GEMMs through torchao's MXFP8 kernel (see _torch_mm_experts_fwd).
        self.use_torch_mm = backend is not None and backend.experts in ("torch_mm", "torch_mm_mxfp8")
        self.use_mxfp8 = backend is not None and backend.experts == "torch_mm_mxfp8"

        # Allocate projection tensor - size depends on whether activation is gated
        # Gated (SwiGLU, Quick-GEGLU): [n_experts, dim, 2*inter_dim]
        # Non-gated (ReLU²): [n_experts, dim, inter_dim]
        up_proj_dim = config.moe_inter_dim * 2 if self.is_gated else config.moe_inter_dim
        self.gate_and_up_projs = nn.Parameter(
            torch.empty(config.n_routed_experts, config.expert_dim, up_proj_dim, dtype=config.dtype)
        )

        self.down_projs = nn.Parameter(
            torch.empty(config.n_routed_experts, config.moe_inter_dim, config.expert_dim, dtype=config.dtype)
        )

        if self.expert_bias:
            self.gate_up_proj_bias = nn.Parameter(torch.empty(config.n_routed_experts, up_proj_dim, dtype=config.dtype))
            self.down_proj_bias = nn.Parameter(
                torch.empty(config.n_routed_experts, config.expert_dim, dtype=config.dtype)
            )
        else:
            self.gate_up_proj_bias = None
            self.down_proj_bias = None

        self.expert_activation_grouped = get_expert_activation_for_deepep(config)

    def forward(
        self,
        x: torch.Tensor,
        token_mask: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for the grouped experts.

        Args:
            x (torch.Tensor): Input tensor. Shape is [num_tokens, model_dim].
            token_mask (torch.Tensor): Boolean mask indicating valid tokens.
                Shape is [num_tokens].
            weights (torch.Tensor): Routing weights for the selected experts.
                Shape is [num_tokens, num_activated_experts].
            indices (torch.Tensor): Indices of the selected experts.
                Shape is [num_tokens, num_activated_experts].

        Returns:
            torch.Tensor: Output tensor after expert computation.
                Shape is [num_tokens, model_dim]
        """
        assert not isinstance(x, DTensor)
        input_dtype = x.dtype

        # Get the projection tensor for EP mesh extraction
        if isinstance(self.gate_and_up_projs, DTensor):
            ep_mesh = self.gate_and_up_projs.device_mesh
            assert ep_mesh is not None
            assert ep_mesh.ndim == 1, "We only support 1D mesh for MoE"
            ep_size = ep_mesh.size()
            ep_rank = ep_mesh.get_local_rank()
        else:
            ep_mesh = None
            ep_size = 1
            ep_rank = 0

        assert self.n_routed_experts % ep_size == 0, (
            f"Number of experts must be divisible by ep_size (ep_size={ep_size})"
        )

        # Cast expert weights to the activation dtype so that fp32-stored
        # parameters (e.g. under fp32 master weights) still work with kernels
        # (grouped_gemm / torch._grouped_mm) that require matching dtypes with
        # the (typically bf16) activations. When the weights are already in the
        # activation dtype these casts are no-ops.
        compute_dtype = x.dtype
        gate_and_up_projs = (
            self.gate_and_up_projs.to_local() if isinstance(self.gate_and_up_projs, DTensor) else self.gate_and_up_projs
        ).to(compute_dtype)
        down_projs = (self.down_projs.to_local() if isinstance(self.down_projs, DTensor) else self.down_projs).to(
            compute_dtype
        )
        gate_up_proj_bias = (
            (
                self.gate_up_proj_bias.to_local()
                if isinstance(self.gate_up_proj_bias, DTensor)
                else self.gate_up_proj_bias
            ).to(compute_dtype)
            if self.expert_bias
            else None
        )
        down_proj_bias = (
            (self.down_proj_bias.to_local() if isinstance(self.down_proj_bias, DTensor) else self.down_proj_bias).to(
                compute_dtype
            )
            if self.expert_bias
            else None
        )

        # EP variable-length all-gather
        if ep_size > 1:
            ep_group = ep_mesh.get_group()
            local_num_tokens = x.size(0)

            # Exchange per-rank token counts
            local_len_t = torch.tensor([local_num_tokens], device=x.device, dtype=torch.int64)
            gathered_len_t = [torch.zeros_like(local_len_t) for _ in range(ep_size)]
            dist.all_gather(gathered_len_t, local_len_t, group=ep_group)
            gathered_lens = [int(t.item()) for t in gathered_len_t]
            max_len = max(gathered_lens)

            def _all_gather_dim0_var(local_tensor: torch.Tensor, *, differentiable: bool) -> torch.Tensor:
                if differentiable:
                    return _AllGatherConcatVarlenFn.apply(local_tensor, ep_group, gathered_lens, max_len)
                if max_len > local_tensor.size(0):
                    pad_shape = (max_len - local_tensor.size(0),) + tuple(local_tensor.shape[1:])
                    pad = torch.zeros(pad_shape, dtype=local_tensor.dtype, device=local_tensor.device)
                    local_padded = torch.cat([local_tensor, pad], dim=0)
                else:
                    local_padded = local_tensor
                gathered = [torch.empty_like(local_padded) for _ in range(ep_size)]
                dist.all_gather(gathered, local_padded, group=ep_group)
                gathered = [g[:n] for g, n in zip(gathered, gathered_lens)]
                return torch.cat(gathered, dim=0)

            x = _all_gather_dim0_var(x, differentiable=True)
            weights = _all_gather_dim0_var(weights.float(), differentiable=False)
            indices = _all_gather_dim0_var(indices, differentiable=False)
            token_mask = _all_gather_dim0_var(token_mask, differentiable=False)

        n_local_experts = self.n_routed_experts // ep_size
        experts_start_idx = ep_rank * n_local_experts
        experts_end_idx = experts_start_idx + n_local_experts

        if self.use_torch_mm:
            y = self._forward_grouped_mm(
                x,
                token_mask,
                weights,
                indices,
                gate_and_up_projs,
                down_projs,
                gate_up_proj_bias,
                down_proj_bias,
                n_local_experts,
                experts_start_idx,
            )
        else:
            y = self._forward_loop(
                x,
                weights,
                indices,
                token_mask,
                gate_and_up_projs,
                down_projs,
                gate_up_proj_bias,
                down_proj_bias,
                n_local_experts,
                experts_start_idx,
                experts_end_idx,
            )

        # Gradient anchor
        if ep_size > 1:
            # Keep the differentiable all-gather path attached to x without materializing a full-size zero tensor.
            y.add_(x.sum(dtype=torch.float32) * 0.0)

        # Variable-length reduce: all_reduce + narrow to original per-rank token boundaries
        if ep_size > 1:
            y = dist_nn_f.all_reduce(y, op=dist.ReduceOp.SUM, group=ep_group)
            start = sum(gathered_lens[:ep_rank])
            y = y.narrow(0, start, local_num_tokens).contiguous()

        return y.to(input_dtype)

    def _forward_loop(
        self,
        x,
        weights,
        indices,
        token_mask,
        gate_and_up_projs,
        down_projs,
        gate_up_proj_bias,
        down_proj_bias,
        n_local_experts,
        experts_start_idx,
        experts_end_idx,
    ):
        """Per-expert loop forward path using gather/scatter."""
        y = torch.zeros(x.shape, dtype=torch.float32, device=x.device)

        active_local_experts = 0
        for i in range(experts_start_idx, experts_end_idx):
            indices_mask = torch.logical_and(indices == i, token_mask.unsqueeze(-1))
            idx, top = torch.where(indices_mask)

            if idx.numel() == 0:
                continue
            active_local_experts += 1

            local_idx = i - experts_start_idx
            down_proj = down_projs[local_idx]
            expert_down_proj_bias = down_proj_bias[local_idx] if down_proj_bias is not None else None

            idx_b = idx[:, None].expand(-1, x.size(1))
            x_idx = x.gather(dim=0, index=idx_b)

            gate_and_up_proj = gate_and_up_projs[local_idx]
            expert_gate_up_proj_bias = gate_up_proj_bias[local_idx] if gate_up_proj_bias is not None else None

            # Up projection (separate from activation, matching DeepEP pattern)
            gate_and_up_out = x_idx @ gate_and_up_proj
            if expert_gate_up_proj_bias is not None:
                gate_and_up_out = gate_and_up_out + expert_gate_up_proj_bias

            # Weighted activation (routing weight applied BETWEEN up and down projections)
            # Uses WeightedSwiGLUFunction with float32 backward precision
            w = weights[idx, top, None]
            activated = self.expert_activation_grouped(gate_and_up_out, w)

            # Down projection
            expert_out = activated @ down_proj
            if expert_down_proj_bias is not None:
                expert_out = expert_out + expert_down_proj_bias * w

            y.scatter_add_(dim=0, index=idx_b, src=expert_out.float())

        # Dummy computation for gradient flow when no tokens routed locally
        if active_local_experts == 0:
            dummy_x = torch.zeros_like(x[0]).unsqueeze(0)
            gate_and_up_out = dummy_x @ gate_and_up_projs[0]
            activated = self.expert_activation_grouped(gate_and_up_out, weights[0, 0, None].unsqueeze(0))
            expert_out = activated @ down_projs[0]
            y[0] += expert_out[0]

        return y

    def _forward_grouped_mm(
        self,
        x,
        token_mask,
        weights,
        indices,
        gate_and_up_projs,
        down_projs,
        gate_up_proj_bias,
        down_proj_bias,
        n_local_experts,
        experts_start_idx,
    ):
        """Grouped GEMM forward path using torch._grouped_mm."""
        sorted_token_ids, sorted_weights, tokens_per_expert, offs = _permute_tokens_for_grouped_mm(
            indices,
            weights,
            token_mask,
            n_local_experts,
            experts_start_idx,
        )

        y = torch.zeros(x.shape, dtype=torch.float32, device=x.device)

        if tokens_per_expert.sum() > 0:
            permuted_x = x[sorted_token_ids]
            permuted_probs = sorted_weights.unsqueeze(-1)

            if self.expert_bias:
                # torch._grouped_mm does not support bias yet (raises
                # "RuntimeError: Bias not supported yet" as of PyTorch 2.10).
                # Apply bias manually after each grouped GEMM via _apply_bias.
                # select_grouped_mm routes through torchao MXFP8 (with the contiguous-
                # operand relayout) when use_mxfp8, else plain torch._grouped_mm.
                # MXFP8: the grouped_mm wrapper clamps its quant input (see
                # select_grouped_mm) so a bias-shifted value can't overflow the e8m0
                # block scale -> nan. The bias-add stays a bf16 separate add (torchao
                # v0.17.0 has no bias arg). bf16 path byte-identical.
                grouped_mm = select_grouped_mm(self.use_mxfp8)
                output1 = grouped_mm(permuted_x, gate_and_up_projs, offs)
                output1 = _apply_bias(output1, gate_up_proj_bias, tokens_per_expert)
                output1 = self.expert_activation_grouped(output1, permuted_probs)
                output2 = grouped_mm(output1, down_projs, offs)
                output2 = _apply_bias(output2, down_proj_bias, tokens_per_expert, permuted_probs)
            else:
                output2 = _torch_mm_experts_fwd(
                    permuted_x,
                    gate_and_up_projs,
                    down_projs,
                    tokens_per_expert,
                    permuted_probs,
                    self.expert_activation_grouped,
                    use_mxfp8=self.use_mxfp8,
                )

            scatter_ids = sorted_token_ids.unsqueeze(1).expand_as(output2)
            y.scatter_add_(0, scatter_ids, output2.float())
        else:
            # Dummy computation for gradient flow
            output1 = torch.matmul(x[0] * 0, gate_and_up_projs[0])
            output1_ = self.expert_activation_grouped(output1, weights[0, 0, None].unsqueeze(0))
            output2 = torch.matmul(output1_, down_projs[0])
            y[0] += output2[0]

        return y

    def init_weights(self, buffer_device: torch.device, init_std: float = 0.02) -> None:
        self.apply(partial(_init_weights, buffer_device=buffer_device, init_std=init_std))


@torch.compile(fullgraph=True, options={"max_autotune": True})
def quick_geglu_deepep(
    x,
    permuted_probs,
    alpha: float = 1.702,
    limit: float = 7.0,
    linear_offset: float = 1.0,
):
    """Apply DeepEP Quick-GEGLU activation and routing probabilities."""

    gate_out, up_out = x[..., ::2], x[..., 1::2]
    # Clamp the input values
    gate_out = gate_out.clamp(min=None, max=limit)
    up_out = up_out.clamp(min=-limit, max=limit)
    out_glu = gate_out * torch.sigmoid(alpha * gate_out)
    # Note we add an extra bias of 1 to the linear layer
    inter = out_glu * (up_out + linear_offset)
    return (inter * permuted_probs).to(x.dtype)


@torch.compile(fullgraph=True, options={"max_autotune": True})
def swiglu_oai_deepep(x, permuted_probs, alpha: float = 1.702, limit: float = 7.0):
    """SwiGLU-OAI (GPT-OSS / MiniMax-M3) activation for grouped experts.

    Computes ``gate * sigmoid(alpha * gate) * (up + 1)`` in fp32 with gate
    clamped ``max=limit`` and up clamped ``+/-limit`` (when ``limit > 0``).

    Unlike :func:`quick_geglu_deepep` (which expects an *interleaved* gate/up
    layout, ``x[..., ::2]`` / ``x[..., 1::2]``), this reads the *concatenated*
    ``[gate | up]`` layout produced by ``MoESplitExpertsStateDictMixin``
    (``torch.cat([gate_t, up_t], dim=-1)``), matching sglang's
    ``swiglu_no_interleaved_with_alpha_and_limit``.
    """
    gate, up = torch.chunk(x, 2, dim=-1)
    gate = gate.float()
    up = up.float()
    if limit > 0.0:
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
    inter = gate * torch.sigmoid(alpha * gate) * (up + 1.0)
    return (inter * permuted_probs).to(x.dtype)


@torch.compile(fullgraph=True, options={"max_autotune": True})
def relu2_deepep(x, permuted_probs):
    """ReLU² activation for DeepEP: relu(x)^2

    For DeepEP with ReLU², x is the output of the up projection (already computed).
    x already has shape [..., inter_dim] from efficient up_proj.
    """
    inter = F.relu(x).pow(2)
    return (inter * permuted_probs).to(x.dtype)


@torch.compile(fullgraph=True, options={"max_autotune": True})
def swiglu_clamped_deepep(x, permuted_probs, limit: float):
    """Clamped SwiGLU (DeepSeek V4 style) for DeepEP.

    Gate is clamped at ``max=limit`` and up at ``(-limit, +limit)`` in FP32
    before ``silu(gate) * up``; the result is multiplied by the permuted
    routing probs and cast back.  Matches the official V4 Expert.forward::

        gate = self.w1(x).float()
        up   = self.w3(x).float()
        if self.swiglu_limit > 0:
            up   = torch.clamp(up,   min=-swiglu_limit, max=swiglu_limit)
            gate = torch.clamp(gate,                     max=swiglu_limit)
        y = F.silu(gate) * up

    ``x`` has shape ``[..., 2 * inter_dim]`` with gate in the first half
    and up in the second half (same layout as ``weighted_bias_swiglu_impl``).
    """
    gate, up = torch.chunk(x, 2, dim=-1)
    gate = gate.float().clamp(max=limit)
    up = up.float().clamp(min=-limit, max=limit)
    inter = F.silu(gate) * up
    return (inter * permuted_probs).to(x.dtype)


@torch.compile(fullgraph=True, options={"max_autotune": True})
def swiglu_step_deepep(x, permuted_probs, limit: float):
    """Clamped SwiGLU with Step3.5/3.7's post-SiLU gate semantics.

    Step first evaluates ``silu(gate)`` in the projection dtype, then clamps
    that activated value at ``max=limit``. The up projection is clamped to
    ``(-limit, +limit)`` before the product. This is intentionally distinct
    from DeepSeek V4's pre-SiLU gate clamp.
    """
    gate, up = torch.chunk(x, 2, dim=-1)
    gate = F.silu(gate).clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    inter = gate * up
    return (inter * permuted_probs).to(x.dtype)


def get_expert_activation_for_deepep(config: MoEConfig):
    """Return the DeepEP expert activation function selected by the MoE config."""

    if config.expert_activation == "swiglu":
        # DeepSeek V4 uses a clamped FP32 variant when swiglu_limit > 0.
        if getattr(config, "swiglu_limit", 0.0) > 0.0:
            return partial(swiglu_clamped_deepep, limit=config.swiglu_limit)
        return weighted_bias_swiglu_impl
    elif config.expert_activation == "swiglu_step":
        if getattr(config, "swiglu_limit", 0.0) > 0.0:
            return partial(swiglu_step_deepep, limit=config.swiglu_limit)
        return weighted_bias_swiglu_impl
    elif config.expert_activation == "swigluoai":
        return partial(
            swiglu_oai_deepep,
            alpha=config.activation_alpha,
            limit=config.activation_limit,
        )
    elif config.expert_activation == "quick_geglu":
        return partial(
            quick_geglu_deepep,
            limit=config.activation_limit,
            alpha=config.activation_alpha,
            linear_offset=1.0,
        )
    elif config.expert_activation == "geglu":
        return weighted_bias_geglu_impl
    elif config.expert_activation == "relu2":
        return relu2_deepep
    else:
        raise ValueError(f"Invalid expert activation: {config.expert_activation}")


class GroupedExpertsDeepEP(nn.Module):
    """
    Sparse MoE implementation using grouped GEMM with DeepEP token dispatch.

    Supports two GEMM backends via BackendConfig.experts:
    - grouped_gemm.ops.gmm (experts="gmm", default)
    - torch._grouped_mm (experts="torch_mm", no external dependency)

    Once the experts for a particular token have been identified, this module
    is invoked to compute and average the output of the activated experts.

    Attributes:
        n_routed_experts (int): Total number of experts in the model.
        gate_and_up_projs (nn.Parameter): Linear layer for gate+up (gated) or just up (non-gated).
        down_projs (nn.Parameter): Linear layer for hidden-to-output transformation.
    """

    def __init__(
        self,
        config: MoEConfig,
        backend: Optional["BackendConfig"] = None,
        dispatcher_backend: str = "deepep",
        dispatcher_num_sms: int = 20,
        dispatcher_num_sms_preprocessing: Optional[int] = None,
        dispatcher_share_token_dispatcher: bool = True,
        dispatcher_async_dispatch: bool = False,
    ):
        """
        Initializes the GroupedExperts module.

        Args:
            config: MoE configuration containing expert parameters.
            backend: Backend configuration. When backend.experts == "torch_mm",
                uses torch._grouped_mm; otherwise uses grouped_gemm.ops.gmm.
            dispatcher_backend: Backend for the flex token dispatcher ("deepep" or "hybridep").
            dispatcher_num_sms: Number of SMs to use for the dispatcher backend.
            dispatcher_num_sms_preprocessing: Optional number of SMs for HybridEP metadata preprocessing.
            dispatcher_share_token_dispatcher: Whether to share a flex dispatcher communication manager across layers.
            dispatcher_async_dispatch: Whether DeepEP/UCCL-EP dispatch should run asynchronously.
        """
        super().__init__()

        self.config = config
        # "torch_mm_mxfp8" dispatches identically to "torch_mm" but routes the grouped
        # GEMMs through torchao's MXFP8 kernel (see _torch_mm_experts_fwd).
        self.use_torch_mm = backend is not None and backend.experts in ("torch_mm", "torch_mm_mxfp8")
        self.use_mxfp8 = backend is not None and backend.experts == "torch_mm_mxfp8"
        self.expert_bias = config.expert_bias
        self.is_gated = is_gated_activation(config.expert_activation)
        self.dispatcher_backend = dispatcher_backend
        self.dispatcher_num_sms = dispatcher_num_sms
        self.dispatcher_num_sms_preprocessing = dispatcher_num_sms_preprocessing
        self.dispatcher_share_token_dispatcher = dispatcher_share_token_dispatcher
        self.dispatcher_async_dispatch = dispatcher_async_dispatch

        # Allocate projection tensor - size depends on whether activation is gated
        # Gated (SwiGLU, Quick-GEGLU): [n_experts, dim, 2*inter_dim]
        # Non-gated (ReLU²): [n_experts, dim, inter_dim]
        up_proj_dim = config.moe_inter_dim * 2 if self.is_gated else config.moe_inter_dim
        self.gate_and_up_projs = nn.Parameter(torch.empty(config.n_routed_experts, config.expert_dim, up_proj_dim))

        self.down_projs = nn.Parameter(torch.empty(config.n_routed_experts, config.moe_inter_dim, config.expert_dim))

        if self.expert_bias:
            self.gate_up_proj_bias = nn.Parameter(torch.empty(config.n_routed_experts, up_proj_dim))
            self.down_proj_bias = nn.Parameter(torch.empty(config.n_routed_experts, config.expert_dim))
        else:
            self.gate_up_proj_bias = None
            self.down_proj_bias = None

        self.expert_activation = get_expert_activation_for_deepep(config)

    def init_token_dispatcher(self, ep_mesh: DeviceMesh):
        self.ep_size = ep_mesh.size()
        self.ep_rank = ep_mesh.get_local_rank()
        ep_group = ep_mesh.get_group()

        config = TokenDispatcherConfig(
            moe_router_topk=self.config.n_activated_experts,
            num_moe_experts=self.config.n_routed_experts,
            moe_permute_fusion=True,
            moe_enable_deepep=True,
            moe_flex_dispatcher_backend=self.dispatcher_backend,
            moe_deepep_num_sms=self.dispatcher_num_sms,
            moe_hybridep_num_sms=self.dispatcher_num_sms,
            moe_hybridep_num_sms_preprocessing=self.dispatcher_num_sms_preprocessing,
            moe_share_token_dispatcher=self.dispatcher_share_token_dispatcher,
            moe_deepep_async_dispatch=self.dispatcher_async_dispatch,
        )

        self.n_routed_experts = self.config.n_routed_experts

        num_local_experts = self.config.n_routed_experts // self.ep_size

        local_expert_indices_offset = self.ep_rank * num_local_experts
        local_expert_indices = [local_expert_indices_offset + i for i in range(num_local_experts)]

        self.token_dispatcher = MoEFlexTokenDispatcher(
            num_local_experts=num_local_experts,
            local_expert_indices=local_expert_indices,
            config=config,
            ep_group=ep_group,
        )
        if self.dispatcher_backend == "deepep":
            self._init_deepep_buffer(ep_group)

    def _init_deepep_buffer(self, ep_group: dist.ProcessGroup) -> None:
        """Initialize DeepEP communication buffers before activation checkpointing."""
        from nemo_automodel.components.moe.megatron.fused_a2a import get_buffer

        dtype_size = max(torch.empty((), dtype=self.config.dtype).element_size(), 2)
        get_buffer(ep_group, self.config.expert_dim * dtype_size)

    def forward(
        self,
        x: torch.Tensor,
        token_mask: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for the grouped experts.

        Args:
            x (torch.Tensor): Input tensor. Shape is [num_tokens, model_dim].
            token_mask (torch.Tensor): Boolean mask indicating valid tokens.
                Shape is [num_tokens].
            weights (torch.Tensor): Routing weights for the selected experts.
                Shape is [num_tokens, num_activated_experts].
            indices (torch.Tensor): Indices of the selected experts.
                Shape is [num_tokens, num_activated_experts].

        Returns:
            torch.Tensor: Output tensor after expert computation.
                Shape is [num_tokens, model_dim]
        """
        assert not isinstance(x, DTensor)

        assert self.n_routed_experts % self.ep_size == 0, (
            f"Number of experts must be divisible by ep_size (ep_size={self.ep_size})"
        )

        weights, indices = _mask_routing_metadata(weights, indices, token_mask)
        (permuted_local_hidden_states, tokens_per_expert, permuted_probs) = self.token_dispatcher.token_permutation2(
            hidden_states=x,
            num_local_tokens=x.size(0),
            token_probs=weights,
            token_indices=indices,
        )
        permuted_probs = permuted_probs.unsqueeze(-1)

        # Cast expert weights to the activation dtype so that fp32-stored
        # parameters (e.g. under fp32 master weights) still work with kernels
        # (grouped_gemm / torch._grouped_mm) that require matching dtypes with
        # the (typically bf16) activations. When the weights are already in the
        # activation dtype these casts are no-ops.
        compute_dtype = permuted_local_hidden_states.dtype
        gate_and_up_projs = self.gate_and_up_projs.to_local().to(compute_dtype)
        down_projs = self.down_projs.to_local().to(compute_dtype)

        if torch.count_nonzero(tokens_per_expert) > 0:
            if self.use_torch_mm:
                tokens_per_expert_gpu = tokens_per_expert.to(
                    device=permuted_local_hidden_states.device, non_blocking=True
                )

                if self.expert_bias:
                    # torch._grouped_mm does not support bias yet (raises
                    # "RuntimeError: Bias not supported yet" as of PyTorch 2.10).
                    # Apply bias manually after each grouped GEMM via _apply_bias.
                    # select_grouped_mm routes through torchao MXFP8 (with the contiguous-
                    # operand relayout) when use_mxfp8, else plain torch._grouped_mm.
                    offs = tokens_per_expert_gpu.cumsum(dim=0).to(torch.int32)
                    grouped_mm = select_grouped_mm(self.use_mxfp8)
                    output1 = grouped_mm(permuted_local_hidden_states, gate_and_up_projs, offs)
                    gate_up_proj_bias = self.gate_up_proj_bias.to_local()
                    # MXFP8: the grouped_mm wrapper clamps its quant input (see
                    # select_grouped_mm) so a bias-shifted value can't overflow the e8m0
                    # block scale -> nan (seen on gpt-oss). The bias-add stays a bf16
                    # separate add (torchao v0.17.0 has no bias arg). bf16 path unchanged.
                    output1 = _apply_bias(output1, gate_up_proj_bias, tokens_per_expert)
                    output1 = self.expert_activation(output1, permuted_probs)
                    output2 = grouped_mm(output1, down_projs, offs)
                    down_bias = self.down_proj_bias.to_local()
                    output2 = _apply_bias(output2, down_bias, tokens_per_expert, permuted_probs)
                else:
                    output2 = _torch_mm_experts_fwd(
                        permuted_local_hidden_states,
                        gate_and_up_projs,
                        down_projs,
                        tokens_per_expert_gpu,
                        permuted_probs,
                        self.expert_activation,
                        use_mxfp8=self.use_mxfp8,
                    )
            else:
                tokens_per_expert = tokens_per_expert.to("cpu")
                output1 = ops.gmm(
                    permuted_local_hidden_states,
                    gate_and_up_projs,
                    tokens_per_expert,
                    trans_b=False,
                )

                if self.expert_bias:
                    gate_up_proj_bias = self.gate_up_proj_bias.to_local().to(compute_dtype)
                    output1 = _apply_bias(output1, gate_up_proj_bias, tokens_per_expert)

                output1 = self.expert_activation(output1, permuted_probs)
                output2 = ops.gmm(output1, down_projs, tokens_per_expert, trans_b=False)

                if self.expert_bias:
                    down_bias = self.down_proj_bias.to_local().to(compute_dtype)
                    output2 = _apply_bias(output2, down_bias, tokens_per_expert, permuted_probs)
        else:
            output1 = torch.matmul(x[0] * 0, gate_and_up_projs[0])
            output1_ = self.expert_activation(output1, permuted_probs)
            output2 = torch.matmul(output1_, down_projs[0])

        y = self.token_dispatcher.token_unpermutation(output2)
        return y

    def init_weights(self, buffer_device: torch.device, init_std: float = 0.02) -> None:
        self.apply(partial(_init_weights, buffer_device=buffer_device, init_std=init_std))


def _torch_mm_experts_fwd(
    hidden_states,
    gate_and_up_projs,
    down_projs,
    tokens_per_expert,
    permuted_probs,
    activation_fn,
    use_mxfp8=False,
):
    # torchao's MXFP8 quantizer (mx_tensor.to_mx) strictly asserts is_contiguous() on each
    # operand it quantizes, unlike torch._grouped_mm. select_grouped_mm returns a wrapper
    # that makes A contiguous and relays out B (so its transpose is contiguous, the layout
    # torchao wants); when mxfp8 is off it returns plain torch._grouped_mm (byte-identical).
    offs = tokens_per_expert.cumsum(dim=0).to(torch.int32)
    grouped_mm = select_grouped_mm(use_mxfp8)
    output1 = grouped_mm(hidden_states, gate_and_up_projs, offs)
    output1 = activation_fn(output1, permuted_probs)
    output2 = grouped_mm(output1, down_projs, offs)
    return output2


class GroupedExpertsTE(nn.Module):
    """
    MoE experts using TE's GroupedLinear module directly.

    Uses TE's native GroupedLinear for computation, providing:
    - Optimized grouped GEMM kernels from TE

    For expert parallelism, each rank creates GroupedLinear with
    num_local_experts = n_routed_experts / ep_size.

    Attributes:
        n_routed_experts (int): Total number of experts in the model.
        gate_up_linear (GroupedLinear): Combined gate and up projection.
        down_linear (GroupedLinear): Down projection.
    """

    def __init__(
        self,
        config: MoEConfig,
        backend: Optional["BackendConfig"] = None,
        dispatcher_backend: str = "deepep",
        dispatcher_num_sms: int = 20,
        dispatcher_num_sms_preprocessing: Optional[int] = None,
        dispatcher_share_token_dispatcher: bool = True,
        dispatcher_async_dispatch: bool = False,
    ):
        """
        Initialize the GroupedExpertsTEGroupedLinear module.

        Args:
            config: MoE configuration containing expert parameters.
            backend: Backend configuration (reserved for future use).
            dispatcher_backend: Backend for the flex token dispatcher ("deepep" or "hybridep").
            dispatcher_num_sms: Number of SMs to use for the dispatcher backend.
            dispatcher_num_sms_preprocessing: Optional number of SMs for HybridEP metadata preprocessing.
            dispatcher_share_token_dispatcher: Whether to share a flex dispatcher communication manager across layers.
            dispatcher_async_dispatch: Whether DeepEP/UCCL-EP dispatch should run asynchronously.
        """
        from nemo_automodel.components.models.common.utils import _patch_te_modules

        _patch_te_modules()

        super().__init__()

        self.config = config
        self.num_local_experts = config.n_routed_experts
        self.expert_bias = config.expert_bias
        self.dim = config.dim
        self.moe_inter_dim = config.moe_inter_dim
        self.is_gated = is_gated_activation(config.expert_activation)
        # Compatibility attributes retained for graph-discovery code written before
        # the TE module and TE-ops expert implementations were split into classes.
        self.__dict__.setdefault("use_te_ops", False)
        self.__dict__.setdefault("_te_ops_mxfp8_fusion_requested", False)
        self.__dict__.setdefault("_te_ops_fusion_checked", False)
        self.dispatcher_backend = dispatcher_backend
        self.dispatcher_num_sms = dispatcher_num_sms
        self.dispatcher_num_sms_preprocessing = dispatcher_num_sms_preprocessing
        self.dispatcher_share_token_dispatcher = dispatcher_share_token_dispatcher
        self.dispatcher_async_dispatch = dispatcher_async_dispatch

        self._build_grouped_linears(config.n_routed_experts)
        self.expert_activation = get_expert_activation_for_deepep(config)

        # FP8 padding/unpadding for GEMM alignment (initialized with full expert count,
        # re-created in init_token_dispatcher with num_local_experts for EP)
        from transformer_engine.pytorch import Fp8Padding, Fp8Unpadding

        self.fp8_padding = Fp8Padding(config.n_routed_experts)
        self.fp8_unpadding = Fp8Unpadding(config.n_routed_experts)

        self.token_dispatcher = None
        self.ep_mesh = None
        self.moe_mesh = None
        self.ep_rank = 0

    def _build_grouped_linears(self, num_experts: int) -> None:
        """Build TE module GroupedLinear projections on the meta device."""
        gate_up_out_features = self.config.moe_inter_dim * 2 if self.is_gated else self.config.moe_inter_dim
        from transformer_engine.pytorch import GroupedLinear

        self.gate_up_linear = GroupedLinear(
            num_gemms=num_experts,
            in_features=self.config.expert_dim,
            out_features=gate_up_out_features,
            bias=self.expert_bias,
            params_dtype=self.config.dtype,
            device="meta",
        )
        self.down_linear = GroupedLinear(
            num_gemms=num_experts,
            in_features=self.config.moe_inter_dim,
            out_features=self.config.expert_dim,
            bias=self.expert_bias,
            params_dtype=self.config.dtype,
            device="meta",
        )
        self.__dict__["_te_grouped_mlp"] = None

    @staticmethod
    def _group_count(linear: nn.Module) -> int:
        num_gemms = getattr(linear, "num_gemms", None)
        return num_gemms if num_gemms is not None else linear.num_groups

    def _get_stacked_weight(self, linear: "GroupedLinear", transpose: bool = False) -> torch.Tensor:
        weights = []
        for i in range(self._group_count(linear)):
            w = getattr(linear, f"weight{i}")
            if isinstance(w, DTensor):
                w = w.to_local()
            weights.append(w)
        stacked = torch.stack(weights, dim=0)  # [num_experts, out, in]
        if transpose:
            stacked = stacked.transpose(-1, -2)  # [num_experts, in, out]
        return stacked

    def _get_stacked_bias(self, linear: "GroupedLinear") -> Optional[torch.Tensor]:
        if not linear.use_bias:
            return None
        biases = []
        for i in range(self._group_count(linear)):
            b = getattr(linear, f"bias{i}")
            if isinstance(b, DTensor):
                b = b.to_local()
            biases.append(b)
        stacked = torch.stack(biases, dim=0)  # [num_experts, out_features]
        return stacked

    def _set_stacked_weight(self, linear: "GroupedLinear", stacked: torch.Tensor, transpose: bool = False):
        if transpose:
            stacked = stacked.transpose(-1, -2)  # [num_experts, out, in]
        for i in range(self._group_count(linear)):
            weight_param = getattr(linear, f"weight{i}")
            if isinstance(weight_param, DTensor):
                weight_param = weight_param.to_local()
            weight_param.data.copy_(stacked[i])

    def _set_stacked_bias(self, linear: "GroupedLinear", stacked: torch.Tensor):
        if not linear.use_bias or stacked is None:
            return
        for i in range(self._group_count(linear)):
            bias_param = getattr(linear, f"bias{i}")
            if isinstance(bias_param, DTensor):
                bias_param = bias_param.to_local()
            bias_param.data.copy_(stacked[i])

    def _to_ep_dtensor(self, tensor: torch.Tensor, *, ep_shard_dim: int = 1) -> torch.Tensor:
        device_mesh = self.moe_mesh or self.ep_mesh
        dtensor = create_dtensor_from_local(
            tensor,
            device_mesh,
            self.ep_rank if device_mesh is not None else None,
            ep_shard_dim=ep_shard_dim,
        )
        return dtensor

    def _canonical_bias_ep_shard_dim(self) -> int:
        """Return the ep_shard dimension of the stacked canonical bias view."""
        # Legacy TE owns one bias vector per expert and FSDP shards each vector,
        # so stacking those local vectors produces an output-dimension shard.
        return 1

    def _normalize_moe_mesh(self, moe_mesh: Optional[DeviceMesh]) -> Optional[DeviceMesh]:
        if moe_mesh is None:
            return None
        allowed_dims = ("ep", "ep_shard", "ep_replicate")
        dims = tuple(dim for dim in moe_mesh.mesh_dim_names if dim in allowed_dims)
        if not dims:
            return None
        if dims == tuple(moe_mesh.mesh_dim_names):
            return moe_mesh
        return moe_mesh[dims]

    def set_moe_mesh(self, moe_mesh: Optional[DeviceMesh]) -> None:
        self.moe_mesh = self._normalize_moe_mesh(moe_mesh)

    def _router_expert_pad_multiple(self) -> int | None:
        """Return the dispatcher padding multiple required by this expert kernel."""
        return None

    @property
    def gate_and_up_projs(self) -> torch.Tensor:
        tensor = self._to_ep_dtensor(self._get_stacked_weight(self.gate_up_linear, transpose=True))
        return tensor

    @gate_and_up_projs.setter
    def gate_and_up_projs(self, value: Optional[torch.Tensor]) -> None:
        if value is None:
            return
        if isinstance(value, DTensor):
            value = value.to_local()
        self._set_stacked_weight(self.gate_up_linear, value, transpose=True)
        self._weights_loaded_from_checkpoint = True

    @property
    def down_projs(self) -> torch.Tensor:
        return self._to_ep_dtensor(self._get_stacked_weight(self.down_linear, transpose=True))

    @down_projs.setter
    def down_projs(self, value: Optional[torch.Tensor]) -> None:
        if value is None:
            return
        if isinstance(value, DTensor):
            value = value.to_local()
        self._set_stacked_weight(self.down_linear, value, transpose=True)
        self._weights_loaded_from_checkpoint = True

    @property
    def gate_up_proj_bias(self) -> Optional[torch.Tensor]:
        if not self.expert_bias:
            return None
        bias = self._get_stacked_bias(self.gate_up_linear)
        if bias is None:
            return None
        return self._to_ep_dtensor(bias, ep_shard_dim=self._canonical_bias_ep_shard_dim())

    @gate_up_proj_bias.setter
    def gate_up_proj_bias(self, value: Optional[torch.Tensor]) -> None:
        if not self.expert_bias or value is None:
            return
        if isinstance(value, DTensor):
            value = value.to_local()
        self._set_stacked_bias(self.gate_up_linear, value)

    @property
    def down_proj_bias(self) -> Optional[torch.Tensor]:
        if not self.expert_bias:
            return None
        bias = self._get_stacked_bias(self.down_linear)
        if bias is None:
            return None
        return self._to_ep_dtensor(bias, ep_shard_dim=self._canonical_bias_ep_shard_dim())

    @down_proj_bias.setter
    def down_proj_bias(self, value: Optional[torch.Tensor]) -> None:
        if not self.expert_bias or value is None:
            return
        if isinstance(value, DTensor):
            value = value.to_local()
        self._set_stacked_bias(self.down_linear, value)

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False, **kwargs) -> Dict[str, Any]:
        """
        Return state dict with stacked tensors in DeepEP format.

        Converts TE GroupedLinear's weight{i} parameters to stacked format:
        - gate_and_up_projs: [num_local_experts, dim, moe_inter_dim * 2]
        - down_projs: [num_local_experts, moe_inter_dim, dim]

        When EP is enabled, returns DTensors sharded on dimension 0.
        """
        gate_and_up_weight = self.gate_and_up_projs
        down_weight = self.down_projs

        def _maybe_detach(t: torch.Tensor) -> torch.Tensor:
            if keep_vars:
                return t
            return t.detach()

        state = {
            f"{prefix}gate_and_up_projs": _maybe_detach(gate_and_up_weight),
            f"{prefix}down_projs": _maybe_detach(down_weight),
        }

        if self.expert_bias:
            gate_up_bias = self.gate_up_proj_bias
            down_bias = self.down_proj_bias
            state[f"{prefix}gate_up_proj_bias"] = _maybe_detach(gate_up_bias)
            state[f"{prefix}down_proj_bias"] = _maybe_detach(down_bias)

        if destination is not None:
            if hasattr(destination, "_metadata"):
                destination._metadata[prefix[:-1]] = dict(version=self._version)
            destination.update(state)
            return destination

        return state

    def _load_from_state_dict(
        self,
        state_dict: Dict[str, Any],
        prefix: str,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """
        Load state dict with stacked tensors in DeepEP format.

        Converts stacked format to TE GroupedLinear's weight{i} parameters:
        - gate_and_up_projs: [num_local_experts, dim, moe_inter_dim * 2]
        - down_projs: [num_local_experts, moe_inter_dim, dim]
        """
        gate_up_key = f"{prefix}gate_and_up_projs"
        down_key = f"{prefix}down_projs"

        if gate_up_key in state_dict:
            gate_up_weight = state_dict[gate_up_key]
            if isinstance(gate_up_weight, DTensor):
                gate_up_weight = gate_up_weight.to_local()
            self._set_stacked_weight(self.gate_up_linear, gate_up_weight, transpose=True)
            self._weights_loaded_from_checkpoint = True
        else:
            missing_keys.append(gate_up_key)

        if down_key in state_dict:
            down_weight = state_dict[down_key]
            if isinstance(down_weight, DTensor):
                down_weight = down_weight.to_local()
            self._set_stacked_weight(self.down_linear, down_weight, transpose=True)
            self._weights_loaded_from_checkpoint = True
        else:
            missing_keys.append(down_key)

        if self.expert_bias:
            gate_up_bias_key = f"{prefix}gate_up_proj_bias"
            down_bias_key = f"{prefix}down_proj_bias"

            if gate_up_bias_key in state_dict:
                gate_up_bias = state_dict[gate_up_bias_key]
                if isinstance(gate_up_bias, DTensor):
                    gate_up_bias = gate_up_bias.to_local()
                self._set_stacked_bias(self.gate_up_linear, gate_up_bias)
            else:
                missing_keys.append(gate_up_bias_key)

            if down_bias_key in state_dict:
                down_bias = state_dict[down_bias_key]
                if isinstance(down_bias, DTensor):
                    down_bias = down_bias.to_local()
                self._set_stacked_bias(self.down_linear, down_bias)
            else:
                missing_keys.append(down_bias_key)

    def init_token_dispatcher(self, ep_mesh: DeviceMesh, moe_mesh: Optional[DeviceMesh] = None):
        """
        Initialize the token dispatcher for expert parallelism.

        Called by the parallelizer after model initialization.

        Args:
            ep_mesh: Device mesh for expert parallelism.
        """
        from nemo_automodel.components.models.common.utils import _patch_te_modules

        _patch_te_modules()

        self.ep_mesh = ep_mesh
        self.ep_rank = ep_mesh.get_local_rank()
        self.ep_size = ep_mesh.size()
        self.set_moe_mesh(moe_mesh if moe_mesh is not None else ep_mesh)

        assert self.config.n_routed_experts % self.ep_size == 0, (
            f"n_routed_experts ({self.config.n_routed_experts}) must be divisible by ep_size ({self.ep_size})"
        )
        self.num_local_experts = self.config.n_routed_experts // self.ep_size

        self._build_grouped_linears(self.num_local_experts)

        token_dispatcher_config = TokenDispatcherConfig(
            moe_router_topk=self.config.n_activated_experts,
            num_moe_experts=self.config.n_routed_experts,
            moe_permute_fusion=True,
            moe_enable_deepep=True,
            moe_flex_dispatcher_backend=self.dispatcher_backend,
            moe_deepep_num_sms=self.dispatcher_num_sms,
            moe_hybridep_num_sms=self.dispatcher_num_sms,
            moe_hybridep_num_sms_preprocessing=self.dispatcher_num_sms_preprocessing,
            moe_share_token_dispatcher=self.dispatcher_share_token_dispatcher,
            moe_deepep_async_dispatch=self.dispatcher_async_dispatch,
            moe_router_expert_pad_multiple=self._router_expert_pad_multiple(),
            moe_expert_rank_capacity_factor=getattr(self, "moe_expert_rank_capacity_factor", None),
        )

        local_expert_indices_offset = self.ep_rank * self.num_local_experts
        local_expert_indices = [local_expert_indices_offset + i for i in range(self.num_local_experts)]

        self.token_dispatcher = MoEFlexTokenDispatcher(
            num_local_experts=self.num_local_experts,
            local_expert_indices=local_expert_indices,
            config=token_dispatcher_config,
            ep_group=ep_mesh.get_group(),
        )

        # Re-create FP8 padding/unpadding with num_local_experts for EP
        from transformer_engine.pytorch import Fp8Padding, Fp8Unpadding

        self.fp8_padding = Fp8Padding(self.num_local_experts)
        self.fp8_unpadding = Fp8Unpadding(self.num_local_experts)

    def forward(
        self,
        x: torch.Tensor,
        token_mask: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass using TE's GroupedLinear with native FP8 support.

        Args:
            x: [num_tokens, model_dim] input tensor
            token_mask: [num_tokens] boolean mask for valid tokens
            weights: [num_tokens, num_activated_experts] routing weights
            indices: [num_tokens, num_activated_experts] expert indices

        Returns:
            [num_tokens, model_dim] output tensor
        """
        assert not isinstance(x, DTensor), "Input should not be a DTensor"
        assert self.config.n_routed_experts % self.ep_size == 0, (
            f"Number of experts must be divisible by ep_size (ep_size={self.ep_size})"
        )

        weights, indices = _mask_routing_metadata(weights, indices, token_mask)

        (permuted_local_hidden_states, tokens_per_expert, permuted_probs) = self.token_dispatcher.token_permutation2(
            hidden_states=x,
            num_local_tokens=x.size(0),
            token_probs=weights,
            token_indices=indices,
        )

        from transformer_engine.pytorch.quantization import FP8GlobalStateManager

        fp8_active = FP8GlobalStateManager.is_fp8_enabled()

        permuted_probs = permuted_probs.unsqueeze(-1)
        if isinstance(tokens_per_expert, torch.Tensor):
            m_splits = tokens_per_expert.tolist()
        else:
            m_splits = list(tokens_per_expert)

        actual_m_splits = None
        if fp8_active:
            actual_m_splits = m_splits
            permuted_local_hidden_states, m_splits = self.fp8_padding(permuted_local_hidden_states, m_splits)
            permuted_probs, _ = self.fp8_padding(permuted_probs, actual_m_splits)

        if sum(m_splits) > 0:
            output1 = self.gate_up_linear(permuted_local_hidden_states, m_splits)
            output1 = self.expert_activation(output1, permuted_probs)
            output2 = self.down_linear(output1, m_splits)
            # The down-projection bias must be weighted by the per-token routing probability
            # (permuted_probs), matching GroupedExperts ("expert_out + down_bias * w") and
            # GroupedExpertsDeepEP ("_apply_bias(output2, down_bias, ..., permuted_probs)").
            # TE's GroupedLinear adds the down bias UNWEIGHTED inside the GEMM, so each of the
            # top-k expert contributions carries a full prob-independent bias that is then
            # summed in the combine step, producing a large systematic offset (e.g. gpt-oss-20b
            # step-0 loss ~8.2 vs the correct ~4.5). Add the missing (prob - 1) * down_bias term
            # so the net down-bias contribution becomes prob * down_bias.
            if self.expert_bias:
                down_bias = self._get_stacked_bias(self.down_linear)
                if down_bias is not None:
                    splits_t = torch.as_tensor(m_splits, device=output2.device)
                    output2 = _apply_bias(output2, down_bias, splits_t, permuted_probs - 1.0)
        else:
            # Handle edge case: no tokens routed to local experts
            # Perform dummy computation for gradient flow
            def to_local(tensor):
                if isinstance(tensor, DTensor):
                    return tensor.to_local()
                else:
                    return tensor

            output1 = torch.matmul(x[0] * 0, to_local(self.gate_up_linear.weight0).T)
            output1_ = self.expert_activation(output1, permuted_probs)
            output2 = torch.matmul(output1_, to_local(self.down_linear.weight0).T)

        if fp8_active and actual_m_splits is not None:
            output2 = self.fp8_unpadding(output2, actual_m_splits)

        y = self.token_dispatcher.token_unpermutation(output2)
        return y

    def init_weights(self, buffer_device: torch.device, init_std: float = 0.02) -> None:
        """Initialize weights using reset_parameters()"""
        self.gate_up_linear.reset_parameters()
        self.down_linear.reset_parameters()


class _TeOpsMXFP8WeightCache:
    """Non-registered MXFP8 compute storage for one stacked owner.

    The high-precision ``nn.Parameter`` remains the sole optimizer and checkpoint
    owner. Transformer Engine's grouped tensor is only a compute cache. Graph-off
    training defaults to lazily refreshed, preallocated GEMM-ready member storage;
    fused grouped quantization remains available as an internal A/B mode. Partial
    expert CUDA graphs use preallocated canonical-scale storage with synchronous
    post-step refresh so every address remains fixed across graph replay.
    """

    GROUP_QUANTIZE_MODE = "group_quantize"
    GEMM_READY_FIXED_MODE = "gemm_ready_fixed"
    FIXED_ADDRESS_MODE = "fixed_address"
    LAZY_MODES = frozenset((GROUP_QUANTIZE_MODE, GEMM_READY_FIXED_MODE))
    PREALLOCATED_MODES = frozenset((GEMM_READY_FIXED_MODE, FIXED_ADDRESS_MODE))
    MODES = LAZY_MODES | PREALLOCATED_MODES

    def __init__(
        self,
        owner: torch.Tensor,
        *,
        num_groups: int,
        out_features: int,
        in_features: int,
        mode: str = FIXED_ADDRESS_MODE,
    ) -> None:
        if mode not in self.MODES:
            raise ValueError(
                f"Unsupported TE-ops MXFP8 weight cache mode {mode!r}; expected one of {sorted(self.MODES)}"
            )
        self.num_groups = num_groups
        self.out_features = out_features
        self.in_features = in_features
        self.mode = mode
        self._validate_owner(owner)

        try:
            import transformer_engine_torch as tex
            from transformer_engine.pytorch.tensor import GroupedTensorStorage, MXFP8Quantizer
        except ImportError as error:
            raise ImportError("TE-ops MXFP8 weight caching requires Transformer Engine 2.16.1 or newer") from error

        refresh_quantizer = MXFP8Quantizer(
            tex.DType.kFloat8E4M3,
            rowwise=True,
            columnwise=True,
        )
        refresh_quantizer.internal = False
        # Graph capture keeps the existing compact-scale representation. The
        # graph-off GEMM-ready mode asks TE to allocate and update the padded,
        # swizzled scale layout consumed directly by GEMM.
        refresh_quantizer.optimize_for_gemm = mode == self.GEMM_READY_FIXED_MODE
        self._tex = tex
        self.refresh_quantizer = refresh_quantizer
        self.tensor = None
        self.members: tuple[Any, ...] = ()
        self.storage_identity: tuple[Any, ...] | None = None
        self.owner_key: tuple[Any, ...] | None = None
        self.invalidated = False
        self.refresh_count = 0
        self.group_quantize_count = 0
        self.member_update_count = 0
        self.buffer_replacement_count = 0

        if mode in self.PREALLOCATED_MODES:
            grouped_tensor = GroupedTensorStorage.make_grouped_tensor_with_shapes(
                num_tensors=num_groups,
                shapes=[(out_features, in_features)] * num_groups,
                quantizer=refresh_quantizer,
                device=owner.device,
                dtype=owner.dtype,
            )
            grouped_tensor.requires_grad_(owner.requires_grad)
            members = tuple(grouped_tensor.quantized_tensors)
            if len(members) != num_groups:
                raise RuntimeError(f"TE MXFP8 cache created {len(members)} members for {num_groups} expert weights")
            self.tensor = grouped_tensor
            self.members = members
            self.storage_identity = self._storage_identity()
        self.refresh(owner, force=True)
        if self.storage_identity is None:
            self.storage_identity = self._storage_identity()

    @property
    def expected_shape(self) -> tuple[int, int, int]:
        return (self.num_groups, self.out_features, self.in_features)

    def _validate_owner(self, owner: torch.Tensor) -> None:
        if tuple(owner.shape) != self.expected_shape:
            raise RuntimeError(
                f"TE-ops MXFP8 cache expected owner shape {self.expected_shape}, got {tuple(owner.shape)}"
            )
        if owner.device.type != "cuda":
            raise RuntimeError(f"TE-ops MXFP8 cache requires a CUDA owner, got {owner.device}")
        if owner.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise RuntimeError(f"TE-ops MXFP8 cache requires FP32/FP16/BF16 owner storage, got {owner.dtype}")
        if not owner.is_contiguous():
            raise RuntimeError("TE-ops MXFP8 cache requires a contiguous stacked owner")
        if self.out_features % 32 != 0 or self.in_features % 32 != 0:
            raise RuntimeError(
                "TE-ops MXFP8 cache requires expert output and input dimensions divisible by 32, "
                f"got ({self.out_features}, {self.in_features})"
            )

    @staticmethod
    def _owner_key(owner: torch.Tensor) -> tuple[Any, ...]:
        return (
            owner.data_ptr(),
            int(owner._version),
            owner.device.type,
            getattr(owner.device, "index", None),
            owner.dtype,
            tuple(owner.shape),
            owner.requires_grad,
        )

    def _storage_identity(self) -> tuple[Any, ...]:
        grouped = self.tensor
        if grouped is None:
            return ()
        members = getattr(grouped, "quantized_tensors", None)

        def offset_identity(offsets: Any) -> Any:
            if isinstance(offsets, torch.Tensor):
                return (offsets.data_ptr(), tuple(offsets.shape), offsets.dtype, offsets.device)
            if offsets is None:
                return None
            return tuple(offsets)

        return (
            id(grouped),
            tuple(id(member) for member in members) if members is not None else (),
            grouped.rowwise_data.data_ptr(),
            tuple(grouped.rowwise_data.shape),
            grouped.columnwise_data.data_ptr(),
            tuple(grouped.columnwise_data.shape),
            grouped.scale_inv.data_ptr(),
            tuple(grouped.scale_inv.shape),
            grouped.columnwise_scale_inv.data_ptr(),
            tuple(grouped.columnwise_scale_inv.shape),
            offset_identity(getattr(grouped, "tensor_offsets", None)),
            offset_identity(getattr(grouped, "offsets", None)),
            offset_identity(getattr(grouped, "scale_inv_offsets", None)),
            offset_identity(getattr(grouped, "columnwise_scale_inv_offsets", None)),
            bool(getattr(grouped, "_with_gemm_swizzled_scales", False)),
        )

    def has_stable_storage_identity(self) -> bool:
        """Return whether every persistent wrapper and backing buffer is unchanged."""
        return self.storage_identity == self._storage_identity()

    def is_current(self, owner: torch.Tensor) -> bool:
        """Return whether the cache already represents this owner generation."""
        return not self.invalidated and self.owner_key == self._owner_key(owner)

    def invalidate(self) -> None:
        """Mark cached compute weights stale without launching quantization."""
        self.invalidated = True

    def _update_preallocated_members(self, owner: torch.Tensor) -> None:
        """Launch in-place quantization into existing member storage."""
        assert self.tensor is not None
        owner_members = owner.unbind(0)
        if len(owner_members) != len(self.members):
            raise RuntimeError(
                f"TE-ops MXFP8 cache has {len(self.members)} members for {len(owner_members)} owner slices"
            )
        with torch.no_grad():
            for source, destination in zip(owner_members, self.members):
                self.refresh_quantizer.update_quantized(source, destination)

    def capture_fixed_address_refresh(self, owner: torch.Tensor) -> None:
        """Launch only fixed-address refresh kernels for CUDA graph capture."""
        self._validate_owner(owner)
        if self.mode != self.FIXED_ADDRESS_MODE:
            raise RuntimeError(f"TE-ops MXFP8 cache-refresh CUDA graphs require fixed_address mode, got {self.mode!r}")
        if self.tensor is None or not self.has_stable_storage_identity():
            raise RuntimeError("TE-ops MXFP8 cache-refresh CUDA graphs require stable preallocated storage")
        if self.tensor.requires_grad != owner.requires_grad:
            raise RuntimeError("TE-ops MXFP8 cache owner and destination requires_grad state changed")
        self._update_preallocated_members(owner)

    def mark_fixed_address_graph_replayed(self, owner: torch.Tensor) -> None:
        """Advance generation and diagnostics after one actual graph launch."""
        self._validate_owner(owner)
        if self.mode != self.FIXED_ADDRESS_MODE or not self.has_stable_storage_identity():
            raise RuntimeError("TE-ops MXFP8 cache changed while its refresh CUDA graph was live")
        self.owner_key = self._owner_key(owner)
        self.invalidated = False
        self.refresh_count += 1
        self.member_update_count += len(self.members)

    def refresh(self, owner: torch.Tensor, *, force: bool = False) -> bool:
        """Refresh the selected MXFP8 representation for a new owner generation."""
        self._validate_owner(owner)
        owner_key = self._owner_key(owner)
        if not force and not self.invalidated and self.owner_key == owner_key:
            return False

        if self.mode == self.GROUP_QUANTIZE_MODE:
            previous_tensor = self.tensor
            with torch.no_grad():
                grouped_tensor = self._tex.group_quantize(
                    owner.view(self.num_groups * self.out_features, self.in_features),
                    self.refresh_quantizer,
                    self.num_groups,
                    None,
                )
            grouped_tensor.requires_grad_(owner.requires_grad)
            if int(grouped_tensor.num_tensors) != self.num_groups:
                raise RuntimeError(
                    f"TE MXFP8 grouped quantization returned {grouped_tensor.num_tensors} tensors, "
                    f"expected {self.num_groups}"
                )
            self.tensor = grouped_tensor
            self.members = tuple(grouped_tensor.quantized_tensors or ())
            self.group_quantize_count += 1
            self.buffer_replacement_count += int(previous_tensor is not None)
        else:
            assert self.tensor is not None
            if self.tensor.requires_grad != owner.requires_grad:
                self.tensor.requires_grad_(owner.requires_grad)
            self._update_preallocated_members(owner)
            self.member_update_count += len(self.members)
        self.owner_key = owner_key
        self.invalidated = False
        self.refresh_count += 1
        return True


@cache
def _get_unstacked_te_ops_grouped_linear_class() -> type[nn.Module]:
    """Build TE's native per-expert-parameter GroupedLinear variant."""
    try:
        from transformer_engine.pytorch import ops as te_ops
    except ImportError as error:
        raise ImportError("experts='te_ops' requires Transformer Engine 2.16.1 or newer") from error

    class _UnstackedTeOpsGroupedLinear(te_ops.GroupedLinear):
        """TE GroupedLinear with ordinary ``weight0`` ... ``weightN`` owners.

        The parent ``GroupedExpertsTeOps`` exposes a canonical virtual state dict,
        so the physical TE parameters must not independently consume checkpoint
        keys during recursive loading.
        """

        def __init__(
            self,
            num_groups: int,
            in_features: int,
            out_features: int,
            *,
            bias: bool,
            dtype: torch.dtype,
            device: torch.device | str,
            scale_bias: bool = False,
        ) -> None:
            super().__init__(
                num_groups=num_groups,
                in_features=in_features,
                out_features=out_features,
                bias=bias,
                dtype=dtype,
                device=device,
                single_grouped_weight=False,
                single_grouped_bias=False,
                scale_bias=scale_bias,
            )
            if self.single_grouped_weight or self.single_grouped_bias:
                raise RuntimeError("TE unexpectedly replaced unstacked expert parameters")

        def _load_from_state_dict(
            self,
            state_dict: Dict[str, Any],
            prefix: str,
            local_metadata,
            strict: bool,
            missing_keys: list[str],
            unexpected_keys: list[str],
            error_msgs: list[str],
        ) -> None:
            """Skip physical keys; the parent loads canonical expert tensors."""
            del state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs

    return _UnstackedTeOpsGroupedLinear


@cache
def _get_stacked_te_ops_grouped_linear_class() -> type[nn.Module]:
    """Build the TE-ops GroupedLinear variant with FSDP-safe stacked owners."""
    try:
        from transformer_engine.pytorch import ops as te_ops
        from transformer_engine.pytorch.tensor import GroupedTensor
    except ImportError as error:
        raise ImportError("experts='te_ops' stacked parameters require Transformer Engine 2.16.1 or newer") from error

    class _StackedTeOpsGroupedLinear(te_ops.GroupedLinear):
        """TE GroupedLinear backed by plain stacked Parameters.

        PyTorch FSDP2 cannot shard TE's GroupedTensor parameter directly. The registered
        owners here are ordinary tensors in TE layout (``[experts, out, in]`` and
        ``[experts, out]``). TE sees transient GroupedTensor aliases over the currently
        unsharded owner storage, while OperationFuser records the plain owners as its
        autograd inputs and maps TE's packed gradients back to them.
        """

        def __init__(
            self,
            num_groups: int,
            in_features: int,
            out_features: int,
            *,
            bias: bool,
            dtype: torch.dtype,
            device: torch.device | str,
            scale_bias: bool = False,
        ) -> None:
            previous_single_param_env = os.environ.get("NVTE_GROUPED_LINEAR_SINGLE_PARAM")
            os.environ["NVTE_GROUPED_LINEAR_SINGLE_PARAM"] = "1"
            try:
                super().__init__(
                    num_groups=num_groups,
                    in_features=in_features,
                    out_features=out_features,
                    bias=bias,
                    dtype=dtype,
                    device=device,
                    single_grouped_weight=True,
                    single_grouped_bias=bias,
                    scale_bias=scale_bias,
                )
            finally:
                if previous_single_param_env is None:
                    del os.environ["NVTE_GROUPED_LINEAR_SINGLE_PARAM"]
                else:
                    os.environ["NVTE_GROUPED_LINEAR_SINGLE_PARAM"] = previous_single_param_env

            if not self.single_grouped_weight or (bias and not self.single_grouped_bias):
                raise RuntimeError(
                    "Transformer Engine disabled single grouped parameters for experts='te_ops'. "
                    "Use a TE build with single_grouped_weight/single_grouped_bias support."
                )
            if self._accumulate_into_main_grad or self.wgrad_store.delay_wgrad_compute():
                raise ValueError("Stacked TE-ops experts do not support main-grad accumulation or delayed wgrad")

            weight_shape = (num_groups, out_features, in_features)
            grouped_weight = self._parameters.pop("weight", None)
            if grouped_weight is not None:
                weight_data = grouped_weight.rowwise_data.view(weight_shape)
            else:
                weight_data = torch.stack(
                    [self._parameters[f"weight{group_idx}"] for group_idx in range(num_groups)], dim=0
                )
            weight_owner = nn.Parameter(weight_data, requires_grad=True)
            weight_owner._te_ops_stacked_weight = True
            for group_idx in range(num_groups):
                self.register_parameter(f"weight{group_idx}", None)
            self.register_parameter("_stacked_weight", weight_owner)

            if bias:
                bias_shape = (num_groups, out_features)
                grouped_bias = self._parameters.pop("bias", None)
                if grouped_bias is not None:
                    bias_data = grouped_bias.rowwise_data.view(bias_shape)
                else:
                    bias_data = torch.stack(
                        [self._parameters[f"bias{group_idx}"] for group_idx in range(num_groups)], dim=0
                    )
                bias_owner = nn.Parameter(bias_data, requires_grad=True)
                bias_owner._te_ops_stacked_bias = True
                for group_idx in range(num_groups):
                    self.register_parameter(f"bias{group_idx}", None)
                self.register_parameter("_stacked_bias", bias_owner)

            self.__dict__["_stacked_weight_alias"] = None
            self.__dict__["_stacked_weight_alias_key"] = None
            self.__dict__["_stacked_bias_alias"] = None
            self.__dict__["_stacked_bias_alias_key"] = None
            self.__dict__["_mxfp8_weight_cache_enabled"] = False
            self.__dict__["_mxfp8_weight_cache"] = None
            self.__dict__["_mxfp8_weight_cache_allocations"] = 0
            self.__dict__["_mxfp8_weight_cache_hits"] = 0
            self.__dict__["_mxfp8_weight_cache_fallbacks"] = 0
            self.__dict__["_mxfp8_weight_cache_optimizer_invalidations"] = 0
            self.__dict__["_mxfp8_weight_cache_optimizer_refreshes"] = 0
            self.__dict__["_mxfp8_weight_cache_fallback_reason"] = "disabled by backend config"
            self.__dict__["_mxfp8_weight_cache_mode"] = _TeOpsMXFP8WeightCache.FIXED_ADDRESS_MODE

        def _owner_local_tensor(self, name: str, expected_shape: tuple[int, ...]) -> torch.Tensor:
            owner = self._parameters[name]
            local_owner = owner.to_local() if isinstance(owner, DTensor) else owner
            if tuple(local_owner.shape) != expected_shape:
                raise RuntimeError(
                    f"TE-ops stacked owner {name} has local shape {tuple(local_owner.shape)}, "
                    f"expected unsharded shape {expected_shape}. The GroupedTensor runtime view "
                    "must be created inside the expert FSDP pre-forward unshard window."
                )
            if not local_owner.is_contiguous():
                raise RuntimeError(f"TE-ops stacked owner {name} must be contiguous after FSDP unshard")
            return local_owner

        def _grouped_alias(
            self,
            owner_name: str,
            member_shape: tuple[int, ...],
            alias_name: str,
        ) -> torch.Tensor:
            expected_shape = (self.num_groups, *member_shape)
            owner = self._parameters[owner_name]
            local_owner = self._owner_local_tensor(owner_name, expected_shape)
            cache_key = (
                id(local_owner),
                local_owner.data_ptr(),
                local_owner.device,
                local_owner.dtype,
                tuple(local_owner.shape),
            )
            alias_key_name = f"{alias_name}_key"
            if self.__dict__.get(alias_key_name) == cache_key:
                return self.__dict__[alias_name]

            alias = GroupedTensor.make_grouped_tensor_from_rowwise_data(
                num_tensors=self.num_groups,
                tensor_shape=member_shape,
                rowwise_data=local_owner.view(-1),
                dtype=local_owner.dtype,
            )
            alias.requires_grad_(owner.requires_grad)
            self.__dict__[alias_name] = alias
            self.__dict__[alias_key_name] = cache_key
            return alias

        def set_mxfp8_weight_cache_enabled(
            self,
            enabled: bool,
            *,
            fallback_reason: str = "",
            mode: str = _TeOpsMXFP8WeightCache.FIXED_ADDRESS_MODE,
        ) -> None:
            """Enable the local compute cache or drop it before expert FSDP wrapping."""
            enabled = bool(enabled)
            if mode not in _TeOpsMXFP8WeightCache.MODES:
                raise ValueError(
                    f"Unsupported TE-ops MXFP8 weight cache mode {mode!r}; "
                    f"expected one of {sorted(_TeOpsMXFP8WeightCache.MODES)}"
                )
            if self.__dict__.get("_mxfp8_weight_cache_mode") != mode:
                self.__dict__["_mxfp8_weight_cache"] = None
            self.__dict__["_mxfp8_weight_cache_mode"] = mode
            self.__dict__["_mxfp8_weight_cache_enabled"] = enabled
            if not enabled:
                self.__dict__["_mxfp8_weight_cache"] = None
                self.__dict__["_mxfp8_weight_cache_fallback_reason"] = fallback_reason or "disabled"
                return
            self.__dict__["_mxfp8_weight_cache_fallback_reason"] = "awaiting owner materialization"
            self.refresh_mxfp8_weight_cache_if_needed()
            if self.__dict__.get("_mxfp8_weight_cache") is not None:
                self.__dict__["_mxfp8_weight_cache_fallback_reason"] = ""

        def refresh_mxfp8_weight_cache_if_needed(self, *, force: bool = False) -> bool:
            """Refresh the compute cache outside the TE Sequential graph target."""
            if not self.__dict__.get("_mxfp8_weight_cache_enabled", False):
                return False
            expected_shape = (self.num_groups, self.out_features, self.in_features)
            owner = self._owner_local_tensor("_stacked_weight", expected_shape)
            if owner.is_meta:
                self.__dict__["_mxfp8_weight_cache_fallback_reason"] = "awaiting owner materialization"
                return False

            cache_state = self.__dict__.get("_mxfp8_weight_cache")
            if cache_state is None:
                cache_state = _TeOpsMXFP8WeightCache(
                    owner,
                    num_groups=self.num_groups,
                    out_features=self.out_features,
                    in_features=self.in_features,
                    mode=self.__dict__["_mxfp8_weight_cache_mode"],
                )
                self.__dict__["_mxfp8_weight_cache"] = cache_state
                self.__dict__["_mxfp8_weight_cache_allocations"] += 1
                self.__dict__["_mxfp8_weight_cache_fallback_reason"] = ""
                return True

            refreshed = cache_state.refresh(owner, force=force)
            if refreshed:
                self.__dict__["_mxfp8_weight_cache_fallback_reason"] = ""
            return refreshed

        def mxfp8_weight_cache_graph_signature(self) -> tuple[Any, ...]:
            """Return immutable owner and destination identities for graph replay."""
            if not self.__dict__.get("_mxfp8_weight_cache_enabled", False):
                raise RuntimeError("TE-ops MXFP8 weight cache is disabled")
            mode = self.__dict__.get("_mxfp8_weight_cache_mode")
            if mode != _TeOpsMXFP8WeightCache.FIXED_ADDRESS_MODE:
                raise RuntimeError(f"TE-ops MXFP8 cache-refresh CUDA graphs require fixed_address mode, got {mode!r}")
            registered_owner = self._parameters["_stacked_weight"]
            owner = self._owner_local_tensor(
                "_stacked_weight",
                (self.num_groups, self.out_features, self.in_features),
            )
            cache_state = self.__dict__.get("_mxfp8_weight_cache")
            if cache_state is None or cache_state.mode != _TeOpsMXFP8WeightCache.FIXED_ADDRESS_MODE:
                raise RuntimeError("TE-ops MXFP8 fixed-address cache has not been materialized")
            if not cache_state.has_stable_storage_identity():
                raise RuntimeError("TE-ops MXFP8 fixed-address cache storage identity changed")
            return (
                id(self),
                id(cache_state),
                id(registered_owner),
                owner.data_ptr(),
                tuple(owner.shape),
                tuple(owner.stride()),
                owner.dtype,
                owner.device,
                owner.requires_grad,
                cache_state._storage_identity(),
            )

        def capture_mxfp8_weight_cache_refresh(self) -> None:
            """Launch the fixed-address refresh kernels without Python bookkeeping."""
            self.mxfp8_weight_cache_graph_signature()
            owner = self._owner_local_tensor(
                "_stacked_weight",
                (self.num_groups, self.out_features, self.in_features),
            )
            self.__dict__["_mxfp8_weight_cache"].capture_fixed_address_refresh(owner)

        def mark_mxfp8_weight_cache_refresh_graph_replayed(self) -> bool:
            """Mark one graph refresh current for eager forwards and diagnostics."""
            self.mxfp8_weight_cache_graph_signature()
            owner = self._owner_local_tensor(
                "_stacked_weight",
                (self.num_groups, self.out_features, self.in_features),
            )
            self.__dict__["_mxfp8_weight_cache"].mark_fixed_address_graph_replayed(owner)
            self.__dict__["_mxfp8_weight_cache_optimizer_refreshes"] += 1
            return True

        def invalidate_mxfp8_weight_cache(self) -> bool:
            """Mark a lazy graph-off cache stale at an optimizer boundary."""
            if not self.__dict__.get("_mxfp8_weight_cache_enabled", False):
                return False
            if self.__dict__.get("_mxfp8_weight_cache_mode") not in _TeOpsMXFP8WeightCache.LAZY_MODES:
                return False
            cache_state = self.__dict__.get("_mxfp8_weight_cache")
            if cache_state is not None:
                cache_state.invalidate()
            return True

        def mxfp8_weight_cache_diagnostics(self) -> dict[str, Any]:
            """Return bounded counters and stable storage identities for profiling."""
            cache_state = self.__dict__.get("_mxfp8_weight_cache")
            storage = None
            refreshes = 0
            current = False
            dirty = False
            if cache_state is not None:
                owner = self._parameters["_stacked_weight"]
                owner = owner.to_local() if isinstance(owner, DTensor) else owner
                current = cache_state.is_current(owner)
                dirty = cache_state.invalidated
                refreshes = cache_state.refresh_count
                grouped = cache_state.tensor
                storage = {
                    "cache_id": id(grouped),
                    "member_ids": tuple(id(member) for member in cache_state.members),
                    "rowwise_data_ptr": grouped.rowwise_data.data_ptr(),
                    "columnwise_data_ptr": grouped.columnwise_data.data_ptr(),
                    "rowwise_scale_ptr": grouped.scale_inv.data_ptr(),
                    "columnwise_scale_ptr": grouped.columnwise_scale_inv.data_ptr(),
                    "rowwise_data_numel": grouped.rowwise_data.numel(),
                    "columnwise_data_numel": grouped.columnwise_data.numel(),
                    "rowwise_scale_numel": grouped.scale_inv.numel(),
                    "columnwise_scale_numel": grouped.columnwise_scale_inv.numel(),
                    "offsets": tuple(getattr(grouped, "offsets", ()) or ()),
                    "rowwise_scale_offsets": tuple(getattr(grouped, "scale_inv_offsets", ()) or ()),
                    "columnwise_scale_offsets": tuple(getattr(grouped, "columnwise_scale_inv_offsets", ()) or ()),
                    "with_gemm_swizzled_scales": bool(getattr(grouped, "_with_gemm_swizzled_scales", False)),
                    "identity_stable": cache_state.has_stable_storage_identity(),
                }
            mode = self.__dict__.get("_mxfp8_weight_cache_mode", _TeOpsMXFP8WeightCache.FIXED_ADDRESS_MODE)
            requires_fixed_identity = mode in _TeOpsMXFP8WeightCache.PREALLOCATED_MODES
            return {
                "enabled": self.__dict__.get("_mxfp8_weight_cache_enabled", False),
                "mode": mode,
                "optimize_for_gemm": False if cache_state is None else cache_state.refresh_quantizer.optimize_for_gemm,
                "allocations": self.__dict__.get("_mxfp8_weight_cache_allocations", 0),
                "refreshes": refreshes,
                "group_quantize_calls": 0 if cache_state is None else cache_state.group_quantize_count,
                "member_update_calls": 0 if cache_state is None else cache_state.member_update_count,
                "buffer_replacements": 0 if cache_state is None else cache_state.buffer_replacement_count,
                "optimizer_invalidations": self.__dict__.get("_mxfp8_weight_cache_optimizer_invalidations", 0),
                "optimizer_refreshes": self.__dict__.get("_mxfp8_weight_cache_optimizer_refreshes", 0),
                "hits": self.__dict__.get("_mxfp8_weight_cache_hits", 0),
                "fallbacks": self.__dict__.get("_mxfp8_weight_cache_fallbacks", 0),
                "current": current,
                "dirty": dirty,
                "requires_fixed_identity": requires_fixed_identity,
                "identity_policy_satisfied": not requires_fixed_identity
                or (storage is not None and storage["identity_stable"]),
                "fallback_reason": self.__dict__.get("_mxfp8_weight_cache_fallback_reason", ""),
                "storage": storage,
            }

        def __getattr__(self, name: str):
            parameters = self.__dict__.get("_parameters", {})
            if name == "weight" and parameters.get("_stacked_weight") is not None:
                if self.__dict__.get("_mxfp8_weight_cache_enabled", False):
                    cache_state = self.__dict__.get("_mxfp8_weight_cache")
                    if cache_state is not None:
                        owner = parameters["_stacked_weight"]
                        owner = owner.to_local() if isinstance(owner, DTensor) else owner
                        if cache_state.is_current(owner):
                            self.__dict__["_mxfp8_weight_cache_hits"] += 1
                            return cache_state.tensor
                        self.__dict__["_mxfp8_weight_cache_fallbacks"] += 1
                        self.__dict__["_mxfp8_weight_cache_fallback_reason"] = (
                            "owner generation changed before cache refresh"
                        )
                return self._grouped_alias(
                    "_stacked_weight",
                    (self.out_features, self.in_features),
                    "_stacked_weight_alias",
                )
            if name == "bias" and parameters.get("_stacked_bias") is not None:
                return self._grouped_alias(
                    "_stacked_bias",
                    (self.out_features,),
                    "_stacked_bias_alias",
                )
            return super().__getattr__(name)

        def stacked_weight_local(self) -> torch.Tensor:
            """Return the registered stacked weight owner's current local tensor."""
            owner = self._parameters["_stacked_weight"]
            return owner.to_local() if isinstance(owner, DTensor) else owner

        def stacked_bias_local(self) -> torch.Tensor | None:
            """Return the registered stacked bias owner's current local tensor."""
            owner = self._parameters.get("_stacked_bias")
            if owner is None:
                return None
            return owner.to_local() if isinstance(owner, DTensor) else owner

        def clear_grouped_aliases(self) -> None:
            """Drop aliases after owner storage is replaced or reshaped."""
            self.__dict__["_stacked_weight_alias"] = None
            self.__dict__["_stacked_weight_alias_key"] = None
            self.__dict__["_stacked_bias_alias"] = None
            self.__dict__["_stacked_bias_alias_key"] = None

        def reset_parameters(self) -> None:
            """Initialize stacked owners in-place without replacing Parameters."""
            if self.__dict__.get("_parameters", {}).get("_stacked_weight") is None:
                super().reset_parameters()
                return
            with torch.no_grad():
                bound = 1 / math.sqrt(self.in_features)
                self.stacked_weight_local().uniform_(-bound, bound)
                bias = self.stacked_bias_local()
                if bias is not None:
                    bias.zero_()
            self.clear_grouped_aliases()
            self.refresh_mxfp8_weight_cache_if_needed(force=True)

        def _load_from_state_dict(
            self,
            state_dict: Dict[str, Any],
            prefix: str,
            local_metadata,
            strict: bool,
            missing_keys: list[str],
            unexpected_keys: list[str],
            error_msgs: list[str],
        ) -> None:
            """Skip internal owner keys; the parent loads canonical expert tensors."""
            del state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs

    return _StackedTeOpsGroupedLinear


def _select_te_ops_activation(
    config: MoEConfig,
    full_mxfp8_fusion_requested: bool = False,
) -> tuple[str, dict[str, Any], bool, bool]:
    """Select the TE activation op without importing the optional TE package.

    Returns the op name, constructor arguments, whether the activation consumes
    routing probabilities itself, and whether TE's full MXFP8 grouped-MLP fusion
    is numerically compatible with the selection.
    """
    activation = config.expert_activation
    full_mxfp8_dims_supported = config.expert_dim % 64 == 0 and config.moe_inter_dim % 64 == 0
    if activation == "swiglu":
        swiglu_limit = float(getattr(config, "swiglu_limit", 0.0))
        if swiglu_limit > 0.0:
            return (
                "exact_gated",
                {"alpha": 1.0, "linear_offset": 0.0, "limit": swiglu_limit},
                False,
                False,
            )
        full_mxfp8_fusion = full_mxfp8_dims_supported
        kwargs = {"glu_interleave_size": 32} if full_mxfp8_fusion_requested and full_mxfp8_fusion else {}
        return "scaled_swiglu", kwargs, True, full_mxfp8_fusion

    if activation == "swiglu_step":
        swiglu_limit = float(getattr(config, "swiglu_limit", 0.0))
        if swiglu_limit > 0.0:
            return (
                "exact_gated",
                {
                    "alpha": 1.0,
                    "linear_offset": 0.0,
                    "limit": swiglu_limit,
                    "clamp_after_gate_activation": True,
                    "use_input_dtype": True,
                },
                False,
                False,
            )
        full_mxfp8_fusion = full_mxfp8_dims_supported
        kwargs = {"glu_interleave_size": 32} if full_mxfp8_fusion_requested and full_mxfp8_fusion else {}
        return "scaled_swiglu", kwargs, True, full_mxfp8_fusion

    if activation == "swigluoai":
        limit = float(config.activation_limit)
        alpha = float(config.activation_alpha)
        if limit <= 0.0:
            return (
                "exact_gated",
                {"alpha": alpha, "linear_offset": 1.0, "limit": None},
                False,
                False,
            )
        full_mxfp8_fusion = (
            full_mxfp8_dims_supported
            and math.isclose(alpha, 1.702, abs_tol=1e-3)
            and math.isclose(limit, 7.0, abs_tol=1e-6)
        )
        kwargs = {"limit": limit, "alpha": alpha}
        if full_mxfp8_fusion_requested and full_mxfp8_fusion:
            kwargs["glu_interleave_size"] = 32
        return (
            "scaled_clamped_qgeglu",
            kwargs,
            True,
            full_mxfp8_fusion,
        )

    if activation == "quick_geglu":
        # GPT-OSS checkpoints use element-interleaved gate/up rows. Unfused TE
        # consumes concatenated rows; the CuTe grouped MLP consumes 32-wide blocks.
        limit = float(config.activation_limit)
        alpha = float(config.activation_alpha)
        full_mxfp8_fusion = (
            full_mxfp8_dims_supported
            and math.isclose(alpha, 1.702, abs_tol=1e-3)
            and math.isclose(limit, 7.0, abs_tol=1e-6)
        )
        kwargs = {"limit": limit, "alpha": alpha}
        if full_mxfp8_fusion_requested and full_mxfp8_fusion:
            kwargs["glu_interleave_size"] = 32
        return (
            "scaled_clamped_qgeglu",
            kwargs,
            True,
            full_mxfp8_fusion,
        )

    if activation == "geglu":
        return "geglu", {}, False, False
    if activation == "relu2":
        return "scaled_srelu", {}, True, full_mxfp8_dims_supported
    raise ValueError(f"experts='te_ops' does not support expert_activation={activation!r}")


@cache
def _get_te_ops_custom_classes() -> tuple[type[nn.Module], type[nn.Module]]:
    """Build the small custom fusible ops used by generic TE experts."""
    try:
        from transformer_engine.pytorch import ops as te_ops
    except ImportError as error:
        raise ImportError("experts='te_ops' requires Transformer Engine 2.16.1 or newer") from error

    class _TeOpsRowScale(te_ops.BasicOperation):
        """Multiply every token row by its routing probability."""

        num_extra_inputs: int = 1

        def op_forward(self, *args, **kwargs) -> None:
            raise RuntimeError("_TeOpsRowScale overrides fuser_forward instead of op_forward")

        def op_backward(self, *args, **kwargs) -> None:
            raise RuntimeError("_TeOpsRowScale overrides fuser_backward instead of op_backward")

        def fuser_forward(
            self,
            basic_op_ctxs,
            input_: torch.Tensor,
            *,
            basic_op_extra_inputs,
            prev_op_grad_output_quantizer,
            next_op_input_quantizer,
            basic_op_kwargs,
        ):
            del prev_op_grad_output_quantizer, next_op_input_quantizer, basic_op_kwargs
            scales = basic_op_extra_inputs[0][0]
            if tuple(scales.shape) != tuple(input_.shape[:-1]):
                raise ValueError(
                    f"TE-ops route scales have shape {tuple(scales.shape)}, expected {tuple(input_.shape[:-1])}"
                )
            ctx = basic_op_ctxs[0]
            if ctx.requires_grad:
                ctx.extra_input_requires_grad = scales.requires_grad
                ctx.save_for_backward(input_, scales)
            return input_ * scales.unsqueeze(-1), [()]

        def fuser_backward(self, basic_op_ctxs, grad_output: torch.Tensor, *, basic_op_grad_extra_outputs):
            del basic_op_grad_extra_outputs
            ctx = basic_op_ctxs[0]
            input_, scales = ctx.saved_tensors
            # Match autograd's promotion for ``input_ * scales``. In the exact
            # activation path ``input_`` is FP32 while the following TE linear
            # returns BF16 dgrad; the canonical explicit BF16 cast promotes that
            # dgrad back to FP32 before differentiating the FP32 route multiply.
            compute_dtype = torch.promote_types(torch.promote_types(input_.dtype, scales.dtype), grad_output.dtype)
            grad_output_compute = grad_output.to(compute_dtype)
            grad_input = (grad_output_compute * scales.to(compute_dtype).unsqueeze(-1)).to(input_.dtype)
            grad_scales = None
            if ctx.extra_input_requires_grad:
                grad_scales = torch.linalg.vecdot(input_.to(compute_dtype), grad_output_compute).to(scales.dtype)
            return grad_input, [()], [(grad_scales,)]

    class _TeOpsExactGatedActivation(te_ops.BasicOperation):
        """Exact FP32 gated activation for variants without a native TE op."""

        def __init__(
            self,
            *,
            alpha: float,
            linear_offset: float,
            limit: float | None,
            clamp_after_gate_activation: bool = False,
            use_input_dtype: bool = False,
        ) -> None:
            super().__init__()
            self.alpha = alpha
            self.linear_offset = linear_offset
            self.limit = limit
            self.clamp_after_gate_activation = clamp_after_gate_activation
            self.use_input_dtype = use_input_dtype

        def op_forward(
            self,
            ctx,
            input_: torch.Tensor,
            prev_op_grad_output_quantizer,
            next_op_input_quantizer,
        ) -> torch.Tensor:
            del prev_op_grad_output_quantizer, next_op_input_quantizer
            if input_.shape[-1] % 2 != 0:
                raise ValueError(f"Gated TE activation requires an even width, got {input_.shape[-1]}")
            gate, up = input_.chunk(2, dim=-1)
            if not self.use_input_dtype:
                gate = gate.float()
                up = up.float()
            if self.limit is not None and not self.clamp_after_gate_activation:
                gate = gate.clamp(max=self.limit)
                up = up.clamp(min=-self.limit, max=self.limit)
            if self.clamp_after_gate_activation or self.alpha == 1.0:
                # Keep exact fused SiLU semantics for Step and DeepSeek V4;
                # decomposing sigmoid + multiply adds a rounding point.
                activated_gate = F.silu(gate)
            else:
                activated_gate = gate * torch.sigmoid(self.alpha * gate)
            if self.limit is not None and self.clamp_after_gate_activation:
                activated_gate = activated_gate.clamp(max=self.limit)
                up = up.clamp(min=-self.limit, max=self.limit)
            output = activated_gate * (up + self.linear_offset)
            if ctx.requires_grad:
                ctx.save_for_backward(input_)
            # The FP32 variants must keep their activation product in FP32 until
            # the following row-scale op applies routing probabilities. The
            # canonical expert path rounds only after that multiply; rounding
            # here would change both outputs and router/parameter gradients.
            # Step intentionally evaluates the whole activation in the input
            # dtype, so preserve its existing BF16/FP16 result.
            return output.to(input_.dtype) if self.use_input_dtype else output

        def op_backward(self, ctx, grad_output: torch.Tensor):
            (input_,) = ctx.saved_tensors
            gate, up = input_.chunk(2, dim=-1)
            if not self.use_input_dtype:
                gate = gate.float()
                up = up.float()
            gate_mask = None
            up_mask = None
            if self.limit is not None and not self.clamp_after_gate_activation:
                gate_mask = gate <= self.limit
                up_mask = (up >= -self.limit) & (up <= self.limit)
                gate = gate.clamp(max=self.limit)
                up = up.clamp(min=-self.limit, max=self.limit)

            sigmoid = torch.sigmoid(self.alpha * gate)
            quick_gelu = gate * sigmoid
            if self.limit is not None and self.clamp_after_gate_activation:
                gate_mask = quick_gelu <= self.limit
                up_mask = (up >= -self.limit) & (up <= self.limit)
                quick_gelu = quick_gelu.clamp(max=self.limit)
                up = up.clamp(min=-self.limit, max=self.limit)
            grad_output = grad_output.float()
            grad_gate = grad_output * (up + self.linear_offset) * sigmoid * (1.0 + self.alpha * gate * (1.0 - sigmoid))
            grad_up = grad_output * quick_gelu
            if gate_mask is not None:
                grad_gate = grad_gate * gate_mask
            if up_mask is not None:
                grad_up = grad_up * up_mask
            return torch.cat((grad_gate, grad_up), dim=-1).to(input_.dtype), ()

    return _TeOpsRowScale, _TeOpsExactGatedActivation


class GroupedExpertsTeOps(GroupedExpertsTE):
    """MoE experts using TE fusible ops with stacked or native owners."""

    def __init__(
        self,
        config: MoEConfig,
        backend: Optional["BackendConfig"] = None,
        dispatcher_backend: str = "deepep",
        dispatcher_num_sms: int = 20,
        dispatcher_num_sms_preprocessing: Optional[int] = None,
        dispatcher_share_token_dispatcher: bool = True,
        dispatcher_async_dispatch: bool = False,
    ) -> None:
        if backend is None or backend.experts != "te_ops":
            raise ValueError("GroupedExpertsTeOps requires BackendConfig(experts='te_ops')")
        configured_fp8_recipe = getattr(getattr(backend, "te_fp8", None), "recipe", None)
        configured_mxfp8 = configured_fp8_recipe == "mxfp8" or (
            callable(getattr(configured_fp8_recipe, "mxfp8", None)) and configured_fp8_recipe.mxfp8()
        )
        self._te_ops_fp8_configured = backend.te_fp8 is not None
        self._te_ops_configured_mxfp8 = configured_mxfp8
        self.moe_expert_rank_capacity_factor = getattr(backend, "moe_expert_rank_capacity_factor", None)
        self.moe_paged_stash = bool(getattr(backend, "moe_paged_stash", False))
        self._te_ops_paged_stash_requested = self.moe_paged_stash
        if self._te_ops_paged_stash_requested and bool(getattr(backend, "partial_cuda_graph_experts", False)):
            raise ValueError("moe_paged_stash and partial_cuda_graph_experts cannot be enabled together")
        if self._te_ops_paged_stash_requested:
            from nemo_automodel.components.moe.paged_stash import get_paged_stash_manager

            if float(getattr(backend, "moe_paged_stash_buffer_size_factor_cpu", 0.0)) != 0.0:
                raise ValueError("AutoModel moe_paged_stash does not yet support pinned-host spill")
            get_paged_stash_manager().configure(
                enabled=True,
                page_size=int(getattr(backend, "moe_paged_stash_page_size", 64)),
                buffer_size_factor=float(getattr(backend, "moe_paged_stash_buffer_size_factor_cuda", 1.1)),
            )
        self._te_ops_unstacked_parameters = bool(getattr(backend, "te_ops_unstacked_parameters", False))
        self._te_ops_mxfp8_fusion_requested = (
            configured_mxfp8 and int(os.environ.get("NVTE_CUTEDSL_FUSED_GROUPED_MLP", "0")) > 0
        )
        self._te_ops_mxfp8_weight_cache_requested = bool(getattr(backend, "te_ops_mxfp8_weight_cache", False))
        self._te_ops_mxfp8_weight_cache_graph_off_mode = _TeOpsMXFP8WeightCache.GEMM_READY_FIXED_MODE
        self._te_ops_mxfp8_weight_cache_mode = (
            _TeOpsMXFP8WeightCache.FIXED_ADDRESS_MODE
            if bool(
                getattr(backend, "partial_cuda_graph_experts", False)
                or getattr(backend, "full_iteration_cuda_graph", False)
            )
            else self._te_ops_mxfp8_weight_cache_graph_off_mode
        )
        # Keep the cache disabled until EP>1 construction reports whether an
        # additional expert FSDP dimension is active. This prevents ep_shard>1
        # jobs from allocating a cache only to discard it before fully_shard.
        self._te_ops_mxfp8_weight_cache_enabled = False
        self._te_ops_mxfp8_weight_cache_fallback_reason = (
            "awaiting EP shard policy" if self._te_ops_mxfp8_weight_cache_requested else "disabled by backend config"
        )
        self._te_ops_full_mxfp8_fusion_eligible = False
        # Graph buckets may overallocate the physical token buffer while
        # retaining real splits whenever TE selects its graph-safe GroupedTensor
        # path. This is deliberately separate from dispatcher padding, which is
        # required only by the full MXFP8 grouped-MLP fusion.
        self._te_ops_graph_uses_paged_capacity = False
        self._te_ops_uses_padded_capacity = False
        self._te_ops_fuser_owner_signature = None
        self._te_ops_fusion_checked = False
        self.use_te_ops = True
        super().__init__(
            config,
            backend=backend,
            dispatcher_backend=dispatcher_backend,
            dispatcher_num_sms=dispatcher_num_sms,
            dispatcher_num_sms_preprocessing=dispatcher_num_sms_preprocessing,
            dispatcher_share_token_dispatcher=dispatcher_share_token_dispatcher,
            dispatcher_async_dispatch=dispatcher_async_dispatch,
        )
        if self._te_ops_paged_stash_requested and not self._te_ops_uses_padded_capacity:
            raise RuntimeError(
                "moe_paged_stash requires the TE CuTe DSL fused MXFP8 grouped-MLP path. "
                "Set te_fp8.recipe='mxfp8' and NVTE_CUTEDSL_FUSED_GROUPED_MLP=1, and use an activation "
                "supported by the fused path. The required TE saved-tensor markers and per-expert "
                "128-row alignment are not available on the generic BF16/GroupedLinear path."
            )

    def _te_ops_dynamic_splits_graph_capability(self) -> tuple[bool, str]:
        """Return whether TE reads grouped split values from device at replay."""
        self._te_ops_graph_uses_paged_capacity = False
        if not torch.cuda.is_available():
            return False, "CUDA is unavailable"
        capability = torch.cuda.get_device_capability()
        if capability < (10, 0):
            return False, f"TE on-device grouped splits require SM100+, got SM{capability[0]}{capability[1]}"
        checker = getattr(self.gate_up_linear, "_is_graph_safe_path_supported", None)
        if not callable(checker):
            return False, "this Transformer Engine build does not expose the graph-safe GroupedTensor path"
        if self._te_ops_fp8_configured:
            if not self._te_ops_configured_mxfp8:
                return False, "only the MXFP8 TE recipe supports on-device grouped splits"
            if capability >= (12, 0):
                return False, (
                    "TE MXFP8 graph-safe GroupedTensor compute requires SM100/SM110, "
                    f"got SM{capability[0]}{capability[1]}"
                )
            # TE's MXFP8 GroupedLinear path supports the same overallocated
            # physical buffer contract even when the surrounding activation is
            # not eligible for the full grouped-MLP fusion.
            self._te_ops_graph_uses_paged_capacity = True
            return True, ""
        if self.config.dtype not in (torch.bfloat16, torch.float16):
            return False, f"BF16/FP16 grouped compute is required, got {self.config.dtype}"
        try:
            supported = checker(
                with_quantized_compute=False,
                input_quantizers=[None] * self.gate_up_linear.num_groups,
                dtype=self.config.dtype,
            )
        except Exception as error:
            return False, f"TE graph-safe GroupedTensor capability check failed: {error}"
        if not supported:
            return False, "TE selected its host-split grouped GEMM path"
        # TE 2.16.1's graph-safe GroupedTensor forward/backward uses device
        # splits as active-region offsets independently of the physical input
        # length. Its own CUDA-graph test covers BF16/FP16 with physical rows
        # greater than sum(split_sizes), including input and parameter grads.
        self._te_ops_graph_uses_paged_capacity = True
        return True, ""

    def _build_grouped_linears(self, num_experts: int) -> None:
        """Build TE-ops projections with the selected parameter layout."""
        from transformer_engine.pytorch import ops as te_ops

        activation_name, activation_kwargs, activation_scales_routes, full_mxfp8_fusion = _select_te_ops_activation(
            self.config,
            full_mxfp8_fusion_requested=self._te_ops_mxfp8_fusion_requested,
        )
        self._te_glu_interleave_size = activation_kwargs.get("glu_interleave_size")

        linear_class = (
            _get_unstacked_te_ops_grouped_linear_class()
            if self._te_ops_unstacked_parameters
            else _get_stacked_te_ops_grouped_linear_class()
        )
        gate_up_out_features = self.config.moe_inter_dim * 2 if self.is_gated else self.config.moe_inter_dim
        self.gate_up_linear = linear_class(
            num_groups=num_experts,
            in_features=self.config.expert_dim,
            out_features=gate_up_out_features,
            bias=self.expert_bias,
            dtype=self.config.dtype,
            device="meta",
        )
        self.down_linear = linear_class(
            num_groups=num_experts,
            in_features=self.config.moe_inter_dim,
            out_features=self.config.expert_dim,
            bias=self.expert_bias,
            dtype=self.config.dtype,
            device="meta",
            scale_bias=self.expert_bias,
        )
        for linear in (self.gate_up_linear, self.down_linear):
            configure_cache = getattr(linear, "set_mxfp8_weight_cache_enabled", None)
            if callable(configure_cache):
                configure_cache(
                    self._te_ops_mxfp8_weight_cache_enabled,
                    fallback_reason=self._te_ops_mxfp8_weight_cache_fallback_reason,
                    mode=self._te_ops_mxfp8_weight_cache_mode,
                )

        if activation_name == "scaled_swiglu":
            activation = te_ops.ScaledSwiGLU(**activation_kwargs)
        elif activation_name == "scaled_clamped_qgeglu":
            activation = te_ops.ScaledClampedQGeGLU(**activation_kwargs)
        elif activation_name == "geglu":
            activation = te_ops.GEGLU(**activation_kwargs)
        elif activation_name == "scaled_srelu":
            scaled_srelu = getattr(te_ops, "ScaledSReLU", None)
            if scaled_srelu is None:
                # ScaledSReLU was added after the other scaled activations. Older
                # TE builds can still run exact BF16 with the generic row scaler.
                activation = te_ops.SReLU(**activation_kwargs)
                activation_scales_routes = False
                full_mxfp8_fusion = False
            else:
                activation = scaled_srelu(**activation_kwargs)
        elif activation_name == "exact_gated":
            _, exact_gated_activation = _get_te_ops_custom_classes()
            activation = exact_gated_activation(**activation_kwargs)
        else:  # pragma: no cover - guarded by _select_te_ops_activation
            raise AssertionError(f"Unhandled TE activation selection {activation_name!r}")

        activation_ops = [activation]
        if not activation_scales_routes:
            row_scale, _ = _get_te_ops_custom_classes()
            activation_ops.append(row_scale())
        self._te_ops_full_mxfp8_fusion_eligible = full_mxfp8_fusion
        self._te_ops_uses_padded_capacity = self._te_ops_mxfp8_fusion_requested and full_mxfp8_fusion
        self._te_ops_fuser_owner_signature = None

        # Sequential is intentionally unregistered because these same linear objects
        # are already registered above. Its OperationFuser is created lazily on the
        # first forward, after FSDP has unsharded the stacked owners.
        self.__dict__["_te_grouped_mlp"] = te_ops.Sequential(
            self.gate_up_linear,
            *activation_ops,
            self.down_linear,
        )

    def configure_mxfp8_weight_cache_for_ep_shard(self, *, ep_shard_enabled: bool) -> None:
        """Enable the cache only when every rank owns complete local expert weights."""
        enabled = self._te_ops_mxfp8_weight_cache_requested and not ep_shard_enabled
        if self._te_ops_mxfp8_weight_cache_requested and ep_shard_enabled:
            fallback_reason = "ep_shard>1 requires FSDP unshard; using eager TE weight quantization"
        elif not self._te_ops_mxfp8_weight_cache_requested:
            fallback_reason = "disabled by backend config"
        else:
            fallback_reason = ""
        self._te_ops_mxfp8_weight_cache_enabled = enabled
        self._te_ops_mxfp8_weight_cache_fallback_reason = fallback_reason
        for linear in (self.gate_up_linear, self.down_linear):
            configure_cache = getattr(linear, "set_mxfp8_weight_cache_enabled", None)
            if callable(configure_cache):
                configure_cache(
                    enabled,
                    fallback_reason=fallback_reason,
                    mode=self._te_ops_mxfp8_weight_cache_mode,
                )

    def configure_mxfp8_weight_cache_for_partial_graph(self, *, captured: bool) -> None:
        """Use fixed storage only for expert layers selected for graph capture."""
        mode = _TeOpsMXFP8WeightCache.FIXED_ADDRESS_MODE if captured else self._te_ops_mxfp8_weight_cache_graph_off_mode
        if mode == self._te_ops_mxfp8_weight_cache_mode:
            return
        self._te_ops_mxfp8_weight_cache_mode = mode
        for linear in (self.gate_up_linear, self.down_linear):
            configure_cache = getattr(linear, "set_mxfp8_weight_cache_enabled", None)
            if callable(configure_cache):
                configure_cache(
                    self._te_ops_mxfp8_weight_cache_enabled,
                    fallback_reason=self._te_ops_mxfp8_weight_cache_fallback_reason,
                    mode=mode,
                )

    def refresh_mxfp8_weight_cache_if_needed(self, *, force: bool = False) -> int:
        """Refresh both projection caches before entering the TE graph target."""
        if not getattr(self, "_te_ops_mxfp8_weight_cache_enabled", False):
            return 0
        refreshes = 0
        for linear in (self.gate_up_linear, self.down_linear):
            refresh = getattr(linear, "refresh_mxfp8_weight_cache_if_needed", None)
            if callable(refresh):
                refreshes += int(refresh(force=force))
        return refreshes

    def mxfp8_weight_cache_diagnostics(self) -> dict[str, Any]:
        """Expose bounded per-projection counters for GB200 validation."""

        def projection_diagnostics(linear: nn.Module) -> dict[str, Any]:
            diagnostics = getattr(linear, "mxfp8_weight_cache_diagnostics", None)
            if callable(diagnostics):
                return diagnostics()
            return {
                "enabled": False,
                "mode": "native_unstacked",
                "optimize_for_gemm": False,
                "allocations": 0,
                "refreshes": 0,
                "group_quantize_calls": 0,
                "member_update_calls": 0,
                "buffer_replacements": 0,
                "optimizer_invalidations": 0,
                "optimizer_refreshes": 0,
                "hits": 0,
                "fallbacks": 0,
                "current": False,
                "dirty": False,
                "requires_fixed_identity": False,
                "identity_policy_satisfied": True,
                "fallback_reason": "native unstacked TE parameters use eager per-expert quantization",
                "storage": None,
            }

        return {
            "requested": getattr(self, "_te_ops_mxfp8_weight_cache_requested", False),
            "enabled": getattr(self, "_te_ops_mxfp8_weight_cache_enabled", False),
            "mode": getattr(
                self,
                "_te_ops_mxfp8_weight_cache_mode",
                _TeOpsMXFP8WeightCache.FIXED_ADDRESS_MODE,
            ),
            "fallback_reason": getattr(self, "_te_ops_mxfp8_weight_cache_fallback_reason", ""),
            "gate_up": projection_diagnostics(self.gate_up_linear),
            "down": projection_diagnostics(self.down_linear),
        }

    def _canonical_bias_ep_shard_dim(self) -> int:
        """Return the canonical bias shard dim for the physical owner layout."""
        return 1 if getattr(self, "_te_ops_unstacked_parameters", False) else 0

    @staticmethod
    def _interleave_glu_blocks(tensor: torch.Tensor, block_size: int = 32) -> torch.Tensor:
        """Convert GPT-OSS ``[gate0, up0, ...]`` rows to TE's 32-wide GLU blocks."""
        if tensor.shape[1] % (2 * block_size) != 0:
            raise ValueError(f"GLU width {tensor.shape[1]} must be divisible by {2 * block_size}")
        tail = tensor.shape[2:]
        pairs = tensor.reshape(tensor.shape[0], -1, 2, *tail)
        blocks = pairs.reshape(tensor.shape[0], -1, block_size, 2, *tail)
        blocks = blocks.permute(0, 1, 3, 2, *range(4, blocks.ndim))
        return blocks.flatten(1, 3)

    @staticmethod
    def _deinterleave_glu_blocks(tensor: torch.Tensor, block_size: int = 32) -> torch.Tensor:
        """Convert TE's 32-wide GLU blocks back to GPT-OSS elementwise gate/up rows."""
        if tensor.shape[1] % (2 * block_size) != 0:
            raise ValueError(f"GLU width {tensor.shape[1]} must be divisible by {2 * block_size}")
        tail = tensor.shape[2:]
        blocks = tensor.reshape(tensor.shape[0], -1, 2, block_size, *tail)
        pairs = blocks.permute(0, 1, 3, 2, *range(4, blocks.ndim))
        return pairs.flatten(1, 3)

    @staticmethod
    def _pair_interleaved_to_concatenated(tensor: torch.Tensor) -> torch.Tensor:
        """Convert GPT-OSS ``[gate0, up0, ...]`` rows to ``[gate | up]``."""
        if tensor.shape[1] % 2 != 0:
            raise ValueError(f"GLU width {tensor.shape[1]} must be even")
        tail = tensor.shape[2:]
        pairs = tensor.reshape(tensor.shape[0], -1, 2, *tail)
        gate = pairs[:, :, 0]
        up = pairs[:, :, 1]
        return torch.cat((gate, up), dim=1)

    @staticmethod
    def _concatenated_to_pair_interleaved(tensor: torch.Tensor) -> torch.Tensor:
        """Convert ``[gate | up]`` rows to GPT-OSS ``[gate0, up0, ...]``."""
        if tensor.shape[1] % 2 != 0:
            raise ValueError(f"GLU width {tensor.shape[1]} must be even")
        gate, up = tensor.chunk(2, dim=1)
        return torch.stack((gate, up), dim=2).flatten(1, 2)

    @staticmethod
    def _interleave_concatenated_glu_blocks(tensor: torch.Tensor, block_size: int = 32) -> torch.Tensor:
        """Convert canonical ``[gate | up]`` rows to TE's block-interleaved layout."""
        if tensor.shape[1] % (2 * block_size) != 0:
            raise ValueError(f"GLU width {tensor.shape[1]} must be divisible by {2 * block_size}")
        tail = tensor.shape[2:]
        gate, up = tensor.chunk(2, dim=1)
        gate = gate.reshape(tensor.shape[0], -1, block_size, *tail)
        up = up.reshape(tensor.shape[0], -1, block_size, *tail)
        return torch.stack((gate, up), dim=2).flatten(1, 3)

    @staticmethod
    def _deinterleave_concatenated_glu_blocks(tensor: torch.Tensor, block_size: int = 32) -> torch.Tensor:
        """Convert TE's block-interleaved rows back to canonical ``[gate | up]``."""
        if tensor.shape[1] % (2 * block_size) != 0:
            raise ValueError(f"GLU width {tensor.shape[1]} must be divisible by {2 * block_size}")
        tail = tensor.shape[2:]
        blocks = tensor.reshape(tensor.shape[0], -1, 2, block_size, *tail)
        gate = blocks[:, :, 0].flatten(1, 2)
        up = blocks[:, :, 1].flatten(1, 2)
        return torch.cat((gate, up), dim=1)

    @staticmethod
    def _to_te_gate_up_layout(tensor: torch.Tensor, activation: str, block_size: int | None) -> torch.Tensor:
        """Convert the canonical checkpoint layout to the selected TE layout."""
        if activation == "quick_geglu":
            if block_size is None:
                return GroupedExpertsTeOps._pair_interleaved_to_concatenated(tensor)
            return GroupedExpertsTeOps._interleave_glu_blocks(tensor, block_size)
        if block_size is None:
            return tensor
        if is_gated_activation(activation):
            return GroupedExpertsTeOps._interleave_concatenated_glu_blocks(tensor, block_size)
        return tensor

    @staticmethod
    def _from_te_gate_up_layout(tensor: torch.Tensor, activation: str, block_size: int | None) -> torch.Tensor:
        """Convert the selected TE layout back to the canonical checkpoint layout."""
        if activation == "quick_geglu":
            if block_size is None:
                return GroupedExpertsTeOps._concatenated_to_pair_interleaved(tensor)
            return GroupedExpertsTeOps._deinterleave_glu_blocks(tensor, block_size)
        if block_size is None:
            return tensor
        if is_gated_activation(activation):
            return GroupedExpertsTeOps._deinterleave_concatenated_glu_blocks(tensor, block_size)
        return tensor

    @staticmethod
    def _uses_single_grouped_weight(linear: nn.Module) -> bool:
        """Return whether a TE op owns one grouped weight parameter."""
        parameters = getattr(linear, "_parameters", {})
        return bool(getattr(linear, "single_grouped_weight", parameters.get("_stacked_weight") is not None))

    @staticmethod
    def _uses_single_grouped_bias(linear: nn.Module) -> bool:
        """Return whether a TE op owns one grouped bias parameter."""
        parameters = getattr(linear, "_parameters", {})
        return bool(getattr(linear, "single_grouped_bias", parameters.get("_stacked_bias") is not None))

    def _get_stacked_weight(self, linear: nn.Module, transpose: bool = False) -> torch.Tensor:
        if self._uses_single_grouped_weight(linear):
            stacked = linear.stacked_weight_local()
        else:
            weights = []
            for group_idx in range(linear.num_groups):
                weight = getattr(linear, f"weight{group_idx}")
                weights.append(weight.to_local() if isinstance(weight, DTensor) else weight)
            stacked = torch.stack(weights, dim=0)
        if linear is self.gate_up_linear:
            stacked = self._from_te_gate_up_layout(
                stacked,
                self.config.expert_activation,
                self._te_glu_interleave_size,
            )
        if transpose:
            stacked = stacked.transpose(-1, -2)
        return stacked

    def _get_stacked_bias(self, linear: nn.Module) -> torch.Tensor | None:
        if not linear.use_bias:
            return None
        if self._uses_single_grouped_bias(linear):
            stacked = linear.stacked_bias_local()
        else:
            biases = []
            for group_idx in range(linear.num_groups):
                bias = getattr(linear, f"bias{group_idx}")
                biases.append(bias.to_local() if isinstance(bias, DTensor) else bias)
            stacked = torch.stack(biases, dim=0)
        if stacked is not None and linear is self.gate_up_linear:
            stacked = self._from_te_gate_up_layout(
                stacked,
                self.config.expert_activation,
                self._te_glu_interleave_size,
            )
        return stacked

    def _set_stacked_weight(self, linear: nn.Module, stacked: torch.Tensor, transpose: bool = False) -> None:
        if transpose:
            stacked = stacked.transpose(-1, -2)
        if linear is self.gate_up_linear:
            stacked = self._to_te_gate_up_layout(
                stacked,
                self.config.expert_activation,
                self._te_glu_interleave_size,
            )
        with torch.no_grad():
            if self._uses_single_grouped_weight(linear):
                linear.stacked_weight_local().copy_(stacked)
            else:
                for group_idx in range(linear.num_groups):
                    weight = getattr(linear, f"weight{group_idx}")
                    weight = weight.to_local() if isinstance(weight, DTensor) else weight
                    weight.copy_(stacked[group_idx])
        clear_aliases = getattr(linear, "clear_grouped_aliases", None)
        if callable(clear_aliases):
            clear_aliases()
        refresh_weight_cache = getattr(linear, "refresh_mxfp8_weight_cache_if_needed", None)
        if callable(refresh_weight_cache):
            refresh_weight_cache(force=True)

    def _set_stacked_bias(self, linear: nn.Module, stacked: torch.Tensor) -> None:
        if not linear.use_bias or stacked is None:
            return
        if linear is self.gate_up_linear:
            stacked = self._to_te_gate_up_layout(
                stacked,
                self.config.expert_activation,
                self._te_glu_interleave_size,
            )
        with torch.no_grad():
            if self._uses_single_grouped_bias(linear):
                bias = linear.stacked_bias_local()
                if bias is None:
                    raise RuntimeError("TE-ops grouped bias owner is missing")
                bias.copy_(stacked)
            else:
                for group_idx in range(linear.num_groups):
                    bias = getattr(linear, f"bias{group_idx}")
                    bias = bias.to_local() if isinstance(bias, DTensor) else bias
                    bias.copy_(stacked[group_idx])
        clear_aliases = getattr(linear, "clear_grouped_aliases", None)
        if callable(clear_aliases):
            clear_aliases()

    def _router_expert_pad_multiple(self) -> int | None:
        if self._te_ops_uses_padded_capacity:
            return 256
        return None

    def _assert_te_fuser_owner_identity(self) -> None:
        """Ensure TE's lazy fuser captured the current parameter owners."""
        module_groups = getattr(self._te_grouped_mlp, "_module_groups", None)
        if module_groups is None:
            # Lightweight test doubles do not construct TE's OperationFuser.
            return
        current_signature = tuple(
            id(parameter) for linear in (self.gate_up_linear, self.down_linear) for parameter in linear.parameters()
        )
        if self.__dict__.get("_te_ops_fuser_owner_signature") == current_signature:
            return

        fuser_groups = [group for group in module_groups if hasattr(group, "_basic_ops")]
        for linear in (self.gate_up_linear, self.down_linear):
            matches = []
            for group in fuser_groups:
                for op_idx, op in enumerate(group._basic_ops):
                    if op is linear:
                        matches.append(tuple(group._basic_op_params[op_idx]))
            if len(matches) != 1:
                raise RuntimeError(
                    "TE-ops fuser did not capture each GroupedLinear exactly once; "
                    "the expert OperationFuser must be created lazily inside the FSDP unshard window"
                )
            current_owners = tuple(linear.parameters())
            if len(matches[0]) != len(current_owners) or any(
                captured is not current for captured, current in zip(matches[0], current_owners)
            ):
                raise RuntimeError(
                    "TE-ops fuser captured stale expert parameters. Keep expert owners stable "
                    "across FSDP unshard and checkpoint recomputation."
                )
        self._te_ops_fuser_owner_signature = current_signature

    def forward(
        self,
        x: torch.Tensor,
        token_mask: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """Run dispatched tokens through the fused TE-ops grouped MLP."""
        assert not isinstance(x, DTensor), "Input should not be a DTensor"
        assert self.config.n_routed_experts % self.ep_size == 0, (
            f"Number of experts must be divisible by ep_size (ep_size={self.ep_size})"
        )
        # This stays outside ``_te_grouped_mlp``, which is the partial CUDA-graph
        # target. For graph-off cache modes, the optimizer hook only
        # marks the cache dirty and this first subsequent expert forward refreshes
        # it. Fixed-address graph caches refresh synchronously in the hook. Owner
        # version checks additionally catch ordinary in-place edits, while GA and
        # checkpoint recomputation reuse the current generation.
        self.refresh_mxfp8_weight_cache_if_needed()

        weights, indices = _mask_routing_metadata(weights, indices, token_mask)
        permuted_local_hidden_states, tokens_per_expert, permuted_probs = self.token_dispatcher.token_permutation2(
            hidden_states=x,
            num_local_tokens=x.size(0),
            token_probs=weights,
            token_indices=indices,
        )
        if not isinstance(tokens_per_expert, torch.Tensor):
            tokens_per_expert = torch.tensor(
                tokens_per_expert,
                dtype=torch.int64,
                device=permuted_local_hidden_states.device,
            )
        else:
            tokens_per_expert = tokens_per_expert.to(
                device=permuted_local_hidden_states.device,
                dtype=torch.int64,
            )
        permuted_probs = permuted_probs.reshape(-1).to(permuted_local_hidden_states.dtype)

        if permuted_local_hidden_states.shape[0] == 0:
            # Keep dispatch and all stacked owners in the graph so empty EP ranks
            # participate in reverse communication and receive explicit zero grads.
            zero = permuted_local_hidden_states.sum() * 0 + permuted_probs.sum() * 0
            for linear in (self.gate_up_linear, self.down_linear):
                for parameter in linear.parameters():
                    local_parameter = parameter.to_local() if isinstance(parameter, DTensor) else parameter
                    zero = zero + local_parameter.reshape(-1)[0].to(zero.dtype) * 0
            output = permuted_local_hidden_states * 0 + zero
            return self.token_dispatcher.token_unpermutation(output)

        from transformer_engine.pytorch.quantization import FP8GlobalStateManager

        fp8_active = FP8GlobalStateManager.is_fp8_enabled()
        fc2_extra_inputs = (tokens_per_expert, permuted_probs) if self.expert_bias else (tokens_per_expert,)
        if getattr(self, "_te_ops_paged_stash_requested", False):
            from nemo_automodel.components.moe.paged_stash import get_paged_stash_manager

            stash_group = get_paged_stash_manager().group(
                name="te_ops_grouped_mlp",
                max_num_tokens=permuted_local_hidden_states.shape[0],
                num_tokens_tensor=tokens_per_expert.sum(),
                tokens_per_expert=tokens_per_expert,
            )
            permuted_local_hidden_states = stash_group.start(permuted_local_hidden_states)
            with stash_group:
                output = self._te_grouped_mlp(
                    permuted_local_hidden_states,
                    tokens_per_expert,
                    permuted_probs,
                    *fc2_extra_inputs,
                )
            output = stash_group.commit(output)
        else:
            output = self._te_grouped_mlp(
                permuted_local_hidden_states,
                tokens_per_expert,
                permuted_probs,
                *fc2_extra_inputs,
            )
        self._assert_te_fuser_owner_identity()
        self._check_te_ops_fusion(fp8_active)
        return self.token_dispatcher.token_unpermutation(output)

    def _check_te_ops_fusion(self, fp8_active: bool) -> None:
        """Fail loudly if an explicitly requested MXFP8 fusion silently fell back."""
        if self._te_ops_fusion_checked or not fp8_active or not self._te_ops_uses_padded_capacity:
            return

        from transformer_engine.pytorch.quantization import FP8GlobalStateManager

        recipe = FP8GlobalStateManager.get_fp8_recipe()
        fusion_requested = (
            recipe is not None and recipe.mxfp8() and int(os.environ.get("NVTE_CUTEDSL_FUSED_GROUPED_MLP", "0")) > 0
        )
        if not fusion_requested:
            return

        self._te_ops_fusion_checked = True
        module_groups = getattr(self._te_grouped_mlp, "_module_groups", None) or []
        forward_ops = getattr(module_groups[0], "_forward_ops", []) if module_groups else []
        backward_ops = getattr(module_groups[0], "_backward_ops", []) if module_groups else []
        forward_fusion_active = any(
            "GroupedMLP_CuTeGEMM" in type(op_and_indices[0]).__name__ for op_and_indices in forward_ops
        )
        backward_fusion_active = any(
            "GroupedMLP_CuTeGEMM" in type(op_and_indices[0]).__name__ for op_and_indices in backward_ops
        )
        if not forward_fusion_active or not backward_fusion_active:
            raise RuntimeError(
                "TE CuTe DSL grouped-MLP forward/backward fusion was requested but did not fully activate. "
                "Check Transformer Engine >=2.16.1, nvidia-cudnn-frontend >=1.23, "
                "SM100, MXFP8 autocast, the selected scaled activation, and 32-wide GLU interleaving where needed."
            )
        if not dist.is_initialized() or dist.get_rank() == 0:
            logger.warning("TE CuTe DSL MXFP8 grouped-MLP forward/backward fusion is active.")

    def init_weights(self, buffer_device: torch.device, init_std: float = 0.02) -> None:
        """Initialize canonical expert weights without replacing parameter owners."""
        del buffer_device
        with torch.no_grad():
            for linear in (self.gate_up_linear, self.down_linear):
                if self._uses_single_grouped_weight(linear):
                    weights = (linear._parameters["_stacked_weight"],)
                else:
                    weights = tuple(getattr(linear, f"weight{idx}") for idx in range(linear.num_groups))
                for weight in weights:
                    weight = weight.to_local() if isinstance(weight, DTensor) else weight
                    weight.normal_(mean=0.0, std=init_std)

                if linear.use_bias:
                    if self._uses_single_grouped_bias(linear):
                        biases = (linear._parameters["_stacked_bias"],)
                    else:
                        biases = tuple(getattr(linear, f"bias{idx}") for idx in range(linear.num_groups))
                    for bias in biases:
                        bias = bias.to_local() if isinstance(bias, DTensor) else bias
                        bias.zero_()
                clear_aliases = getattr(linear, "clear_grouped_aliases", None)
                if callable(clear_aliases):
                    clear_aliases()
        self.refresh_mxfp8_weight_cache_if_needed(force=True)


def _collect_te_ops_mxfp8_weight_cache_linears(
    model_parts: Any,
) -> dict[int, tuple[nn.Module, nn.Parameter]]:
    """Collect each enabled cached projection and its registered owner once."""
    roots = (model_parts,) if isinstance(model_parts, nn.Module) else tuple(model_parts)
    cache_linears: dict[int, tuple[nn.Module, nn.Parameter]] = {}
    seen_experts: set[int] = set()
    for root in roots:
        modules = root.modules() if hasattr(root, "modules") else ()
        for module in modules:
            if not isinstance(module, GroupedExpertsTeOps) or id(module) in seen_experts:
                continue
            seen_experts.add(id(module))
            if not getattr(module, "_te_ops_mxfp8_weight_cache_enabled", False):
                continue
            for linear in (module.gate_up_linear, module.down_linear):
                owner = linear._parameters.get("_stacked_weight")
                if owner is not None:
                    cache_linears[id(linear)] = (linear, owner)
    return cache_linears


class TeOpsMXFP8WeightCacheRefreshTarget:
    """Fixed-address cached projections refreshed after their owning optimizer."""

    def __init__(
        self,
        linears: tuple[nn.Module, ...],
        owners: tuple[nn.Parameter, ...],
        optimizer: torch.optim.Optimizer,
    ) -> None:
        if not linears or len(linears) != len(owners):
            raise ValueError("MXFP8 cache-refresh target requires matching non-empty linears and owners")
        self._linears = linears
        self._owners = owners
        self.optimizer = optimizer
        self._managed_owner_ids = frozenset(id(owner) for owner in owners)

    @property
    def managed_owner_ids(self) -> frozenset[int]:
        """Return owner identities excluded from ordinary optimizer hooks."""
        return self._managed_owner_ids

    def graph_signature(self) -> tuple[Any, ...]:
        """Return every fixed owner and cache destination identity."""
        return tuple(linear.mxfp8_weight_cache_graph_signature() for linear in self._linears)

    def eager_refresh(self) -> int:
        """Refresh synchronously while the forward/backward graph is not ready."""
        refreshed = 0
        for linear in self._linears:
            if linear.refresh_mxfp8_weight_cache_if_needed(force=True):
                linear.__dict__["_mxfp8_weight_cache_optimizer_refreshes"] += 1
                refreshed += 1
        return refreshed

    def capture_refresh(self) -> None:
        """Launch only fixed-address quantization kernels."""
        for linear in self._linears:
            linear.capture_mxfp8_weight_cache_refresh()

    def mark_replayed(self) -> int:
        """Make every replayed cache generation visible to eager forwards."""
        for linear in self._linears:
            linear.mark_mxfp8_weight_cache_refresh_graph_replayed()
        return len(self._linears)


def build_te_ops_mxfp8_weight_cache_refresh_target(
    model_parts: Any,
    optimizers: Any,
) -> TeOpsMXFP8WeightCacheRefreshTarget | None:
    """Build an optimizer-owned target for fixed-address caches only.

    EP-sharded caches are disabled before this point and therefore produce no
    target. Lazy cache modes remain on their ordinary post-step hooks.
    """
    cache_linears = _collect_te_ops_mxfp8_weight_cache_linears(model_parts)
    fixed_linears = []
    fixed_owners = []
    for linear, owner in cache_linears.values():
        mode = linear.__dict__.get("_mxfp8_weight_cache_mode")
        if mode != _TeOpsMXFP8WeightCache.FIXED_ADDRESS_MODE or not owner.requires_grad:
            continue
        fixed_linears.append(linear)
        fixed_owners.append(owner)
    if not fixed_linears:
        return None

    optimizer_list = (optimizers,) if isinstance(optimizers, torch.optim.Optimizer) else tuple(optimizers)
    optimizer_param_ids = [
        {id(parameter) for group in optimizer.param_groups for parameter in group.get("params", ())}
        for optimizer in optimizer_list
    ]
    owner_optimizer_indices = []
    for owner in fixed_owners:
        owner_indices = [
            optimizer_index
            for optimizer_index, parameter_ids in enumerate(optimizer_param_ids)
            if id(owner) in parameter_ids
        ]
        if len(owner_indices) != 1:
            raise RuntimeError(
                "Fixed-address TE-ops MXFP8 cache owner must belong to exactly one optimizer, "
                f"got {len(owner_indices)} owners for shape {tuple(owner.shape)}"
            )
        owner_optimizer_indices.extend(owner_indices)
    if len(set(owner_optimizer_indices)) != 1:
        raise RuntimeError("One MXFP8 cache-refresh CUDA graph cannot span multiple optimizers")

    optimizer = optimizer_list[owner_optimizer_indices[0]]
    return TeOpsMXFP8WeightCacheRefreshTarget(tuple(fixed_linears), tuple(fixed_owners), optimizer)


def register_te_ops_mxfp8_weight_cache_optimizer_hooks(
    model_parts: Any,
    optimizers: Any,
    *,
    excluded_owner_ids: frozenset[int] = frozenset(),
) -> tuple[Any, ...]:
    """Advance enabled compute caches after each owning optimizer step.

    Native fused optimizers may update ``parameter.data`` without incrementing
    the registered owner's PyTorch version counter. A post-step hook is therefore
    the generation boundary for cache correctness. Graph-off grouped caches are
    invalidated without launching quantization and refresh lazily on the first
    subsequent expert forward. Fixed-address partial-graph caches refresh in the
    hook on the current CUDA stream so replay always observes current contents.
    """

    if not isinstance(excluded_owner_ids, frozenset):
        raise TypeError("excluded_owner_ids must be a frozenset")
    cache_linears = {
        linear_id: (linear, owner)
        for linear_id, (linear, owner) in _collect_te_ops_mxfp8_weight_cache_linears(model_parts).items()
        if id(owner) not in excluded_owner_ids
    }

    if isinstance(optimizers, torch.optim.Optimizer):
        optimizer_list = (optimizers,)
    else:
        optimizer_list = tuple(optimizers)

    optimizer_param_ids_by_optimizer = []
    all_optimizer_param_ids: set[int] = set()
    for optimizer in optimizer_list:
        optimizer_param_ids = {
            id(parameter) for group in getattr(optimizer, "param_groups", ()) for parameter in group.get("params", ())
        }
        optimizer_param_ids_by_optimizer.append(optimizer_param_ids)
        all_optimizer_param_ids.update(optimizer_param_ids)

    unowned_trainable = [
        owner for _, owner in cache_linears.values() if owner.requires_grad and id(owner) not in all_optimizer_param_ids
    ]
    if unowned_trainable:
        shapes = sorted(str(tuple(owner.shape)) for owner in unowned_trainable)
        raise RuntimeError(
            "Enabled TE-ops MXFP8 weight cache has trainable stacked owners missing from every "
            f"optimizer param_group (shapes={shapes})"
        )

    overlapping_owners = [
        owner
        for _, owner in cache_linears.values()
        if sum(id(owner) in optimizer_param_ids for optimizer_param_ids in optimizer_param_ids_by_optimizer) > 1
    ]
    if overlapping_owners:
        shapes = sorted(str(tuple(owner.shape)) for owner in overlapping_owners)
        raise RuntimeError(
            f"Enabled TE-ops MXFP8 weight cache owners appear in multiple optimizer param_groups (shapes={shapes})"
        )

    registration_plan = []
    for optimizer, optimizer_param_ids in zip(optimizer_list, optimizer_param_ids_by_optimizer):
        owned_linears = tuple(linear for linear, owner in cache_linears.values() if id(owner) in optimizer_param_ids)
        if not owned_linears:
            continue
        register_hook = getattr(optimizer, "register_step_post_hook", None)
        if not callable(register_hook):
            raise RuntimeError(
                f"{type(optimizer).__name__} owns enabled TE-ops MXFP8 cache parameters but "
                "does not support register_step_post_hook"
            )
        registration_plan.append((register_hook, owned_linears))

    handles = []
    try:
        for register_hook, owned_linears in registration_plan:

            def advance_after_step(_optimizer, _args, _kwargs, *, linears=owned_linears):
                for linear in linears:
                    mode = linear.__dict__.get("_mxfp8_weight_cache_mode", _TeOpsMXFP8WeightCache.FIXED_ADDRESS_MODE)
                    if mode in _TeOpsMXFP8WeightCache.LAZY_MODES:
                        invalidate = getattr(linear, "invalidate_mxfp8_weight_cache", None)
                        if callable(invalidate) and invalidate():
                            linear.__dict__["_mxfp8_weight_cache_optimizer_invalidations"] = (
                                linear.__dict__.get("_mxfp8_weight_cache_optimizer_invalidations", 0) + 1
                            )
                        continue
                    refresh = getattr(linear, "refresh_mxfp8_weight_cache_if_needed", None)
                    if callable(refresh) and refresh(force=True):
                        linear.__dict__["_mxfp8_weight_cache_optimizer_refreshes"] = (
                            linear.__dict__.get("_mxfp8_weight_cache_optimizer_refreshes", 0) + 1
                        )

            handles.append(register_hook(advance_after_step))
    except Exception:
        for handle in handles:
            handle.remove()
        raise

    return tuple(handles)


def collect_te_ops_mxfp8_weight_cache_diagnostics(model_parts: Any) -> dict[str, Any]:
    """Aggregate TE-ops weight-cache counters without emitting per-layer logs."""
    roots = (model_parts,) if isinstance(model_parts, nn.Module) else tuple(model_parts)
    seen_experts: set[int] = set()
    fallback_reasons: dict[str, int] = {}
    cache_ids: list[int] = []
    buffer_identities: list[tuple[int, ...]] = []
    result: dict[str, Any] = {
        "expert_layers": 0,
        "unstacked_layers": 0,
        "requested_layers": 0,
        "enabled_layers": 0,
        "fallback_layers": 0,
        "projection_caches": 0,
        "allocations": 0,
        "refreshes": 0,
        "group_quantize_calls": 0,
        "member_update_calls": 0,
        "buffer_replacements": 0,
        "optimizer_invalidations": 0,
        "optimizer_refreshes": 0,
        "hits": 0,
        "fallbacks": 0,
        "current_caches": 0,
        "dirty_caches": 0,
        "group_quantize_caches": 0,
        "gemm_ready_fixed_caches": 0,
        "fixed_address_caches": 0,
        "gemm_optimized_caches": 0,
        "storage_identity_stable": True,
        "identity_policy_satisfied": True,
    }

    for root in roots:
        modules = root.modules() if hasattr(root, "modules") else ()
        for module in modules:
            if not isinstance(module, GroupedExpertsTeOps) or id(module) in seen_experts:
                continue
            seen_experts.add(id(module))
            diagnostics = module.mxfp8_weight_cache_diagnostics()
            result["expert_layers"] += 1
            result["unstacked_layers"] += int(getattr(module, "_te_ops_unstacked_parameters", False))
            requested = bool(diagnostics["requested"])
            enabled = bool(diagnostics["enabled"])
            result["requested_layers"] += int(requested)
            result["enabled_layers"] += int(enabled)
            fallback_reason = diagnostics["fallback_reason"]
            if requested and fallback_reason:
                result["fallback_layers"] += 1
                fallback_reasons[fallback_reason] = fallback_reasons.get(fallback_reason, 0) + 1

            for projection in (diagnostics["gate_up"], diagnostics["down"]):
                result["allocations"] += int(projection["allocations"])
                result["refreshes"] += int(projection["refreshes"])
                result["group_quantize_calls"] += int(projection["group_quantize_calls"])
                result["member_update_calls"] += int(projection["member_update_calls"])
                result["buffer_replacements"] += int(projection["buffer_replacements"])
                result["optimizer_invalidations"] += int(projection["optimizer_invalidations"])
                result["optimizer_refreshes"] += int(projection["optimizer_refreshes"])
                result["hits"] += int(projection["hits"])
                result["fallbacks"] += int(projection["fallbacks"])
                result["current_caches"] += int(projection["current"])
                result["dirty_caches"] += int(projection["dirty"])
                result["identity_policy_satisfied"] &= bool(projection["identity_policy_satisfied"])
                storage = projection["storage"]
                if storage is None:
                    continue
                result["projection_caches"] += 1
                mode = projection["mode"]
                result["group_quantize_caches"] += int(mode == _TeOpsMXFP8WeightCache.GROUP_QUANTIZE_MODE)
                result["gemm_ready_fixed_caches"] += int(mode == _TeOpsMXFP8WeightCache.GEMM_READY_FIXED_MODE)
                result["fixed_address_caches"] += int(mode == _TeOpsMXFP8WeightCache.FIXED_ADDRESS_MODE)
                result["gemm_optimized_caches"] += int(projection["optimize_for_gemm"])
                result["storage_identity_stable"] &= bool(storage["identity_stable"])
                cache_ids.append(storage["cache_id"])
                buffer_identities.append(
                    (
                        storage["rowwise_data_ptr"],
                        storage["columnwise_data_ptr"],
                        storage["rowwise_scale_ptr"],
                        storage["columnwise_scale_ptr"],
                    )
                )

    result["unique_cache_objects"] = len(set(cache_ids))
    result["unique_buffer_sets"] = len(set(buffer_identities))
    result["fallback_reasons"] = dict(sorted(fallback_reasons.items()))
    return result


def _init_weights(module, buffer_device: torch.device, init_std: float = 0.02):
    def to_local(tensor):
        if isinstance(tensor, DTensor):
            return tensor.to_local()
        else:
            return tensor

    with torch.device(buffer_device):
        if isinstance(module, (GroupedExperts, GroupedExpertsDeepEP)):
            to_local(module.gate_and_up_projs).normal_(mean=0.0, std=init_std)
            to_local(module.down_projs).normal_(mean=0.0, std=init_std)
            if module.expert_bias:
                to_local(module.gate_up_proj_bias).zero_()
                to_local(module.down_proj_bias).zero_()
        elif isinstance(module, GroupedExpertsTE):
            module.init_weights(buffer_device, init_std)
