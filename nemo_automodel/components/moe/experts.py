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

from functools import partial
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Partial, Shard

from nemo_automodel.components.moe.state_dict_utils import create_dtensor_from_local

try:
    from grouped_gemm import ops
except ImportError:
    print("grouped_gemm is not available. Please run:pip install git+https://github.com/fanshiqing/grouped_gemm@v1.1.4")

from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.megatron.moe_utils import (
    weighted_bias_swiglu_impl,
)
from nemo_automodel.components.moe.megatron.token_dispatcher import MoEFlexTokenDispatcher, TokenDispatcherConfig

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
    return activation in ("swiglu", "quick_geglu")


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


@torch.compile
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
        self.use_torch_mm = backend is not None and backend.experts == "torch_mm"

        # Allocate projection tensor - size depends on whether activation is gated
        # Gated (SwiGLU, Quick-GEGLU): [n_experts, dim, 2*inter_dim]
        # Non-gated (ReLU²): [n_experts, dim, inter_dim]
        up_proj_dim = config.moe_inter_dim * 2 if self.is_gated else config.moe_inter_dim
        self.gate_and_up_projs = nn.Parameter(
            torch.empty(config.n_routed_experts, config.dim, up_proj_dim, dtype=config.dtype)
        )

        self.down_projs = nn.Parameter(
            torch.empty(config.n_routed_experts, config.moe_inter_dim, config.dim, dtype=config.dtype)
        )

        if self.expert_bias:
            self.gate_up_proj_bias = nn.Parameter(torch.empty(config.n_routed_experts, up_proj_dim, dtype=config.dtype))
            self.down_proj_bias = nn.Parameter(torch.empty(config.n_routed_experts, config.dim, dtype=config.dtype))
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

        gate_and_up_projs = (
            self.gate_and_up_projs.to_local() if isinstance(self.gate_and_up_projs, DTensor) else self.gate_and_up_projs
        )
        down_projs = self.down_projs.to_local() if isinstance(self.down_projs, DTensor) else self.down_projs

        # DTensor all-gather/reduce-scatter for expert parallelism
        if ep_size > 1:
            # grad_placements=[Partial()] ensures backward does reduce-scatter
            # (default Replicate would just slice, losing cross-rank gradient contributions)
            x = DTensor.from_local(x, device_mesh=ep_mesh, placements=[Shard(0)]).full_tensor(
                grad_placements=[Partial()]
            )
            weights = DTensor.from_local(weights.float(), device_mesh=ep_mesh, placements=[Shard(0)]).full_tensor(
                grad_placements=[Partial()]
            )
            indices = DTensor.from_local(indices, device_mesh=ep_mesh, placements=[Shard(0)]).full_tensor()
            token_mask = DTensor.from_local(token_mask, device_mesh=ep_mesh, placements=[Shard(0)]).full_tensor()

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
                n_local_experts,
                experts_start_idx,
                experts_end_idx,
            )

        if ep_size > 1:
            y = DTensor.from_local(y, device_mesh=ep_mesh, placements=[Partial()])
            y = y.redistribute(placements=[Shard(0)]).to_local()

        return y.to(input_dtype)

    def _forward_loop(
        self,
        x,
        weights,
        indices,
        token_mask,
        gate_and_up_projs,
        down_projs,
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
            down_proj_bias = self.down_proj_bias[local_idx] if self.expert_bias else None

            idx_b = idx[:, None].expand(-1, x.size(1))
            x_idx = x.gather(dim=0, index=idx_b)

            gate_and_up_proj = gate_and_up_projs[local_idx]
            gate_up_proj_bias = self.gate_up_proj_bias[local_idx] if self.expert_bias else None

            # Up projection (separate from activation, matching DeepEP pattern)
            gate_and_up_out = x_idx @ gate_and_up_proj
            if gate_up_proj_bias is not None:
                gate_and_up_out = gate_and_up_out + gate_up_proj_bias

            # Weighted activation (routing weight applied BETWEEN up and down projections)
            # Uses WeightedSwiGLUFunction with float32 backward precision
            w = weights[idx, top, None]
            activated = self.expert_activation_grouped(gate_and_up_out, w)

            # Down projection
            expert_out = activated @ down_proj
            if down_proj_bias is not None:
                expert_out = expert_out + down_proj_bias * w

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
                gate_up_proj_bias = (
                    self.gate_up_proj_bias.to_local()
                    if isinstance(self.gate_up_proj_bias, DTensor)
                    else self.gate_up_proj_bias
                )
                down_proj_bias = (
                    self.down_proj_bias.to_local() if isinstance(self.down_proj_bias, DTensor) else self.down_proj_bias
                )

                output1 = torch._grouped_mm(permuted_x, gate_and_up_projs, offs=offs)
                output1 = _apply_bias(output1, gate_up_proj_bias, tokens_per_expert)
                output1 = self.expert_activation_grouped(output1, permuted_probs)
                output2 = torch._grouped_mm(output1, down_projs, offs=offs)
                output2 = _apply_bias(output2, down_proj_bias, tokens_per_expert, permuted_probs)
            else:
                output2 = _torch_mm_experts_fwd(
                    permuted_x,
                    gate_and_up_projs,
                    down_projs,
                    tokens_per_expert,
                    permuted_probs,
                    self.expert_activation_grouped,
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
    gate_out, up_out = x[..., ::2], x[..., 1::2]
    # Clamp the input values
    gate_out = gate_out.clamp(min=None, max=limit)
    up_out = up_out.clamp(min=-limit, max=limit)
    out_glu = gate_out * torch.sigmoid(alpha * gate_out)
    # Note we add an extra bias of 1 to the linear layer
    inter = out_glu * (up_out + linear_offset)
    return (inter * permuted_probs).to(x.dtype)


@torch.compile(fullgraph=True, options={"max_autotune": True})
def relu2_deepep(x, permuted_probs):
    """ReLU² activation for DeepEP: relu(x)^2

    For DeepEP with ReLU², x is the output of the up projection (already computed).
    x already has shape [..., inter_dim] from efficient up_proj.
    """
    inter = F.relu(x).pow(2)
    return (inter * permuted_probs).to(x.dtype)


def get_expert_activation_for_deepep(config: MoEConfig):
    if config.expert_activation == "swiglu":
        return weighted_bias_swiglu_impl
    elif config.expert_activation == "quick_geglu":
        return partial(
            quick_geglu_deepep,
            limit=config.activation_limit,
            alpha=config.activation_alpha,
            linear_offset=1.0,
        )
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
    ):
        """
        Initializes the GroupedExperts module.

        Args:
            config: MoE configuration containing expert parameters.
            backend: Backend configuration. When backend.experts == "torch_mm",
                uses torch._grouped_mm; otherwise uses grouped_gemm.ops.gmm.
            dispatcher_backend: Backend for the flex token dispatcher ("deepep" or "hybridep").
            dispatcher_num_sms: Number of SMs to use for the dispatcher backend.
        """
        super().__init__()

        self.config = config
        self.use_torch_mm = backend is not None and backend.experts == "torch_mm"
        self.expert_bias = config.expert_bias
        self.is_gated = is_gated_activation(config.expert_activation)
        self.dispatcher_backend = dispatcher_backend
        self.dispatcher_num_sms = dispatcher_num_sms

        # Allocate projection tensor - size depends on whether activation is gated
        # Gated (SwiGLU, Quick-GEGLU): [n_experts, dim, 2*inter_dim]
        # Non-gated (ReLU²): [n_experts, dim, inter_dim]
        up_proj_dim = config.moe_inter_dim * 2 if self.is_gated else config.moe_inter_dim
        self.gate_and_up_projs = nn.Parameter(torch.empty(config.n_routed_experts, config.dim, up_proj_dim))

        self.down_projs = nn.Parameter(torch.empty(config.n_routed_experts, config.moe_inter_dim, config.dim))

        if self.expert_bias:
            self.gate_up_proj_bias = nn.Parameter(torch.empty(config.n_routed_experts, up_proj_dim))
            self.down_proj_bias = nn.Parameter(torch.empty(config.n_routed_experts, config.dim))
        else:
            self.gate_up_proj_bias = None
            self.down_proj_bias = None

        self.expert_activation = get_expert_activation_for_deepep(config)

    def init_token_dispatcher(self, ep_mesh: DeviceMesh):
        self.ep_size = ep_mesh.size()
        self.ep_rank = ep_mesh.get_local_rank()

        config = TokenDispatcherConfig(
            moe_router_topk=self.config.n_activated_experts,
            num_moe_experts=self.config.n_routed_experts,
            moe_permute_fusion=True,
            moe_enable_deepep=True,
            moe_flex_dispatcher_backend=self.dispatcher_backend,
            moe_deepep_num_sms=self.dispatcher_num_sms,
            moe_hybridep_num_sms=self.dispatcher_num_sms,
        )

        self.n_routed_experts = self.config.n_routed_experts

        num_local_experts = self.config.n_routed_experts // self.ep_size

        local_expert_indices_offset = self.ep_rank * num_local_experts
        local_expert_indices = [local_expert_indices_offset + i for i in range(num_local_experts)]

        self.token_dispatcher = MoEFlexTokenDispatcher(
            num_local_experts=num_local_experts,
            local_expert_indices=local_expert_indices,
            config=config,
            ep_group=ep_mesh.get_group(),
        )

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

        indices = indices.masked_fill(~token_mask.unsqueeze(-1), -1)
        (permuted_local_hidden_states, tokens_per_expert, permuted_probs) = self.token_dispatcher.token_permutation2(
            hidden_states=x,
            num_local_tokens=x.size(0),
            token_probs=weights,
            token_indices=indices,
        )
        permuted_probs = permuted_probs.unsqueeze(-1)

        gate_and_up_projs = self.gate_and_up_projs.to_local()
        down_projs = self.down_projs.to_local()

        if torch.count_nonzero(tokens_per_expert) > 0:
            if self.use_torch_mm:
                tokens_per_expert_gpu = tokens_per_expert.to(
                    device=permuted_local_hidden_states.device, non_blocking=True
                )

                if self.expert_bias:
                    # torch._grouped_mm does not support bias yet (raises
                    # "RuntimeError: Bias not supported yet" as of PyTorch 2.10).
                    # Apply bias manually after each grouped GEMM via _apply_bias.
                    offs = tokens_per_expert_gpu.cumsum(dim=0).to(torch.int32)
                    output1 = torch._grouped_mm(permuted_local_hidden_states, gate_and_up_projs, offs=offs)
                    gate_up_proj_bias = self.gate_up_proj_bias.to_local()
                    output1 = _apply_bias(output1, gate_up_proj_bias, tokens_per_expert)
                    output1 = self.expert_activation(output1, permuted_probs)
                    output2 = torch._grouped_mm(output1, down_projs, offs=offs)
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
                    gate_up_proj_bias = self.gate_up_proj_bias.to_local()
                    output1 = _apply_bias(output1, gate_up_proj_bias, tokens_per_expert)

                output1 = self.expert_activation(output1, permuted_probs)
                output2 = ops.gmm(output1, down_projs, tokens_per_expert, trans_b=False)

                if self.expert_bias:
                    down_bias = self.down_proj_bias.to_local()
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
    hidden_states, gate_and_up_projs, down_projs, tokens_per_expert, permuted_probs, activation_fn
):
    offs = tokens_per_expert.cumsum(dim=0).to(torch.int32)
    output1 = torch._grouped_mm(hidden_states, gate_and_up_projs, offs=offs)
    output1 = activation_fn(output1, permuted_probs)
    output2 = torch._grouped_mm(output1, down_projs, offs=offs)
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
    ):
        """
        Initialize the GroupedExpertsTEGroupedLinear module.

        Args:
            config: MoE configuration containing expert parameters.
            backend: Backend configuration (reserved for future use).
            dispatcher_backend: Backend for the flex token dispatcher ("deepep" or "hybridep").
            dispatcher_num_sms: Number of SMs to use for the dispatcher backend.
        """
        from transformer_engine.pytorch import GroupedLinear

        from nemo_automodel.components.models.common.utils import _patch_te_modules

        _patch_te_modules()

        super().__init__()

        self.config = config
        self.num_local_experts = config.n_routed_experts
        self.expert_bias = config.expert_bias
        self.dim = config.dim
        self.moe_inter_dim = config.moe_inter_dim
        self.is_gated = is_gated_activation(config.expert_activation)
        self.dispatcher_backend = dispatcher_backend
        self.dispatcher_num_sms = dispatcher_num_sms

        # Gated (SwiGLU, Quick-GEGLU): out_features = moe_inter_dim * 2
        # Non-gated (ReLU²): out_features = moe_inter_dim
        gate_up_out_features = config.moe_inter_dim * 2 if self.is_gated else config.moe_inter_dim

        # Create TE GroupedLinear layers with full expert count on meta device first
        self.gate_up_linear = GroupedLinear(
            num_gemms=config.n_routed_experts,
            in_features=config.dim,
            out_features=gate_up_out_features,
            bias=self.expert_bias,
            params_dtype=config.dtype,
            device="meta",
        )
        # down_linear: [moe_inter_dim] -> [dim]
        self.down_linear = GroupedLinear(
            num_gemms=config.n_routed_experts,
            in_features=config.moe_inter_dim,
            out_features=config.dim,
            bias=self.expert_bias,
            params_dtype=config.dtype,
            device="meta",
        )

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

    def _get_stacked_weight(self, linear: "GroupedLinear", transpose: bool = False) -> torch.Tensor:
        weights = []
        for i in range(linear.num_gemms):
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
        for i in range(linear.num_gemms):
            b = getattr(linear, f"bias{i}")
            if isinstance(b, DTensor):
                b = b.to_local()
            biases.append(b)
        return torch.stack(biases, dim=0)  # [num_experts, out_features]

    def _set_stacked_weight(self, linear: "GroupedLinear", stacked: torch.Tensor, transpose: bool = False):
        if transpose:
            stacked = stacked.transpose(-1, -2)  # [num_experts, out, in]
        for i in range(linear.num_gemms):
            weight_param = getattr(linear, f"weight{i}")
            if isinstance(weight_param, DTensor):
                weight_param = weight_param.to_local()
            weight_param.data.copy_(stacked[i])

    def _set_stacked_bias(self, linear: "GroupedLinear", stacked: torch.Tensor):
        if not linear.use_bias or stacked is None:
            return
        for i in range(linear.num_gemms):
            bias_param = getattr(linear, f"bias{i}")
            if isinstance(bias_param, DTensor):
                bias_param = bias_param.to_local()
            bias_param.data.copy_(stacked[i])

    def _to_ep_dtensor(self, tensor: torch.Tensor) -> torch.Tensor:
        device_mesh = self.moe_mesh or self.ep_mesh
        dtensor = create_dtensor_from_local(tensor, device_mesh, self.ep_rank if device_mesh is not None else None)
        return dtensor

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
        return self._to_ep_dtensor(bias)

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
        return self._to_ep_dtensor(bias)

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
        from transformer_engine.pytorch import GroupedLinear

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

        gate_up_out_features = self.config.moe_inter_dim * 2 if self.is_gated else self.config.moe_inter_dim

        self.gate_up_linear = GroupedLinear(
            num_gemms=self.num_local_experts,
            in_features=self.config.dim,
            out_features=gate_up_out_features,
            bias=self.expert_bias,
            params_dtype=self.config.dtype,
            device="meta",
        )

        # down_linear: [moe_inter_dim] -> [dim]
        self.down_linear = GroupedLinear(
            num_gemms=self.num_local_experts,
            in_features=self.config.moe_inter_dim,
            out_features=self.config.dim,
            bias=self.expert_bias,
            params_dtype=self.config.dtype,
            device="meta",
        )

        token_dispatcher_config = TokenDispatcherConfig(
            moe_router_topk=self.config.n_activated_experts,
            num_moe_experts=self.config.n_routed_experts,
            moe_permute_fusion=True,
            moe_enable_deepep=True,
            moe_flex_dispatcher_backend=self.dispatcher_backend,
            moe_deepep_num_sms=self.dispatcher_num_sms,
            moe_hybridep_num_sms=self.dispatcher_num_sms,
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

        indices = indices.masked_fill(~token_mask.unsqueeze(-1), -1)

        (permuted_local_hidden_states, tokens_per_expert, permuted_probs) = self.token_dispatcher.token_permutation2(
            hidden_states=x,
            num_local_tokens=x.size(0),
            token_probs=weights,
            token_indices=indices,
        )
        permuted_probs = permuted_probs.unsqueeze(-1)

        if isinstance(tokens_per_expert, torch.Tensor):
            m_splits = tokens_per_expert.tolist()
        else:
            m_splits = list(tokens_per_expert)

        from transformer_engine.pytorch.quantization import FP8GlobalStateManager

        fp8_active = FP8GlobalStateManager.is_fp8_enabled()
        actual_m_splits = None
        if fp8_active:
            actual_m_splits = m_splits
            permuted_local_hidden_states, m_splits = self.fp8_padding(permuted_local_hidden_states, m_splits)
            permuted_probs, _ = self.fp8_padding(permuted_probs, actual_m_splits)

        if sum(m_splits) > 0:
            output1 = self.gate_up_linear(permuted_local_hidden_states, m_splits)
            output1 = self.expert_activation(output1, permuted_probs)
            output2 = self.down_linear(output1, m_splits)
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
