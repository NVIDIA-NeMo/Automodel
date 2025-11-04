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

from dataclasses import dataclass
from functools import partial
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo_automodel.components.moe.utils import BackendConfig, initialize_linear_module
from nemo_automodel.shared.utils import dtype_from_str

try:
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.tensor import DTensor, Partial, Replicate, Shard
except ImportError:
    print("torch.distributed.tensor is not available. DeepSeek model will not work.")

try:
    from grouped_gemm import ops
except ImportError:
    print("grouped_gemm is not available. Please run:pip install git+https://github.com/fanshiqing/grouped_gemm@v1.1.4")

from nemo_automodel.components.moe.megatron.moe_utils import (
    MoEAuxLossAutoScaler,
    weighted_bias_swiglu_impl,
)
from nemo_automodel.components.moe.megatron.token_dispatcher import MoEConfig as MegatronMoEConfig
from nemo_automodel.components.moe.megatron.token_dispatcher import MoEFlexTokenDispatcher

_shared_experts_stream: Optional[torch.cuda.Stream] = None


@dataclass(kw_only=True)
class MoEConfig:
    n_routed_experts: int
    n_shared_experts: int
    n_activated_experts: int
    n_expert_groups: int
    n_limited_groups: int
    train_gate: bool
    gate_bias_update_factor: float
    aux_loss_coeff: float
    score_func: str
    route_scale: float
    dim: int
    inter_dim: int
    moe_inter_dim: int
    norm_topk_prob: bool
    router_bias: bool = False
    expert_bias: bool = False
    expert_activation: Literal["swiglu", "quick_geglu"] = "swiglu"
    activation_alpha: float = 1.702
    activation_limit: float = 7.0
    softmax_before_topk: bool = False
    dtype: str | torch.dtype = torch.bfloat16
    shared_expert_gate: bool = False
    shared_expert_inter_dim: int | None = None

    def __post_init__(self):
        if isinstance(self.dtype, str):
            self.dtype = dtype_from_str(self.dtype, default=torch.bfloat16)


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) used as a feed-forward layer.

    Attributes:
        gate_proj (nn.Module): Linear layer for input-to-hidden transformation.
        down_proj (nn.Module): Linear layer for hidden-to-output transformation.
        up_proj (nn.Module): Additional linear layer for feature transformation.
    """

    def __init__(self, dim: int, inter_dim: int, backend: str, dtype: torch.dtype = torch.bfloat16):
        """
        Initializes the MLP layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        self.gate_proj = initialize_linear_module(
            linear_impl=backend, in_features=dim, out_features=inter_dim, bias=False, dtype=dtype
        )
        self.down_proj = initialize_linear_module(
            linear_impl=backend, in_features=inter_dim, out_features=dim, bias=False, dtype=dtype
        )
        self.up_proj = initialize_linear_module(
            linear_impl=backend, in_features=dim, out_features=inter_dim, bias=False, dtype=dtype
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MLP layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after MLP computation.
        """
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    def init_weights(self, buffer_device: torch.device, init_std: float = 0.02) -> None:
        init_weights_fn = partial(_init_weights, buffer_device=buffer_device, init_std=init_std)
        self.apply(init_weights_fn)


@torch.compile
def swiglu(x, *, gate_and_up_proj, down_proj, gate_up_proj_bias=None, down_proj_bias=None):
    gate_and_up_out = x @ gate_and_up_proj
    if gate_up_proj_bias is not None:
        gate_and_up_out = gate_and_up_out + gate_up_proj_bias
    gate_out, up_out = torch.chunk(gate_and_up_out, 2, -1)
    inter = F.silu(gate_out) * up_out

    inter = inter @ down_proj
    if down_proj_bias is not None:
        inter = inter + down_proj_bias
    return inter


@torch.compile
def quick_geglu(
    x,
    *,
    gate_and_up_proj,
    down_proj,
    gate_up_proj_bias=None,
    down_proj_bias=None,
    alpha: float = 1.702,
    limit: float | None = 7.0,
):
    gate_and_up_out = x @ gate_and_up_proj
    if gate_up_proj_bias is not None:
        gate_and_up_out = gate_and_up_out + gate_up_proj_bias
    gate_out, up_out = gate_and_up_out[:, ::2], gate_and_up_out[:, 1::2]
    # Clamp the input values
    gate_out = gate_out.clamp(min=None, max=limit)
    up_out = up_out.clamp(min=-limit, max=limit)
    out_glu = gate_out * torch.sigmoid(alpha * gate_out)
    # Note we add an extra bias of 1 to the linear layer
    inter = out_glu * (up_out + 1)
    inter = inter @ down_proj
    if down_proj_bias is not None:
        inter = inter + down_proj_bias
    return inter


def get_expert_activation(config: MoEConfig):
    if config.expert_activation == "swiglu":
        return swiglu
    elif config.expert_activation == "quick_geglu":
        return partial(quick_geglu, alpha=config.activation_alpha, limit=config.activation_limit)
    else:
        raise ValueError(f"Invalid expert activation: {config.expert_activation}")


class GroupedExperts(nn.Module):
    """
    Sparse MoE implementation using all-gather/reduce-scatter primitives.

    Once the experts for a particular token have been identified, this module
    is invoked to compute and average the output of the activated experts.

    Attributes:
        n_routed_experts (int): Total number of experts in the model.
        gate_projs (nn.Parameter): Linear layer for input-to-gate transformation.
        up_projs (nn.Parameter): Linear layer for input-to-hidden transformation.
        down_projs (nn.Parameter): Linear layer for hidden-to-output transformation.
    """

    def __init__(self, config: MoEConfig):
        """
        Initializes the GroupedExperts module.

        Args:
            args (MoEArgs): Model arguments containing the number of routed experts,
                model and intermediate dimension parameters.
        """
        super().__init__()
        self.n_routed_experts = config.n_routed_experts
        self.expert_bias = config.expert_bias
        self.gate_and_up_projs = nn.Parameter(
            torch.empty(config.n_routed_experts, config.dim, config.moe_inter_dim * 2, dtype=config.dtype)
        )
        self.down_projs = nn.Parameter(
            torch.empty(config.n_routed_experts, config.moe_inter_dim, config.dim, dtype=config.dtype)
        )

        if self.expert_bias:
            self.gate_up_proj_bias = nn.Parameter(
                torch.empty(config.n_routed_experts, config.moe_inter_dim * 2, dtype=config.dtype)
            )
            self.down_proj_bias = nn.Parameter(torch.empty(config.n_routed_experts, config.dim, dtype=config.dtype))
        else:
            self.gate_up_proj_bias = None
            self.down_proj_bias = None

        self.expert_activation = get_expert_activation(config)

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

        # Replicate the tensor to all experts. This is sub-optimal but is
        # used by this implementation for correctness.
        if ep_size > 1:
            x = DTensor.from_local(x, device_mesh=ep_mesh, placements=[Shard(0)]).full_tensor()
            weights = DTensor.from_local(weights.float(), device_mesh=ep_mesh, placements=[Shard(0)]).full_tensor()
            indices = DTensor.from_local(indices, device_mesh=ep_mesh, placements=[Shard(0)]).full_tensor()
            token_mask = DTensor.from_local(token_mask, device_mesh=ep_mesh, placements=[Shard(0)]).full_tensor()

        n_local_experts = self.n_routed_experts // ep_size
        experts_start_idx = ep_rank * n_local_experts
        experts_end_idx = experts_start_idx + n_local_experts

        def get_local_proj(proj, expert_id):
            local_proj = proj.to_local() if isinstance(proj, DTensor) else proj
            return local_proj[expert_id - experts_start_idx]

        y = torch.zeros_like(x)

        active_local_experts = 0
        for i in range(experts_start_idx, experts_end_idx):
            indices_mask = torch.logical_and(indices == i, token_mask.unsqueeze(-1))
            idx, top = torch.where(indices_mask)

            if idx.numel() == 0:
                continue
            active_local_experts += 1

            gate_and_up_proj = get_local_proj(self.gate_and_up_projs, i)
            down_proj = get_local_proj(self.down_projs, i)

            gate_up_proj_bias = get_local_proj(self.gate_up_proj_bias, i) if self.expert_bias else None
            down_proj_bias = get_local_proj(self.down_proj_bias, i) if self.expert_bias else None

            idx_b = idx[:, None].expand(-1, x.size(1))
            x_idx = x.gather(dim=0, index=idx_b)

            expert_out = (
                self.expert_activation(
                    x_idx,
                    gate_and_up_proj=gate_and_up_proj,
                    down_proj=down_proj,
                    gate_up_proj_bias=gate_up_proj_bias,
                    down_proj_bias=down_proj_bias,
                )
                * weights[idx, top, None]
            )

            y.scatter_add_(dim=0, index=idx_b, src=expert_out.to(x.dtype))

        if ep_size > 1:
            y = DTensor.from_local(y, device_mesh=ep_mesh, placements=[Partial()])
            y = y.redistribute(placements=[Shard(0)]).to_local()

        return y

    def init_weights(self, buffer_device: torch.device, init_std: float = 0.02) -> None:
        self.apply(partial(_init_weights, buffer_device=buffer_device, init_std=init_std))


@torch.compile
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
    else:
        raise ValueError(f"Invalid expert activation: {config.expert_activation}")


class GroupedExpertsDeepEP(nn.Module):
    """
    Sparse MoE implementation using DeepEP.

    Once the experts for a particular token have been identified, this module
    is invoked to compute and average the output of the activated experts.

    Attributes:
        n_routed_experts (int): Total number of experts in the model.
        gate_and_up_projs part1 / gate_projs (nn.Parameter): Linear layer for input-to-gate transformation.
        gate_and_up_projs part2 / up_projs (nn.Parameter): Linear layer for input-to-hidden transformation.
        down_projs (nn.Parameter): Linear layer for hidden-to-output transformation.
    """

    @staticmethod
    def _apply_bias(value, bias, tokens_per_expert, permuted_probs=None):
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
                    [t + b for t, b in zip(torch.split(value.view(-1, shape[-1]), tokens_per_expert.tolist()), bias)]
                )
                .view(shape)
                .to(value.dtype)
            )

        return output

    def __init__(self, config: MoEConfig):
        """
        Initializes the GroupedExperts module.

        Args:
            args (MoEArgs): Model arguments containing the number of routed experts,
                model and intermediate dimension parameters.
        """
        super().__init__()

        self.config = config
        self.expert_bias = config.expert_bias
        self.gate_and_up_projs = nn.Parameter(
            torch.empty(config.n_routed_experts, config.dim, config.moe_inter_dim * 2)
        )
        self.down_projs = nn.Parameter(torch.empty(config.n_routed_experts, config.moe_inter_dim, config.dim))

        if self.expert_bias:
            self.gate_up_proj_bias = nn.Parameter(torch.empty(config.n_routed_experts, config.moe_inter_dim * 2))
            self.down_proj_bias = nn.Parameter(torch.empty(config.n_routed_experts, config.dim))
        else:
            self.gate_up_proj_bias = None
            self.down_proj_bias = None

        self.expert_activation = get_expert_activation_for_deepep(config)

    def init_token_dispatcher(self, ep_mesh: DeviceMesh):
        self.ep_size = ep_mesh.size()
        self.ep_rank = ep_mesh.get_local_rank()

        # TODO: merge with MoEArgs
        config = MegatronMoEConfig(
            moe_router_topk=self.config.n_activated_experts,
            num_moe_experts=self.config.n_routed_experts,
            moe_permute_fusion=True,
            moe_enable_deepep=True,
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

        if torch.count_nonzero(tokens_per_expert) > 0:
            output1 = ops.gmm(
                permuted_local_hidden_states,
                self.gate_and_up_projs.to_local(),
                tokens_per_expert,
                trans_b=False,
            )

            if self.expert_bias:
                gate_and_up_bias = self.gate_up_proj_bias.to_local()
                output1 = self._apply_bias(output1, gate_and_up_bias, tokens_per_expert)
            else:
                gate_and_up_bias = None

            output1 = self.expert_activation(output1, permuted_probs)
            output2 = ops.gmm(output1, self.down_projs.to_local(), tokens_per_expert, trans_b=False)

            if self.expert_bias:
                down_bias = self.down_proj_bias.to_local()
                output2 = self._apply_bias(output2, down_bias, tokens_per_expert, permuted_probs)
        else:
            output1 = torch.matmul(x[0] * 0, self.gate_and_up_projs.to_local()[0])
            output1_ = self.expert_activation(output1, permuted_probs)
            output2 = torch.matmul(output1_, self.down_projs.to_local()[0])

        y = self.token_dispatcher.token_unpermutation(output2)
        return y

    def init_weights(self, buffer_device: torch.device, init_std: float = 0.02) -> None:
        self.apply(partial(_init_weights, buffer_device=buffer_device, init_std=init_std))


class FakeBalancedGate(nn.Module):
    """
    Load balanced gate implementation, spreads tokens uniformly across all experts.
    The rationale for this class is to do performance experiments to understand
    how the load imbalance with real data is impacting end-to-end performance.
    """

    def __init__(self, config: MoEConfig, skip_first_n_experts: int = 0):
        super().__init__()
        self.n_routed_experts = config.n_routed_experts
        self.n_activated_experts = config.n_activated_experts
        self.skip_first_n_experts = skip_first_n_experts

    def forward(
        self,
        x: torch.Tensor,
        token_mask: torch.Tensor,
        cp_mesh: Optional[DeviceMesh],
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for the gating mechanism.

        Args:
            x (torch.Tensor): Input tensor.
            token_mask (torch.Tensor): Boolean mask indicating valid tokens.
            cp_mesh (Optional[DeviceMesh]): Device mesh for context parallel computation.

        Returns:
            weights (torch.Tensor): Routing weights for the selected experts.
            indices (torch.Tensor): Indices of the selected experts.
            aux_loss (Optional[torch.Tensor]): Auxiliary loss for load balancing.
        """
        del token_mask
        del cp_mesh

        n_exp = self.n_routed_experts
        a_exp = self.n_activated_experts
        weights = torch.ones(x.size(0), a_exp, device=x.device) / a_exp
        available_experts = n_exp - self.skip_first_n_experts
        indices = (
            torch.arange(x.size(0) * a_exp, device=x.device).view(-1, a_exp) % available_experts
        ) + self.skip_first_n_experts

        return weights.type_as(x), indices, None

    def update_bias(self) -> None:
        pass

    def init_weights(self, buffer_device: torch.device, init_std: float = 0.02) -> None:
        self.apply(partial(_init_weights, buffer_device=buffer_device, init_std=init_std))


class Gate(nn.Module):
    """
    Gating mechanism for routing inputs in a mixture-of-experts (MoE) model.

    Attributes:
        dim (int): Dimensionality of input features.
        topk (int): Number of top experts activated for each input.
        n_groups (int): Number of groups for routing.
        topk_groups (int): Number of groups to route inputs to.
        score_func (str): Scoring function ('softmax' or 'sigmoid').
        route_scale (float): Scaling factor for routing weights.
        weight (torch.nn.Parameter): Learnable weights for the gate.
        bias (Optional[torch.nn.Parameter]): Optional bias term for the gate.
    """

    def __init__(self, config: MoEConfig, gate_precision: torch.dtype | None = None):
        """
        Initializes the Gate module.

        Args:
            config (MoEConfig): Model configuration containing gating parameters.
            gate_precision (torch.dtype | None): Precision for gate computations (linear, softmax/sigmoid).
        """
        super().__init__()
        self.dim = config.dim
        self.n_experts = config.n_routed_experts
        self.topk = config.n_activated_experts
        self.softmax_before_topk = config.softmax_before_topk
        self.n_groups = config.n_expert_groups
        self.topk_groups = config.n_limited_groups
        self.score_func = config.score_func
        self.route_scale = config.route_scale
        self.train_gate = config.train_gate
        self.bias_update_factor = config.gate_bias_update_factor
        self.aux_loss_coeff = config.aux_loss_coeff
        self.norm_topk_prob = config.norm_topk_prob
        self.gate_precision = gate_precision

        if self.bias_update_factor > 0:
            assert self.train_gate, "Require train_gate to be set to True to apply the bias update"

        self.weight = nn.Parameter(
            torch.empty(config.n_routed_experts, config.dim, dtype=config.dtype), requires_grad=self.train_gate
        )
        if config.router_bias:
            self.bias = nn.Parameter(
                torch.empty(config.n_routed_experts, dtype=config.dtype), requires_grad=self.train_gate
            )
        else:
            self.bias = None

        if self.bias_update_factor > 0:
            self.register_buffer("e_score_correction_bias", torch.zeros((self.n_experts), dtype=config.dtype))
        else:
            self.e_score_correction_bias = None

        self.e_score_correction_bias_master = None

        # Cumulative expert load is a tensor representing the number of tokens
        # routed to each expert on the current rank, accumulated across gradient
        # accumulation steps.
        self._cumulative_expert_load: Optional[torch.Tensor] = None

    def forward(
        self,
        x: torch.Tensor,
        token_mask: torch.Tensor,
        cp_mesh: Optional[DeviceMesh],
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for the gating mechanism.

        Args:
            x (torch.Tensor): Input tensor.
            token_mask (torch.Tensor): Boolean mask indicating valid tokens.
            cp_mesh (Optional[DeviceMesh]): Device mesh for context parallel computation.

        Returns:
            weights (torch.Tensor): Routing weights for the selected experts.
            indices (torch.Tensor): Indices of the selected experts.
            aux_loss (Optional[torch.Tensor]): Auxiliary loss for load balancing.
        """
        original_dtype = x.dtype

        if self.gate_precision is not None:
            x_compute = x.to(dtype=self.gate_precision)
            weight = self.weight.to(dtype=self.gate_precision)
            bias = self.bias.to(dtype=self.gate_precision) if self.bias is not None else None
        else:
            x_compute = x
            weight = self.weight.to(dtype=x.dtype)
            bias = self.bias.to(dtype=x.dtype) if self.bias is not None else None

        scores = F.linear(x_compute, weight, bias=bias)

        if self.score_func == "softmax":
            if self.softmax_before_topk:
                scores = scores.softmax(dim=-1, dtype=self.gate_precision or torch.float32)
                original_scores = scores
                indices = torch.topk(scores, k=self.topk, dim=-1)[1]
                weights = scores.gather(1, indices)
            else:
                values, indices = torch.topk(scores, k=self.topk, dim=-1)
                weights = values.softmax(dim=1, dtype=self.gate_precision or torch.float32)
                original_scores = scores
        else:
            scores = scores.sigmoid()
            original_scores = scores

            # Add correction bias to balance tokens across gates.
            if self.e_score_correction_bias is not None:
                correction_bias = self.e_score_correction_bias
                scores = scores + correction_bias

            if self.n_groups > 1:
                scores = scores.view(x.size(0), self.n_groups, -1)
                if self.e_score_correction_bias is None:
                    group_scores = scores.amax(dim=-1)
                else:
                    group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)

                indices = group_scores.topk(self.topk_groups, dim=-1)[1]
                mask = torch.zeros_like(scores[..., 0]).scatter_(1, indices, True)
                scores = (scores * mask.unsqueeze(-1)).flatten(1)

            indices = torch.topk(scores, self.topk, dim=-1)[1]
            weights = original_scores.gather(1, indices)

        if self.norm_topk_prob and self.topk > 1:
            denom_w = weights.sum(dim=-1, keepdim=True) + 1e-20
            denom_s = original_scores.sum(dim=-1, keepdim=True) + 1e-20
            weights = weights / denom_w
            original_scores = original_scores / denom_s

        weights = weights * self.route_scale

        if self.gate_precision is not None:
            weights = weights.to(dtype=original_dtype)
            original_scores = original_scores.to(dtype=original_dtype)

        if self.bias_update_factor > 0 or self.aux_loss_coeff > 0:
            expert_load = self._compute_expert_load(indices, token_mask)

        if self.bias_update_factor > 0 and self.training:
            if self._cumulative_expert_load is None:
                self._cumulative_expert_load = expert_load.detach()
            else:
                self._cumulative_expert_load += expert_load.detach()

        aux_loss = None
        if self.aux_loss_coeff > 0 and self.training:
            aux_loss = self._compute_aux_loss(original_scores, expert_load, token_mask, cp_mesh)
            # Scale the aux_loss by the number of tokens.
            # Training scales all gradients by 1/(number of tokens).
            # To correct this scaling, we need to scale the aux_loss by number of tokens here.
            MoEAuxLossAutoScaler.apply(weights, aux_loss * weights.shape[0])

        return weights.type_as(x), indices, aux_loss

    def update_bias(self) -> None:
        """
        Updates the correction bias used in the gate based on the popularity of experts.
        This function is a NoOp if the gate is not trained.

        To avoid routing collapse, and to promote better load balance of experts,
        DeepSeek-V3 uses a correction mechanism to adjust the scores of experts using
        a learned bias parameter. The bias parameter is updated based on the popularity
        of experts, i.e., the number of tokens routed to each expert. If an expert is
        more popular than the average, its bias term is decreased, and vice versa.
        This encourages the model to route tokens to less popular experts, promoting
        better load balance.
        """
        assert self.train_gate and self.bias_update_factor > 0, "Gate bias update is disabled"

        assert self.training, "Gate bias update is only supported during training"
        assert self._cumulative_expert_load is not None, (
            "Score correction bias cannot be updated without the current expert load"
        )

        # 1) Compute the expert load across all DP ranks.
        # Copy the cumulative load into a local variable, and set the stored load to None.
        expert_load = self._cumulative_expert_load
        self._cumulative_expert_load = None

        # Place the expert load on the same device mesh as the score correction
        # bias parameter, and sum across all ranks.
        if isinstance(self.e_score_correction_bias, DTensor):
            expert_load = DTensor.from_local(
                expert_load,
                device_mesh=self.e_score_correction_bias.device_mesh,
                placements=[Partial()] * self.e_score_correction_bias.device_mesh.ndim,
            )
            expert_load = expert_load.full_tensor()

        # 2) Compute the bias update by comparing the expert load to the average expert load.
        expert_load = expert_load.float()
        average_expert_load = expert_load.mean()
        bias_update = torch.sign(average_expert_load - expert_load)

        if isinstance(self.e_score_correction_bias, DTensor):
            # Convert the bias update back to a replicated DTensor with the same device
            # mesh as the score correction bias parameter.
            bias_update = DTensor.from_local(
                bias_update,
                device_mesh=self.e_score_correction_bias.device_mesh,
                placements=[Replicate()] * self.e_score_correction_bias.device_mesh.ndim,
            )

            # The score correction bias parameter could be sharded across FSDP
            # ranks (dim=-1), and/or optionally replicated across DDP ranks (dim=0).
            # Redistribute the bias update with the same placement.
            bias_update = bias_update.redistribute(placements=self.e_score_correction_bias.placements)

        # 3) Update the correction bias using the bias update.
        with torch.no_grad():
            # Create full precision master weights
            if self.e_score_correction_bias_master is None:
                self.e_score_correction_bias_master = self.e_score_correction_bias.clone().detach().float()
            self.e_score_correction_bias_master += bias_update * self.bias_update_factor
            self.e_score_correction_bias.copy_(self.e_score_correction_bias_master)

    def _compute_expert_load(
        self,
        indices: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the load of each expert based on the selected indices.
        Args:
            indices (torch.Tensor): Indices of the selected experts.
                Shape is [num_tokens, num_activated_experts].
            token_mask (torch.Tensor): Boolean mask indicating valid tokens.
                Shape is [num_tokens].

        Returns:
            torch.Tensor: Load of each expert (number of tokens routed to each expert).
                Shape is [num_local_experts].
        """
        # Create a mask for the experts based on the selected indices.
        expert_mask = indices.new_zeros((indices.shape[0], self.n_experts))
        contribution = token_mask.to(dtype=expert_mask.dtype).unsqueeze(-1).expand(-1, indices.shape[1])
        expert_mask.scatter_(dim=1, index=indices, src=contribution)
        return expert_mask.sum(dim=0)

    def _compute_aux_loss(
        self,
        original_scores: torch.Tensor,
        expert_load: torch.Tensor,
        token_mask: torch.Tensor,
        cp_mesh: Optional[DeviceMesh],
    ) -> torch.Tensor:
        """
        Computes the auxiliary loss for load balancing.

        **Warning**: Assumes batch size = 1, if batch size > 1, the aux_loss will
        be computed across multiple sequences.

        Args:
            original_scores (torch.Tensor): Original scores from the gating mechanism.
                Shape is [num_tokens, num_experts].
            expert_load (torch.Tensor): Load of each expert (number of tokens routed to each expert).
                Shape is [num_experts].
            token_mask (torch.Tensor): Boolean mask indicating valid tokens.
                Shape is [num_tokens].
            cp_mesh (Optional[DeviceMesh]): Device mesh for context parallel computation.

        Returns:
            torch.Tensor: Auxiliary loss for load balancing.
                Shape is [].
        """
        context_length = token_mask.sum()
        expert_scores = (original_scores * token_mask.unsqueeze(-1)).sum(dim=0)

        if cp_mesh is not None:
            context_length = DTensor.from_local(
                context_length, device_mesh=cp_mesh, placements=[Partial()]
            ).full_tensor()
            expert_load = DTensor.from_local(expert_load, device_mesh=cp_mesh, placements=[Partial()]).full_tensor()
            expert_scores = DTensor.from_local(expert_scores, device_mesh=cp_mesh, placements=[Partial()]).full_tensor()

        # Compute f_i (fraction of tokens dispatched to each expert).
        # If uniform distribution, expert_load will be topk * num_location / n_experts, and f_i will be 1
        # Maximum value f_i entries happens when expert_load = num_location, the value will be n_experts / topk
        f_i = expert_load * self.n_experts / (self.topk * context_length)  # Normalized fraction, (n_experts)

        # Compute P_i (average routing probability per expert)
        P_i = expert_scores / context_length  # (n_experts)

        loss = torch.sum(f_i * P_i)
        return loss

    def init_weights(self, buffer_device: torch.device, init_std: float = 0.02) -> None:
        self.apply(partial(_init_weights, buffer_device=buffer_device, init_std=init_std))


class MoE(nn.Module):
    """
    Mixture-of-Experts (MoE) module.

    Attributes:
        dim (int): Dimensionality of input features.
        n_routed_experts (int): Total number of experts in the model.
        n_local_experts (int): Number of experts handled locally in distributed systems.
        n_activated_experts (int): Number of experts activated for each input.
        gate (nn.Module): Gating mechanism to route inputs to experts.
        experts (nn.ModuleList): List of expert modules.
        shared_experts (nn.Module): Shared experts applied to all inputs.
    """

    def __init__(self, config: MoEConfig, backend: BackendConfig):
        """
        Initializes the MoE module.

        Args:
            args (MoEArgs): Model arguments containing MoE parameters.
        """
        super().__init__()
        self.backend = backend
        self.dim = config.dim
        self.n_routed_experts = config.n_routed_experts
        self.n_activated_experts = config.n_activated_experts

        if backend.fake_balanced_gate:
            self.gate = FakeBalancedGate(config)
        else:
            self.gate = Gate(config, gate_precision=backend.gate_precision)
        if backend.enable_deepep:
            self.experts = GroupedExpertsDeepEP(config)
        else:
            self.experts = GroupedExperts(config)

        if config.n_shared_experts > 0:
            self.shared_experts = MLP(
                config.dim,
                config.n_shared_experts * (config.shared_expert_inter_dim or config.moe_inter_dim),
                backend.linear,
            )
            if config.shared_expert_gate:
                self.shared_expert_gate = initialize_linear_module(backend.linear, config.dim, 1, False)
            else:
                self.shared_expert_gate = None
        else:
            self.shared_experts = None
            self.shared_expert_gate = None

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        cp_mesh: Optional[DeviceMesh] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for the MoE module.

        Args:
            x (torch.Tensor): Input tensor.
            padding_mask (Optional[torch.Tensor]): Boolean mask indicating padding positions.

        Returns:
            torch.Tensor: Output tensor after expert routing and computation.
            Optional[torch.Tensor]: Auxiliary loss for load balancing (if applicable).
        """
        # Reshape the inputs to 2-D since we are just distributing tokens.
        shape = x.size()
        x = x.view(-1, self.dim)
        if padding_mask is not None:
            token_mask = (~padding_mask).flatten()
        else:
            token_mask = torch.ones(x.size(0), dtype=torch.bool, device=x.device)

        weights, indices, aux_loss = self.gate(x, token_mask, cp_mesh)

        if self.shared_experts is None:
            y = self.experts(x, token_mask, weights, indices)
            return y.view(shape)

        # Execute shared experts in a separate stream to overlap compute with the
        # communication for grouped experts.
        global _shared_experts_stream
        if _shared_experts_stream is None:
            _shared_experts_stream = torch.cuda.Stream()

        _shared_experts_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(_shared_experts_stream):
            z = self.shared_experts(x)
            if self.shared_expert_gate is not None:
                z = torch.nn.functional.sigmoid(self.shared_expert_gate(x)) * z

        y = self.experts(x, token_mask, weights, indices)

        # Wait for the shared experts stream to complete all operations before
        # adding together the outputs of grouped experts and shared experts.
        torch.cuda.current_stream().wait_stream(_shared_experts_stream)

        # Reshape the outputs back to 3-D.
        return (y + z).view(shape)

    def init_weights(self, buffer_device: torch.device, init_std: float = 0.02) -> None:
        init_weights_fn = partial(_init_weights, buffer_device=buffer_device, init_std=init_std)
        self.apply(init_weights_fn)


def _init_weights(module, buffer_device: torch.device, init_std: float = 0.02):
    def to_local(tensor):
        if isinstance(tensor, DTensor):
            return tensor.to_local()
        else:
            return tensor

    with torch.device(buffer_device):
        if isinstance(module, Gate):
            to_local(module.weight).normal_(mean=0.0, std=init_std)
            if module.e_score_correction_bias is not None:
                to_local(module.e_score_correction_bias).zero_()
            if module.bias is not None:
                to_local(module.bias).zero_()
        elif isinstance(module, (GroupedExperts, GroupedExpertsDeepEP)):
            to_local(module.gate_and_up_projs).normal_(mean=0.0, std=init_std)
            to_local(module.down_projs).normal_(mean=0.0, std=init_std)
            if module.expert_bias:
                to_local(module.gate_up_proj_bias).zero_()
                to_local(module.down_proj_bias).zero_()
        elif isinstance(module, MLP):
            to_local(module.gate_proj.weight).normal_(mean=0.0, std=init_std)
            to_local(module.down_proj.weight).normal_(mean=0.0, std=init_std)
            to_local(module.up_proj.weight).normal_(mean=0.0, std=init_std)
