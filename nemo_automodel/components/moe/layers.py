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
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo_automodel.components.moe.utils import BackendConfig, initialize_linear_module

try:
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.tensor import DTensor, Partial, Replicate, Shard
except ImportError:
    print("torch.distributed.tensor is not available. DeepSeek model will not work.")

try:
    from grouped_gemm import ops
except ImportError:
    print("grouped_gemm is not available. Please run:pip install git+https://github.com/fanshiqing/grouped_gemm@v1.1.4")

from nemo_automodel.components.moe.megatron.moe_utils import WeightedSwiGLUFunction
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

    def __post_init__(self):
        if self.aux_loss_coeff > 0:
            raise ValueError("aux loss coeff not supported")


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) used as a feed-forward layer.

    Attributes:
        gate_proj (nn.Module): Linear layer for input-to-hidden transformation.
        down_proj (nn.Module): Linear layer for hidden-to-output transformation.
        up_proj (nn.Module): Additional linear layer for feature transformation.
    """

    def __init__(self, dim: int, inter_dim: int, backend: str):
        """
        Initializes the MLP layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        self.gate_proj = initialize_linear_module(
            linear_impl=backend, in_features=dim, out_features=inter_dim, bias=False
        )
        self.down_proj = initialize_linear_module(
            linear_impl=backend, in_features=inter_dim, out_features=dim, bias=False
        )
        self.up_proj = initialize_linear_module(
            linear_impl=backend, in_features=dim, out_features=inter_dim, bias=False
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
        self.gate_projs = nn.Parameter(torch.empty(config.n_routed_experts, config.moe_inter_dim, config.dim))
        self.up_projs = nn.Parameter(torch.empty(config.n_routed_experts, config.moe_inter_dim, config.dim))
        self.down_projs = nn.Parameter(torch.empty(config.n_routed_experts, config.dim, config.moe_inter_dim))

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

        if isinstance(self.gate_projs, DTensor):
            ep_mesh = self.gate_projs.device_mesh
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
            weights = DTensor.from_local(weights, device_mesh=ep_mesh, placements=[Shard(0)]).full_tensor()
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

            gate_proj = get_local_proj(self.gate_projs, i)
            down_proj = get_local_proj(self.down_projs, i)
            up_proj = get_local_proj(self.up_projs, i)

            idx_b = idx[:, None].expand(-1, x.size(1))
            x_idx = x.gather(dim=0, index=idx_b)

            expert_out = swiglu(x_idx, gate_proj, down_proj, up_proj) * weights[idx, top, None]

            y.scatter_add_(dim=0, index=idx_b, src=expert_out)

        if active_local_experts == 0:
            # We need to handle the case where no token selects the experts on this device.
            gate_proj = get_local_proj(self.gate_projs, experts_start_idx)
            down_proj = get_local_proj(self.down_projs, experts_start_idx)
            up_proj = get_local_proj(self.up_projs, experts_start_idx)
            expert_out = swiglu(torch.zeros_like(x[0]), gate_proj, down_proj, up_proj) * weights[0, 0, None]
            y[0] += expert_out

        if ep_size > 1:
            y = DTensor.from_local(y, device_mesh=ep_mesh, placements=[Partial()])
            y = y.redistribute(placements=[Shard(0)]).to_local()

        return y

    def init_weights(self, buffer_device: torch.device, init_std: float = 0.02) -> None:
        self.apply(partial(_init_weights, buffer_device=buffer_device, init_std=init_std))


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

    def __init__(self, config: MoEConfig):
        """
        Initializes the GroupedExperts module.

        Args:
            args (MoEArgs): Model arguments containing the number of routed experts,
                model and intermediate dimension parameters.
        """
        super().__init__()

        self.config = config
        self.gate_and_up_projs = nn.Parameter(
            torch.empty(config.n_routed_experts, config.dim, config.moe_inter_dim * 2)
        )
        self.down_projs = nn.Parameter(torch.empty(config.n_routed_experts, config.moe_inter_dim, config.dim))
        self.config = config

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
            output1_ = WeightedSwiGLUFunction.apply(output1, permuted_probs, False)
            output2 = ops.gmm(output1_, self.down_projs.to_local(), tokens_per_expert, trans_b=False)
        else:
            output1 = torch.matmul(x[0] * 0, self.gate_and_up_projs.to_local()[0])
            output1_ = WeightedSwiGLUFunction.apply(output1, permuted_probs, False)
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

    def __init__(self, config: MoEConfig):
        super().__init__()
        self.n_routed_experts = config.n_routed_experts
        self.n_activated_experts = config.n_activated_experts

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
        indices = torch.arange(x.size(0) * a_exp, device=x.device).view(-1, a_exp) % n_exp

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

    def __init__(self, config: MoEConfig):
        """
        Initializes the Gate module.

        Args:
            args (MoEArgs): Model arguments containing gating parameters.
        """
        super().__init__()
        self.dim = config.dim
        self.n_experts = config.n_routed_experts
        self.topk = config.n_activated_experts
        self.n_groups = config.n_expert_groups
        self.topk_groups = config.n_limited_groups
        self.score_func = config.score_func
        self.route_scale = config.route_scale
        self.train_gate = config.train_gate
        self.bias_update_factor = config.gate_bias_update_factor
        self.aux_loss_coeff = config.aux_loss_coeff
        self.norm_topk_prob = config.norm_topk_prob

        if self.bias_update_factor > 0:
            assert self.train_gate, "Require train_gate to be set to True to apply the bias update"

        self.weight = nn.Parameter(torch.empty(config.n_routed_experts, config.dim), requires_grad=self.train_gate)
        self.e_score_correction_bias = nn.Parameter(torch.empty(config.n_routed_experts), requires_grad=False)
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
        scores = F.linear(x, self.weight)

        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        else:
            scores = scores.sigmoid()
        original_scores = scores

        # Add correction bias to balance tokens across gates.
        if self.e_score_correction_bias is not None:
            scores = scores + self.e_score_correction_bias

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

        if self.score_func == "sigmoid" and self.norm_topk_prob and self.topk > 1:
            # Use out-of-place division to keep autograd versions intact.
            denom_w = weights.sum(dim=-1, keepdim=True) + 1e-20
            denom_s = original_scores.sum(dim=-1, keepdim=True) + 1e-20
            weights = weights / denom_w
            original_scores = original_scores / denom_s
        weights *= self.route_scale

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


@torch.compile
def swiglu(x, gate_proj, down_proj, up_proj):
    inter = F.silu(F.linear(x, gate_proj)) * F.linear(x, up_proj)
    return F.linear(inter, down_proj)


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
            self.gate = Gate(config)
        if backend.enable_deepep:
            self.experts = GroupedExpertsDeepEP(config)
        else:
            self.experts = GroupedExperts(config)
        self.shared_experts = MLP(config.dim, config.n_shared_experts * config.moe_inter_dim, backend.linear)

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

        # Execute shared experts in a separate stream to overlap compute with the
        # communication for grouped experts.
        global _shared_experts_stream
        if _shared_experts_stream is None:
            _shared_experts_stream = torch.cuda.Stream()

        _shared_experts_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(_shared_experts_stream):
            z = self.shared_experts(x)

        y = self.experts(x, token_mask, weights, indices)

        # Wait for the shared experts stream to complete all operations before
        # adding together the outputs of grouped experts and shared experts.
        torch.cuda.current_stream().wait_stream(_shared_experts_stream)

        # Reshape the outputs back to 3-D.
        return (y + z).view(shape), aux_loss

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
            to_local(module.e_score_correction_bias).zero_()
        elif isinstance(module, GroupedExperts):
            to_local(module.gate_projs).normal_(mean=0.0, std=init_std)
            to_local(module.up_projs).normal_(mean=0.0, std=init_std)
            to_local(module.down_projs).normal_(mean=0.0, std=init_std)
        elif isinstance(module, GroupedExpertsDeepEP):
            to_local(module.gate_and_up_projs).normal_(mean=0.0, std=init_std)
            to_local(module.down_projs).normal_(mean=0.0, std=init_std)
        elif isinstance(module, MLP):
            to_local(module.gate_proj.weight).normal_(mean=0.0, std=init_std)
            to_local(module.down_proj.weight).normal_(mean=0.0, std=init_std)
            to_local(module.up_proj.weight).normal_(mean=0.0, std=init_std)
