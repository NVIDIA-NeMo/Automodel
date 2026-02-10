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
import warnings
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Partial, Replicate

from nemo_automodel.components.distributed.init_utils import get_world_size_safe
from nemo_automodel.components.models.common import BackendConfig, initialize_linear_module
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.experts import (
    GroupedExperts,
    GroupedExpertsDeepEP,
    GroupedExpertsTE,
    is_gated_activation,
)
from nemo_automodel.components.moe.experts import (
    _init_weights as _init_expert_weights,
)
from nemo_automodel.components.moe.megatron.moe_utils import (
    MoEAuxLossAutoScaler,
)

_shared_experts_stream: Optional[torch.cuda.Stream] = None


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) used as a feed-forward layer.

    Supports both gated activations (SwiGLU) and simple activations (ReLU²).

    Attributes:
        gate_proj (nn.Module): Linear layer for gate in gated activations (or up_proj for simple).
        down_proj (nn.Module): Linear layer for hidden-to-output transformation.
        up_proj (nn.Module): Additional linear layer for gated activations (None for simple).
    """

    def __init__(
        self,
        dim: int,
        inter_dim: int,
        backend: str,
        dtype: torch.dtype = torch.bfloat16,
        activation: str = "swiglu",
        bias: bool = False,
    ):
        """
        Initializes the MLP layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
            backend (str): Backend for linear layers.
            dtype (torch.dtype): Data type for weights.
            activation (str): Activation function - "swiglu" (default) or "relu2".
            bias (bool): Whether to use bias in linear layers.
        """
        super().__init__()
        if activation not in ("swiglu", "relu2"):
            raise ValueError(f"Unsupported activation: {activation}. Choose 'swiglu' or 'relu2'.")

        self.activation = activation
        self.is_gated = is_gated_activation(activation)

        self.up_proj = initialize_linear_module(
            linear_impl=backend, in_features=dim, out_features=inter_dim, bias=bias, dtype=dtype
        )
        if self.is_gated:
            self.gate_proj = initialize_linear_module(
                linear_impl=backend, in_features=dim, out_features=inter_dim, bias=bias, dtype=dtype
            )
        else:
            self.gate_proj = None

        self.down_proj = initialize_linear_module(
            linear_impl=backend, in_features=inter_dim, out_features=dim, bias=bias, dtype=dtype
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MLP layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after MLP computation.
        """
        if self.is_gated:
            return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        else:
            return self.down_proj(F.relu(self.up_proj(x)).pow(2))

    def init_weights(self, buffer_device: torch.device, init_std: float = 0.02) -> None:
        init_weights_fn = partial(_init_weights, buffer_device=buffer_device, init_std=init_std)
        self.apply(init_weights_fn)


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

    def __init__(
        self,
        config: MoEConfig,
        gate_precision: torch.dtype | None = None,
    ):
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

        # e_score_correction_bias is only created when bias_update_factor > 0 (for training)
        # or when force_e_score_correction_bias is True (for loading HF checkpoints that have it).
        # This flag is useful in cases where we want to load the bias but not update it.
        # Must be float32 when created - small quantization errors in bf16 can cause
        # completely different expert routing.
        if self.bias_update_factor > 0 or config.force_e_score_correction_bias:
            self.register_buffer("e_score_correction_bias", torch.zeros((self.n_experts), dtype=torch.float32))
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
        if backend.dispatcher == "deepep" and get_world_size_safe() == 1:
            warnings.warn(
                "DeepEP dispatcher is enabled in config, but world size is 1. "
                "DeepEP requires multiple GPUs. Falling back to standard GroupedExperts.",
                category=UserWarning,
                stacklevel=2,
            )
            self.experts = GroupedExperts(config)
        elif backend.dispatcher == "deepep":
            # DeepEP dispatcher requires TE or GMM experts (validated in BackendConfig)
            if backend.experts == "gmm":
                self.experts = GroupedExpertsDeepEP(config)
            else:
                # experts == "te"
                self.experts = GroupedExpertsTE(config)
        else:
            # Default to torch experts
            self.experts = GroupedExperts(config)

        if config.n_shared_experts > 0:
            self.shared_experts = MLP(
                config.dim,
                config.n_shared_experts * (config.shared_expert_inter_dim or config.moe_inter_dim),
                backend.linear,
                dtype=config.dtype,
                activation=config.shared_expert_activation,
                bias=config.expert_bias,
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
        elif isinstance(module, (GroupedExperts, GroupedExpertsDeepEP, GroupedExpertsTE)):
            # Delegate expert initialization to experts.py
            _init_expert_weights(module, buffer_device, init_std)
        elif isinstance(module, MLP):
            to_local(module.down_proj.weight).normal_(mean=0.0, std=init_std)
            to_local(module.up_proj.weight).normal_(mean=0.0, std=init_std)
            if module.gate_proj is not None:
                to_local(module.gate_proj.weight).normal_(mean=0.0, std=init_std)
