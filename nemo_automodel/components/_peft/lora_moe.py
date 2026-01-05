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

import math

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor, Partial, Shard

from nemo_automodel.components.moe.layers import GroupedExperts, GroupedExpertsDeepEP
from nemo_automodel.shared.utils import dtype_from_str

try:
    from grouped_gemm import ops
except ImportError:
    ops = None


def swiglu_with_lora(
    x,
    *,
    gate_and_up_proj,
    down_proj,
    lora_gate_and_up_A,
    lora_gate_and_up_B,
    lora_down_A,
    lora_down_B,
    scale,
    gate_up_proj_bias=None,
    down_proj_bias=None,
):
    """SwiGLU expert activation with LoRA injection."""
    # Gate + Up projection with LoRA
    gate_and_up_out = x @ gate_and_up_proj
    gate_and_up_out = gate_and_up_out + (x @ lora_gate_and_up_A @ lora_gate_and_up_B) * scale

    if gate_up_proj_bias is not None:
        gate_and_up_out = gate_and_up_out + gate_up_proj_bias

    gate_out, up_out = torch.chunk(gate_and_up_out, 2, -1)
    inter = torch.nn.functional.silu(gate_out) * up_out

    # Down projection with LoRA
    out = inter @ down_proj
    out = out + (inter @ lora_down_A @ lora_down_B) * scale

    if down_proj_bias is not None:
        out = out + down_proj_bias
    return out


@torch.compile
def quick_geglu_with_lora(
    x,
    *,
    gate_and_up_proj,
    down_proj,
    lora_gate_and_up_A,
    lora_gate_and_up_B,
    lora_down_A,
    lora_down_B,
    scale,
    gate_up_proj_bias=None,
    down_proj_bias=None,
    alpha: float = 1.702,
    limit: float | None = 7.0,
):
    """QuickGEGLU expert activation with LoRA injection."""
    # Gate + Up projection with LoRA
    gate_and_up_out = x @ gate_and_up_proj
    gate_and_up_out = gate_and_up_out + (x @ lora_gate_and_up_A @ lora_gate_and_up_B) * scale

    if gate_up_proj_bias is not None:
        gate_and_up_out = gate_and_up_out + gate_up_proj_bias

    gate_out, up_out = gate_and_up_out[:, ::2], gate_and_up_out[:, 1::2]
    # Clamp the input values
    gate_out = gate_out.clamp(min=None, max=limit)
    up_out = up_out.clamp(min=-limit, max=limit)
    out_glu = gate_out * torch.sigmoid(alpha * gate_out)
    # Note we add an extra bias of 1 to the linear layer
    inter = out_glu * (up_out + 1)

    # Down projection with LoRA
    out = inter @ down_proj
    out = out + (inter @ lora_down_A @ lora_down_B) * scale

    if down_proj_bias is not None:
        out = out + down_proj_bias
    return out


def get_expert_activation_with_lora(config):
    """Get the expert activation function with LoRA support."""
    from functools import partial

    if config.expert_activation == "swiglu":
        return swiglu_with_lora
    elif config.expert_activation == "quick_geglu":
        return partial(quick_geglu_with_lora, alpha=config.activation_alpha, limit=config.activation_limit)
    else:
        raise ValueError(f"Invalid expert activation: {config.expert_activation}")


class GroupedExpertsLoRA(GroupedExperts):
    """
    GroupedExperts + LoRA.

    This class wraps `GroupedExperts` to apply LoRA to the expert weights.

    Attributes:
        lora_dim (int): Rank of the LoRA adapter.
        scale (float): Scaling factor for the LoRA adapter (alpha / dim).
        lora_gate_and_up_A (nn.Parameter): LoRA A matrix for gate and up projections.
        lora_gate_and_up_B (nn.Parameter): LoRA B matrix for gate and up projections.
        lora_down_A (nn.Parameter): LoRA A matrix for down projection.
        lora_down_B (nn.Parameter): LoRA B matrix for down projection.
    """

    def __init__(self, orig_module: GroupedExperts, lora_dim=8, alpha=32, lora_A_init_method="xavier", lora_dtype=None):
        """
        Initializes the GroupedExpertsLoRA module.

        Args:
            orig_module (GroupedExperts): The original module to wrap.
            lora_dim (int): Rank of the LoRA adapter.
            alpha (int): Scaling factor for the LoRA adapter.
            lora_A_init_method (str): Initialization method for LoRA A matrix.
            lora_dtype (torch.dtype): Data type for LoRA weights.
        """
        super().__init__(orig_module.config)

        self.gate_and_up_projs.data.copy_(orig_module.gate_and_up_projs.data)
        self.down_projs.data.copy_(orig_module.down_projs.data)

        if self.expert_bias:
            self.gate_up_proj_bias.data.copy_(orig_module.gate_up_proj_bias.data)
            self.down_proj_bias.data.copy_(orig_module.down_proj_bias.data)

        GroupedExpertsLoRA._init_adapter(
            self,
            lora_dim=lora_dim,
            alpha=alpha,
            lora_A_init_method=lora_A_init_method,
            lora_dtype=lora_dtype,
        )

    @staticmethod
    def _init_adapter(obj, lora_dim=8, alpha=32, lora_A_init_method="xavier", lora_dtype=None):
        obj.lora_dim = lora_dim
        obj.scale = alpha / lora_dim

        # Freeze base weights
        obj.gate_and_up_projs.requires_grad = False
        obj.down_projs.requires_grad = False
        if obj.expert_bias:
            obj.gate_up_proj_bias.requires_grad = False
            obj.down_proj_bias.requires_grad = False

        # Determine dtype
        if isinstance(lora_dtype, str):
            lora_dtype = dtype_from_str(lora_dtype)
        dtype = lora_dtype or obj.gate_and_up_projs.dtype
        device = obj.gate_and_up_projs.device

        # LoRA weights for gate_proj, up_proj, and down_proj
        # We treat gate_and_up as a single block for LoRA as well to match structure
        # Shape: [n_experts, in_dim, lora_dim] and [n_experts, lora_dim, out_dim]

        # gate_and_up: [n_experts, dim, moe_inter_dim * 2]
        obj.lora_gate_and_up_A = nn.Parameter(
            torch.empty(obj.n_routed_experts, obj.config.dim, lora_dim, dtype=dtype, device=device)
        )
        obj.lora_gate_and_up_B = nn.Parameter(
            torch.empty(obj.n_routed_experts, lora_dim, obj.config.moe_inter_dim * 2, dtype=dtype, device=device)
        )

        # down: [n_experts, moe_inter_dim, dim]
        obj.lora_down_A = nn.Parameter(
            torch.empty(obj.n_routed_experts, obj.config.moe_inter_dim, lora_dim, dtype=dtype, device=device)
        )
        obj.lora_down_B = nn.Parameter(
            torch.empty(obj.n_routed_experts, lora_dim, obj.config.dim, dtype=dtype, device=device)
        )

        # Set the expert activation function with LoRA
        obj.expert_activation_with_lora = get_expert_activation_with_lora(obj.config)

        # Initialize LoRA weights
        GroupedExpertsLoRA.init_lora_weights(obj, lora_A_init_method)

    @torch.no_grad
    def init_lora_weights(self, init_method):
        """Initialize LoRA weights.

        IMPORTANT: This method is called by the PEFT framework's `_init_peft_adapters`
        after the model is materialized from meta device to the target device. The method
        name is critical - it serves as a hook for the framework.
        Do not rename or remove this method.

        Args:
            init_method (str): Initialization method ('xavier' or 'kaiming').
        """
        if init_method == "xavier":
            nn.init.xavier_normal_(self.lora_gate_and_up_A)
            nn.init.xavier_normal_(self.lora_down_A)
        else:
            nn.init.kaiming_uniform_(self.lora_gate_and_up_A, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_down_A, a=math.sqrt(5))

        nn.init.zeros_(self.lora_gate_and_up_B)
        nn.init.zeros_(self.lora_down_B)

    def forward(self, x: torch.Tensor, token_mask: torch.Tensor, weights: torch.Tensor, indices: torch.Tensor):
        """Forward pass for GroupedExpertsLoRA with LoRA injection.

        This method duplicates the logic from GroupedExperts.forward but injects LoRA
        computations into the expert processing. This is necessary because the original
        forward doesn't expose hooks for the inner expert computation.

        Args:
            x (torch.Tensor): Input tensor. Shape is [num_tokens, model_dim].
            token_mask (torch.Tensor): Boolean mask indicating valid tokens.
                Shape is [num_tokens].
            weights (torch.Tensor): Routing weights for the selected experts.
                Shape is [num_tokens, num_activated_experts].
            indices (torch.Tensor): Indices of the selected experts.
                Shape is [num_tokens, num_activated_experts].

        Returns:
            torch.Tensor: Output tensor after expert computation with LoRA.
                Shape is [num_tokens, model_dim].
        """

        assert not isinstance(x, DTensor)

        if isinstance(self.gate_and_up_projs, DTensor):
            ep_mesh = self.gate_and_up_projs.device_mesh
            ep_size = ep_mesh.size()
            ep_rank = ep_mesh.get_local_rank()
        else:
            ep_mesh = None
            ep_size = 1
            ep_rank = 0

        if ep_size > 1:
            x = DTensor.from_local(x, device_mesh=ep_mesh, placements=[Shard(0)]).full_tensor()
            weights = DTensor.from_local(weights.float(), device_mesh=ep_mesh, placements=[Shard(0)]).full_tensor()
            indices = DTensor.from_local(indices, device_mesh=ep_mesh, placements=[Shard(0)]).full_tensor()
            token_mask = DTensor.from_local(token_mask, device_mesh=ep_mesh, placements=[Shard(0)]).full_tensor()

        n_local_experts = self.n_routed_experts // ep_size
        experts_start_idx = ep_rank * n_local_experts
        experts_end_idx = experts_start_idx + n_local_experts

        def get_local_proj(proj, expert_id):
            if isinstance(proj, DTensor):
                return proj.to_local()[expert_id - experts_start_idx]
            else:
                return proj[expert_id]

        y = torch.zeros_like(x)

        active_local_experts = 0
        for i in range(experts_start_idx, experts_end_idx):
            indices_mask = torch.logical_and(indices == i, token_mask.unsqueeze(-1))
            idx, top = torch.where(indices_mask)

            if idx.numel() == 0:
                continue
            active_local_experts += 1

            idx_b = idx[:, None].expand(-1, x.size(1))
            x_idx = x.gather(dim=0, index=idx_b)

            expert_out = (
                self.expert_activation_with_lora(
                    x_idx,
                    gate_and_up_proj=get_local_proj(self.gate_and_up_projs, i),
                    down_proj=get_local_proj(self.down_projs, i),
                    lora_gate_and_up_A=get_local_proj(self.lora_gate_and_up_A, i),
                    lora_gate_and_up_B=get_local_proj(self.lora_gate_and_up_B, i),
                    lora_down_A=get_local_proj(self.lora_down_A, i),
                    lora_down_B=get_local_proj(self.lora_down_B, i),
                    scale=self.scale,
                    gate_up_proj_bias=get_local_proj(self.gate_up_proj_bias, i) if self.expert_bias else None,
                    down_proj_bias=get_local_proj(self.down_proj_bias, i) if self.expert_bias else None,
                )
                * weights[idx, top, None]
            )

            y.scatter_add_(dim=0, index=idx_b, src=expert_out.to(x.dtype))

        # Handle the edge case where no tokens are routed to any local experts.
        # This ensures gradient flow through local expert parameters during backprop
        # and proper participation in collective operations (reduce-scatter).
        if active_local_experts == 0:
            expert_out = (
                self.expert_activation_with_lora(
                    torch.zeros_like(x[0]).unsqueeze(0),
                    gate_and_up_proj=get_local_proj(self.gate_and_up_projs, experts_start_idx),
                    down_proj=get_local_proj(self.down_projs, experts_start_idx),
                    lora_gate_and_up_A=get_local_proj(self.lora_gate_and_up_A, experts_start_idx),
                    lora_gate_and_up_B=get_local_proj(self.lora_gate_and_up_B, experts_start_idx),
                    lora_down_A=get_local_proj(self.lora_down_A, experts_start_idx),
                    lora_down_B=get_local_proj(self.lora_down_B, experts_start_idx),
                    scale=self.scale,
                )
                * weights[0, 0, None]
            )
            y[0] += expert_out[0]

        if ep_size > 1:
            y = DTensor.from_local(y, device_mesh=ep_mesh, placements=[Partial()])
            y = y.redistribute(placements=[Shard(0)]).to_local()

        return y


class GroupedExpertsDeepEPLoRA(GroupedExpertsDeepEP):
    """
    GroupedExpertsDeepEP + LoRA.

    This class wraps `GroupedExpertsDeepEP` to apply LoRA to the expert weights using DeepEP kernels.

    Attributes:
        lora_dim (int): Rank of the LoRA adapter.
        scale (float): Scaling factor for the LoRA adapter (alpha / dim).
        lora_gate_and_up_A (nn.Parameter): LoRA A matrix for gate and up projections.
        lora_gate_and_up_B (nn.Parameter): LoRA B matrix for gate and up projections.
        lora_down_A (nn.Parameter): LoRA A matrix for down projection.
        lora_down_B (nn.Parameter): LoRA B matrix for down projection.
    """

    def __init__(
        self, orig_module: GroupedExpertsDeepEP, lora_dim=8, alpha=32, lora_A_init_method="xavier", lora_dtype=None
    ):
        """
        Initializes the GroupedExpertsDeepEPLoRA module.

        Args:
            orig_module (GroupedExpertsDeepEP): The original module to wrap.
            lora_dim (int): Rank of the LoRA adapter.
            alpha (int): Scaling factor for the LoRA adapter.
            lora_A_init_method (str): Initialization method for LoRA A matrix.
            lora_dtype (torch.dtype): Data type for LoRA weights.
        """
        super().__init__(orig_module.config)

        self.gate_and_up_projs.data.copy_(orig_module.gate_and_up_projs.data)
        self.down_projs.data.copy_(orig_module.down_projs.data)

        if self.expert_bias:
            self.gate_up_proj_bias.data.copy_(orig_module.gate_up_proj_bias.data)
            self.down_proj_bias.data.copy_(orig_module.down_proj_bias.data)

        # Copy DeepEP state explicitly from orig_module (no hasattr guards for robustness)
        self.n_routed_experts = getattr(orig_module, "n_routed_experts", self.config.n_routed_experts)
        self.ep_size = getattr(orig_module, "ep_size", 1)
        self.ep_rank = getattr(orig_module, "ep_rank", 0)
        self.token_dispatcher = getattr(orig_module, "token_dispatcher", None)

        GroupedExpertsDeepEPLoRA._init_adapter(
            self,
            lora_dim=lora_dim,
            alpha=alpha,
            lora_A_init_method=lora_A_init_method,
            lora_dtype=lora_dtype,
        )

    @staticmethod
    def _init_adapter(obj, lora_dim=8, alpha=32, lora_A_init_method="xavier", lora_dtype=None):
        obj.lora_dim = lora_dim
        obj.scale = alpha / lora_dim

        obj.gate_and_up_projs.requires_grad = False
        obj.down_projs.requires_grad = False
        if obj.expert_bias:
            obj.gate_up_proj_bias.requires_grad = False
            obj.down_proj_bias.requires_grad = False

        if isinstance(lora_dtype, str):
            lora_dtype = dtype_from_str(lora_dtype)
        dtype = lora_dtype or obj.gate_and_up_projs.dtype
        device = obj.gate_and_up_projs.device

        # LoRA weights
        obj.lora_gate_and_up_A = nn.Parameter(
            torch.empty(obj.config.n_routed_experts, obj.config.dim, lora_dim, dtype=dtype, device=device)
        )
        obj.lora_gate_and_up_B = nn.Parameter(
            torch.empty(obj.config.n_routed_experts, lora_dim, obj.config.moe_inter_dim * 2, dtype=dtype, device=device)
        )

        obj.lora_down_A = nn.Parameter(
            torch.empty(obj.config.n_routed_experts, obj.config.moe_inter_dim, lora_dim, dtype=dtype, device=device)
        )
        obj.lora_down_B = nn.Parameter(
            torch.empty(obj.config.n_routed_experts, lora_dim, obj.config.dim, dtype=dtype, device=device)
        )

        GroupedExpertsDeepEPLoRA.init_lora_weights(obj, lora_A_init_method)

    @torch.no_grad
    def init_lora_weights(self, init_method):
        """Initialize LoRA weights.

        IMPORTANT: This method is called by the PEFT framework's `_init_peft_adapters`
        after the model is materialized from meta device to the target device. The method
        name is critical - it serves as a hook for the framework.
        Do not rename or remove this method.

        Args:
            init_method (str): Initialization method ('xavier' or 'kaiming').
        """
        if init_method == "xavier":
            nn.init.xavier_normal_(self.lora_gate_and_up_A)
            nn.init.xavier_normal_(self.lora_down_A)
        else:
            nn.init.kaiming_uniform_(self.lora_gate_and_up_A, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_down_A, a=math.sqrt(5))

        nn.init.zeros_(self.lora_gate_and_up_B)
        nn.init.zeros_(self.lora_down_B)

    def forward(
        self,
        x: torch.Tensor,
        token_mask: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ):
        """Forward pass for GroupedExpertsDeepEPLoRA with LoRA injection.

        Args:
            x (torch.Tensor): Input tensor. Shape is [num_tokens, model_dim].
            token_mask (torch.Tensor): Boolean mask indicating valid tokens.
                Shape is [num_tokens].
            weights (torch.Tensor): Routing weights for the selected experts.
                Shape is [num_tokens, num_activated_experts].
            indices (torch.Tensor): Indices of the selected experts.
                Shape is [num_tokens, num_activated_experts].

        Returns:
            torch.Tensor: Output tensor after expert computation with LoRA.
                Shape is [num_tokens, model_dim].
        """
        # Duplicated from GroupedExpertsDeepEP.forward with LoRA injection
        assert not isinstance(x, DTensor)
        assert self.n_routed_experts % self.ep_size == 0

        indices = indices.masked_fill(~token_mask.unsqueeze(-1), -1)

        (permuted_local_hidden_states, tokens_per_expert, permuted_probs) = self.token_dispatcher.token_permutation2(
            hidden_states=x,
            num_local_tokens=x.size(0),
            token_probs=weights,
            token_indices=indices,
        )
        permuted_probs = permuted_probs.unsqueeze(-1)

        # Helper to get local weights. In production, all weights are DTensor after parallelization.
        # The isinstance check is for unit tests where parallelizer is mocked and weights remain Parameters.
        def to_local(proj):
            return proj.to_local() if isinstance(proj, DTensor) else proj

        if torch.count_nonzero(tokens_per_expert) > 0:
            # 1. Gate + Up Projection
            output1 = ops.gmm(
                permuted_local_hidden_states,
                to_local(self.gate_and_up_projs),
                tokens_per_expert,
                trans_b=False,
            )

            # Add LoRA
            lora_out1_A = ops.gmm(
                permuted_local_hidden_states,
                to_local(self.lora_gate_and_up_A),
                tokens_per_expert,
                trans_b=False,
            )
            # [T, R] @ [E_local, R, H] -> [T, H]
            lora_out1 = ops.gmm(
                lora_out1_A, to_local(self.lora_gate_and_up_B), tokens_per_expert, trans_b=False
            )

            output1 = output1 + lora_out1 * self.scale

            if self.expert_bias:
                gate_and_up_bias = to_local(self.gate_up_proj_bias)
                output1 = self._apply_bias(output1, gate_and_up_bias, tokens_per_expert)

            output1 = self.expert_activation(output1, permuted_probs)

            # 2. Down Projection
            output2 = ops.gmm(output1, to_local(self.down_projs), tokens_per_expert, trans_b=False)

            # Add LoRA
            lora_out2_A = ops.gmm(output1, to_local(self.lora_down_A), tokens_per_expert, trans_b=False)
            lora_out2 = ops.gmm(lora_out2_A, to_local(self.lora_down_B), tokens_per_expert, trans_b=False)
            output2 = output2 + lora_out2 * self.scale

            if self.expert_bias:
                down_bias = to_local(self.down_proj_bias)
                output2 = self._apply_bias(output2, down_bias, tokens_per_expert, permuted_probs)
        else:
            # Handle empty case for DeepEP - use [0] indexing to match base shapes exactly
            W1 = to_local(self.gate_and_up_projs)[0]  # [dim, 2*inter]
            W2 = to_local(self.down_projs)[0]  # [inter, dim]
            A1 = to_local(self.lora_gate_and_up_A)[0]  # [dim, r]
            B1 = to_local(self.lora_gate_and_up_B)[0]  # [r, 2*inter]
            A2 = to_local(self.lora_down_A)[0]  # [inter, r]
            B2 = to_local(self.lora_down_B)[0]  # [r, dim]

            dummy_x = x[0] * 0  # [dim]

            output1 = torch.matmul(dummy_x, W1)
            output1 = output1 + torch.matmul(torch.matmul(dummy_x, A1), B1) * self.scale

            output1_ = self.expert_activation(output1, permuted_probs)

            output2 = torch.matmul(output1_, W2)
            output2 = output2 + torch.matmul(torch.matmul(output1_ * 0, A2), B2) * self.scale

        y = self.token_dispatcher.token_unpermutation(output2)
        return y
