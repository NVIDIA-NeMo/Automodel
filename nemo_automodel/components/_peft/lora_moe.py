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
from typing import Optional

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor, Partial, Shard

from nemo_automodel.components.moe.layers import GroupedExperts, GroupedExpertsDeepEP
from nemo_automodel.shared.utils import dtype_from_str

try:
    from grouped_gemm import ops
except ImportError:
    ops = None


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

        # Initialize LoRA adapters
        self._init_adapter(
            lora_dim=lora_dim,
            alpha=alpha,
            lora_A_init_method=lora_A_init_method,
            lora_dtype=lora_dtype,
        )

    def _init_adapter(self, lora_dim=8, alpha=32, lora_A_init_method="xavier", lora_dtype=None):
        self.lora_dim = lora_dim
        self.scale = alpha / lora_dim

        # Freeze base weights
        self.gate_and_up_projs.requires_grad = False
        self.down_projs.requires_grad = False
        if self.expert_bias:
            self.gate_up_proj_bias.requires_grad = False
            self.down_proj_bias.requires_grad = False

        # Determine dtype
        if isinstance(lora_dtype, str):
            lora_dtype = dtype_from_str(lora_dtype)
        dtype = lora_dtype or self.gate_and_up_projs.dtype
        device = self.gate_and_up_projs.device

        # LoRA weights for gate_proj, up_proj, and down_proj
        # We treat gate_and_up as a single block for LoRA as well to match structure
        # Shape: [n_experts, in_dim, lora_dim] and [n_experts, lora_dim, out_dim]
        
        # gate_and_up: [n_experts, dim, moe_inter_dim * 2]
        self.lora_gate_and_up_A = nn.Parameter(
            torch.empty(self.n_routed_experts, self.config.dim, lora_dim, dtype=dtype, device=device)
        )
        self.lora_gate_and_up_B = nn.Parameter(
            torch.empty(self.n_routed_experts, lora_dim, self.config.moe_inter_dim * 2, dtype=dtype, device=device)
        )

        # down: [n_experts, moe_inter_dim, dim]
        self.lora_down_A = nn.Parameter(
            torch.empty(self.n_routed_experts, self.config.moe_inter_dim, lora_dim, dtype=dtype, device=device)
        )
        self.lora_down_B = nn.Parameter(
            torch.empty(self.n_routed_experts, lora_dim, self.config.dim, dtype=dtype, device=device)
        )

        self._init_lora_weights(lora_A_init_method)

    def _init_lora_weights(self, init_method):
        # Register hooks to debug parameter gradients
        self.lora_gate_and_up_A.register_hook(lambda grad: print(f"[DEBUG] lora_gate_and_up_A gradient norm: {grad.norm().item():.10f}"))
        self.lora_gate_and_up_B.register_hook(lambda grad: print(f"[DEBUG] lora_gate_and_up_B gradient norm: {grad.norm().item():.10f}"))
        self.lora_down_A.register_hook(lambda grad: print(f"[DEBUG] lora_down_A gradient norm: {grad.norm().item():.10f}"))
        self.lora_down_B.register_hook(lambda grad: print(f"[DEBUG] lora_down_B gradient norm: {grad.norm().item():.10f}"))

        if init_method == "xavier":
            nn.init.xavier_normal_(self.lora_gate_and_up_A)
            nn.init.xavier_normal_(self.lora_down_A)
        else:
            nn.init.kaiming_uniform_(self.lora_gate_and_up_A, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_down_A, a=math.sqrt(5))
            
        nn.init.zeros_(self.lora_gate_and_up_B)
        nn.init.zeros_(self.lora_down_B)

    def forward(self, x, token_mask, weights, indices):
        """
        Forward pass for GroupedExpertsLoRA.

        Args:
            x (torch.Tensor): Input tensor of shape [num_tokens, dim].
            token_mask (torch.Tensor): Boolean mask indicating valid tokens.
            weights (torch.Tensor): Routing weights for the selected experts.
            indices (torch.Tensor): Indices of the selected experts.

        Returns:
            torch.Tensor: Output tensor after expert computation with LoRA.
        """
        # Duplicate logic from GroupedExperts.forward but inject LoRA
        # This is necessary because the original forward doesn't expose hooks for the inner expert computation
        
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

        print(f"[DEBUG] GroupedExpertsLoRA.forward: x stats: mean={x.mean().item():.6f}, max={x.max().item():.6f}, requires_grad={x.requires_grad}, grad_fn={x.grad_fn}")
        print(f"[DEBUG] self.lora_gate_and_up_A.requires_grad={self.lora_gate_and_up_A.requires_grad}")

        n_local_experts = self.n_routed_experts // ep_size
        experts_start_idx = ep_rank * n_local_experts
        experts_end_idx = experts_start_idx + n_local_experts

        def get_local_proj(proj, expert_id):
            if isinstance(proj, DTensor):
                return proj.to_local()[expert_id - experts_start_idx]
            else:
                return proj[expert_id]

        # Helper for LoRA computation: x @ A @ B * scale
        def compute_lora(x, lora_A, lora_B, expert_id):
            local_A = get_local_proj(lora_A, expert_id)
            local_B = get_local_proj(lora_B, expert_id)
            if expert_id == experts_start_idx and x.dim() > 1:
                print(f"[DEBUG] self.scale value: {self.scale}")
            return (x @ local_A @ local_B) * self.scale

        y = torch.zeros_like(x)

        for i in range(experts_start_idx, experts_end_idx):
            indices_mask = torch.logical_and(indices == i, token_mask.unsqueeze(-1))
            idx, top = torch.where(indices_mask)

            if idx.numel() == 0:
                continue

            # DEBUG: Trace Gradient Flow (only for first expert of first layer)
            if i == 0 and idx.numel() > 0 and x.requires_grad:
                print(f"[DEBUG] Expert {i} active. x_idx.requires_grad={x.requires_grad}")

            gate_and_up_proj = get_local_proj(self.gate_and_up_projs, i)
            down_proj = get_local_proj(self.down_projs, i)

            gate_up_proj_bias = get_local_proj(self.gate_up_proj_bias, i) if self.expert_bias else None
            down_proj_bias = get_local_proj(self.down_proj_bias, i) if self.expert_bias else None

            idx_b = idx[:, None].expand(-1, x.size(1))
            x_idx = x.gather(dim=0, index=idx_b)

            # --- Modified Expert Computation with LoRA ---
            # 1. Gate + Up Projection
            gate_and_up_out = x_idx @ gate_and_up_proj
            # Add LoRA
            lora_term_gate = compute_lora(x_idx, self.lora_gate_and_up_A, self.lora_gate_and_up_B, i)
            gate_and_up_out = gate_and_up_out + lora_term_gate
            
            if i == experts_start_idx and x_idx.numel() > 0:
                print(f"[DEBUG] Expert {i}: x_idx stats: mean={x_idx.mean().item():.6f}, max={x_idx.max().item():.6f}")
                print(f"[DEBUG] Expert {i}: lora_term_gate stats: mean={lora_term_gate.mean().item():.6f}, max={lora_term_gate.max().item():.6f}, requires_grad={lora_term_gate.requires_grad}, grad_fn={lora_term_gate.grad_fn}")
            
            # Activation logic (duplicated from layers.py swiglu/quick_geglu but adapted)
            # We need to manually apply activation because we modified the projection output
            if gate_up_proj_bias is not None:
                gate_and_up_out = gate_and_up_out + gate_up_proj_bias
            
            # Correct splitting based on activation type
            if self.config.expert_activation == "swiglu":
                gate_out, up_out = torch.chunk(gate_and_up_out, 2, -1)
            else:
                # for quick_geglu
                gate_out, up_out = gate_and_up_out[..., ::2], gate_and_up_out[..., 1::2]
            
            if self.config.expert_activation == "swiglu":
                inter = torch.nn.functional.silu(gate_out) * up_out
            elif self.config.expert_activation == "quick_geglu":
                limit = self.config.activation_limit
                alpha = self.config.activation_alpha
                gate_out = gate_out.clamp(min=None, max=limit)
                up_out = up_out.clamp(min=-limit, max=limit)
                out_glu = gate_out * torch.sigmoid(alpha * gate_out)
                inter = out_glu * (up_out + 1)

            # 2. Down Projection
            expert_out_val = inter @ down_proj
            # Add LoRA
            expert_out_val = expert_out_val + compute_lora(inter, self.lora_down_A, self.lora_down_B, i)

            if down_proj_bias is not None:
                expert_out_val = expert_out_val + down_proj_bias
            
            expert_out = expert_out_val * weights[idx, top, None]
            # ---------------------------------------------

            y = torch.scatter_add(y, dim=0, index=idx_b, src=expert_out.to(x.dtype))

        if ep_size > 1:
            y = DTensor.from_local(y, device_mesh=ep_mesh, placements=[Partial()])
            y = y.redistribute(placements=[Shard(0)]).to_local()
        
        if y.requires_grad:
            def hook(grad):
                print(f"[DEBUG] GroupedExpertsLoRA output y grad stats: mean={grad.mean().item():.6f}, max={grad.max().item():.6f}, norm={grad.norm().item():.6f}")
            y.register_hook(hook)

        print(f"[DEBUG] GroupedExpertsLoRA.forward output y stats: mean={y.mean().item():.6f}, max={y.max().item():.6f}, requires_grad={y.requires_grad}, grad_fn={y.grad_fn}")

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
    def __init__(self, orig_module: GroupedExpertsDeepEP, lora_dim=8, alpha=32, lora_A_init_method="xavier", lora_dtype=None):
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

        # Ensure n_routed_experts is set (it might not be if init_token_dispatcher wasn't called)
        self.n_routed_experts = self.config.n_routed_experts
        # Also need ep_size for forward check, default to 1 if not set
        if not hasattr(self, "ep_size"):
             self.ep_size = getattr(orig_module, "ep_size", 1)
        if not hasattr(self, "token_dispatcher"):
             self.token_dispatcher = getattr(orig_module, "token_dispatcher", None)

        self._init_adapter(
            lora_dim=lora_dim,
            alpha=alpha,
            lora_A_init_method=lora_A_init_method,
            lora_dtype=lora_dtype,
        )

    def _init_adapter(self, lora_dim=8, alpha=32, lora_A_init_method="xavier", lora_dtype=None):
        self.lora_dim = lora_dim
        self.scale = alpha / lora_dim

        self.gate_and_up_projs.requires_grad = False
        self.down_projs.requires_grad = False
        if self.expert_bias:
            self.gate_up_proj_bias.requires_grad = False
            self.down_proj_bias.requires_grad = False

        if isinstance(lora_dtype, str):
            lora_dtype = dtype_from_str(lora_dtype)
        dtype = lora_dtype or self.gate_and_up_projs.dtype
        device = self.gate_and_up_projs.device

        # LoRA weights
        self.lora_gate_and_up_A = nn.Parameter(
            torch.empty(self.config.n_routed_experts, self.config.dim, lora_dim, dtype=dtype, device=device)
        )
        self.lora_gate_and_up_B = nn.Parameter(
            torch.empty(self.config.n_routed_experts, lora_dim, self.config.moe_inter_dim * 2, dtype=dtype, device=device)
        )

        self.lora_down_A = nn.Parameter(
            torch.empty(self.config.n_routed_experts, self.config.moe_inter_dim, lora_dim, dtype=dtype, device=device)
        )
        self.lora_down_B = nn.Parameter(
            torch.empty(self.config.n_routed_experts, lora_dim, self.config.dim, dtype=dtype, device=device)
        )

        self._init_lora_weights(lora_A_init_method)

    def _init_lora_weights(self, init_method):
        # Register hooks for DeepEP version as well
        self.lora_gate_and_up_A.register_hook(lambda grad: print(f"[DEBUG] DeepEP lora_gate_and_up_A gradient norm: {grad.norm().item():.10f}"))
        self.lora_gate_and_up_B.register_hook(lambda grad: print(f"[DEBUG] DeepEP lora_gate_and_up_B gradient norm: {grad.norm().item():.10f}"))
        self.lora_down_A.register_hook(lambda grad: print(f"[DEBUG] DeepEP lora_down_A gradient norm: {grad.norm().item():.10f}"))
        self.lora_down_B.register_hook(lambda grad: print(f"[DEBUG] DeepEP lora_down_B gradient norm: {grad.norm().item():.10f}"))

        if init_method == "xavier":
            nn.init.xavier_normal_(self.lora_gate_and_up_A)
            nn.init.xavier_normal_(self.lora_down_A)
        else:
            nn.init.kaiming_uniform_(self.lora_gate_and_up_A, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_down_A, a=math.sqrt(5))
            
        nn.init.zeros_(self.lora_gate_and_up_B)
        nn.init.zeros_(self.lora_down_B)

    def forward(self, x, token_mask, weights, indices):
        """
        Forward pass for GroupedExpertsDeepEPLoRA.

        Args:
            x (torch.Tensor): Input tensor of shape [num_tokens, dim].
            token_mask (torch.Tensor): Boolean mask indicating valid tokens.
            weights (torch.Tensor): Routing weights for the selected experts.
            indices (torch.Tensor): Indices of the selected experts.

        Returns:
            torch.Tensor: Output tensor after expert computation with LoRA.
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

        def to_local(tensor):
            return tensor.to_local() if hasattr(tensor, "to_local") else tensor

        n_local_experts = self.n_routed_experts // self.ep_size
        experts_start_idx = self.ep_rank * n_local_experts
        experts_end_idx = experts_start_idx + n_local_experts

        def get_local_weights(proj):
            if isinstance(proj, DTensor):
                return proj.to_local()
            elif self.ep_size > 1:
                return proj[experts_start_idx:experts_end_idx]
            else:
                return proj

        if torch.count_nonzero(tokens_per_expert) > 0:
            print(f"[DEBUG] GroupedExpertsDeepEPLoRA.forward: x stats: mean={x.mean().item():.6f}, max={x.max().item():.6f}, requires_grad={x.requires_grad}, grad_fn={x.grad_fn}")
            print(f"[DEBUG] GroupedExpertsDeepEPLoRA.forward: permuted_local_hidden_states stats: mean={permuted_local_hidden_states.mean().item():.6f}, max={permuted_local_hidden_states.max().item():.6f}")
            print(f"[DEBUG] self.lora_gate_and_up_A stats: mean={to_local(self.lora_gate_and_up_A).mean().item():.6f}, max={to_local(self.lora_gate_and_up_A).max().item():.6f}, requires_grad={self.lora_gate_and_up_A.requires_grad}")
            # 1. Gate + Up Projection
            output1 = ops.gmm(
                permuted_local_hidden_states,
                get_local_weights(self.gate_and_up_projs),
                tokens_per_expert,
                trans_b=False,
            )
            
            # Add LoRA: x @ A @ B
            # We can use ops.gmm for LoRA as well if we chain them
            # A: [E, D, R], B: [E, R, H]
            # x: [T, D], stack of blocks [t_i, D] for each expert
            # x @ A -> [T, R] (stack of blocks [t_i, R] for each expert)
            lora_out1_A = ops.gmm(
                permuted_local_hidden_states,
                get_local_weights(self.lora_gate_and_up_A),
                tokens_per_expert,
                trans_b=False
            )
            # [T, R] @ [E_local, R, H] -> [T, H]
            lora_out1 = ops.gmm(
                lora_out1_A,
                get_local_weights(self.lora_gate_and_up_B),
                tokens_per_expert,
                trans_b=False
            )
            output1 = output1 + lora_out1 * self.scale

            print(f"[DEBUG] output1 stats: mean={output1.mean().item():.6f}, max={output1.max().item():.6f}, requires_grad={output1.requires_grad}, grad_fn={output1.grad_fn}")
            print(f"[DEBUG] lora_out1 stats: mean={lora_out1.mean().item():.6f}, max={lora_out1.max().item():.6f}, requires_grad={lora_out1.requires_grad}, grad_fn={lora_out1.grad_fn}")

            if self.expert_bias:
                gate_and_up_bias = get_local_weights(self.gate_up_proj_bias)
                output1 = self._apply_bias(output1, gate_and_up_bias, tokens_per_expert)
            
            output1 = self.expert_activation(output1, permuted_probs)
            
            # 2. Down Projection
            output2 = ops.gmm(output1, get_local_weights(self.down_projs), tokens_per_expert, trans_b=False)
            
            # Add LoRA
            lora_out2_A = ops.gmm(
                output1,
                get_local_weights(self.lora_down_A),
                tokens_per_expert,
                trans_b=False
            )
            lora_out2 = ops.gmm(
                lora_out2_A,
                get_local_weights(self.lora_down_B),
                tokens_per_expert,
                trans_b=False
            )
            output2 = output2 + lora_out2 * self.scale

            if self.expert_bias:
                down_bias = get_local_weights(self.down_proj_bias)
                output2 = self._apply_bias(output2, down_bias, tokens_per_expert, permuted_probs)
        else:
            # Handle empty case for DeepEP
            # Use slicing compatible with ep_size
            local_base_up = to_local(self.gate_and_up_projs)
            output1 = torch.matmul(x[0:1] * 0, local_base_up[0:1])
            
            # Add LoRA dummy computation
            local_lora_A = get_local_weights(self.lora_gate_and_up_A)
            local_lora_B = get_local_weights(self.lora_gate_and_up_B)
            dummy_lora1 = torch.matmul(torch.matmul(x[0:1] * 0, local_lora_A[0:1]), local_lora_B[0:1])
            output1 = output1 + dummy_lora1 * self.scale
            
            output1_ = self.expert_activation(output1, permuted_probs)
            local_base_down = to_local(self.down_projs)
            output2 = torch.matmul(output1_, local_base_down[0:1])
            
            # Add LoRA dummy computation
            local_lora_down_A = get_local_weights(self.lora_down_A)
            local_lora_down_B = get_local_weights(self.lora_down_B)
            dummy_lora2 = torch.matmul(torch.matmul(output1_ * 0, local_lora_down_A[0:1]), local_lora_down_B[0:1])
            output2 = output2 + dummy_lora2 * self.scale

        y = self.token_dispatcher.token_unpermutation(output2)
        return y
