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
from collections.abc import Callable

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn_f
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

from nemo_automodel.components.moe.experts import (
    GroupedExperts,
    GroupedExpertsDeepEP,
    _apply_bias,
    _apply_router_weight_fp32,
    _permute_tokens_for_grouped_mm,
)
from nemo_automodel.shared.utils import dtype_from_str


def _to_local(proj):
    """Convert DTensor to local tensor, or return as-is."""
    return proj.to_local() if isinstance(proj, DTensor) else proj


def _to_grouped_mm_operand(proj, dtype: torch.dtype):
    """Convert a projection tensor to the dtype/layout expected by grouped MM."""
    return _to_local(proj).to(dtype).contiguous()


def _pad_lora_rank_for_grouped_mm(lora_A: torch.Tensor, lora_B: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad LoRA rank tensors so grouped-mm strides are 16-byte aligned."""
    rank = lora_A.size(-1)
    alignment = max(1, 16 // lora_A.element_size())
    padding = (-rank) % alignment
    if padding == 0:
        return lora_A, lora_B
    return F.pad(lora_A, (0, padding)), F.pad(lora_B, (0, 0, 0, padding))


@torch.no_grad
def _init_stacked_lora_A(lora_A: torch.Tensor, init_method: str) -> None:
    """Initialize a stacked per-expert LoRA A tensor of shape ``[n_experts, in_features, rank]``.

    Every expert's ``[in_features, rank]`` slice is one LoRA A matrix and must be drawn
    like a 2-D ``nn.Linear(in_features, rank)`` weight. Calling ``xavier_normal_`` or
    ``kaiming_uniform_`` on the stacked 3-D tensor would read the fans off the stack
    (``fan_in = in_features * rank``, ``fan_out = n_experts * rank``), which shrinks
    every expert's init by about ``sqrt(rank + n_experts)`` and ties it to the expert
    count. The distributions are therefore resolved from the per-expert fans explicitly:
    xavier -> normal(0, sqrt(2 / (in_features + rank))), kaiming (``a = sqrt(5)``) ->
    uniform(-1 / sqrt(in_features), 1 / sqrt(in_features)).

    Args:
        lora_A: The stacked A tensor to initialize in place.
        init_method: ``"xavier"`` or ``"kaiming"``.
    """
    if lora_A.ndim != 3:
        raise ValueError(f"expected a stacked [n_experts, in_features, rank] tensor, got shape {tuple(lora_A.shape)}")
    in_features, rank = lora_A.shape[1], lora_A.shape[2]
    if init_method == "xavier":
        lora_A.normal_(0.0, math.sqrt(2.0 / (in_features + rank)))
    else:
        bound = 1.0 / math.sqrt(in_features)
        lora_A.uniform_(-bound, bound)


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
        # Build the wrapper without allocating another full set of expert weights.
        with torch.device("meta"):
            super().__init__(orig_module.config)

        self.gate_and_up_projs = orig_module.gate_and_up_projs
        self.down_projs = orig_module.down_projs

        if self.expert_bias:
            self.gate_up_proj_bias = orig_module.gate_up_proj_bias
            self.down_proj_bias = orig_module.down_proj_bias

        # Copy backend setting from original (super().__init__ defaults to False without backend)
        self.use_torch_mm = orig_module.use_torch_mm

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

        up_proj_dim = obj.config.moe_inter_dim * 2 if obj.is_gated else obj.config.moe_inter_dim
        expert_dim = obj.config.expert_dim

        # LoRA weights for gate+up (or just up if non-gated) and down projections
        obj.lora_gate_and_up_A = nn.Parameter(
            torch.empty(obj.n_routed_experts, expert_dim, lora_dim, dtype=dtype, device=device)
        )
        obj.lora_gate_and_up_B = nn.Parameter(
            torch.empty(obj.n_routed_experts, lora_dim, up_proj_dim, dtype=dtype, device=device)
        )

        obj.lora_down_A = nn.Parameter(
            torch.empty(obj.n_routed_experts, obj.config.moe_inter_dim, lora_dim, dtype=dtype, device=device)
        )
        obj.lora_down_B = nn.Parameter(
            torch.empty(obj.n_routed_experts, lora_dim, expert_dim, dtype=dtype, device=device)
        )

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
        _init_stacked_lora_A(self.lora_gate_and_up_A, init_method)
        _init_stacked_lora_A(self.lora_down_A, init_method)

        nn.init.zeros_(self.lora_gate_and_up_B)
        nn.init.zeros_(self.lora_down_B)

    def materialize_effective_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the grouped expert weights with the LoRA updates folded in.

        Returns:
            A pair containing the input-projection tensor of shape
            [experts, expert_dim, fused_intermediate] and the down-projection
            tensor of shape [experts, intermediate, expert_dim].
        """
        gate_and_up_delta = torch.bmm(self.lora_gate_and_up_A, self.lora_gate_and_up_B) * self.scale
        down_delta = torch.bmm(self.lora_down_A, self.lora_down_B) * self.scale
        return self.gate_and_up_projs + gate_and_up_delta, self.down_projs + down_delta

    def forward(
        self,
        x: torch.Tensor,
        token_mask: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        post_expert_output: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Forward pass for GroupedExpertsLoRA with LoRA injection.

        Mirrors GroupedExperts.forward but injects LoRA computations into
        the expert processing at the projection level.

        Args:
            x: Local input activations [local_tokens, expert_dim].
            token_mask: Valid-token flags [local_tokens].
            weights: Selected routing probabilities [local_tokens, topk].
            indices: Global expert IDs [local_tokens, topk].
            post_expert_output: Optional transform applied after each expert's
                down projection and before after-down router weighting.

        Returns:
            Combined local activations [local_tokens, expert_dim], with x's dtype.
        """
        assert not isinstance(x, DTensor)
        input_dtype = x.dtype

        if isinstance(self.gate_and_up_projs, DTensor):
            ep_mesh = self.gate_and_up_projs.device_mesh
            assert ep_mesh is not None
            assert ep_mesh.ndim == 1
            ep_size = ep_mesh.size()
            ep_rank = ep_mesh.get_local_rank()
        else:
            ep_mesh = None
            ep_size = 1
            ep_rank = 0

        assert self.n_routed_experts % ep_size == 0

        compute_dtype = x.dtype
        gate_and_up_projs = _to_grouped_mm_operand(self.gate_and_up_projs, compute_dtype)
        down_projs = _to_grouped_mm_operand(self.down_projs, compute_dtype)
        lora_gate_and_up_A = _to_grouped_mm_operand(self.lora_gate_and_up_A, compute_dtype)
        lora_gate_and_up_B = _to_grouped_mm_operand(self.lora_gate_and_up_B, compute_dtype)
        lora_down_A = _to_grouped_mm_operand(self.lora_down_A, compute_dtype)
        lora_down_B = _to_grouped_mm_operand(self.lora_down_B, compute_dtype)

        local_num_tokens = x.size(0)
        if ep_size > 1:
            ep_group = ep_mesh.get_group()
            x, weights, indices, token_mask, gathered_lens = self._gather_ep_inputs(
                x, token_mask, weights, indices, ep_group=ep_group, ep_size=ep_size
            )

        n_local_experts = self.n_routed_experts // ep_size
        experts_start_idx = ep_rank * n_local_experts
        experts_end_idx = experts_start_idx + n_local_experts

        if self.use_torch_mm:
            lora_gate_and_up_A, lora_gate_and_up_B = _pad_lora_rank_for_grouped_mm(
                lora_gate_and_up_A, lora_gate_and_up_B
            )
            lora_down_A, lora_down_B = _pad_lora_rank_for_grouped_mm(lora_down_A, lora_down_B)
            y = self._forward_grouped_mm(
                x,
                token_mask,
                weights,
                indices,
                gate_and_up_projs,
                down_projs,
                lora_gate_and_up_A,
                lora_gate_and_up_B,
                lora_down_A,
                lora_down_B,
                n_local_experts,
                experts_start_idx,
                post_expert_output,
            )
        else:
            y = self._forward_loop(
                x,
                weights,
                indices,
                token_mask,
                gate_and_up_projs,
                down_projs,
                lora_gate_and_up_A,
                lora_gate_and_up_B,
                lora_down_A,
                lora_down_B,
                n_local_experts,
                experts_start_idx,
                experts_end_idx,
                post_expert_output,
            )

        if ep_size > 1:
            # Ragged-aware combine, mirroring GroupedExperts.forward:
            # redistribute(Shard(0)) would split the gathered output into equal chunks.
            y.add_(x.sum(dtype=torch.float32) * 0.0)  # keep the differentiable gather attached
            y = dist_nn_f.all_reduce(y, op=dist.ReduceOp.SUM, group=ep_group)
            start = sum(gathered_lens[:ep_rank])
            y = y.narrow(0, start, local_num_tokens).contiguous()

        if self.config.apply_router_weight_after_down:
            y = y.sum(dim=1)
        return y.to(input_dtype)

    def _forward_loop(
        self,
        x,
        weights,
        indices,
        token_mask,
        gate_and_up_projs,
        down_projs,
        lora_gate_and_up_A,
        lora_gate_and_up_B,
        lora_down_A,
        lora_down_B,
        n_local_experts,
        experts_start_idx,
        experts_end_idx,
        post_expert_output,
    ):
        """Per-expert loop forward path with LoRA injection."""
        output_shape = (
            (x.shape[0], weights.shape[1], x.shape[1]) if self.config.apply_router_weight_after_down else x.shape
        )
        y = torch.zeros(output_shape, dtype=torch.float32, device=x.device)

        active_local_experts = 0
        for i in range(experts_start_idx, experts_end_idx):
            indices_mask = torch.logical_and(indices == i, token_mask.unsqueeze(-1))
            idx, top = torch.where(indices_mask)

            if idx.numel() == 0:
                continue
            active_local_experts += 1

            local_idx = i - experts_start_idx
            idx_b = idx[:, None].expand(-1, x.size(1))
            x_idx = x.gather(dim=0, index=idx_b)

            # Up projection + LoRA
            gate_and_up_out = x_idx @ gate_and_up_projs[local_idx]
            gate_and_up_out = (
                gate_and_up_out + (x_idx @ lora_gate_and_up_A[local_idx] @ lora_gate_and_up_B[local_idx]) * self.scale
            )

            if self.expert_bias:
                gate_and_up_out = gate_and_up_out + self.gate_up_proj_bias[local_idx]

            w = weights[idx, top, None]
            activation_weight = torch.ones_like(w) if self.config.apply_router_weight_after_down else w
            activated = self.expert_activation_grouped(gate_and_up_out, activation_weight)

            # Down projection + LoRA
            expert_out = activated @ down_projs[local_idx]
            expert_out = expert_out + (activated @ lora_down_A[local_idx] @ lora_down_B[local_idx]) * self.scale

            if self.expert_bias:
                expert_out = expert_out + (
                    self.down_proj_bias[local_idx]
                    if self.config.apply_router_weight_after_down
                    else self.down_proj_bias[local_idx] * w
                )

            if post_expert_output is not None:
                expert_out = post_expert_output(expert_out)

            if self.config.apply_router_weight_after_down:
                expert_out = _apply_router_weight_fp32(expert_out, w, torch.float32)
                slot_ids = idx * weights.shape[1] + top
                slot_ids_b = slot_ids[:, None].expand(-1, x.size(1))
                y.view(-1, x.size(1)).scatter_add_(dim=0, index=slot_ids_b, src=expert_out)
            else:
                y.scatter_add_(dim=0, index=idx_b, src=expert_out.float())

        # Dummy computation for gradient flow when no tokens routed locally
        if active_local_experts == 0:
            dummy_x = torch.zeros_like(x[0]).unsqueeze(0)
            gate_and_up_out = dummy_x @ gate_and_up_projs[0]
            gate_and_up_out = gate_and_up_out + (dummy_x @ lora_gate_and_up_A[0] @ lora_gate_and_up_B[0]) * self.scale
            dummy_weight = weights[0, 0, None].unsqueeze(0)
            activation_weight = (
                torch.ones_like(dummy_weight) if self.config.apply_router_weight_after_down else dummy_weight
            )
            activated = self.expert_activation_grouped(gate_and_up_out, activation_weight)
            expert_out = activated @ down_projs[0]
            expert_out = expert_out + (activated @ lora_down_A[0] @ lora_down_B[0]) * self.scale
            if post_expert_output is not None:
                expert_out = post_expert_output(expert_out)
            if self.config.apply_router_weight_after_down:
                # Apply the routing weight after the down projection, as the token path and
                # GroupedExpertsDeepEPLoRA do. This keeps the weights attached: under EP their
                # all-gather backward is a collective that every rank must enter.
                expert_out = _apply_router_weight_fp32(expert_out, dummy_weight, torch.float32)
                y[0, 0] += expert_out[0]
            else:
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
        lora_gate_and_up_A,
        lora_gate_and_up_B,
        lora_down_A,
        lora_down_B,
        n_local_experts,
        experts_start_idx,
        post_expert_output,
    ):
        """Grouped GEMM forward path with LoRA injection using torch._grouped_mm."""
        sorted_token_ids, sorted_slot_ids, sorted_weights, tokens_per_expert, offs = _permute_tokens_for_grouped_mm(
            indices,
            weights,
            token_mask,
            n_local_experts,
            experts_start_idx,
            return_slot_ids=True,
        )

        output_shape = (
            (x.shape[0], weights.shape[1], x.shape[1]) if self.config.apply_router_weight_after_down else x.shape
        )
        y = torch.zeros(output_shape, dtype=torch.float32, device=x.device)

        if sorted_token_ids.numel() > 0:
            permuted_x = x[sorted_token_ids]
            permuted_probs = sorted_weights.unsqueeze(-1)
            activation_probs = (
                torch.ones_like(permuted_probs) if self.config.apply_router_weight_after_down else permuted_probs
            )

            if self.expert_bias:
                gate_up_proj_bias = _to_local(self.gate_up_proj_bias)
                down_proj_bias = _to_local(self.down_proj_bias)

            # Gate+Up projection + LoRA
            output1 = torch._grouped_mm(permuted_x, gate_and_up_projs, offs=offs)
            lora_out1_A = torch._grouped_mm(permuted_x, lora_gate_and_up_A, offs=offs)
            lora_out1 = torch._grouped_mm(lora_out1_A, lora_gate_and_up_B, offs=offs)
            output1 = output1 + lora_out1 * self.scale

            if self.expert_bias:
                output1 = _apply_bias(output1, gate_up_proj_bias, tokens_per_expert)

            output1 = self.expert_activation_grouped(output1, activation_probs)

            # Down projection + LoRA
            output2 = torch._grouped_mm(output1, down_projs, offs=offs)
            lora_out2_A = torch._grouped_mm(output1, lora_down_A, offs=offs)
            lora_out2 = torch._grouped_mm(lora_out2_A, lora_down_B, offs=offs)
            output2 = output2 + lora_out2 * self.scale

            if self.expert_bias:
                output2 = _apply_bias(
                    output2,
                    down_proj_bias,
                    tokens_per_expert,
                    None if self.config.apply_router_weight_after_down else permuted_probs,
                )

            if post_expert_output is not None:
                output2 = post_expert_output(output2)

            if self.config.apply_router_weight_after_down:
                output2 = _apply_router_weight_fp32(output2, permuted_probs, torch.float32)
                scatter_ids = sorted_slot_ids.unsqueeze(1).expand_as(output2)
                y.view(-1, x.size(1)).scatter_add_(0, scatter_ids, output2)
            else:
                scatter_ids = sorted_token_ids.unsqueeze(1).expand_as(output2)
                y.scatter_add_(0, scatter_ids, output2.float())
        else:
            # Dummy computation for gradient flow
            output1 = torch.matmul(x[0] * 0, gate_and_up_projs[0])
            output1 = (
                output1
                + torch.matmul(torch.matmul(x[0] * 0, lora_gate_and_up_A[0]), lora_gate_and_up_B[0]) * self.scale
            )
            dummy_weight = weights[0, 0, None].unsqueeze(0)
            activation_weight = (
                torch.ones_like(dummy_weight) if self.config.apply_router_weight_after_down else dummy_weight
            )
            output1_ = self.expert_activation_grouped(output1, activation_weight)
            output2 = torch.matmul(output1_, down_projs[0])
            output2 = output2 + torch.matmul(torch.matmul(output1_ * 0, lora_down_A[0]), lora_down_B[0]) * self.scale
            if post_expert_output is not None:
                output2 = post_expert_output(output2)
            if self.config.apply_router_weight_after_down:
                # Same after-down weighting as the token path; see _forward_loop.
                output2 = _apply_router_weight_fp32(output2, dummy_weight, torch.float32)
                y[0, 0] += output2[0]
            else:
                y[0] += output2[0]

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
        # Preserve the existing base storage, including its dtype and device.
        with torch.device("meta"):
            super().__init__(
                orig_module.config,
                dispatcher_backend=orig_module.dispatcher_backend,
                dispatcher_num_sms=orig_module.dispatcher_num_sms,
                dispatcher_share_token_dispatcher=orig_module.dispatcher_share_token_dispatcher,
                dispatcher_async_dispatch=orig_module.dispatcher_async_dispatch,
            )
        self.dispatcher_hybridep_permute_fusion = orig_module.dispatcher_hybridep_permute_fusion
        self.dispatcher_hybridep_compact_routing = orig_module.dispatcher_hybridep_compact_routing
        self.dispatcher_hybridep_num_sms_preprocessing = orig_module.dispatcher_hybridep_num_sms_preprocessing
        self.dispatcher_hybridep_num_blocks_permute = orig_module.dispatcher_hybridep_num_blocks_permute
        self.dispatcher_hybridep_num_blocks_unpermute = orig_module.dispatcher_hybridep_num_blocks_unpermute
        # Sync-free dispatch knobs (BackendConfig.dispatcher_capacity_factor / dispatcher_equal_token_counts) travel
        # with the other HybridEP settings: the shared dispatcher is built from the first (wrapped) layer.
        self.dispatcher_capacity_factor = orig_module.dispatcher_capacity_factor
        self.dispatcher_equal_token_counts = orig_module.dispatcher_equal_token_counts
        self.dispatch_rows_never_empty = orig_module.dispatch_rows_never_empty
        # The combine cache (BackendConfig.dispatcher_recompute_cache) is read by the shared dispatcher's config as
        # well: the replacement must carry it, or the requested mode silently reverts to "none" (review of #4395).
        self.dispatcher_recompute_cache = orig_module.dispatcher_recompute_cache

        self.gate_and_up_projs = orig_module.gate_and_up_projs
        self.down_projs = orig_module.down_projs

        if self.expert_bias:
            self.gate_up_proj_bias = orig_module.gate_up_proj_bias
            self.down_proj_bias = orig_module.down_proj_bias

        # Copy DeepEP state from orig_module (set by init_token_dispatcher, not __init__)
        self.n_routed_experts = getattr(orig_module, "n_routed_experts", self.config.n_routed_experts)
        self.ep_size = getattr(orig_module, "ep_size", 1)
        self.ep_rank = getattr(orig_module, "ep_rank", 0)
        self.token_dispatcher = getattr(orig_module, "token_dispatcher", None)
        self.use_mxfp8 = getattr(orig_module, "use_mxfp8", False)

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

        up_proj_dim = obj.config.moe_inter_dim * 2 if obj.is_gated else obj.config.moe_inter_dim
        expert_dim = obj.config.expert_dim

        # LoRA weights
        obj.lora_gate_and_up_A = nn.Parameter(
            torch.empty(obj.config.n_routed_experts, expert_dim, lora_dim, dtype=dtype, device=device)
        )
        obj.lora_gate_and_up_B = nn.Parameter(
            torch.empty(obj.config.n_routed_experts, lora_dim, up_proj_dim, dtype=dtype, device=device)
        )

        obj.lora_down_A = nn.Parameter(
            torch.empty(obj.config.n_routed_experts, obj.config.moe_inter_dim, lora_dim, dtype=dtype, device=device)
        )
        obj.lora_down_B = nn.Parameter(
            torch.empty(obj.config.n_routed_experts, lora_dim, expert_dim, dtype=dtype, device=device)
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
        _init_stacked_lora_A(self.lora_gate_and_up_A, init_method)
        _init_stacked_lora_A(self.lora_down_A, init_method)

        nn.init.zeros_(self.lora_gate_and_up_B)
        nn.init.zeros_(self.lora_down_B)

    def forward(
        self,
        x: torch.Tensor,
        token_mask: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        post_expert_output: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Apply LoRA while preserving the base expert transform and routing order.

        Args:
            x: Local input activations [local_tokens, expert_dim].
            token_mask: Valid-token flags [local_tokens].
            weights: Selected routing probabilities [local_tokens, topk].
            indices: Global expert IDs [local_tokens, topk].
            post_expert_output: Optional transform from [routed_tokens, expert_dim]
                to the same layout, applied after the down projection and before
                router weighting when apply_router_weight_after_down is enabled.

        Returns:
            Combined local activations [local_tokens, expert_dim], with x's dtype.
        """
        assert not isinstance(x, DTensor)
        assert self.n_routed_experts % self.ep_size == 0

        permuted_local_hidden_states, tokens_per_expert, permuted_probs = self._dispatch_tokens(
            x, token_mask, weights, indices
        )

        activation_probs = (
            torch.ones_like(permuted_probs) if self.config.apply_router_weight_after_down else permuted_probs
        )
        compute_dtype = x.dtype
        gate_and_up_projs = _to_grouped_mm_operand(self.gate_and_up_projs, compute_dtype)
        down_projs = _to_grouped_mm_operand(self.down_projs, compute_dtype)
        lora_gate_and_up_A = _to_grouped_mm_operand(self.lora_gate_and_up_A, compute_dtype)
        lora_gate_and_up_B = _to_grouped_mm_operand(self.lora_gate_and_up_B, compute_dtype)
        lora_down_A = _to_grouped_mm_operand(self.lora_down_A, compute_dtype)
        lora_down_B = _to_grouped_mm_operand(self.lora_down_B, compute_dtype)

        # dispatch_rows_never_empty (static routing or capacity mode): skip the count_nonzero host read, as
        # GroupedExpertsDeepEP.forward does.
        if self.dispatch_rows_never_empty or torch.count_nonzero(tokens_per_expert) > 0:
            lora_gate_and_up_A, lora_gate_and_up_B = _pad_lora_rank_for_grouped_mm(
                lora_gate_and_up_A, lora_gate_and_up_B
            )
            lora_down_A, lora_down_B = _pad_lora_rank_for_grouped_mm(lora_down_A, lora_down_B)
            tokens_per_expert_gpu = tokens_per_expert.to(device=permuted_local_hidden_states.device, non_blocking=True)
            offs = tokens_per_expert_gpu.cumsum(dim=0).to(torch.int32)

            # Gate+Up projection + LoRA
            output1 = torch._grouped_mm(permuted_local_hidden_states, gate_and_up_projs, offs=offs)
            lora_out1_A = torch._grouped_mm(permuted_local_hidden_states, lora_gate_and_up_A, offs=offs)
            lora_out1 = torch._grouped_mm(lora_out1_A, lora_gate_and_up_B, offs=offs)
            output1 = output1 + lora_out1 * self.scale

            if self.expert_bias:
                gate_up_proj_bias = _to_local(self.gate_up_proj_bias)
                output1 = _apply_bias(output1, gate_up_proj_bias, tokens_per_expert)

            output1 = self.expert_activation(output1, activation_probs)

            # Down projection + LoRA
            output2 = torch._grouped_mm(output1, down_projs, offs=offs)
            lora_out2_A = torch._grouped_mm(output1, lora_down_A, offs=offs)
            lora_out2 = torch._grouped_mm(lora_out2_A, lora_down_B, offs=offs)
            output2 = output2 + lora_out2 * self.scale

            if self.expert_bias:
                down_bias = _to_local(self.down_proj_bias)
                output2 = _apply_bias(
                    output2,
                    down_bias,
                    tokens_per_expert,
                    None if self.config.apply_router_weight_after_down else permuted_probs,
                )
        else:
            # Dummy computation for gradient flow
            output1 = torch.matmul(x[0] * 0, gate_and_up_projs[0])
            output1 = (
                output1
                + torch.matmul(torch.matmul(x[0] * 0, lora_gate_and_up_A[0]), lora_gate_and_up_B[0]) * self.scale
            )
            output1_ = self.expert_activation(output1, activation_probs)
            output2 = torch.matmul(output1_, down_projs[0])
            output2 = output2 + torch.matmul(torch.matmul(output1_ * 0, lora_down_A[0]), lora_down_B[0]) * self.scale

        if post_expert_output is not None:
            output2 = post_expert_output(output2)
        if self.config.apply_router_weight_after_down:
            output2 = _apply_router_weight_fp32(output2, permuted_probs, compute_dtype)
        y = self.token_dispatcher.token_unpermutation(output2)
        return y
