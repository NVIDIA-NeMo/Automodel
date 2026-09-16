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
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn_f
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

from nemo_automodel.components.moe.experts import (
    GroupedExperts,
    GroupedExpertsDeepEP,
    _AllGatherConcatVarlenFn,
    _apply_bias,
    _apply_router_weight_fp32,
    _permute_tokens_for_grouped_mm,
)
from nemo_automodel.shared.utils import dtype_from_str

try:
    from grouped_gemm import ops
except ImportError:
    ops = None


def _to_local(proj):
    """Convert DTensor to local tensor, or return as-is."""
    return proj.to_local() if isinstance(proj, DTensor) else proj


def _to_grouped_mm_operand(proj, dtype: torch.dtype):
    """Convert a projection tensor to the dtype/layout expected by grouped MM."""
    return _to_local(proj).to(dtype).contiguous()


def _validate_qat_operand_dtypes(x: torch.Tensor, operands: tuple[torch.Tensor, ...]) -> None:
    """Reject compute casts that would change the exported effective weight.

    Args:
        x: Local activation tensor [tokens, hidden]. CPU and CUDA autocast must
            both be disabled, including when all stored dtypes match.
        operands: Original base and adapter tensors [local_experts, in, out],
            or DTensors sharded on expert axis 0 of a one-dimensional EP mesh.
            Every local dtype must equal x's dtype before any conversion.
    """
    if torch.is_autocast_enabled("cpu") or torch.is_autocast_enabled("cuda"):
        raise ValueError("Expert LoRA weight fake quantization does not support CPU or CUDA autocast")
    if any(_to_local(operand).dtype != x.dtype for operand in operands):
        raise ValueError(
            "Expert LoRA weight fake quantization requires activation dtype to match original local base and LoRA A/B "
            "dtypes so forward and exported effective weights use the same merge dtype"
        )


def _fake_quantize_effective_weight(
    base: torch.Tensor,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    scale: float,
    quantizer: nn.Module,
) -> torch.Tensor:
    """Quantize a local merged expert operand without mutating its parameters.

    Args:
        base: Local tensor of shape [local_experts, in, out], already converted
            by _to_grouped_mm_operand to the activation dtype and device.
        lora_A: Local tensor of shape [local_experts, in, rank], matching base's dtype/device.
        lora_B: Local tensor of shape [local_experts, rank, out], matching base's dtype/device.
        scale: LoRA alpha/rank multiplier, applied exactly once.
        quantizer: Fake quantizer accepting canonical [local_experts, out, in]
            weights and returning the same shape, dtype and device with autograd.

    Returns:
        Contiguous tensor of shape [local_experts, in, out] in the operand
        dtype/device, retaining gradients to both adapters. Inputs are not mutated.
    """
    if any(isinstance(t, DTensor) for t in (base, lora_A, lora_B)):
        raise TypeError("Expert weight fake quantization requires local grouped-MM operands")
    effective_weight = base + scale * (lora_A @ lora_B)
    return quantizer(effective_weight.transpose(-2, -1)).transpose(-2, -1).contiguous()


def _pad_lora_rank_for_grouped_mm(lora_A: torch.Tensor, lora_B: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad LoRA rank tensors so grouped-mm strides are 16-byte aligned."""
    rank = lora_A.size(-1)
    alignment = max(1, 16 // lora_A.element_size())
    padding = (-rank) % alignment
    if padding == 0:
        return lora_A, lora_B
    return F.pad(lora_A, (0, padding)), F.pad(lora_B, (0, 0, 0, padding))


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

    weight_fake_quantizer: nn.Module | None

    def __init__(self, orig_module: GroupedExperts, lora_dim=8, alpha=32, lora_A_init_method="xavier", lora_dtype=None):
        super().__init__(orig_module.config)
        self.to(device=orig_module.gate_and_up_projs.device, dtype=orig_module.gate_and_up_projs.dtype)

        self.gate_and_up_projs.data.copy_(orig_module.gate_and_up_projs.data)
        self.down_projs.data.copy_(orig_module.down_projs.data)

        if self.expert_bias:
            self.gate_up_proj_bias.data.copy_(orig_module.gate_up_proj_bias.data)
            self.down_proj_bias.data.copy_(orig_module.down_proj_bias.data)

        # Copy backend setting from original (super().__init__ defaults to False without backend)
        self.use_torch_mm = orig_module.use_torch_mm
        self.use_mxfp8 = orig_module.use_mxfp8

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
        obj.weight_fake_quantizer = None

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

        QAT merges adapters into local operands and explicitly dispatches the
        parent compute kernels, preserving its routing-weight placement and
        top-k reduction order without changing EP collectives. Non-QAT uses
        additive-LoRA kernels with the same configured routing placement.

        Args:
            x: Local tensor of shape [tokens, hidden], in the compute dtype.
                QAT requires this dtype to match original local base/A/B
                storage dtypes, with CPU and CUDA autocast disabled.
            token_mask: Local boolean tensor of shape [tokens].
            weights: Local routing probabilities of shape [tokens, top_k].
            indices: Local integer expert IDs of shape [tokens, top_k].

        Returns:
            Local tensor of shape [tokens, hidden], in x's dtype/device.
            Expert parameters may be DTensors sharded on expert axis 0 of a
            one-dimensional EP mesh; QAT only sees their local expert shards.
        """
        if self.weight_fake_quantizer is not None:
            if self.use_mxfp8:
                raise NotImplementedError("LoRA weight fake quantization does not support the MXFP8 expert backend")
            _validate_qat_operand_dtypes(
                x,
                (
                    self.gate_and_up_projs,
                    self.down_projs,
                    self.lora_gate_and_up_A,
                    self.lora_gate_and_up_B,
                    self.lora_down_A,
                    self.lora_down_B,
                ),
            )
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

        if self.weight_fake_quantizer is not None:
            gate_and_up_projs = _fake_quantize_effective_weight(
                gate_and_up_projs, lora_gate_and_up_A, lora_gate_and_up_B, self.scale, self.weight_fake_quantizer
            )
            down_projs = _fake_quantize_effective_weight(
                down_projs, lora_down_A, lora_down_B, self.scale, self.weight_fake_quantizer
            )

        local_num_tokens = x.size(0)
        if ep_size > 1:
            # Variable-length EP gather, mirroring GroupedExperts.forward.
            # DTensor.from_local(..., Shard(0)).full_tensor() assumes identical local shapes on
            # every EP rank; with ragged (unpacked / unpadded) batches each rank infers a different
            # global shape and the all_gather never completes. Exchange lengths, pad, gather, narrow.
            ep_group = ep_mesh.get_group()
            local_len_t = torch.tensor([local_num_tokens], device=x.device, dtype=torch.int64)
            gathered_len_t = [torch.zeros_like(local_len_t) for _ in range(ep_size)]
            dist.all_gather(gathered_len_t, local_len_t, group=ep_group)
            gathered_lens = [int(t.item()) for t in gathered_len_t]
            max_len = max(gathered_lens)

            def _gather_var(t: torch.Tensor, *, differentiable: bool) -> torch.Tensor:
                if differentiable:
                    return _AllGatherConcatVarlenFn.apply(t, ep_group, gathered_lens, max_len)
                if max_len > t.size(0):
                    pad = torch.zeros((max_len - t.size(0),) + tuple(t.shape[1:]), dtype=t.dtype, device=t.device)
                    t = torch.cat([t, pad], dim=0)
                gathered = [torch.empty_like(t) for _ in range(ep_size)]
                dist.all_gather(gathered, t, group=ep_group)
                return torch.cat([g[:n] for g, n in zip(gathered, gathered_lens)], dim=0)

            x = _gather_var(x, differentiable=True)
            weights = _gather_var(weights.float(), differentiable=True)
            indices = _gather_var(indices, differentiable=False)
            token_mask = _gather_var(token_mask, differentiable=False)

        n_local_experts = self.n_routed_experts // ep_size
        experts_start_idx = ep_rank * n_local_experts
        experts_end_idx = experts_start_idx + n_local_experts

        if self.weight_fake_quantizer is not None:
            gate_up_proj_bias = _to_local(self.gate_up_proj_bias).to(compute_dtype) if self.expert_bias else None
            down_proj_bias = _to_local(self.down_proj_bias).to(compute_dtype) if self.expert_bias else None
            has_local_tokens = (
                token_mask.unsqueeze(-1) & (indices >= experts_start_idx) & (indices < experts_end_idx)
            ).any()
            if not has_local_tokens:
                # Parent empty paths index x[0]. Keep every QAT gradient and
                # gathered input attached, even for zero tokens or no local routes.
                output_shape = (
                    (x.shape[0], weights.shape[1], x.shape[1])
                    if self.config.apply_router_weight_after_down
                    else x.shape
                )
                y = (
                    torch.zeros(output_shape, dtype=torch.float32, device=x.device)
                    + (
                        x.sum(dtype=torch.float32)
                        + weights.sum(dtype=torch.float32)
                        + gate_and_up_projs.sum(dtype=torch.float32)
                        + down_projs.sum(dtype=torch.float32)
                    )
                    * 0.0
                )
            elif self.use_torch_mm:
                y = GroupedExperts._forward_grouped_mm(
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
                )
            else:
                y = GroupedExperts._forward_loop(
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
                )
        elif self.use_torch_mm:
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
    ):
        """Additive-LoRA loop; QAT dispatches the parent kernel instead.

        Args:
            x: Tensor of shape [tokens, hidden], EP-gathered when needed.
            weights: Tensor of shape [tokens, top_k], routing probabilities.
            indices: Integer tensor of shape [tokens, top_k], global expert IDs.
            token_mask: Boolean tensor of shape [tokens].
            gate_and_up_projs: Local tensor [local_experts, hidden, up], where
                up is intermediate or fused gate+up (2 * intermediate).
            down_projs: Local tensor [local_experts, intermediate, hidden].
            lora_gate_and_up_A: Local tensor [local_experts, hidden, rank].
            lora_gate_and_up_B: Local tensor [local_experts, rank, up].
            lora_down_A: Local tensor [local_experts, intermediate, rank].
            lora_down_B: Local tensor [local_experts, rank, hidden].
            n_local_experts: Number of local experts.
            experts_start_idx: First global expert ID on this rank.
            experts_end_idx: Exclusive last global expert ID on this rank.

        Returns:
            FP32 tensor [tokens, top_k, hidden] for post-down routing, otherwise
            [tokens, hidden], on x's device. Operands have x's dtype/device.
            Slots are reduced by forward after EP combine. No input is mutated.
        """
        apply_after_down = self.config.apply_router_weight_after_down
        output_shape = (x.shape[0], weights.shape[1], x.shape[1]) if apply_after_down else x.shape
        y = torch.zeros(output_shape, dtype=torch.float32, device=x.device)
        gate_up_proj_bias = _to_local(self.gate_up_proj_bias).to(x.dtype) if self.expert_bias else None
        down_proj_bias = _to_local(self.down_proj_bias).to(x.dtype) if self.expert_bias else None

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
                gate_and_up_out = gate_and_up_out + gate_up_proj_bias[local_idx]

            # Preserve the parent's configured routing placement without merging LoRA.
            w = weights[idx, top, None]
            activation_weight = torch.ones_like(w) if apply_after_down else w
            activated = self.expert_activation_grouped(gate_and_up_out, activation_weight)

            # Down projection + LoRA
            expert_out = activated @ down_projs[local_idx]
            expert_out = expert_out + (activated @ lora_down_A[local_idx] @ lora_down_B[local_idx]) * self.scale

            if self.expert_bias:
                expert_out = expert_out + (
                    down_proj_bias[local_idx] if apply_after_down else down_proj_bias[local_idx] * w
                )

            if apply_after_down:
                expert_out = expert_out.float() * w.float()
                slot_ids = idx * weights.shape[1] + top
                slot_ids_b = slot_ids[:, None].expand(-1, x.size(1))
                y.view(-1, x.size(1)).scatter_add_(dim=0, index=slot_ids_b, src=expert_out.float())
            else:
                y.scatter_add_(dim=0, index=idx_b, src=expert_out.float())

        # Keep empty/fully masked ranks attached without indexing nonexistent tokens.
        if active_local_experts == 0:
            y = (
                y
                + (
                    x.sum(dtype=torch.float32)
                    + weights.sum(dtype=torch.float32)
                    + gate_and_up_projs.sum(dtype=torch.float32)
                    + down_projs.sum(dtype=torch.float32)
                    + lora_gate_and_up_A.sum(dtype=torch.float32)
                    + lora_gate_and_up_B.sum(dtype=torch.float32)
                    + lora_down_A.sum(dtype=torch.float32)
                    + lora_down_B.sum(dtype=torch.float32)
                )
                * 0.0
            )

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
    ):
        """Additive-LoRA grouped GEMM; QAT uses the parent kernel instead.

        Args:
            x: Tensor [tokens, hidden], EP-gathered when needed.
            token_mask: Boolean tensor [tokens].
            weights: Routing probability tensor [tokens, top_k].
            indices: Integer tensor [tokens, top_k], global expert IDs.
            gate_and_up_projs: Contiguous local tensor [local_experts, hidden,
                up], where up is intermediate or fused gate+up (2 * intermediate).
            down_projs: Contiguous local tensor [local_experts, intermediate, hidden].
            lora_gate_and_up_A: Contiguous local tensor [local_experts, hidden, rank].
            lora_gate_and_up_B: Contiguous local tensor [local_experts, rank, up].
            lora_down_A: Contiguous local tensor [local_experts, intermediate, rank].
            lora_down_B: Contiguous local tensor [local_experts, rank, hidden].
                Adapter ranks are padded for 16-byte stride alignment outside QAT.
            n_local_experts: Number of local experts.
            experts_start_idx: First global expert ID on this rank.

        Returns:
            FP32 tensor [tokens, top_k, hidden] for post-down routing, otherwise
            [tokens, hidden], on x's device. All operands share x's dtype/device.
            Slots are reduced by forward after EP combine. No input is mutated.
        """
        sorted_token_ids, sorted_slot_ids, sorted_weights, tokens_per_expert, offs = _permute_tokens_for_grouped_mm(
            indices,
            weights,
            token_mask,
            n_local_experts,
            experts_start_idx,
            return_slot_ids=True,
        )

        apply_after_down = self.config.apply_router_weight_after_down
        output_shape = (x.shape[0], weights.shape[1], x.shape[1]) if apply_after_down else x.shape
        y = torch.zeros(output_shape, dtype=torch.float32, device=x.device)

        if sorted_token_ids.numel() > 0:
            permuted_x = x[sorted_token_ids]
            permuted_probs = sorted_weights.unsqueeze(-1)
            activation_probs = torch.ones_like(permuted_probs) if apply_after_down else permuted_probs

            if self.expert_bias:
                gate_up_proj_bias = _to_local(self.gate_up_proj_bias).to(x.dtype)
                down_proj_bias = _to_local(self.down_proj_bias).to(x.dtype)

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
                    output2, down_proj_bias, tokens_per_expert, None if apply_after_down else permuted_probs
                )

            if apply_after_down:
                output2 = _apply_router_weight_fp32(output2, permuted_probs, torch.float32)
                scatter_ids = sorted_slot_ids.unsqueeze(1).expand_as(output2)
                y.view(-1, x.size(1)).scatter_add_(0, scatter_ids, output2.float())
            else:
                scatter_ids = sorted_token_ids.unsqueeze(1).expand_as(output2)
                y.scatter_add_(0, scatter_ids, output2.float())
        else:
            # Preserve the output layout and all gradients even for zero tokens.
            y = (
                y
                + (
                    x.sum(dtype=torch.float32)
                    + weights.sum(dtype=torch.float32)
                    + gate_and_up_projs.sum(dtype=torch.float32)
                    + down_projs.sum(dtype=torch.float32)
                    + lora_gate_and_up_A.sum(dtype=torch.float32)
                    + lora_gate_and_up_B.sum(dtype=torch.float32)
                    + lora_down_A.sum(dtype=torch.float32)
                    + lora_down_B.sum(dtype=torch.float32)
                )
                * 0.0
            )

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

    weight_fake_quantizer: nn.Module | None

    def __init__(
        self, orig_module: GroupedExpertsDeepEP, lora_dim=8, alpha=32, lora_A_init_method="xavier", lora_dtype=None
    ):
        super().__init__(
            orig_module.config,
            dispatcher_backend=orig_module.dispatcher_backend,
            dispatcher_num_sms=orig_module.dispatcher_num_sms,
            dispatcher_share_token_dispatcher=orig_module.dispatcher_share_token_dispatcher,
            dispatcher_async_dispatch=orig_module.dispatcher_async_dispatch,
        )
        self.to(device=orig_module.gate_and_up_projs.device, dtype=orig_module.gate_and_up_projs.dtype)

        self.gate_and_up_projs.data.copy_(orig_module.gate_and_up_projs.data)
        self.down_projs.data.copy_(orig_module.down_projs.data)

        if self.expert_bias:
            self.gate_up_proj_bias.data.copy_(orig_module.gate_up_proj_bias.data)
            self.down_proj_bias.data.copy_(orig_module.down_proj_bias.data)

        # Copy DeepEP state from orig_module (set by init_token_dispatcher, not __init__)
        self.n_routed_experts = getattr(orig_module, "n_routed_experts", self.config.n_routed_experts)
        self.ep_size = getattr(orig_module, "ep_size", 1)
        self.ep_rank = getattr(orig_module, "ep_rank", 0)
        self.token_dispatcher = getattr(orig_module, "token_dispatcher", None)
        self.use_torch_mm = getattr(orig_module, "use_torch_mm", False)
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
        obj.weight_fake_quantizer = None

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
    ) -> torch.Tensor:
        """Forward pass for GroupedExpertsDeepEPLoRA with LoRA injection.

        Mirrors GroupedExpertsDeepEP.forward but injects LoRA computations
        into the expert processing at the projection level. QAT merges local
        operands without changing dispatch or combine. Both paths follow the
        parent's configured pre-down or FP32 post-down routing-weight scaling;
        non-QAT keeps the adapter projections additive.

        Args:
            x: Local tensor [tokens, hidden], in the compute dtype. QAT requires
                original local base/A/B storage dtypes to match, with CPU and
                CUDA autocast disabled. Dispatch must preserve this dtype.
            token_mask: Local boolean tensor [tokens].
            weights: Local routing probability tensor [tokens, top_k].
            indices: Local integer tensor [tokens, top_k], global expert IDs.

        Returns:
            Local tensor [tokens, hidden] on x's device after dispatcher combine.
            Base and adapter parameters may be DTensors sharded on expert axis 0;
            QAT operates only on local [local_experts, in, out] shards, converting
            them to canonical [local_experts, out, in] for the quantizer.
        """
        if self.weight_fake_quantizer is not None:
            if self.use_mxfp8:
                raise NotImplementedError("LoRA weight fake quantization does not support the MXFP8 expert backend")
            _validate_qat_operand_dtypes(
                x,
                (
                    self.gate_and_up_projs,
                    self.down_projs,
                    self.lora_gate_and_up_A,
                    self.lora_gate_and_up_B,
                    self.lora_down_A,
                    self.lora_down_B,
                ),
            )
            if not self.use_torch_mm and ops is None:
                raise RuntimeError("DeepEP LoRA weight fake quantization requires grouped_gemm or use_torch_mm=True")
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
        apply_after_down = self.config.apply_router_weight_after_down
        activation_probs = torch.ones_like(permuted_probs) if apply_after_down else permuted_probs
        if self.weight_fake_quantizer is not None and permuted_local_hidden_states.dtype != x.dtype:
            raise ValueError("DeepEP LoRA weight fake quantization requires dispatch to preserve activation dtype")

        compute_dtype = x.dtype
        gate_and_up_projs = _to_grouped_mm_operand(self.gate_and_up_projs, compute_dtype)
        down_projs = _to_grouped_mm_operand(self.down_projs, compute_dtype)
        lora_gate_and_up_A = _to_grouped_mm_operand(self.lora_gate_and_up_A, compute_dtype)
        lora_gate_and_up_B = _to_grouped_mm_operand(self.lora_gate_and_up_B, compute_dtype)
        lora_down_A = _to_grouped_mm_operand(self.lora_down_A, compute_dtype)
        lora_down_B = _to_grouped_mm_operand(self.lora_down_B, compute_dtype)

        if self.weight_fake_quantizer is not None:
            gate_and_up_projs = _fake_quantize_effective_weight(
                gate_and_up_projs, lora_gate_and_up_A, lora_gate_and_up_B, self.scale, self.weight_fake_quantizer
            )
            down_projs = _fake_quantize_effective_weight(
                down_projs, lora_down_A, lora_down_B, self.scale, self.weight_fake_quantizer
            )

        if torch.count_nonzero(tokens_per_expert) > 0:
            if self.use_torch_mm:
                if self.weight_fake_quantizer is None:
                    lora_gate_and_up_A, lora_gate_and_up_B = _pad_lora_rank_for_grouped_mm(
                        lora_gate_and_up_A, lora_gate_and_up_B
                    )
                    lora_down_A, lora_down_B = _pad_lora_rank_for_grouped_mm(lora_down_A, lora_down_B)
                tokens_per_expert_gpu = tokens_per_expert.to(
                    device=permuted_local_hidden_states.device, non_blocking=True
                )
                offs = tokens_per_expert_gpu.cumsum(dim=0).to(torch.int32)

                # Gate+Up projection + LoRA
                output1 = torch._grouped_mm(permuted_local_hidden_states, gate_and_up_projs, offs=offs)
                if self.weight_fake_quantizer is None:
                    lora_out1_A = torch._grouped_mm(permuted_local_hidden_states, lora_gate_and_up_A, offs=offs)
                    lora_out1 = torch._grouped_mm(lora_out1_A, lora_gate_and_up_B, offs=offs)
                    output1 = output1 + lora_out1 * self.scale

                if self.expert_bias:
                    gate_up_proj_bias = _to_local(self.gate_up_proj_bias)
                    output1 = _apply_bias(output1, gate_up_proj_bias, tokens_per_expert)

                output1 = self.expert_activation(output1, activation_probs)

                # Down projection + LoRA
                output2 = torch._grouped_mm(output1, down_projs, offs=offs)
                if self.weight_fake_quantizer is None:
                    lora_out2_A = torch._grouped_mm(output1, lora_down_A, offs=offs)
                    lora_out2 = torch._grouped_mm(lora_out2_A, lora_down_B, offs=offs)
                    output2 = output2 + lora_out2 * self.scale

                if self.expert_bias:
                    down_bias = _to_local(self.down_proj_bias)
                    output2 = _apply_bias(
                        output2, down_bias, tokens_per_expert, None if apply_after_down else permuted_probs
                    )
            else:
                # Gate+Up projection + LoRA
                output1 = ops.gmm(
                    permuted_local_hidden_states,
                    gate_and_up_projs,
                    tokens_per_expert,
                    trans_b=False,
                )
                if self.weight_fake_quantizer is None:
                    lora_out1_A = ops.gmm(
                        permuted_local_hidden_states,
                        lora_gate_and_up_A,
                        tokens_per_expert,
                        trans_b=False,
                    )
                    lora_out1 = ops.gmm(lora_out1_A, lora_gate_and_up_B, tokens_per_expert, trans_b=False)
                    output1 = output1 + lora_out1 * self.scale

                if self.expert_bias:
                    gate_up_proj_bias = _to_local(self.gate_up_proj_bias).to(compute_dtype)
                    output1 = _apply_bias(output1, gate_up_proj_bias, tokens_per_expert)

                output1 = self.expert_activation(output1, activation_probs)

                # Down projection + LoRA
                output2 = ops.gmm(output1, down_projs, tokens_per_expert, trans_b=False)
                if self.weight_fake_quantizer is None:
                    lora_out2_A = ops.gmm(output1, lora_down_A, tokens_per_expert, trans_b=False)
                    lora_out2 = ops.gmm(lora_out2_A, lora_down_B, tokens_per_expert, trans_b=False)
                    output2 = output2 + lora_out2 * self.scale

                if self.expert_bias:
                    down_bias = _to_local(self.down_proj_bias).to(compute_dtype)
                    output2 = _apply_bias(
                        output2, down_bias, tokens_per_expert, None if apply_after_down else permuted_probs
                    )
        elif self.weight_fake_quantizer is not None:
            # Preserve the empty dispatcher graph and all adapter gradients without x[0].
            output2 = permuted_local_hidden_states + (
                (
                    gate_and_up_projs.sum(dtype=torch.float32)
                    + down_projs.sum(dtype=torch.float32)
                    + permuted_probs.sum(dtype=torch.float32)
                )
                * 0.0
            ).to(compute_dtype)
        else:
            # Preserve the dispatched [0, hidden] layout and additive adapter graph.
            output2 = permuted_local_hidden_states + (
                (
                    gate_and_up_projs.sum(dtype=torch.float32)
                    + down_projs.sum(dtype=torch.float32)
                    + lora_gate_and_up_A.sum(dtype=torch.float32)
                    + lora_gate_and_up_B.sum(dtype=torch.float32)
                    + lora_down_A.sum(dtype=torch.float32)
                    + lora_down_B.sum(dtype=torch.float32)
                    + permuted_probs.sum(dtype=torch.float32)
                )
                * 0.0
            ).to(compute_dtype)

        if apply_after_down:
            output2 = _apply_router_weight_fp32(output2, permuted_probs, compute_dtype)
        y = self.token_dispatcher.token_unpermutation(output2)
        return y
