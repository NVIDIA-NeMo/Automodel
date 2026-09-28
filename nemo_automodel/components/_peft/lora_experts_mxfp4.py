# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""MXFP4-resident expert LoRA implementations."""

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn_f
from torch.distributed.tensor import DTensor

from nemo_automodel.components._peft.lora_experts import (
    GroupedExpertsDeepEPLoRA,
    GroupedExpertsLoRA,
    _pad_lora_rank_for_grouped_mm,
    _to_grouped_mm_operand,
    _to_local,
)
from nemo_automodel.components.moe.experts import (
    GroupedExperts,
    GroupedExpertsDeepEP,
    _apply_bias,
    _permute_tokens_for_grouped_mm,
)
from nemo_automodel.components.moe.quantized_experts import MXFP4ExpertStorageMixin


class GroupedExpertsLoRAMXFP4(MXFP4ExpertStorageMixin, GroupedExpertsLoRA):
    """GroupedExperts + LoRA with the frozen base weights resident in packed mxfp4.

    The base gate/up and down projections are stored as packed fp4-e2m1 int8 plus
    ``float8_e8m0fnu`` block scales (checkpoint orientation ``[n_experts, out_dim,
    in_dim]``) and dequantized on the fly inside ``MXFP4GroupedMM`` during forward
    and backward (see ``MXFP4ExpertStorageMixin``). Only the LoRA adapters (and
    optional expert biases) remain in floating point.

    Meta base weights become packed placeholders for direct checkpoint loading;
    materialized base weights are quantized immediately.
    """

    def __init__(
        self,
        orig_module: GroupedExperts,
        lora_dim: int = 8,
        alpha: int = 32,
        lora_A_init_method: str = "xavier",
        lora_dtype: torch.dtype | str | None = None,
    ) -> None:
        """Initialize the parent's adapters, then replace the frozen base storage."""
        with torch.device(orig_module.gate_and_up_projs.device):
            super().__init__(
                orig_module,
                lora_dim=lora_dim,
                alpha=alpha,
                lora_A_init_method=lora_A_init_method,
                lora_dtype=lora_dtype,
            )
        self._init_mxfp4_storage()

    def forward(
        self, x: torch.Tensor, token_mask: torch.Tensor, weights: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass with mxfp4 base weights and LoRA injection.

        Preserves the tensor and EP contract of GroupedExperts.forward, replacing the base grouped GEMMs with
        MXFP4GroupedMM over the packed weights.
        """
        assert not isinstance(x, DTensor)
        input_dtype = x.dtype

        if isinstance(self.gate_and_up_projs_packed, DTensor):
            ep_mesh = self.gate_and_up_projs_packed.device_mesh
            assert ep_mesh is not None
            assert ep_mesh.ndim == 1
            ep_size = ep_mesh.size()
            ep_rank = ep_mesh.get_local_rank()
        else:
            ep_mesh = None
            ep_size = 1
            ep_rank = 0

        assert self.n_routed_experts % ep_size == 0

        if ep_size > 1:
            ep_group = ep_mesh.get_group()
            local_num_tokens = x.size(0)
            x, weights, indices, token_mask, gathered_lens = self._gather_ep_inputs(
                x, token_mask, weights, indices, ep_group=ep_group, ep_size=ep_size
            )

        n_local_experts = self.n_routed_experts // ep_size
        experts_start_idx = ep_rank * n_local_experts

        y = self._forward_grouped_mm_mxfp4(x, token_mask, weights, indices, n_local_experts, experts_start_idx)

        if ep_size > 1:
            # Keep the differentiable all-gather path attached to x without materializing a full-size zero tensor.
            y.add_(x.sum(dtype=torch.float32) * 0.0)

            # Reduce and narrow to the original per-rank token boundaries.
            y = dist_nn_f.all_reduce(y, op=dist.ReduceOp.SUM, group=ep_group)
            start = sum(gathered_lens[:ep_rank])
            y = y.narrow(0, start, local_num_tokens).contiguous()

        return y.to(input_dtype)

    def _forward_grouped_mm_mxfp4(
        self,
        x: torch.Tensor,
        token_mask: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        n_local_experts: int,
        experts_start_idx: int,
    ) -> torch.Tensor:
        """Compute the local experts' contribution with a packed base and trainable adapters.

        Args:
            x: Tensor of shape [tokens, hidden], gathered across the EP group.
            token_mask: Boolean tensor of shape [tokens] selecting valid tokens.
            weights: Tensor of shape [tokens, top_k] with differentiable routing probabilities.
            indices: Integer tensor of shape [tokens, top_k] with global expert IDs.
            n_local_experts: Number of experts on this rank.
            experts_start_idx: First global expert ID on this rank.

        Returns:
            FP32 tensor of shape [tokens, hidden], before the EP reduction.
        """
        sorted_token_ids, sorted_weights, tokens_per_expert, offs = _permute_tokens_for_grouped_mm(
            indices,
            weights,
            token_mask,
            n_local_experts,
            experts_start_idx,
        )

        lora_gate_and_up_A, lora_gate_and_up_B = _pad_lora_rank_for_grouped_mm(
            _to_grouped_mm_operand(self.lora_gate_and_up_A, x.dtype),
            _to_grouped_mm_operand(self.lora_gate_and_up_B, x.dtype),
        )
        lora_down_A, lora_down_B = _pad_lora_rank_for_grouped_mm(
            _to_grouped_mm_operand(self.lora_down_A, x.dtype),
            _to_grouped_mm_operand(self.lora_down_B, x.dtype),
        )

        y = torch.zeros(x.shape, dtype=torch.float32, device=x.device)

        if tokens_per_expert.sum() > 0:
            permuted_x = x[sorted_token_ids]
            permuted_probs = sorted_weights.unsqueeze(-1)

            if self.expert_bias:
                gate_up_proj_bias = _to_local(self.gate_up_proj_bias)
                down_proj_bias = _to_local(self.down_proj_bias)

            # Gate+Up projection + LoRA
            output1 = self._mxfp4_base_mm(permuted_x, "gate_and_up_projs", offs)
            lora_out1_A = torch._grouped_mm(permuted_x, lora_gate_and_up_A, offs=offs)
            lora_out1 = torch._grouped_mm(lora_out1_A, lora_gate_and_up_B, offs=offs)
            output1 = output1 + lora_out1 * self.scale

            if self.expert_bias:
                output1 = _apply_bias(output1, gate_up_proj_bias, tokens_per_expert)

            output1 = self.expert_activation_grouped(output1, permuted_probs)

            # Down projection + LoRA
            output2 = self._mxfp4_base_mm(output1, "down_projs", offs)
            lora_out2_A = torch._grouped_mm(output1, lora_down_A, offs=offs)
            lora_out2 = torch._grouped_mm(lora_out2_A, lora_down_B, offs=offs)
            output2 = output2 + lora_out2 * self.scale

            if self.expert_bias:
                output2 = _apply_bias(output2, down_proj_bias, tokens_per_expert, permuted_probs)

            scatter_ids = sorted_token_ids.unsqueeze(1).expand_as(output2)
            y.scatter_add_(0, scatter_ids, output2.float())
        else:
            # Dummy computation for gradient flow; dequantize only expert 0.
            gate_up_w0 = self._mxfp4_dequant_expert0("gate_and_up_projs", x.dtype)
            down_w0 = self._mxfp4_dequant_expert0("down_projs", x.dtype)
            output1 = torch.matmul(x[0] * 0, gate_up_w0)
            output1 = (
                output1
                + torch.matmul(torch.matmul(x[0] * 0, lora_gate_and_up_A[0]), lora_gate_and_up_B[0]) * self.scale
            )
            output1_ = self.expert_activation_grouped(output1, weights[0, 0, None].unsqueeze(0))
            output2 = torch.matmul(output1_, down_w0)
            output2 = output2 + torch.matmul(torch.matmul(output1_ * 0, lora_down_A[0]), lora_down_B[0]) * self.scale
            y[0] += output2[0]

        return y


class GroupedExpertsDeepEPLoRAMXFP4(MXFP4ExpertStorageMixin, GroupedExpertsDeepEPLoRA):
    """GroupedExpertsDeepEP + LoRA with the frozen base weights resident in packed mxfp4.

    The DeepEP fused all-to-all token dispatch is reused unchanged from
    ``GroupedExpertsDeepEPLoRA``; only the two frozen base grouped GEMMs read the packed
    fp4-e2m1 + e8m0 base weights via ``MXFP4GroupedMM`` instead of bf16. The LoRA A/B
    adapters (and optional expert biases) stay in floating point and their grouped GEMMs
    are unchanged.

    The DeepEP parent always uses native grouped MM.
    Meta bases become packed checkpoint placeholders; materialized bases
    are quantized immediately.
    """

    def __init__(
        self,
        orig_module: GroupedExpertsDeepEP,
        lora_dim: int = 8,
        alpha: int = 32,
        lora_A_init_method: str = "xavier",
        lora_dtype: torch.dtype | str | None = None,
    ) -> None:
        """Initialize the parent's adapters, then replace the frozen base storage."""
        with torch.device(orig_module.gate_and_up_projs.device):
            super().__init__(
                orig_module,
                lora_dim=lora_dim,
                alpha=alpha,
                lora_A_init_method=lora_A_init_method,
                lora_dtype=lora_dtype,
            )
        self._init_mxfp4_storage()

    def forward(
        self,
        x: torch.Tensor,
        token_mask: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """Forward with mxfp4 base weights, DeepEP dispatch, and LoRA injection.

        Preserves the tensor and EP contract of GroupedExpertsDeepEP.forward, replacing the base
        grouped GEMMs with ``MXFP4GroupedMM`` over the packed weights.
        """
        assert not isinstance(x, DTensor)
        assert self.n_routed_experts % self.ep_size == 0

        permuted_local_hidden_states, tokens_per_expert, permuted_probs = self._dispatch_tokens(
            x, token_mask, weights, indices
        )

        lora_gate_and_up_A, lora_gate_and_up_B = _pad_lora_rank_for_grouped_mm(
            _to_grouped_mm_operand(self.lora_gate_and_up_A, x.dtype),
            _to_grouped_mm_operand(self.lora_gate_and_up_B, x.dtype),
        )
        lora_down_A, lora_down_B = _pad_lora_rank_for_grouped_mm(
            _to_grouped_mm_operand(self.lora_down_A, x.dtype),
            _to_grouped_mm_operand(self.lora_down_B, x.dtype),
        )

        if torch.count_nonzero(tokens_per_expert) > 0:
            tokens_per_expert_gpu = tokens_per_expert.to(device=permuted_local_hidden_states.device, non_blocking=True)
            offs = tokens_per_expert_gpu.cumsum(dim=0).to(torch.int32)

            # Gate+Up projection (mxfp4 base) + LoRA
            output1 = self._mxfp4_base_mm(permuted_local_hidden_states, "gate_and_up_projs", offs)
            lora_out1_A = torch._grouped_mm(permuted_local_hidden_states, lora_gate_and_up_A, offs=offs)
            lora_out1 = torch._grouped_mm(lora_out1_A, lora_gate_and_up_B, offs=offs)
            output1 = output1 + lora_out1 * self.scale

            if self.expert_bias:
                output1 = _apply_bias(output1, _to_local(self.gate_up_proj_bias), tokens_per_expert)

            output1 = self.expert_activation(output1, permuted_probs)

            # Down projection (mxfp4 base) + LoRA
            output2 = self._mxfp4_base_mm(output1, "down_projs", offs)
            lora_out2_A = torch._grouped_mm(output1, lora_down_A, offs=offs)
            lora_out2 = torch._grouped_mm(lora_out2_A, lora_down_B, offs=offs)
            output2 = output2 + lora_out2 * self.scale

            if self.expert_bias:
                output2 = _apply_bias(output2, _to_local(self.down_proj_bias), tokens_per_expert, permuted_probs)
        else:
            # Dummy computation for gradient flow; dequantize only expert 0.
            gate_up_w0 = self._mxfp4_dequant_expert0("gate_and_up_projs", x.dtype)
            down_w0 = self._mxfp4_dequant_expert0("down_projs", x.dtype)
            output1 = torch.matmul(x[0] * 0, gate_up_w0)
            output1 = (
                output1
                + torch.matmul(torch.matmul(x[0] * 0, lora_gate_and_up_A[0]), lora_gate_and_up_B[0]) * self.scale
            )
            output1_ = self.expert_activation(output1, permuted_probs)
            output2 = torch.matmul(output1_, down_w0)
            output2 = output2 + torch.matmul(torch.matmul(output1_ * 0, lora_down_A[0]), lora_down_B[0]) * self.scale

        y = self.token_dispatcher.token_unpermutation(output2)
        return y
