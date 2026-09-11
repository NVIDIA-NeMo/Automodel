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
from torch.distributed.tensor import DTensor, Partial, Shard

from nemo_automodel.components._peft.lora_experts import (
    GroupedExpertsDeepEPLoRA,
    GroupedExpertsLoRA,
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

    Two load paths:
    - ``passthrough=False`` (default): packing is deferred. The base loads as bf16 and
      is packed after the checkpoint load (``pack_base_weights()``). Works with any
      checkpoint, but materializes bf16 experts at load (high peak).
    - ``passthrough=True``: register packed base placeholders at init (no bf16 storage)
      so a packed fp4 checkpoint loads straight into them via the adapter's packed path,
      never materializing bf16 experts. The scale-out path for the full DeepSeek-V4-Flash.
    """

    def __init__(
        self,
        orig_module: GroupedExperts,
        lora_dim=8,
        alpha=32,
        lora_A_init_method="xavier",
        lora_dtype=None,
        passthrough=False,
    ):
        super().__init__(
            orig_module,
            lora_dim=lora_dim,
            alpha=alpha,
            lora_A_init_method=lora_A_init_method,
            lora_dtype=lora_dtype,
        )
        if passthrough:
            # Swap the frozen bf16 base placeholders (from super().__init__) for packed
            # mxfp4 placeholders; the LoRA adapters just built are left untouched. A packed
            # checkpoint then loads straight in with no bf16 expert materialization.
            self._init_packed_placeholders()
        else:
            self._init_mxfp4_storage()

    def forward(self, x: torch.Tensor, token_mask: torch.Tensor, weights: torch.Tensor, indices: torch.Tensor):
        """Forward pass with mxfp4 base weights and LoRA injection.

        Mirrors GroupedExpertsLoRA.forward, replacing the base grouped GEMMs with
        MXFP4GroupedMM over the packed weights. Falls back to the parent (bf16)
        path while packing is still deferred.
        """
        if not self._mxfp4_resident:
            return super().forward(x, token_mask, weights, indices)

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

        y = self._forward_grouped_mm_mxfp4(x, token_mask, weights, indices, n_local_experts, experts_start_idx)

        if ep_size > 1:
            y = DTensor.from_local(y, device_mesh=ep_mesh, placements=[Partial()])
            y = y.redistribute(placements=[Shard(0)]).to_local()

        return y.to(input_dtype)

    def _forward_grouped_mm_mxfp4(self, x, token_mask, weights, indices, n_local_experts, experts_start_idx):
        """Grouped GEMM forward path over packed mxfp4 base weights with LoRA injection."""
        sorted_token_ids, sorted_weights, tokens_per_expert, offs = _permute_tokens_for_grouped_mm(
            indices,
            weights,
            token_mask,
            n_local_experts,
            experts_start_idx,
        )

        # Match the activation dtype for the LoRA grouped GEMMs. The frozen base is
        # dequantized to x.dtype inside MXFP4GroupedMM, but the adapters may be a
        # different dtype (GroupedExperts allocates its base — hence the adapter dtype —
        # as fp32 when no backend dtype is set), which would mismatch torch._grouped_mm.
        lora_gate_and_up_A = _to_local(self.lora_gate_and_up_A).to(x.dtype)
        lora_gate_and_up_B = _to_local(self.lora_gate_and_up_B).to(x.dtype)
        lora_down_A = _to_local(self.lora_down_A).to(x.dtype)
        lora_down_B = _to_local(self.lora_down_B).to(x.dtype)

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

    Requires the torch_mm experts backend; the grouped_gemm (``gmm``) path has no packed
    variant. When constructed from a module whose base weights are still on the meta device,
    packing is deferred until ``pack_base_weights()`` runs after the checkpoint is loaded.
    """

    def __init__(
        self,
        orig_module: GroupedExpertsDeepEP,
        lora_dim=8,
        alpha=32,
        lora_A_init_method="xavier",
        lora_dtype=None,
        passthrough=False,
    ):
        super().__init__(
            orig_module,
            lora_dim=lora_dim,
            alpha=alpha,
            lora_A_init_method=lora_A_init_method,
            lora_dtype=lora_dtype,
        )
        if passthrough:
            # Packed base placeholders (no bf16) so a packed fp4 checkpoint loads straight
            # in; the LoRA adapters from super().__init__ are untouched. See
            # GroupedExpertsLoRAMXFP4 for the passthrough vs deferred distinction.
            self._init_packed_placeholders()
        else:
            self._init_mxfp4_storage()

    def forward(
        self,
        x: torch.Tensor,
        token_mask: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ):
        """Forward with mxfp4 base weights, DeepEP dispatch, and LoRA injection.

        Mirrors ``GroupedExpertsDeepEPLoRA.forward`` (torch_mm branch), replacing the base
        grouped GEMMs with ``MXFP4GroupedMM`` over the packed weights. Falls back to the
        bf16 parent while packing is still deferred.
        """
        if not self._mxfp4_resident:
            return super().forward(x, token_mask, weights, indices)

        assert not isinstance(x, DTensor)
        assert self.use_torch_mm, "mxfp4-resident DeepEP experts require the torch_mm experts backend."
        assert self.n_routed_experts % self.ep_size == 0

        indices = indices.masked_fill(~token_mask.unsqueeze(-1), -1)
        (permuted_local_hidden_states, tokens_per_expert, permuted_probs) = self.token_dispatcher.token_permutation2(
            hidden_states=x,
            num_local_tokens=x.size(0),
            token_probs=weights,
            token_indices=indices,
        )
        permuted_probs = permuted_probs.unsqueeze(-1)

        # Match the activation dtype for the LoRA grouped GEMMs (the base dequantizes to
        # x.dtype inside MXFP4GroupedMM; adapters may be fp32 — see GroupedExpertsLoRAMXFP4).
        lora_gate_and_up_A = _to_local(self.lora_gate_and_up_A).to(x.dtype)
        lora_gate_and_up_B = _to_local(self.lora_gate_and_up_B).to(x.dtype)
        lora_down_A = _to_local(self.lora_down_A).to(x.dtype)
        lora_down_B = _to_local(self.lora_down_B).to(x.dtype)

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
