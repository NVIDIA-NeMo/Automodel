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

"""MXFP4-resident expert storage and model application for frozen MoE experts.

``GroupedExpertsMXFP4`` keeps the frozen routed-expert base weights packed as
fp4-e2m1 + e8m0 block scales (the DeepSeek V4 Flash checkpoint format) and
dequantizes on the fly inside the grouped GEMM, instead of holding them in
bf16. This is the storage win for LoRA / frozen-base training of large MoE
models, where the routed experts dominate parameter memory.

``MXFP4ExpertStorageMixin`` shares packed storage and base GEMMs between the
frozen and LoRA experts, with Torch or DeepEP token dispatch.
"""

import logging

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn_f
import torch.nn as nn
from torch.distributed.tensor import DTensor

from nemo_automodel.components.moe.experts import (
    GroupedExperts,
    GroupedExpertsDeepEP,
    GroupedExpertsTE,
    _apply_bias,
    _permute_tokens_for_grouped_mm,
)
from nemo_automodel.components.quantization.mxfp4 import (
    MXFP4_BLOCK_SIZE,
    MXFP4GroupedMM,
    dequantize_mxfp4,
    quantize_mxfp4,
)

logger = logging.getLogger(__name__)


def _to_local(t):
    """Return the local shard of a DTensor, or the tensor unchanged."""
    return t.to_local() if isinstance(t, DTensor) else t


class MXFP4ExpertStorageMixin:
    """Packed-mxfp4 base-weight storage and grouped GEMM for routed experts.

    Mixed into a ``GroupedExperts`` (or ``GroupedExpertsLoRA``) subclass. The base
    projections ``gate_and_up_projs`` / ``down_projs`` are stored as packed fp4
    (int8, two e2m1 nibbles per byte) plus ``float8_e8m0fnu`` block scales, in
    checkpoint orientation ``[n_experts, out_dim, in_dim]`` so the block scales run
    along the contraction dim. Floating-point base parameters are dropped once packed.

    Meta weights become packed placeholders for direct checkpoint loading.
    Materialized weights are quantized immediately during construction.
    """

    _MXFP4_BASE_NAMES: tuple[str, ...] = ("gate_and_up_projs", "down_projs")

    def _validate_mxfp4_config(self) -> None:
        """Reject execution modes that the packed expert computation does not implement."""
        if self.config.apply_router_weight_after_down:
            raise NotImplementedError("MXFP4 experts do not support apply_router_weight_after_down=True.")
        if not self.use_torch_mm:
            raise NotImplementedError(
                "mxfp4-resident expert weights require the torch_mm experts backend (backend.experts='torch_mm'). "
                "The grouped_gemm path (backend.experts='gmm') has no packed variant; with DeepEP dispatch use "
                "backend.dispatcher='deepep' together with backend.experts='torch_mm'."
            )

    def _init_mxfp4_storage(self) -> None:
        """Replace floating-point bases with packed placeholders or quantized weights."""
        self._validate_mxfp4_config()
        if _to_local(self.gate_and_up_projs).is_meta:
            self._init_packed_placeholders()
        else:
            self._pack_base_weights()
        # The shared expert initializer must skip the removed floating-point bases.
        self._mxfp4_resident = True

    @torch.no_grad()
    def _init_packed_placeholders(self) -> None:
        """Register packed meta parameters in checkpoint orientation [experts, out, in]."""
        cfg = self.config
        block = MXFP4_BLOCK_SIZE
        up_proj_dim = cfg.moe_inter_dim * 2 if self.is_gated else cfg.moe_inter_dim
        expert_dim = cfg.expert_dim
        moe_inter = cfg.moe_inter_dim
        e = cfg.n_routed_experts
        assert expert_dim % block == 0 and moe_inter % block == 0, (
            f"expert dims must be divisible by {block} for mxfp4 (expert_dim={expert_dim}, moe_inter={moe_inter})"
        )
        # Checkpoint orientation [E, out, in], packed along the contraction (in) dim.
        shapes = {
            "gate_and_up_projs": ((e, up_proj_dim, expert_dim // 2), (e, up_proj_dim, expert_dim // block)),
            "down_projs": ((e, expert_dim, moe_inter // 2), (e, expert_dim, moe_inter // block)),
        }
        for name, (packed_shape, scale_shape) in shapes.items():
            packed = torch.empty(packed_shape, dtype=torch.int8, device="meta")
            scales = torch.empty(scale_shape, dtype=torch.float8_e8m0fnu, device="meta")
            self._register_packed_base_weight(name, packed, scales)

    def _register_packed_base_weight(self, name: str, packed: torch.Tensor, scales: torch.Tensor) -> None:
        """Replace one floating-point base projection with frozen packed parameters.

        Args:
            name: Base projection parameter name.
            packed: Int8 tensor of shape [experts, out_dim, in_dim // 2].
            scales: E8M0 tensor of shape [experts, out_dim, in_dim // 32].
                DTensors must retain the base parameter's mesh and expert-axis placement.
        """
        del self._parameters[name]
        self.register_parameter(name + "_packed", nn.Parameter(packed, requires_grad=False))
        self.register_parameter(name + "_scales", nn.Parameter(scales, requires_grad=False))

    @torch.no_grad()
    def _pack_base_weights(self) -> None:
        """Quantize materialized base projections and release their floating-point storage."""
        for name in self._MXFP4_BASE_NAMES:
            param = getattr(self, name)
            # Compute layout [experts, in, out] -> checkpoint layout [experts, out, in].
            packed, scales = quantize_mxfp4(_to_local(param).transpose(-2, -1).contiguous())
            if isinstance(param, DTensor):
                packed = DTensor.from_local(packed, param.device_mesh, param.placements)
                scales = DTensor.from_local(scales, param.device_mesh, param.placements)
            self._register_packed_base_weight(name, packed, scales)

    def _mxfp4_base_mm(self, x: torch.Tensor, name: str, offs: torch.Tensor) -> torch.Tensor:
        """Multiply routed activations by a frozen packed base projection.

        Args:
            x: Tensor of shape [tokens, in_dim], grouped contiguously by local expert.
            name: Base projection parameter name.
            offs: Int32 tensor of shape [local_experts], holding cumulative token counts.

        Returns:
            Tensor of shape [tokens, out_dim] with the activation dtype and device.
        """
        packed = _to_local(getattr(self, name + "_packed"))
        scales = _to_local(getattr(self, name + "_scales"))
        return MXFP4GroupedMM.apply(x, packed, scales, offs)

    def _mxfp4_dequant_expert0(self, name: str, dtype: torch.dtype) -> torch.Tensor:
        """Dequantize expert 0 of base weight ``name`` to compute layout ``[in, out]``."""
        packed = _to_local(getattr(self, name + "_packed"))[0]
        scales = _to_local(getattr(self, name + "_scales"))[0]
        return dequantize_mxfp4(packed, scales, dtype).transpose(-2, -1)


class GroupedExpertsMXFP4(MXFP4ExpertStorageMixin, GroupedExperts):
    """Frozen routed experts with mxfp4-resident base weights and no adapter.

    Drop-in replacement for ``GroupedExperts`` when the experts are frozen (e.g.
    LoRA training that targets only attention). Forward mirrors
    ``GroupedExperts._forward_grouped_mm`` but reads the packed base weights.
    """

    def __init__(self, orig_module: GroupedExperts) -> None:
        """Adopt frozen weights, or create packed placeholders when the original is meta."""
        with torch.device(orig_module.gate_and_up_projs.device):
            super().__init__(orig_module.config, backend=None)
        self.use_torch_mm = orig_module.use_torch_mm
        # These fresh parameters have no autograd history or optimizer references.
        for name in self._MXFP4_BASE_NAMES:
            getattr(self, name).data = _to_local(getattr(orig_module, name)).clone()
        if self.expert_bias:
            self.gate_up_proj_bias.data = _to_local(orig_module.gate_up_proj_bias).clone()
            self.down_proj_bias.data = _to_local(orig_module.down_proj_bias).clone()
        self.requires_grad_(False)
        self._init_mxfp4_storage()

    def forward(
        self,
        x: torch.Tensor,
        token_mask: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute packed experts with the tensor and EP contract of GroupedExperts.forward."""
        assert not isinstance(x, DTensor)
        input_dtype = x.dtype

        if isinstance(self.gate_and_up_projs_packed, DTensor):
            ep_mesh = self.gate_and_up_projs_packed.device_mesh
            assert ep_mesh is not None and ep_mesh.ndim == 1, "We only support 1D mesh for MoE"
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
        """Compute the frozen local experts' contribution before the EP reduction.

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
            indices, weights, token_mask, n_local_experts, experts_start_idx
        )
        y = torch.zeros(x.shape, dtype=torch.float32, device=x.device)

        if tokens_per_expert.sum() > 0:
            permuted_x = x[sorted_token_ids]
            permuted_probs = sorted_weights.unsqueeze(-1)

            output1 = self._mxfp4_base_mm(permuted_x, "gate_and_up_projs", offs)
            if self.expert_bias:
                output1 = _apply_bias(output1, _to_local(self.gate_up_proj_bias), tokens_per_expert)
            output1 = self.expert_activation_grouped(output1, permuted_probs)

            output2 = self._mxfp4_base_mm(output1, "down_projs", offs)
            if self.expert_bias:
                output2 = _apply_bias(output2, _to_local(self.down_proj_bias), tokens_per_expert, permuted_probs)

            scatter_ids = sorted_token_ids.unsqueeze(1).expand_as(output2)
            y.scatter_add_(0, scatter_ids, output2.float())
        else:
            # Dummy computation for gradient flow when no tokens routed locally.
            gate_up_w0 = self._mxfp4_dequant_expert0("gate_and_up_projs", x.dtype)
            down_w0 = self._mxfp4_dequant_expert0("down_projs", x.dtype)
            output1 = torch.matmul(x[0] * 0, gate_up_w0)
            output1_ = self.expert_activation_grouped(output1, weights[0, 0, None].unsqueeze(0))
            output2 = torch.matmul(output1_, down_w0)
            y[0] += output2[0]

        return y


class GroupedExpertsDeepEPMXFP4(MXFP4ExpertStorageMixin, GroupedExpertsDeepEP):
    """Frozen routed experts with mxfp4-resident base weights under DeepEP dispatch.

    Drop-in replacement for ``GroupedExpertsDeepEP`` when the experts are frozen
    (e.g. LoRA on attention only). The DeepEP fused all-to-all token dispatch is reused
    unchanged — mxfp4 only changes the two post-dispatch grouped GEMMs, which read the
    packed base weights via ``MXFP4GroupedMM`` instead of bf16 ``torch._grouped_mm``.

    Requires the torch_mm experts backend (``backend.experts='torch_mm'``); the
    grouped_gemm (``gmm``) path has no packed variant.
    """

    def __init__(self, orig_module: GroupedExpertsDeepEP) -> None:
        """Adopt frozen weights, or create packed placeholders when the original is meta."""
        with torch.device(orig_module.gate_and_up_projs.device):
            super().__init__(
                orig_module.config,
                backend=None,
                dispatcher_backend=orig_module.dispatcher_backend,
                dispatcher_num_sms=orig_module.dispatcher_num_sms,
                dispatcher_share_token_dispatcher=orig_module.dispatcher_share_token_dispatcher,
                dispatcher_async_dispatch=orig_module.dispatcher_async_dispatch,
            )
        self.use_torch_mm = orig_module.use_torch_mm
        # These fresh parameters have no autograd history or optimizer references.
        for name in self._MXFP4_BASE_NAMES:
            getattr(self, name).data = _to_local(getattr(orig_module, name)).clone()
        if self.expert_bias:
            self.gate_up_proj_bias.data = _to_local(orig_module.gate_up_proj_bias).clone()
            self.down_proj_bias.data = _to_local(orig_module.down_proj_bias).clone()
        self.requires_grad_(False)
        self._init_mxfp4_storage()

    def forward(
        self,
        x: torch.Tensor,
        token_mask: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """Forward over mxfp4 base weights with DeepEP dispatch.

        Preserves the tensor and EP contract of ``GroupedExpertsDeepEP.forward``, replacing the two base
        ``torch._grouped_mm`` calls with ``MXFP4GroupedMM`` over the packed weights.
        """
        assert not isinstance(x, DTensor)
        assert self.use_torch_mm, "mxfp4-resident DeepEP experts require the torch_mm experts backend."
        assert self.n_routed_experts % self.ep_size == 0, (
            f"Number of experts must be divisible by ep_size (ep_size={self.ep_size})"
        )

        permuted_local_hidden_states, tokens_per_expert, permuted_probs = self._dispatch_tokens(
            x, token_mask, weights, indices
        )

        if torch.count_nonzero(tokens_per_expert) > 0:
            tokens_per_expert_gpu = tokens_per_expert.to(device=permuted_local_hidden_states.device, non_blocking=True)
            offs = tokens_per_expert_gpu.cumsum(dim=0).to(torch.int32)

            output1 = self._mxfp4_base_mm(permuted_local_hidden_states, "gate_and_up_projs", offs)
            if self.expert_bias:
                output1 = _apply_bias(output1, _to_local(self.gate_up_proj_bias), tokens_per_expert)
            output1 = self.expert_activation(output1, permuted_probs)
            output2 = self._mxfp4_base_mm(output1, "down_projs", offs)
            if self.expert_bias:
                output2 = _apply_bias(output2, _to_local(self.down_proj_bias), tokens_per_expert, permuted_probs)
        else:
            # Dummy computation for gradient flow when no tokens routed locally.
            gate_up_w0 = self._mxfp4_dequant_expert0("gate_and_up_projs", x.dtype)
            down_w0 = self._mxfp4_dequant_expert0("down_projs", x.dtype)
            output1 = torch.matmul(x[0] * 0, gate_up_w0)
            output1_ = self.expert_activation(output1, permuted_probs)
            output2 = torch.matmul(output1_, down_w0)

        y = self.token_dispatcher.token_unpermutation(output2)
        return y


def apply_mxfp4_to_moe_experts(model: nn.Module) -> nn.Module:
    """Apply MXFP4-resident storage to the model's common routed-expert modules.

    Call this after LoRA injection and before distributed sharding. LoRA-targeted
    experts that are already MXFP4-resident are preserved; remaining plain
    ``GroupedExperts`` and ``GroupedExpertsDeepEP`` modules are replaced in place.

    Meta experts receive packed placeholders for direct checkpoint loading, and
    their model state-dict adapter must opt into the MXFP4 expert storage format.
    Already materialized expert weights are quantized without changing the adapter's
    checkpoint loading mode. Model-specific adapters own checkpoint layouts.
    """
    model_parts = list(model.parts) if hasattr(model, "parts") else [model]

    unsupported = [
        name
        for model_part in model_parts
        for name, module in model_part.named_modules()
        if isinstance(module, GroupedExpertsTE)
    ]
    if unsupported:
        raise NotImplementedError(
            "MXFP4-resident expert weights do not support Transformer Engine expert modules; "
            "use backend.experts='torch_mm'."
        )

    for model_part in model_parts:
        needs_checkpoint = any(
            param.is_meta
            for module in model_part.modules()
            if isinstance(module, (GroupedExperts, GroupedExpertsDeepEP))
            for param in module.parameters(recurse=False)
        )
        if needs_checkpoint:
            adapter = getattr(model_part, "state_dict_adapter", None)
            set_storage_format = getattr(adapter, "set_expert_storage_format", None)
            if not callable(set_storage_format):
                raise NotImplementedError(
                    "Loading an MXFP4 expert checkpoint requires the model state-dict adapter to implement "
                    "set_expert_storage_format()."
                )
            set_storage_format("mxfp4")

    frozen_conversions = {
        GroupedExperts: GroupedExpertsMXFP4,
        GroupedExpertsDeepEP: GroupedExpertsDeepEPMXFP4,
    }
    num_converted = 0
    for model_part in model_parts:
        for name, module in list(model_part.named_modules()):
            if isinstance(module, MXFP4ExpertStorageMixin):
                continue
            new_cls = frozen_conversions.get(type(module))
            if new_cls is None:
                continue
            new_module = new_cls(module)
            parent_name, _, child_name = name.rpartition(".")
            parent = model_part.get_submodule(parent_name) if parent_name else model_part
            setattr(parent, child_name, new_module)
            num_converted += 1

    logger.info("Applied MXFP4-resident storage to %d frozen expert module(s)", num_converted)
    return model
