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

import glob
import json
import logging
import math
import os
import re

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
from nemo_automodel.components.quantization.nvfp4 import (
    NVFP4_BLOCK_SIZE,
    NVFP4GroupedMM,
    dequantize_nvfp4,
    quantize_nvfp4,
)

logger = logging.getLogger(__name__)


def _pin_storage_dtype(param: nn.Parameter) -> nn.Parameter:
    """Record a packed-storage parameter's dtype as its compute dtype so the frozen-parameter casts keep it.

    ``prepare_frozen_parameters_for_sharding`` and ``fully_shard_by_dtype`` honour ``_hf_compute_dtype``;
    the frozen-parameter casts also receive the parameters as ``keep_storage`` (``packed_expert_parameters``).
    """
    if param.is_floating_point():
        param._hf_compute_dtype = param.dtype
    return param


def _preserve_packed_storage_dtypes(module: nn.Module, names: tuple[str, ...]) -> None:
    """Record the storage dtypes of packed parameters on ``module`` so whole-model casts keep them.

    ``cast_model_to_dtype`` consults ``_preserve_parameter_dtypes``. Recording the packed names when they are
    registered makes the contract independent of ``prepare_frozen_parameters_for_sharding``, which only FSDP2
    PEFT runs call before ``initialize_weights``.
    """
    preserved = dict(getattr(module, "_preserve_parameter_dtypes", None) or {})
    preserved.update({name: getattr(module, name).dtype for name in names})
    module._preserve_parameter_dtypes = preserved


_PACKED_EXPERT_FORMATS = ("mxfp4", "nf4", "nvfp4")


def validate_packed_expert_topology(
    expert_weight_format: str, *, world_size: int, ep_size: int, pp_size: int = 1
) -> None:
    """Reject distributed layouts the packed-expert storage formats do not implement.

    Packed experts (mxfp4 / nf4 / nvfp4) are owned by their module as plain integer codes plus scale
    tensors, sharded only along the expert axis. They are supported on one rank, and with expert
    parallelism that spans the whole non-pipeline mesh (``ep_size == world_size // pp_size``) where the
    experts are excluded from FSDP. Two layouts are refused: ``ep_size == 1`` on several ranks (FSDP would
    shard the packed tensors by compute dtype and all-gather the scales in bf16) and an additional
    expert-shard axis ``ep_shard > 1`` (FSDP would shard the codes on dim 1 and cast them to the policy
    dtype, which breaks the packed layout).

    Args:
        expert_weight_format: ``PeftConfig.expert_weight_format``; anything but a packed format is accepted.
        world_size: Number of ranks.
        ep_size: Expert-parallel size.
        pp_size: Pipeline-parallel size (tensor and context parallelism are part of the non-pipeline mesh,
            as in ``mesh_utils``, and do not change the expert-shard arithmetic).

    Raises:
        ValueError: If the layout is unsupported for packed experts.
    """
    if expert_weight_format not in _PACKED_EXPERT_FORMATS or world_size <= 1:
        return
    non_pp = max(world_size // max(pp_size, 1), 1)
    if ep_size <= 1:
        raise ValueError(
            f"expert_weight_format={expert_weight_format!r} needs expert parallelism on {world_size} ranks "
            "(distributed.ep_size == 1 would FSDP-shard the packed expert tensors by dtype); set ep_size to the "
            "number of non-pipeline ranks, or use expert_weight_format='unquantized'."
        )
    ep_shard = non_pp // ep_size if ep_size < non_pp else 1
    if ep_shard > 1:
        raise ValueError(
            f"expert_weight_format={expert_weight_format!r} does not support an extra expert-shard axis "
            f"(world_size={world_size}, pp_size={pp_size}, ep_size={ep_size} -> ep_shard={ep_shard}); packed experts are "
            "sharded along the expert axis only, so ep_size must equal the number of non-pipeline ranks."
        )


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
        # DeepEP experts always use native grouped MM; only plain experts can select the loop backend.
        if isinstance(self, GroupedExperts) and not self.use_torch_mm:
            raise NotImplementedError(
                "mxfp4-resident expert weights require the torch_mm experts backend (backend.experts='torch_mm'). "
                "The per-expert loop has no packed variant; with DeepEP dispatch use "
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
        _preserve_packed_storage_dtypes(self, (name + "_packed", name + "_scales"))

    def packed_parameter_names(self) -> tuple[str, ...]:
        """Names of the packed codes and block scales that replace the floating-point base projections."""
        return tuple(name + suffix for name in self._MXFP4_BASE_NAMES for suffix in ("_packed", "_scales"))

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

    The DeepEP parent always uses native grouped MM.
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
        self.dispatcher_hybridep_permute_fusion = orig_module.dispatcher_hybridep_permute_fusion
        self.dispatcher_hybridep_compact_routing = orig_module.dispatcher_hybridep_compact_routing
        self.dispatcher_hybridep_num_sms_preprocessing = orig_module.dispatcher_hybridep_num_sms_preprocessing
        self.dispatcher_hybridep_num_blocks_permute = orig_module.dispatcher_hybridep_num_blocks_permute
        self.dispatcher_hybridep_num_blocks_unpermute = orig_module.dispatcher_hybridep_num_blocks_unpermute
        self.dispatcher_hybridep_max_tokens_per_rank = orig_module.dispatcher_hybridep_max_tokens_per_rank
        # Sync-free dispatch knobs (BackendConfig.dispatcher_capacity_factor / dispatcher_equal_token_counts) travel
        # with the other HybridEP settings: the shared dispatcher is built from the first (wrapped) layer.
        self.dispatcher_capacity_factor = orig_module.dispatcher_capacity_factor
        self.dispatcher_equal_token_counts = orig_module.dispatcher_equal_token_counts
        self.dispatch_rows_never_empty = orig_module.dispatch_rows_never_empty
        # The combine cache (BackendConfig.dispatcher_recompute_cache) is read by the shared dispatcher's config as
        # well: a frozen layer next to a LoRA one must carry the same mode, or the shared-manager check rejects the
        # pair and an all-frozen conversion silently loses the request (review of #4395).
        self.dispatcher_recompute_cache = orig_module.dispatcher_recompute_cache
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
        assert self.n_routed_experts % self.ep_size == 0, (
            f"Number of experts must be divisible by ep_size (ep_size={self.ep_size})"
        )

        permuted_local_hidden_states, tokens_per_expert, permuted_probs = self._dispatch_tokens(
            x, token_mask, weights, indices
        )

        # dispatch_rows_never_empty (static routing or capacity mode): skip the count_nonzero host read, as
        # GroupedExpertsDeepEP.forward does.
        if self.dispatch_rows_never_empty or torch.count_nonzero(tokens_per_expert) > 0:
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


class NVFP4ExpertStorageMixin:
    """Packed-NVFP4 base-weight storage and grouped GEMM for routed experts (ModelOpt NVFP4 checkpoints).

    Mixed into a ``GroupedExperts`` (or ``GroupedExpertsLoRA``) subclass. The base
    projections ``gate_and_up_projs`` / ``down_projs`` are stored as packed fp4-e2m1 codes
    (``uint8``, two per byte), one ``float8_e4m3fn`` scale per ``NVFP4_BLOCK_SIZE`` elements and
    one ``float32`` scale per output row, in checkpoint orientation ``[n_experts, out_dim, in_dim]``
    so the blocks run along the contraction dim: 0.5625 bytes per parameter plus 4 bytes per row.
    Floating-point base parameters are dropped once packed.

    Meta weights become packed placeholders in the shapes and dtypes of a ModelOpt NVFP4 per-expert HF
    checkpoint; the standard base-checkpoint load fills them through the model's state-dict adapter
    (``set_expert_storage_format("nvfp4")`` maps the per-expert ``weight`` / ``weight_scale`` /
    ``weight_scale_2`` tensors onto them), so no quantization runs at load. Materialized
    (floating-point) weights are quantized immediately during construction.
    """

    _NVFP4_BASE_NAMES: tuple[str, ...] = ("gate_and_up_projs", "down_projs")

    @staticmethod
    def nvfp4_parameter_names(name: str) -> tuple[str, str, str]:
        """Names of the packed codes, e4m3 block scales and fp32 row scales replacing base projection ``name``."""
        return name + "_nvfp4", name + "_scales", name + "_global"

    def _validate_nvfp4_config(self) -> None:
        """Reject execution modes that the packed expert computation does not implement."""
        if self.config.apply_router_weight_after_down:
            raise NotImplementedError("NVFP4 experts do not support apply_router_weight_after_down=True.")
        # DeepEP experts always use native grouped MM; only plain experts can select the loop backend.
        if isinstance(self, GroupedExperts) and not self.use_torch_mm:
            raise NotImplementedError(
                "nvfp4-resident expert weights require the torch_mm experts backend (backend.experts='torch_mm'). "
                "The per-expert loop has no packed variant; with DeepEP / HybridEP dispatch use "
                "backend.dispatcher='deepep' or 'hybridep' together with backend.experts='torch_mm'."
            )

    def _init_nvfp4_storage(self) -> None:
        """Replace floating-point bases with packed placeholders or quantized weights."""
        self._validate_nvfp4_config()
        if _to_local(self.gate_and_up_projs).is_meta:
            self._init_nvfp4_placeholders()
        else:
            self._pack_base_weights_nvfp4()
        # The shared expert initializer must skip the removed floating-point bases.
        self._nvfp4_resident = True

    @torch.no_grad()
    def _init_nvfp4_placeholders(self) -> None:
        """Register packed meta parameters in checkpoint orientation [experts, out, in]."""
        cfg = self.config
        block = NVFP4_BLOCK_SIZE
        up_proj_dim = cfg.moe_inter_dim * 2 if self.is_gated else cfg.moe_inter_dim
        expert_dim = cfg.expert_dim
        moe_inter = cfg.moe_inter_dim
        e = cfg.n_routed_experts
        assert expert_dim % block == 0 and moe_inter % block == 0, (
            f"expert dims must be divisible by {block} for nvfp4 (expert_dim={expert_dim}, moe_inter={moe_inter})"
        )
        # Checkpoint orientation [E, out, in], packed along the contraction (in) dim; one row scale per output row.
        shapes = {
            "gate_and_up_projs": (
                (e, up_proj_dim, expert_dim // 2),
                (e, up_proj_dim, expert_dim // block),
                (e, up_proj_dim),
            ),
            "down_projs": ((e, expert_dim, moe_inter // 2), (e, expert_dim, moe_inter // block), (e, expert_dim)),
        }
        for name, (packed_shape, scales_shape, global_shape) in shapes.items():
            packed = torch.empty(packed_shape, dtype=torch.uint8, device="meta")
            scales = torch.empty(scales_shape, dtype=torch.float8_e4m3fn, device="meta")
            global_scale = torch.empty(global_shape, dtype=torch.float32, device="meta")
            self._register_nvfp4_base_weight(name, packed, scales, global_scale)

    def _register_nvfp4_base_weight(
        self, name: str, packed: torch.Tensor, scales: torch.Tensor, global_scale: torch.Tensor
    ) -> None:
        """Replace one floating-point base projection with frozen packed parameters.

        Args:
            name: Base projection parameter name.
            packed: ``uint8`` tensor of shape [experts, out_dim, in_dim // 2].
            scales: ``float8_e4m3fn`` tensor of shape [experts, out_dim, in_dim // NVFP4_BLOCK_SIZE].
            global_scale: ``float32`` tensor of shape [experts, out_dim].
                DTensors must retain the base parameter's mesh and expert-axis placement.
        """
        packed_name, scales_name, global_name = self.nvfp4_parameter_names(name)
        del self._parameters[name]
        self.register_parameter(packed_name, nn.Parameter(packed, requires_grad=False))
        self.register_parameter(scales_name, nn.Parameter(scales, requires_grad=False))
        self.register_parameter(global_name, nn.Parameter(global_scale, requires_grad=False))
        # Pin the registered objects: ``init_empty_weights`` re-creates every registered parameter.
        _pin_storage_dtype(getattr(self, scales_name))
        _pin_storage_dtype(getattr(self, global_name))
        _preserve_packed_storage_dtypes(self, (packed_name, scales_name, global_name))

    def packed_parameter_names(self) -> tuple[str, ...]:
        """Names of the packed codes, block scales and row scales that replace the floating-point base projections."""
        return tuple(param for name in self._NVFP4_BASE_NAMES for param in self.nvfp4_parameter_names(name))

    @torch.no_grad()
    def _pack_base_weights_nvfp4(self) -> None:
        """Quantize materialized base projections and release their floating-point storage."""
        for name in self._NVFP4_BASE_NAMES:
            param = getattr(self, name)
            # Compute layout [experts, in, out] -> checkpoint layout [experts, out, in].
            packed, scales, global_scale = quantize_nvfp4(_to_local(param).transpose(-2, -1).contiguous())
            if isinstance(param, DTensor):
                packed = DTensor.from_local(packed, param.device_mesh, param.placements)
                scales = DTensor.from_local(scales, param.device_mesh, param.placements)
                global_scale = DTensor.from_local(global_scale, param.device_mesh, param.placements)
            self._register_nvfp4_base_weight(name, packed, scales, global_scale)

    def _nvfp4_base_mm(self, x: torch.Tensor, name: str, offs: torch.Tensor) -> torch.Tensor:
        """Multiply routed activations by a frozen packed base projection.

        Args:
            x: Tensor of shape [tokens, in_dim], grouped contiguously by local expert.
            name: Base projection parameter name.
            offs: Int32 tensor of shape [local_experts], holding cumulative token counts.

        Returns:
            Tensor of shape [tokens, out_dim] with the activation dtype and device.
        """
        packed_name, scales_name, global_name = self.nvfp4_parameter_names(name)
        packed = _to_local(getattr(self, packed_name))
        scales = _to_local(getattr(self, scales_name))
        global_scale = _to_local(getattr(self, global_name))
        return NVFP4GroupedMM.apply(x, packed, scales, global_scale, offs)

    def _nvfp4_dequant_expert0(self, name: str, dtype: torch.dtype) -> torch.Tensor:
        """Dequantize expert 0 of base weight ``name`` to compute layout ``[in, out]``."""
        packed_name, scales_name, global_name = self.nvfp4_parameter_names(name)
        packed = _to_local(getattr(self, packed_name))[0]
        scales = _to_local(getattr(self, scales_name))[0]
        global_scale = _to_local(getattr(self, global_name))[0]
        return dequantize_nvfp4(packed, scales, global_scale, dtype).transpose(-2, -1)


class GroupedExpertsNVFP4(NVFP4ExpertStorageMixin, GroupedExperts):
    """Frozen routed experts with NVFP4-resident base weights and no adapter.

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
        for name in self._NVFP4_BASE_NAMES:
            getattr(self, name).data = _to_local(getattr(orig_module, name)).clone()
        if self.expert_bias:
            self.gate_up_proj_bias.data = _to_local(orig_module.gate_up_proj_bias).clone()
            self.down_proj_bias.data = _to_local(orig_module.down_proj_bias).clone()
        self.requires_grad_(False)
        self._init_nvfp4_storage()

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

        if isinstance(self.gate_and_up_projs_nvfp4, DTensor):
            ep_mesh = self.gate_and_up_projs_nvfp4.device_mesh
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

        y = self._forward_grouped_mm_nvfp4(x, token_mask, weights, indices, n_local_experts, experts_start_idx)

        if ep_size > 1:
            # Keep the differentiable all-gather path attached to x without materializing a full-size zero tensor.
            y.add_(x.sum(dtype=torch.float32) * 0.0)

            # Reduce and narrow to the original per-rank token boundaries.
            y = dist_nn_f.all_reduce(y, op=dist.ReduceOp.SUM, group=ep_group)
            start = sum(gathered_lens[:ep_rank])
            y = y.narrow(0, start, local_num_tokens).contiguous()

        return y.to(input_dtype)

    def _forward_grouped_mm_nvfp4(
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

            output1 = self._nvfp4_base_mm(permuted_x, "gate_and_up_projs", offs)
            if self.expert_bias:
                output1 = _apply_bias(output1, _to_local(self.gate_up_proj_bias), tokens_per_expert)
            output1 = self.expert_activation_grouped(output1, permuted_probs)

            output2 = self._nvfp4_base_mm(output1, "down_projs", offs)
            if self.expert_bias:
                output2 = _apply_bias(output2, _to_local(self.down_proj_bias), tokens_per_expert, permuted_probs)

            scatter_ids = sorted_token_ids.unsqueeze(1).expand_as(output2)
            y.scatter_add_(0, scatter_ids, output2.float())
        else:
            # Dummy computation for gradient flow when no tokens routed locally.
            gate_up_w0 = self._nvfp4_dequant_expert0("gate_and_up_projs", x.dtype)
            down_w0 = self._nvfp4_dequant_expert0("down_projs", x.dtype)
            output1 = torch.matmul(x[0] * 0, gate_up_w0)
            output1_ = self.expert_activation_grouped(output1, weights[0, 0, None].unsqueeze(0))
            output2 = torch.matmul(output1_, down_w0)
            y[0] += output2[0]

        return y


class GroupedExpertsDeepEPNVFP4(NVFP4ExpertStorageMixin, GroupedExpertsDeepEP):
    """Frozen routed experts with NVFP4-resident base weights under DeepEP dispatch.

    Drop-in replacement for ``GroupedExpertsDeepEP`` when the experts are frozen
    (e.g. LoRA on attention only). The DeepEP fused all-to-all token dispatch is reused
    unchanged — NVFP4 only changes the two post-dispatch grouped GEMMs, which read the
    packed base weights via ``NVFP4GroupedMM`` instead of bf16 ``torch._grouped_mm``.

    The DeepEP parent always uses native grouped MM.
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
        self.dispatcher_hybridep_permute_fusion = orig_module.dispatcher_hybridep_permute_fusion
        self.dispatcher_hybridep_compact_routing = orig_module.dispatcher_hybridep_compact_routing
        self.dispatcher_hybridep_num_sms_preprocessing = orig_module.dispatcher_hybridep_num_sms_preprocessing
        self.dispatcher_hybridep_num_blocks_permute = orig_module.dispatcher_hybridep_num_blocks_permute
        self.dispatcher_hybridep_num_blocks_unpermute = orig_module.dispatcher_hybridep_num_blocks_unpermute
        self.dispatcher_hybridep_max_tokens_per_rank = orig_module.dispatcher_hybridep_max_tokens_per_rank
        # Sync-free dispatch knobs (BackendConfig.dispatcher_capacity_factor / dispatcher_equal_token_counts) travel
        # with the other HybridEP settings: the shared dispatcher is built from the first (wrapped) layer.
        self.dispatcher_capacity_factor = orig_module.dispatcher_capacity_factor
        self.dispatcher_equal_token_counts = orig_module.dispatcher_equal_token_counts
        self.dispatch_rows_never_empty = orig_module.dispatch_rows_never_empty
        # The combine cache (BackendConfig.dispatcher_recompute_cache) is read by the shared dispatcher's config as
        # well: a frozen layer next to a LoRA one must carry the same mode, or the shared-manager check rejects the
        # pair and an all-frozen conversion silently loses the request (review of #4395).
        self.dispatcher_recompute_cache = orig_module.dispatcher_recompute_cache
        # These fresh parameters have no autograd history or optimizer references.
        for name in self._NVFP4_BASE_NAMES:
            getattr(self, name).data = _to_local(getattr(orig_module, name)).clone()
        if self.expert_bias:
            self.gate_up_proj_bias.data = _to_local(orig_module.gate_up_proj_bias).clone()
            self.down_proj_bias.data = _to_local(orig_module.down_proj_bias).clone()
        self.requires_grad_(False)
        self._init_nvfp4_storage()

    def forward(
        self,
        x: torch.Tensor,
        token_mask: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """Forward over NVFP4 base weights with DeepEP dispatch.

        Preserves the tensor and EP contract of ``GroupedExpertsDeepEP.forward``, replacing the two base
        ``torch._grouped_mm`` calls with ``NVFP4GroupedMM`` over the packed weights.
        """
        assert not isinstance(x, DTensor)
        assert self.n_routed_experts % self.ep_size == 0, (
            f"Number of experts must be divisible by ep_size (ep_size={self.ep_size})"
        )

        permuted_local_hidden_states, tokens_per_expert, permuted_probs = self._dispatch_tokens(
            x, token_mask, weights, indices
        )

        # dispatch_rows_never_empty (static routing or capacity mode): skip the count_nonzero host read, as
        # GroupedExpertsDeepEP.forward does.
        if self.dispatch_rows_never_empty or torch.count_nonzero(tokens_per_expert) > 0:
            tokens_per_expert_gpu = tokens_per_expert.to(device=permuted_local_hidden_states.device, non_blocking=True)
            offs = tokens_per_expert_gpu.cumsum(dim=0).to(torch.int32)

            output1 = self._nvfp4_base_mm(permuted_local_hidden_states, "gate_and_up_projs", offs)
            if self.expert_bias:
                output1 = _apply_bias(output1, _to_local(self.gate_up_proj_bias), tokens_per_expert)
            output1 = self.expert_activation(output1, permuted_probs)
            output2 = self._nvfp4_base_mm(output1, "down_projs", offs)
            if self.expert_bias:
                output2 = _apply_bias(output2, _to_local(self.down_proj_bias), tokens_per_expert, permuted_probs)
        else:
            # Dummy computation for gradient flow when no tokens routed locally.
            gate_up_w0 = self._nvfp4_dequant_expert0("gate_and_up_projs", x.dtype)
            down_w0 = self._nvfp4_dequant_expert0("down_projs", x.dtype)
            output1 = torch.matmul(x[0] * 0, gate_up_w0)
            output1_ = self.expert_activation(output1, permuted_probs)
            output2 = torch.matmul(output1_, down_w0)

        y = self.token_dispatcher.token_unpermutation(output2)
        return y


def apply_nvfp4_to_moe_experts(model: nn.Module) -> nn.Module:
    """Apply NVFP4-resident storage to the model's common routed-expert modules.

    Call this after LoRA injection and before distributed sharding. LoRA-targeted
    experts that are already NVFP4-resident are preserved; remaining plain
    ``GroupedExperts`` and ``GroupedExpertsDeepEP`` modules are replaced in place.
    Transformer Engine experts are rejected: the packed storage implements the
    ``torch_mm`` experts backend (torch, DeepEP or HybridEP dispatch) only.

    Meta experts receive packed placeholders for direct checkpoint loading, and
    their model state-dict adapter must opt into the NVFP4 expert storage format.
    Already materialized expert weights are quantized without changing the adapter's
    checkpoint loading mode. Model-specific adapters own checkpoint layouts.

    Raises:
        NotImplementedError: If a Transformer Engine expert module is present, or the
            model state-dict adapter cannot select the NVFP4 expert storage format.
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
            "NVFP4-resident expert weights do not support Transformer Engine expert modules; use "
            f"backend.experts='torch_mm' (modules: {unsupported[:3]})."
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
                    "Loading an NVFP4 expert checkpoint requires the model state-dict adapter to implement "
                    "set_expert_storage_format()."
                )
            set_storage_format("nvfp4")

    frozen_conversions = {
        GroupedExperts: GroupedExpertsNVFP4,
        GroupedExpertsDeepEP: GroupedExpertsDeepEPNVFP4,
    }
    num_converted = 0
    for model_part in model_parts:
        for name, module in list(model_part.named_modules()):
            if isinstance(module, NVFP4ExpertStorageMixin):
                continue
            new_cls = frozen_conversions.get(type(module))
            if new_cls is None:
                continue
            new_module = new_cls(module)
            parent_name, _, child_name = name.rpartition(".")
            parent = model_part.get_submodule(parent_name) if parent_name else model_part
            setattr(parent, child_name, new_module)
            num_converted += 1

    logger.info("Applied NVFP4-resident storage to %d frozen expert module(s)", num_converted)
    return model


def packed_expert_parameters(model: nn.Module) -> set[nn.Parameter]:
    """Return the packed codes and scales of every mxfp4 / nvfp4 expert module in ``model``.

    Their storage dtype is the weight format, not a master-weight dtype, so callers pass them to the
    frozen-parameter casts as ``keep_storage``.

    Args:
        model: Model or model part, before or after distributed wrapping.

    Returns:
        The currently registered packed parameters; empty when ``model`` has no packed experts.
    """
    return {
        module.get_parameter(name)
        for module in model.modules()
        if isinstance(module, (MXFP4ExpertStorageMixin, NVFP4ExpertStorageMixin))
        for name in module.packed_parameter_names()
    }


def validate_nvfp4_checkpoint_config(checkpoint_dir: str) -> None:
    """Check a checkpoint's quantization metadata and expert tensor layout before its packed experts are loaded.

    ModelOpt writes ``hf_quant_config.json`` (``quantization.quant_algo`` / ``group_size``) and mirrors it in
    ``config.json``'s ``quantization_config``. The packed storage implements NVFP4 with 16-column blocks and
    experts-only quantization (the dense projections, attention, router, norms and embeddings must stay in
    floating point: they are loaded by the ordinary base load). The safetensors headers of a local checkpoint are
    then checked against the per-expert layout the state-dict adapters map onto the packed parameters: every
    routed-expert projection must carry ``weight`` (``U8`` or ``I8`` codes, [out, in / 2]), ``weight_scale``
    (``F8_E4M3``, [out, in / NVFP4_BLOCK_SIZE]) and a scalar ``F32`` ``weight_scale_2``. The checkpoint load
    enforces shapes only and copies with dtype conversion, so a wrongly typed export would otherwise load silently.
    A checkpoint that also quantized the dense layers fails later in the base load.

    Args:
        checkpoint_dir: Local HF checkpoint directory (a snapshot holding only the JSON files skips the header
            check).

    Raises:
        ValueError: If the metadata names another algorithm or block size, or an expert tensor has another dtype or
            an inconsistent shape.
    """
    quant_algo = group_size = None
    hf_quant = os.path.join(checkpoint_dir, "hf_quant_config.json")
    config_json = os.path.join(checkpoint_dir, "config.json")
    if os.path.exists(hf_quant):
        with open(hf_quant) as f:
            q = json.load(f).get("quantization", {})
        quant_algo, group_size = q.get("quant_algo"), q.get("group_size")
    elif os.path.exists(config_json):
        with open(config_json) as f:
            q = json.load(f).get("quantization_config") or {}
        quant_algo = q.get("quant_algo")
        groups = q.get("config_groups") or {}
        sizes = {g.get("weights", {}).get("group_size") for g in groups.values() if isinstance(g, dict)}
        group_size = sizes.pop() if len(sizes) == 1 else None
    if quant_algo is None and group_size is None:
        logger.warning(
            "%s has no hf_quant_config.json / quantization_config: cannot verify it is an NVFP4 (group 16) export",
            checkpoint_dir,
        )
        return
    if str(quant_algo).upper() != "NVFP4":
        raise ValueError(
            f"{checkpoint_dir}: expert_weight_format='nvfp4' needs a ModelOpt NVFP4 checkpoint, found quant_algo="
            f"{quant_algo!r}"
        )
    if group_size is not None and int(group_size) != NVFP4_BLOCK_SIZE:
        raise ValueError(
            f"{checkpoint_dir}: NVFP4 group_size={group_size} is not supported; the packed storage uses "
            f"{NVFP4_BLOCK_SIZE}-column blocks"
        )
    _validate_nvfp4_expert_tensor_headers(checkpoint_dir)


_NVFP4_EXPERT_TENSOR = re.compile(
    r"\.experts\.\d+\.(?:gate_proj|up_proj|down_proj)\.(weight|weight_scale|weight_scale_2)$"
)
_NVFP4_EXPERT_TENSOR_DTYPES = {"weight": ("U8", "I8"), "weight_scale": ("F8_E4M3",), "weight_scale_2": ("F32",)}


def _validate_nvfp4_expert_tensor_headers(checkpoint_dir: str) -> None:
    """Check the dtype and shape of every routed-expert tensor in the checkpoint's safetensors headers."""
    files = sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors")))
    if not files:
        return
    # "<...>.experts.{E}.{proj}" -> {suffix: shape}
    projections: dict[str, dict[str, list[int]]] = {}
    for file_name in files:
        with open(file_name, "rb") as stream:
            header_size = int.from_bytes(stream.read(8), "little")
            header = json.loads(stream.read(header_size))
        for key, entry in header.items():
            match = _NVFP4_EXPERT_TENSOR.search(key)
            if match is None:
                continue
            suffix = match.group(1)
            dtype = entry.get("dtype")
            if dtype not in _NVFP4_EXPERT_TENSOR_DTYPES[suffix]:
                raise ValueError(
                    f"{key}: expected dtype {' or '.join(_NVFP4_EXPERT_TENSOR_DTYPES[suffix])} for a ModelOpt NVFP4 "
                    f"expert {suffix}, found {dtype!r}; expert_weight_format='nvfp4' needs an experts-only NVFP4 export"
                )
            projections.setdefault(key[: -len(suffix) - 1], {})[suffix] = list(entry.get("shape", []))
    if not projections:
        raise ValueError(
            f"{checkpoint_dir}: no routed-expert weight / weight_scale / weight_scale_2 tensors found; "
            "expert_weight_format='nvfp4' needs an experts-only ModelOpt NVFP4 export"
        )
    for name, shapes in projections.items():
        missing = [suffix for suffix in _NVFP4_EXPERT_TENSOR_DTYPES if suffix not in shapes]
        if missing:
            raise ValueError(f"{name}: missing {missing} in the checkpoint (ModelOpt NVFP4 layout)")
        weight, scales, tensor_scale = shapes["weight"], shapes["weight_scale"], shapes["weight_scale_2"]
        if (
            len(weight) != 2
            or len(scales) != 2
            or scales[0] != weight[0]
            or scales[1] * NVFP4_BLOCK_SIZE != weight[1] * 2
        ):
            raise ValueError(
                f"{name}: weight_scale shape {scales} does not match packed weight shape {weight} "
                f"(one float8_e4m3fn scale per {NVFP4_BLOCK_SIZE} columns)"
            )
        if math.prod(tensor_scale) != 1:
            raise ValueError(f"{name}: weight_scale_2 must be a scalar, found shape {tensor_scale}")
