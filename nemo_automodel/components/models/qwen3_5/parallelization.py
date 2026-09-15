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

"""Qwen3.5 dense parallelization: TP plan and mixed-dtype FSDP2 strategy."""

from __future__ import annotations

import logging

from torch import nn

import nemo_automodel.components.distributed.parallelizer as parallelizer
import nemo_automodel.components.distributed.parallelizer_utils as parallelizer_utils
from nemo_automodel.components.distributed.parallel_spec import ParallelSpec
from nemo_automodel.components.distributed.parallelizer import DefaultParallelizationStrategy
from nemo_automodel.shared.multimodal_fsdp import (
    is_multimodal_module_name,
    module_is_fully_frozen,
    module_parameters,
    normalize_frozen_multimodal_sharding,
)

logger = logging.getLogger(__name__)


class Qwen3_5ParallelizationStrategy(DefaultParallelizationStrategy):
    """Parallelization strategy for Qwen3.5 dense models with mixed-dtype GatedDeltaNet.

    Qwen3.5 has linear_attn layers with float32 params (A_log, norm) alongside
    bfloat16 params. Overrides the FSDP sharding step to use fully_shard_by_dtype
    per layer, and sets the CP mesh on CPAwareGatedDeltaNet modules.
    """

    def parallelize(self, model, device_mesh, dp_shard_cp_mesh_name="dp_shard_cp", **kwargs):
        cp_mesh_name = dp_shard_cp_mesh_name.replace("dp_shard_", "")
        cp_enabled = cp_mesh_name in device_mesh.mesh_dim_names and device_mesh[cp_mesh_name].size() > 1

        # The Qwen3.5 model builds CPAwareGatedDeltaNet with a fp32 ``SSMGate``
        # (``_fp32_params``) at construction — no runtime patch needed. Keep those
        # params in their own dtype-uniform fp32 FSDP group (true master weights).
        fp32_compute_module_names = ("_fp32_params",)

        # Delegate TP, AC, mixed precision to the default strategy, but
        # override the FSDP sharding to use fully_shard_by_dtype.
        # Temporarily swap the global — safe because model init is single-threaded
        # (one model is parallelized at a time). Not safe under concurrent calls.
        original_fn = parallelizer.apply_fsdp2_sharding_recursively

        def _fsdp_by_dtype(
            module,
            mesh,
            mp_policy,
            offload_policy=None,
            enable_fsdp2_prefetch=True,
            fsdp2_backward_prefetch_depth=2,
            fsdp2_forward_prefetch_depth=1,
            reshard_after_forward=None,
            fully_shard_fn=None,
            frozen_multimodal_sharding="root",
            ignored_multimodal_params=None,
        ):
            del enable_fsdp2_prefetch, fsdp2_backward_prefetch_depth, fsdp2_forward_prefetch_depth, fully_shard_fn
            frozen_multimodal_sharding = normalize_frozen_multimodal_sharding(frozen_multimodal_sharding)
            pp_enabled = "pp" in mesh.mesh_dim_names and mesh["pp"].size() > 1

            if isinstance(module, (nn.ModuleList, nn.ModuleDict)):
                all_items = list(module.items()) if isinstance(module, nn.ModuleDict) else list(enumerate(module))
                flat_layer_items = [
                    (layer_id, child)
                    for layer_id, child in all_items
                    if not isinstance(child, (nn.ModuleList, nn.ModuleDict))
                ]
                nested_items = [
                    (layer_id, child)
                    for layer_id, child in all_items
                    if isinstance(child, (nn.ModuleList, nn.ModuleDict))
                ]

                for _, child in nested_items:
                    _fsdp_by_dtype(
                        child,
                        mesh,
                        mp_policy,
                        offload_policy,
                        reshard_after_forward=reshard_after_forward,
                        frozen_multimodal_sharding=frozen_multimodal_sharding,
                        ignored_multimodal_params=ignored_multimodal_params,
                    )

                for enum_id, (_, child) in enumerate(flat_layer_items):
                    if reshard_after_forward is not None:
                        layer_reshard_after_forward = reshard_after_forward
                    elif pp_enabled:
                        layer_reshard_after_forward = False
                    else:
                        layer_reshard_after_forward = enum_id < len(flat_layer_items) - 1
                    parallelizer_utils.fully_shard_by_dtype(
                        child,
                        mesh,
                        mp_policy,
                        offload_policy,
                        fp32_compute_module_names=fp32_compute_module_names,
                        reshard_after_forward=layer_reshard_after_forward,
                    )
            else:
                for name, sub in module.named_children():
                    if is_multimodal_module_name(name) and module_is_fully_frozen(sub):
                        if frozen_multimodal_sharding in ("root", "replicate"):
                            logger.info(
                                "Keeping frozen multimodal module %s at FSDP policy %s",
                                name,
                                frozen_multimodal_sharding,
                            )
                            if frozen_multimodal_sharding == "replicate" and ignored_multimodal_params is not None:
                                ignored_multimodal_params.update(module_parameters(sub))
                            continue
                    _fsdp_by_dtype(
                        sub,
                        mesh,
                        mp_policy,
                        offload_policy,
                        reshard_after_forward=reshard_after_forward,
                        frozen_multimodal_sharding=frozen_multimodal_sharding,
                        ignored_multimodal_params=ignored_multimodal_params,
                    )

        parallelizer.apply_fsdp2_sharding_recursively = _fsdp_by_dtype
        try:
            result = super().parallelize(
                model,
                device_mesh,
                dp_shard_cp_mesh_name=dp_shard_cp_mesh_name,
                **kwargs,
            )
        finally:
            parallelizer.apply_fsdp2_sharding_recursively = original_fn

        # Set CP mesh on CPAwareGatedDeltaNet modules
        if cp_enabled:
            from nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn import CPAwareGatedDeltaNet

            cp_mesh = device_mesh[cp_mesh_name]
            for _, mod in model.named_modules():
                if isinstance(mod, CPAwareGatedDeltaNet):
                    mod._cp_mesh = cp_mesh
            # Hand the CP submesh to the model so a forward that embeds and
            # sequence-shards its own primary stream (Megatron-style per-microbatch
            # CP; see shard_sequence_for_cp_round_robin / shard_batch_aux_only) can build this
            # rank's round-robin shard.
            model.cp_mesh = cp_mesh

        return result


QWEN3_5_LAYERS = {"language": ("model.language_model.layers",), "vision": ("model.visual.blocks",)}
# The transformers class and this native port share name, plan and strategy.
# No declared TP plan: Qwen3.5 mixes full self_attn (every 4th layer) with GatedDeltaNet ``linear_attn``, which
# is not TP-shardable with stock kernels, so the loader translates transformers' own ``base_model_tp_plan``
# (self_attn + MLP only).
QWEN3_5_VLM_PARALLEL_SPEC = ParallelSpec(
    layer_groups=QWEN3_5_LAYERS,
    hf_tp_plan_prefix=("model.language_model",),
    strategy=Qwen3_5ParallelizationStrategy(),
)
QWEN3_5_CAUSAL_LM_PARALLEL_SPEC = ParallelSpec(strategy=Qwen3_5ParallelizationStrategy())
