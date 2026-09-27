# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Qwen3.5-owned FSDP2 precision and context-parallel policy."""

from __future__ import annotations

import logging

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy, OffloadPolicy, fully_shard

import nemo_automodel.components.distributed.fsdp2_extensions.utils as parallelizer_utils
from nemo_automodel.components.distributed.fsdp2_extensions.replicated import (
    DEFAULT_MAX_REPLICATED_PARAM_BYTES_PER_MODULE,
    make_fully_shard_with_replicated_parameter_grad_sync,
    replicated_parameters,
    select_small_fp32_parameters,
)
from nemo_automodel.components.distributed.mesh_utils import get_fsdp_dp_mesh
from nemo_automodel.components.distributed.parallelizer import (
    PARALLELIZATION_STRATEGIES,
    DefaultParallelizationStrategy,
)
from nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn import CPAwareGatedDeltaNet
from nemo_automodel.shared.multimodal_fsdp import (
    FrozenMultimodalSharding,
    ignored_params_for_root,
    is_multimodal_module_name,
    module_is_fully_frozen,
    module_parameters,
    normalize_frozen_multimodal_sharding,
)

logger = logging.getLogger(__name__)

_FP32_COMPUTE_MODULE_NAMES = ("_fp32_params",)
_MAX_REPLICATED_FP32_BYTES_PER_MODULE = DEFAULT_MAX_REPLICATED_PARAM_BYTES_PER_MODULE
_QWEN3_5_MODEL_NAMES = ("Qwen3_5ForConditionalGeneration", "Qwen3_5ForCausalLM")


class Qwen3_5ParallelizationStrategy(DefaultParallelizationStrategy):
    """Preserve Qwen3.5's small FP32 GatedDeltaNet parameter islands."""

    def _apply_fsdp_sharding(
        self,
        module: nn.Module,
        mesh: DeviceMesh,
        mp_policy: MixedPrecisionPolicy | None,
        offload_policy: OffloadPolicy | None = None,
        enable_fsdp2_prefetch: bool = True,
        fsdp2_backward_prefetch_depth: int = 2,
        fsdp2_forward_prefetch_depth: int = 1,
        reshard_after_forward: bool | None = None,
        fully_shard_fn=None,
        frozen_multimodal_sharding: FrozenMultimodalSharding = "root",
        ignored_multimodal_params: set[nn.Parameter] | None = None,
    ) -> None:
        """Shard decoder layers by storage/compute dtype under Qwen ownership."""
        del enable_fsdp2_prefetch, fsdp2_backward_prefetch_depth, fsdp2_forward_prefetch_depth
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
                (layer_id, child) for layer_id, child in all_items if isinstance(child, (nn.ModuleList, nn.ModuleDict))
            ]

            for _, child in nested_items:
                self._apply_fsdp_sharding(
                    child,
                    mesh,
                    mp_policy,
                    offload_policy,
                    reshard_after_forward=reshard_after_forward,
                    fully_shard_fn=fully_shard_fn,
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
                ignored_params = ignored_params_for_root(child, ignored_multimodal_params or set())
                parallelizer_utils.fully_shard_by_dtype(
                    child,
                    mesh,
                    mp_policy,
                    offload_policy,
                    fp32_compute_module_names=_FP32_COMPUTE_MODULE_NAMES,
                    reshard_after_forward=layer_reshard_after_forward,
                    ignored_params=ignored_params,
                    fully_shard_fn=fully_shard_fn,
                )
        else:
            for name, submodule in module.named_children():
                if is_multimodal_module_name(name) and module_is_fully_frozen(submodule):
                    if frozen_multimodal_sharding in ("root", "replicate"):
                        logger.info(
                            "Keeping frozen multimodal module %s at FSDP policy %s",
                            name,
                            frozen_multimodal_sharding,
                        )
                        if frozen_multimodal_sharding == "replicate" and ignored_multimodal_params is not None:
                            ignored_multimodal_params.update(module_parameters(submodule))
                        continue
                self._apply_fsdp_sharding(
                    submodule,
                    mesh,
                    mp_policy,
                    offload_policy,
                    reshard_after_forward=reshard_after_forward,
                    fully_shard_fn=fully_shard_fn,
                    frozen_multimodal_sharding=frozen_multimodal_sharding,
                    ignored_multimodal_params=ignored_multimodal_params,
                )

    def parallelize(self, model, device_mesh, dp_shard_cp_mesh_name="dp_shard_cp", **kwargs):
        """Apply Qwen3.5's bounded FP32 replication and CP wiring."""
        cp_mesh_name = dp_shard_cp_mesh_name.replace("dp_shard_", "")
        cp_enabled = cp_mesh_name in device_mesh.mesh_dim_names and device_mesh[cp_mesh_name].size() > 1

        mp_policy = kwargs.get("mp_policy") or MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            output_dtype=torch.float32,
        )
        kwargs["mp_policy"] = mp_policy
        selections = (
            select_small_fp32_parameters(
                model,
                name_fragments=_FP32_COMPUTE_MODULE_NAMES,
                max_bytes_per_module=_MAX_REPLICATED_FP32_BYTES_PER_MODULE,
            )
            if mp_policy.param_dtype not in (None, torch.float32)
            else ()
        )
        replicated_params = replicated_parameters(selections)

        base_fully_shard_fn = kwargs.pop("fully_shard_fn", fully_shard)
        if replicated_params:
            dp_replicate_mesh_name = kwargs.get("dp_replicate_mesh_name", "dp_replicate")
            dp_mesh = get_fsdp_dp_mesh(device_mesh, dp_replicate_mesh_name, dp_shard_cp_mesh_name)
            base_fully_shard_fn = make_fully_shard_with_replicated_parameter_grad_sync(
                model,
                replicated_params,
                dp_mesh,
                fully_shard_fn=base_fully_shard_fn,
            )

        result = super().parallelize(
            model,
            device_mesh,
            dp_shard_cp_mesh_name=dp_shard_cp_mesh_name,
            fully_shard_fn=base_fully_shard_fn,
            replicated_params=set(replicated_params),
            **kwargs,
        )

        if selections:
            replicated = [selection for selection in selections if selection.replicated]
            oversized = [selection for selection in selections if selection.sharded_reason == "size_limit"]
            logger.info(
                "Qwen3.5 FP32 holders: %d module(s) replicated outside FSDP (%d trainable parameters, %d bytes; "
                "gradients use one coalesced FP32 all-reduce), %d kept sharded above the %d-byte limit%s",
                len(replicated),
                sum(parameter.requires_grad for parameter in replicated_params),
                sum(selection.logical_bytes for selection in replicated),
                len(oversized),
                _MAX_REPLICATED_FP32_BYTES_PER_MODULE,
                ": " + ", ".join(f"{selection.name}={selection.logical_bytes} bytes" for selection in oversized)
                if oversized
                else "",
            )
            non_fp32 = [selection.name for selection in selections if selection.sharded_reason == "non_fp32_residency"]
            if non_fp32:
                logger.warning(
                    "Keeping %d Qwen3.5 precision-sensitive module(s) sharded because their resident weights are not "
                    "FP32; loading them below FP32 may already have lost precision: %s",
                    len(non_fp32),
                    ", ".join(non_fp32),
                )

        if cp_enabled:
            cp_mesh = device_mesh[cp_mesh_name]
            for module in model.modules():
                if isinstance(module, CPAwareGatedDeltaNet):
                    module._cp_mesh = cp_mesh
            model.cp_mesh = cp_mesh

        return result


def register_qwen3_5_parallel_strategy() -> None:
    """Register the dense Qwen3.5 strategy once for both native model classes."""
    for name in _QWEN3_5_MODEL_NAMES:
        existing = PARALLELIZATION_STRATEGIES.get(name)
        if existing is None:
            PARALLELIZATION_STRATEGIES[name] = Qwen3_5ParallelizationStrategy()
        elif not isinstance(existing, Qwen3_5ParallelizationStrategy):
            raise RuntimeError(f"parallelization strategy {name} is already registered by {type(existing).__name__}")
