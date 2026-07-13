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

"""Qwen3.5-specific distributed parallelization."""

from __future__ import annotations

from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy, OffloadPolicy

import nemo_automodel.components.distributed.parallelizer_utils as parallelizer_utils
from nemo_automodel.components.distributed import parallelizer
from nemo_automodel.components.distributed.tp_plan import get_hf_tp_shard_plan


class Qwen3_5ParallelizationStrategy(parallelizer.DefaultParallelizationStrategy):
    """Parallelize Qwen3.5's mixed-dtype GatedDeltaNet layers correctly."""

    _FP32_COMPUTE_MODULE_NAMES = ("_fp32_params",)

    def _shard_modules_recursively(
        self,
        module: nn.Module,
        mesh: DeviceMesh,
        mp_policy: MixedPrecisionPolicy | None,
        offload_policy: OffloadPolicy | None = None,
        enable_fsdp2_prefetch: bool = True,
        fsdp2_backward_prefetch_depth: int = 2,
        fsdp2_forward_prefetch_depth: int = 1,
        reshard_after_forward: bool | None = None,
    ) -> None:
        """Shard layers by dtype so fp32 GatedDeltaNet state stays isolated."""
        del enable_fsdp2_prefetch, fsdp2_backward_prefetch_depth, fsdp2_forward_prefetch_depth
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
                self._shard_modules_recursively(
                    child,
                    mesh,
                    mp_policy,
                    offload_policy,
                    reshard_after_forward=reshard_after_forward,
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
                    fp32_compute_module_names=self._FP32_COMPUTE_MODULE_NAMES,
                    reshard_after_forward=layer_reshard_after_forward,
                )
            return

        for _, sub in module.named_children():
            self._shard_modules_recursively(
                sub,
                mesh,
                mp_policy,
                offload_policy,
                reshard_after_forward=reshard_after_forward,
            )

    def parallelize(
        self,
        model: nn.Module,
        device_mesh: DeviceMesh,
        dp_shard_cp_mesh_name: str = "dp_shard_cp",
        **kwargs: object,
    ) -> nn.Module:
        """Apply Qwen3.5-specific FSDP and context-parallel setup."""
        cp_mesh_name = dp_shard_cp_mesh_name.replace("dp_shard_", "")
        cp_enabled = cp_mesh_name in device_mesh.mesh_dim_names and device_mesh[cp_mesh_name].size() > 1

        result = super().parallelize(
            model,
            device_mesh,
            dp_shard_cp_mesh_name=dp_shard_cp_mesh_name,
            **kwargs,
        )

        if cp_enabled:
            from nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn import CPAwareGatedDeltaNet

            cp_mesh = device_mesh[cp_mesh_name]
            for _, mod in model.named_modules():
                if isinstance(mod, CPAwareGatedDeltaNet):
                    mod._cp_mesh = cp_mesh

        return result


_STRATEGY = Qwen3_5ParallelizationStrategy()


def get_parallelization_strategy() -> parallelizer.ParallelizationStrategy:
    """Return the lazily requested Qwen3.5 strategy instance."""
    return _STRATEGY


def get_tp_plan(model: nn.Module, *, sequence_parallel: bool = False):
    """Return Qwen3.5's native Hugging Face tensor-parallel plan."""
    del sequence_parallel
    return get_hf_tp_shard_plan(model)
