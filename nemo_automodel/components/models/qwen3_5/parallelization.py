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

import copy
import logging
from collections.abc import Callable

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy, OffloadPolicy, fully_shard

from nemo_automodel.components.distributed import ModelParallelizer
from nemo_automodel.components.distributed.fsdp2_extensions.compute_dtype import fully_shard_with_compute_dtype_fallback
from nemo_automodel.components.distributed.fsdp2_extensions.replicated import (
    DEFAULT_MAX_REPLICATED_PARAM_BYTES_PER_MODULE,
    make_fully_shard_with_replicated_parameter_grad_sync,
    replicated_parameters,
    select_small_fp32_parameters,
)
from nemo_automodel.components.distributed.multimodal_fsdp import FrozenMultimodalSharding
from nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn import CPAwareGatedDeltaNet

logger = logging.getLogger(__name__)
_FP32_COMPUTE_MODULE_NAMES = ("_fp32_params",)
_MAX_REPLICATED_FP32_BYTES_PER_MODULE = DEFAULT_MAX_REPLICATED_PARAM_BYTES_PER_MODULE


class Qwen3_5ModelParallelizer(ModelParallelizer):
    """Preserve Qwen3.5's small FP32 GatedDeltaNet parameter islands."""

    _shard_module: Callable[..., nn.Module] | None = None

    def _fully_shard_module(self, module: nn.Module, **kwargs) -> nn.Module:
        """Materialize per-parameter compute dtypes while retaining layer ownership."""
        return fully_shard_with_compute_dtype_fallback(
            module,
            fp32_compute_module_names=_FP32_COMPUTE_MODULE_NAMES,
            fully_shard_fn=self._shard_module or fully_shard,
            **kwargs,
        )

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
        frozen_multimodal_sharding: FrozenMultimodalSharding = "root",
        ignored_multimodal_params: set[nn.Parameter] | None = None,
    ) -> None:
        """Select replication after TP and trainability changes, then use the shared layer walk.

        Args:
            module: Model containing parameter tensors of arbitrary shape. TP
                DTensors keep global shapes and TP placements.
            mesh: FSDP or HSDP mesh owning the remaining parameter shards.
            mp_policy: Compute and gradient-reduction policy.
            offload_policy: Optional CPU offload policy.
            enable_fsdp2_prefetch: Whether to prefetch neighboring layer units.
            fsdp2_backward_prefetch_depth: Number of backward prefetch targets.
            fsdp2_forward_prefetch_depth: Number of forward prefetch targets.
            reshard_after_forward: Optional reshard override.
            frozen_multimodal_sharding: Ownership of frozen multimodal towers.
            ignored_multimodal_params: Parameter tensors of arbitrary shape that
                ancestor FSDP units must ignore; updated in place with replication.
        """
        selections = (
            select_small_fp32_parameters(
                module,
                name_fragments=_FP32_COMPUTE_MODULE_NAMES,
                max_bytes_per_module=_MAX_REPLICATED_FP32_BYTES_PER_MODULE,
            )
            if mp_policy is not None and mp_policy.param_dtype not in (None, torch.float32)
            else ()
        )
        parameters = replicated_parameters(selections)
        if parameters:
            self._shard_module = make_fully_shard_with_replicated_parameter_grad_sync(
                module, parameters, mesh, fully_shard_fn=fully_shard
            )
            if ignored_multimodal_params is None:
                raise ValueError("Qwen3.5 replication requires the ancestor ignored-parameter accumulator.")
            ignored_multimodal_params.update(parameters)
        if selections:
            logger.info(
                "Qwen3.5 FP32 holders: %d module(s) replicated outside FSDP (%d bytes), %d kept sharded",
                sum(selection.replicated for selection in selections),
                sum(selection.logical_bytes for selection in selections if selection.replicated),
                sum(not selection.replicated for selection in selections),
            )
            non_fp32 = [selection.name for selection in selections if selection.sharded_reason == "non_fp32_residency"]
            if non_fp32:
                logger.warning(
                    "Keeping precision-sensitive Qwen3.5 modules sharded because resident weights are not FP32: %s",
                    ", ".join(non_fp32),
                )
        super()._apply_fsdp_sharding(
            module,
            mesh,
            mp_policy,
            offload_policy,
            enable_fsdp2_prefetch,
            fsdp2_backward_prefetch_depth,
            fsdp2_forward_prefetch_depth,
            reshard_after_forward,
            frozen_multimodal_sharding=frozen_multimodal_sharding,
            ignored_multimodal_params=ignored_multimodal_params,
        )

    def _apply(self, model: nn.Module, device_mesh: DeviceMesh, **kwargs) -> nn.Module:
        """Apply the shared flow with invocation-local sharding state and Qwen CP wiring."""
        # The class-owned sidecar is shared by all model instances. A per-call
        # copy prevents one model's replicated-gradient callback owning another.
        parallelizer = copy.copy(self)
        parallelizer._shard_module = None
        result = super(Qwen3_5ModelParallelizer, parallelizer)._apply(model, device_mesh, **kwargs)
        cp_mesh_name = kwargs.get("dp_shard_cp_mesh_name", "dp_shard_cp").replace("dp_shard_", "")
        if cp_mesh_name in device_mesh.mesh_dim_names and device_mesh[cp_mesh_name].size() > 1:
            cp_mesh = device_mesh[cp_mesh_name]
            for module in model.modules():
                if isinstance(module, CPAwareGatedDeltaNet):
                    module._cp_mesh = cp_mesh
            model.cp_mesh = cp_mesh
        return result


PARALLELIZER = Qwen3_5ModelParallelizer()

__all__ = ["PARALLELIZER"]
