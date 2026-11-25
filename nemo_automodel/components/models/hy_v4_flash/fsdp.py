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

"""FSDP policy of the Hy4-flash decoder root.

The decoder blocks are sharded by dtype by the shared MoE parallelizer, but the root ``HY4FlashModel`` also owns the
fp32 output hyper-connection (``hc_head``) next to the bf16 final norm. FSDP requires one parameter dtype per unit and
would cast ``hc_head`` to the bf16 policy dtype, so ``hc_head`` gets its own fp32 unit before the root is sharded.
"""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy, OffloadPolicy, fully_shard

from nemo_automodel.components.distributed import ModelParallelizer
from nemo_automodel.components.models.hy_v4_flash.layers import HY4FlashHyperHead


class HY4FlashModelParallelizer(ModelParallelizer):
    """Give the fp32 output hyper-connection a dtype-uniform FSDP unit."""

    _customizes_moe_fsdp = True

    def _fully_shard_module(
        self,
        module: nn.Module,
        *,
        mesh: DeviceMesh,
        mp_policy: MixedPrecisionPolicy,
        offload_policy: OffloadPolicy | None = None,
        reshard_after_forward: bool | int | None = None,
        ignored_params: set[nn.Parameter] | None = None,
    ) -> nn.Module:
        """Shard ``module`` in place, first isolating a directly owned ``hc_head``.

        Args:
            module: Module to shard.
            mesh: Runtime FSDP device mesh.
            mp_policy: Mixed-precision policy of the module's bf16 parameters.
            offload_policy: Parameter/gradient offload policy.
            reshard_after_forward: PyTorch FSDP resharding setting.
            ignored_params: Parameters (arbitrary shapes) already managed outside this unit.

        Returns:
            The input module with FSDP applied.
        """
        options = {} if offload_policy is None else {"offload_policy": offload_policy}
        if ignored_params:
            options["ignored_params"] = ignored_params
        head = getattr(module, "hc_head", None)
        if isinstance(head, HY4FlashHyperHead):
            # Internal unit: fp32 parameters, activations keep the dtype the module produces.
            fully_shard(
                head,
                mesh=mesh,
                mp_policy=replace(mp_policy, param_dtype=torch.float32, output_dtype=None, cast_forward_inputs=False),
                reshard_after_forward=reshard_after_forward,
                **options,
            )
        return fully_shard(
            module, mesh=mesh, mp_policy=mp_policy, reshard_after_forward=reshard_after_forward, **options
        )


PARALLELIZER = HY4FlashModelParallelizer()

__all__ = ["PARALLELIZER", "HY4FlashModelParallelizer"]
