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

"""Model-owned FSDP precision and activation-checkpointing policy."""

from dataclasses import replace

from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, OffloadPolicy, fully_shard

from nemo_automodel.components.distributed.parallelizer import ModelParallelizer
from nemo_automodel.components.distributed.parallelizer_utils import fully_shard_by_dtype


class MistralLarge4Parallelizer(ModelParallelizer):
    """Use existing FSDP machinery while preserving Large 4 precision and sharing.

    Expert ownership and dispatch use the common MoE parallelizer.
    """

    _customizes_moe_fsdp = True

    def _apply(self, model: nn.Module, device_mesh: DeviceMesh, **kwargs) -> nn.Module:
        """Apply the inherited FSDP flow and retain the CP token mesh.

        Args:
            model: Native full model or pruned PP stage.
            device_mesh: Resolved shared mesh, including its named CP axis.
            **kwargs: Existing shared FSDP/AC/compile policies, passed unchanged.

        Returns:
            Parallelized model with its optional CP mesh.
        """
        result = super()._apply(model, device_mesh, **kwargs)
        cp_mesh = device_mesh["cp"] if "cp" in device_mesh.mesh_dim_names else None
        model.cp_mesh = cp_mesh if cp_mesh is not None and cp_mesh.size() > 1 else None
        return result

    def _use_full_layer_activation_checkpointing(self, model: nn.Module) -> bool:
        """Shared KV and index selections are explicit immutable block inputs."""
        return True

    def _fully_shard_module(
        self,
        module: nn.Module,
        *,
        mesh: DeviceMesh,
        mp_policy: MixedPrecisionPolicy | None = None,
        offload_policy: OffloadPolicy | None = None,
        reshard_after_forward: bool | int | None = None,
        ignored_params: set[nn.Parameter] | None = None,
    ) -> nn.Module:
        """Apply existing dtype-aware FSDP grouping to a model-owned boundary.

        Args:
            module: Parameter-owning module; its matrices have shape [output,
                input], vectors [features]. Existing nested FSDP units retain ownership.
            mesh: One-dimensional DP shard mesh or [replicate, shard] HSDP mesh.
            mp_policy: Requested compute policy; sensitive parameters always use FP32.
            offload_policy: Existing FSDP storage policy.
            reshard_after_forward: Existing FSDP reshard setting.
            ignored_params: Optional parameter set with the module's tensor layouts;
                these parameters retain their existing ownership and placements.

        Returns:
            Same module, with global-shaped DTensor parameters sharded on their
            leading axis; HSDP replicates across its first mesh dimension.
        """
        # FP32 LeRoPE tables and router weights cross unit boundaries unchanged.
        if mp_policy is not None:
            mp_policy = replace(mp_policy, cast_forward_inputs=False, output_dtype=None)
        # The generic dtype grouper traverses descendants, including existing
        # FSDP units. Exclude their parameters before grouping this boundary.
        owned = set(ignored_params or ())
        for child in module.modules():
            if child is not module and isinstance(child, FSDPModule):
                owned.update(child.parameters())
        fully_shard_by_dtype(
            module,
            mesh,
            mp_policy,
            offload_policy,
            fp32_compute_module_names=("rotary_emb", "indexer.k_norm", "mlp.gate.weight"),
            reshard_after_forward=reshard_after_forward,
            ignored_params=owned or None,
        )
        # A root with all parameters in nested units still owns the FSDP lifecycle.
        if not isinstance(module, FSDPModule):
            fully_shard(
                module,
                mesh=mesh,
                mp_policy=mp_policy or MixedPrecisionPolicy(),
                offload_policy=offload_policy,
                reshard_after_forward=reshard_after_forward,
                ignored_params=owned or None,
            )
        return module
