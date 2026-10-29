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

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, OffloadPolicy, fully_shard

from nemo_automodel.components.distributed.config import FSDP2Config
from nemo_automodel.components.distributed.mesh import MeshContext
from nemo_automodel.components.distributed.parallelizer import ModelParallelizer
from nemo_automodel.components.distributed.parallelizer_utils import fully_shard_by_dtype
from nemo_automodel.components.moe.layers import MoE


class MistralLarge4Parallelizer(ModelParallelizer):
    """Use existing FSDP machinery while preserving Large 4 precision and sharing.

    Expert ownership and dispatch use the common MoE parallelizer.
    PP composes with EP and contiguous CP; tensor parallelism remains disabled.
    """

    _customizes_moe_fsdp = True

    def parallelize(self, model: nn.Module, mesh_context: MeshContext, /) -> nn.Module:
        """Validate the requested topology before changing parameter ownership."""
        if not isinstance(mesh_context.strategy_config, FSDP2Config):
            raise ValueError("Mistral Large 4 distributed execution currently requires FSDP2")
        grouped = [module for module in model.modules() if isinstance(module, MoE)]
        if mesh_context.pp_size > 1:
            if hasattr(model, "state_dict_adapter"):
                model.state_dict_adapter.pipeline_parallel_mesh = mesh_context.device_mesh["pp"]
        if mesh_context.pp_size > 1 or grouped:
            # PP carries FP32 LeRoPE tables; grouped experts receive FP32 router weights.
            # Preserve each tensor's dtype across both root and block boundaries.
            policy = replace(mesh_context.strategy_config.mp_policy, cast_forward_inputs=False, output_dtype=None)
            mesh_context = replace(
                mesh_context, strategy_config=replace(mesh_context.strategy_config, mp_policy=policy)
            )
        if any(module.backend.dispatcher == "hybridep" for module in grouped) and mesh_context.ep_size <= 1:
            raise ValueError("Mistral HybridEP requires ep_size > 1; use dispatcher='torch' for a single EP rank")
        config = mesh_context.strategy_config
        if grouped and mesh_context.moe_parallel_config is not None:
            moe_policy = mesh_context.moe_parallel_config.mp_policy
            if moe_policy is not None:
                mesh_context = replace(
                    mesh_context,
                    moe_parallel_config=replace(
                        mesh_context.moe_parallel_config,
                        mp_policy=replace(moe_policy, cast_forward_inputs=False, output_dtype=None),
                    ),
                )
        if any(module.backend.dispatcher == "hybridep" for module in grouped):
            policy = (
                mesh_context.moe_parallel_config.mp_policy
                if mesh_context.moe_parallel_config is not None
                and mesh_context.moe_parallel_config.mp_policy is not None
                else config.mp_policy
            )
            if policy.param_dtype not in (None, torch.bfloat16):
                raise ValueError("Mistral HybridEP requires BF16 expert compute in the FSDP policy")
        if config.sequence_parallel or config.enable_compile:
            raise ValueError("Mistral Large 4 FSDP baseline requires sequence_parallel=False and enable_compile=False")
        if mesh_context.activation_checkpointing not in (False, True, "full"):
            raise ValueError("Mistral Large 4 supports full activation checkpointing only")
        if mesh_context.ep_size > 1:
            if not hasattr(model, "state_dict_adapter"):
                raise ValueError("Large 4 EP requires enable_hf_state_dict_adapter=True")
            if mesh_context.dp_replicate_size not in (None, 1):
                raise ValueError("Mistral Large 4 EP does not yet compose with HSDP replication")
            ep_mesh = mesh_context.moe_mesh["ep"]
            if any(module.n_routed_experts % ep_mesh.size() for module in grouped):
                raise ValueError("The expert count must be divisible by ep_size")
            model.state_dict_adapter.expert_parallel_mesh = ep_mesh
        if mesh_context.cp_size > 1:
            model.cp_mesh = mesh_context.device_mesh["cp"]
        return super().parallelize(model, mesh_context)

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
