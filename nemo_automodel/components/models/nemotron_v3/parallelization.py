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

"""Model-owned FSDP placement for Nemotron-H's redundant and TP-sharded blocks."""

import logging

from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy, OffloadPolicy, fully_shard

from nemo_automodel.components.distributed import parallelizer_utils
from nemo_automodel.components.distributed.mesh_utils import get_fsdp_dp_mesh
from nemo_automodel.components.distributed.tp_replicas import mark_tp_replica_gradient_reduction

logger = logging.getLogger(__name__)


def fully_shard_nemotronh(
    model: nn.Module,
    layers: list[nn.Module],
    *,
    device_mesh: DeviceMesh,
    mp_policy: MixedPrecisionPolicy | None,
    offload_policy: OffloadPolicy | None,
    reshard_after_forward: bool | None,
    dp_replicate_mesh_name: str,
    dp_shard_cp_mesh_name: str,
    tp_mesh_name: str,
) -> nn.Module:
    """Assign already TP-parallelized Nemotron-H modules to their FSDP owners.

    The dense TP-only plan runs Mamba/attention blocks, norms, and embeddings
    redundantly. Those parameters use FSDP's replication axis for TP and its
    sharding axis for DP. TP-sharded MLP projections and lm_head keep their
    original DP mesh. FSDP therefore averages each full-computation replica
    gradient during backward, including deferred/accumulated backward.

    Native ownership initially covers the HF dense TP2 layout. Other TP sizes,
    native model layouts, CP, PP, replicated DP, MoE, MTP, tied embeddings, and CPU offload retain
    their existing DP-only placement and generic TP synchronization until their
    composition with the native replica axis is validated. Initialization still
    uses the infrastructure's one-time broadcast; only gradient ownership moves.

    Args:
        model: Nemotron-H model after TP placement and activation checkpointing.
        layers: Decoder blocks in forward order, from either supported layout.
        device_mesh: Root mesh, ordered DP before TP for the dense native path.
        mp_policy: Existing compute and gradient-reduction precision policy.
        offload_policy: Existing FSDP offload policy.
        reshard_after_forward: Existing decoder-unit reshard policy.
        dp_replicate_mesh_name: Root mesh's replicated-DP axis name.
        dp_shard_cp_mesh_name: Flattened DP-shard/context axis name.
        tp_mesh_name: Root mesh's tensor-parallel axis name.

    Returns:
        The same model, mutated in place with FSDP units.
    """
    dp_mesh = get_fsdp_dp_mesh(device_mesh, dp_replicate_mesh_name, dp_shard_cp_mesh_name)
    tp_mesh = device_mesh[tp_mesh_name]
    native_replicas = (
        tp_mesh.size() == 2
        and device_mesh.device_type == "cuda"
        and device_mesh.mesh_dim_names[-1] == tp_mesh_name
        and hasattr(model, "backbone")
        and offload_policy is None
        and all(
            axis not in device_mesh.mesh_dim_names or device_mesh[axis].size() == 1
            for axis in ("cp", "pp", dp_replicate_mesh_name)
        )
        and not getattr(model.config, "n_routed_experts", None)
        and not getattr(model.config, "tie_word_embeddings", False)
        and not getattr(model, "mtp", None)
    )
    replica_mesh = dp_mesh
    if native_replicas:
        replica_mesh = DeviceMesh(
            device_mesh.device_type,
            device_mesh.mesh.reshape(-1, tp_mesh.size()).T.contiguous(),
            mesh_dim_names=(tp_mesh_name, dp_shard_cp_mesh_name),
        )

    fp32_compute_module_names = tuple(getattr(model, "_keep_in_fp32_modules_strict", None) or ())
    for layer in layers:
        layer_mesh = dp_mesh
        if native_replicas:
            if layer.block_type == "mlp":
                parallelizer_utils.fully_shard_by_dtype(
                    layer.norm,
                    mesh=replica_mesh,
                    mp_policy=mp_policy,
                    offload_policy=offload_policy,
                    fp32_compute_module_names=fp32_compute_module_names,
                    reshard_after_forward=reshard_after_forward,
                )
            else:
                layer_mesh = replica_mesh
        parallelizer_utils.fully_shard_by_dtype(
            layer,
            mesh=layer_mesh,
            mp_policy=mp_policy,
            offload_policy=offload_policy,
            fp32_compute_module_names=fp32_compute_module_names,
            reshard_after_forward=reshard_after_forward,
        )

    if native_replicas:
        fully_shard(
            model.lm_head, mesh=dp_mesh, mp_policy=mp_policy, offload_policy=offload_policy, reshard_after_forward=False
        )
    fully_shard(
        model,
        mesh=replica_mesh,
        mp_policy=mp_policy,
        offload_policy=offload_policy,
        reshard_after_forward=False,
    )
    if native_replicas:
        # Keep initialization broadcast, but do not reduce these gradients again
        # in recipes. The marker is module-owned, surviving parameter replacement.
        for module in model.modules():
            mark_tp_replica_gradient_reduction(module, "none")
        logger.info("Native FSDP owns dense Nemotron-H TP replica gradients")
    return model
