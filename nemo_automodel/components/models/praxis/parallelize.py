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
"""Parallelization of :class:`~nemo_automodel.components.models.praxis.PraxisPolicy` for training.

FSDP2 shards every block of the two backbone towers and the action expert over the data-parallel
mesh. With ``distributed.ep_size > 1`` the MoE feed-forwards are expert parallel through the shared
MoE machinery (expert sharding, the backend's token dispatcher, dispatch replay under activation
checkpointing), as in the language-model MoE path; the walk lives here because the Praxis blocks sit
in two stacks under ``backbone`` and one under ``action_expert`` rather than in a single ``layers``
list.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy, OffloadPolicy, fully_shard
from torch.distributed.tensor import DTensor, Replicate, distribute_tensor

from nemo_automodel.components.distributed.activation_checkpointing import (
    ensure_fsdp_ops_sac_ignored,
    ensure_moe_recompute_cache_ops_sac_ignored,
    ensure_profiler_ops_sac_ignored,
)
from nemo_automodel.components.distributed.config import FSDP2Config, MoEParallelizerConfig
from nemo_automodel.components.distributed.fsdp_patches import patch_fsdp_accumulated_grad_guard
from nemo_automodel.components.distributed.mesh import MeshAxisName, MeshContext
from nemo_automodel.components.distributed.mesh_utils import get_fsdp_dp_mesh
from nemo_automodel.components.distributed.parallelizer import ModelParallelizer
from nemo_automodel.components.distributed.parallelizer_utils import get_internal_fsdp_mp_policy
from nemo_automodel.components.models.praxis.layers import TransformerBlock
from nemo_automodel.components.moe.layers import MoE
from nemo_automodel.components.moe.parallelizer import (
    _moe_shard_placement,
    _uses_hybridep_dispatch,
    _uses_moe_recompute_cache,
    apply_ep_to_moe,
    checkpoint_moe_block,
    router_save_checkpoint_policy,
)

if TYPE_CHECKING:
    from nemo_automodel.components.models.praxis.model import PraxisPolicy


def _expert_meshes(moe_mesh: DeviceMesh | None) -> tuple[DeviceMesh | None, DeviceMesh | None]:
    """The expert-parallel mesh and the mesh the experts are FSDP-sharded over; ``None`` when unused."""
    if moe_mesh is None or MeshAxisName.EP not in (moe_mesh.mesh_dim_names or ()):
        return None, None
    ep_mesh = moe_mesh[MeshAxisName.EP]
    if ep_mesh.size() <= 1:
        return None, None
    ep_shard_mesh = None
    if MeshAxisName.EP_SHARD in moe_mesh.mesh_dim_names and moe_mesh[MeshAxisName.EP_SHARD].size() > 1:
        ep_shard_mesh = moe_mesh[MeshAxisName.EP_SHARD]
    return ep_mesh, ep_shard_mesh


def _block_lists(policy: PraxisPolicy) -> tuple[nn.ModuleList, ...]:
    """The block stacks of the model: encoder tower, decoder tower, action expert."""
    return (
        policy.backbone.encoder.core.blocks,
        policy.backbone.decoder.core.blocks,
        policy.action_expert.blocks,
    )


def _moe_of(block: nn.Module) -> MoE | None:
    """The MoE feed-forward of a backbone block, or ``None`` for dense and action-expert blocks."""
    if isinstance(block, TransformerBlock) and isinstance(block.ffn, MoE):
        return block.ffn
    return None


def shard_policy(
    policy: PraxisPolicy,
    mesh: DeviceMesh,
    mp_policy: MixedPrecisionPolicy,
    *,
    moe_mesh: DeviceMesh | None = None,
    activation_checkpointing: bool = True,
    ignore_router_for_ac: bool = True,
    reshard_after_forward: bool = True,
    offload_policy: OffloadPolicy | None = None,
) -> None:
    """Shard the two backbone towers and the action expert block by block, then the root.

    Every transformer block is one FSDP2 unit. With ``activation_checkpointing`` each block recomputes
    its activations in the backward pass; for an MoE block the router projection and its top-k
    selection are saved instead of recomputed when ``ignore_router_for_ac`` is set, which keeps the
    recompute's routing and expert dispatch identical to the forward's. With a ``moe_mesh`` whose
    ``ep`` axis has more than one rank the routed experts are expert parallel: every expert lives
    whole on one rank of that axis, the backend's token dispatcher exchanges the tokens, and the
    experts are FSDP-sharded over the ``ep_shard`` axis when that axis has more than one rank. The
    routers' selection biases become DTensors replicated over ``mesh``
    (:func:`replicate_router_biases`), so that their load-balancing steps use the expert load of all
    ranks.

    With LoRA on the action expert the decoder tower never runs, so its units have no unsharded state
    while the root accumulates gradients across micro-steps; the FSDP guard patch skips them.

    Args:
        policy: The model, built on the meta device.
        mesh: Data-parallel device mesh of the FSDP shards.
        mp_policy: Parameter and reduction dtypes of the shards.
        moe_mesh: MoE mesh with the ``ep`` and, optionally, ``ep_shard`` axes; ``None`` without expert
            parallelism.
        activation_checkpointing: Recompute the blocks' activations in the backward pass.
        ignore_router_for_ac: Save the MoE routing across the recompute instead of recomputing it.
        reshard_after_forward: Free each unit's gathered parameters after its forward.
        offload_policy: FSDP2 offload policy of the units, or ``None`` for none.
    """
    patch_fsdp_accumulated_grad_guard()
    offload = offload_policy if offload_policy is not None else OffloadPolicy()
    ep_mesh, ep_shard_mesh = _expert_meshes(moe_mesh)
    moe_layers = policy.moe_layers()
    if ep_mesh is not None:
        for layer in moe_layers:
            if layer.n_routed_experts % ep_mesh.size():
                raise ValueError(
                    f"{layer.n_routed_experts} routed experts are not divisible by ep_size {ep_mesh.size()}"
                )
            apply_ep_to_moe(layer, ep_mesh, moe_mesh=moe_mesh)
    router_policy = None
    if activation_checkpointing and ignore_router_for_ac and moe_layers:
        gate = moe_layers[0].gate
        router_policy = router_save_checkpoint_policy(hidden_size=gate.dim, num_experts=gate.n_experts)
        ensure_profiler_ops_sac_ignored()
        ensure_fsdp_ops_sac_ignored()
    hybridep = _uses_hybridep_dispatch(policy)
    recompute_cache = _uses_moe_recompute_cache(policy)
    if activation_checkpointing and recompute_cache:
        ensure_moe_recompute_cache_ops_sac_ignored()
    experts_mp_policy = get_internal_fsdp_mp_policy(mp_policy)
    for module_list in _block_lists(policy):
        for i in range(len(module_list)):
            block = module_list[i]
            moe = _moe_of(block)
            if activation_checkpointing:
                block = checkpoint_moe_block(
                    block,
                    router_policy=router_policy if moe is not None else None,
                    uses_hybridep_dispatch=hybridep,
                    uses_moe_recompute_cache=recompute_cache,
                )
            ignored: set[nn.Parameter] | None = None
            if moe is not None and ep_mesh is not None:
                if ep_shard_mesh is not None:
                    fully_shard(
                        moe.experts,
                        mesh=ep_shard_mesh,
                        shard_placement_fn=_moe_shard_placement,
                        reshard_after_forward=reshard_after_forward,
                        mp_policy=experts_mp_policy,
                        offload_policy=offload,
                    )
                else:
                    # Every expert is whole on its rank: nothing of it is FSDP state.
                    ignored = set(moe.experts.parameters())
            fully_shard(
                block,
                mesh=mesh,
                mp_policy=mp_policy,
                reshard_after_forward=reshard_after_forward,
                offload_policy=offload,
                ignored_params=ignored,
            )
            module_list[i] = block
    fully_shard(
        policy, mesh=mesh, mp_policy=mp_policy, reshard_after_forward=reshard_after_forward, offload_policy=offload
    )
    replicate_router_biases(policy, mesh)


def replicate_router_biases(policy: PraxisPolicy, mesh: DeviceMesh) -> None:
    """Turn every router's selection bias into a DTensor replicated over ``mesh``.

    FSDP2 leaves buffers as plain tensors, and the shared gate steps a plain bias from this rank's
    expert load alone. Replicated, its update sums the load over all ranks of ``mesh`` and applies the
    same step on every rank, as the reference trainer does.

    Args:
        policy: The model; its MoE gates hold ``e_score_correction_bias``.
        mesh: Data-parallel device mesh.
    """
    for layer in policy.moe_layers():
        bias = layer.gate.e_score_correction_bias
        if bias is not None and not isinstance(bias, DTensor):
            layer.gate.e_score_correction_bias = distribute_tensor(
                bias, device_mesh=mesh, placements=[Replicate()] * mesh.ndim
            )


class PraxisModelParallelizer(ModelParallelizer):
    """Model-owned parallelization of :class:`PraxisPolicy`, the ``parallelizer`` of the model class.

    The shared flows walk one ``layers`` stack; these hooks walk the Praxis towers with
    :func:`shard_policy` instead, with (``_apply_moe``) and without (``_apply``) expert parallelism,
    and on one rank (``_apply_unsharded``).
    """

    def _apply_moe(self, model: nn.Module, mesh_context: MeshContext) -> nn.Module:
        """Shard ``model`` for the FSDP2 strategy, and the expert parallelism, of ``mesh_context``."""
        policy = _policy(model)
        strategy = mesh_context.strategy_config
        if not isinstance(strategy, FSDP2Config):
            raise TypeError(f"PraxisPolicy trains with distributed.strategy fsdp2, got {type(strategy).__name__}")
        if mesh_context.device_mesh is None:
            raise ValueError("PraxisPolicy parallelization needs a device mesh")
        moe = mesh_context.moe_parallel_config or MoEParallelizerConfig()
        shard_policy(
            policy,
            get_fsdp_dp_mesh(mesh_context.device_mesh),
            _mp_policy(moe.mp_policy or strategy.mp_policy),
            moe_mesh=mesh_context.moe_mesh,
            activation_checkpointing=bool(mesh_context.activation_checkpointing),
            ignore_router_for_ac=moe.ignore_router_for_ac,
            reshard_after_forward=True if strategy.reshard_after_forward is None else strategy.reshard_after_forward,
            offload_policy=strategy.offload_policy,
        )
        return model

    def _apply_unsharded(self, model: nn.Module, mesh_context: MeshContext) -> nn.Module:
        """One data-parallel rank: the per-block FSDP2 wrapping and activation checkpointing over the one-rank mesh."""
        strategy = mesh_context.strategy_config
        if not isinstance(strategy, FSDP2Config):
            raise TypeError(f"PraxisPolicy trains with distributed.strategy fsdp2, got {type(strategy).__name__}")
        if mesh_context.device_mesh is None:
            raise ValueError("PraxisPolicy parallelization needs a device mesh")
        return self._apply(
            model,
            mesh_context.device_mesh,
            mp_policy=strategy.mp_policy,
            offload_policy=strategy.offload_policy,
            activation_checkpointing=bool(mesh_context.activation_checkpointing),
            reshard_after_forward=strategy.reshard_after_forward,
        )

    def _apply(
        self,
        model: nn.Module,
        device_mesh: DeviceMesh,
        mp_policy: MixedPrecisionPolicy | None = None,
        offload_policy: OffloadPolicy | None = None,
        sequence_parallel: bool = False,
        activation_checkpointing: bool = False,
        reshard_after_forward: bool | None = None,
        **kwargs,
    ) -> nn.Module:
        """Shard ``model`` for FSDP2 without expert parallelism; the experts stay inside their blocks."""
        del sequence_parallel, kwargs
        shard_policy(
            _policy(model),
            get_fsdp_dp_mesh(device_mesh),
            _mp_policy(mp_policy),
            activation_checkpointing=bool(activation_checkpointing),
            reshard_after_forward=True if reshard_after_forward is None else reshard_after_forward,
            offload_policy=offload_policy,
        )
        return model


def _policy(model: nn.Module) -> PraxisPolicy:
    from nemo_automodel.components.models.praxis.model import PraxisPolicy

    if not isinstance(model, PraxisPolicy):
        raise TypeError(f"PraxisModelParallelizer parallelizes a PraxisPolicy, got {type(model).__name__}")
    return model


def _mp_policy(mp_policy: MixedPrecisionPolicy | None) -> MixedPrecisionPolicy:
    return mp_policy or MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=False
    )
