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

"""Model-owned distributed parallelization for ERNIE 5.5.

ERNIE 5.5 is distributed with FSDP2 and expert parallelism only. The sidecar rejects every other topology before
the shared dispatcher runs, keeps the shared expert-parallel flow, and adds two things while that flow builds the
FSDP units:

* Every HCA compressor receives the process group of the mesh over which its parameters are sharded. A row shorter
  than one compression window does not use the compressor. With the group, the ranks that share the compressor
  parameters agree, for every microbatch, on whether the compressor takes part in backward, so they issue the same
  FSDP collectives in the same order.
* The hyper-head becomes an FP32 unit of its own before the root unit, which computes in the bulk dtype, claims its
  parameters.

No other trainable parameter of the model is used by some ranks and not by others, so the sidecar does not enable
FSDP's reduction of unused parameters.
"""

import logging
from typing import Required, TypedDict, Unpack

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, OffloadPolicy

from nemo_automodel.components.distributed import FSDP2Config, MeshContext, ModelParallelizer
from nemo_automodel.components.distributed.parallelizer_utils import fully_shard_by_dtype
from nemo_automodel.components.models.ernie5_5.config import Ernie5_5ModelPlan
from nemo_automodel.components.models.ernie5_5.hca import Ernie5_5HCAAttention
from nemo_automodel.components.models.ernie5_5.layers import Ernie5_5HyperHead

logger = logging.getLogger(__name__)

# Parameters of ``Ernie5_5HyperHead``, named relative to that module. All of them compute in FP32.
_HYPER_HEAD_FP32_NAMES = ("hc_fn", "hc_base", "hc_scale")
# Compute dtypes that the model implements; an FSDP policy may also leave the storage dtype in place (None).
_COMPUTE_DTYPES = (None, torch.float32, torch.bfloat16)


class _FullyShardKwargs(TypedDict, total=False):
    """Keyword arguments with which the shared flows call ``_fully_shard_module``."""

    mesh: Required[DeviceMesh]
    mp_policy: MixedPrecisionPolicy | None
    offload_policy: OffloadPolicy | None
    reshard_after_forward: bool | int | None
    ignored_params: set[nn.Parameter] | None


def _axis_names(mesh: DeviceMesh) -> str:
    """Return the axis names of a mesh as text, for example ``dp_replicate, dp_shard``."""
    return ", ".join(str(getattr(name, "value", name)) for name in mesh.mesh_dim_names or ())


def _is_pipeline_stage(model: nn.Module) -> bool:
    """Return whether pipeline-stage surgery removed modules or decoder layers from an ERNIE 5.5 model.

    Args:
        model: The ``Ernie5_5_MoeForCausalLM`` handed to the sidecar.

    Returns:
        True when the embedding, the hyper-head, the final norm, the output head or MultiMax is missing, or the
        decoder holds another number of layers than its configuration declares.
    """
    inner = model.model
    modules = (inner.embed_tokens, inner.hyper_head, inner.norm, inner.lm_head, inner.multimax)
    return any(module is None for module in modules) or len(inner.layers) != model.config.num_hidden_layers


class Ernie5_5ModelParallelizer(ModelParallelizer):
    """Distribute ERNIE 5.5 with FSDP2 and expert parallelism, and reject every other topology.

    The supported topology is ``distributed.strategy: fsdp2`` with ``ep_size`` greater than one, one data-parallel
    sharding dimension over all ranks of the job (``tp_size``, ``cp_size``, ``pp_size`` and ``dp_replicate_size``
    equal to one), FSDP compute in FP32 or BF16, and no activation checkpointing.

    A model with a DSA layer also restricts the number of ranks: it must divide
    ``config.hybrid_mla_num_attention_heads``. A checkpoint stores the key and value projections of a DSA layer as
    one matrix with the rows of each attention head together, and the checkpoint adapter loads and saves that
    matrix only when every rank holds the rows of whole heads. With 64 heads the supported jobs have 2, 4, 8, 16,
    32 or 64 ranks.
    """

    _customizes_moe_fsdp = True

    def _prepare_model(self, model: nn.Module, mesh_context: MeshContext) -> None:
        """Reject every unsupported topology before an executor changes the model.

        The shared dispatcher has not run when this hook is called, so a rejected model keeps its parameters, its
        modules and its synchronization groups as they were handed to the hook. The hook runs for every strategy,
        including DDP and Megatron-FSDP. Under pipeline parallelism the framework has already copied the model
        and cut the copies into stages when it calls the hook, once per stage; the hook rejects the stage.

        One condition is not decided here. The mesh over which the FSDP units are sharded is derived inside the
        shared expert-parallel flow, so ``_fully_shard_module`` checks its number of dimensions as a
        post-condition.

        Args:
            model: The ``Ernie5_5_MoeForCausalLM`` to distribute.
            mesh_context: Meshes, parallelism sizes, strategy, precision policies and activation-checkpointing
                mode of the run.

        Raises:
            ValueError: If the strategy is not FSDP2; if the tensor-, context- or pipeline-parallel size or the
                replicate size is not one; if ``model`` is a pipeline stage; if expert parallelism is off; if
                activation checkpointing is on; if an FSDP compute dtype is neither FP32 nor BF16; if the
                data-parallel mesh does not cover all ranks of the job; or if the model has a DSA layer and the
                number of ranks does not divide its number of DSA attention heads.
        """
        strategy = mesh_context.strategy_config
        if not isinstance(strategy, FSDP2Config):
            raise ValueError(
                f"ERNIE 5.5 requires distributed.strategy fsdp2 with expert parallelism, got {type(strategy).__name__}"
            )
        sizes = {
            "tp_size": mesh_context.tp_size,
            "cp_size": mesh_context.cp_size,
            "pp_size": mesh_context.pp_size,
            "dp_replicate_size": mesh_context.dp_replicate_size or 1,
        }
        unsupported = {name: size for name, size in sizes.items() if size != 1}
        if unsupported:
            raise ValueError(
                "ERNIE 5.5 supports FSDP2 with expert parallelism only; set these distributed sizes to 1: "
                f"{unsupported}"
            )
        if _is_pipeline_stage(model):
            raise ValueError(
                "ERNIE 5.5 does not support pipeline parallelism: the model handed to the parallelizer is a "
                "pipeline stage that lacks modules or decoder layers of the full model"
            )
        ep_size = mesh_context.ep_size
        if ep_size <= 1:
            raise ValueError(
                "ERNIE 5.5 requires expert parallelism: set distributed.ep_size to a value greater than 1, got "
                f"{ep_size}"
            )
        if mesh_context.activation_checkpointing:
            raise ValueError(
                "ERNIE 5.5 does not support activation checkpointing: set distributed.activation_checkpointing to "
                f"false, got {mesh_context.activation_checkpointing!r}"
            )
        moe = mesh_context.moe_parallel_config
        policies = {
            "distributed.mp_policy": strategy.mp_policy,
            "distributed.moe.mp_policy": None if moe is None else moe.mp_policy,
        }
        for name, policy in policies.items():
            if policy is not None and policy.param_dtype not in _COMPUTE_DTYPES:
                raise ValueError(
                    f"ERNIE 5.5 computes in float32 or bfloat16 only: set {name}.param_dtype to one of them, got "
                    f"{policy.param_dtype}"
                )
        device_mesh = mesh_context.device_mesh
        if device_mesh is None:
            raise ValueError("ERNIE 5.5 requires a data-parallel device mesh, got mesh_context.device_mesh None")
        ranks, world = device_mesh.size(), dist.get_world_size()
        if ranks != world:
            raise ValueError(
                f"ERNIE 5.5 requires a data-parallel mesh over all {world} ranks of the job, got a mesh of {ranks} "
                "ranks"
            )
        config = model.config
        if any(layer.attention == "dsa" for layer in Ernie5_5ModelPlan.from_config(config).decoder_layers):
            # The condition of the checkpoint adapter for the key and value projections of a DSA layer, which
            # are sharded over all ranks.
            heads = config.hybrid_mla_num_attention_heads
            if heads % ranks != 0:
                raise ValueError(
                    "ERNIE 5.5 with a DSA layer requires a number of ranks that divides "
                    f"hybrid_mla_num_attention_heads={heads}, so that every rank holds whole attention heads when "
                    f"a checkpoint is loaded or saved; got a job of {ranks} ranks"
                )

    def _apply_moe(self, model: nn.Module, mesh_context: MeshContext) -> nn.Module:
        """Run the shared expert-parallel flow and verify that it sharded the model through this sidecar.

        Args:
            model: The ``Ernie5_5_MoeForCausalLM`` to distribute, admitted by ``_prepare_model``.
            mesh_context: Meshes, parallelism sizes and strategy of the run.

        Returns:
            The model, with its parameters sharded by the shared expert-parallel flow.

        Raises:
            RuntimeError: If the shared flow returned without making the hyper-head an FSDP unit.
        """
        model = super()._apply_moe(model, mesh_context)
        heads = [module for module in model.modules() if isinstance(module, Ernie5_5HyperHead)]
        if not heads or not all(isinstance(head, FSDPModule) for head in heads):
            raise RuntimeError(
                "The expert-parallel flow did not shard ERNIE 5.5 through its sidecar: the hyper-head is not an "
                "FSDP unit, so its FP32 boundary and the HCA synchronization groups are missing"
            )
        return model

    def _fully_shard_module(self, module: nn.Module, **kwargs: Unpack[_FullyShardKwargs]) -> nn.Module:
        """Shard ``module`` after giving ERNIE-specific units inside it their boundary and group.

        The shared flow calls this hook for each decoder block, for the modules inside a block that compute in
        FP32, for the embedding, the output head and the roots. Whichever call contains an HCA compressor sets the
        compressor's synchronization group from the mesh of that call; setting the same group again is harmless.
        The call that contains the hyper-head shards it first as an FP32 unit.

        The check of the mesh is a post-condition of the shared flow, not an admission check: when it fails, the
        flow has already distributed the expert weights. A mesh context built by ``MeshContext.build`` that
        ``_prepare_model`` admitted yields a one-dimensional mesh.

        Args:
            module: Module that becomes an FSDP unit, with parameters of any shape.
            **kwargs: Arguments of ``fully_shard``. ``mesh`` is the one-dimensional data-parallel mesh over which
                the parameters of ``module`` are sharded on their leading axis. ``ignored_params`` holds
                parameters of any shape that another unit already owns; it is passed to ``fully_shard``
                unchanged, as are ``mp_policy``, ``offload_policy`` and ``reshard_after_forward``.

        Returns:
            ``module`` as an FSDP unit.

        Raises:
            ValueError: If ``mesh`` has more than one dimension.
        """
        mesh = kwargs["mesh"]
        if mesh.ndim != 1:
            raise ValueError(
                "ERNIE 5.5 shards its FSDP units over a one-dimensional data-parallel mesh, got a mesh with the "
                f"dimensions ({_axis_names(mesh)})"
            )
        group = mesh.get_group()
        compressors = 0
        head_sharded = False
        for submodule in module.modules():
            if isinstance(submodule, Ernie5_5HCAAttention):
                submodule.compressor._set_hca_param_sync_group(group)
                compressors += 1
            elif isinstance(submodule, Ernie5_5HyperHead) and not isinstance(submodule, FSDPModule):
                fully_shard_by_dtype(
                    submodule,
                    mesh,
                    kwargs.get("mp_policy"),
                    kwargs.get("offload_policy"),
                    fp32_compute_module_names=_HYPER_HEAD_FP32_NAMES,
                    reshard_after_forward=kwargs.get("reshard_after_forward"),
                )
                head_sharded = True
        if head_sharded:
            logger.info(
                "ERNIE 5.5 FSDP on the %d-rank mesh (%s): hyper-head sharded as an FP32 unit, synchronization group "
                "set on %d HCA compressors",
                mesh.size(),
                _axis_names(mesh),
                compressors,
            )
        if isinstance(module, Ernie5_5HyperHead):
            return module
        return super()._fully_shard_module(module, **kwargs)


PARALLELIZER = Ernie5_5ModelParallelizer()

__all__ = ["PARALLELIZER"]
