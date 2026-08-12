# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import logging
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh

from nemo_automodel.components.distributed.config import MegatronFSDPConfig
from nemo_automodel.components.distributed.mesh_utils import get_flat_mesh
from nemo_automodel.components.distributed.parallelizer import (
    _get_parallel_plan,
    megatron_fsdp_strategy_parallelize,
)

if TYPE_CHECKING:
    from nemo_automodel.components.distributed.config import DistributedConfig

logger = logging.getLogger(__name__)

try:
    from megatron_fsdp import MegatronFSDP
    from megatron_fsdp.fully_shard import fully_shard_optimizer as megatron_fsdp_fully_shard_optimizer

    HAS_MEGATRON_FSDP = True
except (ImportError, FileNotFoundError, OSError):
    MegatronFSDP = None
    megatron_fsdp_fully_shard_optimizer = None
    HAS_MEGATRON_FSDP = False


class MegatronFSDPManager:
    """
    Manager for parallelizing models using MegatronFSDP with TP, DP, CP sharding.

    This manager applies parallelization to the model using a prescribed
    TP sharding plan. It supports mixed precision and various FSDP options.

    The device mesh must be created externally and passed in.

    Args:
        config (MegatronFSDPConfig): Configuration for MegatronFSDP distributed training.
        device_mesh (DeviceMesh): Device mesh for distributed operations.

    Example:
        from nemo_automodel.components.distributed.config import MegatronFSDPConfig

        config = MegatronFSDPConfig(zero_dp_strategy=3, overlap_grad_reduce=True)
        # device_mesh created externally via MeshContext.build()
        manager = MegatronFSDPManager(config, device_mesh=device_mesh)
        model, optimizer = manager.parallelize(model, optimizer)
    """

    def __init__(
        self,
        config: MegatronFSDPConfig,
        device_mesh: DeviceMesh,
    ):
        self.config = config
        self.device_mesh = device_mesh

        # Extract config fields for easy access
        self.megatron_fsdp_unit_modules = config.megatron_fsdp_unit_modules
        self.zero_dp_strategy = config.zero_dp_strategy
        self.init_fsdp_with_meta_device = config.init_fsdp_with_meta_device
        self.grad_reduce_in_fp32 = config.grad_reduce_in_fp32
        self.preserve_fp32_weights = config.preserve_fp32_weights
        self.overlap_grad_reduce = config.overlap_grad_reduce
        self.overlap_param_gather = config.overlap_param_gather
        self.check_for_nan_in_grad = config.check_for_nan_in_grad
        self.report_nan_in_param_grad = config.report_nan_in_param_grad
        self.average_in_collective = config.average_in_collective
        self.disable_bucketing = config.disable_bucketing
        self.calculate_per_token_loss = config.calculate_per_token_loss
        self.keep_fp8_transpose_cache = config.keep_fp8_transpose_cache
        self.nccl_ub = config.nccl_ub
        self.fsdp_double_buffer = config.fsdp_double_buffer
        self.activation_checkpointing = config.activation_checkpointing

    def parallelize(self, model, optimizer=None):
        """
        Parallelizes the given model using MegatronFSDP and TP sharding strategies.

        Args:
            model: The model to be parallelized.
            optimizer: The optimizer for the model. If None, user needs to call
                model.finish_grad_sync() before optimizer.step(),
                model.install_optimized_model_weights() and model.zero_grad_buffer()
                after optimizer.zero_grad().

        Returns:
            tuple: (parallelized_model, optimizer)
        """
        if dist.get_world_size() == 1:
            logger.info("World size is 1, skipping parallelization.")
            model = model.to("cuda").to(torch.bfloat16)
            if self.activation_checkpointing:
                if hasattr(model, "gradient_checkpointing_enable"):
                    model.gradient_checkpointing_enable()
                else:
                    logger.error("Model does not support gradient checkpointing. Skipping.")
            return model, optimizer

        if self.activation_checkpointing:
            logger.error("Activation checkpointing is not yet supported with MegatronFSDP. Skipping.")

        if self.zero_dp_strategy != 3:
            if self.device_mesh.get_rank() == 0:
                print("Warning: MegatronFSDP zero_dp_strategy is not 3. Parameters will not be sharded.")

        if self.device_mesh["tp"].size() > 1:
            # Delegate plan selection to central helper. MegatronFSDP currently does not support SP.
            tp_shard_plan = _get_parallel_plan(
                model,
                sequence_parallel=False,  # explicit: SP not supported here
                tp_shard_plan=None,
                tp_size=self.device_mesh["tp"].size(),
            )
        else:
            tp_shard_plan = None

        # ``dp_cp`` is normally produced by DeviceMesh._flatten(), so it lives
        # in the root mesh's private flatten mapping rather than in
        # ``mesh_dim_names``.  A name-only check silently drops CP from the
        # Megatron-FSDP shard group on real MeshContext meshes.  Resolve it
        # through the same compatibility helper used by the rest of the
        # distributed stack; older meshes with a literal ``dp_cp`` dimension
        # continue to work as well.
        try:
            get_flat_mesh(self.device_mesh, "dp_cp")
        except KeyError:
            dp_shard_dim = "dp"
        else:
            dp_shard_dim = "dp_cp"
        tp_dim = "tp"

        model, optimizer = megatron_fsdp_strategy_parallelize(
            model,
            device_mesh=self.device_mesh,
            optimizer=optimizer,
            megatron_fsdp_unit_modules=self.megatron_fsdp_unit_modules,
            tp_shard_plan=tp_shard_plan,
            zero_dp_strategy=self.zero_dp_strategy,
            init_fsdp_with_meta_device=self.init_fsdp_with_meta_device,
            grad_reduce_in_fp32=self.grad_reduce_in_fp32,
            preserve_fp32_weights=self.preserve_fp32_weights,
            overlap_grad_reduce=self.overlap_grad_reduce,
            overlap_param_gather=self.overlap_param_gather,
            check_for_nan_in_grad=self.check_for_nan_in_grad,
            report_nan_in_param_grad=self.report_nan_in_param_grad,
            average_in_collective=self.average_in_collective,
            disable_bucketing=self.disable_bucketing,
            calculate_per_token_loss=self.calculate_per_token_loss,
            keep_fp8_transpose_cache=self.keep_fp8_transpose_cache,
            nccl_ub=self.nccl_ub,
            fsdp_double_buffer=self.fsdp_double_buffer,
            dp_shard_dim=dp_shard_dim,
            tp_dim=tp_dim,
        )

        return model, optimizer


def fully_shard_optimizer(
    model: nn.Module, optimizer: torch.optim.Optimizer, preproc_state_dict_for_dcp_ckpt: bool = True
) -> torch.optim.Optimizer:
    """Register an already-built optimizer with a MegatronFSDP-wrapped model.

    Megatron-FSDP 0.5.0's ``fully_shard_optimizer`` recovers the owning
    ``MegatronFSDP`` from a ``_megatron_fsdp_model`` attribute that
    ``MegatronFSDP.__init__`` stamps onto each distributed ``Parameter``. That
    attribute is a plain Python attribute and does not survive operations that
    rebuild ``Parameter`` objects (e.g. the dtype/device cast the ``from_pretrained``
    load path performs after wrapping). The combined ``fully_shard(model, optimizer)``
    entry point never hits this because it registers the optimizer in the same call,
    before any such op runs; the recipe's separate build-model-then-build-optimizer
    order does, leaving ``fully_shard_optimizer`` unable to find the reference and
    aborting before the first optimizer step. Re-stamp the reference (mirroring the
    wheel's own ``__init__`` logic) on the current distributed params right before
    deferred sharding so the separate sequence matches the combined entry point.
    """
    if not isinstance(model, MegatronFSDP):
        return optimizer
    if not HAS_MEGATRON_FSDP:
        raise ImportError(
            "MegatronFSDP is not installed, please visit https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/distributed/fsdp/src for more information"
        )
    model._replace_param_with_distributed_if_needed()
    for param in model.parameters():
        param._megatron_fsdp_model = model
    return megatron_fsdp_fully_shard_optimizer(optimizer)


def snapshot_distributed_param_attrs(model: nn.Module) -> "dict[str, dict] | None":
    """Snapshot the per-parameter attributes Megatron-FSDP stamps on distributed params.

    ``MegatronFSDP.__init__`` decorates each distributed ``Parameter`` with plain
    Python attributes that later training steps depend on: ``_megatron_fsdp_model``
    (owning-model back-ref used by :func:`fully_shard_optimizer`), ``_is_shared``
    (set on tied parameters so ``_grad_acc`` routes their gradients through the root
    hook instead of double-accumulating), ``orig_param``/``megatron_fsdp_dist_index``/
    ``megatron_fsdp_slice`` and the ``reset_attribute`` closure. These attributes live
    in the ``Parameter.__dict__`` and are silently dropped by any post-wrap operation
    that rebuilds ``Parameter`` objects -- e.g. the ``from_pretrained`` checkpoint
    reload and the ``lm_head`` re-tie the recipe performs after wrapping. Capture them
    here, keyed by parameter name (object identity does not survive the rebuild), so
    :func:`restore_distributed_param_attrs` can re-apply them afterwards.

    ``remove_duplicate=False`` is required so tied parameters (e.g. ``lm_head.weight``
    aliasing ``model.embed_tokens.weight``) are captured under every name they appear
    under, including the ``_is_shared`` marker Megatron-FSDP places on the tied alias.

    Args:
        model: The (possibly Megatron-FSDP-wrapped) model to snapshot.

    Returns:
        A mapping from parameter name to a copy of its ``__dict__``, or ``None`` when
        ``model`` is not a Megatron-FSDP model (nothing to snapshot).
    """
    if not HAS_MEGATRON_FSDP or not isinstance(model, MegatronFSDP):
        return None
    return {name: dict(param.__dict__) for name, param in model.module.named_parameters(remove_duplicate=False)}


def restore_distributed_param_attrs(model: nn.Module, snapshot: "dict[str, dict] | None") -> None:
    """Re-apply Megatron-FSDP per-parameter attributes dropped by a post-wrap rebuild.

    Companion to :func:`snapshot_distributed_param_attrs`. For each current parameter
    (matched by name, since the rebuild replaced the objects) it restores any snapshot
    attribute the rebuilt parameter is missing, following the fix suggested by the
    Megatron-FSDP maintainer on NVIDIA/Megatron-LM#5790: only attributes absent on the
    new parameter are copied, so genuinely re-derived state is never clobbered.

    Args:
        model: The Megatron-FSDP-wrapped model whose parameters were rebuilt.
        snapshot: The mapping returned by :func:`snapshot_distributed_param_attrs`, or
            ``None`` (no-op).
    """
    if snapshot is None or not HAS_MEGATRON_FSDP or not isinstance(model, MegatronFSDP):
        return
    for name, param in model.module.named_parameters(remove_duplicate=False):
        saved = snapshot.get(name)
        if not saved:
            continue
        for key, val in saved.items():
            if not hasattr(param, key):
                setattr(param, key, val)


def maybe_shard_optimizer(
    model_part: nn.Module,
    optimizer: torch.optim.Optimizer,
    distributed_config: "DistributedConfig | None",
    *,
    allow: bool = True,
) -> torch.optim.Optimizer:
    """Shard the optimizer with Megatron-FSDP when the strategy requires it.

    Returns the optimizer unchanged unless ``distributed_config`` is a
    :class:`MegatronFSDPConfig` running in a distributed (world size > 1) job.

    Args:
        model_part: The (already sharded) model part the optimizer belongs to.
        optimizer: The optimizer to (optionally) shard.
        distributed_config: Distributed strategy config; only triggers sharding
            when it is a :class:`MegatronFSDPConfig`.
        allow: Guard for optimizers incompatible with Megatron-FSDP sharding
            (e.g. Dion); asserts when sharding would otherwise apply.
    """
    if isinstance(distributed_config, MegatronFSDPConfig) and dist.get_world_size() > 1:
        assert allow, "Dion optimizer does not support fully_shard_optimizer"
        if not HAS_MEGATRON_FSDP:
            return optimizer
        return fully_shard_optimizer(model_part, optimizer)
    return optimizer
