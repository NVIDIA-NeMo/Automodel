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

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh

from nemo_automodel.components.distributed.config import MegatronFSDPConfig
from nemo_automodel.components.distributed.parallelizer import (
    _get_parallel_plan,
    megatron_fsdp_strategy_parallelize,
)

logger = logging.getLogger(__name__)

try:
    from megatron_fsdp import MegatronFSDP
    from megatron_fsdp.fully_shard import fully_shard_optimizer as megatron_fsdp_fully_shard_optimizer

    HAS_MEGATRON_FSDP = True
except (ImportError, FileNotFoundError):
    # raise FileNotFoundError(
    # E   FileNotFoundError: Could not find shared object file for Transformer Engine torch lib.
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
        # device_mesh created externally via create_device_mesh()
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
        self.sequence_parallel = config.sequence_parallel
        self.megatron_fsdp_unit_modules = config.megatron_fsdp_unit_modules
        self.zero_dp_strategy = config.zero_dp_strategy
        self.init_fsdp_with_meta_device = config.init_fsdp_with_meta_device
        self.grad_reduce_in_fp32 = config.grad_reduce_in_fp32
        self.preserve_fp32_weights = config.preserve_fp32_weights
        self.overlap_grad_reduce = config.overlap_grad_reduce
        self.overlap_param_gather = config.overlap_param_gather
        self.check_for_nan_in_grad = config.check_for_nan_in_grad
        self.average_in_collective = config.average_in_collective
        self.disable_bucketing = config.disable_bucketing
        self.calculate_per_token_loss = config.calculate_per_token_loss
        self.keep_fp8_transpose_cache = config.keep_fp8_transpose_cache
        self.nccl_ub = config.nccl_ub
        self.fsdp_double_buffer = config.fsdp_double_buffer
        self.activation_checkpointing = config.activation_checkpointing
        self.backend = config.backend

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
            )
        else:
            tp_shard_plan = None

        # Determine dp_shard_dim based on whether cp is in mesh
        if "dp_cp" in self.device_mesh.mesh_dim_names:
            dp_shard_dim = "dp_cp"
        else:
            dp_shard_dim = "dp"
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
    """ """
    if not isinstance(model, MegatronFSDP):
        return optimizer
    if not HAS_MEGATRON_FSDP:
        raise ImportError(
            "MegatronFSDP is not installed, please visit https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/distributed/fsdp/src for more information"
        )
    return megatron_fsdp_fully_shard_optimizer(optimizer)
