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
from typing import Optional

from torch.distributed.device_mesh import DeviceMesh

from nemo_automodel.components.distributed.config import FSDP2Config
from nemo_automodel.components.distributed.init_utils import get_world_size_safe
from nemo_automodel.components.distributed.parallelizer import (
    _get_parallel_plan,
    fsdp2_strategy_parallelize,
)

logger = logging.getLogger(__name__)


class FSDP2Manager:
    """
    Manager for parallelizing models using FSDP2 with TP, DP, CP sharding.

    This manager applies parallelization to the model using a prescribed
    TP sharding plan. It supports mixed precision and CPU offloading options.

    The device mesh must be created externally and passed in.

    Args:
        config (FSDP2Config): Configuration for FSDP2 distributed training.
        device_mesh (DeviceMesh): Device mesh for distributed operations.
        moe_mesh (Optional[DeviceMesh]): Optional device mesh for expert parallelism.

    Example:
        from nemo_automodel.components.distributed.config import FSDP2Config

        config = FSDP2Config(sequence_parallel=True, activation_checkpointing=True)
        # device_mesh created externally via create_device_mesh()
        manager = FSDP2Manager(config, device_mesh=device_mesh, moe_mesh=moe_mesh)
        model = manager.parallelize(model)
    """

    def __init__(
        self,
        config: FSDP2Config,
        device_mesh: DeviceMesh,
        moe_mesh: Optional[DeviceMesh] = None,
    ):
        self.config = config
        self.device_mesh = device_mesh
        self.moe_mesh = moe_mesh

        # Extract config fields for easy access
        self.sequence_parallel = config.sequence_parallel
        self.tp_plan = config.tp_plan
        self.mp_policy = config.mp_policy
        self.offload_policy = config.offload_policy
        self.activation_checkpointing = config.activation_checkpointing
        self.defer_fsdp_grad_sync = config.defer_fsdp_grad_sync
        self.backend = config.backend

    def parallelize(self, model):
        """
        Parallelizes the given model using FSDP2 and TP sharding strategies.

        Args:
            model (nn.Module): The model to be parallelized.

        Returns:
            The parallelized model.
        """
        if get_world_size_safe() == 1:
            logger.info("World size is 1, skipping parallelization.")
            if self.activation_checkpointing:
                if hasattr(model, "gradient_checkpointing_enable"):
                    model.gradient_checkpointing_enable()
                else:
                    logger.error("Model does not support gradient checkpointing.")
            return model

        if self.device_mesh["tp"].size() > 1:
            # Delegate plan selection to central helper
            tp_shard_plan = _get_parallel_plan(
                model,
                sequence_parallel=bool(self.sequence_parallel),
                tp_shard_plan=self.tp_plan,
            )
        else:
            tp_shard_plan = None

        fsdp2_strategy_parallelize(
            model,
            device_mesh=self.device_mesh,
            mp_policy=self.mp_policy,
            tp_shard_plan=tp_shard_plan,
            offload_policy=self.offload_policy,
            sequence_parallel=bool(self.sequence_parallel),
            activation_checkpointing=self.activation_checkpointing,
        )
        return model
