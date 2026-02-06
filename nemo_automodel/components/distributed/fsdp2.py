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
        self.use_hf_tp_plan = config.use_hf_tp_plan
        self.custom_tp_plan = config.custom_tp_plan
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
                tp_shard_plan=self.custom_tp_plan,
                use_hf_tp_plan=self.use_hf_tp_plan,
            )
        else:
            tp_shard_plan = None

        fsdp2_strategy_parallelize(
            model,
            device_mesh=self.device_mesh,
            mp_policy=self.mp_policy,
            tp_shard_plan=tp_shard_plan,
            offload_policy=self.offload_policy,
            activation_checkpointing=self.activation_checkpointing,
        )
        return model


# =============================================================================
# REFERENCE: Device mesh creation logic (to be moved to device_mesh.py)
# =============================================================================
#
# def _create_device_mesh(
#     self,
#     dp_size: Optional[int],
#     dp_replicate_size: Optional[int],
#     tp_size: int,
#     cp_size: int,
#     pp_size: int,
#     ep_size: int,
#     world_size: int,
#     backend: str,
# ):
#     """
#     Creates device mesh for FSDP2.
#
#     Mesh shape: (pp_size, dp_replicate_size, dp_shard_size, cp_size, tp_size)
#     Mesh names: ("pp", "dp_replicate", "dp_shard", "cp", "tp")
#     """
#     from torch.distributed.device_mesh import init_device_mesh
#
#     if tp_size is None or tp_size <= 0:
#         tp_size = 1
#     if cp_size is None or cp_size <= 0:
#         cp_size = 1
#     if pp_size is None or pp_size <= 0:
#         pp_size = 1
#     if ep_size is None or ep_size <= 0:
#         ep_size = 1
#
#     # infer dp_size if not provided
#     if dp_size is None or dp_size <= 0:
#         total_parallel_ranks = tp_size * cp_size * pp_size
#         if world_size % total_parallel_ranks != 0:
#             raise ValueError(
#                 f"world_size ({world_size}) must be divisible by (tp_size * cp_size * pp_size) "
#                 f"({tp_size} * {cp_size} * {pp_size} = {total_parallel_ranks})"
#             )
#         dp_size = world_size // total_parallel_ranks
#
#     if dp_replicate_size is None or dp_replicate_size <= 0:
#         dp_replicate_size = 1
#
#     # HSDP usecase
#     # dp_size = dp_replicate_size * dp_shard_size
#     assert dp_size % dp_replicate_size == 0, "dp_size must be a multiple of dp_replicate_size"
#     assert dp_replicate_size < dp_size or dp_replicate_size == 1, (
#         "dp_replicate_size must be less than dp_size since ddp usecase is not supported by FSDP2"
#     )
#     dp_cp_size = dp_size * cp_size
#     assert dp_cp_size % ep_size == 0, f"{dp_cp_size=} must be a multiple of {ep_size=}"
#     if ep_size < dp_cp_size:
#         ep_shard_size = dp_cp_size // ep_size
#     else:
#         ep_shard_size = 1
#
#     dp_shard_size = dp_size // dp_replicate_size
#
#     # Build main device mesh
#     mesh_shape = (pp_size, dp_replicate_size, dp_shard_size, cp_size, tp_size)
#     mesh_names = ("pp", "dp_replicate", "dp_shard", "cp", "tp")
#     for shape, name in zip(mesh_shape, mesh_names):
#         assert isinstance(shape, int), "Expected {} to be an int, but got {}".format(name, type(shape))
#         assert shape > 0, "Expected {} > 0, {}".format(name, shape)
#
#     device_mesh = init_device_mesh(
#         device_type="cuda" if backend == "nccl" else "cpu",
#         mesh_shape=mesh_shape,
#         mesh_dim_names=mesh_names,
#     )
#
#     # Create submeshes
#     # based on https://github.com/pytorch/torchtitan/blob/d282cf2ce9ca8049b4b8423c1d7578c80426576f/torchtitan/distributed/parallel_dims.py#L191
#     # Mesh for data loading (no communication on this mesh)
#     dp_mesh_dim_names = []
#     # Mesh for param sharding
#     dp_shard_cp_mesh_dim_names = []
#     # Mesh for loss all-reduce
#     dp_cp_mesh_dim_names = []
#
#     # for dp_replicate:
#     dp_mesh_dim_names.append("dp_replicate")
#     dp_cp_mesh_dim_names.append("dp_replicate")
#     # for dp_shard:
#     dp_mesh_dim_names.append("dp_shard")
#     dp_shard_cp_mesh_dim_names.append("dp_shard")
#     dp_cp_mesh_dim_names.append("dp_shard")
#     # for cp:
#     dp_shard_cp_mesh_dim_names.append("cp")
#     dp_cp_mesh_dim_names.append("cp")
#
#     # submesh for dp
#     device_mesh[tuple(dp_mesh_dim_names)]._flatten(mesh_dim_name="dp")
#     # submesh for dp_shard_cp
#     device_mesh[tuple(dp_shard_cp_mesh_dim_names)]._flatten(mesh_dim_name="dp_shard_cp")
#     # submesh for dp_cp
#     device_mesh[tuple(dp_cp_mesh_dim_names)]._flatten(mesh_dim_name="dp_cp")
#
#     return device_mesh, dp_size, dp_shard_size, ep_shard_size
#
#
# def _create_moe_mesh(self, pp_size: int, ep_shard_size: int, ep_size: int, backend: str):
#     """
#     Creates MOE mesh for expert parallelism.
#
#     Mesh shape: (pp_size, ep_shard_size, ep_size)
#     Mesh names: ("pp", "ep_shard", "ep")
#     """
#     from torch.distributed.device_mesh import init_device_mesh
#
#     mesh_shape = (pp_size, ep_shard_size, ep_size)
#     mesh_names = ("pp", "ep_shard", "ep")
#     for shape, name in zip(mesh_shape, mesh_names):
#         assert isinstance(shape, int), "Expected {} to be an int, but got {}".format(name, type(shape))
#         assert shape > 0, "Expected {} > 0, {}".format(name, shape)
#
#     moe_mesh = init_device_mesh(
#         device_type="cuda" if backend == "nccl" else "cpu",
#         mesh_shape=mesh_shape,
#         mesh_dim_names=mesh_names,
#     )
#     return moe_mesh
