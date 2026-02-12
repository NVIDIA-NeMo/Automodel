# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Device mesh creation utilities for distributed training.

This module provides a central function to create device meshes based on the
distributed config type (FSDP2, MegatronFSDP, or DDP).

Usage:
    from nemo_automodel.components.distributed.config import FSDP2Config
    from nemo_automodel.components.distributed.device_mesh import create_device_mesh

    config = FSDP2Config(sequence_parallel=True)
    device_mesh, moe_mesh = create_device_mesh(
        config,
        tp_size=2,
        pp_size=1,
        dp_replicate_size=2,
        world_size=8,
    )
"""

from typing import Optional, Tuple, Union

from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from nemo_automodel.components.distributed.config import (
    DDPConfig,
    FSDP2Config,
    MegatronFSDPConfig,
)


def create_device_mesh(
    distributed_config: Union[FSDP2Config, MegatronFSDPConfig, DDPConfig],
    *,
    dp_size: Optional[int] = None,
    dp_replicate_size: Optional[int] = None,
    tp_size: int = 1,
    pp_size: int = 1,
    cp_size: int = 1,
    ep_size: int = 1,
    world_size: int,
) -> Tuple[Optional[DeviceMesh], Optional[DeviceMesh]]:
    """Create device mesh based on distributed config type.

    Routes to the appropriate mesh creation logic based on config type.

    Args:
        distributed_config: The distributed config (FSDP2Config, MegatronFSDPConfig,
            or DDPConfig).
        dp_size: Data parallel size. If None, inferred from world_size and other
            parallelism sizes.
        dp_replicate_size: FSDP2-only. Size of the replication group for HSDP
            (Hybrid Sharded Data Parallel). If None or <= 0, defaults to 1.
            Must be a divisor of dp_size.
        tp_size: Tensor parallel size.
        pp_size: Pipeline parallel size.
        cp_size: Context parallel size.
        ep_size: Expert parallel size (for MoE models).
        world_size: Total number of processes.

    Returns:
        tuple: (device_mesh, moe_mesh)
            - For FSDP2Config: Full device mesh + optional moe_mesh (if ep_size > 1)
            - For MegatronFSDPConfig: Device mesh + None
            - For DDPConfig: (None, None) - DDP doesn't use device mesh

    Raises:
        ValueError: If dp_replicate_size is provided with non-FSDP2 config.
        ValueError: If world_size is not divisible by parallelism sizes.
    """
    # Validate FSDP2-only params
    if dp_replicate_size is not None and dp_replicate_size > 1:
        if not isinstance(distributed_config, FSDP2Config):
            raise ValueError("dp_replicate_size is only supported with FSDP2Config")

    if isinstance(distributed_config, FSDP2Config):
        return _create_fsdp2_device_mesh(
            dp_size=dp_size,
            dp_replicate_size=dp_replicate_size,
            tp_size=tp_size,
            pp_size=pp_size,
            cp_size=cp_size,
            ep_size=ep_size,
            world_size=world_size,
            backend=distributed_config.backend,
        )
    elif isinstance(distributed_config, MegatronFSDPConfig):
        mesh = _create_megatron_fsdp_device_mesh(
            dp_size=dp_size,
            tp_size=tp_size,
            cp_size=cp_size,
            world_size=world_size,
            backend=distributed_config.backend,
        )
        return mesh, None
    elif isinstance(distributed_config, DDPConfig):
        return None, None  # DDP doesn't use device mesh
    else:
        raise ValueError(f"Unknown distributed config type: {type(distributed_config)}")


def _create_fsdp2_device_mesh(
    dp_size: Optional[int],
    dp_replicate_size: Optional[int],
    tp_size: int,
    pp_size: int,
    cp_size: int,
    ep_size: int,
    world_size: int,
    backend: str,
) -> Tuple[DeviceMesh, Optional[DeviceMesh]]:
    """
    Create device mesh for FSDP2.

    Mesh shape: (pp_size, dp_replicate_size, dp_shard_size, cp_size, tp_size)
    Mesh names: ("pp", "dp_replicate", "dp_shard", "cp", "tp")

    Also creates flattened submeshes:
        - "dp": dp_replicate + dp_shard
        - "dp_shard_cp": dp_shard + cp
        - "dp_cp": dp_replicate + dp_shard + cp

    Args:
        dp_size: Data parallel size. If None, inferred from world_size.
        dp_replicate_size: Size of the replication group for HSDP.
        tp_size: Tensor parallel size.
        pp_size: Pipeline parallel size.
        cp_size: Context parallel size.
        ep_size: Expert parallel size (for MoE models).
        world_size: Total number of processes.
        backend: Distributed backend ('nccl' or 'gloo').

    Returns:
        tuple: (device_mesh, moe_mesh)
    """
    # Normalize sizes
    if tp_size is None or tp_size <= 0:
        tp_size = 1
    if cp_size is None or cp_size <= 0:
        cp_size = 1
    if pp_size is None or pp_size <= 0:
        pp_size = 1
    if ep_size is None or ep_size <= 0:
        ep_size = 1

    # Infer dp_size if not provided
    if dp_size is None or dp_size <= 0:
        total_parallel_ranks = tp_size * cp_size * pp_size
        if world_size % total_parallel_ranks != 0:
            raise ValueError(
                f"world_size ({world_size}) must be divisible by (tp_size * cp_size * pp_size) "
                f"({tp_size} * {cp_size} * {pp_size} = {total_parallel_ranks})"
            )
        dp_size = world_size // total_parallel_ranks

    if dp_replicate_size is None or dp_replicate_size <= 0:
        dp_replicate_size = 1

    # HSDP usecase: dp_size = dp_replicate_size * dp_shard_size
    assert dp_size % dp_replicate_size == 0, "dp_size must be a multiple of dp_replicate_size"
    assert dp_replicate_size < dp_size or dp_replicate_size == 1, (
        "dp_replicate_size must be less than dp_size since ddp usecase is not supported by FSDP2"
    )

    # Expert parallelism calculations
    dp_cp_size = dp_size * cp_size
    assert dp_cp_size % ep_size == 0, f"{dp_cp_size=} must be a multiple of {ep_size=}"
    if ep_size < dp_cp_size:
        ep_shard_size = dp_cp_size // ep_size
    else:
        ep_shard_size = 1

    dp_shard_size = dp_size // dp_replicate_size

    # Build main device mesh
    mesh_shape = (pp_size, dp_replicate_size, dp_shard_size, cp_size, tp_size)
    mesh_names = ("pp", "dp_replicate", "dp_shard", "cp", "tp")
    for shape, name in zip(mesh_shape, mesh_names):
        assert isinstance(shape, int), f"Expected {name} to be an int, but got {type(shape)}"
        assert shape > 0, f"Expected {name} > 0, got {shape}"

    device_mesh = init_device_mesh(
        device_type="cuda" if backend == "nccl" else "cpu",
        mesh_shape=mesh_shape,
        mesh_dim_names=mesh_names,
    )

    # Create flattened submeshes
    # Based on https://github.com/pytorch/torchtitan/blob/d282cf2ce9ca8049b4b8423c1d7578c80426576f/torchtitan/distributed/parallel_dims.py#L191
    dp_mesh_dim_names = []  # Mesh for data loading (no communication on this mesh)
    dp_shard_cp_mesh_dim_names = []  # Mesh for param sharding
    dp_cp_mesh_dim_names = []  # Mesh for loss all-reduce

    # for dp_replicate:
    dp_mesh_dim_names.append("dp_replicate")
    dp_cp_mesh_dim_names.append("dp_replicate")
    # for dp_shard:
    dp_mesh_dim_names.append("dp_shard")
    dp_shard_cp_mesh_dim_names.append("dp_shard")
    dp_cp_mesh_dim_names.append("dp_shard")
    # for cp:
    dp_shard_cp_mesh_dim_names.append("cp")
    dp_cp_mesh_dim_names.append("cp")

    # Flatten submeshes
    device_mesh[tuple(dp_mesh_dim_names)]._flatten(mesh_dim_name="dp")
    device_mesh[tuple(dp_shard_cp_mesh_dim_names)]._flatten(mesh_dim_name="dp_shard_cp")
    device_mesh[tuple(dp_cp_mesh_dim_names)]._flatten(mesh_dim_name="dp_cp")

    # Create MOE mesh if ep_size > 1
    moe_mesh = None
    if ep_size > 1:
        moe_mesh = _create_moe_mesh(
            pp_size=pp_size,
            ep_shard_size=ep_shard_size,
            ep_size=ep_size,
            backend=backend,
        )

    return device_mesh, moe_mesh


def _create_megatron_fsdp_device_mesh(
    dp_size: Optional[int],
    tp_size: int,
    cp_size: int,
    world_size: int,
    backend: str,
) -> DeviceMesh:
    """
    Create device mesh for MegatronFSDP.

    Mesh shape: (dp_size, cp_size, tp_size)
    Mesh names: ("dp", "cp", "tp")

    Also creates flattened submesh "dp_cp" if cp_size > 1.

    Args:
        dp_size: Data parallel size. If None, inferred from world_size.
        tp_size: Tensor parallel size.
        cp_size: Context parallel size.
        world_size: Total number of processes.
        backend: Distributed backend ('nccl' or 'gloo').

    Returns:
        DeviceMesh: The device mesh for MegatronFSDP.
    """
    # Normalize sizes
    tp_size = tp_size or 1
    cp_size = cp_size or 1

    # Infer dp_size if not provided
    if dp_size is None or dp_size <= 0:
        total_parallel_ranks = tp_size * cp_size
        if world_size % total_parallel_ranks != 0:
            raise ValueError(
                f"world_size ({world_size}) must be divisible by (tp_size * cp_size) "
                f"({tp_size} * {cp_size} = {total_parallel_ranks})"
            )
        dp_size = world_size // total_parallel_ranks

    mesh_shape = (dp_size, cp_size, tp_size)
    mesh_names = ("dp", "cp", "tp")
    for shape, name in zip(mesh_shape, mesh_names):
        assert isinstance(shape, int), f"Expected {name} to be an int, but got {type(shape)}"
        assert shape > 0, f"Expected {name} > 0, got {shape}"

    # Build mesh [dp, cp, tp]
    device_mesh = init_device_mesh(
        device_type="cuda" if backend == "nccl" else "cpu",
        mesh_shape=mesh_shape,
        mesh_dim_names=mesh_names,
    )

    # Flatten dp+cp if cp > 1
    if cp_size > 1:
        device_mesh[("dp", "cp")]._flatten(mesh_dim_name="dp_cp")

    return device_mesh


def _create_moe_mesh(
    pp_size: int,
    ep_shard_size: int,
    ep_size: int,
    backend: str,
) -> DeviceMesh:
    """
    Create MOE mesh for expert parallelism.

    Mesh shape: (pp_size, ep_shard_size, ep_size)
    Mesh names: ("pp", "ep_shard", "ep")

    Args:
        pp_size: Pipeline parallel size.
        ep_shard_size: Expert shard size (dp_cp_size // ep_size).
        ep_size: Expert parallel size.
        backend: Distributed backend ('nccl' or 'gloo').

    Returns:
        DeviceMesh: The MOE mesh for expert parallelism.
    """
    mesh_shape = (pp_size, ep_shard_size, ep_size)
    mesh_names = ("pp", "ep_shard", "ep")
    for shape, name in zip(mesh_shape, mesh_names):
        assert isinstance(shape, int), f"Expected {name} to be an int, but got {type(shape)}"
        assert shape > 0, f"Expected {name} > 0, got {shape}"

    moe_mesh = init_device_mesh(
        device_type="cuda" if backend == "nccl" else "cpu",
        mesh_shape=mesh_shape,
        mesh_dim_names=mesh_names,
    )
    return moe_mesh
