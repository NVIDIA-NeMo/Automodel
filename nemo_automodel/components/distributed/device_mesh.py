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

"""Mesh context and device mesh creation utilities for distributed training.

This module is the canonical entry point for building the ``MeshContext`` used
by recipe setup and distributed managers. Raw device mesh construction remains
private to keep the public API aligned with its return type. Mesh access helpers
such as ``get_flat_mesh`` live in :mod:`nemo_automodel.components.distributed.mesh_utils`.
"""

import dataclasses
from typing import Any

from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from nemo_automodel.components.distributed.config import (
    DDPConfig,
    FSDP2Config,
    MegatronFSDPConfig,
)
from nemo_automodel.components.distributed.init_utils import get_world_size_safe
from nemo_automodel.components.distributed.mesh import (
    STRATEGY_ALIASES,
    STRATEGY_MAP,
    MeshAxisName,
    MeshContext,
    normalize_strategy_name,
)
from nemo_automodel.components.distributed.pipelining.config import PipelineConfig
from nemo_automodel.components.moe.config import MoEParallelizerConfig


def create_mesh_context(
    strategy: str | FSDP2Config | MegatronFSDPConfig | DDPConfig = "fsdp2",
    *,
    distributed_config: FSDP2Config | MegatronFSDPConfig | DDPConfig | None = None,
    dp_size: int | None = None,
    dp_replicate_size: int | None = None,
    tp_size: int = 1,
    pp_size: int = 1,
    cp_size: int = 1,
    ep_size: int = 1,
    world_size: int | None = None,
    pipeline_config: PipelineConfig | dict | None = None,
    moe_config: MoEParallelizerConfig | dict | None = None,
    **strategy_kwargs: Any,
) -> MeshContext:
    """Create a :class:`MeshContext` from a strategy name or config object.

    Args:
        strategy: Strategy name (``"fsdp2"``, ``"megatron_fsdp"``, ``"mfsdp"``,
            or ``"ddp"``) or an already-instantiated strategy config.
        distributed_config: Backward-compatible keyword alias for passing an
            already-instantiated strategy config.
        dp_size: Data parallel size. If None, inferred from world_size and other
            parallelism sizes.
        dp_replicate_size: FSDP2-only. Size of the replication group for HSDP.
        tp_size: Tensor parallel size.
        pp_size: Pipeline parallel size.
        cp_size: Context parallel size.
        ep_size: Expert parallel size.
        world_size: Total number of processes. If None, inferred from the
            distributed environment.
        pipeline_config: Optional pipeline config object or kwargs dict. A default
            ``PipelineConfig`` is created when ``pp_size > 1`` and no config is
            supplied.
        moe_config: Optional MoE parallelizer config object or kwargs dict. A
            default ``MoEParallelizerConfig`` is created when ``ep_size > 1`` and
            no config is supplied.
        **strategy_kwargs: Strategy-specific config fields used when *strategy*
            is a string.

    Returns:
        A fully initialized ``MeshContext``.

    Raises:
        ValueError: If an unknown strategy name or strategy option is provided.
        ValueError: If dp_replicate_size is provided with non-FSDP2 config.
        ValueError: If world_size is not divisible by requested parallelism sizes.
    """
    if world_size is None:
        world_size = get_world_size_safe()
    if distributed_config is not None:
        if strategy != "fsdp2":
            raise ValueError("Pass either strategy or distributed_config, not both")
        strategy = distributed_config

    strategy_config = _resolve_strategy_config(
        strategy,
        strategy_kwargs=strategy_kwargs,
    )

    device_mesh, moe_mesh = _create_device_meshes(
        strategy_config,
        dp_size=dp_size,
        dp_replicate_size=dp_replicate_size,
        tp_size=tp_size,
        pp_size=pp_size,
        cp_size=cp_size,
        ep_size=ep_size,
        world_size=world_size,
    )

    return MeshContext(
        strategy_config=strategy_config,
        pipeline_config=_resolve_pipeline_config(pipeline_config, pp_size),
        moe_config=_resolve_moe_config(moe_config, ep_size),
        device_mesh=device_mesh,
        moe_mesh=moe_mesh,
    )


def _resolve_strategy_config(
    strategy: str | FSDP2Config | MegatronFSDPConfig | DDPConfig,
    *,
    strategy_kwargs: dict[str, Any],
) -> FSDP2Config | MegatronFSDPConfig | DDPConfig:
    """Resolve a strategy name or config object into a strategy config."""
    if isinstance(strategy, (FSDP2Config, MegatronFSDPConfig, DDPConfig)):
        if strategy_kwargs:
            raise ValueError("Strategy-specific keyword arguments require strategy to be a string")
        return strategy

    if not isinstance(strategy, str):
        raise ValueError(f"Unknown distributed strategy type: {type(strategy)}")

    strategy_name = normalize_strategy_name(strategy)
    if strategy_name not in STRATEGY_MAP:
        valid = sorted(set(STRATEGY_MAP) | set(STRATEGY_ALIASES))
        raise ValueError(f"Unknown strategy: {strategy}. Valid strategies: {valid}")

    strategy_cls = STRATEGY_MAP[strategy_name]
    strategy_kwargs = strategy_kwargs.copy()
    _validate_strategy_kwargs(strategy_name, strategy_cls, strategy_kwargs)
    return strategy_cls(**strategy_kwargs)


def _validate_strategy_kwargs(strategy_name: str, strategy_cls: type, strategy_kwargs: dict[str, Any]) -> None:
    """Check that strategy kwargs are accepted by the strategy dataclass."""
    valid_fields = {field.name for field in dataclasses.fields(strategy_cls)}
    unknown = set(strategy_kwargs) - valid_fields
    if unknown:
        raise ValueError(f"Unknown options for strategy '{strategy_name}': {sorted(unknown)}")


def _resolve_pipeline_config(pipeline_config: PipelineConfig | dict | None, pp_size: int) -> PipelineConfig | None:
    """Resolve pipeline config inputs using recipe-compatible defaults."""
    if (pp_size or 1) <= 1:
        return None
    if pipeline_config is None:
        return PipelineConfig()
    if isinstance(pipeline_config, PipelineConfig):
        return pipeline_config
    return PipelineConfig(**pipeline_config)


def _resolve_moe_config(moe_config: MoEParallelizerConfig | dict | None, ep_size: int) -> MoEParallelizerConfig | None:
    """Resolve MoE config inputs using recipe-compatible defaults."""
    if (ep_size or 1) <= 1:
        return None
    if moe_config is None:
        return MoEParallelizerConfig()
    if isinstance(moe_config, MoEParallelizerConfig):
        return moe_config
    return MoEParallelizerConfig(**moe_config)


def _create_device_meshes(
    distributed_config: FSDP2Config | MegatronFSDPConfig | DDPConfig,
    *,
    dp_size: int | None = None,
    dp_replicate_size: int | None = None,
    tp_size: int = 1,
    pp_size: int = 1,
    cp_size: int = 1,
    ep_size: int = 1,
    world_size: int,
) -> tuple[DeviceMesh | None, DeviceMesh | None]:
    """Create raw device meshes based on distributed config type."""
    if dp_replicate_size is not None and dp_replicate_size > 1 and not isinstance(distributed_config, FSDP2Config):
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
        if (pp_size or 1) > 1:
            raise ValueError("megatron_fsdp does not support pipeline parallelism")
        if (ep_size or 1) > 1:
            raise ValueError("megatron_fsdp does not support expert parallelism")
        mesh = _create_megatron_fsdp_device_mesh(
            dp_size=dp_size,
            tp_size=tp_size,
            cp_size=cp_size,
            world_size=world_size,
            backend=distributed_config.backend,
        )
        return mesh, None
    elif isinstance(distributed_config, DDPConfig):
        if (tp_size or 1) > 1:
            raise ValueError("ddp does not support tensor parallelism")
        if (pp_size or 1) > 1:
            raise ValueError("ddp does not support pipeline parallelism")
        if (cp_size or 1) > 1:
            raise ValueError("ddp does not support context parallelism")
        if (ep_size or 1) > 1:
            raise ValueError("ddp does not support expert parallelism")
        return None, None
    else:
        raise ValueError(f"Unknown distributed config type: {type(distributed_config)}")


def _create_fsdp2_device_mesh(
    dp_size: int | None,
    dp_replicate_size: int | None,
    tp_size: int,
    pp_size: int,
    cp_size: int,
    ep_size: int,
    world_size: int,
    backend: str,
) -> tuple[DeviceMesh, DeviceMesh | None]:
    """Create the FSDP2 root mesh and optional MoE mesh."""
    if tp_size is None or tp_size <= 0:
        tp_size = 1
    if cp_size is None or cp_size <= 0:
        cp_size = 1
    if pp_size is None or pp_size <= 0:
        pp_size = 1
    if ep_size is None or ep_size <= 0:
        ep_size = 1

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

    assert dp_size % dp_replicate_size == 0, "dp_size must be a multiple of dp_replicate_size"
    assert dp_replicate_size < dp_size or dp_replicate_size == 1, (
        f"dp_replicate_size={dp_replicate_size} must be less than dp_size={dp_size} "
        "since DDP usecase is not supported by FSDP2"
    )

    non_pp_size = dp_size * cp_size * tp_size
    assert non_pp_size % ep_size == 0, f"{non_pp_size=} must be a multiple of {ep_size=}"
    ep_shard_size = non_pp_size // ep_size if ep_size < non_pp_size else 1
    dp_shard_size = dp_size // dp_replicate_size

    mesh_shape = (pp_size, dp_replicate_size, dp_shard_size, cp_size, tp_size)
    mesh_names = (
        MeshAxisName.PP,
        MeshAxisName.DP_REPLICATE,
        MeshAxisName.DP_SHARD,
        MeshAxisName.CP,
        MeshAxisName.TP,
    )
    for shape, name in zip(mesh_shape, mesh_names):
        assert isinstance(shape, int), f"Expected {name} to be an int, but got {type(shape)}"
        assert shape > 0, f"Expected {name} > 0, got {shape}"

    device_mesh = init_device_mesh(
        device_type="cuda" if backend == "nccl" else "cpu",
        mesh_shape=mesh_shape,
        mesh_dim_names=mesh_names,
    )

    dp_mesh_dim_names = [MeshAxisName.DP_REPLICATE, MeshAxisName.DP_SHARD]
    dp_shard_cp_mesh_dim_names = [MeshAxisName.DP_SHARD, MeshAxisName.CP]
    dp_cp_mesh_dim_names = [MeshAxisName.DP_REPLICATE, MeshAxisName.DP_SHARD, MeshAxisName.CP]

    _dp_flat = device_mesh[tuple(dp_mesh_dim_names)]._flatten(mesh_dim_name=MeshAxisName.DP)
    _dp_shard_cp_flat = device_mesh[tuple(dp_shard_cp_mesh_dim_names)]._flatten(mesh_dim_name=MeshAxisName.DP_SHARD_CP)
    _dp_cp_flat = device_mesh[tuple(dp_cp_mesh_dim_names)]._flatten(mesh_dim_name=MeshAxisName.DP_CP)
    if not hasattr(device_mesh, "_flatten_mapping"):
        device_mesh._flatten_mapping = {}
    device_mesh._flatten_mapping.setdefault(MeshAxisName.DP, _dp_flat)
    device_mesh._flatten_mapping.setdefault(MeshAxisName.DP_SHARD_CP, _dp_shard_cp_flat)
    device_mesh._flatten_mapping.setdefault(MeshAxisName.DP_CP, _dp_cp_flat)

    moe_mesh = None
    if ep_size > 1:
        non_pp_dims = (MeshAxisName.DP_REPLICATE, MeshAxisName.DP_SHARD, MeshAxisName.CP, MeshAxisName.TP)
        non_pp_mesh = device_mesh[non_pp_dims]._flatten()
        moe_mesh = _unflatten_compat(
            non_pp_mesh,
            0,
            (ep_shard_size, ep_size),
            (MeshAxisName.EP_SHARD, MeshAxisName.EP),
        )

    return device_mesh, moe_mesh


def _create_megatron_fsdp_device_mesh(
    dp_size: int | None,
    tp_size: int,
    cp_size: int,
    world_size: int,
    backend: str,
) -> DeviceMesh:
    """Create the Megatron FSDP mesh."""
    tp_size = tp_size or 1
    cp_size = cp_size or 1

    if dp_size is None or dp_size <= 0:
        total_parallel_ranks = tp_size * cp_size
        if world_size % total_parallel_ranks != 0:
            raise ValueError(
                f"world_size ({world_size}) must be divisible by (tp_size * cp_size) "
                f"({tp_size} * {cp_size} = {total_parallel_ranks})"
            )
        dp_size = world_size // total_parallel_ranks

    mesh_shape = (dp_size, cp_size, tp_size)
    mesh_names = (MeshAxisName.DP, MeshAxisName.CP, MeshAxisName.TP)
    for shape, name in zip(mesh_shape, mesh_names):
        assert isinstance(shape, int), f"Expected {name} to be an int, but got {type(shape)}"
        assert shape > 0, f"Expected {name} > 0, got {shape}"

    device_mesh = init_device_mesh(
        device_type="cuda" if backend == "nccl" else "cpu",
        mesh_shape=mesh_shape,
        mesh_dim_names=mesh_names,
    )

    if cp_size > 1:
        _dp_cp_flat = device_mesh[(MeshAxisName.DP, MeshAxisName.CP)]._flatten(mesh_dim_name=MeshAxisName.DP_CP)
        if not hasattr(device_mesh, "_flatten_mapping"):
            device_mesh._flatten_mapping = {}
        device_mesh._flatten_mapping.setdefault(MeshAxisName.DP_CP, _dp_cp_flat)

    return device_mesh


def _unflatten_compat(flat_mesh: DeviceMesh, dim: int, sizes: tuple, names: tuple) -> DeviceMesh:
    """Compatibility shim for DeviceMesh._unflatten(), added in PyTorch 2.10."""
    if hasattr(flat_mesh, "_unflatten"):
        return flat_mesh._unflatten(dim, sizes, names)
    new_mesh_tensor = flat_mesh.mesh.reshape(sizes)
    from torch.distributed.device_mesh import DeviceMesh as _DeviceMesh

    return _DeviceMesh(flat_mesh.device_type, new_mesh_tensor, mesh_dim_names=names)


__all__ = [
    "create_mesh_context",
    "_create_device_meshes",
    "_create_fsdp2_device_mesh",
    "_create_megatron_fsdp_device_mesh",
    "_unflatten_compat",
]
