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

"""Device mesh construction and access utilities for distributed training."""

import datetime
from dataclasses import dataclass, field

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from nemo_automodel.components.distributed.config import (
    DDPConfig,
    DistributedStrategyConfig,
    FSDP2Config,
    MegatronFSDPConfig,
)
from nemo_automodel.components.distributed.mesh import MeshAxisName, ParallelismSizes

__all__ = [
    "_create_device_meshes",
    "_create_fsdp2_device_mesh",
    "_create_megatron_fsdp_device_mesh",
    "_unflatten_compat",
    "get_flat_mesh",
    "get_submesh",
    "get_fsdp_dp_mesh",
]


def _degree(value: int | None) -> int:
    return value if isinstance(value, int) and value > 0 else 1


def _require_size_one(strategy_name: str, size: int | None, feature_name: str) -> None:
    if _degree(size) > 1:
        raise ValueError(f"{strategy_name} does not support {feature_name}")


@dataclass(frozen=True)
class _MeshSpec:
    """Named mesh shape plus derived flattened axes."""

    shape: tuple[int, ...]
    axes: tuple[MeshAxisName, ...]
    flattened_axes: dict[MeshAxisName, tuple[MeshAxisName, ...]] = field(default_factory=dict)


def _create_device_meshes(
    strategy_config: DistributedStrategyConfig,
    parallelism: ParallelismSizes,
    *,
    world_size: int,
    timeout_minutes: int | None = None,
) -> tuple[DeviceMesh | None, DeviceMesh | None]:
    """Create raw device meshes based on distributed config type."""
    if (
        parallelism.dp_replicate_size is not None
        and parallelism.dp_replicate_size > 1
        and not isinstance(strategy_config, FSDP2Config)
    ):
        raise ValueError("dp_replicate_size is only supported with FSDP2Config")

    if isinstance(strategy_config, FSDP2Config):
        return _create_fsdp2_device_mesh(
            parallelism,
            world_size=world_size,
            timeout_minutes=timeout_minutes,
        )
    elif isinstance(strategy_config, MegatronFSDPConfig):
        _require_size_one("megatron_fsdp", parallelism.pp_size, "pipeline parallelism")
        _require_size_one("megatron_fsdp", parallelism.ep_size, "expert parallelism")
        mesh = _create_megatron_fsdp_device_mesh(
            parallelism,
            world_size=world_size,
            timeout_minutes=timeout_minutes,
        )
        return mesh, None
    elif isinstance(strategy_config, DDPConfig):
        _require_size_one("ddp", parallelism.tp_size, "tensor parallelism")
        _require_size_one("ddp", parallelism.pp_size, "pipeline parallelism")
        _require_size_one("ddp", parallelism.cp_size, "context parallelism")
        _require_size_one("ddp", parallelism.ep_size, "expert parallelism")
        return None, None
    else:
        raise ValueError(f"Unknown distributed strategy config type: {type(strategy_config)}")


def _infer_dp_size(
    dp_size: int | None,
    *,
    world_size: int,
    non_dp_size: int,
    expression: str,
    factors: tuple[int, ...],
) -> int:
    if dp_size is not None and dp_size > 0:
        return dp_size

    if world_size % non_dp_size != 0:
        factors_str = " * ".join(str(factor) for factor in factors)
        raise ValueError(
            f"world_size ({world_size}) must be divisible by ({expression}) ({factors_str} = {non_dp_size})"
        )
    return world_size // non_dp_size


def _mesh_device_type() -> str:
    if dist.is_available() and dist.is_initialized():
        backend = str(dist.get_backend()).lower()
        return "cuda" if "nccl" in backend and torch.cuda.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _init_named_mesh(spec: _MeshSpec, *, timeout_minutes: int | None = None) -> DeviceMesh:
    _validate_mesh_spec(spec)
    device_type = _mesh_device_type()
    device_mesh = init_device_mesh(
        device_type=device_type,
        mesh_shape=spec.shape,
        mesh_dim_names=spec.axes,
        backend_override=_nccl_backend_override(spec.axes, device_type=device_type, timeout_minutes=timeout_minutes),
    )
    _register_flattened_axes(device_mesh, spec.flattened_axes, timeout_minutes=timeout_minutes)
    return device_mesh


def _nccl_backend_override(
    axes: tuple[str, ...],
    *,
    device_type: str,
    timeout_minutes: int | None,
):
    """Create per-axis NCCL options for DeviceMesh subgroups.

    ``init_process_group(timeout=...)`` configures the default process group, but
    ``init_device_mesh`` creates additional per-axis process groups. Without a
    backend override those groups keep PyTorch's default NCCL timeout.
    """
    if timeout_minutes is None or device_type != "cuda":
        return None

    timeout = datetime.timedelta(minutes=timeout_minutes)
    override = {}
    for axis in axes:
        options = dist.ProcessGroupNCCL.Options()
        options._timeout = timeout
        override[axis] = ("nccl", options)
    return override


def _validate_mesh_spec(spec: _MeshSpec) -> None:
    for shape, axis in zip(spec.shape, spec.axes):
        assert isinstance(shape, int), f"Expected {axis} to be an int, but got {type(shape)}"
        assert shape > 0, f"Expected {axis} > 0, got {shape}"


def _register_flattened_axes(
    device_mesh: DeviceMesh,
    flattened_axes: dict[MeshAxisName, tuple[MeshAxisName, ...]],
    *,
    timeout_minutes: int | None = None,
) -> None:
    if not flattened_axes:
        return
    if not hasattr(device_mesh, "_flatten_mapping"):
        device_mesh._flatten_mapping = {}
    backend_overrides = _nccl_backend_override(
        tuple(flattened_axes),
        device_type=device_mesh.device_type,
        timeout_minutes=timeout_minutes,
    )
    for flattened_axis, source_axes in flattened_axes.items():
        flattened_mesh = device_mesh[source_axes]._flatten(
            mesh_dim_name=flattened_axis,
            backend_override=backend_overrides.get(flattened_axis) if backend_overrides else None,
        )
        device_mesh._flatten_mapping.setdefault(flattened_axis, flattened_mesh)


def _create_fsdp2_device_mesh(
    parallelism: ParallelismSizes,
    *,
    world_size: int,
    timeout_minutes: int | None = None,
) -> tuple[DeviceMesh, DeviceMesh | None]:
    """Create the FSDP2 root mesh and optional MoE mesh."""
    tp_size = _degree(parallelism.tp_size)
    cp_size = _degree(parallelism.cp_size)
    pp_size = _degree(parallelism.pp_size)
    ep_size = _degree(parallelism.ep_size)
    dp_replicate_size = _degree(parallelism.dp_replicate_size)
    dp_size = _infer_dp_size(
        parallelism.dp_size,
        world_size=world_size,
        non_dp_size=tp_size * cp_size * pp_size,
        expression="tp_size * cp_size * pp_size",
        factors=(tp_size, cp_size, pp_size),
    )

    if dp_size % dp_replicate_size != 0:
        raise ValueError("dp_size must be a multiple of dp_replicate_size")
    if dp_replicate_size >= dp_size and dp_replicate_size != 1:
        raise ValueError(
            f"dp_replicate_size={dp_replicate_size} must be less than dp_size={dp_size} "
            "since DDP usecase is not supported by FSDP2"
        )

    non_pp_size = dp_size * cp_size * tp_size
    if non_pp_size % ep_size != 0:
        raise ValueError(f"{non_pp_size=} must be a multiple of {ep_size=}")
    ep_shard_size = non_pp_size // ep_size if ep_size < non_pp_size else 1
    dp_shard_size = dp_size // dp_replicate_size

    device_mesh = _init_named_mesh(
        _MeshSpec(
            shape=(pp_size, dp_replicate_size, dp_shard_size, cp_size, tp_size),
            axes=(
                MeshAxisName.PP,
                MeshAxisName.DP_REPLICATE,
                MeshAxisName.DP_SHARD,
                MeshAxisName.CP,
                MeshAxisName.TP,
            ),
            flattened_axes={
                MeshAxisName.DP: (MeshAxisName.DP_REPLICATE, MeshAxisName.DP_SHARD),
                MeshAxisName.DP_SHARD_CP: (MeshAxisName.DP_SHARD, MeshAxisName.CP),
                MeshAxisName.DP_CP: (MeshAxisName.DP_REPLICATE, MeshAxisName.DP_SHARD, MeshAxisName.CP),
            },
        ),
        timeout_minutes=timeout_minutes,
    )

    moe_mesh = None
    if ep_size > 1:
        moe_mesh = _create_moe_mesh(
            device_mesh,
            ep_shard_size=ep_shard_size,
            ep_size=ep_size,
            timeout_minutes=timeout_minutes,
        )

    return device_mesh, moe_mesh


def _create_megatron_fsdp_device_mesh(
    parallelism: ParallelismSizes,
    *,
    world_size: int,
    timeout_minutes: int | None = None,
) -> DeviceMesh:
    """Create the Megatron FSDP mesh."""
    tp_size = _degree(parallelism.tp_size)
    cp_size = _degree(parallelism.cp_size)
    dp_size = _infer_dp_size(
        parallelism.dp_size,
        world_size=world_size,
        non_dp_size=tp_size * cp_size,
        expression="tp_size * cp_size",
        factors=(tp_size, cp_size),
    )

    return _init_named_mesh(
        _MeshSpec(
            shape=(dp_size, cp_size, tp_size),
            axes=(MeshAxisName.DP, MeshAxisName.CP, MeshAxisName.TP),
            flattened_axes={MeshAxisName.DP_CP: (MeshAxisName.DP, MeshAxisName.CP)} if cp_size > 1 else {},
        ),
        timeout_minutes=timeout_minutes,
    )


def _create_moe_mesh(
    device_mesh: DeviceMesh,
    *,
    ep_shard_size: int,
    ep_size: int,
    timeout_minutes: int | None = None,
) -> DeviceMesh:
    non_pp_axes = (MeshAxisName.DP_REPLICATE, MeshAxisName.DP_SHARD, MeshAxisName.CP, MeshAxisName.TP)
    return _unflatten_compat(
        device_mesh[non_pp_axes]._flatten(),
        axis=0,
        sizes=(ep_shard_size, ep_size),
        names=(MeshAxisName.EP_SHARD, MeshAxisName.EP),
        timeout_minutes=timeout_minutes,
    )


def _unflatten_compat(
    flat_mesh: DeviceMesh,
    axis: int,
    sizes: tuple,
    names: tuple,
    *,
    timeout_minutes: int | None = None,
) -> DeviceMesh:
    """Unflatten a mesh with its NCCL timeout, including the PyTorch 2.9 fallback."""
    if hasattr(flat_mesh, "_unflatten"):
        if timeout_minutes is not None and flat_mesh.device_type == "cuda":
            return flat_mesh._unflatten(
                axis,
                sizes,
                names,
                backend_override=_nccl_backend_override(
                    names,
                    device_type=flat_mesh.device_type,
                    timeout_minutes=timeout_minutes,
                ),
            )
        return flat_mesh._unflatten(axis, sizes, names)
    new_mesh_tensor = flat_mesh.mesh.reshape(sizes)
    from torch.distributed.device_mesh import DeviceMesh as _DeviceMesh

    return _DeviceMesh(flat_mesh.device_type, new_mesh_tensor, mesh_dim_names=names)


def get_flat_mesh(device_mesh: "DeviceMesh", name: str) -> "DeviceMesh":
    """Access a 1D submesh by parallelism name (e.g. ``"dp"``, ``"tp"``, ``"dp_cp"``).

    PyTorch 2.11 deprecates ``root_mesh["name"]`` for dimensions created via
    ``_flatten()``.  This reads the ``_flatten()`` result directly.

    Args:
        device_mesh: Any DeviceMesh (root or submesh).
        name: Parallelism dimension name.
    """
    if name in device_mesh.mesh_dim_names:
        return device_mesh[name]
    # _get_root_mesh() was added in PyTorch 2.10; fall back for 2.9.x.
    if hasattr(device_mesh, "_get_root_mesh"):
        root = device_mesh._get_root_mesh()
    else:
        root = device_mesh
    if hasattr(root, "_flatten_mapping") and name in root._flatten_mapping:
        return root._flatten_mapping[name]
    raise KeyError(
        f"Mesh dim {name!r} not found in mesh_dim_names {device_mesh.mesh_dim_names} "
        f"or root _flatten_mapping {set(getattr(root, '_flatten_mapping', {}))}"
    )


def get_submesh(device_mesh: "DeviceMesh", names: tuple) -> "DeviceMesh":
    """Access a submesh by parallelism dim names.

    Handles all cases: single dims, multi-dim slices, and combinations that
    include ``_flatten()``-created dims (e.g. ``("dp_replicate", "dp_shard_cp")``).
    For the latter, finds the parent ``_flatten()`` result and calls ``_unflatten()``
    to decompose it into the requested shape.

    Args:
        device_mesh: Any DeviceMesh (root or submesh).
        names: Tuple of dimension names.
    """
    if len(names) == 1:
        return get_flat_mesh(device_mesh, names[0])
    if all(n in device_mesh.mesh_dim_names for n in names):
        return device_mesh[names]

    # Some dims were created via _flatten(); resolve sizes and unflatten from parent.
    # Strategy: find a parent flattened mesh whose size equals the product of
    # requested dims, unflatten it, then validate that process groups for any
    # dim that also exists on the root mesh are identical (guards against
    # ambiguous size collisions between different flattened meshes).
    from math import prod

    import torch.distributed as dist

    sizes = tuple(get_flat_mesh(device_mesh, n).size() for n in names)
    target = prod(sizes)
    if hasattr(device_mesh, "_get_root_mesh"):
        root = device_mesh._get_root_mesh()
    else:
        root = device_mesh

    for fm in getattr(root, "_flatten_mapping", {}).values():
        if fm.size() != target:
            continue
        try:
            result = _unflatten_compat(fm, 0, sizes, names)
        except (ValueError, RuntimeError):
            continue
        # Validate: for each requested dim, verify its process group matches
        # what get_flat_mesh returns (works for both mesh dims and flattened dims).
        valid = True
        for name in names:
            expected = set(dist.get_process_group_ranks(get_flat_mesh(device_mesh, name).get_group()))
            actual = set(dist.get_process_group_ranks(result[name].get_group()))
            if expected != actual:
                valid = False
                break
        if valid:
            return result
    raise KeyError(
        f"No parent flattened mesh found for dims {names} with target size {target}. "
        f"Available: {set(root._flatten_mapping)}"
    )


def get_fsdp_dp_mesh(
    device_mesh: "DeviceMesh",
    dp_replicate_name: str = MeshAxisName.DP_REPLICATE,
    dp_shard_cp_name: str = MeshAxisName.DP_SHARD_CP,
) -> "DeviceMesh":
    """Return the DP mesh for FSDP2 without losing the original root mesh.

    ``get_submesh()`` may rebuild a fresh DeviceMesh when asked to compose native
    and flattened dims like ``("dp_replicate", "dp_shard_cp")``. That is fine
    for many local operations, but FSDP2 expects its DP mesh to share the same
    root mesh as TP/EP meshes. On multi-node TP runs this can break group
    construction in non-obvious ways.

    Prefer native dimensions whenever possible:
    - cp=1, dp_replicate=1  -> ``device_mesh["dp_shard"]``
    - cp=1, dp_replicate>1  -> ``device_mesh[("dp_replicate", "dp_shard")]``
    - cp>1, dp_replicate=1  -> ``device_mesh["dp_shard_cp"]``

    When both CP and replicated DP are active we fall back to ``get_submesh()``
    because the composed mesh is genuinely multi-level.
    """

    dp_shard_name = MeshAxisName.DP_SHARD
    cp_name = MeshAxisName.CP

    native_dims_available = (
        dp_replicate_name in device_mesh.mesh_dim_names
        and dp_shard_name in device_mesh.mesh_dim_names
        and cp_name in device_mesh.mesh_dim_names
    )
    if native_dims_available:
        cp_size = device_mesh[cp_name].size()
        dp_replicate_size = device_mesh[dp_replicate_name].size()

        if dp_replicate_size > 1 and cp_size > 1:
            pass
        elif dp_replicate_size > 1:
            return device_mesh[(dp_replicate_name, dp_shard_name)]
        elif cp_size > 1:
            return get_flat_mesh(device_mesh, dp_shard_cp_name)
        else:
            return device_mesh[dp_shard_name]

    return get_submesh(device_mesh, (dp_replicate_name, dp_shard_cp_name))
