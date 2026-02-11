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

"""Typed MeshContext dataclass, validation, and strategy map.

``MeshContext`` is the single source of truth for everything related to
distributed training: strategy config, device meshes, and axis names.

Parallelism sizes (``tp_size``, ``pp_size``, etc.) are derived at runtime
from the attached ``DeviceMesh`` objects via ``@property``.  When no mesh
is present the properties return safe defaults (1 for sizes, ``None`` for
dp / hsdp).

All inputs and outputs are typed Python objects (dataclasses, enums, etc.).
YAML / dict parsing belongs in the recipe layer — see
``nemo_automodel.recipes._dist_setup``.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union

from nemo_automodel.components.distributed.config import (
    DDPConfig,
    FSDP2Config,
    MegatronFSDPConfig,
)

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

    from nemo_automodel.components.distributed.pipelining.config import PipelineConfig
    from nemo_automodel.components.moe.config import MoEParallelizerConfig


#: Maps strategy name (from YAML) → strategy dataclass.
STRATEGY_MAP: Dict[str, type] = {
    "fsdp2": FSDP2Config,
    "megatron_fsdp": MegatronFSDPConfig,
    "ddp": DDPConfig,
}


class MeshAxisName(str, Enum):
    """Canonical mesh-dimension names used by ``DeviceMesh`` and helpers.

    Inherits from ``str`` so each member compares equal to (and can be
    used wherever) a plain string — e.g. ``MeshAxisName.TP == "tp"``.
    """

    PP = "pp"
    DP = "dp"
    DP_REPLICATE = "dp_replicate"
    DP_SHARD = "dp_shard"
    DP_SHARD_CP = "dp_shard_cp"
    DP_CP = "dp_cp"
    CP = "cp"
    TP = "tp"
    EP = "ep"
    EP_SHARD = "ep_shard"


#: All values accepted as ``DeviceMesh`` dimension names.
_VALID_AXIS_NAMES: frozenset = frozenset(MeshAxisName)


@dataclass
class MeshContext:
    """Runtime distributed training context: configs + device meshes.

    Parallelism sizes (``tp_size``, ``pp_size``, etc.) are **not** stored as
    fields; they are ``@property`` accessors that read directly from the
    attached ``DeviceMesh`` / ``moe_mesh``.  When no mesh is present the
    properties return safe defaults (``1`` for sizes, ``None`` for dp / hsdp).

    All ``DeviceMesh`` objects passed in must use dimension names from
    :class:`MeshAxisName`; a ``ValueError`` is raised on construction if
    any unknown name is encountered.

    Lifecycle
    ---------
    1. Recipes parse YAML to obtain sizes and strategy configs.
    2. Sizes are passed to ``create_device_mesh`` to build ``DeviceMesh``
       objects.
    3. ``MeshContext`` is created with those meshes; dimension names are
       validated automatically in ``__post_init__``.

    Alternatively, :meth:`from_meshes` constructs an instance directly from
    ``DeviceMesh`` objects (used by ``NeMoAutoModel.from_pretrained``).

    Attributes:
        strategy_config: Strategy-specific config (FSDP2, MegatronFSDP, or DDP).
        device_mesh: Device mesh for distributed training.
        moe_mesh: MoE-specific device mesh.
        pipeline_config: Pipeline-parallel schedule/splitting config.
        moe_config: MoE parallelizer settings.
        activation_checkpointing: Whether activation checkpointing is enabled.
    """

    # config fields
    strategy_config: Optional[Union["FSDP2Config", "MegatronFSDPConfig", "DDPConfig"]] = None
    pipeline_config: Optional["PipelineConfig"] = None
    moe_config: Optional["MoEParallelizerConfig"] = None
    activation_checkpointing: bool = False

    # runtime mesh references
    device_mesh: Optional["DeviceMesh"] = field(default=None, repr=False)
    moe_mesh: Optional["DeviceMesh"] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        _validate_mesh_dim_names(self)
        _validate_distributed_setup(self)

    # Parallelism sizes — derived from the attached meshes
    @property
    def pp_size(self) -> int:
        """Pipeline-parallel degree (from ``device_mesh``, default ``1``)."""
        return _get_axis_size(self.device_mesh, MeshAxisName.PP)

    @property
    def pp_enabled(self) -> bool:
        """``True`` when ``pp_size > 1``."""
        return self.pp_size > 1

    @property
    def tp_size(self) -> int:
        """Tensor-parallel degree (from ``device_mesh``, default ``1``)."""
        return _get_axis_size(self.device_mesh, MeshAxisName.TP)

    @property
    def cp_size(self) -> int:
        """Context-parallel degree (from ``device_mesh``, default ``1``)."""
        return _get_axis_size(self.device_mesh, MeshAxisName.CP)

    @property
    def ep_size(self) -> int:
        """Expert-parallel degree (from ``moe_mesh``, default ``1``)."""
        return _get_axis_size(self.moe_mesh, MeshAxisName.EP)

    @property
    def dp_size(self) -> Optional[int]:
        """Data-parallel degree (from ``device_mesh``, default ``None``)."""
        return _get_axis_size(self.device_mesh, MeshAxisName.DP)

    @property
    def dp_replicate_size(self) -> Optional[int]:
        """HSDP replication degree (from ``device_mesh``, default ``None``)."""
        return _get_axis_size(self.device_mesh, MeshAxisName.DP_REPLICATE)

    # Axis-name helpers (used by AutoPipeline and parallelize_model)
    def _dp_axis_names(self) -> Tuple[str, ...]:
        """DP axis names for FSDP mesh slicing."""
        if self.device_mesh is not None:
            names = self.device_mesh.mesh_dim_names
            if MeshAxisName.DP_REPLICATE in names and MeshAxisName.DP_SHARD_CP in names:
                return (MeshAxisName.DP_REPLICATE, MeshAxisName.DP_SHARD_CP)
        return (MeshAxisName.DP_SHARD_CP,)

    # @akoumpa: we will deprecate `pipeline_axis_kwargs` in 26.04.
    def pipeline_axis_kwargs(self) -> Dict[str, object]:
        """Axis-name kwargs for ``AutoPipeline``."""
        return {
            "pp_axis_name": MeshAxisName.PP,
        } | self.parallelize_axis_kwargs()

    # @akoumpa: we will deprecate `parallelize_axis_kwargs` in 26.04.
    def parallelize_axis_kwargs(self) -> Dict[str, object]:
        """Axis-name kwargs for ``parallelize_fn`` (EP/FSDP, no ``pp_axis_name``)."""
        return {
            "dp_axis_names": self._dp_axis_names(),
            "cp_axis_name": _optional_axis(self.device_mesh, MeshAxisName.CP),
            "tp_axis_name": _optional_axis(self.device_mesh, MeshAxisName.TP),
            "ep_axis_name": _optional_axis(self.moe_mesh, MeshAxisName.EP),
            "ep_shard_axis_names": (MeshAxisName.EP_SHARD,)
            if _optional_axis(self.moe_mesh, MeshAxisName.EP_SHARD)
            else None,
        }

    # Convenience constructor
    @classmethod
    def from_meshes(
        cls,
        device_mesh: Optional["DeviceMesh"],
        moe_mesh: Optional["DeviceMesh"] = None,
        *,
        strategy_config: Optional[Union["FSDP2Config", "MegatronFSDPConfig", "DDPConfig"]] = None,
        pipeline_config: Optional["PipelineConfig"] = None,
        moe_config: Optional["MoEParallelizerConfig"] = None,
        activation_checkpointing: bool = False,
    ) -> "MeshContext":
        """Build a :class:`MeshContext` from ``DeviceMesh`` objects.

        This is the entry-point used by ``NeMoAutoModel.from_pretrained`` /
        ``from_config`` where the caller has raw meshes rather than a parsed
        YAML config.
        """
        return cls(
            strategy_config=strategy_config,
            pipeline_config=pipeline_config,
            moe_config=moe_config,
            activation_checkpointing=activation_checkpointing,
            device_mesh=device_mesh,
            moe_mesh=moe_mesh,
        )


# misc utils
def _get_axis_size(mesh: Optional["DeviceMesh"], axis: MeshAxisName) -> Optional[str]:
    """Return the size of *axis* if present in *mesh*, else ``None``."""
    if mesh is not None and axis in mesh.mesh_dim_names:
        return mesh[axis].size()
    return 1


def _optional_axis(mesh: Optional["DeviceMesh"], axis: MeshAxisName) -> Optional[str]:
    """Return *axis* if present in *mesh*, else ``None``."""
    if mesh is not None and axis in mesh.mesh_dim_names:
        return axis
    return None


# Validation utils
def _validate_mesh_dim_names(mesh_context: "MeshContext") -> None:
    """Ensure every dimension name in the attached meshes is a :class:`MeshAxisName`."""
    for label in ("device_mesh", "moe_mesh"):
        mesh = getattr(mesh_context, label)
        if mesh is None:
            continue
        bad = {n for n in mesh.mesh_dim_names if n not in _VALID_AXIS_NAMES}
        if bad:
            raise ValueError(
                f"{label} contains unknown dimension names {bad}; allowed names are {sorted(_VALID_AXIS_NAMES)}"
            )


def _validate_distributed_setup(mesh_context: "MeshContext") -> None:
    """Validate cross-field constraints on a :class:`MeshContext`.

    Called automatically by ``MeshContext.__post_init__`` when a
    ``strategy_config`` is present.  Can also be invoked explicitly
    after mutating a context.

    Raises:
        ValueError: If any constraint is violated.
    """
    if mesh_context.strategy_config is None:
        return

    if isinstance(mesh_context.strategy_config, MegatronFSDPConfig):
        if mesh_context.pp_size > 1:
            raise ValueError("megatron_fsdp does not support pipeline parallelism")
        if mesh_context.ep_size > 1:
            raise ValueError("megatron_fsdp does not support expert parallelism")
        if mesh_context.strategy_config.sequence_parallel:
            raise ValueError("megatron_fsdp does not yet support sequence_parallel")

    if isinstance(mesh_context.strategy_config, DDPConfig):
        if mesh_context.tp_size > 1:
            raise ValueError("ddp does not support tensor parallelism")
        if mesh_context.pp_size > 1:
            raise ValueError("ddp does not support pipeline parallelism")
        if mesh_context.cp_size > 1:
            raise ValueError("ddp does not support context parallelism")
        if mesh_context.ep_size > 1:
            raise ValueError("ddp does not support expert parallelism")
        if mesh_context.dp_replicate_size is not None and mesh_context.dp_replicate_size > 1:
            raise ValueError("ddp does not support HSDP (dp_replicate_size)")

    if mesh_context.pipeline_config is not None and mesh_context.pp_size <= 1:
        raise ValueError("pipeline config requires pp_size > 1")

    if mesh_context.moe_config is not None and mesh_context.ep_size <= 1:
        raise ValueError("moe config requires ep_size > 1")
