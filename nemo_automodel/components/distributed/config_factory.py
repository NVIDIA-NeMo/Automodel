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

"""Typed distributed-setup dataclass, builder, and cross-field validation.

All inputs and outputs are typed Python objects (dataclasses, enums, etc.).
YAML / dict parsing belongs in the recipe layer — see
``nemo_automodel.recipes._dist_setup``.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union


if TYPE_CHECKING:
    from nemo_automodel.components.distributed.config import (
        DDPConfig,
        FSDP2Config,
        MegatronFSDPConfig,
    )

    from nemo_automodel.components.distributed.pipelining.config import PipelineConfig
    from nemo_automodel.components.moe.config import MoEParallelizerConfig
    from torch.distributed.device_mesh import DeviceMesh


@dataclass
class DistributedSetup:
    """Fully-described distributed training configuration.

    Attributes:
        strategy_config: Strategy-specific configuration (FSDP2, MegatronFSDP, or DDP).
        device_mesh: Device mesh for distributed training.  ``None`` until meshes
            are created by the recipe layer.
        moe_mesh: MoE-specific device mesh.  ``None`` when EP is not used.
        pipeline_config: Pipeline-parallel schedule/splitting config.  ``None``
            when PP is disabled.
        moe_config: MoE parallelizer settings.  ``None`` when EP is not used.
        pp_enabled: Convenience flag — ``True`` when ``pp_size > 1``.
        tp_size: Tensor-parallel degree.
        pp_size: Pipeline-parallel degree.
        cp_size: Context-parallel degree.
        ep_size: Expert-parallel degree.
        dp_size: Data-parallel degree (``None`` = infer from world size).
        dp_replicate_size: HSDP replication degree.
        activation_checkpointing: Whether activation checkpointing is enabled.
    """

    strategy_config: Union["FSDP2Config", "MegatronFSDPConfig", "DDPConfig"]
    device_mesh: Optional["DeviceMesh"]
    moe_mesh: Optional["DeviceMesh"]
    pipeline_config: Optional["PipelineConfig"]
    moe_config: Optional["MoEParallelizerConfig"]
    pp_enabled: bool
    tp_size: int = 1
    pp_size: int = 1
    cp_size: int = 1
    ep_size: int = 1
    dp_size: Optional[int] = None
    dp_replicate_size: Optional[int] = None
    activation_checkpointing: bool = False


def validate_distributed_setup(setup: DistributedSetup) -> None:
    """Validate cross-field constraints on a :class:`DistributedSetup`.

    Raises:
        ValueError: If any constraint is violated.
    """
    cfg = setup.strategy_config

    if isinstance(cfg, MegatronFSDPConfig):
        if setup.pp_size > 1:
            raise ValueError("megatron_fsdp does not support pipeline parallelism")
        if setup.ep_size > 1:
            raise ValueError("megatron_fsdp does not support expert parallelism")
        if cfg.sequence_parallel:
            raise ValueError("megatron_fsdp does not yet support sequence_parallel")

    if isinstance(cfg, DDPConfig):
        if setup.tp_size > 1:
            raise ValueError("ddp does not support tensor parallelism")
        if setup.pp_size > 1:
            raise ValueError("ddp does not support pipeline parallelism")
        if setup.cp_size > 1:
            raise ValueError("ddp does not support context parallelism")
        if setup.ep_size > 1:
            raise ValueError("ddp does not support expert parallelism")
        if setup.dp_replicate_size is not None and setup.dp_replicate_size > 1:
            raise ValueError("ddp does not support HSDP (dp_replicate_size)")

    if setup.pipeline_config is not None and setup.pp_size <= 1:
        raise ValueError("pipeline config requires pp_size > 1")

    if setup.moe_config is not None and setup.ep_size <= 1:
        raise ValueError("moe config requires ep_size > 1")


def build_distributed_setup(
    strategy_config: Union["FSDP2Config", "MegatronFSDPConfig", "DDPConfig" ],
    *,
    tp_size: int = 1,
    pp_size: int = 1,
    cp_size: int = 1,
    ep_size: int = 1,
    dp_size: Optional[int] = None,
    dp_replicate_size: Optional[int] = None,
    activation_checkpointing: bool = False,
    pipeline_config: Optional["PipelineConfig"] = None,
    moe_config: Optional["MoEParallelizerConfig"] = None,
) -> DistributedSetup:
    """Build and validate a :class:`DistributedSetup` from typed components.

    Device meshes (``device_mesh``, ``moe_mesh``) are left as ``None``; they
    are attached later by the recipe layer.

    Raises:
        ValueError: If any cross-field constraint is violated.
    """
    setup = DistributedSetup(
        strategy_config=strategy_config,
        device_mesh=None,
        moe_mesh=None,
        pipeline_config=pipeline_config,
        moe_config=moe_config,
        pp_enabled=pp_size > 1,
        tp_size=tp_size,
        pp_size=pp_size,
        cp_size=cp_size,
        ep_size=ep_size,
        dp_size=dp_size,
        dp_replicate_size=dp_replicate_size,
        activation_checkpointing=activation_checkpointing,
    )
    validate_distributed_setup(setup)
    return setup
