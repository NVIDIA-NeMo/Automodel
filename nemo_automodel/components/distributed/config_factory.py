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

"""Factory for creating distributed configurations from config dictionaries."""

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

from nemo_automodel.components.distributed.config import (
    DDPConfig,
    FSDP2Config,
    MegatronFSDPConfig,
)
from nemo_automodel.components.distributed.pipelining.config import PipelineConfig
from nemo_automodel.components.moe.config import MoEParallelizerConfig

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

STRATEGY_MAP = {"fsdp2": FSDP2Config, "megatron_fsdp": MegatronFSDPConfig, "ddp": DDPConfig}

_PARALLELISM_DEFAULTS = {
    "tp_size": 1,
    "pp_size": 1,
    "cp_size": 1,
    "ep_size": 1,
    "dp_size": None,
    "dp_replicate_size": None,
}


@dataclass
class DistributedSetup:
    strategy_config: Optional[Union[FSDP2Config, MegatronFSDPConfig, DDPConfig]]
    device_mesh: Optional["DeviceMesh"]
    moe_mesh: Optional["DeviceMesh"]
    pipeline_config: Optional[PipelineConfig]
    moe_config: Optional[MoEParallelizerConfig]
    pp_enabled: bool
    tp_size: int = 1
    pp_size: int = 1
    cp_size: int = 1
    ep_size: int = 1
    dp_size: Optional[int] = None
    dp_replicate_size: Optional[int] = None
    activation_checkpointing: bool = False


def _validate(strategy_name, parallelism, strategy_kwargs, pipeline_dict, moe_dict):
    tp_size = parallelism.get("tp_size", 1)
    pp_size = parallelism.get("pp_size", 1)
    cp_size = parallelism.get("cp_size", 1)
    ep_size = parallelism.get("ep_size", 1)
    dp_replicate_size = parallelism.get("dp_replicate_size")

    if strategy_name == "megatron_fsdp":
        if pp_size > 1:
            raise ValueError("megatron_fsdp does not support pipeline parallelism")
        if ep_size > 1:
            raise ValueError("megatron_fsdp does not support expert parallelism")
        if strategy_kwargs.get("sequence_parallel"):
            raise ValueError("megatron_fsdp does not yet support sequence_parallel")

    if strategy_name == "ddp":
        if tp_size > 1:
            raise ValueError("ddp does not support tensor parallelism")
        if pp_size > 1:
            raise ValueError("ddp does not support pipeline parallelism")
        if cp_size > 1:
            raise ValueError("ddp does not support context parallelism")
        if ep_size > 1:
            raise ValueError("ddp does not support expert parallelism")
        if dp_replicate_size is not None and dp_replicate_size > 1:
            raise ValueError("ddp does not support HSDP (dp_replicate_size)")

    if pipeline_dict is not None and pp_size <= 1:
        raise ValueError("pipeline config requires pp_size > 1")

    if moe_dict is not None and ep_size <= 1:
        raise ValueError("moe config requires ep_size > 1")

    strategy_cls = STRATEGY_MAP[strategy_name]
    valid_fields = {f.name for f in dataclasses.fields(strategy_cls)} | {"activation_checkpointing"}
    unknown = set(strategy_kwargs) - valid_fields
    if unknown:
        raise ValueError(f"'{sorted(unknown)[0]}' is not a valid option for strategy '{strategy_name}'")


def parse_distributed_section(cfg_dict: dict) -> DistributedSetup:
    """Parse a flat distributed config dict into a DistributedSetup.

    Meshes (device_mesh, moe_mesh) are left as None; they are created later
    by ``setup_distributed``.
    """
    cfg = cfg_dict.copy()

    strategy_name = cfg.pop("strategy", "fsdp2")
    if strategy_name not in STRATEGY_MAP:
        raise ValueError(f"Unknown strategy: {strategy_name}. Valid strategies: {list(STRATEGY_MAP.keys())}")
    strategy_cls = STRATEGY_MAP[strategy_name]

    parallelism = {k: cfg.pop(k, default) for k, default in _PARALLELISM_DEFAULTS.items()}

    pipeline_dict = cfg.pop("pipeline", None)
    moe_dict = cfg.pop("moe", None)
    activation_checkpointing = cfg.pop("activation_checkpointing", False)

    # Remaining keys are strategy kwargs
    strategy_kwargs = cfg

    _validate(strategy_name, parallelism, strategy_kwargs, pipeline_dict, moe_dict)

    # Route activation_checkpointing: for non-EP configs it goes on the strategy;
    # for EP configs it stays on DistributedSetup (passed to MoE infra separately).
    if parallelism["ep_size"] <= 1:
        strategy_kwargs["activation_checkpointing"] = activation_checkpointing

    strategy_config = strategy_cls(**strategy_kwargs)
    pipeline_config = PipelineConfig(**pipeline_dict) if pipeline_dict is not None and parallelism["pp_size"] > 1 else None
    moe_config = MoEParallelizerConfig(**moe_dict) if moe_dict is not None and parallelism["ep_size"] > 1 else None

    return DistributedSetup(
        strategy_config=strategy_config,
        device_mesh=None,
        moe_mesh=None,
        pipeline_config=pipeline_config,
        moe_config=moe_config,
        pp_enabled=parallelism["pp_size"] > 1,
        activation_checkpointing=activation_checkpointing,
        **parallelism,
    )


def setup_distributed(cfg, world_size: int) -> DistributedSetup:
    """Parse ``cfg.distributed`` and create device meshes.

    Returns a fully-initialized DistributedSetup (including device_mesh/moe_mesh).
    """
    from omegaconf import OmegaConf

    from nemo_automodel.components.distributed.device_mesh import create_device_mesh

    cfg_dict = OmegaConf.to_container(cfg.distributed, resolve=True)
    dist_setup = parse_distributed_section(cfg_dict)

    device_mesh, moe_mesh = create_device_mesh(
        dist_setup.strategy_config,
        dp_size=dist_setup.dp_size,
        dp_replicate_size=dist_setup.dp_replicate_size,
        tp_size=dist_setup.tp_size,
        pp_size=dist_setup.pp_size,
        cp_size=dist_setup.cp_size,
        ep_size=dist_setup.ep_size,
        world_size=world_size,
    )
    dist_setup.device_mesh = device_mesh
    dist_setup.moe_mesh = moe_mesh
    return dist_setup
