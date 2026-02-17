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

import atexit
import datetime
import logging
import os
import signal
from dataclasses import dataclass

import torch
import torch.distributed

logger = logging.getLogger(__name__)

DIST_ENV_RUNTIME_ENV_MAP = {
    "torch_nccl_use_comm_nonblocking": "TORCH_NCCL_USE_COMM_NONBLOCKING",
    "pytorch_alloc_conf": "PYTORCH_ALLOC_CONF",
    "nemotronh_ep_use_deepep_dispatch": "NEMOTRONH_EP_USE_DEEPEP_DISPATCH",
    "nemotronh_ep_require_deepep": "NEMOTRONH_EP_REQUIRE_DEEPEP",
    "nemotronh_ep_physical_partition": "NEMOTRONH_EP_PHYSICAL_PARTITION",
    "nemotronh_ep_sync_inactive_experts": "NEMOTRONH_EP_SYNC_INACTIVE_EXPERTS",
    "nemotronh_ep_expert_reshard_after_forward": "NEMOTRONH_EP_EXPERT_RESHARD_AFTER_FORWARD",
    "nemoautomodel_pp_skip_output_merge": "NEMOAUTOMODEL_PP_SKIP_OUTPUT_MERGE",
}


def _cfg_get(cfg, key, default=None):
    """Safely fetch a key from ConfigNode-like objects or dicts."""
    if cfg is None:
        return default
    getter = getattr(cfg, "get", None)
    if callable(getter):
        return getter(key, default)
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _to_env_str(value) -> str:
    """Convert YAML-friendly values to env-var string format."""
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def apply_runtime_env_from_dist_cfg(cfg_dist_env) -> dict[str, str]:
    """Apply runtime env overrides from `dist_env` config.

    Supports two forms:
    1. First-class dist_env keys in `DIST_ENV_RUNTIME_ENV_MAP`.
    2. Optional passthrough map: `dist_env.runtime_env`.

    Existing process env vars take precedence over config values.
    """
    overrides = {}

    for cfg_key, env_key in DIST_ENV_RUNTIME_ENV_MAP.items():
        value = _cfg_get(cfg_dist_env, cfg_key, None)
        if value is not None:
            overrides[env_key] = value

    runtime_env = _cfg_get(cfg_dist_env, "runtime_env", None)
    if runtime_env is not None:
        runtime_env_dict = runtime_env.to_dict() if hasattr(runtime_env, "to_dict") else runtime_env
        if not isinstance(runtime_env_dict, dict):
            raise TypeError(
                f"dist_env.runtime_env must be a dict-like mapping, but got {type(runtime_env_dict).__name__}"
            )
        overrides.update(runtime_env_dict)

    applied = {}
    for env_key, value in overrides.items():
        env_value = _to_env_str(value)
        existing = os.environ.get(env_key)
        if existing is None:
            os.environ[env_key] = env_value
            applied[env_key] = env_value
        elif existing != env_value:
            logger.info(
                "Keeping pre-set env %s=%s (ignoring config value %s)",
                env_key,
                existing,
                env_value,
            )
    return applied


def get_rank_safe() -> int:
    """
    Get the distributed rank safely, even if torch.distributed is not initialized.

    Returns:
        The current process rank.
    """
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    else:
        return int(os.getenv("RANK", "0"))


def get_world_size_safe() -> int:
    """
    Get the distributed world size safely, even if torch.distributed is not initialized.

    Returns:
        The total number of processes in the distributed job.
    """
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    else:
        return int(os.getenv("WORLD_SIZE", "1"))


def get_local_rank_preinit() -> int:
    """
    Get the local rank from the environment variable, intended for use before full init.

    Returns:
        The local rank of the current process.
    """
    return int(os.getenv("LOCAL_RANK", "0"))


def get_local_world_size_preinit() -> int:
    """
    Get the local world size from the environment variable, intended for use before full init.

    Returns:
        The local world size of the current process.
    """
    return int(os.getenv("LOCAL_WORLD_SIZE", "1"))


@dataclass
class DistInfo:
    """Holds information about the distributed training environment.

    Attributes:
        backend (str): The backend used for torch.distributed (e.g., 'nccl').
        rank (int): The rank of the current process.
        world_size (int): The total number of processes.
        device (torch.device): The device assigned to the current process.
        is_main (bool): True if the process is the main process (rank 0).
    """

    backend: str
    rank: int
    world_size: int
    device: torch.device
    is_main: bool


def initialize_distributed(
    backend,
    timeout_minutes=1,
):
    """Initialize the torch.distributed environment and core model parallel infrastructure.

    This function sets the device based on the local rank, configures the process group,
    and calls torch.distributed.init_process_group with the appropriate parameters.
    It also registers a cleanup function to properly destroy the process group at exit.

    Args:
        backend (str): The backend to use for torch.distributed (e.g., 'nccl').
        timeout_minutes (int, optional): Timeout (in minutes) for distributed initialization. Defaults to 1.

    Returns:
        DistInfo: An instance containing the distributed environment configuration.
    """
    device_count = torch.cuda.device_count()
    device = None
    if torch.distributed.is_initialized():
        if get_rank_safe() == 0:
            print(
                "torch distributed is already initialized, skipping initialization.",
                flush=True,
            )
        if device_count > 0:
            device = torch.cuda.current_device()
    else:
        if get_rank_safe() == 0:
            print("> initializing torch distributed with {} workers.".format(get_world_size_safe()), flush=True)

        # Call the init process
        init_pg_kwargs = {
            "backend": backend,
            "world_size": get_world_size_safe(),
            "rank": get_rank_safe(),
            "timeout": datetime.timedelta(minutes=timeout_minutes),
        }

        # Manually set the device ids.
        device = None
        if device_count > 0:
            rank = get_local_rank_preinit()
            device = torch.device("cuda", rank)
            torch.cuda.set_device(device)

        if get_world_size_safe() == 1:
            init_pg_kwargs["world_size"] = 1
            init_pg_kwargs["rank"] = 0
            init_pg_kwargs["backend"] = "gloo"
            init_pg_kwargs["store"] = torch.distributed.HashStore()

        torch.distributed.init_process_group(**init_pg_kwargs)
        atexit.register(destroy_global_state)
        # Skip barrier in single process case (e.g., torchrun --nproc-per-node=1)
        if get_world_size_safe() > 1:
            torch.distributed.barrier(device_ids=[get_local_rank_preinit()])

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    return DistInfo(backend, rank, world_size, device, rank == 0)


def destroy_global_state():
    """Destroy the torch.distributed process group during cleanup.

    This function is registered to execute at exit to ensure the process group is properly destroyed.
    It temporarily ignores SIGINT to avoid interruption during cleanup.
    """
    # Don't allow Ctrl+C to interrupt this handler
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    signal.signal(signal.SIGINT, signal.SIG_DFL)
