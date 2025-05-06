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

import logging
import os
from datetime import datetime
from typing import Any, Optional

import torch
import torch.distributed
import yaml

from nemo_lm.utils.yaml_utils import safe_yaml_representers


def get_rank_safe() -> int:
    """Get the distributed rank safely, even if torch.distributed is not initialized.

    Returns:
        The current process rank.
    """
    # In megatron init, args.rank comes from the torchrun env var.
    # Once init has been done, args.rank is updated to value of torch get_rank()
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    else:
        return int(os.getenv("RANK", "0"))


def get_world_size_safe() -> int:
    """Get the distributed world size safely, even if torch.distributed is not initialized.

    Returns:
        The total number of processes in the distributed job.
    """
    # In megatron init, args.world_size comes from the torchrun env var.
    # Once init has been done, args.world_size is updated to value of torch get_world_size()
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    else:
        return int(os.getenv("WORLD_SIZE", "1"))


def get_local_rank_preinit() -> int:
    """Get the local rank from the environment variable, intended for use before full init.

    Returns:
        The local rank of the current process.
    """
    return int(os.getenv("LOCAL_RANK", "0"))


def print_rank_0(message: str) -> None:
    """Print a message only on global rank 0.

    Args:
        message: The message string to print.
    """
    rank = get_rank_safe()
    if rank == 0:
        print(message, flush=True)


def is_last_rank() -> bool:
    """Check if the current rank is the last rank in the default process group.

    Returns:
        True if the current rank is the last one, False otherwise.
    """
    return torch.distributed.get_rank() == (torch.distributed.get_world_size() - 1)


def print_rank_last(message: str) -> None:
    """Print a message only on the last rank of the default process group.

    Args:
        message: The message string to print.
    """
    if torch.distributed.is_initialized():
        if is_last_rank():
            print(message, flush=True)
    else:
        print(message, flush=True)


def append_to_progress_log(save_dir: str, string: str, barrier: bool = True) -> None:
    """Append a formatted string to the progress log file (rank 0 only).

    Includes timestamp, job ID, and number of GPUs in the log entry.

    Args:
        save_dir: The directory where the 'progress.txt' file is located.
        string: The message string to append.
        barrier: If True, performs a distributed barrier before writing (rank 0 only).
    """
    if save_dir is None:
        return
    progress_log_filename = os.path.join(save_dir, "progress.txt")
    if barrier and torch.distributed.is_initialized():
        torch.distributed.barrier()
    if get_rank_safe() == 0:
        os.makedirs(os.path.dirname(progress_log_filename), exist_ok=True)
        with open(progress_log_filename, "a+") as f:
            job_id = os.getenv("SLURM_JOB_ID", "")
            num_gpus = get_world_size_safe()
            f.write(
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\tJob ID: {job_id}\t# GPUs: {num_gpus}\t{string}\n"
            )


def barrier_and_log(string: str) -> None:
    """Perform a distributed barrier and then log a message on rank 0.

    Args:
        string: The message string to log.
    """
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print_rank_0(f"[{string}] datetime: {time_str} ")


def log_single_rank(logger: logging.Logger, *args: Any, rank: int = 0, **kwargs: Any):
    """If torch distributed is initialized, log only on rank

    Args:
        logger (logging.Logger): The logger to write the logs

        args (Tuple[Any]): All logging.Logger.log positional arguments

        rank (int, optional): The rank to write on. Defaults to 0.

        kwargs (Dict[str, Any]): All logging.Logger.log keyword arguments
    """
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == rank:
            logger.log(*args, **kwargs)
    else:
        logger.log(*args, **kwargs)


def dump_dataclass_to_yaml(obj: Any, filename: Optional[str] = None) -> Optional[str]:
    """Dump a dataclass object or other Python object to a YAML file or string.

    Uses safe representers to handle common types.

    Args:
        obj: The object to dump.
        filename: If provided, the path to the file where YAML should be written.
                  If None, returns the YAML string directly.

    Returns:
        If filename is None, returns the YAML string representation of the object.
        Otherwise, returns None.
    """
    with safe_yaml_representers():
        if filename is not None:
            with open(filename, "w+") as f:
                yaml.safe_dump(obj, f)
        else:
            return yaml.safe_dump(obj)
