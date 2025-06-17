#!/usr/bin/python3
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
from contextlib import ContextDecorator, nullcontext
from datetime import datetime
from typing import Any, Optional

import torch
import torch.distributed
import torch.distributed as dist
import yaml

from nemo_automodel.utils.yaml_utils import safe_yaml_representers

logger = logging.getLogger(__name__)


class FirstRankPerNode(ContextDecorator):
    """
    Context manager to enforce rank0 to process section over other ranks.

      - Lets LOCAL_RANK==0 run the protected code first on each node.
      - Inserts an extra barrier across *only* the node‑local rank‑0 processes.
      - Works on a single GPU (no env flags, no distributed initialisation).

    Note: it is assumed the scoped code is not torch.distributed heavy.
    """

    def __enter__(self):
        self._created_pg = False
        self._node0_group = None
        self._first = True  # default for single‑GPU / no‑dist case

        # ------------------------------------------------------------------ #
        # 1. Make sure there is at least *some* process‑group initialised
        # ------------------------------------------------------------------ #
        if not dist.is_initialized():
            self._created_pg = self._try_bootstrap_pg()

        if not dist.is_initialized():
            # pure single GPU
            return True

        # ------------------------------------------------------------------ #
        # 2. Figure out local/global ranks
        # ------------------------------------------------------------------ #
        env = os.environ
        global_rank = dist.get_rank()
        local_rank = int(env.get("LOCAL_RANK", global_rank))  # fallback
        self._first = local_rank == 0

        # ------------------------------------------------------------------ #
        # 3. Synchronisation logic
        # ------------------------------------------------------------------ #
        if not self._first:
            # Non‑rank‑0 processes wait for their node‑rank-0
            dist.barrier()

        return self._first

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if self._first and dist.is_initialized():
                # Re‑sync the whole world so that non‑rank‑0s can proceed
                dist.barrier()
                if exc_type is not None:
                    dist.abort()  # propagate failure to the entire job
        finally:
            if self._created_pg:
                dist.destroy_process_group()

        # propagate any exception to outer scope
        return False

    def _try_bootstrap_pg(self) -> bool:
        """Try to create a default pg from env:// variables."""
        env = os.environ
        required = ("WORLD_SIZE", "RANK", "MASTER_ADDR", "MASTER_PORT")
        if all(k in env for k in required):
            dist.init_process_group(
                backend="gloo",
                world_size=int(env.get("WORLD_SIZE")),
                rank=int(env.get("RANK")),
            )
            return True
        return False


def get_rank_safe() -> int:
    """
    Get the distributed rank safely, even if torch.distributed is not initialized.

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
    """
    Get the distributed world size safely, even if torch.distributed is not initialized.

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
    """
    Get the local rank from the environment variable, intended for use before full init.

    Returns:
        The local rank of the current process.
    """
    return int(os.getenv("LOCAL_RANK", "0"))



def is_last_rank() -> bool:
    """
    Check if the current rank is the last rank in the default process group.

    Returns:
        True if the current rank is the last one, False otherwise.
    """
    return torch.distributed.get_rank() == (torch.distributed.get_world_size() - 1)



def append_to_progress_log(save_dir: str, string: str, barrier: bool = True) -> None:
    """
    Append a formatted string to the progress log file (rank 0 only).

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
    """
    Perform a distributed barrier and then log a message on rank 0.

    Args:
        string: The message string to log.
    """
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info("[{}] datetime: {} ".format(string, time_str))


def log_single_rank(logger: logging.Logger, *args: Any, rank: int = 0, **kwargs: Any):
    """
    If torch distributed is initialized, log only on rank.

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
    """
    Dump a dataclass object or other Python object to a YAML file or string.

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


def reduce_loss(
    loss_store: list[torch.Tensor],
    total_num_tokens: torch.Tensor,
    per_token_loss: bool = True,
    dp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce loss across all ranks.

    Args:
        loss_store: List of loss tensors to reduce.
        total_num_tokens: Total number of tokens to divide the loss by.
        per_token_loss: Whether to divide the loss by the number of tokens.
        dp_group: Process group to reduce the loss across.

    Returns:
        Tuple of reduced loss and denominator.
    """
    loss = torch.sum(torch.stack(loss_store).float()).view(1).clone().detach()
    if dp_group is not None:
        dist.all_reduce(loss, op=torch.distributed.ReduceOp.SUM, group=dp_group)

    if per_token_loss:
        denominator = total_num_tokens.clone().detach().to(torch.int)
    else:
        denominator = torch.tensor([len(loss_store)], dtype=torch.int, device="cuda")
    if dp_group is not None:
        dist.all_reduce(denominator, op=torch.distributed.ReduceOp.SUM, group=dp_group)
    return loss, denominator


def get_sync_ctx(model, is_optim_step):
    """Get the synchronization context for the model.

    Args:
        model: The model to synchronize.
        is_optim_step: Whether the current step is an optimizer step.

    Returns:
        A context manager that synchronizes the model.
    """
    # Use `no_sync` on DDP models when we are *not* on the final micro-batch for
    # this gradient update (i.e., when `is_grad` is False). This avoids an
    # all-reduce for every micro-batch and greatly improves throughput.
    if isinstance(model, dist.fsdp._fully_shard._fully_shard.FSDPModule):
        model.set_requires_gradient_sync(is_optim_step)
        sync_ctx = nullcontext()
    elif isinstance(model, torch.nn.parallel.DistributedDataParallel):
        if is_optim_step:
            sync_ctx = nullcontext()
        else:
            sync_ctx = model.no_sync()
    else:
        sync_ctx = nullcontext()
    return sync_ctx

@torch.no_grad()
def rescale_gradients(model, num_tokens_for_grad_scaling, dp_group=None):
    """Rescale gradients across the DP group.

    Args:
        model: The model to rescale.
        num_tokens_for_grad_scaling: The number of tokens to divide the gradients by.
        dp_group: The process group to rescale the gradients across.
    """
    num_tokens_for_grad_scaling = num_tokens_for_grad_scaling.clone().detach()
    dp_group_size = 1
    if dp_group is not None:
       dist.all_reduce(num_tokens_for_grad_scaling, group=dp_group)
       dp_group_size = dist.get_world_size(group=dp_group)
    # DDP/FSDP reduces gradients across ranks, so we need to scale by the world size to inverse it
    scaling_factor = dp_group_size / num_tokens_for_grad_scaling
    for param in model.parameters():
        if param.grad is not None:
            param.grad.data.mul_(scaling_factor)

# based on: https://github.com/pytorch/torchtitan/blob/main/torchtitan/distributed/utils.py#L278
@torch.no_grad()
def clip_gradients(model, clip_norm, foreach=True):
    """Clip gradients across the DP group.

    Args:
        model: The model to clip the gradients of.
        clip_norm: The maximum norm of the gradients.
    """
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    grad_norm = torch.nn.utils.get_total_norm(grads, foreach=foreach)
    if isinstance(grad_norm, torch.distributed.tensor.DTensor):
        grad_norm = grad_norm.full_tensor()
    torch.nn.utils.clip_grads_with_norm_([p for p in model.parameters()], clip_norm, grad_norm, foreach=foreach)
    return grad_norm
