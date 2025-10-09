import logging
import os
import warnings

import torch
import torch.distributed as dist


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def print0(*a, **k):
    if is_main_process():
        print(*a, **k)


def configure_logging():
    level = logging.INFO if is_main_process() else logging.ERROR
    logging.basicConfig(level=level)
    if dist.is_initialized() and dist.get_rank() != 0:
        warnings.filterwarnings("ignore")


def setup_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Set device BEFORE initializing process group
    torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        # Initialize with explicit device specification
        dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
        print0(f"[DIST] Initialized process group with device cuda:{local_rank}")

    configure_logging()

    # BF16 only; WAN models expect this
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # seed
    seed = 42 + (dist.get_rank() if dist.is_initialized() else 0)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    print0(
        f"[DIST] Setup complete - Rank: {dist.get_rank() if dist.is_initialized() else 0}, Device: cuda:{local_rank}"
    )
    return local_rank


def cast_model_to_dtype(module, dtype):
    for p in module.parameters(recurse=True):
        if p.dtype.is_floating_point:
            p.data = p.data.to(dtype)
    for b in module.buffers(recurse=True):
        if b.dtype.is_floating_point:
            b.data = b.data.to(dtype)
