# ddp_utils.py
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, DataLoader
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

@dataclass
class CheckpointIO:
    """Simple disk-based checkpoint I/O."""

    def save(self, state: Dict[str, Any], path: Union[str, Path]):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, path)

    def load(self, path: Union[str, Path]) -> Dict[str, Any]:
        return torch.load(path, map_location="cpu")


@dataclass
class DDPManager:
    checkpoint_io: CheckpointIO = field(
        default_factory=CheckpointIO,
        metadata={"help": "Helper for checkpoint save/load (disk, S3, etc.)."}
    )
    backend: str = field(
        default="nccl",
        default_factory=str,
        metadata={"help": "Distributed backend, e.g. 'nccl' or 'gloo'."}
    )

    # These are populated in setup_distributed(), not by user:
    rank: int = field(
        init=False,
        default_factory=int,
        metadata={"help": "Global rank of this process."}
    )
    world_size: int = field(
        init=False,
        default_factory=int,
        metadata={"help": "Total number of distributed processes."}
    )

    def setup_distributed(self):
        """
        Initialize torch.distributed process group and wrap raw_model in DDP.
        Requires env vars: RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT.
        """
        if not dist.is_initialized():
            rank = int(os.environ["RANK"])
            world = int(os.environ["WORLD_SIZE"])
            os.environ.setdefault("MASTER_ADDR", os.environ.get("MASTER_ADDR", "localhost"))
            os.environ.setdefault("MASTER_PORT", os.environ.get("MASTER_PORT", "29500"))
            dist.init_process_group(self.backend, rank=rank, world_size=world)

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        # Pin GPU if using NCCL
        if self.backend == "nccl":
            local_gpu = self.rank % torch.cuda.device_count()
            torch.cuda.set_device(local_gpu)
            device = torch.device("cuda", index=local_gpu)
        else:
            device = torch.device("cpu")

    def wrap_model(self, model):
        model.to(device)
        # wrap in DDP
        return DDP(model, device_ids=[device] if device.type == "cuda" else None)

    # @contextmanager
    # def no_sync(self):
    #     """
    #     Context manager to skip gradient all‐reduce (for gradient accumulation):
    #         with manager.no_sync():
    #             ...
    #     """
    #     if isinstance(self.model, DDP):
    #         with self.model.no_sync():
    #             yield
    #     else:
    #         yield

    def save_checkpoint(self, state: Dict[str, Any], path: Union[str, Path]):
        """
        Barrier + save on rank 0.
        state: dict of tensors (e.g. {"model": model_state, "opt": opt_state})
        """
        dist.barrier()
        if self.rank == 0:
            self.checkpoint_io.save(state, path)

    def load_checkpoint(self, path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load on rank 0 and broadcast any tensor entries to all ranks.
        Returns the state dict.
        """
        state = self.checkpoint_io.load(path)
        for k, v in state.items():
            if torch.is_tensor(v):
                dist.broadcast(v, src=0)
        return state