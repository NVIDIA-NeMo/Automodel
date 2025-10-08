from torch.utils.data import DataLoader, DistributedSampler
from dataloader import MetaFilesDataset, collate_fn
import torch.distributed as dist
from dist_utils import print0

def create_dataloader(meta_folder: str, batch_size: int, world_size: int):
    dataset = MetaFilesDataset(meta_folder=meta_folder, device="cpu")
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size if dist.is_initialized() else 1,
        rank=dist.get_rank() if dist.is_initialized() else 0,
        shuffle=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )
    return loader, sampler