import torch.distributed as dist
from dataloader import MetaFilesDataset, collate_fn
from torch.utils.data import DataLoader, DistributedSampler


def create_dataloader(meta_folder: str, batch_size: int, num_nodes: int):
    """
    Create a distributed dataloader for multi-node training.
    
    Args:
        meta_folder: Path to folder containing .meta files
        batch_size: Batch size per node
        num_nodes: Number of nodes (for data parallel sampling)
    
    Returns:
        Tuple of (DataLoader, DistributedSampler)
    """
    dataset = MetaFilesDataset(meta_folder=meta_folder, device="cpu")
    
    # Use num_nodes as num_replicas for data parallel across nodes
    # Each node will get different batches
    sampler = DistributedSampler(
        dataset,
        num_replicas=num_nodes if dist.is_initialized() else 1,
        rank=dist.get_rank() // dist.get_world_size() * num_nodes if dist.is_initialized() else 0,
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