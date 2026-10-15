# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise Muse padding preparation through the real two-rank CP sharder."""

import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh

from nemo_automodel.components.distributed.context_parallel.sharder import ContextParallelSharder
from tests.unit_tests.models.muse_spark.conftest import tiny_muse_spark_model


def _worker(rank: int, port: int) -> None:
    """Check local padding, CP-added padding, and independent PP microbatch rows."""
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=120),
    )
    try:
        mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("cp",))
        model = tiny_muse_spark_model()
        model.cp_mesh = mesh["cp"]
        tokens = torch.arange(10, device=rank).expand(2, -1).clone()
        valid = torch.arange(10, device=rank).unsqueeze(0) < torch.tensor([[3], [7]], device=rank)
        batch = {"input_ids": tokens, "labels": tokens.clone(), "attention_mask": valid}
        sharder = ContextParallelSharder(model, mesh, batch, num_chunks=2)
        train_context, batch = sharder.shard(batch)
        expected_indices = torch.tensor([0, 1, 2, 9, 10, 11] if rank == 0 else [3, 4, 5, 6, 7, 8], device=rank)
        expected_padding = torch.nn.functional.pad(~valid, (0, 2), value=True)[:, expected_indices]
        with train_context():
            assert "attention_mask" not in batch
            assert batch["input_ids"].shape == (2, 10)
            assert batch["labels"].shape == batch["padding_mask"].shape == (2, 6)
            torch.testing.assert_close(batch["padding_mask"], expected_padding)
            # A PP schedule chunks only the batch dimension. One CP rank's first
            # microbatch is entirely padding; it must retain that local mask.
            for row, local_mask in enumerate(batch["padding_mask"].chunk(2, dim=0)):
                torch.testing.assert_close(local_mask, expected_padding[row : row + 1])
            valid_count = (~batch["padding_mask"]).sum()
            dist.all_reduce(valid_count)
            assert int(valid_count) == int(valid.sum())
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA GPUs")
def test_cp_padding_matches_local_tokens_and_pipeline_chunks() -> None:
    """Padding survives attention-mask removal and follows CP then PP partitioning."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(port,), nprocs=2, join=True)
