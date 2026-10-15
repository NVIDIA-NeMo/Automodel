# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FP32 Muse router storage must survive initialization after FSDP sharding."""

import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh

from nemo_automodel.components.moe.parallelizer import apply_fsdp
from tests.unit_tests.models.muse_spark.conftest import tiny_muse_spark_model


def _worker(rank: int, port: int) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=180),
    )
    try:
        model = tiny_muse_spark_model().to(device=rank)
        mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("dp",))
        apply_fsdp(model, mesh, ep_enabled=False, ep_shard_enabled=False, reshard_after_forward=True)
        model.initialize_weights(torch.device("cuda", rank), dtype=torch.bfloat16)
        for layer in model.model.layers:
            gate = layer.feed_forward.gate
            assert gate.weight.dtype == torch.float32
            assert gate.weight.to_local().dtype == torch.float32
            assert gate.e_score_correction_bias.dtype == torch.float32
            # DCP copies checkpoint values into these existing local destinations.
            # A BF16 destination would silently round this value to exactly 1.
            with torch.no_grad():
                gate.weight.to_local().fill_(1.0001)
            expected = torch.full(gate.weight.shape, 1.0001, device=rank)
            torch.testing.assert_close(gate.weight.full_tensor(), expected, rtol=0, atol=0)
            assert layer.self_attn.q_proj.weight.dtype == torch.bfloat16
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA GPUs")
def test_router_fp32_survives_sharded_initialization() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(port,), nprocs=2, join=True)
