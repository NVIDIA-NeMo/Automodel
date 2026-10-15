# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Muse PP residual precision over real NCCL sends, receives, and backward."""

import copy
import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from tests.unit_tests.models.muse_spark.conftest import tiny_muse_spark_model


def _worker(rank: int, port: int) -> None:
    """Compare a two-stage four-layer model with its unpartitioned reference."""
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=180),
    )
    try:
        torch.manual_seed(42)
        reference = tiny_muse_spark_model().to(device=rank, dtype=torch.bfloat16).train()
        reference.config.dtype = torch.bfloat16
        stage = copy.deepcopy(reference)
        stage.model.layers = nn.ModuleDict(
            {str(i): layer for i, layer in enumerate(stage.model.layers) if i // 2 == rank}
        )
        if rank == 0:
            stage.model.norm = None
            stage.lm_head = None
        else:
            stage.model.embed_tokens = None
            stage.model.embed_norm = None
        inputs_meta, outputs_meta = stage.get_pipeline_stage_metas(
            is_first=rank == 0, microbatch_size=1, seq_len=8, dtype=torch.bfloat16
        )
        tokens = torch.arange(1, 9, device=rank).unsqueeze(0)
        expected = reference(input_ids=tokens).logits
        upstream = torch.linspace(-1, 1, expected.numel(), device=rank).reshape_as(expected)
        (expected.float() * upstream).sum().backward()

        if rank == 0:
            boundary = stage(input_ids=tokens).logits
            assert boundary.dtype == outputs_meta[0].dtype == torch.float32
            assert boundary.shape == outputs_meta[0].shape
            assert not torch.equal(boundary, boundary.bfloat16().float())
            dist.send(boundary.detach().contiguous(), 1)
            gradient = torch.empty_like(boundary)
            dist.recv(gradient, 1)
            boundary.backward(gradient)
        else:
            boundary = torch.empty_like(inputs_meta[0], device=rank)
            assert boundary.dtype == torch.float32
            dist.recv(boundary, 0)
            boundary.requires_grad_()
            actual = stage(input_ids=boundary).logits
            assert actual.dtype == outputs_meta[0].dtype
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            (actual.float() * upstream).sum().backward()
            assert boundary.grad.dtype == torch.float32
            dist.send(boundary.grad.contiguous(), 0)

        reference_parameters = dict(reference.named_parameters())
        for name, parameter in stage.named_parameters():
            expected_grad = reference_parameters[name].grad
            if expected_grad is None:
                assert parameter.grad is None, name
            else:
                torch.testing.assert_close(parameter.grad, expected_grad, rtol=0, atol=0)
        torch.optim.SGD(stage.parameters(), lr=0.01).step()
        torch.optim.SGD(reference.parameters(), lr=0.01).step()
        for name, parameter in stage.named_parameters():
            torch.testing.assert_close(parameter, reference_parameters[name], rtol=0, atol=0)
        dist.barrier()

        # Production boundary shape: preserve sub-BF16 values and gradients in
        # 128 MiB tensors. Allocate the receiver from the model-owned metadata.
        stage.config.hidden_size = 8192
        inputs_meta, outputs_meta = stage.get_pipeline_stage_metas(
            is_first=rank == 0, microbatch_size=1, seq_len=4096, dtype=torch.bfloat16
        )
        if rank == 0:
            payload = torch.full((1, 4096, 8192), 1.0001, device=rank, requires_grad=True)
            assert payload.dtype == outputs_meta[0].dtype == torch.float32
            dist.send(payload.detach(), 1)
            gradient = torch.empty_like(payload)
            dist.recv(gradient, 1)
            payload.backward(gradient)
            torch.testing.assert_close(payload.grad, 2 * payload.detach(), rtol=0, atol=0)
        else:
            payload = torch.empty_like(inputs_meta[0], device=rank)
            assert payload.dtype == torch.float32
            assert payload.numel() * payload.element_size() == 128 * 1024**2
            dist.recv(payload, 0)
            torch.testing.assert_close(payload, torch.full_like(payload, 1.0001), rtol=0, atol=0)
            assert not torch.equal(payload, payload.bfloat16().float())
            payload.requires_grad_()
            payload.square().sum().backward()
            dist.send(payload.grad, 0)
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA GPUs")
def test_fp32_pipeline_residual_values_and_gradients() -> None:
    """BF16 compute preserves the FP32 residual across model partition boundaries."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(port,), nprocs=2, join=True)
