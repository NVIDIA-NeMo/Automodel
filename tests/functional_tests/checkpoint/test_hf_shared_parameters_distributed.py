# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Exercise standard HF DCP loading with real shared parameters and GPU sharding."""

import socket
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor
from torch.nn.parallel import DistributedDataParallel
from transformers import T5Config, T5ForConditionalGeneration

from nemo_automodel.components.checkpoint.checkpointing import Checkpointer
from nemo_automodel.components.checkpoint.config import CheckpointingConfig


def _worker(rank: int, world_size: int, port: int, checkpoint_root: str, strategy: str, dtype: torch.dtype) -> None:
    """Load a shared-embedding checkpoint and compare weights, gradients, and an optimizer step on each rank."""
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=120),
    )
    try:
        torch.manual_seed(1234)
        config = T5Config(
            vocab_size=32,
            d_model=16,
            d_ff=32,
            d_kv=8,
            num_layers=1,
            num_decoder_layers=1,
            num_heads=2,
            decoder_start_token_id=0,
            dropout_rate=0.0,
        )
        reference = T5ForConditionalGeneration(config).to(device=rank, dtype=dtype).eval()
        model_path = Path(checkpoint_root) / "model"
        if rank == 0:
            reference.save_pretrained(model_path)
        dist.barrier()

        torch.manual_seed(5678 + rank)
        model = T5ForConditionalGeneration(config).to(device=rank, dtype=dtype).eval()
        if strategy == "fsdp2":
            mesh = init_device_mesh("cuda", (world_size,))
            for block in (*model.encoder.block, *model.decoder.block):
                fully_shard(block, mesh=mesh)
            fully_shard(model, mesh=mesh)
        elif strategy == "ddp":
            model = DistributedDataParallel(model, device_ids=[rank])
        unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
        checkpointer = Checkpointer(
            CheckpointingConfig(
                enabled=True,
                checkpoint_dir=str(Path(checkpoint_root) / "checkpoints"),
                model_cache_dir=str(Path(checkpoint_root) / "cache"),
                model_repo_id="test/t5",
                model_save_format="safetensors",
                save_consolidated=False,
            ),
            dp_rank=rank,
            tp_rank=0,
            pp_rank=0,
            moe_mesh=None,
        )
        with patch(
            "nemo_automodel.components.checkpoint.checkpointing._load_hf_checkpoint_preserving_dtype",
            side_effect=AssertionError("Expected DCP loading"),
        ):
            checkpointer.load_model(model, str(model_path), is_init_step=True)

        assert unwrapped.encoder.embed_tokens.weight is unwrapped.decoder.embed_tokens.weight
        assert unwrapped.lm_head.weight is unwrapped.shared.weight
        for name, tensor in unwrapped.state_dict().items():
            full_tensor = tensor.full_tensor() if isinstance(tensor, DTensor) else tensor
            torch.testing.assert_close(full_tensor, reference.state_dict()[name], rtol=0, atol=0)

        inputs = torch.tensor([[3, 4, 5, 2]], device=rank)
        labels = torch.tensor([[6, 7, 2]], device=rank)
        expected = reference(input_ids=inputs, labels=labels)
        actual = model(input_ids=inputs, labels=labels)
        tolerance = 1e-6 if dtype == torch.float32 else 2e-2
        torch.testing.assert_close(actual.logits, expected.logits, rtol=tolerance, atol=tolerance)
        actual.loss.backward()
        expected.loss.backward()
        reference_parameters = dict(reference.named_parameters())
        for name, parameter in unwrapped.named_parameters():
            assert parameter.grad is not None
            gradient = parameter.grad.full_tensor() if isinstance(parameter.grad, DTensor) else parameter.grad
            torch.testing.assert_close(gradient, reference_parameters[name].grad, rtol=tolerance, atol=tolerance)

        torch.optim.SGD(model.parameters(), lr=0.01).step()
        torch.optim.SGD(reference.parameters(), lr=0.01).step()
        for name, parameter in unwrapped.named_parameters():
            full_parameter = parameter.full_tensor() if isinstance(parameter, DTensor) else parameter
            torch.testing.assert_close(full_parameter, reference_parameters[name], rtol=tolerance, atol=tolerance)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="the shared-parameter regression matrix requires two CUDA GPUs"
)
@pytest.mark.parametrize(
    ("world_size", "strategy", "dtype"),
    [
        (1, "unsharded", torch.bfloat16),
        (2, "fsdp2", torch.float32),
        (2, "fsdp2", torch.bfloat16),
        (2, "ddp", torch.float32),
        (2, "ddp", torch.bfloat16),
    ],
)
def test_hf_shared_parameters_load_on_gpu(tmp_path: Path, world_size: int, strategy: str, dtype: torch.dtype) -> None:
    """DCP must restore omitted aliases without breaking distributed gradients or parameter ownership."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(world_size, port, str(tmp_path), strategy, dtype), nprocs=world_size, join=True)
