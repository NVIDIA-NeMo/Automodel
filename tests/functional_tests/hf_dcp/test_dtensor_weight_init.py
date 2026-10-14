# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Check full-source initialization copies and Falcon H1 with real CUDA sharding."""

import socket
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor
from transformers import FalconH1Config, FalconH1ForCausalLM

from nemo_automodel.components.checkpoint.checkpointing import Checkpointer
from nemo_automodel.components.checkpoint.config import CheckpointingConfig


def _worker(rank: int, world_size: int, port: int, root: str, dtype: torch.dtype) -> None:
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world_size, timeout=timedelta(seconds=120)
    )
    try:
        mesh = init_device_mesh("cuda", (world_size,))

        class CopyModel(torch.nn.Module):
            def __init__(self, shape, placement):
                super().__init__()
                self.weight = torch.nn.Parameter(
                    distribute_tensor(torch.empty(shape, device=device, dtype=dtype), mesh, [placement])
                )

            @torch.no_grad()
            def initialize_weights(self):
                # Different rank-local sources must yield the logical rank-zero value.
                source = torch.arange(self.weight.numel(), device=device).reshape(self.weight.shape) + rank * 100
                expected_source = source.clone()
                self.weight.copy_(source)
                torch.testing.assert_close(source, expected_source, rtol=0, atol=0)

        for shape, placement in [((5, 3), Shard(0)), ((3, 5), Shard(1)), ((1, 3), Shard(0)), ((5,), Replicate())]:
            copied = CopyModel(shape, placement)
            Checkpointer.initialize_model_weights(copied, device)
            expected = torch.arange(copied.weight.numel(), device=device, dtype=dtype).reshape(shape)
            torch.testing.assert_close(copied.weight.full_tensor(), expected, rtol=0, atol=0)

        config = FalconH1Config(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            mamba_d_ssm=64,
            mamba_n_heads=8,
            mamba_d_head=8,
            mamba_d_state=8,
            mamba_n_groups=1,
            mamba_chunk_size=4,
            pad_token_id=0,
            ssm_multipliers=[1.0, 2.0, 3.0, 4.0, 5.0],
        )
        torch.manual_seed(42)
        reference = FalconH1ForCausalLM(config).to(device=device, dtype=dtype).eval()
        with torch.device("meta"):
            model = FalconH1ForCausalLM(config).to(dtype=dtype).eval()
        for layer in model.model.layers:
            fully_shard(layer, mesh=mesh)
        fully_shard(model, mesh=mesh)
        Checkpointer.initialize_model_weights(model, device)

        mixer = model.model.layers[0].mamba
        for name in ("A_log", "dt_bias", "D"):
            torch.testing.assert_close(
                getattr(mixer, name).full_tensor(), getattr(reference.model.layers[0].mamba, name), rtol=0, atol=0
            )
        initialized = {
            name: value.full_tensor() if isinstance(value, DTensor) else value
            for name, value in model.state_dict().items()
        }
        assert all(torch.isfinite(value).all() for value in initialized.values())
        reference.load_state_dict(initialized)
        for name, buffer in model.named_buffers():
            torch.testing.assert_close(buffer, reference.get_buffer(name), rtol=0, atol=0)
        inputs = torch.tensor([[1, 3, 4, 5, 6, 7, 8, 2]], device=device)
        expected = reference(input_ids=inputs, labels=inputs, use_cache=False)
        actual = model(input_ids=inputs, labels=inputs, use_cache=False)
        tolerance = 2e-2 if dtype == torch.bfloat16 else 1e-5
        assert torch.isfinite(actual.loss)
        torch.testing.assert_close(actual.logits, expected.logits, rtol=tolerance, atol=tolerance)
        actual.loss.backward()
        expected.loss.backward()
        for name, parameter in model.named_parameters():
            expected_gradient = reference.get_parameter(name).grad
            assert parameter.grad is not None
            torch.testing.assert_close(parameter.grad.full_tensor(), expected_gradient, rtol=tolerance, atol=tolerance)
        torch.optim.SGD(model.parameters(), lr=0.01).step()
        torch.optim.SGD(reference.parameters(), lr=0.01).step()
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(
                parameter.full_tensor(), reference.get_parameter(name), rtol=tolerance, atol=tolerance
            )

        # A subsequent base-checkpoint DCP load must still replace the initialized weights exactly.
        checkpoint = Path(root) / "model"
        if rank == 0:
            reference.save_pretrained(checkpoint)
        dist.barrier()
        Checkpointer(
            CheckpointingConfig(checkpoint_dir=str(Path(root) / "checkpoints"), model_save_format="safetensors"),
            dp_rank=rank,
            tp_rank=0,
            pp_rank=0,
            moe_mesh=None,
        ).load_model(model, str(checkpoint), is_init_step=True)
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter.full_tensor(), reference.get_parameter(name), rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA GPUs for real FSDP2 initialization")
@pytest.mark.parametrize("world_size,dtype", [(1, torch.float32), (2, torch.float32), (2, torch.bfloat16)])
def test_dtensor_initialization_and_falcon_h1_parity(tmp_path: Path, world_size: int, dtype: torch.dtype) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(world_size, port, str(tmp_path), dtype), nprocs=world_size, join=True)
