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

"""Eager parity runner for whole-attention CUDA Graph capture.

Single-GPU mixed PyTorch + TE validation::

    python tests/functional_tests/attention/run_whole_attention_cuda_graph.py

Two-GPU FSDP2 validation::

    torchrun --nproc-per-node=2 tests/functional_tests/attention/run_whole_attention_cuda_graph.py
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FSDPModule, fully_shard
from torch.distributed.tensor import DTensor
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.gpt_oss.rope_utils import RotaryEmbedding, position_ids_to_freqs_cis
from nemo_automodel.components.models.qwen3_moe.layers import Qwen3MoeAttention
from nemo_automodel.recipes.llm.partial_cuda_graphs import PartialCudaGraphManager

HIDDEN_SIZE = 64
HEAD_DIM = 16
NUM_HEADS = 4
NUM_KV_HEADS = 2
SEQUENCE_LENGTH = 128
STEPS = 4
ATOL = 5e-3
RTOL = 5e-3
MAX_ABS_ERRORS: dict[str, float] = {}


class _AttentionLayer(nn.Module):
    def __init__(self, attention: nn.Module) -> None:
        super().__init__()
        self.self_attn = attention

    def forward(self, hidden_states: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        return self.self_attn(hidden_states, freqs_cis=freqs_cis)


class _AttentionModel(nn.Module):
    def __init__(self, layer: nn.Module, backend: BackendConfig) -> None:
        super().__init__()
        self.config = SimpleNamespace(model_type="qwen3_moe_whole_attention_graph")
        self.backend = backend
        self.model = nn.Module()
        self.model.layers = nn.ModuleDict({"0": layer})

    def forward(self, hidden_states: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        return self.model.layers["0"](hidden_states, freqs_cis)


def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    actual_local = _local_tensor(actual).float()
    expected_local = _local_tensor(expected).float()
    MAX_ABS_ERRORS[name] = float((actual_local - expected_local).detach().abs().max())
    torch.testing.assert_close(
        actual_local,
        expected_local,
        atol=ATOL,
        rtol=RTOL,
        msg=lambda message: f"{name}: {message}",
    )


def _build_model(device: torch.device, *, graph: bool) -> _AttentionModel:
    backend = BackendConfig(
        attn="te",
        linear="torch",
        rms_norm="torch",
        rope_fusion=False,
        cuda_graph_modules=["attn"] if graph else [],
    )
    config = Qwen3MoeConfig(
        hidden_size=HIDDEN_SIZE,
        intermediate_size=128,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        rms_norm_eps=1e-6,
        attention_bias=False,
        attention_dropout=0.0,
        torch_dtype=torch.bfloat16,
    )
    with device:
        attention = Qwen3MoeAttention(config, backend)
    return _AttentionModel(_AttentionLayer(attention), backend).train()


def _compare_parameter_state(graph_model: nn.Module, eager_model: nn.Module, *, gradients: bool) -> None:
    graph_parameters = tuple(graph_model.named_parameters())
    eager_parameters = tuple(eager_model.named_parameters())
    assert tuple(name for name, _parameter in graph_parameters) == tuple(name for name, _parameter in eager_parameters)
    for (name, graph_parameter), (_eager_name, eager_parameter) in zip(graph_parameters, eager_parameters):
        if gradients:
            assert graph_parameter.grad is not None, f"missing graph gradient for {name}"
            assert eager_parameter.grad is not None, f"missing eager gradient for {name}"
            _assert_close(f"parameter gradient {name}", graph_parameter.grad, eager_parameter.grad)
        else:
            _assert_close(f"updated parameter {name}", graph_parameter, eager_parameter)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("whole-attention CUDA Graph validation requires CUDA")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    torch.manual_seed(1234)
    eager_model = _build_model(device, graph=False)
    torch.manual_seed(5678)
    graph_model = _build_model(device, graph=True)
    graph_model.load_state_dict(eager_model.state_dict())

    if world_size > 1:
        mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp",))
        fully_shard(eager_model.model.layers["0"], mesh=mesh, reshard_after_forward=True)
        fully_shard(graph_model.model.layers["0"], mesh=mesh, reshard_after_forward=True)

    eager_optimizer = torch.optim.SGD(eager_model.parameters(), lr=1e-3)
    graph_optimizer = torch.optim.SGD(graph_model.parameters(), lr=1e-3)
    graph_manager = PartialCudaGraphManager.from_model_parts([graph_model])
    assert graph_manager is not None
    assert len(graph_manager.entries) == 1
    graph_entry = graph_manager.entries[0]
    assert graph_entry.explicit_parameters is True
    if world_size > 1:
        assert isinstance(graph_entry.capture_owner, FSDPModule)
    else:
        assert graph_entry.capture_owner is None

    rotary = RotaryEmbedding(
        head_dim=HEAD_DIM,
        base=1_000_000,
        dtype=torch.bfloat16,
        device=device,
    )
    position_ids = torch.arange(SEQUENCE_LENGTH, device=device).unsqueeze(0)
    freqs_cis = position_ids_to_freqs_cis(rotary, position_ids, for_fused_rope=False)

    try:
        for step in range(STEPS):
            eager_optimizer.zero_grad(set_to_none=True)
            graph_optimizer.zero_grad(set_to_none=True)
            generator = torch.Generator(device=device).manual_seed(9000 + step + 1000 * local_rank)
            eager_input = torch.randn(
                1,
                SEQUENCE_LENGTH,
                HIDDEN_SIZE,
                generator=generator,
                device=device,
                dtype=torch.bfloat16,
                requires_grad=True,
            )
            graph_input = eager_input.detach().clone().requires_grad_()

            eager_output = eager_model(eager_input, freqs_cis)
            graph_output = graph_model(graph_input, freqs_cis)
            _assert_close(f"step {step} output", graph_output, eager_output)

            eager_loss = eager_output.float().square().mean()
            graph_loss = graph_output.float().square().mean()
            _assert_close(f"step {step} loss", graph_loss, eager_loss)
            eager_loss.backward()
            graph_loss.backward()
            _assert_close(f"step {step} input gradient", graph_input.grad, eager_input.grad)
            _compare_parameter_state(graph_model, eager_model, gradients=True)

            eager_optimizer.step()
            graph_optimizer.step()
            _compare_parameter_state(graph_model, eager_model, gradients=False)
            if step == 0:
                graph_manager.capture()
                if world_size > 1:
                    assert all(isinstance(parameter, DTensor) for parameter in graph_entry.target.parameters())

        expected_replays = STEPS - 1
        stats = graph_manager.stats()
        assert stats == {"captured": 1, "replayed": expected_replays, "fallback": 0}, stats
        if local_rank == 0:
            print(
                json.dumps(
                    {
                        "status": "passed",
                        "scope": "attn",
                        "mixed_backend": "torch projections/rmsnorm + TE DPA",
                        "world_size": world_size,
                        "fsdp2": world_size > 1,
                        "stats": stats,
                        "max_abs_error": max(MAX_ABS_ERRORS.values()),
                    },
                    sort_keys=True,
                )
            )
    finally:
        graph_manager.close()
        if world_size > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
