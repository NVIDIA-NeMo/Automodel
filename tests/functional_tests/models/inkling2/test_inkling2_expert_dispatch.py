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

"""Two-GPU exact parity of Inkling 2's routed experts across the torch and HybridEP dispatchers.

HybridEP's combine only accepts 16-bit tokens, so an FP32 comparison is impossible. Instead inputs,
expert weights, routing weights and the upstream gradient are sparse small integers or powers of two, and
the expert activation is the exact ``(g + u) * probs`` on Inkling 2's interleaved gate/up layout, so every
intermediate is representable in BF16. Every backend must then match an FP64 dense reference bit for bit:
outputs, input gradients, routing-weight gradients (routing weights multiply after the down projection) and
both expert-weight gradients. Before the swap, each backend must hold the Inkling 2 softcapped activation.
"""

import importlib.util
import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.inkling2.configuration import Inkling2Config
from nemo_automodel.components.models.inkling2.layers import Inkling2MoE, inkling2_moe_config
from nemo_automodel.components.moe.experts import GroupedExperts, GroupedExpertsTE
from nemo_automodel.components.moe.parallelizer import apply_ep

_TOKENS = 64
_BACKENDS = (("torch", "torch_mm"), ("hybridep", "torch_mm"), ("hybridep", "te"))


def _text_config():
    text = {
        "hidden_size": 256,
        "num_hidden_layers": 1,
        "local_layer_ids": [0],
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 128,
        "vocab_size": 64,
        "n_routed_experts": 32,
        "num_experts_per_tok": 8,
        "intermediate_size": 128,
        "eos_token_id": 1,
    }
    return Inkling2Config(text_config=text, vision_config=None, eos_token_id=1).text_config


def _exact_activation(gate_and_up: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    """Exact stand-in for the softcapped SwiGLU.

    Args:
        gate_and_up: Tensor of shape [routed_tokens, 2 * intermediate] interleaving gate and up channels.
        probs: Tensor of shape [routed_tokens, 1] (ones when weights apply after the down projection).

    Returns:
        Tensor of shape [routed_tokens, intermediate] in ``gate_and_up``'s dtype.
    """
    return ((gate_and_up[..., 0::2].float() + gate_and_up[..., 1::2].float()) * probs.float()).to(gate_and_up.dtype)


def _sparse_ints(shape: tuple[int, ...], density: float, generator: torch.Generator) -> torch.Tensor:
    values = torch.randint(-1, 2, shape, generator=generator).float()
    return values * (torch.rand(shape, generator=generator) < density).float()


def _rank_inputs(rank: int, config, device: torch.device):
    """Return one rank's tokens [tokens, hidden], expert ids and routing weights [tokens, top_k], and grad."""
    generator = torch.Generator().manual_seed(1000 + rank)
    tokens = _sparse_ints((_TOKENS, config.hidden_size), 0.05, generator)
    indices = torch.stack(
        [
            torch.randperm(config.n_routed_experts, generator=generator)[: config.num_experts_per_tok]
            for _ in range(_TOKENS)
        ]
    )
    weights = torch.tensor([0.25, 0.5, 1.0])[torch.randint(0, 3, indices.shape, generator=generator)]
    grad = _sparse_ints((_TOKENS, config.hidden_size), 0.1, generator)
    return tokens.to(device), indices.to(device), weights.to(device), grad.to(device)


def _reference(config, gate_up: torch.Tensor, down: torch.Tensor, rank: int, device: torch.device) -> dict:
    """FP64 dense reference over both ranks' tokens (expert gradients accumulate across ranks)."""
    gate_up = gate_up.double().to(device).requires_grad_()
    down = down.double().to(device).requires_grad_()
    local = {}
    for source in range(2):
        tokens, indices, weights, grad = _rank_inputs(source, config, device)
        tokens = tokens.double().requires_grad_()
        weights = weights.double().requires_grad_()
        out = torch.zeros_like(tokens)
        for slot in range(indices.shape[1]):
            expert = indices[:, slot]
            projected = torch.einsum("th,thi->ti", tokens, gate_up[expert])
            activated = projected[:, 0::2] + projected[:, 1::2]
            out = out + torch.einsum("ti,tih->th", activated, down[expert]) * weights[:, slot : slot + 1]
        (out * grad.double()).sum().backward()
        if source == rank:
            local = {"out": out.detach(), "tokens": tokens.grad, "weights": weights.grad}
    n_local = config.n_routed_experts // 2
    experts = slice(rank * n_local, (rank + 1) * n_local)
    local["gate_up"] = gate_up.grad[experts]
    local["down"] = down.grad[experts]
    return local


class _Block(nn.Module):
    def __init__(self, moe: Inkling2MoE) -> None:
        super().__init__()
        self.mlp = moe


class _Backbone(nn.Module):
    def __init__(self, moe: Inkling2MoE) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Block(moe)])


class _Model(nn.Module):
    """Minimal model tree in the layout the expert-parallel parallelizer walks."""

    def __init__(self, moe: Inkling2MoE) -> None:
        super().__init__()
        self.model = _Backbone(moe)


def _local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _run_backend(dispatcher: str, experts_backend: str, config, gate_up, down, mesh, device) -> dict:
    rank = dist.get_rank()
    backend = BackendConfig(attn="sdpa", linear="torch", experts=experts_backend, dispatcher=dispatcher)
    with torch.device(device):
        moe = Inkling2MoE(config, backend, inkling2_moe_config(config, torch.bfloat16))
    experts = moe.experts
    hook = "expert_activation_grouped" if isinstance(experts, GroupedExperts) else "expert_activation"
    assert getattr(experts, hook).func.__name__ == "inkling2_softcapped_swiglu"
    apply_ep(_Model(moe), mesh["ep"])
    n_local = config.n_routed_experts // 2
    owned = slice(rank * n_local, (rank + 1) * n_local)
    if isinstance(experts, GroupedExpertsTE):
        experts.to_empty(device=device)
        experts.gate_and_up_projs = gate_up[owned].to(device=device, dtype=torch.bfloat16)
        experts.down_projs = down[owned].to(device=device, dtype=torch.bfloat16)
    else:
        with torch.no_grad():
            _local(experts.gate_and_up_projs).copy_(gate_up[owned])
            _local(experts.down_projs).copy_(down[owned])
    setattr(experts, hook, _exact_activation)

    tokens, indices, weights, grad = _rank_inputs(rank, config, device)
    tokens = tokens.to(torch.bfloat16).requires_grad_()
    weights = weights.requires_grad_()
    out = experts(tokens, torch.ones(_TOKENS, dtype=torch.bool, device=device), weights, indices)
    (out.float() * grad).sum().backward()
    if isinstance(experts, GroupedExpertsTE):
        gate_up_grad = torch.stack([getattr(experts.gate_up_linear, f"weight{i}").grad for i in range(n_local)])
        down_grad = torch.stack([getattr(experts.down_linear, f"weight{i}").grad for i in range(n_local)])
        gate_up_grad, down_grad = gate_up_grad.transpose(-1, -2), down_grad.transpose(-1, -2)
    else:
        gate_up_grad, down_grad = _local(experts.gate_and_up_projs.grad), _local(experts.down_projs.grad)
    return {"out": out, "tokens": tokens.grad, "weights": weights.grad, "gate_up": gate_up_grad, "down": down_grad}


def _worker(rank: int, port: int, backends: tuple[tuple[str, str], ...]) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2, timeout=timedelta(seconds=300)
    )
    try:
        device = torch.device("cuda", rank)
        mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("ep",))
        config = _text_config()
        generator = torch.Generator().manual_seed(7)
        gate_up = _sparse_ints(
            (config.n_routed_experts, config.hidden_size, 2 * config.moe_intermediate_size), 0.05, generator
        )
        down = _sparse_ints(
            (config.n_routed_experts, config.moe_intermediate_size, config.hidden_size), 0.05, generator
        )
        expected = _reference(config, gate_up, down, rank, device)
        for name, value in expected.items():
            assert torch.equal(value.to(torch.bfloat16).double(), value), f"reference {name} is not BF16-exact"
        assert (expected["out"] != 0).float().mean() > 0.5, "the routed outputs must be mostly nonzero"
        for dispatcher, experts_backend in backends:
            got = _run_backend(dispatcher, experts_backend, config, gate_up, down, mesh, device)
            for name, value in expected.items():
                torch.testing.assert_close(
                    got[name].double(), value, rtol=0, atol=0, msg=f"{dispatcher}/{experts_backend} {name}"
                )
            dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA GPUs")
def test_inkling2_experts_match_exactly_across_dispatchers() -> None:
    """Torch, HybridEP + torch_mm and HybridEP + TE give bitwise-identical expert outputs and gradients."""
    backends = tuple(
        (dispatcher, experts_backend)
        for dispatcher, experts_backend in _BACKENDS
        if (dispatcher != "hybridep" or importlib.util.find_spec("deep_ep") is not None)
        and (experts_backend != "te" or importlib.util.find_spec("transformer_engine") is not None)
    )
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(port, backends), nprocs=2, join=True)
