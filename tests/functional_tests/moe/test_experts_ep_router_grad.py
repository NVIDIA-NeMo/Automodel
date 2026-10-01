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

"""Scheduled two-rank regressions for gradients through the EP all-gather.

``GroupedExperts.forward`` all-gathers the per-token routing probabilities across
the expert-parallel group before dispatching tokens to local experts. Routing
probabilities participate in the main-loss gradient, so the gather must be
autograd-safe: a plain ``dist.all_gather`` detaches every gathered tensor and
silently leaves the router trainable only through auxiliary losses.

The NCCL LoRA variant also covers unequal per-rank token counts and checks its
sharded adapter gradients against a single-process fp32 reference.

Post-down routing additionally checks the loop and native grouped-MM backends
against a dense, single-process fp32 oracle, including ranks with no input tokens
or no local expert routes. No DeepEP installation or transport mocks are used.
"""

from __future__ import annotations

import os
import socket
import time
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor

from nemo_automodel.components._peft.lora_experts import GroupedExpertsLoRA
from nemo_automodel.components.models.common.utils import BackendConfig
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.experts import GroupedExperts

_N_EXPERTS = 4
_TOP_K = 2
_DIM = 16
_MOE_INTER_DIM = 32
# Uneven per-rank token counts exercise the variable-length gather path.
_TOKENS_PER_RANK = (3, 2)
_LORA_DIM = 4
_LORA_PARAM_NAMES = (
    "lora_gate_and_up_A",
    "lora_gate_and_up_B",
    "lora_down_A",
    "lora_down_B",
)


def _free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _tiny_moe_config() -> MoEConfig:
    return MoEConfig(
        n_routed_experts=_N_EXPERTS,
        n_shared_experts=0,
        n_activated_experts=_TOP_K,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="softmax",
        route_scale=1.0,
        dim=_DIM,
        inter_dim=_MOE_INTER_DIM,
        moe_inter_dim=_MOE_INTER_DIM,
        norm_topk_prob=False,
        expert_bias=False,
        expert_activation="swiglu",
        dtype=torch.float32,
    )


def _global_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deterministic global batch shared by the reference and EP runs."""
    generator = torch.Generator().manual_seed(1234)
    num_tokens = sum(_TOKENS_PER_RANK)
    x = torch.randn(num_tokens, _DIM, generator=generator)
    router_logits = torch.randn(num_tokens, _N_EXPERTS, generator=generator)
    weights, indices = router_logits.softmax(dim=-1).topk(_TOP_K, dim=-1)
    token_mask = torch.ones(num_tokens, dtype=torch.bool)
    return x, weights, indices, token_mask


def _build_experts(config: MoEConfig, backend: BackendConfig | None = None) -> GroupedExperts:
    generator = torch.Generator().manual_seed(4321)
    experts = GroupedExperts(config, backend=backend)
    with torch.no_grad():
        experts.gate_and_up_projs.copy_(torch.randn(experts.gate_and_up_projs.shape, generator=generator) * 0.05)
        experts.down_projs.copy_(torch.randn(experts.down_projs.shape, generator=generator) * 0.05)
    return experts


def _build_lora_experts(config: MoEConfig, backend: BackendConfig | None = None) -> GroupedExpertsLoRA:
    """Build experts with deterministic, nonzero LoRA weights."""
    experts = GroupedExpertsLoRA(_build_experts(config, backend), lora_dim=_LORA_DIM, alpha=8)
    generator = torch.Generator().manual_seed(9876)
    with torch.no_grad():
        for name in _LORA_PARAM_NAMES:
            param = getattr(experts, name)
            param.copy_(torch.randn(param.shape, generator=generator, dtype=param.dtype) * 0.05)
    return experts


def _lora_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return deterministic inputs and a nonuniform upstream gradient."""
    generator = torch.Generator().manual_seed(2468)
    num_tokens = sum(_TOKENS_PER_RANK)
    x = torch.randn(num_tokens, _DIM, generator=generator)
    weights = torch.rand(num_tokens, _TOP_K, generator=generator)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    indices = torch.tensor([[0, 1], [2, 3], [0, 2], [1, 3], [3, 0]])
    token_mask = torch.ones(num_tokens, dtype=torch.bool)
    output_grad = torch.randn(num_tokens, _DIM, generator=generator)
    return x, weights, indices, token_mask, output_grad


def _lora_reference_forward_backward(
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Run the single-process fp32 LoRA forward/backward reference."""
    experts = _build_lora_experts(_tiny_moe_config()).to(device)
    x, weights, indices, token_mask, output_grad = _lora_inputs()
    x, weights, indices, token_mask, output_grad = (
        tensor.to(device) for tensor in (x, weights, indices, token_mask, output_grad)
    )
    weights = weights.clone().requires_grad_(True)
    y = experts(x, token_mask, weights, indices)
    y.backward(output_grad)

    assert weights.grad is not None
    lora_grads = {}
    for name in _LORA_PARAM_NAMES:
        grad = getattr(experts, name).grad
        assert grad is not None
        assert torch.isfinite(grad).all()
        assert torch.count_nonzero(grad) > 0
        lora_grads[name] = grad.detach()
    return y.detach(), weights.grad.detach(), lora_grads


def _reference_forward_backward() -> tuple[torch.Tensor, torch.Tensor]:
    """Single-process (ep_size=1) forward/backward as ground truth."""
    experts = _build_experts(_tiny_moe_config())
    x, weights, indices, token_mask = _global_inputs()
    weights = weights.clone().requires_grad_(True)
    y = experts(x, token_mask, weights, indices)
    y.sum().backward()
    assert weights.grad is not None
    return y.detach(), weights.grad.detach()


def _ep_router_grad_worker(rank: int, world_size: int, port: int) -> None:
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

        y_ref, weights_grad_ref = _reference_forward_backward()

        ep_mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("ep",))
        experts = _build_experts(_tiny_moe_config())
        experts.gate_and_up_projs = nn.Parameter(
            distribute_tensor(experts.gate_and_up_projs.detach(), ep_mesh, [Shard(0)])
        )
        experts.down_projs = nn.Parameter(distribute_tensor(experts.down_projs.detach(), ep_mesh, [Shard(0)]))

        x, weights, indices, token_mask = _global_inputs()
        start = sum(_TOKENS_PER_RANK[:rank])
        end = start + _TOKENS_PER_RANK[rank]
        local_weights = weights[start:end].clone().requires_grad_(True)

        y_local = experts(x[start:end], token_mask[start:end], local_weights, indices[start:end])
        torch.testing.assert_close(y_local, y_ref[start:end], rtol=1e-4, atol=1e-5)

        y_local.sum().backward()

        # Pre-fix, the routing weights were gathered with a non-differentiable
        # ``dist.all_gather`` and the local router leaf received no gradient.
        assert local_weights.grad is not None, "router weights received no gradient through the EP all-gather"
        torch.testing.assert_close(local_weights.grad, weights_grad_ref[start:end], rtol=1e-4, atol=1e-5)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _lora_ep_ragged_worker(rank: int, world_size: int, port: int) -> None:
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
        dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=timedelta(seconds=60))

        y_ref, weights_grad_ref, lora_grads_ref = _lora_reference_forward_backward(device)

        ep_mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("ep",))
        experts = _build_lora_experts(_tiny_moe_config()).to(device)
        for name, param in list(experts.named_parameters(recurse=False)):
            dist_param = nn.Parameter(distribute_tensor(param.detach(), ep_mesh, [Shard(0)]))
            dist_param.requires_grad = param.requires_grad
            experts.register_parameter(name, dist_param)

        x, weights, indices, token_mask, output_grad = _lora_inputs()
        x, weights, indices, token_mask, output_grad = (
            tensor.to(device) for tensor in (x, weights, indices, token_mask, output_grad)
        )
        token_start = sum(_TOKENS_PER_RANK[:rank])
        token_end = token_start + _TOKENS_PER_RANK[rank]
        local_weights = weights[token_start:token_end].clone().requires_grad_(True)

        y_local = experts(
            x[token_start:token_end],
            token_mask[token_start:token_end],
            local_weights,
            indices[token_start:token_end],
        )
        assert torch.isfinite(y_local).all()
        torch.testing.assert_close(y_local, y_ref[token_start:token_end], rtol=1e-4, atol=1e-5)

        y_local.backward(output_grad[token_start:token_end])

        assert local_weights.grad is not None
        assert torch.isfinite(local_weights.grad).all()
        torch.testing.assert_close(local_weights.grad, weights_grad_ref[token_start:token_end], rtol=1e-4, atol=1e-5)

        n_local_experts = _N_EXPERTS // world_size
        expert_start = rank * n_local_experts
        expert_end = expert_start + n_local_experts
        for name in _LORA_PARAM_NAMES:
            grad = getattr(experts, name).grad
            assert grad is not None
            local_grad = grad.to_local()
            assert torch.isfinite(local_grad).all()
            torch.testing.assert_close(
                local_grad,
                lora_grads_ref[name][expert_start:expert_end],
                rtol=1e-4,
                atol=1e-5,
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is not available")
def test_ep_all_gather_propagates_router_weight_gradients():
    mp.spawn(
        _ep_router_grad_worker, args=(len(_TOKENS_PER_RANK), _free_port()), nprocs=len(_TOKENS_PER_RANK), join=True
    )


@pytest.mark.skipif(
    not dist.is_nccl_available() or torch.cuda.device_count() < len(_TOKENS_PER_RANK),
    reason="requires two CUDA devices and NCCL",
)
def test_lora_ep_ragged_forward_backward_matches_reference():
    world_size = len(_TOKENS_PER_RANK)
    mp.spawn(_lora_ep_ragged_worker, args=(world_size, _free_port()), nprocs=world_size, join=True)


def _dense_post_down_reference(
    experts: GroupedExpertsLoRA,
    x: torch.Tensor,
    token_mask: torch.Tensor,
    router_logits: torch.Tensor,
    indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate every expert with merged LoRA weights, without sparse dispatch.

    Args:
        experts: Unsharded fp32 experts with gate/up weights [experts, hidden,
            2 * intermediate], down weights [experts, intermediate, hidden],
            and their matching low-rank adapters.
        x: Global fp32 tensor of shape [tokens, hidden].
        token_mask: Global boolean tensor of shape [tokens].
        router_logits: Global fp32 tensor of shape [tokens, experts].
        indices: Global integer expert IDs of shape [tokens, top_k].

    Returns:
        FP32 output [tokens, hidden] and selected routing probabilities
        [tokens, top_k], on x's device. No inputs are mutated.
    """
    gate_up = experts.gate_and_up_projs + experts.scale * (experts.lora_gate_and_up_A @ experts.lora_gate_and_up_B)
    down = experts.down_projs + experts.scale * (experts.lora_down_A @ experts.lora_down_B)
    gate, up = torch.einsum("th,ehi->tei", x, gate_up).chunk(2, dim=-1)
    all_outputs = torch.einsum("tei,eih->teh", F.silu(gate) * up, down)
    selected = all_outputs.gather(1, indices.unsqueeze(-1).expand(-1, -1, x.size(-1)))
    weights = router_logits.softmax(dim=-1).gather(1, indices)
    return (selected * weights.unsqueeze(-1)).sum(dim=1) * token_mask.unsqueeze(-1), weights


def _lora_post_down_ep_worker(rank: int, port: int, backend_name: str, route_case: str) -> None:
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = "2"
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
        dist.init_process_group("nccl", rank=rank, world_size=2, timeout=timedelta(seconds=60))

        dtype = torch.bfloat16 if backend_name == "torch_mm" else torch.float32
        config = _tiny_moe_config()
        config.apply_router_weight_after_down = True
        backend = BackendConfig(experts=backend_name, dispatcher="torch")
        experts = _build_lora_experts(config, backend).to(device=device, dtype=dtype)
        # Round parameters to the compute dtype before the fp32 oracle, so the
        # comparison measures compute/reduction error, not initial quantization.
        reference = _build_lora_experts(config).to(dtype=dtype).float().to(device)
        tokens_per_rank = (0, 5) if route_case == "zero_token_rank" else _TOKENS_PER_RANK
        generator = torch.Generator().manual_seed(4567)
        x = torch.randn(5, _DIM, generator=generator).to(device=device, dtype=dtype)
        logits = torch.randn(5, _N_EXPERTS, generator=generator).to(device)
        # Expert 3 receives no routes; expert 2 receives far fewer than expert 0.
        indices = torch.tensor([[0, 2], [0, 1], [0, 2], [1, 0], [0, 1]], device=device)
        if route_case == "no_local_routes":
            indices = torch.tensor([[0, 1], [1, 0], [0, 1], [1, 0], [0, 1]], device=device)
        # Make the chosen routes actual top-k choices, with nonuniform scores.
        logits.scatter_add_(1, indices, torch.full((5, _TOP_K), 8.0, device=device))
        indices = logits.topk(_TOP_K, dim=-1).indices
        token_mask = torch.ones(5, dtype=torch.bool, device=device)
        if route_case == "fully_masked":
            token_mask.zero_()
        output_grad = torch.randn(5, _DIM, generator=generator).to(device=device, dtype=dtype)

        ref_x = x.float().detach().requires_grad_(True)
        ref_logits = logits.detach().clone().requires_grad_(True)
        y_ref, ref_weights = _dense_post_down_reference(reference, ref_x, token_mask, ref_logits, indices)
        ref_weights.retain_grad()
        y_ref.backward(output_grad.float())

        ep_mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("ep",))
        for name, param in list(experts.named_parameters(recurse=False)):
            experts.register_parameter(
                name,
                nn.Parameter(
                    distribute_tensor(param.detach(), ep_mesh, [Shard(0)]),
                    requires_grad=param.requires_grad,
                ),
            )
        assert experts.use_torch_mm == (backend_name == "torch_mm")
        assert experts.gate_and_up_projs.placements == (Shard(0),)
        start = sum(tokens_per_rank[:rank])
        end = start + tokens_per_rank[rank]
        local_x = x[start:end].detach().clone().requires_grad_(True)
        local_logits = logits[start:end].detach().clone().requires_grad_(True)
        local_weights = local_logits.softmax(dim=-1).gather(1, indices[start:end])
        local_weights.retain_grad()
        y = experts(local_x, token_mask[start:end], local_weights, indices[start:end])
        y.backward(output_grad[start:end])

        # BF16 grouped GEMMs round each additive LoRA projection, unlike the
        # merged fp32 oracle. FP32 loop parity uses much tighter tolerances.
        rtol, atol = (4e-2, 2e-4) if dtype == torch.bfloat16 else (1e-4, 1e-6)
        for label, actual, expected in (
            ("output", y, y_ref[start:end]),
            ("input gradient", local_x.grad, ref_x.grad[start:end]),
            ("router gradient", local_logits.grad, ref_logits.grad[start:end]),
            ("routing probability gradient", local_weights.grad, ref_weights.grad[start:end]),
        ):
            assert actual is not None, f"{label} missing on rank {rank}, {route_case}, {backend_name}"
            assert torch.isfinite(actual).all(), label
            assert torch.isfinite(expected).all(), label
            torch.testing.assert_close(actual.float(), expected, rtol=rtol, atol=atol, msg=label)
            if route_case == "fully_masked":
                assert torch.count_nonzero(actual) == 0, label
            elif actual.numel():
                assert torch.count_nonzero(actual) > 0, label

        expert_start = rank * (_N_EXPERTS // 2)
        expert_end = expert_start + _N_EXPERTS // 2
        active_experts = torch.bincount(indices[token_mask].flatten(), minlength=_N_EXPERTS) > 0
        local_active = active_experts[expert_start:expert_end]
        for name in _LORA_PARAM_NAMES:
            param = getattr(experts, name)
            assert param.placements == (Shard(0),)
            assert param.grad is not None, f"{name} missing on rank {rank}, {route_case}, {backend_name}"
            grad = param.grad.to_local()
            ref_grad = getattr(reference, name).grad
            assert ref_grad is not None
            assert torch.isfinite(grad).all()
            assert torch.isfinite(ref_grad).all()
            torch.testing.assert_close(grad.float(), ref_grad[expert_start:expert_end], rtol=rtol, atol=atol, msg=name)
            # Unused experts (including an entirely idle rank) must have exact
            # zero gradients, while routed experts must exercise all adapters.
            assert torch.count_nonzero(grad[~local_active]) == 0, name
            assert torch.count_nonzero(ref_grad[~active_experts]) == 0, name
            for expert_grad in grad[local_active]:
                assert torch.count_nonzero(expert_grad) > 0, name
        for name in ("gate_and_up_projs", "down_projs"):
            assert not getattr(experts, name).requires_grad
            assert getattr(experts, name).grad is None
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_nccl_available() or torch.cuda.device_count() < 2,
    reason="requires two CUDA devices and NCCL",
)
@pytest.mark.parametrize("backend_name", ["torch", "torch_mm"], ids=["loop-fp32", "native-grouped-mm-bf16"])
@pytest.mark.parametrize("route_case", ["uneven_routes", "no_local_routes", "zero_token_rank", "fully_masked"])
def test_lora_post_down_ep_forward_backward_matches_dense_reference(backend_name: str, route_case: str) -> None:
    """Exercise post-down slot reduction with real two-rank NCCL transport."""
    if backend_name == "torch_mm" and (
        not hasattr(torch, "_grouped_mm") or any(torch.cuda.get_device_capability(rank)[0] < 9 for rank in range(2))
    ):
        pytest.skip("native CUDA torch._grouped_mm requires PyTorch support and two SM90+ GPUs")
    context = mp.spawn(_lora_post_down_ep_worker, args=(_free_port(), backend_name, route_case), nprocs=2, join=False)
    deadline = time.monotonic() + 180
    try:
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                pytest.fail(f"two-rank post-down LoRA timed out: {backend_name}, {route_case}")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
