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

"""Two-rank uneven expert ownership, training, and checkpoint regressions."""

import importlib.util
import os
import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor
from transformers import Qwen3MoeConfig

from nemo_automodel.components._peft.lora_experts import GroupedExpertsDeepEPLoRA, GroupedExpertsLoRA
from nemo_automodel.components.models.common.utils import BackendConfig
from nemo_automodel.components.models.qwen3_moe.state_dict_adapter import Qwen3MoeStateDictAdapter
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.experts import GroupedExperts, GroupedExpertsDeepEP
from nemo_automodel.components.moe.megatron.fused_a2a import HAVE_HYBRIDEP, reset_hybrid_ep_buffer
from nemo_automodel.components.moe.state_dict_utils import (
    create_dtensor_from_local,
    get_expert_range_for_rank_from_mesh,
    split_experts_weights_dtensor_aware,
)


def _port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _config(n_experts: int, dtype: torch.dtype) -> MoEConfig:
    return MoEConfig(
        n_routed_experts=n_experts,
        n_shared_experts=0,
        n_activated_experts=min(n_experts, 2),
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="softmax",
        route_scale=1.0,
        dim=128,
        inter_dim=128,
        moe_inter_dim=128,
        norm_topk_prob=False,
        expert_bias=False,
        expert_activation="swiglu",
        dtype=dtype,
    )


def _shard(module: nn.Module, mesh: DeviceMesh) -> None:
    for name, parameter in list(module.named_parameters(recurse=False)):
        module.register_parameter(
            name,
            nn.Parameter(
                distribute_tensor(parameter.detach(), mesh, [Shard(0)]), requires_grad=parameter.requires_grad
            ),
        )


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, dtype: torch.dtype) -> None:
    """Check distributed results against the per-expert FP32 reference.

    Args:
        actual: Tensor of arbitrary shape containing a local output, gradient,
            optimizer update, or scalar gradient norm.
        expected: Reference tensor with the same semantic shape as actual.
        dtype: Expert compute dtype controlling the numerical error bound.
    """
    if dtype == torch.float32:
        torch.testing.assert_close(actual.float(), expected.float(), rtol=1e-4, atol=1e-6)
    else:
        # The unchanged even-EP BF16 control differs from FP32 by about 0.4%
        # relative L2, including values near zero. Check both L2 and max error
        # against tensor magnitude instead of per-element relative error.
        actual, expected = actual.float(), expected.float()
        assert torch.isfinite(actual).all()
        if not expected.numel():
            assert actual.shape == expected.shape
            return
        diff = actual - expected
        torch.testing.assert_close(
            diff.norm(), torch.zeros_like(diff.norm()), rtol=0, atol=0.02 * expected.norm().item() + 1e-7
        )
        torch.testing.assert_close(
            diff.abs().max(), torch.zeros_like(diff.abs().max()), rtol=0, atol=0.02 * expected.abs().max().item() + 1e-7
        )


def _case(rank: int, mesh: DeviceMesh, n_experts: int, backend_name: str, lora: bool) -> None:
    dtype = torch.float32 if backend_name == "torch" else torch.bfloat16
    config = _config(n_experts, dtype)
    torch.manual_seed(71)
    device = torch.device("cuda", rank)
    backend = BackendConfig(experts="torch" if backend_name == "torch" else "torch_mm")
    if backend_name in ("deepep", "hybridep"):
        module = GroupedExpertsDeepEP(
            config, dispatcher_backend=backend_name, dispatcher_share_token_dispatcher=False, backend=backend
        )
    else:
        module = GroupedExperts(config, backend)
    module = module.to(device=device, dtype=dtype)
    if lora:
        wrapper = GroupedExpertsDeepEPLoRA if backend_name in ("deepep", "hybridep") else GroupedExpertsLoRA
        module = wrapper(module, lora_dim=8, alpha=8).to(device=device, dtype=dtype)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(std=0.02)
    reference = GroupedExperts(config).to(device=device, dtype=torch.float32)
    if lora:
        reference = GroupedExpertsLoRA(reference, lora_dim=8, alpha=8).to(device=device, dtype=torch.float32)
    reference.load_state_dict(module.state_dict())
    _shard(module, mesh)
    if backend_name in ("deepep", "hybridep"):
        module.init_token_dispatcher(mesh)
        assert module.token_dispatcher.num_local_experts == (n_experts + 1) // 2
    # Independent torch.chunk oracle: do not derive expected IDs through production helpers.
    chunks = torch.arange(n_experts).chunk(2)
    ids = chunks[rank] if rank < len(chunks) else torch.empty(0, dtype=torch.int64)
    first = sum(chunk.numel() for chunk in chunks[:rank])
    last = first + ids.numel()
    assert get_expert_range_for_rank_from_mesh(mesh, n_experts) == (first, last)
    for name, parameter in module.named_parameters():
        exported, exported_ids = split_experts_weights_dtensor_aware(parameter, n_experts)
        assert exported_ids == ids.tolist()
        for value, expert_id in zip(exported, exported_ids):
            torch.testing.assert_close(value.float(), reference.get_parameter(name)[expert_id])
        restored = create_dtensor_from_local(parameter.to_local().detach().clone(), mesh, n_experts=n_experts)
        assert restored.shape == reference.get_parameter(name).shape
        torch.testing.assert_close(restored.full_tensor().float(), reference.get_parameter(name))
    if not lora:
        adapter = Qwen3MoeStateDictAdapter(Qwen3MoeConfig(num_experts=n_experts), config, backend, dtype=dtype)
        prefix = "model.layers.0.mlp.experts."
        native = {prefix + name: parameter.detach().to(dtype) for name, parameter in reference.named_parameters()}
        hf_state = adapter.to_hf(native)
        restored_native = adapter.from_hf(hf_state, device_mesh=mesh)
        for name, parameter in module.named_parameters():
            restored_parameter = restored_native[prefix + name]
            assert restored_parameter.shape == parameter.shape
            torch.testing.assert_close(restored_parameter.to_local(), parameter.to_local())
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
    lengths = (17, 11)
    start, stop = sum(lengths[:rank]), sum(lengths[: rank + 1])
    topk = config.n_activated_experts
    generator = torch.Generator(device=device).manual_seed(83)
    for masked in (False, True):
        x = torch.randn(sum(lengths), config.dim, generator=generator, device=device, dtype=dtype)
        weights = torch.rand(sum(lengths), topk, generator=generator, device=device)
        weights /= weights.sum(-1, keepdim=True)
        indices = (torch.arange(sum(lengths), device=device)[:, None] + torch.arange(topk, device=device)) % n_experts
        token_mask = torch.full((sum(lengths),), not masked, device=device, dtype=torch.bool)
        upstream = torch.randn(x.shape, device=device, generator=generator, dtype=dtype)
        ref_x, ref_weights = x.float().clone().requires_grad_(), weights.clone().requires_grad_()
        local_x = x[start:stop].clone().requires_grad_()
        local_weights = weights[start:stop].clone().requires_grad_()
        expected = reference(ref_x, token_mask, ref_weights, indices)
        # A fully masked reference is independent of its inputs.
        expected = expected + ref_x[..., :0].sum() + ref_weights[..., :0].sum()
        actual = module(local_x, token_mask[start:stop], local_weights, indices[start:stop])
        _assert_close(actual, expected[start:stop], dtype)
        expected.backward(upstream.float())
        actual.backward(upstream[start:stop])
        _assert_close(local_x.grad, ref_x.grad[start:stop], dtype)
        _assert_close(local_weights.grad, ref_weights.grad[start:stop], dtype)
        local_norm_sq = torch.zeros((), device=device)
        expected_norm_sq = torch.zeros((), device=device)
        for name, parameter in module.named_parameters():
            if not parameter.requires_grad:
                continue
            expected_grad = reference.get_parameter(name).grad
            assert parameter.grad is not None, name
            _assert_close(parameter.grad.to_local(), expected_grad[first:last], dtype)
            local_norm_sq += parameter.grad.to_local().float().square().sum()
            expected_norm_sq += expected_grad.square().sum()
        dist.all_reduce(local_norm_sq)
        _assert_close(local_norm_sq.sqrt(), expected_norm_sq.sqrt(), dtype)
        optimizer.step()
        reference_optimizer.step()
        for name, parameter in module.named_parameters():
            _assert_close(parameter.to_local(), reference.get_parameter(name)[first:last], dtype)
        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
    dist.barrier()


def _worker(rank: int, port: int, backend_name: str, lora: bool) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    dist.init_process_group("nccl", rank=rank, world_size=2, timeout=timedelta(seconds=120))
    try:
        mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("ep",))
        # Equal control, uneven nonempty shards, and a trailing rank with no experts.
        # HybridEP rejects a one-slot communication buffer, including the
        # existing even case E=2/P=2. Empty ownership is covered by the
        # torch and DeepEP backends here.
        for n_experts in (4, 3) if backend_name == "hybridep" else (4, 3, 1):
            _case(rank, mesh, n_experts, backend_name, lora)
    finally:
        if backend_name == "hybridep":
            torch.cuda.synchronize()
            reset_hybrid_ep_buffer()
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA GPUs")
@pytest.mark.parametrize("backend_name", ["torch", "torch_mm", "deepep", "hybridep"])
@pytest.mark.parametrize("lora", [False, True])
def test_uneven_experts_training_and_checkpoint(backend_name: str, lora: bool) -> None:
    if backend_name == "deepep" and importlib.util.find_spec("deep_ep") is None:
        pytest.skip("DeepEP is not installed")
    if backend_name == "hybridep" and not HAVE_HYBRIDEP:
        pytest.skip("HybridEP is not installed")
    mp.spawn(_worker, args=(_port(), backend_name, lora), nprocs=2, join=True)
