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

"""Two-GPU FSDP2 working-weight rounding against an unrounded native AdamW oracle.

This is a distributed optimizer/forward-backward component test, not a model
checkpoint-loading test. Native recipe checkpoint coverage is separate.
"""

from __future__ import annotations

import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor, Shard

from nemo_automodel.components.optim.optimizer import AdamWConfig

ParameterGroup = dict[str, list[nn.Parameter] | bool]


class _TinyModel(nn.Module):
    """Dense fixture with protected, delayed-gradient, and never-used parameters."""

    def __init__(self) -> None:
        super().__init__()
        self.rounded = nn.Linear(16, 16, bias=False)
        self.delayed = nn.Linear(16, 16, bias=False)
        self.protected = nn.Linear(16, 16, bias=False)
        self.unused = nn.Parameter(torch.empty(16, 16))
        with torch.no_grad():
            for index, parameter in enumerate(self.parameters()):
                values = torch.arange(parameter.numel(), dtype=torch.float32).reshape(parameter.shape)
                parameter.copy_(((values - 127) / (512 + index * 64)).bfloat16().float())

    def forward(self, value: torch.Tensor, use_rounded: bool, use_delayed: bool) -> torch.Tensor:
        """Run dense BF16 compute on the active parameter subsets.

        Args:
            value: Input tensor of shape [batch, input_features=16].
            use_rounded: Whether the principal rounded weight receives a gradient.
            use_delayed: Whether the delayed rounded weight receives a gradient.

        Returns:
            Tensor of shape [batch, output_features=16] on the input device.
        """
        output = self.protected(value)
        if use_rounded:
            output = output + self.rounded(value)
        if use_delayed:
            output = output + self.delayed(value)
        return output


def _make_model(mesh: DeviceMesh) -> _TinyModel:
    """Build an actual FSDP2 model with FP32 row shards and BF16 compute."""
    model = _TinyModel().to(torch.device("cuda", torch.cuda.current_device()))
    fully_shard(
        model,
        mesh=mesh,
        mp_policy=MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            output_dtype=torch.bfloat16,
            cast_forward_inputs=True,
        ),
        reshard_after_forward=True,
    )
    assert isinstance(model, FSDPModule)
    assert any(
        isinstance(parameter, DTensor) and any(isinstance(p, Shard) for p in parameter.placements)
        for parameter in model.parameters()
    )
    return model


def _parameters(model: _TinyModel) -> list[nn.Parameter]:
    """Return stable optimizer ordering, including the protected last group."""
    return [model.rounded.weight, model.delayed.weight, model.unused, model.protected.weight]


def _groups(model: _TinyModel) -> list[ParameterGroup]:
    """Preserve explicit quantized/protected group membership."""
    parameters = _parameters(model)
    return [{"params": parameters[:3], "quantize": True}, {"params": parameters[3:], "quantize": False}]


def _local(value: torch.Tensor) -> torch.Tensor:
    """Return a read-only local view of a global or rank-local tensor.

    Args:
        value: Tensor of arbitrary shape [...]; a DTensor may use row-Shard placements.

    Returns:
        Aliasing local tensor with the rank-local shard shape and original dtype/device.
    """
    return value.to_local() if isinstance(value, DTensor) else value


def _assert_bits(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Compare exact bytes, including signed-zero bits, without mutating inputs.

    Args:
        actual: Tensor of arbitrary shape [...], global DTensor or local.
        expected: Tensor whose local shape and dtype match the actual local tensor.
    """
    left, right = _local(actual).detach(), _local(expected).detach()
    assert left.shape == right.shape and left.dtype == right.dtype
    assert torch.equal(
        left.contiguous().reshape(-1).view(torch.uint8), right.contiguous().reshape(-1).view(torch.uint8)
    )


def _backward(model: _TinyModel, step: int, rank: int) -> None:
    """Execute native FSDP2 forward/backward with deterministic rank-dependent input."""
    inputs = torch.arange(64, device="cuda", dtype=torch.float32).reshape(4, 16) / 64
    inputs = inputs + rank / 32 + step / 128
    output = model(inputs, use_rounded=step != 2, use_delayed=step in (1, 2, 3, 5))
    assert output.dtype == torch.bfloat16 and torch.isfinite(output).all()
    coefficients = torch.arange(64, device="cuda", dtype=torch.float32).reshape(4, 16) / 97 + 0.1
    loss = (output.float() * coefficients).square().mean()
    loss.backward()


def _assert_sr_neighbors(actual: torch.Tensor, master: torch.Tensor) -> None:
    """Check SR support using adjacent representable BF16 values, independently of bit-noise code.

    Args:
        actual: Local FP32 working weights of arbitrary shape [...].
        master: Local FP32 master values of matching shape, finite and away from BF16 overflow.
    """
    nearest = master.to(torch.bfloat16)
    lower = torch.where(
        nearest.float() <= master,
        nearest,
        torch.nextafter(nearest, torch.full_like(nearest, -torch.inf)),
    ).float()
    upper = torch.where(
        nearest.float() >= master,
        nearest,
        torch.nextafter(nearest, torch.full_like(nearest, torch.inf)),
    ).float()
    assert torch.all((actual == lower) | (actual == upper))


def _run_oracle_case(rounding: str, seed: int, mesh: DeviceMesh, rank: int) -> None:
    """Compare real FSDP2 gradients/updates to independent local native master AdamW."""
    model = _make_model(mesh)
    parameters = _parameters(model)
    optimizer = AdamWConfig(
        lr=0.0013,
        betas=(0.8, 0.97),
        eps=1e-8,
        weight_decay=0.1,
        weight_rounding=rounding,
        rounding_seed=seed,
    ).build_from_param_groups(_groups(model), device_mesh=mesh)
    assert type(optimizer) is torch.optim.AdamW
    reference = [nn.Parameter(_local(parameter).detach().clone()) for parameter in parameters]
    oracle = torch.optim.AdamW(
        reference,
        lr=0.0013,
        betas=(0.8, 0.97),
        eps=1e-8,
        weight_decay=0.1,
        foreach=False,
        fused=False,
    )
    for step in range(6):
        optimizer.zero_grad(set_to_none=True)
        oracle.zero_grad(set_to_none=True)
        _backward(model, step, rank)
        parameters = _parameters(model)
        for parameter, master in zip(parameters, reference, strict=True):
            master.grad = None if parameter.grad is None else _local(parameter.grad).detach().clone()
        learning_rate = 0.0 if step in (0, 3) else 0.0013
        for group in optimizer.param_groups + oracle.param_groups:
            group["lr"] = learning_rate
        identity = [(id(parameter), _local(parameter).data_ptr()) for parameter in parameters]
        cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
        optimizer.step()
        _assert_bits(torch.get_rng_state(), cpu_rng)
        _assert_bits(torch.cuda.get_rng_state(), cuda_rng)
        oracle.step()
        assert [(id(parameter), _local(parameter).data_ptr()) for parameter in _parameters(model)] == identity
        assert [group["quantization_step"] for group in optimizer.param_groups] == [step + 1, step + 1]
        for index, (parameter, master) in enumerate(zip(parameters, reference, strict=True)):
            actual_state, expected_state = optimizer.state.get(parameter, {}), oracle.state.get(master, {})
            if not expected_state:
                assert not actual_state
                continue
            for key in ("step", "exp_avg", "exp_avg_sq"):
                _assert_bits(actual_state[key], expected_state[key])
            if index == 3:
                assert "master" not in actual_state
                _assert_bits(parameter, master)
            else:
                _assert_bits(actual_state["master"], master)
                _assert_bits(_local(parameter), _local(parameter).bfloat16().float())
                if rounding == "nearest":
                    _assert_bits(_local(parameter), master.bfloat16().float())
                else:
                    _assert_sr_neighbors(_local(parameter), master)
    assert parameters[2] not in optimizer.state


def _worker(rank: int, rendezvous: str, rounding: str, seed: int) -> None:
    """Run a GPU-only two-rank test; CPU DTensor unit coverage is separate."""
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(minutes=3),
    )
    try:
        mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("dp",))
        _run_oracle_case(rounding, seed, mesh, rank)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("rounding", ["nearest", "stochastic"])
@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="actual FSDP2 requires two CUDA GPUs")
def test_native_adamw_rounding_two_rank_fsdp(rounding: str, seed: int) -> None:
    """Exercise actual FSDP2, not a mocked mesh or custom checkpoint loader."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    mp.spawn(_worker, args=(f"tcp://127.0.0.1:{port}", rounding, seed), nprocs=2, join=True)
