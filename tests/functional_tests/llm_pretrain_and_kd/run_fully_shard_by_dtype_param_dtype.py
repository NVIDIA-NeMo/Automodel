#!/usr/bin/env python
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

"""Functional test: fp32 master weights + bf16 FSDP2 policy with per-parameter fp32 compute.

Each KDA-style block (bf16-compute projections, bare fp32 ``A_log``/``dt_bias``)
is one FSDP unit. The sharded model must match an unsharded reference that
emulates the same mixed precision by hand.

Usage:
    torchrun --nproc_per_node=2 tests/functional_tests/llm_pretrain_and_kd/run_fully_shard_by_dtype_param_dtype.py
"""

from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl, checkpoint_wrapper
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor

from nemo_automodel.components.distributed.parallelizer_utils import fully_shard_by_dtype

HIDDEN = 16
NUM_BLOCKS = 2
FP32_TOKENS = ("A_log", "dt_bias")
POLICY = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, output_dtype=torch.bfloat16)


class KDABlock(nn.Module):
    """Kimi-Delta-Attention-style block: bf16 projections and fp32 decay parameters on the block itself."""

    def __init__(self, check_dtypes: bool):
        super().__init__()
        self.check_dtypes = check_dtypes
        self.in_proj = nn.Linear(HIDDEN, HIDDEN, bias=False, dtype=torch.float32)
        self.out_proj = nn.Linear(HIDDEN, HIDDEN, bias=False, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.rand(HIDDEN, dtype=torch.float32).log())
        self.dt_bias = nn.Parameter(torch.rand(HIDDEN, dtype=torch.float32))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Map ``[batch, HIDDEN]`` bf16 activations to ``[batch, HIDDEN]`` bf16 activations."""
        if self.check_dtypes:
            _expect(self.in_proj.weight.dtype == torch.bfloat16, f"in_proj computes {self.in_proj.weight.dtype}")
            _expect(self.out_proj.weight.dtype == torch.bfloat16, f"out_proj computes {self.out_proj.weight.dtype}")
            _expect(self.A_log.dtype == torch.float32, f"A_log computes {self.A_log.dtype}")
            _expect(self.dt_bias.dtype == torch.float32, f"dt_bias computes {self.dt_bias.dtype}")
            _expect(hidden.dtype == torch.bfloat16, f"block input is {hidden.dtype}")
        # The unsharded reference holds fp32 masters; casting mirrors FSDP's bf16 all-gather.
        h = F.linear(hidden.to(torch.bfloat16), self.in_proj.weight.to(torch.bfloat16))
        # Decay gate owns its fp32 casts; FSDP only casts the block inputs.
        gate = torch.exp(-torch.exp(self.A_log) * F.softplus(h.float() + self.dt_bias))
        h = (h.float() * gate).to(torch.bfloat16)
        return F.linear(h, self.out_proj.weight.to(torch.bfloat16))


class TinyModel(nn.Module):
    def __init__(self, check_dtypes: bool):
        super().__init__()
        self.layers = nn.ModuleList([KDABlock(check_dtypes) for _ in range(NUM_BLOCKS)])

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _local(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _build_model(device: torch.device, check_dtypes: bool) -> TinyModel:
    torch.manual_seed(1234)
    return TinyModel(check_dtypes).to(device)


def _shard(
    model: TinyModel, mesh: DeviceMesh, *, reshard_after_forward: bool | None, activation_checkpoint: bool
) -> None:
    for index, block in enumerate(model.layers):
        if activation_checkpoint:
            block = checkpoint_wrapper(block, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
            model.layers[index] = block
        fully_shard_by_dtype(
            block,
            mesh=mesh,
            mp_policy=POLICY,
            offload_policy=None,
            fp32_compute_module_names=FP32_TOKENS,
            reshard_after_forward=reshard_after_forward,
        )
    fully_shard(model, mesh=mesh, mp_policy=POLICY)
    for block in model.layers:
        _expect(isinstance(block, FSDPModule), "each block must be an FSDP unit")
        nested = [name for name, child in block.named_modules() if child is not block and isinstance(child, FSDPModule)]
        _expect(not nested, f"nested FSDP units inside a block: {nested}")


def _batches(device: torch.device) -> list[list[torch.Tensor]]:
    """Three steps of data, identical on every rank; step 2 accumulates two microbatches."""
    generator = torch.Generator(device="cpu").manual_seed(42)

    def batch() -> torch.Tensor:
        return torch.randn(8, HIDDEN, generator=generator).to(device=device, dtype=torch.bfloat16)

    return [[batch()], [batch(), batch()], [batch()]]


def _is_fp32_param(name: str) -> bool:
    return any(token in name for token in FP32_TOKENS)


def _round_bf16_compute_grads(model: nn.Module) -> None:
    """Emulate FSDP2 microbatch accumulation, which sums bf16-compute gradients in bf16."""
    for name, param in model.named_parameters():
        if not _is_fp32_param(name):
            param.grad = param.grad.to(torch.bfloat16).to(torch.float32)


def _train(model: nn.Module, steps: list[list[torch.Tensor]], sharded: bool) -> list[float]:
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    losses: list[float] = []
    for microbatches in steps:
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for index, inputs in enumerate(microbatches):
            if sharded:
                model.set_requires_gradient_sync(index == len(microbatches) - 1)
            output = model(inputs)
            if sharded:
                _expect(output.dtype == torch.bfloat16, f"block output is {output.dtype}")
            loss = output.float().square().mean() / len(microbatches)
            loss.backward()
            step_loss += loss.item()
            if not sharded and len(microbatches) > 1:
                _round_bf16_compute_grads(model)
        if sharded:
            _check_grads(model)
        optimizer.step()
        losses.append(step_loss)
    return losses


def _check_grads(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        _expect(param.grad is not None, f"missing gradient for {name}")
        grad = _local(param.grad)
        _expect(bool(torch.isfinite(grad).all()), f"non-finite gradient for {name}")
        _expect(grad.dtype == torch.float32, f"{name} gradient is {grad.dtype}")


def _assert_parity(sharded: nn.Module, reference: nn.Module, variant: str) -> None:
    reference_params = dict(reference.named_parameters())
    for name, param in sharded.named_parameters():
        full = param.full_tensor() if isinstance(param, DTensor) else param
        canonical = name.replace("_checkpoint_wrapped_module.", "")
        torch.testing.assert_close(full, reference_params[canonical], rtol=1e-4, atol=1e-5, msg=f"{variant}: {name}")


def _run_variant(
    name: str,
    mesh: DeviceMesh,
    device: torch.device,
    *,
    reshard_after_forward: bool | None = None,
    activation_checkpoint: bool = False,
) -> None:
    reference = _build_model(device, check_dtypes=False)
    model = _build_model(device, check_dtypes=True)
    _shard(model, mesh, reshard_after_forward=reshard_after_forward, activation_checkpoint=activation_checkpoint)
    for param in model.parameters():
        _expect(param.dtype == torch.float32, "sharded storage must stay fp32 (master weights)")
    losses = _train(model, _batches(device), sharded=True)
    reference_losses = _train(reference, _batches(device), sharded=False)
    torch.testing.assert_close(losses, reference_losses, rtol=1e-4, atol=1e-5, msg=f"{name}: losses")
    _assert_parity(model, reference, name)
    if dist.get_rank() == 0:
        print(f"PASS: {name} matches the unsharded reference (losses {losses})")


def _run_negative(mesh: DeviceMesh, device: torch.device) -> None:
    model = _build_model(device, check_dtypes=False).to(torch.bfloat16)
    try:
        fully_shard_by_dtype(
            model.layers[0], mesh=mesh, mp_policy=POLICY, offload_policy=None, fp32_compute_module_names=FP32_TOKENS
        )
    except ValueError as error:
        _expect("model.dtype" in str(error), f"unexpected error text: {error}")
    else:
        raise AssertionError("bf16 storage of pinned fp32 parameters must be rejected")
    if dist.get_rank() == 0:
        print("PASS: bf16-storage pinned parameters raise ValueError")


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available", file=sys.stderr)
        return 0
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    if dist.get_world_size() != 2:
        print(f"ERROR: This test requires world_size=2, got {dist.get_world_size()}", file=sys.stderr)
        return 1
    mesh = init_device_mesh(device_type="cuda", mesh_shape=(2,), mesh_dim_names=("dp",))

    _run_variant("default", mesh, device)
    _run_variant("reshard_after_forward=False", mesh, device, reshard_after_forward=False)
    _run_variant("checkpoint_wrapper", mesh, device, activation_checkpoint=True)
    _run_negative(mesh, device)
    torch.cuda.synchronize()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
