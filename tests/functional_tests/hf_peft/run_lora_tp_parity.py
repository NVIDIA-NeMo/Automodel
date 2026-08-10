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

"""Dense-vs-TP correctness checks for LoRA tensor-parallel placements.

Run with ``torchrun --standalone --nproc-per-node=2``.  The test intentionally
uses Adam because replicated LoRA parameters receive Partial gradients: output
and gradient parity alone would not catch an optimizer that applied unreduced
rank-local gradients to replicated parameters.
"""

from __future__ import annotations

import copy

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard
from torch.distributed.tensor.parallel import parallelize_module

from nemo_automodel.components._peft.lora import LinearLoRA
from nemo_automodel.components.distributed.parallel_styles import ColwiseParallelLora, RowwiseParallelLora


def _full_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    low_precision = actual.dtype in (torch.bfloat16, torch.float16)
    atol, rtol = (2e-2, 2e-2) if low_precision else (2e-4, 2e-3)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol, msg=lambda msg: f"{label}: {msg}")


def _assert_replicated_rank_parity(tensor: DTensor, label: str) -> None:
    """Catch silent rank divergence hidden by a Replicate placement."""
    assert tensor.placements == (Replicate(),), f"{label}: expected Replicate, got {tensor.placements}"
    local = tensor.to_local().detach()
    gathered = [torch.empty_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local)
    for rank, rank_tensor in enumerate(gathered[1:], start=1):
        _assert_close(rank_tensor, gathered[0], f"{label}: rank {rank} differs from rank 0")


def _make_lora(device: torch.device, dtype: torch.dtype) -> LinearLoRA:
    base = nn.Linear(32, 48, bias=False, device=device, dtype=dtype)
    module = LinearLoRA(base, dim=8, alpha=8, use_memory_efficient_lora=True).to(device)
    module.weight.requires_grad_(False)
    with torch.no_grad():
        module.lora_B.weight.normal_(mean=0.0, std=0.1)
    return module


def _assert_optimizer_state_parity(
    optimizer: torch.optim.Adam,
    reference_optimizer: torch.optim.Adam,
    module: LinearLoRA,
    reference: LinearLoRA,
) -> None:
    for adapter_name in ("lora_A", "lora_B"):
        param = getattr(module, adapter_name).weight
        reference_param = getattr(reference, adapter_name).weight
        state = optimizer.state[param]
        reference_state = reference_optimizer.state[reference_param]
        for state_name in ("exp_avg", "exp_avg_sq"):
            distributed_state = state[state_name]
            _assert_close(
                _full_tensor(distributed_state),
                reference_state[state_name],
                f"{adapter_name}.{state_name}",
            )
            if param.placements == (Replicate(),):
                assert isinstance(distributed_state, DTensor)
                _assert_replicated_rank_parity(distributed_state, f"{adapter_name}.{state_name}")


def _run_plan(plan_name: str, device: torch.device, dtype: torch.dtype) -> None:
    torch.manual_seed(1234)
    module = _make_lora(device, dtype)
    reference = copy.deepcopy(module)

    mesh = init_device_mesh(device.type, (dist.get_world_size(),), mesh_dim_names=("tp",))
    if plan_name == "colwise":
        plan = ColwiseParallelLora(input_layouts=Replicate(), use_local_output=False)
        expected_param_placements = {"lora_A": (Replicate(),), "lora_B": (Shard(0),)}
        expected_grad_placements = {"lora_A": (Partial(),), "lora_B": (Shard(0),)}
    elif plan_name == "rowwise":
        plan = RowwiseParallelLora(input_layouts=Replicate(), use_local_output=False)
        expected_param_placements = {"lora_A": (Shard(1),), "lora_B": (Replicate(),)}
        expected_grad_placements = {"lora_A": (Shard(1),), "lora_B": (Partial(),)}
    else:
        raise ValueError(f"unknown plan: {plan_name}")

    module = parallelize_module(module, mesh, {"": plan})
    optimizer = torch.optim.Adam((module.lora_A.weight, module.lora_B.weight), lr=1e-3)
    reference_optimizer = torch.optim.Adam((reference.lora_A.weight, reference.lora_B.weight), lr=1e-3)

    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)

        # Exercise the production-risky case: Partial adapter gradients are
        # accumulated across several microbatches before Adam consumes them.
        for microbatch in range(3):
            torch.manual_seed(4321 + 10 * step + microbatch)
            x = torch.randn(3, 5, 32, device=device, dtype=dtype, requires_grad=True)
            target = torch.randn(3, 5, 48, device=device, dtype=dtype)
            reference_x = x.detach().clone().requires_grad_(True)

            output = module(x)
            reference_output = reference(reference_x)
            full_output = output.full_tensor()
            label = f"{plan_name} {dtype} step {step} microbatch {microbatch}"
            _assert_close(full_output, reference_output, f"{label} output")

            F.mse_loss(full_output, target).backward()
            F.mse_loss(reference_output, target).backward()
            _assert_close(_full_tensor(x.grad), reference_x.grad, f"{label} input grad")

        for adapter_name in ("lora_A", "lora_B"):
            param = getattr(module, adapter_name).weight
            reference_param = getattr(reference, adapter_name).weight
            assert param.placements == expected_param_placements[adapter_name]
            assert param.grad.placements == expected_grad_placements[adapter_name]
            if step == 0:
                _assert_close(
                    _full_tensor(param.grad),
                    reference_param.grad,
                    f"{plan_name} step {step} {adapter_name} grad",
                )

        # On step 1, deliberately do not materialize either adapter gradient
        # before Adam.  This ensures the optimizer, rather than the test's
        # full_tensor() comparison, consumes and resolves the Partial gradient.

        optimizer.step()
        reference_optimizer.step()

        for adapter_name in ("lora_A", "lora_B"):
            param = getattr(module, adapter_name).weight
            reference_param = getattr(reference, adapter_name).weight
            _assert_close(
                _full_tensor(param),
                reference_param,
                f"{plan_name} step {step} {adapter_name} parameter",
            )
            if param.placements == (Replicate(),):
                _assert_replicated_rank_parity(param, f"{plan_name} step {step} {adapter_name}")

        _assert_optimizer_state_parity(optimizer, reference_optimizer, module, reference)


def main() -> None:
    dist.init_process_group(backend="nccl")
    if dist.get_world_size() != 2:
        raise RuntimeError(f"LoRA TP parity requires exactly 2 ranks, got {dist.get_world_size()}")

    device = torch.device("cuda", int(torch.distributed.get_rank()))
    torch.cuda.set_device(device)
    try:
        for dtype in (torch.float32, torch.bfloat16):
            _run_plan("colwise", device, dtype)
            _run_plan("rowwise", device, dtype)
        if dist.get_rank() == 0:
            print("LoRA TP dense parity passed for outputs, gradients, Adam states, and parameters")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
