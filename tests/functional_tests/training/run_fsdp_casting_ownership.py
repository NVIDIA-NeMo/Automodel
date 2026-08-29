#!/usr/bin/env python
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

"""FSDP/HSDP regression for ownership with bounded FP32 replication.

Usage:
    torchrun --nproc-per-node=2 tests/functional_tests/training/run_fsdp_casting_ownership.py
    torchrun --nproc-per-node=4 tests/functional_tests/training/run_fsdp_casting_ownership.py
"""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor

from nemo_automodel.components.distributed.fsdp2_extensions.compute_dtype import (
    fully_shard_with_compute_dtype_fallback,
)
from nemo_automodel.components.distributed.fsdp2_extensions.replicated import (
    make_fully_shard_with_replicated_parameter_grad_sync,
    select_small_fp32_parameters,
)
from nemo_automodel.components.optim.optimizer import AdamWConfig, FusedAdamConfig


class _GateParams(nn.Module):
    """Qwen3.5-shaped parameter holder that remains owned by its parent layer."""

    def __init__(self, hidden_size: int, conditional_parameter: bool = False):
        super().__init__()
        self.A_log = nn.Parameter(torch.linspace(-0.2, 0.2, hidden_size))
        self.dt_bias = nn.Parameter(torch.linspace(0.1, 0.3, hidden_size))
        if conditional_parameter:
            self.conditional_scale = nn.Parameter(torch.tensor(0.25))

    def forward(self, projected: torch.Tensor, use_conditional: bool = False) -> torch.Tensor:
        """Compute the precision-sensitive gate.

        Args:
            projected: Tensor of shape ``[batch, hidden]`` in BF16 or FP32.
            use_conditional: Whether to include the optional scalar parameter.

        Returns:
            Tensor of shape ``[batch, hidden]`` in FP32.
        """
        gate = -self.A_log.float().exp() * F.softplus(projected.float() + self.dt_bias)
        if use_conditional:
            gate = gate + self.conditional_scale * projected.float()
        return gate


class _Layer(nn.Module):
    """Tiny layer with ordinary weights and Qwen3.5-shaped gate parameters."""

    def __init__(
        self,
        hidden_size: int = 8,
        bulk_dtype: torch.dtype = torch.float32,
        compute_dtype: torch.dtype = torch.bfloat16,
        conditional_parameter: bool = False,
        use_conditional: bool = False,
    ):
        super().__init__()
        self.projection = nn.Linear(hidden_size, hidden_size, bias=False, dtype=bulk_dtype)
        self._fp32_params = _GateParams(hidden_size, conditional_parameter=conditional_parameter)
        self.compute_dtype = compute_dtype
        self.use_conditional = use_conditional

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the mixed-compute layer.

        Args:
            inputs: Tensor of shape ``[batch, hidden]`` in BF16.

        Returns:
            Tensor of shape ``[batch, hidden]`` in FP32.
        """
        for name, parameter in self.named_parameters():
            expected_dtype = torch.float32 if "_fp32_params" in name else self.compute_dtype
            if parameter.dtype is not expected_dtype:
                raise AssertionError(f"forward parameter {name} should be {expected_dtype}, got {parameter.dtype}")
        projected = self.projection(inputs)
        gate = self._fp32_params(projected, use_conditional=self.use_conditional)
        return projected.float() * gate


class _ReferenceLayer(_Layer):
    """Independent FP32-master reference with explicit transient BF16 copies."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the explicit-cast reference layer.

        Args:
            inputs: Tensor of shape ``[batch, hidden]`` in BF16.

        Returns:
            Tensor of shape ``[batch, hidden]`` in FP32.
        """
        projected = F.linear(inputs, self.projection.weight.to(inputs.dtype))
        gate = self._fp32_params(projected, use_conditional=self.use_conditional)
        return projected.float() * gate


class _Root(nn.Module):
    """Parameter-free root used to exercise already-owned child parameters."""

    def __init__(self, layer: nn.Module):
        super().__init__()
        self.layer = layer

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run the child layer through a separate root FSDP boundary.

        Args:
            inputs: Tensor of shape ``[batch, hidden]`` in BF16 or FP32.

        Returns:
            Tensor of shape ``[batch, hidden]`` in FP32.
        """
        return self.layer(inputs)


@dataclass(frozen=True)
class _Scenario:
    name: str
    bulk_dtype: torch.dtype
    compute_dtype: torch.dtype
    replication_limit: int
    expected_fsdp_units: int
    accumulation_steps: int = 1
    profile: bool = False
    activation_checkpointing: bool = False
    root_boundary: bool = False
    te_optimizer: bool = False
    conditional_parameter: bool = False
    rank_asymmetric_use: bool = False


def _input_for_rank(
    rank: int,
    microbatch: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(9102 + 100 * microbatch + rank)
    return torch.randn(3, 8, generator=generator, device=device, dtype=dtype)


def _run_scenario(
    scenario: _Scenario,
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    mesh,
) -> Counter:
    torch.manual_seed(2026)
    reference_layer = _ReferenceLayer(
        bulk_dtype=scenario.bulk_dtype,
        compute_dtype=scenario.compute_dtype,
        conditional_parameter=scenario.conditional_parameter,
    ).to(device=device)
    layer = _Layer(
        bulk_dtype=scenario.bulk_dtype,
        compute_dtype=scenario.compute_dtype,
        conditional_parameter=scenario.conditional_parameter,
        use_conditional=scenario.rank_asymmetric_use and rank == 0,
    ).to(device=device)
    layer.load_state_dict(reference_layer.state_dict())
    if scenario.activation_checkpointing:
        layer._fp32_params = checkpoint_wrapper(layer._fp32_params)
    reference = _Root(reference_layer) if scenario.root_boundary else reference_layer
    model = _Root(layer) if scenario.root_boundary else layer

    selection = select_small_fp32_parameters(
        model,
        name_fragments=("_fp32_params",),
        max_bytes_per_module=scenario.replication_limit,
    )
    replicated_params = selection.parameters
    fully_shard_fn = fully_shard
    if replicated_params:
        fully_shard_fn = make_fully_shard_with_replicated_parameter_grad_sync(
            model,
            replicated_params,
            mesh,
            fully_shard_fn=fully_shard_fn,
        )
    mp_policy = MixedPrecisionPolicy(
        param_dtype=scenario.compute_dtype,
        reduce_dtype=torch.float32,
        output_dtype=torch.float32,
    )
    candidates = (layer, model) if scenario.root_boundary else (model,)
    for candidate in candidates:
        fully_shard_with_compute_dtype_fallback(
            candidate,
            fp32_compute_module_names=("_fp32_params",),
            mesh=mesh,
            mp_policy=mp_policy,
            reshard_after_forward=not scenario.root_boundary or candidate is not model,
            ignored_params=set(replicated_params),
            fully_shard_fn=fully_shard_fn,
        )

    fsdp_units = [module for module in model.modules() if isinstance(module, FSDPModule)]
    if len(fsdp_units) != scenario.expected_fsdp_units:
        raise AssertionError(
            f"{scenario.name}: expected {scenario.expected_fsdp_units} FSDP unit(s), got {len(fsdp_units)}"
        )
    model.load_state_dict(model.state_dict())
    for name, parameter in model.named_parameters():
        expected_dtype = torch.float32 if "_fp32_params" in name else scenario.bulk_dtype
        if parameter.dtype is not expected_dtype:
            raise AssertionError(
                f"{scenario.name}: resident parameter {name} should be {expected_dtype}, got {parameter.dtype}"
            )
        if "_fp32_params" in name and isinstance(parameter, DTensor) is bool(replicated_params):
            expected_owner = "replicated plain tensor" if replicated_params else "sharded DTensor"
            raise AssertionError(f"{scenario.name}: sensitive parameter {name} should be a {expected_owner}")

    if scenario.te_optimizer:
        optimizer_config = FusedAdamConfig(lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1, master_weights=True)
        optimizer = optimizer_config.build(model, device_mesh=mesh)[0]
        reference_optimizer = optimizer_config.build(reference)[0]
    else:
        optimizer = AdamWConfig(lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1).build(model, device_mesh=mesh)[0]
        reference_optimizer = torch.optim.AdamW(
            reference.parameters(),
            lr=1e-3,
            betas=(0.9, 0.95),
            weight_decay=0.1,
        )

    def assert_te_master_ownership() -> None:
        if not scenario.te_optimizer:
            return
        for name, parameter in model.named_parameters():
            has_master = "master_param" in optimizer.state[parameter]
            if "_fp32_params" in name and has_master:
                raise AssertionError(f"{scenario.name}: TE created a redundant master for resident FP32 {name}")
            if "_fp32_params" not in name and not has_master:
                raise AssertionError(f"{scenario.name}: TE did not retain an FP32 master for BF16 {name}")

    def backward_all_microbatches() -> None:
        for microbatch in range(scenario.accumulation_steps):
            if scenario.accumulation_steps > 1:
                model.set_requires_gradient_sync(microbatch == scenario.accumulation_steps - 1)
            local_inputs = _input_for_rank(rank, microbatch, device, scenario.compute_dtype)
            loss = model(local_inputs).square().mean() / scenario.accumulation_steps
            loss.backward()

    nccl_kernel_counts = Counter()
    dist.barrier()
    torch.cuda.synchronize(device)
    if scenario.profile:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as profiler:
            backward_all_microbatches()
            optimizer.step()
            torch.cuda.synchronize(device)
        nccl_kernel_names = [event.key for event in profiler.events() if event.key.startswith("ncclDevKernel_")]
        for kernel_name in nccl_kernel_names:
            if "AllGather" in kernel_name:
                nccl_kernel_counts["all_gather"] += 1
            elif "ReduceScatter" in kernel_name:
                nccl_kernel_counts["reduce_scatter"] += 1
            elif "AllReduce" in kernel_name:
                nccl_kernel_counts["all_reduce"] += 1
            else:
                nccl_kernel_counts["other"] += 1
    else:
        backward_all_microbatches()
        optimizer.step()

    assert_te_master_ownership()

    for microbatch in range(scenario.accumulation_steps):
        if scenario.rank_asymmetric_use:
            reference_loss = torch.zeros((), device=device)
            for input_rank in range(world_size):
                reference_layer.use_conditional = input_rank == 0
                inputs = _input_for_rank(input_rank, microbatch, device, scenario.compute_dtype)
                reference_loss = reference_loss + reference(inputs).square().mean() / world_size
            (reference_loss / scenario.accumulation_steps).backward()
        else:
            global_inputs = torch.cat(
                [
                    _input_for_rank(input_rank, microbatch, device, scenario.compute_dtype)
                    for input_rank in range(world_size)
                ]
            )
            (reference(global_inputs).square().mean() / scenario.accumulation_steps).backward()
    reference_optimizer.step()

    # Optimizer serialization must retain the same state ownership and remain
    # loadable after an actual distributed step.
    optimizer.load_state_dict(optimizer.state_dict())
    assert_te_master_ownership()

    reference_params = dict(reference.named_parameters())
    shard_world_size = mesh["dp_shard"].size() if world_size == 4 else world_size
    shard_rank = mesh.get_local_rank(mesh_dim="dp_shard") if world_size == 4 else rank
    for name, parameter in model.named_parameters():
        reference_name = name.replace("._checkpoint_wrapped_module", "")
        reference_parameter = reference_params[reference_name]
        tolerance = 2e-3 if reference_parameter.dtype is torch.bfloat16 else 5e-5
        if isinstance(parameter, DTensor):
            local_reference = reference_parameter.chunk(shard_world_size, dim=0)[shard_rank]
            torch.testing.assert_close(parameter.to_local(), local_reference, rtol=0, atol=tolerance)
        else:
            torch.testing.assert_close(parameter, reference_parameter, rtol=0, atol=tolerance)
    return nccl_kernel_counts


def main() -> None:
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available")
        return

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size not in (2, 4):
        raise AssertionError(f"expected world_size=2 (FSDP) or 4 (HSDP), got {world_size}")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if world_size == 4:
        mesh = init_device_mesh(
            "cuda",
            (2, 2),
            mesh_dim_names=("dp_replicate", "dp_shard"),
        )
    else:
        mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp",))

    scenarios = [
        _Scenario("D1-replicated", torch.float32, torch.bfloat16, 64, 1, profile=True),
        _Scenario("D1-single-owner", torch.float32, torch.bfloat16, 0, 1),
        _Scenario("D2-uniform-fp32", torch.float32, torch.float32, 0, 1),
        _Scenario("D4-replicated", torch.bfloat16, torch.bfloat16, 64, 1),
        _Scenario("D4-dtype-split", torch.bfloat16, torch.bfloat16, 0, 2),
        _Scenario("D1-accumulation", torch.float32, torch.bfloat16, 64, 1, accumulation_steps=2),
        _Scenario("D1-activation-checkpoint", torch.float32, torch.bfloat16, 64, 1, activation_checkpointing=True),
        _Scenario("F5-root-after-child", torch.float32, torch.bfloat16, 64, 2, root_boundary=True),
        _Scenario("F16-globally-unused", torch.float32, torch.bfloat16, 128, 1, conditional_parameter=True),
        _Scenario(
            "F16-rank-asymmetric-use",
            torch.float32,
            torch.bfloat16,
            128,
            1,
            conditional_parameter=True,
            rank_asymmetric_use=True,
        ),
    ]
    if os.getenv("RUN_TE_FSDP_CASE") == "1":
        scenarios.append(_Scenario("D3-te-master", torch.bfloat16, torch.bfloat16, 64, 1, te_optimizer=True))
    profiled_counts = Counter()
    for scenario in scenarios:
        counts = _run_scenario(scenario, rank=rank, world_size=world_size, device=device, mesh=mesh)
        if scenario.profile:
            profiled_counts = counts

    expected_nccl_kernel_counts = (
        Counter({"all_gather": 2, "reduce_scatter": 1, "all_reduce": 3})
        if world_size == 4
        else Counter({"all_gather": 2, "reduce_scatter": 1, "all_reduce": 1})
    )
    if profiled_counts != expected_nccl_kernel_counts:
        raise AssertionError(
            f"expected one FSDP unit to launch {dict(expected_nccl_kernel_counts)}, got {dict(profiled_counts)}"
        )

    if rank == 0:
        mode = "HSDP" if world_size == 4 else "FSDP"
        print(
            f"PASS: {len(scenarios)} {mode} dtype/ownership cases, "
            f"{sum(expected_nccl_kernel_counts.values())} profiled NCCL kernels, optimizer parity"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
