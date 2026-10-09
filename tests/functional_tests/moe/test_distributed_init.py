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

"""CUDA initialization must distinguish shards while preserving replicas.

PR tests use two GPUs. Set AUTOMODEL_INIT_TEST_WORLD_SIZE=4 for the real 2x2
EP/FSDP and replicate/shard compositions during scheduled or local validation.
CPU DTensor RNG does not exercise the CUDA placement-offset implementation.
"""

import os
from datetime import timedelta
from pathlib import Path
from typing import Literal, cast

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Placement, Replicate, Shard, distribute_tensor

from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.experts import GroupedExperts
from nemo_automodel.components.moe.layers import MLP, Gate

_Family = Literal["gate", "mlp", "experts"]


def _make_module(family: _Family, dtype: torch.dtype) -> Gate | MLP | GroupedExperts:
    config = MoEConfig(
        n_routed_experts=8,
        n_shared_experts=1,
        n_activated_experts=2,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.001,
        aux_loss_coeff=0.0,
        score_func="sigmoid",
        route_scale=1.0,
        dim=64,
        inter_dim=128,
        moe_inter_dim=128,
        norm_topk_prob=True,
        router_bias=True,
        expert_bias=True,
        dtype=dtype,
    )
    if family == "gate":
        return Gate(config)
    if family == "mlp":
        return MLP(config.dim, config.inter_dim, "torch", dtype=dtype)
    return GroupedExperts(config)


def _local(parameter: torch.Tensor) -> torch.Tensor:
    """Observe local storage without gathering or changing a distributed tensor.

    Args:
        parameter: Tensor of arbitrary shape, or DTensor with global projection
            shape [output, input], [experts, input, fused_gate_up], or
            [experts, intermediate, output]. Biases may have one or two axes.

    Returns:
        Detached per-rank shard, retaining the existing local shape and storage.
    """
    return parameter.to_local().detach() if isinstance(parameter, DTensor) else parameter.detach()


def _check_init(rank: int, world_size: int, rendezvous: str, family: _Family) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=120),
    )
    errors: list[str] = []
    try:
        layouts: list[tuple[str, tuple[int, ...], tuple[Placement, ...]]] = [
            ("replicate", (world_size,), (Replicate(),)),
            ("shard_rows", (world_size,), (Shard(0),)),
            ("expert_fsdp", (world_size // 2, 2), (Shard(1), Shard(0))),
            ("replicate_fsdp", (world_size // 2, 2), (Replicate(), Shard(0))),
        ]
        # With two GPUs, also exercise the first axis independently. With four,
        # the layouts above already exercise both nontrivial axes together.
        if world_size == 2:
            layouts.extend(
                [
                    ("feature_shard", (2, 1), (Shard(1), Shard(0))),
                    ("replica_axis", (2, 1), (Replicate(), Shard(0))),
                ]
            )
        for dtype in (torch.float32, torch.bfloat16):
            for label, mesh_shape, placements in layouts:
                mesh = init_device_mesh("cuda", mesh_shape)
                with torch.device("meta"):
                    module = _make_module(family, dtype)
                for name, parameter in list(module.named_parameters()):
                    # A row bias is replicated across feature-sharding axes.
                    parameter_placements = [
                        Replicate() if isinstance(p, Shard) and p.dim >= parameter.ndim else p for p in placements
                    ]
                    owner_name, _, parameter_name = name.rpartition(".")
                    owner = module.get_submodule(owner_name) if owner_name else module
                    setattr(
                        owner, parameter_name, nn.Parameter(distribute_tensor(parameter, mesh, parameter_placements))
                    )
                module.to_empty(device=torch.device("cuda", rank))
                with torch.no_grad():
                    for parameter in module.parameters():
                        _local(parameter).fill_(float("nan"))
                    for buffer in module.buffers():
                        buffer.fill_(float("nan"))
                originals = {
                    name: (parameter, _local(parameter).data_ptr()) for name, parameter in module.named_parameters()
                }
                # A common seed is required by DTensor. This deliberately does
                # not claim to fix unrelated ranked-seed/repeated-build policy.
                torch.manual_seed(1234)
                with torch.no_grad():
                    module.init_weights(torch.device("cuda", rank), init_std=0.02)
                first = {name: _local(parameter).clone() for name, parameter in module.named_parameters()}
                for name, parameter in module.named_parameters():
                    context = f"{family}/{dtype}/{label}/{name}"
                    original, storage = originals[name]
                    if parameter is not original or _local(parameter).data_ptr() != storage:
                        errors.append(f"{context}: initialization replaced parameter or storage")
                    if not bool(torch.isfinite(_local(parameter)).all()):
                        errors.append(f"{context}: nonfinite initialized values")
                    if "bias" in name:
                        if bool(torch.count_nonzero(_local(parameter))):
                            errors.append(f"{context}: constant bias was not zero")
                        continue
                    local = _local(parameter).contiguous()
                    gathered = [torch.empty_like(local) for _ in range(world_size)]
                    dist.all_gather(gathered, local)
                    coordinates = [
                        (other,) if len(mesh_shape) == 1 else (other // mesh_shape[1], other % mesh_shape[1])
                        for other in range(world_size)
                    ]
                    shard_keys = [
                        tuple(
                            coordinate[axis]
                            for axis, placement in enumerate(placements)
                            if isinstance(placement, Shard)
                        )
                        for coordinate in coordinates
                    ]
                    for left in range(world_size):
                        for right in range(left + 1, world_size):
                            equal = torch.equal(gathered[left], gathered[right])
                            same_shard = shard_keys[left] == shard_keys[right]
                            if equal != same_shard:
                                errors.append(
                                    f"{context}: ranks {left},{right} "
                                    f"{'duplicate different shards' if equal else 'disagree on replicated shard'}"
                                )
                    full = cast(DTensor, parameter).full_tensor().detach().float()
                    if not 0.01 < float(full.std()) < 0.03 or abs(float(full.mean())) > 0.005:
                        errors.append(f"{context}: values do not follow configured normal initialization")
                    if family == "gate":
                        # Full rows correspond to distinct learned expert scores.
                        # This assertion catches repeated router rows independently
                        # of the pairwise local-shard comparison above.
                        rows = full.cpu().contiguous().numpy()
                        if len({row.tobytes() for row in rows}) != full.shape[0]:
                            errors.append(f"{context}: reconstructed global router has duplicate full rows")
                if isinstance(module, Gate):
                    bias = module.e_score_correction_bias
                    if bias is None or bias.dtype != torch.float32 or bool(torch.count_nonzero(bias)):
                        errors.append(f"{family}/{dtype}/{label}: correction bias must remain zero FP32")
                torch.manual_seed(1234)
                with torch.no_grad():
                    module.init_weights(torch.device("cuda", rank), init_std=0.02)
                for name, parameter in module.named_parameters():
                    if not torch.equal(first[name], _local(parameter)):
                        errors.append(f"{family}/{dtype}/{label}/{name}: same-seed initialization is not repeatable")
                torch.manual_seed(1235)
                with torch.no_grad():
                    module.init_weights(torch.device("cuda", rank), init_std=0.02)
                for name, parameter in module.named_parameters():
                    if "bias" not in name and torch.equal(first[name], _local(parameter)):
                        errors.append(f"{family}/{dtype}/{label}/{name}: changed seed did not change random weights")
        # Complete identical collectives on every rank before reporting failures.
        assert not errors, "\n".join(errors)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="DTensor random initialization requires at least two CUDA GPUs"
)
@pytest.mark.parametrize("family", ["gate", "mlp", "experts"])
def test_distributed_moe_init_preserves_shards_and_replicas(tmp_path: Path, family: _Family) -> None:
    """Validate complete initialized shards and RNG reproducibility on CUDA."""
    world_size = int(os.environ.get("AUTOMODEL_INIT_TEST_WORLD_SIZE", "2"))
    if world_size not in (2, 4):
        raise ValueError("AUTOMODEL_INIT_TEST_WORLD_SIZE must be 2 (PR cap) or 4 (composed topology)")
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"Requested {world_size}-GPU composed topology requires {world_size} visible CUDA devices")
    mp.spawn(_check_init, args=(world_size, str(tmp_path / "init"), family), nprocs=world_size, join=True)
