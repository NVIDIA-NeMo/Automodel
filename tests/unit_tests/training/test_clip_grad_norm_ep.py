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

"""Grad clipping must compute ONE global norm with TE grouped experts under EP.

TE grouped expert parameters do not carry the EP axis on their own mesh: with
ep_size equal to the non-pp world they stay plain tensors, and with
ep_shard > 1 they are DTensors sharded only over the ep_shard mesh. Without an
extra reduction over the EP axis every EP group computes a different "global"
norm and clips its shards of the same logical dense FSDP parameter by a
different coefficient. These scenarios drive the real
``scale_grads_and_clip_grad_norm`` across gloo ranks and assert every rank
agrees on the correct global norm and applies the identical clip everywhere.

Expert modules are identified structurally through the
``_nemo_ep_local_expert_params`` marker that ``GroupedExpertsTE`` stamps at
construction, so identification is independent of parameter names, of the MoE
block's attribute name, and of this step's gradient state (collective
participation must be rank-uniform).

Every scenario shares a single ``mp.spawn`` of ``_WORLD`` ranks. Process
start-up dominates the cost of a spawned test (~10 s each, mostly re-importing
torch), so one process group running all scenarios in sequence keeps this file
affordable for the CPU unit-test budget.
"""

import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import DTensor, Shard

pytestmark = pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is required")

MAX_NORM = 1.0
_WORLD = 4
# Bounds a deadlock: without it a regression that hangs the collectives would
# burn the whole job timeout instead of reporting a failed test.
_JOIN_TIMEOUT_S = 300.0


def _expert_model(expert: nn.Parameter, attribute: str = "mlp") -> nn.Module:
    """Build a model whose expert module carries the structural EP marker.

    Args:
        expert: Expert weight parameter of shape ``[out_features, in_features]``
            (a plain tensor or a DTensor sharded over ep_shard only); attached
            as ``<attribute>.experts.weight0``.
        attribute: Attribute name the MoE block hangs off the layer (models use
            both ``mlp`` and ``moe``); identification must not depend on it.
    """
    experts = nn.Module()
    experts._nemo_ep_local_expert_params = True
    experts.register_parameter("weight0", expert)
    block = nn.Module()
    block.add_module("experts", experts)
    model = nn.Module()
    model.add_module(attribute, block)
    return model


def _clip(model: nn.Module, moe_mesh: DeviceMesh | None, **kwargs) -> float:
    from nemo_automodel.components.training.utils import scale_grads_and_clip_grad_norm

    total_norm = scale_grads_and_clip_grad_norm(
        MAX_NORM,
        [model],
        pp_enabled=False,
        device_mesh=None,
        moe_mesh=moe_mesh,
        ep_axis_name="ep" if moe_mesh is not None else None,
        foreach=None,
        **kwargs,
    )
    return float(total_norm)


def _add_dense(model: nn.Module, mesh: DeviceMesh) -> nn.Parameter:
    """Attach a dense param: one logical FSDP tensor sharded across all ranks.

    Local shard shape is ``[2, 4]``, so the logical tensor is ``[2 * world, 4]``;
    the grad is all-ones so any rank-inconsistent clip coefficient shows up as
    differing shard values.
    """
    dense = nn.Parameter(DTensor.from_local(torch.zeros(2, 4), mesh, [Shard(0)]))
    dense.grad = DTensor.from_local(torch.ones(2, 4), mesh, [Shard(0)])
    model.register_parameter("dense", dense)
    return dense


def _scenario_plain_experts(rank: int) -> None:
    """ep_size == world (ep_shard == 1): experts stay plain tensors.

    Passes dp_group_size like the recipes do, so the EP grad scaling
    (division by dp_group_size / ep_shard_size == 4) runs before clipping.
    """
    dp_mesh = init_device_mesh("cpu", (_WORLD,), mesh_dim_names=("dp_shard_cp",))
    moe_mesh = init_device_mesh("cpu", (1, _WORLD), mesh_dim_names=("ep_shard", "ep"))

    expert = nn.Parameter(torch.zeros(2, 8))
    expert.grad = torch.full((2, 8), 40.0 if rank == 0 else 0.4)
    model = _expert_model(expert)
    dense = _add_dense(model, dp_mesh)

    total_norm = _clip(model, moe_mesh, dp_group_size=_WORLD)

    # after ep scaling the expert grads are 10.0 on rank 0 and 0.1 elsewhere:
    # dense 8 * world * 1 + rank-0 experts 16 * 100 + each other rank 16 * 0.01
    correct = (8 * _WORLD * 1.0 + 16 * 100.0 + (_WORLD - 1) * 16 * 0.01) ** 0.5
    coef = MAX_NORM / (correct + 1e-6)
    torch.testing.assert_close(torch.tensor(total_norm), torch.tensor(correct))
    torch.testing.assert_close(dense.grad.to_local(), torch.full((2, 4), coef))
    expected_expert = torch.full((2, 8), (10.0 if rank == 0 else 0.1) * coef)
    torch.testing.assert_close(expert.grad, expected_expert)


def _scenario_ep_shard_experts(rank: int) -> None:
    """ep_size < world (ep_shard > 1): experts are DTensors on the ep_shard mesh only.

    Uses the ``moe`` attribute name to pin attribute-name independence.
    """
    dp_mesh = init_device_mesh("cpu", (_WORLD,), mesh_dim_names=("dp_shard_cp",))
    moe_mesh = DeviceMesh(
        "cpu",
        mesh=torch.tensor([[0, 1], [2, 3]], dtype=torch.int64),
        mesh_dim_names=("ep_shard", "ep"),
    )
    ep_shard_mesh = moe_mesh["ep_shard"]
    ep_index = rank % 2

    gval = 10.0 if ep_index == 0 else 0.1
    expert = nn.Parameter(DTensor.from_local(torch.zeros(2, 4), ep_shard_mesh, [Shard(1)]))
    expert.grad = DTensor.from_local(torch.full((2, 4), gval), ep_shard_mesh, [Shard(1)])
    model = _expert_model(expert, attribute="moe")
    dense = _add_dense(model, dp_mesh)

    total_norm = _clip(model, moe_mesh)

    # dense 8 * world * 1 + ep-group-0 experts 16 * 100 + ep-group-1 experts 16 * 0.01
    correct = (8 * _WORLD * 1.0 + 16 * 100.0 + 16 * 0.01) ** 0.5
    coef = MAX_NORM / (correct + 1e-6)
    torch.testing.assert_close(torch.tensor(total_norm), torch.tensor(correct))
    torch.testing.assert_close(dense.grad.to_local(), torch.full((2, 4), coef))
    torch.testing.assert_close(expert.grad.to_local(), torch.full((2, 4), gval * coef))


def _scenario_rank_without_expert_grads(rank: int) -> None:
    """Only one rank has expert grads: must neither hang nor disagree."""
    dp_mesh = init_device_mesh("cpu", (_WORLD,), mesh_dim_names=("dp_shard_cp",))
    moe_mesh = init_device_mesh("cpu", (1, _WORLD), mesh_dim_names=("ep_shard", "ep"))

    expert = nn.Parameter(torch.zeros(2, 8))
    if rank == 0:
        expert.grad = torch.full((2, 8), 10.0)
    model = _expert_model(expert)
    dense = _add_dense(model, dp_mesh)

    total_norm = _clip(model, moe_mesh)

    # dense 8 * world * 1 + rank-0 experts 16 * 100; the other ranks contribute nothing
    correct = (8 * _WORLD * 1.0 + 16 * 100.0) ** 0.5
    coef = MAX_NORM / (correct + 1e-6)
    torch.testing.assert_close(torch.tensor(total_norm), torch.tensor(correct))
    torch.testing.assert_close(dense.grad.to_local(), torch.full((2, 4), coef))
    if rank != 0:
        assert expert.grad is None


def _scenario_torch_fast_path(rank: int) -> None:
    """use_torch_clip_grad_norm must not bypass the EP reduction."""
    moe_mesh = init_device_mesh("cpu", (1, _WORLD), mesh_dim_names=("ep_shard", "ep"))

    expert = nn.Parameter(torch.zeros(2, 8))
    expert.grad = torch.full((2, 8), 10.0 if rank == 0 else 0.1)
    model = _expert_model(expert)

    total_norm = _clip(model, moe_mesh, use_torch_clip_grad_norm=True)

    correct = (16 * 100.0 + (_WORLD - 1) * 16 * 0.01) ** 0.5
    torch.testing.assert_close(torch.tensor(total_norm), torch.tensor(correct))


def _scenario_inf_norm(rank: int) -> None:
    """The inf-norm path must take the EP-wide max, not the rank-local one."""
    from nemo_automodel.components.training.utils import scale_grads_and_clip_grad_norm

    moe_mesh = init_device_mesh("cpu", (1, _WORLD), mesh_dim_names=("ep_shard", "ep"))

    expert = nn.Parameter(torch.zeros(2, 8))
    expert.grad = torch.full((2, 8), 10.0 if rank == 0 else 0.1)
    model = _expert_model(expert)

    total_norm = float(
        scale_grads_and_clip_grad_norm(
            MAX_NORM,
            [model],
            norm_type=float("inf"),
            moe_mesh=moe_mesh,
            ep_axis_name="ep",
            foreach=None,
        )
    )

    torch.testing.assert_close(torch.tensor(total_norm), torch.tensor(10.0))


def _scenario_ep_size_one(rank: int) -> None:
    """An ep axis of size 1 must behave exactly like moe_mesh=None."""
    moe_mesh = init_device_mesh("cpu", (_WORLD, 1), mesh_dim_names=("ep_shard", "ep"))

    expert = nn.Parameter(torch.zeros(4, 4))
    expert.grad = torch.full((4, 4), 2.0)
    with_mesh = _clip(_expert_model(expert), moe_mesh)

    expert2 = nn.Parameter(torch.zeros(4, 4))
    expert2.grad = torch.full((4, 4), 2.0)
    without_mesh = _clip(_expert_model(expert2), None)

    torch.testing.assert_close(torch.tensor(with_mesh), torch.tensor(without_mesh))
    torch.testing.assert_close(torch.tensor(with_mesh), torch.tensor(8.0))


# Every rank walks this list in the same order, so the collectives each scenario
# issues stay rank-uniform.
_SCENARIOS = (
    _scenario_plain_experts,
    _scenario_ep_shard_experts,
    _scenario_rank_without_expert_grads,
    _scenario_torch_fast_path,
    _scenario_inf_norm,
    _scenario_ep_size_one,
)


def _run_scenarios(rank: int, world: int, store_path: str) -> None:
    dist.init_process_group("gloo", rank=rank, world_size=world, init_method=f"file:///{store_path}")
    try:
        for scenario in _SCENARIOS:
            scenario(rank)
    finally:
        dist.destroy_process_group()


def test_ep_grad_clip_agrees_across_ranks(tmp_path):
    """All EP clipping scenarios, one process group, one spawn."""
    store = str(tmp_path / "s").replace("\\", "/")
    context = mp.spawn(_run_scenarios, args=(_WORLD, store), nprocs=_WORLD, join=False)

    deadline = time.monotonic() + _JOIN_TIMEOUT_S
    while not context.join(timeout=5.0):
        if time.monotonic() > deadline:
            for process in context.processes:
                if process.is_alive():
                    process.terminate()
            pytest.fail(f"ranks did not finish within {_JOIN_TIMEOUT_S:.0f}s; the clip collectives deadlocked")


def test_moe_mesh_none_keeps_single_process_behavior():
    from nemo_automodel.components.training.utils import scale_grads_and_clip_grad_norm

    expert = nn.Parameter(torch.zeros(4, 4))
    expert.grad = torch.full((4, 4), 2.0)
    model = _expert_model(expert)

    total_norm = scale_grads_and_clip_grad_norm(MAX_NORM, [model], moe_mesh=None, foreach=None)

    torch.testing.assert_close(torch.tensor(float(total_norm)), torch.tensor(8.0))
