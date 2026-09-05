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

TE grouped expert parameters don't carry the EP axis on their own mesh: with
ep_size equal to the non-pp world they stay plain tensors, and with
ep_shard > 1 they are DTensors sharded only over the ep_shard mesh. Without an
extra reduction over the EP axis every EP group computes a different "global"
norm and clips its shards of the same logical dense FSDP parameter by a
different coefficient. These tests drive the real
``scale_grads_and_clip_grad_norm`` across gloo ranks and assert every rank
agrees on the correct global norm and applies the identical clip everywhere.

Expert modules are identified structurally through the
``_nemo_ep_local_expert_params`` marker that ``GroupedExpertsTE`` stamps at
construction, so identification is independent of parameter names, of the MoE
block's attribute name, and of this step's gradient state (collective
participation must be rank-uniform).
"""

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import DTensor, Shard

pytestmark = pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is required")

MAX_NORM = 1.0


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

    Local shard shape is ``[2, 4]``; the grad is all-ones so any
    rank-inconsistent clip coefficient shows up as differing shard values.
    """
    dense = nn.Parameter(DTensor.from_local(torch.zeros(2, 4), mesh, [Shard(0)]))
    dense.grad = DTensor.from_local(torch.ones(2, 4), mesh, [Shard(0)])
    model.register_parameter("dense", dense)
    return dense


def _run_plain_expert_rank(rank: int, world: int, store_path: str) -> None:
    """ep_size == world (ep_shard == 1): experts stay plain tensors.

    Passes dp_group_size like the recipes do, so the EP grad scaling
    (division by dp_group_size / ep_shard_size == 2) runs before clipping.
    """
    dist.init_process_group("gloo", rank=rank, world_size=world, init_method=f"file:///{store_path}")
    try:
        dp_mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("dp_shard_cp",))
        moe_mesh = init_device_mesh("cpu", (1, world), mesh_dim_names=("ep_shard", "ep"))

        expert = nn.Parameter(torch.zeros(2, 8))
        expert.grad = torch.full((2, 8), 20.0 if rank == 0 else 0.2)
        model = _expert_model(expert)
        dense = _add_dense(model, dp_mesh)

        total_norm = _clip(model, moe_mesh, dp_group_size=world)

        # after ep scaling the expert grads are 10.0 / 0.1:
        # dense 16 * 1 + rank-0 experts 16 * 100 + rank-1 experts 16 * 0.01
        correct = (16 * 1.0 + 16 * 100.0 + 16 * 0.01) ** 0.5
        coef = MAX_NORM / (correct + 1e-6)
        torch.testing.assert_close(torch.tensor(total_norm), torch.tensor(correct))
        torch.testing.assert_close(dense.grad.to_local(), torch.full((2, 4), coef))
        expected_expert = torch.full((2, 8), (10.0 if rank == 0 else 0.1) * coef)
        torch.testing.assert_close(expert.grad, expected_expert)
    finally:
        dist.destroy_process_group()


def _run_ep_shard_expert_rank(rank: int, world: int, store_path: str) -> None:
    """ep_size < world (ep_shard > 1): experts are DTensors on the ep_shard mesh only.

    Uses the ``moe`` attribute name to pin attribute-name independence.
    """
    dist.init_process_group("gloo", rank=rank, world_size=world, init_method=f"file:///{store_path}")
    try:
        dp_mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("dp_shard_cp",))
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

        # dense 32 * 1 + ep-group-0 experts 16 * 100 + ep-group-1 experts 16 * 0.01
        correct = (32 * 1.0 + 16 * 100.0 + 16 * 0.01) ** 0.5
        coef = MAX_NORM / (correct + 1e-6)
        torch.testing.assert_close(torch.tensor(total_norm), torch.tensor(correct))
        torch.testing.assert_close(dense.grad.to_local(), torch.full((2, 4), coef))
        torch.testing.assert_close(expert.grad.to_local(), torch.full((2, 4), gval * coef))
    finally:
        dist.destroy_process_group()


def _run_asymmetric_grads_rank(rank: int, world: int, store_path: str) -> None:
    """One rank has no expert grads at all: must neither hang nor disagree."""
    dist.init_process_group("gloo", rank=rank, world_size=world, init_method=f"file:///{store_path}")
    try:
        dp_mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("dp_shard_cp",))
        moe_mesh = init_device_mesh("cpu", (1, world), mesh_dim_names=("ep_shard", "ep"))

        expert = nn.Parameter(torch.zeros(2, 8))
        if rank == 0:
            expert.grad = torch.full((2, 8), 10.0)
        model = _expert_model(expert)
        dense = _add_dense(model, dp_mesh)

        total_norm = _clip(model, moe_mesh)

        # dense 16 * 1 + rank-0 experts 16 * 100; rank 1 contributes nothing
        correct = (16 * 1.0 + 16 * 100.0) ** 0.5
        coef = MAX_NORM / (correct + 1e-6)
        torch.testing.assert_close(torch.tensor(total_norm), torch.tensor(correct))
        torch.testing.assert_close(dense.grad.to_local(), torch.full((2, 4), coef))
        if rank == 1:
            assert expert.grad is None
    finally:
        dist.destroy_process_group()


def _run_torch_fast_path_rank(rank: int, world: int, store_path: str) -> None:
    """use_torch_clip_grad_norm must not bypass the EP reduction."""
    dist.init_process_group("gloo", rank=rank, world_size=world, init_method=f"file:///{store_path}")
    try:
        moe_mesh = init_device_mesh("cpu", (1, world), mesh_dim_names=("ep_shard", "ep"))

        expert = nn.Parameter(torch.zeros(2, 8))
        expert.grad = torch.full((2, 8), 10.0 if rank == 0 else 0.1)
        model = _expert_model(expert)

        total_norm = _clip(model, moe_mesh, use_torch_clip_grad_norm=True)

        correct = (16 * 100.0 + 16 * 0.01) ** 0.5
        torch.testing.assert_close(torch.tensor(total_norm), torch.tensor(correct))
    finally:
        dist.destroy_process_group()


def _run_inf_norm_rank(rank: int, world: int, store_path: str) -> None:
    """The inf-norm path must take the EP-wide max, not the rank-local one."""
    dist.init_process_group("gloo", rank=rank, world_size=world, init_method=f"file:///{store_path}")
    try:
        moe_mesh = init_device_mesh("cpu", (1, world), mesh_dim_names=("ep_shard", "ep"))

        expert = nn.Parameter(torch.zeros(2, 8))
        expert.grad = torch.full((2, 8), 10.0 if rank == 0 else 0.1)
        model = _expert_model(expert)

        from nemo_automodel.components.training.utils import scale_grads_and_clip_grad_norm

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
    finally:
        dist.destroy_process_group()


def _run_ep_size_one_rank(rank: int, world: int, store_path: str) -> None:
    """An ep axis of size 1 must behave exactly like moe_mesh=None."""
    dist.init_process_group("gloo", rank=rank, world_size=world, init_method=f"file:///{store_path}")
    try:
        moe_mesh = init_device_mesh("cpu", (1, 1), mesh_dim_names=("ep_shard", "ep"))

        expert = nn.Parameter(torch.zeros(4, 4))
        expert.grad = torch.full((4, 4), 2.0)
        with_mesh = _clip(_expert_model(expert), moe_mesh)

        expert2 = nn.Parameter(torch.zeros(4, 4))
        expert2.grad = torch.full((4, 4), 2.0)
        without_mesh = _clip(_expert_model(expert2), None)

        torch.testing.assert_close(torch.tensor(with_mesh), torch.tensor(without_mesh))
        torch.testing.assert_close(torch.tensor(with_mesh), torch.tensor(8.0))
    finally:
        dist.destroy_process_group()


def test_plain_te_experts_clip_with_one_global_norm(tmp_path):
    store = str(tmp_path / "s").replace("\\", "/")
    mp.spawn(_run_plain_expert_rank, args=(2, store), nprocs=2, join=True)


def test_ep_shard_te_experts_clip_with_one_global_norm(tmp_path):
    store = str(tmp_path / "s").replace("\\", "/")
    mp.spawn(_run_ep_shard_expert_rank, args=(4, store), nprocs=4, join=True)


def test_rank_without_expert_grads_neither_hangs_nor_diverges(tmp_path):
    store = str(tmp_path / "s").replace("\\", "/")
    mp.spawn(_run_asymmetric_grads_rank, args=(2, store), nprocs=2, join=True)


def test_torch_fast_path_does_not_bypass_ep_reduction(tmp_path):
    store = str(tmp_path / "s").replace("\\", "/")
    mp.spawn(_run_torch_fast_path_rank, args=(2, store), nprocs=2, join=True)


def test_inf_norm_takes_the_ep_wide_max(tmp_path):
    store = str(tmp_path / "s").replace("\\", "/")
    mp.spawn(_run_inf_norm_rank, args=(2, store), nprocs=2, join=True)


def test_ep_size_one_falls_back_to_plain_behavior(tmp_path):
    store = str(tmp_path / "s").replace("\\", "/")
    mp.spawn(_run_ep_size_one_rank, args=(1, store), nprocs=1, join=True)


def test_moe_mesh_none_keeps_single_process_behavior():
    from nemo_automodel.components.training.utils import scale_grads_and_clip_grad_norm

    expert = nn.Parameter(torch.zeros(4, 4))
    expert.grad = torch.full((4, 4), 2.0)
    model = _expert_model(expert)

    total_norm = scale_grads_and_clip_grad_norm(MAX_NORM, [model], moe_mesh=None, foreach=None)

    torch.testing.assert_close(torch.tensor(float(total_norm)), torch.tensor(8.0))
