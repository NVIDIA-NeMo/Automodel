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

"""Real-collective tests for tensor-parallel replica synchronization."""

import os
from datetime import timedelta

import torch
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard

from nemo_automodel.components.training.utils import scale_grads_and_clip_grad_norm
from nemo_automodel.shared.tp_replicas import (
    _is_tp_replicated,
    broadcast_tp_replicas,
    exclude_from_tp_replica_sync,
    mark_tp_replica_gradient_reduction,
    synchronize_tp_replica_gradients,
)


class _ParameterHolder(nn.Module):
    """Own one parameter so reduction semantics can be attached per module."""

    def __init__(self, parameter: nn.Parameter) -> None:
        super().__init__()
        self.weight = parameter


class _ReplicaModel(nn.Module):
    """Mix full replicas, partial replicas, a TP shard, and inactive parameters."""

    def __init__(self, rank: int, tp_mesh) -> None:
        super().__init__()
        self.mean_replica = _ParameterHolder(nn.Parameter(torch.tensor([1.0 + rank, 2.0 + rank])))
        self.sum_replica = _ParameterHolder(nn.Parameter(torch.tensor([3.0 + rank])))
        mark_tp_replica_gradient_reduction(self.sum_replica, "sum")

        replicated = DTensor.from_local(
            torch.tensor([4.0 + rank]),
            tp_mesh,
            (Replicate(),),
            run_check=False,
        )
        self.dtensor_replica = _ParameterHolder(nn.Parameter(replicated))

        sharded = DTensor.from_local(
            torch.tensor([10.0 + rank]),
            tp_mesh,
            (Shard(0),),
            run_check=False,
            shape=torch.Size((2,)),
            stride=(1,),
        )
        self.tp_shard = _ParameterHolder(nn.Parameter(sharded))
        self.unused = _ParameterHolder(nn.Parameter(torch.tensor([5.0 + rank])))
        self.frozen = _ParameterHolder(nn.Parameter(torch.tensor([6.0 + rank]), requires_grad=False))
        self.register_buffer("running_value", torch.tensor([7.0 + rank]))


class _RankOrderedBufferModel(nn.Module):
    """Register differently sized buffers in opposite orders on the two ranks."""

    def __init__(self, rank: int) -> None:
        super().__init__()
        buffers = (
            ("full_attention_inv_freq", torch.full((64,), 10.0 + rank)),
            ("sliding_attention_inv_freq", torch.full((32,), 20.0 + rank)),
        )
        ordered_buffers = buffers if rank == 0 else reversed(buffers)
        for name, buffer in ordered_buffers:
            self.register_buffer(name, buffer)


class _RankOrderedParameterModel(nn.Module):
    """Register differently sized parameters in opposite orders on the two ranks."""

    def __init__(self, rank: int) -> None:
        super().__init__()
        parameters = (
            ("full_weight", nn.Parameter(torch.full((64,), 30.0 + rank))),
            ("sliding_weight", nn.Parameter(torch.full((32,), 40.0 + rank))),
        )
        ordered_parameters = parameters if rank == 0 else reversed(parameters)
        for name, parameter in ordered_parameters:
            self.register_parameter(name, parameter)


class _OwnerShardedExpertModel(nn.Module):
    """Plain local tensors owned by different experts on folded TP/EP ranks."""

    def __init__(self, rank: int, owner_mesh) -> None:
        super().__init__()
        self.owner_mesh = owner_mesh
        self.expert = nn.Module()
        self.expert.weight = nn.Parameter(torch.full((4,), 50.0 + rank))
        self.expert.register_buffer("scale", torch.full((2,), 60.0 + rank))
        exclude_from_tp_replica_sync(self.expert)


def _replicated_dtensor_gradient(local_gradient: torch.Tensor, tp_mesh) -> DTensor:
    """Wrap a rank-local gradient as a replicated DTensor without checking peers."""
    return DTensor.from_local(local_gradient, tp_mesh, (Replicate(),), run_check=False)


def _sharded_dtensor_gradient(local_gradient: torch.Tensor, tp_mesh) -> DTensor:
    """Wrap a one-element local gradient as one shard of a two-element tensor."""
    return DTensor.from_local(
        local_gradient,
        tp_mesh,
        (Shard(0),),
        run_check=False,
        shape=torch.Size((2,)),
        stride=(1,),
    )


def _run_replica_sync_worker(rank: int, world_size: int, init_file: str) -> None:
    """Compare two-rank replica synchronization and clipping with an FP32 reference."""
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        tp_mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("tp",))
        folded_mesh = init_device_mesh(
            "cpu",
            (1, world_size),
            mesh_dim_names=("ep_shard", "ep"),
        )
        folded_tp_shard = DTensor.from_local(
            torch.tensor([rank + 1.0]),
            folded_mesh,
            (Replicate(), Shard(0)),
            run_check=False,
            shape=torch.Size((2,)),
            stride=(1,),
        )
        assert not _is_tp_replicated(folded_tp_shard, tuple(range(world_size)), rank, "tp")

        rank_ordered_parameters = _RankOrderedParameterModel(rank)
        synchronized = broadcast_tp_replicas([rank_ordered_parameters], tp_mesh)
        assert synchronized == 2
        torch.testing.assert_close(rank_ordered_parameters.full_weight, torch.full((64,), 30.0))
        torch.testing.assert_close(rank_ordered_parameters.sliding_weight, torch.full((32,), 40.0))
        torch.distributed.barrier()

        rank_ordered_parameters.full_weight.grad = torch.full((64,), 1.0 + rank)
        rank_ordered_parameters.sliding_weight.grad = torch.full((32,), 10.0 + rank)
        synchronized = synchronize_tp_replica_gradients([rank_ordered_parameters], tp_mesh)
        assert synchronized == 2
        torch.testing.assert_close(rank_ordered_parameters.full_weight.grad, torch.full((64,), 1.5))
        torch.testing.assert_close(rank_ordered_parameters.sliding_weight.grad, torch.full((32,), 10.5))
        torch.distributed.barrier()

        owner_sharded_experts = _OwnerShardedExpertModel(rank, folded_mesh)
        owner_sharded_experts.expert.weight.grad = torch.full_like(owner_sharded_experts.expert.weight, 70.0 + rank)
        assert broadcast_tp_replicas([owner_sharded_experts], tp_mesh) == 0
        assert synchronize_tp_replica_gradients([owner_sharded_experts], tp_mesh) == 0
        torch.testing.assert_close(owner_sharded_experts.expert.weight, torch.full((4,), 50.0 + rank))
        torch.testing.assert_close(owner_sharded_experts.expert.scale, torch.full((2,), 60.0 + rank))
        torch.testing.assert_close(owner_sharded_experts.expert.weight.grad, torch.full((4,), 70.0 + rank))
        torch.distributed.barrier()

        rank_ordered_buffers = _RankOrderedBufferModel(rank)
        synchronized = broadcast_tp_replicas([rank_ordered_buffers], tp_mesh)
        assert synchronized == 2
        torch.testing.assert_close(rank_ordered_buffers.full_attention_inv_freq, torch.full((64,), 10.0))
        torch.testing.assert_close(rank_ordered_buffers.sliding_attention_inv_freq, torch.full((32,), 20.0))
        torch.distributed.barrier()

        for accumulation_steps in (1, 2):
            _run_replica_sync_case(rank, world_size, accumulation_steps, tp_mesh)
            torch.distributed.barrier()

        asymmetric_model = _ReplicaModel(rank, tp_mesh)
        if rank == 0:
            asymmetric_model.mean_replica.weight.grad = torch.ones_like(asymmetric_model.mean_replica.weight)
        try:
            synchronize_tp_replica_gradients([asymmetric_model], tp_mesh)
        except RuntimeError as error:
            assert "Gradient presence differs across TP replicas" in str(error)
        else:
            raise AssertionError("Asymmetric TP gradient presence must fail on every rank")
        torch.distributed.barrier()
    finally:
        torch.distributed.destroy_process_group()


def _run_replica_sync_case(rank: int, world_size: int, accumulation_steps: int, tp_mesh) -> None:
    """Run one accumulation-depth case inside an initialized TP process group."""
    model = _ReplicaModel(rank, tp_mesh)

    synchronized = broadcast_tp_replicas([model], tp_mesh)
    assert synchronized == 6
    torch.testing.assert_close(model.mean_replica.weight, torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(model.sum_replica.weight, torch.tensor([3.0]))
    torch.testing.assert_close(model.dtensor_replica.weight.to_local(), torch.tensor([4.0]))
    torch.testing.assert_close(model.tp_shard.weight.to_local(), torch.tensor([10.0 + rank]))
    torch.testing.assert_close(model.unused.weight, torch.tensor([5.0]))
    torch.testing.assert_close(model.frozen.weight, torch.tensor([6.0]))
    torch.testing.assert_close(model.running_value, torch.tensor([7.0]))

    mean_gradient = torch.zeros_like(model.mean_replica.weight)
    sum_gradient = torch.zeros_like(model.sum_replica.weight)
    dtensor_gradient = torch.zeros_like(model.dtensor_replica.weight.to_local())
    shard_gradient = torch.zeros_like(model.tp_shard.weight.to_local())
    for microbatch in range(accumulation_steps):
        mean_gradient.add_(torch.tensor([rank + 1.0 + microbatch, 2.0 * (rank + 1) + microbatch]))
        sum_gradient.add_(rank + 1.0 + microbatch)
        dtensor_gradient.add_(3.0 * (rank + 1) + microbatch)
        shard_gradient.add_(4.0 * (rank + 1) + microbatch)

    model.mean_replica.weight.grad = mean_gradient
    model.sum_replica.weight.grad = sum_gradient
    model.dtensor_replica.weight.grad = _replicated_dtensor_gradient(dtensor_gradient, tp_mesh)
    model.tp_shard.weight.grad = _sharded_dtensor_gradient(shard_gradient, tp_mesh)

    expected_mean = sum(
        (
            torch.tensor([peer + 1.0 + microbatch, 2.0 * (peer + 1) + microbatch])
            for peer in range(world_size)
            for microbatch in range(accumulation_steps)
        ),
        start=torch.zeros(2),
    ).div(world_size)
    expected_sum = torch.tensor(
        [sum(peer + 1.0 + microbatch for peer in range(world_size) for microbatch in range(accumulation_steps))]
    )
    expected_dtensor = torch.tensor(
        [
            sum(3.0 * (peer + 1) + microbatch for peer in range(world_size) for microbatch in range(accumulation_steps))
            / world_size
        ]
    )
    expected_shards = [
        torch.tensor([sum(4.0 * (peer + 1) + microbatch for microbatch in range(accumulation_steps))])
        for peer in range(world_size)
    ]
    reference_gradient = torch.cat([expected_mean, expected_sum, expected_dtensor, *expected_shards])
    expected_norm = torch.linalg.vector_norm(reference_gradient.double())
    clip_coefficient = min(1.0, 1.0 / (expected_norm.item() + 1.0e-6))

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, foreach=False)
    actual_norm = scale_grads_and_clip_grad_norm(
        1.0,
        [model],
        device_mesh=tp_mesh,
        foreach=False,
    )

    torch.testing.assert_close(actual_norm, expected_norm, rtol=1.0e-6, atol=1.0e-7)
    torch.testing.assert_close(model.mean_replica.weight.grad, expected_mean * clip_coefficient)
    torch.testing.assert_close(model.sum_replica.weight.grad, expected_sum * clip_coefficient)
    torch.testing.assert_close(
        model.dtensor_replica.weight.grad.to_local(),
        expected_dtensor * clip_coefficient,
    )
    torch.testing.assert_close(
        model.tp_shard.weight.grad.to_local(),
        expected_shards[rank] * clip_coefficient,
    )
    assert model.unused.weight.grad is None
    assert model.frozen.weight.grad is None

    optimizer.step()
    torch.testing.assert_close(
        model.mean_replica.weight,
        torch.tensor([1.0, 2.0]) - 0.1 * expected_mean * clip_coefficient,
    )
    torch.testing.assert_close(
        model.sum_replica.weight,
        torch.tensor([3.0]) - 0.1 * expected_sum * clip_coefficient,
    )
    torch.testing.assert_close(
        model.dtensor_replica.weight.to_local(),
        torch.tensor([4.0]) - 0.1 * expected_dtensor * clip_coefficient,
    )
    torch.testing.assert_close(
        model.tp_shard.weight.to_local(),
        torch.tensor([10.0 + rank]) - 0.1 * expected_shards[rank] * clip_coefficient,
    )
    torch.testing.assert_close(model.unused.weight, torch.tensor([5.0]))
    torch.testing.assert_close(model.frozen.weight, torch.tensor([6.0]))


def test_tp_replica_sync_matches_reference_before_clipping(tmp_path) -> None:
    """TP replicas match FP32 reference gradients, norm, and post-step weights."""
    torch.multiprocessing.spawn(
        _run_replica_sync_worker,
        args=(2, str(tmp_path / "tp_replica")),
        nprocs=2,
        join=True,
    )
