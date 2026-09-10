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

"""Characterize replicated DTensor parameters; experimental, not a replacement fix.

Equivalent to TorchTitan's NoParallel boundaries for a tiny linear layer. A
controlled rank-dependent upstream gradient stands in for divergent computation;
this test does not claim to reproduce the source of Nano's numerical differences.
"""

import sys
from datetime import timedelta
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import DTensor, Partial, Replicate, distribute_module

from nemo_automodel.components.distributed.tp_replicas import (
    broadcast_tp_replicas,
    mark_tp_replica_gradient_reduction,
    synchronize_tp_replica_gradients,
)


def _replicated_input(module: nn.Module, inputs: tuple[torch.Tensor], mesh: DeviceMesh) -> tuple[DTensor]:
    """Wrap local inputs as TP replicas.

    Args:
        module: Linear being wrapped.
        inputs: Tuple containing a tensor of shape [batch, hidden], identical
            across TP ranks.
        mesh: One-dimensional TP mesh.

    Returns:
        Tuple containing a Replicate DTensor of global/local shape [batch, hidden].
    """
    return (DTensor.from_local(inputs[0], mesh, (Replicate(),), run_check=False),)


def _local_output(module: nn.Module, output: DTensor, mesh: DeviceMesh) -> torch.Tensor:
    """Return the local output without declaring a partial gradient.

    Args:
        module: Linear being wrapped.
        output: Replicate DTensor of global/local shape [batch, output_features].
        mesh: One-dimensional TP mesh.

    Returns:
        Local tensor of shape [batch, output_features], aliasing output storage.
    """
    return output.to_local()


def _replication_worker(rank: int, init_file: str) -> None:
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=2, timeout=timedelta(seconds=45)
    )
    try:
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("tp",))
        for varying_gradient in (False, True):
            # Deliberately different initial weights: distribute_module must
            # canonicalize them, independently of the backward question.
            model = nn.Linear(2, 1, bias=False)
            with torch.no_grad():
                model.weight.fill_(rank + 1.0)
            distribute_module(model, mesh, input_fn=_replicated_input, output_fn=_local_output)
            torch.testing.assert_close(model.weight.to_local(), torch.ones(1, 2), rtol=0, atol=0)
            inputs = torch.tensor([[2.0, 3.0]])
            model(inputs).backward(torch.tensor([[1.0 + rank if varying_gradient else 1.0]]))
            assert model.weight.grad.placements == (Replicate(),)
            # Replicate describes a contract; it is not a mean all-reduce.
            expected = inputs * (1 + rank if varying_gradient else 1)
            torch.testing.assert_close(model.weight.grad.to_local(), expected, rtol=0, atol=0)
            norm = torch.nn.utils.get_total_norm([model.weight.grad]).full_tensor()
            torch.testing.assert_close(norm, torch.linalg.vector_norm(expected), rtol=1e-6, atol=0)
            peers = [torch.empty_like(expected) for _ in range(2)]
            dist.all_gather(peers, model.weight.grad.to_local())
            assert torch.equal(peers[0], peers[1]) is (not varying_gradient)
            # A true partial contribution explicitly describes the missing sum.
            contribution = DTensor.from_local(expected, mesh, (Partial(),), run_check=False)
            total = contribution.redistribute(placements=(Replicate(),)).to_local()
            torch.testing.assert_close(total, inputs * (3 if varying_gradient else 2), rtol=0, atol=0)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.125)
            optimizer.step()
            torch.testing.assert_close(model.weight.to_local(), torch.ones(1, 2) - expected * 0.125, rtol=0, atol=0)
        # Native ownership skips only gradient reduction, not initialization.
        owned = nn.Linear(2, 1, bias=False)
        mark_tp_replica_gradient_reduction(owned, "none")
        # Parameter replacement must not lose the module-owned policy.
        owned.weight = nn.Parameter(torch.full((1, 2), rank + 1.0))
        assert broadcast_tp_replicas([owned], mesh) == 1
        torch.testing.assert_close(owned.weight, torch.ones(1, 2), rtol=0, atol=0)
        owned.weight.grad = torch.tensor([[3.0, 4.0]])
        with patch.object(dist, "all_reduce", side_effect=AssertionError("Unexpected second reduction")):
            assert synchronize_tp_replica_gradients([owned], mesh) == 0
        optimizer = torch.optim.SGD(owned.parameters(), lr=0.125)
        optimizer.step()
        torch.testing.assert_close(owned.weight, torch.tensor([[0.625, 0.5]]), rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_replicated_placement_is_not_a_gradient_consensus_operation(tmp_path, monkeypatch):
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo")
    torch.multiprocessing.spawn(_replication_worker, args=(str(tmp_path / "rdzv"),), nprocs=2, join=True)
