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

"""Real FSDP2 updates and mixed dense/expert DTensor checkpoint round trips."""

import os
import socket
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor, Partial, Shard, distribute_tensor

pytest.importorskip("dion")

from nemo_automodel.components.checkpoint.stateful_wrappers import OptimizerState
from nemo_automodel.components.optim.muown import Muown
from nemo_automodel.components.optim.optimizer import MuownConfig


def _assert_parameters(actual, expected, atol=2e-5):
    for left, right in zip(actual.parameters(), expected.parameters()):
        value = left.full_tensor() if isinstance(left, DTensor) else left
        torch.testing.assert_close(value, right, atol=atol, rtol=2e-5)


def _worker(rank, world_size, port, checkpoint_dir, use_triton=False):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    try:
        with torch._dynamo.config.patch(disable=True):
            _check_mixed_meshes(world_size, checkpoint_dir, use_triton)
            _check_fsdp_training(use_triton)
    finally:
        dist.destroy_process_group()


def _check_mixed_meshes(world_size, checkpoint_dir, use_triton=False):
    mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp",))
    expert_mesh = init_device_mesh("cuda", (world_size // 2, 2), mesh_dim_names=("edp", "ep"))
    for axis in (1, 2):
        torch.manual_seed(31)
        reference = nn.ParameterDict(
            {
                "dense": nn.Parameter(torch.randn(32, 48, device="cuda")),
                "bias": nn.Parameter(torch.randn(16, device="cuda")),
                "experts": nn.Parameter(torch.randn(8, 32, 48, device="cuda")),
                "unused": nn.Parameter(torch.randn(16, 16, device="cuda")),
            }
        )
        layouts = {
            "dense": (mesh, [Shard(0)]),
            "bias": (mesh, [Shard(0)]),
            "experts": (expert_mesh, [Shard(axis), Shard(0)]),
            "unused": (mesh, [Shard(0)]),
        }

        def sharded_copy(source: nn.ParameterDict) -> nn.ParameterDict:
            """Shard a copy of the mixed dense/expert model.

            Args:
                source: Parameter dictionary with dense [32, 48], expert
                    [8, 32, 48], bias [16] and unused [16, 16] weights.
                    Expert axes are [expert, input, output].

            Returns:
                New parameter dictionary with global shapes unchanged; dense
                parameters use DP row shards and experts use EP batch shards
                plus EDP input or output shards.
            """
            return nn.ParameterDict(
                {
                    name: nn.Parameter(distribute_tensor(param.detach().clone(), *layouts[name]))
                    for name, param in source.items()
                }
            )

        def optimizer_for(model: nn.ParameterDict, *, fused: bool = False) -> Muown:
            """Construct one optimizer for the sharded_copy parameter contract."""
            return Muown(
                [
                    {"params": [model["dense"], model["unused"]]},
                    {"params": [model["experts"]], "matrix_transposed": True},
                    {"params": [model["bias"]], "algorithm": "adamw"},
                ],
                distributed_mesh=mesh,
                weight_decay=0.1,
                muon_update_scale=0.5,
                use_triton=fused,
            )

        model = sharded_copy(reference)
        optimizer = optimizer_for(model, fused=use_triton)
        ref_optimizer = optimizer_for(reference)
        # Use AM's exact stateful wrapper, including the untouched-parameter skeleton.
        for step in range(6):
            if step in (0, 5):
                before = [p.detach().clone() for p in model.parameters()]
                path = str(Path(checkpoint_dir) / f"axis{axis}_step{step}")
                dcp.save({"optimizer": OptimizerState(model, optimizer)}, checkpoint_id=path)
                for p, old in zip(model.parameters(), before):
                    torch.testing.assert_close(p.to_local(), old.to_local(), atol=0, rtol=0)
                restored_model = sharded_copy(reference)
                restored_optimizer = optimizer_for(restored_model, fused=use_triton)
                dcp.load({"optimizer": OptimizerState(restored_model, restored_optimizer)}, checkpoint_id=path)
                # Weight checkpoints are orthogonal to optimizer row-state serialization.
                with torch.no_grad():
                    for p, restored in zip(model.parameters(), restored_model.parameters()):
                        restored.copy_(p)
            for name, param in reference.items():
                if name == "unused":
                    continue
                grad = torch.randn_like(param)
                param.grad = grad
                model[name].grad = distribute_tensor(grad.clone(), *layouts[name])
                if step in (0, 5):
                    restored_model[name].grad = model[name].grad.clone()
            optimizer.step()
            ref_optimizer.step()
            _assert_parameters(model, reference)
            if step in (0, 5):
                restored_optimizer.step()
                for p, restored in zip(model.parameters(), restored_model.parameters()):
                    torch.testing.assert_close(p.to_local(), restored.to_local(), atol=0, rtol=0)
        assert optimizer.state[model["unused"]]["muown_step"] == 0

    # A global [3d, d] QKV matrix loses that ratio in its output shard.
    # The fused local update must retain the global QKV learning-rate scale.
    for transposed in (False, True):
        torch.manual_seed(73)
        value = torch.randn(96, 32, device="cuda")
        if transposed:
            value = value.mT.contiguous()
        expected = nn.Parameter(value.clone())
        placement = [Shard(1 if transposed else 0)]
        actual = nn.Parameter(distribute_tensor(value, mesh, placement))
        actual_optimizer = Muown([{"params": [actual], "matrix_transposed": transposed}], use_triton=use_triton)
        reference_optimizer = Muown([{"params": [expected], "matrix_transposed": transposed}])
        for _ in range(3):
            expected.grad = torch.randn_like(expected)
            actual.grad = distribute_tensor(expected.grad.clone(), mesh, placement)
            actual_optimizer.step()
            reference_optimizer.step()
            torch.testing.assert_close(actual.full_tensor(), expected, atol=2e-5, rtol=2e-5)

    uneven = nn.Parameter(distribute_tensor(torch.randn(2 * world_size + 1, 8, device="cuda"), mesh, [Shard(0)]))
    with pytest.raises(ValueError, match="evenly sharded"):
        Muown([uneven])
    partial = nn.Parameter(DTensor.from_local(torch.randn(8, 8, device="cuda"), mesh, [Partial()]))
    with pytest.raises(ValueError, match="Partial"):
        Muown([partial])
    if world_size >= 4:
        both_axes = nn.Parameter(
            distribute_tensor(torch.randn(16, 16, device="cuda"), expert_mesh, [Shard(0), Shard(1)])
        )
        with pytest.raises(NotImplementedError, match="one sharded matrix axis"):
            Muown([both_axes])

    # Plain 2D EP-local expert parameters must not be exchanged over the dense DP mesh.
    torch.manual_seed(100 + dist.get_rank())
    local = nn.Parameter(torch.randn(32, 48, device="cuda"))
    expected = nn.Parameter(local.detach().clone())
    actual_optimizer = Muown([local], distributed_mesh=mesh, use_triton=use_triton)
    expected_optimizer = Muown([expected])
    for _ in range(3):
        local.grad = torch.randn_like(local)
        expected.grad = local.grad.clone()
        actual_optimizer.step()
        expected_optimizer.step()
        torch.testing.assert_close(local, expected, atol=2e-5 if use_triton else 0, rtol=2e-5 if use_triton else 0)


def _check_fsdp_training(use_triton=False):
    torch.manual_seed(41)
    reference = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 8)).cuda()
    model = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 8)).cuda()
    model.load_state_dict(reference.state_dict())
    mesh = init_device_mesh("cuda", (dist.get_world_size(),), mesh_dim_names=("dp",))
    fully_shard(model[0], mesh=mesh)
    fully_shard(model[2], mesh=mesh)
    fully_shard(model, mesh=mesh)
    optimizer = MuownConfig(weight_decay=0.1, use_triton=use_triton, muon_update_scale=0.5).build(
        model, device_mesh=mesh
    )[0]
    reference_optimizer = MuownConfig(weight_decay=0.1, use_triton=use_triton, muon_update_scale=0.5).build(reference)[
        0
    ]
    for _ in range(4):
        inputs = torch.randn(8 * dist.get_world_size(), 16, device="cuda")
        local_inputs = inputs.chunk(dist.get_world_size())[dist.get_rank()]
        for network, batch, opt in [
            (model, local_inputs, optimizer),
            (reference, inputs, reference_optimizer),
        ]:
            opt.zero_grad(set_to_none=True)
            network(batch).square().mean().backward()
        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual.grad.full_tensor(), expected.grad, atol=1e-7, rtol=1e-5)
        optimizer.step()
        reference_optimizer.step()
        _assert_parameters(model, reference)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
@pytest.mark.parametrize("use_triton", [False, True])
def test_muown_fsdp_ep_checkpoint(tmp_path, use_triton):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    mp.spawn(_worker, args=(2, port, str(tmp_path), use_triton), nprocs=2, join=True)
