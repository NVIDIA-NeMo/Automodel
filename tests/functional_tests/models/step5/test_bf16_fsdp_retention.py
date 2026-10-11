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

"""Real BF16 FSDP collectives and gradients, independent of checkpoint loading.

Launch with two CUDA torchrun ranks. Native loading and composed EP/CP/PP
coverage use the project's benchmark recipe separately.
"""

import os

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor
from torch.utils.checkpoint import checkpoint

from nemo_automodel.components.models.step5.parallelizer import _retain_bf16_fsdp_units
from tests.unit_tests.models.step5.test_model import make_model

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or int(os.environ.get("WORLD_SIZE", "1")) != 2,
    reason="Requires an explicit two-rank CUDA torchrun launch",
)


@pytest.fixture(scope="module")
def mesh():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    yield init_device_mesh("cuda", (2,))
    dist.destroy_process_group()


def _model():
    model = make_model(dtype="bfloat16", num_hidden_layers=1, moe_layer_list=[]).cuda()
    mlp = model.model.layers["0"].mlp
    with torch.no_grad():
        mlp.gate_proj.weight.zero_()
        mlp.down_proj.weight.zero_()
        mlp.gate_proj.weight[:32].copy_(torch.eye(32).cuda() / 8)
        mlp.down_proj.weight[:, :32].copy_(torch.eye(32).cuda() / 4)
    return model


def _input(rank: int, microbatch: int) -> torch.Tensor:
    """Return BF16 input [tokens, hidden=32], including a rank with zero tokens."""
    length = 3 + 2 * rank if microbatch == 0 else 2 * (1 - rank)
    return torch.arange(length * 32).reshape(length, 32).remainder(8).cuda().bfloat16() / 32


def _forward(model, hidden: torch.Tensor, *, ac: bool) -> torch.Tensor:
    """Exercise actual model projection modules without an attention kernel.

    Args:
        model: Tiny native model containing the projection units under test.
        hidden: BF16 or FP32 input [tokens, hidden=32]. Zero tokens are supported.
        ac: Whether to replay the first projection with nonreentrant checkpointing.

    Returns:
        Projected tensor [tokens, hidden=32] in the input dtype.
    """
    mlp = model.model.layers["0"].mlp
    hidden = checkpoint(mlp.gate_proj, hidden, use_reentrant=False) if ac else mlp.gate_proj(hidden)
    return mlp.down_proj(hidden)


@pytest.mark.parametrize("ac", [False, True])
@pytest.mark.parametrize("accumulation", [1, 2])
@pytest.mark.parametrize("all_forwards_first", [False, True])
def test_collective_count_gradient_norm_and_update(mesh, ac, accumulation, all_forwards_first):
    reference = _model().float()
    for microbatch in range(accumulation):
        inputs = torch.cat([_input(rank, microbatch).float() for rank in range(2)])
        _forward(reference, inputs, ac=ac).square().mean().backward()
    reference_params = dict(reference.named_parameters())
    expected = {name: p.grad.clone() for name, p in reference_params.items() if p.grad is not None}
    expected_norm = torch.linalg.vector_norm(torch.cat([grad.flatten() for grad in expected.values()]))
    torch.optim.SGD(reference.parameters(), lr=0.125).step()
    results, counts = [], []
    for retain in (False, True):
        model = _model()
        mlp = model.model.layers["0"].mlp
        policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        for unit in (mlp.gate_proj, mlp.down_proj):
            fully_shard(unit, mesh=mesh, mp_policy=policy, reshard_after_forward=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.125)
        original_ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
        _retain_bf16_fsdp_units(model.model, budget_mib=1 if retain else 0)
        assert original_ids == [id(p) for group in optimizer.param_groups for p in group["params"]]
        losses = []
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            for microbatch in range(accumulation):
                hidden = _input(dist.get_rank(), microbatch).requires_grad_()
                output = _forward(model, hidden, ac=ac)
                assert isinstance(mlp.gate_proj.weight, DTensor) != retain
                denominator = sum(_input(rank, microbatch).numel() for rank in range(2))
                loss = output.float().square().sum() * (2 / denominator)
                if all_forwards_first:
                    losses.append(loss)
                else:
                    loss.backward()
            for loss in losses:
                loss.backward()
            torch.cuda.synchronize()
        assert isinstance(mlp.gate_proj.weight, DTensor)
        counts.append(sum(event.count for event in profile.key_averages() if event.key == "c10d::_allgather_base_"))
        gradients = {}
        for name, parameter in model.named_parameters():
            if name not in expected:
                assert parameter.grad is None
                continue
            gradients[name] = parameter.grad.full_tensor().float()
            torch.testing.assert_close(gradients[name], expected[name], rtol=0.01, atol=2e-7)
        norm = torch.linalg.vector_norm(torch.cat([grad.flatten() for grad in gradients.values()]))
        torch.testing.assert_close(norm, expected_norm, rtol=0.01, atol=2e-7)
        optimizer.step()
        parameters = {}
        for name in expected:
            actual = dict(model.named_parameters())[name].full_tensor()
            parameters[name] = actual
            torch.testing.assert_close(actual.float(), reference_params[name], rtol=0.01, atol=2e-7)
        results.append((gradients, parameters))
    assert counts == [4 * accumulation, 2 * accumulation]
    for baseline, retained in zip(results[0], results[1]):
        for name in baseline:
            torch.testing.assert_close(baseline[name], retained[name], rtol=0, atol=0)


def test_shard_padding_is_charged_to_budget(mesh):
    model = _model()
    model.model.layers["0"].mlp.gate_proj = nn.Linear(1026, 511, bias=False, dtype=torch.bfloat16).cuda()
    projection = model.model.layers["0"].mlp.gate_proj
    assert projection.weight.numel() * 2 < 1024**2
    fully_shard(projection, mesh=mesh, reshard_after_forward=True)
    _retain_bf16_fsdp_units(model.model, budget_mib=1)
    assert fully_shard.state(projection)._fsdp_param_group.post_forward_mesh_info is not None
    _retain_bf16_fsdp_units(model.model, budget_mib=2)
    assert fully_shard.state(projection)._fsdp_param_group.post_forward_mesh_info is None
