# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Native optimizer checkpoints retain parameter identity across group splitting."""

from copy import deepcopy

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

from nemo_automodel.components.optim.optimizer import AdamWConfig, OptimizerFromFactoryConfig, ParamGroupOverride


@pytest.fixture
def mixed_model():
    dist.init_process_group("gloo", rank=0, world_size=1, store=dist.HashStore())
    mesh = init_device_mesh("cpu", (1,))
    model = nn.Module()
    # Register ordinary parameters first so splitting changes positional order.
    for index in range(2):
        part = nn.Module()
        part.plain = nn.Parameter(torch.tensor([1.0, 2.0]) + index)
        part.sharded = nn.Parameter(distribute_tensor(torch.tensor([3.0, 4.0]) + index, mesh, [Shard(0)]))
        model.add_module(f"part{index}", part)
    yield model
    dist.destroy_process_group()


def _set_grads(model: nn.Module, step: int) -> None:
    """Assign distinct gradients to parameters of shape [2], local or Shard(0).

    Args:
        model: Module with FP32 ordinary/DTensor parameters of global shape [2].
        step: Update index controlling deterministic, distinct gradients.
    """
    for index, param in enumerate(model.parameters()):
        param.grad = torch.full_like(param, (index + 1) * (step + 1) / 8)


@pytest.mark.parametrize("factory", [False, True])
@pytest.mark.parametrize("explicit_groups", [False, True])
def test_legacy_native_checkpoint_next_update(mixed_model, factory, explicit_groups):
    model = mixed_model
    groups = [
        {
            "params": list(part.parameters()),
            "lr": lr,
            "weight_decay": wd,
            "lr_mult": lr_mult,
            "wd_mult": wd_mult,
            "param_names": [f"part{index}.{name}" for name, _ in part.named_parameters()],
        }
        for index, (part, lr, wd, lr_mult, wd_mult) in enumerate(
            [
                (model.part0, 0.01, 0.02, 1.0, 1.0),
                (model.part1, 0.03, 0.04, 3.0, 2.0),
            ]
        )
    ]
    reference = torch.optim.AdamW(groups, foreach=False)
    for step in range(3):
        _set_grads(model, step)
        reference.step()
    checkpoint = deepcopy(reference.state_dict())
    saved_weights = deepcopy(model.state_dict())
    _set_grads(model, 3)
    reference.step()
    expected = deepcopy(model.state_dict())
    model.load_state_dict(saved_weights)

    config = (
        OptimizerFromFactoryConfig(factory=torch.optim.AdamW, kwargs={"lr": 0.5, "foreach": False})
        if factory
        else AdamWConfig(lr=0.5)
    )
    config.param_group_overrides = [ParamGroupOverride(pattern="part1", lr_mult=3.0, wd_mult=2.0)]
    resumed = config.build_from_param_groups(_copy_groups(groups)) if explicit_groups else config.build(model)[0]
    assert len(resumed.param_groups) == 4
    resumed.load_state_dict(checkpoint)
    assert [group["lr"] for group in resumed.param_groups] == [0.01, 0.01, 0.03, 0.03]
    assert [group["weight_decay"] for group in resumed.param_groups] == [0.02, 0.02, 0.04, 0.04]
    assert [group["lr_mult"] for group in resumed.param_groups] == [1.0, 1.0, 3.0, 3.0]
    assert [group["wd_mult"] for group in resumed.param_groups] == [1.0, 1.0, 2.0, 2.0]
    assert [group["param_names"] for group in resumed.param_groups] == [
        ["part0.sharded"],
        ["part0.plain"],
        ["part1.sharded"],
        ["part1.plain"],
    ]
    # A current-layout checkpoint must bypass the legacy migration unchanged.
    current_checkpoint = deepcopy(resumed.state_dict())
    resumed.load_state_dict(current_checkpoint)
    _set_grads(model, 3)
    resumed.step()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
    # Loading must not mutate the caller's legacy checkpoint.
    assert len(checkpoint["param_groups"]) == 2
    assert checkpoint["param_groups"][0]["params"] == [0, 1]


def _copy_groups(groups: list[dict]) -> list[dict]:
    """Copy group metadata while retaining live parameters of arbitrary shape.

    Args:
        groups: Optimizer groups containing ordinary/DTensor parameters.

    Returns:
        Independent group dicts whose parameter references alias the inputs.
    """
    return [
        {**deepcopy({key: value for key, value in group.items() if key != "params"}), "params": list(group["params"])}
        for group in groups
    ]


def test_mixed_dtensor_foreach_step(mixed_model):
    config = OptimizerFromFactoryConfig(factory=torch.optim.AdamW, kwargs={"lr": 0.01, "foreach": True})
    optimizer = config.build(mixed_model)[0]
    assert len(optimizer.param_groups) == 2
    before = deepcopy(mixed_model.state_dict())
    _set_grads(mixed_model, 0)
    optimizer.step()
    for name, param in mixed_model.named_parameters():
        local = param.to_local() if isinstance(param, DTensor) else param
        old = before[name].to_local() if isinstance(before[name], DTensor) else before[name]
        assert torch.isfinite(local).all()
        assert not torch.equal(local, old)
