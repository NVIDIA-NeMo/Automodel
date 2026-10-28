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

"""Unsharded FSDP2 compute precision must preserve optimizer and buffer precision."""

import copy
import inspect
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.distributed.fsdp import MixedPrecisionPolicy

from nemo_automodel.components.distributed.config import FSDP2Config
from nemo_automodel.components.distributed.model_parallelizer import parallelize_model


class _Projection(nn.Linear):
    def __init__(self):
        super().__init__(4, 4, bias=False)
        self.autocast_states = []

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Project a batch without changing its layout.

        Args:
            inputs: Tensor of shape [batch, hidden], where hidden is 4.

        Returns:
            Tensor of shape [batch, hidden] in the active compute dtype.
        """
        self.autocast_states.append(torch.is_autocast_enabled(inputs.device.type))
        return super().forward(inputs)


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = _Projection()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Transform a batch without changing its layout.

        Args:
            inputs: Tensor of shape [batch, hidden], where hidden is 4.

        Returns:
            Tensor of shape [batch, hidden] in the active compute dtype.
        """
        return self.mlp(inputs).tanh()


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_Block()])
        self.register_buffer("inv_freq", torch.tensor([0.123456789, 0.987654321]))

    def forward(self, inputs: torch.Tensor, *, fail: bool = False) -> torch.Tensor:
        """Transform a batch without changing its layout.

        Args:
            inputs: Tensor of shape [batch, hidden], where hidden is 4.
            fail: Whether to raise before executing the projection.

        Returns:
            Tensor of shape [batch, hidden] in the active compute dtype.
        """
        if fail:
            raise ValueError("forward failed")
        return self.layers[0](inputs)


def _parallelize(model, *, dtype=torch.bfloat16, autocast_dtype=None, checkpointing=False):
    context = SimpleNamespace(
        cp_size=1,
        ep_size=1,
        device_mesh=SimpleNamespace(size=lambda: 1, device_type="cpu"),
        strategy_config=FSDP2Config(
            mp_policy=MixedPrecisionPolicy(param_dtype=dtype),
            autocast_dtype=autocast_dtype,
        ),
        activation_checkpointing=checkpointing,
        reapply_trainability=None,
    )
    return parallelize_model(model, context)


@pytest.mark.parametrize("checkpointing", [False, True])
def test_single_rank_compute_matches_autocast_training(checkpointing):
    torch.manual_seed(42)
    model = _Model()
    reference = copy.deepcopy(model)
    projection = model.layers[0].mlp
    original_buffers = {name: buffer.clone() for name, buffer in model.named_buffers()}
    state_keys = tuple(model.state_dict())
    signature = inspect.signature(model.forward)
    assert _parallelize(model, checkpointing=checkpointing) is model
    assert inspect.signature(model.forward) == signature
    assert tuple(model.state_dict()) == state_keys
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.01)

    for _ in range(2):
        inputs = torch.randn(2, 4)
        output = model(inputs)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            expected = reference(inputs)
        assert output.dtype == torch.bfloat16
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        output.float().square().mean().backward()
        expected.float().square().mean().backward()
        for parameter, oracle in zip(model.parameters(), reference.parameters()):
            assert parameter.dtype == parameter.grad.dtype == torch.float32
            torch.testing.assert_close(parameter.grad, oracle.grad, rtol=0, atol=0)
        optimizer.step()
        reference_optimizer.step()
        optimizer.zero_grad()
        reference_optimizer.zero_grad()

    assert all(projection.autocast_states)
    assert len(projection.autocast_states) >= (4 if checkpointing else 2)
    for parameter, oracle in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(parameter, oracle, rtol=0, atol=0)
        assert optimizer.state[parameter]["exp_avg"].dtype == torch.float32
        assert optimizer.state[parameter]["exp_avg_sq"].dtype == torch.float32
    for name, buffer in model.named_buffers():
        assert buffer.dtype == torch.float32
        torch.testing.assert_close(buffer, original_buffers[name], rtol=0, atol=0)
    assert not torch.is_autocast_enabled("cpu")


@pytest.mark.parametrize(
    ("dtype", "override", "expected"),
    [
        (torch.float32, None, torch.float32),
        (None, None, torch.float32),
        (torch.float16, None, torch.float16),
        (torch.bfloat16, torch.float16, torch.float16),
        (torch.bfloat16, torch.float32, torch.float32),
    ],
)
def test_single_rank_respects_compute_dtype(dtype, override, expected):
    model = _parallelize(_Model(), dtype=dtype, autocast_dtype=override)
    assert model(torch.randn(2, 4)).dtype == expected


def test_single_rank_autocast_restores_context_after_exception_and_deepcopy():
    model = _parallelize(_Model())
    with pytest.raises(ValueError, match="forward failed"):
        model(torch.randn(2, 4), fail=True)
    assert not torch.is_autocast_enabled("cpu")
    cloned = copy.deepcopy(model)
    with torch.no_grad():
        next(cloned.parameters()).zero_()
    assert torch.count_nonzero(cloned(torch.ones(2, 4))) == 0
    assert torch.count_nonzero(model(torch.ones(2, 4))) > 0
