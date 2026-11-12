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

"""Native optimizer checkpoint skeletons for TE FusedAdam, using a stand-in optimizer instead of a TE install."""

from collections.abc import Callable, Iterable
from unittest.mock import patch

import pytest
import torch
from torch import nn

from nemo_automodel.components.checkpoint import stateful_wrappers
from nemo_automodel.components.checkpoint.stateful_wrappers import (
    OptimizerState,
    _materialize_native_fused_adam_state,
)

pytestmark = pytest.mark.cpu_only


class _StubFusedAdam(torch.optim.Optimizer):
    """Stand-in for TE FusedAdam: lazy per-parameter state, a per-group step counter, recorded init calls.

    Args:
        params: Parameters or parameter-group dicts; each parameter keeps its tensor type, shape and dtype.
        capturable: Whether the step counter is an int32 tensor of shape [1], as in TE's capturable mode.
        store_param_remainders: Whether bf16 parameters keep int16 master-weight remainders.
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter] | Iterable[dict[str, list[nn.Parameter] | float]],
        *,
        capturable: bool = False,
        store_param_remainders: bool = False,
    ) -> None:
        super().__init__(params, {"lr": 1e-3})
        self.capturable = capturable
        self.store_param_remainders = store_param_remainders
        # A meta buffer makes the empty-group device fallback observable without CUDA.
        self._dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device="meta")
        self.initialize_calls: list[tuple[nn.Parameter, bool]] = []

    def initialize_state(self, param: nn.Parameter, store_param_remainders: bool) -> None:
        """Record the call and allocate zero moments shaped like ``param``."""
        self.initialize_calls.append((param, store_param_remainders))
        self.state[param]["exp_avg"] = torch.zeros_like(param)
        self.state[param]["exp_avg_sq"] = torch.zeros_like(param)

    def step(self, closure: Callable[[], float] | None = None) -> None:
        raise AssertionError("materializing the checkpoint skeleton must not step the optimizer")


class _MixedDtypeModel(nn.Module):
    """A bf16 linear layer and an fp32 linear layer."""

    def __init__(self) -> None:
        super().__init__()
        self.bf16_layer = nn.Linear(3, 2, dtype=torch.bfloat16)
        self.fp32_layer = nn.Linear(2, 1)


def _param_groups(model: _MixedDtypeModel) -> list[dict[str, list[nn.Parameter] | float]]:
    return [
        {"params": list(model.bf16_layer.parameters())},
        {"params": list(model.fp32_layer.parameters()), "weight_decay": 0.0},
    ]


@pytest.fixture
def stub_fused_adam(monkeypatch: pytest.MonkeyPatch) -> type[_StubFusedAdam]:
    """Make the checkpoint wrapper treat ``_StubFusedAdam`` as TE's FusedAdam."""
    monkeypatch.setattr(stateful_wrappers, "HAS_FUSED_ADAM", True)
    monkeypatch.setattr(stateful_wrappers, "FusedAdam", _StubFusedAdam)
    return _StubFusedAdam


def test_materializes_zero_step_and_only_missing_state(stub_fused_adam):
    model = _MixedDtypeModel()
    optimizer = stub_fused_adam(_param_groups(model), store_param_remainders=True)
    existing = torch.full_like(model.fp32_layer.bias, 7.0)
    optimizer.state[model.fp32_layer.bias]["exp_avg"] = existing

    _materialize_native_fused_adam_state(optimizer)

    assert [group["step"] for group in optimizer.param_groups] == [0, 0]
    # bf16 parameters keep int16 remainders; the fp32 parameter gets a full master copy.
    assert optimizer.initialize_calls == [
        (model.bf16_layer.weight, True),
        (model.bf16_layer.bias, True),
        (model.fp32_layer.weight, False),
    ]
    assert optimizer.state[model.fp32_layer.bias] == {"exp_avg": existing}
    for parameter in (model.bf16_layer.weight, model.bf16_layer.bias, model.fp32_layer.weight):
        assert optimizer.state[parameter].keys() == {"exp_avg", "exp_avg_sq"}


def test_remainders_are_not_requested_when_disabled(stub_fused_adam):
    model = _MixedDtypeModel()
    optimizer = stub_fused_adam(_param_groups(model), store_param_remainders=False)

    _materialize_native_fused_adam_state(optimizer)

    assert [remainder for _, remainder in optimizer.initialize_calls] == [False] * 4


def test_capturable_step_is_int32_tensor_on_the_group_device(stub_fused_adam):
    model = _MixedDtypeModel()
    optimizer = stub_fused_adam([*_param_groups(model), {"params": []}], capturable=True)

    _materialize_native_fused_adam_state(optimizer)

    populated, empty = optimizer.param_groups[:2], optimizer.param_groups[2]
    for group in populated:
        torch.testing.assert_close(group["step"], torch.zeros(1, dtype=torch.int32), rtol=0, atol=0)
    assert empty["step"].dtype == torch.int32
    assert tuple(empty["step"].shape) == (1,)
    assert empty["step"].device == optimizer._dummy_overflow_buf.device


def test_existing_step_counters_are_kept(stub_fused_adam):
    optimizer = stub_fused_adam(_param_groups(_MixedDtypeModel()))
    optimizer.param_groups[0]["step"] = 7

    _materialize_native_fused_adam_state(optimizer)

    assert [group["step"] for group in optimizer.param_groups] == [7, 0]


def test_other_optimizers_are_untouched(stub_fused_adam):
    model = _MixedDtypeModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    _materialize_native_fused_adam_state(optimizer)

    assert all("step" not in group for group in optimizer.param_groups)
    assert not optimizer.state


def test_noop_without_transformer_engine(monkeypatch):
    monkeypatch.setattr(stateful_wrappers, "HAS_FUSED_ADAM", False)
    monkeypatch.setattr(stateful_wrappers, "FusedAdam", None)
    optimizer = _StubFusedAdam(_param_groups(_MixedDtypeModel()))

    _materialize_native_fused_adam_state(optimizer)

    assert optimizer.initialize_calls == []
    assert all("step" not in group for group in optimizer.param_groups)


def test_native_optimizer_state_dict_materializes_fused_adam_skeleton(stub_fused_adam):
    model = _MixedDtypeModel()
    optimizer = stub_fused_adam(_param_groups(model), store_param_remainders=True)

    skeleton = OptimizerState(model, optimizer, is_peft=True, has_expert_parallelism=True).state_dict()["optim"]

    assert skeleton["state"].keys() == {0, 1, 2, 3}
    assert [group["step"] for group in skeleton["param_groups"]] == [0, 0]
    assert len(optimizer.initialize_calls) == 4


def test_flattened_optimizer_state_dict_leaves_fused_adam_alone(stub_fused_adam):
    model = _MixedDtypeModel()
    optimizer = stub_fused_adam(_param_groups(model))

    with patch("nemo_automodel.components.checkpoint.stateful_wrappers.get_optimizer_state_dict", return_value={}):
        OptimizerState(model, optimizer).state_dict()

    assert optimizer.initialize_calls == []
    assert all("step" not in group for group in optimizer.param_groups)
