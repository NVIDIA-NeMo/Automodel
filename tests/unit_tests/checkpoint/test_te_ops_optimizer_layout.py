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

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from nemo_automodel.components.checkpoint.stateful_wrappers import OptimizerState
from nemo_automodel.components.moe.experts import GroupedExpertsTeOps


class _ModelWithTeOps(nn.Module):
    def __init__(self, activation: str = "quick_geglu", block_size: int | None = None) -> None:
        super().__init__()
        self.dense = nn.Linear(2, 2, bias=False)
        experts = GroupedExpertsTeOps.__new__(GroupedExpertsTeOps)
        nn.Module.__init__(experts)
        experts.config = SimpleNamespace(expert_activation=activation)
        experts._te_glu_interleave_size = block_size
        self.mlp = nn.Module()
        self.mlp.experts = experts


def _optimizer_state(model: nn.Module) -> OptimizerState:
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return OptimizerState(model, optimizer)


def test_te_ops_optimizer_state_carries_physical_layout_guard():
    model = _ModelWithTeOps(block_size=32)
    wrapper = _optimizer_state(model)

    with patch(
        "nemo_automodel.components.checkpoint.stateful_wrappers.get_optimizer_state_dict",
        return_value={"state": {}, "param_groups": []},
    ):
        state = wrapper.state_dict()

    guard = state["te_ops_optimizer_layout"]["model0.mlp.experts"]
    assert guard.tolist() == [1, 4, 32, 2, 0]


def test_te_ops_optimizer_load_accepts_matching_layout_and_removes_guard():
    model = _ModelWithTeOps(block_size=32)
    wrapper = _optimizer_state(model)
    state = {
        "optim": {"state": {}, "param_groups": []},
        "te_ops_optimizer_layout": {key: value.clone() for key, value in wrapper._te_ops_optimizer_layout.items()},
    }

    with patch("nemo_automodel.components.checkpoint.stateful_wrappers.set_optimizer_state_dict") as setter:
        wrapper.load_state_dict(state)

    setter.assert_called_once()
    assert "te_ops_optimizer_layout" not in state


@pytest.mark.parametrize("mode", ["missing", "poisoned", "different_block_size"])
def test_te_ops_optimizer_load_rejects_unsafe_or_legacy_layout(mode):
    model = _ModelWithTeOps(block_size=32)
    wrapper = _optimizer_state(model)
    guards = {key: value.clone() for key, value in wrapper._te_ops_optimizer_layout.items()}
    state = {"optim": {"state": {}, "param_groups": []}, "te_ops_optimizer_layout": guards}
    if mode == "missing":
        state.pop("te_ops_optimizer_layout")
    elif mode == "poisoned":
        wrapper.prepare_state_dict_for_load(state)
    else:
        next(iter(guards.values()))[2] = 0

    with (
        patch("nemo_automodel.components.checkpoint.stateful_wrappers.set_optimizer_state_dict") as setter,
        pytest.raises(RuntimeError, match="Restore model state only and reset the optimizer"),
    ):
        wrapper.load_state_dict(state)

    setter.assert_not_called()
