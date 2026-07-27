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

from unittest.mock import patch

import pytest
import torch
from torch import nn

from nemo_automodel.components.checkpoint.checkpointing import Checkpointer, CheckpointingConfig
from nemo_automodel.components.checkpoint.stateful_wrappers import OptimizerState


def _make_stepped_adam_parts(
    count: int = 2,
) -> tuple[list[nn.Module], list[torch.optim.AdamW]]:
    model_parts: list[nn.Module] = []
    optimizers: list[torch.optim.AdamW] = []
    for index in range(count):
        model_part = nn.Linear(2, 1, bias=False)
        optimizer = torch.optim.AdamW(model_part.parameters(), lr=0.01 * (index + 1))
        for parameter in model_part.parameters():
            parameter.grad = torch.full_like(parameter, float(index + 1))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        model_parts.append(model_part)
        optimizers.append(optimizer)
    return model_parts, optimizers


def _assert_adam_states_equal(
    actual: torch.optim.AdamW,
    expected: torch.optim.AdamW,
) -> None:
    actual_parameter = actual.param_groups[0]["params"][0]
    expected_parameter = expected.param_groups[0]["params"][0]
    actual_state = actual.state[actual_parameter]
    expected_state = expected.state[expected_parameter]

    torch.testing.assert_close(actual_state["step"], expected_state["step"])
    torch.testing.assert_close(actual_state["exp_avg"], expected_state["exp_avg"])
    torch.testing.assert_close(actual_state["exp_avg_sq"], expected_state["exp_avg_sq"])
    assert actual.param_groups[0]["lr"] == expected.param_groups[0]["lr"]


def _assert_native_state_dicts_equal(actual: dict, expected: dict) -> None:
    assert actual["param_groups"] == expected["param_groups"]
    assert actual["state"].keys() == expected["state"].keys()
    for parameter_id in actual["state"]:
        assert actual["state"][parameter_id].keys() == expected["state"][parameter_id].keys()
        for state_name, actual_value in actual["state"][parameter_id].items():
            expected_value = expected["state"][parameter_id][state_name]
            if isinstance(actual_value, torch.Tensor):
                torch.testing.assert_close(actual_value, expected_value)
            else:
                assert actual_value == expected_value


def _make_peft_ep_checkpointer(tmp_path) -> Checkpointer:
    config = CheckpointingConfig(
        checkpoint_dir=tmp_path,
        model_save_format="safetensors",
        save_consolidated=False,
        is_peft=True,
    )
    return Checkpointer(config, dp_rank=0, tp_rank=0, pp_rank=0, moe_mesh=object())


def test_checkpointer_uses_native_state_for_all_optimizer_parts_with_ep_topology(tmp_path):
    model_parts, optimizers = _make_stepped_adam_parts()
    checkpointer = _make_peft_ep_checkpointer(tmp_path)

    with patch.object(checkpointer, "_do_save") as do_save:
        checkpointer.save_optimizer(optimizers, model_parts, str(tmp_path))

    state_dict = do_save.call_args.args[0]
    saved_parts = state_dict["optim"]["optimizer_parts"]
    assert len(saved_parts) == len(optimizers)
    for saved_part, optimizer in zip(saved_parts, optimizers, strict=True):
        _assert_native_state_dicts_equal(saved_part, optimizer.state_dict())


def test_native_single_optimizer_state_preserves_existing_checkpoint_schema():
    model_parts, optimizers = _make_stepped_adam_parts(count=1)
    state_dict = OptimizerState(
        model_parts[0],
        optimizers[0],
        is_peft=True,
        has_expert_parallelism=True,
    ).state_dict()

    assert state_dict["optim"].keys() == {"state", "param_groups"}


def test_native_multi_optimizer_state_dcp_round_trip(tmp_path):
    source_models, source_optimizers = _make_stepped_adam_parts()
    checkpointer = _make_peft_ep_checkpointer(tmp_path)
    checkpointer.save_optimizer(source_optimizers, source_models, str(tmp_path))

    target_models = [nn.Linear(2, 1, bias=False) for _ in source_models]
    target_optimizers = [torch.optim.AdamW(model.parameters(), lr=0.5) for model in target_models]
    checkpointer.load_optimizer(target_optimizers, target_models, str(tmp_path))

    for target, source in zip(target_optimizers, source_optimizers, strict=True):
        _assert_adam_states_equal(target, source)


def test_native_multi_optimizer_load_rejects_pipeline_part_count_mismatch():
    model_parts, optimizers = _make_stepped_adam_parts()
    optimizer_state = OptimizerState(
        model_parts,
        optimizers,
        is_peft=True,
        has_expert_parallelism=True,
    )
    state_dict = optimizer_state.state_dict()
    state_dict["optim"]["optimizer_parts"].pop()

    with pytest.raises(ValueError, match="checkpoint has 1 parts, current layout has 2"):
        optimizer_state.load_state_dict(state_dict)
