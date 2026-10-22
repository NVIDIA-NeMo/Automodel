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

"""Exact-name freezing shares the existing trainability and storage lifecycle."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nemo_automodel.components.utils.model_utils import FreezeConfig, apply_parameter_freezing, parse_freeze_config


@pytest.mark.parametrize("names", [["weight", "weight"], [None], [""], "weight", None])
def test_invalid_parameter_exclusions_are_rejected(names):
    with pytest.raises(ValueError, match="freeze_parameters"):
        parse_freeze_config({"freeze_parameters": names})


def test_empty_exclusions_leave_unlisted_parameters_trainable():
    model = nn.Linear(2, 2)
    config = parse_freeze_config({"freeze_parameters": []})
    apply_parameter_freezing(model, config)
    assert all(p.requires_grad for p in model.parameters())
    assert config.freeze_parameters == ()
    assert config.has_generic_selectors()


def test_exact_freezing_preserves_dtype_and_tied_alias_identity():
    model = nn.Module()
    model.encoder = nn.Linear(2, 2, bias=False, dtype=torch.float32)
    model.decoder = nn.Linear(2, 2, bias=False)
    model.decoder.weight = model.encoder.weight
    model.other = nn.Linear(2, 1, dtype=torch.bfloat16)
    parameter = model.encoder.weight
    parameter._no_weight_decay = True
    config = parse_freeze_config({"freeze_parameters": ["decoder.weight"]})
    apply_parameter_freezing(model, config)
    assert model.encoder.weight is model.decoder.weight is parameter
    assert not parameter.requires_grad and parameter.dtype == torch.float32 and parameter._no_weight_decay
    assert all(p.requires_grad and p.dtype == torch.bfloat16 for p in model.other.parameters())


@pytest.mark.parametrize("selector", ["freeze_parameters", "unfreeze_parameters"])
def test_unknown_names_fail_before_any_trainability_mutation(selector):
    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))
    config = parse_freeze_config({"freeze_modules": [{"path": "0"}], selector: ["missing"]})
    with pytest.raises(ValueError, match="Unknown selected parameters"):
        apply_parameter_freezing(model, config)
    assert all(p.requires_grad for p in model.parameters())


def test_parameter_exclusions_win_over_module_unfreezes():
    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))
    config = parse_freeze_config(
        {
            "freeze_modules": [{"path": "0"}],
            "unfreeze_modules": [{"path": "0"}],
            "freeze_parameters": ["0.weight"],
        }
    )
    apply_parameter_freezing(model, config)
    assert not model[0].weight.requires_grad
    assert model[0].bias.requires_grad
    assert all(p.requires_grad for p in model[1].parameters())


def test_unlisted_parameters_keep_existing_frozen_state():
    model = nn.Linear(2, 2)
    model.bias.requires_grad_(False)
    apply_parameter_freezing(model, FreezeConfig(freeze_parameters=("weight",)))
    assert not model.weight.requires_grad and not model.bias.requires_grad


def test_exact_unfreeze_does_not_train_other_parameters_in_the_same_module():
    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))
    config = parse_freeze_config({"freeze_modules": [{"glob": "*"}], "unfreeze_parameters": ["0.weight"]})
    apply_parameter_freezing(model, config)
    assert {name for name, p in model.named_parameters() if p.requires_grad} == {"0.weight"}
    model(torch.ones(1, 2)).sum().backward()
    assert model[0].weight.grad is not None
    assert model[0].bias.grad is None
    assert model[1].weight.grad is None


def test_exact_freeze_wins_over_exact_unfreeze():
    model = nn.Linear(2, 2)
    apply_parameter_freezing(model, FreezeConfig(unfreeze_parameters=("weight",), freeze_parameters=("weight",)))
    assert not model.weight.requires_grad
    assert model.bias.requires_grad


@pytest.mark.parametrize("names", [["weight", "weight"], [None], [""], "weight", None])
def test_invalid_parameter_unfreezes_are_rejected(names):
    with pytest.raises(ValueError, match="unfreeze_parameters"):
        parse_freeze_config({"unfreeze_parameters": names})


def test_parameter_exclusions_rebind_across_wrappers_and_pipeline_partitions():
    model = nn.Module()
    model.layer = nn.Module()
    model.layer._checkpoint_wrapped_module = nn.Linear(2, 2)
    config = FreezeConfig(freeze_parameters=("layer.weight", "other_stage.weight"))
    apply_parameter_freezing(model, config, strict=False)
    assert not model.layer._checkpoint_wrapped_module.weight.requires_grad
    assert model.layer._checkpoint_wrapped_module.bias.requires_grad
    untouched = nn.Linear(2, 2)
    apply_parameter_freezing(untouched, FreezeConfig(freeze_parameters=("other_stage.weight",)), strict=False)
    assert all(p.requires_grad for p in untouched.parameters())


def test_pre_shard_rebinding_freezes_only_explicit_exclusions():
    from nemo_automodel._transformers.infrastructure import _prepare_trainability_before_sharding

    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))
    config = FreezeConfig(freeze_parameters=("1.weight",))
    _prepare_trainability_before_sharding(
        model, peft_enabled=False, freeze_config=config, frozen_compute_dtype=torch.bfloat16, strict=True
    )
    model[1].weight = nn.Parameter(torch.ones_like(model[1].weight, dtype=torch.float32))
    model.register_parameter("new_weight", nn.Parameter(torch.ones(1)))
    _prepare_trainability_before_sharding(
        model, peft_enabled=False, freeze_config=config, frozen_compute_dtype=torch.bfloat16, strict=False
    )
    assert not model[1].weight.requires_grad and model[1].weight.dtype == torch.bfloat16
    assert model[0].weight.requires_grad and model[0].weight.dtype == torch.float32
    assert model[1].bias.requires_grad and model[1].bias.dtype == torch.float32
    assert model.new_weight.requires_grad and model.new_weight.dtype == torch.float32


def test_pre_shard_allowlist_rebinds_replaced_parameters_and_keeps_fp32_values():
    from nemo_automodel._transformers.infrastructure import _prepare_trainability_before_sharding

    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))
    config = parse_freeze_config({"freeze_modules": [{"glob": "*"}], "unfreeze_parameters": ["0.weight"]})
    expected = torch.full_like(model[0].weight, 0.1234567)
    assert not torch.equal(expected, expected.bfloat16().float())
    for strict in (True, False):
        model[0].weight = nn.Parameter(expected.clone(), requires_grad=False)
        model[1].weight = nn.Parameter(torch.ones_like(model[1].weight, dtype=torch.float32))
        _prepare_trainability_before_sharding(
            model, peft_enabled=False, freeze_config=config, frozen_compute_dtype=torch.bfloat16, strict=strict
        )
        assert {name for name, p in model.named_parameters() if p.requires_grad} == {"0.weight"}
        torch.testing.assert_close(model[0].weight, expected, rtol=0, atol=0)
        assert all(p.dtype == torch.bfloat16 for name, p in model.named_parameters() if name != "0.weight")


@pytest.mark.parametrize("selector", ["freeze_parameters", "unfreeze_parameters"])
def test_parameter_exclusions_reject_unsupported_te_expert_renumbering(selector):
    from nemo_automodel._transformers.infrastructure import _validate_parameter_freezing_backend
    from nemo_automodel.components.moe.experts import GroupedExpertsTE

    model = nn.Linear(2, 2)
    expert = GroupedExpertsTE.__new__(GroupedExpertsTE)
    nn.Module.__init__(expert)
    expert.register_parameter("weight0", nn.Parameter(torch.ones(2, 2)))
    model.experts = expert
    with pytest.raises(ValueError, match="TE expert parameters"):
        _validate_parameter_freezing_backend(
            model, SimpleNamespace(ep_size=2), parse_freeze_config({selector: ["experts.weight0"]})
        )

    _validate_parameter_freezing_backend(model, SimpleNamespace(ep_size=2), FreezeConfig(freeze_parameters=("weight",)))
