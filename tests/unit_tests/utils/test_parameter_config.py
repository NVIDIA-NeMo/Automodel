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

import pytest
import torch
from torch import nn

from nemo_automodel.components.utils.parameter_config import ParameterConfig, apply_parameter_config


def test_surface_precision_precedes_fp32_checkpoint_load():
    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1)).to(torch.bfloat16)
    frozen_before = model[1].weight.detach().clone()
    config = ParameterConfig(trainable_names=("0.weight", "0.bias"))
    apply_parameter_config(model, config)
    assert model[0].weight.dtype == torch.float32
    assert model[0].bias.requires_grad
    assert not model[1].weight.requires_grad
    assert model[1].weight.dtype == torch.bfloat16
    assert torch.equal(model[1].weight, frozen_before)
    source = model.state_dict()
    source["0.weight"] = torch.full((2, 2), 0.1234567, dtype=torch.float32)
    model.load_state_dict(source)
    assert torch.equal(model[0].weight, source["0.weight"])


def test_surface_on_meta_before_materialization():
    model = nn.Linear(2, 2, device="meta", dtype=torch.bfloat16)
    apply_parameter_config(model, ParameterConfig(("weight",)))
    assert model.weight.dtype == torch.float32
    assert model.bias.dtype == torch.bfloat16
    assert model.weight.is_meta
    assert not model.bias.requires_grad


@pytest.mark.parametrize("names", [(), ("weight", "weight"), ("missing",)])
def test_surface_rejects_bad_names_before_mutation(names):
    model = nn.Linear(2, 2, dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        apply_parameter_config(model, ParameterConfig(names))
    assert all(p.requires_grad and p.dtype == torch.bfloat16 for p in model.parameters())


def test_explicit_frozen_precision_is_applied_before_loading():
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Linear(2, 1))
    config = ParameterConfig(("0.weight", "0.bias"), torch.float32, torch.bfloat16)
    apply_parameter_config(model, config)
    assert all(parameter.dtype == torch.float32 for parameter in model[0].parameters())
    assert all(parameter.dtype == torch.bfloat16 and not parameter.requires_grad for parameter in model[1].parameters())


def test_tied_parameters_remain_shared_after_selection_and_loading():
    model = nn.Module()
    model.encoder = nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
    model.decoder = nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
    model.decoder.weight = model.encoder.weight
    model.encoder.register_parameter("alias", model.encoder.weight)
    original = model.encoder.weight
    original._no_weight_decay = True
    apply_parameter_config(model, ParameterConfig(("encoder.weight",)))
    assert model.encoder.weight is model.decoder.weight is model.encoder.alias is original
    assert model.encoder.weight._no_weight_decay is True
    source = torch.full((2, 2), 0.1234567, dtype=torch.float32)
    model.load_state_dict({name: source for name in model.state_dict()})
    torch.testing.assert_close(model.decoder.weight, source, rtol=0, atol=0)


@pytest.mark.parametrize("dtype_field", ["trainable_dtype", "frozen_dtype"])
def test_invalid_storage_precision_fails_before_mutation(dtype_field):
    model = nn.Linear(2, 2, dtype=torch.bfloat16)
    config = ParameterConfig(("weight",), **{dtype_field: torch.int32})
    with pytest.raises(ValueError):
        apply_parameter_config(model, config)
    assert all(p.requires_grad and p.dtype == torch.bfloat16 for p in model.parameters())


def test_optimizer_updates_only_the_selected_surface():
    model = nn.Sequential(nn.Linear(2, 2), nn.Tanh(), nn.Linear(2, 1))
    apply_parameter_config(model, ParameterConfig(("0.weight", "0.bias")))
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=0.1)
    loss = (model(torch.ones(3, 2)) - 2).square().mean()
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
    for name, p in model.named_parameters():
        assert bool(torch.equal(p, before[name])) is (not p.requires_grad)
        if not p.requires_grad:
            assert p.grad is None
