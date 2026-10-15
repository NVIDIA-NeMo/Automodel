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

"""Regression coverage for the immutable discrete selector training contract."""

import torch

from nemo_automodel.components.models.common.utils import cast_frozen_modules_to_compute_dtype
from nemo_automodel.components.optim.optimizer import _trainable_params_or_groups
from tests.unit_tests.models.step5.test_model import make_model


def test_indexer_constructor_is_frozen_without_changing_registered_state():
    model = make_model()
    indexer = model.model.layers["3"].self_attn.indexer
    assert set(dict(indexer.named_parameters())) == {
        "q_proj.weight",
        "k_proj.weight",
        "z_proj.weight",
        "w_proj.weight",
        "q_norm.weight",
        "k_norm.weight",
        "k_norm.bias",
    }
    assert all(not parameter.requires_grad for parameter in indexer.parameters())
    assert all(parameter.requires_grad for name, parameter in model.named_parameters() if ".indexer." not in name)
    assert set(dict(indexer.named_parameters())).issubset(indexer.state_dict())
    assert indexer.ssmax_s.dtype == torch.float32
    assert all(parameter.dtype == torch.float32 for parameter in indexer.parameters())


def test_optimizer_exclusion_preserves_indexer_even_with_materialized_zero_gradients():
    model = make_model()
    initial = {name: value.detach().clone() for name, value in model.named_parameters()}
    parameters = _trainable_params_or_groups(model, [])
    optimizer = torch.optim.AdamW(parameters, lr=0.01, weight_decay=0.1)
    indexer = model.model.layers["3"].self_attn.indexer
    assert not set(indexer.parameters()).intersection(parameters)
    for _ in range(2):
        for parameter in model.parameters():
            parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        for name, parameter in model.named_parameters():
            if ".indexer." in name:
                assert torch.equal(parameter, initial[name]), name
                assert parameter not in optimizer.state
    assert not torch.equal(model.model.layers["3"].moe.gate.weight, initial["model.layers.3.moe.gate.weight"])


def test_frozen_cast_keeps_strict_indexer_fp32_values_and_rotary_contract():
    model = make_model()
    indexer = model.model.layers["3"].self_attn.indexer
    with torch.no_grad():
        indexer.w_proj.weight.fill_(0.1234567)
        indexer.q_norm.weight.fill_(0.2345678)
        indexer.k_norm.weight.fill_(1.1234567)
        indexer.k_norm.bias.fill_(0.3456789)
        indexer.ssmax_s.fill_(0.456789)
    before = {name: value.detach().clone() for name, value in indexer.state_dict().items()}
    cast_frozen_modules_to_compute_dtype(model, torch.bfloat16)
    for name, value in indexer.state_dict().items():
        if name in ("q_proj.weight", "k_proj.weight", "z_proj.weight"):
            assert value.dtype == torch.bfloat16
        else:
            assert value.dtype == before[name].dtype, name
            assert torch.equal(value, before[name]), name
    assert all(not parameter.requires_grad for parameter in indexer.parameters())
