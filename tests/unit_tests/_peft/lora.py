# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import torch
import torch.nn as nn
import pytest
from torch.nn.utils import parameters_to_vector

from your_module import LinearLoRA, apply_lora_to_linear_modules  # Replace `your_module` with actual module name


# Dummy model to test patching
class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(16, 16)
        self.linear2 = nn.Linear(16, 16)

    def forward(self, x):
        x = self.linear1(x).relu()
        x = self.linear2(x)
        return x


@pytest.fixture
def dummy_input():
    return torch.randn(2, 16, requires_grad=True)


@pytest.fixture
def model():
    return DummyModel()


def test_lora_patch_applies_to_selected_module(model):
    apply_lora_to_linear_modules(model, target_modules=["linear1"], dim=4, alpha=8)
    assert isinstance(model.linear1, LinearLoRA)
    assert not isinstance(model.linear2, LinearLoRA)


def test_forward_output_consistency(dummy_input):
    base = DummyModel()
    model = DummyModel()
    apply_lora_to_linear_modules(model, target_modules=["linear1"], dim=4, alpha=8)

    base.eval()
    model.eval()

    with torch.no_grad():
        out1 = base(dummy_input)
        out2 = model(dummy_input)

    assert out1.shape == out2.shape
    assert not torch.allclose(out1, out2), "Output should differ due to LoRA injection"


def test_backward_pass(dummy_input):
    model = DummyModel()
    apply_lora_to_linear_modules(model, target_modules=["linear1"], dim=4, alpha=8)
    output = model(dummy_input)
    loss = output.sum()
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None for g in grads), "Some parameters should receive gradients"
    assert all(torch.isfinite(g).all() for g in grads if g is not None), "Gradients should be finite"


def test_lora_layers_are_trainable():
    base = nn.Linear(16, 16)
    lora = LinearLoRA(base, dim=4, alpha=8)

    assert lora.weight.requires_grad is False
    assert lora.lora_a.weight.requires_grad
    assert lora.lora_b.weight.requires_grad
    if lora.bias is not None:
        assert lora.bias.requires_grad is False


def test_dropout_pre_post_effects(dummy_input):
    base = nn.Linear(16, 16)
    lora_pre = LinearLoRA(base, dim=4, alpha=8, dropout=0.5, dropout_position='pre')
    lora_post = LinearLoRA(base, dim=4, alpha=8, dropout=0.5, dropout_position='post')

    lora_pre.train()
    lora_post.train()

    out_pre = lora_pre(dummy_input)
    out_post = lora_post(dummy_input)

    assert out_pre.shape == out_post.shape
    assert not torch.allclose(out_pre, out_post), "Dropout positions should affect output differently"


def test_apply_lora_respects_wildcard(model):
    apply_lora_to_linear_modules(model, target_modules=[".*"], dim=4, alpha=8)
    assert isinstance(model.linear1, LinearLoRA)
    assert isinstance(model.linear2, LinearLoRA)


def test_no_patch_on_non_matching_module(model):
    apply_lora_to_linear_modules(model, target_modules=["nonexistent_module"], dim=4, alpha=8)
    assert not isinstance(model.linear1, LinearLoRA)
    assert not isinstance(model.linear2, LinearLoRA)
