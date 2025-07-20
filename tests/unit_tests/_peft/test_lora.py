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

import pytest
import torch
import torch.nn as nn

from nemo_automodel.components._peft.lora import LinearLoRA, PeftConfig, apply_lora_to_linear_modules


class DummyModel(nn.Module):
    """A dummy neural network model with two linear layers used for testing LoRA injection."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(16, 16)
        self.linear2 = nn.Linear(16, 16)
        self.config = {}

    def forward(self, x):
        """Forward pass through two linear layers with ReLU activation in between."""
        x = self.linear1(x).relu()
        x = self.linear2(x)
        return x


class DummyModelNoConfig(nn.Module):
    """Same as DummyModel but without a `config` attribute."""

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
    """Provides a dummy input tensor for model testing."""
    return torch.randn(2, 16, requires_grad=True)


@pytest.fixture
def model():
    """Instantiates and returns a DummyModel instance."""
    return DummyModel()


@pytest.fixture
def model_no_config():
    """Instantiates a model that has no `config` attr."""
    return DummyModelNoConfig()


def test_lora_patch_on_model_without_config(model_no_config):
    """LoRA should still patch correctly even if the model lacks `config`."""
    apply_lora_to_linear_modules(model_no_config, PeftConfig(target_modules=["linear1"], dim=4, alpha=8))
    assert isinstance(model_no_config.linear1, LinearLoRA)
    assert not isinstance(model_no_config.linear2, LinearLoRA)


def test_backward_pass_without_config(dummy_input, model_no_config):
    """Backward pass must succeed on a model without `config`."""
    apply_lora_to_linear_modules(model_no_config, PeftConfig(target_modules=["linear1"], dim=4, alpha=8))
    out = model_no_config(dummy_input)
    loss = out.sum()
    loss.backward()

    grads = [p.grad for p in model_no_config.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)
    assert all(torch.isfinite(g).all() for g in grads if g is not None)


def test_lora_patch_applies_to_selected_module(model):
    """Tests that LoRA is only applied to specified target modules."""
    apply_lora_to_linear_modules(model, PeftConfig(target_modules=["linear1"], dim=4, alpha=8))
    assert isinstance(model.linear1, LinearLoRA)
    assert not isinstance(model.linear2, LinearLoRA)


def test_lora_patch_applies_to_selected_module_with_str_dtype(model):
    """Tests that LoRA is only applied to specified target modules."""
    apply_lora_to_linear_modules(
        model, PeftConfig(target_modules=["linear1"], dim=4, alpha=8, lora_dtype="torch.bfloat16")
    )
    assert isinstance(model.linear1, LinearLoRA)
    assert model.linear1.lora_A.weight.dtype == torch.bfloat16
    assert model.linear1.lora_B.weight.dtype == torch.bfloat16
    assert not isinstance(model.linear2, LinearLoRA)


def test_forward_output_consistency(dummy_input):
    """Verifies that model output shape remains the same after LoRA patching,
    but values change due to the added LoRA components.
    """
    base = DummyModel()
    model = DummyModel()
    apply_lora_to_linear_modules(model, PeftConfig(target_modules=["linear1"], dim=4, alpha=8))

    base.eval()
    model.eval()

    with torch.no_grad():
        out1 = base(dummy_input)
        out2 = model(dummy_input)

    assert out1.shape == out2.shape
    assert not torch.allclose(out1, out2), "Output should differ due to LoRA injection"


def test_backward_pass(dummy_input):
    """Checks that backpropagation works and gradients are correctly computed
    when LoRA is applied.
    """
    model = DummyModel()
    apply_lora_to_linear_modules(model, PeftConfig(target_modules=["linear1"], dim=4, alpha=8))
    output = model(dummy_input)
    loss = output.sum()
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None for g in grads), "Some parameters should receive gradients"
    assert all(torch.isfinite(g).all() for g in grads if g is not None), "Gradients should be finite"


def test_lora_layers_are_trainable():
    """Ensures that LoRA layers are trainable while base weights remain frozen."""
    base = nn.Linear(16, 16)
    lora = LinearLoRA(base, dim=4, alpha=8)

    assert lora.weight.requires_grad is False
    assert lora.lora_A.weight.requires_grad
    assert lora.lora_B.weight.requires_grad
    if lora.bias is not None:
        assert lora.bias.requires_grad is False


def test_dropout_pre_post_effects(dummy_input):
    """Tests that different dropout positions ('pre' vs 'post') lead to different outputs."""
    base = nn.Linear(16, 16)
    lora_pre = LinearLoRA(base, dim=4, alpha=8, dropout=0.5, dropout_position="pre")
    lora_post = LinearLoRA(base, dim=4, alpha=8, dropout=0.5, dropout_position="post")

    with torch.no_grad():
        lora_pre.lora_A.weight.uniform_()
        lora_pre.lora_B.weight.uniform_()

        lora_post.lora_A.weight.copy_(lora_pre.lora_A.weight)
        lora_post.lora_B.weight.copy_(lora_pre.lora_B.weight)

    lora_pre.train()
    lora_post.train()

    out_pre = lora_pre(dummy_input)
    out_post = lora_post(dummy_input)

    assert out_pre.shape == out_post.shape
    assert not torch.allclose(out_pre, out_post), "Dropout positions should affect output differently"


def test_apply_lora_respects_wildcard(model):
    """Validates that wildcard matching correctly applies LoRA to all matching modules."""
    apply_lora_to_linear_modules(model, PeftConfig(target_modules=[".*"], dim=4, alpha=8))
    assert isinstance(model.linear1, LinearLoRA)
    assert isinstance(model.linear2, LinearLoRA)


def test_no_patch_on_non_matching_module(model):
    """Confirms that no modules are patched if target pattern doesn't match any names."""
    apply_lora_to_linear_modules(model, PeftConfig(target_modules=["nonexistent_module"], dim=4, alpha=8))
    assert not isinstance(model.linear1, LinearLoRA)
    assert not isinstance(model.linear2, LinearLoRA)
