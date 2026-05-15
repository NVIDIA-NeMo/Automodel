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
import torch.nn.functional as F
from torch.autograd.graph import saved_tensors_hooks

from nemo_automodel.components._peft.lora import (
    LinearLoRA,
    LoRATritonFunction,
    PeftConfig,
    apply_lora_to_linear_modules,
    patch_linear_module,
)
from nemo_automodel.shared.import_utils import safe_import_te

HAS_TE, transformer_engine = safe_import_te()


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


def test_peft_config_memory_efficient_lora_round_trip():
    """PeftConfig should default memory-efficient LoRA on and preserve explicit overrides."""
    assert PeftConfig().use_memory_efficient_lora is True

    cfg = PeftConfig.from_dict({"use_memory_efficient_lora": False})
    assert cfg.use_memory_efficient_lora is False
    assert cfg.to_dict()["use_memory_efficient_lora"] is False


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


@pytest.mark.parametrize("input_shape", [(5, 16), (2, 3, 16)])
def test_memory_efficient_lora_matches_legacy_forward_and_backward(input_shape):
    """Custom autograd LoRA should match the legacy two-linear implementation."""
    torch.manual_seed(1234)
    scale = 2.0
    lora_dim = 4
    out_features = 12

    x = torch.randn(*input_shape, requires_grad=True)
    lora_A = torch.randn(lora_dim, input_shape[-1], requires_grad=True)
    lora_B = torch.randn(out_features, lora_dim, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    lora_A_ref = lora_A.detach().clone().requires_grad_(True)
    lora_B_ref = lora_B.detach().clone().requires_grad_(True)

    efficient = LoRATritonFunction.apply(x, lora_A, lora_B, scale, x.dtype, False)
    legacy = F.linear(F.linear(x_ref, lora_A_ref) * scale, lora_B_ref)

    grad = torch.randn_like(legacy)
    efficient.backward(grad)
    legacy.backward(grad)

    assert torch.allclose(efficient, legacy)
    assert torch.allclose(x.grad, x_ref.grad)
    assert torch.allclose(lora_A.grad, lora_A_ref.grad)
    assert torch.allclose(lora_B.grad, lora_B_ref.grad)


def test_memory_efficient_lora_saves_less_forward_state():
    """The custom autograd path should not save the intermediate x @ lora_A.T activation."""
    torch.manual_seed(1234)
    x = torch.randn(8, 16, requires_grad=True)
    lora_A = torch.randn(4, 16, requires_grad=True)
    lora_B = torch.randn(12, 4, requires_grad=True)
    scale = 2.0

    def collect_saved_tensors(fn):
        saved = []

        def pack_hook(tensor):
            saved.append(tuple(tensor.shape))
            return tensor

        with saved_tensors_hooks(pack_hook, lambda tensor: tensor):
            fn()
        return saved

    legacy_saved = collect_saved_tensors(lambda: F.linear(F.linear(x, lora_A) * scale, lora_B))
    efficient_saved = collect_saved_tensors(lambda: LoRATritonFunction.apply(x, lora_A, lora_B, scale, x.dtype, False))

    assert (8, 4) in legacy_saved
    assert (8, 4) not in efficient_saved
    assert sum(torch.tensor(shape).prod().item() for shape in efficient_saved) < sum(
        torch.tensor(shape).prod().item() for shape in legacy_saved
    )


def test_linear_lora_memory_efficient_flag_controls_saved_state():
    """LinearLoRA should use the memory-efficient autograd path when the flag is enabled."""
    torch.manual_seed(1234)
    base = nn.Linear(16, 12, bias=False)
    x = torch.randn(8, 16, requires_grad=True)
    legacy = LinearLoRA(base, dim=4, alpha=8, use_memory_efficient_lora=False)
    efficient = LinearLoRA(base, dim=4, alpha=8, use_memory_efficient_lora=True)

    def collect_saved_tensors(fn):
        saved = []

        def pack_hook(tensor):
            saved.append(tuple(tensor.shape))
            return tensor

        with saved_tensors_hooks(pack_hook, lambda tensor: tensor):
            fn()
        return saved

    legacy_saved = collect_saved_tensors(lambda: legacy(x))
    efficient_saved = collect_saved_tensors(lambda: efficient(x))

    assert (8, 4) in legacy_saved
    assert (8, 4) not in efficient_saved


def test_lora_layers_are_trainable():
    """Ensures that LoRA layers are trainable while base weights remain frozen."""
    base = nn.Linear(16, 16)
    lora = LinearLoRA(base, dim=4, alpha=8)

    assert lora.weight.requires_grad is False
    assert lora.lora_A.weight.requires_grad
    assert lora.lora_B.weight.requires_grad
    if lora.bias is not None:
        assert lora.bias.requires_grad is False


def test_dora_layers_are_trainable_and_forward_works(dummy_input):
    """Ensures DoRA adds a learnable magnitude vector and forward/backward succeed."""
    base = nn.Linear(16, 16)
    dora = LinearLoRA(base, dim=4, alpha=8, use_dora=True, dropout=0.0)

    assert dora.weight.requires_grad is False
    assert dora.lora_A.weight.requires_grad
    assert dora.lora_B.weight.requires_grad
    assert hasattr(dora, "lora_magnitude")
    assert dora.lora_magnitude.requires_grad

    out = dora(dummy_input)
    loss = out.sum()
    loss.backward()

    assert dora.lora_A.weight.grad is not None
    assert dora.lora_B.weight.grad is not None
    assert dora.lora_magnitude.grad is not None
    assert torch.isfinite(dora.lora_magnitude.grad).all()


def test_apply_lora_with_dora_patches_selected_module(model):
    """apply_lora_to_linear_modules should be able to patch a module with DoRA enabled."""
    apply_lora_to_linear_modules(model, PeftConfig(target_modules=["linear1"], dim=4, alpha=8, use_dora=True))
    assert isinstance(model.linear1, LinearLoRA)
    assert getattr(model.linear1, "use_dora", False) is True
    assert hasattr(model.linear1, "lora_magnitude")


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
    assert isinstance(model.linear1, nn.Linear)
    assert isinstance(model.linear2, nn.Linear)
    apply_lora_to_linear_modules(model, PeftConfig(target_modules=[".*"], dim=4, alpha=8))
    assert isinstance(model.linear1, LinearLoRA), type(model.linear1)
    assert isinstance(model.linear2, LinearLoRA)


def test_no_patch_on_non_matching_module(model):
    """Confirms that no modules are patched if target pattern doesn't match any names."""
    assert isinstance(model.linear1, nn.Linear)
    assert isinstance(model.linear2, nn.Linear)
    apply_lora_to_linear_modules(model, PeftConfig(target_modules=["nonexistent_module"], dim=4, alpha=8))
    assert not isinstance(model.linear1, LinearLoRA)
    assert not isinstance(model.linear2, LinearLoRA)


@pytest.mark.skipif(not HAS_TE or not torch.cuda.is_available(), reason="Transformer Engine or CUDA not available")
class TestTELinearLoRA:
    """Tests for LoRA patching of Transformer Engine Linear modules."""

    def test_patch_sets_super_fwd(self):
        """patch_linear_module should set super_fwd for TE Linear."""
        from transformer_engine.pytorch.module.linear import Linear as TELinear

        te_linear = TELinear(in_features=16, out_features=32, bias=False, params_dtype=torch.bfloat16).cuda()
        patched = patch_linear_module(te_linear, dim=4, alpha=8, use_triton=False)
        assert hasattr(patched, "super_fwd"), "super_fwd should be set for TE Linear"
        assert patched.super_fwd is not None
        assert patched.super_fwd != patched.forward

    def test_lora_adapters_are_te_linear(self):
        """lora_A and lora_B should be TE Linear when base module is TE Linear."""
        from transformer_engine.pytorch.module.linear import Linear as TELinear

        te_linear = TELinear(in_features=16, out_features=32, bias=False, params_dtype=torch.bfloat16).cuda()
        patched = patch_linear_module(te_linear, dim=4, alpha=8, use_triton=False)
        assert isinstance(patched.lora_A, TELinear), f"lora_A should be TE Linear, got {type(patched.lora_A)}"
        assert isinstance(patched.lora_B, TELinear), f"lora_B should be TE Linear, got {type(patched.lora_B)}"

    def test_forward_pass(self):
        """Patched TE Linear should produce valid output."""
        from transformer_engine.pytorch.module.linear import Linear as TELinear

        te_linear = TELinear(in_features=16, out_features=32, bias=False, params_dtype=torch.bfloat16).cuda()
        patched = patch_linear_module(te_linear, dim=4, alpha=8, use_triton=False)
        x = torch.randn(2, 16, device="cuda", dtype=torch.bfloat16)
        out = patched(x)
        assert out.shape == (2, 32), f"Expected shape (2, 32), got {out.shape}"
        assert torch.isfinite(out).all(), "Output contains non-finite values"

    def test_backward_pass(self):
        """Backward pass through patched TE Linear should produce gradients on LoRA params."""
        from transformer_engine.pytorch.module.linear import Linear as TELinear

        te_linear = TELinear(in_features=16, out_features=32, bias=False, params_dtype=torch.bfloat16).cuda()
        patched = patch_linear_module(te_linear, dim=4, alpha=8, use_triton=False)
        x = torch.randn(2, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        out = patched(x)
        out.sum().backward()
        assert patched.lora_A.weight.grad is not None, "lora_A should have gradients"
        assert patched.lora_B.weight.grad is not None, "lora_B should have gradients"
        assert torch.isfinite(patched.lora_A.weight.grad).all(), "lora_A gradients should be finite"
        assert torch.isfinite(patched.lora_B.weight.grad).all(), "lora_B gradients should be finite"
