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
from unittest.mock import MagicMock, patch
try:
    import grouped_gemm
except ImportError:
    grouped_gemm = None

from nemo_automodel.components.moe.layers import GroupedExperts, GroupedExpertsDeepEP, MoEConfig
from nemo_automodel.components._peft.lora_moe import GroupedExpertsLoRA, GroupedExpertsDeepEPLoRA
from nemo_automodel.components._peft.lora import patch_moe_module, apply_lora_to_linear_modules, PeftConfig


@pytest.fixture
def device():
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return torch.device("cpu")


@pytest.fixture
def moe_config():
    return MoEConfig(
        n_routed_experts=4,
        n_shared_experts=0,
        n_activated_experts=2,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="softmax",
        route_scale=1.0,
        dim=16,
        inter_dim=32,
        moe_inter_dim=32,
        norm_topk_prob=False,
        expert_activation="swiglu",
        dtype=torch.float32
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_grouped_experts_lora_init(moe_config, device):
    """Test initialization of GroupedExpertsLoRA, verifying shapes and frozen weights."""
    orig_experts = GroupedExperts(moe_config).to(device)
    # Initialize weights to avoid NaNs
    with torch.no_grad():
        orig_experts.init_weights(buffer_device=device)

    lora_experts = GroupedExpertsLoRA(orig_experts, lora_dim=4, alpha=8)

    assert isinstance(lora_experts, GroupedExpertsLoRA)
    assert lora_experts.lora_dim == 4
    assert lora_experts.scale == 2.0

    # Check shapes
    # lora_gate_and_up_A: [n_experts, in_dim, lora_dim] -> [4, 16, 4]
    assert lora_experts.lora_gate_and_up_A.shape == (4, 16, 4)
    # lora_gate_and_up_B: [n_experts, lora_dim, out_dim] -> [4, 4, 64]
    assert lora_experts.lora_gate_and_up_B.shape == (4, 4, 64) # 32 * 2
    # lora_down_A: [n_experts, inter_dim, lora_dim] -> [4, 32, 4]
    assert lora_experts.lora_down_A.shape == (4, 32, 4)
    # lora_down_B: [n_experts, lora_dim, out_dim] -> [4, 4, 16]
    assert lora_experts.lora_down_B.shape == (4, 4, 16)

    # Check requires_grad
    assert not lora_experts.gate_and_up_projs.requires_grad
    assert not lora_experts.down_projs.requires_grad
    assert lora_experts.lora_gate_and_up_A.requires_grad
    assert lora_experts.lora_gate_and_up_B.requires_grad


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_apply_lora_equivalence(moe_config, device):
    """Test that applying LoRA to a model maintains output equivalence upon initialization."""
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = GroupedExperts(moe_config)
            self.linear = nn.Linear(16, 16)

        def forward(self, x, token_mask, weights, indices):
            return self.experts(x, token_mask, weights, indices) + self.linear(x)

    model = MockModel().to(device)
    # Initialize weights
    with torch.no_grad():
        model.experts.init_weights(buffer_device=device)
        nn.init.normal_(model.linear.weight)
        nn.init.zeros_(model.linear.bias)

    # Mock input
    bs = 2
    seq_len = 5
    dim = 16
    model = model.to(device)
    x = torch.randn(bs * seq_len, dim, device=device)
    token_mask = torch.ones(bs * seq_len, dtype=torch.bool, device=device)
    weights = torch.rand(bs * seq_len, 2, device=device)
    indices = torch.randint(0, 4, (bs * seq_len, 2), device=device)

    # Baseline output
    with torch.no_grad():
        out_orig = model(x, token_mask, weights, indices)

    # Apply LoRA
    peft_config = PeftConfig(
        target_modules=["*experts*"],
        dim=4
    )
    apply_lora_to_linear_modules(model, peft_config)
    model = model.to(device)
    # LoRA output
    with torch.no_grad():
        out_lora = model(x, token_mask, weights, indices)

    assert torch.allclose(out_orig, out_lora, atol=1e-6)
    assert isinstance(model.experts, GroupedExpertsLoRA)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_grouped_experts_deepep_lora_init(moe_config, device):
    """Test initialization of GroupedExpertsDeepEPLoRA, verifying shapes."""
    orig_experts = GroupedExpertsDeepEP(moe_config).to(device)
    # Initialize weights
    with torch.no_grad():
        orig_experts.init_weights(buffer_device=device)

    lora_experts = GroupedExpertsDeepEPLoRA(orig_experts, lora_dim=4, alpha=8)

    assert isinstance(lora_experts, GroupedExpertsDeepEPLoRA)
    assert lora_experts.lora_dim == 4

    # Check shapes
    assert lora_experts.lora_gate_and_up_A.shape == (4, 16, 4)
    assert lora_experts.lora_gate_and_up_B.shape == (4, 4, 64)
    assert lora_experts.lora_down_A.shape == (4, 32, 4)
    assert lora_experts.lora_down_B.shape == (4, 4, 16)

    # Check requires_grad
    assert not lora_experts.gate_and_up_projs.requires_grad
    assert not lora_experts.down_projs.requires_grad
    assert lora_experts.lora_gate_and_up_A.requires_grad
    assert lora_experts.lora_gate_and_up_B.requires_grad


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_patch_moe_module(moe_config, device):
    """Test that patch_moe_module correctly wraps the original experts with the appropriate LoRA class."""
    orig_experts = GroupedExperts(moe_config).to(device)
    patched = patch_moe_module(orig_experts, dim=4)
    assert isinstance(patched, GroupedExpertsLoRA)

    orig_experts_deep = GroupedExpertsDeepEP(moe_config).to(device)
    patched_deep = patch_moe_module(orig_experts_deep, dim=4)
    assert isinstance(patched_deep, GroupedExpertsDeepEPLoRA)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_apply_lora_patching_logic(moe_config, device):
    """
    Test the patching logic of apply_lora_to_linear_modules.
    Verifies that:
    1. Exact name matching works for MoE modules.
    2. Wildcard matching works for MoE modules.
    3. Non-target modules (e.g., standard Linear layers not in target list) are NOT patched.
    """
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = GroupedExperts(moe_config)
            self.linear = nn.Linear(16, 16)

    model = MockModel().to(device)
    peft_config = PeftConfig(
        target_modules=["experts"],
        dim=4
    )

    count = apply_lora_to_linear_modules(model, peft_config)
    assert count == 1
    assert isinstance(model.experts, GroupedExpertsLoRA)
    assert isinstance(model.linear, nn.Linear) # Should not be patched

    # Test wildcard matching
    model = MockModel().to(device)
    peft_config = PeftConfig(
        target_modules=["*experts*"],
        dim=4
    )
    count = apply_lora_to_linear_modules(model, peft_config)
    assert count == 1
    assert isinstance(model.experts, GroupedExpertsLoRA)
    assert isinstance(model.linear, nn.Linear) # Should not be patched


class MockDeepEPDispatcher:
    """Mock dispatcher that simulates DeepEP's token permutation locally."""
    def token_permutation2(self, hidden_states, num_local_tokens, token_probs, token_indices):
        # Simply return the hidden states as if it was a single expert local dispatch
        # To make it compatible with ops.gmm, we need a tokens_per_expert tensor
        tokens_per_expert = torch.zeros(4, dtype=torch.long, device=hidden_states.device)
        return hidden_states, tokens_per_expert, token_probs

    def token_unpermutation(self, hidden_states):
        return hidden_states


@pytest.mark.skipif(
    grouped_gemm is None or not torch.cuda.is_available(),
    reason="Requires grouped_gemm and CUDA"
)
def test_grouped_experts_deepep_lora_forward_mocked(moe_config, device):
    """
    Test Forward pass of GroupedExpertsDeepEPLoRA using a Mock Dispatcher.

    This test verifies the LoRA-wrapped gated GEMM logic (using grouped_gemm kernels)
    independently of the DeepEP communication backend. This allows verification on
    non-Hopper (non-sm_90) hardware where DeepEP is physically unavailable.
    """
    moe_config.n_routed_experts = 4
    moe_config.dim = 16
    moe_config.moe_inter_dim = 32
    moe_config.dtype = torch.bfloat16

    orig_experts = GroupedExpertsDeepEP(moe_config).to(device).to(torch.bfloat16)
    # Initialize expert weights BEFORE creating LoRA module so they match after copy
    with torch.no_grad():
        orig_experts.init_weights(device)

    # Manually inject mock state since DeepEP init fails on non-Hopper hardware
    orig_experts.n_routed_experts = 4
    orig_experts.ep_size = 1

    lora_module = GroupedExpertsDeepEPLoRA(orig_experts, lora_dim=4).to(device).to(torch.bfloat16)
    mock_dispatcher = MockDeepEPDispatcher()

    # Mock tokens_per_expert for ops.gmm - needs to sum to num_tokens
    num_tokens = 8
    # One expert gets all tokens for simplicity
    tokens_per_expert = torch.tensor([num_tokens, 0, 0, 0], dtype=torch.long, device="cpu")

    # Capture deterministic data to return from the mock dispatcher
    dtype = torch.bfloat16
    permuted_x = torch.randn(num_tokens, 16, device=device).to(dtype)
    # permuted_probs should be 1D [num_tokens] because it's unsqueezed in forward
    permuted_probs = torch.ones(num_tokens, device=device).to(dtype)

    # Set the same mock on both modules to ensure they see the same "dispatched" data
    mock_dispatcher.token_permutation2 = MagicMock(
        return_value=(permuted_x, tokens_per_expert, permuted_probs)
    )
    lora_module.token_dispatcher = mock_dispatcher
    orig_experts.token_dispatcher = mock_dispatcher

    x = torch.randn(num_tokens, 16, device=device).to(dtype)
    # weights should also be [num_tokens, TopK] where TopK=1
    weights = torch.ones(num_tokens, 1, device=device).to(dtype)
    indices = torch.zeros(num_tokens, 1, dtype=torch.long, device=device)
    token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)

    # This will now reach the lora_module.forward -> ops.gmm calls!
    out = lora_module(x, token_mask, weights, indices)

    # Verify equivalence with zero LoRA weights (DeepEP LoRA B is zero-init by default)
    # GroupedExpertsDeepEP.forward hardcodes a call to .to_local() which regular
    # Parameters don't have. We surgicaly patch it only for this unit test.
    with torch.no_grad(), patch.object(torch.Tensor, "to_local", new=lambda self: self, create=True):
        out_orig = orig_experts(x, token_mask, weights, indices)

    assert out.shape == (num_tokens, 16)
    assert torch.allclose(out, out_orig, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_grouped_experts_lora_forward(moe_config, device):
    """Test forward pass of GroupedExpertsLoRA."""
    orig_experts = GroupedExperts(moe_config).to(device)
    with torch.no_grad():
        orig_experts.init_weights(buffer_device=device)

    lora_experts = GroupedExpertsLoRA(orig_experts, lora_dim=4, alpha=8).to(device)

    # Create input
    num_tokens = 10
    x = torch.randn(num_tokens, 16, device=device)
    token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
    weights = torch.rand(num_tokens, 2, device=device)
    indices = torch.randint(0, 4, (num_tokens, 2), device=device)

    # Forward pass
    out = lora_experts(x, token_mask, weights, indices)

    assert out.shape == (num_tokens, 16)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_lora_weight_initialization_xavier(moe_config, device):
    """Test Xavier initialization for LoRA weights."""
    orig_experts = GroupedExperts(moe_config).to(device)
    with torch.no_grad():
        orig_experts.init_weights(buffer_device=device)

    lora_experts = GroupedExpertsLoRA(orig_experts, lora_dim=4, alpha=8, lora_A_init_method="xavier").to(device)

    # Check that B matrices are zero-initialized
    assert torch.allclose(lora_experts.lora_gate_and_up_B, torch.zeros_like(lora_experts.lora_gate_and_up_B))
    assert torch.allclose(lora_experts.lora_down_B, torch.zeros_like(lora_experts.lora_down_B))

    # Check that A matrices are not zero (xavier initialized)
    assert not torch.allclose(lora_experts.lora_gate_and_up_A, torch.zeros_like(lora_experts.lora_gate_and_up_A))
    assert not torch.allclose(lora_experts.lora_down_A, torch.zeros_like(lora_experts.lora_down_A))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_lora_weight_initialization_kaiming(moe_config, device):
    """Test Kaiming initialization for LoRA weights."""
    orig_experts = GroupedExperts(moe_config).to(device)
    with torch.no_grad():
        orig_experts.init_weights(buffer_device=device)

    lora_experts = GroupedExpertsLoRA(orig_experts, lora_dim=4, alpha=8, lora_A_init_method="kaiming").to(device)

    # Check that B matrices are zero-initialized
    assert torch.allclose(lora_experts.lora_gate_and_up_B, torch.zeros_like(lora_experts.lora_gate_and_up_B))
    assert torch.allclose(lora_experts.lora_down_B, torch.zeros_like(lora_experts.lora_down_B))

    # Check that A matrices are not zero (kaiming initialized)
    assert not torch.allclose(lora_experts.lora_gate_and_up_A, torch.zeros_like(lora_experts.lora_gate_and_up_A))
    assert not torch.allclose(lora_experts.lora_down_A, torch.zeros_like(lora_experts.lora_down_A))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_lora_scale_parameter(moe_config, device):
    """Test that LoRA scale parameter is correctly computed and applied."""
    orig_experts = GroupedExperts(moe_config).to(device)
    with torch.no_grad():
        orig_experts.init_weights(buffer_device=device)

    # Test different alpha/dim combinations
    lora_dim = 4
    alpha = 16
    lora_experts = GroupedExpertsLoRA(orig_experts, lora_dim=lora_dim, alpha=alpha).to(device)

    expected_scale = alpha / lora_dim
    assert lora_experts.scale == expected_scale


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_lora_gradient_flow(moe_config, device):
    """Test that gradients flow only through LoRA parameters, not base weights."""
    orig_experts = GroupedExperts(moe_config).to(device)
    with torch.no_grad():
        orig_experts.init_weights(buffer_device=device)

    lora_experts = GroupedExpertsLoRA(orig_experts, lora_dim=4, alpha=8).to(device)

    # Create input
    num_tokens = 10
    x = torch.randn(num_tokens, 16, device=device, requires_grad=True)
    token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
    weights = torch.rand(num_tokens, 2, device=device)
    indices = torch.randint(0, 4, (num_tokens, 2), device=device)

    # Forward pass
    out = lora_experts(x, token_mask, weights, indices)
    loss = out.sum()

    # Backward pass
    loss.backward()

    # Check gradients
    assert lora_experts.gate_and_up_projs.grad is None
    assert lora_experts.down_projs.grad is None
    assert lora_experts.lora_gate_and_up_A.grad is not None
    assert lora_experts.lora_gate_and_up_B.grad is not None
    assert lora_experts.lora_down_A.grad is not None
    assert lora_experts.lora_down_B.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_lora_zero_active_experts(moe_config, device):
    """Test the edge case where no tokens are routed to any local experts."""
    orig_experts = GroupedExperts(moe_config).to(device)
    with torch.no_grad():
        orig_experts.init_weights(buffer_device=device)

    lora_experts = GroupedExpertsLoRA(orig_experts, lora_dim=4, alpha=8).to(device)

    # Create input where no tokens match any expert
    num_tokens = 10
    x = torch.randn(num_tokens, 16, device=device, requires_grad=True)
    token_mask = torch.zeros(num_tokens, dtype=torch.bool, device=device)  # All masked out
    weights = torch.rand(num_tokens, 2, device=device)
    indices = torch.randint(0, 4, (num_tokens, 2), device=device)

    # Forward pass - should handle gracefully
    out = lora_experts(x, token_mask, weights, indices)
    assert out.shape == (num_tokens, 16)

    # Test backward pass for gradient flow
    loss = out.sum()
    loss.backward()

    # Check that gradients exist for LoRA parameters
    assert lora_experts.lora_gate_and_up_A.grad is not None
    assert lora_experts.lora_gate_and_up_B.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_lora_with_expert_bias(device):
    """Test LoRA with expert bias enabled."""
    moe_config = MoEConfig(
        n_routed_experts=4,
        n_shared_experts=0,
        n_activated_experts=2,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="softmax",
        route_scale=1.0,
        dim=16,
        inter_dim=32,
        moe_inter_dim=32,
        norm_topk_prob=False,
        expert_activation="swiglu",
        dtype=torch.float32,
        expert_bias=True  # Enable bias
    )

    orig_experts = GroupedExperts(moe_config).to(device)
    with torch.no_grad():
        orig_experts.init_weights(buffer_device=device)

    lora_experts = GroupedExpertsLoRA(orig_experts, lora_dim=4, alpha=8).to(device)

    # Check that bias parameters exist and are frozen
    assert hasattr(lora_experts, 'gate_up_proj_bias')
    assert hasattr(lora_experts, 'down_proj_bias')
    assert not lora_experts.gate_up_proj_bias.requires_grad
    assert not lora_experts.down_proj_bias.requires_grad

    # Test forward pass
    num_tokens = 10
    x = torch.randn(num_tokens, 16, device=device)
    token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
    weights = torch.rand(num_tokens, 2, device=device)
    indices = torch.randint(0, 4, (num_tokens, 2), device=device)

    out = lora_experts(x, token_mask, weights, indices)
    assert out.shape == (num_tokens, 16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_quick_geglu_activation_with_lora(device):
    """Test GroupedExpertsLoRA with QuickGEGLU activation."""
    moe_config = MoEConfig(
        n_routed_experts=4,
        n_shared_experts=0,
        n_activated_experts=2,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="softmax",
        route_scale=1.0,
        dim=16,
        inter_dim=32,
        moe_inter_dim=32,
        norm_topk_prob=False,
        expert_activation="quick_geglu",  # Use QuickGEGLU
        activation_alpha=1.702,
        activation_limit=7.0,
        dtype=torch.float32
    )

    orig_experts = GroupedExperts(moe_config).to(device)
    with torch.no_grad():
        orig_experts.init_weights(buffer_device=device)

    lora_experts = GroupedExpertsLoRA(orig_experts, lora_dim=4, alpha=8).to(device)

    # Test forward pass
    num_tokens = 10
    x = torch.randn(num_tokens, 16, device=device)
    token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
    weights = torch.rand(num_tokens, 2, device=device)
    indices = torch.randint(0, 4, (num_tokens, 2), device=device)

    out = lora_experts(x, token_mask, weights, indices)
    assert out.shape == (num_tokens, 16)
    assert not torch.isnan(out).any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_lora_dtype_conversion(moe_config, device):
    """Test LoRA with explicit dtype specification."""
    orig_experts = GroupedExperts(moe_config).to(device)
    with torch.no_grad():
        orig_experts.init_weights(buffer_device=device)

    # Create LoRA with explicit bfloat16 dtype
    if device.type == 'cuda' and torch.cuda.is_bf16_supported():
        lora_experts = GroupedExpertsLoRA(orig_experts, lora_dim=4, alpha=8, lora_dtype="bfloat16").to(device)

        # Check that LoRA weights have the correct dtype
        assert lora_experts.lora_gate_and_up_A.dtype == torch.bfloat16
        assert lora_experts.lora_gate_and_up_B.dtype == torch.bfloat16
        assert lora_experts.lora_down_A.dtype == torch.bfloat16
        assert lora_experts.lora_down_B.dtype == torch.bfloat16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_lora_backward_pass_values(moe_config, device):
    """Test that LoRA backward pass produces non-zero gradients."""
    orig_experts = GroupedExperts(moe_config).to(device)
    with torch.no_grad():
        orig_experts.init_weights(buffer_device=device)

    lora_experts = GroupedExpertsLoRA(orig_experts, lora_dim=4, alpha=8).to(device)

    # Manually set LoRA B weights to non-zero to ensure LoRA contributes to output
    with torch.no_grad():
        lora_experts.lora_gate_and_up_B.normal_(0, 0.01)
        lora_experts.lora_down_B.normal_(0, 0.01)

    # Create input
    num_tokens = 10
    x = torch.randn(num_tokens, 16, device=device, requires_grad=True)
    token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
    weights = torch.rand(num_tokens, 2, device=device)
    indices = torch.randint(0, 4, (num_tokens, 2), device=device)

    # Forward and backward
    out = lora_experts(x, token_mask, weights, indices)
    loss = out.sum()
    loss.backward()

    # Check that gradients are non-zero
    assert not torch.allclose(lora_experts.lora_gate_and_up_A.grad, torch.zeros_like(lora_experts.lora_gate_and_up_A.grad))
    assert not torch.allclose(lora_experts.lora_gate_and_up_B.grad, torch.zeros_like(lora_experts.lora_gate_and_up_B.grad))


@pytest.mark.skipif(
    grouped_gemm is None or not torch.cuda.is_available(),
    reason="Requires grouped_gemm and CUDA"
)
def test_deepep_lora_zero_tokens(moe_config, device):
    """Test DeepEP LoRA forward pass with zero tokens routed to experts."""
    moe_config.n_routed_experts = 4
    moe_config.dim = 16
    moe_config.moe_inter_dim = 32
    moe_config.dtype = torch.bfloat16

    orig_experts = GroupedExpertsDeepEP(moe_config).to(device).to(torch.bfloat16)
    with torch.no_grad():
        orig_experts.init_weights(device)

    orig_experts.n_routed_experts = 4
    orig_experts.ep_size = 1

    lora_module = GroupedExpertsDeepEPLoRA(orig_experts, lora_dim=4).to(device).to(torch.bfloat16)
    mock_dispatcher = MockDeepEPDispatcher()

    num_tokens = 8
    # All experts get zero tokens
    tokens_per_expert = torch.tensor([0, 0, 0, 0], dtype=torch.long, device="cpu")

    dtype = torch.bfloat16
    permuted_x = torch.randn(num_tokens, 16, device=device).to(dtype)
    permuted_probs = torch.ones(num_tokens, device=device).to(dtype)

    mock_dispatcher.token_permutation2 = MagicMock(
        return_value=(permuted_x, tokens_per_expert, permuted_probs)
    )
    lora_module.token_dispatcher = mock_dispatcher

    x = torch.randn(num_tokens, 16, device=device).to(dtype)
    weights = torch.ones(num_tokens, 1, device=device).to(dtype)
    indices = torch.zeros(num_tokens, 1, dtype=torch.long, device=device)
    token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)

    # Should handle zero tokens gracefully
    out = lora_module(x, token_mask, weights, indices)
    assert out.shape == (num_tokens, 16)
