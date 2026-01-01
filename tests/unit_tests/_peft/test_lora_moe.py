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

from nemo_automodel.components.moe.layers import GroupedExperts, GroupedExpertsDeepEP, MoEConfig
from nemo_automodel.components._peft.lora_moe import GroupedExpertsLoRA, GroupedExpertsDeepEPLoRA
from nemo_automodel.components._peft.lora import patch_moe_module, apply_lora_to_linear_modules, PeftConfig

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

def test_grouped_experts_lora_init(moe_config):
    """Test initialization of GroupedExpertsLoRA, verifying shapes and frozen weights."""
    orig_experts = GroupedExperts(moe_config)
    # Initialize weights to avoid NaNs
    with torch.no_grad():
        orig_experts.init_weights(buffer_device=torch.device("cpu"))
    
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

def test_apply_lora_equivalence(moe_config):
    """Test that applying LoRA to a model maintains output equivalence upon initialization."""
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = GroupedExperts(moe_config)
            self.linear = nn.Linear(16, 16)

        def forward(self, x, token_mask, weights, indices):
            return self.experts(x, token_mask, weights, indices) + self.linear(x)

    model = MockModel()
    # Initialize weights
    with torch.no_grad():
        model.experts.init_weights(buffer_device=torch.device("cpu"))
        nn.init.normal_(model.linear.weight)
        nn.init.zeros_(model.linear.bias)

    # Mock input
    bs = 2
    seq_len = 5
    dim = 16
    x = torch.randn(bs * seq_len, dim)
    token_mask = torch.ones(bs * seq_len, dtype=torch.bool)
    weights = torch.rand(bs * seq_len, 2)
    indices = torch.randint(0, 4, (bs * seq_len, 2))

    # Baseline output
    with torch.no_grad():
        out_orig = model(x, token_mask, weights, indices)

    # Apply LoRA
    peft_config = PeftConfig(
        target_modules=["*experts*"],
        dim=4
    )
    apply_lora_to_linear_modules(model, peft_config)
    
    # LoRA output
    with torch.no_grad():
        out_lora = model(x, token_mask, weights, indices)

    assert torch.allclose(out_orig, out_lora, atol=1e-6)
    assert isinstance(model.experts, GroupedExpertsLoRA)

def test_grouped_experts_deepep_lora_init(moe_config):
    """Test initialization of GroupedExpertsDeepEPLoRA, verifying shapes."""
    orig_experts = GroupedExpertsDeepEP(moe_config)
    # Initialize weights
    with torch.no_grad():
        orig_experts.init_weights(buffer_device=torch.device("cpu"))

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

try:
    import grouped_gemm
except ImportError:
    grouped_gemm = None

@pytest.mark.skipif(
    grouped_gemm is None or not torch.cuda.is_available(),
    reason="Requires grouped_gemm and CUDA"
)
def test_grouped_experts_deepep_lora_forward_real(moe_config):
    """Test forward pass correctness of GroupedExpertsDeepEPLoRA using real CUDA kernels (skipped if unavailable)."""
    # Setup for real execution
    device = torch.device("cuda")
    moe_config.dtype = torch.float16 # DeepEP usually requires half precision
    
    orig_experts = GroupedExpertsDeepEP(moe_config).to(device)
    
    # Mock DeviceMesh and group wrapper
    mock_mesh = MagicMock()
    
    # Use real group if running under torchrun (required for DeepEP CUDA kernels)
    if torch.distributed.is_initialized():
        group = torch.distributed.group.WORLD
        mock_mesh.size.return_value = torch.distributed.get_world_size()
        mock_mesh.get_local_rank.return_value = torch.distributed.get_rank()
        mock_mesh.get_group.return_value = group
    else:
        # Fallback to mock for offline skip-checks
        mock_group = MagicMock()
        mock_group.size.return_value = 2
        mock_mesh.size.return_value = 2
        mock_mesh.get_local_rank.return_value = 0
        mock_mesh.get_group.return_value = mock_group
    
    # Let's try to initialize.
    try:
        orig_experts.init_token_dispatcher(mock_mesh)
    except Exception as e:
        pytest.skip(f"Failed to init token dispatcher (likely needs distributed env): {e}")

    # Initialize weights
    with torch.no_grad():
        orig_experts.init_weights(buffer_device=device)
    
    lora_experts = GroupedExpertsDeepEPLoRA(orig_experts, lora_dim=4, alpha=8).to(device)
    
    # Real input
    bs = 2
    seq_len = 128
    dim = 16
    x = torch.randn(bs * seq_len, dim, device=device, dtype=torch.float16)
    token_mask = torch.ones(bs * seq_len, dtype=torch.bool, device=device)
    
    # Weights and indices need to be consistent with n_routed_experts=4
    # We can use a simple routing strategy or random
    weights = torch.rand(bs * seq_len, 2, device=device, dtype=torch.float32)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    indices = torch.randint(0, 4, (bs * seq_len, 2), device=device, dtype=torch.int32)
    
    # Run forward
    out = lora_experts(x, token_mask, weights, indices)
    
    assert out.shape == x.shape
    assert not torch.isnan(out).any()
    
    # Verify equivalence with zero LoRA weights
    # GroupedExpertsDeepEPLoRA initializes B weights to zero by default, so output should match exactly.
    
    with torch.no_grad():
        out_orig = orig_experts(x, token_mask, weights, indices)
    
    # Tolerances for float16
    assert torch.allclose(out, out_orig, atol=1e-3, rtol=1e-3)

def test_patch_moe_module(moe_config):
    """Test that patch_moe_module correctly wraps the original experts with the appropriate LoRA class."""
    orig_experts = GroupedExperts(moe_config)
    patched = patch_moe_module(orig_experts, dim=4)
    assert isinstance(patched, GroupedExpertsLoRA)
    
    orig_experts_deep = GroupedExpertsDeepEP(moe_config)
    patched_deep = patch_moe_module(orig_experts_deep, dim=4)
    assert isinstance(patched_deep, GroupedExpertsDeepEPLoRA)

def test_apply_lora_patching_logic(moe_config):
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

    model = MockModel()
    peft_config = PeftConfig(
        target_modules=["experts"],
        dim=4
    )
    
    count = apply_lora_to_linear_modules(model, peft_config)
    assert count == 1
    assert isinstance(model.experts, GroupedExpertsLoRA)
    assert isinstance(model.linear, nn.Linear) # Should not be patched

    # Test wildcard matching
    model = MockModel()
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

@pytest.mark.skipif(grouped_gemm is None, reason="Requires grouped_gemm")
def test_grouped_experts_deepep_lora_forward_mocked(moe_config):
    """
    Test Forward pass of GroupedExpertsDeepEPLoRA using a Mock Dispatcher.
    
    This test verifies the LoRA-wrapped gated GEMM logic (using grouped_gemm kernels) 
    independently of the DeepEP communication backend. This allows verification on 
    non-Hopper (non-sm_90) hardware where DeepEP is physically unavailable.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    moe_config.n_routed_experts = 4
    moe_config.dim = 16
    moe_config.moe_inter_dim = 32
    moe_config.dtype = torch.bfloat16
    
    orig_experts = GroupedExpertsDeepEP(moe_config).to(device).to(torch.bfloat16)
    # Initialize expert weights BEFORE creating LoRA module so they match after copy
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
    permuted_probs = torch.ones(num_tokens, 1, device=device).to(dtype)
    
    # Set the same mock on both modules to ensure they see the same "dispatched" data
    mock_dispatcher.token_permutation2 = MagicMock(
        return_value=(permuted_x, tokens_per_expert, permuted_probs)
    )
    lora_module.token_dispatcher = mock_dispatcher
    orig_experts.token_dispatcher = mock_dispatcher
    
    x = torch.randn(num_tokens, 16, device=device).to(dtype)
    weights = torch.ones(num_tokens, 1, device=device).to(dtype)
    indices = torch.zeros(num_tokens, 1, dtype=torch.long, device=device)
    token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
    
    # This will now reach the lora_module.forward -> ops.gmm calls!
    out = lora_module(x, token_mask, weights, indices)
    
    # Verify equivalence with zero LoRA weights (DeepEP LoRA B is zero-init by default)
    with torch.no_grad():
        out_orig = orig_experts(x, token_mask, weights, indices)
    
    assert out.shape == (num_tokens, 16)
    assert torch.allclose(out, out_orig, atol=1e-3, rtol=1e-3)
