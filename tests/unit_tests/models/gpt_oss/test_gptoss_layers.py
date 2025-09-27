# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

import math
from unittest.mock import patch

import pytest
import torch
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig

from nemo_automodel.components.models.gpt_oss.layers import (
    GptOssAttention,
    RotaryEmbedding,
    _apply_rotary_emb,
)
from nemo_automodel.components.moe.utils import BackendConfig


@pytest.fixture
def device():
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return torch.device("cpu")


@pytest.fixture
def gpt_config():
    return GptOssConfig(
        vocab_size=1000,
        hidden_size=128,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=32,
        num_hidden_layers=2,
        intermediate_size=256,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        sliding_window=None,
        layer_types=["full_attention", "full_attention"],
        num_local_experts=8,
        num_experts_per_tok=2,
        router_aux_loss_coef=0.01,
    )


@pytest.fixture
def backend_config():
    return BackendConfig(
        linear="torch",
        attn="flex",
        rms_norm="torch",
        enable_deepep=False,
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
    )


class TestApplyRotaryEmbedding:
    """Test _apply_rotary_emb function."""

    def test_apply_rotary_emb_shape_preservation(self, device):
        """Test that rotary embedding preserves tensor shapes."""
        batch_size, seq_len, n_heads, head_dim = 2, 4, 2, 8
        x = torch.randn(batch_size, seq_len, n_heads, head_dim, device=device)
        cos = torch.randn(seq_len, head_dim // 2, device=device)
        sin = torch.randn(seq_len, head_dim // 2, device=device)

        result = _apply_rotary_emb(x, cos, sin)

        assert result.shape == x.shape
        assert result.device == x.device
        assert result.dtype == x.dtype

    def test_apply_rotary_emb_correctness(self, device):
        """Test rotary embedding computation correctness."""
        # Simple test case with known values
        x = torch.ones(1, 2, 1, 4, device=device)
        cos = torch.ones(2, 2, device=device)
        sin = torch.zeros(2, 2, device=device)

        result = _apply_rotary_emb(x, cos, sin)

        # With sin=0 and cos=1, result should be [x1, x2] -> [x1*1-x2*0, x2*1+x1*0] = [x1, x2]
        expected = x.clone()
        torch.testing.assert_close(result, expected)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_apply_rotary_emb_gpu_dtype_handling(self):
        """Test GPU-specific dtype handling."""
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        x = torch.randn(1, 2, 1, 4, dtype=torch.bfloat16, device=device)
        cos = torch.randn(2, 2, dtype=torch.float32, device=device)
        sin = torch.randn(2, 2, dtype=torch.float32, device=device)

        result = _apply_rotary_emb(x, cos, sin)

        assert result.dtype == torch.bfloat16
        assert result.device == device


class TestRotaryEmbedding:
    """Test RotaryEmbedding module."""

    def test_rotary_embedding_init(self, device):
        """Test RotaryEmbedding initialization."""
        head_dim = 32
        base = 10000
        rope = RotaryEmbedding(
            head_dim=head_dim,
            base=base,
            dtype=torch.float32,
            device=device,
        )

        assert rope.head_dim == head_dim
        assert rope.base == base
        assert rope.dtype == torch.float32
        assert rope.device == device
        assert rope.scaling_factor == 1.0
        assert rope.ntk_alpha == 1.0
        assert rope.ntk_beta == 32.0

    def test_compute_concentration_and_inv_freq_no_scaling(self, device):
        """Test concentration and inverse frequency computation without scaling."""
        rope = RotaryEmbedding(
            head_dim=32,
            base=10000,
            dtype=torch.float32,
            scaling_factor=1.0,
            device=device,
        )

        concentration, inv_freq = rope._compute_concentration_and_inv_freq()

        assert concentration == 1.0
        assert inv_freq.shape == (16,)  # head_dim // 2
        assert inv_freq.device == device

    def test_compute_concentration_and_inv_freq_with_scaling(self, device):
        """Test concentration and inverse frequency computation with scaling."""
        rope = RotaryEmbedding(
            head_dim=32,
            base=10000,
            dtype=torch.float32,
            scaling_factor=2.0,
            initial_context_length=512,
            device=device,
        )

        concentration, inv_freq = rope._compute_concentration_and_inv_freq()

        expected_concentration = 0.1 * math.log(2.0) + 1.0
        assert abs(concentration - expected_concentration) < 1e-6
        assert inv_freq.shape == (16,)

    def test_compute_cos_sin(self, device):
        """Test cosine and sine computation."""
        rope = RotaryEmbedding(
            head_dim=32,
            base=10000,
            dtype=torch.float32,
            device=device,
        )

        num_tokens = 8
        cos, sin = rope._compute_cos_sin(num_tokens)

        assert cos.shape == (num_tokens, 16)
        assert sin.shape == (num_tokens, 16)
        assert cos.device == device
        assert sin.device == device

    def test_forward_shape_preservation(self, device):
        """Test that forward pass preserves query and key shapes."""
        rope = RotaryEmbedding(
            head_dim=32,
            base=10000,
            dtype=torch.float32,
            device=device,
        )

        batch_size, seq_len, n_heads, head_dim = 2, 4, 4, 32
        query = torch.randn(batch_size, seq_len, n_heads, head_dim, device=device)
        key = torch.randn(batch_size, seq_len, n_heads, head_dim, device=device)

        q_rot, k_rot = rope(query, key)

        assert q_rot.shape == query.shape
        assert k_rot.shape == key.shape
        assert q_rot.device == query.device
        assert k_rot.device == key.device

    def test_forward_different_seq_lengths(self, device):
        """Test forward pass with different sequence lengths."""
        rope = RotaryEmbedding(
            head_dim=32,
            base=10000,
            dtype=torch.float32,
            device=device,
        )

        for seq_len in [1, 8, 16, 32]:
            query = torch.randn(1, seq_len, 4, 32, device=device)
            key = torch.randn(1, seq_len, 4, 32, device=device)

            q_rot, k_rot = rope(query, key)

            assert q_rot.shape == (1, seq_len, 4, 32)
            assert k_rot.shape == (1, seq_len, 4, 32)


class TestGptOssAttention:
    """Test GptOssAttention module."""

    def test_gpt_oss_attention_init(self, gpt_config, backend_config):
        """Test GptOssAttention initialization."""
        attention = GptOssAttention(gpt_config, backend_config)

        assert attention.head_dim == gpt_config.head_dim
        assert attention.num_attention_heads == gpt_config.num_attention_heads
        assert attention.num_key_value_heads == gpt_config.num_key_value_heads
        assert attention.hidden_size == gpt_config.hidden_size
        assert attention.sliding_window is None
        assert hasattr(attention, "q_proj")
        assert hasattr(attention, "k_proj")
        assert hasattr(attention, "v_proj")
        assert hasattr(attention, "o_proj")
        assert hasattr(attention, "sinks")

    def test_gpt_oss_attention_init_with_sliding_window(self, gpt_config, backend_config):
        """Test GptOssAttention initialization with sliding window."""
        attention = GptOssAttention(gpt_config, backend_config, use_sliding_attention=True)

        assert attention.sliding_window == gpt_config.sliding_window

    def test_gpt_oss_attention_linear_layer_dimensions(self, gpt_config, backend_config):
        """Test that linear layers have correct dimensions."""
        attention = GptOssAttention(gpt_config, backend_config)

        # q_proj: hidden_size -> num_attention_heads * head_dim
        assert attention.q_proj.in_features == gpt_config.hidden_size
        assert attention.q_proj.out_features == gpt_config.num_attention_heads * gpt_config.head_dim

        # k_proj, v_proj: hidden_size -> num_key_value_heads * head_dim
        assert attention.k_proj.in_features == gpt_config.hidden_size
        assert attention.k_proj.out_features == gpt_config.num_key_value_heads * gpt_config.head_dim

        assert attention.v_proj.in_features == gpt_config.hidden_size
        assert attention.v_proj.out_features == gpt_config.num_key_value_heads * gpt_config.head_dim

        # o_proj: num_attention_heads * head_dim -> hidden_size
        assert attention.o_proj.in_features == gpt_config.num_attention_heads * gpt_config.head_dim
        assert attention.o_proj.out_features == gpt_config.hidden_size

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_forward_shape_correctness(self, gpt_config, backend_config, device):
        """Test forward pass output shapes."""
        attention = GptOssAttention(gpt_config, backend_config)
        attention = attention.to(device)

        batch_size, seq_len = 2, 8
        x = torch.randn(batch_size, seq_len, gpt_config.hidden_size, dtype=torch.bfloat16, device=device)
        freqs_cis = torch.randn(batch_size, seq_len, gpt_config.head_dim, dtype=torch.bfloat16, device=device)

        # Mock the attn_module call method instead of replacing the module
        with patch.object(attention.attn_module, "__call__") as mock_attn:
            # Mock attention module to return expected shape
            mock_attn.return_value = torch.randn(
                batch_size, gpt_config.num_attention_heads, seq_len, gpt_config.head_dim, dtype=torch.bfloat16, device=device
            )

            output = attention(x, freqs_cis)

            assert output.shape == (batch_size, seq_len, gpt_config.hidden_size)
            assert output.device == device

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_forward_gpu_execution(self, gpt_config, backend_config):
        """Test forward pass executes correctly on GPU."""
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        attention = GptOssAttention(gpt_config, backend_config)
        attention = attention.to(device)

        batch_size, seq_len = 2, 8
        x = torch.randn(batch_size, seq_len, gpt_config.hidden_size, dtype=torch.bfloat16, device=device)
        freqs_cis = torch.randn(batch_size, seq_len, gpt_config.head_dim, dtype=torch.bfloat16, device=device)

        # Test that the forward pass completes successfully on GPU
        try:
            output = attention(x, freqs_cis)
            assert output.shape == (batch_size, seq_len, gpt_config.hidden_size)
            assert output.device == device
            # Test passes if no exception is raised
        except Exception as e:
            pytest.fail(f"GPU forward pass failed: {e}")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_init_weights(self, gpt_config, backend_config, device):
        """Test weight initialization."""
        attention = GptOssAttention(gpt_config, backend_config)

        # Store original weights to verify they change
        original_q_weight = attention.q_proj.weight.clone()
        original_sinks = attention.sinks.clone()

        attention.init_weights(device, init_std=0.02)

        # Weights should have changed
        assert not torch.equal(attention.q_proj.weight, original_q_weight)
        assert not torch.equal(attention.sinks, original_sinks)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_rotary_embedding_application(self, gpt_config, backend_config, device):
        """Test that rotary embedding is correctly applied to q and k."""
        attention = GptOssAttention(gpt_config, backend_config)
        attention = attention.to(device)

        batch_size, seq_len = 1, 4
        x = torch.randn(batch_size, seq_len, gpt_config.hidden_size, dtype=torch.bfloat16, device=device)

        # Create simple freqs_cis for testing
        cos = torch.ones(batch_size, seq_len, gpt_config.head_dim // 2, dtype=torch.bfloat16, device=device)
        sin = torch.zeros(batch_size, seq_len, gpt_config.head_dim // 2, dtype=torch.bfloat16, device=device)
        freqs_cis = torch.cat([cos, sin], dim=-1)

        # Test that the forward pass completes successfully with valid inputs
        # The main goal is to ensure rotary embedding doesn't crash
        try:
            output = attention(x, freqs_cis)
            assert output.shape == (batch_size, seq_len, gpt_config.hidden_size)
            assert output.device == device
            # Test passes if no exception is raised
        except Exception as e:
            pytest.fail(f"Forward pass failed with rotary embedding: {e}")
