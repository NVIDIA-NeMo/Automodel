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

from unittest.mock import patch

import pytest
import torch
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig

from nemo_automodel.components.models.gpt_oss.layers import (
    GptOssAttention,
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


class TestGptOssAttentionWithTE:
    """Test GptOssAttention with Transformer Engine backend."""

    @pytest.fixture
    def te_backend_config(self):
        return BackendConfig(
            linear="torch",
            attn="te",
            rms_norm="torch",
            enable_deepep=False,
            fake_balanced_gate=False,
            enable_hf_state_dict_adapter=False,
        )

    def test_te_backend_requires_min_version(self, gpt_config):
        """Test that TE backend requires minimum TE version 2.8.0."""
        te_backend = BackendConfig(attn="te")

        # Mock TE version check to simulate old version
        from unittest.mock import patch
        with patch("nemo_automodel.components.models.gpt_oss.layers.is_te_min_version", return_value=False):
            with pytest.raises(ValueError, match="Transformer Engine DotProductAttention for GPT-OSS is only supported for TE version 2.8.0 or higher"):
                GptOssAttention(gpt_config, te_backend)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpt_oss_attention_te_init(self, gpt_config, te_backend_config):
        """Test GptOssAttention initialization with TE backend."""
        pytest.importorskip("transformer_engine")

        # Mock version check to allow initialization
        from unittest.mock import patch
        with patch("nemo_automodel.components.models.gpt_oss.layers.is_te_min_version", return_value=True):
            attention = GptOssAttention(gpt_config, te_backend_config)

            assert attention.backend.attn == "te"
            assert attention.head_dim == gpt_config.head_dim
            assert attention.num_attention_heads == gpt_config.num_attention_heads
            # TE backend should not have sinks as a separate parameter
            assert attention.sinks is None
            # TE backend should have softmax_offset in attn_module
            assert hasattr(attention.attn_module, "softmax_offset")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_te_init_weights(self, gpt_config, te_backend_config, device):
        """Test weight initialization with TE backend."""
        pytest.importorskip("transformer_engine")

        from unittest.mock import patch
        with patch("nemo_automodel.components.models.gpt_oss.layers.is_te_min_version", return_value=True):
            attention = GptOssAttention(gpt_config, te_backend_config)

            # Store original softmax_offset to verify it changes
            original_softmax_offset = attention.attn_module.softmax_offset.clone()

            attention.init_weights(device, init_std=0.02)

            # softmax_offset should have changed
            assert not torch.equal(attention.attn_module.softmax_offset, original_softmax_offset)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_te_forward_with_bshd_format(self, gpt_config, te_backend_config, device):
        """Test TE attention forward pass with BSHD format."""
        pytest.importorskip("transformer_engine")

        from unittest.mock import patch
        with patch("nemo_automodel.components.models.gpt_oss.layers.is_te_min_version", return_value=True):
            attention = GptOssAttention(gpt_config, te_backend_config)
            attention = attention.to(device)

            batch_size, seq_len = 2, 8
            x = torch.randn(batch_size, seq_len, gpt_config.hidden_size, dtype=torch.bfloat16, device=device)
            freqs_cis = torch.randn(batch_size, seq_len, gpt_config.head_dim, dtype=torch.bfloat16, device=device)

            try:
                output = attention(x, freqs_cis)
                assert output.shape == (batch_size, seq_len, gpt_config.hidden_size)
                assert output.device == device
            except Exception as e:
                pytest.fail(f"TE forward pass with BSHD format failed: {e}")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_te_forward_with_thd_format(self, gpt_config, te_backend_config, device):
        """Test TE attention forward pass with THD format."""
        pytest.importorskip("transformer_engine")

        from unittest.mock import patch
        with patch("nemo_automodel.components.models.gpt_oss.layers.is_te_min_version", return_value=True):
            attention = GptOssAttention(gpt_config, te_backend_config)
            attention = attention.to(device)

            # THD format: (num_tokens, hidden_size)
            num_tokens = 16
            x = torch.randn(num_tokens, gpt_config.hidden_size, dtype=torch.bfloat16, device=device)
            freqs_cis = torch.randn(num_tokens, gpt_config.head_dim, dtype=torch.bfloat16, device=device)

            # Provide cu_seqlens for packed sequence
            cu_seqlens = torch.tensor([0, 8, 16], dtype=torch.int32, device=device)

            try:
                output = attention(x, freqs_cis, cu_seqlens=cu_seqlens, max_seqlen=8)
                assert output.shape == (num_tokens, gpt_config.hidden_size)
                assert output.device == device
            except Exception as e:
                pytest.fail(f"TE forward pass with THD format failed: {e}")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_te_forward_with_sliding_window(self, gpt_config, te_backend_config, device):
        """Test TE attention forward pass with sliding window."""
        pytest.importorskip("transformer_engine")

        # Set sliding window in config
        gpt_config.sliding_window = 256

        from unittest.mock import patch
        with patch("nemo_automodel.components.models.gpt_oss.layers.is_te_min_version", return_value=True):
            attention = GptOssAttention(gpt_config, te_backend_config, use_sliding_attention=True)
            attention = attention.to(device)

            assert attention.sliding_window == 256

            batch_size, seq_len = 2, 8
            x = torch.randn(batch_size, seq_len, gpt_config.hidden_size, dtype=torch.bfloat16, device=device)
            freqs_cis = torch.randn(batch_size, seq_len, gpt_config.head_dim, dtype=torch.bfloat16, device=device)

            try:
                output = attention(x, freqs_cis)
                assert output.shape == (batch_size, seq_len, gpt_config.hidden_size)
            except Exception as e:
                pytest.fail(f"TE forward pass with sliding window failed: {e}")
