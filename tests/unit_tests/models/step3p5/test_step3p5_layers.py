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

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.step3p5.layers import (
    Step3p5Attention,
    Step3p5MLP,
    Step3p5RMSNorm,
    Step3p5RotaryEmbedding,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


@dataclass
class MockStep3p5Config:
    """Mock configuration for Step3p5 model."""
    vocab_size: int = 128
    hidden_size: int = 64
    intermediate_size: int = 128
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    num_attention_groups: int = 2
    max_position_embeddings: int = 256
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    partial_rotary_factors: list = None
    layer_types: list = None
    attention_other_setting: dict = None
    sliding_window: int = None
    use_head_wise_attn_gate: bool = False
    use_rope_layers: list = None
    head_dim: int = 16
    attention_bias: bool = False
    torch_dtype: str = "bfloat16"

    def __post_init__(self):
        if self.layer_types is None:
            self.layer_types = ["full_attention", "sliding_attention"]
        if self.attention_other_setting is None:
            self.attention_other_setting = {
                "num_attention_heads": 2,
                "num_attention_groups": 1,
            }


@pytest.fixture
def config():
    return MockStep3p5Config()


@pytest.fixture
def sdpa_backend():
    return BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        rope_fusion=False,
        enable_deepep=False,
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
    )


class TestStep3p5RMSNorm:
    def test_initialization(self):
        hidden_size = 64
        norm = Step3p5RMSNorm(hidden_size, eps=1e-6)

        assert norm.weight.shape == (hidden_size,)
        # Weight should be initialized to zeros
        assert torch.allclose(norm.weight, torch.zeros(hidden_size))
        assert norm.variance_epsilon == 1e-6

    def test_forward_shape_preserved(self):
        hidden_size = 64
        batch, seq = 2, 10
        norm = Step3p5RMSNorm(hidden_size)
        x = torch.randn(batch, seq, hidden_size)

        out = norm(x)

        assert out.shape == x.shape

    def test_weight_plus_one_scaling(self):
        """Verify the (weight + 1) scaling behavior."""
        hidden_size = 8
        norm = Step3p5RMSNorm(hidden_size, eps=0)

        # With weight = 0, output should equal normalized input * 1
        x = torch.randn(1, 1, hidden_size)
        out_zero_weight = norm(x)

        # Set weight to non-zero
        norm.weight.data = torch.ones(hidden_size)
        out_one_weight = norm(x)

        # With weight = 1, scaling is (1 + 1) = 2, so output should be 2x
        assert torch.allclose(out_one_weight, out_zero_weight * 2, atol=1e-6)

    def test_reset_parameters(self):
        hidden_size = 32
        norm = Step3p5RMSNorm(hidden_size)
        norm.weight.data = torch.ones(hidden_size)

        norm.reset_parameters()

        assert torch.allclose(norm.weight, torch.zeros(hidden_size))


class TestStep3p5RotaryEmbedding:
    def test_initialization_with_scalar_theta(self, config):
        rotary = Step3p5RotaryEmbedding(config, layer_idx=0)

        assert rotary.base == config.rope_theta
        assert rotary.partial_rotary_factor == 1.0
        assert rotary.layer_idx == 0

    def test_initialization_with_list_theta(self, config):
        config.rope_theta = [10000.0, 20000.0]
        rotary0 = Step3p5RotaryEmbedding(config, layer_idx=0)
        rotary1 = Step3p5RotaryEmbedding(config, layer_idx=1)

        assert rotary0.base == 10000.0
        assert rotary1.base == 20000.0

    def test_initialization_with_partial_rotary_factors(self, config):
        config.partial_rotary_factors = [0.5, 0.75]
        rotary0 = Step3p5RotaryEmbedding(config, layer_idx=0)
        rotary1 = Step3p5RotaryEmbedding(config, layer_idx=1)

        assert rotary0.partial_rotary_factor == 0.5
        assert rotary1.partial_rotary_factor == 0.75

    def test_forward_returns_cos_sin(self, config):
        rotary = Step3p5RotaryEmbedding(config, layer_idx=0)
        batch, seq = 2, 10
        x = torch.randn(batch, seq, config.hidden_size)
        position_ids = torch.arange(seq).unsqueeze(0).expand(batch, -1)

        cos, sin = rotary(x, position_ids)

        # cos/sin should have shape [batch, seq, rotary_dim]
        expected_shape = (batch, seq, rotary.rotary_dim)
        assert cos.shape == expected_shape
        assert sin.shape == expected_shape


class TestStep3p5MLP:
    def test_initialization(self, config, sdpa_backend):
        mlp = Step3p5MLP(config, sdpa_backend)

        assert mlp.hidden_size == config.hidden_size
        assert mlp.intermediate_size == config.intermediate_size
        assert mlp.swiglu_limit is None

    def test_initialization_with_custom_intermediate_size(self, config, sdpa_backend):
        custom_inter = 256
        mlp = Step3p5MLP(config, sdpa_backend, intermediate_size=custom_inter)

        assert mlp.intermediate_size == custom_inter

    def test_forward_shape_preserved(self, config, sdpa_backend):
        mlp = Step3p5MLP(config, sdpa_backend)
        batch, seq = 2, 10
        x = torch.randn(batch, seq, config.hidden_size).to(torch.bfloat16)

        out = mlp(x)

        assert out.shape == x.shape

    def test_swiglu_clamping_configured(self, config, sdpa_backend):
        """Test that swiglu_limit is properly set."""
        limit = 5.0
        mlp = Step3p5MLP(config, sdpa_backend, swiglu_limit=limit)

        assert mlp.swiglu_limit == limit

        # Forward pass should complete without error
        batch, seq = 1, 1
        x = torch.randn(batch, seq, config.hidden_size).to(torch.bfloat16)
        out = mlp(x)
        assert out is not None


class TestStep3p5Attention:
    def test_initialization_full_attention(self, config, sdpa_backend):
        attention = Step3p5Attention(config, layer_idx=0, backend=sdpa_backend)

        assert attention.num_heads == config.num_attention_heads
        assert attention.num_kv_heads == config.num_attention_groups
        assert attention.sliding_window is None

    def test_initialization_sliding_attention(self, config, sdpa_backend):
        config.sliding_window = 128
        attention = Step3p5Attention(config, layer_idx=1, backend=sdpa_backend)

        # Layer 1 should have sliding_attention type
        assert attention.num_heads == config.attention_other_setting["num_attention_heads"]
        assert attention.num_kv_heads == config.attention_other_setting["num_attention_groups"]
        assert attention.sliding_window == config.sliding_window

    def test_attention_with_step3p5_flash_config(self, sdpa_backend):
        """Test attention with Step-3.5-Flash like config (different head counts)."""
        from dataclasses import dataclass

        @dataclass
        class Step3p5FlashLikeConfig:
            hidden_size: int = 4096
            num_attention_heads: int = 64
            num_attention_groups: int = 8
            head_dim: int = 128
            rms_norm_eps: float = 1e-5
            rope_theta: float = 10000.0
            partial_rotary_factors: list = None
            layer_types: list = None
            attention_other_setting: dict = None
            sliding_window: int = 512
            use_head_wise_attn_gate: bool = True
            use_rope_layers: list = None
            max_position_embeddings: int = 262144
            attention_bias: bool = False

            def __post_init__(self):
                self.layer_types = ["full_attention", "sliding_attention"]
                self.attention_other_setting = {
                    "num_attention_heads": 96,  # Step-3.5-Flash uses 96 for sliding
                    "num_attention_groups": 8,
                }

        config = Step3p5FlashLikeConfig()

        # Full attention layer (layer 0)
        attn_full = Step3p5Attention(config, layer_idx=0, backend=sdpa_backend)
        assert attn_full.num_heads == 64
        assert attn_full.num_kv_heads == 8
        assert attn_full.sliding_window is None

        # Sliding attention layer (layer 1)
        attn_sliding = Step3p5Attention(config, layer_idx=1, backend=sdpa_backend)
        assert attn_sliding.num_heads == 96
        assert attn_sliding.num_kv_heads == 8
        assert attn_sliding.sliding_window == 512

    def test_initialization_with_head_gate(self, config, sdpa_backend):
        config.use_head_wise_attn_gate = True
        attention = Step3p5Attention(config, layer_idx=0, backend=sdpa_backend)

        assert attention.use_head_wise_attn_gate
        assert attention.g_proj is not None

    def test_forward_shape_preserved(self, config, sdpa_backend):
        attention = Step3p5Attention(config, layer_idx=0, backend=sdpa_backend)
        batch, seq = 2, 10
        x = torch.randn(batch, seq, config.hidden_size).to(torch.bfloat16)
        freqs_cis = torch.randn(batch, seq, config.head_dim)

        # Mock the attention function
        fake_attn = torch.zeros(batch, config.num_attention_heads, seq, config.head_dim)
        attention.attn_func = MagicMock(return_value=fake_attn.to(torch.bfloat16))

        with patch("nemo_automodel.components.models.step3p5.layers.apply_rotary_emb_qk", side_effect=lambda q, k, *_, **__: (q, k)):
            out = attention(x, freqs_cis=freqs_cis)

        assert out.shape == (batch, seq, config.hidden_size)

    def test_qk_norms_are_step3p5_rmsnorm(self, config, sdpa_backend):
        attention = Step3p5Attention(config, layer_idx=0, backend=sdpa_backend)

        assert isinstance(attention.q_norm, Step3p5RMSNorm)
        assert isinstance(attention.k_norm, Step3p5RMSNorm)

    def test_init_weights(self, config, sdpa_backend):
        attention = Step3p5Attention(config, layer_idx=0, backend=sdpa_backend)

        with patch("torch.nn.init.trunc_normal_") as mock_trunc:
            with patch.object(attention.q_norm, "reset_parameters") as mock_q_reset:
                with patch.object(attention.k_norm, "reset_parameters") as mock_k_reset:
                    attention.init_weights(torch.device("cpu"), init_std=0.05)

        # 4 linear layers (q, k, v, o)
        assert mock_trunc.call_count == 4
        mock_q_reset.assert_called_once()
        mock_k_reset.assert_called_once()

    def test_use_rope_with_empty_list(self, config, sdpa_backend):
        """Test that empty use_rope_layers list defaults to using RoPE."""
        config.use_rope_layers = []  # Empty list like HF Step-3.5-Flash
        attention = Step3p5Attention(config, layer_idx=0, backend=sdpa_backend)
        assert attention.use_rope is True

    def test_use_rope_with_none(self, config, sdpa_backend):
        """Test that None use_rope_layers defaults to using RoPE."""
        config.use_rope_layers = None
        attention = Step3p5Attention(config, layer_idx=0, backend=sdpa_backend)
        assert attention.use_rope is True

    def test_use_rope_with_explicit_false(self, config, sdpa_backend):
        """Test that explicit False in use_rope_layers disables RoPE."""
        config.use_rope_layers = [False, True]
        attention0 = Step3p5Attention(config, layer_idx=0, backend=sdpa_backend)
        attention1 = Step3p5Attention(config, layer_idx=1, backend=sdpa_backend)
        assert attention0.use_rope is False
        assert attention1.use_rope is True
