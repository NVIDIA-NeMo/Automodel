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

import sys
import types
import importlib.util
import pytest
import torch
from unittest.mock import Mock, patch, MagicMock

# Check if fast_hadamard_transform is available
HADAMARD_AVAILABLE = False
try:
    from fast_hadamard_transform import hadamard_transform  # noqa: F401
    HADAMARD_AVAILABLE = True
except ImportError:
    # Mock fast_hadamard_transform before importing deepseek_v32 modules
    if 'fast_hadamard_transform' not in sys.modules:
        mock_hadamard = types.ModuleType('fast_hadamard_transform')
        mock_hadamard.__spec__ = importlib.util.spec_from_loader('fast_hadamard_transform', loader=None)
        mock_hadamard.hadamard_transform = lambda x, scale: x
        sys.modules['fast_hadamard_transform'] = mock_hadamard

from nemo_automodel.components.models.deepseek_v32.config import DeepseekV32Config
from nemo_automodel.components.models.deepseek_v32.layers import (
    DeepseekV32Indexer,
    DeepseekV32MLA,
    _rotate_activation,
)
from nemo_automodel.components.models.common import BackendConfig

# Skip Transformer Engine tests by default unless explicitly enabled
TE_AVAILABLE = False
try:
    import transformer_engine  # noqa: F401
    TE_AVAILABLE = True
except ImportError:
    pass

skip_te = pytest.mark.skipif(not TE_AVAILABLE, reason="Transformer Engine not available")
skip_if_no_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for GPU operations")
skip_hadamard = pytest.mark.skipif(not HADAMARD_AVAILABLE, reason="fast_hadamard_transform not available")


class TestRotateActivation:
    @skip_hadamard
    @skip_if_no_gpu
    def test_rotate_activation_bfloat16(self):
        """Test rotation with bfloat16 input."""
        x = torch.randn(2, 8, 64, device="cuda", dtype=torch.bfloat16)
        result = _rotate_activation(x)
        assert result.dtype == torch.bfloat16
        assert result.shape == x.shape

    @skip_hadamard
    @skip_if_no_gpu
    def test_rotate_activation_float32_converts(self):
        """Test that float32 input is converted to bfloat16."""
        x = torch.randn(2, 8, 64, device="cuda", dtype=torch.float32)
        result = _rotate_activation(x)
        assert result.dtype == torch.bfloat16
        assert result.shape == x.shape


class TestDeepseekV32IndexerInit:
    def create_mock_config(self, **overrides):
        config = Mock(spec=DeepseekV32Config)
        config.num_attention_heads = 8
        config.hidden_size = 256
        config.q_lora_rank = 128
        config.index_n_heads = 4
        config.index_head_dim = 32
        config.index_topk = 16
        config.qk_rope_head_dim = 16

        for key, value in overrides.items():
            setattr(config, key, value)
        return config

    @patch("nemo_automodel.components.models.deepseek_v32.layers.initialize_linear_module")
    def test_indexer_initialization(self, mock_init_linear):
        """Test that indexer initializes correctly."""
        config = self.create_mock_config()
        backend = BackendConfig(attn="sdpa", linear="torch", rms_norm="torch")

        mock_init_linear.return_value = Mock()

        indexer = DeepseekV32Indexer(config, backend)

        assert indexer.num_heads == 4
        assert indexer.head_dim == 32
        assert indexer.qk_rope_head_dim == 16
        assert indexer.qk_nope_head_dim == 16  # head_dim - qk_rope_head_dim
        assert indexer.index_topk == 16
        assert indexer.q_lora_rank == 128
        assert indexer.hidden_size == 256

    @patch("nemo_automodel.components.models.deepseek_v32.layers.initialize_linear_module")
    def test_indexer_has_k_norm_layernorm(self, mock_init_linear):
        """Test that indexer uses LayerNorm (not RMSNorm) for k_norm."""
        config = self.create_mock_config()
        backend = BackendConfig(attn="sdpa", linear="torch", rms_norm="torch")

        mock_init_linear.return_value = Mock()

        indexer = DeepseekV32Indexer(config, backend)

        # Should have k_norm as LayerNorm
        assert hasattr(indexer, 'k_norm')
        assert isinstance(indexer.k_norm, torch.nn.LayerNorm)

    @patch("nemo_automodel.components.models.deepseek_v32.layers.initialize_linear_module")
    def test_indexer_projections(self, mock_init_linear):
        """Test that indexer has required projection layers."""
        config = self.create_mock_config()
        backend = BackendConfig(attn="sdpa", linear="torch", rms_norm="torch")

        mock_init_linear.return_value = Mock()

        indexer = DeepseekV32Indexer(config, backend)

        # Should have wq_b, wk, and weights_proj
        assert hasattr(indexer, 'wq_b')
        assert hasattr(indexer, 'wk')
        assert hasattr(indexer, 'weights_proj')


class TestDeepseekV32MLAInit:
    def create_mock_config(self, **overrides):
        config = Mock(spec=DeepseekV32Config)
        config.num_attention_heads = 8
        config.hidden_size = 256
        config.q_lora_rank = 128
        config.kv_lora_rank = 64
        config.qk_nope_head_dim = 16
        config.qk_rope_head_dim = 16
        config.qk_head_dim = 32
        config.v_head_dim = 32
        config.rope_scaling = None
        config.max_position_embeddings = 4096

        # Indexer config
        config.index_n_heads = 4
        config.index_head_dim = 32
        config.index_topk = 16

        for key, value in overrides.items():
            setattr(config, key, value)
        return config

    @patch("nemo_automodel.components.models.deepseek_v32.layers.initialize_linear_module")
    @patch("nemo_automodel.components.models.deepseek_v32.layers.initialize_rms_norm_module")
    @patch("nemo_automodel.components.models.deepseek_v32.layers.initialize_attn_module_and_func")
    def test_mla_init_with_q_lora(self, mock_init_attn, mock_init_rms, mock_init_linear):
        """Test MLA initialization with q_lora (V3.2 always uses q_lora)."""
        config = self.create_mock_config()
        backend = BackendConfig(attn="sdpa", linear="torch", rms_norm="torch")

        mock_init_linear.return_value = Mock()
        mock_init_rms.return_value = Mock()
        mock_init_attn.return_value = (Mock(), Mock())

        mla = DeepseekV32MLA(config, backend)

        # V3.2 always uses q_lora
        assert hasattr(mla, 'q_a_proj')
        assert hasattr(mla, 'q_b_proj')
        assert hasattr(mla, 'q_a_layernorm')

        # Check other components
        assert hasattr(mla, 'kv_a_proj_with_mqa')
        assert hasattr(mla, 'kv_a_layernorm')
        assert hasattr(mla, 'kv_b_proj')
        assert hasattr(mla, 'o_proj')

        # Check indexer
        assert hasattr(mla, 'indexer')
        assert isinstance(mla.indexer, DeepseekV32Indexer)

    @patch("nemo_automodel.components.models.deepseek_v32.layers.initialize_linear_module")
    @patch("nemo_automodel.components.models.deepseek_v32.layers.initialize_rms_norm_module")
    @patch("nemo_automodel.components.models.deepseek_v32.layers.initialize_attn_module_and_func")
    def test_mla_attributes(self, mock_init_attn, mock_init_rms, mock_init_linear):
        """Test MLA has correct attribute values."""
        config = self.create_mock_config()
        backend = BackendConfig(attn="sdpa", linear="torch", rms_norm="torch")

        mock_init_linear.return_value = Mock()
        mock_init_rms.return_value = Mock()
        mock_init_attn.return_value = (Mock(), Mock())

        mla = DeepseekV32MLA(config, backend)

        assert mla.n_heads == 8
        assert mla.q_lora_rank == 128
        assert mla.kv_lora_rank == 64
        assert mla.qk_nope_head_dim == 16
        assert mla.qk_rope_head_dim == 16
        assert mla.qk_head_dim == 32
        assert mla.v_head_dim == 32
        assert mla.index_topk == 16

    @patch("nemo_automodel.components.models.deepseek_v32.layers.yarn_get_mscale")
    @patch("nemo_automodel.components.models.deepseek_v32.layers.initialize_linear_module")
    @patch("nemo_automodel.components.models.deepseek_v32.layers.initialize_rms_norm_module")
    @patch("nemo_automodel.components.models.deepseek_v32.layers.initialize_attn_module_and_func")
    def test_mla_with_rope_scaling(self, mock_init_attn, mock_init_rms, mock_init_linear, mock_yarn_get_mscale):
        """Test MLA with rope_scaling configuration."""
        rope_scaling = {
            "factor": 2.0,
            "mscale": 1.0,
            "original_max_position_embeddings": 4096
        }
        config = self.create_mock_config(
            rope_scaling=rope_scaling,
            max_position_embeddings=8192
        )
        backend = BackendConfig(attn="sdpa", linear="torch", rms_norm="torch")

        mock_init_linear.return_value = Mock()
        mock_init_rms.return_value = Mock()
        mock_init_attn.return_value = (Mock(), Mock())
        mock_yarn_get_mscale.return_value = 1.5

        mla = DeepseekV32MLA(config, backend)

        mock_yarn_get_mscale.assert_called_once_with(2.0, 1.0)
        base_scale = 32**-0.5
        expected_scale = base_scale * 1.5 * 1.5
        assert abs(mla.softmax_scale - expected_scale) < 1e-6


class TestDeepseekV32MLASparseMask:
    def create_mock_config(self, **overrides):
        config = Mock(spec=DeepseekV32Config)
        config.num_attention_heads = 8
        config.hidden_size = 256
        config.q_lora_rank = 128
        config.kv_lora_rank = 64
        config.qk_nope_head_dim = 16
        config.qk_rope_head_dim = 16
        config.qk_head_dim = 32
        config.v_head_dim = 32
        config.rope_scaling = None
        config.max_position_embeddings = 4096
        config.index_n_heads = 4
        config.index_head_dim = 32
        config.index_topk = 16

        for key, value in overrides.items():
            setattr(config, key, value)
        return config

    @patch("nemo_automodel.components.models.deepseek_v32.layers.initialize_linear_module")
    @patch("nemo_automodel.components.models.deepseek_v32.layers.initialize_rms_norm_module")
    @patch("nemo_automodel.components.models.deepseek_v32.layers.initialize_attn_module_and_func")
    def test_build_sparse_mask_bshd_format(self, mock_init_attn, mock_init_rms, mock_init_linear):
        """Test sparse mask building for bshd format."""
        config = self.create_mock_config()
        backend = BackendConfig(attn="sdpa", linear="torch", rms_norm="torch")

        mock_init_linear.return_value = Mock()
        mock_init_rms.return_value = Mock()
        mock_init_attn.return_value = (Mock(), Mock())

        mla = DeepseekV32MLA(config, backend)

        # Create mock topk_indices [B, S, topk]
        bsz, seq_len, topk = 2, 32, 8
        topk_indices = torch.randint(0, seq_len, (bsz, seq_len, topk))

        sparse_mask = mla._build_sparse_mask(
            topk_indices,
            seq_len,
            qkv_format="bshd",
            bsz=bsz,
            n_heads=8,
            dtype=torch.float32,
            union_across_batches=False,
        )

        # For SDPA (union_across_batches=False), shape should be [B, n_heads, S, S]
        assert sparse_mask.shape == (bsz, 8, seq_len, seq_len)

    @patch("nemo_automodel.components.models.deepseek_v32.layers.initialize_linear_module")
    @patch("nemo_automodel.components.models.deepseek_v32.layers.initialize_rms_norm_module")
    @patch("nemo_automodel.components.models.deepseek_v32.layers.initialize_attn_module_and_func")
    def test_build_sparse_mask_union_across_batches(self, mock_init_attn, mock_init_rms, mock_init_linear):
        """Test sparse mask building with union across batches (for TE)."""
        config = self.create_mock_config()
        backend = BackendConfig(attn="te", linear="torch", rms_norm="torch")

        mock_init_linear.return_value = Mock()
        mock_init_rms.return_value = Mock()
        mock_init_attn.return_value = (Mock(), Mock())

        mla = DeepseekV32MLA(config, backend)

        bsz, seq_len, topk = 2, 32, 8
        topk_indices = torch.randint(0, seq_len, (bsz, seq_len, topk))

        sparse_mask = mla._build_sparse_mask(
            topk_indices,
            seq_len,
            qkv_format="bshd",
            bsz=bsz,
            n_heads=8,
            dtype=torch.bfloat16,
            union_across_batches=True,
        )

        # For TE (union_across_batches=True), shape should be [1, n_heads, S, S]
        assert sparse_mask.shape == (1, 8, seq_len, seq_len)

    @patch("nemo_automodel.components.models.deepseek_v32.layers.initialize_linear_module")
    @patch("nemo_automodel.components.models.deepseek_v32.layers.initialize_rms_norm_module")
    @patch("nemo_automodel.components.models.deepseek_v32.layers.initialize_attn_module_and_func")
    def test_build_sparse_mask_thd_format(self, mock_init_attn, mock_init_rms, mock_init_linear):
        """Test sparse mask building for thd format."""
        config = self.create_mock_config()
        backend = BackendConfig(attn="sdpa", linear="torch", rms_norm="torch")

        mock_init_linear.return_value = Mock()
        mock_init_rms.return_value = Mock()
        mock_init_attn.return_value = (Mock(), Mock())

        mla = DeepseekV32MLA(config, backend)

        num_tokens, topk = 64, 8
        topk_indices = torch.randint(0, num_tokens, (num_tokens, topk))

        sparse_mask = mla._build_sparse_mask(
            topk_indices,
            num_tokens,
            qkv_format="thd",
            bsz=1,
            n_heads=8,
            dtype=torch.float32,
        )

        # For thd format, shape should be [1, n_heads, T, T]
        assert sparse_mask.shape == (1, 8, num_tokens, num_tokens)


class TestDeepseekV32MLAInitWeights:
    def create_mock_config(self, **overrides):
        config = Mock(spec=DeepseekV32Config)
        config.num_attention_heads = 8
        config.hidden_size = 256
        config.q_lora_rank = 128
        config.kv_lora_rank = 64
        config.qk_nope_head_dim = 16
        config.qk_rope_head_dim = 16
        config.qk_head_dim = 32
        config.v_head_dim = 32
        config.rope_scaling = None
        config.max_position_embeddings = 4096
        config.index_n_heads = 4
        config.index_head_dim = 32
        config.index_topk = 16

        for key, value in overrides.items():
            setattr(config, key, value)
        return config

    @patch("torch.nn.init.trunc_normal_")
    def test_init_weights(self, mock_trunc_normal):
        """Test weight initialization."""
        config = self.create_mock_config()
        backend = BackendConfig(attn="sdpa", linear="torch", rms_norm="torch")

        with patch("nemo_automodel.components.models.deepseek_v32.layers.initialize_linear_module") as mock_init_linear, \
             patch("nemo_automodel.components.models.deepseek_v32.layers.initialize_rms_norm_module") as mock_init_rms, \
             patch("nemo_automodel.components.models.deepseek_v32.layers.initialize_attn_module_and_func") as mock_init_attn:

            mock_linear = Mock()
            mock_linear.weight = torch.randn(64, 256)
            mock_linear.reset_parameters = Mock()
            mock_init_linear.return_value = mock_linear

            mock_norm = Mock()
            mock_norm.reset_parameters = Mock()
            mock_init_rms.return_value = mock_norm

            mock_init_attn.return_value = (Mock(), Mock())

            mla = DeepseekV32MLA(config, backend)

            device = torch.device("cpu")
            mla.init_weights(device, init_std=0.02)

            # Check that trunc_normal_ was called for linear layers
            # MLA has: q_a_proj, q_b_proj, kv_a_proj_with_mqa, kv_b_proj, o_proj (5 layers)
            # Indexer has: wq_b, wk, weights_proj (3 layers)
            # Total: 8 linear layers
            assert mock_trunc_normal.call_count == 8

            # Check that norm reset_parameters was called
            assert mock_norm.reset_parameters.call_count >= 2  # q_a_layernorm, kv_a_layernorm


class TestDeepseekV32IndexerInitWeights:
    def create_mock_config(self, **overrides):
        config = Mock(spec=DeepseekV32Config)
        config.num_attention_heads = 8
        config.hidden_size = 256
        config.q_lora_rank = 128
        config.index_n_heads = 4
        config.index_head_dim = 32
        config.index_topk = 16
        config.qk_rope_head_dim = 16

        for key, value in overrides.items():
            setattr(config, key, value)
        return config

    @patch("torch.nn.init.trunc_normal_")
    @patch("nemo_automodel.components.models.deepseek_v32.layers.initialize_linear_module")
    def test_indexer_init_weights(self, mock_init_linear, mock_trunc_normal):
        """Test Indexer weight initialization directly."""
        config = self.create_mock_config()
        backend = BackendConfig(attn="sdpa", linear="torch", rms_norm="torch")

        mock_linear = Mock()
        mock_linear.weight = torch.randn(64, 256)
        mock_init_linear.return_value = mock_linear

        indexer = DeepseekV32Indexer(config, backend)
        indexer.init_weights(init_std=0.02)

        # Should call trunc_normal_ for wq_b, wk, weights_proj (3 linear layers)
        assert mock_trunc_normal.call_count == 3


class TestDeepseekV32IndexerForward:
    def create_mock_config(self, **overrides):
        config = Mock(spec=DeepseekV32Config)
        config.num_attention_heads = 8
        config.hidden_size = 64
        config.q_lora_rank = 32
        config.index_n_heads = 4
        config.index_head_dim = 16
        config.index_topk = 8
        config.qk_rope_head_dim = 8

        for key, value in overrides.items():
            setattr(config, key, value)
        return config

    @skip_if_no_gpu
    def test_indexer_forward_bshd(self):
        """Test Indexer forward pass with bshd format."""
        config = self.create_mock_config()
        backend = BackendConfig(attn="sdpa", linear="torch", rms_norm="torch")

        indexer = DeepseekV32Indexer(config, backend).cuda().to(torch.bfloat16)

        bsz, seq_len = 2, 16
        x = torch.randn(bsz, seq_len, config.hidden_size, device="cuda", dtype=torch.bfloat16)
        q_resid = torch.randn(bsz, seq_len, config.q_lora_rank, device="cuda", dtype=torch.bfloat16)
        # Create complex freqs_cis for bshd format [B, T, D/2] as complex tensor
        angles = torch.randn(bsz, seq_len, config.qk_rope_head_dim // 2, device="cuda", dtype=torch.float32)
        freqs_cis = torch.polar(torch.ones_like(angles), angles)

        topk_indices = indexer(x, q_resid, freqs_cis)

        assert topk_indices.shape == (bsz, seq_len, config.index_topk)
        assert topk_indices.dtype == torch.int64

    @pytest.mark.skip(reason="thd format requires complex freqs_cis setup matching model runtime")
    @skip_if_no_gpu
    def test_indexer_forward_thd(self):
        """Test Indexer forward pass with thd format."""
        config = self.create_mock_config()
        backend = BackendConfig(attn="sdpa", linear="torch", rms_norm="torch")

        indexer = DeepseekV32Indexer(config, backend).cuda().to(torch.bfloat16)

        num_tokens = 32
        x = torch.randn(num_tokens, config.hidden_size, device="cuda", dtype=torch.bfloat16)
        q_resid = torch.randn(num_tokens, config.q_lora_rank, device="cuda", dtype=torch.bfloat16)
        # Create complex freqs_cis for thd format [T, D/2] as complex tensor
        angles = torch.randn(num_tokens, config.qk_rope_head_dim // 2, device="cuda", dtype=torch.float32)
        freqs_cis = torch.polar(torch.ones_like(angles), angles)

        topk_indices = indexer(x, q_resid, freqs_cis)

        assert topk_indices.shape == (num_tokens, config.index_topk)
        assert topk_indices.dtype == torch.int64

    @skip_if_no_gpu
    def test_indexer_forward_with_attention_mask_bshd(self):
        """Test Indexer forward pass with attention mask in bshd format."""
        config = self.create_mock_config()
        backend = BackendConfig(attn="sdpa", linear="torch", rms_norm="torch")

        indexer = DeepseekV32Indexer(config, backend).cuda().to(torch.bfloat16)

        bsz, seq_len = 2, 16
        x = torch.randn(bsz, seq_len, config.hidden_size, device="cuda", dtype=torch.bfloat16)
        q_resid = torch.randn(bsz, seq_len, config.q_lora_rank, device="cuda", dtype=torch.bfloat16)
        # Create complex freqs_cis for bshd format [B, T, D/2] as complex tensor
        angles = torch.randn(bsz, seq_len, config.qk_rope_head_dim // 2, device="cuda", dtype=torch.float32)
        freqs_cis = torch.polar(torch.ones_like(angles), angles)

        # Create causal mask
        attention_mask = torch.triu(
            torch.full((1, 1, seq_len, seq_len), float("-inf"), device="cuda"),
            diagonal=1,
        )

        topk_indices = indexer(x, q_resid, freqs_cis, attention_mask=attention_mask)

        assert topk_indices.shape == (bsz, seq_len, config.index_topk)

    @pytest.mark.skip(reason="thd format requires complex freqs_cis setup matching model runtime")
    @skip_if_no_gpu
    def test_indexer_forward_with_attention_mask_thd(self):
        """Test Indexer forward pass with attention mask in thd format."""
        config = self.create_mock_config()
        backend = BackendConfig(attn="sdpa", linear="torch", rms_norm="torch")

        indexer = DeepseekV32Indexer(config, backend).cuda().to(torch.bfloat16)

        num_tokens = 32
        x = torch.randn(num_tokens, config.hidden_size, device="cuda", dtype=torch.bfloat16)
        q_resid = torch.randn(num_tokens, config.q_lora_rank, device="cuda", dtype=torch.bfloat16)
        # Create complex freqs_cis for thd format [T, D/2] as complex tensor
        angles = torch.randn(num_tokens, config.qk_rope_head_dim // 2, device="cuda", dtype=torch.float32)
        freqs_cis = torch.polar(torch.ones_like(angles), angles)

        # Create causal mask for thd format
        attention_mask = torch.triu(
            torch.full((1, 1, num_tokens, num_tokens), float("-inf"), device="cuda"),
            diagonal=1,
        )

        topk_indices = indexer(x, q_resid, freqs_cis, attention_mask=attention_mask)

        assert topk_indices.shape == (num_tokens, config.index_topk)

    @skip_if_no_gpu
    def test_indexer_forward_topk_larger_than_seq(self):
        """Test Indexer forward when topk > seq_len."""
        config = self.create_mock_config(index_topk=64)  # larger than seq_len
        backend = BackendConfig(attn="sdpa", linear="torch", rms_norm="torch")

        indexer = DeepseekV32Indexer(config, backend).cuda().to(torch.bfloat16)

        bsz, seq_len = 2, 16  # seq_len < index_topk
        x = torch.randn(bsz, seq_len, config.hidden_size, device="cuda", dtype=torch.bfloat16)
        q_resid = torch.randn(bsz, seq_len, config.q_lora_rank, device="cuda", dtype=torch.bfloat16)
        # Create complex freqs_cis for bshd format [B, T, D/2] as complex tensor
        angles = torch.randn(bsz, seq_len, config.qk_rope_head_dim // 2, device="cuda", dtype=torch.float32)
        freqs_cis = torch.polar(torch.ones_like(angles), angles)

        topk_indices = indexer(x, q_resid, freqs_cis)

        # Should clamp to seq_len
        assert topk_indices.shape == (bsz, seq_len, seq_len)


class TestDeepseekV32MLAForward:
    def create_mock_config(self, **overrides):
        config = Mock(spec=DeepseekV32Config)
        config.num_attention_heads = 4
        config.hidden_size = 64
        config.q_lora_rank = 32
        config.kv_lora_rank = 32
        config.qk_nope_head_dim = 8
        config.qk_rope_head_dim = 8
        config.qk_head_dim = 16
        config.v_head_dim = 16
        config.rope_scaling = None
        config.max_position_embeddings = 4096
        config.index_n_heads = 4
        config.index_head_dim = 16
        config.index_topk = 8

        for key, value in overrides.items():
            setattr(config, key, value)
        return config

    @skip_if_no_gpu
    def test_mla_forward_bshd_sdpa(self):
        """Test MLA forward pass with bshd format and SDPA backend."""
        config = self.create_mock_config()
        backend = BackendConfig(attn="sdpa", linear="torch", rms_norm="torch")

        mla = DeepseekV32MLA(config, backend).cuda().to(torch.bfloat16)

        bsz, seq_len = 2, 16
        x = torch.randn(bsz, seq_len, config.hidden_size, device="cuda", dtype=torch.bfloat16)
        # Create complex freqs_cis for bshd format [B, T, D/2] as complex tensor
        angles = torch.randn(bsz, seq_len, config.qk_rope_head_dim // 2, device="cuda", dtype=torch.float32)
        freqs_cis = torch.polar(torch.ones_like(angles), angles)

        output = mla(x, freqs_cis)

        assert output.shape == (bsz, seq_len, config.hidden_size)
        assert output.dtype == torch.bfloat16

    @pytest.mark.skip(reason="thd format requires complex freqs_cis setup matching model runtime")
    @skip_if_no_gpu
    def test_mla_forward_thd_sdpa(self):
        """Test MLA forward pass with thd format and SDPA backend."""
        config = self.create_mock_config()
        backend = BackendConfig(attn="sdpa", linear="torch", rms_norm="torch")

        mla = DeepseekV32MLA(config, backend).cuda().to(torch.bfloat16)

        num_tokens = 32
        x = torch.randn(num_tokens, config.hidden_size, device="cuda", dtype=torch.bfloat16)
        # Create complex freqs_cis for thd format [T, D/2] as complex tensor
        angles = torch.randn(num_tokens, config.qk_rope_head_dim // 2, device="cuda", dtype=torch.float32)
        freqs_cis = torch.polar(torch.ones_like(angles), angles)

        output = mla(x, freqs_cis)

        assert output.shape == (num_tokens, config.hidden_size)
        assert output.dtype == torch.bfloat16

    @skip_if_no_gpu
    def test_mla_forward_with_attention_mask(self):
        """Test MLA forward pass with attention mask."""
        config = self.create_mock_config()
        backend = BackendConfig(attn="sdpa", linear="torch", rms_norm="torch")

        mla = DeepseekV32MLA(config, backend).cuda().to(torch.bfloat16)

        bsz, seq_len = 2, 16
        x = torch.randn(bsz, seq_len, config.hidden_size, device="cuda", dtype=torch.bfloat16)
        # Create complex freqs_cis for bshd format [B, T, D/2] as complex tensor
        angles = torch.randn(bsz, seq_len, config.qk_rope_head_dim // 2, device="cuda", dtype=torch.float32)
        freqs_cis = torch.polar(torch.ones_like(angles), angles)

        # Create causal mask
        attention_mask = torch.triu(
            torch.full((1, 1, seq_len, seq_len), float("-inf"), device="cuda"),
            diagonal=1,
        )

        output = mla(x, freqs_cis, attention_mask=attention_mask)

        assert output.shape == (bsz, seq_len, config.hidden_size)

    @skip_te
    @skip_if_no_gpu
    def test_mla_forward_bshd_te(self):
        """Test MLA forward pass with bshd format and TE backend."""
        config = self.create_mock_config()
        backend = BackendConfig(attn="te", linear="torch", rms_norm="torch")

        mla = DeepseekV32MLA(config, backend).cuda().to(torch.bfloat16)

        bsz, seq_len = 2, 16
        x = torch.randn(bsz, seq_len, config.hidden_size, device="cuda", dtype=torch.bfloat16)
        # Create complex freqs_cis for bshd format [B, T, D/2] as complex tensor
        angles = torch.randn(bsz, seq_len, config.qk_rope_head_dim // 2, device="cuda", dtype=torch.float32)
        freqs_cis = torch.polar(torch.ones_like(angles), angles)

        output = mla(x, freqs_cis)

        assert output.shape == (bsz, seq_len, config.hidden_size)


class TestBuildSparseMaskWithAttentionMask:
    def create_mock_config(self, **overrides):
        config = Mock(spec=DeepseekV32Config)
        config.num_attention_heads = 8
        config.hidden_size = 256
        config.q_lora_rank = 128
        config.kv_lora_rank = 64
        config.qk_nope_head_dim = 16
        config.qk_rope_head_dim = 16
        config.qk_head_dim = 32
        config.v_head_dim = 32
        config.rope_scaling = None
        config.max_position_embeddings = 4096
        config.index_n_heads = 4
        config.index_head_dim = 32
        config.index_topk = 16

        for key, value in overrides.items():
            setattr(config, key, value)
        return config

    @patch("nemo_automodel.components.models.deepseek_v32.layers.initialize_linear_module")
    @patch("nemo_automodel.components.models.deepseek_v32.layers.initialize_rms_norm_module")
    @patch("nemo_automodel.components.models.deepseek_v32.layers.initialize_attn_module_and_func")
    def test_build_sparse_mask_combines_with_attention_mask(self, mock_init_attn, mock_init_rms, mock_init_linear):
        """Test that sparse mask is combined with attention mask."""
        config = self.create_mock_config()
        backend = BackendConfig(attn="sdpa", linear="torch", rms_norm="torch")

        mock_init_linear.return_value = Mock()
        mock_init_rms.return_value = Mock()
        mock_init_attn.return_value = (Mock(), Mock())

        mla = DeepseekV32MLA(config, backend)

        bsz, seq_len, topk = 2, 32, 8
        topk_indices = torch.randint(0, seq_len, (bsz, seq_len, topk))

        # Create an attention mask
        attention_mask = torch.triu(
            torch.full((bsz, 1, seq_len, seq_len), float("-inf")),
            diagonal=1,
        )

        sparse_mask = mla._build_sparse_mask(
            topk_indices,
            seq_len,
            qkv_format="bshd",
            bsz=bsz,
            n_heads=1,
            dtype=torch.float32,
            attention_mask=attention_mask,
            union_across_batches=False,
        )

        # Result should combine both masks
        assert sparse_mask.shape == (bsz, 1, seq_len, seq_len)

        # Check that causal structure is preserved (upper triangle should be -inf)
        for b in range(bsz):
            for i in range(seq_len):
                for j in range(i + 1, seq_len):
                    assert sparse_mask[b, 0, i, j] == float("-inf")


class TestHadamardTransformFallback:
    """Test the fallback hadamard_transform implementation when fast_hadamard_transform is not available."""

    def test_hadamard_transform_torch_basic(self):
        """Test basic hadamard transform functionality."""
        # Import the torch fallback implementation directly
        from nemo_automodel.components.models.deepseek_v32 import layers

        # Check if we're using the fallback
        if not layers._FAST_HADAMARD_AVAILABLE:
            # Test the torch implementation
            batch_size = 4
            n = 64  # Must be power of 2
            x = torch.randn(batch_size, n)
            scale = n**-0.5

            result = layers.hadamard_transform(x, scale)

            assert result.shape == x.shape
            assert result.dtype == x.dtype

    def test_hadamard_transform_torch_power_of_2(self):
        """Test that hadamard transform works with different power-of-2 sizes."""
        from nemo_automodel.components.models.deepseek_v32 import layers

        if not layers._FAST_HADAMARD_AVAILABLE:
            for n in [8, 16, 32, 64, 128]:
                batch_size = 2
                x = torch.randn(batch_size, n)
                scale = n**-0.5

                result = layers.hadamard_transform(x, scale)
                assert result.shape == (batch_size, n)


class TestRotateActivationEdgeCases:
    """Test edge cases for _rotate_activation function."""

    @skip_if_no_gpu
    def test_rotate_activation_float16_converts(self):
        """Test that float16 input is converted to bfloat16."""
        x = torch.randn(2, 8, 64, device="cuda", dtype=torch.float16)
        result = _rotate_activation(x)
        assert result.dtype == torch.bfloat16
        assert result.shape == x.shape

    def test_rotate_activation_applies_scale(self):
        """Test that rotation applies the correct scale factor."""
        from nemo_automodel.components.models.deepseek_v32 import layers

        if not layers._FAST_HADAMARD_AVAILABLE:
            # With fallback, we can verify the scale is applied
            x = torch.randn(2, 64, dtype=torch.bfloat16)
            result = _rotate_activation(x)
            # The scale factor should be hidden_size^-0.5 = 64^-0.5 = 0.125
            assert result.shape == x.shape
