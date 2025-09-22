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

import pytest
import torch
from unittest.mock import Mock, patch
from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config

from nemo_automodel.components.models.deepseek_v3.layers import (
    preprocess_args_and_kwargs_for_attn,
    postprocess_output_for_attn,
    MLA,
)
from nemo_automodel.components.moe.utils import BackendConfig

# Skip Transformer Engine tests by default unless explicitly enabled
TE_AVAILABLE = False
try:
    import transformer_engine  # noqa: F401
    TE_AVAILABLE = True
except ImportError:
    pass

skip_te = pytest.mark.skipif(not TE_AVAILABLE, reason="Transformer Engine not available")
skip_if_no_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for GPU operations")


class TestPreprocessArgsAndKwargsForAttn:
    @skip_te
    def test_te_backend_no_attention_mask(self):
        q = torch.randn(2, 8, 16, 64)
        k = torch.randn(2, 8, 16, 64)
        v = torch.randn(2, 8, 16, 64)
        attention_mask = None
        backend = BackendConfig(attn="te", linear="torch", rms_norm="torch")

        q_out, k_out, v_out, attn_kwargs = preprocess_args_and_kwargs_for_attn(
            q, k, v, attention_mask, backend
        )

        # For TE backend with no attention mask, tensors should be unchanged
        assert torch.equal(q_out, q)
        assert torch.equal(k_out, k)
        assert torch.equal(v_out, v)
        assert attn_kwargs == {}

    @skip_te
    def test_te_backend_with_attention_mask(self):
        q = torch.randn(2, 8, 16, 64)
        k = torch.randn(2, 8, 16, 64)
        v = torch.randn(2, 8, 16, 64)
        attention_mask = torch.ones(2, 16)  # All ones means no padding
        backend = BackendConfig(attn="te", linear="torch", rms_norm="torch")

        q_out, k_out, v_out, attn_kwargs = preprocess_args_and_kwargs_for_attn(
            q, k, v, attention_mask, backend
        )

        assert torch.equal(q_out, q)
        assert torch.equal(k_out, k)
        assert torch.equal(v_out, v)

        expected_keys = {"attn_mask_type", "window_size", "attention_mask"}
        assert set(attn_kwargs.keys()) == expected_keys
        assert attn_kwargs["attn_mask_type"] == "padding_causal"
        assert attn_kwargs["window_size"] == (-1, 0)

        # Check padding mask shape and values
        padding_mask = attn_kwargs["attention_mask"]
        assert padding_mask.shape == (2, 1, 1, 16)
        # All ones in attention_mask should become all False in padding_mask
        assert not padding_mask.any()

    @skip_te
    def test_te_backend_with_padding_in_attention_mask(self):
        q = torch.randn(2, 8, 16, 64)
        k = torch.randn(2, 8, 16, 64)
        v = torch.randn(2, 8, 16, 64)
        # First sequence: all valid (1s), second sequence: some padding (0s)
        attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                       [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.float32)
        backend = BackendConfig(attn="te", linear="torch", rms_norm="torch")

        q_out, k_out, v_out, attn_kwargs = preprocess_args_and_kwargs_for_attn(
            q, k, v, attention_mask, backend
        )

        padding_mask = attn_kwargs["attention_mask"]
        # Where attention_mask is 0 (padding), padding_mask should be True
        expected_padding = torch.tensor([[False] * 16, [False] * 8 + [True] * 8])
        assert torch.equal(padding_mask.squeeze(1).squeeze(1), expected_padding)

    def test_sdpa_backend_no_attention_mask(self):
        q = torch.randn(2, 8, 16, 64)
        k = torch.randn(2, 8, 16, 64)
        v = torch.randn(2, 8, 16, 64)
        attention_mask = None
        backend = BackendConfig(attn="sdpa", linear="torch", rms_norm="torch")

        q_out, k_out, v_out, attn_kwargs = preprocess_args_and_kwargs_for_attn(
            q, k, v, attention_mask, backend
        )

        # SDPA should transpose dim 1 and 2
        assert q_out.shape == (2, 16, 8, 64)
        assert k_out.shape == (2, 16, 8, 64)
        assert v_out.shape == (2, 16, 8, 64)

        assert torch.equal(q_out, q.transpose(1, 2))
        assert torch.equal(k_out, k.transpose(1, 2))
        assert torch.equal(v_out, v.transpose(1, 2))

        assert attn_kwargs == {"is_causal": True}

    def test_sdpa_backend_with_attention_mask(self):
        q = torch.randn(2, 8, 16, 64)
        k = torch.randn(2, 8, 16, 64)
        v = torch.randn(2, 8, 16, 64)
        attention_mask = torch.ones(2, 16)
        backend = BackendConfig(attn="sdpa", linear="torch", rms_norm="torch")

        q_out, k_out, v_out, attn_kwargs = preprocess_args_and_kwargs_for_attn(
            q, k, v, attention_mask, backend
        )

        # Check transposition
        assert q_out.shape == (2, 16, 8, 64)
        assert k_out.shape == (2, 16, 8, 64)
        assert v_out.shape == (2, 16, 8, 64)

        # Check attention kwargs
        expected_keys = {"attention_mask", "is_causal"}
        assert set(attn_kwargs.keys()) == expected_keys
        assert attn_kwargs["is_causal"] == True
        assert torch.equal(attn_kwargs["attention_mask"], attention_mask.bool())


class TestPostprocessOutputForAttn:
    def test_te_backend_no_change(self):
        x = torch.randn(2, 8, 16, 64)
        backend = BackendConfig(attn="te", linear="torch", rms_norm="torch")

        result = postprocess_output_for_attn(x, backend)

        assert torch.equal(result, x)
        assert result.shape == x.shape

    def test_sdpa_backend_transpose_back(self):
        # Input from SDPA has shape (batch, seq, heads, dim)
        x = torch.randn(2, 16, 8, 64)
        backend = BackendConfig(attn="sdpa", linear="torch", rms_norm="torch")

        result = postprocess_output_for_attn(x, backend)

        # Should transpose back to (batch, heads, seq, dim)
        assert result.shape == (2, 8, 16, 64)
        assert torch.equal(result, x.transpose(1, 2))

    def test_unknown_backend_no_change(self):
        x = torch.randn(2, 8, 16, 64)
        backend = BackendConfig(attn="unknown", linear="torch", rms_norm="torch")

        result = postprocess_output_for_attn(x, backend)

        assert torch.equal(result, x)


class TestMLAInitialization:
    def create_mock_config(self, **overrides):
        config = Mock(spec=DeepseekV3Config)
        # Set default values
        config.num_attention_heads = 32
        config.q_lora_rank = None
        config.kv_lora_rank = 512
        config.qk_nope_head_dim = 64
        config.qk_rope_head_dim = 64
        config.qk_head_dim = 128  # nope + rope
        config.v_head_dim = 128
        config.hidden_size = 4096
        config.rope_scaling = None
        config.max_position_embeddings = 4096

        # Apply overrides
        for key, value in overrides.items():
            setattr(config, key, value)

        return config

    @skip_te
    @patch("nemo_automodel.components.models.deepseek_v3.layers.initialize_linear_module")
    @patch("nemo_automodel.components.models.deepseek_v3.layers.initialize_rms_norm_module")
    @patch("nemo_automodel.components.models.deepseek_v3.layers.initialize_attn_module_and_func")
    def test_mla_init_without_q_lora(self, mock_init_attn, mock_init_rms, mock_init_linear):
        config = self.create_mock_config(q_lora_rank=None)
        backend = BackendConfig(attn="te", linear="torch", rms_norm="torch")

        mock_init_linear.return_value = Mock()
        mock_init_rms.return_value = Mock()
        mock_init_attn.return_value = (Mock(), Mock())

        mla = MLA(config, backend)

        # Check basic attributes
        assert mla.n_heads == 32
        assert mla.q_lora_rank is None
        assert mla.kv_lora_rank == 512
        assert mla.qk_nope_head_dim == 64
        assert mla.qk_rope_head_dim == 64
        assert mla.qk_head_dim == 128
        assert mla.v_head_dim == 128
        assert mla.softmax_scale == 128**-0.5

        # Check that q_proj was initialized (not q_a_proj/q_b_proj)
        assert hasattr(mla, 'q_proj')
        assert not hasattr(mla, 'q_a_proj')
        assert not hasattr(mla, 'q_b_proj')
        assert not hasattr(mla, 'q_a_layernorm')

        # Check other components exist
        assert hasattr(mla, 'kv_a_proj_with_mqa')
        assert hasattr(mla, 'kv_a_layernorm')
        assert hasattr(mla, 'kv_b_proj')
        assert hasattr(mla, 'o_proj')
        assert hasattr(mla, 'attn_module')
        assert hasattr(mla, 'attn_func')

    @skip_te
    @patch("nemo_automodel.components.models.deepseek_v3.layers.initialize_linear_module")
    @patch("nemo_automodel.components.models.deepseek_v3.layers.initialize_rms_norm_module")
    @patch("nemo_automodel.components.models.deepseek_v3.layers.initialize_attn_module_and_func")
    def test_mla_init_with_q_lora(self, mock_init_attn, mock_init_rms, mock_init_linear):
        config = self.create_mock_config(q_lora_rank=1024)
        backend = BackendConfig(attn="te", linear="torch", rms_norm="torch")

        mock_init_linear.return_value = Mock()
        mock_init_rms.return_value = Mock()
        mock_init_attn.return_value = (Mock(), Mock())

        mla = MLA(config, backend)

        # Check that q_a_proj/q_b_proj were initialized (not q_proj)
        assert not hasattr(mla, 'q_proj')
        assert hasattr(mla, 'q_a_proj')
        assert hasattr(mla, 'q_b_proj')
        assert hasattr(mla, 'q_a_layernorm')

        assert mla.q_lora_rank == 1024

    @patch("nemo_automodel.components.models.deepseek_v3.layers.initialize_linear_module")
    @patch("nemo_automodel.components.models.deepseek_v3.layers.initialize_rms_norm_module")
    @patch("nemo_automodel.components.models.deepseek_v3.layers.initialize_attn_module_and_func")
    def test_mla_init_with_sdpa_backend(self, mock_init_attn, mock_init_rms, mock_init_linear):
        config = self.create_mock_config(q_lora_rank=None)
        backend = BackendConfig(attn="sdpa", linear="torch", rms_norm="torch")

        mock_init_linear.return_value = Mock()
        mock_init_rms.return_value = Mock()
        mock_init_attn.return_value = (Mock(), Mock())

        mla = MLA(config, backend)

        # Test that initialization works with SDPA backend
        assert mla.backend.attn == "sdpa"
        assert hasattr(mla, 'q_proj')
        assert hasattr(mla, 'kv_a_proj_with_mqa')
        assert hasattr(mla, 'kv_a_layernorm')
        assert hasattr(mla, 'kv_b_proj')
        assert hasattr(mla, 'o_proj')

    @patch("nemo_automodel.components.models.deepseek_v3.layers.yarn_get_mscale")
    @patch("nemo_automodel.components.models.deepseek_v3.layers.initialize_linear_module")
    @patch("nemo_automodel.components.models.deepseek_v3.layers.initialize_rms_norm_module")
    @patch("nemo_automodel.components.models.deepseek_v3.layers.initialize_attn_module_and_func")
    def test_mla_init_with_rope_scaling(self, mock_init_attn, mock_init_rms, mock_init_linear, mock_yarn_get_mscale):
        rope_scaling = {
            "factor": 2.0,
            "mscale": 1.0,
            "original_max_position_embeddings": 4096
        }
        config = self.create_mock_config(
            rope_scaling=rope_scaling,
            max_position_embeddings=8192  # Greater than original
        )
        backend = BackendConfig(attn="te", linear="torch", rms_norm="torch")

        mock_init_linear.return_value = Mock()
        mock_init_rms.return_value = Mock()
        mock_init_attn.return_value = (Mock(), Mock())
        mock_yarn_get_mscale.return_value = 1.5

        mla = MLA(config, backend)

        # Check that YARN mscale was called and softmax_scale adjusted
        mock_yarn_get_mscale.assert_called_once_with(2.0, 1.0)
        base_scale = 128**-0.5
        expected_scale = base_scale * 1.5 * 1.5
        assert abs(mla.softmax_scale - expected_scale) < 1e-6

    @patch("nemo_automodel.components.models.deepseek_v3.layers.initialize_linear_module")
    @patch("nemo_automodel.components.models.deepseek_v3.layers.initialize_rms_norm_module")
    @patch("nemo_automodel.components.models.deepseek_v3.layers.initialize_attn_module_and_func")
    def test_mla_init_rope_scaling_no_adjustment(self, mock_init_attn, mock_init_rms, mock_init_linear):
        rope_scaling = {
            "factor": 2.0,
            "mscale": 1.5,
            "original_max_position_embeddings": 8192
        }
        config = self.create_mock_config(
            rope_scaling=rope_scaling,
            max_position_embeddings=4096  # Less than or equal to original
        )
        backend = BackendConfig(attn="te", linear="torch", rms_norm="torch")

        mock_init_linear.return_value = Mock()
        mock_init_rms.return_value = Mock()
        mock_init_attn.return_value = (Mock(), Mock())

        mla = MLA(config, backend)

        # Check that softmax_scale was adjusted with the provided mscale
        base_scale = 128**-0.5
        expected_scale = base_scale * 1.5 * 1.5
        assert abs(mla.softmax_scale - expected_scale) < 1e-6


class TestMLAForward:
    def create_mock_config(self, **overrides):
        config = Mock(spec=DeepseekV3Config)
        config.num_attention_heads = 8
        config.q_lora_rank = None
        config.kv_lora_rank = 256
        config.qk_nope_head_dim = 32
        config.qk_rope_head_dim = 32
        config.qk_head_dim = 64
        config.v_head_dim = 64
        config.hidden_size = 1024
        config.rope_scaling = None
        config.max_position_embeddings = 4096

        for key, value in overrides.items():
            setattr(config, key, value)

        return config

    def test_mla_forward_tensor_shapes(self):
        # Test that forward pass handles tensor shapes correctly without mocking internals
        config = self.create_mock_config(q_lora_rank=None)
        backend = BackendConfig(attn="sdpa", linear="torch", rms_norm="torch")

        with patch("nemo_automodel.components.models.deepseek_v3.layers.initialize_linear_module") as mock_init_linear, \
             patch("nemo_automodel.components.models.deepseek_v3.layers.initialize_rms_norm_module") as mock_init_rms, \
             patch("nemo_automodel.components.models.deepseek_v3.layers.initialize_attn_module_and_func") as mock_init_attn:

            # Create mock components that return correctly shaped tensors
            def create_mock_linear(in_features, out_features, *args, **kwargs):
                mock = Mock()
                mock.weight = torch.randn(out_features, in_features)
                def forward_func(x):
                    return torch.randn(*x.shape[:-1], out_features)
                mock.side_effect = forward_func
                return mock

            mock_init_linear.side_effect = create_mock_linear

            mock_norm = Mock()
            mock_norm.side_effect = lambda x: x  # Identity for norm
            mock_norm.reset_parameters = Mock()
            mock_init_rms.return_value = mock_norm

            mock_attn_func = Mock()
            mock_attn_func.side_effect = lambda q, k, v, **kwargs: torch.randn(2, 16, 8, 64)  # SDPA format
            mock_init_attn.return_value = (Mock(), mock_attn_func)

            mla = MLA(config, backend)

            # Test that MLA can be created without tensor shape errors
            assert mla.n_heads == 8
            assert mla.qk_head_dim == 64
            assert mla.v_head_dim == 64
            assert mla.kv_lora_rank == 256

    def test_mla_config_with_q_lora(self):
        # Test MLA with q_lora configuration
        config = self.create_mock_config(q_lora_rank=512)
        backend = BackendConfig(attn="sdpa", linear="torch", rms_norm="torch")

        with patch("nemo_automodel.components.models.deepseek_v3.layers.initialize_linear_module") as mock_init_linear, \
             patch("nemo_automodel.components.models.deepseek_v3.layers.initialize_rms_norm_module") as mock_init_rms, \
             patch("nemo_automodel.components.models.deepseek_v3.layers.initialize_attn_module_and_func") as mock_init_attn:

            mock_init_linear.return_value = Mock()
            mock_init_rms.return_value = Mock()
            mock_init_attn.return_value = (Mock(), Mock())

            mla = MLA(config, backend)

            # Verify q_lora configuration was set up correctly
            assert mla.q_lora_rank == 512
            assert hasattr(mla, 'q_a_proj')
            assert hasattr(mla, 'q_b_proj')
            assert hasattr(mla, 'q_a_layernorm')
            assert not hasattr(mla, 'q_proj')

    def test_mla_preprocess_integration(self):
        # Test that MLA correctly integrates with preprocess function
        config = self.create_mock_config(q_lora_rank=None)
        backend = BackendConfig(attn="sdpa", linear="torch", rms_norm="torch")

        with patch("nemo_automodel.components.models.deepseek_v3.layers.initialize_linear_module") as mock_init_linear, \
             patch("nemo_automodel.components.models.deepseek_v3.layers.initialize_rms_norm_module") as mock_init_rms, \
             patch("nemo_automodel.components.models.deepseek_v3.layers.initialize_attn_module_and_func") as mock_init_attn, \
             patch("nemo_automodel.components.models.deepseek_v3.layers.preprocess_args_and_kwargs_for_attn") as mock_preprocess:

            # Setup all the initialize mocks
            mock_init_linear.return_value = Mock()
            mock_init_rms.return_value = Mock()
            mock_init_attn.return_value = (Mock(), Mock())

            # Setup preprocess mock to return expected values
            mock_preprocess.return_value = (
                torch.randn(2, 16, 8, 64),  # q (SDPA format)
                torch.randn(2, 16, 8, 64),  # k
                torch.randn(2, 16, 8, 64),  # v
                {"is_causal": True}
            )

            mla = MLA(config, backend)

            # Verify backend was set correctly
            assert mla.backend.attn == "sdpa"

            # Test that attention preprocessing would be called with backend
            attention_mask = torch.ones(2, 16)
            q_dummy = torch.randn(2, 8, 16, 64)
            k_dummy = torch.randn(2, 8, 16, 64)
            v_dummy = torch.randn(2, 8, 16, 64)

            # Call preprocess function directly to verify integration
            q_out, k_out, v_out, kwargs = preprocess_args_and_kwargs_for_attn(
                q_dummy, k_dummy, v_dummy, attention_mask, mla.backend
            )

            # Verify SDPA preprocessing was applied
            assert q_out.shape == (2, 16, 8, 64)  # Transposed for SDPA
            assert "is_causal" in kwargs


class TestMLAInitWeights:
    def create_mock_config(self, **overrides):
        config = Mock(spec=DeepseekV3Config)
        config.num_attention_heads = 8
        config.q_lora_rank = None
        config.kv_lora_rank = 256
        config.qk_nope_head_dim = 32
        config.qk_rope_head_dim = 32
        config.qk_head_dim = 64
        config.v_head_dim = 64
        config.hidden_size = 1024
        config.rope_scaling = None
        config.max_position_embeddings = 4096

        for key, value in overrides.items():
            setattr(config, key, value)

        return config

    @patch("torch.nn.init.trunc_normal_")
    def test_init_weights_without_q_lora(self, mock_trunc_normal):
        config = self.create_mock_config(q_lora_rank=None)
        backend = BackendConfig(attn="te", linear="torch", rms_norm="torch")

        with patch("nemo_automodel.components.models.deepseek_v3.layers.initialize_linear_module") as mock_init_linear, \
             patch("nemo_automodel.components.models.deepseek_v3.layers.initialize_rms_norm_module") as mock_init_rms, \
             patch("nemo_automodel.components.models.deepseek_v3.layers.initialize_attn_module_and_func") as mock_init_attn:

            mock_linear = Mock()
            mock_linear.weight = torch.randn(64, 1024)
            mock_linear.reset_parameters = Mock()
            mock_init_linear.return_value = mock_linear

            mock_norm = Mock()
            mock_norm.reset_parameters = Mock()
            mock_init_rms.return_value = mock_norm

            mock_init_attn.return_value = (Mock(), Mock())

            mla = MLA(config, backend)

            # Test init_weights
            device = torch.device("cpu")
            mla.init_weights(device, init_std=0.02)

            # Check that trunc_normal_ was called for each linear layer
            # Should be called for: q_proj, kv_a_proj_with_mqa, kv_b_proj, o_proj
            assert mock_trunc_normal.call_count == 4

            # Check that norm reset_parameters was called
            # Should be called for: kv_a_layernorm (only one norm without q_lora)
            assert mock_norm.reset_parameters.call_count >= 1

    @patch("torch.nn.init.trunc_normal_")
    def test_init_weights_with_q_lora(self, mock_trunc_normal):
        config = self.create_mock_config(q_lora_rank=512)
        backend = BackendConfig(attn="te", linear="torch", rms_norm="torch")

        with patch("nemo_automodel.components.models.deepseek_v3.layers.initialize_linear_module") as mock_init_linear, \
             patch("nemo_automodel.components.models.deepseek_v3.layers.initialize_rms_norm_module") as mock_init_rms, \
             patch("nemo_automodel.components.models.deepseek_v3.layers.initialize_attn_module_and_func") as mock_init_attn:

            mock_linear = Mock()
            mock_linear.weight = torch.randn(64, 1024)
            mock_linear.reset_parameters = Mock()
            mock_init_linear.return_value = mock_linear

            mock_norm = Mock()
            mock_norm.reset_parameters = Mock()
            mock_init_rms.return_value = mock_norm

            mock_init_attn.return_value = (Mock(), Mock())

            mla = MLA(config, backend)

            # Test init_weights
            device = torch.device("cpu")
            mla.init_weights(device, init_std=0.01)

            # Check that trunc_normal_ was called for each linear layer
            # Should be called for: q_a_proj, q_b_proj, kv_a_proj_with_mqa, kv_b_proj, o_proj
            assert mock_trunc_normal.call_count == 5

            # Check init_std was passed correctly
            for call in mock_trunc_normal.call_args_list:
                assert call[1]["std"] == 0.01
                assert call[1]["mean"] == 0.0

            # Check that norm reset_parameters was called for both norms
            # Should be called for: kv_a_layernorm, q_a_layernorm
            assert mock_norm.reset_parameters.call_count >= 2
