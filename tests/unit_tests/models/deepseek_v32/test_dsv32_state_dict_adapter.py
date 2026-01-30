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

# Mock fast_hadamard_transform before importing deepseek_v32 modules
try:
    import fast_hadamard_transform  # noqa: F401
except ImportError:
    if 'fast_hadamard_transform' not in sys.modules:
        mock_hadamard = types.ModuleType('fast_hadamard_transform')
        mock_hadamard.__spec__ = importlib.util.spec_from_loader('fast_hadamard_transform', loader=None)
        mock_hadamard.hadamard_transform = lambda x, scale: x
        sys.modules['fast_hadamard_transform'] = mock_hadamard

from nemo_automodel.components.models.deepseek_v32.config import DeepseekV32Config
from nemo_automodel.components.models.deepseek_v32.state_dict_adapter import DeepSeekV32StateDictAdapter
from nemo_automodel.components.moe.layers import MoEConfig
from nemo_automodel.components.models.common import BackendConfig


class TestDeepSeekV32StateDictAdapter:
    def create_mock_config(self, **overrides):
        config = Mock(spec=DeepseekV32Config)
        config.num_layers = 2
        config.hidden_size = 1024
        config.num_attention_heads = 16
        config.intermediate_size = 2048

        for key, value in overrides.items():
            setattr(config, key, value)

        return config

    def create_mock_moe_config(self, **overrides):
        moe_config = Mock(spec=MoEConfig)
        moe_config.num_experts = 8
        moe_config.n_routed_experts = 8
        moe_config.moe_inter_dim = 512
        moe_config.topk = 2

        for key, value in overrides.items():
            setattr(moe_config, key, value)

        return moe_config

    def create_mock_backend_config(self, **overrides):
        backend = Mock(spec=BackendConfig)
        backend.enable_deepep = False

        for key, value in overrides.items():
            setattr(backend, key, value)

        return backend

    def test_initialization(self):
        """Test adapter initialization."""
        config = self.create_mock_config()
        moe_config = self.create_mock_moe_config()
        backend = self.create_mock_backend_config()

        adapter = DeepSeekV32StateDictAdapter(
            config=config,
            moe_config=moe_config,
            backend=backend,
            dtype=torch.float16
        )

        assert adapter.config == config
        assert adapter.moe_config == moe_config
        assert adapter.backend == backend
        assert adapter.dtype == torch.float16

    def test_non_quantized_keys_includes_indexer(self):
        """Test that non-quantized keys include indexer LayerNorm keys."""
        config = self.create_mock_config()
        moe_config = self.create_mock_moe_config()
        backend = self.create_mock_backend_config()

        adapter = DeepSeekV32StateDictAdapter(config, moe_config, backend)

        non_quantized = adapter._non_quantized_keys

        # Should include base V3 keys
        assert "input_layernorm.weight" in non_quantized
        assert "post_attention_layernorm.weight" in non_quantized
        assert "norm.weight" in non_quantized
        assert "lm_head.weight" in non_quantized
        assert "embed_tokens.weight" in non_quantized
        assert "mlp.gate.weight" in non_quantized

        # Should include V3.2 indexer keys
        assert "indexer.k_norm.weight" in non_quantized
        assert "indexer.k_norm.bias" in non_quantized
        assert "indexer.weights_proj.weight" in non_quantized


class TestDeepSeekV32StateDictAdapterQuantization:
    def create_mock_config(self, **overrides):
        config = Mock(spec=DeepseekV32Config)
        config.num_layers = 2
        config.hidden_size = 1024
        config.num_attention_heads = 16
        config.intermediate_size = 2048

        for key, value in overrides.items():
            setattr(config, key, value)

        return config

    def create_mock_moe_config(self, **overrides):
        moe_config = Mock(spec=MoEConfig)
        moe_config.num_experts = 8
        moe_config.n_routed_experts = 8
        moe_config.moe_inter_dim = 512
        moe_config.topk = 2

        for key, value in overrides.items():
            setattr(moe_config, key, value)

        return moe_config

    def create_mock_backend_config(self, **overrides):
        backend = Mock(spec=BackendConfig)
        backend.enable_deepep = False

        for key, value in overrides.items():
            setattr(backend, key, value)

        return backend

    def test_convert_tensor_quantization_normal_weight(self):
        """Test that normal weights are quantized."""
        config = self.create_mock_config()
        moe_config = self.create_mock_moe_config()
        backend = self.create_mock_backend_config()

        adapter = DeepSeekV32StateDictAdapter(config, moe_config, backend)

        tensor = torch.randn(256, 128)
        fqn = "model.layers.0.self_attn.q_a_proj.weight"

        with patch.object(adapter, '_convert_single_merged_expert_to_hf_split_experts', return_value=None):
            result = adapter.convert_single_tensor_to_hf(fqn, tensor, quantization=True)

            assert len(result) == 2
            assert result[0][0] == fqn
            assert result[0][1].dtype == torch.float8_e4m3fn
            assert result[1][0] == fqn + "_scale_inv"
            assert result[1][1].dtype == torch.float32

    def test_convert_tensor_quantization_skips_layernorm(self):
        """Test that layernorm weights are not quantized."""
        config = self.create_mock_config()
        moe_config = self.create_mock_moe_config()
        backend = self.create_mock_backend_config()

        adapter = DeepSeekV32StateDictAdapter(config, moe_config, backend)

        tensor = torch.randn(256)
        fqn = "model.layers.0.input_layernorm.weight"

        with patch.object(adapter, '_convert_single_merged_expert_to_hf_split_experts', return_value=None):
            result = adapter.convert_single_tensor_to_hf(fqn, tensor, quantization=True)

            assert len(result) == 1
            assert result[0][0] == fqn
            assert result[0][1].dtype == tensor.dtype

    def test_convert_tensor_quantization_skips_indexer_k_norm(self):
        """Test that indexer k_norm weights are not quantized."""
        config = self.create_mock_config()
        moe_config = self.create_mock_moe_config()
        backend = self.create_mock_backend_config()

        adapter = DeepSeekV32StateDictAdapter(config, moe_config, backend)

        # Test k_norm weight
        tensor_weight = torch.randn(64)
        fqn_weight = "model.layers.0.self_attn.indexer.k_norm.weight"

        with patch.object(adapter, '_convert_single_merged_expert_to_hf_split_experts', return_value=None):
            result = adapter.convert_single_tensor_to_hf(fqn_weight, tensor_weight, quantization=True)

            assert len(result) == 1
            assert result[0][0] == fqn_weight
            assert result[0][1].dtype == tensor_weight.dtype

        # Test k_norm bias
        tensor_bias = torch.randn(64)
        fqn_bias = "model.layers.0.self_attn.indexer.k_norm.bias"

        with patch.object(adapter, '_convert_single_merged_expert_to_hf_split_experts', return_value=None):
            result = adapter.convert_single_tensor_to_hf(fqn_bias, tensor_bias, quantization=True)

            assert len(result) == 1
            assert result[0][0] == fqn_bias
            assert result[0][1].dtype == tensor_bias.dtype

    def test_convert_tensor_quantization_indexer_linear_weights(self):
        """Test that indexer linear weights (wq_b, wk) are quantized but weights_proj is not."""
        config = self.create_mock_config()
        moe_config = self.create_mock_moe_config()
        backend = self.create_mock_backend_config()

        adapter = DeepSeekV32StateDictAdapter(config, moe_config, backend)

        # Test wq_b weight - should be quantized
        tensor = torch.randn(256, 128)
        fqn = "model.layers.0.self_attn.indexer.wq_b.weight"

        with patch.object(adapter, '_convert_single_merged_expert_to_hf_split_experts', return_value=None):
            result = adapter.convert_single_tensor_to_hf(fqn, tensor, quantization=True)

            assert len(result) == 2
            assert result[0][0] == fqn
            assert result[0][1].dtype == torch.float8_e4m3fn
            assert result[1][0] == fqn + "_scale_inv"

        # Test wk weight - should be quantized
        tensor_wk = torch.randn(256, 128)
        fqn_wk = "model.layers.0.self_attn.indexer.wk.weight"

        with patch.object(adapter, '_convert_single_merged_expert_to_hf_split_experts', return_value=None):
            result = adapter.convert_single_tensor_to_hf(fqn_wk, tensor_wk, quantization=True)

            assert len(result) == 2
            assert result[0][0] == fqn_wk
            assert result[0][1].dtype == torch.float8_e4m3fn
            assert result[1][0] == fqn_wk + "_scale_inv"

    def test_convert_tensor_quantization_skips_indexer_weights_proj(self):
        """Test that indexer weights_proj is not quantized."""
        config = self.create_mock_config()
        moe_config = self.create_mock_moe_config()
        backend = self.create_mock_backend_config()

        adapter = DeepSeekV32StateDictAdapter(config, moe_config, backend)

        # Test weights_proj weight - should NOT be quantized
        tensor = torch.randn(256, 128)
        fqn = "model.layers.0.self_attn.indexer.weights_proj.weight"

        with patch.object(adapter, '_convert_single_merged_expert_to_hf_split_experts', return_value=None):
            result = adapter.convert_single_tensor_to_hf(fqn, tensor, quantization=True)

            assert len(result) == 1
            assert result[0][0] == fqn
            assert result[0][1].dtype == tensor.dtype  # Should not be quantized


class TestDeepSeekV32StateDictAdapterAddQuantization:
    def create_mock_config(self, **overrides):
        config = Mock(spec=DeepseekV32Config)
        config.num_layers = 2
        config.hidden_size = 1024
        config.num_attention_heads = 16
        config.intermediate_size = 2048

        for key, value in overrides.items():
            setattr(config, key, value)

        return config

    def create_mock_moe_config(self, **overrides):
        moe_config = Mock(spec=MoEConfig)
        moe_config.num_experts = 8
        moe_config.n_routed_experts = 8
        moe_config.moe_inter_dim = 512
        moe_config.topk = 2

        for key, value in overrides.items():
            setattr(moe_config, key, value)

        return moe_config

    def create_mock_backend_config(self, **overrides):
        backend = Mock(spec=BackendConfig)
        backend.enable_deepep = False

        for key, value in overrides.items():
            setattr(backend, key, value)

        return backend

    def test_add_quantization_scale_inv_normal_weights(self):
        """Test that _add_quantization_scale_inv_tensors adds scale_inv for normal weights."""
        config = self.create_mock_config()
        moe_config = self.create_mock_moe_config()
        backend = self.create_mock_backend_config()

        adapter = DeepSeekV32StateDictAdapter(config, moe_config, backend)

        state_dict = {
            "model.layers.0.self_attn.q_a_proj.weight": torch.randn(256, 128),
            "model.layers.0.input_layernorm.weight": torch.randn(256),
        }

        result = adapter._add_quantization_scale_inv_tensors(state_dict)

        # q_a_proj should be quantized
        assert "model.layers.0.self_attn.q_a_proj.weight" in result
        assert result["model.layers.0.self_attn.q_a_proj.weight"].dtype == torch.float8_e4m3fn
        assert "model.layers.0.self_attn.q_a_proj.weight_scale_inv" in result

        # layernorm should NOT be quantized
        assert "model.layers.0.input_layernorm.weight" in result
        assert result["model.layers.0.input_layernorm.weight"].dtype != torch.float8_e4m3fn
        assert "model.layers.0.input_layernorm.weight_scale_inv" not in result

    def test_add_quantization_scale_inv_skips_indexer_k_norm(self):
        """Test that _add_quantization_scale_inv_tensors skips indexer k_norm and weights_proj."""
        config = self.create_mock_config()
        moe_config = self.create_mock_moe_config()
        backend = self.create_mock_backend_config()

        adapter = DeepSeekV32StateDictAdapter(config, moe_config, backend)

        state_dict = {
            "model.layers.0.self_attn.indexer.wq_b.weight": torch.randn(256, 128),
            "model.layers.0.self_attn.indexer.wk.weight": torch.randn(256, 128),
            "model.layers.0.self_attn.indexer.k_norm.weight": torch.randn(64),
            "model.layers.0.self_attn.indexer.k_norm.bias": torch.randn(64),
            "model.layers.0.self_attn.indexer.weights_proj.weight": torch.randn(64, 256),
        }

        result = adapter._add_quantization_scale_inv_tensors(state_dict)

        # wq_b should be quantized
        assert "model.layers.0.self_attn.indexer.wq_b.weight" in result
        assert result["model.layers.0.self_attn.indexer.wq_b.weight"].dtype == torch.float8_e4m3fn
        assert "model.layers.0.self_attn.indexer.wq_b.weight_scale_inv" in result

        # wk should be quantized
        assert "model.layers.0.self_attn.indexer.wk.weight" in result
        assert result["model.layers.0.self_attn.indexer.wk.weight"].dtype == torch.float8_e4m3fn
        assert "model.layers.0.self_attn.indexer.wk.weight_scale_inv" in result

        # k_norm weight should NOT be quantized
        assert "model.layers.0.self_attn.indexer.k_norm.weight" in result
        assert result["model.layers.0.self_attn.indexer.k_norm.weight"].dtype != torch.float8_e4m3fn
        assert "model.layers.0.self_attn.indexer.k_norm.weight_scale_inv" not in result

        # k_norm bias should NOT have scale_inv (not .weight)
        assert "model.layers.0.self_attn.indexer.k_norm.bias" in result
        assert "model.layers.0.self_attn.indexer.k_norm.bias_scale_inv" not in result

        # weights_proj should NOT be quantized
        assert "model.layers.0.self_attn.indexer.weights_proj.weight" in result
        assert result["model.layers.0.self_attn.indexer.weights_proj.weight"].dtype != torch.float8_e4m3fn
        assert "model.layers.0.self_attn.indexer.weights_proj.weight_scale_inv" not in result
