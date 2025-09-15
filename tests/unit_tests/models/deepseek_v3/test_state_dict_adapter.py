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
from unittest.mock import Mock, MagicMock, patch
from transformers import DeepseekV3Config

from nemo_automodel.components.models.deepseek_v3.state_dict_adapter import (
    DeepSeekV3StateDictAdapter,
    calculate_scale_shape,
    dequantize_from_fp8,
    BLOCK_SIZE,
)
from nemo_automodel.components.moe.layers import MoEConfig
from nemo_automodel.components.moe.utils import BackendConfig

skip_if_no_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for GPU operations")


class TestDeepSeekV3StateDictAdapter:
    def create_mock_config(self, **overrides):
        config = Mock(spec=DeepseekV3Config)
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
        config = self.create_mock_config()
        moe_config = self.create_mock_moe_config()
        backend = self.create_mock_backend_config()

        adapter = DeepSeekV3StateDictAdapter(
            config=config,
            moe_config=moe_config,
            backend=backend,
            dtype=torch.float16
        )

        assert adapter.config == config
        assert adapter.moe_config == moe_config
        assert adapter.backend == backend
        assert adapter.dtype == torch.float16
        assert adapter._uses_model_prefix is True
        assert isinstance(adapter.from_hf_map, dict)
        assert len(adapter.from_hf_map) == 3

    def test_from_hf_map_structure(self):
        config = self.create_mock_config()
        moe_config = self.create_mock_moe_config()
        backend = self.create_mock_backend_config()

        adapter = DeepSeekV3StateDictAdapter(config, moe_config, backend)

        expected_keys = [
            "model.layers.{}.mlp.experts.{}.gate_proj.weight",
            "model.layers.{}.mlp.experts.{}.up_proj.weight",
            "model.layers.{}.mlp.experts.{}.down_proj.weight"
        ]

        assert list(adapter.from_hf_map.keys()) == expected_keys

    def test_dequantize_no_scale_inv(self):
        config = self.create_mock_config()
        moe_config = self.create_mock_moe_config()
        backend = self.create_mock_backend_config()

        adapter = DeepSeekV3StateDictAdapter(config, moe_config, backend)

        state_dict = {
            "layer1.weight": torch.randn(64, 32),
            "layer2.weight": torch.randn(128, 64),
        }

        result = adapter._dequantize(state_dict)

        assert len(result) == 2
        assert torch.equal(result["layer1.weight"], state_dict["layer1.weight"])
        assert torch.equal(result["layer2.weight"], state_dict["layer2.weight"])

    def test_dequantize_with_scale_inv(self):
        config = self.create_mock_config()
        moe_config = self.create_mock_moe_config()
        backend = self.create_mock_backend_config()

        adapter = DeepSeekV3StateDictAdapter(config, moe_config, backend, dtype=torch.float32)

        weight = torch.randn(256, 128, dtype=torch.float32).to(torch.float8_e4m3fn)
        scale_inv = torch.ones(2, 1, dtype=torch.float32)

        state_dict = {
            "layer1.weight": weight,
            "layer1.weight_scale_inv": scale_inv,
            "layer2.weight": torch.randn(64, 32),
        }

        with patch('nemo_automodel.components.models.deepseek_v3.state_dict_adapter.dequantize_from_fp8') as mock_dequant:
            mock_dequant.return_value = torch.randn(256, 128, dtype=torch.float32)

            result = adapter._dequantize(state_dict)

            assert len(result) == 2
            assert "layer1.weight_scale_inv" not in result
            assert "layer2.weight" in result
            mock_dequant.assert_called_once_with(weight, scale_inv, dtype=torch.float32)

    def test_add_quantization_scale_inv_tensors_cpu(self):
        config = self.create_mock_config()
        moe_config = self.create_mock_moe_config()
        backend = self.create_mock_backend_config()

        adapter = DeepSeekV3StateDictAdapter(config, moe_config, backend)

        state_dict = {
            "model.layers.0.self_attn.q_proj.weight": torch.randn(512, 256),
            "model.layers.0.input_layernorm.weight": torch.randn(256),
            "model.layers.0.mlp.gate.weight": torch.randn(128, 256),
            "model.embed_tokens.weight": torch.randn(1000, 256),
        }

        result = adapter._add_quantization_scale_inv_tensors(state_dict)

        assert "model.layers.0.self_attn.q_proj.weight_scale_inv" in result
        assert "model.layers.0.input_layernorm.weight_scale_inv" not in result
        assert "model.layers.0.mlp.gate.weight_scale_inv" not in result
        assert "model.embed_tokens.weight_scale_inv" not in result

        assert result["model.layers.0.self_attn.q_proj.weight"].dtype == torch.float8_e4m3fn

    @skip_if_no_gpu
    def test_add_quantization_scale_inv_tensors_gpu(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        config = self.create_mock_config()
        moe_config = self.create_mock_moe_config()
        backend = self.create_mock_backend_config()

        adapter = DeepSeekV3StateDictAdapter(config, moe_config, backend)

        device = torch.device("cuda")
        state_dict = {
            "model.layers.0.self_attn.q_proj.weight": torch.randn(512, 256, device=device),
            "model.layers.0.input_layernorm.weight": torch.randn(256, device=device),
        }

        result = adapter._add_quantization_scale_inv_tensors(state_dict)

        scale_inv = result["model.layers.0.self_attn.q_proj.weight_scale_inv"]
        assert scale_inv.device.type == device.type
        assert scale_inv.dtype == torch.float32

    def test_to_hf_deepep_backend(self):
        config = self.create_mock_config()
        moe_config = self.create_mock_moe_config()
        backend = self.create_mock_backend_config(enable_deepep=True)

        adapter = DeepSeekV3StateDictAdapter(config, moe_config, backend)

        state_dict = {"test_key": torch.randn(10, 10)}

        with patch.object(adapter, '_to_hf_deepep') as mock_deepep:
            mock_deepep.return_value = {"converted_key": torch.randn(10, 10)}

            result = adapter.to_hf(state_dict)

            mock_deepep.assert_called_once_with(state_dict)
            assert "converted_key" in result

    def test_to_hf_grouped_experts_backend(self):
        config = self.create_mock_config()
        moe_config = self.create_mock_moe_config()
        backend = self.create_mock_backend_config(enable_deepep=False)

        adapter = DeepSeekV3StateDictAdapter(config, moe_config, backend)

        state_dict = {"test_key": torch.randn(10, 10)}

        with patch.object(adapter, '_to_hf_grouped_experts') as mock_grouped:
            mock_grouped.return_value = {"converted_key": torch.randn(10, 10)}

            result = adapter.to_hf(state_dict)

            mock_grouped.assert_called_once_with(state_dict)
            assert "converted_key" in result

    def test_to_hf_with_exclude_regex(self):
        config = self.create_mock_config()
        moe_config = self.create_mock_moe_config()
        backend = self.create_mock_backend_config(enable_deepep=False)

        adapter = DeepSeekV3StateDictAdapter(config, moe_config, backend)

        state_dict = {"test_key": torch.randn(10, 10)}

        with patch.object(adapter, '_to_hf_grouped_experts') as mock_grouped:
            mock_grouped.return_value = {
                "keep_this": torch.randn(5, 5),
                "exclude_this": torch.randn(5, 5),
                "also_keep": torch.randn(5, 5)
            }

            result = adapter.to_hf(state_dict, exclude_key_regex=r"exclude.*")

            assert "keep_this" in result
            assert "also_keep" in result
            assert "exclude_this" not in result

    def test_to_hf_is_base_true(self):
        config = self.create_mock_config()
        moe_config = self.create_mock_moe_config()
        backend = self.create_mock_backend_config(enable_deepep=False)

        adapter = DeepSeekV3StateDictAdapter(config, moe_config, backend)

        state_dict = {"test_key": torch.randn(10, 10)}

        with patch.object(adapter, '_to_hf_grouped_experts') as mock_grouped, \
             patch.object(adapter, '_add_quantization_scale_inv_tensors') as mock_add_quant:

            mock_grouped.return_value = {"converted_key": torch.randn(10, 10)}
            mock_add_quant.return_value = {"quantized_key": torch.randn(10, 10)}

            result = adapter.to_hf(state_dict, is_base=True)

            mock_add_quant.assert_called_once()
            assert "quantized_key" in result

    def test_from_hf_detects_model_prefix(self):
        config = self.create_mock_config()
        moe_config = self.create_mock_moe_config()
        backend = self.create_mock_backend_config(enable_deepep=False)

        adapter = DeepSeekV3StateDictAdapter(config, moe_config, backend)

        hf_state_dict = {
            "model.layers.0.mlp.experts.0.gate_proj.weight": torch.randn(128, 256),
            "model.layers.0.attention.weight": torch.randn(256, 256),
        }

        with patch.object(adapter, '_dequantize') as mock_dequant, \
             patch.object(adapter, '_from_hf_grouped_experts') as mock_from_grouped:

            mock_dequant.return_value = hf_state_dict
            mock_from_grouped.return_value = {"converted": torch.randn(10, 10)}

            adapter.from_hf(hf_state_dict)

            assert adapter._uses_model_prefix is True

    def test_from_hf_no_model_prefix(self):
        config = self.create_mock_config()
        moe_config = self.create_mock_moe_config()
        backend = self.create_mock_backend_config(enable_deepep=False)

        adapter = DeepSeekV3StateDictAdapter(config, moe_config, backend)

        hf_state_dict = {
            "layers.0.mlp.experts.0.gate_proj.weight": torch.randn(128, 256),
            "layers.0.attention.weight": torch.randn(256, 256),
        }

        with patch.object(adapter, '_dequantize') as mock_dequant, \
             patch.object(adapter, '_from_hf_grouped_experts') as mock_from_grouped:

            mock_dequant.return_value = hf_state_dict
            mock_from_grouped.return_value = {"converted": torch.randn(10, 10)}

            adapter.from_hf(hf_state_dict)

            assert adapter._uses_model_prefix is False

    def test_from_hf_auto_deepep_backend(self):
        config = self.create_mock_config()
        moe_config = self.create_mock_moe_config()
        backend = self.create_mock_backend_config(enable_deepep=True)

        adapter = DeepSeekV3StateDictAdapter(config, moe_config, backend)

        hf_state_dict = {"test_key": torch.randn(10, 10)}

        with patch.object(adapter, '_dequantize') as mock_dequant, \
             patch.object(adapter, '_from_hf_deepep') as mock_from_deepep:

            mock_dequant.return_value = hf_state_dict
            mock_from_deepep.return_value = {"converted": torch.randn(10, 10)}

            result = adapter.from_hf(hf_state_dict, target_format="auto")

            mock_from_deepep.assert_called_once()
            assert "converted" in result

    def test_from_hf_auto_grouped_experts_backend(self):
        config = self.create_mock_config()
        moe_config = self.create_mock_moe_config()
        backend = self.create_mock_backend_config(enable_deepep=False)

        adapter = DeepSeekV3StateDictAdapter(config, moe_config, backend)

        hf_state_dict = {"test_key": torch.randn(10, 10)}

        with patch.object(adapter, '_dequantize') as mock_dequant, \
             patch.object(adapter, '_from_hf_grouped_experts') as mock_from_grouped:

            mock_dequant.return_value = hf_state_dict
            mock_from_grouped.return_value = {"converted": torch.randn(10, 10)}

            result = adapter.from_hf(hf_state_dict, target_format="auto")

            mock_from_grouped.assert_called_once()
            assert "converted" in result

    def test_from_hf_explicit_target_format_deepep(self):
        config = self.create_mock_config()
        moe_config = self.create_mock_moe_config()
        backend = self.create_mock_backend_config(enable_deepep=False)  # Backend is grouped but override

        adapter = DeepSeekV3StateDictAdapter(config, moe_config, backend)

        hf_state_dict = {"test_key": torch.randn(10, 10)}

        with patch.object(adapter, '_dequantize') as mock_dequant, \
             patch.object(adapter, '_from_hf_deepep') as mock_from_deepep:

            mock_dequant.return_value = hf_state_dict
            mock_from_deepep.return_value = {"converted": torch.randn(10, 10)}

            result = adapter.from_hf(hf_state_dict, target_format="deepep")

            mock_from_deepep.assert_called_once()
            assert "converted" in result

    def test_from_hf_explicit_target_format_grouped_experts(self):
        config = self.create_mock_config()
        moe_config = self.create_mock_moe_config()
        backend = self.create_mock_backend_config(enable_deepep=True)  # Backend is deepep but override

        adapter = DeepSeekV3StateDictAdapter(config, moe_config, backend)

        hf_state_dict = {"test_key": torch.randn(10, 10)}

        with patch.object(adapter, '_dequantize') as mock_dequant, \
             patch.object(adapter, '_from_hf_grouped_experts') as mock_from_grouped:

            mock_dequant.return_value = hf_state_dict
            mock_from_grouped.return_value = {"converted": torch.randn(10, 10)}

            result = adapter.from_hf(hf_state_dict, target_format="grouped_experts")

            mock_from_grouped.assert_called_once()
            assert "converted" in result

    def test_from_hf_invalid_target_format(self):
        config = self.create_mock_config()
        moe_config = self.create_mock_moe_config()
        backend = self.create_mock_backend_config()

        adapter = DeepSeekV3StateDictAdapter(config, moe_config, backend)

        hf_state_dict = {"test_key": torch.randn(10, 10)}

        with patch.object(adapter, '_dequantize') as mock_dequant:
            mock_dequant.return_value = hf_state_dict

            with pytest.raises(ValueError, match="target_format must be"):
                adapter.from_hf(hf_state_dict, target_format="invalid_format")


class TestCalculateScaleShape:
    def test_exact_blocks(self):
        weight = torch.randn(256, 128)  # 2x1 blocks
        expected_shape = torch.Size((2, 1))

        result = calculate_scale_shape(weight)

        assert result == expected_shape

    def test_partial_blocks(self):
        weight = torch.randn(200, 100)  # 2x1 blocks (200/128=1.56->2, 100/128=0.78->1)
        expected_shape = torch.Size((2, 1))

        result = calculate_scale_shape(weight)

        assert result == expected_shape

    def test_single_block(self):
        weight = torch.randn(64, 32)  # 1x1 blocks
        expected_shape = torch.Size((1, 1))

        result = calculate_scale_shape(weight)

        assert result == expected_shape

    def test_large_tensor(self):
        weight = torch.randn(1024, 512)  # 8x4 blocks
        expected_shape = torch.Size((8, 4))

        result = calculate_scale_shape(weight)

        assert result == expected_shape

    def test_custom_block_size(self):
        weight = torch.randn(200, 100)
        custom_block_size = 50
        # 200/50=4, 100/50=2
        expected_shape = torch.Size((4, 2))

        result = calculate_scale_shape(weight, custom_block_size)

        assert result == expected_shape

    def test_minimal_tensor(self):
        weight = torch.randn(1, 1)  # Very small tensor
        expected_shape = torch.Size((1, 1))

        result = calculate_scale_shape(weight)

        assert result == expected_shape


class TestDequantizeFromFp8:
    def test_dequantize_single_block(self):
        weight = torch.randn(64, 32, dtype=torch.float32).to(torch.float8_e4m3fn)
        scale_inv = torch.tensor([[2.0]], dtype=torch.float32)

        result = dequantize_from_fp8(weight, scale_inv, dtype=torch.float32)

        assert result.dtype == torch.float32
        assert result.shape == weight.shape

    def test_dequantize_multiple_blocks(self):
        weight = torch.randn(256, 128, dtype=torch.float32).to(torch.float8_e4m3fn)
        scale_inv = torch.ones((2, 1), dtype=torch.float32) * 1.5

        result = dequantize_from_fp8(weight, scale_inv, dtype=torch.bfloat16)

        assert result.dtype == torch.bfloat16
        assert result.shape == weight.shape

    @skip_if_no_gpu
    def test_dequantize_device_mismatch_handling(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        device = torch.device("cuda")
        weight = torch.randn(128, 64, dtype=torch.float32, device=device).to(torch.float8_e4m3fn)
        scale_inv = torch.ones((1, 1), dtype=torch.float32)  # CPU tensor

        result = dequantize_from_fp8(weight, scale_inv, dtype=torch.float32)

        assert result.device.type == device.type
        assert result.dtype == torch.float32

    def test_dequantize_mismatched_scale_shape_warning(self):
        weight = torch.randn(256, 128, dtype=torch.float32).to(torch.float8_e4m3fn)
        scale_inv = torch.ones((2, 1), dtype=torch.float32)  # Correct shape for 256x128 tensor

        with patch('nemo_automodel.components.models.deepseek_v3.state_dict_adapter.logger') as mock_logger:
            result = dequantize_from_fp8(weight, scale_inv, dtype=torch.float32)

            # No warning should be called for correct shape
            mock_logger.warning.assert_not_called()
            assert result.shape == weight.shape

    def test_dequantize_mismatched_scale_shape_warning_actual_mismatch(self):
        weight = torch.randn(128, 64, dtype=torch.float32).to(torch.float8_e4m3fn)  # Should be (1, 1) scale shape
        scale_inv = torch.ones((2, 1), dtype=torch.float32)  # Wrong shape - too many scale values

        with patch('nemo_automodel.components.models.deepseek_v3.state_dict_adapter.logger') as mock_logger:
            # This will still process but use only the available scale values
            result = dequantize_from_fp8(weight, scale_inv, dtype=torch.float32)

            mock_logger.warning.assert_called_once()
            assert "scale_inv shape" in mock_logger.warning.call_args[0][0]
            assert result.shape == weight.shape

    def test_dequantize_custom_block_size(self):
        weight = torch.randn(100, 50, dtype=torch.float32).to(torch.float8_e4m3fn)
        custom_block_size = 25
        # 100/25=4, 50/25=2
        scale_inv = torch.ones((4, 2), dtype=torch.float32) * 0.5

        result = dequantize_from_fp8(weight, scale_inv, dtype=torch.float32, BLOCK_SIZE=custom_block_size)

        assert result.dtype == torch.float32
        assert result.shape == weight.shape

    def test_dequantize_partial_blocks(self):
        weight = torch.randn(200, 100, dtype=torch.float32).to(torch.float8_e4m3fn)
        scale_inv = torch.tensor([[1.0], [2.0]], dtype=torch.float32)  # 2x1 scale for partial blocks

        result = dequantize_from_fp8(weight, scale_inv, dtype=torch.float16)

        assert result.dtype == torch.float16
        assert result.shape == weight.shape

    def test_dequantize_default_dtype(self):
        weight = torch.randn(128, 128, dtype=torch.float32).to(torch.float8_e4m3fn)
        scale_inv = torch.ones((1, 1), dtype=torch.float32)

        result = dequantize_from_fp8(weight, scale_inv)  # Should default to bfloat16

        assert result.dtype == torch.bfloat16
        assert result.shape == weight.shape

    def test_dequantize_edge_case_small_tensor(self):
        weight = torch.randn(1, 1, dtype=torch.float32).to(torch.float8_e4m3fn)
        scale_inv = torch.tensor([[3.0]], dtype=torch.float32)

        result = dequantize_from_fp8(weight, scale_inv, dtype=torch.float32)

        assert result.dtype == torch.float32
        assert result.shape == (1, 1)
