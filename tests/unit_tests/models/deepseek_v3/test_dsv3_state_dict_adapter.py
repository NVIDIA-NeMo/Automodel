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
    _dequantize_with_torch,
    _dequantize_with_triton,
    _slice_scale_for_dtensor,
    should_quantize_key,
    create_scale_inv_for_weight,
    NON_QUANTIZED_KEY_PATTERNS,
    _TRITON_AVAILABLE,
    BLOCK_SIZE,
)
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.models.common import BackendConfig

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
        moe_config.n_routed_experts = 8
        moe_config.moe_inter_dim = 512
        moe_config.topk = 2

        for key, value in overrides.items():
            setattr(moe_config, key, value)

        return moe_config

    def create_mock_backend_config(self, **overrides):
        backend = Mock(spec=BackendConfig)
        backend.dispatcher = "torch"
        backend.experts = "torch"

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
            mock_dequant.assert_called_once_with(weight, scale_inv, dtype=torch.float32, name="layer1.weight")

    def test_to_hf(self):
        config = self.create_mock_config()
        moe_config = self.create_mock_moe_config()
        backend = self.create_mock_backend_config(dispatcher="deepep", experts="te")

        adapter = DeepSeekV3StateDictAdapter(config, moe_config, backend)

        state_dict = {"test_key": torch.randn(10, 10)}

        with patch.object(adapter, 'convert_single_tensor_to_hf') as mock_convert:
            mock_convert.return_value = [("converted_key", torch.randn(10, 10))]

            result = adapter.to_hf(state_dict)

            mock_convert.assert_called_once()
            assert "converted_key" in result

    def test_to_hf_with_exclude_regex(self):
        config = self.create_mock_config()
        moe_config = self.create_mock_moe_config()
        backend = self.create_mock_backend_config(dispatcher="torch")

        adapter = DeepSeekV3StateDictAdapter(config, moe_config, backend)

        state_dict = {"test_key": torch.randn(10, 10)}

        with patch.object(adapter, 'convert_single_tensor_to_hf') as mock_convert:
            mock_convert.return_value = [
                ("keep_this", torch.randn(5, 5)),
                ("also_keep", torch.randn(5, 5))
            ]

            result = adapter.to_hf(state_dict, exclude_key_regex=r"exclude.*")

            assert "keep_this" in result
            assert "also_keep" in result
            assert "exclude_this" not in result

    def test_to_hf_quantization_true(self):
        config = self.create_mock_config()
        moe_config = self.create_mock_moe_config()
        backend = self.create_mock_backend_config(dispatcher="torch")

        adapter = DeepSeekV3StateDictAdapter(config, moe_config, backend)

        state_dict = {"test_key": torch.randn(10, 10)}

        with patch.object(adapter, 'convert_single_tensor_to_hf') as mock_convert:
            mock_convert.return_value = [("quantized_key", torch.randn(10, 10))]

            result = adapter.to_hf(state_dict, quantization=True)

            mock_convert.assert_called_once()
            assert "quantized_key" in result

    def test_to_hf_quantization_false(self):
        config = self.create_mock_config()
        moe_config = self.create_mock_moe_config()
        backend = self.create_mock_backend_config(dispatcher="torch")

        adapter = DeepSeekV3StateDictAdapter(config, moe_config, backend)

        weight = torch.randn(8, 8)
        state_dict = {"test_key": weight}

        with patch.object(adapter, 'convert_single_tensor_to_hf') as mock_convert:
            mock_convert.return_value = [("keep_key.weight", weight.clone())]

            result = adapter.to_hf(state_dict, quantization=False)

            mock_convert.assert_called_once()
            assert "keep_key.weight" in result
            assert "keep_key.weight_scale_inv" not in result
            assert result["keep_key.weight"].dtype == weight.dtype

    def test_to_hf_exclude_then_quantize(self):
        config = self.create_mock_config()
        moe_config = self.create_mock_moe_config()
        backend = self.create_mock_backend_config(dispatcher="torch")

        adapter = DeepSeekV3StateDictAdapter(config, moe_config, backend)

        state_dict = {"test_key": torch.randn(16, 16)}

        with patch.object(adapter, 'convert_single_tensor_to_hf') as mock_convert:
            mock_convert.return_value = [
                ("keep_key.weight", torch.randn(16, 16).to(torch.float8_e4m3fn)),
                ("keep_key.weight_scale_inv", torch.ones(1, 1))
            ]

            result = adapter.to_hf(state_dict, exclude_key_regex=r"exclude.*", quantization=True)

            assert "exclude_key.weight" not in result
            assert not any(k.startswith("exclude_key.") for k in result.keys())
            assert "keep_key.weight" in result
            assert "keep_key.weight_scale_inv" in result
            assert result["keep_key.weight"].dtype == torch.float8_e4m3fn

    def test_from_hf_detects_model_prefix(self):
        config = self.create_mock_config()
        moe_config = self.create_mock_moe_config()
        backend = self.create_mock_backend_config(dispatcher="torch")

        adapter = DeepSeekV3StateDictAdapter(config, moe_config, backend)

        hf_state_dict = {
            "model.layers.0.mlp.experts.0.gate_proj.weight": torch.randn(128, 256),
            "model.layers.0.attention.weight": torch.randn(256, 256),
        }

        with patch.object(adapter, '_dequantize') as mock_dequant, \
             patch.object(adapter, '_from_hf_w_merged_experts') as mock_from_hf:

            mock_dequant.return_value = hf_state_dict
            mock_from_hf.return_value = {"converted": torch.randn(10, 10)}

            adapter.from_hf(hf_state_dict)

            assert adapter._uses_model_prefix is True

    def test_from_hf_no_model_prefix(self):
        config = self.create_mock_config()
        moe_config = self.create_mock_moe_config()
        backend = self.create_mock_backend_config(dispatcher="torch")

        adapter = DeepSeekV3StateDictAdapter(config, moe_config, backend)

        hf_state_dict = {
            "layers.0.mlp.experts.0.gate_proj.weight": torch.randn(128, 256),
            "layers.0.attention.weight": torch.randn(256, 256),
        }

        with patch.object(adapter, '_dequantize') as mock_dequant, \
             patch.object(adapter, '_from_hf_w_merged_experts') as mock_from_hf:

            mock_dequant.return_value = hf_state_dict
            mock_from_hf.return_value = {"converted": torch.randn(10, 10)}

            adapter.from_hf(hf_state_dict)

            assert adapter._uses_model_prefix is False

    def test_from_hf(self):
        config = self.create_mock_config()
        moe_config = self.create_mock_moe_config()
        backend = self.create_mock_backend_config(dispatcher="deepep", experts="te")

        adapter = DeepSeekV3StateDictAdapter(config, moe_config, backend)

        hf_state_dict = {"test_key": torch.randn(10, 10)}

        with patch.object(adapter, '_dequantize') as mock_dequant, \
             patch.object(adapter, '_from_hf_w_merged_experts') as mock_from_hf:

            mock_dequant.return_value = hf_state_dict
            mock_from_hf.return_value = {"converted": torch.randn(10, 10)}

            result = adapter.from_hf(hf_state_dict)

            mock_from_hf.assert_called_once()
            assert "converted" in result


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


class TestShouldQuantizeKey:
    """Tests for should_quantize_key function."""

    def test_quantizable_weight_keys(self):
        """Test that normal weight keys are marked for quantization."""
        assert should_quantize_key("model.layers.0.self_attn.q_proj.weight") is True
        assert should_quantize_key("model.layers.0.mlp.experts.0.gate_proj.weight") is True
        assert should_quantize_key("model.layers.0.mlp.experts.0.up_proj.weight") is True
        assert should_quantize_key("model.layers.0.mlp.experts.0.down_proj.weight") is True

    def test_non_quantizable_layernorm_keys(self):
        """Test that layernorm weights are not quantized."""
        assert should_quantize_key("model.layers.0.input_layernorm.weight") is False
        assert should_quantize_key("model.layers.0.post_attention_layernorm.weight") is False

    def test_non_quantizable_special_keys(self):
        """Test that special weights are not quantized."""
        assert should_quantize_key("model.norm.weight") is False
        assert should_quantize_key("lm_head.weight") is False
        assert should_quantize_key("model.embed_tokens.weight") is False
        assert should_quantize_key("model.layers.0.mlp.gate.weight") is False

    def test_non_weight_keys(self):
        """Test that non-weight keys are not quantized."""
        assert should_quantize_key("model.layers.0.self_attn.q_proj.bias") is False
        assert should_quantize_key("model.layers.0.input_layernorm.bias") is False
        assert should_quantize_key("some_random_key") is False

    def test_non_quantized_patterns_constant(self):
        """Test that NON_QUANTIZED_KEY_PATTERNS contains expected patterns."""
        assert "input_layernorm.weight" in NON_QUANTIZED_KEY_PATTERNS
        assert "post_attention_layernorm.weight" in NON_QUANTIZED_KEY_PATTERNS
        assert "norm.weight" in NON_QUANTIZED_KEY_PATTERNS
        assert "lm_head.weight" in NON_QUANTIZED_KEY_PATTERNS
        assert "embed_tokens.weight" in NON_QUANTIZED_KEY_PATTERNS
        assert "mlp.gate.weight" in NON_QUANTIZED_KEY_PATTERNS


class TestCreateScaleInvForWeight:
    """Tests for create_scale_inv_for_weight function."""

    def test_regular_tensor(self):
        """Test scale_inv creation for regular tensors."""
        weight = torch.randn(256, 128, dtype=torch.float32)
        scale_inv = create_scale_inv_for_weight(weight)

        expected_shape = calculate_scale_shape(weight)
        assert scale_inv.shape == expected_shape
        assert scale_inv.dtype == torch.float32
        assert torch.all(scale_inv == 1.0)

    def test_regular_tensor_on_gpu(self):
        """Test scale_inv creation for GPU tensors."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        weight = torch.randn(256, 128, dtype=torch.float32, device="cuda")
        scale_inv = create_scale_inv_for_weight(weight)

        assert scale_inv.device.type == "cuda"
        assert scale_inv.dtype == torch.float32

    def test_dtensor_uses_global_shape(self):
        """Test that DTensor weights use global shape for scale_inv."""
        # Create mock DTensor with global shape different from local shape
        local_weight = torch.randn(128, 128, dtype=torch.float32)  # Local portion
        mock_weight = Mock()
        mock_weight.to_local.return_value = local_weight
        mock_weight.shape = torch.Size([256, 128])  # Global shape (2x local in rows)

        with patch('nemo_automodel.components.models.deepseek_v3.state_dict_adapter.is_dtensor') as mock_is_dtensor:
            mock_is_dtensor.return_value = True

            scale_inv = create_scale_inv_for_weight(mock_weight)

            # Should use global shape (256, 128) -> (2, 1) blocks
            assert scale_inv.shape == torch.Size((2, 1))
            assert scale_inv.dtype == torch.float32

    def test_custom_block_size(self):
        """Test scale_inv creation with custom block size."""
        weight = torch.randn(200, 100, dtype=torch.float32)
        scale_inv = create_scale_inv_for_weight(weight, block_size=50)

        # 200/50=4, 100/50=2
        assert scale_inv.shape == torch.Size((4, 2))


class TestSliceScaleForDtensor:
    """Tests for _slice_scale_for_dtensor function."""

    def test_shard_on_dim0(self):
        """Test slicing scale_inv when weight is sharded on dimension 0."""
        from torch.distributed._tensor import Shard

        # Full scale_inv for a 512x256 weight (4x2 blocks with BLOCK_SIZE=128)
        scale_inv = torch.arange(1, 9, dtype=torch.float32).reshape(4, 2)

        # Mock DTensor setup: weight sharded on dim 0 across 2 ranks
        mock_device_mesh = Mock()
        mock_device_mesh.size.return_value = 2  # 2 ranks
        mock_device_mesh.get_local_rank.return_value = 0  # First rank

        mock_weight = Mock()
        mock_weight.device_mesh = mock_device_mesh
        mock_weight.placements = [Shard(0)]  # Sharded on dim 0

        # Local weight is half the rows: 256x256
        weight_local = torch.randn(256, 256, dtype=torch.float32)

        result = _slice_scale_for_dtensor(scale_inv, mock_weight, weight_local)

        # First rank should get first 2 row blocks (rows 0-255 -> blocks 0-1)
        assert result.shape == torch.Size((2, 2))

    def test_shard_on_dim1(self):
        """Test slicing scale_inv when weight is sharded on dimension 1."""
        from torch.distributed._tensor import Shard

        # Full scale_inv for a 256x512 weight (2x4 blocks with BLOCK_SIZE=128)
        scale_inv = torch.arange(1, 9, dtype=torch.float32).reshape(2, 4)

        # Mock DTensor setup: weight sharded on dim 1 across 2 ranks
        mock_device_mesh = Mock()
        mock_device_mesh.size.return_value = 2  # 2 ranks
        mock_device_mesh.get_local_rank.return_value = 1  # Second rank

        mock_weight = Mock()
        mock_weight.device_mesh = mock_device_mesh
        mock_weight.placements = [Shard(1)]  # Sharded on dim 1

        # Local weight is half the cols: 256x256
        weight_local = torch.randn(256, 256, dtype=torch.float32)

        result = _slice_scale_for_dtensor(scale_inv, mock_weight, weight_local)

        # Second rank should get last 2 col blocks (cols 256-511 -> blocks 2-3)
        assert result.shape == torch.Size((2, 2))


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

    def test_dequantize_mismatched_scale_shape_logs_debug(self):
        """Test that mismatched scale shape logs debug message for non-DTensor."""
        weight = torch.randn(128, 64, dtype=torch.float32).to(torch.float8_e4m3fn)  # Should be (1, 1) scale shape
        scale_inv = torch.ones((2, 1), dtype=torch.float32)  # Wrong shape - too many scale values

        with patch('nemo_automodel.components.models.deepseek_v3.state_dict_adapter.logger') as mock_logger:
            # For non-DTensor, mismatched scale shape logs debug (not warning)
            result = dequantize_from_fp8(weight, scale_inv, dtype=torch.float32)

            # Debug should be called for the shape mismatch
            mock_logger.debug.assert_called()
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

    def test_dequantize_non_contiguous_weight(self):
        """Test dequantization with non-contiguous weight tensor."""
        # Create a non-contiguous tensor by transposing
        weight_base = torch.randn(128, 256, dtype=torch.float32).to(torch.float8_e4m3fn)
        weight = weight_base.t()  # Transpose makes it non-contiguous
        assert not weight.is_contiguous()

        scale_inv = torch.ones((2, 1), dtype=torch.float32)

        result = dequantize_from_fp8(weight, scale_inv, dtype=torch.float32)

        assert result.dtype == torch.float32
        assert result.shape == weight.shape

    def test_dequantize_non_contiguous_scale(self):
        """Test dequantization with non-contiguous scale tensor."""
        weight = torch.randn(256, 128, dtype=torch.float32).to(torch.float8_e4m3fn)

        # Create a non-contiguous scale tensor
        scale_base = torch.ones((2, 2), dtype=torch.float32)
        scale_inv = scale_base[:, :1]  # Slice makes it non-contiguous
        assert not scale_inv.is_contiguous()

        result = dequantize_from_fp8(weight, scale_inv, dtype=torch.float32)

        assert result.dtype == torch.float32
        assert result.shape == weight.shape

    @skip_if_no_gpu
    def test_dequantize_triton_fallback_on_exception(self):
        """Test that dequantize_from_fp8 falls back to torch when triton fails."""
        weight = torch.randn(256, 128, dtype=torch.float32, device="cuda").to(torch.float8_e4m3fn)
        scale_inv = torch.ones((2, 1), dtype=torch.float32, device="cuda")

        with patch('nemo_automodel.components.models.deepseek_v3.state_dict_adapter._dequantize_with_triton') as mock_triton, \
             patch('nemo_automodel.components.models.deepseek_v3.state_dict_adapter._dequantize_with_torch') as mock_torch, \
             patch('nemo_automodel.components.models.deepseek_v3.state_dict_adapter.logger') as mock_logger:

            mock_triton.side_effect = RuntimeError("Triton kernel failed")
            mock_torch.return_value = torch.randn(256, 128, dtype=torch.float32, device="cuda")

            result = dequantize_from_fp8(weight, scale_inv, dtype=torch.float32)

            mock_triton.assert_called_once()
            mock_torch.assert_called_once()
            mock_logger.warning.assert_called_once()
            assert "Triton dequant failed" in mock_logger.warning.call_args[0][0]

    def test_dequantize_cpu_uses_torch_implementation(self):
        """Test that CPU tensors use torch implementation (not triton)."""
        weight = torch.randn(256, 128, dtype=torch.float32).to(torch.float8_e4m3fn)
        scale_inv = torch.ones((2, 1), dtype=torch.float32)

        with patch('nemo_automodel.components.models.deepseek_v3.state_dict_adapter._dequantize_with_torch') as mock_torch, \
             patch('nemo_automodel.components.models.deepseek_v3.state_dict_adapter._dequantize_with_triton') as mock_triton:

            mock_torch.return_value = torch.randn(256, 128, dtype=torch.float32)

            result = dequantize_from_fp8(weight, scale_inv, dtype=torch.float32)

            mock_torch.assert_called_once()
            mock_triton.assert_not_called()

    def test_dequantize_dtensor_input(self):
        """Test dequantization with DTensor input returns DTensor output."""
        # Create mock local tensors
        local_weight = torch.randn(256, 128, dtype=torch.float32).to(torch.float8_e4m3fn)
        local_scale = torch.ones((2, 1), dtype=torch.float32)
        dequantized_local = torch.randn(256, 128, dtype=torch.bfloat16)

        # Create mock DTensor weight
        mock_device_mesh = Mock()
        mock_placements = Mock()
        mock_weight = Mock()
        mock_weight.to_local.return_value = local_weight
        mock_weight.device_mesh = mock_device_mesh
        mock_weight.placements = mock_placements

        # Mock DTensor.from_local
        mock_dtensor_result = Mock()

        with patch('nemo_automodel.components.models.deepseek_v3.state_dict_adapter.is_dtensor') as mock_is_dtensor, \
             patch('nemo_automodel.components.models.deepseek_v3.state_dict_adapter._dequantize_with_torch') as mock_dequant, \
             patch('torch.distributed._tensor.DTensor.from_local') as mock_from_local:

            # Configure mocks
            mock_is_dtensor.side_effect = lambda x: x is mock_weight  # weight is DTensor, scale is not
            mock_dequant.return_value = dequantized_local
            mock_from_local.return_value = mock_dtensor_result

            result = dequantize_from_fp8(mock_weight, local_scale, dtype=torch.bfloat16)

            # Verify to_local was called on weight
            mock_weight.to_local.assert_called_once()

            # Verify DTensor.from_local was called with correct args
            mock_from_local.assert_called_once_with(dequantized_local, mock_device_mesh, mock_placements)

            # Verify result is the DTensor
            assert result is mock_dtensor_result

    def test_dequantize_dtensor_with_scale_slicing(self):
        """Test dequantization with DTensor weight and non-DTensor scale triggers slicing."""
        from torch.distributed._tensor import Shard

        # Create local tensors - local weight is 256x128 (2x1 blocks)
        local_weight = torch.randn(256, 128, dtype=torch.float32).to(torch.float8_e4m3fn)
        # Global scale_inv for 512x128 weight (4x1 blocks) - larger than local
        global_scale = torch.arange(1, 5, dtype=torch.float32).reshape(4, 1)
        dequantized_local = torch.randn(256, 128, dtype=torch.bfloat16)

        # Create mock DTensor weight (sharded on dim 0)
        mock_device_mesh = Mock()
        mock_device_mesh.size.return_value = 2  # 2 ranks
        mock_device_mesh.get_local_rank.return_value = 0  # First rank

        mock_placements = [Shard(0)]
        mock_weight = Mock()
        mock_weight.to_local.return_value = local_weight
        mock_weight.device_mesh = mock_device_mesh
        mock_weight.placements = mock_placements

        mock_dtensor_result = Mock()

        with patch('nemo_automodel.components.models.deepseek_v3.state_dict_adapter.is_dtensor') as mock_is_dtensor, \
             patch('nemo_automodel.components.models.deepseek_v3.state_dict_adapter._dequantize_with_torch') as mock_dequant, \
             patch('nemo_automodel.components.models.deepseek_v3.state_dict_adapter._slice_scale_for_dtensor') as mock_slice, \
             patch('torch.distributed._tensor.DTensor.from_local') as mock_from_local:

            # weight is DTensor, scale is not
            mock_is_dtensor.side_effect = lambda x: x is mock_weight
            # Return sliced scale that matches local weight shape
            sliced_scale = torch.ones((2, 1), dtype=torch.float32)
            mock_slice.return_value = sliced_scale
            mock_dequant.return_value = dequantized_local
            mock_from_local.return_value = mock_dtensor_result

            result = dequantize_from_fp8(mock_weight, global_scale, dtype=torch.bfloat16)

            # Verify _slice_scale_for_dtensor was called
            mock_slice.assert_called_once()
            # Verify dequantize was called with sliced scale
            call_args = mock_dequant.call_args
            assert call_args is not None

            # Verify result is DTensor
            assert result is mock_dtensor_result


class TestDequantizeWithTorch:
    """Tests for _dequantize_with_torch function directly."""

    def test_basic_dequantization(self):
        """Test basic dequantization on CPU."""
        weight = torch.randn(128, 128, dtype=torch.float32).to(torch.float8_e4m3fn)
        scale_inv = torch.ones((1, 1), dtype=torch.float32) * 2.0

        result = _dequantize_with_torch(weight, scale_inv, torch.float32, BLOCK_SIZE)

        assert result.dtype == torch.float32
        assert result.shape == weight.shape

    def test_multiple_blocks(self):
        """Test dequantization with multiple blocks."""
        weight = torch.randn(256, 256, dtype=torch.float32).to(torch.float8_e4m3fn)
        scale_inv = torch.ones((2, 2), dtype=torch.float32)

        result = _dequantize_with_torch(weight, scale_inv, torch.bfloat16, BLOCK_SIZE)

        assert result.dtype == torch.bfloat16
        assert result.shape == weight.shape

    def test_partial_blocks(self):
        """Test dequantization with partial blocks."""
        weight = torch.randn(200, 100, dtype=torch.float32).to(torch.float8_e4m3fn)
        scale_inv = torch.ones((2, 1), dtype=torch.float32)

        result = _dequantize_with_torch(weight, scale_inv, torch.float32, BLOCK_SIZE)

        assert result.dtype == torch.float32
        assert result.shape == weight.shape

    def test_varying_scales(self):
        """Test dequantization with different scale values per block."""
        weight = torch.ones(256, 256, dtype=torch.float32).to(torch.float8_e4m3fn)
        scale_inv = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)

        result = _dequantize_with_torch(weight, scale_inv, torch.float32, BLOCK_SIZE)

        assert result.dtype == torch.float32
        assert result.shape == weight.shape


class TestConvertSingleTensorToHf:
    def create_mock_config(self, **overrides):
        config = Mock(spec=DeepseekV3Config)
        config.num_layers = 2
        config.hidden_size = 1024
        for key, value in overrides.items():
            setattr(config, key, value)
        return config

    def create_mock_moe_config(self, **overrides):
        moe_config = Mock(spec=MoEConfig)
        moe_config.n_routed_experts = 2
        moe_config.moe_inter_dim = 512
        for key, value in overrides.items():
            setattr(moe_config, key, value)
        return moe_config

    def create_mock_backend_config(self, **overrides):
        backend = Mock(spec=BackendConfig)
        backend.dispatcher = "torch"
        backend.experts = "torch"
        for key, value in overrides.items():
            setattr(backend, key, value)
        return backend

    def test_expert_tensor_conversion(self):
        config = self.create_mock_config()
        moe_config = self.create_mock_moe_config()
        backend = self.create_mock_backend_config()

        adapter = DeepSeekV3StateDictAdapter(config, moe_config, backend)

        # Create gate_and_up_projs tensor
        tensor = torch.randn(2, 1024, 1024)
        fqn = "model.layers.0.mlp.experts.gate_and_up_projs"

        with patch.object(adapter, '_convert_single_merged_expert_to_hf_split_experts') as mock_convert:
            mock_convert.return_value = [
                ("model.layers.0.mlp.experts.0.gate_proj.weight", torch.randn(512, 1024)),
                ("model.layers.0.mlp.experts.0.up_proj.weight", torch.randn(512, 1024)),
            ]

            result = adapter.convert_single_tensor_to_hf(fqn, tensor)

            mock_convert.assert_called_once_with(fqn, tensor)
            assert len(result) == 2

    def test_non_expert_tensor_conversion(self):
        config = self.create_mock_config()
        moe_config = self.create_mock_moe_config()
        backend = self.create_mock_backend_config()

        adapter = DeepSeekV3StateDictAdapter(config, moe_config, backend)

        tensor = torch.randn(512, 512)
        fqn = "model.layers.0.attention.weight"

        with patch.object(adapter, '_convert_single_merged_expert_to_hf_split_experts') as mock_convert:
            mock_convert.return_value = None

            result = adapter.convert_single_tensor_to_hf(fqn, tensor)

            assert len(result) == 1
            assert result[0][0] == fqn
            assert torch.equal(result[0][1], tensor)

    def test_exclude_key_regex(self):
        config = self.create_mock_config()
        moe_config = self.create_mock_moe_config()
        backend = self.create_mock_backend_config()

        adapter = DeepSeekV3StateDictAdapter(config, moe_config, backend)

        tensor = torch.randn(512, 512)
        fqn = "exclude_this.weight"

        with patch.object(adapter, '_convert_single_merged_expert_to_hf_split_experts', return_value=None):
            result = adapter.convert_single_tensor_to_hf(fqn, tensor, exclude_key_regex=r"exclude.*")

            assert len(result) == 0

    def test_quantization_adds_scale_inv(self):
        config = self.create_mock_config()
        moe_config = self.create_mock_moe_config()
        backend = self.create_mock_backend_config()

        adapter = DeepSeekV3StateDictAdapter(config, moe_config, backend)

        tensor = torch.randn(256, 128)
        fqn = "model.layers.0.self_attn.q_proj.weight"

        with patch.object(adapter, '_convert_single_merged_expert_to_hf_split_experts', return_value=None):
            result = adapter.convert_single_tensor_to_hf(fqn, tensor, quantization=True)

            assert len(result) == 2
            assert result[0][0] == fqn
            assert result[0][1].dtype == torch.float8_e4m3fn
            assert result[1][0] == fqn + "_scale_inv"
            assert result[1][1].dtype == torch.float32

    def test_quantization_skips_non_quantized_keys(self):
        config = self.create_mock_config()
        moe_config = self.create_mock_moe_config()
        backend = self.create_mock_backend_config()

        adapter = DeepSeekV3StateDictAdapter(config, moe_config, backend)

        tensor = torch.randn(256)
        fqn = "model.layers.0.input_layernorm.weight"

        with patch.object(adapter, '_convert_single_merged_expert_to_hf_split_experts', return_value=None):
            result = adapter.convert_single_tensor_to_hf(fqn, tensor, quantization=True)

            assert len(result) == 1
            assert result[0][0] == fqn
            assert result[0][1].dtype == tensor.dtype  # Should not be quantized

    def test_quantization_with_expert_tensors(self):
        config = self.create_mock_config()
        moe_config = self.create_mock_moe_config()
        backend = self.create_mock_backend_config()

        adapter = DeepSeekV3StateDictAdapter(config, moe_config, backend)

        tensor = torch.randn(2, 1024, 1024)
        fqn = "model.layers.0.mlp.experts.gate_and_up_projs"

        expert_results = [
            ("model.layers.0.mlp.experts.0.gate_proj.weight", torch.randn(512, 1024)),
            ("model.layers.0.mlp.experts.0.up_proj.weight", torch.randn(512, 1024)),
        ]

        with patch.object(adapter, '_convert_single_merged_expert_to_hf_split_experts', return_value=expert_results):
            result = adapter.convert_single_tensor_to_hf(fqn, tensor, quantization=True)

            # Each expert weight should be quantized
            assert len(result) == 4  # 2 weights * 2 (weight + scale_inv)
            assert all("_scale_inv" in k or k.endswith(".weight") for k, _ in result)


skip_if_no_triton = pytest.mark.skipif(not _TRITON_AVAILABLE, reason="Triton is required")


def _triton_works_on_current_gpu() -> bool:
    """Check if Triton dequantization kernel works on the current GPU."""
    if not _TRITON_AVAILABLE or not torch.cuda.is_available():
        return False
    try:
        # Run a small test to verify Triton works on this GPU
        weight = torch.randn(128, 128, device="cuda", dtype=torch.float32).to(torch.float8_e4m3fn)
        scale_inv = torch.ones((1, 1), device="cuda", dtype=torch.float32)
        _dequantize_with_triton(weight, scale_inv, torch.bfloat16, BLOCK_SIZE)
        return True
    except Exception:
        return False


skip_if_triton_unsupported = pytest.mark.skipif(
    not _triton_works_on_current_gpu(),
    reason="Triton dequantization kernel not supported on this GPU"
)


class TestDequantizeTritonVsTorch:
    """Tests to verify functional equivalence between _dequantize_with_triton and _dequantize_with_torch."""

    def _create_test_tensors(self, m: int, n: int, device: str = "cuda") -> tuple[torch.Tensor, torch.Tensor]:
        """Create FP8 weight tensor and corresponding scale_inv tensor."""
        weight_float = torch.randn(m, n, device=device, dtype=torch.float32)
        weight_fp8 = weight_float.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)

        scale_shape = calculate_scale_shape(weight_fp8, BLOCK_SIZE)
        scale_inv = torch.rand(scale_shape, device=device, dtype=torch.float32) * 2.0 + 0.5

        return weight_fp8, scale_inv

    @skip_if_triton_unsupported
    def test_equivalence_small_matrix(self):
        """Test equivalence for small matrices (256x256)."""
        weight, scale_inv = self._create_test_tensors(256, 256)
        dtype = torch.bfloat16

        torch_result = _dequantize_with_torch(weight, scale_inv, dtype, BLOCK_SIZE)
        triton_result = _dequantize_with_triton(weight, scale_inv, dtype, BLOCK_SIZE)

        assert torch.allclose(torch_result.float(), triton_result.float(), atol=1e-3, rtol=1e-3)

    @skip_if_triton_unsupported
    def test_equivalence_medium_matrix(self):
        """Test equivalence for medium matrices (512x512)."""
        weight, scale_inv = self._create_test_tensors(512, 512)
        dtype = torch.bfloat16

        torch_result = _dequantize_with_torch(weight, scale_inv, dtype, BLOCK_SIZE)
        triton_result = _dequantize_with_triton(weight, scale_inv, dtype, BLOCK_SIZE)

        assert torch.allclose(torch_result.float(), triton_result.float(), atol=1e-3, rtol=1e-3)

    @skip_if_triton_unsupported
    def test_equivalence_non_square_tall(self):
        """Test equivalence for non-square tall matrices (768x256)."""
        weight, scale_inv = self._create_test_tensors(768, 256)
        dtype = torch.bfloat16

        torch_result = _dequantize_with_torch(weight, scale_inv, dtype, BLOCK_SIZE)
        triton_result = _dequantize_with_triton(weight, scale_inv, dtype, BLOCK_SIZE)

        assert torch.allclose(torch_result.float(), triton_result.float(), atol=1e-3, rtol=1e-3)

    @skip_if_triton_unsupported
    def test_equivalence_non_square_wide(self):
        """Test equivalence for non-square wide matrices (256x768)."""
        weight, scale_inv = self._create_test_tensors(256, 768)
        dtype = torch.bfloat16

        torch_result = _dequantize_with_torch(weight, scale_inv, dtype, BLOCK_SIZE)
        triton_result = _dequantize_with_triton(weight, scale_inv, dtype, BLOCK_SIZE)

        assert torch.allclose(torch_result.float(), triton_result.float(), atol=1e-3, rtol=1e-3)

    @skip_if_triton_unsupported
    def test_equivalence_partial_blocks(self):
        """Test equivalence for matrices with partial blocks (non-divisible by BLOCK_SIZE)."""
        weight, scale_inv = self._create_test_tensors(200, 100)
        dtype = torch.bfloat16

        torch_result = _dequantize_with_torch(weight, scale_inv, dtype, BLOCK_SIZE)
        triton_result = _dequantize_with_triton(weight, scale_inv, dtype, BLOCK_SIZE)

        assert torch.allclose(torch_result.float(), triton_result.float(), atol=1e-3, rtol=1e-3)

    @skip_if_triton_unsupported
    def test_equivalence_float32_output(self):
        """Test equivalence with float32 output dtype."""
        weight, scale_inv = self._create_test_tensors(512, 512)
        dtype = torch.float32

        torch_result = _dequantize_with_torch(weight, scale_inv, dtype, BLOCK_SIZE)
        triton_result = _dequantize_with_triton(weight, scale_inv, dtype, BLOCK_SIZE)

        assert torch_result.dtype == torch.float32
        assert triton_result.dtype == torch.float32
        assert torch.allclose(torch_result, triton_result, atol=1e-5, rtol=1e-5)

    @skip_if_triton_unsupported
    def test_equivalence_float16_output(self):
        """Test equivalence with float16 output dtype."""
        weight, scale_inv = self._create_test_tensors(512, 512)
        dtype = torch.float16

        torch_result = _dequantize_with_torch(weight, scale_inv, dtype, BLOCK_SIZE)
        triton_result = _dequantize_with_triton(weight, scale_inv, dtype, BLOCK_SIZE)

        assert torch_result.dtype == torch.float16
        assert triton_result.dtype == torch.float16
        assert torch.allclose(torch_result.float(), triton_result.float(), atol=1e-3, rtol=1e-3)

    @skip_if_triton_unsupported
    def test_equivalence_uniform_scale(self):
        """Test equivalence with uniform scale_inv values."""
        weight, _ = self._create_test_tensors(512, 512)
        scale_shape = calculate_scale_shape(weight, BLOCK_SIZE)
        scale_inv = torch.ones(scale_shape, device="cuda", dtype=torch.float32) * 2.5
        dtype = torch.bfloat16

        torch_result = _dequantize_with_torch(weight, scale_inv, dtype, BLOCK_SIZE)
        triton_result = _dequantize_with_triton(weight, scale_inv, dtype, BLOCK_SIZE)

        assert torch.allclose(torch_result.float(), triton_result.float(), atol=1e-3, rtol=1e-3)

    @skip_if_triton_unsupported
    def test_equivalence_varying_scales(self):
        """Test equivalence with varying scale_inv values across blocks."""
        weight, _ = self._create_test_tensors(512, 512)
        scale_shape = calculate_scale_shape(weight, BLOCK_SIZE)
        # Create scales that vary significantly across blocks
        scale_inv = torch.arange(1, scale_shape[0] * scale_shape[1] + 1, device="cuda", dtype=torch.float32)
        scale_inv = scale_inv.reshape(scale_shape) * 0.1
        dtype = torch.bfloat16

        torch_result = _dequantize_with_torch(weight, scale_inv, dtype, BLOCK_SIZE)
        triton_result = _dequantize_with_triton(weight, scale_inv, dtype, BLOCK_SIZE)

        assert torch.allclose(torch_result.float(), triton_result.float(), atol=1e-3, rtol=1e-3)

    @skip_if_triton_unsupported
    def test_output_shapes_match(self):
        """Test that output shapes match for both implementations."""
        test_shapes = [(256, 256), (512, 512), (768, 256), (200, 100)]
        dtype = torch.bfloat16

        for m, n in test_shapes:
            weight, scale_inv = self._create_test_tensors(m, n)

            torch_result = _dequantize_with_torch(weight, scale_inv, dtype, BLOCK_SIZE)
            triton_result = _dequantize_with_triton(weight, scale_inv, dtype, BLOCK_SIZE)

            assert torch_result.shape == triton_result.shape == (m, n), f"Shape mismatch for ({m}, {n})"

    @skip_if_triton_unsupported
    def test_output_devices_match(self):
        """Test that output devices match input device."""
        weight, scale_inv = self._create_test_tensors(512, 512, device="cuda")
        dtype = torch.bfloat16

        torch_result = _dequantize_with_torch(weight, scale_inv, dtype, BLOCK_SIZE)
        triton_result = _dequantize_with_triton(weight, scale_inv, dtype, BLOCK_SIZE)

        assert torch_result.device.type == "cuda"
        assert triton_result.device.type == "cuda"
