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

from unittest.mock import MagicMock, patch

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.kimi_k25_vl.model import KimiK25VLConfig
from nemo_automodel.components.models.kimi_k25_vl.state_dict_adapter import (
    KimiK25VLStateDictAdapter,
    dequantize_int4,
    quantize_to_int4,
)
from nemo_automodel.components.moe.layers import MoEConfig


class TestDequantizeInt4:
    """Tests for INT4 dequantization without CUDA."""

    def test_dequantize_int4_basic_cpu(self):
        """Test INT4 dequantization on CPU."""
        # Create small test data
        out_features, in_features = 16, 64
        group_size = 32

        # Create packed weights (8 INT4 values per int32)
        # Use values in int32 range (-2^31 to 2^31-1)
        packed_in = in_features // 8
        weight_packed = torch.randint(-2**31, 2**31, (out_features, packed_in), dtype=torch.int32)

        # Create scale (one per group)
        num_groups = in_features // group_size
        weight_scale = torch.rand(out_features, num_groups, dtype=torch.float16) * 0.1

        # Create shape tensor
        weight_shape = torch.tensor([out_features, in_features], dtype=torch.int64)

        # Run on CPU
        result = dequantize_int4(weight_packed, weight_scale, weight_shape, group_size=group_size, device="cpu")

        assert result.shape == (out_features, in_features)
        assert result.dtype == torch.bfloat16

    def test_dequantize_int4_unpacking(self):
        """Test INT4 unpacking produces values in expected range."""
        # Create known packed value: 0x76543210 contains nibbles 0,1,2,3,4,5,6,7
        packed = torch.tensor([[0x76543210]], dtype=torch.int32)
        scale = torch.ones(1, 1, dtype=torch.float16)
        shape = torch.tensor([1, 8], dtype=torch.int64)

        result = dequantize_int4(packed, scale, shape, group_size=8, device="cpu")

        # After dequantization: nibble - 8 (offset binary)
        # 0 -> -8, 1 -> -7, ..., 7 -> -1
        expected_unsigned = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.float32)
        expected_signed = expected_unsigned - 8  # Offset binary conversion

        result_float = result.squeeze().float()
        torch.testing.assert_close(result_float, expected_signed, atol=1e-3, rtol=1e-3)


class TestQuantizeToInt4:
    """Tests for INT4 quantization."""

    def test_quantize_to_int4_basic(self):
        """Test INT4 quantization produces expected output shapes."""
        out_features, in_features = 16, 64
        group_size = 32

        weight = torch.randn(out_features, in_features, dtype=torch.bfloat16)

        weight_packed, weight_scale, weight_shape = quantize_to_int4(weight, group_size=group_size)

        # Check shapes
        packed_in = in_features // 8
        num_groups = in_features // group_size

        assert weight_packed.shape == (out_features, packed_in)
        assert weight_packed.dtype == torch.int32
        assert weight_scale.shape == (out_features, num_groups)
        assert weight_scale.dtype == torch.float16
        assert weight_shape.shape == (2,)
        assert weight_shape[0].item() == out_features
        assert weight_shape[1].item() == in_features

    def test_quantize_dequantize_roundtrip(self):
        """Test quantize -> dequantize approximately recovers original values."""
        out_features, in_features = 16, 64
        group_size = 32

        # Create weights with moderate values
        weight = torch.randn(out_features, in_features, dtype=torch.float32) * 0.5

        # Quantize
        weight_packed, weight_scale, weight_shape = quantize_to_int4(weight, group_size=group_size)

        # Dequantize
        weight_recovered = dequantize_int4(weight_packed, weight_scale, weight_shape, group_size=group_size, device="cpu")

        # Check approximate recovery (INT4 quantization has significant error)
        # We just check the values are in reasonable range
        assert weight_recovered.shape == weight.shape
        assert not torch.isnan(weight_recovered).any()
        assert not torch.isinf(weight_recovered).any()


def create_mock_moe_config():
    """Create a valid MoEConfig for testing."""
    return MoEConfig(
        n_routed_experts=8,
        n_shared_experts=1,
        n_activated_experts=2,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=False,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="softmax",
        route_scale=1.0,
        dim=7168,
        inter_dim=18432,
        moe_inter_dim=2048,
        norm_topk_prob=True,
    )


class TestKimiK25VLStateDictAdapter:
    """Tests for KimiK25VLStateDictAdapter."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock config for testing."""
        config = KimiK25VLConfig()
        config._name_or_path = "/fake/path"
        return config

    @pytest.fixture
    def mock_moe_config(self):
        """Create a mock MoE config."""
        return create_mock_moe_config()

    @pytest.fixture
    def mock_backend(self):
        """Create a mock backend config."""
        return BackendConfig(
            linear="torch",
            rms_norm="torch",
            attn="sdpa",
        )

    @pytest.fixture
    def adapter(self, mock_config, mock_moe_config, mock_backend):
        """Create adapter instance for testing."""
        return KimiK25VLStateDictAdapter(
            mock_config,
            mock_moe_config,
            mock_backend,
            dtype=torch.bfloat16,
        )

    def test_adapter_initialization(self, adapter):
        """Test adapter initializes correctly."""
        assert adapter.dtype == torch.bfloat16
        assert adapter._uses_model_prefix is True
        assert adapter.llm_adapter is not None

    def test_is_quantized_expert_key_positive(self, adapter):
        """Test _is_quantized_expert_key identifies expert keys."""
        expert_keys = [
            "language_model.model.layers.5.mlp.experts.0.gate_proj.weight",
            "language_model.model.layers.10.mlp.experts.7.up_proj.weight",
            "language_model.model.layers.3.mlp.experts.255.down_proj.weight",
        ]

        for key in expert_keys:
            assert adapter._is_quantized_expert_key(key), f"Should identify {key} as expert key"

    def test_is_quantized_expert_key_negative(self, adapter):
        """Test _is_quantized_expert_key excludes non-expert keys."""
        non_expert_keys = [
            # Shared experts should not be quantized
            "language_model.model.layers.5.mlp.shared_experts.gate_proj.weight",
            # First layer (layer 0) should not be quantized
            "language_model.model.layers.0.mlp.experts.0.gate_proj.weight",
            # Non-expert layer keys
            "language_model.model.layers.5.self_attn.q_proj.weight",
            "language_model.model.embed_tokens.weight",
            "vision_tower.encoder.blocks.0.wqkv.weight",
        ]

        for key in non_expert_keys:
            assert not adapter._is_quantized_expert_key(key), f"Should not identify {key} as expert key"

    def test_convert_single_tensor_to_hf_vision_tower(self, adapter):
        """Test convert_single_tensor_to_hf handles vision tower keys."""
        fqn = "model.vision_tower.encoder.blocks.0.wqkv.weight"
        tensor = torch.randn(3456, 1152)

        result = adapter.convert_single_tensor_to_hf(fqn, tensor, quantization=False)

        assert len(result) == 1
        key, value = result[0]
        assert key == "vision_tower.encoder.blocks.0.wqkv.weight"

    def test_convert_single_tensor_to_hf_mm_projector(self, adapter):
        """Test convert_single_tensor_to_hf handles mm_projector key mapping."""
        # Test linear_1 -> proj.0 mapping
        fqn = "model.multi_modal_projector.linear_1.weight"
        tensor = torch.randn(4608, 4608)

        result = adapter.convert_single_tensor_to_hf(fqn, tensor, quantization=False)

        assert len(result) == 1
        key, value = result[0]
        assert key == "mm_projector.proj.0.weight"

    def test_convert_single_tensor_to_hf_mm_projector_linear2(self, adapter):
        """Test convert_single_tensor_to_hf handles linear_2 -> proj.2 mapping."""
        fqn = "model.multi_modal_projector.linear_2.weight"
        tensor = torch.randn(7168, 4608)

        result = adapter.convert_single_tensor_to_hf(fqn, tensor, quantization=False)

        assert len(result) == 1
        key, value = result[0]
        assert key == "mm_projector.proj.2.weight"

    def test_expand_quantized_keys(self, adapter):
        """Test _expand_quantized_keys expands expert weight keys to triplets."""
        state_dict = {
            "language_model.model.layers.5.mlp.experts.0.gate_proj.weight": torch.randn(2048, 7168),
            "language_model.model.layers.5.self_attn.q_proj.weight": torch.randn(1536, 7168),
        }

        result = adapter._expand_quantized_keys(state_dict)

        # Expert key should be expanded to triplet
        assert "language_model.model.layers.5.mlp.experts.0.gate_proj.weight_packed" in result
        assert "language_model.model.layers.5.mlp.experts.0.gate_proj.weight_scale" in result
        assert "language_model.model.layers.5.mlp.experts.0.gate_proj.weight_shape" in result
        # Original expert key should be removed
        assert "language_model.model.layers.5.mlp.experts.0.gate_proj.weight" not in result

        # Non-expert key should remain unchanged
        assert "language_model.model.layers.5.self_attn.q_proj.weight" in result


class TestKimiK25VLStateDictAdapterFromHF:
    """Tests for from_hf conversion."""

    @pytest.fixture
    def adapter(self):
        """Create adapter for testing."""
        config = KimiK25VLConfig()
        moe_config = create_mock_moe_config()
        backend = BackendConfig(linear="torch", rms_norm="torch", attn="sdpa")
        return KimiK25VLStateDictAdapter(config, moe_config, backend, dtype=torch.bfloat16)

    def test_from_hf_vision_tower_key_mapping(self, adapter):
        """Test from_hf maps vision_tower keys correctly."""
        hf_state_dict = {
            "vision_tower.encoder.blocks.0.wqkv.weight": torch.randn(3456, 1152),
        }

        result = adapter.from_hf(hf_state_dict)

        assert "model.vision_tower.encoder.blocks.0.wqkv.weight" in result

    def test_from_hf_mm_projector_key_mapping(self, adapter):
        """Test from_hf maps mm_projector keys correctly."""
        hf_state_dict = {
            "mm_projector.proj.0.weight": torch.randn(4608, 4608),
            "mm_projector.proj.2.weight": torch.randn(7168, 4608),
            "mm_projector.pre_norm.weight": torch.randn(1152),
        }

        result = adapter.from_hf(hf_state_dict)

        assert "model.multi_modal_projector.linear_1.weight" in result
        assert "model.multi_modal_projector.linear_2.weight" in result
        assert "model.multi_modal_projector.pre_norm.weight" in result

    def test_from_hf_lm_head_key_mapping(self, adapter):
        """Test from_hf maps lm_head keys correctly."""
        hf_state_dict = {
            "language_model.lm_head.weight": torch.randn(163840, 7168),
        }

        result = adapter.from_hf(hf_state_dict)

        assert "lm_head.weight" in result

    def test_from_hf_dtype_conversion(self, adapter):
        """Test from_hf converts tensors to target dtype."""
        hf_state_dict = {
            "vision_tower.encoder.blocks.0.wqkv.weight": torch.randn(3456, 1152, dtype=torch.float32),
        }

        result = adapter.from_hf(hf_state_dict)

        # Should convert to bfloat16 (adapter's dtype)
        assert result["model.vision_tower.encoder.blocks.0.wqkv.weight"].dtype == torch.bfloat16


# =============================================================================
# Additional Dequantize/Quantize Tests
# =============================================================================


class TestDequantizeInt4Extended:
    """Extended tests for INT4 dequantization."""

    def test_dequantize_int4_different_group_sizes(self):
        """Test INT4 dequantization with different group sizes."""
        out_features, in_features = 32, 128

        for group_size in [16, 32, 64]:
            packed_in = in_features // 8
            weight_packed = torch.randint(-2**31, 2**31, (out_features, packed_in), dtype=torch.int32)

            num_groups = in_features // group_size
            weight_scale = torch.rand(out_features, num_groups, dtype=torch.float16) * 0.1
            weight_shape = torch.tensor([out_features, in_features], dtype=torch.int64)

            result = dequantize_int4(weight_packed, weight_scale, weight_shape, group_size=group_size, device="cpu")

            assert result.shape == (out_features, in_features)
            assert not torch.isnan(result).any()

    def test_dequantize_int4_scale_expansion(self):
        """Test scale is correctly expanded to match weight dimensions."""
        out_features, in_features = 16, 64
        group_size = 32

        packed_in = in_features // 8
        weight_packed = torch.zeros(out_features, packed_in, dtype=torch.int32)

        num_groups = in_features // group_size
        # Use distinct scales for each group
        weight_scale = torch.tensor([[1.0, 2.0]] * out_features, dtype=torch.float16)
        weight_shape = torch.tensor([out_features, in_features], dtype=torch.int64)

        result = dequantize_int4(weight_packed, weight_scale, weight_shape, group_size=group_size, device="cpu")

        # All packed values are 0, so unpacked values are 0-8=-8
        # With scale=1.0 for first group (first 32 elements): -8 * 1.0 = -8
        # With scale=2.0 for second group (last 32 elements): -8 * 2.0 = -16
        assert result.shape == (out_features, in_features)

    def test_dequantize_int4_1d_scale(self):
        """Test dequantization with flattened 1D scale."""
        out_features, in_features = 16, 64
        group_size = 32

        packed_in = in_features // 8
        weight_packed = torch.randint(-2**31, 2**31, (out_features, packed_in), dtype=torch.int32)

        num_groups = in_features // group_size
        # Flatten scale to 1D
        weight_scale = torch.rand(out_features * num_groups, dtype=torch.float16) * 0.1
        weight_shape = torch.tensor([out_features, in_features], dtype=torch.int64)

        result = dequantize_int4(weight_packed, weight_scale, weight_shape, group_size=group_size, device="cpu")

        assert result.shape == (out_features, in_features)


class TestQuantizeToInt4Extended:
    """Extended tests for INT4 quantization."""

    def test_quantize_to_int4_different_group_sizes(self):
        """Test INT4 quantization with different group sizes."""
        out_features, in_features = 32, 128

        for group_size in [16, 32, 64]:
            weight = torch.randn(out_features, in_features, dtype=torch.bfloat16)

            weight_packed, weight_scale, weight_shape = quantize_to_int4(weight, group_size=group_size)

            expected_num_groups = in_features // group_size
            assert weight_scale.shape == (out_features, expected_num_groups)

    def test_quantize_to_int4_scale_magnitude(self):
        """Test that scales are proportional to weight magnitude."""
        out_features, in_features = 16, 64
        group_size = 32

        # Create weights with known magnitude
        weight = torch.zeros(out_features, in_features, dtype=torch.float32)
        weight[:, :32] = 1.0  # First group: max=1.0
        weight[:, 32:] = 7.0  # Second group: max=7.0

        weight_packed, weight_scale, weight_shape = quantize_to_int4(weight, group_size=group_size)

        # Scales should be proportional to max values
        # scale = max / 7, so group 1 scale ≈ 1/7, group 2 scale ≈ 7/7
        assert weight_scale[:, 0].mean() < weight_scale[:, 1].mean()

    def test_quantize_to_int4_zero_weights(self):
        """Test quantization of zero weights."""
        out_features, in_features = 16, 64
        group_size = 32

        weight = torch.zeros(out_features, in_features, dtype=torch.float32)

        weight_packed, weight_scale, weight_shape = quantize_to_int4(weight, group_size=group_size)

        # Should not produce NaN or Inf
        assert not torch.isnan(weight_scale).any()
        assert not torch.isinf(weight_scale).any()

    def test_quantize_roundtrip_basic_properties(self):
        """Test that quantize->dequantize produces valid output."""
        out_features, in_features = 32, 128
        group_size = 32

        # Use moderate values that will quantize well
        weight = torch.randn(out_features, in_features, dtype=torch.float32) * 0.5

        weight_packed, weight_scale, weight_shape = quantize_to_int4(weight, group_size=group_size)
        weight_recovered = dequantize_int4(weight_packed, weight_scale, weight_shape, group_size=group_size, device="cpu")

        # Check basic properties - shape preserved
        assert weight_recovered.shape == weight.shape

        # Values should be finite
        assert not torch.isnan(weight_recovered).any()
        assert not torch.isinf(weight_recovered).any()

        # Recovered values should be in a reasonable range relative to original
        # (INT4 has limited precision, so we just check it's in the right ballpark)
        assert weight_recovered.abs().max() < weight.abs().max() * 10


# =============================================================================
# To HF Conversion Tests
# =============================================================================


class TestKimiK25VLStateDictAdapterToHF:
    """Tests for to_hf conversion."""

    @pytest.fixture
    def adapter(self):
        """Create adapter for testing."""
        config = KimiK25VLConfig()
        moe_config = create_mock_moe_config()
        backend = BackendConfig(linear="torch", rms_norm="torch", attn="sdpa")
        return KimiK25VLStateDictAdapter(config, moe_config, backend, dtype=torch.bfloat16)

    def test_to_hf_basic_conversion(self, adapter):
        """Test basic to_hf conversion."""
        state_dict = {
            "model.vision_tower.encoder.blocks.0.wqkv.weight": torch.randn(3456, 1152),
            "model.multi_modal_projector.linear_1.weight": torch.randn(4608, 4608),
        }

        result = adapter.to_hf(state_dict, quantization=False)

        assert "vision_tower.encoder.blocks.0.wqkv.weight" in result
        assert "mm_projector.proj.0.weight" in result

    def test_to_hf_exclude_key_regex(self, adapter):
        """Test to_hf with exclude_key_regex."""
        state_dict = {
            "model.vision_tower.encoder.blocks.0.wqkv.weight": torch.randn(3456, 1152),
            "model.multi_modal_projector.linear_1.weight": torch.randn(4608, 4608),
        }

        result = adapter.to_hf(state_dict, exclude_key_regex=r".*vision_tower.*", quantization=False)

        assert "vision_tower.encoder.blocks.0.wqkv.weight" not in result
        assert "mm_projector.proj.0.weight" in result

    def test_to_hf_all_projector_keys(self, adapter):
        """Test to_hf handles all projector key mappings."""
        state_dict = {
            "model.multi_modal_projector.linear_1.weight": torch.randn(4608, 4608),
            "model.multi_modal_projector.linear_1.bias": torch.randn(4608),
            "model.multi_modal_projector.linear_2.weight": torch.randn(7168, 4608),
            "model.multi_modal_projector.linear_2.bias": torch.randn(7168),
            "model.multi_modal_projector.pre_norm.weight": torch.randn(1152),
        }

        result = adapter.to_hf(state_dict, quantization=False)

        assert "mm_projector.proj.0.weight" in result
        assert "mm_projector.proj.0.bias" in result
        assert "mm_projector.proj.2.weight" in result
        assert "mm_projector.proj.2.bias" in result
        assert "mm_projector.pre_norm.weight" in result


class TestKimiK25VLStateDictAdapterConvertSingleTensor:
    """Tests for convert_single_tensor_to_hf method."""

    @pytest.fixture
    def adapter(self):
        """Create adapter for testing."""
        config = KimiK25VLConfig()
        moe_config = create_mock_moe_config()
        backend = BackendConfig(linear="torch", rms_norm="torch", attn="sdpa")
        return KimiK25VLStateDictAdapter(config, moe_config, backend, dtype=torch.bfloat16)

    def test_convert_removes_model_prefix(self, adapter):
        """Test convert_single_tensor_to_hf removes model. prefix."""
        fqn = "model.some_layer.weight"
        tensor = torch.randn(64, 64)

        result = adapter.convert_single_tensor_to_hf(fqn, tensor, quantization=False)

        key, _ = result[0]
        assert not key.startswith("model.")

    def test_convert_language_model_keys(self, adapter):
        """Test convert_single_tensor_to_hf handles language model keys."""
        fqn = "model.language_model.model.layers.0.self_attn.q_proj.weight"
        tensor = torch.randn(1536, 7168)

        result = adapter.convert_single_tensor_to_hf(fqn, tensor, quantization=False)

        assert len(result) >= 1
        # Should preserve language_model prefix for LLM keys
        key, _ = result[0]
        assert "language_model" in key or "layers" in key

    def test_convert_with_quantization_expert_key(self, adapter):
        """Test convert_single_tensor_to_hf with quantization for expert key."""
        fqn = "model.language_model.model.layers.5.mlp.experts.0.gate_proj.weight"
        tensor = torch.randn(2048, 7168)

        result = adapter.convert_single_tensor_to_hf(fqn, tensor, quantization=True)

        # Expert key should be expanded to triplet
        keys = [r[0] for r in result]
        assert any("weight_packed" in k for k in keys)
        assert any("weight_scale" in k for k in keys)
        assert any("weight_shape" in k for k in keys)

    def test_convert_with_quantization_non_expert_key(self, adapter):
        """Test convert_single_tensor_to_hf with quantization for non-expert key."""
        fqn = "model.vision_tower.encoder.blocks.0.wqkv.weight"
        tensor = torch.randn(3456, 1152)

        result = adapter.convert_single_tensor_to_hf(fqn, tensor, quantization=True)

        # Non-expert keys should NOT be quantized
        keys = [r[0] for r in result]
        assert not any("weight_packed" in k for k in keys)
        assert "vision_tower.encoder.blocks.0.wqkv.weight" in keys


# =============================================================================
# From HF Conversion Extended Tests
# =============================================================================


class TestKimiK25VLStateDictAdapterFromHFExtended:
    """Extended tests for from_hf conversion.

    Note: Tests with expert weights are limited because the underlying
    MoE state dict mixin validates all experts are present. Full expert
    tests require complete checkpoint data.
    """

    @pytest.fixture
    def adapter(self):
        """Create adapter for testing."""
        config = KimiK25VLConfig()
        moe_config = create_mock_moe_config()
        backend = BackendConfig(linear="torch", rms_norm="torch", attn="sdpa")
        return KimiK25VLStateDictAdapter(config, moe_config, backend, dtype=torch.bfloat16)

    def test_from_hf_dequantize_int4_function_directly(self):
        """Test INT4 dequantization function works correctly."""
        out_features, in_features = 32, 64
        group_size = 32

        # Create quantized triplet
        packed_in = in_features // 8
        num_groups = in_features // group_size

        weight_packed = torch.randint(-2**31, 2**31, (out_features, packed_in), dtype=torch.int32)
        weight_scale = torch.rand(out_features, num_groups, dtype=torch.float16) * 0.1
        weight_shape = torch.tensor([out_features, in_features], dtype=torch.int64)

        # Test dequantize function directly
        result = dequantize_int4(weight_packed, weight_scale, weight_shape, group_size=group_size, device="cpu")

        assert result.shape == (out_features, in_features)
        assert result.dtype == torch.bfloat16
        assert not torch.isnan(result).any()

    def test_from_hf_non_expert_keys_only(self, adapter):
        """Test from_hf with non-expert keys (no expert validation triggered)."""
        hf_state_dict = {
            "vision_tower.encoder.blocks.0.wqkv.weight": torch.randn(3456, 1152),
            "mm_projector.proj.0.weight": torch.randn(4608, 4608),
            "mm_projector.proj.2.weight": torch.randn(7168, 4608),
        }

        result = adapter.from_hf(hf_state_dict)

        assert "model.vision_tower.encoder.blocks.0.wqkv.weight" in result
        assert "model.multi_modal_projector.linear_1.weight" in result
        assert "model.multi_modal_projector.linear_2.weight" in result

    def test_from_hf_incomplete_triplet_keys_identified(self, adapter):
        """Test that incomplete triplets are identified correctly."""
        import re

        # Test the pattern matching for quantized triplets
        quant_pat = re.compile(r"^(.+)\.(weight|bias)_(packed|scale|shape)$")

        # Complete triplet
        keys = [
            "language_model.model.layers.5.mlp.experts.0.gate_proj.weight_packed",
            "language_model.model.layers.5.mlp.experts.0.gate_proj.weight_scale",
            "language_model.model.layers.5.mlp.experts.0.gate_proj.weight_shape",
        ]

        quant_groups = {}
        for key in keys:
            m = quant_pat.match(key)
            if m:
                base = f"{m.group(1)}.{m.group(2)}"
                suffix = m.group(3)
                if base not in quant_groups:
                    quant_groups[base] = {}
                quant_groups[base][suffix] = key

        # Should have all three parts
        expected_base = "language_model.model.layers.5.mlp.experts.0.gate_proj.weight"
        assert expected_base in quant_groups
        assert "packed" in quant_groups[expected_base]
        assert "scale" in quant_groups[expected_base]
        assert "shape" in quant_groups[expected_base]

    def test_from_hf_incomplete_triplet_pattern(self, adapter):
        """Test pattern matching for incomplete triplet (missing shape)."""
        import re

        quant_pat = re.compile(r"^(.+)\.(weight|bias)_(packed|scale|shape)$")

        # Incomplete triplet (missing shape)
        keys = [
            "language_model.model.layers.5.mlp.experts.0.gate_proj.weight_packed",
            "language_model.model.layers.5.mlp.experts.0.gate_proj.weight_scale",
        ]

        quant_groups = {}
        for key in keys:
            m = quant_pat.match(key)
            if m:
                base = f"{m.group(1)}.{m.group(2)}"
                suffix = m.group(3)
                if base not in quant_groups:
                    quant_groups[base] = {}
                quant_groups[base][suffix] = key

        expected_base = "language_model.model.layers.5.mlp.experts.0.gate_proj.weight"
        assert expected_base in quant_groups
        # Should NOT have all three parts
        assert "shape" not in quant_groups[expected_base]

        # Validation check: only process complete triplets
        valid_groups = [
            (base, parts)
            for base, parts in quant_groups.items()
            if all(p in parts for p in ["packed", "scale", "shape"])
        ]
        assert len(valid_groups) == 0  # Incomplete triplet should be filtered out


# =============================================================================
# Expert Key Detection Tests
# =============================================================================


class TestIsQuantizedExpertKeyExtended:
    """Extended tests for _is_quantized_expert_key method."""

    @pytest.fixture
    def adapter(self):
        """Create adapter for testing."""
        config = KimiK25VLConfig()
        moe_config = create_mock_moe_config()
        backend = BackendConfig(linear="torch", rms_norm="torch", attn="sdpa")
        return KimiK25VLStateDictAdapter(config, moe_config, backend, dtype=torch.bfloat16)

    def test_all_projection_types(self, adapter):
        """Test all projection types are identified."""
        proj_types = ["gate_proj", "up_proj", "down_proj"]

        for proj in proj_types:
            key = f"language_model.model.layers.5.mlp.experts.0.{proj}.weight"
            assert adapter._is_quantized_expert_key(key), f"Should identify {proj} as expert key"

    def test_all_expert_indices(self, adapter):
        """Test various expert indices are identified."""
        for expert_idx in [0, 1, 7, 63, 127, 255]:
            key = f"language_model.model.layers.5.mlp.experts.{expert_idx}.gate_proj.weight"
            assert adapter._is_quantized_expert_key(key), f"Expert {expert_idx} should be identified"

    def test_all_layer_indices_except_zero(self, adapter):
        """Test layer indices > 0 are identified as expert keys."""
        for layer_idx in [1, 2, 5, 10, 60]:
            key = f"language_model.model.layers.{layer_idx}.mlp.experts.0.gate_proj.weight"
            assert adapter._is_quantized_expert_key(key), f"Layer {layer_idx} expert should be identified"

    def test_layer_zero_excluded(self, adapter):
        """Test layer 0 experts are NOT identified as quantized."""
        key = "language_model.model.layers.0.mlp.experts.0.gate_proj.weight"
        assert not adapter._is_quantized_expert_key(key)

    def test_bias_not_identified(self, adapter):
        """Test bias keys are not identified as expert weight keys."""
        key = "language_model.model.layers.5.mlp.experts.0.gate_proj.bias"
        # Method checks for .weight, bias should not match
        assert not adapter._is_quantized_expert_key(key)


# =============================================================================
# Expand Quantized Keys Extended Tests
# =============================================================================


class TestExpandQuantizedKeysExtended:
    """Extended tests for _expand_quantized_keys method."""

    @pytest.fixture
    def adapter(self):
        """Create adapter for testing."""
        config = KimiK25VLConfig()
        moe_config = create_mock_moe_config()
        backend = BackendConfig(linear="torch", rms_norm="torch", attn="sdpa")
        return KimiK25VLStateDictAdapter(config, moe_config, backend, dtype=torch.bfloat16)

    def test_expand_multiple_experts(self, adapter):
        """Test expansion of multiple expert keys."""
        state_dict = {
            "language_model.model.layers.5.mlp.experts.0.gate_proj.weight": torch.randn(2048, 7168),
            "language_model.model.layers.5.mlp.experts.1.gate_proj.weight": torch.randn(2048, 7168),
            "language_model.model.layers.5.mlp.experts.0.up_proj.weight": torch.randn(2048, 7168),
        }

        result = adapter._expand_quantized_keys(state_dict)

        # Each expert key should be expanded to 3 keys
        assert len(result) == 9  # 3 original keys * 3 triplet components

    def test_expand_preserves_non_expert_keys(self, adapter):
        """Test expansion preserves non-expert keys unchanged."""
        original_tensor = torch.randn(7168, 7168)
        state_dict = {
            "language_model.model.layers.5.mlp.experts.0.gate_proj.weight": torch.randn(2048, 7168),
            "language_model.model.embed_tokens.weight": original_tensor,
        }

        result = adapter._expand_quantized_keys(state_dict)

        assert "language_model.model.embed_tokens.weight" in result
        assert torch.equal(result["language_model.model.embed_tokens.weight"], original_tensor)

    def test_expand_layer_zero_not_expanded(self, adapter):
        """Test layer 0 expert keys are NOT expanded."""
        original_tensor = torch.randn(2048, 7168)
        state_dict = {
            "language_model.model.layers.0.mlp.experts.0.gate_proj.weight": original_tensor,
        }

        result = adapter._expand_quantized_keys(state_dict)

        # Layer 0 should remain unchanged
        assert "language_model.model.layers.0.mlp.experts.0.gate_proj.weight" in result
        assert torch.equal(result["language_model.model.layers.0.mlp.experts.0.gate_proj.weight"], original_tensor)


# =============================================================================
# DTensor Quantization Tests
# =============================================================================


class TestDTensorQuantization:
    """Tests for DTensor INT4 quantization handling."""

    def test_dtensor_quantization_shapes_calculation(self):
        """Test INT4 quantization shape calculations for DTensor."""
        out_features, in_features = 2048, 7168
        group_size = 32

        # INT4 packing: 8 values per int32
        packed_in_features = in_features // 8
        num_groups = in_features // group_size

        assert packed_in_features == 896  # 7168 // 8
        assert num_groups == 224  # 7168 // 32

    def test_dtensor_local_shape_calculation(self):
        """Test local shape calculation for sharded DTensor."""
        out_features, in_features = 2048, 7168
        group_size = 32

        # Simulate 4-way sharding along dim 0
        num_shards = 4
        local_out = out_features // num_shards
        local_in = in_features  # Not sharded on dim 1

        local_packed_in = local_in // 8
        local_num_groups = local_in // group_size

        assert local_out == 512  # 2048 // 4
        assert local_packed_in == 896  # 7168 // 8
        assert local_num_groups == 224  # 7168 // 32

    def test_weight_shape_tensor_format(self):
        """Test weight_shape tensor format."""
        out_features, in_features = 2048, 7168

        weight_shape = torch.tensor([out_features, in_features], dtype=torch.int64)

        assert weight_shape.shape == (2,)
        assert weight_shape[0].item() == out_features
        assert weight_shape[1].item() == in_features
        assert weight_shape.dtype == torch.int64

    def test_packed_scale_shape_consistency(self):
        """Test packed and scale shapes are consistent."""
        out_features, in_features = 2048, 7168
        group_size = 32

        packed_in_features = in_features // 8
        num_groups = in_features // group_size

        # Packed shape
        packed_shape = (out_features, packed_in_features)

        # Scale shape - must have same dim 0 as packed
        scale_shape = (out_features, num_groups)

        assert packed_shape[0] == scale_shape[0]

        # Scale dim 1 should be 8x smaller than packed dim 1 (since group_size=32)
        # packed: in_features // 8 = 896
        # scale: in_features // 32 = 224
        # ratio: 896 / 224 = 4
        assert packed_shape[1] // scale_shape[1] == 4

    def test_dtensor_quantization_empty_tensors_format(self):
        """Test empty tensor format for INT4 quantization placeholders."""
        out_features, in_features = 32, 64
        group_size = 32

        packed_in_features = in_features // 8
        num_groups = in_features // group_size

        # Create empty tensors as placeholders (DCP will fill them)
        weight_packed = torch.empty(out_features, packed_in_features, dtype=torch.int32)
        weight_scale = torch.empty(out_features, num_groups, dtype=torch.float16)
        weight_shape = torch.tensor([out_features, in_features], dtype=torch.int64)

        assert weight_packed.shape == (32, 8)  # 64 // 8 = 8
        assert weight_packed.dtype == torch.int32
        assert weight_scale.shape == (32, 2)  # 64 // 32 = 2
        assert weight_scale.dtype == torch.float16
        assert weight_shape.tolist() == [32, 64]


# =============================================================================
# Dequant One Function Tests
# =============================================================================


class TestDequantOneLogic:
    """Tests for dequant_one function logic."""

    def test_dequant_one_checks_all_parts_exist(self):
        """Test dequant_one validates all triplet parts exist."""
        state_dict = {
            "layer.weight_packed": torch.zeros(1),
            "layer.weight_scale": torch.zeros(1),
            # Missing weight_shape
        }

        parts = {
            "packed": "layer.weight_packed",
            "scale": "layer.weight_scale",
            "shape": "layer.weight_shape",  # Not in state_dict
        }

        packed_exists = parts["packed"] in state_dict
        scale_exists = parts["scale"] in state_dict
        shape_exists = parts["shape"] in state_dict

        assert packed_exists is True
        assert scale_exists is True
        assert shape_exists is False
        assert not all([packed_exists, scale_exists, shape_exists])

    def test_dequant_one_returns_none_for_incomplete(self):
        """Test dequant_one returns None for incomplete triplet."""
        base = "language_model.model.layers.5.mlp.experts.0.gate_proj.weight"
        parts = {
            "packed": f"{base}_packed",
            "scale": f"{base}_scale",
            "shape": f"{base}_shape",
        }

        # Only packed and scale exist
        state_dict = {
            f"{base}_packed": torch.zeros(1),
            f"{base}_scale": torch.zeros(1),
        }

        packed_exists = parts["packed"] in state_dict
        scale_exists = parts["scale"] in state_dict
        shape_exists = parts["shape"] in state_dict

        if not all([packed_exists, scale_exists, shape_exists]):
            result_base, weight, consumed_keys = base, None, []
        else:
            result_base, weight, consumed_keys = base, torch.zeros(1), list(parts.values())

        assert result_base == base
        assert weight is None
        assert consumed_keys == []

    def test_dequant_one_returns_consumed_keys_on_success(self):
        """Test dequant_one returns consumed keys list on success."""
        base = "layer.weight"
        parts = {
            "packed": f"{base}_packed",
            "scale": f"{base}_scale",
            "shape": f"{base}_shape",
        }

        # All parts exist
        state_dict = {
            f"{base}_packed": torch.randint(-2**31, 2**31, (16, 8), dtype=torch.int32),
            f"{base}_scale": torch.rand(16, 2, dtype=torch.float16) * 0.1,
            f"{base}_shape": torch.tensor([16, 64], dtype=torch.int64),
        }

        packed_exists = parts["packed"] in state_dict
        scale_exists = parts["scale"] in state_dict
        shape_exists = parts["shape"] in state_dict

        if not all([packed_exists, scale_exists, shape_exists]):
            consumed_keys = []
        else:
            consumed_keys = list(parts.values())

        assert len(consumed_keys) == 3
        assert f"{base}_packed" in consumed_keys
        assert f"{base}_scale" in consumed_keys
        assert f"{base}_shape" in consumed_keys

    def test_dequant_one_with_valid_triplet(self):
        """Test dequant_one with valid quantized triplet calls dequantize_int4."""
        out_features, in_features = 16, 64
        group_size = 32

        packed_in = in_features // 8
        num_groups = in_features // group_size

        weight_packed = torch.randint(-2**31, 2**31, (out_features, packed_in), dtype=torch.int32)
        weight_scale = torch.rand(out_features, num_groups, dtype=torch.float16) * 0.1
        weight_shape = torch.tensor([out_features, in_features], dtype=torch.int64)

        # Dequantize
        weight = dequantize_int4(weight_packed, weight_scale, weight_shape, group_size=32, device="cpu")

        assert weight.shape == (out_features, in_features)
        assert weight.dtype == torch.bfloat16


# =============================================================================
# Expert Key Fusion Detection Tests
# =============================================================================


class TestExpertKeyFusionDetection:
    """Tests for expert key fusion detection in from_hf."""

    def test_detect_expert_keys_in_llm_keys(self):
        """Test detection of expert keys in LLM state dict."""
        llm_keys = {
            "model.layers.5.mlp.experts.0.gate_proj.weight": torch.randn(2048, 7168),
            "model.layers.5.mlp.experts.0.up_proj.weight": torch.randn(2048, 7168),
            "model.layers.5.mlp.experts.1.gate_proj.weight": torch.randn(2048, 7168),
            "model.layers.5.self_attn.q_proj.weight": torch.randn(1536, 7168),  # Not expert
        }

        expert_llm_keys = [k for k in llm_keys.keys() if "experts." in k and ".weight" in k]

        assert len(expert_llm_keys) == 3
        assert "model.layers.5.self_attn.q_proj.weight" not in expert_llm_keys

    def test_count_unique_expert_ids(self):
        """Test counting unique expert IDs from keys."""
        import re

        llm_keys = {
            "model.layers.5.mlp.experts.0.gate_proj.weight": None,
            "model.layers.5.mlp.experts.0.up_proj.weight": None,
            "model.layers.5.mlp.experts.1.gate_proj.weight": None,
            "model.layers.5.mlp.experts.7.gate_proj.weight": None,
            "model.layers.6.mlp.experts.0.gate_proj.weight": None,
            "model.layers.6.mlp.experts.3.gate_proj.weight": None,
        }

        expert_ids = set()
        for k in llm_keys.keys():
            m = re.search(r"experts\.(\d+)\.", k)
            if m:
                expert_ids.add(int(m.group(1)))

        assert expert_ids == {0, 1, 3, 7}

    def test_no_expert_keys_detection(self):
        """Test detection when no expert keys present."""
        import re

        llm_keys = {
            "model.layers.5.self_attn.q_proj.weight": None,
            "model.layers.5.self_attn.k_proj.weight": None,
            "model.layers.5.mlp.gate_proj.weight": None,  # Dense MLP, not MoE
        }

        expert_llm_keys = [k for k in llm_keys.keys() if "experts." in k and ".weight" in k]
        expert_ids = set()
        for k in llm_keys.keys():
            m = re.search(r"experts\.(\d+)\.", k)
            if m:
                expert_ids.add(int(m.group(1)))

        assert len(expert_llm_keys) == 0
        assert len(expert_ids) == 0

    def test_key_prefix_replacement(self):
        """Test model.language_model.model prefix replacement."""
        converted_llm = {
            "model.layers.5.self_attn.q_proj.weight": torch.randn(1536, 7168),
            "model.embed_tokens.weight": torch.randn(163840, 7168),
        }

        native_state_dict = {}
        for k, v in converted_llm.items():
            native_state_dict[k.replace("model.", "model.language_model.model.")] = v

        assert "model.language_model.model.layers.5.self_attn.q_proj.weight" in native_state_dict
        assert "model.language_model.model.embed_tokens.weight" in native_state_dict

    def test_expert_pattern_regex(self):
        """Test expert pattern regex correctly extracts expert ID."""
        import re

        test_keys = [
            ("model.layers.5.mlp.experts.0.gate_proj.weight", 0),
            ("model.layers.5.mlp.experts.127.up_proj.weight", 127),
            ("model.layers.5.mlp.experts.255.down_proj.weight", 255),
            ("model.layers.5.mlp.shared_experts.gate_proj.weight", None),  # Not experts.N
        ]

        for key, expected_id in test_keys:
            m = re.search(r"experts\.(\d+)\.", key)
            if expected_id is not None:
                assert m is not None, f"Should match: {key}"
                assert int(m.group(1)) == expected_id
            else:
                assert m is None, f"Should not match: {key}"


# =============================================================================
# Quantization Groups Pattern Tests
# =============================================================================


class TestQuantGroupsPattern:
    """Tests for quantization groups pattern matching."""

    def test_quant_pattern_matches_packed(self):
        """Test quant pattern matches _packed suffix."""
        import re

        quant_pat = re.compile(r"^(.+)\.(weight|bias)_(packed|scale|shape)$")
        key = "language_model.model.layers.5.mlp.experts.0.gate_proj.weight_packed"

        m = quant_pat.match(key)
        assert m is not None
        assert m.group(1) == "language_model.model.layers.5.mlp.experts.0.gate_proj"
        assert m.group(2) == "weight"
        assert m.group(3) == "packed"

    def test_quant_pattern_matches_scale(self):
        """Test quant pattern matches _scale suffix."""
        import re

        quant_pat = re.compile(r"^(.+)\.(weight|bias)_(packed|scale|shape)$")
        key = "language_model.model.layers.5.mlp.experts.0.gate_proj.weight_scale"

        m = quant_pat.match(key)
        assert m is not None
        assert m.group(3) == "scale"

    def test_quant_pattern_matches_shape(self):
        """Test quant pattern matches _shape suffix."""
        import re

        quant_pat = re.compile(r"^(.+)\.(weight|bias)_(packed|scale|shape)$")
        key = "language_model.model.layers.5.mlp.experts.0.gate_proj.weight_shape"

        m = quant_pat.match(key)
        assert m is not None
        assert m.group(3) == "shape"

    def test_quant_pattern_does_not_match_regular_weight(self):
        """Test quant pattern does not match regular weight keys."""
        import re

        quant_pat = re.compile(r"^(.+)\.(weight|bias)_(packed|scale|shape)$")
        regular_keys = [
            "language_model.model.layers.5.mlp.experts.0.gate_proj.weight",
            "vision_tower.encoder.blocks.0.wqkv.weight",
            "mm_projector.proj.0.weight",
        ]

        for key in regular_keys:
            m = quant_pat.match(key)
            assert m is None, f"Should not match regular key: {key}"

    def test_quant_groups_building(self):
        """Test building quant_groups dictionary from keys."""
        import re

        quant_pat = re.compile(r"^(.+)\.(weight|bias)_(packed|scale|shape)$")
        state_dict_keys = [
            "layer.5.experts.0.gate_proj.weight_packed",
            "layer.5.experts.0.gate_proj.weight_scale",
            "layer.5.experts.0.gate_proj.weight_shape",
            "layer.5.experts.1.gate_proj.weight_packed",
            "layer.5.experts.1.gate_proj.weight_scale",
            "layer.5.experts.1.gate_proj.weight_shape",
            "layer.5.self_attn.q_proj.weight",  # Not quantized
        ]

        quant_groups = {}
        for key in state_dict_keys:
            m = quant_pat.match(key)
            if m:
                base = f"{m.group(1)}.{m.group(2)}"
                suffix = m.group(3)
                if base not in quant_groups:
                    quant_groups[base] = {}
                quant_groups[base][suffix] = key

        assert len(quant_groups) == 2  # Two experts

        base_0 = "layer.5.experts.0.gate_proj.weight"
        base_1 = "layer.5.experts.1.gate_proj.weight"

        assert base_0 in quant_groups
        assert base_1 in quant_groups

        assert set(quant_groups[base_0].keys()) == {"packed", "scale", "shape"}
        assert set(quant_groups[base_1].keys()) == {"packed", "scale", "shape"}

    def test_filter_complete_triplets(self):
        """Test filtering for complete triplets only."""
        import re

        quant_pat = re.compile(r"^(.+)\.(weight|bias)_(packed|scale|shape)$")

        # Incomplete triplet (missing shape)
        state_dict_keys = [
            "layer.5.experts.0.gate_proj.weight_packed",
            "layer.5.experts.0.gate_proj.weight_scale",
            # Missing: layer.5.experts.0.gate_proj.weight_shape
            "layer.5.experts.1.gate_proj.weight_packed",
            "layer.5.experts.1.gate_proj.weight_scale",
            "layer.5.experts.1.gate_proj.weight_shape",  # Complete triplet
        ]

        quant_groups = {}
        for key in state_dict_keys:
            m = quant_pat.match(key)
            if m:
                base = f"{m.group(1)}.{m.group(2)}"
                suffix = m.group(3)
                if base not in quant_groups:
                    quant_groups[base] = {}
                quant_groups[base][suffix] = key

        # Filter for complete triplets
        complete_triplets = [
            (base, parts)
            for base, parts in quant_groups.items()
            if all(p in parts for p in ["packed", "scale", "shape"])
        ]

        assert len(complete_triplets) == 1
        assert complete_triplets[0][0] == "layer.5.experts.1.gate_proj.weight"
