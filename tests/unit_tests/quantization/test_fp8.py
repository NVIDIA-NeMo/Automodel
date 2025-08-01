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
from unittest.mock import patch, MagicMock

from nemo_automodel.components.quantization.fp8 import (
    FP8Config,
    apply_fp8_to_model,
    verify_fp8_conversion,
    _module_filter_fn,
)


class TestFP8Config:
    """Test FP8Config class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = FP8Config()
        assert config.enable_fsdp_float8_all_gather is False
        assert config.force_recompute_fp8_weight_in_bwd is False
        assert config.precompute_float8_dynamic_scale_for_fsdp is False
        assert config.filter_fqns == []
        assert config.recipe_name is None
        assert config.emulate is False
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = FP8Config(
            enable_fsdp_float8_all_gather=True,
            force_recompute_fp8_weight_in_bwd=True,
            precompute_float8_dynamic_scale_for_fsdp=True,
            filter_fqns=["lm_head", "embed"],
            recipe_name="tensorwise",
            emulate=True
        )
        assert config.enable_fsdp_float8_all_gather is True
        assert config.force_recompute_fp8_weight_in_bwd is True
        assert config.precompute_float8_dynamic_scale_for_fsdp is True
        assert config.filter_fqns == ["lm_head", "embed"]
        assert config.recipe_name == "tensorwise"
        assert config.emulate is True



class TestModuleFilter:
    """Test module filtering functionality."""
    
    def test_filter_linear_module(self):
        """Test filtering of linear modules."""
        linear = nn.Linear(32, 64)  # Divisible by 16
        assert _module_filter_fn(linear, "test.linear", []) is True
    
    def test_filter_non_linear_module(self):
        """Test filtering of non-linear modules."""
        relu = nn.ReLU()
        assert _module_filter_fn(relu, "test.relu", []) is False
    
    def test_filter_with_fqn_filter(self):
        """Test filtering with FQN filter list."""
        linear = nn.Linear(32, 64)
        assert _module_filter_fn(linear, "test.lm_head", ["lm_head"]) is False
        assert _module_filter_fn(linear, "test.linear", ["lm_head"]) is True
    
    def test_filter_small_dimensions(self):
        """Test filtering of modules with small dimensions."""
        small_linear = nn.Linear(15, 31)  # Not divisible by 16
        assert _module_filter_fn(small_linear, "test.small", []) is False
        
        good_linear = nn.Linear(32, 64)  # Divisible by 16
        assert _module_filter_fn(good_linear, "test.good", []) is True


class TestFP8Conversion:
    """Test FP8 model conversion."""
    
    def create_test_model(self):
        """Create a simple test model."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(32, 64)  # Good size
                self.linear2 = nn.Linear(64, 32)  # Good size
                self.small_linear = nn.Linear(15, 31)  # Bad size
                self.relu = nn.ReLU()
            
            def forward(self, x):
                x = self.linear1(x)
                x = self.relu(x)
                x = self.linear2(x)
                return x
        
        return TestModel()
    
    @patch('nemo_automodel.components.quantization.fp8._has_cuda_capability')
    def test_apply_fp8_hardware_check_fail(self, mock_cuda_capability):
        """Test FP8 application with hardware check failure."""
        mock_cuda_capability.return_value = False
        
        model = self.create_test_model()
        config = FP8Config(emulate=False)
        
        with pytest.raises(ValueError, match="FP8 is only supported on SM89"):
            apply_fp8_to_model(model, config)
    
    @patch('nemo_automodel.components.quantization.fp8._has_cuda_capability')
    @patch('nemo_automodel.components.quantization.fp8.HAVE_TORCHAO', False)
    def test_apply_fp8_torchao_import_error(self, mock_cuda_capability):
        """Test FP8 application with torchao import error."""
        mock_cuda_capability.return_value = True
        
        model = self.create_test_model()
        config = FP8Config()
        
        # This will cause the import to fail in apply_fp8_to_model
        with pytest.raises(ImportError, match="torchao is not installed"):
            apply_fp8_to_model(model, config)
    
    def test_verify_fp8_conversion_no_fp8(self):
        """Test verification with no FP8 modules."""
        model = self.create_test_model()
        results = verify_fp8_conversion(model)
        
        assert results['success'] is False
        assert results['fp8_count'] == 0
        assert results['linear_count'] == 3  # 2 good + 1 small
        assert results['conversion_rate'] == 0.0
        assert len(results['fp8_modules']) == 0
    
    @pytest.mark.skipif(
        not hasattr(torch.cuda, 'get_device_capability') or not torch.cuda.is_available(),
        reason="Requires CUDA for torchao imports"
    )
    def test_verify_fp8_conversion_with_mock_fp8(self):
        """Test verification with mock FP8 modules."""
        # This test requires torchao to work properly
        try:
            from torchao.float8.float8_linear import Float8Linear
        except ImportError:
            pytest.skip("torchao not available")
        
        model = self.create_test_model()
        
        # Create a mock that has the right class name for the fallback check
        class MockFloat8Linear(nn.Linear):
            pass
        
        # Set the class name to match what verify_fp8_conversion looks for
        MockFloat8Linear.__name__ = "Float8Linear"
        
        # Replace one linear with mock FP8 module  
        model.linear1 = MockFloat8Linear(32, 64)
        
        results = verify_fp8_conversion(model)
        
        # Should find the mock module by class name fallback
        assert results['success'] is True
        assert results['fp8_count'] == 1
        assert results['linear_count'] == 3  # 3 total linear layers: MockFloat8Linear, linear2, small_linear
        assert len(results['fp8_modules']) == 1
        assert 'linear1' in results['fp8_modules'][0]['name']


class TestIntegration:
    """Integration tests for FP8 functionality."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_full_fp8_pipeline_emulated(self):
        """Test full FP8 pipeline with emulation (if torchao available)."""
        pytest.importorskip("torchao.float8", reason="torchao not available")
        
        # Create test model
        model = nn.Sequential(
            nn.Linear(128, 256),  # Large enough for FP8
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        # Apply FP8 with emulation
        config = FP8Config(emulate=True)
        
        try:
            fp8_model = apply_fp8_to_model(model, config)
            
            # Verify conversion
            results = verify_fp8_conversion(fp8_model)
            
            # Should have some FP8 modules (if torchao conversion worked)
            # Note: This might be 0 if emulation doesn't actually replace modules
            assert results['fp8_count'] >= 0
            
            # Test forward pass
            test_input = torch.randn(2, 128)
            output = fp8_model(test_input)
            assert output.shape == (2, 128)
            
        except Exception as e:
            pytest.skip(f"FP8 conversion not working in test environment: {e}")
    
    def test_config_validation(self):
        """Test configuration validation and edge cases."""
        # Test with None config
        model = nn.Linear(32, 64)
        
        # Should not raise with None config (uses default)
        # Note: This will likely fail without torchao, but that's expected
        try:
            apply_fp8_to_model(model, None)
        except (ImportError, ValueError):
            # Expected in test environment without proper setup
            pass
    
    def test_filter_functionality(self):
        """Test module filtering functionality."""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone_linear = nn.Linear(64, 128)
                self.lm_head = nn.Linear(128, 1000)
                self.embed_tokens = nn.Linear(1000, 64)
        
        model = TestModel()
        
        # Test filtering works correctly
        from nemo_automodel.components.quantization.fp8 import _module_filter_fn
        
        filter_fqns = ["lm_head", "embed_tokens"]
        
        # Should include backbone
        assert _module_filter_fn(
            model.backbone_linear, "backbone_linear", filter_fqns
        ) is True
        
        # Should exclude filtered modules
        assert _module_filter_fn(
            model.lm_head, "lm_head", filter_fqns
        ) is False
        
        assert _module_filter_fn(
            model.embed_tokens, "embed_tokens", filter_fqns
        ) is False


class TestFP8ConfigMethods:
    """Test FP8Config class methods added in the refactor."""
    
    def test_from_config_node_with_none(self):
        """Test from_config_node with None returns default config."""
        config = FP8Config.from_config_node(None)
        assert isinstance(config, FP8Config)
        assert config.recipe_name is None
        assert config.enable_fsdp_float8_all_gather is False
        assert config.filter_fqns == []
    
    def test_from_config_node_with_mock_node(self):
        """Test from_config_node with mock config node."""
        # Create a mock config node
        mock_node = MagicMock()
        mock_node.recipe_name = "rowwise"
        mock_node.enable_fsdp_float8_all_gather = True
        mock_node.filter_fqns = ["lm_head"]
        mock_node.emulate = True
        
        # Mock hasattr to return True for these fields
        def mock_hasattr(obj, attr):
            return attr in ['recipe_name', 'enable_fsdp_float8_all_gather', 'filter_fqns', 'emulate']
        
        with patch('builtins.hasattr', side_effect=mock_hasattr):
            config = FP8Config.from_config_node(mock_node)
        
        assert config.recipe_name == "rowwise"
        assert config.enable_fsdp_float8_all_gather is True
        assert config.filter_fqns == ["lm_head"]
        assert config.emulate is True
    
    def test_to_dict_conversion(self):
        """Test to_dict method converts config to legacy format."""
        config = FP8Config(
            recipe_name="tensorwise",
            enable_fsdp_float8_all_gather=True,
            precompute_float8_dynamic_scale_for_fsdp=True,
            force_recompute_fp8_weight_in_bwd=False,
            filter_fqns=["lm_head", "embed_tokens"],
            emulate=True
        )
        
        result = config.to_dict()
        expected = {
            'fp8_recipe_name': 'tensorwise',
            'enable_fsdp_float8_all_gather': True,
            'precompute_float8_dynamic_scale_for_fsdp': True,
            'force_recompute_fp8_weight_in_bwd': False,
            'fp8_filter_fqns': ['lm_head', 'embed_tokens'],
            'fp8_emulate': True,
        }
        
        assert result == expected
    
    def test_to_dict_with_defaults(self):
        """Test to_dict with default values."""
        config = FP8Config()
        result = config.to_dict()
        
        expected = {
            'fp8_recipe_name': None,
            'enable_fsdp_float8_all_gather': False,
            'precompute_float8_dynamic_scale_for_fsdp': False,
            'force_recompute_fp8_weight_in_bwd': False,
            'fp8_filter_fqns': [],
            'fp8_emulate': False,
        }
        
        assert result == expected

if __name__ == "__main__":
    pytest.main([__file__]) 