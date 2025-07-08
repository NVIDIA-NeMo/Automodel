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

import logging
import pytest
from unittest.mock import Mock, patch
import transformers
from transformers import AutoConfig

from nemo_automodel._transformers.auto_model import (
    NeMoAutoModelForCausalLM,
    NeMoAutoModelForImageTextToText,
    patch_attention,
)


class TestNeMoAutoModelForCausalLM:
    """Test cases for NeMoAutoModelForCausalLM class."""

    def test_from_pretrained_liger_kernel_not_available(self, caplog):
        """Test warning when Liger kernel is not available."""
        with patch('nemo_automodel._transformers.auto_model.HAS_LIGER_KERNEL', False):
            with patch.object(transformers.AutoModelForCausalLM, 'from_pretrained') as mock_from_pretrained:
                mock_model = Mock()
                mock_model.config = Mock()
                mock_from_pretrained.return_value = mock_model
                
                # Test line 208 - warning when HAS_LIGER_KERNEL is False
                with caplog.at_level(logging.WARNING):
                    model = NeMoAutoModelForCausalLM.from_pretrained("dummy_model")
                
                assert "Asked to use Liger Kernel, but could not import" in caplog.text
                assert model is mock_model
                mock_from_pretrained.assert_called_once()


    def test_from_config_liger_kernel_not_available(self, caplog):
        """Test warning when Liger kernel is not available in from_config."""
        with patch('nemo_automodel._transformers.auto_model.HAS_LIGER_KERNEL', False):
            with patch.object(transformers.AutoModelForCausalLM, 'from_config') as mock_from_config:
                mock_model = Mock()
                mock_model.config = Mock()
                mock_from_config.return_value = mock_model
                
                config = AutoConfig.from_pretrained("gpt2")
                
                # Test line 297 - warning when HAS_LIGER_KERNEL is False
                with caplog.at_level(logging.WARNING):
                    model = NeMoAutoModelForCausalLM.from_config(config)
                
                assert "Asked to use Liger Kernel, but could not import" in caplog.text
                assert model is mock_model
                mock_from_config.assert_called_once()

class TestNeMoAutoModelForImageTextToText:
    """Test cases for NeMoAutoModelForImageTextToText class."""

    def test_from_pretrained_liger_kernel_not_available(self, caplog):
        """Test warning when Liger kernel is not available."""
        with patch('nemo_automodel._transformers.auto_model.HAS_LIGER_KERNEL', False):
            with patch.object(transformers.AutoModelForImageTextToText, 'from_pretrained') as mock_from_pretrained:
                mock_model = Mock()
                mock_model.config = Mock()
                mock_from_pretrained.return_value = mock_model
                
                # Test line 356 - warning when HAS_LIGER_KERNEL is False
                with caplog.at_level(logging.WARNING):
                    model = NeMoAutoModelForImageTextToText.from_pretrained("dummy_model")
                
                assert "Asked to use Liger Kernel, but could not import" in caplog.text
                assert model is mock_model
                mock_from_pretrained.assert_called_once()

    def test_from_config_liger_kernel_not_available(self, caplog):
        """Test warning when Liger kernel is not available in from_config."""
        with patch('nemo_automodel._transformers.auto_model.HAS_LIGER_KERNEL', False):
            with patch.object(transformers.AutoModelForImageTextToText, 'from_config') as mock_from_config:
                mock_model = Mock()
                mock_model.config = Mock()
                mock_from_config.return_value = mock_model
                
                config = AutoConfig.from_pretrained("gpt2")
                
                # Test warning when HAS_LIGER_KERNEL is False
                with caplog.at_level(logging.WARNING):
                    model = NeMoAutoModelForImageTextToText.from_config(config)
                
                assert "Asked to use Liger Kernel, but could not import" in caplog.text
                assert model is mock_model
                mock_from_config.assert_called_once()

    
class TestPatchAttention:
    """Test cases for patch_attention function."""

    def test_patch_attention_basic(self):
        """Test basic patch_attention functionality."""
        # Create a mock object with a forward method
        mock_obj = Mock()
        mock_forward = Mock()
        mock_obj.forward = mock_forward
        
        # Mock the forward method to be a bound method
        mock_forward.__func__ = Mock()
        mock_forward.__self__ = mock_obj
        
        with patch('nemo_automodel._transformers.auto_model.sdpa_kernel') as mock_sdpa_kernel:
            with patch('nemo_automodel._transformers.auto_model._assert_same_signature'):
                result = patch_attention(mock_obj)
                
                assert result is mock_obj
                # Verify that the forward method was replaced
                assert mock_obj.forward != mock_forward

    def test_patch_attention_with_custom_sdpa_method(self):
        """Test patch_attention with custom SDPA method."""
        from torch.nn.attention import SDPBackend
        
        mock_obj = Mock()
        mock_forward = Mock()
        mock_obj.forward = mock_forward
        
        # Mock the forward method to be a bound method
        mock_forward.__func__ = Mock()
        mock_forward.__self__ = mock_obj
        
        custom_sdpa_method = [SDPBackend.FLASH_ATTENTION]
        
        with patch('nemo_automodel._transformers.auto_model.sdpa_kernel') as mock_sdpa_kernel:
            with patch('nemo_automodel._transformers.auto_model._assert_same_signature'):
                result = patch_attention(mock_obj, custom_sdpa_method)
                
                assert result is mock_obj
                # Verify that the forward method was replaced
                assert mock_obj.forward != mock_forward


class TestUtilityFunctions:
    """Test cases for utility functions."""

    def test_assert_same_signature_matching(self):
        """Test _assert_same_signature with matching signatures."""
        from nemo_automodel._transformers.auto_model import _assert_same_signature
        
        def func1(a, b, c=None):
            pass
        
        def func2(a, b, c=None):
            pass
        
        # Should not raise an exception
        _assert_same_signature(func1, func2)

    def test_assert_same_signature_different(self):
        """Test _assert_same_signature with different signatures."""
        from nemo_automodel._transformers.auto_model import _assert_same_signature
        
        def func1(a, b, c=None):
            pass
        
        def func2(a, b, d=None):
            pass
        
        # Should raise an AssertionError
        with pytest.raises(AssertionError):
            _assert_same_signature(func1, func2) 