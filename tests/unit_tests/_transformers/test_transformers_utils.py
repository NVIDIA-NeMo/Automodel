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

from unittest.mock import Mock, patch

import pytest

from nemo_automodel.components._transformers.utils import sliding_window_overwrite


class TestSlidingWindowOverwrite:
    """Test cases for sliding_window_overwrite function."""

    @patch("nemo_automodel.components._transformers.utils.AutoConfig.from_pretrained")
    def test_sliding_window_overwrite_use_sliding_window_false(self, mock_from_pretrained, capsys):
        """Test sliding_window is set to None when use_sliding_window is False."""
        # Create a mock config with use_sliding_window=False and sliding_window attribute
        mock_config = Mock()
        mock_config.use_sliding_window = False
        mock_config.sliding_window = 4096  # Some default value
        mock_from_pretrained.return_value = mock_config

        model_name = "test-model"
        result = sliding_window_overwrite(model_name)

        # Assert the function was called correctly
        mock_from_pretrained.assert_called_once_with(model_name, trust_remote_code=True)

        # Assert the expected override dictionary is returned
        expected_result = {"sliding_window": None}
        assert result == expected_result

        # Assert the print statement was called
        captured = capsys.readouterr()
        assert "use_sliding_window=False in config - overriding sliding_window parameter to None" in captured.out
        assert str(expected_result) in captured.out

    @patch("nemo_automodel.components._transformers.utils.AutoConfig.from_pretrained")
    def test_sliding_window_overwrite_use_sliding_window_true(self, mock_from_pretrained):
        """Test no override when use_sliding_window is True."""
        # Create a mock config with use_sliding_window=True
        mock_config = Mock()
        mock_config.use_sliding_window = True
        mock_config.sliding_window = 4096
        mock_from_pretrained.return_value = mock_config

        model_name = "test-model"
        result = sliding_window_overwrite(model_name)

        # Assert the function was called correctly
        mock_from_pretrained.assert_called_once_with(model_name, trust_remote_code=True)

        # Assert empty dictionary is returned (no override needed)
        assert result == {}

    @patch("nemo_automodel.components._transformers.utils.AutoConfig.from_pretrained")
    def test_sliding_window_overwrite_no_use_sliding_window_attribute(self, mock_from_pretrained):
        """Test no override when use_sliding_window attribute doesn't exist."""
        # Create a mock config without use_sliding_window attribute
        mock_config = Mock()
        del mock_config.use_sliding_window  # Remove the attribute
        mock_from_pretrained.return_value = mock_config

        model_name = "test-model"
        result = sliding_window_overwrite(model_name)

        # Assert the function was called correctly
        mock_from_pretrained.assert_called_once_with(model_name, trust_remote_code=True)

        # Assert empty dictionary is returned (no override needed)
        assert result == {}

    @patch("nemo_automodel.components._transformers.utils.AutoConfig.from_pretrained")
    def test_sliding_window_overwrite_missing_sliding_window_attribute(self, mock_from_pretrained):
        """Test assertion error when use_sliding_window is False but sliding_window attribute missing."""
        # Create a mock config with use_sliding_window=False but no sliding_window attribute
        mock_config = Mock()
        mock_config.use_sliding_window = False
        del mock_config.sliding_window  # Remove the sliding_window attribute
        mock_from_pretrained.return_value = mock_config

        model_name = "test-model"

        # This should raise an AssertionError due to the assertion in the function
        with pytest.raises(AssertionError):
            sliding_window_overwrite(model_name)

        # Assert the function was called correctly
        mock_from_pretrained.assert_called_once_with(model_name, trust_remote_code=True)

    @patch("nemo_automodel.components._transformers.utils.AutoConfig.from_pretrained")
    def test_sliding_window_overwrite_different_model_names(self, mock_from_pretrained):
        """Test function works with different model names."""
        # Create a mock config with use_sliding_window=False
        mock_config = Mock()
        mock_config.use_sliding_window = False
        mock_config.sliding_window = 2048
        mock_from_pretrained.return_value = mock_config

        # Test with different model names
        model_names = [
            "microsoft/DialoGPT-medium",
            "huggingface/CodeBERTa-small-v1",
            "/path/to/local/model",
            "my-org/custom-model",
        ]

        for model_name in model_names:
            result = sliding_window_overwrite(model_name)

            # Assert the expected override dictionary is returned
            assert result == {"sliding_window": None}

        # Assert the function was called for each model
        assert mock_from_pretrained.call_count == len(model_names)

    @patch("nemo_automodel.components._transformers.utils.AutoConfig.from_pretrained")
    def test_sliding_window_overwrite_trust_remote_code_parameter(self, mock_from_pretrained):
        """Test that trust_remote_code=True is always passed to AutoConfig.from_pretrained."""
        # Create a mock config
        mock_config = Mock()
        mock_config.use_sliding_window = True
        mock_from_pretrained.return_value = mock_config

        model_name = "test-model"
        sliding_window_overwrite(model_name)

        # Verify trust_remote_code=True was passed
        mock_from_pretrained.assert_called_once_with(model_name, trust_remote_code=True)

    @patch("nemo_automodel.components._transformers.utils.AutoConfig.from_pretrained")
    def test_sliding_window_overwrite_hasattr_behavior(self, mock_from_pretrained):
        """Test that hasattr is used correctly to check for attributes."""
        # Create a mock config that behaves correctly with hasattr
        mock_config = Mock()

        # Test case where hasattr returns False for use_sliding_window
        mock_config.configure_mock(**{"use_sliding_window": Mock(side_effect=AttributeError)})
        mock_from_pretrained.return_value = mock_config

        model_name = "test-model"
        result = sliding_window_overwrite(model_name)

        # Should return empty dict when use_sliding_window attribute doesn't exist
        assert result == {}

        # Reset and test case where hasattr returns True but value is not False
        mock_config = Mock()
        mock_config.use_sliding_window = None  # Neither True nor False
        mock_config.sliding_window = 1024
        mock_from_pretrained.return_value = mock_config

        result = sliding_window_overwrite(model_name)

        # Should return empty dict when use_sliding_window is not exactly False
        assert result == {}
