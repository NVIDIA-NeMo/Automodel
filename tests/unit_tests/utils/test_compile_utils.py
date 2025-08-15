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

import os
import unittest.mock as mock
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

# Disable torch.compile for testing to avoid compilation overhead
os.environ["TORCH_COMPILE_DISABLE"] = "1"

from nemo_automodel.components.utils.compile_utils import (
    CompileConfig,
    build_compile_config,
    compile_model,
    configure_torch_dynamo,
    create_compile_config_from_dict,
    enable_torch_dynamo_scalar_outputs,
    apply_flash_attention_compile_fix,
    patch_prepare_fa2_from_position_ids,
)


class TestCompileConfig:
    """Test CompileConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CompileConfig()
        assert config.enabled is False
        assert config.mode == "default"
        assert config.fullgraph is False
        assert config.dynamic is False
        assert config.backend is None
        assert config.options == {}
        assert config.dynamo_cache_size_limit == 256

    def test_custom_config(self):
        """Test custom configuration values."""
        options = {"some_option": "value"}
        config = CompileConfig(
            enabled=True,
            mode="reduce-overhead",
            fullgraph=True,
            dynamic=True,
            backend="inductor",
            options=options,
            dynamo_cache_size_limit=512,
        )
        assert config.enabled is True
        assert config.mode == "reduce-overhead"
        assert config.fullgraph is True
        assert config.dynamic is True
        assert config.backend == "inductor"
        assert config.options == options
        assert config.dynamo_cache_size_limit == 512

    def test_to_dict(self):
        """Test to_dict method."""
        config = CompileConfig(
            enabled=True,
            mode="max-autotune",
            dynamo_cache_size_limit=128,
        )
        result = config.to_dict()
        expected = {
            "enabled": True,
            "mode": "max-autotune",
            "fullgraph": False,
            "dynamic": False,
            "backend": None,
            "options": {},
            "dynamo_cache_size_limit": 128,
        }
        assert result == expected


class TestBuildCompileConfig:
    """Test build_compile_config function."""

    def test_build_from_dict(self):
        """Test building config from dictionary."""
        cfg_dict = {
            "enabled": True,
            "mode": "reduce-overhead",
            "backend": "inductor",
            "dynamo_cache_size_limit": 512,
        }
        config = build_compile_config(cfg_dict)
        assert isinstance(config, CompileConfig)
        assert config.enabled is True
        assert config.mode == "reduce-overhead"
        assert config.backend == "inductor"
        assert config.dynamo_cache_size_limit == 512

    def test_build_from_config_object(self):
        """Test building config from existing CompileConfig object."""
        original_config = CompileConfig(enabled=True, mode="max-autotune")
        config = build_compile_config(original_config)
        assert config is original_config  # Should return the same object


class TestCreateCompileConfigFromDict:
    """Test create_compile_config_from_dict function."""

    def test_create_with_all_options(self):
        """Test creating config with all options."""
        config_dict = {
            "enabled": True,
            "mode": "max-autotune",
            "fullgraph": True,
            "dynamic": False,
            "backend": "inductor",
            "options": {"test": "value"},
            "dynamo_cache_size_limit": 1024,
        }
        config = create_compile_config_from_dict(config_dict)
        assert config.enabled is True
        assert config.mode == "max-autotune"
        assert config.fullgraph is True
        assert config.dynamic is False
        assert config.backend == "inductor"
        assert config.options == {"test": "value"}
        assert config.dynamo_cache_size_limit == 1024

    def test_create_with_defaults(self):
        """Test creating config with default values."""
        config_dict = {}
        config = create_compile_config_from_dict(config_dict)
        assert config.enabled is False
        assert config.mode == "default"
        assert config.fullgraph is False
        assert config.dynamic is False
        assert config.backend is None
        assert config.options == {}
        assert config.dynamo_cache_size_limit == 256


class TestConfigureTorchDynamo:
    """Test configure_torch_dynamo function."""

    @patch("nemo_automodel.components.utils.compile_utils.torch._dynamo")
    def test_configure_dynamo_success(self, mock_dynamo):
        """Test successful dynamo configuration."""
        mock_config = MagicMock()
        mock_dynamo.config = mock_config

        configure_torch_dynamo(cache_size_limit=512, capture_scalar_outputs=True)

        assert mock_config.cache_size_limit == 512
        assert mock_config.capture_scalar_outputs is True
        assert mock_config.suppress_errors is True
        assert mock_config.assume_static_by_default is True

    @patch("nemo_automodel.components.utils.compile_utils.torch._dynamo")
    def test_configure_dynamo_without_scalar_outputs(self, mock_dynamo):
        """Test dynamo configuration without scalar outputs."""
        mock_config = MagicMock()
        mock_dynamo.config = mock_config

        configure_torch_dynamo(cache_size_limit=256, capture_scalar_outputs=False)

        assert mock_config.cache_size_limit == 256
        assert mock_config.capture_scalar_outputs is False

    def test_configure_dynamo_import_error(self):
        """Test handling of ImportError when torch._dynamo is not available."""
        with patch("nemo_automodel.components.utils.compile_utils.torch._dynamo") as mock_dynamo:
            mock_dynamo.side_effect = ImportError("Module not found")
            # Should not raise an exception
            configure_torch_dynamo()


class TestEnableTorchDynamoScalarOutputs:
    """Test enable_torch_dynamo_scalar_outputs function."""

    @patch("nemo_automodel.components.utils.compile_utils.torch._dynamo")
    def test_enable_scalar_outputs_success(self, mock_dynamo):
        """Test successful enabling of scalar outputs."""
        mock_config = MagicMock()
        mock_dynamo.config = mock_config

        enable_torch_dynamo_scalar_outputs()

        assert mock_config.capture_scalar_outputs is True

    def test_enable_scalar_outputs_import_error(self):
        """Test handling of ImportError."""
        with patch("nemo_automodel.components.utils.compile_utils.torch._dynamo") as mock_dynamo:
            mock_dynamo.side_effect = ImportError("Module not found")
            # Should not raise an exception
            enable_torch_dynamo_scalar_outputs()


class TestPatchPrepareFA2FromPositionIds:
    """Test patch_prepare_fa2_from_position_ids function."""

    @patch("nemo_automodel.components.utils.compile_utils.transformers.modeling_flash_attention_utils")
    def test_patch_success(self, mock_fa_utils):
        """Test successful patching."""
        result = patch_prepare_fa2_from_position_ids()
        assert result is True
        # Verify that the function was patched
        assert hasattr(mock_fa_utils, "prepare_fa2_from_position_ids")

    def test_patch_import_error(self):
        """Test handling of import error."""
        with patch(
            "nemo_automodel.components.utils.compile_utils.transformers.modeling_flash_attention_utils"
        ) as mock_fa_utils:
            mock_fa_utils.side_effect = ImportError("Module not found")
            result = patch_prepare_fa2_from_position_ids()
            assert result is False

    def test_patch_general_exception(self):
        """Test handling of general exception."""
        with patch(
            "nemo_automodel.components.utils.compile_utils.transformers.modeling_flash_attention_utils"
        ) as mock_fa_utils:
            mock_fa_utils.side_effect = Exception("Some error")
            result = patch_prepare_fa2_from_position_ids()
            assert result is False


class TestApplyFlashAttentionCompileFix:
    """Test apply_flash_attention_compile_fix function."""

    @patch("nemo_automodel.components.utils.compile_utils.enable_torch_dynamo_scalar_outputs")
    @patch("nemo_automodel.components.utils.compile_utils.patch_prepare_fa2_from_position_ids")
    def test_apply_fix_success(self, mock_patch, mock_enable):
        """Test successful application of fix."""
        mock_patch.return_value = True

        result = apply_flash_attention_compile_fix()

        mock_enable.assert_called_once()
        mock_patch.assert_called_once()
        assert result is True

    @patch("nemo_automodel.components.utils.compile_utils.enable_torch_dynamo_scalar_outputs")
    @patch("nemo_automodel.components.utils.compile_utils.patch_prepare_fa2_from_position_ids")
    def test_apply_fix_failure(self, mock_patch, mock_enable):
        """Test failure in applying fix."""
        mock_patch.return_value = False

        result = apply_flash_attention_compile_fix()

        mock_enable.assert_called_once()
        mock_patch.assert_called_once()
        assert result is False


class TestCompileModel:
    """Test compile_model function."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))

    def test_compile_disabled(self):
        """Test that model is returned unchanged when compilation is disabled."""
        config = CompileConfig(enabled=False)
        result = compile_model(self.model, config)
        assert result is self.model

    @patch("nemo_automodel.components.utils.compile_utils.configure_torch_dynamo")
    @patch("nemo_automodel.components.utils.compile_utils.apply_flash_attention_compile_fix")
    @patch("nemo_automodel.components.utils.compile_utils.torch.compile")
    def test_compile_enabled_simple_model(self, mock_torch_compile, mock_fa_fix, mock_configure):
        """Test compilation with a simple model (no transformer blocks)."""
        config = CompileConfig(enabled=True, mode="default", dynamo_cache_size_limit=512)
        mock_compiled_model = MagicMock()
        mock_torch_compile.return_value = mock_compiled_model

        result = compile_model(self.model, config)

        mock_configure.assert_called_once_with(cache_size_limit=512)
        mock_fa_fix.assert_called_once()
        mock_torch_compile.assert_called_once()
        assert result is mock_compiled_model

    def test_compile_model_with_layers_attribute(self):
        """Test compilation with a model that has layers attribute."""
        # Create a mock model with layers
        mock_model = MagicMock()
        mock_layers = MagicMock()
        mock_layers.named_children.return_value = [("0", MagicMock()), ("1", MagicMock())]
        mock_model.layers = mock_layers

        config = CompileConfig(enabled=True)

        with (
            patch("nemo_automodel.components.utils.compile_utils.configure_torch_dynamo"),
            patch("nemo_automodel.components.utils.compile_utils.apply_flash_attention_compile_fix"),
            patch("nemo_automodel.components.utils.compile_utils.torch.compile") as mock_compile,
        ):
            result = compile_model(mock_model, config)

            # Should compile individual blocks, not the whole model
            assert mock_compile.call_count == 2  # Two layers
            assert result is mock_model

    @patch("nemo_automodel.components.utils.compile_utils.configure_torch_dynamo")
    @patch("nemo_automodel.components.utils.compile_utils.apply_flash_attention_compile_fix")
    @patch("nemo_automodel.components.utils.compile_utils.torch.compile")
    def test_compile_with_exception_fallback(self, mock_torch_compile, mock_fa_fix, mock_configure):
        """Test compilation fallback when torch.compile raises exception."""
        config = CompileConfig(enabled=True)
        mock_torch_compile.side_effect = [
            Exception("symbolic evaluation error"),  # First attempt fails
            MagicMock(),  # Fallback succeeds
        ]

        result = compile_model(self.model, config)

        # Should attempt compilation twice (original + fallback)
        assert mock_torch_compile.call_count == 2
        # First call with original settings, second with dynamic=False
        calls = mock_torch_compile.call_args_list
        assert calls[0][1]["dynamic"] is False  # Original config
        assert calls[1][1]["dynamic"] is False  # Fallback with dynamic=False

    @patch("nemo_automodel.components.utils.compile_utils.configure_torch_dynamo")
    @patch("nemo_automodel.components.utils.compile_utils.apply_flash_attention_compile_fix")
    @patch("nemo_automodel.components.utils.compile_utils.torch.compile")
    def test_compile_hidden_states_error_fallback(self, mock_torch_compile, mock_fa_fix, mock_configure):
        """Test compilation fallback for hidden_states KeyError."""
        config = CompileConfig(enabled=True)
        mock_torch_compile.side_effect = [
            KeyError("hidden_states"),  # Specific error
            Exception("Still failing"),  # Fallback 1 fails
            Exception("Still failing"),  # Fallback 2 fails
        ]

        result = compile_model(self.model, config)

        # Should try 3 times and then return original model
        assert mock_torch_compile.call_count == 3
        assert result is self.model

    def test_compile_all_fallbacks_fail(self):
        """Test that original model is returned when all compilation attempts fail."""
        config = CompileConfig(enabled=True)

        with (
            patch("nemo_automodel.components.utils.compile_utils.configure_torch_dynamo"),
            patch("nemo_automodel.components.utils.compile_utils.apply_flash_attention_compile_fix"),
            patch("nemo_automodel.components.utils.compile_utils.torch.compile") as mock_compile,
        ):
            mock_compile.side_effect = Exception("Compilation always fails")

            result = compile_model(self.model, config)

            # Should return original model when compilation fails
            assert result is self.model
