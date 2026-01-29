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

"""Unit tests for KimiVL model components."""

from unittest.mock import MagicMock, patch

import pytest
from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config

from nemo_automodel.components.models.kimivl.model import (
    KimiVLConfig,
    KimiVLForConditionalGeneration,
    MoonViTConfig,
)


class TestMoonViTConfig:
    """Tests for MoonViTConfig."""

    def test_default_initialization(self):
        """Test MoonViTConfig initializes with correct defaults."""
        config = MoonViTConfig()
        
        assert config.patch_size == 14
        assert config.init_pos_emb_height == 64
        assert config.init_pos_emb_width == 64
        assert config.num_attention_heads == 16
        assert config.num_hidden_layers == 27
        assert config.hidden_size == 1152
        assert config.intermediate_size == 4304
        assert config.merge_kernel_size == [2, 2]
        assert config.model_type == "moonvit"

    def test_custom_initialization(self):
        """Test MoonViTConfig with custom values."""
        config = MoonViTConfig(
            patch_size=16,
            hidden_size=768,
            num_hidden_layers=12,
            merge_kernel_size=(4, 4),
        )
        
        assert config.patch_size == 16
        assert config.hidden_size == 768
        assert config.num_hidden_layers == 12
        assert config.merge_kernel_size == [4, 4]

    def test_merge_kernel_size_tuple_to_list(self):
        """Test that tuple merge_kernel_size is converted to list."""
        config = MoonViTConfig(merge_kernel_size=(3, 3))
        assert config.merge_kernel_size == [3, 3]
        assert isinstance(config.merge_kernel_size, list)


class TestKimiVLConfig:
    """Tests for KimiVLConfig."""

    def test_default_initialization(self):
        """Test KimiVLConfig initializes with defaults."""
        config = KimiVLConfig()
        
        assert isinstance(config.vision_config, MoonViTConfig)
        assert isinstance(config.text_config, DeepseekV3Config)
        assert config.ignore_index == -100
        assert config.media_placeholder_token_id == 163605
        assert config.pad_token_id == 0
        assert config.architectures == ["KimiVLForConditionalGeneration"]
        assert config.model_type == "kimi_vl"

    def test_initialization_with_dict_configs(self):
        """Test KimiVLConfig initializes correctly from dict configs."""
        vision_dict = {"hidden_size": 768, "patch_size": 16}
        text_dict = {"hidden_size": 1024, "vocab_size": 50000}
        
        config = KimiVLConfig(
            vision_config=vision_dict,
            text_config=text_dict,
        )
        
        assert isinstance(config.vision_config, MoonViTConfig)
        assert config.vision_config.hidden_size == 768
        assert config.vision_config.patch_size == 16
        
        assert isinstance(config.text_config, DeepseekV3Config)
        assert config.text_config.hidden_size == 1024
        assert config.text_config.vocab_size == 50000

    def test_initialization_with_config_objects(self):
        """Test KimiVLConfig initializes correctly from config objects."""
        vision_config = MoonViTConfig(hidden_size=512)
        text_config = DeepseekV3Config(hidden_size=2048)
        
        config = KimiVLConfig(
            vision_config=vision_config,
            text_config=text_config,
        )
        
        assert config.vision_config is vision_config
        assert config.text_config is text_config

    def test_to_dict(self):
        """Test KimiVLConfig.to_dict() includes nested configs."""
        config = KimiVLConfig()
        config_dict = config.to_dict()
        
        assert "vision_config" in config_dict
        assert "text_config" in config_dict
        assert isinstance(config_dict["vision_config"], dict)
        assert isinstance(config_dict["text_config"], dict)
        assert config_dict["vision_config"]["model_type"] == "moonvit"

    def test_custom_architectures(self):
        """Test KimiVLConfig with custom architectures."""
        config = KimiVLConfig(architectures=["CustomArch"])
        assert config.architectures == ["CustomArch"]


class TestKimiVLForConditionalGeneration:
    """Tests for KimiVLForConditionalGeneration."""

    def test_from_pretrained_delegates_to_from_config(self):
        """Test from_pretrained loads config and delegates to from_config."""
        mock_config = MagicMock(spec=KimiVLConfig)
        mock_config.vision_config = MoonViTConfig()
        mock_config.text_config = DeepseekV3Config(
            vocab_size=100,
            hidden_size=64,
            num_attention_heads=4,
            num_hidden_layers=1,
            intermediate_size=128,
            qk_rope_head_dim=16,
            v_head_dim=16,
            qk_nope_head_dim=16,
        )
        mock_config.media_placeholder_token_id = 163605
        
        with patch.object(KimiVLConfig, "from_pretrained", return_value=mock_config):
            with patch.object(
                KimiVLForConditionalGeneration, "from_config"
            ) as mock_from_config:
                mock_from_config.return_value = MagicMock()
                
                KimiVLForConditionalGeneration.from_pretrained("dummy/path")
                
                KimiVLConfig.from_pretrained.assert_called_once_with("dummy/path")
                mock_from_config.assert_called_once()
                assert mock_from_config.call_args[0][0] is mock_config

    def test_modelclass_export_exists(self):
        """Test ModelClass is exported and points to correct class."""
        from nemo_automodel.components.models.kimivl import model as kimivl_mod
        
        assert hasattr(kimivl_mod, "ModelClass")
        assert kimivl_mod.ModelClass is KimiVLForConditionalGeneration


class TestKimiVLUsesDeepseekV3Config:
    """Tests to verify KimiVL properly uses HuggingFace's DeepseekV3Config."""

    def test_text_config_is_hf_deepseek_v3_config(self):
        """Verify text_config uses HF's DeepseekV3Config, not a custom class."""
        config = KimiVLConfig()
        
        # Should be the actual HuggingFace DeepseekV3Config class
        assert type(config.text_config).__name__ == "DeepseekV3Config"
        assert type(config.text_config).__module__ == "transformers.models.deepseek_v3.configuration_deepseek_v3"

    def test_text_config_from_dict_creates_hf_config(self):
        """Verify creating from dict still uses HF's DeepseekV3Config."""
        config = KimiVLConfig(text_config={"hidden_size": 512})
        
        assert type(config.text_config).__name__ == "DeepseekV3Config"
        assert config.text_config.hidden_size == 512
