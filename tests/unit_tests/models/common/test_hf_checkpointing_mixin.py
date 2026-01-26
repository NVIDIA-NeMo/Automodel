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

"""Unit tests for HFCheckpointingMixin."""

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch

from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin


class MockConfig:
    """Mock config for testing."""

    def __init__(self, tie_word_embeddings=True):
        self.tie_word_embeddings = tie_word_embeddings
        self.num_hidden_layers = 2
        self.num_attention_heads = 4
        self.num_key_value_heads = 4
        self.hidden_size = 32
        self.intermediate_size = 64

    def save_pretrained(self, path):
        """Mock save_pretrained."""
        pass


class MockCheckpointerConfig:
    """Mock checkpointer config."""

    def __init__(self, is_peft=False):
        self.is_peft = is_peft


class MockCheckpointer:
    """Mock checkpointer for testing."""

    def __init__(self, is_peft=False):
        self.config = MockCheckpointerConfig(is_peft=is_peft)
        self.moe_mesh = None

    def save_model(self, model, weights_path, peft_config=None, tokenizer=None):
        """Mock save_model."""
        pass


class MockStateDictAdapter:
    """Mock state dict adapter for testing."""

    def __init__(self, config):
        self.config = config

    def to_hf(self, state_dict, **kwargs):
        """Convert NeMo format to HF format (split qkv_proj)."""
        hf_state_dict = {}
        for key, value in state_dict.items():
            if "qkv_proj.weight" in key:
                # Split combined QKV into separate Q, K, V
                q_size = self.config.num_attention_heads * (self.config.hidden_size // self.config.num_attention_heads)
                kv_size = self.config.num_key_value_heads * (self.config.hidden_size // self.config.num_attention_heads)
                q, k, v = value.split([q_size, kv_size, kv_size], dim=0)
                prefix = key.replace("qkv_proj.weight", "")
                hf_state_dict[f"{prefix}q_proj.weight"] = q
                hf_state_dict[f"{prefix}k_proj.weight"] = k
                hf_state_dict[f"{prefix}v_proj.weight"] = v
            else:
                hf_state_dict[key] = value
        return hf_state_dict

    def from_hf(self, state_dict, **kwargs):
        """Convert HF format to NeMo format (combine q/k/v into qkv_proj)."""
        nemo_state_dict = {}
        processed_keys = set()

        for key, value in state_dict.items():
            if key in processed_keys:
                continue
            if "q_proj.weight" in key:
                prefix = key.replace("q_proj.weight", "")
                k_key = f"{prefix}k_proj.weight"
                v_key = f"{prefix}v_proj.weight"
                if k_key in state_dict and v_key in state_dict:
                    qkv = torch.cat([state_dict[key], state_dict[k_key], state_dict[v_key]], dim=0)
                    nemo_state_dict[f"{prefix}qkv_proj.weight"] = qkv
                    processed_keys.update([key, k_key, v_key])
            elif "k_proj.weight" not in key and "v_proj.weight" not in key:
                nemo_state_dict[key] = value

        return nemo_state_dict


class SimpleModelWithMixin(HFCheckpointingMixin, nn.Module):
    """Simple model using the mixin for testing."""

    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        self.linear = nn.Linear(config.hidden_size, config.hidden_size)
        # No adapter - should pass through unchanged

    def forward(self, x):
        return self.linear(x)


class ModelWithAdapter(HFCheckpointingMixin, nn.Module):
    """Model with state_dict_adapter for testing conversion."""

    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        # Combined QKV projection (NeMo internal format)
        qkv_size = config.hidden_size + 2 * config.hidden_size  # Q + K + V
        self.qkv_proj = nn.Linear(config.hidden_size, qkv_size, bias=False)
        self.state_dict_adapter = MockStateDictAdapter(config)

    def forward(self, x):
        return self.qkv_proj(x)


class TestHFCheckpointingMixinStateDict:
    """Tests for state_dict() method."""

    @patch("nemo_automodel.components.models.common.hf_checkpointing_mixin.ModelState")
    @patch("nemo_automodel.components.models.common.hf_checkpointing_mixin._maybe_adapt_state_dict_to_hf")
    def test_state_dict_uses_model_state(self, mock_adapt, mock_model_state_cls):
        """Test that state_dict() uses ModelState with checkpointer config."""
        config = MockConfig()
        model = SimpleModelWithMixin(config)
        model._checkpointer = MockCheckpointer(is_peft=False)

        # Setup mocks
        mock_model_state = MagicMock()
        mock_model_state.model = [model]
        mock_model_state.state_dict.return_value = {"linear.weight": torch.randn(32, 32)}
        mock_model_state_cls.return_value = mock_model_state
        mock_adapt.return_value = {"linear.weight": torch.randn(32, 32)}

        # Call state_dict
        model.state_dict()

        # Verify ModelState was created with is_peft from checkpointer config
        mock_model_state_cls.assert_called_once_with(model, is_peft=False)
        mock_model_state.state_dict.assert_called_once()

    @patch("nemo_automodel.components.models.common.hf_checkpointing_mixin.ModelState")
    @patch("nemo_automodel.components.models.common.hf_checkpointing_mixin._maybe_adapt_state_dict_to_hf")
    def test_state_dict_calls_adapter_with_moe_mesh(self, mock_adapt, mock_model_state_cls):
        """Test that state_dict() calls _maybe_adapt_state_dict_to_hf with quantization=False and device_mesh."""
        config = MockConfig()
        model = ModelWithAdapter(config)
        model._checkpointer = MockCheckpointer(is_peft=False)

        # Setup mocks
        native_state = {"qkv_proj.weight": torch.randn(96, 32)}
        mock_model_state = MagicMock()
        mock_model_state.model = [model]
        mock_model_state.state_dict.return_value = native_state
        mock_model_state_cls.return_value = mock_model_state

        hf_state = {"q_proj.weight": torch.randn(32, 32)}
        mock_adapt.return_value = hf_state

        # Call state_dict
        result = model.state_dict()

        # Verify adapter conversion was called with correct args
        mock_adapt.assert_called_once_with(model, native_state, quantization=False, device_mesh=None)
        assert result == hf_state


class TestHFCheckpointingMixinLoadStateDict:
    """Tests for load_state_dict() method."""

    @patch("nemo_automodel.components.models.common.hf_checkpointing_mixin.ModelState")
    @patch("nemo_automodel.components.models.common.hf_checkpointing_mixin._maybe_adapt_state_dict_from_hf")
    def test_load_state_dict_uses_model_state(self, mock_adapt, mock_model_state_cls):
        """Test that load_state_dict() uses ModelState to load."""
        config = MockConfig()
        model = SimpleModelWithMixin(config)
        model._checkpointer = MockCheckpointer(is_peft=False)

        # Setup mocks
        hf_state = {"linear.weight": torch.randn(32, 32)}
        native_state = {"linear.weight": torch.randn(32, 32)}
        mock_adapt.return_value = native_state

        mock_model_state = MagicMock()
        mock_model_state.model = [model]
        mock_model_state_cls.return_value = mock_model_state

        # Call load_state_dict
        model.load_state_dict(hf_state)

        # Verify ModelState was used for loading
        mock_model_state_cls.assert_called_once_with(model, is_peft=False)
        mock_model_state.load_state_dict.assert_called_once()

    @patch("nemo_automodel.components.models.common.hf_checkpointing_mixin.ModelState")
    @patch("nemo_automodel.components.models.common.hf_checkpointing_mixin._maybe_adapt_state_dict_from_hf")
    def test_load_state_dict_calls_adapter_with_moe_mesh(self, mock_adapt, mock_model_state_cls):
        """Test that load_state_dict() calls _maybe_adapt_state_dict_from_hf with moe_mesh."""
        config = MockConfig()
        model = ModelWithAdapter(config)
        model._checkpointer = MockCheckpointer(is_peft=False)

        # Setup mocks
        hf_state = {"q_proj.weight": torch.randn(32, 32)}
        native_state = {"qkv_proj.weight": torch.randn(96, 32)}
        mock_adapt.return_value = native_state

        mock_model_state = MagicMock()
        mock_model_state.model = [model]
        mock_model_state_cls.return_value = mock_model_state

        # Call load_state_dict
        model.load_state_dict(hf_state)

        # Verify adapter conversion was called with moe_mesh
        mock_adapt.assert_called_once_with(model, hf_state, moe_mesh=None)

    @patch("nemo_automodel.components.models.common.hf_checkpointing_mixin.ModelState")
    @patch("nemo_automodel.components.models.common.hf_checkpointing_mixin._maybe_adapt_state_dict_from_hf")
    def test_load_state_dict_strict_false_with_adapter(self, mock_adapt, mock_model_state_cls):
        """Test that load_state_dict() uses strict=False when model has state_dict_adapter."""
        config = MockConfig()
        model = ModelWithAdapter(config)
        model._checkpointer = MockCheckpointer(is_peft=False)

        # Setup mocks
        mock_adapt.return_value = {}
        mock_model_state = MagicMock()
        mock_model_state.model = [model]
        mock_model_state_cls.return_value = mock_model_state

        # Call load_state_dict
        model.load_state_dict({})

        # Verify strict=False was used (because model has state_dict_adapter)
        mock_model_state.load_state_dict.assert_called_once_with({}, strict=False)


class TestHFCheckpointingMixinSavePretrained:
    """Tests for save_pretrained() method."""

    def test_save_pretrained_requires_checkpointer(self):
        """Test that save_pretrained() raises error without checkpointer."""
        config = MockConfig()
        model = SimpleModelWithMixin(config)

        with pytest.raises(ValueError, match="No checkpointer provided"):
            model.save_pretrained("/tmp/test")

    def test_save_pretrained_uses_checkpointer(self):
        """Test that save_pretrained() uses Checkpointer.save_model()."""
        config = MockConfig()
        model = SimpleModelWithMixin(config)

        # Create mock checkpointer
        mock_checkpointer = MagicMock()
        model._checkpointer = mock_checkpointer

        # Call save_pretrained
        model.save_pretrained("/tmp/test")

        # Verify Checkpointer.save_model was called
        mock_checkpointer.save_model.assert_called_once_with(
            model=model,
            weights_path="/tmp/test",
            peft_config=None,
            tokenizer=None,
        )

    def test_save_pretrained_accepts_explicit_checkpointer(self):
        """Test that save_pretrained() accepts explicit checkpointer argument."""
        config = MockConfig()
        model = SimpleModelWithMixin(config)

        # Create mock checkpointer
        mock_checkpointer = MagicMock()

        # Call save_pretrained with explicit checkpointer
        model.save_pretrained("/tmp/test", checkpointer=mock_checkpointer)

        # Verify Checkpointer.save_model was called
        mock_checkpointer.save_model.assert_called_once()

    def test_save_pretrained_passes_peft_config(self):
        """Test that save_pretrained() passes peft_config from kwargs."""
        config = MockConfig()
        model = SimpleModelWithMixin(config)

        mock_checkpointer = MagicMock()
        model._checkpointer = mock_checkpointer
        peft_config = {"type": "lora"}

        model.save_pretrained("/tmp/test", peft_config=peft_config)

        mock_checkpointer.save_model.assert_called_once_with(
            model=model,
            weights_path="/tmp/test",
            peft_config=peft_config,
            tokenizer=None,
        )


class TestHFCheckpointingMixinFromPretrained:
    """Tests for from_pretrained() class method."""

    def test_from_pretrained_requires_checkpointer(self):
        """Test that from_pretrained() raises error without checkpointer."""
        with pytest.raises(ValueError, match="No checkpointer provided"):
            SimpleModelWithMixin.from_pretrained("test-model")


class TestHFCheckpointingMixinFromConfig:
    """Tests for from_config() class method."""

    @patch("nemo_automodel.components.models.common.hf_checkpointing_mixin.Checkpointer")
    def test_from_config_creates_model_with_config(self, mock_checkpointer_cls):
        """Test that from_config() creates a model with the given config."""
        # Setup mock checkpointer
        mock_checkpointer = MagicMock()
        mock_checkpointer_cls.return_value = mock_checkpointer

        config = MockConfig()
        model = SimpleModelWithMixin.from_config(config)

        # Verify model was created with config
        assert model.config is config
        assert isinstance(model, SimpleModelWithMixin)

    @patch("nemo_automodel.components.models.common.hf_checkpointing_mixin.Checkpointer")
    def test_from_config_calls_load_base_model_without_loading(self, mock_checkpointer_cls):
        """Test that from_config() calls load_base_model with load_base_model=False."""
        # Setup mock checkpointer
        mock_checkpointer = MagicMock()
        mock_checkpointer_cls.return_value = mock_checkpointer

        config = MockConfig()
        model = SimpleModelWithMixin.from_config(config)

        # Verify load_base_model was called with load_base_model=False
        mock_checkpointer.load_base_model.assert_called_once()
        call_kwargs = mock_checkpointer.load_base_model.call_args[1]
        assert call_kwargs["load_base_model"] is False

    @patch("nemo_automodel.components.models.common.hf_checkpointing_mixin.Checkpointer")
    def test_from_config_does_not_set_checkpointer_attribute(self, mock_checkpointer_cls):
        """Test that from_config() does not set _checkpointer on the model."""
        # Setup mock checkpointer
        mock_checkpointer = MagicMock()
        mock_checkpointer_cls.return_value = mock_checkpointer

        config = MockConfig()
        model = SimpleModelWithMixin.from_config(config)

        # Verify _checkpointer is NOT set (remains None) since it's internal
        assert model._checkpointer is None

    @patch("nemo_automodel.components.models.common.hf_checkpointing_mixin.Checkpointer")
    def test_from_config_creates_checkpointer_with_correct_config(self, mock_checkpointer_cls):
        """Test that from_config() creates Checkpointer with correct CheckpointingConfig."""
        # Setup mock checkpointer
        mock_checkpointer = MagicMock()
        mock_checkpointer_cls.return_value = mock_checkpointer

        config = MockConfig()
        SimpleModelWithMixin.from_config(config)

        # Verify Checkpointer was created with correct config
        mock_checkpointer_cls.assert_called_once()
        call_kwargs = mock_checkpointer_cls.call_args[1]
        assert "config" in call_kwargs
        checkpointing_config = call_kwargs["config"]
        assert checkpointing_config.enabled is True
        assert checkpointing_config.is_peft is False


class TestHFCheckpointingMixinCheckpointerAttribute:
    """Tests for _checkpointer attribute."""

    def test_checkpointer_default_none(self):
        """Test that _checkpointer is None by default."""
        config = MockConfig()
        model = SimpleModelWithMixin(config)
        assert model._checkpointer is None

    def test_checkpointer_can_be_set(self):
        """Test that _checkpointer can be set."""
        config = MockConfig()
        model = SimpleModelWithMixin(config)

        mock_checkpointer = MockCheckpointer()
        model._checkpointer = mock_checkpointer

        assert model._checkpointer is mock_checkpointer


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
