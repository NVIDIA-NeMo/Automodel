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

import logging
from unittest.mock import MagicMock, Mock, patch
import pytest
import torch
import torch.nn as nn
from nemo_automodel.recipes.llm.train_ft import _get_packed_sequence_config, build_model_and_optimizer


@pytest.mark.parametrize(
    "has_packed_sequence, is_hf_model, cp_size, return_val, raises",
    [
        (True, True, 1, {"attn_implementation": "flash_attention_2"}, None),
        (True, True, 2, {"attn_implementation": "sdpa"}, ValueError),
        (True, False, 1, {}, None),
        (True, False, 2, {}, None),
        (False, True, 1, {}, None),
        (False, True, 2, {'attn_implementation': 'sdpa'}, None),
        (False, False, 1, {}, None),
        (False, False, 2, {}, None),
    ],
)
def test_get_packed_sequence_config(has_packed_sequence, is_hf_model, cp_size, return_val, raises):
    if raises:
        with pytest.raises(raises):
            _get_packed_sequence_config(has_packed_sequence, is_hf_model, cp_size)
    else:
        assert _get_packed_sequence_config(has_packed_sequence, is_hf_model, cp_size) == return_val


class DummyLinear(nn.Module):
    """Simple linear layer for testing"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.in_features = in_features
        self.out_features = out_features


class DummyModel(nn.Module):
    """Simple model for testing PEFT + PP"""
    def __init__(self):
        super().__init__()
        self.layer1 = DummyLinear(10, 10)
        self.layer2 = DummyLinear(10, 10)

    def forward(self, x):
        x = self.layer1.weight @ x
        x = self.layer2.weight @ x
        return x


class DummyPeftConfig:
    """Mock PEFT config"""
    def __init__(self):
        self.use_triton = True
        self.dim = 8
        self.alpha = 32
        self.match_all_linear = True


class DummyOptConfig:
    """Mock optimizer config"""
    def instantiate(self, params):
        return torch.optim.SGD(params, lr=0.01)


class DummyModelConfig:
    """Mock model config"""
    def __init__(self):
        self.pretrained_model_name_or_path = None

    def instantiate(self, **kwargs):
        return DummyModel()

    def get(self, key, default=None):
        return default


def test_peft_with_pipeline_parallelism_enabled(caplog):
    """Test that PEFT can be applied with pipeline parallelism enabled"""

    # Create mock configs
    device = torch.device("cpu")
    cfg_model = DummyModelConfig()
    cfg_opt = DummyOptConfig()
    cfg_peft = DummyPeftConfig()

    # Create mock autopipeline
    mock_autopipeline = MagicMock()
    mock_autopipeline.parts = []

    # Create mock checkpointer
    mock_checkpointer = MagicMock()
    mock_checkpointer.load_base_model = MagicMock()

    # Mock the apply_lora_to_linear_modules function
    with patch('nemo_automodel.recipes.llm.train_ft.apply_lora_to_linear_modules') as mock_apply_lora:
        with patch('nemo_automodel.recipes.llm.train_ft.print_trainable_parameters'):
            with patch('nemo_automodel.recipes.llm.train_ft._supports_logits_to_keep', return_value=True):
                with caplog.at_level(logging.INFO):
                    # This should NOT raise an assertion error
                    model, state_dict_keys, optimizer, loss_fn = build_model_and_optimizer(
                        device=device,
                        cfg_model=cfg_model,
                        cfg_opt=cfg_opt,
                        cfg_peft=cfg_peft,
                        model_wrapper=None,
                        seed=42,
                        checkpointer=mock_checkpointer,
                        autopipeline=mock_autopipeline,
                        loss_fn=None,
                    )

                    # Verify that apply_lora was called
                    assert mock_apply_lora.called, "apply_lora_to_linear_modules should be called"

                    # Verify that use_triton was disabled
                    assert cfg_peft.use_triton == False, "use_triton should be disabled for PP"

                    # Verify the log message was generated
                    assert "Enabling PEFT with Pipeline Parallelism" in caplog.text


def test_peft_without_pipeline_parallelism(caplog):
    """Test that PEFT works correctly without pipeline parallelism"""

    # Create mock configs
    device = torch.device("cpu")
    cfg_model = DummyModelConfig()
    cfg_opt = DummyOptConfig()
    cfg_peft = DummyPeftConfig()

    # Create mock checkpointer
    mock_checkpointer = MagicMock()
    mock_checkpointer.load_base_model = MagicMock()

    # Mock the apply_lora_to_linear_modules function
    with patch('nemo_automodel.recipes.llm.train_ft.apply_lora_to_linear_modules') as mock_apply_lora:
        with patch('nemo_automodel.recipes.llm.train_ft.print_trainable_parameters'):
            with patch('nemo_automodel.recipes.llm.train_ft._supports_logits_to_keep', return_value=True):
                with caplog.at_level(logging.INFO):
                    # This should work fine without PP
                    model, state_dict_keys, optimizer, loss_fn = build_model_and_optimizer(
                        device=device,
                        cfg_model=cfg_model,
                        cfg_opt=cfg_opt,
                        cfg_peft=cfg_peft,
                        model_wrapper=None,
                        seed=42,
                        checkpointer=mock_checkpointer,
                        autopipeline=None,  # No pipeline parallelism
                        loss_fn=None,
                    )

                    # Verify that apply_lora was called
                    assert mock_apply_lora.called, "apply_lora_to_linear_modules should be called"

                    # use_triton could still be True (not disabled by PP)
                    # The PP-specific log should not appear
                    assert "Enabling PEFT with Pipeline Parallelism" not in caplog.text


def test_peft_with_tp_disables_triton(caplog):
    """Test that PEFT with tensor parallelism disables triton"""

    # Create mock configs
    device = torch.device("cpu")
    cfg_model = DummyModelConfig()
    cfg_opt = DummyOptConfig()
    cfg_peft = DummyPeftConfig()

    # Create mock checkpointer
    mock_checkpointer = MagicMock()
    mock_checkpointer.load_base_model = MagicMock()

    # Mock the apply_lora_to_linear_modules function
    with patch('nemo_automodel.recipes.llm.train_ft.apply_lora_to_linear_modules') as mock_apply_lora:
        with patch('nemo_automodel.recipes.llm.train_ft.print_trainable_parameters'):
            with patch('nemo_automodel.recipes.llm.train_ft._supports_logits_to_keep', return_value=True):
                with caplog.at_level(logging.INFO):
                    # Test with TP > 1
                    model, state_dict_keys, optimizer, loss_fn = build_model_and_optimizer(
                        device=device,
                        cfg_model=cfg_model,
                        cfg_opt=cfg_opt,
                        cfg_peft=cfg_peft,
                        model_wrapper=None,
                        seed=42,
                        checkpointer=mock_checkpointer,
                        tp_size=2,  # Enable TP
                        autopipeline=None,
                        loss_fn=None,
                    )

                    # Verify that use_triton was disabled
                    assert cfg_peft.use_triton == False, "use_triton should be disabled for TP"

                    # Verify the TP log message was generated
                    assert "Disabling Triton with TP" in caplog.text
