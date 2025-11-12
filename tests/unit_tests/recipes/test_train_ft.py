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
from nemo_automodel.recipes.llm.train_ft import _get_packed_sequence_config, build_validation_dataloader, build_dataloader, build_model_and_optimizer
from nemo_automodel.components.config.loader import ConfigNode
from unittest.mock import patch
import importlib
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset


class DummyIterableDataset(IterableDataset):  # noqa: D401
    """Minimal iterable dataset with shard/shuffle hooks for testing build_dataloader."""

    def __init__(self, items=None, num_shards=1, tokenizer=None, **kwargs):
        super().__init__()
        self.items = items or list(range(10))
        self.num_shards = num_shards
        self._shard = None
        self._shuffle_calls = []
        self.dataset = self.items  # mimic underlying HF dataset holder

    def __iter__(self):  # pragma: no cover - iteration not needed in these tests
        it = self.items
        if self._shard is not None:
            n, idx = self._shard
            it = [x for i, x in enumerate(it) if i % n == idx]
        for x in it:
            yield x

    def shard(self, num_shards, index):
        self._shard = (num_shards, index)
        return self

    def shuffle(self, buffer_size: int, seed: int):
        self._shuffle_calls.append((buffer_size, seed))
        return self


def dl_factory_capture(**kwargs):  # returns a sentinel while exposing passed kwargs via attribute
    dl_factory_capture.captured = kwargs
    return "dl"


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

def test_build_validation_dataloader_pp_enabled(caplog):
    cfg = ConfigNode(
        {
            "model": {},
            "validation_dataloader": {},
        }
    )

    with caplog.at_level(logging.WARNING):
        result = build_validation_dataloader(cfg, dp_world_size=2, dp_rank=0, pp_enabled=True)

    assert result == {}


def test_build_validation_dataloader_collects_and_names_properly():
    # Multiple validation dataset keys with different separators
    cfg = ConfigNode(
        {
            "model": {},
            "validation_dataloader": {},
            "distributed": {"cp_size": 3},
            "step_scheduler": {
                "local_batch_size": 8,
                "global_batch_size": 16,
                "max_steps": 123,
                "val_every_steps": 10,
            },
            # Keys to be discovered via cfg.to_dict().keys()
            "validation_dataset": {"some": "cfg"},
            "validation_dataset_val": {"some": "cfg"},
            "validation_dataset-test": {"some": "cfg"},
            "validation_dataset.foo": {"some": "cfg"},
        }
    )

    expected_names = {"default", "val", "test", "foo"}

    with patch("nemo_automodel.recipes.llm.train_ft.build_dataloader", return_value=("dl", "tok")) as mock_build:
        result = build_validation_dataloader(cfg, dp_world_size=4, dp_rank=1, pp_enabled=False)

    # Assert keys are correctly generated
    assert set(result.keys()) == expected_names
    # Values should be the first element of the tuple returned by build_dataloader
    assert set(result.values()) == {"dl"}
    # build_dataloader called once per validation dataset
    assert mock_build.call_count == 4

    # Inspect one call for important kwargs
    _, kwargs = mock_build.call_args
    assert kwargs["dp_world_size"] == 4
    assert kwargs["dp_rank"] == 1
    assert kwargs["pp_enabled"] is False
    assert kwargs["supports_seq_lens"] is True
    assert kwargs["cp_size"] == 3


def test_build_validation_dataloader_no_validation_keys():
    cfg = ConfigNode(
        {
            "model": {},
            "validation_dataloader": {},
        }
    )

    with patch("nemo_automodel.recipes.llm.train_ft.build_dataloader") as mock_build:
        result = build_validation_dataloader(cfg, dp_world_size=1, dp_rank=0, pp_enabled=False)

    assert result == {}
    mock_build.assert_not_called()

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
        with patch('nemo_automodel.recipes.llm.train_ft.print_trainable_parameters', return_value=(100, 1000)):
            with patch('nemo_automodel.recipes.llm.train_ft._supports_logits_to_keep', return_value=True):
                with caplog.at_level(logging.INFO):
                    # This should NOT raise an assertion error
                    model, state_dict_keys, optimizer, loss_fn, param_info = build_model_and_optimizer(
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

                    # Verify that the param_info is correct
                    assert param_info == {"trainable_params": 100, "total_params": 1000}


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
        with patch('nemo_automodel.recipes.llm.train_ft.print_trainable_parameters', return_value=(100, 1000)):
            with patch('nemo_automodel.recipes.llm.train_ft._supports_logits_to_keep', return_value=True):
                with caplog.at_level(logging.INFO):
                    # This should work fine without PP
                    model, state_dict_keys, optimizer, loss_fn, param_info = build_model_and_optimizer(
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

                    # Verify that the param_info is correct
                    assert param_info == {"trainable_params": 100, "total_params": 1000}


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
        with patch('nemo_automodel.recipes.llm.train_ft.print_trainable_parameters', return_value=(100, 1000)):
            with patch('nemo_automodel.recipes.llm.train_ft._supports_logits_to_keep', return_value=True):
                with caplog.at_level(logging.INFO):
                    # Test with TP > 1
                    model, state_dict_keys, optimizer, loss_fn, param_info = build_model_and_optimizer(
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

                    # Verify that the param_info is correct
                    assert param_info == {"trainable_params": 100, "total_params": 1000}


def test_build_dataloader_iterable_shard_and_shuffle_removed_from_cfg(monkeypatch):
    # cfg_ds: target resolves to this test module dataset class
    cfg_ds = ConfigNode(
        {
            "_target_": "tests.unit_tests.recipes.test_train_ft.DummyIterableDataset",
            "tokenizer": None,
            "num_shards": 4,
        }
    )
    # cfg_dl: target captures kwargs and returns sentinel
    cfg_dl = ConfigNode(
        {
            "_target_": "tests.unit_tests.recipes.test_train_ft.dl_factory_capture",
            "shuffle": True,
            "shuffle_buffer_size": 8,
            "num_workers": 0,
        }
    )
    cfg_model = ConfigNode({})
    cfg_ps = ConfigNode({})

    dl, tok = build_dataloader(
        cfg_ds=cfg_ds,
        cfg_dl=cfg_dl,
        cfg_model=cfg_model,
        cfg_ps=cfg_ps,
        seed=123,
        local_batch_size=2,
        global_batch_size=4,
        max_steps=None,
        val_check_interval=None,
        dp_rank=1,
        dp_world_size=2,
        pp_enabled=False,
        supports_seq_lens=True,
        cp_size=1,
    )

    assert dl == "dl"
    assert tok is None
    mod = importlib.import_module("tests.unit_tests.recipes.test_train_ft")
    captured = getattr(mod.dl_factory_capture, "captured")
    # Ensure shuffle-related keys are not forwarded to DataLoader instantiation
    assert "shuffle" not in captured and "shuffle_buffer_size" not in captured
    ds = captured["dataset"]
    # Avoid fragile identity issues from re-imports; validate by name and interface
    assert ds.__class__.__name__ == "DummyIterableDataset"
    # Shard path used when num_shards >= dp_world_size
    assert ds._shard == (2, 1)
    # Shuffle called with buffer size and seed
    assert ds._shuffle_calls and ds._shuffle_calls[-1] == (8, 123)


    
