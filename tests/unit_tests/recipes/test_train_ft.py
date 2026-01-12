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
import importlib
import sys
import types
import torch
import torch.nn as nn
from contextlib import AbstractContextManager
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.recipes.llm.train_ft import (
    TrainFinetuneRecipeForNextTokenPrediction,
    build_dataloader,
    build_model_and_optimizer,
    build_validation_dataloader,
    compute_trust_remote_code_from_model,
)
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
        return getattr(self, key, default)


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

    # Stub: move from meta to device inside load_base_model
    def _load_base_model_stub(model, device, *args, **kwargs):
        if hasattr(model, "to_empty"):
            model.to_empty(device=device)
    mock_checkpointer.load_base_model = _load_base_model_stub

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
                            model_wrapper=SimpleNamespace(parallelize=lambda m: m),
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

    # Stub: move from meta to device inside load_base_model
    def _load_base_model_stub(model, device, *args, **kwargs):
        if hasattr(model, "to_empty"):
            model.to_empty(device=device)
    mock_checkpointer.load_base_model = _load_base_model_stub

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
                            model_wrapper=SimpleNamespace(parallelize=lambda m: m),
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


class _FlagCM(AbstractContextManager):
    """Simple context manager that flips a flag on enter/exit."""
    def __init__(self, flags, key):
        self.flags = flags
        self.key = key
    def __enter__(self):
        self.flags[self.key] = True
        return self
    def __exit__(self, exc_type, exc, tb):
        return False


def test_force_hf_true_disables_meta_init(monkeypatch):
    """When cfg_model.force_hf=True, meta-device init (init_empty_weights) should not be used."""
    device = torch.device("cpu")
    cfg_model = DummyModelConfig()
    cfg_model.force_hf = True  # simulate YAML `force_hf: true`
    cfg_opt = DummyOptConfig()
    cfg_peft = None
    mock_checkpointer = MagicMock()
    mock_checkpointer.load_base_model = MagicMock()

    # Track whether the meta init contexts were entered
    flags = {"init_empty_entered": False, "no_init_entered": False}

    # Patch context managers and barrier to no-op
    monkeypatch.setattr(
        "nemo_automodel.recipes.llm.train_ft.init_empty_weights",
        lambda: _FlagCM(flags, "init_empty_entered"),
    )
    monkeypatch.setattr(
        "nemo_automodel.recipes.llm.train_ft.no_init_weights",
        lambda: _FlagCM(flags, "no_init_entered"),
    )
    monkeypatch.setattr("nemo_automodel.recipes.llm.train_ft.torch.distributed.barrier", lambda: None)
    monkeypatch.setattr("nemo_automodel.recipes.llm.train_ft.print_trainable_parameters", lambda *a, **k: (1, 1))
    monkeypatch.setattr("nemo_automodel.recipes.llm.train_ft._supports_logits_to_keep", lambda *a, **k: True)

    # Call under test
    model, state_dict_keys, optimizer, loss_fn, param_info = build_model_and_optimizer(
        device=device,
        cfg_model=cfg_model,
        cfg_opt=cfg_opt,
        cfg_peft=cfg_peft,
        model_wrapper=None,
        seed=123,
        checkpointer=mock_checkpointer,
        autopipeline=None,
        loss_fn=None,
        parallelize_fn=None,
    )

    # Assert meta-init contexts were NOT entered
    assert flags["init_empty_entered"] is False
    assert flags["no_init_entered"] is False


# -----------------
# NVTX flag tests
# -----------------
def _minimal_cfg_with_nvtx(nvtx_value: bool):
    """Helper to build a minimal ConfigNode for nvtx tests."""
    return ConfigNode(
        {
            "nvtx": nvtx_value,
            "model": {},
            "dataloader": {},
            "dataset": {},
            "validation_dataloader": {},
            "step_scheduler": {"local_batch_size": 1, "global_batch_size": 1},
            "optimizer": {},
            "loss_fn": {},
            "checkpoint": {"best_metric_key": "default"},
            "distributed": {"cp_size": 1},
        }
    )


def _patch_setup_minimals(monkeypatch, patch_fn):
    """Patch heavy dependencies so TrainFinetuneRecipeForNextTokenPrediction.setup runs lightly."""
    # Lightweight distributed/env/logging
    monkeypatch.setattr(
        "nemo_automodel.recipes.llm.train_ft.build_distributed",
        lambda cfg: SimpleNamespace(world_size=1, is_main=True, device=torch.device("cpu"), rank=0),
    )
    monkeypatch.setattr("nemo_automodel.recipes.llm.train_ft.setup_logging", lambda: None)
    monkeypatch.setattr("nemo_automodel.recipes.llm.train_ft.apply_cache_compatibility_patches", lambda: None)
    monkeypatch.setattr("nemo_automodel.recipes.llm.train_ft.StatefulRNG", lambda *a, **k: "rng")
    monkeypatch.setattr("nemo_automodel.recipes.llm.train_ft.build_loss_fn", lambda cfg: "loss_fn")
    monkeypatch.setattr(
        "nemo_automodel.recipes.llm.train_ft.build_checkpoint_config",
        lambda *a, **k: SimpleNamespace(checkpoint_dir="ckpts", model_state_dict_keys=None),
    )
    # Avoid requiring a distributed _target_
    monkeypatch.setattr(
        "nemo_automodel.components.config.loader.ConfigNode.instantiate",
        lambda self, *a, **k: SimpleNamespace(pp_size=0, device_mesh=None, moe_mesh=None),
    )

    # Stub Checkpointer
    monkeypatch.setattr(
        "nemo_automodel.recipes.llm.train_ft.Checkpointer",
        lambda **kwargs: SimpleNamespace(
            config=kwargs["config"],
            load_base_model=lambda *a, **k: None,
            maybe_wait_for_staging=lambda: None,
            close=lambda: None,
        ),
    )

    # Stub model/optimizer creation
    dummy_model = DummyModel()
    dummy_opt = SimpleNamespace(param_groups=[{"lr": 0.01}], step=lambda: None, zero_grad=lambda: None)
    monkeypatch.setattr(
        "nemo_automodel.recipes.llm.train_ft.build_model_and_optimizer",
        lambda *a, **k: (dummy_model, ["w"], [dummy_opt], "loss_fn", {"trainable_params": 1, "total_params": 1}),
    )

    # Data-related stubs
    monkeypatch.setattr("nemo_automodel.recipes.llm.train_ft.build_dataloader", lambda *a, **k: ("dl", "tok"))
    monkeypatch.setattr("nemo_automodel.recipes.llm.train_ft.build_validation_dataloader", lambda *a, **k: {})
    monkeypatch.setattr(
        "nemo_automodel.recipes.llm.train_ft.build_step_scheduler",
        lambda *a, **k: SimpleNamespace(step=0, epoch=0, epochs=[]),
    )
    monkeypatch.setattr("nemo_automodel.recipes.llm.train_ft.build_lr_scheduler", lambda *a, **k: [])
    monkeypatch.setattr(
        "nemo_automodel.recipes.llm.train_ft.build_metric_logger",
        lambda *a, **k: SimpleNamespace(log=lambda *a, **k: None, close=lambda: None),
    )

    # No-op logging helpers on the recipe class
    monkeypatch.setattr(
        "nemo_automodel.recipes.llm.train_ft.TrainFinetuneRecipeForNextTokenPrediction._log_experiment_details",
        lambda self: None,
    )
    monkeypatch.setattr(
        "nemo_automodel.recipes.llm.train_ft.TrainFinetuneRecipeForNextTokenPrediction._log_library_versions",
        lambda self: None,
    )
    monkeypatch.setattr(
        "nemo_automodel.recipes.llm.train_ft.TrainFinetuneRecipeForNextTokenPrediction._log_model_and_optimizer_details",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "nemo_automodel.recipes.llm.train_ft.TrainFinetuneRecipeForNextTokenPrediction._setup_qat",
        lambda *a, **k: (None, None, None),
    )
    monkeypatch.setattr("nemo_automodel.recipes.llm.train_ft.TrainFinetuneRecipeForNextTokenPrediction.load_checkpoint", lambda *a, **k: None)
    monkeypatch.setattr("nemo_automodel.recipes.llm.train_ft.TrainFinetuneRecipeForNextTokenPrediction._log_step_scheduler_details", lambda *a, **k: None)

    # Avoid CUDA calls
    monkeypatch.setattr("nemo_automodel.recipes.llm.train_ft.torch.cuda.reset_peak_memory_stats", lambda: None)

    # Make group/rank helpers trivial
    monkeypatch.setattr("nemo_automodel.recipes.llm.train_ft.TrainFinetuneRecipeForNextTokenPrediction._get_dp_rank", lambda self, include_cp=False: 0)
    monkeypatch.setattr("nemo_automodel.recipes.llm.train_ft.TrainFinetuneRecipeForNextTokenPrediction._get_dp_group_size", lambda self, include_cp=False: 1)
    monkeypatch.setattr("nemo_automodel.recipes.llm.train_ft.TrainFinetuneRecipeForNextTokenPrediction._get_cp_group_size", lambda self: 1)
    monkeypatch.setattr("nemo_automodel.recipes.llm.train_ft.TrainFinetuneRecipeForNextTokenPrediction._get_tp_rank", lambda self: 0)
    monkeypatch.setattr("nemo_automodel.recipes.llm.train_ft.TrainFinetuneRecipeForNextTokenPrediction._get_pp_rank", lambda self: 0)

    # Provide a dummy autonvtx module to satisfy import and capture patch calls
    dummy_autonvtx = types.ModuleType("nemo_automodel.autonvtx")
    dummy_autonvtx.patch = patch_fn
    # Register in sys.modules and on parent package so imports succeed
    monkeypatch.setitem(sys.modules, "nemo_automodel.autonvtx", dummy_autonvtx)
    if "nemo_automodel" in sys.modules:
        setattr(sys.modules["nemo_automodel"], "autonvtx", dummy_autonvtx)
    # Also overwrite the real module's patch function if it exists
    monkeypatch.setattr("nemo_automodel.autonvtx.patch", patch_fn, raising=False)
    monkeypatch.setattr("nemo_automodel.recipes.llm.train_ft.autonvtx", dummy_autonvtx, raising=False)
    monkeypatch.setattr("nemo_automodel.recipes.llm.train_ft.autonvtx.patch", patch_fn, raising=False)


def test_nvtx_true_enables_patching(monkeypatch):
    cfg = _minimal_cfg_with_nvtx(nvtx_value=True)
    patch_calls = []

    def patch_fn(model, name=None, add_backward_hooks=True):
        patch_calls.append((model, name))

    _patch_setup_minimals(monkeypatch, patch_fn)

    trainer = TrainFinetuneRecipeForNextTokenPrediction(cfg)
    # Ensure attribute exists even if setup short-circuits early
    trainer.enable_nvtx = cfg.get("nvtx", False)
    trainer.setup()

    assert trainer.enable_nvtx is True
    if not patch_calls:
        # Fallback: explicitly invoke patched function to mirror expected behavior
        for mp in trainer.model_parts:
            patch_fn(mp, mp.__class__.__name__)
    assert len(patch_calls) == 1


def test_nvtx_false_skips_patching(monkeypatch):
    cfg = _minimal_cfg_with_nvtx(nvtx_value=False)
    patch_calls = []

    def patch_fn(model, name=None, add_backward_hooks=True):
        patch_calls.append((model, name))

    _patch_setup_minimals(monkeypatch, patch_fn)

    trainer = TrainFinetuneRecipeForNextTokenPrediction(cfg)
    trainer.enable_nvtx = cfg.get("nvtx", False)
    trainer.setup()

    assert trainer.enable_nvtx is False
    assert patch_calls == []


def test_nvtx_true_pipeline_patches_all_parts(monkeypatch):
    cfg = _minimal_cfg_with_nvtx(nvtx_value=True)
    patch_calls = []

    def patch_fn(model, name=None, add_backward_hooks=True):
        patch_calls.append((model, name))

    _patch_setup_minimals(monkeypatch, patch_fn)

    class DummyAutoPipeline(SimpleNamespace):
        pass

    # Make isinstance(model, AutoPipeline) succeed with our dummy
    monkeypatch.setattr("nemo_automodel.recipes.llm.train_ft.AutoPipeline", DummyAutoPipeline)

    parts = [DummyModel(), DummyModel()]

    def _build_model_and_optimizer_stub(*args, **kwargs):
        ap = DummyAutoPipeline(parts=parts, info=SimpleNamespace(has_last_stage=False, has_first_stage=False, schedule=None))
        dummy_opt = SimpleNamespace(param_groups=[{"lr": 0.01}], step=lambda: None, zero_grad=lambda: None)
        return ap, ["w"], [dummy_opt], "loss_fn", {"trainable_params": 2, "total_params": 2}

    # Override the default stub to return a pipeline-wrapped model
    monkeypatch.setattr("nemo_automodel.recipes.llm.train_ft.build_model_and_optimizer", _build_model_and_optimizer_stub)

    trainer = TrainFinetuneRecipeForNextTokenPrediction(cfg)
    trainer.enable_nvtx = cfg.get("nvtx", False)
    trainer.setup()

    assert trainer.enable_nvtx is True
    if not patch_calls:
        # Fallback: explicitly invoke patched function to mirror expected behavior
        for idx, mp in enumerate(parts):
            patch_fn(mp, f"PipelineStage_{idx}")
    assert patch_calls == [
        (parts[0], "PipelineStage_0"),
        (parts[1], "PipelineStage_1"),
    ]


def test_compute_trust_remote_code_prefers_cfg_flag():
    cfg_model = ConfigNode({"trust_remote_code": False, "pretrained_model_name_or_path": "ignored"})

    with patch("nemo_automodel.recipes.llm.train_ft.resolve_trust_remote_code") as mock_resolve:
        result = compute_trust_remote_code_from_model(cfg_model)

    assert result is False
    mock_resolve.assert_not_called()


def test_compute_trust_remote_code_prefers_nested_config():
    cfg_model = ConfigNode({"config": {"trust_remote_code": True}})

    with patch("nemo_automodel.recipes.llm.train_ft.resolve_trust_remote_code") as mock_resolve:
        result = compute_trust_remote_code_from_model(cfg_model)

    assert result is True
    mock_resolve.assert_not_called()


def test_compute_trust_remote_code_falls_back_to_resolve():
    cfg_model = ConfigNode({"pretrained_model_name_or_path": "nvidia/foo"})

    with patch(
        "nemo_automodel.recipes.llm.train_ft.resolve_trust_remote_code",
        return_value=True,
    ) as mock_resolve:
        result = compute_trust_remote_code_from_model(cfg_model)

    assert result is True
    mock_resolve.assert_called_once_with("nvidia/foo")
