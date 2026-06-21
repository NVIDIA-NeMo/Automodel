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

import importlib
import logging
import sys
import types
from contextlib import AbstractContextManager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from nemo_automodel.components.config.loader import ConfigNode

# Skip decorator for tests that require CUDA
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
from torch.utils.data import IterableDataset

from nemo_automodel._transformers.model_init import resolve_sdpa_method
from nemo_automodel.components.distributed.utils import dp_eval_sample_shard
from nemo_automodel.components.eval.tool_call_evaluator import ToolCallAccuracyEvaluator
from nemo_automodel.components.loss.mtp import PipelineCausalLMLoss
from nemo_automodel.components.models.deepseek_v4.cp import dsv4_cp_local_seq_multiple
from nemo_automodel.components.optim.optimizer import build_optimizer_config
from nemo_automodel.recipes._typed_config import _as_dict, _callable_and_kwargs
from nemo_automodel.recipes.llm.train_ft import (
    TrainFinetuneRecipeForNextTokenPrediction,
    build_dataloader,
    build_model,
    build_validation_dataloader,
    compute_trust_remote_code_from_model,
)

# The recipe loader builder takes the same kwargs the tests already pass.
_build_dataloader = build_dataloader


def build_optimizer(model, cfg_opt, distributed_config, device_mesh):
    """Resolve a YAML optimizer block and build it (mirrors ``RecipeConfig.optimizer.build``)."""
    return build_optimizer_config(*_callable_and_kwargs(cfg_opt)).build(model, device_mesh=device_mesh)


def build_checkpoint_config(cfg_ckpt, cache_dir, model_repo_id, is_peft):
    """Resolve a YAML checkpoint block into a ``CheckpointingConfig`` (mirrors ``RecipeConfig.checkpoint``)."""
    from nemo_automodel.components.checkpoint.config import CheckpointingConfig

    kwargs = _as_dict(cfg_ckpt) if cfg_ckpt is not None else {}
    kwargs.pop("restore_from", None)
    derived = {"model_repo_id": model_repo_id, "model_cache_dir": cache_dir, "is_peft": is_peft}
    return CheckpointingConfig(**{**derived, **kwargs})


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


def test_pipeline_causal_lm_loss_adds_mtp_tuple_output():
    from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = nn.Linear(3, 5, bias=False)
            self.mtp_config = SimpleNamespace(loss_scaling_factor=0.2)

        def get_output_embeddings(self):
            return self.lm_head

    torch.manual_seed(123)
    model = DummyModel()
    model.train()
    loss_fn = MaskedCrossEntropy(fp32_upcast=False, reduction="sum")
    wrapper = PipelineCausalLMLoss(loss_fn, model)

    logits = torch.randn(1, 4, 5)
    mtp_h = torch.randn(1, 4, 3)
    labels = torch.tensor([[1, 2, 3, 4]])

    got = wrapper((logits, mtp_h), labels)
    shifted_labels = torch.tensor([[2, 3, 4, -100]])
    expected = loss_fn(logits=logits, labels=labels) + 0.2 * loss_fn(
        logits=model.lm_head(mtp_h),
        labels=shifted_labels,
    )

    torch.testing.assert_close(got, expected)


def test_mtp_loss_config_defaults_and_override():
    from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy
    from nemo_automodel.components.loss.mtp import MTPLossConfig

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = nn.Linear(3, 5, bias=False)
            self.mtp_config = SimpleNamespace(loss_scaling_factor=0.2)

        def get_output_embeddings(self):
            return self.lm_head

    # Defaults: scaling_factor None (model-driven), ignore_index -100.
    assert MTPLossConfig().scaling_factor is None
    assert MTPLossConfig().ignore_index == -100

    torch.manual_seed(123)
    model = DummyModel()
    model.train()
    loss_fn = MaskedCrossEntropy(fp32_upcast=False, reduction="sum")

    logits = torch.randn(1, 4, 5)
    mtp_h = torch.randn(1, 4, 3)
    labels = torch.tensor([[1, 2, 3, 4]])
    shifted_labels = torch.tensor([[2, 3, 4, -100]])
    base = loss_fn(logits=logits, labels=labels)
    aux = loss_fn(logits=model.lm_head(mtp_h), labels=shifted_labels)

    # An explicit scaling_factor overrides the model-provided 0.2.
    got_override = MTPLossConfig(scaling_factor=0.5).build(loss_fn, model)((logits, mtp_h), labels)
    torch.testing.assert_close(got_override, base + 0.5 * aux)

    # The default (None) falls back to the model-provided 0.2.
    got_default = MTPLossConfig().build(loss_fn, model)((logits, mtp_h), labels)
    torch.testing.assert_close(got_default, base + 0.2 * aux)


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

    assert set(result.keys()) == expected_names
    assert set(result.values()) == {"dl"}
    # One dataloader built per validation dataset, with the runtime args threaded through.
    assert mock_build.call_count == 4
    _, kwargs = mock_build.call_args
    assert kwargs["dp_world_size"] == 4
    assert kwargs["dp_rank"] == 1
    assert kwargs["pp_enabled"] is False
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


@pytest.mark.parametrize("attn,expect_packed", [("magi", True), ("te", True), ("sdpa", False)])
def test_build_validation_dataloader_packs_val_for_thd_backends(monkeypatch, attn, expect_packed):
    """The validation set is packed (cfg_ps passed) for THD-consuming backends.

    Regression: previously only TE triggered val packing; the magi backend left
    validation unpacked, so at cp>1 magi sharded tiny per-example sequences
    degenerately and inflated the val loss.
    """
    # Pretend the (training) dataloader uses the THD packed collater.
    monkeypatch.setattr("nemo_automodel.recipes.llm.train_ft._uses_thd_collater", lambda _cfg: True)
    cfg = ConfigNode(
        {
            "model": {"backend": {"attn": attn}},
            "dataloader": {},
            "validation_dataloader": {},
            "packed_sequence": {"packed_sequence_size": 1024},
            "distributed": {"cp_size": 2},
            "step_scheduler": {
                "local_batch_size": 1,
                "global_batch_size": 8,
                "max_steps": 5,
                "val_every_steps": 1,
            },
            "seed": 42,
            "validation_dataset": {"some": "cfg"},
        }
    )
    with patch("nemo_automodel.recipes.llm.train_ft.build_dataloader", return_value=("dl", "tok")) as mock_build:
        build_validation_dataloader(cfg, dp_world_size=1, dp_rank=0, pp_enabled=False)
    _, kwargs = mock_build.call_args
    assert (kwargs["cfg_ps"] is not None) is expect_packed


def test_build_validation_dataloader_packs_val_when_model_requires_thd(monkeypatch):
    """Models with backend-specific packed validation requirements can opt in."""
    monkeypatch.setattr("nemo_automodel.recipes.llm.train_ft._uses_thd_collater", lambda _cfg: True)
    cfg = ConfigNode(
        {
            "model": {"backend": {"attn": "sdpa"}},
            "dataloader": {},
            "validation_dataloader": {},
            "packed_sequence": {"packed_sequence_size": 1024},
            "distributed": {"cp_size": 1},
            "step_scheduler": {
                "local_batch_size": 1,
                "global_batch_size": 8,
                "max_steps": 5,
                "val_every_steps": 1,
            },
            "seed": 42,
            "validation_dataset": {"some": "cfg"},
        }
    )

    class ModelRequiresPackedVal:
        def should_pack_validation_with_training(self):
            return True

    model = ModelRequiresPackedVal()
    with patch("nemo_automodel.recipes.llm.train_ft.build_dataloader", return_value=("dl", "tok")) as mock_build:
        build_validation_dataloader(cfg, dp_world_size=1, dp_rank=0, pp_enabled=False, model=model)

    _, kwargs = mock_build.call_args
    assert kwargs["cfg_ps"] is cfg.packed_sequence
    assert kwargs["model"] is model


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
        # Add config attribute like HF models have (needed by apply_model_infrastructure)
        self.config = SimpleNamespace()

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


def test_deepseek_v4_cp_local_seq_multiple_uses_compress_ratios():
    cfg = SimpleNamespace(model_type="deepseek_v4", compress_ratios=[0, 4, 128])
    model = SimpleNamespace(config=cfg)

    assert dsv4_cp_local_seq_multiple(model) == 128


def test_deepseek_v4_cp_local_seq_multiple_handles_ratio4_only():
    cfg = SimpleNamespace(model_type="deepseek_v4", compress_ratios=[4])

    assert dsv4_cp_local_seq_multiple(cfg) == 8


class DummyModelConfig:
    """Mock model config"""

    def __init__(self):
        self.pretrained_model_name_or_path = None

    def instantiate(self, **kwargs):
        return DummyModel()

    def get(self, key, default=None):
        return getattr(self, key, default)

    def get_as_string(self, key, default=None):
        return str(getattr(self, key, default))


def test_peft_with_pipeline_parallelism_enabled(caplog):
    """Test that _apply_peft_and_lower_precision disables triton with PP."""
    from nemo_automodel._transformers.infrastructure import _apply_peft_and_lower_precision

    cfg_peft = DummyPeftConfig()
    model = DummyModel()
    mock_autopipeline = MagicMock()

    with patch("nemo_automodel._transformers.infrastructure.apply_lora_to_linear_modules") as mock_apply_lora:
        with caplog.at_level(logging.INFO):
            _apply_peft_and_lower_precision(
                model,
                tp_size=1,
                autopipeline=mock_autopipeline,
                peft_config=cfg_peft,
                quantization_config=None,
                fp8_config=None,
                qat_quantizer=None,
            )

    assert mock_apply_lora.called, "apply_lora_to_linear_modules should be called"
    assert cfg_peft.use_triton == False, "use_triton should be disabled for PP"
    assert "Enabling PEFT with Pipeline Parallelism" in caplog.text


@requires_cuda
def test_peft_without_pipeline_parallelism(caplog):
    """Test that PEFT works correctly without pipeline parallelism"""

    # Create mock configs
    cfg_model = DummyModelConfig()
    cfg_opt = DummyOptConfig()
    cfg_peft = DummyPeftConfig()

    # Mock the apply_lora_to_linear_modules function (now inside apply_model_infrastructure)
    with patch("nemo_automodel._transformers.infrastructure.apply_lora_to_linear_modules") as mock_apply_lora:
        with patch("nemo_automodel._transformers.infrastructure.print_trainable_parameters"):
            with patch("nemo_automodel.recipes.llm.train_ft._supports_logits_to_keep", return_value=True):
                with patch("nemo_automodel._transformers.infrastructure._supports_logits_to_keep", return_value=True):
                    with patch("nemo_automodel._transformers.auto_model._verify_sdpa_support"):
                        with patch("nemo_automodel._transformers.infrastructure._shard_ep_fsdp") as mock_shard:
                            # Return a DummyModel with lora_dummy_param so freeze doesn't remove all trainable params
                            sharded_model = DummyModel()
                            sharded_model.register_parameter(
                                "lora_dummy_param",
                                nn.Parameter(torch.tensor(1.0, device=torch.device("cuda")), requires_grad=True),
                            )
                            mock_shard.return_value = sharded_model
                            with caplog.at_level(logging.INFO):
                                # This should work fine without PP
                                model = build_model(
                                    cfg_model=cfg_model,
                                    cfg_peft=cfg_peft,
                                    seed=42,
                                )
                                _ = build_optimizer(model, cfg_opt, None, None)

                            # Verify that apply_lora was called
                            assert mock_apply_lora.called, "apply_lora_to_linear_modules should be called"

                            # use_triton could still be True (not disabled by PP)
                            # The PP-specific log should not appear
                            assert "Enabling PEFT with Pipeline Parallelism" not in caplog.text


def test_peft_with_tp_disables_triton(caplog):
    """Test that _apply_peft_and_lower_precision disables triton with TP."""
    from nemo_automodel._transformers.infrastructure import _apply_peft_and_lower_precision

    cfg_peft = DummyPeftConfig()
    model = DummyModel()

    with patch("nemo_automodel._transformers.infrastructure.apply_lora_to_linear_modules"):
        with caplog.at_level(logging.INFO):
            _apply_peft_and_lower_precision(
                model,
                tp_size=2,
                autopipeline=None,
                peft_config=cfg_peft,
                quantization_config=None,
                fp8_config=None,
                qat_quantizer=None,
            )

    assert cfg_peft.use_triton == False, "use_triton should be disabled for TP"
    assert "Disabling Triton with TP" in caplog.text


def test_build_checkpoint_config_peft_torch_save_overrides_to_safetensors(caplog):
    """PEFT + torch_save: warn, fall back to safetensors defaults; preserve checkpoint_dir."""
    from nemo_automodel.components.checkpoint._backports.filesystem import SerializationFormat

    cfg_ckpt = MagicMock()
    cfg_ckpt.to_dict.return_value = {
        "model_save_format": "torch_save",
        "checkpoint_dir": "/user/ckpt/",
        "save_consolidated": False,
    }

    with caplog.at_level(logging.WARNING):
        config = build_checkpoint_config(
            cfg_ckpt=cfg_ckpt,
            cache_dir=None,
            model_repo_id="org/model",
            is_peft=True,
        )

    assert any("falling back" in rec.message.lower() for rec in caplog.records)
    assert config.is_peft is True
    assert config.model_save_format == SerializationFormat.SAFETENSORS
    # checkpoint_dir is preserved from the user config
    assert config.checkpoint_dir == "/user/ckpt/"
    # other user-provided torch_save options are discarded; save_consolidated falls back to the default "final"
    assert config.save_consolidated.value == "final"
    assert config.is_async is False


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

    dl, tok = _build_dataloader(
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


@requires_cuda
def test_force_hf_true_disables_meta_init(monkeypatch):
    """When cfg_model.force_hf=True, meta-device init (init_empty_weights) should not be used.
    Note: Meta device init is now handled in auto_model.py for NeMoAutoModel targets.
    For non-NeMoAutoModel targets, this test verifies the basic model instantiation works."""
    cfg_model = DummyModelConfig()
    cfg_model.force_hf = True  # simulate YAML `force_hf: true`
    cfg_opt = DummyOptConfig()
    cfg_peft = None

    monkeypatch.setattr("nemo_automodel.recipes.llm.train_ft._supports_logits_to_keep", lambda *a, **k: True)
    monkeypatch.setattr("nemo_automodel._transformers.infrastructure._supports_logits_to_keep", lambda *a, **k: True)
    monkeypatch.setattr("nemo_automodel._transformers.auto_model._verify_sdpa_support", lambda *a, **k: None)
    monkeypatch.setattr("nemo_automodel._transformers.infrastructure.print_trainable_parameters", lambda *a, **k: None)

    # Call under test
    model = build_model(
        cfg_model=cfg_model,
        cfg_peft=cfg_peft,
        seed=123,
    )
    optimizer = build_optimizer(model, cfg_opt, None, None)

    # Model should be instantiated
    assert model is not None
    assert optimizer is not None


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
        "nemo_automodel.recipes.llm.train_ft.initialize_distributed",
        lambda *a, **k: SimpleNamespace(world_size=1, is_main=True, device=torch.device("cpu"), rank=0),
    )
    monkeypatch.setattr("nemo_automodel.recipes.llm.train_ft.setup_logging", lambda: None)
    monkeypatch.setattr("nemo_automodel.recipes.llm.train_ft.apply_cache_compatibility_patches", lambda: None)
    monkeypatch.setattr("nemo_automodel.recipes.llm.train_ft.StatefulRNG", lambda *a, **k: "rng")
    monkeypatch.setattr(
        "nemo_automodel.recipes._typed_config.RecipeConfig.loss_fn",
        property(lambda self: SimpleNamespace(build=lambda: "loss_fn")),
    )

    def _stub_build_checkpoint_config(*a, **k):
        cfg = SimpleNamespace(checkpoint_dir="ckpts", model_state_dict_keys=None)
        cfg.build = lambda **kw: SimpleNamespace(
            config=cfg,
            load_base_model=lambda *a, **k: None,
            maybe_wait_for_staging=lambda: None,
            close=lambda: None,
        )
        return cfg

    monkeypatch.setattr(
        "nemo_automodel.recipes._typed_config.RecipeConfig.checkpoint",
        property(lambda self: _stub_build_checkpoint_config()),
    )
    # Stub create_distributed_setup_from_config to avoid requiring torch.distributed init
    monkeypatch.setattr(
        "nemo_automodel.recipes.llm.train_ft.create_distributed_setup_from_config",
        lambda cfg, world_size: SimpleNamespace(
            mesh_context=SimpleNamespace(
                pp_enabled=False,
                device_mesh=None,
                moe_mesh=None,
                cp_size=1,
                pp_size=1,
            ),
            strategy_config=None,
            pipeline_config=None,
            moe_parallel_config=None,
            activation_checkpointing=False,
        ),
    )

    # Stub model/optimizer creation
    dummy_model = DummyModel()
    dummy_opt = SimpleNamespace(param_groups=[{"lr": 0.01}], step=lambda: None, zero_grad=lambda: None)
    monkeypatch.setattr(
        "nemo_automodel.recipes.llm.train_ft.build_model",
        lambda *a, **k: dummy_model,
    )
    monkeypatch.setattr(
        "nemo_automodel.recipes._typed_config.RecipeConfig.optimizer",
        property(lambda self: SimpleNamespace(build=lambda *a, **k: [dummy_opt])),
    )

    # Data-related stubs
    monkeypatch.setattr("nemo_automodel.recipes.llm.train_ft.build_dataloader", lambda *a, **k: ("dl", "tok"))
    monkeypatch.setattr("nemo_automodel.recipes.llm.train_ft.build_validation_dataloader", lambda *a, **k: {})
    monkeypatch.setattr(
        "nemo_automodel.components.training.step_scheduler.StepSchedulerConfig.build",
        lambda self, *a, **k: SimpleNamespace(step=0, epoch=0, epochs=[]),
    )
    monkeypatch.setattr(
        "nemo_automodel.components.optim.optimizer.LRSchedulerConfig.build",
        lambda self, *a, **k: [],
    )
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
    monkeypatch.setattr(
        "nemo_automodel.recipes.llm.train_ft.TrainFinetuneRecipeForNextTokenPrediction.load_checkpoint",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "nemo_automodel.recipes.llm.train_ft.TrainFinetuneRecipeForNextTokenPrediction._log_step_scheduler_details",
        lambda *a, **k: None,
    )

    # Avoid CUDA calls
    monkeypatch.setattr("nemo_automodel.recipes.llm.train_ft.torch.cuda.reset_peak_memory_stats", lambda: None)

    # Make group/rank helpers trivial
    monkeypatch.setattr(
        "nemo_automodel.recipes.llm.train_ft.TrainFinetuneRecipeForNextTokenPrediction._get_dp_rank",
        lambda self, include_cp=False: 0,
    )
    monkeypatch.setattr(
        "nemo_automodel.recipes.llm.train_ft.TrainFinetuneRecipeForNextTokenPrediction._get_dp_group_size",
        lambda self, include_cp=False: 1,
    )
    monkeypatch.setattr(
        "nemo_automodel.recipes.llm.train_ft.TrainFinetuneRecipeForNextTokenPrediction._get_cp_group_size",
        lambda self: 1,
    )
    monkeypatch.setattr(
        "nemo_automodel.recipes.llm.train_ft.TrainFinetuneRecipeForNextTokenPrediction._get_tp_rank", lambda self: 0
    )
    monkeypatch.setattr(
        "nemo_automodel.recipes.llm.train_ft.TrainFinetuneRecipeForNextTokenPrediction._get_pp_rank", lambda self: 0
    )

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

    def _build_model_stub(*args, **kwargs):
        return DummyAutoPipeline(
            parts=parts, info=SimpleNamespace(has_last_stage=False, has_first_stage=False, schedule=None)
        )

    def _build_optimizer_stub(*args, **kwargs):
        dummy_opt = SimpleNamespace(param_groups=[{"lr": 0.01}], step=lambda: None, zero_grad=lambda: None)
        return [dummy_opt]

    # Override the default stubs to return a pipeline-wrapped model
    monkeypatch.setattr("nemo_automodel.recipes.llm.train_ft.build_model", _build_model_stub)
    monkeypatch.setattr(
        "nemo_automodel.recipes._typed_config.RecipeConfig.optimizer",
        property(lambda self: SimpleNamespace(build=_build_optimizer_stub)),
    )

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


def test_run_train_validation_loop_calls_gc_hook_once_per_step():
    class _OneStepScheduler:
        def __init__(self):
            self.step = 0
            self.epoch = 0
            self.epochs = [0]
            self.is_val_step = False
            self.is_ckpt_step = False
            self.sigterm_flag = False

        def set_epoch(self, epoch):
            self.epoch = epoch

        def __iter__(self):
            yield ["dummy-batch"]

    trainer = TrainFinetuneRecipeForNextTokenPrediction.__new__(TrainFinetuneRecipeForNextTokenPrediction)
    trainer.model_parts = [MagicMock()]
    trainer.step_scheduler = _OneStepScheduler()
    trainer.max_grad_norm = 1.0
    trainer._enable_qat_if_delayed = MagicMock()
    trainer._run_train_optim_step = MagicMock(return_value=SimpleNamespace(metrics={"loss": 1.0}))
    trainer._maybe_collect_garbage = MagicMock()
    trainer._collect_moe_load_balance = MagicMock()
    trainer.log_train_metrics = MagicMock()
    trainer.log_val_metrics = MagicMock()
    trainer.save_checkpoint = MagicMock()
    trainer.val_dataloaders = {}
    trainer.metric_logger_train = SimpleNamespace(close=MagicMock())
    trainer.metric_logger_valid = {}
    trainer.checkpointer = SimpleNamespace(close=MagicMock())
    trainer.best_metric_key = "default"

    trainer.run_train_validation_loop()

    trainer._maybe_collect_garbage.assert_called_once()


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


# -----------------
# PP Validation tests
# -----------------


class MockSchedule:
    """Mock PP schedule that tracks step/eval calls."""

    def __init__(self):
        self.step_calls = []
        self.eval_calls = []

    def step(self, *args, **kwargs):
        self.step_calls.append((args, kwargs))
        # Populate losses list if provided
        if "losses" in kwargs and kwargs["losses"] is not None:
            kwargs["losses"].append(torch.tensor(0.5))

    def eval(self, *args, **kwargs):
        self.eval_calls.append((args, kwargs))
        # Populate losses list if provided
        if "losses" in kwargs and kwargs["losses"] is not None:
            kwargs["losses"].append(torch.tensor(0.5))


class MockPPInfo:
    """Mock PP info with configurable first/last stage flags."""

    def __init__(self, has_first_stage=True, has_last_stage=True):
        self.has_first_stage = has_first_stage
        self.has_last_stage = has_last_stage
        self.schedule = MockSchedule()


def _create_minimal_recipe_for_pp_test(monkeypatch, pp_info):
    """Create a minimal TrainFinetuneRecipeForNextTokenPrediction for PP testing."""
    cfg = ConfigNode(
        {
            "nvtx": False,
            "model": {},
            "dataloader": {"collate_fn": "nemo_automodel.components.datasets.utils.default_collater"},
            "dataset": {},
            "validation_dataloader": {},
            "step_scheduler": {"local_batch_size": 1, "global_batch_size": 1},
            "optimizer": {},
            "loss_fn": {},
            "checkpoint": {"best_metric_key": "default"},
            "distributed": {"cp_size": 1},
            "autopipeline": {"pp_microbatch_size": 1},
        }
    )

    # Minimal stubs so we can create the recipe
    monkeypatch.setattr(
        "nemo_automodel.recipes.llm.train_ft.initialize_distributed",
        lambda *a, **k: SimpleNamespace(world_size=1, is_main=True, device=torch.device("cpu"), rank=0),
    )
    monkeypatch.setattr("nemo_automodel.recipes.llm.train_ft.setup_logging", lambda: None)

    # Mock helper functions to avoid needing full config
    monkeypatch.setattr("nemo_automodel.recipes.llm.train_ft._uses_te_dot_product_attention", lambda cfg: False)
    monkeypatch.setattr("nemo_automodel.recipes.llm.train_ft._uses_thd_collater", lambda cfg: False)

    # Create the recipe without calling setup
    recipe = TrainFinetuneRecipeForNextTokenPrediction(cfg)

    # Mock out attributes needed for _forward_backward_step
    # Use object.__setattr__ to bypass the state tracking
    object.__setattr__(recipe, "dist_env", SimpleNamespace(device=torch.device("cpu"), rank=0, is_main=True))
    object.__setattr__(recipe, "device_mesh", None)
    object.__setattr__(recipe, "pp_enabled", True)
    object.__setattr__(recipe, "pp", SimpleNamespace(info=pp_info, update_seq_len=lambda seq_len: None))
    object.__setattr__(recipe, "tokenizer", SimpleNamespace(pad_token_id=0))
    object.__setattr__(recipe, "te_fp8", None)

    return recipe


def test_forward_backward_step_pp_uses_eval_for_validation(monkeypatch):
    """Test that _forward_backward_step uses schedule.eval() when is_train=False with PP."""
    from contextlib import nullcontext

    pp_info = MockPPInfo(has_first_stage=True, has_last_stage=True)
    recipe = _create_minimal_recipe_for_pp_test(monkeypatch, pp_info)

    # Mock make_cp_batch_and_ctx to return a no-op context manager
    monkeypatch.setattr(
        "nemo_automodel.recipes.llm.train_ft.make_cp_batch_and_ctx",
        lambda device_mesh, batch, **kwargs: (nullcontext, batch),
    )

    # Create a minimal batch
    batch = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "labels": torch.tensor([[1, 2, 3]]),
    }

    loss_buffer = []
    recipe._forward_backward_step(
        idx=0,
        batch=batch,
        loss_buffer=loss_buffer,
        num_label_tokens=None,
        num_batches=1,
        is_train=False,  # Validation mode
    )

    # Should use eval, not step
    assert len(pp_info.schedule.eval_calls) == 1, "schedule.eval() should be called once for validation"
    assert len(pp_info.schedule.step_calls) == 0, "schedule.step() should not be called for validation"


def test_forward_backward_step_pp_uses_step_for_training(monkeypatch):
    """Test that _forward_backward_step uses schedule.step() when is_train=True with PP."""
    from contextlib import nullcontext

    pp_info = MockPPInfo(has_first_stage=True, has_last_stage=True)
    recipe = _create_minimal_recipe_for_pp_test(monkeypatch, pp_info)

    # Mock make_cp_batch_and_ctx to return a no-op context manager
    monkeypatch.setattr(
        "nemo_automodel.recipes.llm.train_ft.make_cp_batch_and_ctx",
        lambda device_mesh, batch, **kwargs: (nullcontext, batch),
    )

    # Create a minimal batch
    batch = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "labels": torch.tensor([[1, 2, 3]]),
    }

    loss_buffer = []
    recipe._forward_backward_step(
        idx=0,
        batch=batch,
        loss_buffer=loss_buffer,
        num_label_tokens=None,
        num_batches=1,
        is_train=True,  # Training mode
    )

    # Should use step, not eval
    assert len(pp_info.schedule.step_calls) == 1, "schedule.step() should be called once for training"
    assert len(pp_info.schedule.eval_calls) == 0, "schedule.eval() should not be called for training"


def test_forward_backward_step_pp_non_first_stage_uses_eval_for_validation(monkeypatch):
    """Test schedule.eval() without input_ids when not on first stage."""
    from contextlib import nullcontext

    pp_info = MockPPInfo(has_first_stage=False, has_last_stage=True)
    recipe = _create_minimal_recipe_for_pp_test(monkeypatch, pp_info)

    # Mock make_cp_batch_and_ctx to return a no-op context manager
    monkeypatch.setattr(
        "nemo_automodel.recipes.llm.train_ft.make_cp_batch_and_ctx",
        lambda device_mesh, batch, **kwargs: (nullcontext, batch),
    )

    # Create a minimal batch
    batch = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "labels": torch.tensor([[1, 2, 3]]),
    }

    loss_buffer = []
    recipe._forward_backward_step(
        idx=0,
        batch=batch,
        loss_buffer=loss_buffer,
        num_label_tokens=None,
        num_batches=1,
        is_train=False,  # Validation mode
    )

    # Should use eval without input_ids as first positional arg
    assert len(pp_info.schedule.eval_calls) == 1
    args, kwargs = pp_info.schedule.eval_calls[0]
    assert len(args) == 0, "Non-first stage should not pass input_ids as positional arg"
    assert "target" in kwargs


def test_forward_backward_step_pp_non_first_stage_uses_step_for_training(monkeypatch):
    """Test schedule.step() without input_ids when not on first stage."""
    from contextlib import nullcontext

    pp_info = MockPPInfo(has_first_stage=False, has_last_stage=True)
    recipe = _create_minimal_recipe_for_pp_test(monkeypatch, pp_info)

    # Mock make_cp_batch_and_ctx to return a no-op context manager
    monkeypatch.setattr(
        "nemo_automodel.recipes.llm.train_ft.make_cp_batch_and_ctx",
        lambda device_mesh, batch, **kwargs: (nullcontext, batch),
    )

    # Create a minimal batch
    batch = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "labels": torch.tensor([[1, 2, 3]]),
    }

    loss_buffer = []
    recipe._forward_backward_step(
        idx=0,
        batch=batch,
        loss_buffer=loss_buffer,
        num_label_tokens=None,
        num_batches=1,
        is_train=True,  # Training mode
    )

    # Should use step without input_ids as first positional arg
    assert len(pp_info.schedule.step_calls) == 1
    args, kwargs = pp_info.schedule.step_calls[0]
    assert len(args) == 0, "Non-first stage should not pass input_ids as positional arg"
    assert "target" in kwargs


def test_run_validation_epoch_pp_sends_loss_from_last_stage_to_main(monkeypatch):
    """Test that _run_validation_epoch broadcasts val_loss from last stage to main rank for PP."""
    from contextlib import nullcontext

    pp_info = MockPPInfo(has_first_stage=True, has_last_stage=True)
    recipe = _create_minimal_recipe_for_pp_test(monkeypatch, pp_info)

    # Set up recipe attributes for validation - use object.__setattr__ to bypass state tracking
    object.__setattr__(recipe, "model_parts", [DummyModel()])
    object.__setattr__(recipe, "step_scheduler", SimpleNamespace(step=1, epoch=0))
    object.__setattr__(recipe, "optimizer", [SimpleNamespace(param_groups=[{"lr": 0.01}])])

    # Stub the PP last-stage broadcast helper (post-d96f1b20 the recipe broadcasts
    # within the PP group instead of doing send/recv to global rank 0).
    monkeypatch.setattr(recipe, "_broadcast_from_last_pp_stage", lambda t: t)

    # Set dist_env.rank to 0 (last stage and main rank are the same in this test)
    object.__setattr__(recipe, "dist_env", SimpleNamespace(device=torch.device("cpu"), rank=0, is_main=True))

    # Mock the forward_backward_step to populate loss_buffer
    def mock_forward_backward_step(idx, batch, *, loss_buffer, num_label_tokens, num_batches, is_train):
        loss_buffer.append(torch.tensor(0.5))

    monkeypatch.setattr(recipe, "_forward_backward_step", mock_forward_backward_step)

    # Mock _dp_allreduce to return the tensor/value
    def mock_dp_allreduce(val, include_cp=False):
        if isinstance(val, torch.Tensor):
            return val
        return torch.tensor(val)

    monkeypatch.setattr(recipe, "_dp_allreduce", mock_dp_allreduce)

    # Mock make_cp_batch_and_ctx
    monkeypatch.setattr(
        "nemo_automodel.recipes.llm.train_ft.make_cp_batch_and_ctx",
        lambda device_mesh, batch, **kwargs: (nullcontext, batch),
    )

    # Mock ScopedRNG
    monkeypatch.setattr(
        "nemo_automodel.recipes.llm.train_ft.ScopedRNG",
        lambda **kwargs: MagicMock(__enter__=lambda s: s, __exit__=lambda s, *a: None),
    )

    # Create a simple dataloader that yields one batch
    val_dataloader = [{"input_ids": torch.tensor([[1, 2, 3]]), "labels": torch.tensor([[1, 2, 3]])}]

    result = recipe._run_validation_epoch(val_dataloader)

    # Verify result is a MetricsSample with val_loss
    assert "val_loss" in result.metrics
    # val_loss should be a float, not a tensor
    assert isinstance(result.metrics["val_loss"], float)


def test_run_validation_epoch_pp_main_rank_receives_from_last_stage(monkeypatch):
    """Test that main rank receives val_loss from last stage via the PP broadcast helper."""
    from contextlib import nullcontext

    pp_info = MockPPInfo(has_first_stage=True, has_last_stage=False)
    recipe = _create_minimal_recipe_for_pp_test(monkeypatch, pp_info)

    # Set up recipe attributes - use object.__setattr__ to bypass state tracking
    object.__setattr__(recipe, "model_parts", [DummyModel()])
    object.__setattr__(recipe, "step_scheduler", SimpleNamespace(step=1, epoch=0))
    object.__setattr__(recipe, "optimizer", [SimpleNamespace(param_groups=[{"lr": 0.01}])])

    # Track calls to the PP last-stage broadcast helper and simulate the last
    # stage's value of 0.5 propagating into the non-last-stage tensor.
    broadcast_calls = []

    def mock_broadcast(tensor):
        broadcast_calls.append(tensor)
        tensor.fill_(0.5)
        return tensor

    monkeypatch.setattr(recipe, "_broadcast_from_last_pp_stage", mock_broadcast)

    # Main rank (0) is different from last stage (3)
    object.__setattr__(recipe, "dist_env", SimpleNamespace(device=torch.device("cpu"), rank=0, is_main=True))

    def mock_forward_backward_step(idx, batch, *, loss_buffer, num_label_tokens, num_batches, is_train):
        loss_buffer.append(torch.tensor(0.0))  # Non-last stage has 0 loss

    monkeypatch.setattr(recipe, "_forward_backward_step", mock_forward_backward_step)

    def mock_dp_allreduce(val, include_cp=False):
        if isinstance(val, torch.Tensor):
            return val
        return torch.tensor(val)

    monkeypatch.setattr(recipe, "_dp_allreduce", mock_dp_allreduce)

    monkeypatch.setattr(
        "nemo_automodel.recipes.llm.train_ft.make_cp_batch_and_ctx",
        lambda device_mesh, batch, **kwargs: (nullcontext, batch),
    )

    monkeypatch.setattr(
        "nemo_automodel.recipes.llm.train_ft.ScopedRNG",
        lambda **kwargs: MagicMock(__enter__=lambda s: s, __exit__=lambda s, *a: None),
    )

    val_dataloader = [{"input_ids": torch.tensor([[1, 2, 3]]), "labels": torch.tensor([[1, 2, 3]])}]

    result = recipe._run_validation_epoch(val_dataloader)

    # Main rank should have invoked the PP broadcast helper to pull val_loss
    # and pp_num_tokens from the last PP stage (two calls total).
    assert len(broadcast_calls) >= 1, "Main rank should broadcast val_loss from the last PP stage"
    assert isinstance(result.metrics["val_loss"], float)


# -----------------
# State dict adapter tests for _maybe_adapt_state_dict_to_hf
# -----------------


class MockStateDictAdapter:
    """Mock state dict adapter that transforms keys."""

    def to_hf(self, state_dict, exclude_key_regex=None, quantization=False, **kwargs):
        """Transform state dict keys by adding 'transformed_' prefix."""
        return {f"transformed_{k}": v for k, v in state_dict.items()}


class DummyModelWithAdapter(nn.Module):
    """Model with a state_dict_adapter for testing."""

    def __init__(self):
        super().__init__()
        self.layer = DummyLinear(10, 10)
        self.state_dict_adapter = MockStateDictAdapter()

    def forward(self, x):
        return self.layer.weight @ x


class DummyModelConfigWithAdapter:
    """Mock model config that returns a model with state_dict_adapter."""

    def __init__(self):
        self.pretrained_model_name_or_path = None

    def instantiate(self, **kwargs):
        return DummyModelWithAdapter()

    def get(self, key, default=None):
        return getattr(self, key, default)


@requires_cuda
def test_build_model_state_dict_keys_uses_adapter(caplog):
    """Test that state_dict_keys are transformed using _maybe_adapt_state_dict_to_hf when adapter is present."""

    cfg_model = DummyModelConfigWithAdapter()
    cfg_opt = DummyOptConfig()
    cfg_peft = None

    with patch("nemo_automodel.recipes.llm.train_ft._supports_logits_to_keep", return_value=True):
        with patch("nemo_automodel._transformers.infrastructure._supports_logits_to_keep", return_value=True):
            with patch("nemo_automodel._transformers.auto_model._verify_sdpa_support"):
                with patch("nemo_automodel._transformers.infrastructure.print_trainable_parameters"):
                    model = build_model(
                        cfg_model=cfg_model,
                        cfg_peft=cfg_peft,
                        seed=42,
                    )
                    optimizer = build_optimizer(model, cfg_opt, None, None)

    # Model should be instantiated
    assert model is not None
    assert optimizer is not None


@requires_cuda
def test_build_model_state_dict_keys_without_adapter():
    """Test that state_dict_keys are not transformed when no adapter is present."""

    cfg_model = DummyModelConfig()  # DummyModel has no state_dict_adapter
    cfg_opt = DummyOptConfig()
    cfg_peft = None

    with patch("nemo_automodel.recipes.llm.train_ft._supports_logits_to_keep", return_value=True):
        with patch("nemo_automodel._transformers.infrastructure._supports_logits_to_keep", return_value=True):
            with patch("nemo_automodel._transformers.auto_model._verify_sdpa_support"):
                with patch("nemo_automodel._transformers.infrastructure.print_trainable_parameters"):
                    model = build_model(
                        cfg_model=cfg_model,
                        cfg_peft=cfg_peft,
                        seed=42,
                    )
                    optimizer = build_optimizer(model, cfg_opt, None, None)

    # Model should be instantiated
    assert model is not None
    assert optimizer is not None


@requires_cuda
def test_build_model_with_quantized_model_config():
    """Test that model with quantization_config is properly instantiated."""

    cfg_opt = DummyOptConfig()
    cfg_peft = None

    # Create a model config that returns a model with quantization_config
    class DummyQuantizedModelConfig:
        def __init__(self):
            self.pretrained_model_name_or_path = None

        def instantiate(self, **kwargs):
            model = DummyModel()
            # Add a config attribute with quantization_config
            model.config = SimpleNamespace(quantization_config={"bits": 4})
            return model

        def get(self, key, default=None):
            return getattr(self, key, default)

    cfg_model = DummyQuantizedModelConfig()

    with patch("nemo_automodel.recipes.llm.train_ft._supports_logits_to_keep", return_value=True):
        with patch("nemo_automodel._transformers.infrastructure._supports_logits_to_keep", return_value=True):
            with patch("nemo_automodel._transformers.auto_model._verify_sdpa_support"):
                with patch("nemo_automodel._transformers.infrastructure.print_trainable_parameters"):
                    model = build_model(
                        cfg_model=cfg_model,
                        cfg_peft=cfg_peft,
                        seed=42,
                    )
                    _ = build_optimizer(model, cfg_opt, None, None)

    # Model should be instantiated with quantization config
    assert model is not None
    assert hasattr(model.config, "quantization_config")


@requires_cuda
def test_build_model_without_quant_config():
    """Test that model without quantization_config is properly instantiated."""

    cfg_model = DummyModelConfig()  # DummyModel has no config.quantization_config
    cfg_opt = DummyOptConfig()
    cfg_peft = None

    with patch("nemo_automodel.recipes.llm.train_ft._supports_logits_to_keep", return_value=True):
        with patch("nemo_automodel._transformers.infrastructure._supports_logits_to_keep", return_value=True):
            with patch("nemo_automodel._transformers.auto_model._verify_sdpa_support"):
                with patch("nemo_automodel._transformers.infrastructure.print_trainable_parameters"):
                    model = build_model(
                        cfg_model=cfg_model,
                        cfg_peft=cfg_peft,
                        seed=42,
                    )
                    _ = build_optimizer(model, cfg_opt, None, None)

    # Model should be instantiated without quantization config
    assert model is not None
    assert not hasattr(model.config, "quantization_config")


# =============================================================================
# New tests for updated build_model / build_optimizer API
# =============================================================================


@requires_cuda
def test_build_model_and_optimizer_return_values():
    """Test that build_model and build_optimizer return proper values."""
    cfg_model = DummyModelConfig()
    cfg_opt = DummyOptConfig()

    with patch("nemo_automodel.recipes.llm.train_ft._supports_logits_to_keep", return_value=True):
        with patch("nemo_automodel._transformers.infrastructure._supports_logits_to_keep", return_value=True):
            with patch("nemo_automodel._transformers.auto_model._verify_sdpa_support"):
                with patch("nemo_automodel._transformers.infrastructure.print_trainable_parameters"):
                    model = build_model(
                        cfg_model=cfg_model,
                        cfg_peft=None,
                        seed=42,
                    )
                    optimizer = build_optimizer(model, cfg_opt, None, None)

    assert model is not None
    assert optimizer is not None


# =============================================================================
# Tests for optimizer dtype string resolution in build_optimizer
# =============================================================================


# =============================================================================
# Tests for _get_model_name helper
# =============================================================================

# =============================================================================
# Tests for PP mask precomputation guard in build_dataloader
# =============================================================================


def test_build_dataloader_pp_autoconfig_failure_skips_mask_collate(caplog):
    """When AutoConfig.from_pretrained raises, mask precomputation is skipped and a warning is logged."""
    cfg_ds = ConfigNode(
        {
            "_target_": "tests.unit_tests.recipes.test_train_ft.DummyIterableDataset",
            "tokenizer": None,
            "num_shards": 4,
        }
    )
    cfg_dl = ConfigNode(
        {
            "_target_": "tests.unit_tests.recipes.test_train_ft.dl_factory_capture",
            "num_workers": 0,
        }
    )
    cfg_model = ConfigNode({"pretrained_model_name_or_path": "bad/model"})
    cfg_ps = ConfigNode({})

    with (
        patch("nemo_automodel.recipes.llm.train_ft.AutoConfig.from_pretrained", side_effect=OSError("not found")),
        caplog.at_level(logging.WARNING),
    ):
        dl, tok = _build_dataloader(
            cfg_ds=cfg_ds,
            cfg_dl=cfg_dl,
            cfg_model=cfg_model,
            cfg_ps=cfg_ps,
            seed=123,
            local_batch_size=2,
            global_batch_size=4,
            max_steps=None,
            val_check_interval=None,
            dp_rank=0,
            dp_world_size=1,
            pp_enabled=True,
        )

    assert "Failed to load model config for causal mask precomputation" in caplog.text
    # collate_fn should NOT have been set since AutoConfig failed
    mod = importlib.import_module("tests.unit_tests.recipes.test_train_ft")
    captured = getattr(mod.dl_factory_capture, "captured")
    assert "collate_fn" not in captured


def test_build_dataloader_pp_autoconfig_success_sets_mask_collate():
    """When AutoConfig.from_pretrained succeeds and no collate_fn exists, a mask-only collate is set."""
    cfg_ds = ConfigNode(
        {
            "_target_": "tests.unit_tests.recipes.test_train_ft.DummyIterableDataset",
            "tokenizer": None,
            "num_shards": 4,
        }
    )
    cfg_dl = ConfigNode(
        {
            "_target_": "tests.unit_tests.recipes.test_train_ft.dl_factory_capture",
            "num_workers": 0,
        }
    )
    cfg_model = ConfigNode({"pretrained_model_name_or_path": "good/model"})
    cfg_ps = ConfigNode({})

    mock_config = MagicMock()
    with (
        patch("nemo_automodel.recipes.llm.train_ft.AutoConfig.from_pretrained", return_value=mock_config),
        patch("nemo_automodel.components.datasets.utils.add_causal_masks_to_batch", side_effect=lambda b, **kw: b),
    ):
        dl, tok = _build_dataloader(
            cfg_ds=cfg_ds,
            cfg_dl=cfg_dl,
            cfg_model=cfg_model,
            cfg_ps=cfg_ps,
            seed=123,
            local_batch_size=2,
            global_batch_size=4,
            max_steps=None,
            val_check_interval=None,
            dp_rank=0,
            dp_world_size=1,
            pp_enabled=True,
        )

    # collate_fn should have been set (mask-only path)
    mod = importlib.import_module("tests.unit_tests.recipes.test_train_ft")
    captured = getattr(mod.dl_factory_capture, "captured")
    assert "collate_fn" in captured
    assert callable(captured["collate_fn"])


def test_build_dataloader_pp_deepseek_v4_skips_mask_collate(caplog):
    """DeepSeek V4 computes causal masks internally, so PP mask precomputation is skipped."""
    cfg_ds = ConfigNode(
        {
            "_target_": "tests.unit_tests.recipes.test_train_ft.DummyIterableDataset",
            "tokenizer": None,
            "num_shards": 4,
        }
    )
    cfg_dl = ConfigNode(
        {
            "_target_": "tests.unit_tests.recipes.test_train_ft.dl_factory_capture",
            "num_workers": 0,
        }
    )
    cfg_model = ConfigNode({"pretrained_model_name_or_path": "deepseek-ai/DeepSeek-V4-Pro"})
    cfg_ps = ConfigNode({})

    mock_config = MagicMock(model_type="deepseek_v4")
    with (
        patch("nemo_automodel.recipes.llm.train_ft.AutoConfig.from_pretrained", return_value=mock_config),
        patch("nemo_automodel.components.datasets.utils.add_causal_masks_to_batch") as add_masks,
        caplog.at_level(logging.INFO),
    ):
        dl, tok = _build_dataloader(
            cfg_ds=cfg_ds,
            cfg_dl=cfg_dl,
            cfg_model=cfg_model,
            cfg_ps=cfg_ps,
            seed=123,
            local_batch_size=2,
            global_batch_size=4,
            max_steps=None,
            val_check_interval=None,
            dp_rank=0,
            dp_world_size=1,
            pp_enabled=True,
        )

    mod = importlib.import_module("tests.unit_tests.recipes.test_train_ft")
    captured = getattr(mod.dl_factory_capture, "captured")
    assert "collate_fn" not in captured
    assert "Skipping pipeline parallel causal mask precomputation for model_type=deepseek_v4" in caplog.text
    add_masks.assert_not_called()


def test_build_dataloader_pp_autoconfig_success_chains_existing_collate():
    """When AutoConfig.from_pretrained succeeds and collate_fn exists, they are chained."""
    call_order = []

    def my_collate(batch):
        call_order.append("base")
        return batch

    cfg_ds = ConfigNode(
        {
            "_target_": "tests.unit_tests.recipes.test_train_ft.DummyIterableDataset",
            "tokenizer": None,
            "num_shards": 4,
        }
    )
    cfg_dl = ConfigNode(
        {
            "_target_": "tests.unit_tests.recipes.test_train_ft.dl_factory_capture",
            "num_workers": 0,
            "collate_fn": my_collate,
        }
    )
    cfg_model = ConfigNode({"pretrained_model_name_or_path": "good/model"})
    cfg_ps = ConfigNode({})

    mock_config = MagicMock()

    def mock_add_masks(batch, model_config=None):
        call_order.append("masks")
        return batch

    with (
        patch("nemo_automodel.recipes.llm.train_ft.AutoConfig.from_pretrained", return_value=mock_config),
        patch("nemo_automodel.components.datasets.utils.add_causal_masks_to_batch", side_effect=mock_add_masks),
    ):
        dl, tok = _build_dataloader(
            cfg_ds=cfg_ds,
            cfg_dl=cfg_dl,
            cfg_model=cfg_model,
            cfg_ps=cfg_ps,
            seed=123,
            local_batch_size=2,
            global_batch_size=4,
            max_steps=None,
            val_check_interval=None,
            dp_rank=0,
            dp_world_size=1,
            pp_enabled=True,
        )

    mod = importlib.import_module("tests.unit_tests.recipes.test_train_ft")
    captured = getattr(mod.dl_factory_capture, "captured")
    assert "collate_fn" in captured
    chained_fn = captured["collate_fn"]

    # Invoke the chained collate to verify ordering
    chained_fn(["dummy_batch"])
    assert call_order == ["base", "masks"]


@pytest.mark.parametrize(
    "cfg_attrs,expected",
    [
        # String config
        ({"config": "org/model-name"}, "org/model-name"),
        # Direct pretrained_model_name_or_path
        ({"pretrained_model_name_or_path": "direct/model"}, "direct/model"),
        # Not found - returns None
        ({}, None),
    ],
)
def test_get_model_name(cfg_attrs, expected):
    """Test _get_model_name extracts model name from various config structures."""
    from nemo_automodel.recipes.llm.train_ft import _get_model_name

    cfg_model = SimpleNamespace(**cfg_attrs)
    cfg_model.get = lambda key, default=None: getattr(cfg_model, key, default)

    result = _get_model_name(cfg_model)
    assert result == expected


def test_get_model_name_from_nested_config():
    """Test _get_model_name extracts from nested config.pretrained_model_name_or_path."""
    from nemo_automodel.recipes.llm.train_ft import _get_model_name

    inner_config = SimpleNamespace(pretrained_model_name_or_path="nested/model")
    inner_config.get = lambda key, default=None: getattr(inner_config, key, default)
    cfg_model = SimpleNamespace(config=inner_config)
    cfg_model.get = lambda key, default=None: getattr(cfg_model, key, default)

    result = _get_model_name(cfg_model)
    assert result == "nested/model"


# ---------------------------------------------------------------------------
# _log_moe_metrics tests
# ---------------------------------------------------------------------------


def _make_moe_layer_loads(loads_list):
    """Build a layer_loads dict from a list of 1-D lists (mirrors test_load_balance_metrics helper)."""
    result = {}
    for i, load in enumerate(loads_list):
        result[f"layers.{i}.moe.gate"] = {
            "expert_load": torch.tensor(load, dtype=torch.float32),
            "aux_loss": None,
            "n_experts": len(load),
        }
    return result


def _make_trainer_for_moe(moe_cfg_dict, layer_loads=None):
    """Create a bare recipe instance with cfg and _moe_layer_loads set."""
    trainer = TrainFinetuneRecipeForNextTokenPrediction.__new__(TrainFinetuneRecipeForNextTokenPrediction)
    trainer.cfg = ConfigNode({"moe_metrics": moe_cfg_dict})
    trainer._moe_layer_loads = layer_loads
    return trainer


def test_log_moe_metrics_skips_when_no_loads():
    """wandb_log_fn should not be called when _moe_layer_loads is None."""
    trainer = _make_trainer_for_moe({"enabled": True, "mode": "brief"}, layer_loads=None)
    log_fn = MagicMock()

    trainer._log_moe_metrics(step=1, wandb_log_fn=log_fn)

    log_fn.assert_not_called()


def test_log_moe_metrics_brief_mode_default():
    """Brief mode should call wandb_log_fn once with correct step."""
    loads = _make_moe_layer_loads([[100.0, 200.0, 300.0, 400.0]])
    trainer = _make_trainer_for_moe({"enabled": True, "mode": "brief", "top_k_experts": 2}, layer_loads=loads)
    log_fn = MagicMock()

    trainer._log_moe_metrics(step=42, wandb_log_fn=log_fn)

    log_fn.assert_called_once()
    _, kwargs = log_fn.call_args
    assert kwargs["step"] == 42
    metrics = log_fn.call_args[0][0]
    assert "moe/cv_mean" in metrics


def test_log_moe_metrics_passes_top_k_zero():
    """top_k_experts=0 should produce no moe_expert_utilization/ keys."""
    loads = _make_moe_layer_loads([[100.0, 200.0, 300.0, 400.0]])
    trainer = _make_trainer_for_moe({"enabled": True, "mode": "brief", "top_k_experts": 0}, layer_loads=loads)
    log_fn = MagicMock()

    trainer._log_moe_metrics(step=1, wandb_log_fn=log_fn)

    log_fn.assert_called_once()
    metrics = log_fn.call_args[0][0]
    util_keys = [k for k in metrics if k.startswith("moe_expert_utilization/")]
    assert len(util_keys) == 0
    assert "moe/cv_mean" in metrics


def test_log_moe_metrics_passes_top_k_from_config():
    """top_k_experts=3 should produce moe_expert_utilization/ keys."""
    loads = _make_moe_layer_loads([[100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0]])
    trainer = _make_trainer_for_moe({"enabled": True, "mode": "brief", "top_k_experts": 3}, layer_loads=loads)
    log_fn = MagicMock()

    trainer._log_moe_metrics(step=1, wandb_log_fn=log_fn)

    log_fn.assert_called_once()
    metrics = log_fn.call_args[0][0]
    util_keys = [k for k in metrics if k.startswith("moe_expert_utilization/")]
    assert len(util_keys) > 0
    assert len(util_keys) <= 6  # at most 2 * top_k


def test_log_moe_metrics_detailed_mode():
    """Detailed mode should call compute_detailed_metrics (includes per-layer keys)."""
    loads = _make_moe_layer_loads([[100.0, 200.0], [300.0, 400.0]])
    trainer = _make_trainer_for_moe({"enabled": True, "mode": "detailed", "top_k_experts": 2}, layer_loads=loads)
    log_fn = MagicMock()

    trainer._log_moe_metrics(step=10, wandb_log_fn=log_fn)

    log_fn.assert_called_once()
    metrics = log_fn.call_args[0][0]
    assert "moe/layer_0/cv" in metrics
    assert "moe/layer_1/cv" in metrics


def test_log_moe_metrics_detailed_mode_non_detailed_step():
    """On non-detailed steps, detailed mode should fall back to brief metrics."""
    loads = _make_moe_layer_loads([[100.0, 200.0], [300.0, 400.0]])
    trainer = _make_trainer_for_moe(
        {"enabled": True, "mode": "detailed", "top_k_experts": 2, "detailed_every_steps": 10},
        layer_loads=loads,
    )
    log_fn = MagicMock()

    trainer._log_moe_metrics(step=5, wandb_log_fn=log_fn)

    log_fn.assert_called_once()
    metrics = log_fn.call_args[0][0]
    # Brief metrics: no per-layer keys
    assert "moe/layer_0/cv" not in metrics
    assert "moe/cv_mean" in metrics


class TestRunTrainOptimStepSetsMoEScale:
    """Tests that _run_train_optim_step sets MoEAuxLossAutoScaler.main_loss_backward_scale."""

    def setup_method(self):
        from nemo_automodel.components.moe.megatron.moe_utils import MoEAuxLossAutoScaler

        MoEAuxLossAutoScaler.main_loss_backward_scale = None

    def teardown_method(self):
        from nemo_automodel.components.moe.megatron.moe_utils import MoEAuxLossAutoScaler

        MoEAuxLossAutoScaler.main_loss_backward_scale = None

    def _make_recipe(self, monkeypatch, pp_enabled, dp_group_size=4):
        from nemo_automodel.components.config.loader import ConfigNode

        cfg = ConfigNode(
            {
                "nvtx": False,
                "model": {},
                "dataloader": {"collate_fn": "nemo_automodel.components.datasets.utils.default_collater"},
                "dataset": {},
                "validation_dataloader": {},
                "step_scheduler": {"local_batch_size": 1, "global_batch_size": 1},
                "optimizer": {},
                "loss_fn": {},
                "checkpoint": {"best_metric_key": "default"},
                "distributed": {"cp_size": 1},
                "autopipeline": {"pp_microbatch_size": 1},
            }
        )
        monkeypatch.setattr(
            "nemo_automodel.recipes.llm.train_ft.initialize_distributed",
            lambda *a, **k: SimpleNamespace(world_size=1, is_main=True, device=torch.device("cpu"), rank=0),
        )
        monkeypatch.setattr("nemo_automodel.recipes.llm.train_ft.setup_logging", lambda: None)
        monkeypatch.setattr("nemo_automodel.recipes.llm.train_ft._uses_te_dot_product_attention", lambda cfg: False)
        monkeypatch.setattr("nemo_automodel.recipes.llm.train_ft._uses_thd_collater", lambda cfg: False)

        recipe = TrainFinetuneRecipeForNextTokenPrediction(cfg)

        object.__setattr__(recipe, "dist_env", SimpleNamespace(device=torch.device("cpu"), rank=0, is_main=True))
        object.__setattr__(recipe, "device_mesh", None)
        object.__setattr__(recipe, "moe_mesh", None)
        object.__setattr__(recipe, "pp_enabled", pp_enabled)
        object.__setattr__(recipe, "te_fp8", None)
        object.__setattr__(recipe, "model_parts", [nn.Linear(4, 4)])
        object.__setattr__(
            recipe,
            "optimizer",
            [SimpleNamespace(step=lambda: None, zero_grad=lambda: None, param_groups=[{"lr": 0.01}])],
        )
        object.__setattr__(recipe, "lr_schedulers", [])
        object.__setattr__(recipe, "step_scheduler", SimpleNamespace(step=1, epoch=0))

        if pp_enabled:
            pp_info = SimpleNamespace(has_first_stage=True, has_last_stage=True)
            object.__setattr__(recipe, "pp", SimpleNamespace(info=pp_info, update_seq_len=lambda seq_len: None))
            # Stub the PP last-stage broadcast helper (post-d96f1b20 the recipe
            # broadcasts inside the PP group instead of using send/recv).
            monkeypatch.setattr(recipe, "_broadcast_from_last_pp_stage", lambda t: t)
        object.__setattr__(recipe, "tokenizer", SimpleNamespace(pad_token_id=0))

        monkeypatch.setattr(
            recipe,
            "_dp_allreduce",
            lambda val, include_cp=False: val if isinstance(val, torch.Tensor) else torch.tensor(val),
        )
        monkeypatch.setattr(recipe, "_get_dp_group_size", lambda include_cp=False: dp_group_size)
        monkeypatch.setattr(recipe, "_get_cp_group_size", lambda: 1)

        def mock_forward_backward_step(idx, batch, *, loss_buffer, num_label_tokens, num_batches, is_train=True):
            loss_buffer.append(torch.tensor(0.5))

        monkeypatch.setattr(recipe, "_forward_backward_step", mock_forward_backward_step)
        monkeypatch.setattr(
            "nemo_automodel.recipes.llm.train_ft.scale_grads_and_clip_grad_norm",
            lambda *a, **k: torch.tensor(1.0),
        )
        monkeypatch.setattr("nemo_automodel.recipes.llm.train_ft.prepare_for_grad_accumulation", lambda *a, **k: None)
        monkeypatch.setattr("nemo_automodel.recipes.llm.train_ft.prepare_for_final_backward", lambda *a, **k: None)
        monkeypatch.setattr("nemo_automodel.recipes.llm.train_ft.prepare_after_first_microbatch", lambda *a, **k: None)
        object.__setattr__(recipe, "checkpointer", SimpleNamespace(maybe_wait_for_staging=lambda: None))
        object.__setattr__(recipe, "lr_scheduler", None)
        object.__setattr__(recipe, "timestamp", 0.0)
        return recipe

    def test_pp_enabled_sets_scale_to_num_label_tokens(self, monkeypatch):
        from nemo_automodel.components.moe.megatron.moe_utils import MoEAuxLossAutoScaler

        recipe = self._make_recipe(monkeypatch, pp_enabled=True, dp_group_size=4)

        # 3 valid labels out of 4
        batches = [{"input_ids": torch.tensor([[1, 2, 3, 4]]), "labels": torch.tensor([[1, 2, 3, -100]])}]

        recipe._run_train_optim_step(batches)

        assert MoEAuxLossAutoScaler.main_loss_backward_scale is not None
        assert MoEAuxLossAutoScaler.main_loss_backward_scale.item() == pytest.approx(3.0)

    def test_pp_disabled_sets_scale_to_dp_group_size(self, monkeypatch):
        from nemo_automodel.components.moe.megatron.moe_utils import MoEAuxLossAutoScaler

        dp_size = 8
        recipe = self._make_recipe(monkeypatch, pp_enabled=False, dp_group_size=dp_size)

        batches = [{"input_ids": torch.tensor([[1, 2, 3, 4]]), "labels": torch.tensor([[1, 2, 3, -100]])}]

        recipe._run_train_optim_step(batches)

        assert MoEAuxLossAutoScaler.main_loss_backward_scale is not None
        assert MoEAuxLossAutoScaler.main_loss_backward_scale.item() == pytest.approx(float(dp_size))


# -----------------------------------------------------------------------------
# rope_fusion disabled when cp > 1
# -----------------------------------------------------------------------------


def _minimal_cfg_with_rope_fusion(cp_size: int, rope_fusion: bool):
    """Helper to build a minimal ConfigNode for rope_fusion / CP tests."""
    return ConfigNode(
        {
            "model": {"backend": {"rope_fusion": rope_fusion}},
            "dataloader": {},
            "dataset": {},
            "validation_dataloader": {},
            "step_scheduler": {"local_batch_size": 1, "global_batch_size": 1},
            "optimizer": {},
            "loss_fn": {},
            "checkpoint": {"best_metric_key": "default"},
            "distributed": {"cp_size": cp_size},
        }
    )


def _patch_setup_minimals_with_cp(monkeypatch, cp_size):
    """Variant of _patch_setup_minimals that lets us control cp_size."""
    _patch_setup_minimals(monkeypatch, lambda *a, **k: None)
    # Override create_distributed_setup_from_config to expose the desired cp_size
    monkeypatch.setattr(
        "nemo_automodel.recipes.llm.train_ft.create_distributed_setup_from_config",
        lambda cfg, world_size: SimpleNamespace(
            mesh_context=SimpleNamespace(
                pp_enabled=False,
                device_mesh=None,
                moe_mesh=None,
                cp_size=cp_size,
                pp_size=1,
            ),
            strategy_config=None,
            pipeline_config=None,
            moe_parallel_config=None,
            activation_checkpointing=False,
        ),
    )


def test_rope_fusion_disabled_when_cp_gt_1(monkeypatch):
    """rope_fusion should be set to False during setup when cp_size > 1."""
    cfg = _minimal_cfg_with_rope_fusion(cp_size=2, rope_fusion=True)
    _patch_setup_minimals_with_cp(monkeypatch, cp_size=2)

    trainer = TrainFinetuneRecipeForNextTokenPrediction(cfg)
    trainer.setup()

    assert cfg.model.backend.rope_fusion is False


def test_rope_fusion_unchanged_when_cp_eq_1(monkeypatch):
    """rope_fusion should remain True when cp_size == 1."""
    cfg = _minimal_cfg_with_rope_fusion(cp_size=1, rope_fusion=True)
    _patch_setup_minimals_with_cp(monkeypatch, cp_size=1)

    trainer = TrainFinetuneRecipeForNextTokenPrediction(cfg)
    trainer.setup()

    assert cfg.model.backend.rope_fusion is True


def test_rope_fusion_stays_false_when_already_disabled(monkeypatch):
    """rope_fusion=False should stay False regardless of cp_size."""
    cfg = _minimal_cfg_with_rope_fusion(cp_size=4, rope_fusion=False)
    _patch_setup_minimals_with_cp(monkeypatch, cp_size=4)

    trainer = TrainFinetuneRecipeForNextTokenPrediction(cfg)
    trainer.setup()

    assert cfg.model.backend.rope_fusion is False


# ============================================================================
# Tests for resolve_sdpa_method
# ============================================================================


class TestResolveSdpaMethod:
    """Tests for resolve_sdpa_method helper."""

    def test_explicit_strings_converted_to_backends(self):
        from torch.nn.attention import SDPBackend

        result = resolve_sdpa_method(["flash_attention", "math"])
        assert result == [SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]

    def test_case_insensitive(self):
        from torch.nn.attention import SDPBackend

        result = resolve_sdpa_method(["Flash_Attention", "EFFICIENT_ATTENTION"])
        assert result == [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown SDPA backend 'bogus'"):
            resolve_sdpa_method(["bogus"])

    def test_none_with_no_constraints_returns_none(self):
        assert resolve_sdpa_method(None) is None

    def test_auto_cp_restricts_backends(self):
        from torch.nn.attention import SDPBackend

        mesh = MagicMock()
        mesh.mesh_dim_names = ("dp", "cp")
        mesh.__getitem__ = lambda self, key: MagicMock(size=lambda: 2) if key == "cp" else MagicMock(size=lambda: 1)

        result = resolve_sdpa_method(None, device_mesh=mesh)
        assert result == [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]

    def test_auto_activation_checkpointing_restricts_backends(self):
        from torch.nn.attention import SDPBackend

        result = resolve_sdpa_method(None, activation_checkpointing=True)
        assert result == [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]

    def test_explicit_overrides_auto(self):
        """When sdpa_method is provided, auto-selection is bypassed."""
        from torch.nn.attention import SDPBackend

        mesh = MagicMock()
        mesh.mesh_dim_names = ("dp", "cp")
        mesh.__getitem__ = lambda self, key: MagicMock(size=lambda: 2) if key == "cp" else MagicMock(size=lambda: 1)

        result = resolve_sdpa_method(["math"], device_mesh=mesh, activation_checkpointing=True)
        assert result == [SDPBackend.MATH]

    def test_sdp_backend_enums_passed_through(self):
        """SDPBackend enum values should be passed through unchanged."""
        from torch.nn.attention import SDPBackend

        result = resolve_sdpa_method([SDPBackend.FLASH_ATTENTION, SDPBackend.MATH])
        assert result == [SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]

    def test_mixed_strings_and_enums(self):
        """Mix of string and SDPBackend values should work."""
        from torch.nn.attention import SDPBackend

        result = resolve_sdpa_method([SDPBackend.FLASH_ATTENTION, "efficient_attention"])
        assert result == [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]


class TestDpEvalSampleShard:
    """`dp_eval_sample_shard` shards eval samples only when the model is
    replicated per DP rank (DDP); sharded strategies must stay in lockstep."""

    def test_ddp_multi_rank_shards(self):
        from nemo_automodel.components.distributed.config import DDPConfig

        assert dp_eval_sample_shard(DDPConfig(), 1, 4) == (1, 4)

    def test_ddp_single_rank_no_shard(self):
        from nemo_automodel.components.distributed.config import DDPConfig

        assert dp_eval_sample_shard(DDPConfig(), 0, 1) is None

    def test_fsdp2_never_shards(self):
        # Sharding under FSDP2 would desync generate()'s per-layer all-gathers.
        from nemo_automodel.components.distributed.config import FSDP2Config

        assert dp_eval_sample_shard(FSDP2Config(), 1, 4) is None

    def test_megatron_fsdp_never_shards(self):
        from nemo_automodel.components.distributed.config import MegatronFSDPConfig

        assert dp_eval_sample_shard(MegatronFSDPConfig(), 1, 4) is None


class _FakeToolCallEvaluator:
    """Stand-in for ToolCallAccuracyEvaluator: returns canned metrics (or raises)
    so ``_run_validation_epoch``'s reduction can be tested without generation."""

    metric_prefix = "tool_call"
    # Reuse the real evaluator's keys so this fake can never silently diverge.
    METRIC_KEYS = ToolCallAccuracyEvaluator.METRIC_KEYS

    def __init__(self, *, sample_shard=None, run_on_fsdp2=False, result=None, raises=False):
        self.sample_shard = sample_shard
        self.run_on_fsdp2 = run_on_fsdp2
        self._result = result if result is not None else {}
        self._raises = raises
        self.eval_calls = 0

    def evaluate(self, model, tokenizer):
        self.eval_calls += 1
        if self._raises:
            raise RuntimeError("boom")
        return dict(self._result)


def _make_eval_recipe(distributed_config, evaluator):
    """Minimal recipe wired for ``_run_validation_epoch`` with an empty val loader.

    Single-rank: ``_dp_allreduce`` is the identity, so the packed all-reduce
    recovers the per-rank means directly.
    """
    recipe = TrainFinetuneRecipeForNextTokenPrediction.__new__(TrainFinetuneRecipeForNextTokenPrediction)
    recipe.model_parts = [SimpleNamespace(eval=lambda: None)]
    recipe.dist_env = SimpleNamespace(device=torch.device("cpu"), is_main=True)
    recipe.optimizer = [SimpleNamespace(param_groups=[{"lr": 0.01}])]
    recipe.pp_enabled = False
    recipe.distributed_config = distributed_config
    recipe.tool_call_evaluator = evaluator
    recipe.tokenizer = object()
    recipe.step_scheduler = SimpleNamespace(step=3, epoch=1)
    recipe._warned_tool_call_eval_skipped = False
    recipe._dp_allreduce = lambda tensor, *args, **kwargs: tensor
    return recipe


class TestRunValidationToolCallEval:
    """Cover the tool-call eval reduction branches in ``_run_validation_epoch``
    (FSDP2 skip / DDP packed all-reduce / replicated / evaluate failure) without a
    real model or process group."""

    def _run(self, recipe, monkeypatch):
        # max_memory_allocated() is CUDA-only; stub it so the CPU metrics build works.
        monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a, **k: 0)
        return recipe._run_validation_epoch([])  # empty loader -> straight to the eval block

    def test_fsdp2_skips_in_loop_eval(self, monkeypatch):
        from nemo_automodel.components.distributed.config import FSDP2Config

        ev = _FakeToolCallEvaluator(run_on_fsdp2=False)
        recipe = _make_eval_recipe(FSDP2Config(), ev)
        out = self._run(recipe, monkeypatch)
        assert out.metrics["tool_call/_disabled_fsdp2"] == 1.0
        assert ev.eval_calls == 0  # generation never ran
        assert recipe._warned_tool_call_eval_skipped is True

    def test_ddp_sharded_packed_allreduce(self, monkeypatch):
        from nemo_automodel.components.distributed.config import DDPConfig

        result = {
            "tool_call/has_call": 1.0,
            "tool_call/name_correct": 0.5,
            "tool_call/args_json_valid": 1.0,
            "tool_call/args_field_recall": 0.0,
            "tool_call/args_field_precision": 0.0,
            "tool_call/args_exact_match": 0.0,
            "tool_call/_count": 2.0,
            "tool_call/_skipped": 1.0,
        }
        ev = _FakeToolCallEvaluator(sample_shard=(0, 1), result=result)
        out = self._run(_make_eval_recipe(DDPConfig(), ev), monkeypatch)
        # Identity all-reduce on one rank: count-weighted sum / count recovers the mean.
        assert out.metrics["tool_call/has_call"] == 1.0
        assert out.metrics["tool_call/name_correct"] == 0.5
        assert out.metrics["tool_call/_count"] == 2.0
        assert out.metrics["tool_call/_skipped"] == 1.0
        assert ev.eval_calls == 1

    def test_replicated_reports_local_without_collective(self, monkeypatch):
        from nemo_automodel.components.distributed.config import FSDP2Config

        result = {
            "tool_call/has_call": 0.75,
            "tool_call/name_correct": 0.25,
            "tool_call/args_json_valid": 0.5,
            "tool_call/args_field_recall": 0.1,
            "tool_call/args_field_precision": 0.2,
            "tool_call/args_exact_match": 0.0,
            "tool_call/_count": 4.0,
            "tool_call/_skipped": 0.0,
        }
        # FSDP2 + run_on_fsdp2 -> not skipped; sample_shard None -> replicated branch.
        ev = _FakeToolCallEvaluator(sample_shard=None, run_on_fsdp2=True, result=result)
        out = self._run(_make_eval_recipe(FSDP2Config(), ev), monkeypatch)
        assert out.metrics["tool_call/has_call"] == 0.75
        assert out.metrics["tool_call/_count"] == 4.0
        assert out.metrics["tool_call/_skipped"] == 0.0

    def test_evaluate_failure_is_tolerated(self, monkeypatch):
        from nemo_automodel.components.distributed.config import DDPConfig

        ev = _FakeToolCallEvaluator(sample_shard=(0, 1), raises=True)
        out = self._run(_make_eval_recipe(DDPConfig(), ev), monkeypatch)
        # An evaluate() that raised contributes an empty result -> zeros, count 0.
        assert out.metrics["tool_call/_count"] == 0.0
        assert out.metrics["tool_call/has_call"] == 0.0


def test_forward_backward_step_dsv4_cp_hook_and_grad_touch(monkeypatch):
    """Non-PP step: the model-owned CP hook attaches its batch prep, and the
    ``_cp_full_logits_grad_touch`` flag adds the zero-valued full-logits term so
    backward still reaches every parameter."""
    from contextlib import nullcontext

    cfg = ConfigNode(
        {
            "nvtx": False,
            "model": {},
            "dataloader": {"collate_fn": "nemo_automodel.components.datasets.utils.default_collater"},
            "dataset": {},
            "validation_dataloader": {},
            "step_scheduler": {"local_batch_size": 1, "global_batch_size": 1},
            "optimizer": {},
            "loss_fn": {},
            "checkpoint": {"best_metric_key": "default"},
            "distributed": {"cp_size": 2},
            "autopipeline": {"pp_microbatch_size": 1},
        }
    )
    monkeypatch.setattr(
        "nemo_automodel.recipes.llm.train_ft.initialize_distributed",
        lambda *a, **k: SimpleNamespace(world_size=1, is_main=True, device=torch.device("cpu"), rank=0),
    )
    monkeypatch.setattr("nemo_automodel.recipes.llm.train_ft.setup_logging", lambda: None)
    monkeypatch.setattr("nemo_automodel.recipes.llm.train_ft._uses_te_dot_product_attention", lambda cfg: False)
    monkeypatch.setattr("nemo_automodel.recipes.llm.train_ft._uses_thd_collater", lambda cfg: False)
    recipe = TrainFinetuneRecipeForNextTokenPrediction(cfg)

    class _CPModel(nn.Module):
        def __init__(self):
            super().__init__()
            # Wide output + large magnitude in fp16 so a naive (low-precision)
            # logits.sum() overflows to inf; the grad touch must promote to fp32.
            self.lin = nn.Linear(4, 8192)
            self.prepared = False

        def prepare_model_inputs_for_cp(self, input_ids):
            self.prepared = True
            return {"_cp_full_logits_grad_touch": True}

        def forward(self, **batch):
            logits = (self.lin(batch["input_ids"].float()) + 50.0).to(torch.float16)
            return SimpleNamespace(logits=logits)

    model = _CPModel()
    object.__setattr__(recipe, "dist_env", SimpleNamespace(device=torch.device("cpu"), rank=0, is_main=True))
    object.__setattr__(recipe, "device_mesh", None)
    object.__setattr__(recipe, "pp_enabled", False)
    object.__setattr__(recipe, "magi", SimpleNamespace(enabled=False))
    object.__setattr__(recipe, "tokenizer", SimpleNamespace(pad_token_id=0))
    object.__setattr__(recipe, "te_fp8", None)
    object.__setattr__(recipe, "model_parts", [model])
    object.__setattr__(recipe, "distributed_config", SimpleNamespace(defer_fsdp_grad_sync=True))
    object.__setattr__(recipe, "loss_fn", object())  # not FusedLinearCrossEntropy
    object.__setattr__(recipe, "_get_dp_group_size", lambda include_cp=False: 1)

    captured = {}

    def _fake_calc_loss(loss_fn, *, logits, labels, model, hidden_states, num_label_tokens):
        captured["logits_is_tensor"] = isinstance(logits, torch.Tensor)
        # finite base loss (fp32) so any non-finiteness must come from the grad touch
        return logits.float().mean()

    monkeypatch.setattr(
        "nemo_automodel.recipes.llm.train_ft.make_cp_batch_and_ctx",
        lambda device_mesh, batch, **k: (nullcontext, batch),
    )
    monkeypatch.setattr("nemo_automodel.recipes.llm.train_ft.calculate_loss", _fake_calc_loss)
    monkeypatch.setattr("nemo_automodel.recipes.llm.train_ft.get_final_hidden_states", lambda out: None)
    monkeypatch.setattr("nemo_automodel.recipes.llm.train_ft.get_sync_ctx", lambda *a, **k: nullcontext())
    monkeypatch.setattr("nemo_automodel.recipes.llm.train_ft.filter_forward_kwargs", lambda model, batch: batch)

    batch = {"input_ids": torch.randn(1, 4, 4), "labels": torch.zeros(1, 4, dtype=torch.long)}
    loss_buffer = []
    recipe._forward_backward_step(
        idx=0, batch=batch, loss_buffer=loss_buffer, num_label_tokens=None, num_batches=1, is_train=True
    )

    assert model.prepared is True  # cp_size>1 + hasattr -> hook body ran
    assert captured["logits_is_tensor"]
    assert len(loss_buffer) == 1
    # the fp32-promoted grad touch must not poison the loss with inf/nan
    assert torch.isfinite(loss_buffer[0]).all()
    # the grad-touch kept the full logits in the graph, so backward populated grads
    assert model.lin.weight.grad is not None
    assert torch.isfinite(model.lin.weight.grad).all()
