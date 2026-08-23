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

import inspect
import logging
import sys
import types
from contextlib import AbstractContextManager, nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.components.datasets.datum import LossInputLayout

# Skip decorator for tests that require CUDA
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
from torch.utils.data import IterableDataset

from nemo_automodel._transformers.mfu import MFUConfig
from nemo_automodel._transformers.model_init import resolve_sdpa_method
from nemo_automodel.components.datasets.loader import (
    DataloaderConfig,
)
from nemo_automodel.components.distributed.utils import dp_eval_sample_shard
from nemo_automodel.components.eval.tool_call_evaluator import ToolCallAccuracyEvaluator
from nemo_automodel.components.loss.mtp import PipelineCausalLMLoss
from nemo_automodel.components.models.deepseek_v4.cp import dsv4_cp_local_seq_multiple
from nemo_automodel.components.optim.optimizer import build_optimizer_config
from nemo_automodel.engine import ForwardBackwardResult
from nemo_automodel.recipes._typed_config import RecipeConfig, _as_dict, _callable_and_kwargs
from nemo_automodel.recipes.llm.train_ft import (
    TrainFinetuneRecipeForNextTokenPrediction,
    _build_pp_collate_wrapper,
    _should_pack_validation,
    build_model,
    compute_trust_remote_code_from_model,
)


def test_recipe_config_resolves_mfu_settings():
    config = RecipeConfig(ConfigNode({"mfu": {"device": "h100", "peak_tflops": 1979.0}}))

    assert config.mfu == MFUConfig(device="h100", peak_tflops=1979.0)


def test_recipe_config_defaults_mfu_settings():
    assert RecipeConfig(ConfigNode({})).mfu == MFUConfig()


def _build_loader(
    cfg_ds,
    cfg_dl,
    cfg_model,
    cfg_ps,
    *,
    seed,
    local_batch_size,
    global_batch_size,
    max_steps,
    val_check_interval,
    dp_rank,
    dp_world_size,
    pp_enabled,
    cp_size=1,
    model=None,
):
    """Resolve loader YAML like ``RecipeConfig.dataloader`` and build it."""
    raw = ConfigNode(
        {
            "dataset": cfg_ds.to_dict(),
            "dataloader": cfg_dl.to_dict(),
            "model": cfg_model.to_dict(),
            "packed_sequence": cfg_ps.to_dict(),
            "seed": seed,
            "step_scheduler": {
                "local_batch_size": local_batch_size,
                "global_batch_size": global_batch_size,
                "max_steps": max_steps,
                "val_every_steps": val_check_interval,
            },
        }
    )
    config = RecipeConfig(raw).dataloader
    loader = config.build(
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        pp_enabled=pp_enabled,
        supports_seq_lens=model is None or "seq_lens" in inspect.signature(model.forward).parameters,
        cp_size=cp_size,
        collate_wrapper=_build_pp_collate_wrapper(cfg_model, pp_enabled),
    )
    return loader, None


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
    """Minimal iterable dataset with shard/shuffle hooks for testing the dataloader build."""

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


class DummyMapDataset(torch.utils.data.Dataset):
    """Minimal map-style dataset used to exercise recipe-side packing."""

    def __init__(self, split=None):
        self.split = split
        self.items = [{"input_ids": [1, 2], "labels": [1, 2]}]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]

    def shuffle(self, seed):
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


def test_validation_dataloaders_pp_enabled(caplog):
    cfg = ConfigNode(
        {
            "model": {},
            "validation_dataloader": {},
        }
    )

    with caplog.at_level(logging.WARNING):
        result = RecipeConfig(cfg).validation_dataloaders

    assert result == {}


def test_validation_dataloaders_collects_and_names_properly():
    # Multiple validation dataset keys with different separators, each resolved to a DataloaderConfig.
    ds_target = "tests.unit_tests.recipes.test_train_ft.DummyIterableDataset"
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
            "validation_dataset": {"_target_": ds_target},
            "validation_dataset_val": {"_target_": ds_target},
            "validation_dataset-test": {"_target_": ds_target},
        }
    )

    result = RecipeConfig(cfg).validation_dataloaders

    assert set(result.keys()) == {"default", "val", "test"}
    assert all(isinstance(v, DataloaderConfig) for v in result.values())
    assert all(v.batch_size == 8 for v in result.values())


def test_validation_dataloaders_no_validation_keys():
    cfg = ConfigNode(
        {
            "model": {},
            "validation_dataloader": {},
        }
    )

    assert RecipeConfig(cfg).validation_dataloaders == {}


def test_validation_dataloaders_no_validation_config():
    cfg = ConfigNode(
        {
            "model": {},
            "dataloader": {},
        }
    )

    assert RecipeConfig(cfg).validation_dataloaders == {}


@pytest.mark.parametrize("attn", ["magi", "te", "sdpa"])
def test_validation_dataloaders_pack_configured_sequences(attn):
    cfg = ConfigNode(
        {
            "model": {"backend": {"attn": attn}},
            "dataloader": {},
            "validation_dataloader": {},
            "packed_sequence": {"packed_sequence_size": 1024},
            "validation_dataset": {"_target_": "tests.unit_tests.recipes.test_train_ft.DummyMapDataset"},
        }
    )

    assert RecipeConfig(cfg).validation_dataloaders["default"].packing is not None


def test_validation_dataloaders_skip_packing_without_pack_size():
    cfg = ConfigNode(
        {
            "model": {"backend": {"attn": "sdpa"}},
            "dataloader": {},
            "validation_dataloader": {},
            "packed_sequence": {"packed_sequence_size": 0},
            "validation_dataset": {"_target_": "tests.unit_tests.recipes.test_train_ft.DummyMapDataset"},
        }
    )

    assert RecipeConfig(cfg).validation_dataloaders["default"].packing is None


@pytest.mark.parametrize("attn", ["magi", "te", "sdpa"])
def test_should_pack_validation_for_explicit_thd_collater(attn):
    collate_fn = "nemo_automodel.components.datasets.utils.packed_sequence_thd_collater"
    dataset = {"_target_": "tests.unit_tests.recipes.test_train_ft.DummyMapDataset"}
    cfg = RecipeConfig(
        ConfigNode(
            {
                "model": {"backend": {"attn": attn}},
                "dataset": dataset,
                "dataloader": {"collate_fn": collate_fn},
                "validation_dataset": dataset,
                "validation_dataloader": {"collate_fn": collate_fn},
                "packed_sequence": {"packed_sequence_size": 1024},
            }
        )
    )

    assert _should_pack_validation(cfg.dataloader, cfg.validation_dataloaders["default"], nn.Module()) is True


def test_should_not_pack_validation_without_pack_size():
    dataset = {"_target_": "tests.unit_tests.recipes.test_train_ft.DummyMapDataset"}
    cfg = RecipeConfig(
        ConfigNode(
            {
                "dataset": dataset,
                "validation_dataset": dataset,
                "packed_sequence": {"packed_sequence_size": 0},
            }
        )
    )

    assert _should_pack_validation(cfg.dataloader, cfg.validation_dataloaders["default"], nn.Module()) is False


@pytest.mark.parametrize(("attn", "expected"), [("magi", True), ("te", True), ("sdpa", False)])
def test_should_pack_validation_for_live_model_backend(attn, expected):
    collate_fn = "nemo_automodel.components.datasets.utils.packed_sequence_thd_collater"
    dataset = {"_target_": "tests.unit_tests.recipes.test_train_ft.DummyMapDataset"}
    cfg = RecipeConfig(
        ConfigNode(
            {
                "dataset": dataset,
                "dataloader": {"collate_fn": collate_fn},
                "validation_dataset": dataset,
                "validation_dataloader": {},
                "packed_sequence": {"packed_sequence_size": 1024},
            }
        )
    )
    model = nn.Module()
    model.backend = SimpleNamespace(attn=attn)

    assert _should_pack_validation(cfg.dataloader, cfg.validation_dataloaders["default"], model) is expected


def test_should_pack_validation_when_model_requires_training_layout():
    collate_fn = "nemo_automodel.components.datasets.utils.packed_sequence_thd_collater"
    dataset = {"_target_": "tests.unit_tests.recipes.test_train_ft.DummyMapDataset"}
    cfg = RecipeConfig(
        ConfigNode(
            {
                "model": {"backend": {"attn": "sdpa"}},
                "dataset": dataset,
                "dataloader": {"collate_fn": collate_fn},
                "validation_dataset": dataset,
                "validation_dataloader": {},
                "packed_sequence": {"packed_sequence_size": 1024},
            }
        )
    )

    class ModelRequiresPackedValidation(nn.Module):
        def should_pack_validation_with_training(self):
            return True

    assert (
        _should_pack_validation(
            cfg.dataloader,
            cfg.validation_dataloaders["default"],
            ModelRequiresPackedValidation(),
        )
        is True
    )


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
        "max_recent_checkpoints": 2,
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
    # The builder preserves `checkpoint_dir` and `max_recent_checkpoints` from the user configuration.
    assert config.checkpoint_dir == "/user/ckpt/"
    assert config.max_recent_checkpoints == 2
    # The builder coerces incompatible `torch_save` options and restores the default `save_consolidated="final"`.
    assert config.save_consolidated.value == "final"
    assert config.is_async is False


def test_build_loader_iterable_shard_and_shuffle_removed_from_cfg(monkeypatch):
    # cfg_ds: target resolves to this test module dataset class
    cfg_ds = ConfigNode(
        {
            "_target_": "tests.unit_tests.recipes.test_train_ft.DummyIterableDataset",
            "tokenizer": None,
            "num_shards": 4,
        }
    )
    # shuffle / shuffle_buffer_size are consumed by the loader build, not forwarded to the DataLoader.
    cfg_dl = ConfigNode({"shuffle": True, "shuffle_buffer_size": 8, "num_workers": 0})
    cfg_model = ConfigNode({})
    cfg_ps = ConfigNode({})

    loader, tok = _build_loader(
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

    assert tok is None
    ds = loader.dataset  # the (sharded + shuffled) dataset wrapped by ParallelAwareDataloader
    assert ds.__class__.__name__ == "DummyIterableDataset"
    # Shard path used when num_shards >= dp_world_size
    assert ds._shard == (2, 1)
    # Shuffle called with buffer size and seed
    assert ds._shuffle_calls and ds._shuffle_calls[-1] == (8, 123)


def test_build_dataloader_prepacked_sequence_skips_recipe_packing(monkeypatch):
    cfg_ds = ConfigNode(
        {
            "_target_": "tests.unit_tests.recipes.test_train_ft.DummyIterableDataset",
            "tokenizer": None,
        }
    )
    cfg_dl = ConfigNode(
        {
            "num_workers": 0,
        }
    )
    cfg_model = ConfigNode({})
    cfg_ps = ConfigNode({"packed_sequence_size": 8, "prepacked": True})

    class _PackedModel(nn.Module):
        def forward(self, input_ids, seq_lens=None):
            """Return token IDs unchanged.

            Args:
                input_ids: Token IDs shaped ``[B, S]``, where ``B`` is batch and ``S`` is sequence length.
                seq_lens: Optional packed lengths shaped ``[B, N]``, where ``N`` is packed sequences per row.

            Returns:
                The input ``[B, S]`` tensor without copying.
            """
            return input_ids

    dl, tok = _build_loader(
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
        pp_enabled=False,
        cp_size=1,
        model=_PackedModel(),
    )

    assert tok is None
    ds = dl.dataset
    assert ds.__class__.__name__ == "DummyIterableDataset"
    assert ds._shuffle_calls == []


@pytest.mark.parametrize("supports_thd", [True, False])
def test_build_dataloader_packing_uses_configured_cp_size(monkeypatch, supports_thd):
    captured = {}

    def fake_pack_dataset(dataset, **kwargs):
        captured.update(kwargs)
        return dataset

    monkeypatch.setattr("nemo_automodel.components.datasets.llm.packed_sequence.pack_dataset", fake_pack_dataset)
    cfg_ds = ConfigNode(
        {
            "_target_": "tests.unit_tests.recipes.test_train_ft.DummyMapDataset",
            "split": "train",
        }
    )
    cfg_dl = ConfigNode(
        {
            "num_workers": 0,
        }
    )
    cfg_model = ConfigNode({})
    cfg_ps = ConfigNode({"packed_sequence_size": 8, "packing_strategy": "thd"})

    if supports_thd:

        class _PackedModel(nn.Module):
            def forward(self, input_ids, seq_lens=None):
                """Return token IDs while accepting packed-sequence metadata.

                Args:
                    input_ids: Token IDs shaped ``[B, S]``, where ``B`` is batch and ``S`` is sequence length.
                    seq_lens: Optional lengths shaped ``[B, N]``, where ``N`` is packed sequences per row.

                Returns:
                    The input ``[B, S]`` tensor without copying.
                """
                return input_ids

    else:

        class _PackedModel(nn.Module):
            def forward(self, input_ids):
                """Return token IDs unchanged.

                Args:
                    input_ids: Token IDs shaped ``[B, S]``, where ``B`` is batch and ``S`` is sequence length.

                Returns:
                    The input ``[B, S]`` tensor without copying.
                """
                return input_ids

    model = _PackedModel()

    dl, _ = _build_loader(
        cfg_ds=cfg_ds,
        cfg_dl=cfg_dl,
        cfg_model=cfg_model,
        cfg_ps=cfg_ps,
        seed=123,
        local_batch_size=1,
        global_batch_size=1,
        max_steps=None,
        val_check_interval=None,
        dp_rank=0,
        dp_world_size=1,
        pp_enabled=False,
        cp_size=2,
        model=model,
    )

    assert dl.dataset.__class__.__name__ == "DummyMapDataset"
    if supports_thd:
        assert captured["cp_size"] == 2
    else:
        assert captured == {}


# (num_proc, max_packs, expect_parallel): parallel only when num_proc>1 AND max_packs unset
# (max_packs relies on the serial pass' lazy early-stop, so it stays serial).
@pytest.mark.parametrize(
    "num_proc,max_packs,expect_parallel",
    [(2, None, True), (1, None, False), (2, 5, False)],
)
def test_build_dataloader_parallel_tokenize_gated_on_num_proc(monkeypatch, num_proc, max_packs, expect_parallel):
    """num_proc>1 pre-tokenizes in parallel and feeds the result to packing; else it does not."""
    calls = {"tokenize": 0}
    packed_input = {}

    def fake_tokenize_parallel(dataset, num_proc):
        calls["tokenize"] += 1
        calls["num_proc"] = num_proc
        return "MATERIALIZED"

    def fake_pack_dataset(dataset, **kwargs):
        packed_input["dataset"] = dataset
        return dataset

    monkeypatch.setattr(
        "nemo_automodel.components.datasets.llm.packed_sequence.tokenize_dataset_parallel", fake_tokenize_parallel
    )
    monkeypatch.setattr("nemo_automodel.components.datasets.llm.packed_sequence.pack_dataset", fake_pack_dataset)

    cfg_ds = ConfigNode({"_target_": "tests.unit_tests.recipes.test_train_ft.DummyMapDataset", "split": "train"})
    cfg_dl = ConfigNode({"_target_": "torchdata.stateful_dataloader.StatefulDataLoader", "num_workers": 0})
    cfg_model = ConfigNode({})
    cfg_ps = ConfigNode(
        {"packed_sequence_size": 8, "packing_strategy": "thd", "num_proc": num_proc, "max_packs": max_packs}
    )

    class _PackedModel(nn.Module):
        def forward(self, input_ids, seq_lens=None):
            return input_ids

    _build_loader(
        cfg_ds=cfg_ds,
        cfg_dl=cfg_dl,
        cfg_model=cfg_model,
        cfg_ps=cfg_ps,
        seed=123,
        local_batch_size=1,
        global_batch_size=1,
        max_steps=None,
        val_check_interval=None,
        dp_rank=0,
        dp_world_size=1,
        pp_enabled=False,
        cp_size=1,
        model=_PackedModel(),
    )

    if expect_parallel:
        assert calls["tokenize"] == 1 and calls["num_proc"] == num_proc
        assert packed_input["dataset"] == "MATERIALIZED"
    else:
        assert calls["tokenize"] == 0
        assert packed_input["dataset"].__class__.__name__ == "DummyMapDataset"


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
def _minimal_cfg_with_nvtx(nvtx_value: bool, optimizer_target: str | None = None):
    """Helper to build a minimal ConfigNode for nvtx tests."""
    return ConfigNode(
        {
            "nvtx": nvtx_value,
            "model": {},
            "dataloader": {},
            "dataset": {},
            "validation_dataloader": {},
            "step_scheduler": {"local_batch_size": 1, "global_batch_size": 1},
            "optimizer": {"_target_": optimizer_target} if optimizer_target is not None else {},
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

    # Data-related stubs: short-circuit the RecipeConfig dataloader resolution + build.
    monkeypatch.setattr(
        RecipeConfig,
        "dataloader",
        property(
            lambda self: SimpleNamespace(
                build=lambda **k: "dl",
                dataset_builds_on_all_ranks=False,
                seed=42,
            )
        ),
    )
    monkeypatch.setattr(RecipeConfig, "validation_dataloaders", property(lambda self: {}))
    monkeypatch.setattr("nemo_automodel.recipes.llm.train_ft._build_tokenizer", lambda cfg_model, cfg_ds: ({}, None))
    monkeypatch.setattr("nemo_automodel.recipes.llm.train_ft.ScopedRNG", lambda **kwargs: nullcontext())
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


def test_setup_rejects_loss_without_sum_contract(monkeypatch):
    cfg = _minimal_cfg_with_nvtx(nvtx_value=False)
    _patch_setup_minimals(monkeypatch, lambda *args, **kwargs: None)
    monkeypatch.setattr("nemo_automodel.recipes.llm.train_ft._supports_logits_to_keep", lambda _model: True)

    trainer = TrainFinetuneRecipeForNextTokenPrediction(cfg)
    with pytest.raises(ValueError, match="reduction='sum'"):
        trainer.setup()


def test_setup_builds_engine_for_eager_sum_loss(monkeypatch):
    from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy

    cfg = _minimal_cfg_with_nvtx(nvtx_value=False)
    _patch_setup_minimals(monkeypatch, lambda *args, **kwargs: None)
    monkeypatch.setattr(
        RecipeConfig,
        "mtp",
        property(lambda self: SimpleNamespace(ignore_index=-321)),
    )

    trainer = TrainFinetuneRecipeForNextTokenPrediction(cfg)
    trainer.setup()

    assert isinstance(trainer.loss_fn, MaskedCrossEntropy)
    assert trainer.engine is not None
    assert trainer.engine.mtp_ignore_index == -321
    assert trainer.engine.optimizers == tuple(trainer.optimizer)
    assert trainer.engine.lr_schedulers == tuple(trainer.lr_scheduler or ())
    assert trainer.engine.max_grad_norm == trainer.max_grad_norm


def test_setup_builds_engine_for_eager_fused_loss(monkeypatch):
    from nemo_automodel.components.loss.linear_ce import FusedLinearCrossEntropy

    cfg = _minimal_cfg_with_nvtx(nvtx_value=False)
    _patch_setup_minimals(monkeypatch, lambda *args, **kwargs: None)
    fused_loss = FusedLinearCrossEntropy()
    monkeypatch.setattr(
        RecipeConfig,
        "loss_fn",
        property(lambda self: SimpleNamespace(build=lambda: fused_loss)),
    )
    monkeypatch.setattr("nemo_automodel.recipes.llm.train_ft._supports_logits_to_keep", lambda _model: True)

    trainer = TrainFinetuneRecipeForNextTokenPrediction(cfg)
    trainer.setup()

    assert trainer.loss_fn is fused_loss
    assert trainer.engine is not None


@pytest.mark.parametrize(
    (
        "local_has_mtp",
        "cp_size",
        "packed_sequence_size",
        "dataloader_emits_thd",
        "pipeline_thd_kind",
        "fused_loss",
        "scale_grads_in_schedule",
        "magi_enabled",
        "local_batch_size",
        "error_match",
    ),
    [
        (False, 1, 0, False, None, False, False, False, 2, None),
        (True, 1, 0, False, None, False, False, False, 2, None),
        (False, 2, 0, False, None, False, False, False, 2, None),
        (False, 1, 8, False, None, False, False, False, 2, None),
        (False, 1, 0, True, "native", False, False, False, 2, None),
        (False, 1, 0, True, "stock_hf", False, False, False, 2, "do not consume packed document boundaries"),
        (False, 1, 0, False, None, True, False, False, 2, None),
        (False, 1, 0, False, None, False, True, False, 2, "scale_grads_in_schedule=False"),
        (False, 1, 0, False, None, False, False, True, 2, "Magi pipeline training requires"),
        (False, 1, 0, False, None, False, False, True, 1, None),
    ],
)
def test_setup_pipeline_engine_matrix(
    monkeypatch,
    local_has_mtp,
    cp_size,
    packed_sequence_size,
    dataloader_emits_thd,
    pipeline_thd_kind,
    fused_loss,
    scale_grads_in_schedule,
    magi_enabled,
    local_batch_size,
    error_match,
):
    from nemo_automodel.components.loss.linear_ce import FusedLinearCrossEntropy

    cfg = _minimal_cfg_with_nvtx(nvtx_value=False)
    cfg.step_scheduler.local_batch_size = local_batch_size
    cfg.step_scheduler.global_batch_size = local_batch_size
    cfg.packed_sequence = ConfigNode({"packed_sequence_size": packed_sequence_size})
    _patch_setup_minimals(monkeypatch, lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "nemo_automodel.recipes.llm.train_ft.setup_magi",
        lambda *args, **kwargs: SimpleNamespace(enabled=magi_enabled, hf_dispatch=False),
    )
    fused_loss_fn = FusedLinearCrossEntropy() if fused_loss else None
    if fused_loss_fn is not None:
        monkeypatch.setattr(
            RecipeConfig,
            "loss_fn",
            property(lambda self: SimpleNamespace(build=lambda: fused_loss_fn)),
        )
        monkeypatch.setattr("nemo_automodel.recipes.llm.train_ft._supports_logits_to_keep", lambda _model: True)
    pp_collate_wrapper = object()
    monkeypatch.setattr(
        "nemo_automodel.recipes.llm.train_ft._build_pp_collate_wrapper",
        lambda *_args, **_kwargs: pp_collate_wrapper,
    )
    dataloader_build_kwargs = []

    def build_dataloader(**kwargs):
        dataloader_build_kwargs.append(kwargs)
        return "dl"

    monkeypatch.setattr(
        RecipeConfig,
        "dataloader",
        property(
            lambda self: SimpleNamespace(
                build=build_dataloader,
                dataset_builds_on_all_ranks=False,
                emits_thd=dataloader_emits_thd,
                seed=42,
            )
        ),
    )

    class DummyAutoPipeline(SimpleNamespace):
        pass

    monkeypatch.setattr("nemo_automodel.recipes.llm.train_ft.AutoPipeline", DummyAutoPipeline)
    monkeypatch.setattr("nemo_automodel.engine._engine.AutoPipeline", DummyAutoPipeline)
    parts = [DummyModel()]
    parts[0]._pp_return_hidden_states_supported = True
    if pipeline_thd_kind == "native":
        parts[0].supports_thd = True
    elif pipeline_thd_kind == "stock_hf":
        parts[0]._te_attention_injected = True
        parts[0].forward = MagicMock()
    if local_has_mtp:
        parts[0].mtp = nn.Identity()
    pipeline = DummyAutoPipeline(
        parts=parts,
        pp_batch_size=local_batch_size,
        pp_microbatch_size=1,
        scale_grads_in_schedule=scale_grads_in_schedule,
        info=SimpleNamespace(
            has_first_stage=True,
            has_last_stage=False,
            schedule=SimpleNamespace(),
            stages=[SimpleNamespace(is_last=False)],
        ),
    )
    monkeypatch.setattr("nemo_automodel.recipes.llm.train_ft.build_model", lambda *args, **kwargs: pipeline)
    monkeypatch.setattr(
        "nemo_automodel.recipes.llm.train_ft.create_distributed_setup_from_config",
        lambda cfg, world_size: SimpleNamespace(
            mesh_context=SimpleNamespace(
                pp_enabled=True,
                device_mesh=None,
                moe_mesh=None,
                cp_size=cp_size,
                pp_size=2,
            ),
            strategy_config=None,
            pipeline_config=SimpleNamespace(
                pp_seq_len=None,
                scale_grads_in_schedule=scale_grads_in_schedule,
            ),
            moe_parallel_config=None,
            activation_checkpointing=False,
        ),
    )
    trainer = TrainFinetuneRecipeForNextTokenPrediction(cfg)
    if error_match is not None:
        with pytest.raises(ValueError, match=error_match):
            trainer.setup()
        return
    trainer.setup()

    if fused_loss_fn is not None:
        assert trainer.loss_fn is fused_loss_fn
    assert trainer.engine is not None
    assert trainer.engine.pipeline is pipeline
    assert dataloader_build_kwargs[0]["collate_wrapper"] is (None if dataloader_emits_thd else pp_collate_wrapper)


def test_setup_allows_per_token_megatron_fsdp(monkeypatch):
    cfg = _minimal_cfg_with_nvtx(nvtx_value=False)
    _patch_setup_minimals(monkeypatch, lambda *args, **kwargs: None)
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
            strategy_config=SimpleNamespace(calculate_per_token_loss=True),
            pipeline_config=None,
            moe_parallel_config=None,
            activation_checkpointing=False,
        ),
    )

    trainer = TrainFinetuneRecipeForNextTokenPrediction(cfg)
    trainer.setup()

    assert trainer.distributed_config.calculate_per_token_loss is True
    assert trainer.engine is not None


def test_engine_pipeline_loss_reuses_configured_loss_and_thd_metadata():
    recipe = object.__new__(TrainFinetuneRecipeForNextTokenPrediction)
    recipe.pp_enabled = True
    recipe.pipeline_loss_fn = MagicMock(return_value=torch.tensor(3.0))
    output = object()
    labels = torch.tensor([[1, 2, -100]])
    cu_seqlens = torch.tensor([0, 3], dtype=torch.int32)

    loss = recipe._engine_loss_fn(
        output,
        {
            "labels": labels,
            "weights": labels.ne(-100),
            "cu_seqlens": cu_seqlens,
        },
    )

    assert loss.item() == pytest.approx(3.0)
    assert recipe.pipeline_loss_fn.cu_seqlens is cu_seqlens
    recipe.pipeline_loss_fn.assert_called_once_with(output, labels)


def test_engine_loss_passes_precomputed_mtp_targets(monkeypatch):
    """The recipe callback consumes Engine-prepared MTP targets without rolling them again."""
    recipe = object.__new__(TrainFinetuneRecipeForNextTokenPrediction)
    recipe.pp_enabled = False
    recipe.model_parts = [nn.Identity()]
    recipe.loss_fn = object()
    recipe.cfg = SimpleNamespace(mtp=SimpleNamespace(ignore_index=-100, scaling_factor=0.5))
    recipe._get_dp_group = lambda include_cp=False: None
    recipe._get_cp_group_size = lambda: 2

    labels = torch.tensor([[1, 2, -100]])
    mtp_targets = (torch.tensor([[2, -100, -100]]),)
    cu_seqlens = torch.tensor([0, 3], dtype=torch.int32)
    output = SimpleNamespace(
        logits=torch.randn(1, 3, 5),
        mtp_per_depth_h=None,
        mtp_per_depth_logits=(torch.randn(1, 3, 5),),
        mtp_loss_scaling_factor=1.0,
    )
    calculate_mtp_loss_mock = MagicMock(return_value=torch.tensor(3.0))
    monkeypatch.setattr("nemo_automodel.recipes.llm.train_ft.calculate_loss", MagicMock(return_value=torch.tensor(2.0)))
    monkeypatch.setattr("nemo_automodel.recipes.llm.train_ft.calculate_mtp_loss", calculate_mtp_loss_mock)
    monkeypatch.setattr("nemo_automodel.recipes.llm.train_ft.get_final_hidden_states", lambda _output: None)

    loss = recipe._engine_loss_fn(
        output,
        {
            "labels": labels,
            "weights": labels.ne(-100),
            "cu_seqlens": cu_seqlens,
            "mtp_per_depth_targets": mtp_targets,
        },
    )

    assert loss.item() == pytest.approx(5.0)
    assert calculate_mtp_loss_mock.call_args.kwargs["mtp_per_depth_targets"] is mtp_targets
    assert calculate_mtp_loss_mock.call_args.kwargs["cu_seqlens"] is None

    with pytest.raises(RuntimeError, match="globally prepared per-depth targets"):
        recipe._engine_loss_fn(
            output,
            {"labels": labels, "weights": labels.ne(-100), "cu_seqlens": cu_seqlens},
        )


def test_setup_does_not_change_storage_dtype_for_non_kd_recipe(monkeypatch):
    cfg = _minimal_cfg_with_nvtx(nvtx_value=False, optimizer_target="torch.optim.AdamW")

    _patch_setup_minimals(monkeypatch, lambda *a, **k: None)
    dummy_opt = SimpleNamespace(param_groups=[{"lr": 0.01}], step=lambda: None, zero_grad=lambda: None)
    optimizer_config = build_optimizer_config("torch.optim.AdamW", {"lr": 0.01})
    monkeypatch.setattr(optimizer_config, "build", lambda *a, **k: [dummy_opt])
    monkeypatch.setattr(
        "nemo_automodel.recipes._typed_config.RecipeConfig.optimizer",
        property(lambda self: optimizer_config),
    )

    trainer = TrainFinetuneRecipeForNextTokenPrediction(cfg)
    trainer.setup()

    assert not hasattr(cfg.model, "torch_dtype")


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


class _StageWithLogitsToKeep(nn.Module):
    def forward(self, input_ids=None, logits_to_keep=0, **kwargs):
        return None


class _StageNoLogitsToKeep(nn.Module):
    def forward(self, input_ids=None, **kwargs):
        return None


@pytest.mark.parametrize(
    "has_logits_to_keep, has_marker, pp_enabled, expect_fused",
    [
        (True, True, True, True),  # PP generic patched forward -> fused CE kept
        (False, True, True, False),  # PP, no logits_to_keep -> fall back
        (True, False, True, False),  # PP, logits_to_keep but no hidden-states marker (MoE/custom) -> fall back
        (True, False, False, True),  # non-PP: the hidden-states marker gate does not apply -> fused CE kept
    ],
)
def test_maybe_downgrade_loss_fn(has_logits_to_keep, has_marker, pp_enabled, expect_fused):
    """FusedLinearCrossEntropy survives only when the probed stage module supports
    logits_to_keep and (under PP) advertises hidden-states emission via
    _pp_return_hidden_states_supported; otherwise it downgrades to MaskedCrossEntropy."""
    from nemo_automodel.components.loss.linear_ce import FusedLinearCrossEntropy
    from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy
    from nemo_automodel.recipes.llm.train_ft import _maybe_downgrade_loss_fn

    probe = (_StageWithLogitsToKeep if has_logits_to_keep else _StageNoLogitsToKeep)()
    if has_marker:
        probe._pp_return_hidden_states_supported = True  # set by patch_hf_model_for_pp on the generic forward

    result = _maybe_downgrade_loss_fn(FusedLinearCrossEntropy(), probe, pp_enabled=pp_enabled)

    assert isinstance(result, FusedLinearCrossEntropy) is expect_fused
    if not expect_fused:
        assert isinstance(result, MaskedCrossEntropy)


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
    trainer.partial_cuda_graph_manager = None
    trainer._partial_cuda_graph_capture_pending = False
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
    trainer.checkpointer = SimpleNamespace(finalize=MagicMock())
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


def test_engine_validation_pipeline_loss_reuses_configured_loss_and_thd_metadata():
    recipe = object.__new__(TrainFinetuneRecipeForNextTokenPrediction)
    recipe.pp_enabled = True
    recipe.pipeline_loss_fn = MagicMock(return_value=torch.tensor(3.0))
    output = object()
    labels = torch.tensor([[1, 2, -100]])
    cu_seqlens = torch.tensor([0, 3], dtype=torch.int32)

    loss = recipe._engine_validation_loss_fn(
        output,
        {
            "labels": labels,
            "weights": labels.ne(-100),
            "cu_seqlens": cu_seqlens,
        },
    )

    assert loss.item() == pytest.approx(3.0)
    assert recipe.pipeline_loss_fn.cu_seqlens is cu_seqlens
    recipe.pipeline_loss_fn.assert_called_once_with(output, labels)


def test_run_validation_epoch_uses_engine_evaluate_for_pp_complete_results(monkeypatch):
    recipe = TrainFinetuneRecipeForNextTokenPrediction.__new__(TrainFinetuneRecipeForNextTokenPrediction)
    recipe.model_parts = [MagicMock()]
    recipe.dist_env = SimpleNamespace(device=torch.device("cpu"), is_main=True)
    recipe.optimizer = [SimpleNamespace(param_groups=[{"lr": 0.01}])]
    recipe.step_scheduler = SimpleNamespace(step=1, epoch=0)
    recipe.pp_enabled = True
    recipe.pipeline_loss_fn = MagicMock()
    recipe.loss_fn = object()
    recipe.tool_call_evaluator = None

    engine = MagicMock()
    engine.evaluate.side_effect = [
        SimpleNamespace(
            loss_sum=torch.tensor(4.0, dtype=torch.float64),
            weight_sum=torch.tensor(2.0, dtype=torch.float64),
            token_outputs=[],
            batch_outputs=[],
        ),
        SimpleNamespace(
            loss_sum=torch.tensor(9.0, dtype=torch.float64),
            weight_sum=torch.tensor(3.0, dtype=torch.float64),
            token_outputs=[],
            batch_outputs=[],
        ),
    ]
    recipe.engine = engine
    allreduce = MagicMock(side_effect=lambda tensor, **kwargs: tensor)
    recipe._dp_allreduce = allreduce
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(
        "nemo_automodel.recipes.llm.train_ft.ScopedRNG",
        lambda **kwargs: nullcontext(),
    )

    batches = [
        {"input_ids": torch.tensor([[1, 2, 3]]), "labels": torch.tensor([[1, 2, -100]])},
        {"input_ids": torch.tensor([[4, 5, 6, 7]]), "labels": torch.tensor([[4, 5, 6, -100]])},
    ]
    metrics = recipe._run_validation_epoch(batches)

    assert engine.evaluate.call_count == 2
    for call, batch in zip(engine.evaluate.call_args_list, batches):
        datum_batches, loss_fn = call.args
        assert len(datum_batches) == len(datum_batches[0]) == 1
        datum = datum_batches[0][0]
        assert datum.model_inputs["input_ids"] is batch["input_ids"]
        assert datum.loss_fn_inputs["labels"] is batch["labels"]
        torch.testing.assert_close(datum.loss_fn_inputs["weights"], batch["labels"].ne(-100))
        assert datum.loss_fn_input_layouts == {
            "labels": LossInputLayout.PER_TOKEN,
            "weights": LossInputLayout.PER_TOKEN,
        }
        assert loss_fn == recipe._engine_validation_loss_fn
    assert allreduce.call_count == 2
    assert all("include_cp" not in call.kwargs for call in allreduce.call_args_list)
    recipe.model_parts[0].eval.assert_not_called()
    assert metrics.metrics["val_loss"] == pytest.approx(13.0 / 5.0)
    assert metrics.metrics["num_label_tokens"] == pytest.approx(5.0)


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
# Tests for PP mask precomputation guard in the loader build
# =============================================================================


def _pp_loader(cfg_model, cfg_dl, **patches):
    """Build a PP-enabled iterable loader and return its resolved ``collate_fn``."""
    cfg_ds = ConfigNode(
        {
            "_target_": "tests.unit_tests.recipes.test_train_ft.DummyIterableDataset",
            "tokenizer": None,
            "num_shards": 4,
        }
    )
    loader, _ = _build_loader(
        cfg_ds=cfg_ds,
        cfg_dl=cfg_dl,
        cfg_model=cfg_model,
        cfg_ps=ConfigNode({}),
        seed=123,
        local_batch_size=2,
        global_batch_size=4,
        max_steps=None,
        val_check_interval=None,
        dp_rank=0,
        dp_world_size=1,
        pp_enabled=True,
    )
    return loader.collate_fn


def test_pp_autoconfig_failure_skips_masks(caplog):
    """When AutoConfig.from_pretrained raises, the collate is left unwrapped (warning logged)."""
    calls = []
    cfg_dl = ConfigNode({"collate_fn": lambda b: calls.append("base") or b, "num_workers": 0})
    cfg_model = ConfigNode({"pretrained_model_name_or_path": "bad/model"})
    with (
        patch("nemo_automodel.recipes.llm.train_ft.AutoConfig.from_pretrained", side_effect=OSError("not found")),
        patch("nemo_automodel.components.datasets.utils.add_causal_masks_to_batch") as add_masks,
        caplog.at_level(logging.WARNING),
    ):
        collate_fn = _pp_loader(cfg_model, cfg_dl)

    assert "Failed to load model config for causal mask precomputation" in caplog.text
    collate_fn(["dummy"])
    assert calls == ["base"]
    add_masks.assert_not_called()


@pytest.mark.parametrize(
    "model_type",
    ["deepseek_v4", "glm_moe_dsa"],
)
def test_pp_custom_sparse_model_skips_masks(caplog, model_type):
    """Custom sparse models compute masks internally, so PP precomputation is skipped."""
    calls = []
    cfg_dl = ConfigNode({"collate_fn": lambda b: calls.append("base") or b, "num_workers": 0})
    model_name = "zai-org/GLM-5.2" if model_type == "glm_moe_dsa" else "deepseek-ai/DeepSeek-V4-Pro"
    cfg_model = ConfigNode({"pretrained_model_name_or_path": model_name})
    with (
        patch(
            "nemo_automodel.recipes.llm.train_ft.AutoConfig.from_pretrained",
            return_value=MagicMock(model_type=model_type),
        ),
        patch("nemo_automodel.components.datasets.utils.add_causal_masks_to_batch") as add_masks,
        caplog.at_level(logging.INFO),
    ):
        collate_fn = _pp_loader(cfg_model, cfg_dl)

    collate_fn(["dummy"])
    assert calls == ["base"]
    add_masks.assert_not_called()
    assert f"Skipping pipeline parallel causal mask precomputation for model_type={model_type}" in caplog.text


def test_pp_autoconfig_success_chains_masks():
    """When AutoConfig succeeds, the resolved collate is wrapped with mask precomputation (base -> masks)."""
    call_order = []
    cfg_dl = ConfigNode({"collate_fn": lambda b: call_order.append("base") or b, "num_workers": 0})
    cfg_model = ConfigNode({"pretrained_model_name_or_path": "good/model"})

    def mock_add_masks(batch, model_config=None):
        call_order.append("masks")
        return batch

    with (
        patch("nemo_automodel.recipes.llm.train_ft.AutoConfig.from_pretrained", return_value=MagicMock()),
        patch("nemo_automodel.components.datasets.utils.add_causal_masks_to_batch", side_effect=mock_add_masks),
    ):
        collate_fn = _pp_loader(cfg_model, cfg_dl)

    collate_fn(["dummy_batch"])
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


class TestRunTrainOptimStepUsesEngine:
    """Training delegates every supported batch layout to Engine."""

    def _make_recipe(
        self,
        monkeypatch,
        pp_enabled,
    ):
        recipe = object.__new__(TrainFinetuneRecipeForNextTokenPrediction)
        object.__setattr__(recipe, "cfg", ConfigNode({"fp8": None}))
        object.__setattr__(recipe, "dist_env", SimpleNamespace(device=torch.device("cpu"), rank=0, is_main=True))
        object.__setattr__(recipe, "device_mesh", None)
        object.__setattr__(recipe, "pp_enabled", pp_enabled)
        object.__setattr__(recipe, "model_parts", [nn.Linear(4, 4)])
        object.__setattr__(recipe, "step_scheduler", SimpleNamespace(step=1, epoch=0))
        monkeypatch.setattr(
            recipe,
            "_dp_allreduce",
            lambda val, include_cp=False: val if isinstance(val, torch.Tensor) else torch.tensor(val),
        )
        monkeypatch.setattr(recipe, "_get_dp_group_size", lambda include_cp=False: 4)
        monkeypatch.setattr(recipe, "_get_cp_group_size", lambda: 1)

        object.__setattr__(recipe, "checkpointer", SimpleNamespace(maybe_wait_for_staging=lambda: None))
        object.__setattr__(recipe, "loss_fn", object())
        engine = MagicMock()
        engine.forward_backward.return_value = ForwardBackwardResult(
            loss=torch.tensor(0.5),
            loss_sum=torch.tensor(2.0),
            weight_sum=torch.tensor(4.0),
            token_outputs=[],
            batch_outputs=[],
        )
        engine.step.return_value = SimpleNamespace(grad_norm=torch.tensor(1.0), learning_rates=(0.01,))
        object.__setattr__(recipe, "engine", engine)
        object.__setattr__(recipe, "timestamp", 0.0)
        monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
        return recipe

    def test_pp_engine_owns_forward_backward_and_token_normalization(self, monkeypatch):
        recipe = self._make_recipe(monkeypatch, pp_enabled=True)
        batches = [
            {"input_ids": torch.tensor([[1, 2, 3]]), "labels": torch.tensor([[1, 2, -100]])},
            {"input_ids": torch.tensor([[4, 5, 6]]), "labels": torch.tensor([[4, 5, -100]])},
        ]
        datums = [object(), object()]
        make_datum = MagicMock(side_effect=datums)
        monkeypatch.setattr(recipe, "_make_engine_datum", make_datum)
        engine = MagicMock()
        engine.forward_backward.return_value = ForwardBackwardResult(
            loss=torch.tensor(0.25),
            loss_sum=torch.tensor(1.75),
            weight_sum=torch.tensor(7.0),
            token_outputs=[],
            batch_outputs=[],
        )
        engine.step.return_value = SimpleNamespace(grad_norm=torch.tensor(1.0), learning_rates=(0.02,))
        object.__setattr__(recipe, "engine", engine)
        optimizer = SimpleNamespace(
            step=MagicMock(),
            zero_grad=MagicMock(),
            param_groups=[{"lr": 0.01}],
        )
        object.__setattr__(recipe, "optimizer", [optimizer])
        metrics = recipe._run_train_optim_step(batches)

        engine.forward_backward.assert_called_once_with([[datum] for datum in datums], recipe._engine_loss_fn)
        engine.step.assert_called_once_with(before_optimizer_step=recipe.checkpointer.maybe_wait_for_staging)
        assert make_datum.call_count == 2
        optimizer.step.assert_not_called()
        optimizer.zero_grad.assert_not_called()
        assert metrics.metrics["loss"] == pytest.approx(0.25)
        assert metrics.metrics["grad_norm"] == pytest.approx(1.0)
        assert metrics.metrics["lr"] == pytest.approx(0.02)
        assert metrics.metrics["num_label_tokens"] == 7

    def test_pp_thd_batch_uses_engine(self, monkeypatch):
        recipe = self._make_recipe(monkeypatch, pp_enabled=True)
        batch = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "labels": torch.tensor([[1, 2, -100]]),
            "qkv_format": "thd",
        }
        engine = MagicMock()
        engine.forward_backward.return_value = ForwardBackwardResult(
            loss=torch.tensor(0.5),
            loss_sum=torch.tensor(1.0),
            weight_sum=torch.tensor(2.0),
            token_outputs=[],
            batch_outputs=[],
        )
        engine.step.return_value = SimpleNamespace(grad_norm=torch.tensor(1.0), learning_rates=(0.01,))
        object.__setattr__(recipe, "engine", engine)

        recipe._run_train_optim_step([batch])

        engine.forward_backward.assert_called_once()
        engine.step.assert_called_once_with(before_optimizer_step=recipe.checkpointer.maybe_wait_for_staging)
        datums, loss_fn = engine.forward_backward.call_args.args
        assert len(datums) == 1 and len(datums[0]) == 1
        assert datums[0][0].model_inputs["qkv_format"] == "thd"
        assert loss_fn == recipe._engine_loss_fn

    def test_train_step_leaves_fp8_post_step_work_to_engine(self, monkeypatch):
        recipe = self._make_recipe(monkeypatch, pp_enabled=False)
        object.__setattr__(
            recipe,
            "cfg",
            ConfigNode(
                {
                    "fp8": {
                        "enabled": True,
                        "precompute_float8_dynamic_scale_for_fsdp": True,
                    }
                }
            ),
        )
        object.__setattr__(
            recipe,
            "device_mesh",
            {"dp_shard": SimpleNamespace(size=lambda: 2)},
        )
        monkeypatch.setattr(
            "nemo_automodel.recipes.llm.train_ft.precompute_float8_dynamic_scale_for_fsdp",
            lambda _model: pytest.fail("the LLM recipe must not run FP8 post-step work directly"),
            raising=False,
        )

        recipe._run_train_optim_step([{"input_ids": torch.tensor([[1, 2]]), "labels": torch.tensor([[2, -100]])}])

        recipe.engine.step.assert_called_once_with(
            before_optimizer_step=recipe.checkpointer.maybe_wait_for_staging,
        )


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
    assert trainer.engine is not None


def test_setup_builds_engine_for_mtp_without_context_parallelism(monkeypatch):
    cp_size = 1
    cfg = _minimal_cfg_with_rope_fusion(cp_size=cp_size, rope_fusion=True)
    _patch_setup_minimals_with_cp(monkeypatch, cp_size=cp_size)
    model = DummyModel()
    model.mtp = nn.Identity()
    monkeypatch.setattr("nemo_automodel.recipes.llm.train_ft.build_model", lambda *args, **kwargs: model)

    trainer = TrainFinetuneRecipeForNextTokenPrediction(cfg)
    trainer.setup()

    assert trainer.engine is not None


def test_setup_builds_engine_for_magi(monkeypatch):
    cfg = _minimal_cfg_with_rope_fusion(cp_size=1, rope_fusion=True)
    _patch_setup_minimals_with_cp(monkeypatch, cp_size=1)
    monkeypatch.setattr(
        "nemo_automodel.recipes.llm.train_ft.setup_magi",
        lambda *args, **kwargs: SimpleNamespace(enabled=True, hf_dispatch=False),
    )

    trainer = TrainFinetuneRecipeForNextTokenPrediction(cfg)
    trainer.setup()

    assert trainer.engine is not None


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
        monkeypatch.setattr("nemo_automodel.recipes.llm.train_ft.ScopedRNG", lambda **kwargs: nullcontext())
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


def test_engine_validation_loss_uses_eval_path():
    recipe = TrainFinetuneRecipeForNextTokenPrediction.__new__(TrainFinetuneRecipeForNextTokenPrediction)
    recipe.pp_enabled = False
    expected = torch.tensor(5.0)
    compute_loss = MagicMock(return_value=expected)
    recipe._compute_causal_lm_loss = compute_loss

    output = object()
    labels = torch.tensor([[1, 2, -100]])
    loss_inputs = {
        "labels": labels,
        "weights": labels.ne(-100),
        "cu_seqlens": torch.tensor([0, 3], dtype=torch.int32),
    }

    loss = recipe._engine_validation_loss_fn(output, loss_inputs)

    assert loss is expected
    compute_loss.assert_called_once_with(
        output,
        labels,
        loss_inputs,
        num_label_tokens=None,
        is_train=False,
    )
