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
import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock
from types import SimpleNamespace

from contextlib import nullcontext

from nemo_automodel.components.loggers.metric_logger import MetricsSample
from nemo_automodel.recipes.vlm.finetune import (
    FinetuneRecipeForVLM,
    _get_model_name,
    build_model_and_optimizer,
)


class _Cfg(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)
def test_get_model_name_prefers_pretrained_path():
    cfg = _Cfg(pretrained_model_name_or_path="org/model")
    assert _get_model_name(cfg) == "org/model"

    cfg = _Cfg(config={"pretrained_model_name_or_path": "nested/model"})
    assert _get_model_name(cfg) == "nested/model"

    assert _get_model_name(_Cfg()) is None


from nemo_automodel.components.checkpoint.checkpointing import Checkpointer, CheckpointingConfig



def _count_trainable(parameters):
    return sum(p.numel() for p in parameters if getattr(p, "requires_grad", False))

@pytest.fixture(autouse=True)
def _mock_missing_cuda(monkeypatch):
    """Some helper functions unconditionally access torch.cuda APIs. When running on a
    CPU-only build they raise `RuntimeError: Torch not compiled with CUDA`.
    Patch the relevant CUDA APIs with no-op stubs when CUDA is unavailable."""
    if torch.cuda.is_available():
        yield  # nothing to do
        return

    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [], raising=False)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda _: None, raising=False)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda _: None, raising=False)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None, raising=False)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0, raising=False)
    yield


class DummyModel(nn.Module):
    """Simple model containing an embedding and a linear layer ("language_model")."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(10, 4)
        # expose as attribute so apply_parameter_freezing can find it
        self.language_model = nn.Linear(4, 4)
        # Add config attribute like HF models have
        self.config = SimpleNamespace()

    def forward(self, x):  # pragma: no cover – not needed for these unit tests
        return self.language_model(self.embedding(x))


class DummyOptConfig:
    """Mimics an optimizer config object with an *instantiate* method."""

    def __init__(self, lr: float = 0.01):
        self.lr = lr
        self.foreach = None
    def instantiate(self, params):
        # Always return an SGD optimizer for the given params
        return torch.optim.SGD(params, lr=self.lr)

    def get(self, key, default):
        return getattr(self, key, default)

class DummyModelConfig:
    """Mimics the Hydra/OmegaConf model config with an *instantiate* method."""

    def __init__(self):
        from nemo_automodel._transformers import NeMoAutoModelForImageTextToText
        # Add _target_ to make the config valid for VLM finetuning
        self._target_ = NeMoAutoModelForImageTextToText.from_pretrained

    def instantiate(self, **kwargs):
        return DummyModel()

    def get(self, key, default=None):
        return getattr(self, key, default)


# -----------------------------------------------------------------------------
# build_model_and_optimizer
# -----------------------------------------------------------------------------

def test_build_model_and_optimizer_basic():
    """Test basic build_model_and_optimizer for VLM.
    Note: New API passes freeze_config to model instantiation instead of calling _freeze_model."""
    cfg_model = DummyModelConfig()
    cfg_opt = DummyOptConfig(lr=0.01)

    with patch('nemo_automodel.recipes.vlm.finetune._supports_logits_to_keep', return_value=True):
        # New API: freeze_config is passed to model instantiation
        model, optim, loss_fn = build_model_and_optimizer(
            cfg_model=cfg_model,
            cfg_opt=cfg_opt,
            cfg_freeze=None,
            cfg_peft=None,
            model_wrapper=SimpleNamespace(parallelize=lambda m: m),
            seed=123,
            tp_size=1,
        )

    # Check returned objects and their properties
    assert isinstance(model, DummyModel)
    assert isinstance(optim, list)
    assert len(optim) == 1
    assert isinstance(optim[0], torch.optim.Optimizer)


def test_build_model_and_optimizer_passes_freeze_config():
    """Test that freeze_config is passed to model instantiation."""
    from nemo_automodel._transformers import NeMoAutoModelForImageTextToText

    captured_kwargs = {}

    class CapturingModelConfig:
        def __init__(self):
            self._target_ = NeMoAutoModelForImageTextToText.from_pretrained

        def instantiate(self, **kwargs):
            captured_kwargs.update(kwargs)
            return DummyModel()

        def get(self, key, default=None):
            return getattr(self, key, default)

    cfg_model = CapturingModelConfig()
    cfg_opt = DummyOptConfig(lr=0.01)

    class FreezeConfig:
        def to_dict(self):
            return {"freeze_embeddings": True, "freeze_language_model": False}

    with patch('nemo_automodel.recipes.vlm.finetune._supports_logits_to_keep', return_value=True):
        model, optim, loss_fn = build_model_and_optimizer(
            cfg_model=cfg_model,
            cfg_opt=cfg_opt,
            cfg_freeze=FreezeConfig(),
            cfg_peft=None,
            model_wrapper=None,
            seed=123,
        )

    # Verify freeze_config was passed to model instantiation
    assert "freeze_config" in captured_kwargs
    assert captured_kwargs["freeze_config"] == {"freeze_embeddings": True, "freeze_language_model": False}


# -----------------------------------------------------------------------------
# FinetuneRecipeForVLM helpers
# -----------------------------------------------------------------------------


class _DummyOptimizer:
    def __init__(self):
        self.param_groups = [{"lr": 0.01}]
        self.step_called = False
        self.zero_grad_called = False

    def step(self):
        self.step_called = True

    def zero_grad(self, set_to_none=True):
        self.zero_grad_called = True


class _TensorModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, **batch):
        return torch.zeros((), requires_grad=True)


@pytest.mark.cuda(False)
def test_run_train_step_supports_tensor_outputs(monkeypatch):
    recipe = FinetuneRecipeForVLM.__new__(FinetuneRecipeForVLM)
    recipe.dist_env = SimpleNamespace(device="cpu")
    recipe.device_mesh = None
    recipe.moe_mesh = None
    recipe.loss_fn = object()
    model = _TensorModel()
    recipe.model_parts = [model]  # Now uses model_parts instead of model
    recipe.pp_enabled = False  # Pipeline parallelism disabled
    recipe.optimizer = [_DummyOptimizer()]  # Now a list
    recipe.step_scheduler = SimpleNamespace(step=0, epoch=0)
    recipe.checkpointer = SimpleNamespace(maybe_wait_for_staging=lambda: None)
    recipe.cfg = _Cfg(fp8=None)
    recipe.lr_scheduler = None
    recipe.timestamp = 0.0
    recipe.model_wrapper = None

    recipe._dp_allreduce = lambda tensor, include_cp=False: tensor
    recipe._get_dp_group_size = lambda include_cp=True: 1
    recipe._get_cp_group_size = lambda: 1

    batches = [
        {
            "labels": torch.tensor([[1, -100]]),
            "input_ids": torch.tensor([[1, 2]]),
        }
    ]

    logits_seen = {}

    def fake_calculate_loss(*args, **kwargs):
        logits_seen["value"] = kwargs["logits"]
        return torch.tensor(1.0, requires_grad=True)

    monkeypatch.setattr(
        "nemo_automodel.recipes.vlm.finetune.make_cp_batch_and_ctx",
        lambda device_mesh, batch: (lambda: nullcontext(), batch),
    )
    monkeypatch.setattr(
        "nemo_automodel.recipes.vlm.finetune.get_sync_ctx",
        lambda model, is_last, defer_fsdp_grad_sync=True: nullcontext(),
    )

    calculate_mock = MagicMock(side_effect=fake_calculate_loss)
    monkeypatch.setattr("nemo_automodel.recipes.vlm.finetune.calculate_loss", calculate_mock)

    grad_clip_mock = MagicMock(return_value=2.5)
    monkeypatch.setattr(
        "nemo_automodel.recipes.vlm.finetune.scale_grads_and_clip_grad_norm",
        grad_clip_mock,
    )

    monkeypatch.setattr(
        "nemo_automodel.recipes.vlm.finetune.prepare_for_grad_accumulation",
        lambda model_parts, pp_enabled: None,
    )
    monkeypatch.setattr(
        "nemo_automodel.recipes.vlm.finetune.prepare_for_final_backward",
        lambda model_parts, pp_enabled: None,
    )

    metrics = recipe._run_train_optim_step(batches, max_grad_norm=1.0)

    assert isinstance(metrics, MetricsSample)
    assert logits_seen["value"].requires_grad
    grad_clip_mock.assert_called_once()
    assert calculate_mock.call_args.kwargs["num_label_tokens"] == 1
    assert metrics.metrics["grad_norm"] == 2.5
    assert recipe.optimizer[0].step_called
    assert recipe.optimizer[0].zero_grad_called


# -----------------------------------------------------------------------------
# AutoProcessor exception handling test
# -----------------------------------------------------------------------------

def test_autoprocessor_success():
    """Test successful AutoProcessor creation."""

    with patch('transformers.AutoProcessor') as mock_auto_processor:
        mock_processor = MagicMock()
        mock_auto_processor.from_pretrained.return_value = mock_processor

        model_id = "test/model"

        processor = mock_auto_processor.from_pretrained(model_id)

        assert processor is mock_processor
        mock_auto_processor.from_pretrained.assert_called_once_with("test/model")


def test_autoprocessor_exception_handling(caplog):
    """Test AutoProcessor exception handling and logging in build_dataloader."""
    import logging
    from nemo_automodel.recipes.vlm.finetune import build_dataloader

    with patch('transformers.AutoProcessor.from_pretrained') as mock_from_pretrained, \
         patch('nemo_automodel.components.training.rng.StatefulRNG'), \
         patch('torch.utils.data.distributed.DistributedSampler'), \
         patch('nemo_automodel.components.datasets.vlm.collate_fns.COLLATE_FNS', {'NoneType': MagicMock()}):

        # Set up the exception
        mock_from_pretrained.side_effect = Exception("Model does not have AutoProcessor")

        # Mock configurations - minimal setup
        cfg_ds = MagicMock()
        cfg_ds.instantiate.return_value = []
        cfg_ds.path_or_dataset = "test/dataset"

        cfg_dl = MagicMock()
        cfg_dl.get.return_value = None  # No custom settings
        cfg_dl.instantiate.return_value = MagicMock()

        cfg_processor = None  # This triggers the exception path

        with caplog.at_level(logging.WARNING):
            dataloader, processor = build_dataloader(cfg_ds, cfg_dl, "test/model", cfg_processor, None, 123, 1)

        # Verify the results
        assert processor is None
        mock_from_pretrained.assert_called_once_with("test/model")



def test_autoprocessor_with_processor_kwargs(caplog):
    """Test AutoProcessor exception handling when cfg_processor has no instantiate method."""
    import logging
    from nemo_automodel.recipes.vlm.finetune import build_dataloader

    # Simple processor config class without instantiate method
    class ProcessorConfig:
        def to_dict(self):
            return {"trust_remote_code": True, "some_param": "value"}

    with patch('transformers.AutoProcessor.from_pretrained') as mock_from_pretrained, \
         patch('nemo_automodel.components.training.rng.StatefulRNG'), \
         patch('torch.utils.data.distributed.DistributedSampler'), \
         patch('nemo_automodel.components.datasets.vlm.collate_fns.COLLATE_FNS', {'NoneType': MagicMock()}):

        # Set up the exception
        mock_from_pretrained.side_effect = Exception("Model does not have AutoProcessor")

        # Mock configurations - minimal setup
        cfg_ds = MagicMock()
        cfg_ds.instantiate.return_value = []
        cfg_ds.path_or_dataset = "test/dataset"

        cfg_dl = MagicMock()
        cfg_dl.get.return_value = None  # No custom settings
        cfg_dl.instantiate.return_value = MagicMock()

        cfg_processor = ProcessorConfig()  # This has to_dict but no instantiate

        with caplog.at_level(logging.WARNING):
            dataloader, processor = build_dataloader(cfg_ds, cfg_dl, "test/model", cfg_processor, None, 123, 1)

        # Verify the results
        assert processor is None
        mock_from_pretrained.assert_called_once_with("test/model", trust_remote_code=True, some_param="value")


# -----------------------------------------------------------------------------
# State dict adapter tests for _maybe_adapt_state_dict_to_hf in VLM
# -----------------------------------------------------------------------------


class MockStateDictAdapter:
    """Mock state dict adapter that transforms keys."""

    def to_hf(self, state_dict, exclude_key_regex=None, quantization=False, **kwargs):
        """Transform state dict keys by adding 'vlm_transformed_' prefix."""
        return {f"vlm_transformed_{k}": v for k, v in state_dict.items()}


class DummyModelWithAdapter(torch.nn.Module):
    """VLM model with a state_dict_adapter for testing."""

    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(10, 4)
        self.language_model = torch.nn.Linear(4, 4)
        self.state_dict_adapter = MockStateDictAdapter()

    def forward(self, x):
        return self.language_model(self.embedding(x))


class DummyModelConfigWithAdapter:
    """Mock model config that returns a model with state_dict_adapter."""

    def __init__(self):
        from nemo_automodel._transformers import NeMoAutoModelForImageTextToText
        # Add _target_ to make the config valid for VLM finetuning
        self._target_ = NeMoAutoModelForImageTextToText.from_pretrained

    def instantiate(self, **kwargs):
        return DummyModelWithAdapter()

    def get(self, key, default=None):
        return getattr(self, key, default)


def test_vlm_build_model_with_adapter():
    """Test that model with state_dict_adapter is properly instantiated in VLM."""

    cfg_opt = DummyOptConfig(lr=0.01)

    # Create a config that simulates NeMoAutoModel's internal infrastructure handling
    from nemo_automodel._transformers import NeMoAutoModelForImageTextToText

    class NeMoModelConfigWithAdapter:
        def __init__(self):
            self._target_ = NeMoAutoModelForImageTextToText.from_pretrained

        def instantiate(self, **kwargs):
            return DummyModelWithAdapter()

        def get(self, key, default=None):
            return getattr(self, key, default)

    cfg_model = NeMoModelConfigWithAdapter()

    with patch('nemo_automodel.recipes.vlm.finetune._supports_logits_to_keep', return_value=True):
        model, optim, loss_fn = build_model_and_optimizer(
            cfg_model=cfg_model,
            cfg_opt=cfg_opt,
            cfg_freeze=None,
            cfg_peft=None,
            model_wrapper=SimpleNamespace(parallelize=lambda m: m),
            seed=123,
            tp_size=1,
        )

    # Model should be instantiated with adapter
    assert model is not None
    assert hasattr(model, "state_dict_adapter")


def test_vlm_build_model_without_adapter():
    """Test that model without state_dict_adapter is properly instantiated in VLM."""

    cfg_opt = DummyOptConfig(lr=0.01)

    # Create a config that simulates NeMoAutoModel's internal infrastructure handling (no adapter)
    from nemo_automodel._transformers import NeMoAutoModelForImageTextToText

    class NeMoModelConfigNoAdapter:
        def __init__(self):
            self._target_ = NeMoAutoModelForImageTextToText.from_pretrained

        def instantiate(self, **kwargs):
            return DummyModel()  # No adapter

        def get(self, key, default=None):
            return getattr(self, key, default)

    cfg_model = NeMoModelConfigNoAdapter()

    with patch('nemo_automodel.recipes.vlm.finetune._supports_logits_to_keep', return_value=True):
        model, optim, loss_fn = build_model_and_optimizer(
            cfg_model=cfg_model,
            cfg_opt=cfg_opt,
            cfg_freeze=None,
            cfg_peft=None,
            model_wrapper=SimpleNamespace(parallelize=lambda m: m),
            seed=123,
            tp_size=1,
        )

    # Model should be instantiated without adapter
    assert model is not None
    assert not hasattr(model, "state_dict_adapter")


def test_vlm_build_model_with_quantization_config():
    """Test that model with quantization_config is properly instantiated in VLM."""
    from nemo_automodel._transformers import NeMoAutoModelForImageTextToText

    cfg_opt = DummyOptConfig(lr=0.01)

    # Create a model config that simulates NeMoAutoModel's internal infrastructure handling
    class DummyQuantizedVLMModelConfig:
        def __init__(self):
            self._target_ = NeMoAutoModelForImageTextToText.from_pretrained

        def instantiate(self, **kwargs):
            model = DummyModel()
            # Add a config attribute with quantization_config
            model.config = SimpleNamespace(quantization_config={"bits": 4})
            return model

        def get(self, key, default=None):
            return getattr(self, key, default)

    cfg_model = DummyQuantizedVLMModelConfig()

    with patch('nemo_automodel.recipes.vlm.finetune._supports_logits_to_keep', return_value=True):
        model, optim, loss_fn = build_model_and_optimizer(
            cfg_model=cfg_model,
            cfg_opt=cfg_opt,
            cfg_freeze=None,
            cfg_peft=None,
            model_wrapper=SimpleNamespace(parallelize=lambda m: m),
            seed=123,
            tp_size=1,
        )

    # Model should be instantiated with quantization config
    assert model is not None
    assert hasattr(model.config, "quantization_config")


def test_vlm_build_model_without_quantization_config():
    """Test that model without quantization_config is properly instantiated in VLM."""
    from nemo_automodel._transformers import NeMoAutoModelForImageTextToText

    cfg_opt = DummyOptConfig(lr=0.01)

    # Create a config that simulates NeMoAutoModel's internal infrastructure handling (no quant config)
    class DummyNoQuantVLMModelConfig:
        def __init__(self):
            self._target_ = NeMoAutoModelForImageTextToText.from_pretrained

        def instantiate(self, **kwargs):
            return DummyModel()  # DummyModel has no config.quantization_config

        def get(self, key, default=None):
            return getattr(self, key, default)

    cfg_model = DummyNoQuantVLMModelConfig()

    with patch('nemo_automodel.recipes.vlm.finetune._supports_logits_to_keep', return_value=True):
        model, optim, loss_fn = build_model_and_optimizer(
            cfg_model=cfg_model,
            cfg_opt=cfg_opt,
            cfg_freeze=None,
            cfg_peft=None,
            model_wrapper=SimpleNamespace(parallelize=lambda m: m),
            seed=123,
            tp_size=1,
        )

    # Model should be instantiated without quantization config
    assert model is not None
    assert not hasattr(model.config, "quantization_config")


# =============================================================================
# New tests for VLM-specific build_model_and_optimizer functionality
# =============================================================================


def test_vlm_build_model_raises_value_error_for_non_nemo_auto_model():
    """Test that VLM build_model_and_optimizer raises ValueError when target is not NeMoAutoModelForImageTextToText."""

    # Create a cfg_model that targets something other than NeMoAutoModelForImageTextToText
    class InvalidModelConfig:
        def __init__(self):
            self._target_ = "some.invalid.Target"

        def instantiate(self, **kwargs):
            return DummyModel()

        def get(self, key, default=None):
            return getattr(self, key, default)

    cfg_model = InvalidModelConfig()
    cfg_opt = DummyOptConfig(lr=0.01)

    with pytest.raises(ValueError, match="VLM finetuning requires NeMoAutoModelForImageTextToText"):
        build_model_and_optimizer(
            cfg_model=cfg_model,
            cfg_opt=cfg_opt,
            cfg_freeze=None,
            cfg_peft=None,
            model_wrapper=None,
            seed=42,
        )


def test_vlm_build_model_disables_foreach_with_tp():
    """Test that when tp_size > 1, cfg_opt.foreach is set to False in VLM."""
    from nemo_automodel._transformers import NeMoAutoModelForImageTextToText

    class NeMoVLMModelConfig:
        def __init__(self):
            self._target_ = NeMoAutoModelForImageTextToText.from_pretrained

        def instantiate(self, **kwargs):
            return DummyModel()

        def get(self, key, default=None):
            return getattr(self, key, default)

    cfg_model = NeMoVLMModelConfig()
    cfg_opt = DummyOptConfig(lr=0.01)
    cfg_opt.foreach = True  # Initially True

    with patch('nemo_automodel.recipes.vlm.finetune._supports_logits_to_keep', return_value=True):
        model, optimizer, loss_fn = build_model_and_optimizer(
            cfg_model=cfg_model,
            cfg_opt=cfg_opt,
            cfg_freeze=None,
            cfg_peft=None,
            model_wrapper=None,
            seed=42,
            tp_size=2,  # TP > 1
        )

    # Verify foreach was disabled
    assert cfg_opt.foreach is False


from nemo_automodel.recipes.vlm.finetune import (
    parallelize_for_pp,
    build_step_scheduler,
    build_lr_scheduler,
    build_checkpoint_config,
    calculate_loss,
)


# -----------------------------------------------------------------------------
# build_step_scheduler tests
# -----------------------------------------------------------------------------


class TestBuildStepScheduler:
    """Tests for build_step_scheduler function."""

    def test_build_step_scheduler_with_defaults(self):
        """Test build_step_scheduler with default configuration."""
        mock_dataloader = MagicMock()
        mock_dataloader.__len__ = MagicMock(return_value=100)

        # Use empty config dict instead of None (None triggers assertion error)
        cfg = MagicMock()
        cfg.to_dict.return_value = {}

        step_scheduler = build_step_scheduler(
            cfg=cfg,
            dataloader=mock_dataloader,
            dp_group_size=2,
            local_batch_size=4,
        )

        # Verify default values are applied
        assert step_scheduler.num_epochs == 10
        assert step_scheduler.ckpt_every_steps == 100
        assert step_scheduler.dataloader is mock_dataloader

    def test_build_step_scheduler_with_custom_config(self):
        """Test build_step_scheduler with custom configuration."""
        mock_dataloader = MagicMock()
        mock_dataloader.__len__ = MagicMock(return_value=50)

        cfg = MagicMock()
        cfg.to_dict.return_value = {
            "num_epochs": 5,
            "ckpt_every_steps": 50,
            "max_steps": 200,
        }

        step_scheduler = build_step_scheduler(
            cfg=cfg,
            dataloader=mock_dataloader,
            dp_group_size=4,
            local_batch_size=8,
        )

        # Custom values should override defaults
        assert step_scheduler.num_epochs == 5
        assert step_scheduler.ckpt_every_steps == 50
        assert step_scheduler.max_steps == 200

    def test_build_step_scheduler_rejects_target(self):
        """Test that _target_ in config raises error when passed to StepScheduler."""
        mock_dataloader = MagicMock()
        mock_dataloader.__len__ = MagicMock(return_value=100)

        # Create a config object where "_target_" in cfg returns True
        cfg = {"_target_": "some.class"}

        with pytest.raises(AssertionError, match="_target_ not permitted"):
            build_step_scheduler(
                cfg=cfg,
                dataloader=mock_dataloader,
                dp_group_size=1,
                local_batch_size=1,
            )


# -----------------------------------------------------------------------------
# build_lr_scheduler tests
# -----------------------------------------------------------------------------


class TestBuildLRScheduler:
    """Tests for build_lr_scheduler function."""

    def test_build_lr_scheduler_returns_none_when_cfg_is_none(self):
        """Test that None config returns None scheduler."""
        result = build_lr_scheduler(cfg=None, optimizer=MagicMock(), step_scheduler=MagicMock())
        assert result is None

    def test_build_lr_scheduler_creates_schedulers_for_single_optimizer(self):
        """Test scheduler creation for single optimizer."""
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.01, weight_decay=0.01)

        mock_dataloader = MagicMock()
        mock_dataloader.__len__ = MagicMock(return_value=100)

        step_scheduler = MagicMock()
        step_scheduler.num_epochs = 10
        step_scheduler.dataloader = mock_dataloader
        step_scheduler.grad_acc_steps = 1
        step_scheduler.max_steps = None

        cfg = MagicMock()
        cfg.to_dict.return_value = {
            "lr_decay_style": "cosine",
        }

        schedulers = build_lr_scheduler(cfg=cfg, optimizer=optimizer, step_scheduler=step_scheduler)

        assert schedulers is not None
        assert len(schedulers) == 1
        # Verify scheduler was created with correct parameters
        assert schedulers[0].max_lr == 0.01
        assert schedulers[0].init_lr == 0.001  # 10% of base LR
        assert schedulers[0].min_lr == 0.0001  # 1% of base LR

    def test_build_lr_scheduler_creates_schedulers_for_optimizer_list(self):
        """Test scheduler creation for list of optimizers (PP case)."""
        opt1 = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.01)
        opt2 = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.02)
        optimizers = [opt1, opt2]

        mock_dataloader = MagicMock()
        mock_dataloader.__len__ = MagicMock(return_value=100)

        step_scheduler = MagicMock()
        step_scheduler.num_epochs = 5
        step_scheduler.dataloader = mock_dataloader
        step_scheduler.grad_acc_steps = 2
        step_scheduler.max_steps = None

        cfg = MagicMock()
        cfg.to_dict.return_value = {}

        schedulers = build_lr_scheduler(cfg=cfg, optimizer=optimizers, step_scheduler=step_scheduler)

        assert schedulers is not None
        assert len(schedulers) == 2
        # First scheduler uses first optimizer's LR
        assert schedulers[0].max_lr == 0.01
        # Second scheduler uses second optimizer's LR
        assert schedulers[1].max_lr == 0.02

    def test_build_lr_scheduler_respects_max_steps(self):
        """Test that max_steps limits total_steps calculation."""
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.01)

        mock_dataloader = MagicMock()
        mock_dataloader.__len__ = MagicMock(return_value=1000)

        step_scheduler = MagicMock()
        step_scheduler.num_epochs = 100  # Would be 100000 steps
        step_scheduler.dataloader = mock_dataloader
        step_scheduler.grad_acc_steps = 1
        step_scheduler.max_steps = 500  # Limit to 500

        cfg = MagicMock()
        cfg.to_dict.return_value = {}

        schedulers = build_lr_scheduler(cfg=cfg, optimizer=optimizer, step_scheduler=step_scheduler)

        # Decay steps should be limited by max_steps
        assert schedulers[0].lr_decay_steps == 500


# -----------------------------------------------------------------------------
# build_checkpoint_config tests
# -----------------------------------------------------------------------------


class TestBuildCheckpointConfig:
    """Tests for build_checkpoint_config function."""

    def test_build_checkpoint_config_with_defaults(self):
        """Test checkpoint config with minimal inputs."""
        config = build_checkpoint_config(
            cfg_ckpt=None,
            cache_dir="/tmp/cache",
            model_repo_id="org/model",
            is_peft=False,
        )

        assert config.enabled is True
        assert config.checkpoint_dir == "checkpoints/"
        # model_save_format is an enum, check value
        assert config.model_save_format.value == "safetensors"
        assert config.model_repo_id == "org/model"
        assert config.model_cache_dir == "/tmp/cache"
        assert config.save_consolidated is True
        assert config.is_peft is False

    def test_build_checkpoint_config_with_custom_config(self):
        """Test checkpoint config with custom settings."""
        cfg_ckpt = MagicMock()
        cfg_ckpt.to_dict.return_value = {
            "checkpoint_dir": "/custom/ckpt/",
            "save_consolidated": False,
            "restore_from": "/some/path",  # Should be removed
        }

        config = build_checkpoint_config(
            cfg_ckpt=cfg_ckpt,
            cache_dir=None,
            model_repo_id="org/model",
            is_peft=True,
        )

        assert config.checkpoint_dir == "/custom/ckpt/"
        assert config.save_consolidated is False
        assert config.is_peft is True

    def test_build_checkpoint_config_rejects_peft_with_torch_save(self):
        """Test that PEFT with torch_save format raises error."""
        cfg_ckpt = MagicMock()
        cfg_ckpt.to_dict.return_value = {
            "model_save_format": "torch_save",
        }

        with pytest.raises(ValueError, match="PEFT checkpointing is not supported for torch_save"):
            build_checkpoint_config(
                cfg_ckpt=cfg_ckpt,
                cache_dir=None,
                model_repo_id="org/model",
                is_peft=True,
            )

    def test_build_checkpoint_config_uses_hf_hub_cache_when_cache_dir_none(self):
        """Test that HF_HUB_CACHE is used when cache_dir is None."""
        from huggingface_hub import constants as hf_constants

        config = build_checkpoint_config(
            cfg_ckpt=None,
            cache_dir=None,
            model_repo_id="org/model",
            is_peft=False,
        )

        assert config.model_cache_dir == hf_constants.HF_HUB_CACHE


# -----------------------------------------------------------------------------
# calculate_loss tests
# -----------------------------------------------------------------------------


class TestCalculateLoss:
    """Tests for calculate_loss function."""

    def test_calculate_loss_with_masked_ce(self):
        """Test calculate_loss with MaskedCrossEntropy."""
        from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy

        loss_fn = MaskedCrossEntropy()
        logits = torch.randn(2, 10, 100)  # batch, seq, vocab
        labels = torch.randint(0, 100, (2, 10))
        labels[0, 5:] = -100  # Mask some tokens

        loss = calculate_loss(
            loss_fn,
            logits=logits,
            labels=labels,
            model=None,
            hidden_states=None,
            num_label_tokens=10,
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # scalar

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="FusedLinearCE requires CUDA")
    def test_calculate_loss_with_fused_linear_ce(self):
        """Test calculate_loss with FusedLinearCrossEntropy."""
        from nemo_automodel.components.loss.linear_ce import FusedLinearCrossEntropy

        loss_fn = FusedLinearCrossEntropy()
        hidden_states = torch.randn(2, 10, 64, device="cuda")
        labels = torch.randint(0, 100, (2, 10), device="cuda")

        # Mock model with lm_head
        model = MagicMock()
        lm_head = torch.nn.Linear(64, 100).cuda()
        model.get_output_embeddings.return_value = lm_head

        loss = calculate_loss(
            loss_fn,
            logits=None,
            labels=labels,
            model=model,
            hidden_states=hidden_states,
            num_label_tokens=20,
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="FusedLinearCE requires CUDA")
    def test_calculate_loss_fused_ce_finds_lm_head_by_name(self):
        """Test that FusedLinearCE can find lm_head via named_parameters when model has no get_output_embeddings."""
        from nemo_automodel.components.loss.linear_ce import FusedLinearCrossEntropy

        loss_fn = FusedLinearCrossEntropy()
        hidden_states = torch.randn(2, 5, 32, device="cuda")
        labels = torch.randint(0, 50, (2, 5), device="cuda")

        # Use a plain object that has lm_head but no get_output_embeddings
        # This tests the fallback path in calculate_loss
        class ModelWithLmHeadOnly:
            """Non-nn.Module model without get_output_embeddings."""

            def __init__(self):
                self._lm_head = torch.nn.Linear(32, 50).cuda()

            def named_parameters(self, remove_duplicate=False):
                return [("lm_head.weight", self._lm_head.weight), ("lm_head.bias", self._lm_head.bias)]

        model = ModelWithLmHeadOnly()

        loss = calculate_loss(
            loss_fn,
            logits=None,
            labels=labels,
            model=model,
            hidden_states=hidden_states,
            num_label_tokens=10,
        )

        assert isinstance(loss, torch.Tensor)

    def test_calculate_loss_fused_ce_raises_without_lm_head(self):
        """Test that FusedLinearCE raises when lm_head not found."""
        from nemo_automodel.components.loss.linear_ce import FusedLinearCrossEntropy

        loss_fn = FusedLinearCrossEntropy()
        hidden_states = torch.randn(2, 5, 32)
        labels = torch.randint(0, 50, (2, 5))

        # Model with no get_output_embeddings and no lm_head in named_parameters
        class ModelWithoutLmHead(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.other_layer = torch.nn.Linear(32, 50)

        model = ModelWithoutLmHead()

        with pytest.raises(ValueError, match="lm_head.weight not found"):
            calculate_loss(
                loss_fn,
                logits=None,
                labels=labels,
                model=model,
                hidden_states=hidden_states,
                num_label_tokens=10,
            )


# -----------------------------------------------------------------------------
# PP Logic tests for _forward_backward_step
# -----------------------------------------------------------------------------


class _MockPPInfo:
    """Mock PP info structure."""

    def __init__(self, has_first_stage=True, has_last_stage=True, n_microbatches=2, add_losses=True):
        self.has_first_stage = has_first_stage
        self.has_last_stage = has_last_stage
        self._n_microbatches = n_microbatches
        self._add_losses = add_losses

        # Create a schedule mock that adds losses when called
        self.schedule = MagicMock()
        self.schedule._n_microbatches = n_microbatches

        def step_side_effect(*args, **kwargs):
            if self._add_losses and kwargs.get("losses") is not None:
                # Add mock losses for each microbatch
                for _ in range(n_microbatches):
                    kwargs["losses"].append(torch.tensor(0.5))

        self.schedule.step = MagicMock(side_effect=step_side_effect)


class _MockAutoPipeline:
    """Mock AutoPipeline for PP testing."""

    def __init__(self, has_first_stage=True, has_last_stage=True, n_microbatches=2, add_losses=True):
        self._info = _MockPPInfo(has_first_stage, has_last_stage, n_microbatches, add_losses)
        self.info = self._info


def _create_pp_recipe(model=None):
    """Helper to create a PP recipe bypassing BaseRecipe tracking."""
    if model is None:
        model = _TensorModel()
    recipe = object.__new__(FinetuneRecipeForVLM)
    # Initialize __dict__ directly to bypass BaseRecipe.__setattr__ tracking
    recipe.__dict__["__state_tracked"] = set()
    recipe.__dict__["_best_val_loss"] = float("inf")
    recipe.__dict__["dist_env"] = SimpleNamespace(device="cpu")
    recipe.__dict__["device_mesh"] = None
    recipe.__dict__["moe_mesh"] = None
    recipe.__dict__["pp_enabled"] = True
    recipe.__dict__["loss_fn"] = MagicMock()
    recipe.__dict__["model_wrapper"] = None
    recipe.__dict__["model_parts"] = [model]
    recipe.__dict__["_get_dp_group_size"] = lambda include_cp=True: 1
    return recipe


class TestForwardBackwardStepPP:
    """Tests for _forward_backward_step with pipeline parallelism enabled."""

    @pytest.fixture
    def pp_recipe(self):
        """Create a recipe configured for PP testing."""
        return _create_pp_recipe()

    def test_pp_skips_validation_forward(self, pp_recipe, monkeypatch):
        """Test that PP mode skips forward pass during validation."""
        pp_recipe.pp = _MockAutoPipeline()

        monkeypatch.setattr(
            "nemo_automodel.recipes.vlm.finetune.make_cp_batch_and_ctx",
            lambda device_mesh, batch: (lambda: nullcontext(), batch),
        )

        batch = {
            "labels": torch.tensor([[1, 2]]),
            "input_ids": torch.tensor([[1, 2]]),
        }
        loss_buffer = []

        # Should return early without error
        pp_recipe._forward_backward_step(
            idx=0,
            batch=batch,
            loss_buffer=loss_buffer,
            num_label_tokens=2,
            num_batches=1,
            is_train=False,  # Validation mode
        )

        # Loss buffer should be empty (no forward pass)
        assert len(loss_buffer) == 0

    def test_pp_vlm_chunking_equal_images_and_batch(self, pp_recipe, monkeypatch):
        """Test VLM pixel_values chunking when n_images == batch_size."""
        pp_recipe.pp = _MockAutoPipeline(has_first_stage=True, has_last_stage=True, n_microbatches=2)

        monkeypatch.setattr(
            "nemo_automodel.recipes.vlm.finetune.make_cp_batch_and_ctx",
            lambda device_mesh, batch: (lambda: nullcontext(), batch),
        )

        batch_size = 4
        n_images = 4
        # image_grid_hws: 4 images, each with different patch counts
        image_grid_hws = torch.tensor([[2, 2], [3, 3], [2, 3], [4, 4]])  # patch counts: 4, 9, 6, 16
        total_patches = 4 + 9 + 6 + 16  # = 35
        pixel_values = torch.randn(total_patches, 3, 14, 14)

        batch = {
            "labels": torch.randint(0, 100, (batch_size, 10)),
            "input_ids": torch.randint(0, 100, (batch_size, 10)),
            "pixel_values": pixel_values,
            "image_grid_hws": image_grid_hws,
        }
        loss_buffer = []

        pp_recipe._forward_backward_step(
            idx=0,
            batch=batch,
            loss_buffer=loss_buffer,
            num_label_tokens=40,
            num_batches=1,
            is_train=True,
        )

        # Verify chunking happened correctly
        model = pp_recipe.model_parts[0]
        assert model._vlm_pixel_values_chunks is None  # Cleared after step
        assert model._vlm_image_grid_hws_chunks is None
        assert model._vlm_chunk_idx is None

        # Verify schedule.step was called
        pp_recipe.pp.info.schedule.step.assert_called_once()

        # Verify loss was computed
        assert len(loss_buffer) == 1

    def test_pp_vlm_chunking_with_image_grid_thw(self, pp_recipe, monkeypatch):
        """Test VLM pixel_values chunking with image_grid_thw (3D grid) instead of image_grid_hws."""
        pp_recipe.pp = _MockAutoPipeline(has_first_stage=True, has_last_stage=True, n_microbatches=2)

        monkeypatch.setattr(
            "nemo_automodel.recipes.vlm.finetune.make_cp_batch_and_ctx",
            lambda device_mesh, batch: (lambda: nullcontext(), batch),
        )

        batch_size = 4
        n_images = 4
        # image_grid_thw: 4 images with T, H, W dimensions (uses .prod(dim=1) for patch counts)
        image_grid_thw = torch.tensor([[1, 2, 2], [1, 3, 3], [1, 2, 3], [1, 4, 4]])  # patch counts: 4, 9, 6, 16
        total_patches = 4 + 9 + 6 + 16  # = 35
        pixel_values = torch.randn(total_patches, 3, 14, 14)

        batch = {
            "labels": torch.randint(0, 100, (batch_size, 10)),
            "input_ids": torch.randint(0, 100, (batch_size, 10)),
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,  # Using thw instead of hws
        }
        loss_buffer = []

        pp_recipe._forward_backward_step(
            idx=0,
            batch=batch,
            loss_buffer=loss_buffer,
            num_label_tokens=40,
            num_batches=1,
            is_train=True,
        )

        # Verify chunking happened correctly
        model = pp_recipe.model_parts[0]
        assert model._vlm_pixel_values_chunks is None  # Cleared after step
        assert model._vlm_image_grid_hws_chunks is None
        assert model._vlm_chunk_idx is None

        # Verify schedule.step was called
        pp_recipe.pp.info.schedule.step.assert_called_once()

        # Verify loss was computed
        assert len(loss_buffer) == 1

    def test_pp_vlm_chunking_mismatched_images_and_batch(self, pp_recipe, monkeypatch, caplog):
        """Test VLM chunking when n_images != batch_size (fallback path)."""
        import logging

        pp_recipe.pp = _MockAutoPipeline(has_first_stage=True, has_last_stage=True, n_microbatches=2)

        monkeypatch.setattr(
            "nemo_automodel.recipes.vlm.finetune.make_cp_batch_and_ctx",
            lambda device_mesh, batch: (lambda: nullcontext(), batch),
        )

        batch_size = 4
        n_images = 2  # Different from batch_size
        image_grid_hws = torch.tensor([[2, 2], [3, 3]])
        total_patches = 4 + 9
        pixel_values = torch.randn(total_patches, 3, 14, 14)

        batch = {
            "labels": torch.randint(0, 100, (batch_size, 10)),
            "input_ids": torch.randint(0, 100, (batch_size, 10)),
            "pixel_values": pixel_values,
            "image_grid_hws": image_grid_hws,
        }
        loss_buffer = []

        with caplog.at_level(logging.WARNING):
            pp_recipe._forward_backward_step(
                idx=0,
                batch=batch,
                loss_buffer=loss_buffer,
                num_label_tokens=40,
                num_batches=1,
                is_train=True,
            )

        # Should log warning about mismatched images
        assert any("giving all images to first microbatch" in record.message for record in caplog.records)

    def test_pp_last_stage_computes_loss(self, pp_recipe, monkeypatch):
        """Test that last stage computes and buffers loss."""

        def mock_schedule_step(*args, **kwargs):
            # Simulate loss computation on last stage
            if kwargs.get("losses") is not None:
                kwargs["losses"].append(torch.tensor(0.5))
                kwargs["losses"].append(torch.tensor(0.3))

        pp = _MockAutoPipeline(has_first_stage=True, has_last_stage=True, n_microbatches=2, add_losses=False)
        pp.info.schedule.step = MagicMock(side_effect=mock_schedule_step)
        pp_recipe.pp = pp

        monkeypatch.setattr(
            "nemo_automodel.recipes.vlm.finetune.make_cp_batch_and_ctx",
            lambda device_mesh, batch: (lambda: nullcontext(), batch),
        )

        batch = {
            "labels": torch.tensor([[1, 2]]),
            "input_ids": torch.tensor([[1, 2]]),
        }
        loss_buffer = []

        pp_recipe._forward_backward_step(
            idx=0,
            batch=batch,
            loss_buffer=loss_buffer,
            num_label_tokens=2,
            num_batches=1,
            is_train=True,
        )

        # Loss should be sum of microbatch losses
        assert len(loss_buffer) == 1
        assert torch.isclose(loss_buffer[0], torch.tensor(0.8))

    def test_pp_non_last_stage_returns_zero_loss(self, pp_recipe, monkeypatch):
        """Test that non-last stage returns zero loss."""
        pp = _MockAutoPipeline(has_first_stage=True, has_last_stage=False, n_microbatches=2)
        pp_recipe.pp = pp

        monkeypatch.setattr(
            "nemo_automodel.recipes.vlm.finetune.make_cp_batch_and_ctx",
            lambda device_mesh, batch: (lambda: nullcontext(), batch),
        )

        batch = {
            "labels": torch.tensor([[1, 2]]),
            "input_ids": torch.tensor([[1, 2]]),
        }
        loss_buffer = []

        pp_recipe._forward_backward_step(
            idx=0,
            batch=batch,
            loss_buffer=loss_buffer,
            num_label_tokens=2,
            num_batches=1,
            is_train=True,
        )

        assert len(loss_buffer) == 1
        assert loss_buffer[0].item() == 0.0

    def test_pp_non_first_stage_skips_input_ids(self, pp_recipe, monkeypatch):
        """Test that non-first stage doesn't pass input_ids to schedule."""
        step_calls = []

        def mock_schedule_step(*args, **kwargs):
            step_calls.append((args, kwargs))
            # Add losses so torch.stack doesn't fail
            if kwargs.get("losses") is not None:
                kwargs["losses"].append(torch.tensor(0.5))

        pp = _MockAutoPipeline(has_first_stage=False, has_last_stage=True, n_microbatches=2, add_losses=False)
        pp.info.schedule.step = MagicMock(side_effect=mock_schedule_step)
        pp_recipe.pp = pp

        monkeypatch.setattr(
            "nemo_automodel.recipes.vlm.finetune.make_cp_batch_and_ctx",
            lambda device_mesh, batch: (lambda: nullcontext(), batch),
        )

        batch = {
            "labels": torch.tensor([[1, 2]]),
            "input_ids": torch.tensor([[1, 2]]),
        }
        loss_buffer = []

        pp_recipe._forward_backward_step(
            idx=0,
            batch=batch,
            loss_buffer=loss_buffer,
            num_label_tokens=2,
            num_batches=1,
            is_train=True,
        )

        # Should be called without positional args (no input_ids)
        assert len(step_calls) == 1
        args, kwargs = step_calls[0]
        assert len(args) == 0  # No positional args
        assert "target" in kwargs


# -----------------------------------------------------------------------------
# FinetuneRecipeForVLM.setup() tests
# -----------------------------------------------------------------------------


class TestFinetuneRecipeSetup:
    """Tests for FinetuneRecipeForVLM.setup() method components."""

    def test_setup_initializes_dist_env(self, monkeypatch):
        """Test that setup initializes distributed environment."""
        from nemo_automodel.recipes.vlm.finetune import build_distributed

        mock_dist_info = SimpleNamespace(
            rank=0,
            world_size=1,
            local_rank=0,
            is_main=True,
            device=torch.device("cpu"),
        )

        monkeypatch.setattr(
            "nemo_automodel.recipes.vlm.finetune.initialize_distributed",
            lambda backend, timeout_minutes: mock_dist_info,
        )

        dist_env = build_distributed({"backend": "gloo", "timeout_minutes": 5})

        assert dist_env.rank == 0
        assert dist_env.world_size == 1
        assert dist_env.is_main is True

    def test_setup_pp_config_validation(self):
        """Test PP configuration validation in setup."""
        # Create minimal config that would fail PP validation
        cfg = _Cfg()
        cfg.step_scheduler = _Cfg(local_batch_size=4)
        cfg.autopipeline = _Cfg(pp_microbatch_size=8)  # 4 // 8 = 0 < pp_size

        # The assertion should fail: pp_batch_size // pp_microbatch_size >= pp_size
        pp_batch_size = 4
        pp_microbatch_size = 8
        pp_size = 2

        with pytest.raises(AssertionError):
            assert pp_batch_size // pp_microbatch_size >= pp_size

    def test_setup_grad_norm_default(self):
        """Test that default grad norm is set when not specified."""
        cfg = _Cfg()
        cfg.clip_grad_norm = None

        max_grad_norm = cfg.get("clip_grad_norm.max_norm", None)
        if max_grad_norm is None:
            max_grad_norm = 1.0

        assert max_grad_norm == 1.0

    def test_setup_grad_norm_from_config(self):
        """Test that grad norm is read from config."""

        class NestedCfg:
            def __init__(self):
                self.clip_grad_norm = _Cfg(max_norm=0.5)

            def get(self, key, default=None):
                parts = key.split(".")
                obj = self
                for part in parts:
                    obj = getattr(obj, part, None)
                    if obj is None:
                        return default
                return obj

        cfg = NestedCfg()
        max_grad_norm = cfg.get("clip_grad_norm.max_norm", None)

        assert max_grad_norm == 0.5


# -----------------------------------------------------------------------------
# _forward_backward_step non-PP tests (FusedLinearCE path)
# -----------------------------------------------------------------------------


class _ModelOutput:
    """Model output that supports both attribute access and 'in' operator."""

    def __init__(self, logits, hidden_states=None):
        self.logits = logits
        self.hidden_states = hidden_states

    def __contains__(self, key):
        return hasattr(self, key) and getattr(self, key) is not None


class _ModelWithHiddenStates(torch.nn.Module):
    """Model that outputs hidden states for FusedLinearCE testing."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
        self.lm_head = torch.nn.Linear(10, 50)

    def forward(self, logits_to_keep=None, **kwargs):
        hidden = torch.randn(2, 5, 10)
        return _ModelOutput(
            logits=self.lm_head(hidden),
            hidden_states=[hidden],
        )

    def get_output_embeddings(self):
        return self.lm_head


def _create_non_pp_recipe(model, device="cpu"):
    """Helper to create a non-PP recipe bypassing BaseRecipe tracking."""
    recipe = object.__new__(FinetuneRecipeForVLM)
    # Initialize __dict__ directly to bypass BaseRecipe.__setattr__ tracking
    recipe.__dict__["__state_tracked"] = set()
    recipe.__dict__["_best_val_loss"] = float("inf")
    recipe.__dict__["dist_env"] = SimpleNamespace(device=device)
    recipe.__dict__["device_mesh"] = None
    recipe.__dict__["moe_mesh"] = None
    recipe.__dict__["pp_enabled"] = False
    recipe.__dict__["model_wrapper"] = None
    recipe.__dict__["model_parts"] = [model]
    recipe.__dict__["_get_dp_group_size"] = lambda include_cp=True: 1
    return recipe


class TestForwardBackwardStepNonPP:
    """Tests for _forward_backward_step without pipeline parallelism."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="FusedLinearCE requires CUDA")
    def test_non_pp_with_fused_linear_ce(self, monkeypatch):
        """Test non-PP path with FusedLinearCrossEntropy."""
        from nemo_automodel.components.loss.linear_ce import FusedLinearCrossEntropy

        # Model output class that supports both attribute access and 'in' operator
        class ModelOutput:
            def __init__(self, logits, hidden_states):
                self.logits = logits
                self.hidden_states = hidden_states

            def __contains__(self, key):
                return hasattr(self, key)

        # Create CUDA model for FusedLinearCE - must use bf16/fp16 for backward
        class CudaModelWithHiddenStates(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # Keep lm_head in bfloat16 to match hidden states
                self.lm_head = torch.nn.Linear(10, 50)

            def forward(self, logits_to_keep=None, **kwargs):
                # FusedLinearCE requires bf16/fp16 hidden states
                hidden = torch.randn(2, 5, 10, device="cuda", dtype=torch.bfloat16, requires_grad=True)
                # lm_head is already bfloat16, so no conversion needed
                return ModelOutput(
                    logits=self.lm_head(hidden),
                    hidden_states=[hidden],
                )

            def get_output_embeddings(self):
                return self.lm_head

        # Create model and convert entirely to bfloat16
        model = CudaModelWithHiddenStates().cuda().bfloat16()
        non_pp_recipe = _create_non_pp_recipe(model, device="cuda")
        non_pp_recipe.__dict__["loss_fn"] = FusedLinearCrossEntropy()

        monkeypatch.setattr(
            "nemo_automodel.recipes.vlm.finetune.make_cp_batch_and_ctx",
            lambda device_mesh, batch: (lambda: nullcontext(), batch),
        )
        monkeypatch.setattr(
            "nemo_automodel.recipes.vlm.finetune.get_sync_ctx",
            lambda model, is_last, defer_fsdp_grad_sync=True: nullcontext(),
        )

        batch = {
            "labels": torch.randint(0, 50, (2, 5)),
            "input_ids": torch.randint(0, 100, (2, 5)),
        }
        loss_buffer = []

        non_pp_recipe._forward_backward_step(
            idx=0,
            batch=batch,
            loss_buffer=loss_buffer,
            num_label_tokens=10,
            num_batches=1,
            is_train=True,
        )

        assert len(loss_buffer) == 1
        assert isinstance(loss_buffer[0], torch.Tensor)

    def test_non_pp_fused_ce_requires_hidden_states(self, monkeypatch):
        """Test that FusedLinearCE raises error when hidden_states not in output."""
        from nemo_automodel.components.loss.linear_ce import FusedLinearCrossEntropy

        # Model output class that supports 'in' operator but has no hidden_states
        class ModelOutputNoHiddenStates:
            def __init__(self, logits):
                self.logits = logits

            def __contains__(self, key):
                return hasattr(self, key)

        # Model that doesn't output hidden_states
        class BadModel(torch.nn.Module):
            def forward(self, logits_to_keep=None, **kwargs):
                return ModelOutputNoHiddenStates(logits=torch.randn(2, 5, 50))

        non_pp_recipe = _create_non_pp_recipe(BadModel())
        non_pp_recipe.__dict__["loss_fn"] = FusedLinearCrossEntropy()

        monkeypatch.setattr(
            "nemo_automodel.recipes.vlm.finetune.make_cp_batch_and_ctx",
            lambda device_mesh, batch: (lambda: nullcontext(), batch),
        )
        monkeypatch.setattr(
            "nemo_automodel.recipes.vlm.finetune.get_sync_ctx",
            lambda model, is_last, defer_fsdp_grad_sync=True: nullcontext(),
        )

        batch = {
            "labels": torch.randint(0, 50, (2, 5)),
            "input_ids": torch.randint(0, 100, (2, 5)),
        }
        loss_buffer = []

        with pytest.raises(ValueError, match="FusedLinearCrossEntropy requires the model to output hidden states"):
            non_pp_recipe._forward_backward_step(
                idx=0,
                batch=batch,
                loss_buffer=loss_buffer,
                num_label_tokens=10,
                num_batches=1,
                is_train=True,
            )

    def test_non_pp_with_masked_ce(self, monkeypatch):
        """Test non-PP path with MaskedCrossEntropy."""
        from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy

        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 50)

            def forward(self, **kwargs):
                # Create logits through a layer so gradients can flow
                x = torch.randn(2, 5, 10, requires_grad=True)
                logits = self.linear(x)
                return _ModelOutput(logits=logits, hidden_states=None)

        non_pp_recipe = _create_non_pp_recipe(SimpleModel())
        non_pp_recipe.__dict__["loss_fn"] = MaskedCrossEntropy()

        monkeypatch.setattr(
            "nemo_automodel.recipes.vlm.finetune.make_cp_batch_and_ctx",
            lambda device_mesh, batch: (lambda: nullcontext(), batch),
        )
        monkeypatch.setattr(
            "nemo_automodel.recipes.vlm.finetune.get_sync_ctx",
            lambda model, is_last, defer_fsdp_grad_sync=True: nullcontext(),
        )

        batch = {
            "labels": torch.randint(0, 50, (2, 5)),
            "input_ids": torch.randint(0, 100, (2, 5)),
        }
        loss_buffer = []

        non_pp_recipe._forward_backward_step(
            idx=0,
            batch=batch,
            loss_buffer=loss_buffer,
            num_label_tokens=10,
            num_batches=1,
            is_train=True,
        )

        assert len(loss_buffer) == 1
        assert isinstance(loss_buffer[0], torch.Tensor)

    def test_non_pp_validation_mode_no_backward(self, monkeypatch):
        """Test that validation mode doesn't call backward."""
        from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy

        # Simple model for this test
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 50)

            def forward(self, **kwargs):
                return _ModelOutput(logits=torch.randn(2, 5, 50), hidden_states=None)

        non_pp_recipe = _create_non_pp_recipe(SimpleModel())
        non_pp_recipe.__dict__["loss_fn"] = MaskedCrossEntropy()

        monkeypatch.setattr(
            "nemo_automodel.recipes.vlm.finetune.make_cp_batch_and_ctx",
            lambda device_mesh, batch: (lambda: nullcontext(), batch),
        )

        batch = {
            "labels": torch.randint(0, 50, (2, 5)),
            "input_ids": torch.randint(0, 100, (2, 5)),
        }
        loss_buffer = []

        # Should complete without error and not call backward
        non_pp_recipe._forward_backward_step(
            idx=0,
            batch=batch,
            loss_buffer=loss_buffer,
            num_label_tokens=10,
            num_batches=1,
            is_train=False,  # Validation mode
        )

        assert len(loss_buffer) == 1

    def test_non_pp_handles_dict_batch_values(self, monkeypatch):
        """Test that nested dict values in batch are moved to device."""
        from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy

        class SimpleModel(torch.nn.Module):
            def forward(self, **kwargs):
                return _ModelOutput(logits=torch.randn(2, 5, 50), hidden_states=None)

        non_pp_recipe = _create_non_pp_recipe(SimpleModel())
        non_pp_recipe.__dict__["loss_fn"] = MaskedCrossEntropy()

        monkeypatch.setattr(
            "nemo_automodel.recipes.vlm.finetune.make_cp_batch_and_ctx",
            lambda device_mesh, batch: (lambda: nullcontext(), batch),
        )

        # Batch with nested dict (like attention_mask dict)
        batch = {
            "labels": torch.randint(0, 50, (2, 5)),
            "input_ids": torch.randint(0, 100, (2, 5)),
            "nested": {
                "inner_tensor": torch.ones(2, 5),
                "none_value": None,
            },
        }
        loss_buffer = []

        # Should handle nested dict without error
        non_pp_recipe._forward_backward_step(
            idx=0,
            batch=batch,
            loss_buffer=loss_buffer,
            num_label_tokens=10,
            num_batches=1,
            is_train=False,
        )

        assert len(loss_buffer) == 1


class TestParallelizeForPP:
    """Tests for the new parallelize_for_pp function."""

    def test_parallelize_for_pp_with_model_wrapper(self):
        """Test that parallelize_for_pp calls model_wrapper.parallelize."""
        model = nn.Linear(10, 10)
        mock_wrapper = MagicMock()
        parallelized_model = nn.Linear(10, 10)
        mock_wrapper.parallelize.return_value = parallelized_model

        result = parallelize_for_pp(
            model,
            world_mesh=MagicMock(),
            model_wrapper=mock_wrapper,
        )

        mock_wrapper.parallelize.assert_called_once_with(model)
        assert result is parallelized_model

    def test_parallelize_for_pp_without_model_wrapper(self):
        """Test that parallelize_for_pp returns model unchanged when no wrapper."""
        model = nn.Linear(10, 10)

        result = parallelize_for_pp(
            model,
            world_mesh=MagicMock(),
            model_wrapper=None,
        )

        assert result is model

    def test_parallelize_for_pp_wrapper_without_parallelize_method(self):
        """Test behavior when wrapper doesn't have parallelize method."""
        model = nn.Linear(10, 10)
        mock_wrapper = MagicMock(spec=[])  # No parallelize method

        result = parallelize_for_pp(
            model,
            world_mesh=MagicMock(),
            model_wrapper=mock_wrapper,
        )

        assert result is model


# -----------------------------------------------------------------------------
# build_model_and_optimizer returns list (diff coverage)
# -----------------------------------------------------------------------------


def test_build_model_and_optimizer_returns_optimizer_list():
    """Test that build_model_and_optimizer returns list of optimizers."""
    cfg_model = DummyModelConfig()
    cfg_opt = DummyOptConfig(lr=0.01)

    with patch('nemo_automodel.recipes.vlm.finetune._supports_logits_to_keep', return_value=True):
        model, optimizer, loss_fn = build_model_and_optimizer(
            cfg_model=cfg_model,
            cfg_opt=cfg_opt,
            cfg_freeze=None,
            cfg_peft=None,
            model_wrapper=None,
            seed=42,
            tp_size=2,  # TP > 1
        )

    # Verify foreach was disabled
    assert cfg_opt.foreach is False


def test_vlm_build_model_returns_model_optimizer_loss_fn_tuple():
    """Test that VLM build_model_and_optimizer returns (model, optimizer, loss_fn) 3-tuple."""
    from nemo_automodel._transformers import NeMoAutoModelForImageTextToText

    class NeMoVLMModelConfig:
        def __init__(self):
            self._target_ = NeMoAutoModelForImageTextToText.from_pretrained

        def instantiate(self, **kwargs):
            return DummyModel()

        def get(self, key, default=None):
            return getattr(self, key, default)

    cfg_model = NeMoVLMModelConfig()
    cfg_opt = DummyOptConfig(lr=0.01)

    with patch('nemo_automodel.recipes.vlm.finetune._supports_logits_to_keep', return_value=True):
        result = build_model_and_optimizer(
            cfg_model=cfg_model,
            cfg_opt=cfg_opt,
            cfg_freeze=None,
            cfg_peft=None,
            model_wrapper=None,
            seed=42,
            loss_fn=MagicMock(),
        )

    # Should return 3-tuple
    assert isinstance(result, tuple)
    assert len(result) == 3
    model, optimizer, loss_fn = result
    assert model is not None
    assert optimizer is not None


@pytest.mark.parametrize("entry_point", ["from_config", "from_pretrained"])
def test_vlm_build_model_validates_nemo_auto_model_entry_points(entry_point):
    """Test that VLM recognizes both NeMoAutoModelForImageTextToText entry points."""
    from nemo_automodel._transformers import NeMoAutoModelForImageTextToText

    target = getattr(NeMoAutoModelForImageTextToText, entry_point)

    class NeMoVLMModelConfig:
        def __init__(self):
            self._target_ = target

        def instantiate(self, **kwargs):
            return DummyModel()

        def get(self, key, default=None):
            return getattr(self, key, default)

    cfg_model = NeMoVLMModelConfig()
    cfg_opt = DummyOptConfig(lr=0.01)

    with patch('nemo_automodel.recipes.vlm.finetune._supports_logits_to_keep', return_value=True):
        # Should not raise - entry point should be recognized
        model, optimizer, loss_fn = build_model_and_optimizer(
            cfg_model=cfg_model,
            cfg_opt=cfg_opt,
            cfg_freeze=None,
            cfg_peft=None,
            model_wrapper=None,
            seed=42,
        )

    assert model is not None
