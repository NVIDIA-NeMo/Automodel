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
import sys
import types
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.components.datasets.vlm.pp_media import (
    VLM_PP_MEDIA_KEY,
    chunk_step3_media,
    chunk_vlm_media,
    prepare_vlm_media_for_pp,
    stage_vlm_media_for_pp,
)
from nemo_automodel.components.loggers.metric_logger import MetricsSample
from nemo_automodel.components.optim.optimizer import LRSchedulerConfig, build_optimizer_config
from nemo_automodel.components.training.step_scheduler import StepSchedulerConfig
from nemo_automodel.recipes._typed_config import (
    _STEP_SCHEDULER_RUNTIME_KEYS,
    _as_dict,
    _callable_and_kwargs,
    _section_kwargs,
)
from nemo_automodel.recipes.vlm.finetune import (
    FinetuneRecipeForVLM,
    _get_model_name,
    build_model,
)


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


def build_step_scheduler(cfg, dataloader, dp_group_size, local_batch_size):
    """Build a StepScheduler from a YAML block (mirrors ``RecipeConfig.step_scheduler.build``)."""
    kwargs = {k: v for k, v in _section_kwargs(cfg).items() if k not in _STEP_SCHEDULER_RUNTIME_KEYS}
    return StepSchedulerConfig(**kwargs).build(dataloader, dp_group_size, local_batch_size)


def build_lr_scheduler(cfg, optimizer, step_scheduler):
    """Build an LR scheduler from a YAML block (mirrors ``RecipeConfig.lr_scheduler.build``)."""
    if cfg is None:
        return None
    return LRSchedulerConfig(**_section_kwargs(cfg)).build(optimizer, step_scheduler)


class _Cfg(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)


def test_get_model_name_prefers_pretrained_path():
    cfg = _Cfg(pretrained_model_name_or_path="org/model")
    assert _get_model_name(cfg) == "org/model"

    cfg = _Cfg(config={"pretrained_model_name_or_path": "nested/model"})
    assert _get_model_name(cfg) == "nested/model"

    assert _get_model_name(_Cfg()) is None


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
# build_model / build_optimizer
# -----------------------------------------------------------------------------


def test_build_model_and_optimizer_basic():
    """Test basic build_model and build_optimizer for VLM."""
    cfg_model = DummyModelConfig()
    cfg_opt = DummyOptConfig(lr=0.01)

    with patch("nemo_automodel.recipes.vlm.finetune._supports_logits_to_keep", return_value=True):
        model = build_model(
            cfg_model=cfg_model,
            cfg_freeze=None,
            cfg_peft=None,
            seed=123,
        )
        optim = build_optimizer(model, cfg_opt, None, None)

    # Check returned objects and their properties
    assert isinstance(model, DummyModel)
    assert isinstance(optim, list)
    assert len(optim) == 1
    assert isinstance(optim[0], torch.optim.Optimizer)


def test_build_model_passes_freeze_config():
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

    class FreezeConfig:
        def to_dict(self):
            return {"freeze_language_model": False, "freeze_vision_tower": True}

    with patch("nemo_automodel.recipes.vlm.finetune._supports_logits_to_keep", return_value=True):
        build_model(
            cfg_model=cfg_model,
            cfg_freeze=FreezeConfig(),
            cfg_peft=None,
            seed=123,
        )

    # Verify freeze_config was passed to model instantiation
    assert "freeze_config" in captured_kwargs
    assert captured_kwargs["freeze_config"] == {"freeze_language_model": False, "freeze_vision_tower": True}


def test_build_model_passes_distributed_setup():
    """Distributed policy is passed through the single setup object."""
    from nemo_automodel._transformers import NeMoAutoModelForImageTextToText
    from nemo_automodel.components.distributed.config import DistributedSetup
    from nemo_automodel.components.distributed.mesh import MeshContext

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
    distributed_setup = DistributedSetup(mesh_context=MeshContext())

    with patch("nemo_automodel.recipes.vlm.finetune._supports_logits_to_keep", return_value=True):
        build_model(
            cfg_model=cfg_model,
            cfg_freeze=None,
            cfg_peft=None,
            seed=123,
            distributed_setup=distributed_setup,
        )

    assert captured_kwargs["distributed_setup"] is distributed_setup
    assert "moe_config" not in captured_kwargs
    assert "activation_checkpointing" not in captured_kwargs


def test_build_model_no_moe_config_when_cfg_moe_is_none():
    """Test that moe_config and activation_checkpointing are not in kwargs when cfg_moe is None."""
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

    with patch("nemo_automodel.recipes.vlm.finetune._supports_logits_to_keep", return_value=True):
        build_model(
            cfg_model=cfg_model,
            cfg_freeze=None,
            cfg_peft=None,
            seed=123,
        )

    assert "moe_config" not in captured_kwargs
    assert "activation_checkpointing" not in captured_kwargs


def test_build_model_passes_quantization_config():
    """cfg_quantization is converted via create_bnb_config and forwarded as quantization_config."""
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
    cfg_quantization = SimpleNamespace(load_in_4bit=True)
    sentinel_bnb = object()

    with (
        patch("nemo_automodel.recipes.vlm.finetune._supports_logits_to_keep", return_value=True),
        patch(
            "nemo_automodel.components.quantization.qlora.create_bnb_config",
            return_value=sentinel_bnb,
        ) as mock_create,
    ):
        build_model(
            cfg_model=cfg_model,
            cfg_freeze=None,
            cfg_peft=None,
            seed=123,
            cfg_quantization=cfg_quantization,
        )

    # Wiring: cfg_quantization -> create_bnb_config(...) -> kwargs["quantization_config"]
    mock_create.assert_called_once_with(cfg_quantization)
    assert captured_kwargs.get("quantization_config") is sentinel_bnb


def test_build_model_no_quantization_config_when_none():
    """No quantization_config kwarg when cfg_quantization is None (the default)."""
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

    with patch("nemo_automodel.recipes.vlm.finetune._supports_logits_to_keep", return_value=True):
        build_model(
            cfg_model=cfg_model,
            cfg_freeze=None,
            cfg_peft=None,
            seed=123,
        )

    assert "quantization_config" not in captured_kwargs


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
    # ``is_remote_logging_step`` is read by ``_forward_backward_step`` when the
    # composite (gemma4 joint drafter) attaches drafter logits; default False
    # so non-drafter test paths skip the log line.
    recipe.step_scheduler = SimpleNamespace(step=0, epoch=0, is_remote_logging_step=False)
    recipe.checkpointer = SimpleNamespace(maybe_wait_for_staging=lambda: None)
    recipe.cfg = _Cfg(fp8=None)
    recipe.lr_scheduler = None
    recipe.timestamp = 0.0
    recipe.distributed_config = None

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
        "nemo_automodel.components.distributed.cp_utils._make_cp_batch_and_ctx",
        lambda device_mesh, batch, *a, **k: (lambda: nullcontext(), batch, None),
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


def _build_pp_recipe_for_optim_step(num_label_tokens_in_batch: int):
    """Shared setup for _run_train_optim_step tests with pp_enabled=True."""
    recipe = FinetuneRecipeForVLM.__new__(FinetuneRecipeForVLM)
    recipe.dist_env = SimpleNamespace(device="cpu", rank=0, is_main=False)
    # No "pp" in dim_names -> src_rank = mesh.reshape(-1)[-1].item(). With rank != src_rank
    # and is_main=False, neither distributed send nor recv branch fires.
    recipe.device_mesh = SimpleNamespace(mesh=torch.tensor([1]), mesh_dim_names=("dp",))
    recipe.moe_mesh = None
    recipe.loss_fn = object()
    recipe.model_parts = [_TensorModel()]
    recipe.pp_enabled = True
    recipe.optimizer = [_DummyOptimizer()]
    recipe.step_scheduler = SimpleNamespace(step=0, epoch=0, is_remote_logging_step=False)
    recipe.checkpointer = SimpleNamespace(maybe_wait_for_staging=lambda: None)
    recipe.cfg = _Cfg(fp8=None)
    recipe.lr_scheduler = None
    recipe.timestamp = 0.0
    recipe.distributed_config = None
    recipe._dp_allreduce = lambda tensor, include_cp=False: tensor
    recipe._get_dp_group_size = lambda include_cp=True: 1
    recipe._get_cp_group_size = lambda: 1

    # Build a batch whose (labels != -100).sum() == num_label_tokens_in_batch.
    seq = [1] * num_label_tokens_in_batch + [-100] * (4 - num_label_tokens_in_batch)
    batches = [
        {
            "labels": torch.tensor([seq]),
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
        }
    ]
    return recipe, batches


def _patch_pp_optim_step_dependencies(monkeypatch):
    monkeypatch.setattr(
        "nemo_automodel.recipes.vlm.finetune.scale_grads_and_clip_grad_norm",
        lambda **kwargs: 0.0,
    )
    monkeypatch.setattr(
        "nemo_automodel.recipes.vlm.finetune.prepare_for_grad_accumulation",
        lambda model_parts, pp_enabled: None,
    )
    monkeypatch.setattr(
        "nemo_automodel.recipes.vlm.finetune.prepare_for_final_backward",
        lambda model_parts, pp_enabled: None,
    )
    monkeypatch.setattr(
        "nemo_automodel.recipes.vlm.finetune.prepare_after_first_microbatch",
        lambda: None,
    )


@pytest.mark.cuda(False)
def test_run_train_step_clears_first_microbatch_after_first_batch(monkeypatch):
    recipe, _ = _build_pp_recipe_for_optim_step(num_label_tokens_in_batch=2)
    batches = [
        {
            "labels": torch.tensor([[1, -100, 2, -100]]),
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
        },
        {
            "labels": torch.tensor([[-100, 3, -100, 4]]),
            "input_ids": torch.tensor([[5, 6, 7, 8]]),
        },
    ]
    events = []

    monkeypatch.setattr(
        "nemo_automodel.recipes.vlm.finetune.prepare_for_grad_accumulation",
        lambda model_parts, pp_enabled: events.append("prepare"),
    )
    monkeypatch.setattr(
        "nemo_automodel.recipes.vlm.finetune.prepare_for_final_backward",
        lambda model_parts, pp_enabled: events.append("final"),
    )
    monkeypatch.setattr(
        "nemo_automodel.recipes.vlm.finetune.prepare_after_first_microbatch",
        lambda: events.append("after_first"),
    )
    monkeypatch.setattr(
        "nemo_automodel.recipes.vlm.finetune.scale_grads_and_clip_grad_norm",
        lambda **kwargs: 0.0,
    )

    def fake_forward_backward_step(idx, batch, loss_buffer, num_label_tokens, num_batches):
        events.append(f"forward_{idx}")
        loss_buffer.append(torch.tensor(1.0))

    recipe._forward_backward_step = fake_forward_backward_step

    recipe._run_train_optim_step(batches, max_grad_norm=1.0)

    assert events == ["prepare", "forward_0", "after_first", "final", "forward_1"]


@pytest.mark.cuda(False)
def test_run_train_step_pp_zero_label_tokens_no_nan(monkeypatch):
    """Regression for PR #1985: PP reporting loss must be 0.0 (not NaN) when num_label_tokens=0.

    With pipeline parallelism enabled, _run_train_optim_step divides reporting_loss by
    num_label_tokens. If every label in the batch is the ignore_index (-100), the divisor
    is zero and the reported metric would be NaN without the guard at finetune.py:1136.
    """
    recipe, batches = _build_pp_recipe_for_optim_step(num_label_tokens_in_batch=0)

    def fake_forward_backward_step(idx, batch, loss_buffer, num_label_tokens, num_batches):
        # Mirror the PP path: append a finite per-microbatch sum loss. With the guard,
        # this must still yield reporting_loss == 0.0.
        loss_buffer.append(torch.tensor(5.0))

    recipe._forward_backward_step = fake_forward_backward_step
    _patch_pp_optim_step_dependencies(monkeypatch)

    metrics = recipe._run_train_optim_step(batches, max_grad_norm=1.0)

    assert isinstance(metrics, MetricsSample)
    assert metrics.metrics["num_label_tokens"] == 0
    loss = metrics.metrics["loss"]
    assert loss == loss, f"reporting loss must not be NaN, got {loss}"
    assert loss == 0.0, f"reporting loss must be 0.0 when num_label_tokens=0, got {loss}"


@pytest.mark.cuda(False)
def test_run_train_step_pp_nonzero_label_tokens_divides(monkeypatch):
    """PP reporting loss is the summed microbatch loss divided by num_label_tokens."""
    recipe, batches = _build_pp_recipe_for_optim_step(num_label_tokens_in_batch=4)

    def fake_forward_backward_step(idx, batch, loss_buffer, num_label_tokens, num_batches):
        loss_buffer.append(torch.tensor(8.0))

    recipe._forward_backward_step = fake_forward_backward_step
    _patch_pp_optim_step_dependencies(monkeypatch)

    metrics = recipe._run_train_optim_step(batches, max_grad_norm=1.0)

    assert metrics.metrics["num_label_tokens"] == 4
    assert metrics.metrics["loss"] == pytest.approx(8.0 / 4)


# -----------------------------------------------------------------------------
# AutoProcessor exception handling test
# -----------------------------------------------------------------------------


def _test_vlm_dataset(path_or_dataset=None, split=None):
    return [{"path": path_or_dataset, "split": split}]


def _test_vlm_collate(examples, processor=None):
    return examples


def _vlm_dataloader_cfg():
    return ConfigNode(
        {
            "_target_": "torchdata.stateful_dataloader.StatefulDataLoader",
            "collate_fn": {"_target_": _test_vlm_collate},
            "num_workers": 0,
        }
    )


def test_autoprocessor_success():
    """Test successful AutoProcessor creation."""

    with patch("transformers.AutoProcessor") as mock_auto_processor:
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

    with (
        patch("transformers.AutoProcessor.from_pretrained") as mock_from_pretrained,
        patch("nemo_automodel.components.training.rng.StatefulRNG"),
        patch("torch.utils.data.distributed.DistributedSampler"),
        patch("nemo_automodel.components.datasets.vlm.collate_fns.COLLATE_FNS", {"NoneType": MagicMock()}),
    ):
        # Set up the exception
        mock_from_pretrained.side_effect = Exception("Model does not have AutoProcessor")

        # Mock configurations - minimal setup
        cfg_ds = ConfigNode({"_target_": _test_vlm_dataset, "path_or_dataset": "test/dataset"})
        cfg_dl = _vlm_dataloader_cfg()

        cfg_processor = None  # This triggers the exception path

        with caplog.at_level(logging.WARNING):
            dataloader, processor = build_dataloader(cfg_ds, cfg_dl, "test/model", cfg_processor, None, 123, 1)

        # Verify the results
        assert processor is None
        mock_from_pretrained.assert_called_once_with("test/model")


def test_autoprocessor_retries_on_layer_types_mismatch():
    """On StrictDataclassClassValidationError from validate_layer_type,
    relax the validator globally and retry AutoProcessor.from_pretrained once."""
    from huggingface_hub.errors import StrictDataclassClassValidationError

    from nemo_automodel.recipes.vlm.finetune import build_dataloader

    stub_processor = MagicMock()
    calls = {"n": 0}

    def fake_from_pretrained(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            cause = ValueError("`num_hidden_layers` (45) must be equal to the number of layer types (48).")
            raise StrictDataclassClassValidationError(validator="validate_layer_type", cause=cause)
        return stub_processor

    with (
        patch("transformers.AutoProcessor.from_pretrained", side_effect=fake_from_pretrained),
        patch(
            "nemo_automodel._transformers.v4_patches.layer_types.relax_layer_types_validator", return_value=True
        ) as mock_relax,
        patch("nemo_automodel.components.training.rng.StatefulRNG"),
        patch("torch.utils.data.distributed.DistributedSampler"),
        patch("nemo_automodel.components.datasets.vlm.collate_fns.COLLATE_FNS", {"MagicMock": MagicMock()}),
    ):
        cfg_ds = ConfigNode({"_target_": _test_vlm_dataset, "path_or_dataset": "test/dataset"})
        cfg_dl = _vlm_dataloader_cfg()

        dataloader, processor = build_dataloader(cfg_ds, cfg_dl, "stepfun-ai/Step-3.5-Flash", None, None, 123, 1)

        assert processor is stub_processor
        assert calls["n"] == 2
        mock_relax.assert_called_once()


def test_autoprocessor_loads_inside_first_rank_per_node():
    """Test that processor instantiation happens inside the FirstRankPerNode context."""

    from nemo_automodel.recipes.vlm.finetune import build_dataloader

    call_order = []

    class TrackingFirstRankPerNode:
        def __enter__(self):
            call_order.append("enter_first_rank")
            return self

        def __exit__(self, *args):
            call_order.append("exit_first_rank")
            return False

    def tracking_from_pretrained(*args, **kwargs):
        call_order.append("autoprocessor")
        return MagicMock()

    with (
        patch("nemo_automodel.recipes.vlm.finetune.FirstRankPerNode", TrackingFirstRankPerNode),
        patch("transformers.AutoProcessor.from_pretrained", side_effect=tracking_from_pretrained),
        patch("nemo_automodel.components.training.rng.StatefulRNG"),
        patch("torch.utils.data.distributed.DistributedSampler"),
        patch("nemo_automodel.components.datasets.vlm.collate_fns.COLLATE_FNS", {"NoneType": MagicMock()}),
    ):
        cfg_ds = ConfigNode({"_target_": _test_vlm_dataset, "path_or_dataset": "test/dataset"})
        cfg_dl = _vlm_dataloader_cfg()

        build_dataloader(cfg_ds, cfg_dl, "test/model", None, None, 123, 1)

    assert "enter_first_rank" in call_order
    assert "autoprocessor" in call_order
    assert "exit_first_rank" in call_order
    first_rank_idx = call_order.index("enter_first_rank")
    processor_idx = call_order.index("autoprocessor")
    exit_idx = call_order.index("exit_first_rank")
    assert first_rank_idx < processor_idx < exit_idx, (
        f"AutoProcessor must load inside FirstRankPerNode context, got order: {call_order}"
    )


def test_autoprocessor_with_processor_kwargs(caplog):
    """Test AutoProcessor exception handling when cfg_processor has no instantiate method."""
    import logging

    from nemo_automodel.recipes.vlm.finetune import build_dataloader

    # Simple processor config class without instantiate method
    class ProcessorConfig:
        def to_dict(self):
            return {"trust_remote_code": True, "some_param": "value"}

    with (
        patch("transformers.AutoProcessor.from_pretrained") as mock_from_pretrained,
        patch("nemo_automodel.components.training.rng.StatefulRNG"),
        patch("torch.utils.data.distributed.DistributedSampler"),
        patch("nemo_automodel.components.datasets.vlm.collate_fns.COLLATE_FNS", {"NoneType": MagicMock()}),
    ):
        # Set up the exception
        mock_from_pretrained.side_effect = Exception("Model does not have AutoProcessor")

        # Mock configurations - minimal setup
        cfg_ds = ConfigNode({"_target_": _test_vlm_dataset, "path_or_dataset": "test/dataset"})
        cfg_dl = _vlm_dataloader_cfg()

        cfg_processor = ProcessorConfig()  # This has to_dict but no instantiate

        with caplog.at_level(logging.WARNING):
            dataloader, processor = build_dataloader(cfg_ds, cfg_dl, "test/model", cfg_processor, None, 123, 1)

        # Verify the results
        assert processor is None
        mock_from_pretrained.assert_called_once_with("test/model", trust_remote_code=True, some_param="value")


# -----------------------------------------------------------------------------
# chat_template override tests for build_dataloader
# -----------------------------------------------------------------------------


def test_build_dataloader_chat_template_applied():
    """chat_template in dataset config is applied to processor and not leaked to dataset target."""
    from nemo_automodel.recipes.vlm.finetune import build_dataloader

    ds_calls = []

    def ds_factory(path_or_dataset, split=None):
        ds_calls.append({"path_or_dataset": path_or_dataset, "split": split})
        return [{}]

    class DummyProcessor:
        def __init__(self):
            self.chat_template = "{{ default }}"
            self.tokenizer = SimpleNamespace(chat_template="{{ default }}")

    processor = DummyProcessor()
    cfg_ds = ConfigNode(
        {"_target_": ds_factory, "path_or_dataset": "ds/path", "split": "train", "chat_template": "{{ custom }}"}
    )
    cfg_dl = _vlm_dataloader_cfg()

    with (
        pytest.warns(DeprecationWarning, match="RecipeConfig.vlm_dataloader"),
        patch("transformers.AutoProcessor.from_pretrained", return_value=processor),
        patch("torch.utils.data.distributed.DistributedSampler"),
        patch("nemo_automodel.components.datasets.vlm.collate_fns.COLLATE_FNS", {"default": MagicMock()}),
    ):
        _, built_processor = build_dataloader(cfg_ds, cfg_dl, "model", None, None, 42, 1)

    assert built_processor.chat_template == "{{ custom }}"
    assert built_processor.tokenizer.chat_template == "{{ custom }}"
    assert ds_calls == [{"path_or_dataset": "ds/path", "split": "train"}]


def test_build_dataloader_no_chat_template():
    """Without chat_template, processor template stays unchanged."""
    from nemo_automodel.recipes.vlm.finetune import build_dataloader

    def ds_factory(path_or_dataset, split=None):
        return [{}]

    class DummyProcessor:
        def __init__(self):
            self.chat_template = "{{ original }}"
            self.tokenizer = SimpleNamespace(chat_template="{{ original }}")

    processor = DummyProcessor()
    cfg_ds = ConfigNode({"_target_": ds_factory, "path_or_dataset": "ds/path", "split": "train"})
    cfg_dl = _vlm_dataloader_cfg()

    with (
        patch("transformers.AutoProcessor.from_pretrained", return_value=processor),
        patch("torch.utils.data.distributed.DistributedSampler"),
        patch("nemo_automodel.components.datasets.vlm.collate_fns.COLLATE_FNS", {"default": MagicMock()}),
    ):
        _, built_processor = build_dataloader(cfg_ds, cfg_dl, "model", None, None, 42, 1)

    assert built_processor.chat_template == "{{ original }}"
    assert built_processor.tokenizer.chat_template == "{{ original }}"


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

    with patch("nemo_automodel.recipes.vlm.finetune._supports_logits_to_keep", return_value=True):
        model = build_model(
            cfg_model=cfg_model,
            cfg_freeze=None,
            cfg_peft=None,
            seed=123,
        )

    # Model should be instantiated with adapter
    assert model is not None
    assert hasattr(model, "state_dict_adapter")


def test_vlm_build_model_without_adapter():
    """Test that model without state_dict_adapter is properly instantiated in VLM."""

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

    with patch("nemo_automodel.recipes.vlm.finetune._supports_logits_to_keep", return_value=True):
        model = build_model(
            cfg_model=cfg_model,
            cfg_freeze=None,
            cfg_peft=None,
            seed=123,
        )

    # Model should be instantiated without adapter
    assert model is not None
    assert not hasattr(model, "state_dict_adapter")


def test_vlm_build_model_with_quantization_config():
    """Test that model with quantization_config is properly instantiated in VLM."""
    from nemo_automodel._transformers import NeMoAutoModelForImageTextToText

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

    with patch("nemo_automodel.recipes.vlm.finetune._supports_logits_to_keep", return_value=True):
        model = build_model(
            cfg_model=cfg_model,
            cfg_freeze=None,
            cfg_peft=None,
            seed=123,
        )

    # Model should be instantiated with quantization config
    assert model is not None
    assert hasattr(model.config, "quantization_config")


def test_vlm_build_model_without_quantization_config():
    """Test that model without quantization_config is properly instantiated in VLM."""
    from nemo_automodel._transformers import NeMoAutoModelForImageTextToText

    # Create a config that simulates NeMoAutoModel's internal infrastructure handling (no quant config)
    class DummyNoQuantVLMModelConfig:
        def __init__(self):
            self._target_ = NeMoAutoModelForImageTextToText.from_pretrained

        def instantiate(self, **kwargs):
            return DummyModel()  # DummyModel has no config.quantization_config

        def get(self, key, default=None):
            return getattr(self, key, default)

    cfg_model = DummyNoQuantVLMModelConfig()

    with patch("nemo_automodel.recipes.vlm.finetune._supports_logits_to_keep", return_value=True):
        model = build_model(
            cfg_model=cfg_model,
            cfg_freeze=None,
            cfg_peft=None,
            seed=123,
        )

    # Model should be instantiated without quantization config
    assert model is not None
    assert not hasattr(model.config, "quantization_config")


# =============================================================================
# New tests for VLM-specific build_model / build_optimizer functionality
# =============================================================================


def test_vlm_build_model_raises_value_error_for_non_nemo_auto_model():
    """Test that VLM build_model raises ValueError when target is not NeMoAutoModelForImageTextToText."""

    # Create a cfg_model that targets something other than NeMoAutoModelForImageTextToText
    class InvalidModelConfig:
        def __init__(self):
            self._target_ = "some.invalid.Target"

        def instantiate(self, **kwargs):
            return DummyModel()

        def get(self, key, default=None):
            return getattr(self, key, default)

    cfg_model = InvalidModelConfig()

    with pytest.raises(ValueError, match="VLM finetuning requires a recipe-compatible model target"):
        build_model(
            cfg_model=cfg_model,
            cfg_freeze=None,
            cfg_peft=None,
            seed=42,
        )


from nemo_automodel.recipes.vlm.finetune import calculate_loss

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

    def test_build_step_scheduler_ignores_local_batch_size_in_yaml_block(self):
        """Real YAML step_scheduler blocks carry local_batch_size (a runtime arg
        passed separately); it must not crash StepSchedulerConfig construction."""
        mock_dataloader = MagicMock()
        mock_dataloader.__len__ = MagicMock(return_value=50)

        cfg = MagicMock()
        cfg.to_dict.return_value = {
            "global_batch_size": 256,
            "local_batch_size": 2,  # runtime arg present in the YAML block
            "ckpt_every_steps": 50,
        }

        step_scheduler = build_step_scheduler(
            cfg=cfg,
            dataloader=mock_dataloader,
            dp_group_size=4,
            local_batch_size=2,
        )

        # global_batch_size (256, from config) // (local_batch_size 2 * dp_size 4) = 32
        assert step_scheduler.grad_acc_steps == 32
        assert step_scheduler.ckpt_every_steps == 50

    def test_build_step_scheduler_ignores_target(self):
        """``_target_`` in the step_scheduler block is dropped by the typed boundary
        (RecipeConfig.step_scheduler), not passed into StepSchedulerConfig."""
        mock_dataloader = MagicMock()
        mock_dataloader.__len__ = MagicMock(return_value=100)

        cfg = {"_target_": "some.class"}

        step_scheduler = build_step_scheduler(
            cfg=cfg,
            dataloader=mock_dataloader,
            dp_group_size=1,
            local_batch_size=1,
        )
        assert step_scheduler is not None


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
        step_scheduler.epoch_len = 100  # ceil(len(dataloader)=100 / grad_acc=1)
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
        step_scheduler.epoch_len = 50  # ceil(len(dataloader)=100 / grad_acc=2)
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
        step_scheduler.epoch_len = 1000  # ceil(len(dataloader)=1000 / grad_acc=1)
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
        assert config.save_consolidated.value == "final"
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
        assert config.save_consolidated.value == "false"
        assert config.is_peft is True

    def test_build_checkpoint_config_warns_on_peft_with_torch_save(self, caplog):
        """PEFT + torch_save: warn, fall back to safetensors defaults; preserve checkpoint_dir."""
        from nemo_automodel.components.checkpoint._backports.filesystem import SerializationFormat

        cfg_ckpt = MagicMock()
        cfg_ckpt.to_dict.return_value = {
            "model_save_format": "torch_save",
            "checkpoint_dir": "/user/ckpt/",
            "save_consolidated": False,
        }

        with caplog.at_level("WARNING"):
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

    def update_seq_len(self, seq_len: int) -> None:
        # Dynamic seq-len hook is a no-op in tests; AutoPipeline exposes this for
        # variable-length VLM batches.
        return None


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
    recipe.__dict__["distributed_config"] = None
    recipe.__dict__["model_parts"] = [model]
    recipe.__dict__["_get_dp_group_size"] = lambda include_cp=True: 1
    return recipe


def _prepare_pp_vlm_batch(batch, n_microbatches=2):
    return prepare_vlm_media_for_pp(
        batch,
        batch_size=batch["input_ids"].shape[0],
        n_microbatches=n_microbatches,
    )


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
            "nemo_automodel.components.distributed.cp_utils._make_cp_batch_and_ctx",
            lambda device_mesh, batch, *a, **k: (lambda: nullcontext(), batch, None),
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
            "nemo_automodel.components.distributed.cp_utils._make_cp_batch_and_ctx",
            lambda device_mesh, batch, *a, **k: (lambda: nullcontext(), batch, None),
        )

        batch_size = 4
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
        _prepare_pp_vlm_batch(batch)
        loss_buffer = []
        captured_chunks = {}

        def step_side_effect(*args, **kwargs):
            model = pp_recipe.model_parts[0]
            captured_chunks["pixel_values"] = [chunk.clone() for chunk in model._vlm_pixel_values_chunks]
            captured_chunks["image_grid"] = [chunk.clone() for chunk in model._vlm_image_grid_hws_chunks]
            captured_chunks["chunk_idx"] = model._vlm_chunk_idx
            for _ in range(2):
                kwargs["losses"].append(torch.tensor(0.5))

        pp_recipe.pp.info.schedule.step = MagicMock(side_effect=step_side_effect)

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
        assert captured_chunks["chunk_idx"] == 0
        assert torch.equal(captured_chunks["pixel_values"][0], pixel_values[:13])
        assert torch.equal(captured_chunks["pixel_values"][1], pixel_values[13:])
        assert torch.equal(captured_chunks["image_grid"][0], image_grid_hws[:2])
        assert torch.equal(captured_chunks["image_grid"][1], image_grid_hws[2:])
        assert model._vlm_pixel_values_chunks is None  # Cleared after step
        assert model._vlm_image_grid_hws_chunks is None
        assert model._vlm_chunk_idx is None

        # Verify schedule.step was called
        pp_recipe.pp.info.schedule.step.assert_called_once()

        # Verify loss was computed
        assert len(loss_buffer) == 1

    def test_pp_vlm_chunking_videos_uses_video_grid_and_counts(self, pp_recipe, monkeypatch):
        """Video tensors are chunked by per-sample video counts before schedule.step."""
        pp_recipe.pp = _MockAutoPipeline(has_first_stage=True, has_last_stage=True, n_microbatches=2)

        monkeypatch.setattr(
            "nemo_automodel.components.distributed.cp_utils._make_cp_batch_and_ctx",
            lambda device_mesh, batch, *a, **k: (lambda: nullcontext(), batch, None),
        )

        batch_size = 4
        video_grid_thw = torch.tensor([[1, 2, 2], [1, 3, 3], [1, 2, 3], [1, 4, 4]])
        pixel_values_videos = torch.randn(int(video_grid_thw.prod(dim=1).sum().item()), 64)
        n_videos_per_sample = torch.tensor([1, 0, 2, 1])

        def step_side_effect(*args, **kwargs):
            model = pp_recipe.model_parts[0]
            assert "pixel_values_videos" not in kwargs
            assert "video_grid_thw" not in kwargs
            assert len(model._vlm_pixel_values_videos_chunks) == 2
            assert len(model._vlm_video_grid_thw_chunks) == 2
            assert model._vlm_video_grid_thw_chunks[0].shape[0] == 1
            assert model._vlm_video_grid_thw_chunks[1].shape[0] == 3
            assert model._vlm_pixel_values_videos_chunks[0].shape[0] == 4
            assert model._vlm_pixel_values_videos_chunks[1].shape[0] == 9 + 6 + 16
            for _ in range(2):
                kwargs["losses"].append(torch.tensor(0.5))

        pp_recipe.pp.info.schedule.step.side_effect = step_side_effect

        batch = {
            "labels": torch.randint(0, 100, (batch_size, 10)),
            "input_ids": torch.randint(0, 100, (batch_size, 10)),
            "pixel_values_videos": pixel_values_videos,
            "video_grid_thw": video_grid_thw,
            "n_videos_per_sample": n_videos_per_sample,
        }
        _prepare_pp_vlm_batch(batch)
        loss_buffer = []

        pp_recipe._forward_backward_step(
            idx=0,
            batch=batch,
            loss_buffer=loss_buffer,
            num_label_tokens=40,
            num_batches=1,
            is_train=True,
        )

        model = pp_recipe.model_parts[0]
        assert model._vlm_pixel_values_videos_chunks is None
        assert model._vlm_video_grid_thw_chunks is None
        assert model._vlm_chunk_idx is None
        assert len(loss_buffer) == 1

    def test_pp_vlm_chunking_image_and_video_mixed(self, pp_recipe, monkeypatch):
        """When a batch carries both images and videos, both streams chunk independently
        but share a single _vlm_chunk_idx initialized once at 0; both clean up to None."""
        pp_recipe.pp = _MockAutoPipeline(has_first_stage=True, has_last_stage=True, n_microbatches=2)

        monkeypatch.setattr(
            "nemo_automodel.components.distributed.cp_utils._make_cp_batch_and_ctx",
            lambda device_mesh, batch, *a, **k: (lambda: nullcontext(), batch, None),
        )

        batch_size = 4

        # n_images_per_sample=[2,0,1,0]: mb0 (samples 0..1) covers images 0..1; mb1 covers image 2.
        image_grid_thw = torch.tensor([[1, 2, 2], [1, 3, 3], [1, 2, 3]])  # patch counts: 4, 9, 6
        pixel_values = torch.randn(int(image_grid_thw.prod(dim=1).sum().item()), 32)
        n_images_per_sample = torch.tensor([2, 0, 1, 0])

        # n_videos_per_sample=[1,0,2,1]: mb0 covers video 0; mb1 covers videos 1..3.
        video_grid_thw = torch.tensor([[1, 2, 2], [1, 3, 3], [1, 2, 3], [1, 4, 4]])  # patch counts: 4, 9, 6, 16
        pixel_values_videos = torch.randn(int(video_grid_thw.prod(dim=1).sum().item()), 64)
        n_videos_per_sample = torch.tensor([1, 0, 2, 1])

        def step_side_effect(*args, **kwargs):
            model = pp_recipe.model_parts[0]

            # Both modalities are popped before schedule.step so the schedule never
            # tries to chunk the misaligned multimodal tensors along dim 0.
            assert "pixel_values" not in kwargs
            assert "image_grid_hws" not in kwargs
            assert "image_grid_thw" not in kwargs
            assert "pixel_values_videos" not in kwargs
            assert "video_grid_thw" not in kwargs

            assert len(model._vlm_pixel_values_chunks) == 2
            assert len(model._vlm_image_grid_hws_chunks) == 2
            assert model._vlm_image_grid_hws_chunks[0].shape[0] == 2
            assert model._vlm_image_grid_hws_chunks[1].shape[0] == 1
            assert model._vlm_pixel_values_chunks[0].shape[0] == 4 + 9
            assert model._vlm_pixel_values_chunks[1].shape[0] == 6

            assert len(model._vlm_pixel_values_videos_chunks) == 2
            assert len(model._vlm_video_grid_thw_chunks) == 2
            assert model._vlm_video_grid_thw_chunks[0].shape[0] == 1
            assert model._vlm_video_grid_thw_chunks[1].shape[0] == 3
            assert model._vlm_pixel_values_videos_chunks[0].shape[0] == 4
            assert model._vlm_pixel_values_videos_chunks[1].shape[0] == 9 + 6 + 16

            # Single shared cursor: image-branch sets it to 0 first, video branch resets to 0 again.
            assert model._vlm_chunk_idx == 0

            for _ in range(2):
                kwargs["losses"].append(torch.tensor(0.5))

        pp_recipe.pp.info.schedule.step.side_effect = step_side_effect

        batch = {
            "labels": torch.randint(0, 100, (batch_size, 10)),
            "input_ids": torch.randint(0, 100, (batch_size, 10)),
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "n_images_per_sample": n_images_per_sample,
            "pixel_values_videos": pixel_values_videos,
            "video_grid_thw": video_grid_thw,
            "n_videos_per_sample": n_videos_per_sample,
        }
        _prepare_pp_vlm_batch(batch)
        loss_buffer = []

        pp_recipe._forward_backward_step(
            idx=0,
            batch=batch,
            loss_buffer=loss_buffer,
            num_label_tokens=40,
            num_batches=1,
            is_train=True,
        )

        model = pp_recipe.model_parts[0]
        assert model._vlm_pixel_values_chunks is None
        assert model._vlm_image_grid_hws_chunks is None
        assert model._vlm_pixel_values_videos_chunks is None
        assert model._vlm_video_grid_thw_chunks is None
        assert model._vlm_chunk_idx is None
        assert len(loss_buffer) == 1

    def test_pp_vlm_chunking_with_image_grid_thw(self, pp_recipe, monkeypatch):
        """Test VLM pixel_values chunking with image_grid_thw (3D grid) instead of image_grid_hws."""
        pp_recipe.pp = _MockAutoPipeline(has_first_stage=True, has_last_stage=True, n_microbatches=2)

        monkeypatch.setattr(
            "nemo_automodel.components.distributed.cp_utils._make_cp_batch_and_ctx",
            lambda device_mesh, batch, *a, **k: (lambda: nullcontext(), batch, None),
        )

        batch_size = 4
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
        _prepare_pp_vlm_batch(batch)
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

    def test_pp_vlm_chunking_qwen35_ep4_pp2_local_batch_images(self, pp_recipe, monkeypatch):
        """Qwen3.5 35B EP4/PP2-style local batch keeps proper image chunks during schedule.step."""
        pp_recipe.pp = _MockAutoPipeline(has_first_stage=True, has_last_stage=True, n_microbatches=2)

        monkeypatch.setattr(
            "nemo_automodel.components.distributed.cp_utils._make_cp_batch_and_ctx",
            lambda device_mesh, batch, *a, **k: (lambda: nullcontext(), batch, None),
        )

        image_grid_thw = torch.tensor([[1, 2, 2], [1, 3, 3]])
        patch_counts = image_grid_thw.prod(dim=1)
        pixel_values = torch.arange(int(patch_counts.sum()) * 4, dtype=torch.float32).reshape(-1, 4)
        batch = {
            "labels": torch.randint(0, 100, (2, 10)),
            "input_ids": torch.randint(0, 100, (2, 10)),
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "n_images_per_sample": torch.tensor([1, 1]),
        }
        _prepare_pp_vlm_batch(batch)
        loss_buffer = []
        captured_chunks = {}

        def step_side_effect(*args, **kwargs):
            model = pp_recipe.model_parts[0]
            captured_chunks["pixel_values"] = [chunk.clone() for chunk in model._vlm_pixel_values_chunks]
            captured_chunks["image_grid"] = [chunk.clone() for chunk in model._vlm_image_grid_hws_chunks]
            for _ in range(2):
                kwargs["losses"].append(torch.tensor(0.5))

        pp_recipe.pp.info.schedule.step = MagicMock(side_effect=step_side_effect)

        pp_recipe._forward_backward_step(
            idx=0,
            batch=batch,
            loss_buffer=loss_buffer,
            num_label_tokens=20,
            num_batches=1,
            is_train=True,
        )

        split_at = int(patch_counts[0].item())
        assert torch.equal(captured_chunks["pixel_values"][0], pixel_values[:split_at])
        assert torch.equal(captured_chunks["pixel_values"][1], pixel_values[split_at:])
        assert torch.equal(captured_chunks["image_grid"][0], image_grid_thw[:1])
        assert torch.equal(captured_chunks["image_grid"][1], image_grid_thw[1:])
        assert pp_recipe.model_parts[0]._vlm_pixel_values_chunks is None
        assert len(loss_buffer) == 1

    def test_pp_vlm_chunking_mismatched_images_raises(self):
        """When media cannot be aligned to samples, VLM PP data prep raises."""
        batch_size = 4
        image_grid_hws = torch.tensor([[2, 2], [3, 3]])
        total_patches = 4 + 9
        pixel_values = torch.randn(total_patches, 3, 14, 14)

        batch = {
            "labels": torch.randint(0, 100, (batch_size, 10)),
            "input_ids": torch.randint(0, 100, (batch_size, 10)),
            "pixel_values": pixel_values,
            "image_grid_hws": image_grid_hws,
        }

        with pytest.raises(ValueError, match="VLM PP chunking cannot align"):
            _prepare_pp_vlm_batch(batch)

    def test_pp_vlm_chunking_with_image_sizes(self, pp_recipe, monkeypatch):
        """Test VLM pixel_values chunking with image_sizes fallback (e.g., Mistral4-style)."""
        pp_recipe.pp = _MockAutoPipeline(has_first_stage=True, has_last_stage=True, n_microbatches=2)

        monkeypatch.setattr(
            "nemo_automodel.components.distributed.cp_utils._make_cp_batch_and_ctx",
            lambda device_mesh, batch, *a, **k: (lambda: nullcontext(), batch, None),
        )

        batch_size = 4
        # image_sizes: [N_images, 2] — no image_grid_hws or image_grid_thw
        image_sizes = torch.tensor([[224, 224], [224, 224], [224, 224], [224, 224]])
        # 4D pixel_values: [N_images, C, H, W]
        pixel_values = torch.randn(batch_size, 3, 224, 224)

        batch = {
            "labels": torch.randint(0, 100, (batch_size, 10)),
            "input_ids": torch.randint(0, 100, (batch_size, 10)),
            "pixel_values": pixel_values,
            "image_sizes": image_sizes,
        }
        _prepare_pp_vlm_batch(batch)
        loss_buffer = []

        pp_recipe._forward_backward_step(
            idx=0,
            batch=batch,
            loss_buffer=loss_buffer,
            num_label_tokens=40,
            num_batches=1,
            is_train=True,
        )

        model = pp_recipe.model_parts[0]
        assert model._vlm_pixel_values_chunks is None  # Cleared after step
        assert model._vlm_image_grid_hws_chunks is None
        assert model._vlm_chunk_idx is None
        pp_recipe.pp.info.schedule.step.assert_called_once()
        assert len(loss_buffer) == 1

    def test_pp_vlm_chunking_4d_pixel_values(self, pp_recipe, monkeypatch):
        """Test VLM pixel_values chunking when pixel_values is 4D [N, C, H, W]."""
        pp_recipe.pp = _MockAutoPipeline(has_first_stage=True, has_last_stage=True, n_microbatches=2)

        monkeypatch.setattr(
            "nemo_automodel.components.distributed.cp_utils._make_cp_batch_and_ctx",
            lambda device_mesh, batch, *a, **k: (lambda: nullcontext(), batch, None),
        )

        batch_size = 4
        image_grid_hws = torch.tensor([[224, 224], [224, 224], [224, 224], [224, 224]])
        # 4D pixel_values — triggers the new dim==4 chunking path
        pixel_values = torch.randn(batch_size, 3, 224, 224)

        batch = {
            "labels": torch.randint(0, 100, (batch_size, 10)),
            "input_ids": torch.randint(0, 100, (batch_size, 10)),
            "pixel_values": pixel_values,
            "image_grid_hws": image_grid_hws,
        }
        _prepare_pp_vlm_batch(batch)
        loss_buffer = []

        pp_recipe._forward_backward_step(
            idx=0,
            batch=batch,
            loss_buffer=loss_buffer,
            num_label_tokens=40,
            num_batches=1,
            is_train=True,
        )

        model = pp_recipe.model_parts[0]
        assert model._vlm_pixel_values_chunks is None  # Cleared after step
        assert model._vlm_image_grid_hws_chunks is None
        assert model._vlm_chunk_idx is None
        pp_recipe.pp.info.schedule.step.assert_called_once()
        assert len(loss_buffer) == 1

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
            "nemo_automodel.components.distributed.cp_utils._make_cp_batch_and_ctx",
            lambda device_mesh, batch, *a, **k: (lambda: nullcontext(), batch, None),
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
            "nemo_automodel.components.distributed.cp_utils._make_cp_batch_and_ctx",
            lambda device_mesh, batch, *a, **k: (lambda: nullcontext(), batch, None),
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
            "nemo_automodel.components.distributed.cp_utils._make_cp_batch_and_ctx",
            lambda device_mesh, batch, *a, **k: (lambda: nullcontext(), batch, None),
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
    recipe.__dict__["distributed_config"] = None
    recipe.__dict__["model_parts"] = [model]
    recipe.__dict__["_get_dp_group_size"] = lambda include_cp=True: 1
    # ``is_remote_logging_step`` is read by ``_forward_backward_step`` to
    # gate the joint-drafter loss-log line; default False so non-drafter
    # test paths don't trip on the new attribute.
    recipe.__dict__["step_scheduler"] = SimpleNamespace(is_remote_logging_step=False)
    return recipe


class _DummyCPSubMesh:
    def __init__(self, size: int):
        self._size = size

    def size(self) -> int:
        return self._size


class _DummyCPDeviceMesh(dict):
    def __init__(self, cp_size: int):
        super().__init__()
        self["cp"] = _DummyCPSubMesh(cp_size)
        self.mesh_dim_names = ["cp"]


class _CPPreEmbedModel(torch.nn.Module):
    def __init__(self, *, return_mm_token_type_ids: bool = True):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.return_mm_token_type_ids = return_mm_token_type_ids

    def prepare_model_inputs_for_cp(self, *args, **kwargs):
        raise AssertionError("prepare_model_inputs_for_cp should be invoked through model.__call__")

    def forward(self, *, _pre_embed_only=False, **kwargs):
        assert _pre_embed_only is True
        # Production contract: read the live batch from _cp_batch, remove the
        # raw inputs consumed into inputs_embeds.
        cp_batch = kwargs["_cp_batch"]
        input_ids = cp_batch["input_ids"]
        mm_token_type_ids = cp_batch.get("mm_token_type_ids")
        consumed = {k: None for k in ("input_ids", "pixel_values", "mm_token_type_ids", "image_position_ids")}
        cp_batch.pop("image_position_ids", None)
        batch, seq = input_ids.shape
        inputs_embeds = self.scale * torch.ones(batch, seq, 4)
        per_layer_inputs = self.scale * torch.ones(batch, seq, 2, 3)
        prepared = {
            **consumed,
            "inputs_embeds": inputs_embeds,
            "per_layer_inputs": per_layer_inputs,
        }
        if self.return_mm_token_type_ids and mm_token_type_ids is not None:
            prepared["mm_token_type_ids"] = mm_token_type_ids
        return prepared


class _CPPreEmbedStop(RuntimeError):
    pass


class TestForwardBackwardStepNonPP:
    """Tests for _forward_backward_step without pipeline parallelism."""

    def test_non_pp_cp_pre_embed_uses_model_returned_mm_token_type_ids_and_grad(self, monkeypatch):
        model = _CPPreEmbedModel()
        non_pp_recipe = _create_non_pp_recipe(model)
        non_pp_recipe.__dict__["device_mesh"] = _DummyCPDeviceMesh(cp_size=2)

        mm_token_type_ids = torch.tensor([[1, 1, 0, 0]])

        def _capture_cp_batch(device_mesh, batch, loss_mask=None, **kwargs):
            assert "input_ids" not in batch
            assert "pixel_values" not in batch
            assert "image_position_ids" not in batch
            assert "mm_token_type_ids" in batch
            assert "per_layer_inputs" in batch
            torch.testing.assert_close(batch["mm_token_type_ids"], mm_token_type_ids)
            assert batch["inputs_embeds"].requires_grad
            raise _CPPreEmbedStop

        monkeypatch.setattr(
            "nemo_automodel.components.distributed.cp_utils._make_cp_batch_and_ctx",
            _capture_cp_batch,
        )

        batch = {
            "labels": torch.randint(0, 50, (1, 4)),
            "input_ids": torch.randint(0, 100, (1, 4)),
            "pixel_values": torch.randn(1, 3, 8, 8),
            "image_position_ids": torch.zeros(1, 1, 2, dtype=torch.long),
            "mm_token_type_ids": mm_token_type_ids,
        }

        with pytest.raises(_CPPreEmbedStop):
            non_pp_recipe._forward_backward_step(
                idx=0,
                batch=batch,
                loss_buffer=[],
                num_label_tokens=4,
                num_batches=1,
                is_train=False,
            )

    def test_non_pp_cp_pre_embed_drops_mm_token_type_ids_when_model_does_not_return_it(self, monkeypatch):
        model = _CPPreEmbedModel(return_mm_token_type_ids=False)
        non_pp_recipe = _create_non_pp_recipe(model)
        non_pp_recipe.__dict__["device_mesh"] = _DummyCPDeviceMesh(cp_size=2)

        def _capture_cp_batch(device_mesh, batch, loss_mask=None, **kwargs):
            assert "input_ids" not in batch
            assert "pixel_values" not in batch
            assert "image_position_ids" not in batch
            assert "mm_token_type_ids" not in batch
            assert "per_layer_inputs" in batch
            assert batch["inputs_embeds"].requires_grad
            raise _CPPreEmbedStop

        monkeypatch.setattr(
            "nemo_automodel.components.distributed.cp_utils._make_cp_batch_and_ctx",
            _capture_cp_batch,
        )

        batch = {
            "labels": torch.randint(0, 50, (1, 4)),
            "input_ids": torch.randint(0, 100, (1, 4)),
            "pixel_values": torch.randn(1, 3, 8, 8),
            "image_position_ids": torch.zeros(1, 1, 2, dtype=torch.long),
            "mm_token_type_ids": torch.tensor([[1, 1, 0, 0]]),
        }

        with pytest.raises(_CPPreEmbedStop):
            non_pp_recipe._forward_backward_step(
                idx=0,
                batch=batch,
                loss_buffer=[],
                num_label_tokens=4,
                num_batches=1,
                is_train=False,
            )

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
            "nemo_automodel.components.distributed.cp_utils._make_cp_batch_and_ctx",
            lambda device_mesh, batch, *a, **k: (lambda: nullcontext(), batch, None),
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
            "nemo_automodel.components.distributed.cp_utils._make_cp_batch_and_ctx",
            lambda device_mesh, batch, *a, **k: (lambda: nullcontext(), batch, None),
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
            "nemo_automodel.components.distributed.cp_utils._make_cp_batch_and_ctx",
            lambda device_mesh, batch, *a, **k: (lambda: nullcontext(), batch, None),
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
            "nemo_automodel.components.distributed.cp_utils._make_cp_batch_and_ctx",
            lambda device_mesh, batch, *a, **k: (lambda: nullcontext(), batch, None),
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
            "nemo_automodel.components.distributed.cp_utils._make_cp_batch_and_ctx",
            lambda device_mesh, batch, *a, **k: (lambda: nullcontext(), batch, None),
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


# -----------------------------------------------------------------------------
# build_optimizer returns correct type (diff coverage)
# -----------------------------------------------------------------------------


def test_vlm_build_model_and_optimizer_return_values():
    """Test that VLM build_model and build_optimizer return proper values."""
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

    with patch("nemo_automodel.recipes.vlm.finetune._supports_logits_to_keep", return_value=True):
        model = build_model(
            cfg_model=cfg_model,
            cfg_freeze=None,
            cfg_peft=None,
            seed=42,
        )
        optimizer = build_optimizer(model, cfg_opt, None, None)

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

    with patch("nemo_automodel.recipes.vlm.finetune._supports_logits_to_keep", return_value=True):
        # Should not raise - entry point should be recognized
        model = build_model(
            cfg_model=cfg_model,
            cfg_freeze=None,
            cfg_peft=None,
            seed=42,
        )

    assert model is not None


@pytest.mark.parametrize("entry_point", ["from_config", "from_pretrained"])
def test_vlm_build_model_accepts_multimodal_lm_entry_points(entry_point):
    """Test that VLM build_model accepts NeMoAutoModelForMultimodalLM entry points."""
    from nemo_automodel._transformers import NeMoAutoModelForMultimodalLM

    target = getattr(NeMoAutoModelForMultimodalLM, entry_point)

    class NeMoVLMModelConfig:
        def __init__(self):
            self._target_ = target

        def instantiate(self, **kwargs):
            return DummyModel()

        def get(self, key, default=None):
            return getattr(self, key, default)

    cfg_model = NeMoVLMModelConfig()

    with patch("nemo_automodel.recipes.vlm.finetune._supports_logits_to_keep", return_value=True):
        model = build_model(
            cfg_model=cfg_model,
            cfg_freeze=None,
            cfg_peft=None,
            seed=42,
        )

    assert model is not None


_GEMMA4_COMPOSITE_MOD = "nemo_automodel.components.models.gemma4_drafter.composite"


def _target_owner_names(targets):
    """Class names of the objects that own each allowlisted classmethod target."""
    names = set()
    for t in targets:
        owner = getattr(t, "__self__", None)
        if owner is not None:
            names.add(getattr(owner, "__name__", str(owner)))
    return names


class TestRecipeTargetAllowlist:
    """Coverage for the recipe-side model-target allowlist that ``build_model``
    gates on (``_accepted_targets`` / ``_is_recipe_target`` in
    ``recipes/vlm/finetune.py``).

    ``_accepted_targets`` adds the optional Gemma4 composite behind a
    ``try/except ImportError``. Existing ``build_model`` tests only exercise
    whichever branch matches the installed deps, so these tests force *both*
    branches (import present and absent) deterministically, plus the
    ``target is None`` short-circuit -- all without depending on whether the
    optional ``transformers.models.gemma4_assistant`` dep is installed.
    """

    def test_accepted_targets_contains_all_nemo_auto_entrypoints(self):
        from nemo_automodel._transformers import (
            NeMoAutoModelForCausalLM,
            NeMoAutoModelForImageTextToText,
            NeMoAutoModelForMultimodalLM,
        )
        from nemo_automodel.recipes.vlm.finetune import _accepted_targets

        targets = _accepted_targets()
        assert isinstance(targets, set)
        for cls in (
            NeMoAutoModelForCausalLM,
            NeMoAutoModelForImageTextToText,
            NeMoAutoModelForMultimodalLM,
        ):
            assert cls.from_pretrained in targets
            assert cls.from_config in targets

    def test_accepted_targets_missing_gemma4_dep_takes_except_branch(self, monkeypatch):
        """Force the optional composite import to fail: the ``except ImportError``
        branch runs and the set still holds the NeMoAuto entrypoints while the
        Gemma4 composite is absent."""
        from nemo_automodel._transformers import NeMoAutoModelForCausalLM
        from nemo_automodel.recipes.vlm.finetune import _accepted_targets

        # A ``None`` entry in sys.modules makes ``from <mod> import X`` raise ImportError.
        monkeypatch.setitem(sys.modules, _GEMMA4_COMPOSITE_MOD, None)

        targets = _accepted_targets()
        assert NeMoAutoModelForCausalLM.from_pretrained in targets
        assert "Gemma4WithDrafter" not in _target_owner_names(targets)

    def test_accepted_targets_present_gemma4_dep_adds_composite(self, monkeypatch):
        """Inject a fake composite module so the ``accepted.add(...)`` branch runs
        regardless of whether the real optional dep is installed."""
        from nemo_automodel.recipes.vlm.finetune import _accepted_targets

        class Gemma4WithDrafter:
            @classmethod
            def from_pretrained(cls):
                return cls()

        fake_mod = types.ModuleType(_GEMMA4_COMPOSITE_MOD)
        fake_mod.Gemma4WithDrafter = Gemma4WithDrafter
        monkeypatch.setitem(sys.modules, _GEMMA4_COMPOSITE_MOD, fake_mod)

        targets = _accepted_targets()
        assert Gemma4WithDrafter.from_pretrained in targets

    def test_is_recipe_target_none_returns_false(self):
        from nemo_automodel.recipes.vlm.finetune import _is_recipe_target

        assert _is_recipe_target(None) is False

    def test_is_recipe_target_accepts_nemo_auto_and_rejects_others(self):
        from nemo_automodel._transformers import NeMoAutoModelForImageTextToText
        from nemo_automodel.recipes.vlm.finetune import _is_recipe_target

        assert _is_recipe_target(NeMoAutoModelForImageTextToText.from_pretrained) is True
        assert _is_recipe_target(NeMoAutoModelForImageTextToText.from_config) is True
        assert _is_recipe_target("some.invalid.Target") is False
        assert _is_recipe_target(lambda: None) is False


# -----------------------------------------------------------------------------
# rope_fusion disabled when cp > 1
# -----------------------------------------------------------------------------


def _patch_vlm_setup_minimals(monkeypatch, cp_size):
    """Patch heavy dependencies so FinetuneRecipeForVLM.setup() runs lightly."""
    monkeypatch.setattr(
        "nemo_automodel.recipes.vlm.finetune.initialize_distributed",
        lambda *a, **k: SimpleNamespace(world_size=1, is_main=True, device=torch.device("cpu"), rank=0),
    )
    monkeypatch.setattr("nemo_automodel.recipes.vlm.finetune.setup_logging", lambda: None)
    monkeypatch.setattr("nemo_automodel.recipes.vlm.finetune.apply_cache_compatibility_patches", lambda: None)
    monkeypatch.setattr("nemo_automodel.recipes.vlm.finetune.StatefulRNG", lambda *a, **k: "rng")
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
    monkeypatch.setattr(
        "nemo_automodel.recipes.vlm.finetune.create_distributed_setup_from_config",
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
    dummy_model = DummyModel()
    dummy_opt = SimpleNamespace(param_groups=[{"lr": 0.01}], step=lambda: None, zero_grad=lambda **k: None)
    monkeypatch.setattr("nemo_automodel.recipes.vlm.finetune.build_model", lambda *a, **k: dummy_model)
    monkeypatch.setattr(
        "nemo_automodel.recipes._typed_config.RecipeConfig.optimizer",
        property(lambda self: SimpleNamespace(build=lambda *a, **k: [dummy_opt])),
    )
    loader_config = SimpleNamespace(
        packing=None,
        resolve_packing_attn_implementation=lambda **kwargs: None,
        build=lambda **kwargs: SimpleNamespace(dataloader="dl", processor="proc"),
    )
    monkeypatch.setattr(
        "nemo_automodel.recipes._typed_config.RecipeConfig.vlm_dataloader",
        property(lambda self: loader_config),
    )
    monkeypatch.setattr(
        "nemo_automodel.recipes._typed_config.RecipeConfig.vlm_validation_dataloader",
        property(lambda self: None),
    )
    monkeypatch.setattr("nemo_automodel.recipes.vlm.finetune.ScopedRNG", lambda **kwargs: nullcontext())
    monkeypatch.setattr(
        "nemo_automodel.components.training.step_scheduler.StepSchedulerConfig.build",
        lambda self, *a, **k: SimpleNamespace(step=0, epoch=0, epochs=[]),
    )
    monkeypatch.setattr("nemo_automodel.components.optim.optimizer.LRSchedulerConfig.build", lambda self, *a, **k: [])
    monkeypatch.setattr(
        "nemo_automodel.recipes.vlm.finetune.build_metric_logger",
        lambda *a, **k: SimpleNamespace(log=lambda *a, **k: None, close=lambda: None),
    )
    monkeypatch.setattr(
        "nemo_automodel.recipes.vlm.finetune.FinetuneRecipeForVLM._log_experiment_details",
        lambda self: None,
    )
    monkeypatch.setattr(
        "nemo_automodel.recipes.vlm.finetune.FinetuneRecipeForVLM._log_library_versions",
        lambda self: None,
    )
    monkeypatch.setattr(
        "nemo_automodel.recipes.vlm.finetune.FinetuneRecipeForVLM._log_model_and_optimizer_details",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "nemo_automodel.recipes.vlm.finetune.FinetuneRecipeForVLM.load_checkpoint",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "nemo_automodel.recipes.vlm.finetune.FinetuneRecipeForVLM._log_step_scheduler_details",
        lambda *a, **k: None,
    )
    monkeypatch.setattr("nemo_automodel.recipes.vlm.finetune.torch.cuda.reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(
        "nemo_automodel.recipes.vlm.finetune.FinetuneRecipeForVLM._get_dp_rank", lambda self, include_cp=False: 0
    )
    monkeypatch.setattr(
        "nemo_automodel.recipes.vlm.finetune.FinetuneRecipeForVLM._get_dp_group_size", lambda self, include_cp=False: 1
    )
    monkeypatch.setattr("nemo_automodel.recipes.vlm.finetune.FinetuneRecipeForVLM._get_cp_group_size", lambda self: 1)
    monkeypatch.setattr("nemo_automodel.recipes.vlm.finetune.FinetuneRecipeForVLM._get_tp_rank", lambda self: 0)
    monkeypatch.setattr("nemo_automodel.recipes.vlm.finetune.FinetuneRecipeForVLM._get_pp_rank", lambda self: 0)


def _minimal_vlm_cfg(cp_size: int, rope_fusion: bool, prewarm: dict[str, bool] | None = None) -> ConfigNode:
    cfg = {
        "model": {"backend": {"rope_fusion": rope_fusion}},
        "dataloader": {},
        "dataset": {"path_or_dataset": "dummy"},
        "validation_dataloader": {},
        "step_scheduler": {"local_batch_size": 1, "global_batch_size": 1},
        "optimizer": {},
        "loss_fn": {},
        "checkpoint": {"best_metric_key": "default"},
        "distributed": {"cp_size": cp_size},
    }
    if prewarm is not None:
        cfg["prewarm"] = prewarm
    return ConfigNode(cfg)


def test_vlm_setup_applies_prewarm_config(monkeypatch):
    """VLM setup should apply the typed prewarm config to its parallelized model parts."""
    cfg = _minimal_vlm_cfg(cp_size=1, rope_fusion=False, prewarm={"comm_groups": True})
    _patch_vlm_setup_minimals(monkeypatch, cp_size=1)
    calls = []

    def _record_apply(self, *, model_parts, device, pp_mesh=None):
        calls.append((self, model_parts, device, pp_mesh))

    monkeypatch.setattr("nemo_automodel.components.training.prewarm.PrewarmConfig.apply", _record_apply)

    trainer = FinetuneRecipeForVLM(cfg)
    trainer.setup()

    assert len(calls) == 1
    prewarm, model_parts, device, pp_mesh = calls[0]
    assert prewarm.comm_groups is True
    assert model_parts == trainer.model_parts
    assert device == torch.device("cpu")
    assert pp_mesh is None


def test_vlm_rope_fusion_disabled_when_cp_gt_1(monkeypatch):
    """rope_fusion should be set to False during VLM setup when cp_size > 1."""
    cfg = _minimal_vlm_cfg(cp_size=2, rope_fusion=True)
    _patch_vlm_setup_minimals(monkeypatch, cp_size=2)

    trainer = FinetuneRecipeForVLM(cfg)
    trainer.setup()

    assert cfg.model.backend.rope_fusion is False


def test_vlm_rope_fusion_unchanged_when_cp_eq_1(monkeypatch):
    """rope_fusion should remain True in VLM setup when cp_size == 1."""
    cfg = _minimal_vlm_cfg(cp_size=1, rope_fusion=True)
    _patch_vlm_setup_minimals(monkeypatch, cp_size=1)

    trainer = FinetuneRecipeForVLM(cfg)
    trainer.setup()

    assert cfg.model.backend.rope_fusion is True


def test_vlm_rope_fusion_stays_false_when_already_disabled(monkeypatch):
    """rope_fusion=False should stay False in VLM setup regardless of cp_size."""
    cfg = _minimal_vlm_cfg(cp_size=4, rope_fusion=False)
    _patch_vlm_setup_minimals(monkeypatch, cp_size=4)

    trainer = FinetuneRecipeForVLM(cfg)
    trainer.setup()

    assert cfg.model.backend.rope_fusion is False


# ---------------------------------------------------------------------------
# chunk_vlm_media tests
# ---------------------------------------------------------------------------


class TestChunkVlmMedia:
    """Tests for PP VLM media microbatch splitting."""

    def test_4d_pixel_values_simple_chunk(self):
        pixel_values = torch.randn(4, 3, 56, 56)
        image_grid = torch.tensor([[1, 2, 2]] * 4)
        pv_chunks, ig_chunks = chunk_vlm_media(pixel_values, image_grid, batch_size=4, n_microbatches=2)
        assert len(pv_chunks) == 2
        assert pv_chunks[0].shape[0] == 2
        assert pv_chunks[1].shape[0] == 2

    def test_n_images_per_sample_packed(self):
        """Packed sequences: each batch item has variable number of images."""
        # 2 batch items: first has 3 images, second has 1 image
        # image_grid: 4 images total, each 2x2 patches = 4 patches each
        image_grid = torch.tensor([[1, 2, 2], [1, 2, 2], [1, 2, 2], [1, 2, 2]])
        pixel_values = torch.randn(16, 64)  # 4 images * 4 patches = 16 patches
        n_images_per_sample = torch.tensor([3, 1])

        pv_chunks, ig_chunks = chunk_vlm_media(
            pixel_values,
            image_grid,
            batch_size=2,
            n_microbatches=2,
            n_images_per_sample=n_images_per_sample,
        )
        assert len(pv_chunks) == 2
        assert ig_chunks[0].shape[0] == 3  # first batch item: 3 images
        assert ig_chunks[1].shape[0] == 1  # second batch item: 1 image
        assert pv_chunks[0].shape[0] == 12  # 3 images * 4 patches
        assert pv_chunks[1].shape[0] == 4  # 1 image * 4 patches

    def test_legacy_one_image_per_sample(self):
        # 4 samples, 1 image each with different patch counts
        image_grid = torch.tensor([[1, 2, 2], [1, 3, 3], [1, 2, 2], [1, 3, 3]])
        patch_counts = image_grid.prod(dim=1)  # [4, 9, 4, 9] = 26 total
        pixel_values = torch.randn(int(patch_counts.sum()), 64)

        pv_chunks, ig_chunks = chunk_vlm_media(
            pixel_values,
            image_grid,
            batch_size=4,
            n_microbatches=2,
        )
        assert len(pv_chunks) == 2
        assert ig_chunks[0].shape[0] == 2
        assert ig_chunks[1].shape[0] == 2
        assert pv_chunks[0].shape[0] == 4 + 9  # first 2 images
        assert pv_chunks[1].shape[0] == 4 + 9  # last 2 images

    def test_qwen35_ep4_pp2_style_n_images_per_sample(self):
        """EP does not affect chunking; PP2 should split media by batch sample ownership."""
        image_grid = torch.tensor([[1, 2, 2], [1, 1, 3], [1, 3, 3], [1, 2, 4]])
        patch_counts = image_grid.prod(dim=1)
        pixel_values = torch.randn(int(patch_counts.sum()), 64)
        n_images_per_sample = torch.tensor([1, 0, 2, 1])

        pv_chunks, ig_chunks = chunk_vlm_media(
            pixel_values,
            image_grid,
            batch_size=4,
            n_microbatches=2,
            n_images_per_sample=n_images_per_sample,
        )

        assert len(pv_chunks) == 2
        assert torch.equal(ig_chunks[0], image_grid[:1])
        assert torch.equal(ig_chunks[1], image_grid[1:])
        assert pv_chunks[0].shape[0] == int(patch_counts[:1].sum())
        assert pv_chunks[1].shape[0] == int(patch_counts[1:].sum())

    def test_fallback_mismatched_images_raises(self):
        """n_images != batch_size with no n_images_per_sample now raises rather
        than silently emptying mb1..N (which previously caused trailing microbatches
        to scatter media tokens into empty pixel_values)."""
        image_grid = torch.tensor([[1, 2, 2], [1, 2, 2], [1, 2, 2]])
        pixel_values = torch.randn(12, 64)  # 3 images but batch_size=2

        with pytest.raises(ValueError, match="VLM PP chunking cannot align"):
            chunk_vlm_media(
                pixel_values,
                image_grid,
                batch_size=2,
                n_microbatches=2,
            )

    def test_n_videos_per_sample_packed(self):
        """The media chunk helper also handles video grids/counts."""

        video_grid = torch.tensor([[1, 2, 2], [1, 3, 3], [1, 2, 3], [1, 4, 4]])
        pixel_values_videos = torch.randn(int(video_grid.prod(dim=1).sum().item()), 64)
        n_videos_per_sample = torch.tensor([1, 0, 2, 1])

        pv_chunks, vg_chunks = chunk_vlm_media(
            pixel_values_videos,
            video_grid,
            batch_size=4,
            n_microbatches=2,
            n_images_per_sample=n_videos_per_sample,
        )

        assert len(pv_chunks) == 2
        assert vg_chunks[0].shape[0] == 1
        assert vg_chunks[1].shape[0] == 3
        assert pv_chunks[0].shape[0] == 4
        assert pv_chunks[1].shape[0] == 9 + 6 + 16

    def test_uneven_batch_size_general_branch_covers_all_samples(self):
        """batch_size not divisible by n_microbatches must not drop trailing samples.

        torch.tensor.chunk(n) used by schedule.step on input_ids returns ceil-sized
        chunks. chunk_vlm_media must mirror that or the last sample's images are
        silently lost while its text still flows through the schedule.
        """

        # 7 samples across 3 microbatches: ceil(7/3)=3, expect splits [3, 3, 1].
        batch_size, n_microbatches = 7, 3
        image_grid = torch.tensor([[1, 2, 2]] * batch_size)  # 4 patches/image
        pixel_values = torch.randn(int(image_grid.prod(dim=1).sum().item()), 64)
        n_images_per_sample = torch.tensor([1] * batch_size)

        pv_chunks, ig_chunks = chunk_vlm_media(
            pixel_values,
            image_grid,
            batch_size=batch_size,
            n_microbatches=n_microbatches,
            n_images_per_sample=n_images_per_sample,
        )

        assert len(ig_chunks) == n_microbatches
        assert [c.shape[0] for c in ig_chunks] == [3, 3, 1]
        assert sum(c.shape[0] for c in ig_chunks) == batch_size  # no sample dropped
        assert sum(c.shape[0] for c in pv_chunks) == pixel_values.shape[0]

    def test_uneven_batch_size_legacy_branch_covers_all_images(self):
        """Legacy 1-image-per-sample branch must also use ceil division."""

        # 5 images across 3 microbatches: ceil(5/3)=2, expect splits [2, 2, 1].
        batch_size, n_microbatches = 5, 3
        image_grid = torch.tensor([[1, 2, 2]] * batch_size)
        pixel_values = torch.randn(int(image_grid.prod(dim=1).sum().item()), 64)

        pv_chunks, ig_chunks = chunk_vlm_media(
            pixel_values,
            image_grid,
            batch_size=batch_size,
            n_microbatches=n_microbatches,
        )

        assert len(ig_chunks) == n_microbatches
        assert [c.shape[0] for c in ig_chunks] == [2, 2, 1]
        assert sum(c.shape[0] for c in ig_chunks) == batch_size

    def test_uneven_batch_size_gemma4_multi_image_branch_covers_all_samples(self):
        """Gemma4 multi-image branch (3D pixel_values + counts) must also use ceil."""
        # 7 samples across 3 microbatches: ceil(7/3)=3, expect sample splits [3, 3, 1].
        # Image counts per split are [2 + 1 + 0, 3 + 1 + 2, 1] = [3, 6, 1].
        batch_size, n_microbatches = 7, 3
        max_patches = 4
        n_images_per_sample = torch.tensor([2, 1, 0, 3, 1, 2, 1])
        n_images = int(n_images_per_sample.sum().item())
        image_grid = torch.tensor([[1, 2, 2]] * n_images)
        pixel_values = torch.randn(n_images, max_patches, 64)  # 3D, one row per image.

        pv_chunks, ig_chunks = chunk_vlm_media(
            pixel_values,
            image_grid,
            batch_size=batch_size,
            n_microbatches=n_microbatches,
            n_images_per_sample=n_images_per_sample,
        )

        assert len(ig_chunks) == n_microbatches
        assert [c.shape[0] for c in ig_chunks] == [3, 6, 1]
        assert [c.shape[0] for c in pv_chunks] == [3, 6, 1]
        assert sum(c.shape[0] for c in pv_chunks) == n_images

    def test_step3_media_chunks_full_images_and_flat_patches(self):
        pixel_values = torch.arange(4 * 3, dtype=torch.float32).reshape(4, 3)
        patch_pixel_values = torch.arange(6 * 2, dtype=torch.float32).reshape(6, 2)
        patch_newline_mask = torch.tensor([True, False, False, True, False, True])
        num_patches = torch.tensor([2, 0, 3, 1])

        chunks = chunk_step3_media(
            pixel_values,
            batch_size=4,
            n_microbatches=2,
            num_patches=num_patches,
            patch_pixel_values=patch_pixel_values,
            patch_newline_mask=patch_newline_mask,
        )

        assert torch.equal(chunks["pixel_values"][0], pixel_values[:2])
        assert torch.equal(chunks["pixel_values"][1], pixel_values[2:])
        assert torch.equal(chunks["num_patches"][0], torch.tensor([2, 0]))
        assert torch.equal(chunks["num_patches"][1], torch.tensor([3, 1]))
        assert torch.equal(chunks["patch_pixel_values"][0], patch_pixel_values[:2])
        assert torch.equal(chunks["patch_pixel_values"][1], patch_pixel_values[2:])
        assert torch.equal(chunks["patch_newline_mask"][0], patch_newline_mask[:2])
        assert torch.equal(chunks["patch_newline_mask"][1], patch_newline_mask[2:])

    def test_step3_media_defaults_num_patches_and_validates_shapes(self):
        pixel_values = torch.randn(3, 2)
        chunks = chunk_step3_media(pixel_values, batch_size=3, n_microbatches=2)
        assert [chunk.tolist() for chunk in chunks["num_patches"]] == [[0, 0], [0]]
        assert "patch_pixel_values" not in chunks
        assert "patch_newline_mask" not in chunks

        with pytest.raises(ValueError, match="one full image tensor per sample"):
            chunk_step3_media(pixel_values[:2], batch_size=3, n_microbatches=2)
        with pytest.raises(ValueError, match="num_patches must have length"):
            chunk_step3_media(pixel_values, batch_size=3, n_microbatches=2, num_patches=torch.tensor([1, 2]))

    def test_prepare_step3_media_without_image_grid_and_stage_cleanup(self):
        model = SimpleNamespace()
        pp = SimpleNamespace(info=SimpleNamespace(has_first_stage=True))
        batch = {
            "input_ids": torch.ones(4, 3, dtype=torch.long),
            "pixel_values": torch.arange(4 * 3, dtype=torch.float32).reshape(4, 3),
            "patch_pixel_values": torch.arange(4 * 2, dtype=torch.float32).reshape(4, 2),
            "num_patches": torch.tensor([1, 0, 2, 1]),
            "patch_newline_mask": torch.tensor([True, False, True, False]),
        }

        prepared = prepare_vlm_media_for_pp(batch, batch_size=4, n_microbatches=2)

        assert "pixel_values" not in prepared
        assert "patch_pixel_values" not in prepared
        assert "num_patches" not in prepared
        assert "patch_newline_mask" not in prepared
        assert VLM_PP_MEDIA_KEY in prepared

        with stage_vlm_media_for_pp(pp, [model], prepared):
            assert len(model._vlm_pixel_values_chunks) == 2
            assert len(model._vlm_patch_pixel_values_chunks) == 2
            assert len(model._vlm_num_patches_chunks) == 2
            assert len(model._vlm_patch_newline_mask_chunks) == 2
            assert model._vlm_chunk_idx == 0

        assert model._vlm_pixel_values_chunks is None
        assert model._vlm_patch_pixel_values_chunks is None
        assert model._vlm_num_patches_chunks is None
        assert model._vlm_patch_newline_mask_chunks is None
        assert model._vlm_chunk_idx is None


# -----------------------------------------------------------------------------
# get_rope_index forwarding tests for build_dataloader
#
# Guard against a regression where the VLM recipe forgot to pass
# get_rope_index to neat_pack_dataset_vlm, silently degrading mRoPE to
# plain 1D positions for packed Qwen2.5-VL / Qwen3-VL training.
# -----------------------------------------------------------------------------


def _make_packing_cfg(pack_size=128):
    return ConfigNode(
        {
            "pack_size": pack_size,
            "pretokenize": True,
            "max_length": pack_size,
            "drop_long_samples": True,
            "max_packs": None,
            "packing_ratio": 1.0,
            "balance_media_tokens": True,
            "collate_max_length": None,
            "post_tokenize_hook_fn": None,
        }
    )


def _make_dataset_cfg():
    return ConfigNode({"_target_": _test_vlm_dataset, "truncate": True})


def _patches_for_packing(neat_pack_side_effect):
    processor = MagicMock()
    processor.tokenizer.pad_token_id = 0
    processor.chat_template = "{{ x }}"
    return processor, [
        patch("transformers.AutoProcessor.from_pretrained", return_value=processor),
        patch("torch.utils.data.distributed.DistributedSampler"),
        patch(
            "nemo_automodel.components.datasets.vlm.datasets.PreTokenizedDatasetWrapper",
            return_value=MagicMock(),
        ),
        patch(
            "nemo_automodel.components.datasets.vlm.neat_packing_vlm.neat_pack_dataset_vlm",
            side_effect=neat_pack_side_effect,
        ),
        patch("nemo_automodel.components.datasets.vlm.loader.StatefulDataLoader", return_value=MagicMock()),
        patch("nemo_automodel.components.models.common.packing.configure_packing"),
        patch(
            "nemo_automodel.components.models.common.packing.get_attn_implementation",
            return_value="sdpa",
        ),
    ]


def test_build_dataloader_forwards_get_rope_index_to_packing():
    """get_rope_index passed to build_dataloader must reach neat_pack_dataset_vlm."""
    from contextlib import ExitStack

    from nemo_automodel.recipes.vlm.finetune import build_dataloader

    sentinel = MagicMock(name="get_rope_index")
    captured = {}

    def fake_neat_pack(*args, **kwargs):
        captured.update(kwargs)
        return MagicMock()

    _, ctx_managers = _patches_for_packing(fake_neat_pack)

    with ExitStack() as stack:
        for cm in ctx_managers:
            stack.enter_context(cm)
        build_dataloader(
            _make_dataset_cfg(),
            _vlm_dataloader_cfg(),
            "test/model",
            None,
            None,
            42,
            1,
            cfg_ps=_make_packing_cfg(pack_size=64),
            get_rope_index=sentinel,
        )

    assert captured.get("get_rope_index") is sentinel, (
        f"build_dataloader must forward get_rope_index to neat_pack_dataset_vlm; got kwargs={list(captured.keys())}"
    )


def test_build_dataloader_default_get_rope_index_is_none():
    """When the model does not expose get_rope_index, packing must receive None."""
    from contextlib import ExitStack

    from nemo_automodel.recipes.vlm.finetune import build_dataloader

    captured = {}

    def fake_neat_pack(*args, **kwargs):
        captured.update(kwargs)
        return MagicMock()

    _, ctx_managers = _patches_for_packing(fake_neat_pack)

    with ExitStack() as stack:
        for cm in ctx_managers:
            stack.enter_context(cm)
        build_dataloader(
            _make_dataset_cfg(),
            _vlm_dataloader_cfg(),
            "test/model",
            None,
            None,
            42,
            1,
            cfg_ps=_make_packing_cfg(pack_size=64),
        )

    assert "get_rope_index" in captured, "neat_pack_dataset_vlm must receive get_rope_index kwarg even when None"
    assert captured["get_rope_index"] is None


def _run_build_dataloader_capturing_wrapper(dataset_cfg):
    """Run build_dataloader (pretokenize path) and return the PreTokenizedDatasetWrapper mock."""
    from contextlib import ExitStack

    from nemo_automodel.recipes.vlm.finetune import build_dataloader

    wrapper_mock = MagicMock(return_value=MagicMock())
    _, ctx_managers = _patches_for_packing(lambda *a, **k: MagicMock())

    with ExitStack() as stack:
        for cm in ctx_managers:
            stack.enter_context(cm)
        # Override the wrapper patch from _patches_for_packing so we can inspect call kwargs.
        stack.enter_context(
            patch(
                "nemo_automodel.components.datasets.vlm.datasets.PreTokenizedDatasetWrapper",
                wrapper_mock,
            )
        )
        build_dataloader(
            dataset_cfg,
            _vlm_dataloader_cfg(),
            "test/model",
            None,
            None,
            42,
            1,
            cfg_ps=_make_packing_cfg(pack_size=64),
        )
    return wrapper_mock


def test_build_dataloader_inject_fake_images_defaults_true():
    """When dataset cfg omits inject_fake_images, the wrapper defaults to True."""
    wrapper_mock = _run_build_dataloader_capturing_wrapper(_make_dataset_cfg())
    assert wrapper_mock.call_args.kwargs["inject_fake_images"] is True


def test_build_dataloader_forwards_inject_fake_images_false():
    """inject_fake_images=False in dataset cfg must reach PreTokenizedDatasetWrapper."""
    cfg = ConfigNode({"_target_": _test_vlm_dataset, "truncate": True, "inject_fake_images": False})

    wrapper_mock = _run_build_dataloader_capturing_wrapper(cfg)
    assert wrapper_mock.call_args.kwargs["inject_fake_images"] is False
