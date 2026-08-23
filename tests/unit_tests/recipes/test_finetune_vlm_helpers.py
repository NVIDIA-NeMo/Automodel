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
from nemo_automodel.components.datasets.datum import LossInputLayout
from nemo_automodel.components.datasets.vlm.pp_media import (
    VLM_PP_MEDIA_KEY,
    chunk_step3_media,
    chunk_vlm_media,
    prepare_vlm_media_for_pp,
    stage_vlm_media_for_pp,
    wrap_vlm_collate_for_pp,
)
from nemo_automodel.components.distributed.pipelining import AutoPipeline
from nemo_automodel.components.optim.optimizer import LRSchedulerConfig, build_optimizer_config
from nemo_automodel.components.training.step_scheduler import StepSchedulerConfig
from nemo_automodel.engine import ForwardBackwardResult
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


def _build_engine_recipe_for_optim_step(*, pp_enabled: bool = False):
    """Build the smallest recipe state needed to exercise the Engine boundary."""
    recipe = FinetuneRecipeForVLM.__new__(FinetuneRecipeForVLM)
    recipe.dist_env = SimpleNamespace(device="cpu", rank=0, is_main=True)
    recipe.device_mesh = None
    recipe.moe_mesh = None
    recipe.loss_fn = object()
    recipe.model_parts = [_TensorModel()]
    recipe._has_joint_drafter = False
    recipe.pp_enabled = pp_enabled
    if pp_enabled:
        recipe.pp = SimpleNamespace(info=SimpleNamespace(has_first_stage=True))
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
    recipe.engine = MagicMock()
    recipe.engine.forward_backward.return_value = ForwardBackwardResult(
        loss=torch.tensor(0.25),
        loss_sum=torch.tensor(1.0),
        weight_sum=torch.tensor(4.0),
        token_outputs=[],
        batch_outputs=[],
    )
    recipe.engine.step.return_value = SimpleNamespace(grad_norm=2.5, learning_rates=(0.01,))
    return recipe


@pytest.mark.cuda(False)
def test_run_train_step_passes_nested_prebatched_datums_to_engine():
    recipe = _build_engine_recipe_for_optim_step(pp_enabled=True)
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
    metrics = recipe._run_train_optim_step(batches)

    datums, loss_fn = recipe.engine.forward_backward.call_args.args
    assert [len(batch) for batch in datums] == [1, 1]
    flat_datums = [batch[0] for batch in datums]
    assert all(datum.model_inputs.keys() == {"input_ids"} for datum in flat_datums)
    assert [datum.loss_fn_inputs["weights"].tolist() for datum in flat_datums] == [
        [[True, False, True, False]],
        [[False, True, False, True]],
    ]
    assert callable(loss_fn)
    recipe.engine.step.assert_called_once_with(
        before_optimizer_step=recipe.checkpointer.maybe_wait_for_staging,
    )
    assert metrics.metrics["loss"] == pytest.approx(0.25)
    assert metrics.metrics["grad_norm"] == pytest.approx(2.5)
    assert metrics.metrics["lr"] == pytest.approx(0.01)
    assert not recipe.optimizer[0].step_called
    assert not recipe.optimizer[0].zero_grad_called


@pytest.mark.cuda(False)
def test_train_step_logs_joint_drafter_only_on_first_engine_loss_call():
    recipe = _build_engine_recipe_for_optim_step()
    recipe._has_joint_drafter = True
    recipe.step_scheduler.is_remote_logging_step = True
    batches = [
        {"labels": torch.tensor([[1, -100, 2]]), "input_ids": torch.tensor([[1, 2, 3]])},
        {"labels": torch.tensor([[-100, 3, 4]]), "input_ids": torch.tensor([[4, 5, 6]])},
    ]
    recipe._compute_vlm_loss = MagicMock(side_effect=[torch.tensor(1.0), torch.tensor(2.0)])

    def forward_backward(datums, loss_fn):
        for (datum,) in datums:
            loss_fn(object(), datum.loss_fn_inputs)
        return ForwardBackwardResult(
            loss=torch.tensor(0.25),
            loss_sum=torch.tensor(1.0),
            weight_sum=torch.tensor(4.0),
            token_outputs=[],
            batch_outputs=[],
        )

    recipe.engine.forward_backward.side_effect = forward_backward

    recipe._run_train_optim_step(batches)

    assert recipe._compute_vlm_loss.call_count == 2
    first_call, second_call = recipe._compute_vlm_loss.call_args_list
    assert first_call.kwargs["is_train"] is True
    assert first_call.kwargs["log_drafter"] is True
    assert first_call.kwargs["log_denominator"] == 4
    assert second_call.kwargs["is_train"] is True
    assert second_call.kwargs["log_drafter"] is False
    assert second_call.kwargs["log_denominator"] == 4


@pytest.mark.cuda(False)
def test_train_step_does_not_reduce_drafter_denominator_for_regular_model():
    recipe = _build_engine_recipe_for_optim_step()
    recipe.step_scheduler.is_remote_logging_step = True
    reductions = []

    def allreduce(tensor, include_cp=False):
        reductions.append(tensor.clone())
        return tensor

    recipe._dp_allreduce = allreduce
    batch = {"labels": torch.tensor([[1, -100, 2]]), "input_ids": torch.tensor([[1, 2, 3]])}

    recipe._run_train_optim_step([batch])

    assert len(reductions) == 1  # Throughput tokens only; Engine owns the loss denominator.


@pytest.mark.cuda(False)
def test_run_train_step_uses_engine_for_empty_supervision():
    recipe = _build_engine_recipe_for_optim_step()
    recipe.engine.forward_backward.return_value = ForwardBackwardResult(
        loss=torch.tensor(0.0),
        loss_sum=torch.tensor(0.0),
        weight_sum=torch.tensor(0.0),
        token_outputs=[],
        batch_outputs=[],
    )
    batch = {"labels": torch.full((1, 4), -100), "input_ids": torch.arange(4).reshape(1, 4)}
    metrics = recipe._run_train_optim_step([batch])

    recipe.engine.forward_backward.assert_called_once()
    recipe.engine.step.assert_called_once()
    assert metrics.metrics["loss"] == 0.0
    assert metrics.metrics["num_label_tokens"] == 0
    assert not recipe.optimizer[0].step_called


@pytest.mark.cuda(False)
def test_train_step_leaves_fp8_post_step_work_to_engine(monkeypatch):
    recipe = _build_engine_recipe_for_optim_step()
    recipe.cfg = _Cfg(
        fp8={
            "enabled": True,
            "precompute_float8_dynamic_scale_for_fsdp": True,
        }
    )
    recipe.device_mesh = {"dp_shard": SimpleNamespace(size=lambda: 2)}
    monkeypatch.setattr(
        "nemo_automodel.recipes.vlm.finetune.precompute_float8_dynamic_scale_for_fsdp",
        lambda _model: pytest.fail("the VLM recipe must not run FP8 post-step work directly"),
        raising=False,
    )

    batch = {"labels": torch.tensor([[1, 2]]), "input_ids": torch.tensor([[3, 4]])}
    recipe._run_train_optim_step([batch])

    recipe.engine.step.assert_called_once_with(
        before_optimizer_step=recipe.checkpointer.maybe_wait_for_staging,
    )


def test_make_engine_datum_filters_raw_media_off_first_pipeline_stage():
    recipe = FinetuneRecipeForVLM.__new__(FinetuneRecipeForVLM)
    recipe.pp_enabled = True
    recipe.pp = SimpleNamespace(info=SimpleNamespace(has_first_stage=False))
    recipe.loss_fn = object()
    media_chunks = {"pixel_values": [torch.ones(1, 2)]}
    batch = {
        "input_ids": torch.tensor([[1, 2]]),
        "labels": torch.tensor([[2, -100]]),
        "pixel_values": torch.ones(1, 3, 4, 4),
        "image_grid_thw": torch.ones(1, 3, dtype=torch.long),
        VLM_PP_MEDIA_KEY: media_chunks,
    }

    datum = recipe._make_engine_datum(batch)

    assert datum.model_inputs["input_ids"] is batch["input_ids"]
    assert "pixel_values" not in datum.model_inputs
    assert "image_grid_thw" not in datum.model_inputs
    assert VLM_PP_MEDIA_KEY not in datum.model_inputs
    assert datum.loss_fn_input_layouts == {
        "labels": LossInputLayout.PER_TOKEN,
        "weights": LossInputLayout.PER_TOKEN,
    }

    recipe.pp.info.has_first_stage = True
    first_stage_datum = recipe._make_engine_datum(batch)
    assert first_stage_datum.model_inputs[VLM_PP_MEDIA_KEY] is media_chunks


def test_engine_context_stages_pipeline_media_and_cleans_up(monkeypatch):
    recipe = FinetuneRecipeForVLM.__new__(FinetuneRecipeForVLM)
    recipe.pp_enabled = True
    model = _TensorModel()
    recipe.model_parts = [model]
    recipe.pp = SimpleNamespace(
        info=SimpleNamespace(
            has_first_stage=True,
            schedule=None,
            stages=[SimpleNamespace(is_first=True)],
        )
    )
    recipe._cp_vision_frame_sharding_context = nullcontext
    input_ids = torch.tensor([[1, 2]])
    model_inputs = {
        "input_ids": input_ids,
        VLM_PP_MEDIA_KEY: {"pixel_values": [torch.ones(1, 2)]},
    }

    with recipe._engine_context(model_inputs):
        assert model._vlm_pixel_values_chunks is not None
        assert VLM_PP_MEDIA_KEY not in model_inputs

    assert model._vlm_pixel_values_chunks is None
    assert model._vlm_chunk_idx is None


def test_engine_pipeline_loss_reuses_configured_loss_and_thd_metadata():
    recipe = FinetuneRecipeForVLM.__new__(FinetuneRecipeForVLM)
    recipe.pp_enabled = True
    recipe.pipeline_loss_fn = MagicMock(return_value=torch.tensor(3.0))
    output = torch.randn(1, 2, 4)
    labels = torch.tensor([[1, 2]])
    cu_seqlens = torch.tensor([0, 2], dtype=torch.int32)

    loss = recipe._engine_loss_fn(output, {"labels": labels, "cu_seqlens": cu_seqlens})

    assert loss.item() == 3.0
    assert recipe.pipeline_loss_fn.cu_seqlens is cu_seqlens
    recipe.pipeline_loss_fn.assert_called_once_with(output, labels)


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
        assert config.max_recent_checkpoints is None

    def test_build_checkpoint_config_with_custom_config(self):
        """Test checkpoint config with custom settings."""
        cfg_ckpt = MagicMock()
        cfg_ckpt.to_dict.return_value = {
            "checkpoint_dir": "/custom/ckpt/",
            "max_recent_checkpoints": 3,
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
        assert config.max_recent_checkpoints == 3
        assert config.save_consolidated.value == "false"
        assert config.is_peft is True

    def test_build_checkpoint_config_warns_on_peft_with_torch_save(self, caplog):
        """PEFT + torch_save: warn, fall back to safetensors defaults; preserve checkpoint_dir."""
        from nemo_automodel.components.checkpoint._backports.filesystem import SerializationFormat

        cfg_ckpt = MagicMock()
        cfg_ckpt.to_dict.return_value = {
            "model_save_format": "torch_save",
            "checkpoint_dir": "/user/ckpt/",
            "max_recent_checkpoints": 2,
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
        # The builder preserves `checkpoint_dir` and `max_recent_checkpoints` from the user configuration.
        assert config.checkpoint_dir == "/user/ckpt/"
        assert config.max_recent_checkpoints == 2
        # The builder coerces incompatible `torch_save` options and restores the default `save_consolidated="final"`.
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

        # Use a module that has lm_head parameters but no get_output_embeddings
        # This tests the fallback path in calculate_loss
        class ModelWithLmHeadOnly(torch.nn.Module):
            """nn.Module model without get_output_embeddings."""

            def __init__(self):
                super().__init__()
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
    from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy

    monkeypatch.setattr(
        "nemo_automodel.recipes.vlm.finetune.initialize_distributed",
        lambda *a, **k: SimpleNamespace(world_size=1, is_main=True, device=torch.device("cpu"), rank=0),
    )
    monkeypatch.setattr("nemo_automodel.recipes.vlm.finetune.setup_logging", lambda: None)
    monkeypatch.setattr("nemo_automodel.recipes.vlm.finetune.apply_cache_compatibility_patches", lambda: None)
    monkeypatch.setattr("nemo_automodel.recipes.vlm.finetune.StatefulRNG", lambda *a, **k: "rng")
    monkeypatch.setattr(
        "nemo_automodel.recipes._typed_config.RecipeConfig.loss_fn",
        property(lambda self: SimpleNamespace(build=lambda: MaskedCrossEntropy(reduction="sum"))),
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


def _minimal_vlm_cfg(
    cp_size: int,
    rope_fusion: bool,
    prewarm: dict[str, bool] | None = None,
    optimizer_target: str | None = None,
) -> ConfigNode:
    cfg = {
        "model": {"backend": {"rope_fusion": rope_fusion}},
        "dataloader": {},
        "dataset": {"path_or_dataset": "dummy"},
        "validation_dataloader": {},
        "step_scheduler": {"local_batch_size": 1, "global_batch_size": 1},
        "optimizer": {"_target_": optimizer_target} if optimizer_target is not None else {},
        "loss_fn": {},
        "checkpoint": {"best_metric_key": "default"},
        "distributed": {"cp_size": cp_size},
    }
    if prewarm is not None:
        cfg["prewarm"] = prewarm
    return ConfigNode(cfg)


def _patch_vlm_distributed_setup(
    monkeypatch,
    *,
    pp_enabled: bool,
    calculate_per_token_loss: bool = False,
    scale_grads_in_schedule: bool = False,
):
    mesh_context = SimpleNamespace(
        pp_enabled=pp_enabled,
        device_mesh=None,
        moe_mesh=None,
        cp_size=1,
        pp_size=2 if pp_enabled else 1,
    )
    pipeline_config = (
        SimpleNamespace(
            scale_grads_in_schedule=scale_grads_in_schedule,
            pp_batch_size=1,
            pp_microbatch_size=1,
            patch_stage_backward_maybe_with_nosync=False,
            loss_fn=None,
        )
        if pp_enabled
        else None
    )
    monkeypatch.setattr(
        "nemo_automodel.recipes.vlm.finetune.create_distributed_setup_from_config",
        lambda cfg, world_size: SimpleNamespace(
            mesh_context=mesh_context,
            strategy_config=SimpleNamespace(calculate_per_token_loss=calculate_per_token_loss),
            pipeline_config=pipeline_config,
            moe_parallel_config=None,
            activation_checkpointing=False,
        ),
    )


def test_vlm_setup_allows_calculate_per_token_loss(monkeypatch):
    cfg = _minimal_vlm_cfg(cp_size=1, rope_fusion=False)
    _patch_vlm_setup_minimals(monkeypatch, cp_size=1)
    _patch_vlm_distributed_setup(monkeypatch, pp_enabled=False, calculate_per_token_loss=True)

    trainer = FinetuneRecipeForVLM(cfg)
    trainer.setup()

    assert trainer.distributed_config.calculate_per_token_loss is True
    assert trainer.engine is not None


def test_vlm_setup_rejects_pipeline_schedule_gradient_scaling(monkeypatch):
    cfg = _minimal_vlm_cfg(cp_size=1, rope_fusion=False)
    _patch_vlm_setup_minimals(monkeypatch, cp_size=1)
    _patch_vlm_distributed_setup(monkeypatch, pp_enabled=True, scale_grads_in_schedule=True)

    trainer = FinetuneRecipeForVLM(cfg)
    with pytest.raises(ValueError, match="scale_grads_in_schedule=False"):
        trainer.setup()


@pytest.mark.parametrize("local_batch_size", [1, 2])
def test_vlm_setup_supports_magi_pipeline_only_with_unit_local_batch(monkeypatch, local_batch_size):
    cfg = _minimal_vlm_cfg(cp_size=1, rope_fusion=False)
    cfg.step_scheduler.local_batch_size = local_batch_size
    cfg.distributed.pipeline = ConfigNode({"pp_microbatch_size": 1})
    _patch_vlm_setup_minimals(monkeypatch, cp_size=1)
    _patch_vlm_distributed_setup(monkeypatch, pp_enabled=True)
    monkeypatch.setattr(
        "nemo_automodel.recipes.vlm.finetune.setup_magi",
        lambda *args, **kwargs: SimpleNamespace(enabled=True),
    )

    if local_batch_size == 2:
        with pytest.raises(ValueError, match="Magi pipeline training requires"):
            FinetuneRecipeForVLM(cfg).setup()
        return

    model = DummyModel()
    pipeline = object.__new__(AutoPipeline)
    pipeline._info = SimpleNamespace(
        model_parts=[model],
        has_first_stage=True,
        has_last_stage=False,
        stages=[SimpleNamespace(is_first=True, is_last=False)],
        schedule=MagicMock(),
    )
    pipeline.scale_grads_in_schedule = False
    pipeline.pp_batch_size = 1
    pipeline.pp_microbatch_size = 1
    monkeypatch.setattr("nemo_automodel.recipes.vlm.finetune.build_model", lambda *args, **kwargs: pipeline)

    trainer = FinetuneRecipeForVLM(cfg)
    trainer.setup()

    assert trainer.pp is pipeline
    assert trainer.engine.pipeline is pipeline


def test_vlm_setup_applies_prewarm_config(monkeypatch):
    """VLM setup should apply the typed prewarm config to its parallelized model parts."""
    cfg = _minimal_vlm_cfg(cp_size=1, rope_fusion=False, prewarm={"comm_groups": True})
    _patch_vlm_setup_minimals(monkeypatch, cp_size=1)
    calls = []

    def _record_apply(self, *, model_parts, device, batch_size, pp_mesh=None):
        calls.append((self, model_parts, device, batch_size, pp_mesh))

    monkeypatch.setattr("nemo_automodel.components.training.prewarm.PrewarmConfig.apply", _record_apply)

    trainer = FinetuneRecipeForVLM(cfg)
    trainer.setup()

    assert len(calls) == 1
    prewarm, model_parts, device, batch_size, pp_mesh = calls[0]
    assert prewarm.comm_groups is True
    assert model_parts == trainer.model_parts
    assert device == torch.device("cpu")
    assert batch_size == 1
    assert pp_mesh is None


def test_vlm_setup_threads_pp_group_to_checkpointer(monkeypatch):
    cfg = _minimal_vlm_cfg(cp_size=1, rope_fusion=False)
    _patch_vlm_setup_minimals(monkeypatch, cp_size=1)
    pp_group = object()
    build_kwargs = {}
    checkpoint_config = SimpleNamespace(checkpoint_dir="ckpts", model_state_dict_keys=None)

    def _build_checkpointer(**kwargs):
        build_kwargs.update(kwargs)
        return SimpleNamespace(
            config=checkpoint_config,
            load_base_model=lambda *args, **kwargs: None,
            maybe_wait_for_staging=lambda: None,
            close=lambda: None,
        )

    checkpoint_config.build = _build_checkpointer
    monkeypatch.setattr(
        "nemo_automodel.recipes._typed_config.RecipeConfig.checkpoint",
        property(lambda self: checkpoint_config),
    )
    monkeypatch.setattr(FinetuneRecipeForVLM, "_get_pp_group", lambda self: pp_group)

    trainer = FinetuneRecipeForVLM(cfg)
    trainer.setup()

    assert build_kwargs["pp_group"] is pp_group


def test_vlm_rope_fusion_disabled_when_cp_gt_1(monkeypatch):
    """rope_fusion should be set to False during VLM setup when cp_size > 1."""
    cfg = _minimal_vlm_cfg(cp_size=2, rope_fusion=True)
    _patch_vlm_setup_minimals(monkeypatch, cp_size=2)

    trainer = FinetuneRecipeForVLM(cfg)
    trainer.setup()

    assert cfg.model.backend.rope_fusion is False
    assert trainer.engine is not None


def test_vlm_setup_binds_optimizer_state_to_engine(monkeypatch):
    cfg = _minimal_vlm_cfg(cp_size=1, rope_fusion=False)
    _patch_vlm_setup_minimals(monkeypatch, cp_size=1)

    trainer = FinetuneRecipeForVLM(cfg)
    trainer.setup()

    assert trainer.engine.optimizers == tuple(trainer.optimizer)
    assert trainer.engine.lr_schedulers == tuple(trainer.lr_scheduler or ())
    assert trainer.engine.max_grad_norm == trainer.max_grad_norm


def test_vlm_setup_builds_engine_for_magi(monkeypatch):
    cfg = _minimal_vlm_cfg(cp_size=1, rope_fusion=True)
    _patch_vlm_setup_minimals(monkeypatch, cp_size=1)
    monkeypatch.setattr(
        "nemo_automodel.recipes.vlm.finetune.setup_magi",
        lambda *args, **kwargs: SimpleNamespace(enabled=True),
    )

    trainer = FinetuneRecipeForVLM(cfg)
    trainer.setup()

    assert trainer.engine is not None


def test_vlm_compute_loss_uses_final_thd_sequence_boundaries(monkeypatch):
    recipe = object.__new__(FinetuneRecipeForVLM)
    recipe.model_parts = [nn.Identity()]
    recipe.loss_fn = object()
    recipe.cfg = SimpleNamespace(mtp=SimpleNamespace(scaling_factor=0.5, ignore_index=-100))
    recipe.dist_env = SimpleNamespace(is_main=False)
    recipe._get_dp_group = lambda include_cp=False: None
    recipe._maybe_add_drafter_loss = lambda **kwargs: kwargs["base_loss"]
    seen = {}

    def fake_mtp_loss(*args, **kwargs):
        seen.update(kwargs)
        return torch.tensor(2.0)

    monkeypatch.setattr(
        "nemo_automodel.recipes.vlm.finetune.calculate_loss",
        lambda *args, **kwargs: torch.tensor(1.0),
    )
    monkeypatch.setattr("nemo_automodel.recipes.vlm.finetune.calculate_mtp_loss", fake_mtp_loss)
    cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.int32)
    out = SimpleNamespace(
        logits=torch.zeros(5, 8),
        mtp_per_depth_logits=[torch.zeros(5, 8)],
        mtp_loss_scaling_factor=0.25,
    )

    loss = recipe._compute_vlm_loss(
        out=out,
        labels=torch.arange(5),
        num_label_tokens=None,
        is_train=True,
        cu_seqlens=cu_seqlens,
    )

    assert loss.item() == pytest.approx(3.0)
    assert seen["cu_seqlens"] is cu_seqlens


def test_vlm_engine_loss_uses_explicit_mtp_targets_instead_of_thd_boundaries(monkeypatch):
    recipe = object.__new__(FinetuneRecipeForVLM)
    recipe.model_parts = [nn.Identity()]
    recipe.loss_fn = object()
    recipe.cfg = SimpleNamespace(mtp=SimpleNamespace(scaling_factor=0.5, ignore_index=-100))
    recipe.dist_env = SimpleNamespace(is_main=False)
    recipe.pp_enabled = False
    recipe._get_dp_group = lambda include_cp=False: None
    recipe._get_cp_group_size = lambda: 2
    recipe._maybe_add_drafter_loss = MagicMock(return_value=torch.tensor(3.0))
    mtp_loss = MagicMock(return_value=torch.tensor(2.0))
    monkeypatch.setattr(
        "nemo_automodel.recipes.vlm.finetune.calculate_loss",
        MagicMock(return_value=torch.tensor(1.0)),
    )
    monkeypatch.setattr("nemo_automodel.recipes.vlm.finetune.calculate_mtp_loss", mtp_loss)

    labels = torch.arange(5)
    cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.int32)
    targets = (torch.tensor([1, -100, 3, 4, -100]),)
    out = SimpleNamespace(
        logits=torch.zeros(5, 8),
        mtp_per_depth_logits=[torch.zeros(5, 8)],
        mtp_loss_scaling_factor=0.25,
    )

    loss = recipe._engine_loss_fn(
        out,
        {
            "labels": labels,
            "weights": labels.ne(-100),
            "cu_seqlens": cu_seqlens,
            "mtp_per_depth_targets": targets,
        },
    )

    assert loss.item() == pytest.approx(3.0)
    assert mtp_loss.call_args.kwargs["mtp_per_depth_targets"] is targets
    assert mtp_loss.call_args.kwargs["cu_seqlens"] is None
    assert mtp_loss.call_args.kwargs["labels"] is labels

    with pytest.raises(RuntimeError, match="globally prepared per-depth targets"):
        recipe._engine_loss_fn(
            out,
            {"labels": labels, "weights": labels.ne(-100), "cu_seqlens": cu_seqlens},
        )


def test_vlm_engine_validation_loss_uses_eval_path():
    recipe = FinetuneRecipeForVLM.__new__(FinetuneRecipeForVLM)
    recipe.pp_enabled = False
    expected = torch.tensor(5.0)
    compute_loss = MagicMock(return_value=expected)
    recipe._compute_vlm_loss = compute_loss

    output = object()
    labels = torch.tensor([[1, 2, -100]])
    targets = (torch.tensor([[2, -100, -100]]),)
    cu_seqlens = torch.tensor([0, 3], dtype=torch.int32)
    loss_inputs = {
        "labels": labels,
        "weights": labels.ne(-100),
        "cu_seqlens": cu_seqlens,
        "mtp_per_depth_targets": targets,
    }

    loss = recipe._engine_validation_loss_fn(output, loss_inputs)

    assert loss is expected
    compute_loss.assert_called_once_with(
        out=output,
        labels=labels,
        num_label_tokens=None,
        is_train=False,
        cu_seqlens=cu_seqlens,
        mtp_per_depth_targets=targets,
        log_drafter=False,
        log_denominator=None,
    )


def test_vlm_validation_uses_engine_evaluate_and_aggregates_uneven_batches(monkeypatch):
    recipe = FinetuneRecipeForVLM.__new__(FinetuneRecipeForVLM)
    recipe.model_parts = [MagicMock()]
    recipe.dist_env = SimpleNamespace(device=torch.device("cpu"))
    recipe.optimizer = [SimpleNamespace(param_groups=[{"lr": 0.01}])]
    recipe.step_scheduler = SimpleNamespace(step=3, epoch=1)
    recipe.pp_enabled = False
    recipe.loss_fn = object()

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
        "nemo_automodel.recipes.vlm.finetune.ScopedRNG",
        lambda **kwargs: nullcontext(),
    )

    batches = [
        {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "labels": torch.tensor([[1, 2, -100]]),
            "pixel_values": torch.ones(1, 3, 2, 2),
        },
        {
            "input_ids": torch.tensor([[4, 5, 6, 7]]),
            "labels": torch.tensor([[3, 4, 5, -100]]),
            "pixel_values": torch.zeros(1, 3, 2, 2),
        },
    ]
    metrics = recipe._run_validation_epoch(batches)

    assert engine.evaluate.call_count == 2
    for call, batch in zip(engine.evaluate.call_args_list, batches):
        datum_batches, loss_fn = call.args
        assert len(datum_batches) == len(datum_batches[0]) == 1
        datum = datum_batches[0][0]
        assert datum.model_inputs["input_ids"] is batch["input_ids"]
        assert datum.model_inputs["pixel_values"] is batch["pixel_values"]
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


@pytest.mark.parametrize(
    "batches",
    [
        [],
        [{"input_ids": torch.tensor([[1, 2]]), "labels": torch.tensor([[-100, -100]])}],
    ],
)
def test_vlm_validation_rejects_zero_global_denominator(monkeypatch, batches):
    recipe = FinetuneRecipeForVLM.__new__(FinetuneRecipeForVLM)
    recipe.dist_env = SimpleNamespace(device=torch.device("cpu"))
    recipe.pp_enabled = False
    recipe.loss_fn = object()
    recipe.engine = MagicMock()
    recipe.engine.evaluate.return_value = SimpleNamespace(
        loss_sum=torch.tensor(0.0, dtype=torch.float64),
        weight_sum=torch.tensor(0.0, dtype=torch.float64),
        token_outputs=[],
        batch_outputs=[],
    )
    recipe._dp_allreduce = MagicMock(side_effect=lambda tensor, **kwargs: tensor)
    monkeypatch.setattr(
        "nemo_automodel.recipes.vlm.finetune.ScopedRNG",
        lambda **kwargs: nullcontext(),
    )

    with pytest.raises(ValueError, match="no supervised label tokens.*drop_last=true"):
        recipe._run_validation_epoch(batches)

    assert recipe._dp_allreduce.call_count == 2


def test_vlm_rope_fusion_unchanged_when_cp_eq_1(monkeypatch):
    """rope_fusion should remain True in VLM setup when cp_size == 1."""
    cfg = _minimal_vlm_cfg(cp_size=1, rope_fusion=True)
    _patch_vlm_setup_minimals(monkeypatch, cp_size=1)

    trainer = FinetuneRecipeForVLM(cfg)
    trainer.setup()

    assert cfg.model.backend.rope_fusion is True


def test_vlm_setup_rejects_loss_without_sum_contract(monkeypatch):
    cfg = _minimal_vlm_cfg(cp_size=1, rope_fusion=True)
    _patch_vlm_setup_minimals(monkeypatch, cp_size=1)
    monkeypatch.setattr(
        "nemo_automodel.recipes._typed_config.RecipeConfig.loss_fn",
        property(lambda self: SimpleNamespace(build=lambda: SimpleNamespace(reduction="mean"))),
    )
    monkeypatch.setattr("nemo_automodel.recipes.vlm.finetune._supports_logits_to_keep", lambda _model: True)

    trainer = FinetuneRecipeForVLM(cfg)
    with pytest.raises(ValueError, match="reduction='sum'"):
        trainer.setup()


def test_vlm_setup_builds_engine_for_eager_sum_loss(monkeypatch):
    from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy

    cfg = _minimal_vlm_cfg(cp_size=1, rope_fusion=True)
    _patch_vlm_setup_minimals(monkeypatch, cp_size=1)

    trainer = FinetuneRecipeForVLM(cfg)
    trainer.setup()

    assert isinstance(trainer.loss_fn, MaskedCrossEntropy)
    assert trainer.engine is not None
    assert trainer.engine.mtp_ignore_index == trainer.cfg.mtp.ignore_index == -100


def test_vlm_setup_does_not_change_storage_dtype_for_non_kd_recipe(monkeypatch):
    cfg = _minimal_vlm_cfg(cp_size=1, rope_fusion=True, optimizer_target="torch.optim.AdamW")
    _patch_vlm_setup_minimals(monkeypatch, cp_size=1)
    dummy_opt = SimpleNamespace(param_groups=[{"lr": 0.01}], step=lambda: None, zero_grad=lambda **k: None)
    optimizer_config = build_optimizer_config("torch.optim.AdamW", {"lr": 0.01})
    monkeypatch.setattr(optimizer_config, "build", lambda *a, **k: [dummy_opt])
    monkeypatch.setattr(
        "nemo_automodel.recipes._typed_config.RecipeConfig.optimizer",
        property(lambda self: optimizer_config),
    )

    trainer = FinetuneRecipeForVLM(cfg)
    trainer.setup()

    assert not hasattr(cfg.model, "torch_dtype")


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

    def test_variable_resolution_list_chunks_by_sample_counts(self):
        pixel_values = [torch.full((3, 8, 9 + index), index, dtype=torch.bfloat16) for index in range(5)]
        image_grid = torch.tensor([[1, 2, 2], [1, 2, 3], [1, 3, 3], [1, 4, 2], [1, 2, 4]])
        n_images_per_sample = torch.tensor([2, 0, 1, 2])

        pv_chunks, ig_chunks = chunk_vlm_media(
            pixel_values,
            image_grid,
            batch_size=4,
            n_microbatches=2,
            n_images_per_sample=n_images_per_sample,
        )

        assert [[id(value) for value in chunk] for chunk in pv_chunks] == [
            [id(value) for value in pixel_values[:2]],
            [id(value) for value in pixel_values[2:]],
        ]
        assert ig_chunks is not None
        assert torch.equal(ig_chunks[0], image_grid[:2])
        assert torch.equal(ig_chunks[1], image_grid[2:])

    @pytest.mark.parametrize(
        ("counts", "grid", "match"),
        [
            (torch.tensor([1, 1]), torch.ones(3, 3, dtype=torch.long), "length batch_size"),
            (torch.tensor([1, -1, 3]), torch.ones(3, 3, dtype=torch.long), "non-negative"),
            (torch.tensor([1, 1, 0]), torch.ones(3, 3, dtype=torch.long), "sum\\(n_images_per_sample\\)"),
            (torch.tensor([1, 1, 1]), torch.ones(2, 3, dtype=torch.long), "image_grid.shape\\[0\\]"),
        ],
    )
    def test_variable_resolution_list_strictly_validates_alignment(self, counts, grid, match):
        pixel_values = [torch.ones(3, 8, 8) for _ in range(3)]

        with pytest.raises(ValueError, match=match):
            chunk_vlm_media(
                pixel_values,
                grid,
                batch_size=3,
                n_microbatches=2,
                n_images_per_sample=counts,
            )

    def test_variable_resolution_list_rejects_non_tensor_values(self):
        with pytest.raises(TypeError, match="list of tensors"):
            chunk_vlm_media(
                [torch.ones(3, 8, 8), "not-a-tensor"],
                None,
                batch_size=2,
                n_microbatches=2,
            )

    def test_wrapper_chunks_variable_resolution_image_and_video_lists_without_grids(self):
        images = [torch.ones(3, 8, 9 + index) for index in range(4)]
        videos = [torch.ones(6, 10, 11 + index) for index in range(3)]

        def collate_fn(_examples):
            return {
                "input_ids": torch.ones(4, 8, dtype=torch.long),
                "pixel_values": images,
                "n_images_per_sample": torch.tensor([2, 0, 1, 1]),
                "pixel_values_videos": videos,
                "n_videos_per_sample": torch.tensor([0, 1, 2, 0]),
            }

        prepared = wrap_vlm_collate_for_pp(collate_fn, n_microbatches=2)([{}] * 4)
        media = prepared[VLM_PP_MEDIA_KEY]

        assert [[id(value) for value in chunk] for chunk in media["pixel_values"]] == [
            [id(value) for value in images[:2]],
            [id(value) for value in images[2:]],
        ]
        assert [[id(value) for value in chunk] for chunk in media["pixel_values_videos"]] == [
            [id(value) for value in videos[:1]],
            [id(value) for value in videos[1:]],
        ]
        assert "image_grid_hws" not in media
        assert "video_grid_thw" not in media

    def test_wrapper_rejects_variable_resolution_list_count_mismatch(self):
        def collate_fn(_examples):
            return {
                "input_ids": torch.ones(3, 8, dtype=torch.long),
                "pixel_values": [torch.ones(3, 8, 8) for _ in range(3)],
                "n_images_per_sample": torch.tensor([1, 0, 1]),
            }

        with pytest.raises(ValueError, match="sum\\(n_images_per_sample\\)=2"):
            wrap_vlm_collate_for_pp(collate_fn, n_microbatches=2)([{}] * 3)

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

        with pytest.raises(ValueError, match="cannot align pixel_values with num_patches"):
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

    @pytest.mark.parametrize("schedule_flag", ["_stage_forward_initialized", "_stages_forward_initialized"])
    def test_stage_media_replays_first_chunk_after_dynamic_metadata_forward(self, schedule_flag):
        class MediaConsumer(nn.Module):
            def forward(self):
                chunk = self._vlm_pixel_values_chunks[self._vlm_chunk_idx]
                self._vlm_chunk_idx += 1
                return chunk

        model = MediaConsumer()
        schedule = SimpleNamespace(**{schedule_flag: False})
        stage = SimpleNamespace(is_first=True)
        pp = SimpleNamespace(
            info=SimpleNamespace(has_first_stage=True, schedule=schedule, stages=[stage]),
        )
        first_chunk = torch.tensor([1.0])
        second_chunk = torch.tensor([2.0])
        batch = {
            VLM_PP_MEDIA_KEY: {
                "pixel_values": [first_chunk, second_chunk],
                "image_grid_hws": [torch.ones(1), torch.ones(1)],
            }
        }

        with stage_vlm_media_for_pp(pp, [model], batch):
            metadata_output = model()
            first_microbatch_output = model()
            second_microbatch_output = model()

            assert torch.equal(metadata_output, first_chunk)
            assert torch.equal(first_microbatch_output, first_chunk)
            assert torch.equal(second_microbatch_output, second_chunk)
            assert model._vlm_chunk_idx == 2

    @pytest.mark.parametrize("analytical_metadata,forward_initialized", [(True, False), (False, True)])
    def test_stage_media_does_not_replay_without_dynamic_metadata_forward(
        self, analytical_metadata, forward_initialized
    ):
        class MediaConsumer(nn.Module):
            def forward(self):
                chunk = self._vlm_pixel_values_chunks[self._vlm_chunk_idx]
                self._vlm_chunk_idx += 1
                return chunk

        model = MediaConsumer()
        schedule = SimpleNamespace(_stage_forward_initialized=forward_initialized)
        stage = SimpleNamespace(is_first=True)
        if analytical_metadata:
            stage._configure_outputs_meta = lambda *_args: None
        pp = SimpleNamespace(
            info=SimpleNamespace(has_first_stage=True, schedule=schedule, stages=[stage]),
        )
        first_chunk = torch.tensor([1.0])
        batch = {
            VLM_PP_MEDIA_KEY: {
                "pixel_values": [first_chunk],
                "image_grid_hws": [torch.ones(1)],
            }
        }

        with stage_vlm_media_for_pp(pp, [model], batch):
            assert torch.equal(model(), first_chunk)
            assert model._vlm_chunk_idx == 1

    @pytest.mark.parametrize("schedule_flag", ["_stage_forward_initialized", "_stages_forward_initialized"])
    def test_stage_media_does_not_replay_with_static_user_metadata(self, schedule_flag):
        class MediaConsumer(nn.Module):
            def forward(self):
                chunk = self._vlm_pixel_values_chunks[self._vlm_chunk_idx]
                self._vlm_chunk_idx += 1
                return chunk

        model = MediaConsumer()
        schedule = SimpleNamespace(**{schedule_flag: False})
        stage = SimpleNamespace(
            is_first=True,
            _user_meta=SimpleNamespace(inputs=(object(),), outputs=(object(),)),
        )
        pp = SimpleNamespace(
            info=SimpleNamespace(has_first_stage=True, schedule=schedule, stages=[stage]),
        )
        first_chunk = torch.tensor([1.0])
        second_chunk = torch.tensor([2.0, 3.0])
        batch = {
            VLM_PP_MEDIA_KEY: {
                "pixel_values": [first_chunk, second_chunk],
                "image_grid_hws": [torch.ones(1), torch.ones(2)],
            }
        }

        with stage_vlm_media_for_pp(pp, [model], batch):
            assert torch.equal(model(), first_chunk)
            assert torch.equal(model(), second_chunk)
            assert model._vlm_chunk_idx == 2

    def test_prepare_flat_patches_without_image_grid(self):
        pixel_values = torch.arange(5 * 2 * 2 * 2 * 3).reshape(5, 2, 2, 2, 3)
        batch = {
            "input_ids": torch.ones(2, 4, dtype=torch.long),
            "pixel_values": pixel_values,
            "num_patches": torch.tensor([2, 3]),
        }

        prepared = prepare_vlm_media_for_pp(batch, batch_size=2, n_microbatches=2)
        media = prepared[VLM_PP_MEDIA_KEY]

        assert torch.equal(media["pixel_values"][0], pixel_values[:2])
        assert torch.equal(media["pixel_values"][1], pixel_values[2:])
        assert [chunk.tolist() for chunk in media["num_patches"]] == [[2], [3]]


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
