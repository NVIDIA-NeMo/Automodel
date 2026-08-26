# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from collections import deque
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from nemo_automodel.components.distributed.config import DDPConfig, FSDP2Config
from nemo_automodel.recipes.retrieval import train_bi_encoder
from nemo_automodel.recipes.retrieval.train_bi_encoder import (
    TrainBiEncoderRecipe,
    _configure_sentence_transformer_export,
    _get_autocast_ctx,
    _get_model_instantiate_kwargs,
    _unwrap_model_for_attrs,
    _uses_multi_vector_scoring,
)


class _RetrieverAttrs(torch.nn.Module):
    pooling = "multi_vector"
    l2_normalize = True
    do_distributed_inbatch_negative = True
    detach_distributed_inbatch_negatives = False


class _DDPLikeWrapper(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module


class _DictLikeConfig(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)


def test_retrieval_attrs_unwrap_ddp_like_wrapper():
    inner = _RetrieverAttrs()
    wrapped = _DDPLikeWrapper(inner)

    attr_model = _unwrap_model_for_attrs(wrapped)

    assert attr_model is inner
    assert attr_model.l2_normalize is True
    assert attr_model.do_distributed_inbatch_negative is True
    assert attr_model.detach_distributed_inbatch_negatives is False
    assert _uses_multi_vector_scoring(wrapped) is True


def test_retrieval_attrs_accept_unwrapped_model():
    inner = _RetrieverAttrs()

    assert _unwrap_model_for_attrs(inner) is inner
    assert _uses_multi_vector_scoring(inner) is True


def test_configure_sentence_transformer_export_binds_exact_static_collator_prompts():
    captured = {}

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def configure_sentence_transformer_prompts(self, **kwargs):
            captured.update(kwargs)

    collator = SimpleNamespace(
        query_prefix="query:",
        passage_prefix="passage:",
        use_dataset_instruction=False,
    )

    wrapped = _DDPLikeWrapper(_Model())
    _configure_sentence_transformer_export(wrapped, collator)

    assert captured == {"query_prompt": "query: ", "document_prompt": "passage: "}


def test_configure_sentence_transformer_export_ignores_collator_when_export_is_disabled():
    class _Model:
        sentence_transformer_export_config = None

        def configure_sentence_transformer_prompts(self, **kwargs):
            raise AssertionError(f"disabled export must not configure prompts: {kwargs}")

    _configure_sentence_transformer_export(_Model(), SimpleNamespace())


def test_configure_sentence_transformer_export_disables_export_for_dataset_instructions(caplog):
    class _Model:
        sentence_transformer_export_config = object()

        def configure_sentence_transformer_prompts(self, **kwargs):
            raise AssertionError(f"static prompts must not be configured: {kwargs}")

        def disable_sentence_transformer_export(self):
            self.sentence_transformer_export_config = None

    collator = SimpleNamespace(
        query_prefix="ignored query:",
        passage_prefix="ignored passage:",
        use_dataset_instruction=True,
    )

    model = _Model()
    _configure_sentence_transformer_export(model, collator)

    assert model.sentence_transformer_export_config is None
    assert "per-example dataset instructions" in caplog.text


def test_configure_sentence_transformer_export_disables_export_for_custom_collator(caplog):
    class _Model:
        sentence_transformer_export_config = object()

        def configure_sentence_transformer_prompts(self, **kwargs):
            raise AssertionError(f"unknown custom preprocessing must not configure prompts: {kwargs}")

        def disable_sentence_transformer_export(self):
            self.sentence_transformer_export_config = None

    class _CustomCollator:
        def __call__(self, examples):
            return examples

    model = _Model()
    collator = _CustomCollator()
    assert callable(collator)

    _configure_sentence_transformer_export(model, collator)

    assert model.sentence_transformer_export_config is None
    assert "does not expose static query_prefix and passage_prefix metadata" in caplog.text


def test_retrieval_autocast_ctx_disabled_by_default(monkeypatch):
    def _unexpected_autocast(*args, **kwargs):
        raise AssertionError("autocast should be disabled when autocast_dtype is unset")

    monkeypatch.setattr(train_bi_encoder.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(train_bi_encoder.torch, "autocast", _unexpected_autocast)

    with _get_autocast_ctx(SimpleNamespace(autocast_dtype=None)):
        pass


def test_retrieval_autocast_ctx_uses_configured_dtype(monkeypatch):
    captured = {}

    def _fake_autocast(*, device_type, dtype):
        captured["device_type"] = device_type
        captured["dtype"] = dtype
        return nullcontext()

    monkeypatch.setattr(train_bi_encoder.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(train_bi_encoder.torch, "autocast", _fake_autocast)

    with _get_autocast_ctx(SimpleNamespace(autocast_dtype=torch.bfloat16)):
        pass

    assert captured == {"device_type": "cuda", "dtype": torch.bfloat16}


def test_retrieval_model_instantiate_kwargs_include_compile_config():
    distributed_setup = object()
    peft_config = object()
    cfg = _DictLikeConfig(compile={"enabled": True, "mode": "reduce-overhead", "dynamic": False})

    kwargs = _get_model_instantiate_kwargs(cfg, distributed_setup, peft_config)

    assert kwargs["distributed_setup"] is distributed_setup
    assert kwargs["peft_config"] is peft_config
    assert kwargs["compile_config"].enabled is True
    assert kwargs["compile_config"].mode == "reduce-overhead"
    assert kwargs["compile_config"].dynamic is False


class _FakeCheckpointer:
    def maybe_wait_for_staging(self):
        pass


class _FakeOptimizer:
    param_groups = [{"lr": 1e-5}]

    def step(self):
        pass

    def zero_grad(self):
        pass


class _FakeStepScheduler:
    step = 1
    epoch = 0


def _make_recipe_for_optim_step(distributed_config):
    recipe = TrainBiEncoderRecipe.__new__(TrainBiEncoderRecipe)
    recipe.distributed_config = distributed_config
    recipe.model_parts = [torch.nn.Linear(1, 1)]
    recipe.pp_enabled = False
    recipe.device_mesh = None
    recipe.moe_mesh = None
    recipe.checkpointer = _FakeCheckpointer()
    recipe.optimizer = [_FakeOptimizer()]
    recipe.lr_scheduler = None
    recipe.step_scheduler = _FakeStepScheduler()
    recipe.loss_average_window = deque(maxlen=50)
    recipe.timestamp = 0.0
    recipe._get_dp_group_size = lambda include_cp=True: 1

    def _fake_forward_backward_step(*args, **kwargs):
        kwargs["loss_buffer"].append(torch.tensor(1.0))

    recipe._forward_backward_step = _fake_forward_backward_step
    return recipe


def test_retrieval_optim_step_uses_torch_clip_fast_path_for_ddp(monkeypatch):
    captured = {}

    def _fake_scale_grads_and_clip_grad_norm(*args, **kwargs):
        captured.update(kwargs)
        return 0.0

    monkeypatch.setattr(train_bi_encoder, "scale_grads_and_clip_grad_norm", _fake_scale_grads_and_clip_grad_norm)

    recipe = _make_recipe_for_optim_step(DDPConfig())
    recipe._run_train_optim_step([{}], max_grad_norm=1.0)

    assert captured["use_torch_clip_grad_norm"] is True


def test_retrieval_optim_step_keeps_sharded_clip_path_for_fsdp2(monkeypatch):
    captured = {}

    def _fake_scale_grads_and_clip_grad_norm(*args, **kwargs):
        captured.update(kwargs)
        return 0.0

    monkeypatch.setattr(train_bi_encoder, "scale_grads_and_clip_grad_norm", _fake_scale_grads_and_clip_grad_norm)

    recipe = _make_recipe_for_optim_step(FSDP2Config())
    recipe._run_train_optim_step([{}], max_grad_norm=1.0)

    assert captured["use_torch_clip_grad_norm"] is False


class _RecordingMetricLogger:
    """Buffers like MetricLogger but records what actually reached "disk" and when."""

    def __init__(self):
        self.buffer = []
        self.persisted = []
        self.closed = False

    def log(self, record):
        self.buffer.append(record)

    def flush(self):
        self.persisted.extend(self.buffer)
        self.buffer = []

    def close(self):
        self.flush()
        self.closed = True


class _LoopStepScheduler:
    """Minimal StepScheduler stand-in driving a fixed number of steps.

    ``sigterm_at`` makes the signal appear at that step, mirroring the real scheduler's
    sticky flag: once raised it stays raised, and the epoch/step iterators stop on it.
    """

    def __init__(self, n_steps, ckpt_steps=(), sigterm_at=None, val_steps=()):
        self.step = 0
        self.epoch = 0
        self._n = n_steps
        self._ckpt = set(ckpt_steps)
        self._val = set(val_steps)
        self._sigterm_at = sigterm_at
        self.sigterm_flag = False
        self.steps_run = []

    @property
    def epochs(self):
        for e in range(1):
            if self.sigterm_received:
                return
            yield e

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        while self.step < self._n:
            self.step += 1
            if self.sigterm_flag:
                return
            self.steps_run.append(self.step)
            yield [{}]

    @property
    def sigterm_received(self):
        if self.sigterm_flag:
            return True
        if self._sigterm_at is not None and self.step >= self._sigterm_at:
            self.sigterm_flag = True
        return self.sigterm_flag

    @property
    def is_ckpt_step(self):
        return self.step in self._ckpt or self.sigterm_received

    @property
    def is_val_step(self):
        return self.step in self._val


def _make_loop_recipe(step_scheduler, *, raise_at=None):
    recipe = TrainBiEncoderRecipe.__new__(TrainBiEncoderRecipe)
    recipe.model_parts = [torch.nn.Linear(1, 1)]
    recipe.step_scheduler = step_scheduler
    recipe.dataloader = SimpleNamespace(dataset=SimpleNamespace())
    recipe.val_dataloader = None
    recipe.max_grad_norm = 1.0
    recipe.timestamp = 0.0
    recipe.metric_logger_train = _RecordingMetricLogger()
    recipe.metric_logger_valid = _RecordingMetricLogger()
    recipe.saved_at = []

    def _optim_step(batches, max_grad_norm):
        if raise_at is not None and step_scheduler.step == raise_at:
            raise RuntimeError("simulated CUDA OOM")
        return SimpleNamespace(metrics={"loss": 1.0})

    recipe._run_train_optim_step = _optim_step
    recipe.log_train_metrics = lambda d: recipe.metric_logger_train.log(d)
    recipe._make_progress_bar = lambda: None
    recipe._update_progress_bar = lambda pbar, metrics: None
    recipe._maybe_collect_garbage = lambda: None
    recipe._finalize_and_close_checkpointer = lambda: None
    recipe.save_checkpoint = lambda *a, **k: recipe.saved_at.append(step_scheduler.step)
    return recipe


def test_sigterm_on_checkpoint_step_saves_then_stops_the_loop():
    """A signal on a scheduled checkpoint step: the checkpoint is written, then the
    post-checkpoint poll stops the loop instead of running further steps."""
    sched = _LoopStepScheduler(n_steps=10, ckpt_steps={4}, sigterm_at=4)
    recipe = _make_loop_recipe(sched)

    recipe.run_train_validation_loop()

    assert sched.steps_run == [1, 2, 3, 4], "loop must stop after the signalled step"
    assert recipe.saved_at == [4], "checkpoint must still be taken"
    assert recipe.metric_logger_train.closed
    assert len(recipe.metric_logger_train.persisted) == 4, "every step's metrics must survive"


def test_metrics_persisted_when_the_loop_raises():
    """An exception mid-loop must still persist buffered metrics, because the loggers are
    closed in the finally rather than after it."""
    sched = _LoopStepScheduler(n_steps=10)
    recipe = _make_loop_recipe(sched, raise_at=3)

    with pytest.raises(RuntimeError, match="simulated CUDA OOM"):
        recipe.run_train_validation_loop()

    assert recipe.metric_logger_train.closed, "logger must be closed from the finally"
    assert recipe.metric_logger_valid.closed
    # steps 1 and 2 logged before the raise; both must have reached disk
    assert len(recipe.metric_logger_train.persisted) == 2


def test_checkpoint_flushes_metrics_when_not_buffer_aligned():
    """The durability guarantee comes from the explicit flush, not from record counting.

    The checkpoint here lands at step 3, which no record-count boundary coincides with --
    the situation after resuming at a step that is not a multiple of ckpt_every_steps, or at
    an epoch-boundary checkpoint. The metrics for the covered steps must be on disk by the
    time the checkpoint is taken.
    """
    sched = _LoopStepScheduler(n_steps=6, ckpt_steps={3})
    recipe = _make_loop_recipe(sched)
    persisted_at_ckpt = {}
    original_save = recipe.save_checkpoint

    def _save(*a, **k):
        original_save(*a, **k)
        # sample AFTER the recipe's flush by deferring the read to the next step
        persisted_at_ckpt["at_save"] = len(recipe.metric_logger_train.persisted)

    recipe.save_checkpoint = _save
    recipe.run_train_validation_loop()

    # the flush runs immediately after save_checkpoint returns, so by step 4 the first three
    # steps are durable rather than sitting in the buffer
    assert recipe.saved_at == [3]
    assert len(recipe.metric_logger_train.persisted) == 6
    assert persisted_at_ckpt["at_save"] == 0, "flush is after the save, not before"
