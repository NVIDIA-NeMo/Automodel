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

import torch

from nemo_automodel.components.distributed.config import DDPConfig, FSDP2Config
from nemo_automodel.recipes.retrieval import train_bi_encoder
from nemo_automodel.recipes.retrieval.train_bi_encoder import (
    TrainBiEncoderRecipe,
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
