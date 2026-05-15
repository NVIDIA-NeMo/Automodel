# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from nemo_automodel.recipes.diffusion import train as diffusion_train
from nemo_automodel.recipes.diffusion.train import (
    TrainDiffusionRecipe,
    _calculate_throughput_metrics,
    _count_local_batch_group_samples,
    _get_diffusion_microbatch_size,
    build_model_and_optimizer,
)


def test_get_diffusion_microbatch_size_prefers_video_latents():
    batch = {
        "video_latents": torch.zeros(3, 16, 2, 4, 4),
        "text_embeddings": torch.zeros(9, 5, 64),
    }

    assert _get_diffusion_microbatch_size(batch) == 3


def test_get_diffusion_microbatch_size_uses_image_latents():
    assert _get_diffusion_microbatch_size({"image_latents": torch.zeros(2, 16, 8, 8)}) == 2


def test_get_diffusion_microbatch_size_uses_fallback_keys_and_zero_for_empty_batches():
    assert _get_diffusion_microbatch_size({"latents": torch.zeros(4, 16)}) == 4
    assert _get_diffusion_microbatch_size({"text_embeddings_2": torch.zeros(5, 7, 8)}) == 5
    assert _get_diffusion_microbatch_size({"metadata": "not-a-tensor"}) == 0


def test_count_local_batch_group_samples_sums_microbatches():
    batch_group = [
        {"image_latents": torch.zeros(2, 16, 8, 8)},
        {"video_latents": torch.zeros(3, 16, 1, 4, 4)},
    ]

    assert _count_local_batch_group_samples(batch_group) == 5


def test_calculate_throughput_metrics_uses_measured_counts():
    metrics = _calculate_throughput_metrics(
        elapsed_seconds=2.0,
        optimizer_steps=4,
        global_samples=32,
        world_size=8,
    )

    assert metrics["step_time"] == pytest.approx(0.5)
    assert metrics["optimizer_steps_per_sec"] == pytest.approx(2.0)
    assert metrics["samples_per_sec"] == pytest.approx(16.0)
    assert metrics["samples_per_sec_per_gpu"] == pytest.approx(2.0)
    assert metrics["samples_per_step"] == pytest.approx(8.0)
    assert metrics["log_window_seconds"] == pytest.approx(2.0)
    assert metrics["log_window_steps"] == pytest.approx(4.0)
    assert metrics["log_window_samples"] == pytest.approx(32.0)


def test_calculate_throughput_metrics_clamps_invalid_inputs():
    metrics = _calculate_throughput_metrics(
        elapsed_seconds=0.0,
        optimizer_steps=-1,
        global_samples=-32,
        world_size=0,
    )

    assert metrics["step_time"] == pytest.approx(1e-12)
    assert metrics["optimizer_steps_per_sec"] == pytest.approx(0.0)
    assert metrics["samples_per_sec"] == pytest.approx(0.0)
    assert metrics["samples_per_sec_per_gpu"] == pytest.approx(0.0)
    assert metrics["samples_per_step"] == pytest.approx(0.0)
    assert metrics["log_window_steps"] == pytest.approx(0.0)
    assert metrics["log_window_samples"] == pytest.approx(0.0)


class _TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.attention_backend = None

    def set_attention_backend(self, attention_backend):
        self.attention_backend = attention_backend


def test_build_model_and_optimizer_forwards_perf_options_and_optimizer_kwargs(monkeypatch):
    pipe = SimpleNamespace(transformer=_TinyTransformer())
    manager = SimpleNamespace(device_mesh="mesh")
    calls = {}

    def fake_from_pretrained(model_id, **kwargs):
        calls["model_id"] = model_id
        calls.update(kwargs)
        return pipe, {"transformer": manager}

    monkeypatch.setattr(
        diffusion_train.NeMoAutoDiffusionPipeline,
        "from_pretrained",
        staticmethod(fake_from_pretrained),
    )
    monkeypatch.setattr(diffusion_train.torch.cuda, "is_available", lambda: False)

    _, optimizer, device_mesh = build_model_and_optimizer(
        model_id="dummy-model",
        finetune_mode=True,
        learning_rate=0.125,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        fsdp_cfg={
            "dp_size": 1,
            "sequence_parallel": True,
            "tp_plan": {"layers.0": "colwise"},
            "patch_is_packed_sequence": True,
            "defer_fsdp_grad_sync": False,
            "enable_async_tensor_parallel": True,
            "enable_compile": True,
            "enable_fsdp2_prefetch": False,
            "fsdp2_backward_prefetch_depth": 4,
            "fsdp2_forward_prefetch_depth": 3,
        },
        attention_backend="flash",
        optimizer_cfg={
            "weight_decay": 0.25,
            "betas": [0.8, 0.95],
            "eps": 1e-7,
            "amsgrad": True,
            "foreach": False,
            "maximize": True,
        },
    )

    manager_args = calls["parallel_scheme"]["transformer"]
    assert manager_args["sequence_parallel"] is True
    assert manager_args["tp_plan"] == {"layers.0": "colwise"}
    assert manager_args["patch_is_packed_sequence"] is True
    assert manager_args["defer_fsdp_grad_sync"] is False
    assert manager_args["enable_async_tensor_parallel"] is True
    assert manager_args["enable_compile"] is True
    assert manager_args["enable_fsdp2_prefetch"] is False
    assert manager_args["fsdp2_backward_prefetch_depth"] == 4
    assert manager_args["fsdp2_forward_prefetch_depth"] == 3
    assert pipe.transformer.attention_backend == "flash"
    assert device_mesh == "mesh"
    assert optimizer.defaults["lr"] == pytest.approx(0.125)
    assert optimizer.defaults["weight_decay"] == pytest.approx(0.25)
    assert optimizer.defaults["betas"] == (0.8, 0.95)
    assert optimizer.defaults["eps"] == pytest.approx(1e-7)
    assert optimizer.defaults["amsgrad"] is True
    assert optimizer.defaults["foreach"] is False
    assert optimizer.defaults["maximize"] is True


def test_build_model_and_optimizer_rejects_foreach_and_fused_together(monkeypatch):
    pipe = SimpleNamespace(transformer=_TinyTransformer())
    manager = SimpleNamespace(device_mesh=None)

    monkeypatch.setattr(
        diffusion_train.NeMoAutoDiffusionPipeline,
        "from_pretrained",
        staticmethod(lambda *_args, **_kwargs: (pipe, {"transformer": manager})),
    )

    with pytest.raises(ValueError, match="foreach=True and fused=True"):
        build_model_and_optimizer(
            model_id="dummy-model",
            finetune_mode=True,
            learning_rate=0.125,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
            fsdp_cfg={"dp_size": 1},
            optimizer_cfg={"foreach": True, "fused": True},
        )


def test_recipe_timing_and_global_sample_helpers_use_distributed_reductions(monkeypatch):
    recipe = object.__new__(TrainDiffusionRecipe)
    recipe.device = torch.device("cpu")
    recipe._sync_device = MagicMock()
    recipe._get_dp_group = MagicMock(return_value="dp-group")
    all_reduce_calls = []

    def fake_all_reduce(tensor, op=None, group=None):
        all_reduce_calls.append((op, group))
        if tensor.dtype.is_floating_point:
            tensor.fill_(7.5)
        else:
            tensor.fill_(23)

    monkeypatch.setattr(diffusion_train.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(diffusion_train.dist, "get_backend", lambda: "gloo")
    monkeypatch.setattr(diffusion_train.dist, "all_reduce", fake_all_reduce)
    monkeypatch.setattr(diffusion_train.time, "perf_counter", lambda: 12.0)

    elapsed_seconds, end_time = recipe._elapsed_seconds_since(3.0)
    global_samples = recipe._count_global_samples(11)

    assert elapsed_seconds == pytest.approx(7.5)
    assert end_time == pytest.approx(12.0)
    assert global_samples == 23
    assert all_reduce_calls[0] == (diffusion_train.dist.ReduceOp.MAX, None)
    assert all_reduce_calls[1] == (diffusion_train.dist.ReduceOp.SUM, "dp-group")


def test_recipe_cuda_helpers_can_be_exercised_without_cuda_runtime(monkeypatch):
    recipe = object.__new__(TrainDiffusionRecipe)
    recipe.device = torch.device("cuda", 0)
    synchronize = MagicMock()

    monkeypatch.setattr(diffusion_train.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(diffusion_train.torch.cuda, "synchronize", synchronize)
    monkeypatch.setattr(diffusion_train.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(diffusion_train.dist, "get_backend", lambda: "nccl")

    recipe._sync_device()

    synchronize.assert_called_once_with(recipe.device)
    assert recipe._get_collective_device() == recipe.device


def test_recipe_memory_metrics_use_cuda_counters_and_rank_max(monkeypatch):
    recipe = object.__new__(TrainDiffusionRecipe)
    recipe.device = torch.device("cuda", 0)
    scale = 1024**3

    def fake_all_reduce(tensor, op=None, group=None):
        assert op == diffusion_train.dist.ReduceOp.MAX
        assert group is None
        tensor.add_(0.5)

    monkeypatch.setattr(diffusion_train.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(diffusion_train.torch.cuda, "memory_allocated", lambda device: 1 * scale)
    monkeypatch.setattr(diffusion_train.torch.cuda, "memory_reserved", lambda device: 2 * scale)
    monkeypatch.setattr(diffusion_train.torch.cuda, "max_memory_allocated", lambda device: 3 * scale)
    monkeypatch.setattr(diffusion_train.torch.cuda, "max_memory_reserved", lambda device: 4 * scale)
    monkeypatch.setattr(diffusion_train.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(diffusion_train.dist, "get_backend", lambda: "gloo")
    monkeypatch.setattr(diffusion_train.dist, "all_reduce", fake_all_reduce)

    metrics = recipe._get_memory_metrics()

    assert metrics["memory_allocated_gb"] == pytest.approx(1.5)
    assert metrics["memory_reserved_gb"] == pytest.approx(2.5)
    assert metrics["max_memory_allocated_gb"] == pytest.approx(3.5)
    assert metrics["max_memory_reserved_gb"] == pytest.approx(4.5)
    assert metrics["mem"] == pytest.approx(3.5)


class _FakeProgressBar:
    def __init__(self, iterable, desc):
        self.iterable = iterable
        self.desc = desc
        self.postfix = None

    def set_postfix(self, postfix):
        self.postfix = postfix


class _FakeStepScheduler:
    def __init__(self, batch_group):
        self.step = 0
        self.epochs = [0]
        self.dataloader = None
        self.is_ckpt_step = False
        self._batch_group = batch_group

    def __iter__(self):
        self.step = 1
        yield self._batch_group


def test_run_train_validation_loop_uses_hot_path_and_logs_perf_metrics(monkeypatch):
    recipe = object.__new__(TrainDiffusionRecipe)
    model = nn.Linear(1, 1)
    batch_group = [
        {"video_latents": torch.zeros(2, 1), "text_embeddings": torch.zeros(2, 1)},
        {"image_latents": torch.zeros(3, 1), "text_embeddings": torch.zeros(3, 1)},
    ]
    progress_bars = []

    def fake_tqdm(iterable, desc):
        progress_bar = _FakeProgressBar(iterable, desc)
        progress_bars.append(progress_bar)
        return progress_bar

    monkeypatch.setitem(sys.modules, "tqdm", SimpleNamespace(tqdm=fake_tqdm))
    monkeypatch.setattr(diffusion_train, "prepare_for_grad_accumulation", MagicMock())
    monkeypatch.setattr(diffusion_train, "prepare_for_final_backward", MagicMock())
    monkeypatch.setattr(diffusion_train, "prepare_after_first_microbatch", MagicMock())
    monkeypatch.setattr(diffusion_train, "clip_grad_norm", MagicMock(return_value=torch.tensor(0.25)))
    monkeypatch.setattr(diffusion_train.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(diffusion_train, "is_main_process", lambda: True)
    monkeypatch.setattr(diffusion_train.wandb, "run", None, raising=False)

    recipe.global_batch_size = 5
    recipe.local_batch_size = 2
    recipe.num_nodes = 1
    recipe.dp_size = 1
    recipe.world_size = 1
    recipe.num_epochs = 1
    recipe.sampler = SimpleNamespace(set_epoch=MagicMock())
    recipe.dataloader = [object()]
    recipe.step_scheduler = _FakeStepScheduler(batch_group)
    recipe.optimizer = SimpleNamespace(
        zero_grad=MagicMock(),
        step=MagicMock(),
        param_groups=[{"lr": 0.01}],
    )
    recipe.lr_scheduler = [SimpleNamespace(step=MagicMock())]
    recipe.model = model
    recipe.device = torch.device("cpu")
    recipe.bf16 = torch.float32
    recipe.check_loss = True
    recipe.clip_grad_max_norm = 0.5
    recipe.grad_clip_foreach = False
    recipe.peft_cfg = None
    recipe.log_every = 1
    recipe._elapsed_seconds_since = MagicMock(return_value=(2.0, 10.0))
    recipe._count_global_samples = MagicMock(return_value=5)
    recipe._get_memory_metrics = MagicMock(
        return_value={
            "mem": 0.0,
            "memory_allocated_gb": 0.0,
            "memory_reserved_gb": 0.0,
            "max_memory_allocated_gb": 0.0,
            "max_memory_reserved_gb": 0.0,
        }
    )
    recipe.save_checkpoint = MagicMock()
    recipe.flow_matching_pipeline = SimpleNamespace(
        step=MagicMock(
            side_effect=[
                (None, torch.tensor(2.0, requires_grad=True), None, {}),
                (None, torch.tensor(4.0, requires_grad=True), None, {}),
            ]
        )
    )

    recipe.run_train_validation_loop()

    diffusion_train.prepare_for_grad_accumulation.assert_called_once_with([model], pp_enabled=False)
    diffusion_train.prepare_for_final_backward.assert_called_once_with([model], pp_enabled=False)
    diffusion_train.prepare_after_first_microbatch.assert_called_once()
    diffusion_train.clip_grad_norm.assert_called_once_with(0.5, [model], foreach=False)
    recipe.optimizer.zero_grad.assert_called_once_with(set_to_none=True)
    recipe.optimizer.step.assert_called_once()
    recipe.lr_scheduler[0].step.assert_called_once_with(1)
    recipe._count_global_samples.assert_called_once_with(5)
    recipe.save_checkpoint.assert_not_called()
    assert recipe.flow_matching_pipeline.step.call_args_list[0].kwargs["collect_metrics"] is False
    assert recipe.flow_matching_pipeline.step.call_args_list[0].kwargs["check_loss"] is True
    assert progress_bars[0].postfix == {
        "loss": "3.0000",
        "avg": "3.0000",
        "lr": "1.00e-02",
        "gn": "0.25",
        "s/s": "2.5",
        "s/s/gpu": "2.50",
    }
