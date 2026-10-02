# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Diffusion recipe wiring for custom-model MoE transformers with expert parallelism."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest
import torch
import torch.nn as nn

from nemo_automodel.components.checkpoint.config import CheckpointingConfig
from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.components.distributed.config import MoEParallelizerConfig
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.moe.megatron.moe_utils import MoEAuxLossAutoScaler
from nemo_automodel.recipes.diffusion import train as diffusion_train
from nemo_automodel.recipes.diffusion.train import TrainDiffusionRecipe, _build_diffusion_mesh_context

# =============================================================================
# Mesh construction
# =============================================================================


def test_build_diffusion_mesh_context_forwards_moe_parallel_config_with_ep():
    with patch.object(diffusion_train.MeshContext, "build") as build_mesh:
        _build_diffusion_mesh_context(
            fsdp_cfg={"ep_size": 2, "moe": {"reshard_after_forward": True}},
            ddp_cfg=None,
            world_size=4,
            dtype=torch.bfloat16,
            lora_enabled=False,
        )

    kwargs = build_mesh.call_args.kwargs
    assert kwargs["parallelism_sizes"].ep_size == 2
    moe_parallel_config = kwargs["moe_parallel_config"]
    assert isinstance(moe_parallel_config, MoEParallelizerConfig)
    assert moe_parallel_config.reshard_after_forward is True


def test_build_diffusion_mesh_context_has_no_moe_parallel_config_without_ep():
    with patch.object(diffusion_train.MeshContext, "build") as build_mesh:
        _build_diffusion_mesh_context(
            fsdp_cfg={"ep_size": 1},
            ddp_cfg=None,
            world_size=4,
            dtype=torch.bfloat16,
            lora_enabled=False,
        )

    kwargs = build_mesh.call_args.kwargs
    assert kwargs["parallelism_sizes"].ep_size == 1
    assert kwargs["moe_parallel_config"] is None


# =============================================================================
# Recipe setup
# =============================================================================


class _StopAfterCheckpointer(Exception):
    pass


def _moe_recipe_cfg(**model_overrides):
    model = {"pretrained_model_name_or_path": "dummy-custom-moe", "mode": "finetune", **model_overrides}
    return ConfigNode(
        {
            "model": model,
            "flow_matching": {"adapter_type": "simple"},
            "optimizer": {"_target_": "torch.optim.AdamW", "lr": 1.0e-4},
            "performance": {},
            "fsdp": {"ep_size": 2},
            "step_scheduler": {"num_epochs": 1, "local_batch_size": 1, "global_batch_size": 1},
            "checkpoint": {"enabled": False, "checkpoint_dir": "/tmp/unused-diffusion-moe-ckpt"},
        }
    )


def _patch_setup_until_checkpointer(monkeypatch, mesh_context):
    monkeypatch.setattr(
        diffusion_train, "initialize_distributed", lambda *args, **kwargs: SimpleNamespace(is_main=False)
    )
    monkeypatch.setattr(diffusion_train, "setup_logging", lambda: None)
    monkeypatch.setattr(diffusion_train, "StatefulRNG", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(diffusion_train.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(diffusion_train.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(diffusion_train, "broadcast_tp_replicas", MagicMock())
    build_pipeline = MagicMock(return_value=(SimpleNamespace(transformer=nn.Linear(1, 1)), mesh_context))
    monkeypatch.setattr(diffusion_train, "build_diffusion_pipeline", build_pipeline)
    checkpointer_build = MagicMock(side_effect=_StopAfterCheckpointer)
    monkeypatch.setattr(CheckpointingConfig, "build", checkpointer_build)
    return build_pipeline, checkpointer_build


def test_setup_keeps_moe_mesh_and_passes_it_to_the_checkpointer(monkeypatch):
    moe_mesh = SimpleNamespace(mesh_dim_names=("ep_shard", "ep"))
    mesh_context = SimpleNamespace(device_mesh=None, moe_mesh=moe_mesh)
    build_pipeline, checkpointer_build = _patch_setup_until_checkpointer(monkeypatch, mesh_context)

    recipe = TrainDiffusionRecipe(
        _moe_recipe_cfg(
            backend={"experts": "torch", "dispatcher": "torch"},
            config_overrides={"num_hidden_layers": 2},
        )
    )
    with pytest.raises(_StopAfterCheckpointer):
        recipe.setup()

    pipeline_kwargs = build_pipeline.call_args.kwargs
    # The YAML backend section becomes a BackendConfig at the recipe boundary; overrides stay a plain dict.
    assert pipeline_kwargs["backend"] == BackendConfig(experts="torch", dispatcher="torch")
    assert pipeline_kwargs["config_overrides"] == {"num_hidden_layers": 2}
    assert type(pipeline_kwargs["config_overrides"]) is dict
    assert recipe.moe_mesh is moe_mesh
    assert recipe.device_mesh is None
    assert recipe.pp_enabled is False
    assert checkpointer_build.call_args.kwargs["moe_mesh"] is moe_mesh


def test_setup_without_custom_options_or_moe_mesh(monkeypatch):
    mesh_context = SimpleNamespace(device_mesh=None, moe_mesh=None)
    build_pipeline, checkpointer_build = _patch_setup_until_checkpointer(monkeypatch, mesh_context)

    recipe = TrainDiffusionRecipe(_moe_recipe_cfg())
    with pytest.raises(_StopAfterCheckpointer):
        recipe.setup()

    assert build_pipeline.call_args.kwargs["backend"] is None
    assert build_pipeline.call_args.kwargs["config_overrides"] is None
    assert recipe.moe_mesh is None
    assert checkpointer_build.call_args.kwargs["moe_mesh"] is None


# =============================================================================
# Training step
# =============================================================================


class _GateBiasModel(nn.Linear):
    def __init__(self):
        super().__init__(1, 1)
        self.update_moe_gate_bias = MagicMock()


class _StepScheduler:
    """Yields one accumulation window (list of microbatches) per optimizer step."""

    def __init__(self, batch_groups):
        self.step = 0
        self.epochs = [0]
        self.dataloader = None
        self.is_ckpt_step = False
        self.is_val_step = False
        self.log_remote_every_steps = 0
        self._batch_groups = batch_groups

    def __iter__(self):
        for group in self._batch_groups:
            self.step += 1
            yield group


class _FakeDDP(nn.Module):
    """Stands in for DistributedDataParallel: the recipe must look through ``.module``."""

    def __init__(self, module):
        super().__init__()
        self.module = module


def _microbatch():
    return {"video_latents": torch.zeros(1, 1), "text_embeddings": torch.zeros(1, 1)}


def _loop_recipe(monkeypatch, *, model, moe_mesh, microbatches_per_step=(2,)):
    monkeypatch.setitem(sys.modules, "tqdm", SimpleNamespace(tqdm=lambda iterable, desc: iterable))
    for name in (
        "prepare_for_grad_accumulation",
        "prepare_for_final_backward",
        "prepare_after_first_microbatch",
        "synchronize_tp_replica_gradients",
    ):
        monkeypatch.setattr(diffusion_train, name, MagicMock())
    monkeypatch.setattr(diffusion_train, "clip_grad_norm", MagicMock(return_value=torch.tensor(0.25)))
    monkeypatch.setattr(diffusion_train, "scale_grads_and_clip_grad_norm", MagicMock(return_value=torch.tensor(0.5)))
    monkeypatch.setattr(diffusion_train, "get_expert_tp_replication_factor", MagicMock(return_value=1))
    monkeypatch.setattr(diffusion_train.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(diffusion_train.wandb, "run", None, raising=False)

    recipe = object.__new__(TrainDiffusionRecipe)
    batch_groups = [[_microbatch() for _ in range(n)] for n in microbatches_per_step]
    recipe.dist_env = SimpleNamespace(is_main=False)
    recipe.global_batch_size = 2
    recipe.local_batch_size = 1
    recipe.num_nodes = 1
    recipe.dp_size = 2
    recipe.cp_size = 1
    recipe.world_size = 2
    recipe.num_epochs = 1
    recipe.sampler = None
    recipe.dataloader = [object()]
    recipe.step_scheduler = _StepScheduler(batch_groups)
    recipe.val_dataloader = None
    recipe.optimizer = [SimpleNamespace(zero_grad=MagicMock(), step=MagicMock(), param_groups=[{"lr": 0.01}])]
    recipe.lr_scheduler = None
    recipe.model = model
    recipe.device_mesh = object()
    recipe.moe_mesh = moe_mesh
    recipe.pp_enabled = False
    recipe.device = torch.device("cpu")
    recipe.compute_dtype = torch.float32
    recipe.check_loss = False
    recipe.clip_grad_max_norm = 1.0
    recipe.grad_clip_foreach = True
    recipe.defer_fsdp_grad_sync = True
    recipe.transformer_engine_fp8 = False
    recipe._autocast_dtype = None
    recipe.peft_cfg = None
    recipe._get_dp_group_size = MagicMock(return_value=4)
    recipe._get_cp_group_size = MagicMock(return_value=1)
    recipe.save_checkpoint = MagicMock()
    recipe._finalize_and_close_checkpointer = MagicMock()
    aux_scales = []

    def step(*args, **kwargs):
        # Record the auxiliary-loss scale in effect for every microbatch's backward.
        aux_scales.append(MoEAuxLossAutoScaler.main_loss_backward_scale.item())
        return (None, torch.tensor(float(len(aux_scales)), requires_grad=True), None, {})

    recipe.flow_matching_pipeline = SimpleNamespace(step=MagicMock(side_effect=step))
    recipe.aux_scales_seen = aux_scales
    return recipe


def test_train_step_with_moe_mesh_uses_ep_aware_clipping_and_updates_gate_bias(monkeypatch):
    monkeypatch.setattr(MoEAuxLossAutoScaler, "main_loss_backward_scale", torch.tensor(1.0))
    model = _GateBiasModel()
    moe_mesh = SimpleNamespace(mesh_dim_names=("ep_shard", "ep"))
    recipe = _loop_recipe(monkeypatch, model=model, moe_mesh=moe_mesh)
    order = MagicMock()
    order.attach_mock(recipe.optimizer[0].step, "optimizer_step")
    order.attach_mock(model.update_moe_gate_bias, "update_moe_gate_bias")

    recipe.run_train_validation_loop()

    diffusion_train.scale_grads_and_clip_grad_norm.assert_called_once_with(
        1.0,
        [model],
        device_mesh=recipe.device_mesh,
        moe_mesh=moe_mesh,
        ep_axis_name="ep",
        foreach=True,
        dp_group_size=4,
        expert_tp_replication_factor=1,
    )
    recipe._get_dp_group_size.assert_called_with(include_cp=True)
    diffusion_train.get_expert_tp_replication_factor.assert_called_once_with([model], recipe.device_mesh)
    diffusion_train.clip_grad_norm.assert_not_called()
    # Router aux losses are averaged over the 2 accumulation microbatches (cp_size 1, no PP).
    assert MoEAuxLossAutoScaler.main_loss_backward_scale.item() == pytest.approx(0.5)
    assert order.mock_calls == [call.optimizer_step(), call.update_moe_gate_bias()]


def test_train_step_with_moe_mesh_without_ep_axis_passes_no_axis_name(monkeypatch):
    monkeypatch.setattr(MoEAuxLossAutoScaler, "main_loss_backward_scale", torch.tensor(1.0))
    recipe = _loop_recipe(monkeypatch, model=nn.Linear(1, 1), moe_mesh=SimpleNamespace(mesh_dim_names=None))

    recipe.run_train_validation_loop()

    assert diffusion_train.scale_grads_and_clip_grad_norm.call_args.kwargs["ep_axis_name"] is None


def test_train_step_without_moe_mesh_uses_dense_clipping(monkeypatch):
    monkeypatch.setattr(MoEAuxLossAutoScaler, "main_loss_backward_scale", torch.tensor(1.0))
    model = _GateBiasModel()
    recipe = _loop_recipe(monkeypatch, model=model, moe_mesh=None)

    recipe.run_train_validation_loop()

    diffusion_train.clip_grad_norm.assert_called_once_with(
        1.0,
        [model],
        device_mesh=recipe.device_mesh,
        foreach=True,
    )
    diffusion_train.scale_grads_and_clip_grad_norm.assert_not_called()
    # Routers inject auxiliary gradients at EP=1 too: the scale averages the 2 microbatches regardless of EP.
    assert recipe.aux_scales_seen == pytest.approx([0.5, 0.5])
    # Custom MoE models without EP still balance routers with gate biases.
    model.update_moe_gate_bias.assert_called_once_with()


@pytest.mark.parametrize("moe_mesh", [None, SimpleNamespace(mesh_dim_names=("ep_shard", "ep"))])
def test_aux_loss_scale_follows_each_accumulation_window(monkeypatch, moe_mesh):
    """Scale = 1 / microbatches of the current window, including a shorter final window."""
    monkeypatch.setattr(MoEAuxLossAutoScaler, "main_loss_backward_scale", torch.tensor(1.0))
    recipe = _loop_recipe(monkeypatch, model=_GateBiasModel(), moe_mesh=moe_mesh, microbatches_per_step=(3, 2, 1))

    recipe.run_train_validation_loop()

    assert recipe.aux_scales_seen == pytest.approx([1 / 3, 1 / 3, 1 / 3, 0.5, 0.5, 1.0])
    assert recipe.optimizer[0].step.call_count == 3


def test_train_step_updates_gate_bias_through_ddp_wrapper(monkeypatch):
    monkeypatch.setattr(MoEAuxLossAutoScaler, "main_loss_backward_scale", torch.tensor(1.0))
    monkeypatch.setattr(diffusion_train, "DistributedDataParallel", _FakeDDP)
    inner = _GateBiasModel()
    recipe = _loop_recipe(monkeypatch, model=_FakeDDP(inner), moe_mesh=None)

    recipe.run_train_validation_loop()

    inner.update_moe_gate_bias.assert_called_once_with()


def test_train_step_skips_gate_bias_update_for_models_without_it(monkeypatch):
    monkeypatch.setattr(MoEAuxLossAutoScaler, "main_loss_backward_scale", torch.tensor(1.0))
    model = nn.Linear(1, 1)
    recipe = _loop_recipe(monkeypatch, model=model, moe_mesh=None)

    recipe.run_train_validation_loop()

    recipe.optimizer[0].step.assert_called_once_with()
