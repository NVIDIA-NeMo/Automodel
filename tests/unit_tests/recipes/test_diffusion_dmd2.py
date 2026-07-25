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

"""Lightweight contracts for DMD2 in the native diffusion trainer."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import yaml

from nemo_automodel.recipes.base_recipe import is_distributed_stateful
from nemo_automodel.recipes.diffusion import dmd2 as dmd2_module
from nemo_automodel.recipes.diffusion.dmd2 import DMD2Objective, _load_negative_prompt_embedding
from nemo_automodel.recipes.diffusion.train import TrainDiffusionRecipe

_REPO_ROOT = Path(__file__).resolve().parents[3]
_EXAMPLE_CONFIG = _REPO_ROOT / "examples" / "diffusion" / "dmd2" / "qwen_image_dmd2.yaml"


class _ObjectiveConfig(dict):
    def __init__(self, **values) -> None:
        super().__init__(values)
        self.modelopt_config = SimpleNamespace(to_dict=dict)

    __getattr__ = dict.__getitem__


def _objective(*, student_update_freq: int = 5) -> DMD2Objective:
    config = SimpleNamespace(
        student_update_freq=student_update_freq,
        guidance_scale=None,
        gan_loss_weight_gen=0.03,
        ema=None,
    )
    fastgen = SimpleNamespace(DMDConfig=Mock(return_value=config))
    with patch.object(dmd2_module, "_require_qwen_fastgen", return_value=(fastgen, None, None)):
        return DMD2Objective(_ObjectiveConfig())


def _linear() -> torch.nn.Linear:
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    return model


def _loss(model, name):
    value = model.weight.square().sum()
    return {"total": value, name: value}


def _training_fixture():
    objective = _objective()
    student = _linear()
    fake_score = _linear()
    discriminator = _linear()

    student_optimizer = torch.optim.SGD(student.parameters(), lr=0.1)
    fake_score_optimizer = torch.optim.SGD(fake_score.parameters(), lr=0.1)
    discriminator_optimizer = torch.optim.SGD(discriminator.parameters(), lr=0.1)

    def _discriminator_loss(*_args, **_kwargs):
        assert fake_score.weight.grad is not None
        return _loss(discriminator, "gan")

    modelopt_pipeline = SimpleNamespace(
        compute_student_loss=Mock(side_effect=lambda *_args, **_kwargs: _loss(student, "vsd")),
        compute_fake_score_loss=Mock(side_effect=lambda *_args, **_kwargs: _loss(fake_score, "dsm")),
        compute_discriminator_loss=Mock(side_effect=_discriminator_loss),
        update_ema=Mock(),
        ema=None,
    )
    objective.modelopt_pipeline = modelopt_pipeline
    objective.teacher = torch.nn.Identity()
    objective.fake_score = fake_score
    objective.fake_score_optimizer = fake_score_optimizer
    objective.discriminator = discriminator
    objective.discriminator_optimizer = discriminator_optimizer
    objective._feature_capture_shape = (1, 1)

    scheduler = SimpleNamespace(num_steps=0)
    scheduler.step = lambda increment: setattr(scheduler, "num_steps", scheduler.num_steps + increment)
    recipe = SimpleNamespace(
        model=student,
        optimizer=student_optimizer,
        lr_scheduler=[scheduler],
        device=torch.device("cpu"),
        compute_dtype=torch.float32,
        clip_grad_max_norm=100.0,
        grad_clip_foreach=False,
        check_loss=True,
    )
    micro_batch = {
        "image_latents": torch.zeros(1, 1, 1, 1),
        "text_embeddings": torch.zeros(1, 1, 1),
    }
    return objective, recipe, [micro_batch, micro_batch], modelopt_pipeline, scheduler


def test_load_negative_prompt_embedding_accepts_only_canonical_tensor(tmp_path):
    path = tmp_path / "negative.pt"
    embedding = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    torch.save(embedding, path)
    loaded = _load_negative_prompt_embedding(str(path))
    torch.testing.assert_close(loaded, embedding)
    assert loaded.is_contiguous()

    torch.save({"embedding": embedding}, path)
    with pytest.raises(ValueError, match="floating-point tensor|non-finite"):
        _load_negative_prompt_embedding(str(path))


def test_objective_is_tracked_as_distributed_checkpoint_state():
    objective = _objective()
    recipe = TrainDiffusionRecipe.__new__(TrainDiffusionRecipe)

    recipe.dmd2 = objective

    assert is_distributed_stateful(objective)
    assert "dmd2" in recipe.__dict__["__state_tracked"]


def test_objective_checkpoint_contains_all_trainable_auxiliary_state():
    objective, *_ = _training_fixture()
    wrappers = [Mock() for _ in range(4)]
    for index, wrapper in enumerate(wrappers):
        wrapper.state_dict.return_value = {"value": torch.tensor(index)}
    (
        objective._fake_score_state,
        objective._fake_score_optimizer_state,
        objective._discriminator_state,
        objective._discriminator_optimizer_state,
    ) = wrappers
    objective.modelopt_pipeline.ema = Mock()
    objective.modelopt_pipeline.ema.state_dict.return_value = {"shadow": torch.ones(1)}

    state = objective.state_dict()
    assert set(state) == {"fake_score", "fake_score_optimizer", "discriminator", "discriminator_optimizer", "ema"}
    objective.load_state_dict(state)
    for key, wrapper in zip(state, (*wrappers, objective.modelopt_pipeline.ema), strict=True):
        wrapper.load_state_dict.assert_called_once_with(state[key])


def test_discriminator_gradients_are_averaged_over_data_parallel_group():
    objective, recipe, *_ = _training_fixture()
    for parameter in objective.discriminator.parameters():
        parameter.grad = torch.full_like(parameter, 2.0)
    recipe._get_dp_group_size = Mock(return_value=2)
    recipe._get_dp_group = Mock(return_value=object())

    def _all_reduce(tensor, **_kwargs):
        tensor.add_(4.0)

    with (
        patch.object(dmd2_module.dist, "is_initialized", return_value=True),
        patch.object(dmd2_module.dist, "all_reduce", side_effect=_all_reduce) as all_reduce,
    ):
        objective._synchronize_discriminator_gradients(recipe)

    assert all_reduce.call_count == len(list(objective.discriminator.parameters()))
    for parameter in objective.discriminator.parameters():
        torch.testing.assert_close(parameter.grad, torch.full_like(parameter, 3.0))


def test_train_batch_group_alternates_modelopt_updates_and_active_optimizers():
    objective, recipe, batch_group, modelopt_pipeline, scheduler = _training_fixture()
    assert objective.primary_optimizer_steps(6) == 2

    with (
        patch.object(dmd2_module, "prepare_for_grad_accumulation"),
        patch.object(dmd2_module, "prepare_for_final_backward"),
        patch.object(dmd2_module, "prepare_after_first_microbatch"),
    ):
        objective.train_batch_group(recipe, batch_group, global_step=0)

        assert modelopt_pipeline.compute_student_loss.call_count == 2
        modelopt_pipeline.compute_fake_score_loss.assert_not_called()
        modelopt_pipeline.compute_discriminator_loss.assert_not_called()
        modelopt_pipeline.update_ema.assert_called_once_with(iteration=1)
        assert scheduler.num_steps == 1

        objective.train_batch_group(recipe, batch_group, global_step=1)

    assert modelopt_pipeline.compute_fake_score_loss.call_count == 2
    assert modelopt_pipeline.compute_discriminator_loss.call_count == 2
    assert modelopt_pipeline.update_ema.call_count == 1
    assert scheduler.num_steps == 1
    torch.testing.assert_close(recipe.model.weight, torch.tensor([[0.8]]))
    torch.testing.assert_close(objective.fake_score.weight, torch.tensor([[0.8]]))
    torch.testing.assert_close(objective.discriminator.weight, torch.tensor([[0.8]]))


def test_native_trainer_delegates_dmd2():
    recipe = TrainDiffusionRecipe.__new__(TrainDiffusionRecipe)
    expected = (1.0, 2.0)
    train_batch_group = Mock(return_value=expected)
    recipe.dmd2 = SimpleNamespace(train_batch_group=train_batch_group)
    batch_group = [{"image_latents": torch.zeros(1)}]

    result = recipe._train_batch_group(batch_group, global_step=7)

    assert result is expected
    train_batch_group.assert_called_once_with(recipe, batch_group, 7)


def test_qwen_image_dmd2_yaml_uses_the_native_trainer_contract():
    config = yaml.safe_load(_EXAMPLE_CONFIG.read_text(encoding="utf-8"))

    assert "recipe" not in config
    assert config["model"]["pretrained_model_name_or_path"] == "Qwen/Qwen-Image"

    dmd2 = config["dmd2"]
    modelopt_config = dmd2["modelopt_config"]
    assert "_target_" not in modelopt_config
    assert set(dmd2["discriminator"]) == {"feature_indices", "num_blocks", "inner_dim"}
    assert (
        modelopt_config["guidance_scale"],
        modelopt_config["student_sample_steps"],
        modelopt_config["student_update_freq"],
        modelopt_config["gan_loss_weight_gen"],
    ) == (4.0, 4, 5, 0.03)
    assert dmd2["negative_prompt_embedding_path"] == "PATH_TO_NEGATIVE_PROMPT_EMBEDDING"
    assert len(modelopt_config["sample_t_cfg"]["t_list"]) == 5
    assert "pipeline" not in dmd2
    assert "feature_capture_fn" not in dmd2
    assert "register_parallel_strategy_fn" not in dmd2

    dataloader = config["data"]["dataloader"]
    assert dataloader["_target_"].endswith("build_text_to_image_multiresolution_dataloader")
    assert "negative_prompt_embedding_path" not in dataloader


def test_qwen_image_strategy_checkpoints_complete_blocks(monkeypatch):
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl

    from nemo_automodel.components.distributed.parallelizer import (
        PARALLELIZATION_STRATEGIES,
        DefaultParallelizationStrategy,
    )
    from nemo_automodel.components.models.qwen_image import fsdp

    name = "QwenImageTransformer2DModel"
    monkeypatch.delitem(PARALLELIZATION_STRATEGIES, name, raising=False)
    wrapped = []
    monkeypatch.setattr(fsdp, "checkpoint_wrapper", lambda block, **kwargs: wrapped.append(kwargs) or block)
    monkeypatch.setattr(DefaultParallelizationStrategy, "parallelize", lambda self, model, *_args, **_kwargs: model)

    fsdp.register_qwen_image_parallel_strategy()
    model = SimpleNamespace(transformer_blocks=[object(), object()])
    assert PARALLELIZATION_STRATEGIES[name].parallelize(model, object(), activation_checkpointing=True) is model
    assert [call["checkpoint_impl"] for call in wrapped] == [CheckpointImpl.NO_REENTRANT] * 2
