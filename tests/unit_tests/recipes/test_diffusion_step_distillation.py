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

from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml

from nemo_automodel.components.checkpoint import CheckpointingConfig
from nemo_automodel.components.checkpoint.checkpointing import Checkpointer
from nemo_automodel.components.checkpoint.stateful_wrappers import ModelState, OptimizerState
from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.components.optim.scheduler import OptimizerParamScheduler
from nemo_automodel.components.training.ema import EMAManager
from nemo_automodel.components.training.step_scheduler import StepScheduler
from nemo_automodel.recipes.base_recipe import is_distributed_stateful
from nemo_automodel.recipes.diffusion import step_distillation as dmd2_module
from nemo_automodel.recipes.diffusion.step_distillation import _DMD2Objective, _load_negative_prompt_embedding
from nemo_automodel.recipes.diffusion.train import TrainDiffusionRecipe

_REPO_ROOT = Path(__file__).resolve().parents[3]
_EXAMPLE_CONFIG = _REPO_ROOT / "examples" / "diffusion" / "dmd2" / "qwen_image_dmd2.yaml"


class _ObjectiveConfig(dict):
    __getattr__ = dict.__getitem__

    def to_dict(self) -> dict:
        return dict(self)


def _objective(
    *,
    student_update_freq: int = 5,
    guidance_scale: float | None = None,
    gan_loss_weight_gen: float = 0.03,
    ema: SimpleNamespace | None = None,
    **cfg_values,
) -> _DMD2Objective:
    config = SimpleNamespace(
        student_update_freq=student_update_freq,
        guidance_scale=guidance_scale,
        gan_loss_weight_gen=gan_loss_weight_gen,
        ema=ema,
    )
    fastgen = SimpleNamespace(DMDConfig=Mock(return_value=config))
    with patch.object(dmd2_module, "_require_fastgen", return_value=fastgen):
        return _DMD2Objective(_ObjectiveConfig(**cfg_values))


def _configure_recipe(values: dict, optimizer: object | None = None) -> SimpleNamespace:
    cfg = Mock()
    cfg.get.side_effect = lambda key, default=None: values.get(key, default)
    cfg.optimizer = optimizer
    return SimpleNamespace(cfg=cfg)


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

    def _discriminator_loss(
        latents: torch.Tensor,
        noise: torch.Tensor,
        **_kwargs: object,
    ) -> dict[str, torch.Tensor]:
        """Return scalar discriminator losses.

        Args:
            latents: Tensor of shape ``[B,C,H,W]``.
            noise: Tensor matching ``latents`` in shape, dtype, and device.
            **_kwargs: Additional conditioning arguments.

        Returns:
            Mapping containing scalar ``total`` and ``gan`` loss tensors.
        """
        assert fake_score.weight.grad is not None
        return _loss(discriminator, "gan")

    dmd_pipeline = SimpleNamespace(
        compute_student_loss=Mock(side_effect=lambda *_args, **_kwargs: _loss(student, "vsd")),
        compute_fake_score_loss=Mock(side_effect=lambda *_args, **_kwargs: _loss(fake_score, "dsm")),
        compute_discriminator_loss=Mock(side_effect=_discriminator_loss),
        update_ema=Mock(),
        ema=None,
    )
    objective.dmd_pipeline = dmd_pipeline
    objective.teacher = torch.nn.Identity()
    objective.fake_score = fake_score
    objective.student_optimizer = student_optimizer
    objective.fake_score_optimizer = fake_score_optimizer
    objective.discriminator = discriminator
    objective.discriminator_optimizer = discriminator_optimizer
    objective._feature_capture_shape = (1, 1)

    scheduler = SimpleNamespace(num_steps=0)
    scheduler.step = lambda increment: setattr(scheduler, "num_steps", scheduler.num_steps + increment)
    recipe = SimpleNamespace(
        model=student,
        optimizer=[student_optimizer],
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
        "text_embeddings_mask": torch.ones(1, 1, dtype=torch.bool),
    }
    return objective, recipe, [micro_batch, micro_batch], dmd_pipeline, scheduler


class _QuadraticDMDLoss:
    def __init__(self, model: torch.nn.Linear, name: str) -> None:
        self.model = model
        self.name = name

    def __call__(
        self,
        latents: torch.Tensor,
        noise: torch.Tensor,
        **conditioning: object,
    ) -> dict[str, torch.Tensor]:
        """Return a scalar loss for one DMD2 microbatch.

        Args:
            latents: Tensor of shape ``[B,C,H,W]``.
            noise: Tensor matching ``latents`` in shape, dtype, and device.
            **conditioning: Text embeddings ``[B,S,D]`` and masks ``[B,S]``.

        Returns:
            Mapping containing scalar ``total`` and named loss tensors.
        """
        del latents, noise, conditioning
        total = self.model.weight.square().sum()
        return {"total": total, self.name: total}


class _UpdateEMA:
    def __init__(self, ema: EMAManager, student: torch.nn.Module) -> None:
        self.ema = ema
        self.student = student

    def __call__(self, *, iteration: int) -> None:
        del iteration
        self.ema.update(self.student)


def _checkpoint_recipe(checkpoint_dir: Path, base_weight: float) -> TrainDiffusionRecipe:
    recipe = TrainDiffusionRecipe(
        ConfigNode(
            {
                "model": {"pretrained_model_name_or_path": "dmd2-checkpoint-test"},
                "dmd2": {"student_update_freq": 2},
            }
        )
    )
    recipe.checkpointer = Checkpointer(
        CheckpointingConfig(
            checkpoint_dir=checkpoint_dir,
            model_save_format="torch_save",
            save_consolidated=False,
        ),
        dp_rank=0,
        tp_rank=0,
        pp_rank=0,
    )
    recipe.device = torch.device("cpu")
    recipe.compute_dtype = torch.float32
    recipe.clip_grad_max_norm = 100.0
    recipe.grad_clip_foreach = False
    recipe.check_loss = True
    recipe.cpu_offload = False
    recipe.peft_config = None

    student, fake_score, discriminator = (_linear() for _ in range(3))
    with torch.no_grad():
        for model, weight in zip(
            (student, fake_score, discriminator),
            (base_weight, base_weight + 1.0, base_weight + 2.0),
            strict=True,
        ):
            model.weight.fill_(weight)
    student_optimizer = torch.optim.SGD(student.parameters(), lr=0.1, momentum=0.9)
    fake_score_optimizer = torch.optim.SGD(fake_score.parameters(), lr=0.1, momentum=0.9)
    discriminator_optimizer = torch.optim.SGD(discriminator.parameters(), lr=0.1, momentum=0.9)
    recipe.model = student
    recipe.optimizer = [student_optimizer]
    recipe.lr_scheduler = [
        OptimizerParamScheduler(
            student_optimizer,
            init_lr=0.1,
            max_lr=0.1,
            min_lr=0.1,
            lr_warmup_steps=0,
            lr_decay_steps=4,
            lr_decay_style="constant",
            start_wd=0.0,
            end_wd=0.0,
            wd_incr_steps=4,
            wd_incr_style="constant",
        )
    ]
    micro_batch = {
        "image_latents": torch.zeros(1, 1, 1, 1),
        "text_embeddings": torch.zeros(1, 1, 1),
        "text_embeddings_mask": torch.ones(1, 1, dtype=torch.bool),
    }
    recipe.step_scheduler = StepScheduler(
        global_batch_size=1,
        local_batch_size=1,
        dp_size=1,
        dataloader=[micro_batch] * 4,
        ckpt_every_steps=2,
        save_checkpoint_every_epoch=False,
        loss_average_window_steps=1,
        num_epochs=1,
        max_steps=4,
        preemption_signal=None,
    )

    ema = EMAManager(student, decay=0.5)
    objective = _DMD2Objective.__new__(_DMD2Objective)
    objective.cfg = recipe.cfg.dmd2
    objective.dmd_config = SimpleNamespace(student_update_freq=2)
    objective.dmd_pipeline = SimpleNamespace(
        compute_student_loss=_QuadraticDMDLoss(student, "vsd"),
        compute_fake_score_loss=_QuadraticDMDLoss(fake_score, "dsm"),
        compute_discriminator_loss=_QuadraticDMDLoss(discriminator, "gan"),
        update_ema=_UpdateEMA(ema, student),
        ema=ema,
    )
    objective.teacher = torch.nn.Identity()
    objective.fake_score = fake_score
    objective.student_optimizer = student_optimizer
    objective.fake_score_optimizer = fake_score_optimizer
    objective.discriminator = discriminator
    objective.discriminator_optimizer = discriminator_optimizer
    objective.negative_prompt_embedding = None
    objective._feature_capture_shape = (1, 1)
    objective._fake_score_state = ModelState(fake_score)
    objective._fake_score_optimizer_state = OptimizerState(fake_score, fake_score_optimizer)
    objective._discriminator_state = ModelState(discriminator)
    objective._discriminator_optimizer_state = OptimizerState(
        discriminator,
        discriminator_optimizer,
    )
    recipe._dmd2 = objective
    return recipe


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


def test_prepare_batch_preserves_positive_lengths_and_builds_negative_mask():
    objective = _objective()
    objective.negative_prompt_embedding = torch.zeros(3, 4)
    recipe = SimpleNamespace(device=torch.device("cpu"), compute_dtype=torch.float32)
    batch = {
        "image_latents": torch.zeros(2, 1, 2, 2),
        "text_embeddings": torch.zeros(2, 5, 4),
        "text_embeddings_mask": torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]),
    }

    _, _, text_embeddings, text_mask, negative_embeddings, negative_mask = objective._prepare_batch(recipe, batch)

    assert text_embeddings.shape == (2, 5, 4)
    torch.testing.assert_close(text_mask, batch["text_embeddings_mask"])
    assert negative_embeddings.shape == (2, 3, 4)
    torch.testing.assert_close(negative_mask, torch.ones(2, 3, dtype=torch.long))


def test_configure_rejects_flash_attention_with_required_text_mask():
    objective = _objective()
    values = {
        "model.mode": "finetune",
        "fsdp": {},
        "model.attention_backend": "flash",
    }
    cfg = Mock()
    cfg.get.side_effect = lambda key, default=None: values.get(key, default)
    cfg.optimizer = object()

    with pytest.raises(ValueError, match="flash-attn 2"):
        objective.configure(SimpleNamespace(cfg=cfg))


def test_objective_is_tracked_as_distributed_checkpoint_state():
    objective = _objective()
    recipe = TrainDiffusionRecipe.__new__(TrainDiffusionRecipe)

    recipe._dmd2 = objective

    assert is_distributed_stateful(objective)
    assert "_dmd2" in recipe.__dict__["__state_tracked"]


def test_dmd2_production_checkpoint_resumes_all_training_state(tmp_path):
    source = _checkpoint_recipe(tmp_path, base_weight=1.0)
    restored = _checkpoint_recipe(tmp_path, base_weight=-4.0)
    try:
        source_steps = iter(source.step_scheduler)
        source._train_batch_group(next(source_steps), global_step=-1)
        source._train_batch_group(next(source_steps), global_step=-1)

        source_models = (source.model, source._dmd2.fake_score, source._dmd2.discriminator)
        restored_models = (restored.model, restored._dmd2.fake_score, restored._dmd2.discriminator)
        source_optimizers = (
            source.optimizer[0],
            source._dmd2.fake_score_optimizer,
            source._dmd2.discriminator_optimizer,
        )
        restored_optimizers = (
            restored.optimizer[0],
            restored._dmd2.fake_score_optimizer,
            restored._dmd2.discriminator_optimizer,
        )
        assert all(optimizer.state for optimizer in source_optimizers)
        assert not any(optimizer.state for optimizer in restored_optimizers)
        assert source.lr_scheduler[0].num_steps == 1
        assert source.step_scheduler.state_dict() == {"step": 2, "epoch": 0}
        source.save_checkpoint(epoch=0, step=2, train_loss=0.0)

        restored.load_checkpoint("LATEST")
        restored._dmd2.after_restore(restored)
        assert (restored.step_scheduler.step, restored.step_scheduler.epoch) == (2, 0)
        assert restored.lr_scheduler[0].num_steps == 1
        for actual, expected in zip(restored_models, source_models, strict=True):
            torch.testing.assert_close(actual.state_dict(), expected.state_dict())
        for actual, expected in zip(restored_optimizers, source_optimizers, strict=True):
            torch.testing.assert_close(actual.state_dict(), expected.state_dict())
        torch.testing.assert_close(
            restored._dmd2.dmd_pipeline.ema.state_dict(),
            source._dmd2.dmd_pipeline.ema.state_dict(),
        )
        assert restored.lr_scheduler[0].state_dict() == source.lr_scheduler[0].state_dict()

        student_before = source.model.weight.detach().clone()
        fake_score_before = source._dmd2.fake_score.weight.detach().clone()
        discriminator_before = source._dmd2.discriminator.weight.detach().clone()
        ema_before = source._dmd2.dmd_pipeline.ema.state_dict()
        source_batch = next(source_steps)
        restored_steps = iter(restored.step_scheduler)
        restored_batch = next(restored_steps)
        assert source.step_scheduler.step == restored.step_scheduler.step == 2
        source._train_batch_group(source_batch, global_step=-1)
        restored._train_batch_group(restored_batch, global_step=-1)

        assert not torch.equal(source.model.weight, student_before)
        torch.testing.assert_close(source._dmd2.fake_score.weight, fake_score_before)
        torch.testing.assert_close(source._dmd2.discriminator.weight, discriminator_before)
        ema_after = source._dmd2.dmd_pipeline.ema.state_dict()
        assert any(not torch.equal(ema_after[name], value) for name, value in ema_before.items())
        assert source.lr_scheduler[0].num_steps == restored.lr_scheduler[0].num_steps == 2
        for actual, expected in zip(restored_models, source_models, strict=True):
            torch.testing.assert_close(actual.state_dict(), expected.state_dict())
        for actual, expected in zip(restored_optimizers, source_optimizers, strict=True):
            torch.testing.assert_close(actual.state_dict(), expected.state_dict())
        torch.testing.assert_close(
            restored._dmd2.dmd_pipeline.ema.state_dict(),
            source._dmd2.dmd_pipeline.ema.state_dict(),
        )
    finally:
        source.checkpointer.close()
        restored.checkpointer.close()


def test_set_phase_keeps_sharded_transformers_trainable():
    """The student and fake score must stay ``requires_grad=True`` in both phases.

    FSDP2 caches a parameter group's ``_orig_dtype``/``_reduce_dtype`` from its *trainable*
    parameters at the module's first forward. Step 0 is a student phase and ModelOpt
    evaluates the fake score inside ``compute_student_loss``, so freezing either sharded
    transformer here would poison that cache for the rest of the run — silently reducing
    gradients in the parameter dtype instead of ``fsdp.reduce_dtype``, and raising
    "attempting to assign a gradient with dtype 'c10::BFloat16'" under fp32 master weights.
    Only the replicated discriminator, which has no such cached state, is toggled.
    """
    objective = _objective()
    objective.fake_score = _linear()
    objective.discriminator = _linear()
    recipe = SimpleNamespace(model=_linear())

    for student_phase in (True, False, True):
        objective._set_phase(recipe, student_phase)

        assert all(parameter.requires_grad for parameter in recipe.model.parameters())
        assert all(parameter.requires_grad for parameter in objective.fake_score.parameters())
        assert recipe.model.training is student_phase
        assert objective.fake_score.training is not student_phase

        assert objective.discriminator.training is not student_phase
        assert all(parameter.requires_grad is not student_phase for parameter in objective.discriminator.parameters())


def test_checkpoint_taken_during_student_phase_resumes(tmp_path):
    """A checkpoint saved before the first fake-score phase must stay resumable.

    ``_set_phase`` leaves the discriminator frozen after a student phase, and DCP's
    ``_init_optim_state`` only materializes optimizer state for trainable parameters. Without
    :meth:`_DMD2Objective._discriminator_trainable` the saved discriminator optimizer carries
    ``param_groups`` and no ``state``, and the resume-side load fails with
    "Missing key in checkpoint state_dict: discriminator_optimizer.optim.state...".
    """
    source = _checkpoint_recipe(tmp_path, base_weight=1.0)
    restored = _checkpoint_recipe(tmp_path, base_weight=-4.0)
    try:
        # One outer step only: with student_update_freq=2 that is the student phase, so the
        # discriminator optimizer has never stepped and the discriminator is frozen.
        source._train_batch_group(next(iter(source.step_scheduler)), global_step=-1)
        assert not any(parameter.requires_grad for parameter in source._dmd2.discriminator.parameters())
        assert not source._dmd2.discriminator_optimizer.state
        source.save_checkpoint(epoch=0, step=1, train_loss=0.0)

        restored.load_checkpoint("LATEST")
        restored._dmd2.after_restore(restored)

        assert restored.step_scheduler.step == 1
        for name in ("fake_score", "discriminator"):
            torch.testing.assert_close(
                getattr(restored._dmd2, name).state_dict(),
                getattr(source._dmd2, name).state_dict(),
            )
        # The checkpoint hook restores the phase toggle rather than leaving it enabled.
        assert not any(parameter.requires_grad for parameter in source._dmd2.discriminator.parameters())
    finally:
        source.checkpointer.close()
        restored.checkpointer.close()


def _run_discriminator_sync(rank: int, world_size: int, init_file: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        discriminator = _linear()
        objective = _DMD2Objective.__new__(_DMD2Objective)
        objective.discriminator = discriminator
        discriminator.weight.grad = torch.full_like(discriminator.weight, rank + 1.0)
        recipe = SimpleNamespace(
            _get_dp_group_size=lambda: world_size,
            _get_dp_group=lambda: dist.group.WORLD,
        )

        objective._synchronize_discriminator_gradients(recipe)
        reference = _linear()
        reference.weight.grad = torch.full_like(reference.weight, 1.5)
        torch.optim.SGD(discriminator.parameters(), lr=0.1).step()
        torch.optim.SGD(reference.parameters(), lr=0.1).step()

        torch.testing.assert_close(discriminator.weight.grad, reference.weight.grad)
        torch.testing.assert_close(discriminator.weight, reference.weight)
    finally:
        dist.destroy_process_group()


def test_discriminator_gradients_match_single_process_reference(tmp_path):
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("Gloo is unavailable")
    mp.spawn(_run_discriminator_sync, args=(2, str(tmp_path / "gloo_init")), nprocs=2, join=True)


def test_train_batch_group_alternates_modelopt_updates_and_active_optimizers():
    objective, recipe, batch_group, dmd_pipeline, scheduler = _training_fixture()
    assert objective.primary_optimizer_steps(6) == 2

    with (
        patch.object(dmd2_module, "prepare_for_grad_accumulation"),
        patch.object(dmd2_module, "prepare_for_final_backward"),
        patch.object(dmd2_module, "prepare_after_first_microbatch"),
    ):
        objective.train_batch_group(recipe, batch_group, global_step=0)

        assert dmd_pipeline.compute_student_loss.call_count == 2
        dmd_pipeline.compute_fake_score_loss.assert_not_called()
        dmd_pipeline.compute_discriminator_loss.assert_not_called()
        dmd_pipeline.update_ema.assert_called_once_with(iteration=1)
        assert scheduler.num_steps == 1

        objective.train_batch_group(recipe, batch_group, global_step=1)

    assert dmd_pipeline.compute_fake_score_loss.call_count == 2
    assert dmd_pipeline.compute_discriminator_loss.call_count == 2
    assert dmd_pipeline.update_ema.call_count == 1
    assert scheduler.num_steps == 1
    torch.testing.assert_close(recipe.model.weight, torch.tensor([[0.8]]))
    torch.testing.assert_close(objective.fake_score.weight, torch.tensor([[0.8]]))
    torch.testing.assert_close(objective.discriminator.weight, torch.tensor([[0.8]]))


def test_student_scheduler_uses_student_update_budget():
    objective = _objective(student_update_freq=5)
    lr_scheduler_config = Mock()
    expected = [object()]
    lr_scheduler_config.build.return_value = expected
    optimizer = [Mock()]
    step_scheduler = SimpleNamespace(epoch_len=10, num_epochs=3, max_steps=22)
    recipe = SimpleNamespace(
        cfg=SimpleNamespace(lr_scheduler=lr_scheduler_config),
        optimizer=optimizer,
        step_scheduler=step_scheduler,
    )

    result = objective.build_lr_scheduler(recipe)

    assert result is expected
    lr_scheduler_config.build.assert_called_once_with(optimizer, step_scheduler, total_steps=5)


def test_native_trainer_delegates_dmd2():
    recipe = TrainDiffusionRecipe.__new__(TrainDiffusionRecipe)
    expected = (1.0, 2.0)
    train_batch_group = Mock(return_value=expected)
    recipe._dmd2 = SimpleNamespace(train_batch_group=train_batch_group)
    recipe.step_scheduler = SimpleNamespace(step=7)
    batch_group = [{"image_latents": torch.zeros(1)}]

    result = recipe._train_batch_group(batch_group, global_step=99)

    assert result is expected
    train_batch_group.assert_called_once_with(recipe, batch_group, global_step=7)


def test_qwen_image_dmd2_yaml_uses_the_native_trainer_contract():
    config = yaml.safe_load(_EXAMPLE_CONFIG.read_text(encoding="utf-8"))

    assert "recipe" not in config
    assert config["model"]["pretrained_model_name_or_path"] == "Qwen/Qwen-Image"
    assert "attention_backend" not in config["model"]

    dmd2 = config["dmd2"]
    assert "_target_" not in dmd2
    assert (
        dmd2["guidance_scale"],
        dmd2["student_sample_steps"],
        dmd2["student_update_freq"],
        dmd2["gan_loss_weight_gen"],
    ) == (4.0, 4, 5, 0.03)
    assert dmd2["negative_prompt_embedding_path"] == "PATH_TO_NEGATIVE_PROMPT_EMBEDDING"
    assert len(dmd2["sample_t_cfg"]["t_list"]) == 5
    assert dmd2["pipeline"]["_target_"].endswith("QwenImageDMDPipeline")
    assert dmd2["discriminator"]["_target_"].endswith("Discriminator_ImageDiT")
    assert dmd2["feature_capture"]["_target_"].endswith("attach_feature_capture")
    assert config["fsdp"]["activation_checkpointing"] == "selective"
    assert "optim" not in config
    assert config["optimizer"]["_target_"] == "torch.optim.AdamW"
    assert config["optimizer"]["lr"] == 2.0e-6
    assert config["step_scheduler"]["log_remote_every_steps"] == 1

    dataloader = config["data"]["dataloader"]
    assert dataloader["_target_"].endswith("build_text_to_image_multiresolution_dataloader")
    assert "negative_prompt_embedding_path" not in dataloader


def test_require_fastgen_returns_module_or_points_to_dmd2_extra():
    fastgen = SimpleNamespace()
    with patch.object(dmd2_module, "safe_import", return_value=(True, fastgen)):
        assert dmd2_module._require_fastgen() is fastgen

    with patch.object(dmd2_module, "safe_import", return_value=(False, None)):
        with pytest.raises(ImportError, match="uv sync --extra dmd2"):
            dmd2_module._require_fastgen()


def test_load_negative_prompt_embedding_rejects_missing_and_non_finite_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        _load_negative_prompt_embedding(str(tmp_path / "absent.pt"))

    path = tmp_path / "negative.pt"
    torch.save(torch.full((2, 3), torch.inf), path)
    with pytest.raises(ValueError, match="non-finite"):
        _load_negative_prompt_embedding(str(path))


@pytest.mark.parametrize("student_update_freq", [0, 1])
def test_init_rejects_student_update_freq_below_two(student_update_freq):
    # freq=1 makes `step % freq == 0` true on every step, so the fake score and
    # discriminator would never be trained while still feeding the student loss.
    with pytest.raises(ValueError, match="student_update_freq"):
        _objective(student_update_freq=student_update_freq)


@pytest.mark.parametrize(
    ("values", "match"),
    [
        ({"model.mode": "pretrain"}, "model.mode=finetune"),
        ({"peft": {}}, "full-parameter"),
        ({"ddp": {}, "fsdp": {}}, "does not support DDP"),
        ({}, "does not support DDP"),
        ({"fsdp": {"tp_size": 2, "cp_size": 2}}, "tp_size=2, cp_size=2"),
        ({"fsdp": {}, "data.dataloader.train_text_encoder": True}, "precomputed text embeddings"),
        ({"fsdp": {}, "model.stage": "high_noise"}, "model.stage"),
        ({"fsdp": {}, "model.transformer_engine_fp8": True}, "Transformer Engine FP8"),
    ],
)
def test_configure_rejects_unsupported_topologies(values, match):
    with pytest.raises(ValueError, match=match):
        _objective().configure(_configure_recipe({"model.mode": "finetune", **values}, optimizer=object()))


def test_configure_requires_native_optimizer_and_resumable_ema():
    recipe = _configure_recipe({"model.mode": "finetune", "fsdp": {}}, optimizer=None)
    with pytest.raises(ValueError, match="top-level `optimizer`"):
        _objective().configure(recipe)

    recipe.cfg.optimizer = object()
    with pytest.raises(ValueError, match="fsdp2=true and mode=full_tensor"):
        _objective(ema=SimpleNamespace(fsdp2=False, mode="full_tensor")).configure(recipe)

    _objective(ema=SimpleNamespace(fsdp2=True, mode="full_tensor")).configure(recipe)


def test_build_lr_scheduler_handles_streaming_and_missing_budgets():
    objective = _objective(student_update_freq=5)
    assert objective.build_lr_scheduler(SimpleNamespace(cfg=SimpleNamespace(lr_scheduler=None))) is None

    lr_scheduler_config = Mock()
    recipe = SimpleNamespace(
        cfg=SimpleNamespace(lr_scheduler=lr_scheduler_config),
        optimizer=[Mock()],
        step_scheduler=SimpleNamespace(epoch_len=None, num_epochs=1, max_steps=7),
    )
    objective.build_lr_scheduler(recipe)
    lr_scheduler_config.build.assert_called_once_with(recipe.optimizer, recipe.step_scheduler, total_steps=2)

    recipe.step_scheduler.max_steps = None
    with pytest.raises(ValueError, match="positive outer-step budget"):
        objective.build_lr_scheduler(recipe)


def _setup_recipe(cfg_optimizer: Mock) -> SimpleNamespace:
    return SimpleNamespace(
        model=_linear(),
        optimizer=[torch.optim.SGD(_linear().parameters(), lr=0.1)],
        device=torch.device("cpu"),
        compute_dtype=torch.float32,
        model_dtype=torch.float32,
        cpu_offload=False,
        model_id="Qwen/Qwen-Image",
        attention_backend=None,
        transformer_engine_linear=False,
        transformer_engine_fp8=False,
        fuse_qkv_projections=False,
        compact_fused_qkv_projections=False,
        active_transformer=None,
        device_mesh=None,
        cfg=SimpleNamespace(optimizer=cfg_optimizer, get=lambda key, default=None: {"fsdp": {}}.get(key, default)),
    )


def _cfg_optimizer() -> Mock:
    cfg_optimizer = Mock()
    cfg_optimizer.build.side_effect = lambda model, device_mesh=None: [torch.optim.SGD(model.parameters(), lr=0.1)]
    return cfg_optimizer


def test_setup_builds_auxiliaries_discriminator_and_checkpoint_state(tmp_path):
    negative_path = tmp_path / "negative.pt"
    torch.save(torch.zeros(3, 4), negative_path)
    discriminator = _linear()
    discriminator_cfg = Mock()
    discriminator_cfg.instantiate.return_value = discriminator
    pipeline_cfg = Mock()
    pipeline_cfg.instantiate.return_value = SimpleNamespace(ema=None)
    objective = _objective(
        guidance_scale=4.0,
        negative_prompt_embedding_path=str(negative_path),
        discriminator=discriminator_cfg,
        feature_capture=Mock(),
        pipeline=pipeline_cfg,
    )
    recipe = _setup_recipe(_cfg_optimizer())

    def _build_pipeline(**kwargs):
        assert kwargs["model_id"] == "Qwen/Qwen-Image"
        return SimpleNamespace(transformer=_linear()), None

    with (
        patch.object(dmd2_module, "_require_fastgen", return_value=SimpleNamespace()),
        patch.object(dmd2_module, "build_diffusion_pipeline", side_effect=_build_pipeline) as build_pipeline,
    ):
        objective.setup(recipe)

    assert build_pipeline.call_count == 2
    assert objective.teacher is not None and not objective.teacher.training
    assert not any(parameter.requires_grad for parameter in objective.teacher.parameters())
    assert objective.fake_score is not None and objective.fake_score.training
    assert all(parameter.requires_grad for parameter in objective.fake_score.parameters())
    torch.testing.assert_close(objective.negative_prompt_embedding, torch.zeros(3, 4))

    discriminator_cfg.instantiate.assert_called_once_with()
    assert objective.discriminator is discriminator
    assert objective.student_optimizer is recipe.optimizer[0]
    assert recipe.cfg.optimizer.build.call_count == 2  # fake score + discriminator

    pipeline_cfg.instantiate.assert_called_once_with(
        student=recipe.model,
        teacher=objective.teacher,
        fake_score=objective.fake_score,
        config=objective.dmd_config,
        discriminator=discriminator,
    )
    assert objective._fake_score_state is not None
    assert objective._fake_score_optimizer_state is not None
    assert objective._discriminator_state is not None
    assert objective._discriminator_optimizer_state is not None


def test_setup_requires_negative_embedding_path_when_cfg_is_enabled():
    objective = _objective(guidance_scale=4.0)
    with patch.object(dmd2_module, "_require_fastgen", return_value=SimpleNamespace()):
        with pytest.raises(ValueError, match="negative_prompt_embedding_path"):
            objective.setup(SimpleNamespace())


def test_setup_requires_discriminator_arguments_when_gan_is_enabled():
    objective = _objective(gan_loss_weight_gen=0.03)
    recipe = _setup_recipe(_cfg_optimizer())
    with (
        patch.object(dmd2_module, "_require_fastgen", return_value=SimpleNamespace()),
        patch.object(
            dmd2_module,
            "build_diffusion_pipeline",
            side_effect=lambda **_kwargs: (SimpleNamespace(transformer=_linear()), None),
        ),
    ):
        with pytest.raises(ValueError, match="dmd2.discriminator"):
            objective.setup(recipe)


def test_after_restore_validates_student_scheduler_phase_boundary():
    objective = _objective(student_update_freq=5)
    objective.after_restore(SimpleNamespace(lr_scheduler=None))

    recipe = SimpleNamespace(
        lr_scheduler=[SimpleNamespace(num_steps=2)],
        step_scheduler=SimpleNamespace(step=7),
    )
    objective.after_restore(recipe)

    recipe.lr_scheduler[0].num_steps = 1
    with pytest.raises(ValueError, match="inconsistent student LR scheduler"):
        objective.after_restore(recipe)


def test_train_batch_group_rejects_empty_group_and_requires_setup():
    objective, recipe, *_ = _training_fixture()
    with pytest.raises(RuntimeError, match="empty gradient-accumulation group"):
        objective.train_batch_group(recipe, [], global_step=0)

    with pytest.raises(RuntimeError, match="has not been set up"):
        _objective().state_dict()


@pytest.mark.parametrize("student_phase", [True, False])
def test_train_batch_group_raises_on_non_finite_losses(student_phase):
    objective, recipe, batch_group, dmd_pipeline, _ = _training_fixture()
    nan_loss = {"total": torch.tensor(torch.nan, requires_grad=True)}
    if student_phase:
        dmd_pipeline.compute_student_loss.side_effect = lambda *_args, **_kwargs: nan_loss
        match = "student"
    else:
        dmd_pipeline.compute_discriminator_loss.side_effect = lambda *_args, **_kwargs: nan_loss
        match = "discriminator"

    with (
        patch.object(dmd2_module, "prepare_for_grad_accumulation"),
        patch.object(dmd2_module, "prepare_for_final_backward"),
        patch.object(dmd2_module, "prepare_after_first_microbatch"),
    ):
        with pytest.raises(FloatingPointError, match=match):
            objective.train_batch_group(recipe, batch_group, global_step=0 if student_phase else 1)


def test_only_optimizer_unwraps_single_instances_and_rejects_groups():
    optimizer = torch.optim.SGD(_linear().parameters(), lr=0.1)
    assert _DMD2Objective._only_optimizer(optimizer, "student") is optimizer
    assert _DMD2Objective._only_optimizer([optimizer], "student") is optimizer
    with pytest.raises(ValueError, match="one student optimizer, got 2"):
        _DMD2Objective._only_optimizer([optimizer, optimizer], "student")


def test_prepare_batch_promotes_unbatched_embeddings_and_masks():
    objective = _objective()
    recipe = SimpleNamespace(device=torch.device("cpu"), compute_dtype=torch.float32)
    batch = {
        "image_latents": torch.zeros(1, 1, 2, 2),
        "text_embeddings": torch.zeros(5, 4),
        "text_embeddings_mask": torch.ones(5, dtype=torch.long),
    }
    _, noise, text_embeddings, text_mask, negative, negative_mask = objective._prepare_batch(recipe, batch)
    assert text_embeddings.shape == (1, 5, 4)
    assert text_mask.shape == (1, 5)
    assert noise.shape == (1, 1, 2, 2)
    assert negative is None and negative_mask is None


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"text_embeddings_mask": None}, "requires text_embeddings_mask"),
        ({"image_latents": torch.zeros(1, 2, 2)}, r"image_latents \[B,C,H,W\]"),
        ({"text_embeddings_mask": torch.ones(1, 3, dtype=torch.long)}, "matching text_embeddings"),
    ],
)
def test_prepare_batch_rejects_malformed_batches(overrides, match):
    objective = _objective()
    recipe = SimpleNamespace(device=torch.device("cpu"), compute_dtype=torch.float32)
    batch = {
        "image_latents": torch.zeros(1, 1, 2, 2),
        "text_embeddings": torch.zeros(1, 5, 4),
        "text_embeddings_mask": torch.ones(1, 5, dtype=torch.long),
    }
    batch.update(overrides)
    batch = {key: value for key, value in batch.items() if value is not None}
    with pytest.raises(ValueError, match=match):
        objective._prepare_batch(recipe, batch)


def test_prepare_batch_rejects_mismatched_negative_embedding_width():
    objective = _objective()
    objective.negative_prompt_embedding = torch.zeros(3, 8)
    recipe = SimpleNamespace(device=torch.device("cpu"), compute_dtype=torch.float32)
    batch = {
        "image_latents": torch.zeros(1, 1, 2, 2),
        "text_embeddings": torch.zeros(1, 5, 4),
        "text_embeddings_mask": torch.ones(1, 5, dtype=torch.long),
    }
    with pytest.raises(ValueError, match="dimensions must match"):
        objective._prepare_batch(recipe, batch)


def test_prepare_batch_attaches_feature_capture_once_per_latent_shape():
    objective = _objective()
    objective.teacher = object()
    objective.discriminator = SimpleNamespace(feature_indices={3, 1})
    feature_capture = Mock()
    objective.cfg.feature_capture = feature_capture
    recipe = SimpleNamespace(device=torch.device("cpu"), compute_dtype=torch.float32)
    batch = {
        "image_latents": torch.zeros(1, 1, 2, 4),
        "text_embeddings": torch.zeros(1, 5, 4),
        "text_embeddings_mask": torch.ones(1, 5, dtype=torch.long),
    }

    objective._prepare_batch(recipe, batch)
    objective._prepare_batch(recipe, batch)

    feature_capture.instantiate.assert_called_once_with(
        teacher=objective.teacher,
        feature_indices=[1, 3],
        h_lat=2,
        w_lat=4,
    )
    assert objective._feature_capture_shape == (2, 4)


def test_discriminator_broadcast_replicates_rank_zero_tensors():
    objective, recipe, *_ = _training_fixture()
    recipe._get_dp_group_size = Mock(return_value=2)
    recipe._get_dp_group = Mock(return_value=object())

    with (
        patch.object(dmd2_module.dist, "is_initialized", return_value=True),
        patch.object(dmd2_module.dist, "broadcast") as broadcast,
    ):
        objective._broadcast_discriminator(recipe)

    tensors = [*objective.discriminator.parameters(), *objective.discriminator.buffers()]
    assert broadcast.call_count == len(tensors)
    for call in broadcast.call_args_list:
        assert call.kwargs["src"] == 0


class _StopSetup(Exception):
    pass


def test_native_trainer_setup_constructs_and_configures_dmd2_objective():
    from nemo_automodel.components.config.loader import ConfigNode
    from nemo_automodel.recipes.diffusion import train as train_module

    recipe = TrainDiffusionRecipe(
        ConfigNode(
            {
                "model": {"pretrained_model_name_or_path": "Qwen/Qwen-Image"},
                "fsdp": {},
                "dmd2": {"student_update_freq": 5},
            }
        )
    )
    objective = Mock()
    objective.configure.side_effect = _StopSetup
    objective_cls = Mock(return_value=objective)

    with (
        patch.object(train_module, "initialize_distributed", return_value=SimpleNamespace(is_main=False)),
        patch.object(dmd2_module, "_DMD2Objective", objective_cls),
        pytest.raises(_StopSetup),
    ):
        recipe.setup()

    assert objective_cls.call_count == 1
    objective.configure.assert_called_once_with(recipe)
    assert recipe._dmd2 is objective
