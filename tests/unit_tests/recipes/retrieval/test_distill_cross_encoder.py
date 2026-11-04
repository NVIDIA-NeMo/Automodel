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
"""Unit tests for :class:`DistillCrossEncoderRecipe`.

CPU-only. The distillation-specific methods are exercised directly rather than through
``setup()``, so no real model, device mesh, or process group is needed.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
import yaml

from nemo_automodel.components.distributed.mesh import MeshContext
from nemo_automodel.components.distributed.config import DDPConfig, DistributedSetup, FSDP2Config
from nemo_automodel.components.loss.cross_encoder_distill import GroupedScoreKLLoss
from nemo_automodel.recipes.retrieval.distill_cross_encoder import DistillCrossEncoderRecipe
from nemo_automodel.recipes.retrieval.train_cross_encoder import TrainCrossEncoderRecipe

REPO_ROOT = Path(__file__).resolve().parents[4]
CONFIG_PATH = REPO_ROOT / "examples" / "retrieval" / "distillation" / "ministral3_vl_ranker_distill.yaml"

_HIDDEN = 6
_N_PASSAGES = 4
_N_QUERIES = 3
_STUDENT_TEMP = 0.02
_KD_TEMP = 0.1
_VOCAB_SIZE = 131072
# Deliberately not 1.0: a unit value would neutralize the recipe-level division and hide any
# divergence between the distillation CE and TrainCrossEncoderRecipe.
_RECIPE_TEMP = 0.5


class _StubCrossEncoder(nn.Module):
    """Emits ``[num_pairs, 1]`` logits divided by ``config.temperature``, like the real model."""

    def __init__(self, scale: float, temperature: float = _STUDENT_TEMP, vocab_size: int = _VOCAB_SIZE):
        super().__init__()
        self.score = nn.Linear(_HIDDEN, 1, bias=False)
        with torch.no_grad():
            self.score.weight.fill_(scale)
        self.config = SimpleNamespace(temperature=temperature, vocab_size=vocab_size)
        self.seen_batches: list[dict] = []

    def forward(self, return_dict: bool = False, **batch):
        self.seen_batches.append(batch)
        return SimpleNamespace(logits=self.score(batch["features"]) / self.config.temperature)


class _RecordingKDLoss(nn.Module):
    """Wraps the real loss and records the score tensors it was handed."""

    def __init__(self, temperature: float = _KD_TEMP, reference_temperature: float | None = _STUDENT_TEMP):
        super().__init__()
        self.inner = GroupedScoreKLLoss(temperature=temperature, reference_temperature=reference_temperature)
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def forward(self, student_scores: torch.Tensor, teacher_scores: torch.Tensor) -> torch.Tensor:
        self.calls.append((student_scores.detach().clone(), teacher_scores.detach().clone()))
        return self.inner(student_scores, teacher_scores)


def _make_batch(seed: int = 0) -> dict:
    generator = torch.Generator().manual_seed(seed)
    return {
        "features": torch.randn(_N_QUERIES * _N_PASSAGES, _HIDDEN, generator=generator),
        "pixel_values": torch.randn(_N_QUERIES, 3, generator=generator),
        "labels": torch.zeros(_N_QUERIES, dtype=torch.long),
    }


def _make_recipe(
    *,
    ce_weight: float = 0.5,
    kd_weight: float = 0.5,
    with_teacher: bool = True,
    kd_loss: nn.Module | None = None,
    teacher_temperature: float = _STUDENT_TEMP,
) -> DistillCrossEncoderRecipe:
    recipe = object.__new__(DistillCrossEncoderRecipe)
    recipe.dist_env = SimpleNamespace(device=torch.device("cpu"))
    recipe.distributed_config = SimpleNamespace(autocast_dtype=None, defer_fsdp_grad_sync=False)
    recipe.model_parts = [_StubCrossEncoder(scale=0.5)]
    recipe.teacher_model = _StubCrossEncoder(scale=-0.8, temperature=teacher_temperature) if with_teacher else None
    recipe.train_n_passages = _N_PASSAGES
    recipe.temperature = _RECIPE_TEMP
    recipe.student_internal_temperature = _STUDENT_TEMP
    recipe.teacher_internal_temperature = teacher_temperature
    recipe.loss_weights = {"ce": ce_weight, "kd": kd_weight}
    recipe.kd_loss = kd_loss if kd_loss is not None else _RecordingKDLoss()
    recipe._acc_buffer = []
    recipe._loss_component_buffer = []
    return recipe


def _run_step(recipe, batch: dict, is_train: bool = True) -> list[torch.Tensor]:
    loss_buffer: list[torch.Tensor] = []
    recipe._forward_backward_step(
        0,
        batch,
        loss_buffer=loss_buffer,
        num_batches=1,
        is_train=is_train,
        modality_loss_buffers=None,
    )
    return loss_buffer


# ---------------------------------------------------------------------------
# Loss blending
# ---------------------------------------------------------------------------
def test_loss_is_the_configured_weighted_sum():
    recipe = _make_recipe(ce_weight=0.3, kd_weight=0.7)

    loss_buffer = _run_step(recipe, _make_batch())

    components = recipe._loss_component_buffer[0]
    expected = 0.3 * components["loss_ce"] + 0.7 * components["loss_kd"]
    assert torch.allclose(loss_buffer[0], expected, atol=1e-5)
    assert components["loss_kd"] > 0
    assert torch.isfinite(loss_buffer[0])


def test_component_and_accuracy_buffers_are_populated():
    recipe = _make_recipe()

    _run_step(recipe, _make_batch())

    assert set(recipe._loss_component_buffer[0]) == {"loss_ce", "loss_kd"}
    assert all(value.dtype == torch.float32 for value in recipe._loss_component_buffer[0].values())
    # TrainCrossEncoderRecipe._run_train_optim_step stacks this buffer unconditionally.
    assert len(recipe._acc_buffer) == 1
    num_correct, batch_size = recipe._acc_buffer[0]
    assert batch_size == _N_QUERIES
    assert 0 <= int(num_correct) <= _N_QUERIES


def test_grouped_scores_have_query_by_passage_shape():
    """Without view(-1, n_passages) the softmax spans a size-1 dim and both losses hit zero."""
    kd_loss = _RecordingKDLoss()
    recipe = _make_recipe(kd_loss=kd_loss)

    _run_step(recipe, _make_batch())

    student_scores, teacher_scores = kd_loss.calls[0]
    assert student_scores.shape == (_N_QUERIES, _N_PASSAGES)
    assert teacher_scores.shape == (_N_QUERIES, _N_PASSAGES)


def test_raw_scores_use_each_models_own_internal_temperature():
    kd_loss = _RecordingKDLoss()
    recipe = _make_recipe(kd_loss=kd_loss, teacher_temperature=0.05)
    batch = _make_batch()

    _run_step(recipe, batch)

    student_scores, teacher_scores = kd_loss.calls[0]
    with torch.no_grad():
        # logits * own temperature undoes the model's internal division, recovering score().
        expected_student = recipe.model_parts[0].score(batch["features"]).view(_N_QUERIES, _N_PASSAGES)
        expected_teacher = recipe.teacher_model.score(batch["features"]).view(_N_QUERIES, _N_PASSAGES)
    assert torch.allclose(student_scores, expected_student, atol=1e-5)
    assert torch.allclose(teacher_scores, expected_teacher, atol=1e-5)
    assert student_scores.dtype == torch.float32 and teacher_scores.dtype == torch.float32


# ---------------------------------------------------------------------------
# Teacher isolation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("with_images", [False, True])
def test_teacher_sees_same_inputs_with_teacher_only_dummy_vision(with_images):
    """The frozen teacher keeps vision collectives active without altering student inputs."""
    recipe = _make_recipe()
    batch = _make_batch()
    if not with_images:
        batch.pop("pixel_values")

    _run_step(recipe, batch)

    student_batch = recipe.model_parts[0].seen_batches[0]
    teacher_batch = recipe.teacher_model.seen_batches[0]
    expected_inputs = {"features", "pixel_values"} if with_images else {"features"}
    assert set(student_batch) == expected_inputs
    assert set(teacher_batch) == expected_inputs | {"force_dummy_vision"}
    assert teacher_batch["force_dummy_vision"] is True
    for key in student_batch:
        assert torch.equal(student_batch[key], teacher_batch[key])


def test_gradients_reach_the_student_only():
    recipe = _make_recipe()

    _run_step(recipe, _make_batch())

    for param in recipe.model_parts[0].parameters():
        assert param.grad is not None and param.grad.abs().sum() > 0
    for param in recipe.teacher_model.parameters():
        assert param.grad is None


def test_no_backward_when_not_training():
    recipe = _make_recipe()

    _run_step(recipe, _make_batch(), is_train=False)

    assert all(param.grad is None for param in recipe.model_parts[0].parameters())


# ---------------------------------------------------------------------------
# SFT fallback
# ---------------------------------------------------------------------------
def test_ce_only_weights_reproduce_plain_cross_encoder_loss_and_gradients():
    """ce_loss_weight 1.0 with kd_loss_weight 0.0 must match TrainCrossEncoderRecipe exactly."""
    batch = _make_batch()
    recipe = _make_recipe(ce_weight=1.0, kd_weight=0.0, with_teacher=False)
    distill_loss = _run_step(recipe, batch)[0]

    reference = _make_recipe()
    reference.model_parts[0].load_state_dict(recipe.model_parts[0].state_dict())
    sft_loss_buffer: list[torch.Tensor] = []
    TrainCrossEncoderRecipe._forward_backward_step(
        reference,
        0,
        batch,
        loss_buffer=sft_loss_buffer,
        num_batches=1,
        is_train=True,
        modality_loss_buffers=None,
    )

    assert torch.allclose(distill_loss, sft_loss_buffer[0], atol=1e-6)
    for distill_param, sft_param in zip(recipe.model_parts[0].parameters(), reference.model_parts[0].parameters()):
        assert torch.allclose(distill_param.grad, sft_param.grad, atol=1e-6)


def test_zero_kd_weight_skips_the_teacher_forward():
    recipe = _make_recipe(ce_weight=1.0, kd_weight=0.0, with_teacher=True)

    _run_step(recipe, _make_batch())

    assert recipe.teacher_model.seen_batches == []
    assert recipe._loss_component_buffer[0]["loss_kd"].item() == 0.0


@pytest.mark.parametrize(
    "weights",
    [
        {"ce_loss_weight": 1.0, "kd_loss_weight": -0.5},
        {"ce_loss_weight": 0.0, "kd_loss_weight": 0.0},
        {"ce_loss_weight": 1.0, "kd_loss_weight": float("nan")},
        {"ce_loss_weight": float("inf"), "kd_loss_weight": 0.5},
    ],
)
def test_invalid_loss_weights_raise(weights):
    """A negative weight reaches a None teacher; two zeros make the loss constant."""
    recipe = object.__new__(DistillCrossEncoderRecipe)
    recipe.cfg = SimpleNamespace(get=lambda key, default=None: weights.get(key, default))

    with patch.object(TrainCrossEncoderRecipe, "setup", lambda self: None), pytest.raises(ValueError):
        recipe.setup()


# ---------------------------------------------------------------------------
# Metric reporting
# ---------------------------------------------------------------------------
def test_run_train_optim_step_delegates_and_attaches_component_metrics():
    recipe = _make_recipe()
    parent_result = SimpleNamespace(metrics={"loss": 1.0, "train_accuracy": 0.25})

    def _fake_parent(self, batches, max_grad_norm=None):
        # Stands in for the inherited body, which fills the buffer via _forward_backward_step.
        self._loss_component_buffer.append({"loss_ce": torch.tensor(2.0), "loss_kd": torch.tensor(4.0)})
        self._loss_component_buffer.append({"loss_ce": torch.tensor(4.0), "loss_kd": torch.tensor(6.0)})
        return parent_result

    with patch.object(TrainCrossEncoderRecipe, "_run_train_optim_step", _fake_parent):
        result = recipe._run_train_optim_step([], max_grad_norm=1.0)

    assert result is parent_result
    assert result.metrics["loss_ce"] == pytest.approx(3.0)
    assert result.metrics["loss_kd"] == pytest.approx(5.0)
    # train_accuracy only survives when the override delegates to super().
    assert result.metrics["train_accuracy"] == pytest.approx(0.25)


def test_kd_loss_is_not_tracked_for_checkpointing():
    """BaseRecipe would otherwise treat the loss module as a model to save."""
    recipe = _make_recipe()

    tracked = recipe.__dict__["__state_tracked"]
    assert "kd_loss" not in tracked
    assert "model_parts" in tracked


# ---------------------------------------------------------------------------
# Teacher construction
# ---------------------------------------------------------------------------
class _StubTeacherConfig:
    """Mimics the ``teacher_model`` config node, capturing the kwargs it is built with."""

    def __init__(self, factory):
        self.factory = factory
        self.kwargs: dict | None = None

    def instantiate(self, **kwargs):
        self.kwargs = kwargs
        return self.factory()


class _StubCfg:
    def __init__(self, teacher_config: _StubTeacherConfig):
        self.teacher_model = teacher_config

    def get(self, key, default=None):
        return default


def _build_teacher_on(strategy, factory=None):
    recipe = object.__new__(DistillCrossEncoderRecipe)
    recipe.student_internal_temperature = _STUDENT_TEMP
    recipe.student_vocab_size = _VOCAB_SIZE
    recipe.distributed_setup = DistributedSetup(
        mesh_context=MeshContext(
            device_mesh=SimpleNamespace(mesh_dim_names=("dp",)), process_group=object()
        ),
        strategy_config=strategy,
        activation_checkpointing=True,
    )
    teacher_config = _StubTeacherConfig(factory or (lambda: _StubCrossEncoder(scale=-0.8)))
    recipe.cfg = _StubCfg(teacher_config)

    recipe._build_teacher()
    return recipe, teacher_config


@pytest.mark.parametrize("reshard_after_forward", [True, False])
def test_teacher_setup_inherits_reshard_and_disables_activation_checkpointing(reshard_after_forward):
    """The teacher follows the config's reshard policy; only activation checkpointing is forced off."""
    _, teacher_config = _build_teacher_on(
        FSDP2Config(activation_checkpointing=True, reshard_after_forward=reshard_after_forward)
    )

    teacher_setup = teacher_config.kwargs["distributed_setup"]
    assert teacher_config.kwargs["peft_config"] is None
    assert teacher_setup.strategy_config.reshard_after_forward is reshard_after_forward
    assert teacher_setup.strategy_config.activation_checkpointing is False
    assert teacher_setup.activation_checkpointing is False


def test_teacher_setup_shares_the_student_mesh_and_leaves_it_unchanged():
    strategy = FSDP2Config(activation_checkpointing=True, reshard_after_forward=False)
    recipe, teacher_config = _build_teacher_on(strategy)

    teacher_setup = teacher_config.kwargs["distributed_setup"]
    # The policy context is copied, but meshes and process groups must be reused.
    assert teacher_setup.mesh_context.device_mesh is recipe.distributed_setup.mesh_context.device_mesh
    assert teacher_setup.mesh_context.moe_mesh is recipe.distributed_setup.mesh_context.moe_mesh
    assert teacher_setup.mesh_context.process_group is recipe.distributed_setup.mesh_context.process_group
    assert recipe.distributed_setup.strategy_config is strategy
    assert recipe.distributed_setup.activation_checkpointing is True
    assert strategy.reshard_after_forward is False
    assert strategy.activation_checkpointing is True


def test_non_fsdp2_strategy_only_disables_activation_checkpointing():
    """DDPConfig has no reshard_after_forward field, so the teacher setup must not require one."""
    _, teacher_config = _build_teacher_on(DDPConfig(activation_checkpointing=True))

    teacher_setup = teacher_config.kwargs["distributed_setup"]
    assert teacher_setup.strategy_config.activation_checkpointing is False
    assert not hasattr(teacher_setup.strategy_config, "reshard_after_forward")


def test_teacher_is_frozen_and_in_eval_mode():
    recipe, _ = _build_teacher_on(FSDP2Config())

    assert not recipe.teacher_model.training
    assert all(not param.requires_grad for param in recipe.teacher_model.parameters())


def test_teacher_construction_preserves_the_rng_stream():
    """Teacher init must not shift the student's dropout sequence relative to a fresh run."""
    torch.manual_seed(1234)
    expected = torch.rand(4)

    torch.manual_seed(1234)
    _build_teacher_on(FSDP2Config())
    actual = torch.rand(4)

    assert torch.equal(expected, actual)


def test_mismatched_vocab_size_raises():
    """Both models read one tokenization, so a vocabulary mismatch must fail loudly at setup."""
    with pytest.raises(ValueError, match="vocab_size"):
        _build_teacher_on(
            FSDP2Config(),
            factory=lambda: _StubCrossEncoder(scale=-0.8, vocab_size=_VOCAB_SIZE + 1),
        )


def test_matching_vocab_size_is_accepted():
    recipe, _ = _build_teacher_on(FSDP2Config())

    assert recipe.teacher_model is not None


def test_differing_internal_temperatures_warn_instead_of_raising(caplog):
    with caplog.at_level("WARNING"):
        recipe, _ = _build_teacher_on(FSDP2Config(), factory=lambda: _StubCrossEncoder(scale=-0.8, temperature=0.05))

    assert recipe.teacher_internal_temperature == pytest.approx(0.05)
    assert recipe.student_internal_temperature == pytest.approx(_STUDENT_TEMP)
    assert "config.temperature differ" in caplog.text


# ---------------------------------------------------------------------------
# Shipped config
# ---------------------------------------------------------------------------
def test_example_config_wires_the_distillation_recipe():
    raw_config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    assert raw_config["recipe"] == "DistillCrossEncoderRecipe"
    assert raw_config["ce_loss_weight"] + raw_config["kd_loss_weight"] == pytest.approx(1.0)
    assert raw_config["kd_temperature"] > 0
    # Same target as the student, so --teacher_model.pretrained_model_name_or_path overrides work.
    assert raw_config["teacher_model"]["_target_"] == raw_config["model"]["_target_"]
    assert "pretrained_model_name_or_path" in raw_config["teacher_model"]
    assert raw_config["teacher_model"]["num_labels"] == raw_config["model"]["num_labels"] == 1
    assert raw_config["dataloader"]["collate_fn"]["collator_fn_name"] == "process_queries_documents_crossencoder"
