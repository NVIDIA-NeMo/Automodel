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

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nemo_automodel.recipes.llm import kd as llm_kd
from nemo_automodel.recipes.vlm import kd as vlm_kd


class _Cfg:
    def __init__(self, **values):
        self._values = values

    def get(self, key, default=None):
        return self._values.get(key, default)


class _Teacher(nn.Module):
    def forward(self, input_ids):
        return SimpleNamespace(logits=torch.nn.functional.one_hot(input_ids, num_classes=4).float())


_RECIPE_CASES = (
    pytest.param(
        llm_kd,
        llm_kd.KnowledgeDistillationRecipeForNextTokenPrediction,
        llm_kd.TrainFinetuneRecipeForNextTokenPrediction,
        id="llm",
    ),
    pytest.param(
        vlm_kd,
        vlm_kd.KnowledgeDistillationRecipeForVLM,
        vlm_kd.FinetuneRecipeForVLM,
        id="vlm",
    ),
)


@pytest.mark.parametrize("recipe_module,recipe_cls,_", _RECIPE_CASES)
@pytest.mark.parametrize("is_student", (True, False), ids=("student", "teacher"))
def test_create_distributed_setup_assigns_the_role_process_group(monkeypatch, recipe_module, recipe_cls, _, is_student):
    student_setup = SimpleNamespace(mesh_context=SimpleNamespace(process_group=None))
    teacher_setup = SimpleNamespace(mesh_context=SimpleNamespace(process_group=None))
    setups = SimpleNamespace(separate=True, student=student_setup, teacher=teacher_setup)
    bridge = SimpleNamespace(
        is_student=is_student,
        is_teacher=not is_student,
        student_group="student-group",
        teacher_group="teacher-group",
    )
    monkeypatch.setattr(recipe_module, "create_kd_distributed_setups", lambda cfg, world_size: setups)
    monkeypatch.setattr(recipe_module, "KDMeshBridge", lambda built_setups, device: bridge)

    recipe = object.__new__(recipe_cls)
    recipe.cfg = object()
    recipe.dist_env = SimpleNamespace(world_size=4, device="cpu")

    result = recipe._create_distributed_setup()

    assert result is (student_setup if is_student else teacher_setup)
    assert recipe._training_process_group == "student-group"
    expected_group = "student-group" if is_student else "teacher-group"
    assert result.mesh_context.process_group == expected_group
    assert recipe._should_setup_training_components() is is_student


@pytest.mark.parametrize("recipe_module,recipe_cls,_", _RECIPE_CASES)
def test_create_distributed_setup_reuses_the_shared_student_mesh(monkeypatch, recipe_module, recipe_cls, _):
    student_setup = object()
    setups = SimpleNamespace(separate=False, student=student_setup)
    monkeypatch.setattr(recipe_module, "create_kd_distributed_setups", lambda cfg, world_size: setups)

    recipe = object.__new__(recipe_cls)
    recipe.cfg = object()
    recipe.dist_env = SimpleNamespace(world_size=2, device="cpu")

    assert recipe._create_distributed_setup() is student_setup
    assert recipe.kd_mesh_bridge is None
    assert recipe._should_setup_training_components() is True


@pytest.mark.parametrize("recipe_module,recipe_cls,_", _RECIPE_CASES)
def test_setup_kd_state_builds_loss_and_resets_buffers(monkeypatch, recipe_module, recipe_cls, _):
    loss = object()
    monkeypatch.setattr(recipe_module, "_build_kd_loss_fn", lambda cfg: loss)
    recipe = object.__new__(recipe_cls)
    recipe.cfg = _Cfg(kd_loss_fn="loss-config", kd_ratio=0.75)

    recipe._setup_kd_state()

    assert recipe.kd_loss_fn is loss
    assert recipe.kd_ratio == 0.75
    assert recipe._kd_loss_buffer == []
    assert recipe._ce_loss_buffer == []


@pytest.mark.parametrize("_,recipe_cls,__", _RECIPE_CASES)
def test_get_separate_teacher_logits_returns_the_received_wave(_, recipe_cls, __):
    expected = torch.ones(1, 2, 4)
    received = iter((None, expected))
    calls = []
    bridge = SimpleNamespace(
        num_waves=2,
        broadcast_command=lambda command: calls.append(("command", command)),
        send_batch=lambda wave, batch: calls.append(("batch", wave, batch)),
        send_logits=lambda wave, logits: next(received),
    )
    recipe = object.__new__(recipe_cls)
    recipe.kd_mesh_bridge = bridge
    batch = {"input_ids": torch.tensor([[1, 2]]), "labels": torch.tensor([[1, 2]])}

    assert recipe._get_separate_teacher_logits(batch) is expected
    assert calls[0][0] == "command"
    assert [call[1] for call in calls if call[0] == "batch"] == [0, 1]


@pytest.mark.parametrize("_,recipe_cls,__", _RECIPE_CASES)
def test_get_separate_teacher_logits_rejects_missing_output(_, recipe_cls, __):
    recipe = object.__new__(recipe_cls)
    recipe.kd_mesh_bridge = SimpleNamespace(
        num_waves=1,
        broadcast_command=lambda command: None,
        send_batch=lambda wave, batch: None,
        send_logits=lambda wave, logits: None,
    )

    with pytest.raises(RuntimeError, match="did not receive teacher logits"):
        recipe._get_separate_teacher_logits({})


@pytest.mark.parametrize("recipe_module,recipe_cls,_", _RECIPE_CASES)
def test_teacher_worker_serves_each_wave_until_stop(monkeypatch, recipe_module, recipe_cls, _):
    commands = iter((recipe_module.RUN_TEACHER, recipe_module.STOP_TEACHER))
    sent = []

    class _SignalHandler:
        def __init__(self, group):
            assert group == "teacher-group"

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    monkeypatch.setattr(recipe_module, "DistributedSignalHandler", _SignalHandler)
    recipe = object.__new__(recipe_cls)
    recipe.kd_mesh_bridge = SimpleNamespace(
        teacher_group="teacher-group",
        num_waves=2,
        broadcast_command=lambda: next(commands),
        send_batch=lambda wave, batch: {"wave": wave},
        send_logits=lambda wave, logits: sent.append((wave, logits)),
    )
    recipe._teacher_forward_separate = lambda batch: torch.tensor(batch["wave"])

    recipe._run_teacher_worker()

    assert [wave for wave, _ in sent] == [0, 1]


@pytest.mark.parametrize("recipe_module,recipe_cls,_", _RECIPE_CASES)
def test_teacher_forward_separate_materializes_logits(monkeypatch, recipe_module, recipe_cls, _):
    materialized = []
    tokens = object()
    monkeypatch.setattr(
        recipe_module,
        "materialize_teacher_logits",
        lambda logits, *, tokens, sequence_length: materialized.append((tokens, sequence_length)) or logits,
    )
    recipe = object.__new__(recipe_cls)
    recipe.kd_mesh_bridge = SimpleNamespace(move_to_device=lambda batch: batch)
    recipe.device_mesh = None
    recipe.cp_runtime = SimpleNamespace(
        prepare_forward=lambda model, batch, **kwargs: SimpleNamespace(
            context=nullcontext(), batch=batch, tokens=tokens
        )
    )
    recipe.teacher_model = _Teacher()
    if recipe_module is llm_kd:
        recipe.pp_enabled = False

    logits = recipe._teacher_forward_separate({"input_ids": torch.tensor([[1, 2]]), "labels": torch.tensor([[1, 2]])})

    assert logits.shape == (1, 2, 4)
    assert materialized == [(tokens, 2)]


@pytest.mark.parametrize("recipe_module,recipe_cls,base_cls", _RECIPE_CASES)
def test_run_loop_routes_teacher_and_stops_after_student(monkeypatch, recipe_module, recipe_cls, base_cls):
    parent_calls = []
    monkeypatch.setattr(base_cls, "run_train_validation_loop", lambda self: parent_calls.append(self) or "trained")

    teacher = object.__new__(recipe_cls)
    teacher.separate_meshes = True
    teacher.kd_mesh_bridge = SimpleNamespace(is_teacher=True)
    teacher_calls = []
    teacher._run_teacher_worker = lambda: teacher_calls.append("served")
    assert teacher.run_train_validation_loop() is None
    assert teacher_calls == ["served"]

    commands = []
    student = object.__new__(recipe_cls)
    student.separate_meshes = True
    student.pp_enabled = False
    student.kd_mesh_bridge = SimpleNamespace(
        is_teacher=False,
        broadcast_command=lambda command: commands.append(command),
    )
    assert student.run_train_validation_loop() == "trained"
    assert commands == [recipe_module.STOP_TEACHER]

    shared = object.__new__(recipe_cls)
    shared.separate_meshes = False
    shared.pp_enabled = False
    assert shared.run_train_validation_loop() == "trained"
    assert parent_calls == [student, shared]
