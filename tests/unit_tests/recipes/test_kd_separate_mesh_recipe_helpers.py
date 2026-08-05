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
from unittest.mock import Mock

import pytest
import torch
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.components.loss.kd_loss import KDLoss
from nemo_automodel.components.loss.linear_ce import FusedLinearCrossEntropy
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


class _TensorHiddenStateStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("hidden_states", torch.arange(24, dtype=torch.float32).reshape(2, 3, 4))
        self.logits_to_keep = None

    def forward(self, input_ids: torch.Tensor, logits_to_keep: int | None = None) -> CausalLMOutputWithPast:
        """Return logits and a tensor-valued final hidden state.

        Args:
            input_ids: Tensor of shape [batch, sequence].
            logits_to_keep: Number of trailing logits requested by the fused-loss path.

        Returns:
            Model output with logits of shape [batch, sequence, vocab] and
            tensor-valued hidden states of shape [batch, sequence, hidden].
        """
        self.logits_to_keep = logits_to_keep
        logits = torch.ones(*input_ids.shape, 4, dtype=torch.float32) * self.scale
        if logits_to_keep is not None:
            logits = logits[:, -logits_to_keep:]
        return CausalLMOutputWithPast(logits=logits, hidden_states=self.hidden_states)


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
def test_kd_fused_loss_preserves_tensor_valued_hidden_states(monkeypatch, recipe_module, recipe_cls, _):
    def no_op_sharder(model, mesh, batch, **kwargs):
        """Return a context-parallel sharder that preserves the input batch.

        Args:
            model: Student model under test.
            mesh: Unused device mesh.
            batch: Mapping containing tensors of shape [batch, sequence].
            **kwargs: Unused context-parallel options.

        Returns:
            Object whose shard operation returns the unchanged batch.
        """
        del model, mesh, kwargs
        return SimpleNamespace(shard=lambda actual_batch: (nullcontext, actual_batch))

    calculate_loss = Mock(return_value=torch.tensor(2.0))
    monkeypatch.setattr(recipe_module, "ContextParallelSharder", no_op_sharder)
    monkeypatch.setattr(recipe_module, "calculate_loss", calculate_loss)

    student = _TensorHiddenStateStudent()
    recipe = object.__new__(recipe_cls)
    recipe.dist_env = SimpleNamespace(device="cpu")
    recipe.device_mesh = None
    recipe.pp_enabled = False
    recipe.distributed_config = SimpleNamespace(defer_fsdp_grad_sync=True)
    recipe.model_parts = [student]
    recipe.teacher_model = _Teacher()
    recipe.loss_fn = FusedLinearCrossEntropy()
    recipe.kd_loss_fn = KDLoss()
    recipe.kd_ratio = 0.5
    recipe._offload_teacher_model = False
    recipe.separate_meshes = False
    recipe._get_dp_group_size = lambda include_cp=True: 1

    batch = {
        "input_ids": torch.tensor([[0, 1, 2], [1, 2, 3]]),
        "labels": torch.tensor([[0, 1, 2], [1, 2, -100]]),
    }
    if recipe_module is vlm_kd:
        recipe._ce_loss_buffer = []
        recipe._kd_loss_buffer = []
        recipe._forward_backward_step(
            0,
            batch,
            loss_buffer=[],
            num_label_tokens=5,
            num_batches=1,
            is_train=False,
        )
    else:
        recipe._forward_backward_step(
            0,
            batch,
            num_label_tokens=5,
            num_batches=1,
            is_train=False,
        )

    received_hidden_states = calculate_loss.call_args.kwargs["hidden_states"]
    assert student.logits_to_keep is None
    assert received_hidden_states is student.hidden_states
    assert received_hidden_states.shape[:-1] == batch["labels"].shape


@pytest.mark.parametrize("recipe_module,recipe_cls,parent_cls", _RECIPE_CASES)
def test_kd_setup_defaults_torch_optimizer_storage_to_fp32(monkeypatch, recipe_module, recipe_cls, parent_cls):
    class SetupStopped(Exception):
        pass

    cfg = ConfigNode(
        {
            "model": {},
            "teacher_model": {},
            "optimizer": {"_target_": "torch.optim.AdamW", "lr": 0.01},
        }
    )

    def stop_parent_setup(self):
        raise SetupStopped

    monkeypatch.setattr(recipe_module, "_verify_tokenizer_compatibility", lambda *args, **kwargs: None)
    monkeypatch.setattr(parent_cls, "setup", stop_parent_setup)

    recipe = object.__new__(recipe_cls)
    recipe.cfg = cfg

    with pytest.raises(SetupStopped):
        recipe.setup()

    assert cfg.model.torch_dtype == "float32"


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
    monkeypatch.setattr(
        recipe_module,
        "ContextParallelSharder",
        lambda model, mesh, batch, **kwargs: SimpleNamespace(shard=lambda actual: (nullcontext, actual)),
    )
    materialized = []
    monkeypatch.setattr(
        recipe_module,
        "materialize_teacher_logits",
        lambda logits, *, device_mesh, sequence_length: materialized.append((device_mesh, sequence_length)) or logits,
    )
    recipe = object.__new__(recipe_cls)
    recipe.kd_mesh_bridge = SimpleNamespace(move_to_device=lambda batch: batch)
    recipe.device_mesh = None
    recipe.teacher_model = _Teacher()
    if recipe_module is llm_kd:
        recipe.pp_enabled = False

    logits = recipe._teacher_forward_separate({"input_ids": torch.tensor([[1, 2]]), "labels": torch.tensor([[1, 2]])})

    assert logits.shape == (1, 2, 4)
    assert materialized == [(None, 2)]


def test_llm_teacher_pp_updates_stage_shapes_for_variable_length_waves(monkeypatch):
    monkeypatch.setattr(
        llm_kd,
        "ContextParallelSharder",
        lambda model, mesh, batch, **kwargs: SimpleNamespace(shard=lambda actual: (nullcontext, actual)),
    )
    monkeypatch.setattr(llm_kd, "materialize_teacher_logits", lambda logits, **kwargs: logits)

    calls = []

    def schedule_eval(input_ids, **kwargs):
        calls.append(("eval", input_ids.shape[1]))

    teacher_model = SimpleNamespace(_teacher_logits_capture=[None])
    teacher_pp = SimpleNamespace(
        info=SimpleNamespace(
            has_first_stage=True,
            has_last_stage=True,
            schedule=SimpleNamespace(eval=schedule_eval),
        ),
        update_seq_len=lambda sequence_length: calls.append(("update", sequence_length)),
    )
    recipe = object.__new__(llm_kd.KnowledgeDistillationRecipeForNextTokenPrediction)
    recipe.kd_mesh_bridge = SimpleNamespace(move_to_device=lambda batch: batch)
    recipe.device_mesh = None
    recipe.teacher_model = teacher_model
    recipe.teacher_pp = teacher_pp
    recipe.pp_enabled = True

    for sequence_length in (195, 186):
        teacher_model._teacher_logits_capture[0] = [torch.ones(1, sequence_length, 4)]
        logits = recipe._teacher_forward_separate(
            {
                "input_ids": torch.ones(1, sequence_length, dtype=torch.long),
                "attention_mask": torch.ones(1, sequence_length, dtype=torch.long),
                "labels": torch.ones(1, sequence_length, dtype=torch.long),
            }
        )
        assert logits.shape == (1, sequence_length, 4)

    assert calls == [("update", 195), ("eval", 195), ("update", 186), ("eval", 186)]


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
