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

"""The KD step without pipeline parallelism runs the student, and only the student, under its TE quantization."""

from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nemo_automodel.components.loss.kd_loss import KDLoss
from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy
from nemo_automodel.components.models.common.utils import TEFp8Config, TENvfp4Config, TEQuantizationConfig
from nemo_automodel.recipes.llm import kd as llm_kd

VOCAB, HIDDEN = 11, 8


class _RecordingModel(nn.Module):
    """A tiny language model that records whether the TE quantization context is entered."""

    def __init__(self, entered: list[bool]) -> None:
        super().__init__()
        self.embed = nn.Embedding(VOCAB, HIDDEN)
        self.head = nn.Linear(HIDDEN, VOCAB)
        self.entered = entered
        self.forward_seen: list[bool] = []
        self.backward_seen: list[bool] = []

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Record the context and return the logits.

        Args:
            input_ids: Tensor of shape [batch, sequence].

        Returns:
            Tensor of shape [batch, sequence, VOCAB].
        """
        self.forward_seen.append(self.entered[0])
        logits = self.head(self.embed(input_ids))

        def record_backward(grad: torch.Tensor) -> None:
            """Record the context when the gradient of the logits arrives.

            Args:
                grad: Tensor of shape [batch, sequence, VOCAB].
            """
            self.backward_seen.append(self.entered[0])

        if logits.requires_grad:
            logits.register_hook(record_backward)
        return logits


def _step(
    monkeypatch: pytest.MonkeyPatch, te_quantization: TEQuantizationConfig | None
) -> tuple[torch.Tensor, torch.Tensor, _RecordingModel, _RecordingModel, list[bool]]:
    """Run one KD step without pipeline parallelism on fixed weights and data.

    Returns:
        The scalar loss, the gradient of the student head of shape [VOCAB, HIDDEN], the student, the teacher,
        and the flag that is true while the TE quantization context is entered.
    """
    entered = [False]

    @contextmanager
    def autocast() -> Iterator[None]:
        entered[0] = True
        try:
            yield
        finally:
            entered[0] = False

    monkeypatch.setattr(TEQuantizationConfig, "maybe_te_autocast", lambda self: autocast())
    torch.manual_seed(0)
    student, teacher = _RecordingModel(entered), _RecordingModel(entered)
    recipe = object.__new__(llm_kd.KnowledgeDistillationRecipeForNextTokenPrediction)
    recipe.pp_enabled = False
    recipe.model_parts = [student]
    recipe.teacher_model = teacher
    recipe._offload_teacher_model = False
    recipe.device_mesh = None
    recipe.distributed_config = None
    recipe.dist_env = SimpleNamespace(device=torch.device("cpu"))
    recipe.loss_fn = MaskedCrossEntropy()
    recipe.kd_loss_fn = KDLoss()
    recipe.kd_ratio = 1.0
    recipe.te_quantization = te_quantization
    recipe._get_dp_group_size = lambda include_cp=False: 1
    batch = {"input_ids": torch.randint(1, VOCAB, (2, 5)), "labels": torch.randint(1, VOCAB, (2, 5))}

    loss, _, _ = recipe._forward_backward_step(0, batch, num_label_tokens=10, num_batches=1)

    return loss, student.head.weight.grad, student, teacher, entered


@pytest.mark.parametrize("te_quantization", [TEFp8Config(), TENvfp4Config()], ids=["te_fp8", "te_nvfp4"])
def test_the_student_runs_under_its_te_quantization(
    monkeypatch: pytest.MonkeyPatch, te_quantization: TEQuantizationConfig
) -> None:
    loss, gradient, student, teacher, entered = _step(monkeypatch, te_quantization)

    assert student.forward_seen == [True]
    assert student.backward_seen == [True]
    # The teacher is a full-precision model: its forward stays outside.
    assert teacher.forward_seen == [False]
    assert not teacher.backward_seen
    assert not entered[0]
    # The context is all that differs from a step without TE quantization.
    loss_plain, gradient_plain, _, _, _ = _step(monkeypatch, None)
    assert torch.equal(loss, loss_plain)
    assert torch.equal(gradient, gradient_plain)


def test_no_te_quantization_enters_no_context(monkeypatch: pytest.MonkeyPatch) -> None:
    loss, gradient, student, teacher, _ = _step(monkeypatch, None)

    assert student.forward_seen == [False]
    assert student.backward_seen == [False]
    assert teacher.forward_seen == [False]
    assert torch.isfinite(loss)
    assert gradient is not None and gradient.abs().sum() > 0
