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

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nemo_automodel.recipes.vlm import kd as vlm_kd


class _Output:
    def __init__(self, logits: torch.Tensor):
        self.logits = logits
        self.hidden_states = None


class _Teacher(nn.Module):
    def forward(self, input_ids):
        vocab = 4
        logits = torch.arange(input_ids.numel() * vocab, dtype=torch.float32).view(*input_ids.shape, vocab)
        return _Output(logits)


class _Student(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, input_ids):
        vocab = 4
        logits = torch.ones(*input_ids.shape, vocab, dtype=torch.float32) * self.weight
        return _Output(logits)


class _KD:
    temperature = 1.0

    def __call__(self, student_logits, teacher_logits, labels, num_batch_labels=None):
        del teacher_logits, labels, num_batch_labels
        return student_logits.mean()


@pytest.mark.cuda(False)
def test_vlm_kd_teacher_forward_uses_scoped_offloading_when_enabled(monkeypatch):
    seen = {}

    class _FakeScopedModuleOffloading:
        def __init__(self, model, enabled=False):
            seen["model"] = model
            seen["enabled"] = enabled

        def __enter__(self):
            seen["entered"] = True
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            seen["exited"] = True
            return False

    monkeypatch.setattr(vlm_kd, "ScopedModuleOffloading", _FakeScopedModuleOffloading)
    monkeypatch.setattr(
        vlm_kd,
        "calculate_loss",
        lambda *args, **kwargs: kwargs["logits"].mean(),
    )

    recipe = vlm_kd.KnowledgeDistillationRecipeForVLM.__new__(vlm_kd.KnowledgeDistillationRecipeForVLM)
    recipe.dist_env = SimpleNamespace(device="cpu")
    recipe.device_mesh = None
    recipe.pp_enabled = False
    recipe.distributed_config = SimpleNamespace(defer_fsdp_grad_sync=True)
    recipe.model_parts = [_Student()]
    recipe.teacher_model = _Teacher()
    recipe.loss_fn = object()
    recipe.kd_loss_fn = _KD()
    recipe.kd_ratio = 0.5
    recipe._kd_loss_buffer = []
    recipe._ce_loss_buffer = []
    recipe._offload_teacher_model = True
    recipe._get_dp_group_size = lambda include_cp=True: 1

    loss_buffer = []
    batch = {
        "input_ids": torch.tensor([[1, 2]]),
        "labels": torch.tensor([[1, -100]]),
    }

    recipe._forward_backward_step(
        0,
        batch,
        loss_buffer=loss_buffer,
        num_label_tokens=1,
        num_batches=1,
    )

    assert seen == {
        "model": recipe.teacher_model,
        "enabled": True,
        "entered": True,
        "exited": True,
    }
    assert len(loss_buffer) == 1
    assert len(recipe._ce_loss_buffer) == 1
    assert len(recipe._kd_loss_buffer) == 1
