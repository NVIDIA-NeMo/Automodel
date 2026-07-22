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

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from nemo_automodel.components.loss import kd_loss as kd_loss_module
from nemo_automodel.components.loss.kd_loss import KDLoss
from nemo_automodel.recipes.vlm import kd as vlm_kd


class _MeshDim:
    def __init__(self, size: int):
        self._size = size

    def size(self) -> int:
        return self._size


class _DeviceMesh:
    mesh_dim_names = ("cp", "tp")

    def __init__(self, *, cp_size: int = 1, tp_size: int = 1):
        self._dims = {"cp": _MeshDim(cp_size), "tp": _MeshDim(tp_size)}

    def __getitem__(self, key: str) -> _MeshDim:
        return self._dims[key]


class _StudentVLM(nn.Module):
    def __init__(self, *, hidden_size: int = 8, vocab_size: int = 11):
        super().__init__()
        self.embedding = nn.Embedding(32, hidden_size)
        self.proj = nn.Linear(hidden_size, vocab_size, bias=False)
        self.pre_embed_calls = []
        self.forward_calls = []

    def get_input_embeddings(self):
        return self.embedding

    def prepare_model_inputs_for_cp(self, batch, *, num_chunks=1):
        # Sharder-only CP hook: consumes nothing (the forward embeds + shards).
        self.pre_embed_calls.append(dict(batch))
        return {}

    def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        self.forward_calls.append({"input_ids": input_ids, "inputs_embeds": inputs_embeds, **kwargs})
        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)
        return SimpleNamespace(logits=self.proj(inputs_embeds), hidden_states=None)


class _TeacherVLM(nn.Module):
    def __init__(self, *, hidden_size: int = 8, vocab_size: int = 11):
        super().__init__()
        self.embedding = nn.Embedding(32, hidden_size)
        self.proj = nn.Linear(hidden_size, vocab_size, bias=False)
        self.forward_calls = []

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        self.forward_calls.append({"input_ids": input_ids, "inputs_embeds": inputs_embeds, **kwargs})
        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)
        return SimpleNamespace(logits=self.proj(inputs_embeds), hidden_states=None)


def _make_recipe(*, student: nn.Module, teacher: nn.Module, kd_loss_fn: KDLoss, device_mesh=None):
    recipe = object.__new__(vlm_kd.KnowledgeDistillationRecipeForVLM)
    recipe.dist_env = SimpleNamespace(device=torch.device("cpu"))
    recipe.device_mesh = device_mesh
    recipe.pp_enabled = False
    recipe.model_parts = [student]
    recipe.teacher_model = teacher
    recipe.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    recipe.kd_loss_fn = kd_loss_fn
    recipe.kd_ratio = 1.0
    recipe.distributed_config = SimpleNamespace(defer_fsdp_grad_sync=False)
    recipe._ce_loss_buffer = []
    recipe._kd_loss_buffer = []
    recipe._offload_teacher_model = False
    recipe._get_dp_group_size = lambda include_cp=False: 1
    return recipe


def _batch():
    return {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "pixel_values": torch.ones(1, 3, 2, 2),
        "attention_mask": torch.ones(1, 4),
        "labels": torch.tensor([[1, 2, -100, 4]]),
    }


@pytest.fixture(scope="module")
def trivial_pg(tmp_path_factory):
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed not available")
    if not torch.distributed.is_initialized():
        store_path = tmp_path_factory.mktemp("dist") / "store"
        torch.distributed.init_process_group(
            backend="gloo",
            init_method=f"file://{store_path}",
            rank=0,
            world_size=1,
        )
    return torch.distributed.group.WORLD


def test_vlm_kd_uses_tp_kd_loss_path(monkeypatch, trivial_pg):
    original_kl_forward_tp = kd_loss_module._kl_forward_tp
    tp_calls = []

    def wrapped_kl_forward_tp(t_logits, s_logits, tp_group):
        tp_calls.append((t_logits.shape, s_logits.shape, tp_group))
        return original_kl_forward_tp(t_logits, s_logits, tp_group)

    monkeypatch.setattr(kd_loss_module, "_kl_forward_tp", wrapped_kl_forward_tp)
    monkeypatch.setattr(vlm_kd, "get_sync_ctx", lambda *args, **kwargs: nullcontext())

    student = _StudentVLM()
    teacher = _TeacherVLM()
    recipe = _make_recipe(student=student, teacher=teacher, kd_loss_fn=KDLoss(tp_group=trivial_pg))
    loss_buffer = []

    recipe._forward_backward_step(
        0,
        _batch(),
        loss_buffer=loss_buffer,
        num_label_tokens=3,
        num_batches=1,
        is_train=True,
    )

    assert len(tp_calls) == 1
    assert tp_calls[0][2] is trivial_pg
    assert student.proj.weight.grad is not None
    assert len(loss_buffer) == 1
    assert torch.isfinite(loss_buffer[0])


def test_vlm_kd_cp_prepare_shards_input_ids_and_teacher_embeds_them(monkeypatch):
    make_cp_calls = []

    def fake_shard(sharder, batch):
        """Capture the unsharded model-input mapping at the public CP seam.

        Args:
            sharder: Sharder configured with the recipe's device mesh.
            batch: Mutable model-input mapping whose tensors have global batch
                and sequence extents.

        Returns:
            The null context factory and the same input mapping.
        """
        make_cp_calls.append((sharder, dict(batch)))
        return nullcontext, batch

    monkeypatch.setattr(vlm_kd.ContextParallelSharder, "shard", fake_shard)

    student = _StudentVLM(hidden_size=8)
    teacher = _TeacherVLM(hidden_size=8)
    recipe = _make_recipe(
        student=student,
        teacher=teacher,
        kd_loss_fn=KDLoss(),
        device_mesh=_DeviceMesh(cp_size=2),
    )
    loss_buffer = []

    recipe._forward_backward_step(
        0,
        _batch(),
        loss_buffer=loss_buffer,
        num_label_tokens=3,
        num_batches=1,
        is_train=False,
    )

    # Sunk student: the sharder-only hook consumes nothing, so input_ids (not
    # inputs_embeds) flows to CP; the teacher embeds those input_ids itself.
    assert len(student.pre_embed_calls) == 1
    assert len(make_cp_calls) == 1
    cp_batch = make_cp_calls[0][1]
    assert "input_ids" in cp_batch
    assert "inputs_embeds" not in cp_batch
    assert "labels" in cp_batch
    assert teacher.forward_calls[0]["input_ids"] is not None
    assert teacher.forward_calls[0]["inputs_embeds"] is None
    assert torch.isfinite(loss_buffer[0])
