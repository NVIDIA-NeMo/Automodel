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

"""Real-CUDA functional coverage for cached-teacher bi-encoder distillation."""

from __future__ import annotations

import time
from collections import deque
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from nemo_automodel.components.loss.embedding_distill import EmbeddingDistillLoss, EmbeddingMSELoss
from nemo_automodel.components.loss.infonce import InfoNCEDistillLoss, InfoNCELoss
from nemo_automodel.recipes.retrieval.distill_bi_encoder import EmbeddingDistillRecipe

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


class _TinyStudent(nn.Module):
    """Embedding student with native and teacher-space representations."""

    def __init__(self, vocab_size: int, hidden_size: int, teacher_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.projection = nn.Linear(hidden_size, teacher_size)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Pool token embeddings from tensors shaped ``[batch, sequence]``."""
        hidden = self.embedding(batch["input_ids"])
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return F.normalize(pooled, dim=-1), self.projection(pooled), {}


def test_cached_teacher_distillation_updates_backbone_and_projection() -> None:
    """Run four production loss branches, backward, clipping, and AdamW on CUDA."""
    torch.manual_seed(11)
    device = torch.device("cuda")
    student = _TinyStudent(vocab_size=48, hidden_size=8, teacher_size=6).to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-2)

    recipe = EmbeddingDistillRecipe.__new__(EmbeddingDistillRecipe)
    recipe.loss_weights = {"distill": 1.0, "mse": 0.5, "score": 0.0, "intermediate": 0.0, "nce": 0.2, "nce_kd": 0.2}
    recipe.distill_loss = EmbeddingDistillLoss()
    recipe.mse_loss = EmbeddingMSELoss()
    recipe.infonce_loss = InfoNCELoss(cross_device_negatives=False)
    recipe.infonce_distill_loss = InfoNCEDistillLoss(cross_device_negatives=False)
    recipe.use_cached_teacher = True
    recipe.model_parts = [student]
    recipe.optimizer = [optimizer]
    recipe.lr_scheduler = None
    recipe.checkpointer = SimpleNamespace(maybe_wait_for_staging=lambda: None)
    recipe.dist_env = SimpleNamespace(device=device)
    recipe.distributed_config = SimpleNamespace(defer_fsdp_grad_sync=True)
    recipe.device_mesh = None
    recipe.moe_mesh = None
    recipe.pp_enabled = False
    recipe.step_scheduler = SimpleNamespace(step=0, epoch=0)
    recipe.loss_average_window = deque(maxlen=8)
    recipe.timestamp = time.perf_counter()
    recipe._get_dp_group_size = lambda include_cp=False: 1

    batch_size, num_negatives, sequence_length = 2, 2, 4
    batch = {
        "q_input_ids": torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]),
        "q_attention_mask": torch.ones(batch_size, sequence_length, dtype=torch.long),
        "d_input_ids": torch.tensor([[9, 10, 11, 12], [13, 14, 15, 16]]),
        "d_attention_mask": torch.ones(batch_size, sequence_length, dtype=torch.long),
        "n_input_ids": torch.tensor([[[17, 18, 19, 20], [21, 22, 23, 24]], [[25, 26, 27, 28], [29, 30, 31, 32]]]),
        "n_attention_mask": torch.ones(batch_size, num_negatives, sequence_length, dtype=torch.long),
        "n_mask": torch.tensor([[1, 1], [1, 0]], dtype=torch.long),
        "t_q_pool": torch.randn(batch_size, 6),
        "t_d_pool": torch.randn(batch_size, 6),
        "t_n_pool": torch.randn(batch_size, num_negatives, 6),
    }
    before_backbone = student.embedding.weight.detach().clone()
    before_projection = student.projection.weight.detach().clone()

    metrics = recipe._run_train_optim_step([batch], max_grad_norm=1.0)

    for key in ("loss", "loss_distill", "loss_mse", "loss_nce", "loss_nce_distill"):
        assert torch.isfinite(torch.tensor(metrics.metrics[key]))
    assert metrics.metrics["loss"] > 0
    assert torch.isfinite(torch.as_tensor(metrics.metrics["grad_norm"]))
    assert not torch.equal(student.embedding.weight, before_backbone)
    assert not torch.equal(student.projection.weight, before_projection)
