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

"""Real TP coverage of the separate-mesh KD bridge and its pipeline wrapper."""

import datetime
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor

pytestmark = pytest.mark.cpu_only


def _worker(rank: int, init_method: str) -> None:
    """Compare bridge and PP loss/gradients on even, uneven, and empty TP shards."""
    from nemo_automodel.components.loss.kd_loss import KDLoss
    from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy
    from nemo_automodel.recipes.kd_utils import KDMeshBridge
    from nemo_automodel.recipes.llm.kd import KnowledgeDistillationRecipeForNextTokenPrediction

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=init_method, rank=rank, world_size=4, timeout=datetime.timedelta(seconds=45)
    )
    try:
        mesh = init_device_mesh("cpu", (4,))
        generator = torch.Generator().manual_seed(437)
        for vocab in (12, 13, 9):
            full_student = torch.randn(1, 2, vocab, generator=generator)
            teacher = torch.randn(1, 2, vocab, generator=generator)
            labels = torch.tensor([[0, vocab - 1]])
            rows = -(-vocab // 4)
            start = min(rows * rank, vocab)
            end = min(start + rows, vocab)
            for ratio in (None, 0.5, 1.0):
                local = full_student[..., start:end].clone().requires_grad_()
                student = DTensor.from_local(
                    local, mesh, [Shard(-1)], shape=full_student.shape, stride=full_student.stride()
                )
                dense = full_student.clone().requires_grad_()
                if ratio is None:
                    matched = KDMeshBridge.match_student_vocab_shard(student, teacher)
                    assert isinstance(matched, DTensor)
                    assert matched.shape == student.shape and matched.placements == student.placements
                    torch.testing.assert_close(matched.to_local(), teacher[..., start:end])
                    actual = KDLoss()(student, matched, labels)
                    expected = KDLoss()(dense, teacher, labels)
                    explicit = KDLoss(tp_group=mesh.get_group())(local, matched.to_local(), labels)
                    torch.testing.assert_close(explicit, actual)
                else:
                    recipe = object.__new__(KnowledgeDistillationRecipeForNextTokenPrediction)
                    recipe.kd_ratio = ratio
                    recipe.loss_fn = MaskedCrossEntropy()
                    recipe.kd_loss_fn = KDLoss()
                    recipe._kd_loss_buffer, recipe._ce_loss_buffer = [], []
                    recipe.separate_meshes = True
                    recipe.kd_mesh_bridge = KDMeshBridge
                    recipe._current_teacher_logits = [teacher]
                    actual = recipe._make_pp_kd_loss_wrapper()(student, labels)
                    expected = (1 - ratio) * MaskedCrossEntropy()(dense, labels)
                    expected = expected + ratio * KDLoss()(dense, teacher, labels, num_batch_labels=1)
                    assert recipe._current_teacher_logits == []
                actual.backward()
                expected.backward()
                actual_local = actual.to_local() if isinstance(actual, DTensor) else actual
                torch.testing.assert_close(actual_local, expected)
                torch.testing.assert_close(local.grad, dense.grad[..., start:end], atol=1e-6, rtol=1e-5)
                torch.testing.assert_close(
                    local.detach() - 0.1 * local.grad, (dense.detach() - 0.1 * dense.grad)[..., start:end]
                )
        replicated = distribute_tensor(full_student, mesh, [Replicate()])
        assert KDMeshBridge.match_student_vocab_shard(replicated, teacher) is teacher
        assert KDMeshBridge.match_student_vocab_shard(full_student, teacher) is teacher
    finally:
        dist.destroy_process_group()


@pytest.mark.runtime_budget(30, hard_timeout=90, reason="one four-rank Gloo spawn tests real KD and PP gradients")
def test_kd_vocab_bridge_and_pp_wrapper(tmp_path: Path) -> None:
    """The bridge preserves the loss contract and torch.chunk vocabulary boundaries."""
    mp.spawn(_worker, args=((tmp_path / "rdzv").as_uri(),), nprocs=4, join=True)
