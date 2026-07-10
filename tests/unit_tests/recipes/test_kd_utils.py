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

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nemo_automodel.components.distributed.config import DDPConfig, FSDP2Config
from nemo_automodel.components.loss.kd_loss import KDLoss
from nemo_automodel.recipes import kd_utils


def test_materialize_teacher_logits_unshards_cp_and_removes_padding(monkeypatch):
    cp_mesh = SimpleNamespace(size=lambda: 2)

    class Mesh:
        mesh_dim_names = ("cp",)

        def __getitem__(self, name):
            assert name == "cp"
            return cp_mesh

    local = torch.tensor([[[1.0], [2.0]]])

    def unshard(mesh, tensor, *, seq_dim):
        """Expand local ``[B=1, S_local=2, V=1]`` logits to full sequence."""
        assert mesh is cp_mesh
        assert tensor is local
        assert seq_dim == 1
        return torch.tensor([[[1.0], [2.0], [3.0], [0.0]]])

    monkeypatch.setattr(kd_utils, "unshard_context_parallel_tensor", unshard)

    result = kd_utils.materialize_teacher_logits(local, device_mesh=Mesh(), sequence_length=3)

    assert torch.equal(result, torch.tensor([[[1.0], [2.0], [3.0]]]))


def test_shared_kd_setup_keeps_existing_world(monkeypatch):
    shared = SimpleNamespace(strategy_config=FSDP2Config())

    def build(cfg, **kwargs):
        return shared

    monkeypatch.setattr(kd_utils, "create_distributed_setup_from_config", build)

    setups = kd_utils.create_kd_distributed_setups({"distributed": {}}, world_size=4)

    assert setups.student is shared
    assert setups.teacher is shared
    assert setups.student_ranks == (0, 1, 2, 3)
    assert setups.teacher_ranks == (0, 1, 2, 3)
    assert not setups.separate


def test_separate_kd_setup_assigns_contiguous_disjoint_ranks(monkeypatch):
    calls = []

    def build(cfg, **kwargs):
        calls.append((cfg, kwargs))
        return SimpleNamespace(strategy_config=FSDP2Config())

    monkeypatch.setattr(kd_utils, "create_distributed_setup_from_config", build)
    cfg = {
        "separate_meshes": True,
        "distributed": {"strategy": "fsdp2", "dp_size": 2},
        "teacher_distributed": {"strategy": "fsdp2", "dp_size": 1, "tp_size": 2},
    }

    setups = kd_utils.create_kd_distributed_setups(cfg, world_size=4)

    assert setups.student_ranks == (0, 1)
    assert setups.teacher_ranks == (2, 3)
    assert setups.separate
    assert calls[0][1] == {"world_size": 2, "ranks": (0, 1)}
    assert calls[1][1] == {"world_size": 2, "ranks": (2, 3)}


@pytest.mark.parametrize(
    ("cfg", "message"),
    [
        (
            {"distributed": {}, "teacher_distributed": {"dp_size": 1}},
            "teacher_distributed requires separate_meshes=true",
        ),
        (
            {"separate_meshes": True, "distributed": {"dp_size": 2}},
            "requires a teacher_distributed section",
        ),
        (
            {
                "separate_meshes": True,
                "distributed": {"tp_size": 2},
                "teacher_distributed": {"dp_size": 2},
            },
            "distributed.dp_size must be set",
        ),
        (
            {
                "separate_meshes": True,
                "distributed": {"dp_size": 2},
                "teacher_distributed": {"dp_size": 1},
            },
            "student=2.*teacher=1 != world_size=4",
        ),
    ],
)
def test_kd_setup_rejects_ambiguous_rank_splits(cfg, message):
    with pytest.raises(ValueError, match=message):
        kd_utils.create_kd_distributed_setups(cfg, world_size=4)


def test_separate_kd_setup_rejects_ddp(monkeypatch):
    monkeypatch.setattr(
        kd_utils,
        "create_distributed_setup_from_config",
        lambda cfg, **kwargs: SimpleNamespace(strategy_config=DDPConfig()),
    )
    cfg = {
        "separate_meshes": True,
        "distributed": {"strategy": "ddp", "dp_size": 2},
        "teacher_distributed": {"strategy": "fsdp2", "dp_size": 2},
    }

    with pytest.raises(ValueError, match="DDP is not supported"):
        kd_utils.create_kd_distributed_setups(cfg, world_size=4)


def _teacher_logits(input_ids: torch.Tensor) -> torch.Tensor:
    """Return deterministic logits ``[B, S, V=3]`` for ids ``[B, S]``."""
    values = input_ids.float()
    return torch.stack((values * 0.5, values * -0.25 + 1.0, values * 0.125 - 0.5), dim=-1)


def _run_kd_bridge_worker(rank: int, world_size: int, init_file: str) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        cfg = {
            "separate_meshes": True,
            "distributed": {"strategy": "fsdp2", "dp_size": 2},
            "teacher_distributed": {"strategy": "fsdp2", "dp_size": 1, "tp_size": 2},
        }
        setups = kd_utils.create_kd_distributed_setups(cfg, world_size=world_size)
        bridge = kd_utils.KDMeshBridge(setups, device=torch.device("cpu"))
        batch = None
        if bridge.is_student:
            input_ids = torch.tensor([[rank + 1, rank + 2]], dtype=torch.long)
            batch = {"input_ids": input_ids, "labels": input_ids.clone()}

        bridge.broadcast_command(kd_utils.RUN_TEACHER if bridge.is_student else None)
        received_teacher_logits = None
        for wave in range(bridge.num_waves):
            teacher_batch = bridge.send_batch(wave, batch)
            logits = _teacher_logits(teacher_batch["input_ids"]) if bridge.is_teacher else None
            received = bridge.send_logits(wave, logits)
            if received is not None:
                received_teacher_logits = received

        if bridge.is_student:
            assert received_teacher_logits is not None
            expected = _teacher_logits(batch["input_ids"])
            torch.testing.assert_close(received_teacher_logits, expected)
            scale = torch.tensor(0.75, requires_grad=True)
            loss = KDLoss()(expected.detach() * scale, received_teacher_logits, batch["labels"])
            loss.backward()
            assert torch.isfinite(loss)
            assert scale.grad is not None and torch.isfinite(scale.grad)

            from nemo_automodel.components.checkpoint.checkpointing import Checkpointer
            from nemo_automodel.components.checkpoint.config import CheckpointingConfig

            checkpoint_config = CheckpointingConfig(enabled=False, is_async=True)
            checkpointer = Checkpointer(
                checkpoint_config,
                dp_rank=rank,
                tp_rank=0,
                pp_rank=0,
                process_group=bridge.student_group,
            )
            checkpointer.close()
        bridge.synchronize()
    finally:
        dist.destroy_process_group()


def test_kd_mesh_bridge_routes_two_student_replicas_through_teacher_tp2(tmp_path):
    """Exercise explicit subset meshes and bridge collectives on four CPU ranks."""
    mp.spawn(_run_kd_bridge_worker, args=(4, str(tmp_path / "process_group")), nprocs=4, join=True)


def _run_subset_ep_mesh_worker(rank: int, world_size: int, init_file: str) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        cfg = {
            "separate_meshes": True,
            "distributed": {"strategy": "fsdp2", "dp_size": 2},
            "teacher_distributed": {
                "strategy": "fsdp2",
                "dp_size": 4,
                "pp_size": 2,
                "ep_size": 2,
                "pipeline": {},
            },
        }
        setups = kd_utils.create_kd_distributed_setups(cfg, world_size=world_size)
        if rank in setups.teacher_ranks:
            device_mesh = setups.teacher.mesh_context.device_mesh
            moe_mesh = setups.teacher.mesh_context.moe_mesh
            assert device_mesh["pp"].size() == 2
            assert moe_mesh is not None
            assert moe_mesh["ep"].size() == 2
            assert set(dist.get_process_group_ranks(moe_mesh["ep"].get_group())).issubset(setups.teacher_ranks)
    finally:
        dist.destroy_process_group()


def test_separate_kd_setup_builds_teacher_ep_mesh_on_rank_subset(tmp_path):
    """Teacher PP and EP groups may occupy ranks disjoint from the student mesh."""
    mp.spawn(_run_subset_ep_mesh_worker, args=(10, str(tmp_path / "ep_process_group")), nprocs=10, join=True)


def test_pp_kd_wrapper_consumes_teacher_microbatches_in_order():
    """Shared-mesh PP uses the matching captured teacher logits for each microbatch."""
    from nemo_automodel.recipes.llm import kd as llm_kd

    recipe = object.__new__(llm_kd.KnowledgeDistillationRecipeForNextTokenPrediction)
    recipe.kd_ratio = 1.0
    recipe.kd_loss_fn = KDLoss()
    recipe._kd_loss_buffer = []
    recipe._ce_loss_buffer = []
    recipe.separate_meshes = False

    teacher_logits = [
        torch.tensor([[[1.0, 0.0, -1.0], [0.5, 0.0, -0.5]]]),
        torch.tensor([[[-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]]]),
    ]
    recipe._current_teacher_logits = list(teacher_logits)
    loss_fn = recipe._make_pp_kd_loss_wrapper()
    labels = torch.tensor([[0, 1]])
    student_logits = torch.zeros(1, 2, 3, requires_grad=True)

    first = loss_fn(student_logits, labels)
    second = loss_fn(student_logits, labels)

    torch.testing.assert_close(first, KDLoss()(student_logits, teacher_logits[0], labels, num_batch_labels=1))
    torch.testing.assert_close(second, KDLoss()(student_logits, teacher_logits[1], labels, num_batch_labels=1))
    assert recipe._current_teacher_logits == []
