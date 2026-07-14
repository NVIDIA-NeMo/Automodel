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

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml

from nemo_automodel.components.checkpoint.config import CheckpointingConfig
from nemo_automodel.components.distributed.config import DDPConfig, FSDP2Config
from nemo_automodel.components.loss.kd_loss import KDLoss
from nemo_automodel.recipes import kd_utils
from tests.functional_tests.llm_pretrain_and_kd import kd_separate_mesh_test_utils
from tests.functional_tests.llm_pretrain_and_kd.compare_kd_sep_mesh_losses import _compare_pair
from tests.functional_tests.llm_pretrain_and_kd.kd_separate_mesh_test_utils import TinyKDDataset

_REPO_ROOT = Path(__file__).resolve().parents[3]
_KD_FP32_MASTER_YAMLS = (
    "examples/llm_kd/llama3_2/llama3_2_1b_kd_separate_mesh_teacher_cp2.yaml",
    "examples/llm_kd/llama3_2/llama3_2_1b_kd_separate_mesh_teacher_pp2.yaml",
    "examples/llm_kd/llama3_2/llama3_2_1b_kd_separate_mesh_teacher_tp2.yaml",
    "examples/vlm_kd/qwen3_5/qwen3_5_vl_4b_kd_separate_mesh_teacher_cp2.yaml",
    "examples/vlm_kd/qwen3_5/qwen3_5_vl_4b_kd_separate_mesh_teacher_dp2.yaml",
    "examples/vlm_kd/qwen3_5/qwen3_5_vl_4b_kd_separate_mesh_teacher_tp2.yaml",
    "tests/functional_tests/llm_pretrain_and_kd/kd_separate_mesh.yaml",
    "tests/functional_tests/llm_pretrain_and_kd/kd_sep_mesh_gemma_dense.yaml",
    "tests/functional_tests/llm_pretrain_and_kd/kd_sep_mesh_gemma_moe.yaml",
)


def test_tiny_kd_dataset_requests_flat_chat_template_token_ids():
    class Tokenizer:
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, return_dict):
            assert messages == [
                {"role": "user", "content": "Explain knowledge distillation and distributed validation."}
            ]
            assert tokenize is True
            assert add_generation_prompt is True
            assert return_dict is False
            return [10, 11, 12, 13]

        def encode(self, text, *, add_special_tokens):
            assert text.startswith("Knowledge distillation")
            assert add_special_tokens is False
            return list(range(20, 84))

    dataset = TinyKDDataset(Tokenizer(), num_samples=1, seq_length=8, use_chat_template=True)

    assert dataset[0]["input_ids"] == [10, 11, 12, 13, 20, 21, 22, 23]
    assert dataset[0]["labels"] == [-100, -100, -100, 20, 21, 22, 23, 24]


def test_tiny_kd_dataset_preserves_non_chat_first_label_mask():
    class Tokenizer:
        def encode(self, text, *, add_special_tokens):
            assert text.startswith("Knowledge distillation")
            assert add_special_tokens is True
            return list(range(64))

    dataset = TinyKDDataset(Tokenizer(), num_samples=1, seq_length=8)

    assert dataset[0]["input_ids"] == list(range(8))
    assert dataset[0]["labels"] == [-100, 2, 3, 4, 5, 6, 7, 8]


def test_tiny_kd_sft_dataset_masks_generation_prompt():
    class Tokenizer:
        eos_token_id = 2

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, return_dict):
            assert messages == [{"role": "user", "content": "What is KD?"}]
            assert tokenize is True
            assert add_generation_prompt is True
            assert return_dict is False
            return [10, 11, 12, 13]

        def encode(self, text, *, add_special_tokens):
            assert text == "KD transfers teacher behavior to a student."
            assert add_special_tokens is False
            return [20, 21, 22, 23, 24, 25, 26, 27]

    dataset = kd_separate_mesh_test_utils.make_tiny_kd_sft_dataset(
        tokenizer=Tokenizer(),
        prompt="What is KD?",
        completion="KD transfers teacher behavior to a student.",
        num_samples=2,
        seq_length=8,
    )

    assert dataset[0]["input_ids"] == [10, 11, 12, 13, 20, 21, 22, 23]
    assert dataset[0]["labels"] == [-100, -100, -100, 20, 21, 22, 23, 24]
    assert dataset[0]["attention_mask"] == [1] * 8
    dataset[0]["labels"][3] = -100
    assert dataset[1]["labels"][3] == 20


def test_kd_yamls_use_fp32_optimizer_master_weights():
    for relative_path in _KD_FP32_MASTER_YAMLS:
        cfg = yaml.safe_load((_REPO_ROOT / relative_path).read_text())

        optimizer = cfg["optimizer"]
        assert optimizer["_target_"] == "transformer_engine.pytorch.optimizers.fused_adam.FusedAdam"
        assert optimizer["master_weights"] is True
        assert optimizer["master_weight_dtype"] == "torch.float32"

        assert cfg["model"]["torch_dtype"] == "bfloat16"
        assert cfg["teacher_model"]["torch_dtype"] == "bfloat16"


def test_loss_comparator_rejects_mismatched_kd_settings():
    baseline = {
        "step": 0,
        "loss": 1.0,
        "ce_loss": 1.0,
        "kd_loss": 1.0,
        "grad_norm": 1.0,
        "kd_ratio": 0.5,
        "temperature": 1.0,
    }
    candidate = baseline | {"temperature": 2.0}

    with pytest.raises(ValueError, match="mismatched run settings"):
        _compare_pair("teacher_tp2", [baseline], [candidate], 0.01, 0.01)


def test_materialize_teacher_logits_unshards_cp_and_removes_padding(monkeypatch):
    cp_mesh = SimpleNamespace(size=lambda: 2)

    class Mesh:
        mesh_dim_names = ("cp",)

        def __getitem__(self, name):
            assert name == "cp"
            return cp_mesh

    local = torch.tensor([[[1.0], [2.0]]])

    def unshard(mesh, tensor, *, seq_dim):
        """Expand local logits to the full sequence.

        Args:
            mesh: Mock context-parallel mesh.
            tensor: Tensor of shape ``[batch=1, local_sequence=2, vocab=1]``
                containing local logits.
            seq_dim: Sequence axis, which must be one.

        Returns:
            Tensor of shape ``[batch=1, sequence=4, vocab=1]`` containing full
            logits.
        """
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
    """Return deterministic teacher logits.

    Args:
        input_ids: Tensor of shape ``[batch, sequence]`` containing token ids.

    Returns:
        Tensor of shape ``[batch, sequence, vocab]`` containing logits with
        ``vocab = 3``.
    """
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
        checkpointer = None
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

            checkpoint_config = CheckpointingConfig(enabled=False, is_async=True)

            checkpointer = Checkpointer(
                checkpoint_config,
                dp_rank=rank,
                tp_rank=0,
                pp_rank=0,
                process_group=bridge.student_group,
            )
        bridge.synchronize()
        # Barrier completion only guarantees every rank entered the collective;
        # use the default group to ensure teacher ranks have returned before
        # students destroy their async checkpoint groups.
        dist.barrier()
        if checkpointer is not None:
            checkpointer.close()
        # Hold teacher ranks until the student-only checkpoint groups are closed
        # before tearing down the default process group.
        dist.barrier()
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
