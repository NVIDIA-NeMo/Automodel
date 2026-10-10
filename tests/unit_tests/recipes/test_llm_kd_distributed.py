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

import time
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate

from nemo_automodel.components.loss.kd_loss import KDLoss
from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy
from nemo_automodel.components.moe.megatron.moe_utils import MoEAuxLossAutoScaler
from nemo_automodel.recipes.llm import kd as llm_kd


class _MixedMeshModel(nn.Module):
    """Two scalar parameters that live on different device meshes.

    Args:
        expert_mesh: Mesh carrying the expert-parallel axis; the ``expert``
            parameter is a replicated ``DTensor`` of shape [1] on it.
        dense_mesh: Mesh carrying the data-parallel axis; the ``dense``
            parameter is a replicated ``DTensor`` of shape [1] on it.
    """

    def __init__(self, expert_mesh: DeviceMesh, dense_mesh: DeviceMesh) -> None:
        super().__init__()
        self.register_parameter(
            "expert",
            nn.Parameter(
                DTensor.from_local(
                    torch.tensor([10.0]),
                    expert_mesh,
                    (Replicate(),),
                    run_check=False,
                )
            ),
        )
        self.register_parameter(
            "dense",
            nn.Parameter(
                DTensor.from_local(
                    torch.tensor([20.0]),
                    dense_mesh,
                    (Replicate(),),
                    run_check=False,
                )
            ),
        )


def _set_grads(model: _MixedMeshModel, expert_grad: float, dense_grad: float) -> None:
    """Attach replicated scalar gradients, each of shape [1], on the parameter's own mesh."""
    model.expert.grad = DTensor.from_local(
        torch.tensor([expert_grad]),
        model.expert.device_mesh,
        (Replicate(),),
        run_check=False,
    )
    model.dense.grad = DTensor.from_local(
        torch.tensor([dense_grad]),
        model.dense.device_mesh,
        (Replicate(),),
        run_check=False,
    )


def _make_recipe(model, device_mesh, moe_mesh):
    """Build a bare KD recipe carrying only the state the optimizer step touches."""
    recipe = object.__new__(llm_kd.KnowledgeDistillationRecipeForNextTokenPrediction)
    recipe.model_parts = [model]
    recipe.pp_enabled = False
    recipe.device_mesh = device_mesh
    recipe.moe_mesh = moe_mesh
    recipe.optimizer = [torch.optim.SGD(model.parameters(), lr=0.1)]
    recipe.lr_scheduler = None
    recipe.checkpointer = SimpleNamespace(maybe_wait_for_staging=lambda: None)
    recipe.cfg = {}
    recipe.timestamp = time.perf_counter() - 1.0
    recipe.step_scheduler = SimpleNamespace(step=1, epoch=0)
    recipe.loss_fn = MaskedCrossEntropy(ignore_index=0)
    recipe.kd_ratio = 0.5
    recipe.kd_loss_fn = SimpleNamespace(temperature=1.0)
    recipe._ce_loss_buffer = []
    recipe._kd_loss_buffer = []
    recipe._dp_allreduce = lambda tensor, include_cp=False: tensor
    recipe._get_dp_group_size = lambda include_cp=False: 1
    recipe._get_cp_group_size = lambda: 1
    return recipe


def _tp_only_mesh(tp_size: int):
    """Mock a world mesh whose only queried axis is ``tp`` of the requested size."""
    mesh = Mock()
    mesh.mesh_dim_names = ("dp_shard_cp", "tp")
    mesh.mesh = torch.tensor([0])
    mesh.__getitem__ = Mock(return_value=Mock(size=Mock(return_value=tp_size)))
    return mesh


@pytest.fixture
def single_rank_gloo():
    """Single-rank gloo process group so DTensor collectives resolve locally."""
    assert not dist.is_initialized()
    dist.init_process_group("gloo", rank=0, world_size=1, store=dist.HashStore())
    try:
        yield
    finally:
        dist.destroy_process_group()


def test_llm_kd_non_pp_step_clips_gradients_across_device_meshes(single_rank_gloo):
    """The non-PP KD step clips mixed EP and DP DTensor gradients together."""
    expert_mesh = DeviceMesh("cpu", torch.tensor([0]), mesh_dim_names=("ep",))
    dense_mesh = DeviceMesh("cpu", torch.tensor([0]), mesh_dim_names=("dp_shard_cp",))
    device_mesh = DeviceMesh("cpu", torch.tensor([0]), mesh_dim_names=("tp",))

    model = _MixedMeshModel(expert_mesh, dense_mesh)
    _set_grads(model, expert_grad=3.0, dense_grad=4.0)

    recipe = _make_recipe(model, device_mesh, expert_mesh)
    recipe._forward_backward_step = Mock(return_value=(torch.tensor(1.0), torch.tensor(0.75), torch.tensor(0.25)))

    metrics = recipe._run_train_optim_step(
        [{"labels": torch.tensor([[0, 1], [0, 2]])}],
        max_grad_norm=2.5,
    )

    assert metrics.metrics["grad_norm"] == pytest.approx(5.0)
    assert metrics.metrics["num_label_tokens"] == 2
    assert MoEAuxLossAutoScaler.main_loss_backward_scale.item() == pytest.approx(1.0)
    torch.testing.assert_close(model.expert.to_local().detach(), torch.tensor([9.85]))
    torch.testing.assert_close(model.dense.to_local().detach(), torch.tensor([19.8]))
    assert model.expert.grad is None
    assert model.dense.grad is None


def test_llm_kd_non_pp_step_undoes_expert_tp_replication(single_rank_gloo):
    """Custom-MoE TP replicas are divided out of expert grads before clipping."""
    expert_mesh = DeviceMesh("cpu", torch.tensor([0]), mesh_dim_names=("ep",))
    dense_mesh = DeviceMesh("cpu", torch.tensor([0]), mesh_dim_names=("dp_shard_cp",))

    model = _MixedMeshModel(expert_mesh, dense_mesh)
    # The custom-MoE TP path marks the model when it replicates the token path,
    # which accumulates every expert gradient tp_size times.
    model._nemo_moe_tp_requires_replica_sync = True
    _set_grads(model, expert_grad=6.0, dense_grad=4.0)

    recipe = _make_recipe(model, _tp_only_mesh(tp_size=2), expert_mesh)
    recipe._forward_backward_step = Mock(return_value=(torch.tensor(1.0), torch.tensor(0.75), torch.tensor(0.25)))

    metrics = recipe._run_train_optim_step(
        [{"labels": torch.tensor([[1, 2]])}],
        max_grad_norm=2.5,
    )

    # Expert grad 6.0 is divided by the tp_size=2 replication factor to 3.0,
    # the dense grad is untouched, so the global norm is √(3²+4²)=5.0.
    assert metrics.metrics["grad_norm"] == pytest.approx(5.0)
    assert MoEAuxLossAutoScaler.main_loss_backward_scale.item() == pytest.approx(1.0)
    # Clipping halves both grads (2.5/5.0), then SGD(lr=0.1) applies them.
    torch.testing.assert_close(model.expert.to_local().detach(), torch.tensor([9.85]))
    torch.testing.assert_close(model.dense.to_local().detach(), torch.tensor([19.8]))


def test_llm_kd_pp_step_undoes_expert_tp_replication(single_rank_gloo):
    """The PP KD step applies the same expert TP replication correction."""
    expert_mesh = DeviceMesh("cpu", torch.tensor([0]), mesh_dim_names=("ep",))
    dense_mesh = DeviceMesh("cpu", torch.tensor([0]), mesh_dim_names=("dp_shard_cp",))
    pp_mesh = DeviceMesh("cpu", torch.tensor([0]), mesh_dim_names=("pp",))

    model = _MixedMeshModel(expert_mesh, dense_mesh)
    model._nemo_moe_tp_requires_replica_sync = True
    _set_grads(model, expert_grad=6.0, dense_grad=4.0)

    device_mesh = Mock()
    device_mesh.mesh_dim_names = ("pp", "tp")
    device_mesh.mesh = torch.tensor([0])
    device_mesh.__getitem__ = Mock(
        side_effect=lambda axis: pp_mesh if axis == "pp" else Mock(size=Mock(return_value=2))
    )

    recipe = _make_recipe(model, device_mesh, expert_mesh)
    recipe.pp_enabled = True
    recipe.pp = SimpleNamespace(pp_batch_size=2, pp_microbatch_size=1)
    recipe.dist_env = SimpleNamespace(device=torch.device("cpu"), rank=0, is_main=True)
    recipe._forward_backward_step_pp = Mock(
        side_effect=lambda i, batch, loss_buffer, num_label_tokens, num_batches: loss_buffer.append(torch.tensor(1.0))
    )

    metrics = recipe._run_train_optim_step(
        [{"labels": torch.tensor([[0, 1], [0, 2]])}],
        max_grad_norm=1.25,
    )

    # PP first divides every grad by num_label_tokens/dp_group_size = 2 (6→3, 4→2),
    # then the tp_size=2 replication factor halves the expert grad (3→1.5),
    # so the global norm is √(1.5²+2²)=2.5.
    assert metrics.metrics["grad_norm"] == pytest.approx(2.5)
    assert metrics.metrics["num_label_tokens"] == 2
    assert MoEAuxLossAutoScaler.main_loss_backward_scale.item() == pytest.approx(1.0)
    # Clipping halves both grads (1.25/2.5), then SGD(lr=0.1) applies them.
    torch.testing.assert_close(model.expert.to_local().detach(), torch.tensor([9.925]))
    torch.testing.assert_close(model.dense.to_local().detach(), torch.tensor([19.9]))


def test_llm_kd_validation_uses_loss_ignore_index(monkeypatch):
    monkeypatch.setattr(llm_kd, "ScopedRNG", lambda **kwargs: nullcontext())
    monkeypatch.setattr(llm_kd.torch.cuda, "max_memory_allocated", lambda: 0)
    recipe = object.__new__(llm_kd.KnowledgeDistillationRecipeForNextTokenPrediction)
    recipe.pp_enabled = False
    recipe.model_parts = [nn.Linear(1, 1)]
    recipe.loss_fn = MaskedCrossEntropy(ignore_index=0)
    recipe.dist_env = SimpleNamespace(device=torch.device("cpu"))
    recipe.step_scheduler = SimpleNamespace(step=1, epoch=0)
    recipe.optimizer = [SimpleNamespace(param_groups=[{"lr": 0.1}])]
    recipe._dp_allreduce = lambda tensor, include_cp=False: tensor
    seen_num_label_tokens = []

    def _forward_backward_step(idx, batch, *, num_label_tokens, num_batches, is_train=True):
        seen_num_label_tokens.append(num_label_tokens)
        return torch.tensor(1.0), torch.tensor(0.75), torch.tensor(0.25)

    recipe._forward_backward_step = _forward_backward_step

    metrics = recipe._run_validation_epoch([{"labels": torch.tensor([[0, 1], [0, 2]])}])

    assert seen_num_label_tokens == [2]
    assert metrics.metrics["num_label_tokens"] == 2
    assert metrics.metrics["val_loss"] == pytest.approx(1.0)


class _TokenLogitsModel(nn.Module):
    """Look up per-token logits, so a sharded ``input_ids`` yields sharded logits.

    Args:
        table: Tensor of shape [vocab, vocab] holding the logits of each token.
        trainable: Whether to register ``table`` as a parameter.
    """

    def __init__(self, table: torch.Tensor, *, trainable: bool) -> None:
        super().__init__()
        if trainable:
            self.table = nn.Parameter(table)
        else:
            self.register_buffer("table", table)

    def forward(self, input_ids: torch.Tensor) -> SimpleNamespace:
        """Return the logits of each input token.

        Args:
            input_ids: Tensor of shape [batch, sequence].

        Returns:
            Namespace whose ``logits`` is a tensor of shape [batch, sequence, vocab].
        """
        return SimpleNamespace(logits=self.table[input_ids])


_STUDENT_TABLE = torch.linspace(-1.0, 1.0, 16).reshape(4, 4)
_TEACHER_TABLE = torch.linspace(1.0, -1.0, 16).reshape(4, 4)


def _make_step_recipe():
    """Build a bare KD recipe carrying only the state the non-PP student step reads."""
    recipe = object.__new__(llm_kd.KnowledgeDistillationRecipeForNextTokenPrediction)
    recipe.dist_env = SimpleNamespace(device="cpu")
    recipe.device_mesh = None
    recipe.pp_enabled = False
    recipe.separate_meshes = False
    recipe.distributed_config = SimpleNamespace(defer_fsdp_grad_sync=True)
    recipe.model_parts = [_TokenLogitsModel(_STUDENT_TABLE.clone(), trainable=True)]
    recipe.teacher_model = _TokenLogitsModel(_TEACHER_TABLE, trainable=False)
    recipe._offload_teacher_model = False
    recipe.loss_fn = MaskedCrossEntropy()
    recipe.kd_loss_fn = KDLoss()
    recipe.kd_ratio = 0.5
    recipe._get_dp_group = lambda include_cp=False: None
    recipe._get_dp_group_size = lambda include_cp=False: 1
    return recipe


def _expected_step_losses(input_ids, labels, num_label_tokens):
    """Compute one KD step's losses directly from the student and teacher tables.

    Args:
        input_ids: Tensor of shape [batch, sequence] holding this rank's tokens.
        labels: Tensor of shape [batch, sequence] aligned with ``input_ids``.
        num_label_tokens: Valid-label count across the full sequence.

    Returns:
        Tuple of scalar tensors containing mixed, KL, and CE loss.
    """
    student_logits = _STUDENT_TABLE[input_ids]
    ce = MaskedCrossEntropy()(student_logits, labels, num_label_tokens=num_label_tokens)
    kd = KDLoss()(student_logits, _TEACHER_TABLE[input_ids], labels, num_batch_labels=num_label_tokens)
    return 0.5 * ce + 0.5 * kd, kd, ce


def test_llm_kd_non_pp_step_without_cp_uses_full_sequence_labels():
    """At cp=1 the real sharder is the identity and the step sees every label."""
    batch = {"input_ids": torch.tensor([[1, 2, 3]]), "labels": torch.tensor([[2, 3, -100]])}

    losses = _make_step_recipe()._forward_backward_step(0, batch, num_label_tokens=2, num_batches=1)

    for actual, expected in zip(losses, _expected_step_losses(batch["input_ids"], batch["labels"], 2)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_llm_kd_non_pp_step_shards_labels_with_the_batch_under_cp(monkeypatch):
    """Under CP the step leaves labels in the batch for the sharder and uses its local labels."""

    def load_balanced_sharder(model, mesh, batch, **kwargs):
        """Mimic rank 0 of the cp=2 load-balanced sharder on a length-3 batch.

        Like ``shard_batch_load_balanced``, it requires ``batch["labels"]``, pads
        the sequence to ``2 * cp`` tokens (labels with -100) and keeps rank 0's
        head and tail chunks, positions 0 and 3.

        Args:
            model: Unused student model.
            mesh: Unused device mesh.
            batch: Mapping containing ``input_ids`` and ``labels`` as tensors of
                shape [batch, sequence].
            **kwargs: Unused context-parallel options.

        Returns:
            Object whose shard operation shards the batch in place.
        """
        del model, mesh, kwargs

        def shard(actual_batch):
            """Replace ``input_ids`` and ``labels`` with rank 0's padded local tokens.

            Args:
                actual_batch: Mapping containing ``input_ids`` and ``labels`` as
                    tensors of shape [batch, 3].

            Returns:
                Context factory and the same mapping, now holding tensors of shape [batch, 2].
            """
            for key, fill in (("input_ids", 0), ("labels", -100)):
                actual_batch[key] = nn.functional.pad(actual_batch[key], (0, 1), value=fill)[:, [0, 3]]
            return nullcontext, actual_batch

        return SimpleNamespace(shard=shard)

    monkeypatch.setattr(llm_kd, "ContextParallelSharder", load_balanced_sharder)
    batch = {"input_ids": torch.tensor([[1, 2, 3]]), "labels": torch.tensor([[2, 3, 1]])}

    losses = _make_step_recipe()._forward_backward_step(0, batch, num_label_tokens=3, num_batches=1)

    local_input_ids, local_labels = torch.tensor([[1, 0]]), torch.tensor([[2, -100]])
    for actual, expected in zip(losses, _expected_step_losses(local_input_ids, local_labels, 3)):
        torch.testing.assert_close(actual, expected)
