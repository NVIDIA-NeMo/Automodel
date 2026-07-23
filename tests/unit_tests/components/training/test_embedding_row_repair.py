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

import logging
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist

from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.components.training.embedding_row_repair import (
    EmbeddingRowRepairConfig,
    repair_input_embedding_rows,
)
from nemo_automodel.recipes._typed_config import RecipeConfig


class _TinyLM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(6, 4)
        self.lm_head = torch.nn.Linear(4, 6, bias=False)

    def get_input_embeddings(self) -> torch.nn.Embedding:
        return self.embed_tokens

    def get_output_embeddings(self) -> torch.nn.Linear:
        return self.lm_head


def _make_damaged_model() -> _TinyLM:
    model = _TinyLM()
    with torch.no_grad():
        model.embed_tokens.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 2.0, 0.0, 0.0],
                    [1.0e-8, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 3.0, 0.0],
                    [0.0, 0.0, 0.0, 4.0],
                ]
            )
        )
        model.lm_head.weight.copy_(
            torch.tensor(
                [
                    [1.0, 1.0, 0.0, 0.0],
                    [1.0, 2.0, 3.0, 4.0],
                    [0.0, 1.0, 1.0, 0.0],
                    [4.0, 3.0, 2.0, 1.0],
                    [1.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0, 1.0],
                ]
            )
        )
    return model


@pytest.fixture
def damaged_model():
    return _make_damaged_model()


def test_repair_identifies_rows_and_restores_healthy_scale(damaged_model, caplog):
    before = damaged_model.embed_tokens.weight.detach().clone()
    source = damaged_model.lm_head.weight.detach().clone()
    expected_target = torch.tensor([1.0, 2.0, 3.0, 4.0]).square().mean().sqrt()

    with caplog.at_level(logging.WARNING):
        report = repair_input_embedding_rows(damaged_model, min_norm=1.0e-4)

    assert report.repaired_row_ids == (1, 3)
    assert report.min_norm_before == 0.0
    assert report.target_norm == pytest.approx(expected_target.item())
    torch.testing.assert_close(damaged_model.embed_tokens.weight[[0, 2, 4, 5]], before[[0, 2, 4, 5]])

    repaired = damaged_model.embed_tokens.weight[[1, 3]]
    repaired_norms = torch.linalg.vector_norm(repaired, dim=1)
    torch.testing.assert_close(repaired_norms, expected_target.expand_as(repaired_norms))
    torch.testing.assert_close(
        torch.nn.functional.normalize(repaired, dim=1),
        torch.nn.functional.normalize(source[[1, 3]], dim=1),
    )
    assert "ids=[1, 3]" in caplog.text


def test_repair_is_noop_for_healthy_embeddings(damaged_model):
    with torch.no_grad():
        damaged_model.embed_tokens.weight[1].fill_(1.0)
        damaged_model.embed_tokens.weight[3].fill_(1.0)
    before = damaged_model.embed_tokens.weight.detach().clone()

    report = repair_input_embedding_rows(damaged_model, min_norm=1.0e-4)

    assert report.repaired_row_ids == ()
    assert report.target_norm is None
    torch.testing.assert_close(damaged_model.embed_tokens.weight, before)


def test_repair_supports_bfloat16_weights(damaged_model):
    damaged_model.to(torch.bfloat16)

    report = repair_input_embedding_rows(damaged_model, min_norm=1.0e-4)

    assert report.repaired_row_ids == (1, 3)
    repaired_norms = torch.linalg.vector_norm(damaged_model.embed_tokens.weight[[1, 3]].float(), dim=1)
    torch.testing.assert_close(
        repaired_norms,
        torch.full_like(repaired_norms, report.target_norm),
        rtol=5.0e-3,
        atol=5.0e-3,
    )


def test_repair_aborts_before_mutation_when_too_many_rows_are_damaged(damaged_model):
    before = damaged_model.embed_tokens.weight.detach().clone()

    with pytest.raises(ValueError, match="Refusing to repair 2 input embedding rows"):
        repair_input_embedding_rows(damaged_model, min_norm=1.0e-4, max_rows=1)

    torch.testing.assert_close(damaged_model.embed_tokens.weight, before)


def test_repair_rejects_damaged_output_source(damaged_model):
    before = damaged_model.embed_tokens.weight.detach().clone()
    with torch.no_grad():
        damaged_model.lm_head.weight[1].zero_()

    with pytest.raises(ValueError, match=r"output rows are also damaged: \[1\]"):
        repair_input_embedding_rows(damaged_model, min_norm=1.0e-4)

    torch.testing.assert_close(damaged_model.embed_tokens.weight, before)


def test_config_validation_and_recipe_typed_view():
    with pytest.raises(ValueError, match="min_norm"):
        EmbeddingRowRepairConfig(min_norm=-1.0)
    with pytest.raises(ValueError, match="max_rows"):
        EmbeddingRowRepairConfig(max_rows=0)

    recipe = RecipeConfig(ConfigNode({"embedding_row_repair": {"min_norm": 1.0e-4, "max_rows": 8}}))
    assert recipe.embedding_row_repair == EmbeddingRowRepairConfig(min_norm=1.0e-4, max_rows=8)
    assert EmbeddingRowRepairConfig(enabled=False).apply(_TinyLM()) is None


def test_repair_supports_single_rank_sharded_dtensors(damaged_model, single_rank_gloo):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor

    mesh = init_device_mesh("cpu", (1,))
    damaged_model.embed_tokens.weight = torch.nn.Parameter(
        distribute_tensor(damaged_model.embed_tokens.weight.detach(), mesh, [Shard(0)])
    )
    damaged_model.lm_head.weight = torch.nn.Parameter(
        distribute_tensor(damaged_model.lm_head.weight.detach(), mesh, [Shard(0)])
    )

    report = repair_input_embedding_rows(damaged_model, min_norm=1.0e-4)

    assert report.repaired_row_ids == (1, 3)
    repaired_norms = torch.linalg.vector_norm(damaged_model.embed_tokens.weight.full_tensor()[[1, 3]], dim=1)
    assert torch.all(repaired_norms > 1.0e-4)


def _run_two_rank_sharded_dtensor_repair(rank: int, world_size: int, init_file: str, shard_dim: int) -> None:
    """Exercise embedding repair with real two-rank DTensor collectives."""
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor

    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        mesh = init_device_mesh("cpu", (world_size,))
        model = _make_damaged_model()
        before = model.embed_tokens.weight.detach().clone()
        source = model.lm_head.weight.detach().clone()
        expected_target = torch.tensor([1.0, 2.0, 3.0, 4.0]).square().mean().sqrt()

        model.embed_tokens.weight = torch.nn.Parameter(
            distribute_tensor(model.embed_tokens.weight.detach(), mesh, [Shard(shard_dim)])
        )
        model.lm_head.weight = torch.nn.Parameter(
            distribute_tensor(model.lm_head.weight.detach(), mesh, [Shard(shard_dim)])
        )

        report = repair_input_embedding_rows(model, min_norm=1.0e-4)

        assert report.repaired_row_ids == (1, 3)
        assert report.min_norm_before == 0.0
        assert report.target_norm == pytest.approx(expected_target.item())

        repaired_full = model.embed_tokens.weight.full_tensor()
        torch.testing.assert_close(repaired_full[[0, 2, 4, 5]], before[[0, 2, 4, 5]])
        repaired_rows = repaired_full[[1, 3]]
        repaired_norms = torch.linalg.vector_norm(repaired_rows, dim=1)
        torch.testing.assert_close(repaired_norms, expected_target.expand_as(repaired_norms))
        torch.testing.assert_close(
            torch.nn.functional.normalize(repaired_rows, dim=1),
            torch.nn.functional.normalize(source[[1, 3]], dim=1),
        )
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("shard_dim", [0, 1])
def test_repair_supports_two_rank_sharded_dtensors(tmp_path, shard_dim):
    """Two-rank CPU DTensor repair matches the unsharded row IDs and target norm."""
    torch.multiprocessing.spawn(
        _run_two_rank_sharded_dtensor_repair,
        args=(2, str(tmp_path / f"repair_pg_shard_{shard_dim}"), shard_dim),
        nprocs=2,
        join=True,
    )


@pytest.fixture
def single_rank_gloo():
    if dist.is_initialized():
        pytest.skip("a process group is already initialized in this session")
    dist.init_process_group(backend="gloo", rank=0, world_size=1, store=dist.HashStore())
    try:
        yield
    finally:
        dist.destroy_process_group()
