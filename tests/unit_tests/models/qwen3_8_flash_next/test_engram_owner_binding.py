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

"""Late owner-group binding of the Engram table for pipeline stages (CPU, gloo)."""

from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.qwen3_8_flash_next.config import (
    Qwen3_8_FlashNextConfig,
    Qwen3_8_FlashNextTextConfig,
)
from nemo_automodel.components.models.qwen3_8_flash_next.engram import Qwen3_8_FlashNextEngramTableConfig
from nemo_automodel.components.models.qwen3_8_flash_next.model import Qwen3_8_FlashNextForConditionalGeneration

# Over the default budget on purpose: the module spawns gloo workers that re-import torch.
pytestmark = pytest.mark.timeout(300)

TABLE = Qwen3_8_FlashNextEngramTableConfig(num_embeddings=12, embedding_dim=3, initializer_range=0.02)


def test_pending_table_holds_no_rows_until_bound() -> None:
    table = TABLE.build(process_group=None, bind_owner_group_later=True)

    assert table.owner_group_pending is True
    assert table.process_group is None
    assert tuple(table.weight.shape) == (0, 3)
    with pytest.raises(RuntimeError, match="before its owner group was bound"):
        table(torch.tensor([0, 1]))


def test_pending_table_rejects_an_explicit_group() -> None:
    with pytest.raises(ValueError, match="requires process_group=None"):
        TABLE.build(process_group=object(), bind_owner_group_later=True)  # type: ignore[arg-type]


def test_eager_table_is_unchanged() -> None:
    table = TABLE.build(process_group=None)

    assert table.owner_group_pending is False
    assert tuple(table.weight.shape) == (12, 3)
    assert table(torch.tensor([[0, 11]])).shape == (1, 2, 3)


def _stage_mesh() -> DeviceMesh:
    return DeviceMesh.from_group(dist.group.WORLD, device_type="cpu", mesh_dim_names=("dp_shard_cp",))


def _pending_table_binds_stage_mesh_worker(rank: int, world_size: int, store_path: str) -> None:
    try:
        torch.set_num_threads(1)
        dist.init_process_group("gloo", init_method=f"file://{store_path}", rank=rank, world_size=world_size)
        with torch.device("meta"):
            table = TABLE.build(process_group=None, dtype=torch.float32, bind_owner_group_later=True)
        assert tuple(table.weight.shape) == (0, 3)

        parameter = table.parallelize_weight(_stage_mesh())

        assert table.owner_group_pending is False
        assert table.owner_world_size == world_size
        assert (table.global_row_start, table.global_row_end) == (6 * rank, 6 * rank + 6)
        assert isinstance(parameter, DTensor)
        assert tuple(parameter.shape) == (12, 3)
        assert tuple(parameter.placements) == (Shard(0),)
        assert parameter.to_local().device.type == "meta"
        assert tuple(parameter.to_local().shape) == (6, 3)

        from nemo_automodel.components.checkpoint.checkpointing import to_empty_parameters_only

        to_empty_parameters_only(table, device=torch.device("cpu"))
        with torch.no_grad():
            # Row r stores the value r in every column, so a lookup returns its own id.
            table.weight.to_local().copy_(
                torch.arange(table.global_row_start, table.global_row_end, dtype=torch.float32)
                .unsqueeze(1)
                .expand(-1, 3)
            )
        table.mark_sharding_contract()
        ids = torch.tensor([[11, 0, 7], [6, 5, 1]]) if rank == 0 else torch.tensor([[2, 9, 4], [8, 3, 10]])
        values = table(ids)
        torch.testing.assert_close(values, ids.unsqueeze(-1).expand(-1, -1, 3).to(torch.float32))
        with pytest.raises(RuntimeError, match="already bound"):
            table.bind_owner_group(dist.group.WORLD)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_pending_table_binds_the_stage_mesh_and_looks_up_rows(tmp_path: Path) -> None:
    mp.spawn(
        _pending_table_binds_stage_mesh_worker,
        args=(2, str(tmp_path / "engram-pending-pg")),
        nprocs=2,
        join=True,
    )


def _tiny_ple_config() -> Qwen3_8_FlashNextConfig:
    text = Qwen3_8_FlashNextTextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        layer_types=["full_attention"],
        full_attention_interval=1,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        num_experts=2,
        num_experts_per_tok=1,
        hc_count=2,
        hc_lowrank=4,
        ple_layer_ids=[1],
        ple_embed_dim=48,
        ngram_size=3,
        heads_per_ngram=8,
        split_ngram_parts=2,
        indexer_budget=8,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=8,
        max_position_embeddings=16,
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default", "partial_rotary_factor": 1.0},
        partial_rotary_factor=1.0,
        dtype="float32",
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=1,
        tie_word_embeddings=False,
    )
    return Qwen3_8_FlashNextConfig(text_config=text, language_model_only=True, tie_word_embeddings=False)


def _model_binds_table_during_parallelization_worker(rank: int, world_size: int, store_path: str) -> None:
    try:
        torch.set_num_threads(1)
        dist.init_process_group("gloo", init_method=f"file://{store_path}", rank=rank, world_size=world_size)
        config = _tiny_ple_config()
        with torch.device("meta"):
            model = Qwen3_8_FlashNextForConditionalGeneration.from_config(
                config,
                backend=BackendConfig(linear="torch", attn="sdpa", experts="torch", dispatcher="torch"),
                engram_table_config=TABLE,
            )
        table = model.model.language_model.layers["0"].ple.ple_embedding.ngram_embedding
        # In a distributed run the owner group is not the world at construction time.
        assert table.owner_group_pending is True
        assert table.weight.numel() == 0
        with pytest.raises(RuntimeError, match="must be bound"):
            model.state_dict_adapter._local_checkpoint_shard_range()

        owned = model._nemo_prepare_model_owned_dtensors(_stage_mesh())

        assert owned == {table.weight}
        assert table.owner_group_pending is False
        assert tuple(dist.get_process_group_ranks(table.process_group)) == tuple(range(world_size))
        assert tuple(table.weight.to_local().shape) == (6, 3)
        # Two physical checkpoint shards of six rows: rank r owns shard r.
        assert model.state_dict_adapter._local_checkpoint_shard_range() == (rank, rank + 1)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_model_binds_the_table_to_its_stage_mesh(tmp_path: Path) -> None:
    mp.spawn(
        _model_binds_table_during_parallelization_worker,
        args=(2, str(tmp_path / "engram-model-pg")),
        nprocs=2,
        join=True,
    )
