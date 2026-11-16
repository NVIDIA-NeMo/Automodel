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

"""CPU tests for the frozen FP8 Qwen3.8-Flash-Next PLE table with its global scale."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import NamedTuple

import pytest
import torch
import torch.multiprocessing as mp
from huggingface_hub import constants as huggingface_constants
from safetensors.torch import save_file
from torch import nn

from nemo_automodel.components.checkpoint.checkpointing import to_empty_parameters_only
from nemo_automodel.components.checkpoint.config import CheckpointingConfig
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.qwen3_8_flash_next import engram as engram_module
from nemo_automodel.components.models.qwen3_8_flash_next.config import (
    Qwen3_8_FlashNextConfig,
    Qwen3_8_FlashNextTextConfig,
)
from nemo_automodel.components.models.qwen3_8_flash_next.engram import (
    Qwen3_8_FlashNextEngramTableConfig,
    Qwen3_8_FlashNextHostNGramTable,
    Qwen3_8_FlashNextNGramEmbedding,
    Qwen3_8_FlashNextPLELayer,
)
from nemo_automodel.components.models.qwen3_8_flash_next.model import Qwen3_8_FlashNextForConditionalGeneration
from nemo_automodel.components.models.qwen3_8_flash_next.state_dict_adapter import Qwen3_8_FlashNextStateDictAdapter
from nemo_automodel.components.moe.layers import MoEConfig

# A representative global scale, stored as BF16 [1].
_CHECKPOINT_SCALE = 0.00019931793
_PREFIX = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
_SCALE_KEY = f"{_PREFIX}.weight_scale"
_SPLIT_PARTS = 4
_ROWS = 72
_DIM = 4
_FP8 = "float8_e4m3fn"


class _Checkpoint(NamedTuple):
    path: Path
    table: torch.Tensor
    scale: torch.Tensor


def _finite_fp8_table(rows: int, dim: int, seed: int = 0) -> torch.Tensor:
    """Return an FP8 table of finite values, with every finite E4M3 byte pattern when it has room.

    Returns:
        Tensor of shape ``[rows, dim]`` and dtype ``float8_e4m3fn``.
    """
    finite_bytes = torch.tensor([value for value in range(256) if value & 0x7F != 0x7F], dtype=torch.uint8)
    generator = torch.Generator().manual_seed(seed)
    flat = finite_bytes[torch.randint(finite_bytes.numel(), (rows * dim,), generator=generator)]
    covered = min(flat.numel(), finite_bytes.numel())
    flat[:covered] = finite_bytes[:covered]
    return flat.view(rows, dim).view(torch.float8_e4m3fn)


def _write_checkpoint(
    path: Path,
    *,
    table: torch.Tensor,
    scale: torch.Tensor | None,
    drop_shards: tuple[int, ...] = (),
    prefix: str = _PREFIX,
    extra_tensors: dict[str, torch.Tensor] | None = None,
) -> Path:
    """Write a two-file sharded PLE table, an optional scale file, and an index.

    Args:
        path: Checkpoint directory to create.
        table: Tensor of shape ``[rows, dim]`` split row-wise into ``_SPLIT_PARTS`` shards.
        scale: Optional tensor of shape ``[1]`` stored under the scale key.
        drop_shards: Shard indices to leave out of the checkpoint.
        prefix: Checkpoint name prefix of the PLE table.
        extra_tensors: Other tensors of the checkpoint; ``lm_head.weight`` when omitted.
    """
    path.mkdir(parents=True)
    weight_map: dict[str, str] = {}
    files: dict[str, dict[str, torch.Tensor]] = {}
    for shard_idx, shard in enumerate(table.chunk(_SPLIT_PARTS, dim=0)):
        if shard_idx in drop_shards:
            continue
        file_name = f"model-ple-{shard_idx // 2:03d}.safetensors"
        key = f"{prefix}.shard_{shard_idx}.weight"
        files.setdefault(file_name, {})[key] = shard.contiguous()
        weight_map[key] = file_name
    if scale is not None:
        files["model-ple-scale.safetensors"] = {f"{prefix}.weight_scale": scale}
        weight_map[f"{prefix}.weight_scale"] = "model-ple-scale.safetensors"
    if extra_tensors is None:
        extra_tensors = {"lm_head.weight": torch.zeros(2, 3, dtype=torch.bfloat16)}
    files["model-embed.safetensors"] = dict(extra_tensors)
    weight_map.update(dict.fromkeys(extra_tensors, "model-embed.safetensors"))
    for file_name, tensors in files.items():
        save_file(tensors, str(path / file_name))
    (path / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    return path


@pytest.fixture
def fp8_checkpoint(tmp_path: Path) -> _Checkpoint:
    table = _finite_fp8_table(_ROWS, _DIM)
    scale = torch.tensor([_CHECKPOINT_SCALE], dtype=torch.bfloat16)
    return _Checkpoint(_write_checkpoint(tmp_path / "ckpt", table=table, scale=scale), table, scale)


def _text_config(ple_embedding_dtype: str | None) -> Qwen3_8_FlashNextTextConfig:
    return Qwen3_8_FlashNextTextConfig(
        vocab_size=32,
        hidden_size=3,
        intermediate_size=4,
        num_hidden_layers=2,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=3,
        moe_intermediate_size=2,
        shared_expert_intermediate_size=2,
        num_experts=2,
        num_experts_per_tok=1,
        hc_count=2,
        hc_lowrank=1,
        ple_layer_ids=[2],
        ple_embed_dim=3,
        split_ngram_parts=_SPLIT_PARTS,
        ple_embedding_dtype=ple_embedding_dtype,
        dtype="float32",
    )


def _moe_config() -> MoEConfig:
    return MoEConfig(
        dim=3,
        inter_dim=4,
        moe_inter_dim=2,
        n_routed_experts=2,
        n_shared_experts=1,
        n_activated_experts=1,
        n_expert_groups=0,
        n_limited_groups=0,
        train_gate=True,
        gate_bias_update_factor=0.0,
        score_func="softmax",
        route_scale=1.0,
        aux_loss_coeff=0.0,
        norm_topk_prob=True,
        expert_bias=False,
        router_bias=False,
        expert_activation="swiglu",
        softmax_before_topk=True,
        shared_expert_gate=True,
        shared_expert_inter_dim=2,
    )


def _backend() -> BackendConfig:
    return BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        dispatcher="torch",
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
    )


def _host_table(host_dir: Path, rows: int = _ROWS) -> Qwen3_8_FlashNextHostNGramTable:
    host_dir.mkdir(exist_ok=True)
    return Qwen3_8_FlashNextHostNGramTable(num_embeddings=rows, embedding_dim=_DIM, host_dir=host_dir)


def _adapter(
    checkpoint: Path | None,
    table: nn.Module | Qwen3_8_FlashNextHostNGramTable,
    ple_embedding_dtype: str | None,
) -> Qwen3_8_FlashNextStateDictAdapter:
    return Qwen3_8_FlashNextStateDictAdapter(
        _text_config(ple_embedding_dtype),
        _moe_config(),
        _backend(),
        table,
        dtype=torch.float32,
        pretrained_model_name_or_path=None if checkpoint is None else str(checkpoint),
    )


def _load_fp8(checkpoint: Path, host_dir: Path) -> Qwen3_8_FlashNextHostNGramTable:
    table = _host_table(host_dir)
    assert _adapter(checkpoint, table, _FP8).from_hf({}) == {}
    return table


def _bits(values: torch.Tensor) -> torch.Tensor:
    """Reinterpret a BF16 tensor of any shape as int16 for bitwise comparison."""
    return values.view(torch.int16)


def test_fp8_lookup_is_bitwise_equal_to_engine_order(fp8_checkpoint: _Checkpoint, tmp_path: Path) -> None:
    table = _load_fp8(fp8_checkpoint.path, tmp_path / "shm")
    ids = torch.cat(
        [torch.arange(_ROWS).flip(0), torch.randint(_ROWS, (48,), generator=torch.Generator().manual_seed(1))]
    )
    ids = ids.view(2, 4, 15)

    actual = table(ids)

    assert actual.dtype == torch.bfloat16
    assert actual.shape == (2, 4, 15, _DIM)
    assert not actual.requires_grad
    # The engine: gather FP8 -> cast to BF16 -> multiply by the BF16 [1] scale.
    engine_order = fp8_checkpoint.table[ids].to(torch.bfloat16) * fp8_checkpoint.scale
    assert torch.equal(_bits(actual), _bits(engine_order))
    # Independent invariant: the exact product, rounded once to BF16.
    exact = fp8_checkpoint.table.to(torch.float64)[ids] * fp8_checkpoint.scale.to(torch.float64)
    assert torch.equal(_bits(actual), _bits(exact.to(torch.bfloat16)))
    # The unscaled rows (what the BF16 path would load) are about 1 / scale = 5,000x larger.
    assert actual.abs().max() <= 448 * _CHECKPOINT_SCALE * 1.01


def test_fp8_table_through_ngram_hashing(tmp_path: Path) -> None:
    rows = 36
    table_values = _finite_fp8_table(rows, _DIM, seed=3)
    scale = torch.tensor([_CHECKPOINT_SCALE], dtype=torch.bfloat16)
    checkpoint = _write_checkpoint(tmp_path / "ckpt", table=table_values, scale=scale)
    table = _host_table(tmp_path / "shm", rows=rows)
    table.load(
        checkpoint,
        weight_map=json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"],
        shard_keys=[f"{_PREFIX}.shard_{idx}.weight" for idx in range(_SPLIT_PARTS)],
        scale_key=_SCALE_KEY,
    )
    embedding = Qwen3_8_FlashNextNGramEmbedding(
        table,
        ngram_size=3,
        heads_per_ngram=2,
        eos_token_id=99,
        layer_multipliers=(3, 5, 7),
        ngram_heads_vocab_sizes=(5, 7, 11, 13),
        ngram_heads_offsets=(0, 5, 12, 23),
    )
    input_ids = torch.tensor([[4, 8, 99, 15, 16, 23, 42]])

    actual = embedding(input_ids)

    ngram_ids = embedding._hash_input_ids(input_ids)
    expected = (table_values[ngram_ids].to(torch.bfloat16) * scale).flatten(start_dim=-2)
    assert actual.shape == (1, 7, 4 * _DIM)
    assert torch.equal(_bits(actual), _bits(expected))
    assert "ngram_embedding" not in dict(embedding.named_children())


def test_fp8_table_is_frozen_and_outside_the_module_tree(fp8_checkpoint: _Checkpoint, tmp_path: Path) -> None:
    table = _load_fp8(fp8_checkpoint.path, tmp_path / "shm")
    rows_before = table.rows.clone()
    embedding = Qwen3_8_FlashNextNGramEmbedding(
        table,
        ngram_size=3,
        heads_per_ngram=2,
        eos_token_id=99,
        layer_multipliers=(3, 5, 7),
        ngram_heads_vocab_sizes=(5, 7, 11, 13),
        ngram_heads_offsets=(0, 5, 12, 23),
    )
    ple = Qwen3_8_FlashNextPLELayer(
        embedding,
        hidden_size=2,
        hc_count=2,
        ple_embed_dim=4 * _DIM,
        backend=BackendConfig(linear="torch"),
        dtype=torch.bfloat16,
    )
    # A module-wide cast must not touch the FP8 bytes or the BF16 scale.
    ple.to(torch.float32)
    assert table.rows.dtype == torch.uint8
    assert table.scale.dtype == torch.bfloat16
    assert all("ngram_embedding" not in name for name in ple.state_dict())
    assert all(parameter.numel() != table.rows.numel() for parameter in ple.parameters())

    hidden_states = torch.randn(1, 6, 4, requires_grad=True)
    optimizer = torch.optim.SGD(ple.parameters(), lr=1.0)
    ple(hidden_states, torch.tensor([[1, 2, 3, 99, 4, 5]])).square().sum().backward()
    optimizer.step()

    assert ple.key_proj.weight.grad is not None
    assert hidden_states.grad is not None
    assert torch.equal(table.rows, rows_before)


def test_fp8_model_keeps_the_table_out_of_state_dict_and_init(fp8_checkpoint: _Checkpoint, tmp_path: Path) -> None:
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
        ple_embed_dim=32,
        heads_per_ngram=8,
        split_ngram_parts=_SPLIT_PARTS,
        ple_embedding_dtype=_FP8,
        indexer_budget=8,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=8,
        max_position_embeddings=16,
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default", "partial_rotary_factor": 1.0},
        partial_rotary_factor=1.0,
        dtype="bfloat16",
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=1,
    )
    # A checkpoint directory with this config and the table under layer 0 (ple_layer_ids=[1]).
    checkpoint = _write_checkpoint(
        tmp_path / "ckpt0",
        table=fp8_checkpoint.table,
        scale=fp8_checkpoint.scale,
        prefix=_PREFIX.replace("layers.1.", "layers.0."),
    )
    Qwen3_8_FlashNextConfig(text_config=text, language_model_only=True).save_pretrained(str(checkpoint))
    host_dir = tmp_path / "shm"
    host_dir.mkdir()
    with torch.device("meta"):
        model = Qwen3_8_FlashNextForConditionalGeneration.from_pretrained(
            str(checkpoint),
            moe_config=MoEConfig(
                dim=16,
                inter_dim=16,
                moe_inter_dim=8,
                n_routed_experts=2,
                n_shared_experts=1,
                n_activated_experts=1,
                n_expert_groups=0,
                n_limited_groups=0,
                train_gate=True,
                gate_bias_update_factor=0.0,
                score_func="softmax",
                route_scale=1.0,
                aux_loss_coeff=0.0,
                norm_topk_prob=True,
                expert_bias=False,
                router_bias=False,
                expert_activation="swiglu",
                softmax_before_topk=True,
                shared_expert_gate=True,
                shared_expert_inter_dim=8,
            ),
            backend=BackendConfig(linear="torch", attn="sdpa", experts="torch", dispatcher="torch"),
            engram_table_config=Qwen3_8_FlashNextEngramTableConfig(num_embeddings=_ROWS, embedding_dim=_DIM),
            engram_host_dir=str(host_dir),
        )
    table = model.model.language_model.layers["0"].ple.ple_embedding.ngram_embedding
    assert isinstance(table, Qwen3_8_FlashNextHostNGramTable)
    assert all("ngram_embedding" not in name for name in model.state_dict())
    # The model-owned DTensor hook does not touch the mesh when no table parameter exists.
    assert model._nemo_prepare_model_owned_dtensors(None) == set()

    to_empty_parameters_only(model, device=torch.device("cpu"))
    model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.bfloat16)
    assert table.rows is None

    # The adapter's checkpoint load finds the checkpoint directory by itself and fills the table.
    assert model.state_dict_adapter.pretrained_model_name_or_path == str(checkpoint)
    model.state_dict_adapter.from_hf({})
    ids = torch.arange(_ROWS).view(8, 9)
    expected = fp8_checkpoint.table[ids].to(torch.bfloat16) * fp8_checkpoint.scale
    assert torch.equal(_bits(table(ids)), _bits(expected))


def _fp8_load_error(tmp_path: Path, **checkpoint_kwargs: object) -> str:
    """Load an FP8-config table from a checkpoint built with ``checkpoint_kwargs``; return the error."""
    checkpoint = _write_checkpoint(tmp_path / "ckpt", **checkpoint_kwargs)
    with pytest.raises(RuntimeError) as error:
        _load_fp8(checkpoint, tmp_path / "shm")
    return str(error.value)


def test_fp8_table_without_scale_is_a_load_error(tmp_path: Path) -> None:
    message = _fp8_load_error(tmp_path, table=_finite_fp8_table(_ROWS, _DIM), scale=None)
    assert "FP8 PLE table but no" in message and _SCALE_KEY in message


def test_fp8_config_with_bf16_shards_is_a_load_error(tmp_path: Path) -> None:
    message = _fp8_load_error(
        tmp_path,
        table=torch.ones(_ROWS, _DIM, dtype=torch.bfloat16),
        scale=torch.tensor([_CHECKPOINT_SCALE], dtype=torch.bfloat16),
    )
    assert "stores the PLE table shards as ['BF16']" in message


@pytest.mark.parametrize(
    ("scale", "match"),
    [
        (torch.tensor([_CHECKPOINT_SCALE], dtype=torch.float32), "must be a bfloat16 tensor of shape \\[1\\]"),
        (torch.tensor([_CHECKPOINT_SCALE, 1.0], dtype=torch.bfloat16), "must be a bfloat16 tensor of shape \\[1\\]"),
        (torch.tensor([0.0], dtype=torch.bfloat16), "finite and positive"),
        (torch.tensor([float("nan")], dtype=torch.bfloat16), "finite and positive"),
    ],
)
def test_fp8_invalid_scale_is_a_load_error(tmp_path: Path, scale: torch.Tensor, match: str) -> None:
    checkpoint = _write_checkpoint(tmp_path / "ckpt", table=_finite_fp8_table(_ROWS, _DIM), scale=scale)
    with pytest.raises(RuntimeError, match=match):
        _load_fp8(checkpoint, tmp_path / "shm")


def test_fp8_missing_shard_is_a_load_error(tmp_path: Path) -> None:
    message = _fp8_load_error(
        tmp_path,
        table=_finite_fp8_table(_ROWS, _DIM),
        scale=torch.tensor([_CHECKPOINT_SCALE], dtype=torch.bfloat16),
        drop_shards=(2,),
    )
    assert f"has 3 of the {_SPLIT_PARTS} PLE table shards" in message


def test_fp8_shard_rows_must_fill_the_table(fp8_checkpoint: _Checkpoint, tmp_path: Path) -> None:
    table = _host_table(tmp_path / "shm", rows=_ROWS + _SPLIT_PARTS)
    with pytest.raises(RuntimeError, match="PLE table shards whose shape is not \\[19, 4\\]"):
        _adapter(fp8_checkpoint.path, table, _FP8).from_hf({})
    assert not list((tmp_path / "shm").iterdir())


def _write_unequal_checkpoint(path: Path, table: torch.Tensor, scale: torch.Tensor) -> Path:
    """Write ``_SPLIT_PARTS`` shards of 19, 17, 18 and 18 rows: the right total, shifted rows."""
    path.mkdir(parents=True)
    bounds = [0, 19, 36, 54, _ROWS]
    tensors = {f"{_PREFIX}.shard_{k}.weight": table[bounds[k] : bounds[k + 1]].contiguous() for k in range(4)}
    tensors[_SCALE_KEY] = scale
    save_file(tensors, str(path / "model.safetensors"))
    return path


def test_fp8_unequal_shards_are_a_load_error(fp8_checkpoint: _Checkpoint, tmp_path: Path) -> None:
    checkpoint = _write_unequal_checkpoint(tmp_path / "unequal", fp8_checkpoint.table, fp8_checkpoint.scale)
    with pytest.raises(RuntimeError, match="whose shape is not \\[18, 4\\].*shard_0.*\\[19, 4\\]"):
        _load_fp8(checkpoint, tmp_path / "shm")
    assert not list((tmp_path / "shm").iterdir())


def test_fp8_writer_rejects_unequal_shards_on_its_own(fp8_checkpoint: _Checkpoint, tmp_path: Path) -> None:
    checkpoint = _write_unequal_checkpoint(tmp_path / "unequal", fp8_checkpoint.table, fp8_checkpoint.scale)
    table = _host_table(tmp_path / "shm")
    with pytest.raises(RuntimeError, match="must be float8_e4m3fn of shape \\[18, 4\\]; got .* \\[19, 4\\]"):
        table.load(
            checkpoint,
            weight_map={
                key: "model.safetensors" for key in (*[f"{_PREFIX}.shard_{k}.weight" for k in range(4)], _SCALE_KEY)
            },
            shard_keys=[f"{_PREFIX}.shard_{k}.weight" for k in range(4)],
            scale_key=_SCALE_KEY,
        )
    assert not list((tmp_path / "shm").glob("*.bin"))


def test_fp8_without_host_dir_is_a_load_error(fp8_checkpoint: _Checkpoint) -> None:
    table = Qwen3_8_FlashNextHostNGramTable(num_embeddings=_ROWS, embedding_dim=_DIM, host_dir=None)
    with pytest.raises(RuntimeError, match="needs an explicit host directory.*engram_host_dir"):
        _adapter(fp8_checkpoint.path, table, _FP8).from_hf({})


def test_replaced_shard_with_same_size_and_mtime_is_read_again(fp8_checkpoint: _Checkpoint, tmp_path: Path) -> None:
    host_dir = tmp_path / "shm"
    first = _load_fp8(fp8_checkpoint.path, host_dir)
    shard_file = fp8_checkpoint.path / "model-ple-000.safetensors"
    stat = shard_file.stat()
    new_table = _finite_fp8_table(_ROWS, _DIM, seed=9)
    rows = _ROWS // _SPLIT_PARTS
    save_file(
        {f"{_PREFIX}.shard_{k}.weight": new_table[k * rows : (k + 1) * rows].contiguous() for k in (0, 1)},
        str(shard_file),
    )
    os.utime(shard_file, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert shard_file.stat().st_size == stat.st_size and shard_file.stat().st_mtime_ns == stat.st_mtime_ns

    second = _load_fp8(fp8_checkpoint.path, host_dir)

    assert second.source != first.source
    ids = torch.arange(2 * rows)
    expected = new_table[ids].to(torch.bfloat16) * fp8_checkpoint.scale
    assert torch.equal(_bits(second(ids)), _bits(expected))


def test_host_file_that_differs_from_its_source_is_written_again(
    fp8_checkpoint: _Checkpoint, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    host_dir = tmp_path / "shm"
    first = _load_fp8(fp8_checkpoint.path, host_dir)
    manifest = json.loads(first.source.with_name(first.source.name + ".ok").read_text())
    assert len(manifest["shards"]) == _SPLIT_PARTS and manifest["rows_per_shard"] == _ROWS // _SPLIT_PARTS
    first.source.write_bytes(bytes(_ROWS * _DIM))  # same size, other bytes, marker kept

    with caplog.at_level(logging.WARNING):
        second = _load_fp8(fp8_checkpoint.path, host_dir)

    assert "does not match its source shards; writing it again" in caplog.text
    ids = torch.arange(_ROWS)
    expected = fp8_checkpoint.table[ids].to(torch.bfloat16) * fp8_checkpoint.scale
    assert torch.equal(_bits(second(ids)), _bits(expected))


def test_supplied_scale_with_bf16_config_is_an_error_without_checkpoint_dir(tmp_path: Path) -> None:
    table = Qwen3_8_FlashNextEngramTableConfig(num_embeddings=16, embedding_dim=_DIM).build(process_group=None)
    scale = torch.tensor([_CHECKPOINT_SCALE], dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="a scale with a BF16 PLE table"):
        _adapter(None, table, None).from_hf({_SCALE_KEY: scale})


def test_supplied_fp8_shard_with_bf16_config_is_an_error_without_checkpoint_dir(tmp_path: Path) -> None:
    table = Qwen3_8_FlashNextEngramTableConfig(num_embeddings=16, embedding_dim=_DIM).build(process_group=None)
    shard = _finite_fp8_table(4, _DIM)
    with pytest.raises(RuntimeError, match="is float8_e4m3fn, but the config does not set"):
        _adapter(None, table, None).from_hf({f"{_PREFIX}.shard_0.weight": shard})


def test_supplied_fp8_tensors_must_agree_with_the_loaded_table(fp8_checkpoint: _Checkpoint, tmp_path: Path) -> None:
    adapter = _adapter(fp8_checkpoint.path, _host_table(tmp_path / "shm"), _FP8)
    other_scale = torch.tensor([2 * _CHECKPOINT_SCALE], dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="differs from the scale loaded"):
        adapter.from_hf({_SCALE_KEY: other_scale})
    with pytest.raises(RuntimeError, match="supplied .*shard_1.weight is torch.bfloat16"):
        adapter.from_hf({f"{_PREFIX}.shard_1.weight": torch.zeros(18, _DIM, dtype=torch.bfloat16)})
    native = adapter.from_hf(
        {_SCALE_KEY: fp8_checkpoint.scale, f"{_PREFIX}.shard_1.weight": fp8_checkpoint.table[18:36].contiguous()}
    )
    assert native == {}


def test_bf16_views_of_model_built_from_config_warn(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    table = Qwen3_8_FlashNextEngramTableConfig(num_embeddings=16, embedding_dim=_DIM).build(process_group=None)
    adapter = _adapter(None, table, None)
    views = dict(adapter.convert_single_tensor_to_hf(f"{_PREFIX}.weight", table.weight))
    with caplog.at_level(logging.WARNING):
        assert adapter.from_hf(views) == {}
    assert "model built from a config (no base checkpoint)" in caplog.text


def test_fp8_needs_a_local_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="read from a local safetensors checkpoint"):
        _adapter(None, _host_table(tmp_path / "shm"), _FP8).from_hf({})


def test_named_checkpoint_without_local_copy_is_an_error(tmp_path: Path) -> None:
    table = Qwen3_8_FlashNextEngramTableConfig(num_embeddings=16, embedding_dim=_DIM).build(process_group=None)
    with pytest.raises(RuntimeError, match="is not a local directory"):
        _adapter(tmp_path / "missing-checkpoint", table, None).from_hf({})


def test_hub_id_is_read_from_the_directory_the_checkpointer_names(fp8_checkpoint: _Checkpoint, tmp_path: Path) -> None:
    table = _host_table(tmp_path / "shm")
    adapter = Qwen3_8_FlashNextStateDictAdapter(
        _text_config(_FP8), _moe_config(), _backend(), table, pretrained_model_name_or_path="org/model"
    )
    adapter.set_base_checkpoint_dir(fp8_checkpoint.path)
    adapter.from_hf({})
    assert torch.equal(table.rows, fp8_checkpoint.table.view(torch.uint8))


def test_bf16_views_of_named_checkpoint_without_safetensors_are_an_error(tmp_path: Path) -> None:
    checkpoint = tmp_path / "bin-only"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    table = Qwen3_8_FlashNextEngramTableConfig(num_embeddings=16, embedding_dim=_DIM).build(process_group=None)
    adapter = _adapter(checkpoint, table, None)
    views = dict(adapter.convert_single_tensor_to_hf(f"{_PREFIX}.weight", table.weight))
    with pytest.raises(RuntimeError, match="Cannot check the stored format of 4 PLE table shards"):
        adapter.from_hf(views)


def test_fp8_host_dir_without_room_is_a_load_error(
    fp8_checkpoint: _Checkpoint, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _Usage(NamedTuple):
        total: int
        used: int
        free: int

    monkeypatch.setattr(engram_module.shutil, "disk_usage", lambda path: _Usage(10**6, 10**6 - 7, 7))
    with pytest.raises(RuntimeError, match="needs .* GiB \\(table plus 5%\\) in .* which has .* GiB free"):
        _load_fp8(fp8_checkpoint.path, tmp_path / "shm")


def test_bf16_config_with_scale_is_a_load_error(tmp_path: Path) -> None:
    checkpoint = _write_checkpoint(
        tmp_path / "ckpt",
        table=torch.ones(16, _DIM, dtype=torch.float32),
        scale=torch.tensor([_CHECKPOINT_SCALE], dtype=torch.bfloat16),
    )
    table = Qwen3_8_FlashNextEngramTableConfig(num_embeddings=16, embedding_dim=_DIM).build(process_group=None)
    with pytest.raises(RuntimeError, match="a scale with a BF16 PLE table"):
        _adapter(checkpoint, table, None).from_hf({})


def test_bf16_config_with_fp8_shards_is_a_load_error(tmp_path: Path) -> None:
    checkpoint = _write_checkpoint(tmp_path / "ckpt", table=_finite_fp8_table(16, _DIM), scale=None)
    table = Qwen3_8_FlashNextEngramTableConfig(num_embeddings=16, embedding_dim=_DIM).build(process_group=None)
    with pytest.raises(RuntimeError, match="stores the PLE table shards as F8_E4M3"):
        _adapter(checkpoint, table, "bfloat16").from_hf({})


def test_bf16_table_without_scale_keeps_view_loading(tmp_path: Path) -> None:
    checkpoint = _write_checkpoint(tmp_path / "ckpt", table=torch.ones(16, _DIM, dtype=torch.float32), scale=None)
    table = Qwen3_8_FlashNextEngramTableConfig(num_embeddings=16, embedding_dim=_DIM).build(process_group=None)
    adapter = _adapter(checkpoint, table, None)
    views = dict(adapter.convert_single_tensor_to_hf(f"{_PREFIX}.weight", table.weight))

    native = adapter.from_hf(views)

    assert sorted(views) == [f"{_PREFIX}.shard_{idx}.weight" for idx in range(_SPLIT_PARTS)]
    assert native == {}
    assert adapter.view_loaded_native_keys == {f"{_PREFIX}.weight"}


def test_ple_embedding_dtype_rejects_other_dtypes() -> None:
    with pytest.raises(ValueError, match="ple_embedding_dtype must be"):
        _text_config("float16")
    nested = Qwen3_8_FlashNextConfig(text_config={"ple_layer_ids": [2], "ple_embedding_dtype": _FP8})
    assert nested.text_config.ple_embedding_dtype == _FP8


def test_unloaded_fp8_table_refuses_lookups(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="not loaded"):
        _host_table(tmp_path / "shm")(torch.tensor([0]))


def test_fp8_lookup_rejects_out_of_range_ids(fp8_checkpoint: _Checkpoint, tmp_path: Path) -> None:
    table = _load_fp8(fp8_checkpoint.path, tmp_path / "shm")
    with pytest.raises(ValueError, match=f"must lie in \\[0, {_ROWS}\\)"):
        table(torch.tensor([0, _ROWS]))


def test_fp8_host_file_is_written_once_and_reused(
    fp8_checkpoint: _Checkpoint, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    host_dir = tmp_path / "shm"
    first = _load_fp8(fp8_checkpoint.path, host_dir)
    table_files = sorted(host_dir.glob("*.bin"))
    assert len(table_files) == 1 and table_files[0].stat().st_size == _ROWS * _DIM
    assert table_files[0].with_name(table_files[0].name + ".ok").is_file()

    def _no_write(*args: object) -> None:
        raise AssertionError("the complete host file must be reused, not rewritten")

    monkeypatch.setattr(Qwen3_8_FlashNextHostNGramTable, "_write_rows", _no_write)
    second = _load_fp8(fp8_checkpoint.path, host_dir)
    _adapter(fp8_checkpoint.path, second, _FP8).from_hf({})
    assert torch.equal(first.rows, second.rows)
    assert second.source == table_files[0]

    # A file without its completion marker (a killed writer) is written again.
    monkeypatch.undo()
    table_files[0].with_name(table_files[0].name + ".ok").unlink()
    third = _load_fp8(fp8_checkpoint.path, host_dir)
    assert torch.equal(third.rows, first.rows)


def test_fp8_host_file_is_shared_through_a_symlinked_checkpoint_view(
    fp8_checkpoint: _Checkpoint, tmp_path: Path
) -> None:
    host_dir = tmp_path / "shm"
    first = _load_fp8(fp8_checkpoint.path, host_dir)
    view = tmp_path / "view"
    view.mkdir()
    for file in fp8_checkpoint.path.iterdir():
        (view / file.name).symlink_to(file.resolve())

    second = _load_fp8(view, host_dir)

    assert second.source == first.source
    assert len(list(host_dir.glob("*.bin"))) == 1


def test_fp8_host_file_is_shared_through_another_path_to_the_same_files(
    fp8_checkpoint: _Checkpoint, tmp_path: Path
) -> None:
    # Hard links reach the same inodes under another resolved path, as a second mount path of a shared
    # filesystem does.
    # Linking changes the inode's ctime, so link first and load after.
    linked = tmp_path / "linked"
    linked.mkdir()
    for file in fp8_checkpoint.path.iterdir():
        os.link(file, linked / file.name)
    assert (linked / "model-ple-000.safetensors").resolve() != (
        fp8_checkpoint.path / "model-ple-000.safetensors"
    ).resolve()
    host_dir = tmp_path / "shm"

    first = _load_fp8(fp8_checkpoint.path, host_dir)
    second = _load_fp8(linked, host_dir)

    assert second.source == first.source
    assert len(list(host_dir.glob("*.bin"))) == 1


def _load_in_process(rank: int, checkpoint: str, host_dir: str, results: dict[int, bytes]) -> None:
    table = _load_fp8(Path(checkpoint), Path(host_dir))
    results[rank] = table.rows.numpy().tobytes()


def test_concurrent_loads_share_one_host_file(fp8_checkpoint: _Checkpoint, tmp_path: Path) -> None:
    host_dir = tmp_path / "shm"
    host_dir.mkdir()
    with mp.Manager() as manager:
        results = manager.dict()
        mp.spawn(_load_in_process, args=(str(fp8_checkpoint.path), str(host_dir), results), nprocs=3, join=True)
        loaded = dict(results)
    assert len(loaded) == 3
    expected = fp8_checkpoint.table.view(torch.uint8).numpy().tobytes()
    assert all(rows == expected for rows in loaded.values())
    assert len(list(host_dir.glob("*.bin"))) == 1
    assert not list(host_dir.glob("*.tmp*"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_fp8_cuda_lookup_matches_cpu_bitwise(fp8_checkpoint: _Checkpoint, tmp_path: Path) -> None:
    table = _load_fp8(fp8_checkpoint.path, tmp_path / "shm")
    ids = torch.arange(_ROWS).view(6, 12)
    cuda_values = table(ids.cuda())
    assert cuda_values.device.type == "cuda"
    assert torch.equal(_bits(cuda_values.cpu()), _bits(table(ids)))


class _TinyFP8CheckpointModel(HFCheckpointingMixin, nn.Module):
    """Minimal model with the production PLE hierarchy, an FP8 host table, and one trainable parameter."""

    def __init__(self, name_or_path: str, host_dir: Path) -> None:
        super().__init__()
        self.config = _text_config(_FP8)
        self.ordinary = nn.Parameter(torch.zeros(3))
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.layers = nn.ModuleList([nn.Module(), nn.Module()])
        ple = nn.Module()
        ple.ple_embedding = nn.Module()
        # A plain attribute, as in the model: the host table is not a submodule.
        ple.ple_embedding.ngram_embedding = _host_table(host_dir)
        self.model.language_model.layers[1].ple = ple
        self.state_dict_adapter = Qwen3_8_FlashNextStateDictAdapter(
            _text_config(_FP8),
            _moe_config(),
            _backend(),
            ple.ple_embedding.ngram_embedding,
            dtype=torch.float32,
            pretrained_model_name_or_path=name_or_path,
        )
        # ``apply_model_infrastructure`` records this key list before sharding in production.
        self._pre_shard_hf_state_dict_keys = self.state_dict_adapter.get_hf_state_dict_keys(self.state_dict())

    @property
    def table(self) -> Qwen3_8_FlashNextHostNGramTable:
        return self.model.language_model.layers[1].ple.ple_embedding.ngram_embedding


def _write_model_checkpoint(path: Path, table: torch.Tensor, scale: torch.Tensor, ordinary: torch.Tensor) -> Path:
    """Write an FP8-table checkpoint that also holds the tiny model's trainable parameter."""
    _write_checkpoint(path, table=table, scale=scale, extra_tensors={"ordinary": ordinary})
    return path


def _checkpointer(checkpoint_dir: Path, *, repo_id: str | None = None):
    return CheckpointingConfig(
        enabled=True,
        checkpoint_dir=str(checkpoint_dir),
        model_save_format="safetensors",
        model_repo_id=repo_id,
        save_consolidated=True,
        is_peft=False,
    ).build(dp_rank=0, tp_rank=0, pp_rank=0)


def test_base_load_reads_the_table_from_the_snapshot_the_checkpointer_loads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Two cached snapshots of one Hub repository: ``main`` points at A, the run requests B.
    repo = tmp_path / "hub" / "models--org--tiny"
    sha_a, sha_b = "a" * 40, "b" * 40
    table_a = _finite_fp8_table(_ROWS, _DIM, seed=11)
    table_b = (-table_a.to(torch.float32)).to(torch.float8_e4m3fn)
    scale_b = torch.tensor([_CHECKPOINT_SCALE], dtype=torch.bfloat16)
    _write_model_checkpoint(repo / "snapshots" / sha_a, table_a, scale_b * 2, torch.full((3,), 1.0))
    snapshot_b = _write_model_checkpoint(repo / "snapshots" / sha_b, table_b, scale_b, torch.full((3,), 2.0))
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text(sha_a)
    monkeypatch.setattr(huggingface_constants, "HF_HUB_CACHE", str(tmp_path / "hub"))

    model = _TinyFP8CheckpointModel("org/tiny", tmp_path / "shm")
    checkpointer = _checkpointer(tmp_path / "out")
    try:
        # The native loader resolves revision B to its snapshot and passes that directory.
        checkpointer.load_base_model(model, torch.device("cpu"), str(tmp_path / "hub"), str(snapshot_b))
    finally:
        checkpointer.close()

    assert torch.equal(model.ordinary.detach(), torch.full((3,), 2.0))
    assert torch.equal(model.table.rows, table_b.view(torch.uint8))
    assert torch.equal(model.table.scale, scale_b)


def test_named_checkpoint_needs_the_checkpointer_snapshot(tmp_path: Path) -> None:
    # The adapter never resolves a Hub ID by itself; only the checkpointer names the snapshot.
    adapter = _adapter(None, _host_table(tmp_path / "shm"), _FP8)
    adapter.pretrained_model_name_or_path = "org/tiny"
    with pytest.raises(RuntimeError, match="is not a local directory"):
        adapter.from_hf({})
