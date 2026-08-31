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

"""CPU-runnable tests for HY4 packed context-parallel batch helpers."""

from __future__ import annotations

import contextlib

import pytest
import torch

from nemo_automodel.components.distributed.context_parallel.sharder import contiguous_local_indices
from nemo_automodel.components.models.hy_v4 import cp as hy_cp


class _FakeMesh:
    """Minimal CPU mesh implementing the sharder rank/size contract."""

    def __init__(self, size: int = 2, group: str = "cp-group", rank: int = 0) -> None:
        self._size = size
        self._group = group
        self._rank = rank

    def size(self) -> int:
        return self._size

    def get_group(self) -> str:
        return self._group

    def get_local_rank(self) -> int:
        return self._rank


def _thd_chunk(num_tokens: int = 6) -> dict[str, torch.Tensor]:
    """Build global THD fields with token axis ``[num_tokens]``."""
    return {
        "input_ids": torch.arange(num_tokens),
        "labels": torch.arange(num_tokens) + 100,
        "position_ids": torch.arange(num_tokens) + 200,
        "cu_seqlens": torch.tensor([0, num_tokens // 2, num_tokens], dtype=torch.int64),
        "max_seqlen": torch.tensor(num_tokens // 2, dtype=torch.int64),
        "cu_seqlens_padded": torch.tensor([0, num_tokens // 2 + 1, num_tokens + 2], dtype=torch.int64),
        "padding_mask": torch.tensor([index == 3 for index in range(num_tokens)]),
    }


def test_cp_enabled_checks_group_and_world_size(monkeypatch):
    monkeypatch.setattr(hy_cp.dist, "is_available", lambda: True)
    monkeypatch.setattr(hy_cp.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(hy_cp.dist, "get_world_size", lambda group: 2 if group == "cp-group" else 1)

    assert hy_cp.hy_v4_cp_enabled(None) is False
    assert hy_cp.hy_v4_cp_enabled("other-group") is False
    assert hy_cp.hy_v4_cp_enabled("cp-group") is True


def test_cp_all_gather_noops_when_cp_disabled():
    tensor = torch.randn(2, 3)

    assert hy_cp.hy_v4_cp_all_gather(tensor, dim=0, cp_group=None) is tensor


def test_cp_all_gather_concatenates_differentiable_rank_outputs(monkeypatch):
    tensor = torch.arange(4).view(2, 2)
    monkeypatch.setattr(hy_cp, "hy_v4_cp_enabled", lambda cp_group: True)

    def fake_all_gather(received: torch.Tensor, group: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Return two rank shards shaped like the input without copying rank 0."""
        assert group == "cp-group"
        return received, received + 10

    monkeypatch.setattr(hy_cp, "all_gather", fake_all_gather)

    gathered = hy_cp.hy_v4_cp_all_gather(tensor, dim=0, cp_group="cp-group")

    assert gathered.tolist() == [[0, 1], [2, 3], [10, 11], [12, 13]]


def test_slice_thd_chunk_requires_even_divisibility():
    with pytest.raises(ValueError, match="divisible by cp_size"):
        hy_cp._slice_thd_chunk_for_cp(
            _thd_chunk(num_tokens=5),
            cp_mesh=_FakeMesh(size=2),
            cp_group="cp-group",
            cp_size=2,
            cp_rank=0,
            padding_token_id=120002,
        )


def test_slice_thd_chunk_preserves_global_metadata_and_pack_mask():
    local = hy_cp._slice_thd_chunk_for_cp(
        _thd_chunk(),
        cp_mesh=_FakeMesh(size=3, rank=1),
        cp_group="cp-group",
        cp_size=3,
        cp_rank=1,
        padding_token_id=120002,
    )

    assert local["input_ids"].tolist() == [2, 3]
    assert local["labels"].tolist() == [102, 103]
    assert local["position_ids"].tolist() == [202, 203]
    assert local["cu_seqlens"].dtype == torch.int32
    assert local["cu_seqlens"].tolist() == [0, 3, 6]
    assert local["max_seqlen"].dtype == torch.int32
    assert local["cu_seqlens_padded"].tolist() == [0, 4, 8]
    assert local["hy_v4_cp_query_indices"].dtype == torch.int32
    assert local["hy_v4_cp_query_indices"].tolist() == [2, 3]
    assert local["padding_mask"].tolist() == [False, True]
    assert local["qkv_format"] == "thd"
    assert local["_hy_v4_cp_group"] == "cp-group"


def test_make_packed_cp_batch_single_chunk(monkeypatch):
    captured = {}

    def fake_split(batch, **kwargs):
        captured.update(batch=batch, kwargs=kwargs)
        return _thd_chunk()

    monkeypatch.setattr(hy_cp, "split_batch_into_thd_chunks", fake_split)
    monkeypatch.setattr(hy_cp.dist, "is_available", lambda: False)
    monkeypatch.setattr(hy_cp.dist, "is_initialized", lambda: False)

    ctx, local = hy_cp.make_hy_v4_packed_cp_batch_and_ctx(
        _FakeMesh(size=3),
        None,
        {"input_ids": torch.arange(6).view(1, 6)},
        padding_token_id=120002,
        num_chunks=1,
        seq_lens_padding_value=-77,
    )

    assert ctx is contextlib.nullcontext
    assert captured["kwargs"]["num_chunks"] == 1
    assert captured["kwargs"]["seq_lens_padding_value"] == -77
    assert captured["kwargs"]["padding_token_id"] == 120002
    assert local["input_ids"].tolist() == [0, 1]
    assert local["cp_size"] == 3
    assert local["cp_rank"] == 0


def test_shard_packed_cp_batch_reports_single_chunk_layout(monkeypatch):
    monkeypatch.setattr(hy_cp, "split_batch_into_thd_chunks", lambda *args, **kwargs: _thd_chunk())
    monkeypatch.setattr(hy_cp.dist, "is_available", lambda: False)
    monkeypatch.setattr(hy_cp.dist, "is_initialized", lambda: False)

    ctx, local, layout = hy_cp.shard_hy_v4_packed_cp_batch(
        _FakeMesh(size=3),
        None,
        {"input_ids": torch.arange(6).view(1, 6)},
        padding_token_id=120002,
    )

    assert ctx is contextlib.nullcontext
    assert local["input_ids"].tolist() == [0, 1]
    assert layout is not None
    assert layout.padded_seq_len == 6
    assert layout.input_row_shape == (1, 6)


def test_make_packed_cp_batch_stacks_pipeline_chunks(monkeypatch):
    thd_batch = {
        "input_ids": torch.tensor([[0, 1, 2, 3], [10, 11, 12, 13]]),
        "labels": torch.tensor([[100, 101, 102, 103], [110, 111, 112, 113]]),
        "position_ids": torch.tensor([[200, 201, 202, 203], [210, 211, 212, 213]]),
        "cu_seqlens": torch.tensor([[0, 4], [0, 4]], dtype=torch.int64),
        "max_seqlen": torch.tensor([4, 4], dtype=torch.int64),
        "cu_seqlens_padded": torch.tensor([[0, 4], [0, 4]], dtype=torch.int64),
        "padding_mask": torch.tensor([[False, False, False, False], [False, False, True, False]]),
    }
    monkeypatch.setattr(hy_cp, "split_batch_into_thd_chunks", lambda *args, **kwargs: thd_batch)
    monkeypatch.setattr(hy_cp.dist, "is_available", lambda: True)
    monkeypatch.setattr(hy_cp.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(hy_cp.dist, "get_rank", lambda group: 1)

    ctx, local = hy_cp.make_hy_v4_packed_cp_batch_and_ctx(
        _FakeMesh(size=2),
        None,
        {"input_ids": torch.arange(8).view(2, 4)},
        padding_token_id=120002,
        num_chunks=2,
    )

    assert ctx is contextlib.nullcontext
    assert local["input_ids"].tolist() == [[2, 3], [12, 13]]
    assert local["labels"].tolist() == [[102, 103], [112, 113]]
    assert local["position_ids"].tolist() == [[202, 203], [212, 213]]
    assert local["hy_v4_cp_query_indices"].tolist() == [[2, 3], [2, 3]]
    assert local["padding_mask"].tolist() == [[False, False], [True, False]]
    assert local["cp_rank"] == 1
    assert local["qkv_format"] == "thd"


def test_model_registers_contiguous_model_owned_cp_sharder(tiny_hy_v4_model):
    sharder = tiny_hy_v4_model.prepare_model_inputs_for_cp({"input_ids": torch.arange(8).unsqueeze(0)})["cp_sharder"]

    assert sharder.local_token_global_indices is contiguous_local_indices
    assert tiny_hy_v4_model._owns_cp_attention
