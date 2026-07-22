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

"""CPU-runnable tests for GLM MoE DSA context-parallel batch helpers."""

from __future__ import annotations

import contextlib

import pytest
import torch

from nemo_automodel.components.models.glm_moe_dsa import cp as glm_cp


class _FakeMesh:
    def __init__(self, size=2, group="cp-group"):
        self._size = size
        self._group = group

    def size(self):
        return self._size

    def get_group(self):
        return self._group


def _thd_chunk(num_tokens=6):
    return {
        "input_ids": torch.arange(num_tokens),
        "labels": torch.arange(num_tokens) + 100,
        "position_ids": torch.arange(num_tokens) + 200,
        "cu_seqlens": torch.tensor([0, num_tokens // 2, num_tokens], dtype=torch.int64),
        "max_seqlen": torch.tensor(num_tokens // 2, dtype=torch.int64),
        "cu_seqlens_padded": torch.tensor([0, num_tokens // 2 + 1, num_tokens + 2], dtype=torch.int64),
    }


def test_glm_dsa_cp_enabled_checks_group_and_world_size(monkeypatch):
    monkeypatch.setattr(glm_cp.dist, "is_available", lambda: True)
    monkeypatch.setattr(glm_cp.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(glm_cp.dist, "get_world_size", lambda group: 2 if group == "cp-group" else 1)

    assert glm_cp.glm_dsa_cp_enabled(None) is False
    assert glm_cp.glm_dsa_cp_enabled("other-group") is False
    assert glm_cp.glm_dsa_cp_enabled("cp-group") is True


def test_glm_dsa_cp_all_gather_noops_when_cp_disabled():
    tensor = torch.randn(2, 3)
    assert glm_cp.glm_dsa_cp_all_gather(tensor, dim=0, cp_group=None) is tensor


def test_glm_dsa_cp_all_gather_concatenates_autograd_gather(monkeypatch):
    tensor = torch.arange(4).view(2, 2)
    monkeypatch.setattr(glm_cp, "glm_dsa_cp_enabled", lambda cp_group: True)

    import torch.distributed.nn.functional as dist_nn_f

    def fake_all_gather(received, group):
        assert group == "cp-group"
        return (received, received + 10)

    monkeypatch.setattr(dist_nn_f, "all_gather", fake_all_gather)

    out = glm_cp.glm_dsa_cp_all_gather(tensor, dim=0, cp_group="cp-group")

    assert out.tolist() == [[0, 1], [2, 3], [10, 11], [12, 13]]


def test_contiguous_cp_indices_requires_even_divisibility():
    with pytest.raises(ValueError, match="total tokens divisible"):
        glm_cp._contiguous_cp_indices(total_tokens=5, cp_size=2, cp_rank=0, device=torch.device("cpu"))

    assert glm_cp._contiguous_cp_indices(6, 3, 1, torch.device("cpu")).tolist() == [2, 3]


def test_slice_thd_chunk_for_cp_preserves_global_metadata_and_padding_mask():
    out = glm_cp._slice_thd_chunk_for_cp(
        _thd_chunk(),
        cp_group="cp-group",
        cp_size=3,
        cp_rank=1,
        padding_token_id=3,
    )

    assert out["input_ids"].tolist() == [2, 3]
    assert out["labels"].tolist() == [102, 103]
    assert out["position_ids"].tolist() == [202, 203]
    assert out["cu_seqlens"].dtype == torch.int32
    assert out["cu_seqlens"].tolist() == [0, 3, 6]
    assert out["max_seqlen"].dtype == torch.int32
    assert out["cu_seqlens_padded"].tolist() == [0, 4, 8]
    assert out["glm_dsa_cp_query_indices"].dtype == torch.int32
    assert out["glm_dsa_cp_query_indices"].tolist() == [2, 3]
    assert out["padding_mask"].tolist() == [False, True]
    assert out["qkv_format"] == "thd"
    assert out["_glm_dsa_cp_group"] == "cp-group"


def test_make_glm_dsa_packed_cp_batch_single_chunk(monkeypatch):
    captured = {}

    def fake_split(batch, **kwargs):
        captured.update(batch=batch, kwargs=kwargs)
        return _thd_chunk()

    monkeypatch.setattr(glm_cp, "split_batch_into_thd_chunks", fake_split)
    monkeypatch.setattr(glm_cp.dist, "is_available", lambda: False)
    monkeypatch.setattr(glm_cp.dist, "is_initialized", lambda: False)

    out = glm_cp._prepare_glm_dsa_packed_cp_batch(
        _FakeMesh(size=3),
        None,
        {"input_ids": torch.arange(6).view(1, 6)},
        padding_token_id=3,
        num_chunks=1,
        seq_lens_padding_value=-77,
    )

    assert captured["kwargs"]["num_chunks"] == 1
    assert captured["kwargs"]["seq_lens_padding_value"] == -77
    assert captured["kwargs"]["padding_token_id"] == 3
    assert out["input_ids"].tolist() == [0, 1]
    assert out["cp_size"] == 3
    assert out["cp_rank"] == 0


def test_shard_glm_dsa_packed_cp_batch_reports_single_chunk_layout(monkeypatch):
    monkeypatch.setattr(glm_cp, "split_batch_into_thd_chunks", lambda *args, **kwargs: _thd_chunk())
    monkeypatch.setattr(glm_cp.dist, "is_available", lambda: False)
    monkeypatch.setattr(glm_cp.dist, "is_initialized", lambda: False)

    prepared = glm_cp.shard_glm_dsa_packed_cp_batch(
        _FakeMesh(size=3),
        None,
        {"input_ids": torch.arange(6).view(1, 6)},
        padding_token_id=3,
        num_chunks=1,
    )

    out = prepared.batch
    assert isinstance(prepared.context, contextlib.nullcontext)
    assert out["input_ids"].tolist() == [0, 1]
    assert prepared.padded_seq_len == 6
    assert prepared.input_row_shape == (1, 6)


def test_make_glm_dsa_packed_cp_batch_stacks_pipeline_chunks(monkeypatch):
    thd_batch = {
        "input_ids": torch.tensor([[0, 1, 2, 3], [10, 11, 12, 13]]),
        "labels": torch.tensor([[100, 101, 102, 103], [110, 111, 112, 113]]),
        "position_ids": torch.tensor([[200, 201, 202, 203], [210, 211, 212, 213]]),
        "cu_seqlens": torch.tensor([[0, 4], [0, 4]], dtype=torch.int64),
        "max_seqlen": torch.tensor([4, 4], dtype=torch.int64),
        "cu_seqlens_padded": torch.tensor([[0, 4], [0, 4]], dtype=torch.int64),
    }
    monkeypatch.setattr(glm_cp, "split_batch_into_thd_chunks", lambda *args, **kwargs: thd_batch)
    monkeypatch.setattr(glm_cp.dist, "is_available", lambda: True)
    monkeypatch.setattr(glm_cp.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(glm_cp.dist, "get_rank", lambda group: 1)

    out = glm_cp._prepare_glm_dsa_packed_cp_batch(
        _FakeMesh(size=2),
        None,
        {"input_ids": torch.arange(8).view(2, 4)},
        padding_token_id=12,
        num_chunks=2,
    )

    assert out["input_ids"].tolist() == [[2, 3], [12, 13]]
    assert out["labels"].tolist() == [[102, 103], [112, 113]]
    assert out["position_ids"].tolist() == [[202, 203], [212, 213]]
    assert out["glm_dsa_cp_query_indices"].tolist() == [[2, 3], [2, 3]]
    assert out["padding_mask"].tolist() == [[False, False], [True, False]]
    assert out["cp_rank"] == 1
    assert out["qkv_format"] == "thd"
