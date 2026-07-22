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

"""Tests for GLM MoE DSA context-parallel helpers."""

from __future__ import annotations

import contextlib
import math
import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nemo_automodel.components.models.glm_moe_dsa import cp as glm_cp


def _free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _nccl_all_gather_worker(rank: int, world_size: int, port: int) -> None:
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)

        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

        # Rank-dependent loss weights make the expected gradient sensitive to
        # both gather ordering and routing gradients back to the source rank.
        local = torch.arange(rank * 4, rank * 4 + 4, device=device, dtype=torch.float32).view(2, 2)
        local.requires_grad_(True)
        gathered = glm_cp.glm_dsa_cp_all_gather(local, dim=0, cp_group=dist.group.WORLD)
        expected = torch.arange(world_size * 4, device=device, dtype=torch.float32).view(world_size * 2, 2)
        torch.testing.assert_close(gathered, expected)

        weights = torch.arange(1, gathered.numel() + 1, device=device, dtype=torch.float32).view_as(gathered)
        ((rank + 1) * gathered * weights).sum().backward()
        rank_scale_sum = world_size * (world_size + 1) // 2
        expected_grad = rank_scale_sum * weights[rank * 2 : (rank + 1) * 2]
        torch.testing.assert_close(local.grad, expected_grad)

        # Match the production CP shape: local Q attends causally to gathered
        # global K/V, while K/V gradients accumulate losses from every rank.
        global_seq, head_dim = world_size * 2, 4
        start, end = rank * 2, (rank + 1) * 2
        full_q_data = torch.sin(torch.arange(global_seq * head_dim, device=device).view(global_seq, head_dim) / 7)
        full_k_data = torch.cos(torch.arange(global_seq * head_dim, device=device).view(global_seq, head_dim) / 5)
        full_v_data = torch.tanh(torch.arange(global_seq * head_dim, device=device).view(global_seq, head_dim) / 9)

        local_q = full_q_data[start:end].clone().requires_grad_(True)
        local_k = full_k_data[start:end].clone().requires_grad_(True)
        local_v = full_v_data[start:end].clone().requires_grad_(True)
        global_k = glm_cp.glm_dsa_cp_all_gather(local_k, dim=0, cp_group=dist.group.WORLD)
        global_v = glm_cp.glm_dsa_cp_all_gather(local_v, dim=0, cp_group=dist.group.WORLD)

        query_positions = torch.arange(start, end, device=device)
        key_positions = torch.arange(global_seq, device=device)
        local_scores = local_q @ global_k.transpose(0, 1) / math.sqrt(head_dim)
        local_scores = local_scores.masked_fill(key_positions[None, :] > query_positions[:, None], -torch.inf)
        local_out = torch.softmax(local_scores, dim=-1) @ global_v

        ref_q = full_q_data.clone().requires_grad_(True)
        ref_k = full_k_data.clone().requires_grad_(True)
        ref_v = full_v_data.clone().requires_grad_(True)
        ref_scores = ref_q @ ref_k.transpose(0, 1) / math.sqrt(head_dim)
        causal_mask = key_positions[None, :] > key_positions[:, None]
        ref_scores = ref_scores.masked_fill(causal_mask, -torch.inf)
        ref_out = torch.softmax(ref_scores, dim=-1) @ ref_v
        torch.testing.assert_close(local_out, ref_out[start:end], rtol=1e-5, atol=1e-6)

        output_weights = torch.linspace(0.25, 2.0, global_seq * head_dim, device=device).view(global_seq, head_dim)
        (local_out * output_weights[start:end]).sum().backward()
        (ref_out * output_weights).sum().backward()
        torch.testing.assert_close(local_q.grad, ref_q.grad[start:end], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(local_k.grad, ref_k.grad[start:end], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(local_v.grad, ref_v.grad[start:end], rtol=1e-5, atol=1e-6)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


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


@pytest.mark.run_only_on("GPU")
@pytest.mark.skipif(
    not dist.is_nccl_available() or torch.cuda.device_count() < 2,
    reason="GLM DSA CP all-gather requires two CUDA devices and NCCL",
)
def test_glm_dsa_cp_all_gather_nccl_forward_backward():
    world_size = 2
    mp.spawn(_nccl_all_gather_worker, args=(world_size, _free_port()), nprocs=world_size, join=True)


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


def test_slice_thd_chunk_for_cp_preserves_explicit_padding_mask():
    chunk = _thd_chunk()
    chunk["padding_mask"] = torch.tensor([False, True, False, False, True, False])

    out = glm_cp._slice_thd_chunk_for_cp(
        chunk,
        cp_group="cp-group",
        cp_size=3,
        cp_rank=1,
        padding_token_id=3,
    )

    assert out["input_ids"].tolist() == [2, 3]
    assert out["padding_mask"].tolist() == [False, False]


def test_slice_thd_chunk_for_cp_omits_identity_query_indices_at_cp1():
    chunk = _thd_chunk()

    out = glm_cp._slice_thd_chunk_for_cp(
        chunk,
        cp_group="cp-group",
        cp_size=1,
        cp_rank=0,
        padding_token_id=3,
    )

    assert out["input_ids"].data_ptr() == chunk["input_ids"].data_ptr()
    assert "glm_dsa_cp_query_indices" not in out


def test_make_glm_dsa_packed_cp_batch_single_chunk(monkeypatch):
    captured = {}

    def fake_split(batch, **kwargs):
        captured.update(batch=batch, kwargs=kwargs)
        return _thd_chunk()

    monkeypatch.setattr(glm_cp, "split_batch_into_thd_chunks", fake_split)
    monkeypatch.setattr(glm_cp.dist, "is_available", lambda: False)
    monkeypatch.setattr(glm_cp.dist, "is_initialized", lambda: False)

    ctx, out = glm_cp.make_glm_dsa_packed_cp_batch_and_ctx(
        _FakeMesh(size=3),
        None,
        {"input_ids": torch.arange(6).view(1, 6)},
        padding_token_id=3,
        num_chunks=1,
        seq_lens_padding_value=-77,
    )

    assert ctx is contextlib.nullcontext
    assert captured["kwargs"]["num_chunks"] == 1
    assert captured["kwargs"]["seq_lens_padding_value"] == -77
    assert captured["kwargs"]["padding_token_id"] == 3
    assert out["input_ids"].tolist() == [0, 1]
    assert out["cp_size"] == 3
    assert out["cp_rank"] == 0


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

    ctx, out = glm_cp.make_glm_dsa_packed_cp_batch_and_ctx(
        _FakeMesh(size=2),
        None,
        {"input_ids": torch.arange(8).view(2, 4)},
        padding_token_id=12,
        num_chunks=2,
    )

    assert ctx is contextlib.nullcontext
    assert out["input_ids"].tolist() == [[2, 3], [12, 13]]
    assert out["labels"].tolist() == [[102, 103], [112, 113]]
    assert out["position_ids"].tolist() == [[202, 203], [212, 213]]
    assert out["glm_dsa_cp_query_indices"].tolist() == [[2, 3], [2, 3]]
    assert out["padding_mask"].tolist() == [[False, False], [True, False]]
    assert out["cp_rank"] == 1
    assert out["qkv_format"] == "thd"
