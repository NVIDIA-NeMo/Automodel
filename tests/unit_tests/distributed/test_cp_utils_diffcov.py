# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""Targeted unit tests for context-parallel helper paths in cp_utils.

These exercise the CP attention SDPA-swap hooks, the linear-attn position hook,
the classic DTensor _cp_sdpa path (DTensor mocked), padding/prepare helpers, and
the manual contiguous-shard batch builder branches that the broader suite leaves
uncovered. No GPU or real process group is required.
"""

from types import MethodType
from unittest import mock

import pytest
import torch
import torch.nn.functional as F

from nemo_automodel.components.distributed import cp_contiguous_shard as cm
from nemo_automodel.components.distributed import cp_utils as cu


class _FakeMesh:
    def __init__(self, size=2, rank=0):
        self._size = size
        self._rank = rank

    def size(self):
        return self._size

    def get_group(self):
        return object()

    def get_local_rank(self):
        return self._rank


class _Attn(torch.nn.Module):
    def forward(self, query, key, value, attention_mask=None, **kwargs):
        return F.scaled_dot_product_attention(query, key, value)


class _Wrapper(torch.nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.self_attn = attn


def _qkv(seq=4, d=8, heads=2):
    torch.manual_seed(0)
    q = torch.randn(1, heads, seq, d)
    k = torch.randn(1, heads, seq, d)
    v = torch.randn(1, heads, seq, d)
    return q, k, v


# ---------------------------------------------------------------------------
# attach_context_parallel_hooks: the _cp_uses_attention_hook fast path
# ---------------------------------------------------------------------------
def test_context_parallel_hook_uses_attention_hook_branch():
    attn = _Attn()
    model = _Wrapper(attn)
    cu.attach_context_parallel_hooks(model)
    attn._cp_uses_attention_hook = True
    captured = {}

    def spy(_m, args, kwargs):
        captured["mask"] = kwargs.get("attention_mask", "absent")
        captured["causal"] = kwargs.get("is_causal")

    attn.register_forward_pre_hook(spy, with_kwargs=True)
    q, k, v = _qkv()
    attn(q, k, v, attention_mask=torch.ones(1, 1, 4, 4))
    # the CP hook forced attention_mask=None and is_causal=True
    assert captured["mask"] is None
    assert captured["causal"] is True


# ---------------------------------------------------------------------------
# attach_cp_attention_hooks: cp_size<=1 passthrough, manual path, DTensor path
# ---------------------------------------------------------------------------
def test_cp_attention_hooks_cp_size_one_passthrough():
    attn = _Attn()
    model = _Wrapper(attn)
    cu.attach_cp_attention_hooks(model, _FakeMesh(size=1))
    q, k, v = _qkv()
    out = attn(q, k, v)
    ref = F.scaled_dot_product_attention(q, k, v)
    assert torch.allclose(out, ref)
    # SDPA must be restored after the forward (post-hook)
    assert F.scaled_dot_product_attention is not None


def test_cp_attention_hooks_manual_path():
    attn = _Attn()
    calls = {}

    def manual(self, query, key, value, *, cp_mesh, attn_mask, dropout_p, is_causal, scale, enable_gqa, kwargs):
        calls["meta"] = self._cp_manual_metadata
        calls["active"] = self._cp_manual_active
        return torch.zeros_like(query)

    attn.run_cp_manual_attention = MethodType(manual, attn)
    attn._cp_manual_metadata_keys = ("foo",)
    model = _Wrapper(attn)
    cu.attach_cp_attention_hooks(model, _FakeMesh(size=2))
    q, k, v = _qkv()
    out = attn(q, k, v, foo=torch.tensor([1, 2, 3]))
    assert torch.equal(out, torch.zeros_like(q))
    assert calls["active"] is True
    assert torch.equal(calls["meta"]["foo"], torch.tensor([1, 2, 3]))
    # post-hook cleared the per-call state
    assert attn._cp_manual_active is False
    assert attn._cp_manual_metadata == {}


def test_cp_attention_hooks_dtensor_sdpa_path():
    attn = _Attn()  # no run_cp_manual_attention -> classic DTensor _cp_sdpa
    model = _Wrapper(attn)
    q, k, v = _qkv()

    class _FakeDTensor:
        @staticmethod
        def from_local(t, device_mesh=None, placements=None):
            return t  # identity: treat local tensor as already-distributed

    # attach_cp_attention_hooks imports DTensor/Shard at attach time, so patch
    # before attaching.
    with (
        mock.patch("torch.distributed.tensor.DTensor", _FakeDTensor),
        mock.patch("torch.distributed.tensor.Shard", lambda d: ("shard", d)),
    ):
        cu.attach_cp_attention_hooks(model, _FakeMesh(size=2))
        out = attn(q, k, v)
    ref = F.scaled_dot_product_attention(q, k, v)
    assert torch.allclose(out, ref)


def test_cp_attention_hooks_restores_sdpa_on_exception():
    class _BoomAttn(torch.nn.Module):
        def forward(self, query, key, value, **kwargs):
            F.scaled_dot_product_attention  # trigger nothing; just raise after swap
            raise RuntimeError("boom")

    original = F.scaled_dot_product_attention
    attn = _BoomAttn()
    model = _Wrapper(attn)
    cu.attach_cp_attention_hooks(model, _FakeMesh(size=2))
    q, k, v = _qkv()
    with pytest.raises(RuntimeError, match="boom"):
        attn(q, k, v)
    # always_call post-hook restored the real SDPA
    assert F.scaled_dot_product_attention is original


# ---------------------------------------------------------------------------
# attach_linear_attn_position_hooks
# ---------------------------------------------------------------------------
def test_linear_attn_position_hook_caches_position_ids():
    class _LinAttn(torch.nn.Module):
        def forward(self, x):
            return x

    class _DecoderLayer(torch.nn.Module):
        layer_type = "linear_attention"

        def __init__(self):
            super().__init__()
            self.linear_attn = _LinAttn()

        def forward(self, x, position_ids=None):
            return self.linear_attn(x)

    layer = _DecoderLayer()
    model = torch.nn.Module()
    model.add_module("layer", layer)
    cu.attach_linear_attn_position_hooks(model)
    assert layer._linear_attn_pos_hook_registered is True
    pos = torch.arange(4).unsqueeze(0)
    layer(torch.randn(1, 4), position_ids=pos)
    assert torch.equal(layer.linear_attn._cached_position_ids, pos)
    # idempotent: re-attaching does not double-register
    cu.attach_linear_attn_position_hooks(model)


# ---------------------------------------------------------------------------
# _pad_tensor_seq_dim_ / _pad_position_ids_seq_dim_ pad_len<=0 no-ops
# ---------------------------------------------------------------------------
def test_pad_tensor_noop_when_pad_len_zero():
    t = torch.randn(2, 4)
    assert cm._pad_tensor_seq_dim_(t, 1, 0) is t


def test_pad_position_ids_noop_when_pad_len_zero():
    p = torch.arange(4).unsqueeze(0)
    assert cm._pad_position_ids_seq_dim_(p, 1, 0) is p


def test_pad_position_ids_extends_monotonically():
    p = torch.tensor([[0, 1, 2, 3]])
    out = cm._pad_position_ids_seq_dim_(p, 1, 2)
    assert out.tolist() == [[0, 1, 2, 3, 4, 5]]


# ---------------------------------------------------------------------------
# _prepare_cp_batch_common branches
# ---------------------------------------------------------------------------
def test_prepare_cp_batch_common_builds_padding_mask_from_4d_mask():
    mask = torch.ones(1, 1, 4, 4, dtype=torch.bool)
    batch = {
        "input_ids": torch.zeros(1, 4, dtype=torch.long),
        "labels": torch.zeros(1, 4, dtype=torch.long),
        "attention_mask": mask,
    }
    cu._prepare_cp_batch_common(_FakeMesh(size=2), None, batch, None)
    assert "padding_mask" in batch and batch["padding_mask"].shape == (1, 4)


def test_prepare_cp_batch_common_labels_from_loss_mask():
    batch = {"input_ids": torch.zeros(1, 4, dtype=torch.long)}
    loss_mask = torch.ones(1, 4, dtype=torch.long)
    res = cu._prepare_cp_batch_common(_FakeMesh(size=2), None, batch, loss_mask)
    labels = res[3]
    assert torch.equal(labels, loss_mask)


def test_prepare_cp_batch_common_raises_without_labels():
    batch = {"input_ids": torch.zeros(1, 4, dtype=torch.long)}
    with pytest.raises(KeyError, match="labels"):
        cu._prepare_cp_batch_common(_FakeMesh(size=2), None, batch, None)


# ---------------------------------------------------------------------------
# _make_contiguous_shard_cp_batch: pad branches for every sequence key + extra
# metadata, plus the contiguous slice.
# ---------------------------------------------------------------------------
def test_make_contiguous_shard_pads_and_slices_all_keys():
    cp_size = 2
    seq = 6  # pad_len = (-6) % 4 = 2 -> padded to 8, local 4
    d = 8
    batch = {
        "inputs_embeds": torch.randn(1, seq, d),
        "mm_token_type_ids": torch.zeros(1, seq, dtype=torch.long),
        "per_layer_inputs": torch.randn(1, seq, d),
        "padding_mask": torch.zeros(1, seq, dtype=torch.bool),
        "vision_extra": torch.zeros(1, seq, dtype=torch.long),
        "_cp_metadata_seq_dims": {"vision_extra": 1},
        "_cp_metadata_pad_values": {"vision_extra": 0},
    }
    labels = torch.zeros(1, seq, dtype=torch.long)
    position_ids = torch.arange(seq).unsqueeze(0)
    loss_mask = torch.ones(1, seq, dtype=torch.long)

    _ctx, out = cm._make_contiguous_shard_cp_batch(
        _FakeMesh(size=cp_size, rank=0),
        batch,
        primary_key="inputs_embeds",
        seq_len=seq,
        labels=labels,
        position_ids=position_ids,
        pos_seq_dim=1,
        loss_mask=loss_mask,
        padding_token_id=0,
    )
    # rank 0 keeps the first local_seq_len=4 positions
    assert out["inputs_embeds"].shape == (1, 4, d)
    assert out["labels"].shape == (1, 4)
    assert out["per_layer_inputs"].shape == (1, 4, d)
    assert out["padding_mask"].shape == (1, 4)
    assert out["vision_extra"].shape == (1, 4)
    assert out["loss_mask"].shape == (1, 4)
    assert out["_packed_seq_ids"].shape == (1, 4)


def test_make_contiguous_shard_uses_dist_rank_when_initialized():
    batch = {"input_ids": torch.zeros(1, 4, dtype=torch.long)}
    labels = torch.zeros(1, 4, dtype=torch.long)
    position_ids = torch.arange(4).unsqueeze(0)
    with (
        mock.patch("torch.distributed.is_available", return_value=True),
        mock.patch("torch.distributed.is_initialized", return_value=True),
        mock.patch("torch.distributed.get_rank", return_value=0),
    ):
        _ctx, out = cm._make_contiguous_shard_cp_batch(
            _FakeMesh(size=2, rank=0),
            batch,
            primary_key="input_ids",
            seq_len=4,
            labels=labels,
            position_ids=position_ids,
            pos_seq_dim=1,
            loss_mask=None,
            padding_token_id=0,
        )
    assert out["input_ids"].shape == (1, 2)


def test_make_contiguous_shard_raises_when_not_divisible(monkeypatch):
    # Force padding to be a no-op so the post-pad divisibility check trips.
    monkeypatch.setattr(cm, "_pad_tensor_seq_dim_", lambda t, *a, **k: t)
    monkeypatch.setattr(cm, "_pad_position_ids_seq_dim_", lambda p, *a, **k: p)
    batch = {"input_ids": torch.zeros(1, 6, dtype=torch.long)}
    labels = torch.zeros(1, 6, dtype=torch.long)
    position_ids = torch.arange(6).unsqueeze(0)
    with pytest.raises(ValueError, match="divisible by cp_size"):
        cm._make_contiguous_shard_cp_batch(
            _FakeMesh(size=4, rank=0),
            batch,
            primary_key="input_ids",
            seq_len=6,
            labels=labels,
            position_ids=position_ids,
            pos_seq_dim=1,
            loss_mask=None,
            padding_token_id=0,
        )


# ---------------------------------------------------------------------------
# make_cp_batch_and_ctx use_te routing
# ---------------------------------------------------------------------------
def test_make_cp_batch_and_ctx_use_te_routes_to_te_builder():
    sentinel = {"te": True}

    class _DM(dict):
        mesh_dim_names = ["cp"]

    dm = _DM(cp=_FakeMesh(size=2))
    with mock.patch.object(cu, "make_cp_batch_for_te", return_value=sentinel) as te:
        ctx, out = cu.make_cp_batch_and_ctx(dm, {"input_ids": torch.zeros(1, 4, dtype=torch.long)}, use_te=True)
    te.assert_called_once()
    assert out is sentinel
