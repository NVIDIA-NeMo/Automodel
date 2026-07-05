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

"""CPU unit tests for DeepSeek V4 model-owned context-parallel batch prep.

Covers the model-owned CP path that runs without a real process group:
``make_dsv4_contiguous_shard_cp_batch_and_ctx`` (the ``_cp_make_batch_fn``
callable), the scalar group helpers, ``dsv4_cp_local_seq_multiple``, and the
``DeepseekV4ForCausalLM`` CP-prep hook (``prepare_model_inputs_for_cp`` and the
``_pre_embed_only`` forward branch).
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import pytest
import torch

from nemo_automodel.components.models.deepseek_v4 import cp as cpmod
from nemo_automodel.components.models.deepseek_v4.cp import (
    dsv4_cp_enabled,
    dsv4_cp_local_seq_multiple,
    dsv4_cp_rank,
    dsv4_cp_size,
    make_dsv4_contiguous_shard_cp_batch_and_ctx,
)
from nemo_automodel.components.models.deepseek_v4.model import DeepseekV4ForCausalLM


@pytest.fixture(autouse=True)
def _force_no_dist(monkeypatch):
    """Exercise the single-process (no process group) path deterministically.

    These tests drive the CP helpers/callable with a fake mesh whose ``get_group``
    returns a sentinel, not a real ProcessGroup. If another test in the same pytest
    worker left ``torch.distributed`` initialized, the helpers would otherwise call
    ``dist.get_rank(group=<sentinel>)`` and raise. Pin ``is_initialized`` to False so
    rank resolution falls back to ``cp_mesh.get_local_rank()``.
    """
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)


class _FakeMesh:
    """Minimal stand-in for a CP device-mesh slice (no real process group)."""

    def __init__(self, size: int, local_rank: int = 0, group: object = "cp_group_sentinel"):
        self._size = size
        self._local_rank = local_rank
        self._group = group

    def size(self) -> int:
        return self._size

    def get_local_rank(self) -> int:
        return self._local_rank

    def get_group(self) -> object:
        return self._group


def _shard(batch, *, cp_size, local_rank, **kwargs):
    mesh = _FakeMesh(cp_size, local_rank)
    return make_dsv4_contiguous_shard_cp_batch_and_ctx(mesh, None, batch, **kwargs)


# --------------------------------------------------------------------------- #
# Scalar group helpers (no dist initialized -> degenerate single-rank values)  #
# --------------------------------------------------------------------------- #
def test_group_helpers_without_dist():
    assert dsv4_cp_enabled(None) is False
    assert dsv4_cp_enabled("anything") is False
    assert dsv4_cp_rank(None) == 0
    assert dsv4_cp_size(None) == 1


# --------------------------------------------------------------------------- #
# dsv4_cp_local_seq_multiple                                                    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "ratios,expected",
    [
        (None, 1),  # no ratios configured
        ([], 1),
        ([0, 0], 1),  # only non-positive ratios -> filtered out
        ([2], 2),  # plain ratio
        ([4], 8),  # ratio-4 layers need 2*ratio for the cross-window overlap
        ([0, 4, 128], 128),  # lcm(8, 128)
        ([2, 3], 6),  # lcm of plain ratios
    ],
)
def test_local_seq_multiple(ratios, expected):
    cfg = SimpleNamespace(compress_ratios=ratios)
    assert dsv4_cp_local_seq_multiple(cfg) == expected
    # also accepts an object carrying `.config`
    assert dsv4_cp_local_seq_multiple(SimpleNamespace(config=cfg)) == expected


# --------------------------------------------------------------------------- #
# make_dsv4_contiguous_shard_cp_batch_and_ctx                                   #
# --------------------------------------------------------------------------- #
def test_contiguous_shard_basic_input_ids():
    seq = 8
    batch = {
        "input_ids": torch.arange(seq).view(1, seq),
        "labels": torch.arange(seq).view(1, seq),
    }
    ctx, out = _shard(batch, cp_size=2, local_rank=1)
    # context manager factory is the nullcontext class (instantiated by the caller)
    assert ctx is contextlib.nullcontext
    # divisor = cp_size * max(pad_multiple or 2, 2) = 4; seq=8 already divisible -> no pad.
    # local_seq = 8 / 2 = 4; rank 1 owns [4:8].
    assert out["input_ids"].shape == (1, 4)
    torch.testing.assert_close(out["input_ids"], torch.arange(4, 8).view(1, 4))
    torch.testing.assert_close(out["labels"], torch.arange(4, 8).view(1, 4))
    # position_ids were synthesized over the global sequence then sharded.
    torch.testing.assert_close(out["position_ids"], torch.arange(4, 8).view(1, 4))
    # the CP process group is handed to the forward.
    assert out["_dsv4_cp_group"] == "cp_group_sentinel"


def test_contiguous_shard_rank0_slice():
    batch = {"input_ids": torch.arange(8).view(1, 8), "labels": torch.arange(8).view(1, 8)}
    _, out = _shard(batch, cp_size=2, local_rank=0)
    torch.testing.assert_close(out["input_ids"], torch.arange(0, 4).view(1, 4))


def test_contiguous_shard_pads_to_divisor():
    # seq=5, cp_size=2, pad_multiple=2 -> divisor=2*max(2,2)=4 -> pad to 8.
    batch = {"input_ids": torch.arange(5).view(1, 5), "labels": torch.arange(5).view(1, 5)}
    _, out = _shard(batch, cp_size=2, local_rank=0, pad_multiple=2)
    assert out["input_ids"].shape == (1, 4)  # padded global 8 // cp_size 2
    # labels pad uses ignore_index -100
    batch2 = {"input_ids": torch.arange(5).view(1, 5), "labels": torch.arange(5).view(1, 5)}
    _, out2 = _shard(batch2, cp_size=2, local_rank=1, pad_multiple=2)
    assert (out2["labels"] == -100).any()


def test_contiguous_shard_pad_multiple_controls_shard_size():
    # pad_multiple=4 -> divisor=cp_size*max(4,2)=8; seq=8 already divisible.
    batch = {"input_ids": torch.arange(8).view(1, 8), "labels": torch.arange(8).view(1, 8)}
    _, out = _shard(batch, cp_size=2, local_rank=0, pad_multiple=4)
    assert out["input_ids"].shape == (1, 4)
    # seq=8, pad_multiple=8 -> divisor=16 -> pad to 16, local=8.
    batch2 = {"input_ids": torch.arange(8).view(1, 8), "labels": torch.arange(8).view(1, 8)}
    _, out2 = _shard(batch2, cp_size=2, local_rank=0, pad_multiple=8)
    assert out2["input_ids"].shape == (1, 8)


def test_contiguous_shard_attention_mask_2d_to_padding_mask():
    seq = 8
    attn = torch.ones(1, seq, dtype=torch.long)
    attn[0, -2:] = 0  # last two are padding
    batch = {
        "input_ids": torch.arange(seq).view(1, seq),
        "labels": torch.arange(seq).view(1, seq),
        "attention_mask": attn,
    }
    _, out = _shard(batch, cp_size=2, local_rank=1)
    assert "attention_mask" not in out  # consumed
    # rank 1 owns [4:8]; padding_mask True == pad on the last two positions.
    torch.testing.assert_close(out["padding_mask"], torch.tensor([[False, False, True, True]]))


def test_contiguous_shard_attention_mask_4d_to_padding_mask():
    seq = 4
    # 4D additive mask: diagonal 0 == attend, nonzero (e.g. -inf penalty) == padded.
    attn = torch.zeros(1, 1, seq, seq)
    attn[0, 0, range(seq), range(seq)] = torch.tensor([0.0, 0.0, -1e9, -1e9])
    batch = {
        "input_ids": torch.arange(seq).view(1, seq),
        "labels": torch.arange(seq).view(1, seq),
        "attention_mask": attn,
    }
    _, out = _shard(batch, cp_size=1, local_rank=0)
    torch.testing.assert_close(out["padding_mask"], torch.tensor([[False, False, True, True]]))


def test_contiguous_shard_attention_mask_4d_bool():
    seq = 4
    # 4D boolean mask: diagonal True == attend, so logical_not() == padded.
    attn = torch.zeros(1, 1, seq, seq, dtype=torch.bool)
    attn[0, 0, range(seq), range(seq)] = torch.tensor([True, True, True, False])
    batch = {
        "input_ids": torch.arange(seq).view(1, seq),
        "labels": torch.arange(seq).view(1, seq),
        "attention_mask": attn,
    }
    _, out = _shard(batch, cp_size=1, local_rank=0)
    torch.testing.assert_close(out["padding_mask"], torch.tensor([[False, False, False, True]]))


def test_contiguous_shard_inputs_embeds_path():
    seq, hidden = 8, 3
    batch = {
        "inputs_embeds": torch.randn(1, seq, hidden),
        "labels": torch.arange(seq).view(1, seq),
    }
    _, out = _shard(batch, cp_size=2, local_rank=0)
    assert "inputs_embeds" in out and out["inputs_embeds"].shape == (1, 4, hidden)
    assert "input_ids" not in out


def test_contiguous_shard_loss_mask_becomes_labels_when_labels_absent():
    seq = 8
    batch = {"input_ids": torch.arange(seq).view(1, seq)}
    loss_mask = torch.ones(1, seq)
    _, out = _shard(batch, cp_size=2, local_rank=0, loss_mask=loss_mask)
    # loss_mask was promoted to labels then sharded.
    assert out["labels"].shape == (1, 4)


def test_contiguous_shard_loss_mask_kept_alongside_labels():
    seq = 8
    batch = {"input_ids": torch.arange(seq).view(1, seq), "labels": torch.arange(seq).view(1, seq)}
    loss_mask = torch.ones(1, seq)
    _, out = _shard(batch, cp_size=2, local_rank=1, loss_mask=loss_mask)
    assert out["loss_mask"].shape == (1, 4)


def test_contiguous_shard_position_ids_3d():
    seq = 8
    pos = torch.arange(seq).view(1, 1, seq).expand(1, 3, seq).contiguous()
    batch = {"input_ids": torch.arange(seq).view(1, seq), "labels": torch.arange(seq).view(1, seq), "position_ids": pos}
    _, out = _shard(batch, cp_size=2, local_rank=1)
    assert out["position_ids"].shape == (1, 3, 4)
    torch.testing.assert_close(out["position_ids"][0, 0], torch.arange(4, 8))


def test_contiguous_shard_packed_sequence_guards():
    base = {"input_ids": torch.arange(8).view(1, 8), "labels": torch.arange(8).view(1, 8)}
    with pytest.raises(NotImplementedError, match="packed sequences"):
        _shard({**base, "cu_seqlens": torch.tensor([0, 8])}, cp_size=2, local_rank=0)
    with pytest.raises(NotImplementedError, match="packed sequences"):
        _shard({**base, "qkv_format": "thd"}, cp_size=2, local_rank=0)


def test_contiguous_shard_requires_exactly_one_primary_key():
    # both input_ids and inputs_embeds -> assertion
    with pytest.raises(AssertionError):
        _shard(
            {
                "input_ids": torch.arange(8).view(1, 8),
                "inputs_embeds": torch.randn(1, 8, 2),
                "labels": torch.arange(8).view(1, 8),
            },
            cp_size=2,
            local_rank=0,
        )
    # neither -> assertion
    with pytest.raises(AssertionError):
        _shard({"labels": torch.arange(8).view(1, 8)}, cp_size=2, local_rank=0)


def test_contiguous_shard_requires_labels():
    with pytest.raises(KeyError, match="labels"):
        _shard({"input_ids": torch.arange(8).view(1, 8)}, cp_size=2, local_rank=0)


# --------------------------------------------------------------------------- #
# DeepseekV4ForCausalLM CP-prep hook                                           #
# --------------------------------------------------------------------------- #
def test_prepare_model_inputs_for_cp_returns_make_batch_fn_and_flag():
    from nemo_automodel.components.distributed.cp_sharder import full_logits_grad_touch

    # The method only reads self.config, so a lightweight stand-in suffices.
    cfg = SimpleNamespace(compress_ratios=[0, 4, 128])
    fake_self = SimpleNamespace(config=cfg)
    prepared = DeepseekV4ForCausalLM.prepare_model_inputs_for_cp(fake_self, input_ids=torch.arange(8).view(1, 8))

    sharder = prepared["cp_sharder"]
    assert sharder.finalize_loss_fn is full_logits_grad_touch
    assert sharder.layout == "contiguous"
    fn = sharder.shard_batch
    # the partial binds the config-derived per-rank multiple (lcm(8,128) == 128)
    assert fn.keywords["pad_multiple"] == 128
    assert fn.func is make_dsv4_contiguous_shard_cp_batch_and_ctx

    # the bound fn shards a batch end-to-end with a real (fake-mesh) divisor.
    batch = {"input_ids": torch.arange(256).view(1, 256), "labels": torch.arange(256).view(1, 256)}
    _, out = fn(_FakeMesh(2, 0), None, batch)
    assert out["input_ids"].shape == (1, 128)


def test_forward_pre_embed_only_branch_delegates_to_prepare():
    # forward()'s first statement short-circuits to prepare_model_inputs_for_cp
    # before any model compute, so a fake self exercises it without a build.
    cfg = SimpleNamespace(compress_ratios=[4])
    fake_self = SimpleNamespace(config=cfg)
    fake_self.prepare_model_inputs_for_cp = lambda input_ids: DeepseekV4ForCausalLM.prepare_model_inputs_for_cp(
        fake_self, input_ids=input_ids
    )
    out = DeepseekV4ForCausalLM.forward(fake_self, torch.arange(8).view(1, 8), _pre_embed_only=True)
    assert out["cp_sharder"].finalize_loss_fn is not None
    assert out["cp_sharder"].shard_batch.keywords["pad_multiple"] == 8


def test_setup_cp_attention_stores_group():
    from nemo_automodel.components.models.deepseek_v4.layers import DeepseekV4Attention

    fake_attn = SimpleNamespace()
    DeepseekV4Attention.setup_cp_attention(fake_attn, _FakeMesh(2, 0, group="grp"))
    assert fake_attn._cp_group == "grp"


def test_module_exposes_pad_helper_noops():
    from nemo_automodel.components.distributed import cp_sharder

    # pad_len <= 0 is a no-op (returns the same tensor object) for both pad helpers.
    t = torch.arange(6).view(1, 6)
    assert cp_sharder._pad_tensor_seq_dim_(t, 1, 0, 0) is t
    pos = torch.arange(6).view(1, 6)
    assert cp_sharder._pad_position_ids_seq_dim_(pos, 1, 0) is pos
