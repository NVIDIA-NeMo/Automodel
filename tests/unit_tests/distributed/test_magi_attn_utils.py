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

"""Unit tests for the MagiAttention integration helpers.

The mask-spec builders and the MagiState/setup_magi wiring are pure Python and
need neither a GPU nor the optional ``magi_attention`` package, so they run on a
CPU CI runner. The FFA kernel parity tests at the bottom are gated behind CUDA +
``magi_attention``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import nemo_automodel.components.distributed.magi_attn_utils as mu
from nemo_automodel.components.distributed.magi_attn_utils import AttnMaskSpec, MagiState, setup_magi


class _FakeCfg:
    """Minimal stand-in for the recipe ConfigNode (dotted ``.get``)."""

    def __init__(self, values: dict):
        self._values = values

    def get(self, key, default=None):
        return self._values.get(key, default)


class _FakeGroup:
    def __init__(self, size):
        self._size = size

    def size(self):
        return self._size


class _FakeMesh:
    """Stand-in for a DeviceMesh exposing ``mesh_dim_names`` and ``mesh["cp"]``."""

    def __init__(self, dim_names, group=None):
        self.mesh_dim_names = dim_names
        self._group = group

    def __getitem__(self, name):
        assert name == "cp"
        return SimpleNamespace(get_group=lambda: self._group)


class _FakeAttention(nn.Module):
    """Module whose class name contains 'Attention' (matched by the stampers)."""


class _LM(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _FakeAttention()


class _VLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = _LM()
        self.visual = nn.Linear(4, 4)  # vision tower: must NOT be stamped


# --------------------------------------------------------------------------- #
# AttnMaskSpec builders (pure Python)
# --------------------------------------------------------------------------- #
class TestAttnMaskSpec:
    def test_causal(self):
        spec = AttnMaskSpec.causal(10)
        assert spec.q_ranges == [[0, 10]]
        assert spec.k_ranges == [[0, 10]]
        assert spec.mask_types == ["causal"]
        assert spec.total_seqlen == 10

    def test_varlen_causal(self):
        spec = AttnMaskSpec.varlen([3, 5, 2], causal=True)
        assert spec.q_ranges == [[0, 3], [3, 8], [8, 10]]
        assert spec.k_ranges == [[0, 3], [3, 8], [8, 10]]
        assert spec.mask_types == ["causal", "causal", "causal"]
        assert spec.total_seqlen == 10

    def test_varlen_full(self):
        spec = AttnMaskSpec.varlen([4, 4], causal=False)
        assert spec.mask_types == ["full", "full"]

    def test_prefix_tree_shared_prefix(self):
        # flat layout [node0 | node1 | node2], samples S0={n0,n1}, S1={n0,n2}.
        node_lengths = [4, 2, 3]
        spec, sample_ranges = AttnMaskSpec.prefix_tree(node_lengths, [[0, 1], [0, 2]])
        # each node attends CAUSAL to itself and FULL to its ancestors; deduped.
        assert spec.q_ranges == [[0, 4], [4, 6], [4, 6], [6, 9], [6, 9]]
        assert spec.k_ranges == [[0, 4], [4, 6], [0, 4], [6, 9], [0, 4]]
        assert spec.mask_types == ["causal", "causal", "full", "causal", "full"]
        assert spec.total_seqlen == 9
        # the shared prefix (node0) self-rectangle appears exactly once.
        assert spec.q_ranges.count([0, 4]) == 1
        # per-sample reconstruction ranges (prefix + own leaf).
        assert sample_ranges == [[[0, 4], [4, 6]], [[0, 4], [6, 9]]]

    def test_fingerprint_distinguishes_masks(self):
        a = AttnMaskSpec.causal(8)
        b = AttnMaskSpec.varlen([4, 4])
        assert a.fingerprint() != b.fingerprint()
        assert a.fingerprint() == AttnMaskSpec.causal(8).fingerprint()


# --------------------------------------------------------------------------- #
# MagiState
# --------------------------------------------------------------------------- #
class TestMagiState:
    def test_defaults_disabled(self):
        st = MagiState()
        assert st.enabled is False
        assert st.custom is False
        assert st.cp_group is None
        assert st.cp_size == 1
        assert st.hf_dispatch is False

    @pytest.mark.parametrize(
        "enabled,custom,expected",
        [(True, False, True), (True, True, False), (False, False, False), (False, True, False)],
    )
    def test_hf_dispatch(self, enabled, custom, expected):
        assert MagiState(enabled=enabled, custom=custom).hf_dispatch is expected

    def test_undispatch_logits_identity_when_not_hf_dispatch(self):
        sentinel = object()
        # disabled and custom paths never undispatch -> identity (no magi import needed).
        assert MagiState().undispatch_logits(sentinel) is sentinel
        assert MagiState(enabled=True, custom=True).undispatch_logits(sentinel) is sentinel


# --------------------------------------------------------------------------- #
# setup_magi
# --------------------------------------------------------------------------- #
class TestSetupMagi:
    def test_disabled_when_not_configured(self):
        st = setup_magi(_FakeCfg({}), device_mesh=None)
        assert st.enabled is False and st.custom is False and st.cp_size == 1

    def test_hf_backend_enabled(self, monkeypatch):
        calls = {"register": 0, "active": []}
        monkeypatch.setattr(mu, "register_magi_attention", lambda: calls.__setitem__("register", calls["register"] + 1))
        monkeypatch.setattr(mu, "get_cp_group", lambda mesh: _FakeGroup(1))
        monkeypatch.setattr(mu, "set_active_cp_group", lambda g: calls["active"].append(g))

        st = setup_magi(_FakeCfg({"model.attn_implementation": "magi"}), device_mesh=object())
        assert st.enabled and not st.custom and st.hf_dispatch
        assert st.cp_size == 1
        assert calls["register"] == 1
        assert calls["active"] == []  # HF path does not set the active cp_group

    def test_custom_backend_enabled_sets_active_group(self, monkeypatch):
        grp = _FakeGroup(2)
        active = []
        monkeypatch.setattr(mu, "register_magi_attention", lambda: None)
        monkeypatch.setattr(mu, "get_cp_group", lambda mesh: grp)
        monkeypatch.setattr(mu, "set_active_cp_group", lambda g: active.append(g))

        st = setup_magi(_FakeCfg({"model.backend.attn": "magi"}), device_mesh=object())
        assert st.enabled and st.custom and not st.hf_dispatch
        assert st.cp_size == 2
        assert active == [grp]  # custom path wires the active cp_group


# --------------------------------------------------------------------------- #
# CPU-coverable helpers (no GPU / no magi_attention package)
# --------------------------------------------------------------------------- #
class TestGetCpGroup:
    def test_none_mesh(self):
        assert mu.get_cp_group(None) is None

    def test_mesh_without_cp_dim(self):
        assert mu.get_cp_group(_FakeMesh(("dp", "tp"))) is None

    def test_mesh_with_cp_dim(self):
        grp = object()
        assert mu.get_cp_group(_FakeMesh(("dp", "cp"), grp)) is grp


class TestGetHeadConfig:
    def test_plain_llm_config(self):
        cfg = SimpleNamespace(num_attention_heads=32, num_key_value_heads=8, head_dim=128)
        assert mu._get_head_config(SimpleNamespace(config=cfg)) == (32, 8, 128)

    def test_head_dim_derived_from_hidden_size(self):
        cfg = SimpleNamespace(num_attention_heads=16, hidden_size=2048, head_dim=None)
        # no num_key_value_heads -> falls back to num_attention_heads; head_dim = 2048/16.
        assert mu._get_head_config(SimpleNamespace(config=cfg)) == (16, 16, 128)

    def test_vlm_uses_text_config(self):
        text = SimpleNamespace(num_attention_heads=24, num_key_value_heads=4, head_dim=64)
        outer = SimpleNamespace(text_config=text)  # no top-level num_attention_heads
        assert mu._get_head_config(SimpleNamespace(config=outer)) == (24, 4, 64)


class TestStampCpGroup:
    def test_set_cp_group_on_all_attention(self):
        model = _VLM()
        sentinel = object()
        mu._set_cp_group_on_attention(model, sentinel)
        # every "*Attention" module is stamped, regardless of subtree.
        assert model.language_model.self_attn.cp_group is sentinel
        assert not hasattr(model.visual, "cp_group")

    def test_iter_language_model_attention_skips_vision(self):
        model = _VLM()
        mods = list(mu._iter_language_model_attention(model))
        assert mods == [model.language_model.self_attn]


class TestMagiPrepareVlm:
    """magi_prepare_vlm is pure Python (no magi import) for the cp_size==1 path."""

    def test_stamps_language_backbone_only(self):
        model = _VLM()
        batch = {"input_ids": torch.zeros(1, 8, dtype=torch.long)}
        out_batch, key = mu.magi_prepare_vlm(model, batch, cp_group=None)
        assert key is None
        assert out_batch is batch
        attn = model.language_model.self_attn
        assert attn.cp_group is None and attn._magi_self_key is True
        assert not hasattr(model.visual, "cp_group")

    def test_rejects_batch_dim_gt_1(self):
        model = _VLM()
        with pytest.raises(ValueError, match=r"\[1, S\]"):
            mu.magi_prepare_vlm(model, {"input_ids": torch.zeros(2, 8, dtype=torch.long)}, cp_group=None)

    def test_rejects_cp_gt_1(self):
        model = _VLM()
        with pytest.raises(NotImplementedError, match="cp_size==1"):
            mu.magi_prepare_vlm(model, {"input_ids": torch.zeros(1, 8, dtype=torch.long)}, cp_group=_FakeGroup(2))


class TestMagiStateVlmBatch:
    def test_prepare_vlm_batch_non_custom_stamps_backbone(self):
        model = _VLM()
        st = MagiState(enabled=True, custom=False, cp_group=None, cp_size=1)
        train_ctx, batch = st.prepare_vlm_batch(model, {"input_ids": torch.zeros(1, 4, dtype=torch.long)})
        from contextlib import nullcontext

        assert train_ctx is nullcontext
        assert model.language_model.self_attn._magi_self_key is True

    def test_prepare_vlm_batch_custom_is_noop(self):
        # custom VLMs use the factory attn_func (active cp_group); no stamping here.
        model = _VLM()
        st = MagiState(enabled=True, custom=True)
        _, batch = st.prepare_vlm_batch(model, {"input_ids": torch.zeros(1, 4, dtype=torch.long)})
        assert not hasattr(model.language_model.self_attn, "_magi_self_key")


class TestPackedCpDocSeqlens:
    """Regression for the packed-CP key spanning the padded THD layout.

    The custom-model CP packed path must build its dist key from
    ``cu_seqlens_padded`` (spans the flat input) not ``cu_seqlens`` (real tokens
    only); otherwise the dispatched shard length disagrees with get_position_ids
    and the model hits a RoPE q vs cos/sin length mismatch.
    """

    def test_prefers_cu_seqlens_padded(self):
        # padded layout sums to 1024 (matches input); real cu_seqlens sums to 944.
        batch = {
            "cu_seqlens": torch.tensor([0, 400, 944]),
            "cu_seqlens_padded": torch.tensor([0, 440, 1024]),
        }
        assert mu._packed_cp_doc_seqlens(batch, 1024) == [440, 584]

    def test_falls_back_to_cu_seqlens(self):
        batch = {"cu_seqlens": torch.tensor([0, 300, 1024])}
        assert mu._packed_cp_doc_seqlens(batch, 1024) == [300, 724]

    def test_drops_padding_sentinels(self):
        batch = {"cu_seqlens_padded": torch.tensor([0, 512, 1024, -1000, -1000])}
        assert mu._packed_cp_doc_seqlens(batch, 1024) == [512, 512]

    def test_raises_when_layout_mismatches_input(self):
        # cu_seqlens (944) would not span the 1024 flat input -> guard fires.
        batch = {"cu_seqlens": torch.tensor([0, 400, 944])}
        with pytest.raises(ValueError, match="!= flat input length 1024"):
            mu._packed_cp_doc_seqlens(batch, 1024)


class TestActiveStateAccessors:
    def test_attn_spec_roundtrip(self):
        spec = AttnMaskSpec.causal(4)
        mu.set_active_attn_spec(spec)
        try:
            assert mu.get_active_attn_spec() is spec
        finally:
            mu.set_active_attn_spec(None)
        assert mu.get_active_attn_spec() is None

    def test_cp_group_roundtrip(self):
        grp = object()
        mu.set_active_cp_group(grp)
        try:
            assert mu.get_active_cp_group() is grp
        finally:
            mu.set_active_cp_group(None)
        assert mu.get_active_cp_group() is None

    def test_is_magi_available_returns_bool(self):
        assert isinstance(mu.is_magi_available(), bool)


# --------------------------------------------------------------------------- #
# FFA kernel parity (requires CUDA + magi_attention)
# --------------------------------------------------------------------------- #
def _init_single_rank_cp_group():
    import os

    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29591")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", rank=0, world_size=1)
    return dist.new_group([0], backend="nccl")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_make_magi_attn_func_matches_sdpa_causal():
    """The custom-factory attn_func (cp=1 causal self-key) matches SDPA causal."""
    pytest.importorskip("magi_attention")
    import torch.distributed as dist
    import torch.nn.functional as F
    from einops import rearrange

    torch.cuda.set_device(0)
    cp_group = _init_single_rank_cp_group()
    try:
        mu.set_active_cp_group(cp_group)
        mu.set_active_attn_spec(None)
        nh_q, nh_kv, hd, S = 16, 4, 128, 384
        torch.manual_seed(0)
        q = torch.randn(S, nh_q, hd, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(S, nh_kv, hd, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(S, nh_kv, hd, device="cuda", dtype=torch.bfloat16)

        attn = mu.make_magi_attn_func(softmax_scale=hd**-0.5)
        out = attn(q, k, v)  # thd layout [S, nh, hd]

        qh = rearrange(q, "s nh hd -> 1 nh s hd")
        kh = rearrange(k, "s nh hd -> 1 nh s hd").repeat_interleave(nh_q // nh_kv, 1)
        vh = rearrange(v, "s nh hd -> 1 nh s hd").repeat_interleave(nh_q // nh_kv, 1)
        ref = rearrange(F.scaled_dot_product_attention(qh, kh, vh, is_causal=True), "1 nh s hd -> s nh hd")

        rel = (out.float() - ref.float()).abs().max() / (ref.float().abs().max() + 1e-6)
        assert rel.item() < 5e-3, f"magi vs sdpa rel={rel.item()}"
        assert not torch.isnan(out).any()
    finally:
        mu.set_active_cp_group(None)
        if dist.is_initialized():
            dist.destroy_process_group()
