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

import pytest
import torch

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
