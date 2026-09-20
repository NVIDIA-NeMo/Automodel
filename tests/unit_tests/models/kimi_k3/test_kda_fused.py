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

"""Kimi-K3 ``kda_chunk_impl = "fused"``: config knob, support checks, routing and the autograd wrapper (CPU, kernels stubbed)."""

from __future__ import annotations

import sys
import types

import pytest
import torch
from torch import nn

from nemo_automodel.components.models.kimi_k3 import kda_fused
from nemo_automodel.components.models.kimi_k3 import model as kmod
from nemo_automodel.components.models.kimi_k3.config import KimiK3TextConfig

pytest.importorskip("fla")


def _small_config(**overrides) -> KimiK3TextConfig:
    kwargs = dict(
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        linear_attn_config={
            "head_dim": 16,
            "num_heads": 4,
            "short_conv_kernel_size": 4,
            "kda_layers": [1, 2, 3],
            "full_attn_layers": [4],
        },
        kda_use_fused_gate=False,  # CPU path for the gate in these stubbed-kernel tests
    )
    kwargs.update(overrides)
    return KimiK3TextConfig(**kwargs)


def _k3_shaped(T=256, H=2, docs=(0, 128, 256)):
    q = torch.randn(1, T, H, 128, dtype=torch.bfloat16)
    k = torch.randn(1, T, H, 128, dtype=torch.bfloat16)
    v = torch.randn(1, T, H, 128, dtype=torch.bfloat16)
    g = -torch.rand(1, T, H, 128, dtype=torch.float32)
    beta = torch.rand(1, T, H, dtype=torch.float32)
    cu = torch.tensor(docs, dtype=torch.int32)
    return q, k, v, g, beta, cu


def test_config_default_is_fla_and_rejects_unknown_impl():
    assert _small_config().kda_chunk_impl == "fla"
    assert _small_config(kda_chunk_impl="fused").kda_chunk_impl == "fused"
    with pytest.raises(ValueError, match="kda_chunk_impl"):
        _small_config(kda_chunk_impl="cutlass")


def test_config_rejects_kf_with_incompatible_knobs_at_config_time():
    with pytest.raises(ValueError, match="kda_mode='chunk'"):
        _small_config(kda_chunk_impl="fused", kda_mode="fused_recurrent")


def test_supported_call_has_no_reason():
    assert kda_fused.fused_kda_unsupported_reason(*_k3_shaped()) is None


@pytest.mark.parametrize(
    "mutate, needle",
    [
        (lambda t: dict(mode="fused_recurrent"), "kda_mode"),
        (lambda t: dict(cp_context=object()), "context parallelism"),
        (lambda t: dict(use_qk_l2norm_in_kernel=False), "l2norm"),
        (lambda t: dict(safe_gate=False), "bounded log gate"),
    ],
)
def test_unsupported_options_are_named(mutate, needle):
    t = _k3_shaped()
    assert needle in kda_fused.fused_kda_unsupported_reason(*t, **mutate(t))


def test_unsupported_tensors_are_named():
    q, k, v, g, beta, cu = _k3_shaped()
    assert kda_fused.fused_kda_unsupported_reason(q, k, v, g, beta, None) is None  # dense batch: packed by the wrapper
    assert "batch of 1" in kda_fused.fused_kda_unsupported_reason(q.expand(2, -1, -1, -1), k, v, g, beta, cu)
    assert "head dims" in kda_fused.fused_kda_unsupported_reason(q[..., :64], k[..., :64], v, g, beta, cu)
    assert "bfloat16" in kda_fused.fused_kda_unsupported_reason(q.float(), k, v, g, beta, cu)
    assert "float32" in kda_fused.fused_kda_unsupported_reason(q, k, v, g.bfloat16(), beta, cu)
    assert "beta" in kda_fused.fused_kda_unsupported_reason(q, k, v, g, beta[..., :1], cu)


class _PassThroughConv(nn.Module):
    def forward(self, x, **kwargs):  # noqa: D102
        return x, None


class _PassThroughNorm(nn.Module):
    def forward(self, o, gate):  # noqa: D102
        return o


def _layer_with_stubs(monkeypatch, cfg, reason=None):
    seen = {}

    def fake_fused(q, k, v, g, beta, cu_seqlens):
        seen["fused"] = dict(q=q.shape, g_dtype=g.dtype, beta_dtype=beta.dtype)
        return torch.zeros_like(v), None

    def fake_fla(*args, **kwargs):
        seen["fla"] = True
        return torch.zeros_like(kwargs["v"]), None

    monkeypatch.setattr(kmod, "fused_chunk_kda", fake_fused)
    monkeypatch.setattr(kmod, "fused_kda_unsupported_reason", lambda *a, **k: reason)
    monkeypatch.setattr(kmod, "chunk_kda", fake_fla)
    layer = kmod.KimiDeltaAttention(cfg, layer_idx=0)
    layer.q_conv1d = layer.k_conv1d = layer.v_conv1d = _PassThroughConv()
    layer.o_norm = _PassThroughNorm()
    x = torch.randn(1, 8, cfg.hidden_size, dtype=torch.bfloat16)
    return layer, x, seen


def test_layer_routes_to_the_fused_kernels_when_enabled(monkeypatch):
    layer, x, seen = _layer_with_stubs(monkeypatch, _small_config(kda_chunk_impl="fused"))
    out = layer(x)
    assert out.shape == x.shape
    assert "fused" in seen and "fla" not in seen
    assert seen["fused"]["g_dtype"] == torch.float32 and seen["fused"]["beta_dtype"] == torch.float32


def test_layer_keeps_fla_by_default(monkeypatch):
    layer, x, seen = _layer_with_stubs(monkeypatch, _small_config())
    layer(x)
    assert "fla" in seen and "fused" not in seen


def test_layer_raises_instead_of_falling_back(monkeypatch):
    layer, x, _ = _layer_with_stubs(monkeypatch, _small_config(kda_chunk_impl="fused"), reason="head dims must be")
    with pytest.raises(ValueError, match="head dims must be"):
        layer(x)


def test_autograd_wrapper_saves_raw_inputs_and_uses_the_triton_backward(monkeypatch):
    calls = {}

    class _FakeExt:
        @staticmethod
        def run(q, k, v, g, beta, cu, o):
            calls["fwd"] = dict(cu_dtype=cu.dtype, contiguous=all(t.is_contiguous() for t in (q, k, v, g, beta)))
            o.copy_(v * 2)

    fake_bwd = types.ModuleType("nemo_automodel.components.models.kimi_k3.kda_fused.chunk_kda_bwd_triton")

    def fake_run(q, k, v, g, beta, cu, do, dq, dk, dv, dg, dbeta, cu_seqlens_cpu=None):
        calls["bwd"] = dict(
            do_dtype=do.dtype, saved=(q.shape, k.shape, v.shape, g.shape, beta.shape, cu.shape), cpu=cu_seqlens_cpu
        )
        dq.copy_(q * 3)
        dk.copy_(k * 5)
        dv.copy_(do)
        dg.fill_(7.0)
        dbeta.fill_(11.0)

    fake_bwd.run = fake_run
    monkeypatch.setitem(sys.modules, fake_bwd.__name__, fake_bwd)
    monkeypatch.setattr(kda_fused, "_forward_ext", lambda: _FakeExt)

    q, k, v, g, beta, cu = _k3_shaped(T=64, H=1, docs=(0, 64))
    leaves = [t.clone().requires_grad_(True) for t in (q, k, v, g, beta)]
    o, final_state = kda_fused.fused_chunk_kda(*leaves, cu.to(torch.int64))
    assert final_state is None
    torch.testing.assert_close(o, (v * 2).to(o.dtype))
    do = torch.randn_like(o)
    o.backward(do)
    assert calls["fwd"]["cu_dtype"] == torch.int32 and calls["fwd"]["contiguous"]
    assert calls["bwd"]["do_dtype"] == torch.bfloat16
    assert calls["bwd"]["cpu"] is None  # packed input: offsets come from the device tensor (one sync, as in FLA)
    torch.testing.assert_close(leaves[0].grad, (q * 3).to(torch.bfloat16))
    torch.testing.assert_close(leaves[1].grad, (k * 5).to(torch.bfloat16))
    torch.testing.assert_close(leaves[2].grad, do.to(torch.bfloat16))
    assert torch.all(leaves[3].grad == 7.0) and leaves[3].grad.dtype == torch.float32
    assert torch.all(leaves[4].grad == 11.0) and leaves[4].grad.dtype == torch.float32


def test_dense_batch_is_packed_into_documents(monkeypatch):
    seen = {}

    class _FakeExt:
        @staticmethod
        def run(q, k, v, g, beta, cu, o):
            seen["shape"] = tuple(q.shape)
            seen["cu"] = cu.tolist()
            o.copy_(v)

    fake_bwd = types.ModuleType("nemo_automodel.components.models.kimi_k3.kda_fused.chunk_kda_bwd_triton")

    def fake_run(q, k, v, g, beta, cu, do, dq, dk, dv, dg, dbeta, cu_seqlens_cpu=None):
        seen["bwd_shape"] = tuple(do.shape)
        seen["cpu"] = cu_seqlens_cpu
        dq.zero_(), dk.zero_(), dv.copy_(do), dg.zero_(), dbeta.zero_()

    fake_bwd.run = fake_run
    monkeypatch.setitem(sys.modules, fake_bwd.__name__, fake_bwd)
    monkeypatch.setattr(kda_fused, "_forward_ext", lambda: _FakeExt)
    B, T, H = 3, 32, 2
    q = torch.randn(B, T, H, 128, dtype=torch.bfloat16)
    k, v = torch.randn_like(q), torch.randn_like(q).requires_grad_(True)
    g = -torch.rand(B, T, H, 128)
    beta = torch.rand(B, T, H)
    o, _ = kda_fused.fused_chunk_kda(q, k, v, g, beta, None)
    assert o.shape == (B, T, H, 128) and seen["shape"] == (1, B * T, H, 128)
    assert seen["cu"] == [0, T, 2 * T, 3 * T]
    o.backward(torch.ones_like(o))
    assert seen["bwd_shape"] == (1, B * T, H, 128) and v.grad.shape == (B, T, H, 128)
    assert seen["cpu"] == (0, T, 2 * T, 3 * T)  # dense input: host offsets from the shape, no device->host sync


def test_stale_build_lock_is_cleared_and_a_fresh_one_is_kept(tmp_path, monkeypatch):
    # no .so -> None (caller falls back to torch's load); a stale lock older than the threshold is removed
    assert kda_fused._import_prebuilt(str(tmp_path)) is None
    lock = tmp_path / "lock"
    lock.write_text("")
    import os as _os
    import time as _time

    old = _time.time() - 2 * kda_fused._STALE_LOCK_SECONDS
    _os.utime(lock, (old, old))
    with pytest.warns(UserWarning, match="build lock"):
        kda_fused._clear_stale_lock(str(tmp_path))
    assert not lock.exists()
    lock.write_text("")  # a fresh lock (a build in progress) is left alone
    kda_fused._clear_stale_lock(str(tmp_path))
    assert lock.exists()


def test_public_entry_point_validates_before_touching_the_kernels(monkeypatch):
    monkeypatch.setattr(kda_fused, "_forward_ext", lambda: pytest.fail("kernel must not be reached"))
    q, k, v, g, beta, cu = _k3_shaped()
    with pytest.raises(ValueError, match="head dims"):
        kda_fused.fused_chunk_kda(q[..., :64], k[..., :64], v, g, beta, cu)


def test_prebuilt_import_waits_while_a_fresh_lock_shows_a_build_in_progress(tmp_path):
    (tmp_path / f"{kda_fused._EXT_NAME}.so").write_bytes(b"not a real module")
    (tmp_path / "lock").write_text("")  # fresh: a builder may still be linking the .so
    assert kda_fused._import_prebuilt(str(tmp_path)) is None


def test_backward_module_imports_without_triton(monkeypatch):
    """CPU-only installs (the import check on macOS runners) must be able to import the kernel module; run() raises."""
    import importlib

    monkeypatch.setitem(sys.modules, "triton", None)  # makes `import triton` raise ImportError
    monkeypatch.setitem(sys.modules, "triton.language", None)
    name = "nemo_automodel.components.models.kimi_k3.kda_fused.chunk_kda_bwd_triton"
    monkeypatch.delitem(sys.modules, name, raising=False)
    mod = importlib.import_module(name)
    assert mod.HAVE_TRITON is False
    with pytest.raises(ImportError):
        mod.run(*([None] * 12))
    monkeypatch.delitem(sys.modules, name, raising=False)  # do not leave the stubbed module cached for other tests
