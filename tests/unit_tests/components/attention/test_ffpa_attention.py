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

"""Unit tests for nemo_automodel/components/attention/ffpa_attention.py."""

import logging
import sys
from types import ModuleType, SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.nn as nn

from nemo_automodel.components.attention import ffpa_attention as ffpa_mod
from nemo_automodel.components.attention.ffpa_attention import (
    ffpa_attention_forward,
    register_ffpa_attention,
    setup_ffpa_backend,
)


@pytest.fixture(autouse=True)
def _reset_module_state():
    ffpa_mod._FALLBACK_WARNED.clear()
    ffpa_mod._SDPA_FN = None
    ffpa_mod._EAGER_FN = None
    ffpa_mod._FLEX_FN = None
    ffpa_mod._FFPA_HIGH_LEVEL = None
    ffpa_mod._FFPA_LOW_LEVEL_READY = None
    ffpa_mod._ffpa_sm80_split_d_backward.cache_clear()
    yield
    ffpa_mod._FALLBACK_WARNED.clear()
    ffpa_mod._ffpa_sm80_split_d_backward.cache_clear()


def _module(head_dim, training=False):
    m = nn.Module()
    m.head_dim = head_dim
    m.training = training
    return m


def _qkv(B, Hq, Hkv, S, D, dtype=torch.bfloat16):
    return (
        torch.randn(B, Hq, S, D, dtype=dtype),
        torch.randn(B, Hkv, S, D, dtype=dtype),
        torch.randn(B, Hkv, S, D, dtype=dtype),
    )


def _patch_sdpa(rv=None):
    rv = rv if rv is not None else (torch.zeros(1, 16, 8, 512), None)
    return mock.patch("transformers.integrations.sdpa_attention.sdpa_attention_forward", return_value=rv)


def _patch_eager(rv=None):
    rv = rv if rv is not None else (torch.zeros(1, 16, 8, 512), None)
    return mock.patch("transformers.models.gemma4.modeling_gemma4.eager_attention_forward", return_value=rv)


def _warns(caplog, substr):
    return [r for r in caplog.records if r.name == ffpa_mod.__name__ and substr in r.getMessage()]


def test_module_exports_only_public_entry_points():
    assert ffpa_mod.__all__ == [
        "ffpa_attention_forward",
        "ffpa_mask",
        "register_ffpa_attention",
        "setup_ffpa_backend",
    ]


def test_registration_idempotent_and_visible():
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    ffpa_mod._REGISTERED = False
    ALL_ATTENTION_FUNCTIONS._global_mapping.pop("ffpa", None)

    assert register_ffpa_attention() is True
    assert ALL_ATTENTION_FUNCTIONS._global_mapping["ffpa"] is ffpa_attention_forward
    assert register_ffpa_attention() is False


# The HF 'ffpa' backend calls the op directly, bypassing the SDPA swap ring CP installs
# and packed-sequence masking; both must be rejected (for custom models like Gemma4 this
# is the only check before a silently-wrong run).
@pytest.mark.parametrize(
    "cp_size, has_packed_sequence, match",
    [(8, False, "context parallelism"), (1, True, "packed sequences")],
)
def test_setup_ffpa_backend_rejects_unsupported(cp_size, has_packed_sequence, match):
    with pytest.raises(ValueError, match=match):
        setup_ffpa_backend(cp_size, has_packed_sequence)


def test_setup_ffpa_backend_registers_when_eligible():
    # Patch the registrar so the eligible path doesn't mutate the global HF registry.
    with mock.patch.object(ffpa_mod, "register_ffpa_attention") as reg:
        setup_ffpa_backend(cp_size=1, has_packed_sequence=False)
    reg.assert_called_once_with()


def test_non_512_head_dim_routes_to_sdpa_silently(caplog):
    q, k, v = _qkv(1, 4, 4, 8, 64)
    sentinel = (torch.zeros(1, 8, 4, 64), None)
    with _patch_sdpa(sentinel) as mock_sdpa, caplog.at_level(logging.WARNING, logger=ffpa_mod.__name__):
        out, weights = ffpa_attention_forward(_module(64), q, k, v, None, scaling=0.125)
    mock_sdpa.assert_called_once()
    assert out is sentinel[0]
    assert weights is sentinel[1]
    assert not _warns(caplog, "fallback")


def test_fp32_routes_to_sdpa_with_warning(caplog):
    q, k, v = _qkv(1, 8, 4, 8, 512, dtype=torch.float32)
    with _patch_sdpa() as mock_sdpa, caplog.at_level(logging.WARNING, logger=ffpa_mod.__name__):
        ffpa_attention_forward(_module(512), q, k, v, None, scaling=0.0442)
    mock_sdpa.assert_called_once()
    assert len(_warns(caplog, "dtype")) == 1


# softcap layers must stay on eager: SDPA silently drops the softcap kwarg.
def test_softcap_routes_to_eager_with_warning(caplog):
    q, k, v = _qkv(1, 8, 4, 8, 512)
    with (
        _patch_sdpa() as mock_sdpa,
        _patch_eager() as mock_eager,
        caplog.at_level(logging.WARNING, logger=ffpa_mod.__name__),
    ):
        ffpa_attention_forward(_module(512), q, k, v, None, softcap=50.0, scaling=0.0442)
    mock_eager.assert_called_once()
    mock_sdpa.assert_not_called()
    assert len(_warns(caplog, "softcap")) == 1


def test_4d_mask_routes_to_sdpa_silently():
    q, k, v = _qkv(1, 8, 4, 16, 512)
    mask = torch.zeros(1, 1, 16, 16, dtype=torch.bfloat16)
    with (
        mock.patch.object(ffpa_mod, "_ffpa_low_level_ready", return_value=True),
        _patch_sdpa() as mock_sdpa,
    ):
        ffpa_attention_forward(_module(512), q, k, v, attention_mask=mask, scaling=0.0442)
    mock_sdpa.assert_called_once()


def test_training_dropout_routes_to_sdpa_with_warning(caplog):
    q, k, v = _qkv(1, 8, 4, 16, 512)
    with _patch_sdpa() as mock_sdpa, caplog.at_level(logging.WARNING, logger=ffpa_mod.__name__):
        ffpa_attention_forward(_module(512, training=True), q, k, v, None, dropout=0.1, scaling=0.0442)
    mock_sdpa.assert_called_once()
    assert len(_warns(caplog, "dropout")) == 1


def test_none_scaling_routes_to_sdpa_with_warning(caplog):
    q, k, v = _qkv(1, 8, 4, 16, 512)
    with _patch_sdpa() as mock_sdpa, caplog.at_level(logging.WARNING, logger=ffpa_mod.__name__):
        ffpa_attention_forward(_module(512), q, k, v, None, scaling=None)
    mock_sdpa.assert_called_once()
    assert len(_warns(caplog, "scaling")) == 1


def test_missing_ffpa_routes_to_sdpa_with_warning(caplog):
    q, k, v = _qkv(1, 8, 4, 8, 512)
    with (
        mock.patch.object(ffpa_mod, "_ffpa_low_level_ready", return_value=False),
        _patch_sdpa() as mock_sdpa,
        caplog.at_level(logging.WARNING, logger=ffpa_mod.__name__),
    ):
        ffpa_attention_forward(_module(512), q, k, v, None, scaling=0.0442)
    mock_sdpa.assert_called_once()
    assert len(_warns(caplog, "ffpa_unavailable")) == 1


def test_degrades_to_eager_when_sdpa_unimportable():
    q, k, v = _qkv(1, 4, 4, 8, 64)
    with (
        mock.patch.object(ffpa_mod, "_get_sdpa", return_value=None),
        _patch_eager() as mock_eager,
    ):
        ffpa_attention_forward(_module(64), q, k, v, None, scaling=0.125)
    mock_eager.assert_called_once()


def test_warning_is_deduped_across_calls(caplog):
    q, k, v = _qkv(1, 8, 4, 16, 512, dtype=torch.float32)
    mask = torch.zeros(1, 1, 16, 16, dtype=torch.bfloat16)
    with _patch_sdpa(), caplog.at_level(logging.WARNING, logger=ffpa_mod.__name__):
        for _ in range(5):
            ffpa_attention_forward(_module(512), q, k, v, attention_mask=mask, scaling=0.0442)
    assert len(_warns(caplog, "dtype")) == 1


def test_dense_path_calls_high_level_func_with_fa_layout():
    B, Hq, Hkv, S, D = 1, 8, 4, 16, 512
    q, k, v = _qkv(B, Hq, Hkv, S, D)
    backend = mock.MagicMock()
    ffpa_fn = mock.Mock(return_value=torch.zeros(B, Hq, S, D, dtype=torch.bfloat16))

    with (
        mock.patch.object(ffpa_mod, "_ffpa_low_level_ready", return_value=True),
        mock.patch.object(ffpa_mod, "_get_ffpa_high_level", return_value=(ffpa_fn, backend)),
    ):
        out, weights = ffpa_attention_forward(_module(512), q, k, v, attention_mask=None, dropout=0.0, scaling=0.0442)

    ffpa_fn.assert_called_once()
    args, kwargs = ffpa_fn.call_args
    assert args[0] is q and args[1] is k and args[2] is v
    assert kwargs == {
        "attn_mask": None,
        "dropout_p": 0.0,
        "is_causal": True,
        "scale": 0.0442,
        "enable_gqa": True,
        "backend": backend,
    }
    assert out.shape == (B, S, Hq, D)
    assert weights is None


def test_2d_bool_mask_returns_padded_output():
    B, Hq, Hkv, S, D = 1, 2, 1, 2, 512
    q, k, v = _qkv(B, Hq, Hkv, S, D)
    attention_mask = torch.tensor([[True, False]])
    out_pack = q.transpose(1, 2)[attention_mask]
    expected = torch.zeros_like(q.transpose(1, 2))
    expected[attention_mask] = out_pack

    with (
        mock.patch.object(ffpa_mod, "_ffpa_low_level_ready", return_value=True),
        mock.patch.object(ffpa_mod, "_ffpa_varlen_fwd", return_value=(out_pack, torch.empty(Hq, 1))),
    ):
        out, weights = ffpa_attention_forward(_module(D), q, k, v, attention_mask, scaling=0.0442)

    assert torch.equal(out, expected)
    assert weights is None


def test_ffpa_mask_returns_only_nontrivial_padding_mask():
    attention_mask = torch.ones(1, 4, dtype=torch.bool)
    assert ffpa_mod.ffpa_mask(1, 4, 4, attention_mask=attention_mask) is None
    attention_mask[0, 1] = False
    assert torch.equal(ffpa_mod.ffpa_mask(1, 4, 4, attention_mask=attention_mask), attention_mask)


@pytest.mark.parametrize("causal", [False, True])
def test_ffpa_mask_routes_only_sliding_to_flex(causal):
    from transformers.masking_utils import causal_mask_function

    mask_function = causal_mask_function if causal else (lambda b, h, qi, ki: ki <= qi)
    sentinel = object()
    with mock.patch("transformers.masking_utils.flex_attention_mask", return_value=sentinel) as mock_flex_mask:
        out = ffpa_mod.ffpa_mask(
            batch_size=1, q_length=8, kv_length=8, mask_function=mask_function, attention_mask=None, device="cpu"
        )
    assert (mock_flex_mask.call_count == 1) is (not causal)
    assert out is (None if causal else sentinel)


def test_ffpa_mask_can_route_sliding_to_sdpa():
    from transformers.masking_utils import sliding_window_causal_mask_function

    config = SimpleNamespace(ffpa_sliding_attn_backend="sdpa")
    out = ffpa_mod.ffpa_mask(
        batch_size=1,
        q_length=4,
        kv_length=4,
        mask_function=sliding_window_causal_mask_function(2),
        device="cpu",
        config=config,
        local_size=2,
    )

    expected = torch.tensor(
        [
            [
                [
                    [True, False, False, False],
                    [True, True, False, False],
                    [False, True, True, False],
                    [False, False, True, True],
                ]
            ]
        ]
    )
    assert torch.equal(out, expected)


def test_ffpa_mask_rejects_unknown_sliding_backend():
    with pytest.raises(ValueError, match="ffpa_sliding_attn_backend must be 'flex' or 'sdpa', got 'invalid'"):
        ffpa_mod.ffpa_mask(
            batch_size=1,
            q_length=4,
            kv_length=4,
            mask_function=lambda b, h, qi, ki: ki <= qi,
            config=SimpleNamespace(ffpa_sliding_attn_backend="invalid"),
        )


def test_block_mask_routes_to_flex_not_sdpa():
    q, k, v = _qkv(1, 8, 4, 16, 256)
    flex_out = (torch.zeros(1, 16, 8, 256), None)
    fake_flex = mock.Mock(return_value=flex_out)
    with (
        mock.patch.object(ffpa_mod, "_is_block_mask", return_value=True),
        mock.patch.object(ffpa_mod, "_get_flex", return_value=fake_flex),
        _patch_sdpa() as mock_sdpa,
    ):
        out, weights = ffpa_attention_forward(_module(256), q, k, v, attention_mask=object(), scaling=0.0625)
    fake_flex.assert_called_once()
    mock_sdpa.assert_not_called()
    assert out is flex_out[0]
    assert weights is None


def test_block_mask_requires_flex_no_sdpa_fallback():
    q, k, v = _qkv(1, 8, 4, 16, 256)
    with (
        mock.patch.object(ffpa_mod, "_is_block_mask", return_value=True),
        mock.patch.object(ffpa_mod, "_get_flex", return_value=None),
        _patch_sdpa() as mock_sdpa,
        pytest.raises(RuntimeError),
    ):
        ffpa_attention_forward(_module(256), q, k, v, attention_mask=object(), scaling=0.0625)
    mock_sdpa.assert_not_called()


# --------------------------------------------------------------------------- #
# Backward dK/dV storage on the generic SM80 Split-D path                      #
# --------------------------------------------------------------------------- #
def _fake_ffpa_cute(**attrs):
    """Stand-in ``ffpa_attn.cute`` module for the function-local imports."""
    cute = ModuleType("ffpa_attn.cute")
    vars(cute).update(attrs)
    pkg = ModuleType("ffpa_attn")
    pkg.cute = cute
    return mock.patch.dict(sys.modules, {"ffpa_attn": pkg, "ffpa_attn.cute": cute})


def _packed(T, Hq=4, Hkv=2, D=8):
    """(grad_out, q, k, v, out, lse) in packed THD layout for T tokens."""
    return (
        torch.randn(T, Hq, D),
        torch.randn(T, Hq, D),
        torch.randn(T, Hkv, D),
        torch.randn(T, Hkv, D),
        torch.randn(T, Hq, D),
        torch.randn(Hq, T),
    )


def test_sm80_split_d_backward_follows_ffpa_dispatch():
    sm80, sm90 = object(), object()
    device = torch.device("cpu")
    with _fake_ffpa_cute(_backward_impl_for_device=lambda *_: sm80, _ffpa_attn_backward_sm80=sm80):
        assert ffpa_mod._ffpa_sm80_split_d_backward(device, 512, 512)
    ffpa_mod._ffpa_sm80_split_d_backward.cache_clear()
    with _fake_ffpa_cute(_backward_impl_for_device=lambda *_: sm90, _ffpa_attn_backward_sm80=sm80):
        assert not ffpa_mod._ffpa_sm80_split_d_backward(device, 512, 512)


def test_sm80_split_d_backward_is_cached_per_device_and_head_dims():
    sm80 = object()
    dispatch = mock.Mock(return_value=sm80)
    with _fake_ffpa_cute(_backward_impl_for_device=dispatch, _ffpa_attn_backward_sm80=sm80):
        for _ in range(3):
            assert ffpa_mod._ffpa_sm80_split_d_backward(torch.device("cpu"), 512, 512)
        assert ffpa_mod._ffpa_sm80_split_d_backward(torch.device("cpu"), 256, 256)
    assert dispatch.call_count == 2


@pytest.mark.parametrize("sm80", [True, False])
def test_dense_bwd_requests_fp32_kv_storage_only_on_sm80(sm80):
    q, k, v = _qkv(1, 8, 4, 16, 512)
    grad_out, out, lse = torch.randn_like(q), torch.randn_like(q), torch.randn(1, 8, 16)
    fake_bwd = mock.Mock(return_value=("dq", "dk", "dv"))
    with (
        _fake_ffpa_cute(_ffpa_attn_backward_cute=fake_bwd),
        mock.patch.object(ffpa_mod, "_ffpa_sm80_split_d_backward", return_value=sm80) as dispatch,
    ):
        assert ffpa_mod._ffpa_dense_bwd(grad_out, q, k, v, out, lse, scale=0.0625, causal=True) == ("dq", "dk", "dv")
    dispatch.assert_called_once_with(q.device, 512, 512)
    storage = {"grad_kv_storage_dtype": torch.float32} if sm80 else {}
    fake_bwd.assert_called_once_with(grad_out, q, k, v, out, lse, 0.0625, True, **storage)


def test_sm80_split_d_backward_keeps_defaults_when_dispatch_is_unimportable(caplog):
    caplog.set_level(logging.WARNING, logger=ffpa_mod.__name__)
    with _fake_ffpa_cute():  # no private dispatch names exported
        assert ffpa_mod._ffpa_sm80_split_d_backward(torch.device("cpu"), 512, 512) is False
        ffpa_mod._ffpa_sm80_split_d_backward.cache_clear()
        assert ffpa_mod._ffpa_sm80_split_d_backward(torch.device("cpu"), 256, 256) is False
    assert len(_warns(caplog, "cannot inspect the ffpa-attn backward dispatch")) == 1


def test_varlen_bwd_sm80_runs_dense_backward_per_document():
    Hq, Hkv, D = 4, 2, 8
    grad_out, q_pack, k_pack, v_pack, out, lse = _packed(10, Hq, Hkv, D)
    calls = []

    def fake_dense_bwd(dout, q, k, v, o, lse_seg, *, scale, causal):
        calls.append((tuple(q.shape), tuple(k.shape), tuple(lse_seg.shape), scale, causal))
        assert dout.shape == o.shape == q.shape
        return q * 2, k * 3, v * 4  # SDPA layout, marked per input

    with mock.patch.object(ffpa_mod, "_ffpa_dense_bwd", fake_dense_bwd):
        dq, dk, dv = ffpa_mod._ffpa_varlen_bwd_sm80(
            grad_out, q_pack, k_pack, v_pack, out, lse, [0, 6, 10], [0, 6, 10], scale=0.0625, causal=True
        )
    assert calls == [
        ((1, Hq, 6, D), (1, Hkv, 6, D), (1, Hq, 6), 0.0625, True),
        ((1, Hq, 4, D), (1, Hkv, 4, D), (1, Hq, 4), 0.0625, True),
    ]
    assert torch.equal(dq, q_pack * 2)
    assert torch.equal(dk, k_pack * 3)
    assert torch.equal(dv, v_pack * 4)


def test_varlen_bwd_sm80_skips_empty_segments():
    grad_out, q_pack, k_pack, v_pack, out, lse = _packed(5)
    fake_dense_bwd = mock.Mock(side_effect=AssertionError("no segment should run"))
    with mock.patch.object(ffpa_mod, "_ffpa_dense_bwd", fake_dense_bwd):
        # doc 1 has no queries, doc 2 has no keys
        dq, dk, dv = ffpa_mod._ffpa_varlen_bwd_sm80(
            grad_out, q_pack, k_pack, v_pack, out, lse, [0, 0, 5], [0, 3, 3], scale=0.0625, causal=False
        )
    fake_dense_bwd.assert_not_called()
    assert not dq.any() and not dk.any() and not dv.any()


@pytest.mark.parametrize("host_bounds", [True, False])
def test_varlen_bwd_on_sm80_uses_segment_loop_with_host_bounds(host_bounds):
    T = 6
    grad_out, q_pack, k_pack, v_pack, out, lse = _packed(T)
    cu_q, cu_k = mock.Mock(name="cu_q"), mock.Mock(name="cu_k")  # only .tolist() is reachable on this path
    cu_q.tolist.return_value = cu_k.tolist.return_value = [0, T]
    fake_op = mock.Mock()
    fake_loop = mock.Mock(return_value=("ldq", "ldk", "ldv"))
    host = {"cu_q_host": [0, T], "cu_k_host": [0, T]} if host_bounds else {}
    with (
        mock.patch.object(ffpa_mod, "_ffpa_sm80_split_d_backward", return_value=True),
        mock.patch.object(ffpa_mod, "_ffpa_varlen_bwd_sm80", fake_loop),
        mock.patch.object(torch.ops, "ffpa_attn", SimpleNamespace(_varlen_bwd_cute=fake_op), create=True),
    ):
        result = ffpa_mod._ffpa_varlen_bwd(
            grad_out, q_pack, k_pack, v_pack, out, lse, cu_q, cu_k, T, T, scale=0.0625, causal=False, **host
        )
    assert result == ("ldq", "ldk", "ldv")
    fake_op.assert_not_called()
    fake_loop.assert_called_once_with(
        grad_out, q_pack, k_pack, v_pack, out, lse, [0, T], [0, T], scale=0.0625, causal=False
    )
    assert cu_q.tolist.call_count == cu_k.tolist.call_count == (0 if host_bounds else 1)


def test_varlen_bwd_off_sm80_keeps_the_varlen_op():
    T = 6
    grad_out, q_pack, k_pack, v_pack, out, lse = _packed(T)
    cu = torch.tensor([0, T], dtype=torch.int32)
    fake_op = mock.Mock(return_value=("dq", "dk", "dv"))
    fake_loop = mock.Mock()
    with (
        mock.patch.object(ffpa_mod, "_ffpa_sm80_split_d_backward", return_value=False),
        mock.patch.object(ffpa_mod, "_ffpa_varlen_bwd_sm80", fake_loop),
        mock.patch.object(torch.ops, "ffpa_attn", SimpleNamespace(_varlen_bwd_cute=fake_op), create=True),
    ):
        result = ffpa_mod._ffpa_varlen_bwd(
            grad_out, q_pack, k_pack, v_pack, out, lse, cu, cu, T, T, scale=0.0625, causal=False
        )
    assert result == ("dq", "dk", "dv")
    fake_loop.assert_not_called()
    fake_op.assert_called_once()
