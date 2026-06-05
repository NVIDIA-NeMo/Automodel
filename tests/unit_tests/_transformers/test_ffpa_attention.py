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

"""Unit tests for nemo_automodel/_transformers/ffpa_attention.py."""

import logging
from unittest import mock

import pytest
import torch
import torch.nn as nn

from nemo_automodel._transformers import ffpa_attention as ffpa_mod
from nemo_automodel._transformers.ffpa_attention import ffpa_attention_forward, register_ffpa_attention


@pytest.fixture(autouse=True)
def _reset_module_state():
    ffpa_mod._FALLBACK_WARNED.clear()
    ffpa_mod._SDPA_FN = None
    ffpa_mod._EAGER_FN = None
    ffpa_mod._FFPA_FN = None
    ffpa_mod._CUTEDSL_BACKEND = None
    ffpa_mod._FFPA_LOW_LEVEL_READY = None
    yield
    ffpa_mod._FALLBACK_WARNED.clear()


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


def test_registration_idempotent_and_visible():
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    ffpa_mod._REGISTERED = False
    ALL_ATTENTION_FUNCTIONS._global_mapping.pop("ffpa", None)

    assert register_ffpa_attention() is True
    assert ALL_ATTENTION_FUNCTIONS._global_mapping["ffpa"] is ffpa_attention_forward
    assert register_ffpa_attention() is False


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
    fwd_calls = []
    backend = mock.MagicMock()

    def fake_ffpa(qn, kn, vn, attn_mask, dropout_p, is_causal, scale, enable_gqa, backend):
        fwd_calls.append((qn, kn, vn, attn_mask, dropout_p, is_causal, scale, enable_gqa, backend))
        return torch.zeros(B, Hq, S, D, dtype=torch.bfloat16)

    with (
        mock.patch.object(ffpa_mod, "_ffpa_low_level_ready", return_value=True),
        mock.patch.object(ffpa_mod, "_get_ffpa_high_level", return_value=(fake_ffpa, backend)),
    ):
        out, weights = ffpa_attention_forward(_module(512), q, k, v, attention_mask=None, dropout=0.0, scaling=0.0442)

    assert len(fwd_calls) == 1
    qn, kn, vn, attn_mask, dropout_p, is_causal, scale, enable_gqa, used_backend = fwd_calls[0]
    assert qn.shape == (B, Hq, S, D)
    assert kn.shape == (B, Hkv, S, D)
    assert vn.shape == (B, Hkv, S, D)
    assert attn_mask is None
    assert dropout_p == 0.0
    assert is_causal is True
    assert scale == 0.0442
    assert enable_gqa is True
    assert used_backend is backend
    assert out.shape == (B, S, Hq, D)
    assert weights is None
