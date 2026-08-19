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

"""Capability gates for the MagiAttention custom-model backend.

MagiAttention implements context parallelism (and CP + sequence packing) via its
own dispatch, so the gates that previously admitted only TE must also admit
``backend.attn == "magi"``. These are pure introspection checks — no GPU or
``magi_attention`` package required.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch.nn as nn

from nemo_automodel._transformers.capabilities import ModelSupports, _uses_magi_attention


class _BackendModel(nn.Module):
    """Custom-model stand-in carrying a BackendConfig-like ``backend.attn``."""

    def __init__(self, attn: str):
        super().__init__()
        self.backend = SimpleNamespace(attn=attn)
        self.config = SimpleNamespace()  # no hybrid markers

    def forward(self, **kwargs):  # VAR_KEYWORD -> _supports_seq_lens is True
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# _uses_magi_attention
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "attn,expected",
    [("magi", True), ("te", False), ("sdpa", False), ("flex", False)],
)
def test_uses_magi_attention(attn, expected):
    assert _uses_magi_attention(_BackendModel(attn)) is expected


def test_uses_magi_attention_no_backend():
    assert _uses_magi_attention(SimpleNamespace()) is False


# --------------------------------------------------------------------------- #
# supports_cp / supports_sequence_packing / supports_cp_with_sequence_packing
# --------------------------------------------------------------------------- #
def _supports(attn, cp_size=1):
    # ModelSupports holds the model weakly (in production the model owns it as
    # ``model._supports``), so the caller must keep ``model`` alive for the
    # duration of the capability check -- return it alongside.
    model = _BackendModel(attn)
    mesh = SimpleNamespace(cp_size=cp_size)
    return model, ModelSupports(model, mesh)


def test_supports_cp_admits_magi():
    model, supports = _supports("magi")
    assert supports.supports_cp is True


def test_supports_cp_rejects_flex_backend():
    """Regression: the gate was not broadened to every custom backend."""
    model, supports = _supports("flex")
    assert supports.supports_cp is False


def test_supports_sequence_packing_admits_magi():
    model, supports = _supports("magi")
    assert supports.supports_sequence_packing is True


def test_supports_cp_with_sequence_packing_admits_magi_at_cp2():
    model, supports = _supports("magi", cp_size=2)
    assert supports.supports_cp_with_sequence_packing is True


def test_supports_cp_with_sequence_packing_rejects_flex_at_cp2():
    model, supports = _supports("flex", cp_size=2)
    assert supports.supports_cp_with_sequence_packing is False


def test_supports_cp_with_sequence_packing_cp1_falls_back_to_packing():
    # at cp_size<=1 it reduces to plain sequence-packing support (magi qualifies).
    model, supports = _supports("magi", cp_size=1)
    assert supports.supports_cp_with_sequence_packing is True
