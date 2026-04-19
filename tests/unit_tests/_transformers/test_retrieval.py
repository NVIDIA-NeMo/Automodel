# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Tests for nemo_automodel._transformers.retrieval (build_encoder_backbone, BiEncoderModel, etc.)."""

from unittest.mock import MagicMock

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPast

from nemo_automodel._transformers.retrieval import BiEncoderModel

# ---------------------------------------------------------------------------
# BiEncoderModel.encode() passes is_causal=False
# ---------------------------------------------------------------------------


class _SpyModel(nn.Module):
    """A fake model that records the kwargs passed to forward()."""

    def __init__(self, hidden_size=16):
        super().__init__()
        self.config = MagicMock(hidden_size=hidden_size)
        self.captured_kwargs = {}

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        self.captured_kwargs = dict(kwargs)
        bsz, seq = input_ids.shape
        h = self.config.hidden_size
        last = torch.ones(bsz, seq, h)
        return BaseModelOutputWithPast(
            last_hidden_state=last,
            hidden_states=[last],
        )


def test_encode_passes_is_causal_false():
    """BiEncoderModel.encode() must pass is_causal=False to the model's
    forward() so FA2/SDPA kernels don't apply causal masking."""
    spy = _SpyModel(hidden_size=16)
    encoder = BiEncoderModel(model=spy, pooling="avg", l2_normalize=False)

    input_dict = {
        "input_ids": torch.ones(2, 4, dtype=torch.long),
        "attention_mask": torch.ones(2, 4, dtype=torch.long),
    }
    encoder.encode(input_dict)

    assert "is_causal" in spy.captured_kwargs, "encode() must pass is_causal kwarg to model forward"
    assert spy.captured_kwargs["is_causal"] is False


# ---------------------------------------------------------------------------
# build_encoder_backbone sets is_causal flags on generic models
# ---------------------------------------------------------------------------


class _FakeAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.is_causal = True


class _FakeDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _FakeAttention()


class _FakeDecoderModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type("Cfg", (), {"model_type": "fake_decoder"})()
        self.layers = nn.ModuleList([_FakeDecoderLayer(), _FakeDecoderLayer()])


def _mock_generic_automodel(monkeypatch):
    from nemo_automodel._transformers import retrieval

    fake_model = _FakeDecoderModel()
    monkeypatch.setattr(
        retrieval,
        "AutoModel",
        type("AutoModel", (), {"from_pretrained": staticmethod(lambda *a, **kw: fake_model)}),
    )
    monkeypatch.setattr(
        retrieval,
        "AutoConfig",
        type("AutoConfig", (), {"from_pretrained": staticmethod(lambda *a, **kw: type("Cfg", (), {"model_type": "fake_decoder"})())}),
    )
    return fake_model


def test_build_encoder_backbone_sets_config_is_causal(monkeypatch):
    """build_encoder_backbone must set config.is_causal = False on generic models."""
    _mock_generic_automodel(monkeypatch)
    from nemo_automodel._transformers import retrieval

    result = retrieval.build_encoder_backbone("fake/path", task="embedding")
    assert result.config.is_causal is False


def test_build_encoder_backbone_sets_attention_is_causal(monkeypatch):
    """build_encoder_backbone must set module.is_causal = False on all attention layers."""
    _mock_generic_automodel(monkeypatch)
    from nemo_automodel._transformers import retrieval

    result = retrieval.build_encoder_backbone("fake/path", task="embedding")
    for layer in result.layers:
        assert layer.self_attn.is_causal is False
