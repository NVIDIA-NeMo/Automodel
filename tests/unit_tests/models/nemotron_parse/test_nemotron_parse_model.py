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

import types
from unittest.mock import MagicMock

import pytest
import torch
from transformers.models.donut.modeling_donut_swin import DonutSwinModelOutput

from nemo_automodel.components.models.nemotron_parse import model as np_model


def test_nemotron_parse_forward_with_stub_encoder(monkeypatch):
    """
    Smoke-test NemotronParseForConditionalGeneration by stubbing the vision encoder
    to avoid heavy RADIO dependencies. Ensures forward + loss computation works.
    """

    decoder_dim = 32
    vocab_size = 50

    # Stub the underlying RADIO encoder creation to return a lightweight module.
    class DummyEncoder(torch.nn.Module):
        def forward(self, pixel_values, *args, **kwargs):
            batch = pixel_values.shape[0]
            # Return shapes compatible with RadioWithNeck but cheap to compute.
            summary = torch.zeros(batch, 3840)
            feature = torch.zeros(batch, 16, 1280)
            return summary, feature

    class DummyEncoderConfig:
        def __init__(self, patch_size=16, max_resolution=64):
            self.patch_size = patch_size
            self.max_resolution = max_resolution

        def to_dict(self):
            return {"patch_size": self.patch_size, "max_resolution": self.max_resolution}

    # Avoid downloading RADIO config (which requires open_clip) by returning a stub.
    dummy_encoder_config = DummyEncoderConfig()
    monkeypatch.setattr(np_model.AutoConfig, "from_pretrained", lambda *args, **kwargs: dummy_encoder_config)
    monkeypatch.setattr(np_model.AutoModel, "from_config", lambda config, trust_remote_code=True: DummyEncoder())

    # Ensure decoder outputs carry inputs_embeds (not present in BaseModelOutput).
    original_decoder_forward = np_model.NemotronParseDecoder.forward

    def decoder_forward_with_inputs(self, *args, **kwargs):
        outputs = original_decoder_forward(self, *args, **kwargs)
        # Prefer passed embeds; otherwise derive from input_ids for the test.
        inputs_embeds = kwargs.get("inputs_embeds")
        if inputs_embeds is None and kwargs.get("input_ids") is not None:
            inputs_embeds = self.embed_tokens(kwargs["input_ids"])
        outputs.inputs_embeds = inputs_embeds
        return outputs

    monkeypatch.setattr(np_model.NemotronParseDecoder, "forward", decoder_forward_with_inputs, raising=True)

    # Bypass RadioWithNeck heavy convs by returning a small hidden state directly.
    def fake_forward(self, pixel_values, *args, **kwargs):
        batch = pixel_values.shape[0]
        hidden = torch.zeros(batch, 2, decoder_dim, dtype=torch.bfloat16)
        return DonutSwinModelOutput(last_hidden_state=hidden)

    monkeypatch.setattr(np_model.RadioWithNeck, "forward", fake_forward, raising=True)

    config = np_model.NemotronParseConfig(
        encoder={"patch_size": 16, "max_resolution": 64},
        decoder={
            "vocab_size": vocab_size,
            "d_model": decoder_dim,
            "encoder_attention_heads": 4,
            "decoder_attention_heads": 4,
            "decoder_ffn_dim": 64,
            "encoder_ffn_dim": 64,
        },
        max_sequence_length=32,
    )

    model = np_model.NemotronParseForConditionalGeneration(config)

    pixel_values = torch.zeros(1, 3, 4, 4, dtype=torch.bfloat16)
    labels = torch.tensor([[1, 2]])

    outputs = model(pixel_values=pixel_values, labels=labels, return_dict=True)

    assert outputs.logits.shape == (1, labels.shape[1], vocab_size)
    assert outputs.loss is not None
    assert torch.isfinite(outputs.loss)

