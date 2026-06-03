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

"""CPU tiny-config tests for NemotronParse memory-efficient fused cross-entropy support.

The training recipe (``nemo_automodel/recipes/llm/train_ft.py``) only takes the
``FusedLinearCrossEntropy`` (cut-CE) path when:
  (a) ``_supports_logits_to_keep(model)`` is True (a ``logits_to_keep`` parameter
      exists on ``forward``), AND
  (b) ``model(logits_to_keep=1, **batch)`` returns an output where
      ``"hidden_states" in out`` is True and ``get_final_hidden_states(out)``
      yields the final (pre-lm_head) hidden states spanning the full sequence.

These tests assert NemotronParseForConditionalGeneration satisfies that contract
without disturbing the default (logits_to_keep=0) behavior. They mirror the
stub-encoder tiny-config approach from ``test_nemotron_parse_model.py`` so they
run on CPU without the heavy RADIO vision dependency.
"""

import torch
from transformers.models.donut.modeling_donut_swin import DonutSwinModelOutput

from nemo_automodel.components.models.nemotron_parse import model as np_model
from nemo_automodel.components.training.model_output_utils import get_final_hidden_states
from nemo_automodel.components.utils.model_utils import _supports_logits_to_keep

DECODER_DIM = 32
VOCAB_SIZE = 50


def _build_tiny_model(monkeypatch):
    """Build a tiny NemotronParse model on CPU with a stubbed vision encoder."""

    class DummyEncoder(torch.nn.Module):
        def forward(self, pixel_values, *args, **kwargs):
            batch = pixel_values.shape[0]
            summary = torch.zeros(batch, 3840)
            feature = torch.zeros(batch, 16, 1280)
            return summary, feature

    class DummyEncoderConfig:
        def __init__(self, patch_size=16, max_resolution=64):
            self.patch_size = patch_size
            self.max_resolution = max_resolution

        def to_dict(self):
            return {"patch_size": self.patch_size, "max_resolution": self.max_resolution}

    dummy_encoder_config = DummyEncoderConfig()
    monkeypatch.setattr(np_model.AutoConfig, "from_pretrained", lambda *args, **kwargs: dummy_encoder_config)
    monkeypatch.setattr(np_model.AutoModel, "from_config", lambda config, trust_remote_code=True: DummyEncoder())

    # Bypass the heavy RADIO neck convs: return a small encoder hidden state directly.
    def fake_forward(self, pixel_values, *args, **kwargs):
        batch = pixel_values.shape[0]
        hidden = torch.zeros(batch, 2, DECODER_DIM, dtype=torch.bfloat16)
        return DonutSwinModelOutput(last_hidden_state=hidden)

    monkeypatch.setattr(np_model.RadioWithNeck, "forward", fake_forward, raising=True)

    config = np_model.NemotronParseConfig(
        encoder={"patch_size": 16, "max_resolution": 64},
        decoder={
            "vocab_size": VOCAB_SIZE,
            "d_model": DECODER_DIM,
            "encoder_attention_heads": 4,
            "decoder_attention_heads": 4,
            "decoder_ffn_dim": 64,
            "encoder_ffn_dim": 64,
        },
        max_sequence_length=32,
    )

    model = np_model.NemotronParseForConditionalGeneration(config)
    model.eval()
    return model


def test_supports_logits_to_keep(monkeypatch):
    """forward exposes a ``logits_to_keep`` parameter (gate (a) in the recipe)."""
    model = _build_tiny_model(monkeypatch)
    assert _supports_logits_to_keep(model) is True


def test_cut_ce_contract_logits_to_keep_and_hidden_states(monkeypatch):
    """logits_to_keep=1 + output_hidden_states=True satisfies gate (b)."""
    model = _build_tiny_model(monkeypatch)

    pixel_values = torch.zeros(1, 3, 4, 4, dtype=torch.bfloat16)
    decoder_input_ids = torch.tensor([[1, 2, 3, 4]])
    seq_len = decoder_input_ids.shape[1]

    with torch.no_grad():
        out = model(
            pixel_values=pixel_values,
            decoder_input_ids=decoder_input_ids,
            logits_to_keep=1,
            output_hidden_states=True,
            return_dict=True,
        )

    # (b1) The recipe gates on ``"hidden_states" in out``; also accept a populated attr.
    assert ("hidden_states" in out) or (out.hidden_states is not None)

    # (b2) The final hidden states span the FULL sequence (not sliced like logits).
    final_hs = get_final_hidden_states(out)
    assert final_hs is not None
    assert final_hs.shape[0] == 1
    assert final_hs.shape[1] == seq_len
    assert final_hs.shape[-1] == DECODER_DIM

    # (b3) Logits correspond to only the last token (logits_to_keep=1).
    assert out.logits.ndim == 3
    assert out.logits.shape == (1, 1, VOCAB_SIZE)


def test_default_forward_full_length_logits(monkeypatch):
    """Default call (logits_to_keep=0) still yields full-length logits, no hidden_states."""
    model = _build_tiny_model(monkeypatch)

    pixel_values = torch.zeros(1, 3, 4, 4, dtype=torch.bfloat16)
    decoder_input_ids = torch.tensor([[1, 2, 3, 4]])
    seq_len = decoder_input_ids.shape[1]

    with torch.no_grad():
        out = model(pixel_values=pixel_values, decoder_input_ids=decoder_input_ids, return_dict=True)

    # Full-sequence logits, unchanged from the pre-cut-CE behavior.
    assert out.logits.shape == (1, seq_len, VOCAB_SIZE)
    # With output_hidden_states unset/falsy the key must not be registered, so the
    # recipe leaves models that don't ask for hidden states on the masked-CE path.
    assert "hidden_states" not in out


def test_logits_to_keep_matches_full_slice(monkeypatch):
    """logits_to_keep=1 logits equal the last position of the full (logits_to_keep=0) logits."""
    model = _build_tiny_model(monkeypatch)

    pixel_values = torch.zeros(1, 3, 4, 4, dtype=torch.bfloat16)
    decoder_input_ids = torch.tensor([[1, 2, 3, 4]])

    with torch.no_grad():
        full = model(pixel_values=pixel_values, decoder_input_ids=decoder_input_ids, return_dict=True)
        last = model(
            pixel_values=pixel_values,
            decoder_input_ids=decoder_input_ids,
            logits_to_keep=1,
            return_dict=True,
        )

    torch.testing.assert_close(last.logits[:, -1, :], full.logits[:, -1, :])
