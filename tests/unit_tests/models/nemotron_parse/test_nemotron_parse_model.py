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

import tempfile
import types
from unittest.mock import MagicMock

import pytest
import torch
from transformers.models.donut.modeling_donut_swin import DonutSwinModelOutput

from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
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


class TestNemotronParseHFCheckpointingMixin:
    """Tests for HFCheckpointingMixin integration."""

    def test_model_inherits_hf_checkpointing_mixin(self):
        """Test that NemotronParseForConditionalGeneration inherits from HFCheckpointingMixin."""
        # Verify class inheritance
        assert issubclass(np_model.NemotronParseForConditionalGeneration, HFCheckpointingMixin), (
            "NemotronParseForConditionalGeneration should inherit from HFCheckpointingMixin"
        )

        # Verify MRO has mixin before PreTrainedModel
        mro_names = [cls.__name__ for cls in np_model.NemotronParseForConditionalGeneration.__mro__]
        mixin_idx = mro_names.index("HFCheckpointingMixin")
        pretrained_idx = mro_names.index("PreTrainedModel")
        assert mixin_idx < pretrained_idx, (
            "HFCheckpointingMixin should come before PreTrainedModel in MRO"
        )

    def test_model_has_checkpointer_attribute(self, monkeypatch):
        """Test that model has _checkpointer attribute."""
        # Stub encoder config to avoid heavy RADIO dependencies
        class DummyEncoderConfig:
            def __init__(self):
                self.patch_size = 16
                self.max_resolution = 64

            def to_dict(self):
                return {"patch_size": self.patch_size, "max_resolution": self.max_resolution}

        class DummyEncoder(torch.nn.Module):
            def forward(self, pixel_values, *args, **kwargs):
                batch = pixel_values.shape[0]
                summary = torch.zeros(batch, 3840)
                feature = torch.zeros(batch, 16, 1280)
                return summary, feature

        monkeypatch.setattr(np_model.AutoConfig, "from_pretrained", lambda *args, **kwargs: DummyEncoderConfig())
        monkeypatch.setattr(np_model.AutoModel, "from_config", lambda config, trust_remote_code=True: DummyEncoder())

        config = np_model.NemotronParseConfig(
            encoder={"patch_size": 16, "max_resolution": 64},
            decoder={"vocab_size": 50, "d_model": 32, "decoder_ffn_dim": 64, "encoder_ffn_dim": 64},
        )
        model = np_model.NemotronParseForConditionalGeneration(config)

        # Model should have _checkpointer attribute (may be None if not set)
        assert hasattr(model, "_checkpointer"), (
            "Model should have _checkpointer attribute from HFCheckpointingMixin"
        )

    def test_save_pretrained_requires_checkpointer(self, monkeypatch):
        """Test that save_pretrained raises error without checkpointer."""
        # Stub encoder config to avoid heavy RADIO dependencies
        class DummyEncoderConfig:
            def __init__(self):
                self.patch_size = 16
                self.max_resolution = 64

            def to_dict(self):
                return {"patch_size": self.patch_size, "max_resolution": self.max_resolution}

        class DummyEncoder(torch.nn.Module):
            def forward(self, pixel_values, *args, **kwargs):
                batch = pixel_values.shape[0]
                summary = torch.zeros(batch, 3840)
                feature = torch.zeros(batch, 16, 1280)
                return summary, feature

        monkeypatch.setattr(np_model.AutoConfig, "from_pretrained", lambda *args, **kwargs: DummyEncoderConfig())
        monkeypatch.setattr(np_model.AutoModel, "from_config", lambda config, trust_remote_code=True: DummyEncoder())

        config = np_model.NemotronParseConfig(
            encoder={"patch_size": 16, "max_resolution": 64},
            decoder={"vocab_size": 50, "d_model": 32, "decoder_ffn_dim": 64, "encoder_ffn_dim": 64},
        )
        model = np_model.NemotronParseForConditionalGeneration(config)

        # Clear checkpointer if set
        model._checkpointer = None

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="No checkpointer provided"):
                model.save_pretrained(tmpdir)

    def test_save_pretrained_uses_checkpointer(self, monkeypatch):
        """Test that save_pretrained delegates to Checkpointer.save_model."""
        # Stub encoder config to avoid heavy RADIO dependencies
        class DummyEncoderConfig:
            def __init__(self):
                self.patch_size = 16
                self.max_resolution = 64

            def to_dict(self):
                return {"patch_size": self.patch_size, "max_resolution": self.max_resolution}

        class DummyEncoder(torch.nn.Module):
            def forward(self, pixel_values, *args, **kwargs):
                batch = pixel_values.shape[0]
                summary = torch.zeros(batch, 3840)
                feature = torch.zeros(batch, 16, 1280)
                return summary, feature

        monkeypatch.setattr(np_model.AutoConfig, "from_pretrained", lambda *args, **kwargs: DummyEncoderConfig())
        monkeypatch.setattr(np_model.AutoModel, "from_config", lambda config, trust_remote_code=True: DummyEncoder())

        config = np_model.NemotronParseConfig(
            encoder={"patch_size": 16, "max_resolution": 64},
            decoder={"vocab_size": 50, "d_model": 32, "decoder_ffn_dim": 64, "encoder_ffn_dim": 64},
        )
        model = np_model.NemotronParseForConditionalGeneration(config)

        # Create mock checkpointer
        mock_checkpointer = MagicMock()
        model._checkpointer = mock_checkpointer

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)

            # Verify Checkpointer.save_model was called
            mock_checkpointer.save_model.assert_called_once()
            call_kwargs = mock_checkpointer.save_model.call_args[1]
            assert call_kwargs["model"] is model
            assert call_kwargs["weights_path"] == tmpdir

