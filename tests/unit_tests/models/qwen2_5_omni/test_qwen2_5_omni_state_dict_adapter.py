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

"""Tests for Qwen2.5-Omni state dict adapter."""

from unittest.mock import Mock

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.qwen2_5_omni.state_dict_adapter import (
    Qwen2_5OmniStateDictAdapter,
)


@pytest.fixture
def config():
    cfg = Mock()
    cfg.num_hidden_layers = 2
    cfg.hidden_size = 32
    return cfg


@pytest.fixture
def backend_config():
    return BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        enable_hf_state_dict_adapter=False,
    )


@pytest.fixture
def adapter(config, backend_config):
    return Qwen2_5OmniStateDictAdapter(config=config, backend=backend_config, dtype=torch.float32)


class TestInitialization:
    def test_defaults(self, adapter, config, backend_config):
        assert adapter.config is config
        assert adapter.backend is backend_config
        assert adapter.dtype == torch.float32
        assert adapter._uses_thinker_prefix is True


class TestFromHF:
    def test_strips_thinker_prefix(self, adapter):
        hf_state = {
            "thinker.model.embed_tokens.weight": torch.randn(4, 4),
            "thinker.audio_tower.conv1.weight": torch.randn(2, 2),
            "thinker.visual.blocks.0.attn.q.weight": torch.randn(2, 2),
            "thinker.lm_head.weight": torch.randn(4, 4),
        }
        out = adapter.from_hf(hf_state)
        # All thinker.* keys must be stripped of the prefix
        assert "model.embed_tokens.weight" in out
        assert "audio_tower.conv1.weight" in out
        assert "visual.blocks.0.attn.q.weight" in out
        assert "lm_head.weight" in out
        assert all(not k.startswith("thinker.") for k in out)
        # Flag should remain True so to_hf re-adds the prefix
        assert adapter._uses_thinker_prefix is True

    def test_drops_talker_and_token2wav(self, adapter):
        hf_state = {
            "thinker.model.embed_tokens.weight": torch.randn(2, 2),
            "talker.model.embed_tokens.weight": torch.randn(2, 2),
            "talker.codec_head.weight": torch.randn(2, 2),
            "token2wav.code2wav_bigvgan_model.layers.0.weight": torch.randn(2, 2),
            "token2wav.code2wav_dit_model.layers.0.weight": torch.randn(2, 2),
        }
        out = adapter.from_hf(hf_state)
        assert "model.embed_tokens.weight" in out
        for key in out:
            assert not key.startswith("talker."), f"talker key leaked through: {key}"
            assert not key.startswith("token2wav."), f"token2wav key leaked through: {key}"

    def test_handles_state_without_thinker_prefix(self, adapter):
        # If a checkpoint already lacks the thinker prefix (e.g. a NeMo-saved one),
        # keys pass through unchanged.
        hf_state = {"model.embed_tokens.weight": torch.randn(2, 2)}
        out = adapter.from_hf(hf_state)
        assert "model.embed_tokens.weight" in out


class TestToHF:
    def test_adds_thinker_prefix(self, adapter):
        nemo_state = {
            "model.embed_tokens.weight": torch.randn(2, 2),
            "audio_tower.conv1.weight": torch.randn(2, 2),
            "lm_head.weight": torch.randn(2, 2),
        }
        out = adapter.to_hf(nemo_state)
        assert all(k.startswith("thinker.") for k in out.keys())
        assert "thinker.model.embed_tokens.weight" in out
        assert "thinker.audio_tower.conv1.weight" in out
        assert "thinker.lm_head.weight" in out

    def test_respects_exclude_regex(self, adapter):
        nemo_state = {"keep.me": torch.ones(1), "drop.me": torch.ones(1)}
        out = adapter.to_hf(nemo_state, exclude_key_regex=r"^thinker\.drop")
        assert "thinker.drop.me" not in out
        assert "thinker.keep.me" in out

    def test_round_trip(self, adapter):
        nemo_state = {
            "model.embed_tokens.weight": torch.randn(4, 4),
            "audio_tower.conv1.weight": torch.randn(2, 2),
            "visual.blocks.0.attn.q.weight": torch.randn(2, 2),
            "lm_head.weight": torch.randn(4, 4),
        }
        # nemo → hf
        hf_state = adapter.to_hf(nemo_state)
        # hf → nemo
        restored = adapter.from_hf(hf_state)
        for key, value in nemo_state.items():
            assert key in restored, f"missing key after round-trip: {key}"
            assert torch.equal(restored[key], value), f"value mismatch for {key}"


class TestSingleTensorConversion:
    def test_prefixes_key(self, adapter):
        out = adapter.convert_single_tensor_to_hf("model.embed_tokens.weight", torch.zeros(2, 2))
        assert out == [("thinker.model.embed_tokens.weight", out[0][1])]

    def test_respects_exclude_regex(self, adapter):
        out = adapter.convert_single_tensor_to_hf("drop.me", torch.zeros(1), exclude_key_regex=r"^thinker\.drop")
        assert out == []
