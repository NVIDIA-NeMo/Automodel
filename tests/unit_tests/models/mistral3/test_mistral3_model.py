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

from unittest.mock import patch

import torch
from transformers import AutoConfig, AutoModel
from transformers.modeling_outputs import BaseModelOutputWithPast

from nemo_automodel.components.models.mistral3 import model as mistral_mod
from nemo_automodel.components.models.mistral3.model import (
    Ministral3Config,
    Ministral3ForCausalLM,
    Ministral3Model,
    Mistral3ForConditionalGeneration,
)


def tiny_config() -> Ministral3Config:
    cfg = Ministral3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=64,
        attention_dropout=0.0,
    )
    # Ensure eager attention path in tests to avoid optional backends.
    cfg._attn_implementation = "eager"
    return cfg


class TestConfigAndAutoIntegration:
    def test_auto_config_registration(self):
        cfg = AutoConfig.for_model("ministral3")
        assert isinstance(cfg, Ministral3Config)

    def test_auto_model_from_config_returns_ministral3_model(self):
        cfg = tiny_config()
        model = AutoModel.from_config(cfg)
        assert isinstance(model, Ministral3Model)

    def test_auto_model_for_causal_lm_registration(self):
        cfg = tiny_config()
        lm = mistral_mod.AutoModelForCausalLM.from_config(cfg)  # type: ignore[attr-defined]
        assert isinstance(lm, Ministral3ForCausalLM)


class TestMinistral3Model:
    def test_initialization_sets_components(self):
        cfg = tiny_config()
        model = Ministral3Model(cfg)

        assert model.embed_tokens.num_embeddings == cfg.vocab_size
        assert len(model.layers) == cfg.num_hidden_layers
        assert model.rotary_emb.max_seq_len_cached == cfg.max_position_embeddings

    def test_forward_runs_layers_and_returns_last_hidden_state(self):
        cfg = tiny_config()
        model = Ministral3Model(cfg)
        batch, seq_len = 2, 3
        input_ids = torch.randint(0, cfg.vocab_size, (batch, seq_len))
        dummy_hidden = torch.zeros(batch, seq_len, cfg.hidden_size)

        with patch.object(model.layers[0], "forward", return_value=dummy_hidden) as mock_layer:
            outputs = model(input_ids, use_cache=False)

        assert outputs.last_hidden_state.shape == (batch, seq_len, cfg.hidden_size)
        mock_layer.assert_called_once()


class TestMinistral3ForCausalLM:
    def test_forward_emits_logits(self):
        cfg = tiny_config()
        model = Ministral3ForCausalLM(cfg)
        batch, seq_len = 2, 4
        input_ids = torch.randint(0, cfg.vocab_size, (batch, seq_len))
        fake_hidden = torch.randn(batch, seq_len, cfg.hidden_size)
        fake_output = BaseModelOutputWithPast(last_hidden_state=fake_hidden)

        with patch.object(model.model, "forward", return_value=fake_output) as mock_forward:
            outputs = model(input_ids, logits_to_keep=0)

        assert outputs.logits.shape == (batch, seq_len, cfg.vocab_size)
        mock_forward.assert_called_once()


class TestModelClassExport:
    def test_model_class_points_to_models(self):
        assert hasattr(mistral_mod, "ModelClass")
        assert mistral_mod.Ministral3ForCausalLM in mistral_mod.ModelClass
        assert Mistral3ForConditionalGeneration in mistral_mod.ModelClass

