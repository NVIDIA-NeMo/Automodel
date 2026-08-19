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

import pytest
import torch
from transformers import AutoConfig, AutoModel
from transformers.modeling_outputs import BaseModelOutputWithPast

from nemo_automodel._transformers.model_init import _resolve_custom_model_cls_for_config
from nemo_automodel.components.models.mistral3 import model as mistral_mod
from nemo_automodel.components.models.mistral3.model import (
    Ministral3Config,
    Ministral3ForCausalLM,
    Ministral3Model,
)
from nemo_automodel.components.models.mistral3.state_dict_adapter import Mistral3FP8StateDictAdapter


def tiny_config(**overrides) -> Ministral3Config:
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
        **overrides,
    )
    # Ensure eager attention path in tests to avoid optional backends.
    cfg._attn_implementation = "eager"
    return cfg


class TestConfigAndAutoIntegration:
    def test_auto_config_registration(self):
        cfg = AutoConfig.for_model("ministral3")
        # Check by class name since transformers may return a subclass
        assert type(cfg).__name__ == "Ministral3Config"

    def test_auto_model_from_config_returns_ministral3_model(self):
        cfg = tiny_config()
        model = AutoModel.from_config(cfg)
        # May return transformers or nemo_automodel version, check by class name
        assert type(model).__name__ == "Ministral3Model"

    def test_auto_model_for_causal_lm_registration(self):
        cfg = tiny_config()
        lm = mistral_mod.AutoModelForCausalLM.from_config(cfg)  # type: ignore[attr-defined]
        # May return transformers or nemo_automodel version, check by class name
        assert type(lm).__name__ == "Ministral3ForCausalLM"

    def test_fp8_devstral_config_resolves_streaming_custom_model(self):
        cfg = tiny_config(
            architectures=["Ministral3ForCausalLM"],
            quantization_config={
                "quant_method": "fp8",
                "dequantize": False,
                "weight_block_size": None,
            },
        )

        model_cls = _resolve_custom_model_cls_for_config(cfg)

        assert model_cls is Ministral3ForCausalLM
        assert isinstance(model_cls(cfg).state_dict_adapter, Mistral3FP8StateDictAdapter)


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
    def test_rejects_tied_word_embeddings(self):
        # UNTIED_ONLY: the guard raises at the top of __init__ before construction.
        with pytest.raises(NotImplementedError, match="does not support tie_word_embeddings=True"):
            Ministral3ForCausalLM(tiny_config(tie_word_embeddings=True))

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

    @pytest.mark.parametrize("dequantize", [False, True])
    def test_per_tensor_fp8_checkpoint_attaches_streaming_adapter(self, dequantize):
        cfg = tiny_config(
            quantization_config={
                "quant_method": "fp8",
                "dequantize": dequantize,
                "weight_block_size": None,
            }
        )

        model = Ministral3ForCausalLM(cfg)

        assert isinstance(model.state_dict_adapter, Mistral3FP8StateDictAdapter)
        assert model.state_dict_adapter._layout_name == "causal_lm"
        assert isinstance(model.model.layers[0].self_attn.q_proj, torch.nn.Linear)

    def test_non_fp8_checkpoint_does_not_attach_fp8_adapter(self):
        model = Ministral3ForCausalLM(tiny_config())

        assert not hasattr(model, "state_dict_adapter")

    def test_per_block_fp8_checkpoint_is_rejected(self):
        cfg = tiny_config(
            quantization_config={
                "quant_method": "fp8",
                "dequantize": False,
                "weight_block_size": [128, 128],
            }
        )

        with pytest.raises(NotImplementedError, match="supports per-tensor checkpoints only"):
            Ministral3ForCausalLM(cfg)


# NOTE: HFCheckpointingMixin tests are now in tests/unit_tests/models/common/test_hf_checkpointing_mixin.py
