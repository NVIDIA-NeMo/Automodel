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

from __future__ import annotations

import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.qwen3_5.model import Qwen3_5ForCausalLM, build_mtp_config_from_hf


def _tiny_config(**kwargs):
    layer_types = kwargs.pop("layer_types", ["full_attention"])
    cfg = Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=32,
        max_position_embeddings=16,
        rms_norm_eps=1e-6,
        pad_token_id=0,
        layer_types=layer_types,
        attn_implementation="eager",
        **kwargs,
    )
    return cfg


def _backend():
    return BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        enable_deepep=False,
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=True,
    )


class TestQwen3_5MTPConfig:
    def test_uses_mtp_num_hidden_layers(self):
        cfg = _tiny_config(mtp_num_hidden_layers=2)
        mtp_config = build_mtp_config_from_hf(cfg)

        assert mtp_config.enabled
        assert mtp_config.num_layers == 2
        assert mtp_config.layer_pattern == "*"

    def test_num_nextn_override_takes_precedence(self):
        cfg = _tiny_config(mtp_num_hidden_layers=1)
        mtp_config = build_mtp_config_from_hf(cfg, num_nextn_predict_layers=3)

        assert mtp_config.num_layers == 3


class TestQwen3_5MTPModel:
    def test_dense_mtp_builds_full_attention_block(self):
        cfg = _tiny_config(mtp_num_hidden_layers=1, layer_types=["linear_attention"])
        model = Qwen3_5ForCausalLM(cfg, backend=_backend())

        assert model.mtp is not None
        mtp_layer = model.mtp.layers[0]
        assert mtp_layer.layer_type == "full_attention"
        assert hasattr(mtp_layer, "self_attn")
        assert hasattr(mtp_layer, "mlp")
        assert hasattr(mtp_layer, "eh_proj")
        assert hasattr(mtp_layer, "final_layernorm")

    def test_forward_emits_mtp_hidden_states_in_training(self):
        cfg = _tiny_config(mtp_num_hidden_layers=1)
        model = Qwen3_5ForCausalLM(cfg, backend=_backend())
        model.train()

        input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        out = model(input_ids=input_ids)

        assert out.logits.shape == (1, 4, cfg.vocab_size)
        assert out.mtp_per_depth_h is not None
        assert len(out.mtp_per_depth_h) == 1
        assert out.mtp_per_depth_h[0].shape == (1, 4, cfg.hidden_size)
        assert out.mtp_loss_scaling_factor == 0.1

    def test_eval_does_not_emit_mtp_hidden_states(self):
        cfg = _tiny_config(mtp_num_hidden_layers=1)
        model = Qwen3_5ForCausalLM(cfg, backend=_backend())
        model.eval()

        out = model(input_ids=torch.tensor([[1, 2, 3, 4]], dtype=torch.long))

        assert out.mtp_per_depth_h is None
        assert out.mtp_loss_scaling_factor is None
