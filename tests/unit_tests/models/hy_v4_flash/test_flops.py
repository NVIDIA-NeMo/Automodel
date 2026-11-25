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

import json
import os

import pytest

from nemo_automodel.components.models.hy_v4_flash.config import HY4FlashConfig
from nemo_automodel.components.utils.flops_utils import get_flops_formula_for_hf_config

from .tiny import tiny_config


def _brute_force(cfg: HY4FlashConfig, seq_len: int) -> float:
    """Count every GEMM and attention key one query at a time, without the production closed forms."""
    t = cfg.text_config
    total = 0.0
    hs, hc = t.hidden_size, t.hc_mult
    for layer, kind in enumerate(t.layer_types):
        heads = t.attention_heads(kind)
        nope, rope, v = t.attention_dims(kind)
        linear = hs * t.q_lora_rank + t.q_lora_rank * heads * (nope + rope)
        linear += hs * (t.kv_lora_rank + rope) + t.kv_lora_rank * heads * (nope + v)
        linear += 2 * hs * heads * v
        if t.mlp_layer_types[layer] == "sparse":
            linear += hs * t.n_routed_experts + 2 * hs * t.routed_expert_hidden_size
            linear += (t.num_experts_per_tok * 3 * t.routed_expert_hidden_size + 3 * hs) * t.moe_intermediate_size
        else:
            linear += 3 * hs * t.intermediate_size
        linear += 2 * 2 * hc * hc * hs
        total += 6 * seq_len * linear
        for i in range(seq_len):
            keys = min(i + 1, t.sliding_window if kind == "sliding_attention" else t.index_topk)
            total += 6 * heads * ((nope + rope) + v) * keys
            if kind != "sliding_attention":
                n, d = t.index_n_heads, t.index_head_dim
                total += 2 * (t.q_lora_rank * n * d + hs * n + hs * d + 2 * d * d)
                total += 2 * n * d * ((i + 1) // t.index_kpool)
    total += 6 * seq_len * (hc * hc * hs + len(t.engram_layer_ids) * (hc + 1) * 2 * t.engram_n_embed_per_ngram * hs + hs * t.vocab_size)
    return total


@pytest.mark.parametrize("seq_len", [1, 7, 40])
def test_formula_matches_per_query_counting(seq_len):
    cfg = tiny_config()
    formula = get_flops_formula_for_hf_config(cfg)
    assert formula(cfg, gbs=3, seq_len=seq_len) == pytest.approx(3 * _brute_force(cfg, seq_len), rel=1e-12)


def test_text_config_resolves_the_same_formula():
    cfg = tiny_config()
    assert get_flops_formula_for_hf_config(cfg.text_config)(cfg.text_config, 1, 16) == get_flops_formula_for_hf_config(cfg)(cfg, 1, 16)


def test_released_config_has_roughly_the_model_card_active_parameters():
    path = os.environ.get("HY4_CONFIG_JSON")
    if not path or not os.path.exists(path):
        pytest.skip("HY4_CONFIG_JSON not provided")
    cfg = HY4FlashConfig(**json.load(open(path)))
    per_token_at_one = cfg.flops_formula(cfg, 1, 1) / 6
    # model card: "Activated Parameters per Token: Text 11.3B"
    assert 11.2e9 < per_token_at_one < 11.4e9, per_token_at_one
