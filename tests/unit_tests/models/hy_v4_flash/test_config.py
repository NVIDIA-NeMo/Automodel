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
import torch
from transformers import AutoConfig

import nemo_automodel._transformers.registry  # noqa: F401  (registers the custom configs with AutoConfig)
from nemo_automodel.components.models.hy_v4_flash.config import HY4FlashConfig, HY4FlashTextConfig
from nemo_automodel.components.models.hy_v4_flash.engram import HY4FlashEngramLayer

from .tiny import tiny_config, tiny_text_config

RELEASED = os.environ.get("HY4_CONFIG_JSON")
needs_released = pytest.mark.skipif(
    RELEASED is None or not os.path.exists(RELEASED), reason="HY4_CONFIG_JSON not provided"
)


def test_autoconfig_resolves_the_local_class_from_a_checkpoint_style_config(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(tiny_config().to_dict()))
    loaded = AutoConfig.from_pretrained(tmp_path)
    assert type(loaded) is HY4FlashConfig and type(loaded.text_config) is HY4FlashTextConfig


def test_round_trip_keeps_the_per_type_rope_table(tmp_path):
    cfg = tiny_config()
    cfg.save_pretrained(tmp_path)
    again = HY4FlashConfig.from_pretrained(tmp_path).text_config
    assert again.layer_rope_parameters == cfg.text_config.layer_rope_parameters
    assert again.layer_rope_parameters["sliding_attention"]["rope_theta"] == 1.0e4
    assert again.layer_rope_parameters["deepseek_sparse_attention"]["rope_theta"] == 1.0e7


@needs_released
def test_released_config_semantics():
    text = HY4FlashConfig(**json.load(open(RELEASED))).text_config
    # config ids are one-based: zero-based layers 1, 11, 21, 31 (confirmed against the weight names)
    assert text.engram_layer_ids == [2, 12, 22, 32] and text.engram_layer_indices == [1, 11, 21, 31]
    assert text.engram_fp8_embedding is False
    assert (
        text.layer_types.count("deepseek_sparse_attention") == 12 and text.layer_types.count("sliding_attention") == 30
    )
    assert [i for i, t in enumerate(text.layer_types) if t == "deepseek_sparse_attention"][:4] == [0, 1, 5, 9]
    assert text.index_kpool == 4 and text.index_topk // text.index_kpool == 512
    assert text.attention_heads("sliding_attention") == 64 and text.attention_heads("deepseek_sparse_attention") == 32
    assert text.layer_rope_parameters["sliding_attention"]["rope_theta"] == 10000
    assert text.layer_rope_parameters["deepseek_sparse_attention"]["rope_theta"] == 10000000


def test_engram_fp8_lookup_is_rejected_instead_of_ignored():
    with pytest.raises(NotImplementedError, match="engram_fp8_embedding"):
        HY4FlashEngramLayer(tiny_text_config(engram_fp8_embedding=True), 1, dtype=torch.float32)


def test_unsupported_layer_type_is_rejected():
    with pytest.raises(ValueError, match="unsupported layer type"):
        tiny_text_config(layer_types=["full_attention"] * 6)


def test_wrong_engram_layer_offset_is_detected_by_the_layer_id_check():
    # a zero-based reading of the ids [2, 4] would put Engram on layers 2 and 4, which the config lookup rejects
    cfg = tiny_text_config()
    assert cfg.engram_layer_indices == [1, 3]
    with pytest.raises(KeyError):
        HY4FlashEngramLayer(cfg, 2, dtype=torch.float32)  # layer_id 3 has no head-vocab table in the config


def test_reduced_depth_keeps_the_released_schedule_prefix():
    text = tiny_text_config(num_hidden_layers=3)
    assert text.layer_types == ["deepseek_sparse_attention", "deepseek_sparse_attention", "sliding_attention"]
    assert len(text.mlp_layer_types) == 3
    assert text.engram_layer_indices == [1]  # id 4 (layer 3) is not built
