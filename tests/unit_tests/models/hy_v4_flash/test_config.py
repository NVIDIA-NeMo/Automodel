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
from pathlib import Path

import pytest
from transformers import AutoConfig

import nemo_automodel._transformers.registry  # noqa: F401  (registers the custom configs with AutoConfig)
from nemo_automodel.components.models.hy_v4_flash.config import HYV4FlashConfig, HYV4FlashTextConfig
from tests.unit_tests.models.hy_v4_flash.conftest import (
    CPU_BACKEND,
    TINY_TEXT_KWARGS,
    released_config,
    tiny_config,
    tiny_text_config,
)


def test_autoconfig_resolves_the_local_class_from_a_checkpoint_style_config(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps(tiny_config().to_dict()))
    loaded = AutoConfig.from_pretrained(tmp_path)
    assert type(loaded) is HYV4FlashConfig and type(loaded.text_config) is HYV4FlashTextConfig


def test_round_trip_keeps_the_per_type_rope_table(tmp_path: Path) -> None:
    cfg = tiny_config()
    cfg.save_pretrained(tmp_path)
    again = HYV4FlashConfig.from_pretrained(tmp_path).text_config
    assert again.layer_rope_parameters == cfg.text_config.layer_rope_parameters
    assert again.layer_rope_parameters["sliding_attention"]["rope_theta"] == 1.0e4
    assert again.layer_rope_parameters["indexed_attention"]["rope_theta"] == 1.0e7


def test_released_config_semantics() -> None:
    text = released_config().text_config
    # config ids are one-based: zero-based layers 1, 11, 21, 31 (confirmed against the weight names)
    assert text.engram_layer_ids == [2, 12, 22, 32] and text.engram_layer_indices == [1, 11, 21, 31]
    assert text.engram_fp8_embedding is False
    assert (
        text.attention_types.count("indexed_attention") == 12 and text.attention_types.count("sliding_attention") == 30
    )
    assert [i for i, t in enumerate(text.attention_types) if t == "indexed_attention"][:4] == [0, 1, 5, 9]
    assert text.index_kpool == 4 and text.index_topk // text.index_kpool == 512
    assert text.attention_heads("sliding_attention") == 64 and text.attention_heads("indexed_attention") == 32
    assert text.layer_rope_parameters["sliding_attention"]["rope_theta"] == 10000
    assert text.layer_rope_parameters["indexed_attention"]["rope_theta"] == 10000000


@pytest.mark.parametrize("spelling", ["deepseek_sparse_attention", "indexed_attention"])
def test_released_and_reference_layer_type_spellings_build_the_same_layers(spelling: str) -> None:
    # The released config says "deepseek_sparse_attention"; newer transformers base classes rewrite it to
    # "indexed_attention". Either way the indexer layers must get the sparse head count, rotary base and an indexer.
    text = tiny_text_config(num_hidden_layers=3, layer_types=[spelling, "sliding_attention", spelling])
    assert text.attention_types == ["indexed_attention", "sliding_attention", "indexed_attention"]
    assert [text.attention_heads(t) for t in text.attention_types] == [4, 8, 4]
    again = HYV4FlashTextConfig(**text.to_dict())
    assert again.attention_types == text.attention_types
    assert again.layer_rope_parameters["indexed_attention"]["rope_theta"] == 1.0e7


def _rope_parameters(**sliding_overrides: str | bool) -> dict[str, dict[str, str | float | bool | list[int]]]:
    """The tiny per-type rotary table with fields of the sliding-window entry replaced."""
    entry = {"rope_type": "default", "mrope_section": [2, 1, 1]}
    return {
        "sliding_attention": {**entry, "rope_theta": 1.0e4, **sliding_overrides},
        "deepseek_sparse_attention": {**entry, "rope_theta": 1.0e7},
    }


def test_unsupported_layer_type_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported layer type"):
        tiny_text_config(layer_types=["full_attention"] * 6)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"attention_bias": True}, "attention_bias=True is not supported"),
        ({"attention_dropout": 0.1}, "attention_dropout must be 0"),
        ({"dtype": "float32"}, "built in bfloat16"),
        ({"hidden_act": "gelu"}, "silu activation only"),
        ({"rope_parameters": _rope_parameters(rope_type="yarn")}, "rope_type must be 'default'"),
        ({"rope_parameters": _rope_parameters(mrope_interleaved=False)}, "only interleaved M-RoPE"),
    ],
)
def test_settings_the_model_does_not_implement_are_rejected(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        tiny_text_config(**overrides)


@pytest.mark.parametrize(("field", "value"), [("attn", "flex"), ("rms_norm", "te")])
def test_unsupported_backend_choices_are_rejected(field: str, value: str) -> None:
    import dataclasses

    from nemo_automodel.components.models.hy_v4_flash.attention import HYV4FlashAttention

    with pytest.raises(ValueError, match=f"backend.{field}"):
        HYV4FlashAttention(tiny_text_config(), 0, dataclasses.replace(CPU_BACKEND, **{field: value}))


def test_reduced_depth_keeps_the_released_schedule_prefix() -> None:
    text = tiny_text_config(num_hidden_layers=3)
    assert text.attention_types == ["indexed_attention", "indexed_attention", "sliding_attention"]
    assert len(text.mlp_layer_types) == 3
    assert text.engram_layer_indices == [1]  # id 4 (layer 3) is not built


def test_fp32_lm_head_flag_cannot_be_silently_disabled() -> None:
    assert tiny_text_config().enable_lm_head_fp32 is True
    with pytest.raises(ValueError, match="enable_lm_head_fp32"):
        tiny_text_config(enable_lm_head_fp32=False)


def test_top_k_of_a_single_group_is_rejected() -> None:
    # One group leaves no recalled group next to the query's own, which the selection does not implement.
    with pytest.raises(ValueError, match="at least two groups"):
        tiny_text_config(index_topk=2, compressed_indexer_compress_ratio=2)
    assert tiny_text_config(index_topk=4, compressed_indexer_compress_ratio=2).index_topk == 4


def test_reference_name_of_the_latent_expert_width_is_accepted() -> None:
    kwargs = {name: value for name, value in TINY_TEXT_KWARGS.items() if name != "routed_expert_hidden_size"}
    assert HYV4FlashTextConfig(**kwargs, moe_hidden_size=20).routed_expert_hidden_size == 20


def test_engram_config_without_the_fp8_flag_is_rejected() -> None:
    # The reference reads an absent flag as True (not implemented here); only an explicit False may build Engram.
    kwargs = {name: value for name, value in TINY_TEXT_KWARGS.items() if name != "engram_fp8_embedding"}
    with pytest.raises(ValueError, match="engram_fp8_embedding is missing"):
        HYV4FlashTextConfig(**kwargs)
    assert HYV4FlashTextConfig(**{**kwargs, "engram_layer_ids": None}).engram_fp8_embedding is False
