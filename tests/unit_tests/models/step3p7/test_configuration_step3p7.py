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

from nemo_automodel.components.models.step3p7.configuration_step3p7 import (
    Step3p5VConfig,
    Step3p7Config,
    Step3p7TextConfig,
    StepRoboticsVisionEncoderConfig,
    _json_safe_value,
    _normalize_per_layer_values,
)


def test_json_safe_value_handles_dtype_dict_and_tuple():
    assert _json_safe_value(torch.bfloat16) == "bfloat16"
    assert _json_safe_value({"dtype": torch.float32, "items": (torch.float16, 3)}) == {
        "dtype": "float32",
        "items": ["float16", 3],
    }


def test_vision_config_preserves_aliases_and_explicit_cls_token():
    config = StepRoboticsVisionEncoderConfig(ues_cls_token=True)
    assert config.use_cls_token is True
    assert config.ues_cls_token is True

    explicit = StepRoboticsVisionEncoderConfig(ues_cls_token=True, use_cls_token=False)
    assert explicit.use_cls_token is False
    assert explicit.ues_cls_token is False


def test_text_config_normalizes_layer_values_and_torch_dtype():
    rope_scaling = {
        "type": "yarn",
        "factor": 2.0,
        "original_max_position_embeddings": 4096,
        "rope_theta": 10000.0,
    }
    config = Step3p7TextConfig(
        num_hidden_layers=3,
        layer_types=["full_attention"],
        swiglu_limits=[1.0, 2.0, 3.0, 4.0],
        swiglu_limits_shared=[None],
        partial_rotary_factors=[0.5],
        rope_theta=[10000.0],
        rope_scaling=rope_scaling,
        use_rope_layers=[True],
        share_expert_dims=32,
        torch_dtype=torch.bfloat16,
    )

    assert config.layer_types == ["full_attention", "full_attention", "full_attention"]
    assert config.swiglu_limits == [1.0, 2.0, 3.0]
    assert config.swiglu_limits_shared == [None, None, None]
    assert config.partial_rotary_factors == [0.5, 0.5, 0.5]
    assert config.rope_theta == [10000.0, 10000.0, 10000.0]
    assert config.rope_scaling["factor"] == rope_scaling["factor"]
    assert config.rope_scaling["original_max_position_embeddings"] == rope_scaling["original_max_position_embeddings"]
    assert config.rope_scaling["rope_type"] == "yarn"
    assert config.use_rope_layers == [True, True, True]
    assert config.share_expert_dim == 32
    assert config.to_dict()["torch_dtype"] == "bfloat16"


def test_text_config_preserves_explicit_mtp_base_layer_idx():
    config = Step3p7TextConfig(
        num_hidden_layers=4,
        num_nextn_predict_layers=2,
        mtp_base_layer_idx=6,
        layer_types=[
            "full_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
        ],
        swiglu_limits=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    )

    assert config.mtp_base_layer_idx == 6
    assert config.layer_types == ["full_attention", "sliding_attention", "full_attention", "sliding_attention"]
    assert config.mtp_layer_types == ["full_attention", "sliding_attention"]
    assert config.mtp_swiglu_limits == [6.0, 7.0]


def test_normalize_per_layer_values_branches():
    assert _normalize_per_layer_values(None, 2) is None
    assert _normalize_per_layer_values([], 2) == []
    assert _normalize_per_layer_values([1, 2, 3], 2) == [1, 2]
    assert _normalize_per_layer_values([1], 3) == [1, 1, 1]


def test_step3p7_config_builds_nested_defaults_and_json_safe_dict():
    rope_scaling = {
        "type": "yarn",
        "factor": 2.0,
        "original_max_position_embeddings": 4096,
        "rope_theta": 10000.0,
    }
    config = Step3p7Config(rope_scaling=rope_scaling, torch_dtype=torch.float16)

    assert isinstance(config.vision_config, StepRoboticsVisionEncoderConfig)
    assert isinstance(config.text_config, Step3p7TextConfig)
    assert config.text_config.rope_scaling["rope_type"] == "yarn"
    assert config.text_config.rope_scaling["factor"] == 2.0
    assert config.hidden_size == config.text_config.hidden_size
    assert config.max_position_embeddings == config.text_config.max_position_embeddings
    assert config.to_dict()["dtype"] == "float16"
    assert config.model_type == "step3p7"


def test_step3p5v_config_preserves_legacy_model_type():
    config = Step3p5VConfig()

    assert isinstance(config, Step3p7Config)
    assert config.model_type == "step3p5v"


def test_step3p7_config_accepts_dicts_and_existing_text_config():
    config = Step3p7Config(
        vision_config={"width": 8, "layers": 1},
        text_config={"hidden_size": 16, "max_position_embeddings": 32},
        rope_scaling={"type": "linear", "factor": 2.0, "rope_theta": 10000.0},
        image_token_id=7,
    )
    assert config.vision_config.width == 8
    assert config.text_config.hidden_size == 16
    assert config.text_config.rope_scaling["rope_type"] == "linear"
    assert config.text_config.rope_scaling["factor"] == 2.0
    assert config.image_token_id == 7

    text_config = Step3p7TextConfig(hidden_size=12, rope_scaling=None)
    text_config.rope_scaling = None
    reused = Step3p7Config(
        text_config=text_config,
        rope_scaling={"type": "linear", "factor": 2.0, "rope_theta": 10000.0},
    )
    assert reused.text_config is text_config
    assert text_config.rope_scaling["type"] == "linear"
    assert text_config.rope_scaling["factor"] == 2.0


def test_step3p7_config_yarn_without_original_max_position_embeddings():
    """Regression for NVBug 6333464: a minimal yarn rope_scaling (no
    ``original_max_position_embeddings``) must still construct, with the key
    defaulted from the text config's ``max_position_embeddings``."""
    config = Step3p7Config(
        text_config={"hidden_size": 16, "max_position_embeddings": 4096},
        rope_scaling={"rope_type": "yarn", "factor": 2.0},
    )
    assert config.rope_scaling["rope_type"] == "yarn"
    assert config.rope_scaling["factor"] == 2.0
    assert config.rope_scaling["original_max_position_embeddings"] == 4096
