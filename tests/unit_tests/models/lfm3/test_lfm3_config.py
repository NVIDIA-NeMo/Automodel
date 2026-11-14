# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""LFM3 config resolution from checkpoint-style ``config.json`` files and RoPE labelling."""

import copy
import json

import pytest

from nemo_automodel._transformers.model_init import get_hf_config
from nemo_automodel._transformers.registry import ModelRegistry
from nemo_automodel.components.models.lfm3.config import Lfm3Config, Lfm3VlConfig, Lfm3VlVisionConfig

# The RoPE entry and remote-code map of LiquidAI/LFM3-1.7B's config.json.
PUBLISHED_ROPE = {
    "full_attention": {"partial_rotary_factor": 0.5, "rope_theta": 1000000.0, "rope_type": "default"},
    "sliding_attention": {"rope_theta": 10000, "rope_type": "default"},
}
PUBLISHED_TEXT = {
    "architectures": ["Lfm3ForCausalLM"],
    "auto_map": {
        "AutoConfig": "configuration_lfm3.Lfm3Config",
        "AutoModelForCausalLM": "modeling_lfm3.Lfm3ForCausalLM",
    },
    "model_type": "lfm3",
    "hidden_size": 32,
    "num_hidden_layers": 5,
    "rope_parameters": PUBLISHED_ROPE,
    "transformers_version": "5.12.1",
}


def _checkpoint(tmp_path, config):
    (tmp_path / "config.json").write_text(json.dumps(config))
    return str(tmp_path)


def test_published_text_config_resolves_locally_without_remote_code(tmp_path):
    config = get_hf_config(_checkpoint(tmp_path, PUBLISHED_TEXT), "sdpa")

    assert type(config) is Lfm3Config
    assert config.layer_types == ["conv", "conv", "full_attention", "conv", "sliding_attention"]
    assert ModelRegistry.resolve_custom_model_cls("Lfm3ForCausalLM", config).__name__ == "Lfm3ForCausalLM"


def test_published_vl_config_resolves_local_sub_configs(tmp_path):
    published = {
        "architectures": ["Lfm3VlForConditionalGeneration"],
        "auto_map": {"AutoConfig": "configuration_lfm3_vl.Lfm3VlConfig"},
        "model_type": "lfm3_vl",
        "downsample_factor": 2,
        "projector_layer_norm_eps": 1e-06,
        "text_config": {
            key: value for key, value in PUBLISHED_TEXT.items() if key not in ("architectures", "auto_map")
        },
        "vision_config": {
            "model_type": "lfm3_vl_vision",
            "num_hidden_layers": 4,
            "rope_parameters": {"rope_theta": 1000000.0, "rope_type": "axial"},
        },
    }

    config = get_hf_config(_checkpoint(tmp_path, published), "sdpa")

    assert (type(config), type(config.text_config), type(config.vision_config)) == (
        Lfm3VlConfig,
        Lfm3Config,
        Lfm3VlVisionConfig,
    )
    assert config.text_config.rope_parameters["full_attention"]["rope_type"] == "proportional"
    assert config.vision_config.layer_types == ["conv", "conv", "full_attention", "conv"]
    assert ModelRegistry.resolve_custom_model_cls("Lfm3VlForConditionalGeneration", config).__name__ == (
        "Lfm3VlForConditionalGeneration"
    )


def test_partial_rotation_labelled_default_is_relabelled_without_touching_the_input():
    rope = copy.deepcopy(PUBLISHED_ROPE)

    config = Lfm3Config(rope_parameters=rope)

    assert config.rope_parameters["full_attention"] == {**PUBLISHED_ROPE["full_attention"], "rope_type": "proportional"}
    assert config.rope_parameters["sliding_attention"]["rope_type"] == "default"
    assert rope["full_attention"]["rope_type"] == "default"
    assert Lfm3Config().rope_parameters["full_attention"] == config.rope_parameters["full_attention"]


def test_relabelled_config_survives_a_save_and_reload(tmp_path):
    Lfm3Config(rope_parameters=copy.deepcopy(PUBLISHED_ROPE)).save_pretrained(tmp_path)

    saved = json.loads((tmp_path / "config.json").read_text())

    assert saved["rope_parameters"]["full_attention"]["rope_type"] == "proportional"
    assert Lfm3Config.from_pretrained(tmp_path).rope_parameters == saved["rope_parameters"]


@pytest.mark.parametrize(
    ("rope_parameters", "message"),
    [
        (
            {"sliding_attention": {"rope_type": "default", "rope_theta": 10000.0}},
            "no entry for the \\['full_attention'\\]",
        ),
        ({**PUBLISHED_ROPE, "sliding_attention": {"rope_type": "yarn", "rope_theta": 10000.0, "factor": 2.0}}, "yarn"),
    ],
)
def test_unusable_rope_parameters_are_rejected(rope_parameters, message):
    with pytest.raises(ValueError, match=message):
        Lfm3Config(rope_parameters=rope_parameters)
