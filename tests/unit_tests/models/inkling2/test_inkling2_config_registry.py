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
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from nemo_automodel._transformers.registry import ModelRegistry
from nemo_automodel.components.models.inkling2.configuration import (
    Inkling2Config,
    Inkling2TextConfig,
    Inkling2VisionConfig,
)
from nemo_automodel.components.models.inkling2.model import Inkling2ForConditionalGeneration

pytestmark = pytest.mark.cpu_only

_LOCAL_LAYER_IDS = [i for i in range(48) if (i + 1) % 6]

# Text, vision and drafter fields copied from the released thinkingmachines/Inkling-2-Small config.json.
_RELEASED_CONFIG = {
    "architectures": ["Inkling2ForConditionalGeneration"],
    "model_type": "inkling2",
    "eos_token_id": 200006,
    "text_config": {
        "model_max_length": 1048576,
        "torch_dtype": "bfloat16",
        "hidden_size": 4608,
        "num_hidden_layers": 48,
        "vocab_size": 201024,
        "num_attention_heads": 48,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "d_rel": 16,
        "rel_extent": 1024,
        "log_scaling_n_floor": 128000,
        "log_scaling_alpha": 0.1,
        "rms_norm_eps": 1e-06,
        "use_value_norm_no_gamma": True,
        "qk_l2_clip_threshold": 22.627416997969522,
        "rel_l2_clip_threshold": 8.0,
        "local_layer_ids": _LOCAL_LAYER_IDS,
        "sconv_kernel_size": 4,
        "unpadded_vocab_size": 200058,
        "final_logit_softcapping": 30.0,
        "swa_head_dim": 128,
        "swa_num_attention_heads": 48,
        "swa_num_key_value_heads": 8,
        "sliding_window_size": 1024,
        "n_layers_per_attn_res_block": 6,
        "attn_res_normalize_values": True,
        "n_routed_experts": 384,
        "num_experts_per_tok": 8,
        "n_shared_experts": 0,
        "intermediate_size": 2304,
        "mlp_gate_softcap": 4.0,
        "mlp_up_softcap": 25.0,
        "route_scale": 8.0,
        "use_gate_bias": True,
    },
    "vision_config": {
        "temporal_patch_size": 2,
        "native_patch_size": 20,
        "adapter_merge_size": 2,
        "projection_hidden_dim": 4096,
        "projector_bias": False,
        "hidden_size": 1024,
        "num_hidden_layers": 23,
        "num_attention_heads": 16,
        "mlp_ratio": 4.0,
        "posemb_grid_size": 32,
        "layer_norm_eps": 1e-05,
    },
    "mtp_config": {"num_hidden_layers": 5, "block_size": 8, "target_layer_ids": [5, 11, 17, 23, 29, 35, 41, 48]},
}


def _write_config(directory: Path) -> Path:
    (directory / "config.json").write_text(json.dumps(_RELEASED_CONFIG))
    return directory


def test_inkling2_model_type_resolves_to_local_config_class() -> None:
    assert CONFIG_MAPPING["inkling2"] is Inkling2Config
    assert ModelRegistry.has_custom_model("Inkling2ForConditionalGeneration")


def test_released_config_json_resolves_through_auto_config(tmp_path: Path) -> None:
    config = AutoConfig.from_pretrained(_write_config(tmp_path))

    assert isinstance(config, Inkling2Config)
    text, vision = config.text_config, config.vision_config
    assert isinstance(text, Inkling2TextConfig)
    assert isinstance(vision, Inkling2VisionConfig)
    # Checkpoint aliases.
    assert text.moe_intermediate_size == 2304
    assert text.conv_kernel_size == 4
    assert text.max_position_embeddings == 1048576
    assert vision.patch_size == 20
    assert vision.spatial_merge_size == 2
    assert vision.intermediate_size == 4096
    # Full attention on the layers missing from local_layer_ids.
    full_layers = [i for i, layer_type in enumerate(text.layer_types) if layer_type == "hybrid"]
    assert full_layers == [5, 11, 17, 23, 29, 35, 41, 47]
    assert text.unpadded_vocab_size == 200058
    assert text.final_logit_softcapping == 30.0
    assert config.image_token_id == 200054
    assert config.mtp_config["block_size"] == 8


def test_config_save_and_reload_preserves_fields(tmp_path: Path) -> None:
    config = AutoConfig.from_pretrained(_write_config(tmp_path))
    config.save_pretrained(tmp_path / "roundtrip")
    reloaded = AutoConfig.from_pretrained(tmp_path / "roundtrip")

    assert isinstance(reloaded, Inkling2Config)
    assert reloaded.text_config.layer_types == config.text_config.layer_types
    assert reloaded.text_config.moe_intermediate_size == 2304
    assert reloaded.vision_config.patch_size == 20
    assert reloaded.mtp_config == config.mtp_config


def test_text_only_config_has_no_vision_tower() -> None:
    config = Inkling2Config(text_config={"hidden_size": 64}, vision_config=None)
    assert config.vision_config is None


def test_vision_width_must_match_text_width() -> None:
    with pytest.raises(ValueError, match="text_hidden_size"):
        Inkling2Config(text_config={"hidden_size": 64}, vision_config={"text_hidden_size": 128})


@pytest.mark.parametrize("layer_types", [["hybrid_sliding"], ["full_attention", "hybrid"]])
def test_invalid_layer_types_are_rejected(layer_types: list[str]) -> None:
    with pytest.raises(ValueError, match="layer_types"):
        Inkling2TextConfig(num_hidden_layers=2, layer_types=layer_types)


def test_tied_embeddings_are_rejected() -> None:
    config = Inkling2Config(text_config={"hidden_size": 64, "num_hidden_layers": 1}, tie_word_embeddings=True)
    with pytest.raises(NotImplementedError, match="tie_word_embeddings"):
        Inkling2ForConditionalGeneration(config)
