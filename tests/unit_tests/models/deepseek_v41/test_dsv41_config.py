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

import json

import pytest
import torch
from transformers import AutoConfig

from nemo_automodel._transformers.registry import (
    _CUSTOM_CONFIG_REGISTRATIONS,
    MODEL_ARCH_MAPPING,
    resolve_custom_config_cls,
)
from nemo_automodel.components.models.deepseek_v41.config import DeepseekV41Config
from nemo_automodel.components.models.deepseek_v41.model import DeepseekV41ForCausalLM
from tests.unit_tests.models.deepseek_v41.conftest import tiny_config

# Mirrors the released ``config.json`` (nested text_config / vision_config).
_RELEASED_STYLE_CONFIG = {
    "architectures": ["DeepseekV41ForCausalLM"],
    "model_type": "deepseek_v41",
    "dtype": "bfloat16",
    "bos_token_id": 0,
    "eos_token_id": 1,
    "pad_token_id": 2,
    "image_token_id": 129264,
    "quantization_config": {"quant_method": "fp8", "weight_block_size": [32, 32], "scale_fmt": "ue8m0"},
    "text_config": {
        "model_type": "deepseek_v41_text",
        "vocab_size": 129280,
        "hidden_size": 5120,
        "num_hidden_layers": 40,
        "rms_norm_eps": 1e-20,
        "rope_scaling": {"rope_type": "yarn", "factor": 16, "original_max_position_embeddings": 65536},
        "compress_ratios": [0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0],
        "kv_source_layer_ids": [2, 8, 14, 20],
        "index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36],
        "candidate_source_layer_id": 20,
        "candidate_topk_blocks": 2048,
        "candidate_block_size": 8,
        "engram_layer_ids": [1, 14],
        "engram_num_embeddings": [384006168, 384016682],
        "engram_compressed_vocab_size": 99092,
        "num_nextn_predict_layers": 3,
        "dspark_target_layer_ids": [37, 38, 39],
    },
    "vision_config": {"model_type": "deepseek_v41_vision", "num_hidden_layers": 32},
}


class TestConfigFlattening:
    def test_text_config_is_hoisted(self):
        config = DeepseekV41Config.from_dict(_RELEASED_STYLE_CONFIG)
        assert config.model_type == "deepseek_v41"
        assert config.hidden_size == 5120
        assert config.num_hidden_layers == 40
        assert config.rms_norm_eps == 1e-20
        assert config.kv_source_layer_ids == [2, 8, 14, 20]
        assert config.engram_num_embeddings == [384006168, 384016682]
        assert config.vision_config["num_hidden_layers"] == 32
        assert config.image_token_id == 129264
        assert config.pad_token_id == 2
        assert not hasattr(config, "quantization_config") or config.quantization_config is None

    def test_dtype_resolution(self):
        assert DeepseekV41Config.from_dict(_RELEASED_STYLE_CONFIG).torch_dtype == torch.bfloat16
        assert DeepseekV41Config(torch_dtype="float32").torch_dtype == torch.float32
        assert DeepseekV41Config().torch_dtype == torch.bfloat16

    def test_from_pretrained_directory(self, tmp_path):
        (tmp_path / "config.json").write_text(json.dumps(_RELEASED_STYLE_CONFIG))
        config = DeepseekV41Config.from_pretrained(tmp_path)
        assert isinstance(config, DeepseekV41Config)
        assert config.compress_ratios[:2] == [0, 0]
        assert config.compress_ratio(2) == 2 and config.compress_ratio(20) == 1 and config.compress_ratio(41) == 0

    def test_auto_config_resolves_custom_class(self, tmp_path):
        (tmp_path / "config.json").write_text(json.dumps(_RELEASED_STYLE_CONFIG))
        config = AutoConfig.from_pretrained(tmp_path)
        assert isinstance(config, DeepseekV41Config)
        assert config.index_source_layer_ids == [2, 8, 14, 20, 24, 28, 32, 36]

    def test_engram_length_mismatch_rejected(self):
        with pytest.raises(ValueError, match="engram_layer_ids"):
            DeepseekV41Config(engram_layer_ids=[1, 14], engram_num_embeddings=[10])


class TestLayerLayout:
    def test_released_layout_modes(self):
        config = DeepseekV41Config.from_dict(_RELEASED_STYLE_CONFIG)
        config.validate_layer_layout()
        modes = [config.csa2_mode(i) for i in range(40)]
        assert modes[:2] == ["swa", "swa"]
        assert [i for i, m in enumerate(modes) if m == "full"] == [2, 8, 14, 20]
        assert [i for i, m in enumerate(modes) if m == "reindex"] == [24, 28, 32, 36]
        assert modes.count("reuse") == 40 - 2 - 4 - 4

    def test_tiny_layout_modes(self):
        config = tiny_config()
        assert [config.csa2_mode(i) for i in range(6)] == ["swa", "swa", "full", "reuse", "full", "reindex"]

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            (dict(compress_ratios=[0, 0, 2, 2, 1, 1], kv_source_layer_ids=[0, 4]), "compress_ratio 0"),
            (dict(kv_source_layer_ids=[4], index_source_layer_ids=[4, 5]), "no KV source precedes"),
            (dict(compress_ratios=[0, 0, 2, 1, 1, 1]), "reuses KV from layer"),
            (dict(index_source_layer_ids=[2, 5]), "must also be an index source"),
            (dict(kv_source_layer_ids=[2, 4], index_source_layer_ids=[2, 4], candidate_source_layer_id=5), "candidate"),
        ],
    )
    def test_invalid_layouts_rejected(self, overrides, match):
        config = tiny_config(**overrides)
        with pytest.raises(ValueError, match=match):
            config.validate_layer_layout()

    def test_valid_layout_variants(self):
        # No Reindex layer and no candidate pool: every non-source CSA2 layer is Reuse.
        config = tiny_config(index_source_layer_ids=[2, 4], candidate_source_layer_id=-1)
        config.validate_layer_layout()
        assert config.csa2_mode(5) == "reuse"
        # Candidate pool built by the last KV source, consumed by the trailing Reindex layer.
        config = tiny_config(candidate_source_layer_id=4)
        config.validate_layer_layout()
        assert config.csa2_mode(5) == "reindex"


class TestRegistry:
    def test_arch_and_config_registered(self):
        assert MODEL_ARCH_MAPPING["DeepseekV41ForCausalLM"] == (
            "nemo_automodel.components.models.deepseek_v41.model",
            "DeepseekV41ForCausalLM",
        )
        assert _CUSTOM_CONFIG_REGISTRATIONS["deepseek_v41"] == (
            "nemo_automodel.components.models.deepseek_v41.config",
            "DeepseekV41Config",
        )
        assert resolve_custom_config_cls("deepseek_v41") is DeepseekV41Config

    def test_capabilities(self):
        capabilities = DeepseekV41ForCausalLM.ModelCapabilities()
        assert capabilities.supports_ep and capabilities.supports_thd
        assert not (capabilities.supports_tp or capabilities.supports_cp or capabilities.supports_pp)
