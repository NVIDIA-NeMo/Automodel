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

from nemo_automodel._transformers import registry
from nemo_automodel._transformers.model_init import get_hf_config
from nemo_automodel.components.models.llama_nemotron_vl.model import LlamaNemotronVLConfig


@pytest.mark.parametrize("trust_remote_code", [False, True])
def test_registered_vl_config_loads_without_remote_code(tmp_path, trust_remote_code):
    # HF registration may already be warm from imports; require Automodel's own resolver too.
    assert registry.resolve_custom_config_cls("llama_nemotron_vl") is LlamaNemotronVLConfig
    checkpoint_config = {
        "model_type": "llama_nemotron_vl",
        "vision_config": {
            "model_type": "siglip_vision_model",
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "image_size": 4,
            "patch_size": 2,
        },
        "llm_config": {
            "model_type": "llama",
            "architectures": ["LlamaBidirectionalModel"],
            "vocab_size": 16,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
        },
        # Match Hub metadata but omit the remote source: Automodel owns the config contract.
        "auto_map": {"AutoConfig": "configuration_llama_nemotron_vl.LlamaNemotronVLConfig"},
    }
    (tmp_path / "config.json").write_text(json.dumps(checkpoint_config))

    config = get_hf_config(str(tmp_path), "eager", trust_remote_code=trust_remote_code, local_files_only=True)

    assert type(config) is LlamaNemotronVLConfig
    assert config.get_text_config(decoder=True) is config.llm_config
