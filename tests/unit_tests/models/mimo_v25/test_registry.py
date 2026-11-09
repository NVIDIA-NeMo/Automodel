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

import yaml

from nemo_automodel._transformers.model_init import _resolve_custom_model_cls_for_config, get_hf_config
from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.mimo_v2_flash.config import MiMoV2Config as SharedMiMoV2Config
from nemo_automodel.components.models.mimo_v2_flash.model import MiMoV2ForCausalLM as SharedMiMoV2ForCausalLM
from nemo_automodel.components.models.mimo_v25.config import MiMoV2Config
from nemo_automodel.components.models.mimo_v25.model import MiMoV2ForCausalLM
from nemo_automodel.components.models.mimo_v25.state_dict_adapter import MiMoV2StateDictAdapter


def test_v25_recipe_selects_its_own_model_without_replacing_shared_registration(tmp_path):
    config_dict = {
        "model_type": "mimo_v2",
        "architectures": ["MiMoV2ForCausalLM"],
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 32,
        "moe_intermediate_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "v_head_dim": 4,
        "swa_num_attention_heads": 4,
        "swa_num_key_value_heads": 2,
        "swa_head_dim": 8,
        "swa_v_head_dim": 4,
        "hybrid_layer_pattern": [0, 1],
        "moe_layer_freq": [0, 1],
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "n_group": 1,
        "topk_group": 1,
        "norm_topk_prob": True,
        "sliding_window": 4,
        "partial_rotary_factor": 0.5,
        "attention_projection_layout": "fused_qkv",
        "torch_dtype": "float32",
    }
    (tmp_path / "config.json").write_text(json.dumps(config_dict))
    repo_root = Path(__file__).resolve().parents[4]
    recipe = yaml.safe_load((repo_root / "examples/llm_finetune/mimo_v25/mimo_v25_pro_hellaswag.yaml").read_text())
    recipe_config = ConfigNode(recipe["model"]["config"])
    config = recipe_config.instantiate(pretrained_model_name_or_path=str(tmp_path))

    assert isinstance(config, MiMoV2Config)
    model_cls = _resolve_custom_model_cls_for_config(config)
    assert model_cls is MiMoV2ForCausalLM
    backend = BackendConfig(
        attn="sdpa",
        linear="torch",
        rms_norm="torch_fp32",
        experts="torch",
        dispatcher="torch",
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=True,
    )
    model = model_cls.from_config(config, backend=backend)
    assert isinstance(model.state_dict_adapter, MiMoV2StateDictAdapter)

    shared_config = get_hf_config(tmp_path, attn_implementation="sdpa")
    assert isinstance(shared_config, SharedMiMoV2Config)
    assert shared_config.architectures == ["MiMoV2ForCausalLM"]
    assert _resolve_custom_model_cls_for_config(shared_config) is SharedMiMoV2ForCausalLM
