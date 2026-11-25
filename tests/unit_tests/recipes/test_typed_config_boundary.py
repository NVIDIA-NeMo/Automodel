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

"""The recipe layer turns YAML sections into the typed configs that components take."""

import pytest
import torch

from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.components.models.bagel.configuration import BagelBackendConfig
from nemo_automodel.components.models.bagel.hf_backbone_loader import BagelBackbonesConfig
from nemo_automodel.components.quantization.fp8 import FP8Config
from nemo_automodel.components.quantization.qlora import BnbQuantizationConfig
from nemo_automodel.components.utils.compile_utils import CompileConfig
from nemo_automodel.recipes._typed_config import (
    RecipeConfig,
    bnb_quantization_config_from_node,
    compile_config_from_node,
    fp8_config_from_node,
)


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ({}, None),
        ({"attn_implementation": "sdpa"}, None),
        ({"attn_implementation": "magi"}, "hf"),
        ({"backend": {"attn": "magi"}}, "custom"),
        # The custom-model backend wins, as before the YAML read moved to the recipe layer.
        ({"attn_implementation": "magi", "backend": {"attn": "magi"}}, "custom"),
    ],
)
def test_magi_backend(model, expected):
    assert RecipeConfig(ConfigNode({"model": model})).magi_backend == expected


def test_compile_section():
    node = ConfigNode({"enabled": True, "mode": "max-autotune", "options": {"max_autotune": True}})
    config = compile_config_from_node(node)
    assert isinstance(config, CompileConfig)
    assert (config.enabled, config.mode, dict(config.options)) == (True, "max-autotune", {"max_autotune": True})


def test_fp8_section():
    config = fp8_config_from_node(ConfigNode({"enabled": True, "recipe_name": "rowwise", "filter_fqns": ["lm_head"]}))
    assert isinstance(config, FP8Config)
    assert (config.enabled, config.recipe_name, config.filter_fqns) == (True, "rowwise", ["lm_head"])


def test_quantization_section():
    config = bnb_quantization_config_from_node(ConfigNode({"load_in_4bit": True, "bnb_4bit_quant_type": "fp4"}))
    assert config == BnbQuantizationConfig(load_in_4bit=True, bnb_4bit_quant_type="fp4")


def test_quantization_null_load_flag_is_disabled():
    config = bnb_quantization_config_from_node(ConfigNode({"load_in_4bit": None}))
    assert config.load_in_4bit is False


@pytest.mark.parametrize(
    ("convert", "section"),
    [
        (compile_config_from_node, {"enabled": True, "modee": "default"}),
        (fp8_config_from_node, {"enabled": True, "recipe": "rowwise"}),
        (bnb_quantization_config_from_node, {"load_in_4bit": True, "bnb_4bit_quant": "nf4"}),
    ],
)
def test_sections_reject_unknown_keys(convert, section):
    with pytest.raises(TypeError):
        convert(ConfigNode(section))


def test_bagel_sections():
    raw = ConfigNode(
        {
            "model": {
                "init_mode": "hf_backbones",
                "stage": 2,
                "llm_path": "/llm",
                "vit_path": "/vit",
                "layer_module": "Qwen2MoTDecoderLayer",
                "torch_dtype": "bfloat16",
                "max_latent_size": 32,
                "backend": {"linear": "te", "fused_rope": True},
                "vae_config": {"z_channels": 16, "downsample": 8},
            }
        }
    )
    config = RecipeConfig(raw)

    assert config.bagel_backbones == BagelBackbonesConfig(
        llm_path="/llm",
        vit_path="/vit",
        layer_module="Qwen2MoTDecoderLayer",
        torch_dtype=torch.bfloat16,
        max_latent_size=32,
    )
    assert config.bagel_backend == BagelBackendConfig(linear="te", fused_rope=True)
    assert config.bagel_vae_config == {"z_channels": 16, "downsample": 8}


def test_bagel_backbone_scalar_overrides_are_normalized():
    raw = ConfigNode(
        {
            "model": {
                "llm_path": "/llm",
                "visual_und": None,
                "latent_patch_size": 2.0,
                "max_latent_size": "064",
                "timestep_shift": "2",
            }
        }
    )
    backbones = RecipeConfig(raw).bagel_backbones
    assert backbones.visual_und is False
    assert backbones.latent_patch_size == 2 and isinstance(backbones.latent_patch_size, int)
    assert backbones.max_latent_size == 64 and isinstance(backbones.max_latent_size, int)
    assert backbones.timestep_shift == 2.0 and isinstance(backbones.timestep_shift, float)


def test_bagel_defaults_and_validation():
    config = RecipeConfig(ConfigNode({"model": {"llm_path": "/llm", "visual_und": False}}))
    assert config.bagel_backend == BagelBackendConfig()
    assert config.bagel_vae_config is None
    assert config.bagel_backbones.torch_dtype == torch.float32

    with pytest.raises(ValueError, match="requires model.llm_path"):
        RecipeConfig(ConfigNode({"model": {"vit_path": "/vit"}})).bagel_backbones
    with pytest.raises(ValueError, match="requires model.vit_path"):
        RecipeConfig(ConfigNode({"model": {"llm_path": "/llm"}})).bagel_backbones
    with pytest.raises(TypeError):
        RecipeConfig(ConfigNode({"model": {"backend": {"unknown": True}}})).bagel_backend
