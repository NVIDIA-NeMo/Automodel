# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json

import pytest

from nemo_automodel._transformers.registry import (
    _CUSTOM_CONFIG_REGISTRATIONS,
    MODEL_ARCH_MAPPING,
    resolve_custom_config_cls,
)
from nemo_automodel.components.models.glm5_next.config import Glm5NextConfig, Glm5NextTextConfig
from nemo_automodel.components.models.glm5_next.model import Glm5NextForConditionalGeneration


def test_checkpoint_style_nested_config_resolves_hybrid_patterns():
    config = Glm5NextConfig.from_dict(
        {
            "model_type": "glm5_next",
            "architectures": ["Glm5NextForConditionalGeneration"],
            "text_config": {
                "num_hidden_layers": 4,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "q_lora_rank": 8,
                "qk_rope_head_dim": 0,
                "qk_nope_head_dim": 4,
                "v_head_dim": 4,
                "index_topk": 8,
                "index_kpool": 4,
                "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
                "linear_attn_config": {
                    "head_dim": 4,
                    "num_heads": 2,
                    "short_conv_kernel_size": 2,
                    "gate_lower_bound": -5.0,
                },
            },
            "vision_config": {"depth": 1},
        }
    )

    assert config.text_config.layer_types[-1] == "deepseek_sparse_attention"
    assert config.text_config.linear_lower_bound == -5.0
    assert config.text_config.linear_conv_kernel_dim == 2


def _checkpoint_style_config(**linear_attn_overrides):
    return {
        "model_type": "glm5_next",
        "architectures": ["Glm5NextForConditionalGeneration"],
        "text_config": {
            "num_hidden_layers": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "q_lora_rank": 8,
            "qk_rope_head_dim": 0,
            "qk_nope_head_dim": 4,
            "v_head_dim": 4,
            "index_topk": 8,
            "index_kpool": 4,
            "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
            "linear_attn_config": {
                "head_dim": 4,
                "num_heads": 2,
                "short_conv_kernel_size": 2,
                "gate_lower_bound": -5.0,
                **linear_attn_overrides,
            },
        },
        "vision_config": {"depth": 1},
    }


@pytest.mark.parametrize(
    "released_split",
    [{"kda_layers": [0, 1, 2], "full_attn_layers": [3]}, {}],
    ids=["released-lists", "derived-from-layer-types"],
)
def test_saved_config_keeps_kda_layer_split(tmp_path, released_split):
    config = Glm5NextConfig.from_dict(_checkpoint_style_config(**released_split))
    config.save_pretrained(tmp_path)

    saved = json.loads((tmp_path / "config.json").read_text())["text_config"]["linear_attn_config"]
    assert saved["kda_layers"] == [0, 1, 2]
    assert saved["full_attn_layers"] == [3]
    reloaded = Glm5NextConfig.from_pretrained(tmp_path)
    assert reloaded.text_config.linear_attn_config["kda_layers"] == [0, 1, 2]
    assert reloaded.text_config.layer_types[-1] == "deepseek_sparse_attention"


def test_config_rejects_kda_layers_that_disagree_with_layer_types():
    with pytest.raises(ValueError, match="kda_layers does not match layer_types"):
        Glm5NextConfig.from_dict(_checkpoint_style_config(kda_layers=[0, 1, 3], full_attn_layers=[2]))


def test_config_rejects_rope_and_invalid_kpool_contracts():
    with pytest.raises(ValueError, match="NoPE"):
        Glm5NextTextConfig(qk_rope_head_dim=16)
    with pytest.raises(ValueError, match="divide index_topk"):
        Glm5NextTextConfig(index_topk=7, index_kpool=4)


def test_registry_resolves_native_config_and_model():
    assert MODEL_ARCH_MAPPING["Glm5NextForConditionalGeneration"] == (
        "nemo_automodel.components.models.glm5_next.model",
        "Glm5NextForConditionalGeneration",
    )
    assert _CUSTOM_CONFIG_REGISTRATIONS["glm5_next"] == (
        "nemo_automodel.components.models.glm5_next.config",
        "Glm5NextConfig",
    )
    assert resolve_custom_config_cls("glm5_next") is Glm5NextConfig
    capabilities = Glm5NextForConditionalGeneration.ModelCapabilities()
    assert capabilities.supports_ep and capabilities.supports_cp and capabilities.supports_thd
    assert not capabilities.supports_tp and not capabilities.supports_pp
