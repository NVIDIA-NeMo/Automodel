# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
from types import SimpleNamespace

import pytest

from nemo_automodel._transformers.model_init import _consume_config_overrides, get_hf_config
from nemo_automodel._transformers.registry import (
    _CUSTOM_CONFIG_REGISTRATIONS,
    MODEL_ARCH_MAPPING,
    resolve_custom_config_cls,
)
from nemo_automodel.components.checkpoint.addons import ConsolidatedHFAddon
from nemo_automodel.components.models.glm5_next.config import Glm5NextConfig, Glm5NextTextConfig
from nemo_automodel.components.models.glm5_next.model import Glm5NextForConditionalGeneration
from tests.unit_tests.models.glm5_next.conftest import tiny_backend, tiny_glm5_next_config


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


def test_truncated_config_keeps_released_lists_for_its_layers(tmp_path):
    # The first four layers of the release: layer_types shortened, the released lists left full.
    config = Glm5NextConfig.from_dict(
        _checkpoint_style_config(kda_layers=[0, 1, 2, 4, 5, 6, 8], full_attn_layers=[3, 7, 11])
    )
    config.save_pretrained(tmp_path)

    saved = json.loads((tmp_path / "config.json").read_text())["text_config"]["linear_attn_config"]
    assert saved["kda_layers"] == [0, 1, 2]
    assert saved["full_attn_layers"] == [3]


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


def test_export_config_json_keeps_released_fields():
    # Consolidated HF exports serialize the full config (``to_json_string(use_diff=False)``).
    config = Glm5NextConfig.from_dict(_checkpoint_style_config(kda_layers=[0, 1, 2], full_attn_layers=[3]))
    exported = json.loads(config.to_json_string(use_diff=False))

    assert exported["architectures"] == ["Glm5NextForConditionalGeneration"]
    assert "architectures" not in exported["text_config"]
    assert "architectures" not in exported["vision_config"]
    linear_attn_config = exported["text_config"]["linear_attn_config"]
    assert linear_attn_config["kda_layers"] == [0, 1, 2]
    assert linear_attn_config["full_attn_layers"] == [3]


def test_export_config_json_top_level_architectures_survive_text_config_merge():
    # GLM-5.3 loaders such as SGLang's config merge top-level keys into ``text_config`` and read
    # ``architectures`` back from it; a null entry in ``text_config`` would replace the list.
    exported = json.loads(Glm5NextConfig.from_dict(_checkpoint_style_config()).to_json_string(use_diff=False))
    merged = {"architectures": exported["architectures"], **exported["text_config"]}

    assert merged["architectures"] == ["Glm5NextForConditionalGeneration"]


_UNSET = object()


@pytest.mark.parametrize("sub_config", ["text_config", "vision_config"])
@pytest.mark.parametrize(
    "architectures",
    [_UNSET, None, [], ["Glm5NextForConditionalGeneration"]],
    ids=["unset", "null", "empty", "nonempty"],
)
def test_nested_architectures_serialize_and_reload(tmp_path, sub_config, architectures):
    # Transformers' nested diff looks up each serialized sub-config key in the sub-config's
    # default ``to_dict``, so an explicit nested list must not lose that key.
    raw = _checkpoint_style_config()
    if architectures is not _UNSET:
        raw[sub_config]["architectures"] = architectures
    expected = None if architectures is _UNSET else architectures
    config = Glm5NextConfig.from_dict(raw)

    for use_diff in (False, True):
        serialized = json.loads(config.to_json_string(use_diff=use_diff))
        assert serialized["architectures"] == ["Glm5NextForConditionalGeneration"]
        if expected is None:
            assert "architectures" not in serialized[sub_config]
        else:
            assert serialized[sub_config]["architectures"] == expected

    config.save_pretrained(tmp_path)
    saved = json.loads((tmp_path / "config.json").read_text())
    for reloaded in (
        Glm5NextConfig.from_pretrained(tmp_path),
        Glm5NextConfig.from_dict(saved),
        get_hf_config(str(tmp_path), "sdpa"),
    ):
        assert isinstance(reloaded, Glm5NextConfig)
        assert reloaded.architectures == ["Glm5NextForConditionalGeneration"]
        assert getattr(reloaded, sub_config).architectures == expected


@pytest.mark.parametrize(
    ("text_config_override", "kda_layers", "full_attn_layers"),
    [
        (
            {
                "layer_types": [
                    "deepseek_sparse_attention",
                    "linear_attention",
                    "linear_attention",
                    "linear_attention",
                ]
            },
            [1, 2, 3],
            [0],
        ),
        (
            {
                "num_hidden_layers": 2,
                "layer_types": ["linear_attention", "linear_attention"],
                "mlp_layer_types": ["dense", "dense"],
                "indexer_types": ["full", "full"],
            },
            [0, 1],
            [],
        ),
    ],
    ids=["reordered-schedule", "two-layer-prefix"],
)
def test_overridden_layer_schedule_exports_its_own_split(tmp_path, text_config_override, kda_layers, full_attn_layers):
    # Recipe overrides reach the built config through ``setattr`` after construction.
    config = tiny_glm5_next_config()
    _consume_config_overrides(config, {"text_config": text_config_override})
    model = Glm5NextForConditionalGeneration(config, backend=tiny_backend())
    assert sorted(model.model.language_model.layers.keys(), key=int) == [
        str(idx) for idx in range(text_config_override.get("num_hidden_layers", 4))
    ]

    ConsolidatedHFAddon().pre_save(
        model_state=SimpleNamespace(model=[model]),
        hf_metadata_dir=str(tmp_path),
        tokenizer=None,
        fqn_to_file_index_mapping={},
        fqn_to_dtype_mapping=None,
        original_model_path=None,
        v4_compatible=False,
    )

    saved = json.loads((tmp_path / "config.json").read_text())["text_config"]["linear_attn_config"]
    assert saved["kda_layers"] == kda_layers
    assert saved["full_attn_layers"] == full_attn_layers
    reloaded = Glm5NextConfig.from_pretrained(tmp_path)
    assert reloaded.text_config.layer_types == text_config_override["layer_types"]
    assert reloaded.text_config.linear_attn_config["kda_layers"] == kda_layers
