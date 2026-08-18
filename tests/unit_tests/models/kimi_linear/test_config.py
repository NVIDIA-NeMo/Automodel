# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest

from nemo_automodel._transformers.models.kimi_linear.config import KimiLinear48BConfig
from nemo_automodel._transformers.models.kimi_linear.model import KimiLinear48BForCausalLM
from nemo_automodel._transformers.registry import (
    _CUSTOM_CONFIG_REGISTRATIONS,
    MODEL_ARCH_MAPPING,
    resolve_custom_config_cls,
)


def test_kimi_linear_config_flags():
    config = KimiLinear48BConfig(
        hidden_size=64,
        num_attention_heads=4,
        num_hidden_layers=3,
        num_experts=8,
        num_experts_per_token=2,
        moe_intermediate_size=32,
        q_lora_rank=None,
        kv_lora_rank=16,
        qk_nope_head_dim=8,
        qk_rope_head_dim=4,
        v_head_dim=8,
        mla_use_nope=True,
        linear_attn_config={"kda_layers": [1, 3], "full_attn_layers": [2]},
    )

    assert config.model_type == "kimi_linear_48b_a3b"
    assert config.is_moe
    assert config.is_mla
    assert config.is_linear_attn
    assert config.kda_use_qk_l2norm_in_kernel
    assert config.is_kda_layer(0)
    assert not config.is_kda_layer(1)
    assert config.is_kda_layer(2)


def test_kimi_linear_config_rejects_missing_layer_lists():
    with pytest.raises(ValueError, match="kda_layers and full_attn_layers"):
        KimiLinear48BConfig(linear_attn_config={"kda_layers": [1]})


def test_kimi_linear_config_stamps_automodel_identity(tmp_path):
    """A published checkpoint's shared identity must not survive onto the config."""
    import json

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "kimi_linear",
                "architectures": ["KimiLinearForCausalLM"],
                "hidden_size": 64,
                "num_attention_heads": 4,
                "num_hidden_layers": 3,
                "linear_attn_config": {"kda_layers": [1, 3], "full_attn_layers": [2]},
            }
        )
    )

    config = KimiLinear48BConfig.from_pretrained(tmp_path)

    assert config.model_type == "kimi_linear_48b_a3b"
    assert config.architectures == ["KimiLinear48BForCausalLM"]
    # The rest of the checkpoint is still honored.
    assert config.hidden_size == 64
    assert config.is_kda_layer(0)


def test_kimi_linear_registry_and_capabilities():
    assert MODEL_ARCH_MAPPING["KimiLinear48BForCausalLM"] == (
        "nemo_automodel._transformers.models.kimi_linear.model",
        "KimiLinear48BForCausalLM",
    )
    # Moonshot publishes this model and the Kimi K3 text backbone under the same
    # model_type and the same architecture name, so Automodel gives this one its own
    # identity instead of disambiguating between them.
    assert _CUSTOM_CONFIG_REGISTRATIONS["kimi_linear_48b_a3b"] == (
        "nemo_automodel._transformers.models.kimi_linear.config",
        "KimiLinear48BConfig",
    )
    assert resolve_custom_config_cls("kimi_linear_48b_a3b") is KimiLinear48BConfig

    capabilities = KimiLinear48BForCausalLM.ModelCapabilities()
    assert capabilities.supports_ep
    assert not capabilities.supports_pp
    assert not capabilities.supports_tp
    assert capabilities.supports_cp
