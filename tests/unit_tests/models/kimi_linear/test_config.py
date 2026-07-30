# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest

from nemo_automodel._transformers.registry import (
    _CUSTOM_CONFIG_ARCH_REGISTRATIONS,
    MODEL_ARCH_MAPPING,
    resolve_custom_config_cls,
)
from nemo_automodel.components.models.kimi_linear.config import KimiLinearConfig
from nemo_automodel.components.models.kimi_linear.model import KimiLinearForCausalLM


def test_kimi_linear_config_flags():
    config = KimiLinearConfig(
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

    assert config.model_type == "kimi_linear"
    assert config.is_moe
    assert config.is_mla
    assert config.is_linear_attn
    assert config.kda_use_qk_l2norm_in_kernel
    assert config.is_kda_layer(0)
    assert not config.is_kda_layer(1)
    assert config.is_kda_layer(2)


def test_kimi_linear_config_rejects_missing_layer_lists():
    with pytest.raises(ValueError, match="kda_layers and full_attn_layers"):
        KimiLinearConfig(linear_attn_config={"kda_layers": [1]})


def test_kimi_linear_registry_and_capabilities():
    assert MODEL_ARCH_MAPPING["KimiLinearForCausalLM"] == (
        "nemo_automodel.components.models.kimi_linear.model",
        "KimiLinearForCausalLM",
    )
    # The Kimi K3 text backbone publishes the same model_type, so the architecture
    # name is what selects this config (see _CUSTOM_CONFIG_ARCH_REGISTRATIONS).
    assert _CUSTOM_CONFIG_ARCH_REGISTRATIONS["kimi_linear"]["KimiLinearForCausalLM"] == (
        "nemo_automodel.components.models.kimi_linear.config",
        "KimiLinearConfig",
    )
    assert resolve_custom_config_cls("kimi_linear", ["KimiLinearForCausalLM"]) is KimiLinearConfig

    capabilities = KimiLinearForCausalLM.ModelCapabilities()
    assert capabilities.supports_ep
    assert not capabilities.supports_pp
    assert not capabilities.supports_tp
    assert capabilities.supports_cp
