# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Muse Spark Open configuration and native-registry contracts."""

from nemo_automodel._transformers.registry import (
    _CUSTOM_CONFIG_REGISTRATIONS,
    MODEL_ARCH_MAPPING,
    resolve_custom_config_cls,
)
from nemo_automodel.components.models.muse_spark.config import (
    MuseSpark1xOmniConfig,
    expand_global_attention_schedule,
)
from nemo_automodel.components.models.muse_spark.model import MuseSpark1xOmniForCausalLM


def test_released_config_preserves_layer_phase_and_metap_scales() -> None:
    """A diagnostic prefix keeps the 62-layer checkpoint's periodic phase."""
    config = MuseSpark1xOmniConfig(num_hidden_layers=4, schedule_num_hidden_layers=62)

    assert expand_global_attention_schedule([2048, 2048, 2048, 0], 62, 4)[:4] == [2048, 0, 2048, 2048]
    assert config.layer_sliding_windows == [2048, 0, 2048, 2048]
    assert config.no_rope_layers == [False, True, False, False]
    assert config.patch_token_id == 200092
    assert config.router_output_multiplier == (8192 / 256) ** -0.5
    assert config.output_multiplier == config.router_output_multiplier


def test_registry_resolves_native_config_and_model() -> None:
    """The immutable checkpoint resolves without remote modeling code."""
    assert MODEL_ARCH_MAPPING["MuseSpark1xOmniForCausalLM"] == (
        "nemo_automodel.components.models.muse_spark.model",
        "MuseSpark1xOmniForCausalLM",
    )
    assert _CUSTOM_CONFIG_REGISTRATIONS["musespark1x_omni"] == (
        "nemo_automodel.components.models.muse_spark.config",
        "MuseSpark1xOmniConfig",
    )
    assert resolve_custom_config_cls("musespark1x_omni") is MuseSpark1xOmniConfig
    capabilities = MuseSpark1xOmniForCausalLM.ModelCapabilities()
    assert capabilities.supports_ep and capabilities.supports_cp and capabilities.supports_pp
    assert capabilities.supports_thd and not capabilities.supports_tp
