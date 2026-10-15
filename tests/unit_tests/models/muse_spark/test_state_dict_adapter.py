# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Round-trip tests for released Muse Spark Open tensor layouts."""

import torch

from nemo_automodel.components.models.muse_spark.state_dict_adapter import MuseSparkStateDictAdapter
from tests.unit_tests.models.muse_spark.conftest import tiny_backend, tiny_muse_spark_config


def test_released_router_and_grouped_experts_round_trip() -> None:
    """Raw [H,E]/[E,out,in] tensors become native [E,H]/[E,in,out] tensors."""
    config = tiny_muse_spark_config()
    model_backend = tiny_backend()
    from nemo_automodel.components.models.muse_spark.layers import build_muse_spark_moe_config

    adapter = MuseSparkStateDictAdapter(
        config,
        build_muse_spark_moe_config(config, torch.float32),
        model_backend,
        dtype=torch.float32,
    )
    raw = {
        "model.layers.0.feed_forward.router.weight": torch.arange(64.0).reshape(16, 4),
        "model.layers.0.feed_forward.expert_bias": torch.arange(4.0),
        "model.layers.0.feed_forward.experts.w13_weight": torch.arange(512.0).reshape(4, 16, 8),
        "model.layers.0.feed_forward.experts.w2_weight": torch.arange(256.0).reshape(4, 8, 8),
        "model.layers.0.feed_forward.pooler.weight": torch.arange(128.0).reshape(8, 16),
        "model.layers.0.feed_forward.unpooler.weight": torch.arange(128.0).reshape(16, 8),
    }

    native = adapter.from_hf(raw)

    assert native["model.layers.0.feed_forward.gate.weight"].shape == (4, 16)
    assert native["model.layers.0.feed_forward.experts.gate_and_up_projs"].shape == (4, 8, 16)
    assert native["model.layers.0.feed_forward.experts.down_projs"].shape == (4, 8, 8)
    assert native["model.layers.0.feed_forward.gate.weight"].dtype is torch.float32
    round_trip = adapter.to_hf(native)
    for key, value in raw.items():
        torch.testing.assert_close(round_trip[key], value)


def test_adapter_skips_redundant_vision_and_uninstantiated_layers() -> None:
    """Duplicate perception tensors and layers beyond a diagnostic prefix are ignored."""
    config = tiny_muse_spark_config()
    from nemo_automodel.components.models.muse_spark.layers import build_muse_spark_moe_config

    adapter = MuseSparkStateDictAdapter(
        config,
        build_muse_spark_moe_config(config, torch.float32),
        tiny_backend(),
        dtype=torch.float32,
    )
    converted = adapter.from_hf(
        {
            "modalities.vision_perception.encoder.vision_model.conv1_linear.weight": torch.ones(1),
            "model.layers.4.self_attn.q_proj.weight": torch.ones(1),
            "model.embed_tokens.weight": torch.ones(64, 16),
        }
    )

    assert "model.embed_tokens.weight" in converted
    assert all("layers.4" not in key for key in converted)
    assert all("modalities.vision_perception" not in key for key in converted)
