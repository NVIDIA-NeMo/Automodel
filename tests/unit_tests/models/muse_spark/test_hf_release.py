# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HF-v2 configuration, checkpoint storage and numerical conversion contracts."""

import copy
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from nemo_automodel._transformers.model_init import get_hf_config
from nemo_automodel._transformers.registry import (
    _CUSTOM_CONFIG_REGISTRATIONS,
    MODEL_ARCH_MAPPING,
    resolve_custom_config_cls,
)
from nemo_automodel.components.checkpoint.checkpointing import Checkpointer
from nemo_automodel.components.checkpoint.config import CheckpointingConfig
from nemo_automodel.components.models.muse_spark.config import (
    MuseSpark1xOmniConfig,
    expand_global_attention_schedule,
)
from nemo_automodel.components.models.muse_spark.configuration_muse_spark import MuseSparkConfig
from nemo_automodel.components.models.muse_spark.model import MuseSpark1xOmniForCausalLM
from nemo_automodel.components.models.muse_spark.state_dict_adapter import MuseSparkStateDictAdapter
from safetensors.torch import save_file
from tests.unit_tests.models.muse_spark.conftest import tiny_backend, tiny_muse_spark_config


def tiny_hf_config() -> MuseSparkConfig:
    """Return a small v2 configuration with all three modalities present."""
    return MuseSparkConfig(
        text_config={
            "vocab_size": 64,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "moe_hidden_size": 8,
            "moe_intermediate_size": 8,
            "num_local_experts": 4,
            "num_experts_per_tok": 2,
            "sliding_window": 4,
            "max_position_embeddings": 128,
            "output_multiplier": 0.5,
            "residual_gate_temperature": 0.7,
            "layer_types": ["sliding_attention", "full_attention", "sliding_attention", "sliding_attention"],
            "layer_rope_theta": [500000.0, 0, 500000.0, 500000.0],
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
        },
        vision_config={
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_attention_heads": 2,
            "num_hidden_layers": 2,
            "patch_size": 2,
            "pos_emb_height": 2,
            "pos_emb_width": 3,
            "max_position_embeddings": 6,
            "layer_types": ["window_attention", "full_attention"],
        },
        audio_config={
            "hidden_sizes": [4, 8],
            "depths": [1, 1],
            "patch_sizes": [2, 2],
            "num_hidden_layers": 2,
            "out_hidden_size": 16,
        },
        projector_hidden_size=16,
        image_token_id=61,
        video_token_id=62,
        audio_token_id=63,
        dtype="float32",
    )


def test_v2_config_registry_serialization_and_prefix(tmp_path: Path) -> None:
    config = tiny_hf_config()
    config.save_pretrained(tmp_path)
    raw = json.loads((tmp_path / "config.json").read_text())
    assert raw["model_type"] == "muse_spark"
    assert "hidden_size" not in raw and raw["text_config"]["hidden_size"] == 16
    assert raw["vision_config"]["pos_emb_width"] == 3
    assert "navit_config" not in raw["vision_config"]
    assert resolve_custom_config_cls("muse_spark") is MuseSparkConfig
    assert MODEL_ARCH_MAPPING["MuseSparkForConditionalGeneration"][1] == "MuseSpark1xOmniForCausalLM"
    restored = get_hf_config(str(tmp_path), "sdpa", num_hidden_layers=2)
    assert isinstance(restored, MuseSparkConfig)
    assert restored.num_hidden_layers == 2
    assert restored.no_rope_layers[:2] == [False, True]
    restored.save_pretrained(tmp_path / "prefix")
    prefix = get_hf_config(str(tmp_path / "prefix"), "sdpa")
    assert prefix.num_hidden_layers == 2
    assert prefix.layer_types == ["sliding_attention", "full_attention"]
    assert prefix.no_rope_layers == [False, True]
    assert prefix.output_multiplier == prefix.router_output_multiplier == 0.5
    assert prefix.residual_gate_temperature == 0.7


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("num_shared_experts", 1, "shared experts"),
        ("hidden_act", "gelu", "SwiGLU"),
        ("layer_rope_theta", [1, 2, 3, 4], "RoPE theta"),
        ("moe_hidden_size", 7, "divide hidden_size"),
    ],
)
def test_v2_config_rejects_unsupported_text(field: str, value: object, match: str) -> None:
    data = tiny_hf_config().to_dict()
    data["text_config"][field] = value
    with pytest.raises(ValueError, match=match):
        MuseSparkConfig(**data)


def test_v2_literal_key_mapping_and_expert_math() -> None:
    model = MuseSpark1xOmniForCausalLM(tiny_hf_config(), backend=tiny_backend())
    adapter = model.state_dict_adapter
    raw = {
        "model.language_model.layers.0.mlp.experts.gate_up_proj": torch.randn(4, 16, 8),
        "model.language_model.layers.0.mlp.experts.down_proj": torch.randn(4, 8, 8),
        "model.language_model.layers.0.mlp.gate.weight": torch.randn(4, 16),
        "model.language_model.layers.0.mlp.gate.e_score_correction_bias": torch.randn(4),
        "model.vision_tower.layers.0.attn.q_proj.bias": torch.arange(8.0),
        "model.audio_tower.layers.1.conv_blocks.0.mlp.fc1.weight": torch.randn(32, 8),
    }
    native = adapter.from_hf(raw)
    torch.testing.assert_close(
        native["model.layers.0.feed_forward.gate.weight"],
        raw["model.language_model.layers.0.mlp.gate.weight"],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        native["vision_model.encoder.layers.0.attn.q_proj.bias"],
        torch.tensor([0.0, 2.0, 1.0, 3.0, 4.0, 6.0, 5.0, 7.0]),
        rtol=0,
        atol=0,
    )
    assert "audio_encoder.stages.1.1.linear1.weight" in native
    x = torch.randn(3, 8)
    gate, up = F.linear(x, raw["model.language_model.layers.0.mlp.experts.gate_up_proj"][2]).chunk(2, dim=-1)
    expected = F.linear(F.silu(gate) * up, raw["model.language_model.layers.0.mlp.experts.down_proj"][2])
    gate, up = (x @ native["model.layers.0.feed_forward.experts.gate_and_up_projs"][2]).chunk(2, dim=-1)
    actual = (F.silu(gate) * up) @ native["model.layers.0.feed_forward.experts.down_projs"][2]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    exported = adapter.to_hf(native)
    assert set(exported) == set(raw)
    for key in raw:
        torch.testing.assert_close(exported[key], raw[key], rtol=0, atol=0)


def test_v2_rope_conversion_forward_and_backward() -> None:
    torch.manual_seed(11)
    model = MuseSpark1xOmniForCausalLM(tiny_hf_config(), backend=tiny_backend())
    raw_weight = torch.randn(16, 16, dtype=torch.float64, requires_grad=True)
    x = torch.randn(3, 16, dtype=torch.float64, requires_grad=True)
    native = model.state_dict_adapter.from_hf({"model.language_model.layers.0.self_attn.q_proj.weight": raw_weight})
    phase = torch.randn(3, 2, dtype=torch.float64)
    half_q = F.linear(x, raw_weight).reshape(3, 4, 4)
    first, second = half_q.chunk(2, -1)
    cosine, sine = phase.cos()[:, None], phase.sin()[:, None]
    expected = torch.cat((first * cosine - second * sine, second * cosine + first * sine), -1)
    adjacent_q = F.linear(x, native["model.layers.0.self_attn.q_proj.weight"]).reshape(3, 4, 2, 2)
    complex_q = torch.view_as_complex(adjacent_q.contiguous())
    rotated = torch.view_as_real(complex_q * torch.polar(torch.ones_like(phase), phase)[:, None])
    actual = rotated.transpose(-1, -2).flatten(-2)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    upstream = torch.randn_like(actual)
    expected_grads = torch.autograd.grad((expected * upstream).sum(), (x, raw_weight), retain_graph=True)
    actual_grads = torch.autograd.grad((actual * upstream).sum(), (x, raw_weight))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, atol=1e-12, rtol=1e-12)


def test_v2_direct_destinations_reach_model_storage() -> None:
    model = MuseSpark1xOmniForCausalLM(tiny_hf_config(), backend=tiny_backend())
    model.initialize_weights(torch.device("cpu"), dtype=torch.float32)
    adapter = model.state_dict_adapter
    assert adapter.supports_low_memory_dcp_load
    state = model.state_dict()
    destinations = adapter.to_hf(state, for_checkpoint_load=True)
    raw = {}
    with torch.no_grad():
        for i, (key, tensor) in enumerate(destinations.items()):
            value = torch.arange(tensor.numel(), dtype=tensor.dtype).reshape(tensor.shape) / max(tensor.numel(), 1) + i
            tensor.copy_(value)
            raw[key] = value
    expert = state["model.layers.0.feed_forward.experts.gate_and_up_projs"]
    hf_expert = destinations["model.language_model.layers.0.mlp.experts.gate_up_proj"]
    assert hf_expert.untyped_storage().data_ptr() == expert.untyped_storage().data_ptr()
    torch.testing.assert_close(
        expert.transpose(-1, -2), raw["model.language_model.layers.0.mlp.experts.gate_up_proj"], rtol=0, atol=0
    )
    converted = adapter.from_hf(destinations)
    assert (
        converted["model.layers.0.feed_forward.experts.gate_and_up_projs"].untyped_storage().data_ptr()
        == expert.untyped_storage().data_ptr()
    )
    model.load_state_dict(converted, strict=True)
    exported = adapter.to_hf(model.state_dict())
    for key in raw:
        torch.testing.assert_close(exported[key], raw[key], rtol=0, atol=0)


@pytest.mark.runtime_budget(30, reason="Real DCP load, multimodal forward/backward, and consolidated save/reload")
def test_v2_real_dcp_load_multimodal_gradients_and_save_reload(tmp_path: Path) -> None:
    torch.manual_seed(19)
    config = tiny_hf_config()
    reference = MuseSpark1xOmniForCausalLM(config, backend=tiny_backend())
    reference.initialize_weights(torch.device("cpu"), dtype=torch.float32)
    source = tmp_path / "source"
    source.mkdir()
    raw = {
        key: value.detach().contiguous()
        for key, value in reference.state_dict_adapter.to_hf(reference.state_dict()).items()
    }
    save_file(raw, source / "model.safetensors")
    config.save_pretrained(source)
    actual = MuseSpark1xOmniForCausalLM(get_hf_config(str(source), "sdpa"), backend=tiny_backend())
    actual.initialize_weights(torch.device("cpu"), dtype=torch.float32)
    checkpointer = Checkpointer(
        CheckpointingConfig(
            enabled=True,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            model_save_format="safetensors",
            save_consolidated=True,
        ),
        dp_rank=0,
        tp_rank=0,
        pp_rank=0,
    )
    with patch(
        "nemo_automodel.components.checkpoint.checkpointing._load_hf_checkpoint_preserving_dtype",
        side_effect=AssertionError("v2 must use direct DCP destinations"),
    ):
        checkpointer.load_model(actual, str(source), is_init_step=True)
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(actual.state_dict()[name], value, rtol=0, atol=0)

    inputs = dict(
        input_ids=torch.tensor([[1, 61, 2, 63, 63, 3]]),
        pixel_values=torch.randn(4, 24),
        image_grid_thw=torch.tensor([[1, 2, 2]]),
        audio_values=[torch.randn(8)],
    )
    expected = reference(**inputs).logits
    observed = actual(**inputs).logits
    torch.testing.assert_close(observed, expected, atol=1e-6, rtol=1e-6)
    upstream = torch.randn_like(expected)
    expected.backward(upstream)
    observed.backward(upstream)
    expected_params = dict(reference.named_parameters())
    for name, parameter in actual.named_parameters():
        assert parameter.grad is not None, name
        torch.testing.assert_close(parameter.grad, expected_params[name].grad, atol=1e-6, rtol=1e-6)
    torch.optim.SGD(actual.parameters(), lr=1e-4).step()
    checkpointer.save_model(actual, str(tmp_path / "saved"))
    reloaded = MuseSpark1xOmniForCausalLM(copy.deepcopy(config), backend=tiny_backend())
    reloaded.initialize_weights(torch.device("cpu"), dtype=torch.float32)
    checkpointer.load_model(reloaded, str(tmp_path / "saved/model"))
    for name, value in actual.state_dict().items():
        torch.testing.assert_close(reloaded.state_dict()[name], value, rtol=0, atol=0)


def test_v2_lora_native_state_roundtrip_and_base_key_discovery() -> None:
    """Preserve native adapter factors while converting the base checkpoint."""
    model = MuseSpark1xOmniForCausalLM(tiny_hf_config(), backend=tiny_backend())
    adapter = model.state_dict_adapter
    factors = {
        "model.layers.0.self_attn.q_proj.lora_A.weight": torch.randn(3, 16),
        "model.layers.0.self_attn.q_proj.lora_B.weight": torch.randn(16, 3),
        "model.layers.0.self_attn.k_proj.lora_A.weight": torch.randn(3, 16),
        "model.layers.0.self_attn.k_proj.lora_B.weight": torch.randn(8, 3),
        "model.layers.0.feed_forward.experts.lora_gate_and_up_A": torch.randn(4, 8, 3),
        "model.layers.0.feed_forward.experts.lora_gate_and_up_B": torch.randn(4, 3, 16),
        "model.layers.0.feed_forward.experts.lora_down_A": torch.randn(4, 8, 3),
        "model.layers.0.feed_forward.experts.lora_down_B": torch.randn(4, 3, 8),
    }
    base_name = "model.layers.0.self_attn.q_proj.weight"
    state = {base_name: torch.randn(16, 16), **factors}
    converted = adapter.to_hf(state)
    assert set(adapter.get_hf_state_dict_keys(state)) == set(converted)
    for key, tensor in factors.items():
        assert converted[key] is tensor
    restored = adapter.from_hf(converted)
    for key, tensor in state.items():
        torch.testing.assert_close(restored[key], tensor, rtol=0, atol=0)
    base_only = adapter.to_hf(state, exclude_key_regex=r".*\.lora_.*")
    assert set(base_only) == {"model.language_model.layers.0.self_attn.q_proj.weight"}


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
