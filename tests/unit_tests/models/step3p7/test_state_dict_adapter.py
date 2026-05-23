from __future__ import annotations

import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.step3p7.configuration_step3p7 import Step3p7Config
from nemo_automodel.components.models.step3p7.state_dict_adapter import Step3p7StateDictAdapter
from nemo_automodel.components.moe.config import MoEConfig


def _adapter(dtype=torch.float32):
    config = Step3p7Config(
        text_config={
            "hidden_size": 4,
            "intermediate_size": 8,
            "moe_intermediate_size": 2,
            "moe_num_experts": 3,
            "moe_top_k": 1,
            "num_hidden_layers": 1,
            "vocab_size": 16,
            "head_dim": 2,
        }
    )
    moe_config = MoEConfig(
        dim=4,
        inter_dim=8,
        moe_inter_dim=2,
        n_routed_experts=3,
        n_shared_experts=0,
        n_activated_experts=1,
        n_expert_groups=0,
        n_limited_groups=0,
        train_gate=True,
        gate_bias_update_factor=0.0,
        score_func="softmax",
        route_scale=1.0,
        aux_loss_coeff=0.0,
        norm_topk_prob=True,
        router_bias=False,
        expert_bias=False,
        expert_activation="swiglu",
        dtype=dtype,
    )
    backend = BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        enable_hf_state_dict_adapter=True,
    )
    return Step3p7StateDictAdapter(config, moe_config, backend, dtype=dtype)


def test_static_key_mapping_helpers_cover_all_prefixes():
    assert Step3p7StateDictAdapter._is_text_key("model.layers.0.weight")
    assert Step3p7StateDictAdapter._is_text_key("model.language_model.layers.0.weight")
    assert Step3p7StateDictAdapter._is_text_key("language_model.layers.0.weight")
    assert not Step3p7StateDictAdapter._is_text_key("vision_model.weight")

    assert Step3p7StateDictAdapter._to_text_hf_key("model.language_model.layers.0.weight") == ("model.layers.0.weight")
    assert Step3p7StateDictAdapter._to_text_hf_key("language_model.layers.0.weight") == "model.layers.0.weight"
    assert Step3p7StateDictAdapter._to_text_hf_key("model.layers.0.weight") == "model.layers.0.weight"

    assert Step3p7StateDictAdapter._to_native_text_key("model.layers.0.weight") == (
        "model.language_model.layers.0.weight"
    )
    assert Step3p7StateDictAdapter._to_native_text_key("language_model.layers.0.weight") == (
        "model.language_model.layers.0.weight"
    )
    assert Step3p7StateDictAdapter._to_native_text_key("layers.0.weight") == ("model.language_model.layers.0.weight")

    assert Step3p7StateDictAdapter._map_non_text_from_hf("foo.weight_scale_inv") is None
    assert Step3p7StateDictAdapter._map_non_text_from_hf("vision_model.conv.weight") == (
        "model.vision_model.conv.weight"
    )
    assert Step3p7StateDictAdapter._map_non_text_from_hf("vit_large_projector.weight") == (
        "model.vit_large_projector.weight"
    )
    assert Step3p7StateDictAdapter._map_non_text_from_hf("other.weight") == "other.weight"

    assert Step3p7StateDictAdapter._map_non_text_to_hf("model.vision_model.conv.weight") == ("vision_model.conv.weight")
    assert Step3p7StateDictAdapter._map_non_text_to_hf("model.vit_large_projector.weight") == (
        "vit_large_projector.weight"
    )
    assert Step3p7StateDictAdapter._map_non_text_to_hf("other.weight") == "other.weight"


def test_from_hf_maps_text_vision_projector_router_bias_and_ignores_fp8_scale():
    adapter = _adapter(dtype=torch.float32)
    gate = torch.arange(3 * 2 * 4, dtype=torch.float32).reshape(3, 2, 4)
    up = gate + 100
    router_bias = torch.tensor([1.0, 2.0, 3.0])
    vision_weight = torch.randn(2, 2)
    projector_weight = torch.randn(4, 4)

    native = adapter.from_hf(
        {
            "model.layers.0.moe.gate_proj.weight": gate,
            "model.layers.0.moe.up_proj.weight": up,
            "model.layers.0.moe.router_bias": router_bias,
            "model.norm.weight": torch.ones(4),
            "language_model.embed_tokens.weight": torch.ones(16, 4),
            "vision_model.conv.weight": vision_weight,
            "vit_large_projector.weight": projector_weight,
            "vit_large_projector.weight_scale_inv": torch.ones(1),
        }
    )

    assert "model.language_model.layers.0.moe.experts.gate_and_up_projs" in native
    assert "model.language_model.layers.0.moe.gate.e_score_correction_bias" in native
    torch.testing.assert_close(
        native["model.language_model.layers.0.moe.gate.e_score_correction_bias"],
        router_bias,
    )
    assert "model.language_model.norm.weight" in native
    assert "model.language_model.embed_tokens.weight" in native
    assert native["model.vision_model.conv.weight"] is vision_weight
    assert native["model.vit_large_projector.weight"] is projector_weight
    assert all("weight_scale_inv" not in key for key in native)


def test_to_hf_maps_text_and_non_text_keys_with_exclude_filter():
    adapter = _adapter(dtype=torch.float32)
    native = {
        "model.language_model.layers.0.moe.gate.e_score_correction_bias": torch.tensor([1.0, 2.0, 3.0]),
        "model.vision_model.conv.weight": torch.randn(2, 2),
        "model.vit_large_projector.weight": torch.randn(4, 4),
        "other.weight": torch.randn(1),
    }

    hf = adapter.to_hf(native, exclude_key_regex=r"vision_model\..*")

    assert "model.layers.0.moe.router_bias" in hf
    assert "vision_model.conv.weight" not in hf
    assert "vit_large_projector.weight" in hf
    assert "other.weight" in hf


def test_convert_single_tensor_to_hf_passthrough_and_exclude():
    adapter = _adapter(dtype=torch.float32)
    tensor = torch.ones(1)

    assert adapter.convert_single_tensor_to_hf("model.vision_model.weight", tensor) == [("vision_model.weight", tensor)]
    assert (
        adapter.convert_single_tensor_to_hf(
            "model.vision_model.weight",
            tensor,
            exclude_key_regex=r"vision_model\..*",
        )
        == []
    )
