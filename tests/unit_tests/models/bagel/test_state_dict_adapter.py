# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from nemo_automodel.components.models.bagel.state_dict_adapter import (
    BagelStateDictAdapter,
    fuse_qwen_split_projections,
)


def test_bagel_adapter_roots_upstream_keys_for_am_model():
    adapter = BagelStateDictAdapter(stage="stage2")
    weight = torch.ones(2, 2)
    bias = torch.zeros(2)

    converted = adapter.from_hf(
        {
            "language_model.model.embed_tokens.weight": weight,
            "time_embedder.mlp.0.bias": bias,
        }
    )

    assert set(converted) == {
        "model.language_model.model.embed_tokens.weight",
        "model.time_embedder.mlp.0.bias",
    }
    assert converted["model.language_model.model.embed_tokens.weight"] is weight
    assert converted["model.time_embedder.mlp.0.bias"] is bias


def test_bagel_adapter_accepts_already_rooted_am_keys():
    adapter = BagelStateDictAdapter(stage="stage2")
    weight = torch.ones(2, 2)

    converted = adapter.from_hf({"model.language_model.model.embed_tokens.weight": weight})

    assert set(converted) == {"model.language_model.model.embed_tokens.weight"}
    assert converted["model.language_model.model.embed_tokens.weight"] is weight


def test_bagel_adapter_unroots_am_keys_when_saving():
    adapter = BagelStateDictAdapter(stage="stage2")
    weight = torch.ones(2, 2)

    converted = adapter.to_hf(
        {
            "model.language_model.model.embed_tokens.weight": weight,
            "model._extra_state": torch.zeros(1),
        },
        exclude_key_regex=r".*_extra_state.*",
    )

    assert set(converted) == {"language_model.model.embed_tokens.weight"}
    assert converted["language_model.model.embed_tokens.weight"] is weight


def _projection_adapter(*, fused: bool) -> BagelStateDictAdapter:
    text_config = SimpleNamespace(
        hidden_size=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=2,
        fused_projections=fused,
    )
    return BagelStateDictAdapter(config=SimpleNamespace(text_config=text_config), stage="stage1")


def _split_projection_state() -> dict[str, torch.Tensor]:
    prefix = "language_model.model.layers.0"
    return {
        f"{prefix}.self_attn.q_proj.weight": torch.full((8, 8), 1.0),
        f"{prefix}.self_attn.k_proj.weight": torch.full((4, 8), 2.0),
        f"{prefix}.self_attn.v_proj.weight": torch.full((4, 8), 3.0),
        f"{prefix}.self_attn.q_proj.bias": torch.full((8,), 4.0),
        f"{prefix}.self_attn.k_proj.bias": torch.full((4,), 5.0),
        f"{prefix}.self_attn.v_proj.bias": torch.full((4,), 6.0),
        f"{prefix}.mlp.gate_proj.weight": torch.full((16, 8), 7.0),
        f"{prefix}.mlp.up_proj.weight": torch.full((16, 8), 8.0),
    }


def test_qwen_backbone_projection_fusion_accepts_raw_hf_keys():
    prefix = "model.layers.0"
    state = {
        f"{prefix}.self_attn.q_proj.weight": torch.full((8, 8), 1.0),
        f"{prefix}.self_attn.k_proj.weight": torch.full((4, 8), 2.0),
        f"{prefix}.self_attn.v_proj.weight": torch.full((4, 8), 3.0),
        f"{prefix}.mlp.gate_proj.weight": torch.full((16, 8), 4.0),
        f"{prefix}.mlp.up_proj.weight": torch.full((16, 8), 5.0),
    }

    fuse_qwen_split_projections(state)

    assert set(state) == {
        f"{prefix}.self_attn.qkv_proj.weight",
        f"{prefix}.mlp.gate_up_proj.weight",
    }
    torch.testing.assert_close(state[f"{prefix}.self_attn.qkv_proj.weight"][:8], torch.full((8, 8), 1.0))
    torch.testing.assert_close(state[f"{prefix}.mlp.gate_up_proj.weight"][16:], torch.full((16, 8), 5.0))


def test_bagel_adapter_keeps_split_projections_when_fusion_is_disabled():
    state = _split_projection_state()

    converted = _projection_adapter(fused=False).from_hf(state)

    assert set(converted) == {f"model.{key}" for key in state}
    assert not any("qkv_proj" in key or "gate_up_proj" in key for key in converted)


def test_bagel_adapter_fused_projection_round_trip_is_exact():
    state = _split_projection_state()
    adapter = _projection_adapter(fused=True)

    converted = adapter.from_hf(state)

    qkv = converted["model.language_model.model.layers.0.self_attn.qkv_proj.weight"]
    gate_up = converted["model.language_model.model.layers.0.mlp.gate_up_proj.weight"]
    assert qkv.shape == (16, 8)
    assert gate_up.shape == (32, 8)
    torch.testing.assert_close(qkv[:8], state["language_model.model.layers.0.self_attn.q_proj.weight"])
    torch.testing.assert_close(qkv[8:12], state["language_model.model.layers.0.self_attn.k_proj.weight"])
    torch.testing.assert_close(qkv[12:], state["language_model.model.layers.0.self_attn.v_proj.weight"])

    restored = adapter.to_hf(converted)
    assert set(restored) == set(state)
    for key, tensor in state.items():
        torch.testing.assert_close(restored[key], tensor)


def test_bagel_adapter_splits_fused_checkpoint_for_split_model():
    state = _split_projection_state()
    fused_state = _projection_adapter(fused=True).from_hf(state)

    converted = _projection_adapter(fused=False).from_hf(fused_state)

    assert set(converted) == {f"model.{key}" for key in state}
    for key, tensor in state.items():
        torch.testing.assert_close(converted[f"model.{key}"], tensor)


def test_bagel_adapter_does_not_fuse_siglip_attention_projections():
    prefix = "vit_model.vision_model.encoder.layers.0.self_attn"
    state = {
        f"{prefix}.q_proj.weight": torch.full((4, 4), 1.0),
        f"{prefix}.k_proj.weight": torch.full((4, 4), 2.0),
        f"{prefix}.v_proj.weight": torch.full((4, 4), 3.0),
    }

    converted = _projection_adapter(fused=True).from_hf(state)

    assert set(converted) == {f"model.{key}" for key in state}
    assert not any("qkv_proj" in key for key in converted)


def test_bagel_adapter_rejects_mixed_split_and_fused_projection_layouts():
    state = _split_projection_state()
    state["language_model.model.layers.0.self_attn.qkv_proj.weight"] = torch.zeros(16, 8)

    with pytest.raises(KeyError, match="both"):
        _projection_adapter(fused=True).from_hf(state)


def test_bagel_adapter_rejects_partial_projection_groups_when_fused():
    state = _split_projection_state()
    del state["language_model.model.layers.0.self_attn.v_proj.weight"]

    with pytest.raises(KeyError, match="v_proj"):
        _projection_adapter(fused=True).from_hf(state)


def test_bagel_adapter_round_trips_generation_projection_names():
    prefix = "language_model.model.layers.0"
    state = {
        f"{prefix}.self_attn.q_proj_moe_gen.weight": torch.full((8, 8), 1.0),
        f"{prefix}.self_attn.k_proj_moe_gen.weight": torch.full((4, 8), 2.0),
        f"{prefix}.self_attn.v_proj_moe_gen.weight": torch.full((4, 8), 3.0),
        f"{prefix}.mlp_moe_gen.gate_proj.weight": torch.full((16, 8), 4.0),
        f"{prefix}.mlp_moe_gen.up_proj.weight": torch.full((16, 8), 5.0),
    }
    adapter = _projection_adapter(fused=True)
    adapter.stage = "stage2"

    converted = adapter.from_hf(state)

    assert f"model.{prefix}.self_attn.qkv_proj_moe_gen.weight" in converted
    assert f"model.{prefix}.mlp_moe_gen.gate_up_proj.weight" in converted
    restored = adapter.to_hf(converted)
    assert set(restored) == set(state)
    for key, tensor in state.items():
        torch.testing.assert_close(restored[key], tensor)


def test_bagel_adapter_native_checkpoint_keeps_fused_tensor_identity():
    adapter = _projection_adapter(fused=True)
    key = "model.language_model.model.layers.0.self_attn.qkv_proj.weight"
    extra_state_key = "model.language_model.model.layers.0.self_attn.qkv_proj._extra_state"
    tensor = torch.randn(16, 8)
    extra_state = torch.tensor([7])

    checkpoint_state = adapter.to_hf(
        {key: tensor, extra_state_key: extra_state},
        native_checkpoint=True,
        exclude_key_regex=r".*_extra_state.*",
    )
    restored = adapter.from_hf(checkpoint_state, native_checkpoint=True)

    assert set(checkpoint_state) == {key, extra_state_key}
    assert checkpoint_state[key] is tensor
    assert checkpoint_state[extra_state_key] is extra_state
    assert restored[key] is tensor
    assert restored[extra_state_key] is extra_state


def test_bagel_adapter_rejects_row_sharded_fused_export():
    adapter = _projection_adapter(fused=True)
    key = "model.language_model.model.layers.0.self_attn.qkv_proj.weight"
    fake_dtensor = SimpleNamespace(placements=(SimpleNamespace(dim=0),))

    with pytest.raises(RuntimeError, match="row-sharded DTensor"):
        adapter.to_hf({key: fake_dtensor})


def test_bagel_adapter_rejects_head_dim_that_disagrees_with_model():
    text_config = SimpleNamespace(
        hidden_size=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        fused_projections=True,
    )
    adapter = BagelStateDictAdapter(config=SimpleNamespace(text_config=text_config), stage="stage1")
    key = "model.language_model.model.layers.0.self_attn.qkv_proj.weight"

    with pytest.raises(ValueError, match="disagrees"):
        adapter.to_hf({key: torch.zeros(16, 8)})


def test_bagel_adapter_converts_single_fused_tensor_for_hf_export():
    adapter = _projection_adapter(fused=True)
    tensor = torch.arange(16 * 8, dtype=torch.float32).reshape(16, 8)

    converted = adapter.convert_single_tensor_to_hf(
        "model.language_model.model.layers.0.self_attn.qkv_proj.weight",
        tensor,
    )

    assert [key for key, _ in converted] == [
        "language_model.model.layers.0.self_attn.q_proj.weight",
        "language_model.model.layers.0.self_attn.k_proj.weight",
        "language_model.model.layers.0.self_attn.v_proj.weight",
    ]
    torch.testing.assert_close(converted[0][1], tensor[:8])
    torch.testing.assert_close(converted[1][1], tensor[8:12])
    torch.testing.assert_close(converted[2][1], tensor[12:])
    assert adapter.convert_single_tensor_to_hf("model.layer._extra_state", torch.zeros(1)) == []


def test_bagel_adapter_drops_te_extra_state():
    adapter = _projection_adapter(fused=False)

    converted = adapter.from_hf(
        {
            "language_model.model.embed_tokens.weight": torch.ones(2, 2),
            "language_model.model.layers.0.self_attn.q_proj._extra_state": torch.zeros(1),
        }
    )

    assert set(converted) == {"model.language_model.model.embed_tokens.weight"}
