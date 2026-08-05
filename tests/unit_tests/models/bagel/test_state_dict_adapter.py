# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import torch

from nemo_automodel.components.models.bagel.state_dict_adapter import BagelStateDictAdapter


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


def test_bagel_adapter_drops_te_extra_state():
    adapter = BagelStateDictAdapter(stage="stage1")

    converted = adapter.from_hf(
        {
            "language_model.model.embed_tokens.weight": torch.ones(2, 2),
            "language_model.model.layers.0.self_attn.q_proj._extra_state": torch.zeros(1),
        }
    )

    assert set(converted) == {"model.language_model.model.embed_tokens.weight"}
    assert adapter.convert_single_tensor_to_hf("model.layer._extra_state", torch.zeros(1)) == []
