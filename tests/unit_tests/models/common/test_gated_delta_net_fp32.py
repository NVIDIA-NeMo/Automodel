# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for shared GatedDeltaNet fp32 checkpoint helpers."""

from __future__ import annotations

import torch

from nemo_automodel.components.models.common.gated_delta_net_fp32 import (
    FP32_GDN_PARAM_NAMES,
    has_gated_delta_net_fp32_checkpoint_contract,
    is_gated_delta_net_fp32_param_key,
    route_fp32_holder_key,
    strip_fp32_holder_key,
    upcast_gated_delta_net_fp32_state_tensor,
)


def test_strip_fp32_holder_key():
    assert strip_fp32_holder_key("model.layers.0.linear_attn._fp32_params.A_log") == (
        "model.layers.0.linear_attn.A_log"
    )
    assert strip_fp32_holder_key("model.layers.0.linear_attn._fp32_params.dt_bias") == (
        "model.layers.0.linear_attn.dt_bias"
    )
    assert strip_fp32_holder_key("model.layers.0.linear_attn.A_log") == "model.layers.0.linear_attn.A_log"
    assert strip_fp32_holder_key("model.layers.0.mlp.gate.weight") == "model.layers.0.mlp.gate.weight"


def test_route_fp32_holder_key():
    for name in FP32_GDN_PARAM_NAMES:
        assert route_fp32_holder_key(f"model.layers.0.linear_attn.{name}") == (
            f"model.layers.0.linear_attn._fp32_params.{name}"
        )

    assert route_fp32_holder_key("model.layers.0.linear_attn._fp32_params.A_log") == (
        "model.layers.0.linear_attn._fp32_params.A_log"
    )
    assert route_fp32_holder_key("model.layers.0.linear_attn.conv1d.weight") == (
        "model.layers.0.linear_attn.conv1d.weight"
    )
    assert route_fp32_holder_key("model.layers.0.self_attn.A_log") == "model.layers.0.self_attn.A_log"


def test_strip_route_round_trip():
    bare = "model.layers.3.linear_attn.A_log"
    routed = route_fp32_holder_key(bare)
    assert routed == "model.layers.3.linear_attn._fp32_params.A_log"
    assert strip_fp32_holder_key(routed) == bare


def test_is_gated_delta_net_fp32_param_key():
    assert is_gated_delta_net_fp32_param_key("model.layers.0.linear_attn.A_log")
    assert is_gated_delta_net_fp32_param_key("model.layers.0.linear_attn._fp32_params.dt_bias")
    assert not is_gated_delta_net_fp32_param_key("model.layers.0.self_attn.A_log")
    assert not is_gated_delta_net_fp32_param_key("model.layers.0.linear_attn.conv1d.weight")


def test_has_gated_delta_net_fp32_checkpoint_contract():
    assert has_gated_delta_net_fp32_checkpoint_contract(type("Cfg", (), {"architectures": ["Qwen3NextForCausalLM"]})())
    assert has_gated_delta_net_fp32_checkpoint_contract(
        type("Cfg", (), {"architectures": ["Qwen3_5MoeForConditionalGeneration"]})()
    )
    assert not has_gated_delta_net_fp32_checkpoint_contract(type("Cfg", (), {"architectures": ["LlamaForCausalLM"]})())


def test_upcast_gated_delta_net_fp32_state_tensor():
    tensor = torch.ones(4, dtype=torch.bfloat16)
    out = upcast_gated_delta_net_fp32_state_tensor("model.layers.0.linear_attn._fp32_params.dt_bias", tensor)
    assert out.dtype == torch.float32
    assert torch.equal(out, tensor.float())


def test_upcast_gated_delta_net_fp32_state_tensor_leaves_unrelated_state():
    tensor = torch.ones(4, dtype=torch.bfloat16)
    out = upcast_gated_delta_net_fp32_state_tensor("model.layers.0.self_attn.q_proj.weight", tensor)
    assert out is tensor


def test_upcast_gated_delta_net_fp32_state_tensor_handles_bare_key():
    tensor = torch.ones(4, dtype=torch.bfloat16)
    out = upcast_gated_delta_net_fp32_state_tensor("model.layers.0.linear_attn.A_log", tensor)
    assert out.dtype == torch.float32
