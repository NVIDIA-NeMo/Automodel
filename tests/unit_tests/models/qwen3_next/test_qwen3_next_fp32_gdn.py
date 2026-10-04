# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Precision-contract tests for Qwen3-Next's fp32 GatedDeltaNet gate parameters."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F
from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.utils import cast_model_to_dtype
from nemo_automodel.components.models.qwen3_next.layers import Qwen3NextFp32GatedDeltaNet
from nemo_automodel.components.models.qwen3_next.model import Qwen3NextForCausalLM
from nemo_automodel.shared.parameter_names import canonical_parameter_fqn

GATE_PARAM_KEYS = frozenset({"model.layers.0.linear_attn.A_log", "model.layers.0.linear_attn.dt_bias"})


def _tiny_config(layer_types: list[str]) -> Qwen3NextConfig:
    return Qwen3NextConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        num_experts=4,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        num_hidden_layers=len(layer_types),
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        max_position_embeddings=16,
        rms_norm_eps=1e-6,
        layer_types=layer_types,
    )


def _backend() -> BackendConfig:
    return BackendConfig(
        attn="sdpa",
        linear="torch",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        rope_fusion=False,
        enable_hf_state_dict_adapter=True,
    )


def _tiny_model() -> Qwen3NextForCausalLM:
    return Qwen3NextForCausalLM(_tiny_config(["linear_attention", "full_attention"]), backend=_backend())


def _gdn(default_dtype: torch.dtype = torch.float32) -> Qwen3NextFp32GatedDeltaNet:
    cfg = _tiny_config(["linear_attention"])
    cfg.torch_dtype = default_dtype
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(default_dtype)
        return Qwen3NextFp32GatedDeltaNet(cfg, layer_idx=0)
    finally:
        torch.set_default_dtype(previous)


def _assert_hf_init_values(a_log: torch.Tensor, dt_bias: torch.Tensor) -> None:
    assert a_log.dtype == torch.float32
    assert dt_bias.dtype == torch.float32
    assert torch.equal(dt_bias, torch.ones_like(dt_bias))
    assert torch.all(a_log <= math.log(16.0))
    assert torch.isfinite(a_log).all()


def test_constructor_keeps_gate_params_fp32_under_bf16_default_dtype():
    gdn = _gdn(torch.bfloat16)

    assert "A_log" in gdn._parameters
    assert "dt_bias" in gdn._parameters
    assert "_fp32_params" not in gdn._modules
    assert set(gdn.state_dict()) >= {"A_log", "dt_bias"}
    assert gdn.in_proj_qkvz.weight.dtype == torch.bfloat16
    _assert_hf_init_values(gdn.A_log.detach(), gdn.dt_bias.detach())


def test_compute_gate_matches_hf_formula_in_fp32():
    gdn = _gdn()
    a = torch.randn(2, 3, gdn.num_v_heads)

    g = gdn._compute_gate(a)

    expected = -gdn.A_log.exp() * F.softplus(a + gdn.dt_bias)
    assert g.dtype == torch.float32
    torch.testing.assert_close(g, expected)


def test_compute_gate_casts_bf16_input_to_fp32_and_backpropagates():
    gdn = _gdn()
    a = torch.randn(2, 3, gdn.num_v_heads).to(torch.bfloat16)

    g = gdn._compute_gate(a)
    g.sum().backward()

    expected = -gdn.A_log.detach().exp() * F.softplus(a.float() + gdn.dt_bias.detach())
    assert g.dtype == torch.float32
    torch.testing.assert_close(g, expected)
    for param in (gdn.A_log, gdn.dt_bias):
        assert param.grad is not None
        assert param.grad.dtype == torch.float32
        assert torch.isfinite(param.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Qwen3NextModel builds on the current CUDA device")
def test_strict_fp32_tokens_match_exactly_the_gate_params():
    model = _tiny_model()
    tokens = Qwen3NextForCausalLM._keep_in_fp32_modules_strict

    matched = {
        name for name, _ in model.named_parameters() if any(tok in canonical_parameter_fqn(name) for tok in tokens)
    }

    assert tokens == ["linear_attn.A_log", "linear_attn.dt_bias"]
    assert matched == GATE_PARAM_KEYS
    assert not any("_fp32_params" in key for key in model.state_dict())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Qwen3NextModel builds on the current CUDA device")
def test_cast_model_to_dtype_restores_gate_params_fp32():
    model = _tiny_model()

    cast_model_to_dtype(model, torch.bfloat16)

    linear_attn = model.model.layers["0"].linear_attn
    assert linear_attn.A_log.dtype == torch.float32
    assert linear_attn.dt_bias.dtype == torch.float32
    assert linear_attn.in_proj_qkvz.weight.dtype == torch.bfloat16
    assert model.model.embed_tokens.weight.dtype == torch.bfloat16
    assert model.lm_head.weight.dtype == torch.bfloat16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Qwen3NextModel builds on the current CUDA device")
def test_initialize_weights_bf16_keeps_gate_params_fp32_with_exact_values():
    model = _tiny_model()

    model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.bfloat16)

    linear_attn = model.model.layers["0"].linear_attn
    _assert_hf_init_values(linear_attn.A_log.detach(), linear_attn.dt_bias.detach())
    assert linear_attn.in_proj_qkvz.weight.dtype == torch.bfloat16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Qwen3NextModel builds on the current CUDA device")
def test_state_dict_round_trips_to_hf_keys():
    model = _tiny_model()
    state_dict = model.state_dict()

    hf_state_dict = model.state_dict_adapter.to_hf(state_dict)
    native_state_dict = model.state_dict_adapter.from_hf(hf_state_dict)

    assert GATE_PARAM_KEYS <= set(hf_state_dict)
    assert all(hf_state_dict[key].dtype == torch.float32 for key in GATE_PARAM_KEYS)
    assert set(native_state_dict) == set(state_dict)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Qwen3NextModel builds on the current CUDA device")
@pytest.mark.parametrize("key", sorted(GATE_PARAM_KEYS))
def test_from_hf_upcasts_bf16_gate_params(key):
    model = _tiny_model()
    hf_state_dict = model.state_dict_adapter.to_hf(model.state_dict())
    hf_state_dict[key] = hf_state_dict[key].to(torch.bfloat16)

    native_state_dict = model.state_dict_adapter.from_hf(hf_state_dict)

    assert native_state_dict[key].dtype == torch.float32
