# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the Qwen3.5-MoE fp32 GatedDeltaNet decay-gate contract."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("transformers.models.qwen3_5_moe")

from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.utils import cast_model_to_dtype
from nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn import CPAwareGatedDeltaNet
from nemo_automodel.components.models.qwen3_5_moe.model import Qwen3_5MoeForCausalLM
from nemo_automodel.components.moe.layers import MoEConfig


def _text_config(layer_types: list[str] | None = None) -> Qwen3_5MoeTextConfig:
    layer_types = layer_types or ["linear_attention"]
    return Qwen3_5MoeTextConfig(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=len(layer_types),
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=64,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=16,
        rms_norm_eps=1e-6,
        router_aux_loss_coef=0.01,
        pad_token_id=0,
        tie_word_embeddings=False,
        layer_types=layer_types,
    )


def _moe_config(config: Qwen3_5MoeTextConfig) -> MoEConfig:
    return MoEConfig(
        dim=config.hidden_size,
        inter_dim=config.hidden_size,
        moe_inter_dim=config.moe_intermediate_size,
        n_routed_experts=config.num_experts,
        n_shared_experts=1,
        n_activated_experts=config.num_experts_per_tok,
        n_expert_groups=0,
        n_limited_groups=0,
        train_gate=True,
        gate_bias_update_factor=0.0,
        score_func="softmax",
        route_scale=1.0,
        aux_loss_coeff=config.router_aux_loss_coef,
        norm_topk_prob=True,
        expert_bias=False,
        router_bias=False,
        expert_activation="swiglu",
        softmax_before_topk=True,
        shared_expert_gate=True,
        shared_expert_inter_dim=config.shared_expert_intermediate_size,
    )


def _tiny_model(layer_types: list[str]) -> Qwen3_5MoeForCausalLM:
    config = _text_config(layer_types)
    backend = BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        dispatcher="torch",
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
    )
    return Qwen3_5MoeForCausalLM.from_config(config, moe_config=_moe_config(config), backend=backend)


def test_constructor_pins_gate_params_fp32_under_bf16_default_dtype():
    cfg = _text_config()
    cfg.torch_dtype = torch.bfloat16

    old_default_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        gdn = CPAwareGatedDeltaNet(cfg, layer_idx=0)
    finally:
        torch.set_default_dtype(old_default_dtype)

    # Bare parameters with the HF names: no holder submodule, no descriptor.
    assert gdn._parameters["A_log"] is gdn.A_log
    assert gdn._parameters["dt_bias"] is gdn.dt_bias
    assert "_fp32_params" not in gdn._modules
    assert gdn.A_log.dtype == torch.float32
    assert gdn.dt_bias.dtype == torch.float32
    assert gdn.in_proj_qkv.weight.dtype == torch.bfloat16


def test_compute_gate_runs_in_fp32_from_bf16_input():
    gdn = CPAwareGatedDeltaNet(_text_config(), layer_idx=0)
    with torch.no_grad():
        gdn.A_log.zero_()  # exp(0) = 1
        gdn.dt_bias.zero_()
    a = torch.randn(2, 3, gdn.num_v_heads, dtype=torch.bfloat16)

    out = gdn._compute_gate(a)

    assert out.dtype == torch.float32
    torch.testing.assert_close(out, -F.softplus(a.float()))


def test_strict_fp32_tokens_match_exactly_the_gate_params():
    model = _tiny_model(["linear_attention", "full_attention"])
    tokens = model._keep_in_fp32_modules_strict

    matched = {name for name, _ in model.named_parameters() if any(token in name for token in tokens)}
    assert matched == {"model.layers.0.linear_attn.A_log", "model.layers.0.linear_attn.dt_bias"}

    # FSDP matches the same tokens against layer-relative names.
    layer_matched = {
        name for name, _ in model.model.layers["0"].named_parameters() if any(token in name for token in tokens)
    }
    assert layer_matched == {"linear_attn.A_log", "linear_attn.dt_bias"}
    assert not any(any(token in name for token in tokens) for name, _ in model.model.layers["1"].named_parameters())


def test_cast_model_to_dtype_keeps_gate_params_fp32_with_exact_values():
    model = _tiny_model(["linear_attention"])
    model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.float32)
    linear_attn = model.model.layers["0"].linear_attn
    a_log_before = linear_attn.A_log.detach().clone()
    dt_bias_before = linear_attn.dt_bias.detach().clone()

    cast_model_to_dtype(model, torch.bfloat16)

    assert linear_attn.in_proj_qkv.weight.dtype == torch.bfloat16
    assert linear_attn.A_log.dtype == torch.float32
    assert linear_attn.dt_bias.dtype == torch.float32
    torch.testing.assert_close(linear_attn.A_log.detach(), a_log_before, rtol=0, atol=0)
    torch.testing.assert_close(linear_attn.dt_bias.detach(), dt_bias_before, rtol=0, atol=0)
    assert "model.layers.0.linear_attn.A_log" in model.state_dict()
