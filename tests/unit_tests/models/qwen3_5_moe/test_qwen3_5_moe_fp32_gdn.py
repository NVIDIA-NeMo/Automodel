# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the Qwen3.5-MoE fp32 GatedDeltaNet decay-gate contract."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("transformers.models.qwen3_5_moe")

from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig

from nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn import CPAwareGatedDeltaNet


def _text_config() -> Qwen3_5MoeTextConfig:
    layer_types = ["linear_attention"]
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
