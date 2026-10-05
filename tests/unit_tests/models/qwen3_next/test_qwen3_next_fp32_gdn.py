# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Precision-contract tests for Qwen3-Next's fp32 GatedDeltaNet gate parameters."""

from __future__ import annotations

import math

import torch
from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig

from nemo_automodel.components.models.qwen3_next.layers import Qwen3NextFp32GatedDeltaNet


def _tiny_config() -> Qwen3NextConfig:
    return Qwen3NextConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        num_experts=4,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        num_hidden_layers=1,
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
        layer_types=["linear_attention"],
    )


def test_constructor_keeps_gate_params_fp32_under_bf16_default_dtype():
    """The override rebuilds ``A_log`` / ``dt_bias`` fp32 with HF's init values under a bf16 default dtype.

    The strict-contract and gate tests cover the model-level behaviour; this is the CPU check
    of the layer itself (``Qwen3NextForCausalLM`` builds on the current CUDA device).
    """
    cfg = _tiny_config()
    cfg.torch_dtype = torch.bfloat16
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        gdn = Qwen3NextFp32GatedDeltaNet(cfg, layer_idx=0)
    finally:
        torch.set_default_dtype(previous)

    assert "A_log" in gdn._parameters and "dt_bias" in gdn._parameters
    assert gdn.A_log.dtype == torch.float32 and gdn.dt_bias.dtype == torch.float32
    assert gdn.in_proj_qkvz.weight.dtype == torch.bfloat16
    assert torch.equal(gdn.dt_bias, torch.ones_like(gdn.dt_bias))
    assert torch.all(gdn.A_log <= math.log(16.0)) and torch.isfinite(gdn.A_log).all()
