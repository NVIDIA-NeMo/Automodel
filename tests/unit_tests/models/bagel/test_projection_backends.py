# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

import torch
from transformers import Qwen2Config

from nemo_automodel.components.models.bagel.modeling_qwen2_packed import PackedAttention, Qwen2MLP
from nemo_automodel.components.models.common import BackendConfig

_TORCH_BACKEND = BackendConfig(attn="flex", linear="torch", rms_norm="torch_fp32", rope_fusion=False)


def _config(*, fused: bool) -> Qwen2Config:
    config = Qwen2Config(
        vocab_size=32,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        attention_dropout=0.0,
    )
    config.qk_norm = True
    config.fused_projections = fused
    return config


def test_mlp_projection_fusion_is_configurable_and_numerically_equivalent() -> None:
    torch.manual_seed(7)
    split = Qwen2MLP(_config(fused=False), backend=_TORCH_BACKEND)
    fused = Qwen2MLP(_config(fused=True), backend=_TORCH_BACKEND)
    with torch.no_grad():
        fused.gate_up_proj.weight.copy_(torch.cat([split.gate_proj.weight, split.up_proj.weight], dim=0))
        fused.down_proj.weight.copy_(split.down_proj.weight)

    hidden_states = torch.randn(3, 5, split.hidden_size)

    torch.testing.assert_close(fused(hidden_states), split(hidden_states), atol=1e-6, rtol=1e-5)
    assert hasattr(split, "gate_proj") and not hasattr(split, "gate_up_proj")
    assert hasattr(fused, "gate_up_proj") and not hasattr(fused, "gate_proj")


def test_attention_projection_fusion_uses_q_k_v_row_order() -> None:
    torch.manual_seed(11)
    split = PackedAttention(_config(fused=False), layer_idx=0, backend=_TORCH_BACKEND)
    fused = PackedAttention(_config(fused=True), layer_idx=0, backend=_TORCH_BACKEND)
    with torch.no_grad():
        fused.qkv_proj.weight.copy_(torch.cat([split.q_proj.weight, split.k_proj.weight, split.v_proj.weight], dim=0))
        fused.qkv_proj.bias.copy_(torch.cat([split.q_proj.bias, split.k_proj.bias, split.v_proj.bias], dim=0))

    hidden_states = torch.randn(9, split.hidden_size)
    qkv = fused.qkv_proj(hidden_states)
    fused_q, fused_k, fused_v = qkv.split([fused.q_size, fused.k_size, fused.v_size], dim=-1)

    torch.testing.assert_close(fused_q, split.q_proj(hidden_states), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(fused_k, split.k_proj(hidden_states), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(fused_v, split.v_proj(hidden_states), atol=1e-6, rtol=1e-5)
    assert hasattr(split, "q_proj") and not hasattr(split, "qkv_proj")
    assert hasattr(fused, "qkv_proj") and not hasattr(fused, "q_proj")
