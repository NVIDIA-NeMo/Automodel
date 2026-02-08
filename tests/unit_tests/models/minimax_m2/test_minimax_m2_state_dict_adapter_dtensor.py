# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.minimax_m2.state_dict_adapter import MiniMaxM2StateDictAdapter
from nemo_automodel.components.moe.layers import MoEConfig


@dataclass
class MockMiniMaxM2Config:
    hidden_size: int = 64
    intermediate_size: int = 32
    num_local_experts: int = 2
    torch_dtype: str = "bfloat16"


@pytest.fixture
def adapter():
    config = MockMiniMaxM2Config()
    moe_config = MoEConfig(
        dim=64,
        inter_dim=32,
        moe_inter_dim=32,
        n_routed_experts=2,
        n_shared_experts=0,
        n_activated_experts=2,
        n_expert_groups=0,
        n_limited_groups=0,
        train_gate=True,
        gate_bias_update_factor=0.0,
        score_func="sigmoid",
        route_scale=1.0,
        aux_loss_coeff=0.0,
        norm_topk_prob=True,
        router_bias=False,
        expert_bias=False,
        expert_activation="swiglu",
        dtype=torch.bfloat16,
        force_e_score_correction_bias=True,
    )
    backend = BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        enable_deepep=False,
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=True,
    )
    return MiniMaxM2StateDictAdapter(config, moe_config, backend, dtype=torch.bfloat16)


def test_to_hf_quantization_adds_scale_inv_for_quantized_weights(adapter):
    native_state_dict = {
        "model.layers.0.self_attn.q_proj.weight": torch.randn(64, 64),
        "model.layers.0.self_attn.q_norm.weight": torch.randn(64),
    }

    hf_state_dict = adapter.to_hf(native_state_dict, quantization=True)

    assert "model.layers.0.self_attn.q_proj.weight" in hf_state_dict
    assert "model.layers.0.self_attn.q_proj.weight_scale_inv" in hf_state_dict
    # q_norm is explicitly excluded from quantization
    assert "model.layers.0.self_attn.q_norm.weight_scale_inv" not in hf_state_dict
