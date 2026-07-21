# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from unittest.mock import patch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.kimi_linear.config import KimiLinearConfig
from nemo_automodel.components.models.kimi_linear.model import KimiLinearForCausalLM


def _tiny_kimi_config() -> KimiLinearConfig:
    return KimiLinearConfig(
        vocab_size=128,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        q_lora_rank=None,
        kv_lora_rank=8,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=8,
        mla_use_nope=True,
        num_experts=4,
        num_experts_per_token=2,
        num_shared_experts=0,
        moe_intermediate_size=8,
        first_k_dense_replace=1,
        linear_attn_config={
            "kda_layers": [1],
            "full_attn_layers": [2],
            "num_heads": 2,
            "head_dim": 8,
            "short_conv_kernel_size": 4,
        },
        torch_dtype="float32",
    )


def _backend_config() -> BackendConfig:
    return BackendConfig(
        attn="eager",
        linear="torch",
        rms_norm="torch_fp32",
        experts="torch",
        dispatcher="torch",
        enable_hf_state_dict_adapter=True,
    )


def test_update_moe_gate_bias_no_op_when_factor_zero():
    model = KimiLinearForCausalLM(_tiny_kimi_config(), backend=_backend_config())
    moe_layer = model.model.layers["1"].block_sparse_moe

    assert moe_layer.gate.bias_update_factor == 0.0
    with patch.object(moe_layer.gate, "update_bias") as mock_update_bias:
        model.update_moe_gate_bias()

    mock_update_bias.assert_not_called()


def test_kimi_moe_uses_hf_routing_numerics():
    model = KimiLinearForCausalLM(_tiny_kimi_config(), backend=_backend_config())
    moe_layer = model.model.layers["1"].block_sparse_moe

    assert not moe_layer.gate.router_topk_sorted
    assert moe_layer.gate.router_weights_fp32
    assert moe_layer.gate.router_weight_uses_score_correction_bias
    assert moe_layer.experts.route_weight_after_down_proj
