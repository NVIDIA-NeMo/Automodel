# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from unittest.mock import patch

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.kimi_linear.config import KimiLinearConfig
from nemo_automodel.components.models.kimi_linear.model import KimiLinearForCausalLM


def _tiny_kimi_config(*, use_kda: bool = False) -> KimiLinearConfig:
    linear_attn_config = (
        {"kda_layers": [1], "full_attn_layers": [2]} if use_kda else {"kda_layers": [], "full_attn_layers": [1, 2]}
    )
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
            **linear_attn_config,
            "num_heads": 2,
            "head_dim": 8,
            "short_conv_kernel_size": 4,
        },
        torch_dtype="float32",
    )


def _require_fla() -> None:
    pytest.importorskip("fla")
    pytest.importorskip("fla.modules")
    pytest.importorskip("fla.ops.kda")
    pytest.importorskip("fla.ops.kda.gate")


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


def test_tiny_kimi_with_kda_constructs_when_fla_available():
    _require_fla()

    model = KimiLinearForCausalLM(_tiny_kimi_config(use_kda=True), backend=_backend_config())

    assert model.model.layers["0"].is_linear_attn


def test_kimi_moe_uses_hf_routing_numerics():
    model = KimiLinearForCausalLM(_tiny_kimi_config(), backend=_backend_config())
    moe_layer = model.model.layers["1"].block_sparse_moe

    assert not moe_layer.gate.router_topk_sorted
    assert moe_layer.gate.router_weights_fp32
    assert moe_layer.gate.router_weight_uses_score_correction_bias
    assert moe_layer.experts.route_weight_after_down_proj


def test_initialize_weights_respects_explicit_buffer_device_on_cpu():
    model = KimiLinearForCausalLM(_tiny_kimi_config(), backend=_backend_config())
    explicit_device = torch.device("meta")

    with (
        patch("torch.cuda.is_available", return_value=False),
        patch.object(model.model, "init_weights") as mock_model_init,
    ):
        model.initialize_weights(buffer_device=explicit_device, dtype=torch.float32)

    mock_model_init.assert_called_once()
    assert mock_model_init.call_args.args[0] == explicit_device


def test_checkpoint_free_initialize_and_eval_forward_runs_hf_order_moe():
    model = KimiLinearForCausalLM(_tiny_kimi_config(), backend=_backend_config())
    model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.float32)
    model.eval()

    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        if tensor.is_floating_point():
            assert torch.isfinite(tensor).all(), name

    moe_layer = model.model.layers["1"]
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    with (
        patch.object(moe_layer, "_moe_infer_hf_order", wraps=moe_layer._moe_infer_hf_order) as mock_hf_order,
        torch.inference_mode(),
    ):
        output = model(input_ids=input_ids)

    mock_hf_order.assert_called_once()
    assert output.logits.shape == (2, 3, model.vocab_size)
    assert torch.isfinite(output.logits).all()
