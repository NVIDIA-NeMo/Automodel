# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Regression tests for Mistral4 adaptive routing-bias precision."""

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.mistral4.configuration import Mistral4Config
from nemo_automodel.components.models.mistral4.model import (
    _HF_MISTRAL3_AVAILABLE,
    Mistral4ForCausalLM,
)


def _tiny_text_config() -> Mistral4Config:
    return Mistral4Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        moe_intermediate_size=4,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        n_shared_experts=1,
        n_routed_experts=2,
        kv_lora_rank=2,
        q_lora_rank=None,
        qk_rope_head_dim=2,
        v_head_dim=2,
        qk_nope_head_dim=2,
        n_group=1,
        topk_group=1,
        num_experts_per_tok=1,
        max_position_embeddings=16,
        torch_dtype=torch.float32,
    )


def _torch_backend() -> BackendConfig:
    return BackendConfig(
        attn="sdpa",
        linear="torch",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        rope_fusion=False,
        enable_hf_state_dict_adapter=False,
    )


def _assert_router_biases_are_fp32(model: torch.nn.Module) -> None:
    biases = [(name, buffer) for name, buffer in model.named_buffers() if name.endswith("e_score_correction_bias")]

    assert biases, "Mistral4 should create adaptive routing-bias buffers"
    for name, bias in biases:
        assert bias.dtype == torch.float32, f"routing bias {name} was cast to {bias.dtype}"
        torch.testing.assert_close(bias, torch.zeros_like(bias))


def test_text_initialize_weights_bf16_keeps_router_bias_fp32() -> None:
    model = Mistral4ForCausalLM(_tiny_text_config(), backend=_torch_backend())
    model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.bfloat16)

    _assert_router_biases_are_fp32(model)
    assert model.lm_head.weight.dtype == torch.bfloat16


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("gate_precision", [None, torch.float32])
def test_router_matches_hf_precision(dtype: torch.dtype, gate_precision: torch.dtype | None) -> None:
    """Compare selected experts, routing weights, and gradients with the HF router."""
    from transformers.models.mistral4.configuration_mistral4 import Mistral4Config as HFMistral4Config
    from transformers.models.mistral4.modeling_mistral4 import Mistral4TopkRouter

    torch.manual_seed(7)
    config = _tiny_text_config()
    config.torch_dtype = dtype
    config.n_routed_experts = 128
    config.num_experts_per_tok = 4
    config.v_head_dim = config.head_dim = config.qk_head_dim
    backend = _torch_backend()
    backend.gate_precision = gate_precision
    model = Mistral4ForCausalLM(config, backend=backend).eval()
    gate = model.model.layers["0"].mlp.gate
    reference_dtype = gate_precision or dtype
    reference = Mistral4TopkRouter(HFMistral4Config(**config.to_dict())).to(dtype=reference_dtype).eval()
    with torch.no_grad():
        gate.weight.normal_(std=0.5)
        gate.e_score_correction_bias.zero_()
        reference.weight.copy_(gate.weight)

    hidden_states = torch.randn(128, config.hidden_size, dtype=dtype, requires_grad=True)
    reference_hidden_states = hidden_states.detach().to(reference_dtype).requires_grad_(True)
    token_mask = torch.ones(hidden_states.shape[0], dtype=torch.bool)
    actual_weights, actual_indices, _ = gate(hidden_states, token_mask, None)
    _, expected_weights, expected_indices = reference(reference_hidden_states)
    # Expert order may differ; compare the complete sparse routing assignment.
    actual = torch.zeros(128, config.n_routed_experts, dtype=dtype).scatter(1, actual_indices, actual_weights)
    expected = torch.zeros(128, config.n_routed_experts, dtype=dtype).scatter(
        1, expected_indices, expected_weights.to(dtype)
    )
    torch.testing.assert_close(actual_indices.sort(-1).values, expected_indices.sort(-1).values, atol=0, rtol=0)
    # FP32 normalization can sum the selected weights in a different expert order.
    torch.testing.assert_close(actual, expected, atol=0 if dtype == torch.bfloat16 else 1e-7, rtol=1e-6)
    gradient = torch.randn_like(actual)
    (actual * gradient).sum().backward()
    (expected * gradient).sum().backward()
    tolerance = 0.008 if dtype == torch.bfloat16 else 1e-6
    torch.testing.assert_close(
        hidden_states.grad, reference_hidden_states.grad.to(dtype), atol=tolerance, rtol=tolerance
    )
    torch.testing.assert_close(gate.weight.grad, reference.weight.grad.to(dtype), atol=tolerance, rtol=tolerance)
    assert backend.gate_precision is gate_precision


@pytest.mark.skipif(not _HF_MISTRAL3_AVAILABLE, reason="transformers Mistral3 model is unavailable")
def test_multimodal_initialize_weights_bf16_keeps_router_bias_fp32() -> None:
    from transformers.models.mistral3.configuration_mistral3 import Mistral3Config

    from nemo_automodel.components.models.mistral4.model import Mistral3ForConditionalGeneration

    config = Mistral3Config(
        text_config=_tiny_text_config().to_dict(),
        vision_config={
            "model_type": "pixtral",
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_channels": 3,
            "image_size": 4,
            "patch_size": 2,
        },
        image_token_index=10,
        spatial_merge_size=2,
    )
    model = Mistral3ForConditionalGeneration(config, backend=_torch_backend())
    model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.bfloat16)

    _assert_router_biases_are_fp32(model)
    assert model.lm_head.weight.dtype == torch.bfloat16
