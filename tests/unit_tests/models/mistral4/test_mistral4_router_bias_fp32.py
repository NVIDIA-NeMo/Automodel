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
        n_routed_experts=4,
        kv_lora_rank=2,
        q_lora_rank=None,
        qk_rope_head_dim=2,
        v_head_dim=2,
        qk_nope_head_dim=2,
        n_group=1,
        topk_group=1,
        num_experts_per_tok=2,
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


def test_default_training_preserves_hf_router_choices() -> None:
    from unittest.mock import patch

    from transformers.models.mistral4.configuration_mistral4 import Mistral4Config as HFConfig
    from transformers.models.mistral4.modeling_mistral4 import Mistral4MoE

    config = _tiny_text_config()
    model = Mistral4ForCausalLM(config, backend=_torch_backend()).train()
    gate = model.model.layers["0"].mlp.gate
    reference = Mistral4MoE(HFConfig(**config.to_dict()))
    with torch.no_grad():
        gate.weight.zero_()
        for parameter in reference.parameters():
            parameter.zero_()
    hidden = torch.ones(8, config.hidden_size)
    token_mask = torch.ones(8, dtype=torch.bool)
    # The experts' input contract is stable across HF's router refactoring.
    with patch.object(reference.experts, "forward", wraps=reference.experts.forward) as experts_forward:
        reference(hidden)
    _, expected_indices, expected_weights = experts_forward.call_args.args
    for _ in range(7):
        gate(hidden, token_mask, None)
        model.update_moe_gate_bias()
    weights, indices, _ = gate(hidden, token_mask, None)

    torch.testing.assert_close(indices, expected_indices)
    torch.testing.assert_close(weights, expected_weights)
    torch.testing.assert_close(gate.e_score_correction_bias, torch.zeros(config.n_routed_experts))
    assert gate.weight.requires_grad


def test_explicit_adaptive_router_bias_remains_supported() -> None:
    config = _tiny_text_config()
    model = Mistral4ForCausalLM(
        config, backend=_torch_backend(), moe_overrides={"gate_bias_update_factor": 1e-3}
    ).train()
    gate = model.model.layers["0"].mlp.gate
    with torch.no_grad():
        gate.weight.zero_()
    hidden = torch.ones(8, config.hidden_size)
    token_mask = torch.ones(8, dtype=torch.bool)
    _, before_indices, _ = gate(hidden, token_mask, None)
    model.update_moe_gate_bias()
    _, after_indices, _ = gate(hidden, token_mask, None)

    assert not torch.equal(before_indices, after_indices)
    assert torch.count_nonzero(gate.e_score_correction_bias) == config.n_routed_experts

    restored = Mistral4ForCausalLM(config, backend=_torch_backend()).train()
    restored.load_state_dict(model.state_dict())
    restored.update_moe_gate_bias()
    restored_gate = restored.model.layers["0"].mlp.gate
    _, restored_indices, _ = restored_gate(hidden, token_mask, None)
    torch.testing.assert_close(restored_gate.e_score_correction_bias, gate.e_score_correction_bias)
    torch.testing.assert_close(restored_indices, after_indices)


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
