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

import pytest
import torch
from transformers.models.mistral4.configuration_mistral4 import Mistral4Config as HFMistral4Config
from transformers.models.mistral4.modeling_mistral4 import Mistral4Attention as HFMistral4Attention
from transformers.models.mistral4.modeling_mistral4 import Mistral4RotaryEmbedding

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v3.rope_utils import (
    freqs_cis_from_position_ids,
    precompute_freqs_cis,
)
from nemo_automodel.components.models.mistral4.configuration import Mistral4Config
from nemo_automodel.components.models.mistral4.model import Mistral4MLA, _get_llama_4_attn_scale


def _tiny_long_context_config() -> Mistral4Config:
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
        v_head_dim=4,
        qk_nope_head_dim=2,
        n_group=1,
        topk_group=1,
        num_experts_per_tok=1,
        max_position_embeddings=1048576,
        torch_dtype=torch.float32,
    )


def _torch_backend() -> BackendConfig:
    return BackendConfig(
        attn="sdpa",
        linear="torch",
        rms_norm="torch",
        rope_fusion=False,
        enable_hf_state_dict_adapter=False,
    )


@pytest.mark.parametrize(
    "rope_overrides",
    [
        {"rope_type": "yarn"},
        {"rope_type": "yarn", "mscale": 0.5, "mscale_all_dim": 1.0},
        {"rope_type": "yarn", "mscale_all_dim": 0.0},
        {"rope_type": "yarn", "factor": 1.0},
        {"rope_type": "default"},
        {"type": "yarn"},
    ],
)
def test_mistral4_attention_scale_matches_hf(rope_overrides: dict[str, str | float]) -> None:
    config = _tiny_long_context_config()
    config.rope_parameters.update(rope_overrides)
    attention = Mistral4MLA(config, _torch_backend())
    reference = HFMistral4Attention(HFMistral4Config(**config.to_dict()), layer_idx=0)
    assert attention.softmax_scale == pytest.approx(reference.scaling)


@pytest.mark.parametrize("position_offset", [0, 8192])
@pytest.mark.parametrize("q_lora_rank", [None, 2])
def test_mistral4_attention_forward_backward_matches_hf(position_offset: int, q_lora_rank: int | None) -> None:
    """Compare real attention outputs and gradients with the pinned HF implementation."""
    torch.manual_seed(42)
    config = _tiny_long_context_config()
    config.q_lora_rank = q_lora_rank
    hf_config = HFMistral4Config(**config.to_dict())
    hf_config._attn_implementation = "sdpa"
    reference = HFMistral4Attention(hf_config, layer_idx=0).float()
    attention = Mistral4MLA(config, _torch_backend()).float()
    attention.load_state_dict(reference.state_dict(), strict=True)

    hidden_states = torch.randn(2, 8, config.hidden_size, requires_grad=True)
    reference_hidden_states = hidden_states.detach().clone().requires_grad_(True)
    position_ids = torch.arange(position_offset, position_offset + 8).unsqueeze(0).expand(2, -1)
    frequencies = precompute_freqs_cis(
        config.qk_rope_head_dim,
        config.max_position_embeddings,
        config.rope_parameters["rope_theta"],
        config.rope_parameters,
    )
    freqs_cis = freqs_cis_from_position_ids(position_ids, frequencies)
    position_embeddings = Mistral4RotaryEmbedding(hf_config)(reference_hidden_states, position_ids)

    actual = attention(hidden_states, freqs_cis, position_ids=position_ids)
    expected, _ = reference(
        reference_hidden_states,
        position_embeddings=position_embeddings,
        attention_mask=None,
        position_ids=position_ids,
    )
    # FP32 uses equivalent rotary layouts and SDPA with different operation order.
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    gradient = torch.randn_like(expected)
    actual.backward(gradient)
    expected.backward(gradient)
    torch.testing.assert_close(hidden_states.grad, reference_hidden_states.grad, atol=2e-6, rtol=2e-5)
    reference_parameters = dict(reference.named_parameters())
    for name, parameter in attention.named_parameters():
        torch.testing.assert_close(parameter.grad, reference_parameters[name].grad, atol=2e-6, rtol=2e-5)


def test_mistral4_long_context_scale_applies_to_full_query() -> None:
    config = _tiny_long_context_config()
    attention = Mistral4MLA(config, _torch_backend())
    captured_queries: list[torch.Tensor] = []

    def capture_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        """Capture SDPA inputs while returning a shape-compatible result.

        Args:
            query: Tensor of shape [batch, heads, sequence, qk_head_dim].
            key: Tensor of shape [batch, heads, sequence, qk_head_dim].
            value: Tensor of shape [batch, heads, sequence, v_head_dim].
            **kwargs: Attention metadata unused by this test stand-in.

        Returns:
            Zero tensor of shape [batch, heads, sequence, v_head_dim].
        """
        del key, kwargs
        captured_queries.append(query.detach().clone())
        return torch.zeros_like(value)

    attention.attn_func = capture_attention
    with torch.no_grad():
        attention.q_proj.weight.zero_()
        attention.q_proj.weight[: config.qk_head_dim, : config.qk_head_dim].copy_(torch.eye(config.qk_head_dim))

    hidden_states = torch.ones(1, 1, config.hidden_size)
    freqs_cis = torch.ones(1, 1, config.qk_rope_head_dim // 2, dtype=torch.complex64)
    original_max_position = config.rope_parameters["original_max_position_embeddings"]
    low_position_ids = torch.zeros(1, 1, dtype=torch.long)
    high_position_ids = torch.full((1, 1), original_max_position, dtype=torch.long)

    attention(hidden_states, freqs_cis, position_ids=low_position_ids)
    attention(hidden_states, freqs_cis, position_ids=high_position_ids)

    expected_scale = _get_llama_4_attn_scale(
        high_position_ids,
        config.rope_parameters["llama_4_scaling_beta"],
        original_max_position,
    ).item()
    torch.testing.assert_close(captured_queries[1], captured_queries[0] * expected_scale)
